"""Agente con grafo de estados (LangGraph) — triaje + RAG.

Modela el flujo como un grafo con `StateGraph`, nodos y aristas (incluida una
arista condicional), tal como se ve en el curso:

    START → triaje → (condicional)
                       ├─ auto_resolver  (responde con el RAG)
                       ├─ pedir_info     (pide más detalles)
                       └─ abrir_ticket   (solicitud para gestión humana)
                                          → END

Conecta lo construido antes: el **triaje** usa `with_structured_output` (salida
validada por Pydantic con `Literal`) y `auto_resolver` usa el **RAG** (`app/rag.py`).
Es una vía **paralela** (endpoint `/grafo/ask`): no altera los demás endpoints.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from . import llm

PROMPT_TRIAJE = (
    "Eres un especialista en triaje de un asistente turístico de Santander, Colombia. "
    "Dado el mensaje del usuario, clasifícalo. Reglas:\n"
    "- auto_resolver: preguntas claras sobre turismo en Santander (destinos, gastronomía, "
    "aventura, transporte, clima). Ej: '¿Dónde hago rafting?'.\n"
    "- pedir_info: mensajes vagos o sin contexto suficiente. Ej: 'Necesito ayuda con un viaje'.\n"
    "- abrir_ticket: solicitudes que requieren gestión humana (reservar, contactar un guía, "
    "quejas, personalizar un tour). Ej: 'Quiero reservar un tour a Barichara para 4 personas'."
)


class TriajeOut(BaseModel):
    """Salida estructurada del triaje (con Literal para restringir valores)."""

    decision: Literal["auto_resolver", "pedir_info", "abrir_ticket"]
    urgencia: Literal["baja", "mediana", "alta"] = "baja"
    campos_faltantes: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    """Estado que se propaga por los nodos del grafo."""

    pregunta: str
    triaje: dict
    respuesta: Optional[str]
    citaciones: Optional[list]
    documentos_encontrados: Optional[bool]
    accion_final: str


# --------------------------------------------------------------------- #
# Nodos
# --------------------------------------------------------------------- #
def nodo_triaje(state: AgentState) -> dict:
    """Clasifica la pregunta. Si el LLM falla, cae a 'auto_resolver' (RAG)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        modelo = llm.construir_chat_model(temperature=0, con_respaldo=False)
        salida = modelo.with_structured_output(TriajeOut).invoke(
            [SystemMessage(content=PROMPT_TRIAJE), HumanMessage(content=state["pregunta"])]
        )
        return {"triaje": salida.model_dump()}
    except Exception as exc:  # noqa: BLE001 - por defecto intentamos resolver con el RAG
        print(f"[grafo] triaje falló ({exc}); se usa auto_resolver por defecto.")
        return {"triaje": {"decision": "auto_resolver", "urgencia": "baja", "campos_faltantes": []}}


def nodo_auto_resolver(state: AgentState) -> dict:
    """Responde con el RAG (recuperación + generación con citaciones)."""
    from .rag import rag

    r = rag.preguntar(state["pregunta"])
    return {
        "respuesta": r["respuesta"],
        "citaciones": r["citaciones"],
        "documentos_encontrados": r["documentos_encontrados"],
        "accion_final": "AUTO_RESOLVER",
    }


def nodo_pedir_info(state: AgentState) -> dict:
    """Pide más detalles al usuario cuando la consulta es imprecisa."""
    faltantes = state.get("triaje", {}).get("campos_faltantes") or []
    extra = f" ¿Podrías precisar: {', '.join(faltantes)}?" if faltantes else ""
    return {
        "respuesta": "Necesito un poco más de contexto para ayudarte con tu viaje a "
        "Santander." + extra,
        "accion_final": "PEDIR_INFO",
    }


def nodo_abrir_ticket(state: AgentState) -> dict:
    """Registra la solicitud como un 'ticket' para gestión humana."""
    return {
        "respuesta": "He registrado tu solicitud como una petición para nuestro equipo. "
        "Un asesor turístico la atenderá pronto. 🎫",
        "accion_final": "ABRIR_TICKET",
    }


def arista_decision_triaje(state: AgentState) -> str:
    """Decide el siguiente nodo según la decisión del triaje."""
    decision = (state.get("triaje") or {}).get("decision", "auto_resolver")
    return {"auto_resolver": "rag", "pedir_info": "info", "abrir_ticket": "ticket"}.get(
        decision, "rag"
    )


# --------------------------------------------------------------------- #
# Construcción del grafo
# --------------------------------------------------------------------- #
_grafo = None


def _construir_grafo():
    from langgraph.graph import END, START, StateGraph

    workflow = StateGraph(AgentState)
    workflow.add_node("triaje", nodo_triaje)
    workflow.add_node("auto_resolver", nodo_auto_resolver)
    workflow.add_node("pedir_info", nodo_pedir_info)
    workflow.add_node("abrir_ticket", nodo_abrir_ticket)

    workflow.add_edge(START, "triaje")
    workflow.add_conditional_edges(
        "triaje",
        arista_decision_triaje,
        {"rag": "auto_resolver", "info": "pedir_info", "ticket": "abrir_ticket"},
    )
    workflow.add_edge("auto_resolver", END)
    workflow.add_edge("pedir_info", END)
    workflow.add_edge("abrir_ticket", END)
    return workflow.compile()


def responder(pregunta: str) -> dict:
    """Ejecuta el grafo y devuelve la respuesta + la decisión del triaje."""
    global _grafo
    if _grafo is None:
        _grafo = _construir_grafo()
    estado = _grafo.invoke({"pregunta": pregunta})
    return {
        "respuesta": estado.get("respuesta", ""),
        "decision": (estado.get("triaje") or {}).get("decision"),
        "accion_final": estado.get("accion_final"),
        "citaciones": estado.get("citaciones") or [],
    }
