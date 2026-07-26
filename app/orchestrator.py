"""Agente orquestador (ReAct) con LangGraph.

Un agente ReAct **razona y decide** qué herramienta usar según la consulta, en
lugar de un ruteo fijo. En LangChain 1.x, el agente ReAct se crea con
`langgraph.prebuilt.create_react_agent` (tool-calling), que es el equivalente
moderno del `create_react_agent` clásico del curso.

Este orquestador es **paralelo** a los endpoints existentes: se expone en
`/agente` sin alterar `/ask` ni `/vision`.
"""
from __future__ import annotations

from . import llm
from .tools import HERRAMIENTAS

SYSTEM_ORQUESTADOR = (
    "Eres un asistente turístico experto en Santander, Colombia. Responde en español, "
    "de forma clara y amable. Usa las herramientas disponibles cuando ayuden a responder:\n"
    "- Para lo ESTÁTICO de Santander (destinos, gastronomía, rutas, mejor época) usa "
    "'guia_turistica' (la guía oficial en PDF).\n"
    "- Para información ACTUAL o en tiempo real (eventos y ferias próximas, clima de hoy, "
    "precios vigentes, noticias) usa 'busca_web'; cuando la uses, cita la fuente (el enlace).\n"
    "- Prioriza la guía para lo que ella cubre; recurre a la web solo para lo que la guía no "
    "puede saber. Si una herramienta no trae información, dilo con honestidad."
)

_agente_cache = None


def _get_agente():
    """Crea (una sola vez) el agente ReAct con el modelo y las herramientas."""
    global _agente_cache
    if _agente_cache is None:
        from langgraph.prebuilt import create_react_agent

        # Un solo modelo (los agentes usan bind_tools; sin .with_fallbacks()).
        modelo = llm.construir_chat_model(temperature=0, con_respaldo=False)
        _agente_cache = create_react_agent(modelo, HERRAMIENTAS, prompt=SYSTEM_ORQUESTADOR)
    return _agente_cache


def responder(pregunta: str) -> dict:
    """Ejecuta el agente orquestador y devuelve la respuesta + herramientas usadas."""
    agente = _get_agente()
    resultado = agente.invoke({"messages": [("user", pregunta)]})
    mensajes = resultado.get("messages", [])

    respuesta = mensajes[-1].content if mensajes else ""
    # Registra qué herramientas usó el agente (para mostrar su razonamiento).
    herramientas_usadas = [
        getattr(m, "name", "") for m in mensajes if getattr(m, "type", "") == "tool"
    ]
    return {"respuesta": respuesta, "herramientas_usadas": herramientas_usadas}
