"""Agente orquestador (ReAct) con LangGraph.

Un agente ReAct **razona y decide** qué herramienta usar según la consulta, en
lugar de un ruteo fijo. En LangChain 1.x, el agente ReAct se crea con
`langgraph.prebuilt.create_react_agent` (tool-calling), que es el equivalente
moderno del `create_react_agent` clásico del curso.

Este orquestador es **paralelo** a los endpoints existentes: se expone en
`/agente` sin alterar `/ask` ni `/vision`.
"""
from __future__ import annotations

from .config import settings
from .tools import HERRAMIENTAS

SYSTEM_ORQUESTADOR = (
    "Eres un asistente turístico experto en Santander, Colombia. Responde en español, "
    "de forma clara y amable. Usa las herramientas disponibles cuando ayuden a responder; "
    "para preguntas sobre turismo usa la guía. Si la herramienta no trae información, dilo "
    "con honestidad."
)


def _modelo_chat():
    """Construye un chat model de LangChain (con soporte de tool-calling)."""
    prov = settings.llm_provider.lower()
    if prov == "cohere" and settings.cohere_api_key:
        from langchain_cohere import ChatCohere

        return ChatCohere(
            model=settings.cohere_model, cohere_api_key=settings.cohere_api_key, temperature=0
        )
    if prov == "gemini" and settings.gemini_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.chat_model, google_api_key=settings.gemini_api_key, temperature=0
        )
    # Por defecto Groq (Llama soporta tool-calling y tiene buen free tier).
    from langchain_groq import ChatGroq

    return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0)


_agente_cache = None


def _get_agente():
    """Crea (una sola vez) el agente ReAct con el modelo y las herramientas."""
    global _agente_cache
    if _agente_cache is None:
        from langgraph.prebuilt import create_react_agent

        _agente_cache = create_react_agent(
            _modelo_chat(), HERRAMIENTAS, prompt=SYSTEM_ORQUESTADOR
        )
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
