"""Herramientas (Tools) para el agente orquestador.

Cada herramienta es una función que el agente ReAct puede decidir usar. Se
definen con el decorador `@tool` de LangChain; su docstring es la "descripción"
que el agente lee para decidir cuándo usarla.

Este módulo es **paralelo**: no toca los endpoints existentes (`/ask`, `/vision`).
Está pensado para crecer (p. ej. una futura herramienta de base de datos).
"""
from __future__ import annotations

from langchain_core.tools import tool

from .agent import TourismAgent
from .memory import ConversationMemory

# Instancias propias del orquestador (aisladas del resto de la app).
_agente = TourismAgent()
_memoria = ConversationMemory()


def _asegurar_indexado() -> None:
    if not _agente.esta_listo():
        _agente.indexar()


@tool
def guia_turistica(pregunta: str) -> str:
    """Responde preguntas sobre turismo en Santander, Colombia (destinos, gastronomía,
    deportes de aventura, transporte, mejor época) usando la guía oficial en PDF.
    Úsala para cualquier consulta sobre qué visitar, comer o hacer en Santander."""
    _asegurar_indexado()
    return _agente.preguntar(pregunta)["respuesta"]


@tool
def buscar_historial(termino: str) -> str:
    """Busca en el historial de conversaciones guardadas las que mencionan un término.
    Úsala cuando pregunten qué han consultado antes los usuarios o para recordar
    conversaciones previas."""
    resultados = _memoria.buscar(termino)
    if not resultados:
        return f"No hay conversaciones guardadas que mencionen '{termino}'."
    return "\n".join(
        f"- {r['pregunta']} -> {r['respuesta'][:80]}" for r in resultados[:5]
    )


# Lista de herramientas disponibles para el orquestador.
HERRAMIENTAS = [guia_turistica, buscar_historial]
