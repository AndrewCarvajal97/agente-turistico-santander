"""Generación de embeddings usando Google Gemini (modelo text-embedding-004).

Se distingue entre embeddings de documentos ("RETRIEVAL_DOCUMENT") y de la
consulta ("RETRIEVAL_QUERY"), lo que mejora la calidad de la recuperación en el RAG.
Los textos se procesan por lotes para respetar los límites de la API.
"""
from __future__ import annotations

from google.genai import types

from .config import settings
from .llm_client import build_client

_LOTE_MAX = 100


def _embed(textos: list[str], task_type: str) -> list[list[float]]:
    """Llama a Gemini para obtener los embeddings de una lista de textos."""
    if not textos:
        return []

    cliente = build_client()
    resultado: list[list[float]] = []

    for i in range(0, len(textos), _LOTE_MAX):
        lote = textos[i : i + _LOTE_MAX]
        respuesta = cliente.models.embed_content(
            model=settings.embed_model,
            contents=lote,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        resultado.extend(emb.values for emb in respuesta.embeddings)

    return resultado


def embed_documentos(textos: list[str]) -> list[list[float]]:
    """Embeddings para los fragmentos del documento (indexación)."""
    return _embed(textos, task_type="RETRIEVAL_DOCUMENT")


def embed_consulta(texto: str) -> list[float]:
    """Embedding para la pregunta del usuario (búsqueda)."""
    vectores = _embed([texto], task_type="RETRIEVAL_QUERY")
    return vectores[0]
