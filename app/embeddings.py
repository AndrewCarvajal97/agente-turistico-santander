"""Generación de embeddings usando OCI Generative AI (modelos Cohere).

OCI limita cada petición de embeddings a 96 textos, por lo que se procesa por lotes.
Se distingue entre embeddings de documentos ("SEARCH_DOCUMENT") y de la consulta
("SEARCH_QUERY"), tal como recomienda Cohere para mejorar la recuperación.
"""
from __future__ import annotations

import oci

from .config import settings
from .oci_client import build_genai_client

_LOTE_MAX = 96


def _serving_mode(model_id: str):
    return oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_id)


def _embed(textos: list[str], input_type: str) -> list[list[float]]:
    """Llama a OCI para obtener los embeddings de una lista de textos."""
    if not textos:
        return []

    cliente = build_genai_client()
    resultado: list[list[float]] = []

    for i in range(0, len(textos), _LOTE_MAX):
        lote = textos[i : i + _LOTE_MAX]
        detalles = oci.generative_ai_inference.models.EmbedTextDetails(
            inputs=lote,
            serving_mode=_serving_mode(settings.embed_model),
            compartment_id=settings.compartment_id,
            input_type=input_type,
            truncate="END",
        )
        respuesta = cliente.embed_text(detalles)
        resultado.extend(respuesta.data.embeddings)

    return resultado


def embed_documentos(textos: list[str]) -> list[list[float]]:
    """Embeddings para los fragmentos del documento (indexación)."""
    return _embed(textos, input_type="SEARCH_DOCUMENT")


def embed_consulta(texto: str) -> list[float]:
    """Embedding para la pregunta del usuario (búsqueda)."""
    vectores = _embed([texto], input_type="SEARCH_QUERY")
    return vectores[0]
