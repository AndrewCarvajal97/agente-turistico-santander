"""Orquestación del agente RAG.

Flujo:
  1. Indexar (una sola vez): PDF -> texto -> chunks -> embeddings -> VectorStore.
  2. Preguntar: pregunta -> embedding -> recuperar contexto -> prompt -> LLM -> respuesta.
"""
from __future__ import annotations

import oci

from .chunker import dividir_en_chunks
from .config import settings
from .embeddings import embed_consulta, embed_documentos
from .oci_client import build_genai_client
from .pdf_loader import leer_pdf
from .vector_store import VectorStore

SYSTEM_PROMPT = (
    "Eres un asistente turístico experto en el departamento de Santander, Colombia. "
    "Responde de forma clara, amable y concisa, ÚNICAMENTE con la información del "
    "contexto proporcionado. Si la respuesta no está en el contexto, indica con "
    "honestidad que no cuentas con esa información en la guía. Responde en español."
)


class TourismAgent:
    """Agente que responde preguntas sobre la guía turística de Santander."""

    def __init__(self) -> None:
        self.store: VectorStore | None = None

    # ------------------------------------------------------------------ #
    # Indexación
    # ------------------------------------------------------------------ #
    def indexar(self, pdf_path: str | None = None, forzar: bool = False) -> int:
        """Construye (o carga) el índice vectorial del documento.

        Returns:
            Número de fragmentos indexados.
        """
        ruta_pdf = pdf_path or settings.pdf_path

        if not forzar and VectorStore.existe(settings.index_path):
            self.store = VectorStore.cargar(settings.index_path)
            return len(self.store.textos)

        texto = leer_pdf(ruta_pdf)
        chunks = dividir_en_chunks(
            texto, tamano=settings.chunk_size, solapamiento=settings.chunk_overlap
        )
        vectores = embed_documentos(chunks)

        self.store = VectorStore(chunks, vectores)
        self.store.guardar(settings.index_path)
        return len(chunks)

    def esta_listo(self) -> bool:
        return self.store is not None and len(self.store.textos) > 0

    # ------------------------------------------------------------------ #
    # Consulta
    # ------------------------------------------------------------------ #
    def preguntar(self, pregunta: str) -> dict:
        """Responde una pregunta usando RAG.

        Returns:
            {"respuesta": str, "fuentes": list[dict]}
        """
        if not self.esta_listo():
            raise RuntimeError("El índice no está construido. Llama a indexar() primero.")

        pregunta = (pregunta or "").strip()
        if not pregunta:
            return {"respuesta": "Por favor, escribe una pregunta.", "fuentes": []}

        # 1) Recuperar los fragmentos más relevantes.
        emb = embed_consulta(pregunta)
        recuperados = self.store.buscar(emb, k=settings.top_k)
        contexto = "\n\n---\n\n".join(r["texto"] for r in recuperados)

        # 2) Construir el prompt y llamar al LLM.
        respuesta = self._generar(pregunta, contexto)

        return {
            "respuesta": respuesta,
            "fuentes": [
                {"fragmento": r["texto"][:220] + "…", "score": round(r["score"], 3)}
                for r in recuperados
            ],
        }

    def _generar(self, pregunta: str, contexto: str) -> str:
        """Llama al modelo de chat de OCI con el contexto recuperado."""
        cliente = build_genai_client()

        mensaje = (
            f"{SYSTEM_PROMPT}\n\n"
            f"### Contexto de la guía:\n{contexto}\n\n"
            f"### Pregunta del usuario:\n{pregunta}"
        )

        chat_request = oci.generative_ai_inference.models.CohereChatRequest(
            message=mensaje,
            max_tokens=600,
            temperature=0.2,
        )
        detalles = oci.generative_ai_inference.models.ChatDetails(
            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                model_id=settings.chat_model
            ),
            chat_request=chat_request,
            compartment_id=settings.compartment_id,
        )
        respuesta = cliente.chat(detalles)
        return respuesta.data.chat_response.text.strip()
