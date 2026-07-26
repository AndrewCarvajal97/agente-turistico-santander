"""RAG real (Retrieval-Augmented Generation) con embeddings y FAISS.

A diferencia del agente por **contexto completo** (`app/agent.py`), aquí el flujo es
el RAG clásico:

  1. **Cargar** el PDF y dividirlo en *chunks* (`RecursiveCharacterTextSplitter`).
  2. **Embeddings**: convertir cada chunk en un vector semántico (Cohere).
  3. **Indexar** los vectores en **FAISS** (base vectorial en memoria).
  4. **Recuperar** (retrieval): por cada pregunta se buscan los chunks más
     similares y solo esos se pasan al LLM para **generar** la respuesta.

Es una vía **paralela** (endpoint `/rag/ask`): no altera `/ask`. Sirve para escalar
a documentos grandes o a múltiples fuentes, donde inyectar todo el texto no es viable.
"""
from __future__ import annotations

from . import llm
from .config import settings
from .pdf_loader import leer_pdf

SYSTEM_PROMPT_RAG = (
    "Eres un asistente turístico experto en Santander, Colombia. Responde en español, "
    "de forma clara y concisa, ÚNICAMENTE con la información del contexto recuperado. "
    "Si no está en el contexto, indícalo con honestidad."
)


class RagSantander:
    """Índice vectorial (FAISS) del documento y recuperación semántica."""

    def __init__(self) -> None:
        self.retriever = None

    def _embeddings(self):
        """Modelo de embeddings: Cohere por defecto (gratis en el trial) o Gemini."""
        if settings.rag_embed_provider.lower() == "gemini" and settings.gemini_api_key:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001", google_api_key=settings.gemini_api_key
            )
        from langchain_cohere import CohereEmbeddings

        if not settings.cohere_api_key:
            raise ValueError("El RAG necesita COHERE_API_KEY (o RAG_EMBED_PROVIDER=gemini).")
        return CohereEmbeddings(
            cohere_api_key=settings.cohere_api_key, model=settings.cohere_embed_model
        )

    def indexar(self, pdf_path: str | None = None) -> int:
        """Carga el PDF, lo divide en chunks, construye FAISS y crea el retriever."""
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        texto = leer_pdf(pdf_path or settings.pdf_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap
        )
        fragmentos = splitter.split_text(texto)
        documentos = [Document(page_content=f) for f in fragmentos]
        vectorstore = FAISS.from_documents(documentos, self._embeddings())
        # Retriever con umbral de similitud: descarta fragmentos poco relevantes.
        self.retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": settings.rag_score_threshold,
                "k": settings.rag_top_k,
            },
        )
        return len(fragmentos)

    def esta_listo(self) -> bool:
        return self.retriever is not None

    def preguntar(self, pregunta: str) -> dict:
        """Recupera los chunks más relevantes (retriever) y genera la respuesta."""
        if not self.esta_listo():
            self.indexar()  # indexación perezosa (en la primera consulta)

        pregunta = (pregunta or "").strip()
        if not pregunta:
            return {"respuesta": "Por favor, escribe una pregunta.", "fragmentos": []}

        encontrados = self.retriever.invoke(pregunta)
        if not encontrados:
            return {
                "respuesta": "No encontré información suficientemente relevante en la guía "
                "para responder esa pregunta.",
                "fragmentos": [],
            }
        contexto = "\n\n---\n\n".join(d.page_content for d in encontrados)

        mensaje = f"### Contexto recuperado:\n{contexto}\n\n### Pregunta:\n{pregunta}"
        respuesta = llm.generar_texto(mensaje, SYSTEM_PROMPT_RAG, settings.max_output_tokens)
        return {
            "respuesta": respuesta,
            "fragmentos": [d.page_content[:180].strip() + "…" for d in encontrados],
        }


# Instancia única (el índice se construye de forma perezosa en la primera pregunta).
rag = RagSantander()
