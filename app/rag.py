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

import os

from . import llm
from .config import settings

SYSTEM_PROMPT_RAG = (
    "Eres un asistente turístico experto en Santander, Colombia. Responde en español, de "
    "forma clara y concisa, ÚNICAMENTE con la información del contexto proporcionado. "
    "Si la respuesta no está en el contexto, responde exactamente 'No lo sé'."
)


class RagSantander:
    """Índice vectorial (FAISS) del documento y recuperación semántica."""

    def __init__(self) -> None:
        self.retriever = None
        self.document_chain = None

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

    def indexar(self, docs_dir: str | None = None) -> int:
        """Indexa TODOS los PDFs del directorio: chunks, FAISS, retriever y cadena."""
        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
        from langchain_community.vectorstores import FAISS
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # DirectoryLoader + PyPDFLoader: carga todos los PDFs del directorio (una página
        # por Document con metadata source/page), conservada al dividir -> citaciones
        # trazables. Escalable: agregar más PDFs a la carpeta los indexa automáticamente.
        directorio = docs_dir or settings.rag_docs_dir
        paginas = DirectoryLoader(
            directorio, glob="**/*.pdf", loader_cls=PyPDFLoader
        ).load()
        if not paginas:
            raise FileNotFoundError(f"No se encontraron PDFs en '{directorio}' para el RAG.")

        # Chunking: "semantic" divide por significado (usa embeddings); "recursive"
        # (por defecto) divide por caracteres con solapamiento.
        if settings.rag_chunking.lower() == "semantic":
            from langchain_experimental.text_splitter import SemanticChunker

            splitter = SemanticChunker(self._embeddings())
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap
            )
        fragmentos = splitter.split_documents(paginas)
        vectorstore = FAISS.from_documents(fragmentos, self._embeddings())
        # Retriever con umbral de similitud: descarta fragmentos poco relevantes.
        self.retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": settings.rag_score_threshold,
                "k": settings.rag_top_k,
            },
        )
        # Cadena "stuff" (equivalente moderno de create_stuff_documents_chain en LCEL):
        # inserta los documentos recuperados en {context} y genera la respuesta.
        prompt_rag = ChatPromptTemplate(
            [
                ("system", SYSTEM_PROMPT_RAG),
                ("human", "Contexto:\n{context}\n\nPregunta: {input}"),
            ]
        )
        self.document_chain = (
            prompt_rag | llm.construir_chat_model(temperature=0.2) | StrOutputParser()
        )
        return len(fragmentos)

    def esta_listo(self) -> bool:
        return self.retriever is not None

    @staticmethod
    def _no_encontrado(respuesta: str = "No lo sé.") -> dict:
        return {"respuesta": respuesta, "citaciones": [], "documentos_encontrados": False}

    def preguntar(self, pregunta: str) -> dict:
        """Recupera los chunks relevantes y genera la respuesta con citaciones.

        Returns:
            {"respuesta": str, "citaciones": list[str], "documentos_encontrados": bool}
        """
        if not self.esta_listo():
            self.indexar()  # indexación perezosa (en la primera consulta)

        pregunta = (pregunta or "").strip()
        if not pregunta:
            return self._no_encontrado("Por favor, escribe una pregunta.")

        # 1) Recuperación: si nada supera el umbral, respondemos "No lo sé".
        documentos = self.retriever.invoke(pregunta)
        if not documentos:
            return self._no_encontrado()

        # 2) Generación: se "rellenan" (stuff) los documentos recuperados en el contexto.
        contexto = "\n\n".join(d.page_content for d in documentos)
        respuesta = self.document_chain.invoke({"input": pregunta, "context": contexto})

        # 3) El modelo también puede decir "No lo sé" si el contexto no sirve.
        if respuesta.strip().rstrip(".!?¡¿").lower() in ("no lo sé", "no lo se"):
            return self._no_encontrado()

        # Citaciones trazables: fragmento + página de origen (metadata del loader).
        citaciones = [
            {
                "contenido": d.page_content[:180].strip() + "…",
                "pagina": (d.metadata.get("page", 0) or 0) + 1,
                "fuente": os.path.basename(d.metadata.get("source", "")),
            }
            for d in documentos
        ]
        return {
            "respuesta": respuesta,
            "citaciones": citaciones,
            "documentos_encontrados": True,
        }


# Instancia única (el índice se construye de forma perezosa en la primera pregunta).
rag = RagSantander()
