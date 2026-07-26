"""RAG real (Retrieval-Augmented Generation) con embeddings y base vectorial.

A diferencia del agente por **contexto completo** (`app/agent.py`), aquí el flujo es
el RAG clásico:

  1. **Cargar** los PDFs y dividirlos en *chunks* (`RecursiveCharacterTextSplitter`).
  2. **Embeddings**: convertir cada chunk en un vector semántico (Cohere).
  3. **Indexar** los vectores en una **base vectorial** (strategy configurable):
       - ``faiss``    → local, **persistida en disco** (no re-indexa en cada arranque).
       - ``pinecone`` → en la **nube** (persistente y escalable a muchos documentos).
  4. **Recuperar** (retrieval): por cada pregunta se buscan los chunks más
     similares y solo esos se pasan al LLM para **generar** la respuesta.

El backend se elige con ``RAG_VECTORSTORE`` sin tocar el resto del pipeline. Es una
vía **paralela** (endpoint `/rag/ask`): no altera `/ask`. Sirve para escalar a
documentos grandes o a múltiples fuentes, donde inyectar todo el texto no es viable.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import llm
from .config import settings

SYSTEM_PROMPT_RAG = (
    "Eres un asistente turístico experto en Santander, Colombia. Responde en español, de "
    "forma clara y concisa, ÚNICAMENTE con la información del contexto proporcionado. "
    "Si la respuesta no está en el contexto, responde exactamente 'No lo sé'."
)


class RagSantander:
    """Base vectorial (FAISS o Pinecone) de los documentos y recuperación semántica."""

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

    def _cargar_fragmentos(self, embeddings, docs_dir: str | None):
        """Carga los PDFs del directorio y los divide en chunks (loader + splitter)."""
        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # DirectoryLoader + PyPDFLoader: carga todos los PDFs del directorio (una página
        # por Document con metadata source/page), conservada al dividir -> citaciones
        # trazables. Agregar más PDFs a la carpeta los indexa automáticamente.
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

            splitter = SemanticChunker(embeddings)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap
            )
        return splitter.split_documents(paginas)

    def _vs_faiss(self, embeddings, docs_dir, forzar):
        """Backend FAISS: base vectorial local **persistida en disco**.

        Si el índice ya existe y ``forzar`` es False, se **carga** (rápido y sin gastar
        cuota de embeddings); si no, se construye desde los PDFs y se guarda.
        """
        from langchain_community.vectorstores import FAISS

        ruta = Path(settings.rag_index_dir)
        if not forzar and (ruta / "index.faiss").exists():
            vs = FAISS.load_local(
                str(ruta), embeddings, allow_dangerous_deserialization=True
            )
            return vs, vs.index.ntotal
        fragmentos = self._cargar_fragmentos(embeddings, docs_dir)
        vs = FAISS.from_documents(fragmentos, embeddings)
        ruta.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(ruta))  # index.faiss + index.pkl
        return vs, len(fragmentos)

    def _vs_pinecone(self, embeddings, docs_dir, forzar):
        """Backend Pinecone: base vectorial en la **nube** (persistente y escalable).

        El índice vive en Pinecone, así que los vectores sobreviven a los reinicios y
        el proyecto puede crecer a muchos documentos sin recalcular todo. Solo se suben
        los fragmentos si el índice está vacío (o si ``forzar`` es True).
        """
        import time

        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec

        if not settings.pinecone_api_key:
            raise ValueError(
                "RAG_VECTORSTORE=pinecone requiere PINECONE_API_KEY "
                "(consíguela en https://app.pinecone.io)."
            )
        pc = Pinecone(api_key=settings.pinecone_api_key)
        nombre = settings.pinecone_index

        # Crea el índice serverless si no existe. La dimensión debe coincidir con el
        # modelo de embeddings (Cohere embed-multilingual-v3.0 = 1024); se calcula
        # embebiendo una cadena de prueba para no hardcodearla.
        existentes = [i["name"] for i in pc.list_indexes()]
        if nombre not in existentes:
            dimension = len(embeddings.embed_query("dimensión"))
            pc.create_index(
                name=nombre,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud, region=settings.pinecone_region
                ),
            )
            # El índice recién creado tarda unos segundos en quedar "listo".
            for _ in range(30):
                estado = pc.describe_index(nombre).status
                listo = getattr(estado, "ready", None)
                if listo is None and hasattr(estado, "get"):
                    listo = estado.get("ready")
                if listo:
                    break
                time.sleep(1)

        index = pc.Index(nombre)
        vs = PineconeVectorStore(index=index, embedding=embeddings)

        # Solo indexa si el índice está vacío o si se fuerza (evita re-subir en cada
        # arranque: los vectores ya persisten en la nube).
        stats = index.describe_index_stats()
        total = getattr(stats, "total_vector_count", None)
        if total is None:
            total = stats.get("total_vector_count", 0) if hasattr(stats, "get") else 0
        if forzar or not total:
            fragmentos = self._cargar_fragmentos(embeddings, docs_dir)
            vs.add_documents(fragmentos)
            return vs, len(fragmentos)
        return vs, total

    def indexar(self, docs_dir: str | None = None, forzar: bool = False) -> int:
        """Indexa los PDFs y arma retriever + cadena, según el backend elegido.

        **Strategy**: ``settings.rag_vectorstore`` selecciona la base vectorial
        (``faiss`` local/persistido o ``pinecone`` en la nube) sin tocar el resto del
        pipeline. Cambiar de una a otra es solo una variable de entorno.
        """
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        embeddings = self._embeddings()
        backends = {"faiss": self._vs_faiss, "pinecone": self._vs_pinecone}
        backend = settings.rag_vectorstore.lower()
        if backend not in backends:
            raise ValueError(
                f"RAG_VECTORSTORE inválido: '{backend}'. Usa 'faiss' o 'pinecone'."
            )
        vectorstore, n_fragmentos = backends[backend](embeddings, docs_dir, forzar)

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
        return n_fragmentos

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
