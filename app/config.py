"""Carga y centraliza la configuración del proyecto desde variables de entorno."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Carga las variables definidas en el archivo .env (si existe).
load_dotenv()


def _get(nombre: str, por_defecto: str = "") -> str:
    return os.getenv(nombre, por_defecto).strip()


@dataclass(frozen=True)
class Settings:
    """Configuración inmutable de la aplicación."""

    # Proveedor de LLM: "gemini", "groq" o "cohere"
    llm_provider: str = _get("LLM_PROVIDER", "gemini")

    # --- Google Gemini ---
    gemini_api_key: str = _get("GEMINI_API_KEY")
    # Modelo de chat (alias "latest" para no depender de versiones que se deprecan)
    chat_model: str = _get("GEMINI_CHAT_MODEL", "gemini-flash-latest")
    # Tope de tokens de salida (amplio para que el "pensamiento" del modelo no
    # trunque la respuesta final).
    max_output_tokens: int = int(_get("MAX_OUTPUT_TOKENS", "50000") or 50000)

    # --- Groq (modelos open source: Llama, Gemma) ---
    groq_api_key: str = _get("GROQ_API_KEY")
    groq_model: str = _get("GROQ_MODEL", "llama-3.3-70b-versatile")
    # Modelos de respaldo de Groq (cada uno con su cuota diaria propia). Si el
    # modelo principal se queda sin cupo, se intenta con estos, en orden.
    groq_fallback_models: str = _get(
        "GROQ_FALLBACK_MODELS", "llama-3.1-8b-instant,gemma2-9b-it"
    )

    # --- Cohere (fuerte en contexto en español/latinoamericano) ---
    cohere_api_key: str = _get("COHERE_API_KEY")
    cohere_model: str = _get("COHERE_MODEL", "command-r-08-2024")
    # Modelo de embeddings de Cohere (para el RAG real).
    cohere_embed_model: str = _get("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")

    # --- RAG (recuperación por embeddings + FAISS), vía paralela a /ask ---
    # Proveedor de embeddings: "cohere" (gratis en el trial) o "gemini".
    rag_embed_provider: str = _get("RAG_EMBED_PROVIDER", "cohere")
    rag_chunk_size: int = int(_get("RAG_CHUNK_SIZE", "800") or 800)
    rag_chunk_overlap: int = int(_get("RAG_CHUNK_OVERLAP", "100") or 100)
    rag_top_k: int = int(_get("RAG_TOP_K", "4") or 4)
    # Umbral mínimo de similitud (0-1) para considerar un fragmento relevante.
    rag_score_threshold: float = float(_get("RAG_SCORE_THRESHOLD", "0.3") or 0.3)

    # Documento fuente
    pdf_path: str = _get("PDF_PATH", "data/guia_turistica_santander.pdf")

    # Memoria de conversaciones: un único CSV para todas las sesiones.
    history_csv: str = _get("HISTORY_CSV", "data/historial.csv")
    # Cuántos intercambios recientes de la sesión se usan como contexto.
    memory_max_turns: int = int(_get("MEMORY_MAX_TURNS", "6") or 6)

    # Clave de administrador para acciones protegidas (p. ej. análisis de datos).
    admin_key: str = _get("ADMIN_KEY")

    def validar(self) -> None:
        """Lanza un error claro si falta la API key del proveedor activo."""
        prov = self.llm_provider.lower()
        claves = {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "cohere": self.cohere_api_key,
        }
        urls = {
            "groq": "https://console.groq.com/keys",
            "gemini": "https://aistudio.google.com/app/apikey",
            "cohere": "https://dashboard.cohere.com/api-keys",
        }
        if prov not in claves:
            raise ValueError(
                f"LLM_PROVIDER inválido: '{self.llm_provider}'. Usa 'groq', 'gemini' o 'cohere'."
            )
        if not claves[prov]:
            raise ValueError(
                f"Falta la API key para LLM_PROVIDER={prov}. Consíguela en {urls[prov]}"
            )


settings = Settings()
