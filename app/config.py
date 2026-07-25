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

    # Proveedor de LLM: "gemini" o "groq"
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

    # Documento fuente
    pdf_path: str = _get("PDF_PATH", "data/guia_turistica_santander.pdf")

    # Memoria de conversaciones: un único CSV para todas las sesiones.
    history_csv: str = _get("HISTORY_CSV", "data/historial.csv")
    # Cuántos intercambios recientes de la sesión se usan como contexto.
    memory_max_turns: int = int(_get("MEMORY_MAX_TURNS", "6") or 6)

    # Clave de administrador para acciones protegidas (p. ej. análisis de datos).
    admin_key: str = _get("ADMIN_KEY")

    def validar(self) -> None:
        """Lanza un error claro si falta configuración esencial del proveedor activo."""
        prov = self.llm_provider.lower()
        if prov == "groq" and not self.groq_api_key:
            raise ValueError(
                "Falta GROQ_API_KEY (LLM_PROVIDER=groq). Obtén una gratis en "
                "https://console.groq.com/keys"
            )
        if prov == "gemini" and not self.gemini_api_key:
            raise ValueError(
                "Falta GEMINI_API_KEY (LLM_PROVIDER=gemini). Copia .env.example a .env "
                "y complétalo con tu clave de https://aistudio.google.com/app/apikey"
            )


settings = Settings()
