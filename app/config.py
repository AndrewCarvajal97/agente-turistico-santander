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

    # Proveedor de IA: Google Gemini
    gemini_api_key: str = _get("GEMINI_API_KEY")

    # Modelos
    chat_model: str = _get("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
    embed_model: str = _get("GEMINI_EMBED_MODEL", "text-embedding-004")

    # Documento y parámetros de RAG
    pdf_path: str = _get("PDF_PATH", "data/guia_turistica_santander.pdf")
    index_path: str = _get("INDEX_PATH", "data/index.npz")
    top_k: int = int(_get("TOP_K", "4") or 4)
    chunk_size: int = int(_get("CHUNK_SIZE", "900") or 900)
    chunk_overlap: int = int(_get("CHUNK_OVERLAP", "150") or 150)

    def validar(self) -> None:
        """Lanza un error claro si falta configuración esencial."""
        if not self.gemini_api_key:
            raise ValueError(
                "Falta GEMINI_API_KEY. Copia .env.example a .env y complétalo "
                "con tu clave de https://aistudio.google.com/app/apikey"
            )


settings = Settings()
