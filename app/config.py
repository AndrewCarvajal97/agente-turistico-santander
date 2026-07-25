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

    # Modelo de chat (alias "latest" para no depender de versiones que se deprecan)
    chat_model: str = _get("GEMINI_CHAT_MODEL", "gemini-flash-latest")
    # Tope de tokens de salida (amplio para que el "pensamiento" del modelo no
    # trunque la respuesta final).
    max_output_tokens: int = int(_get("MAX_OUTPUT_TOKENS", "50000") or 50000)

    # Documento fuente
    pdf_path: str = _get("PDF_PATH", "data/guia_turistica_santander.pdf")

    # Memoria de conversaciones por sesión (persistencia en archivos)
    memory_dir: str = _get("MEMORY_DIR", "data/memoria")
    # Tamaño (en caracteres) del bloque reciente antes de resumir la memoria.
    memory_max_chars: int = int(_get("MEMORY_MAX_CHARS", "2500") or 2500)
    # Mínimo de intercambios recientes que se conservan siempre de forma literal.
    memory_min_turns: int = int(_get("MEMORY_MIN_TURNS", "3") or 3)

    def validar(self) -> None:
        """Lanza un error claro si falta configuración esencial."""
        if not self.gemini_api_key:
            raise ValueError(
                "Falta GEMINI_API_KEY. Copia .env.example a .env y complétalo "
                "con tu clave de https://aistudio.google.com/app/apikey"
            )


settings = Settings()
