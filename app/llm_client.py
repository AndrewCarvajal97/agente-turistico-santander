"""Construye el cliente de Google Gemini (SDK google-genai).

La autenticación es simplemente una API key obtenida en:
    https://aistudio.google.com/app/apikey
"""
from __future__ import annotations

from google import genai

from .config import settings


def build_client() -> "genai.Client":
    """Devuelve un cliente de Gemini autenticado con la API key."""
    if not settings.gemini_api_key:
        raise ValueError(
            "Falta GEMINI_API_KEY. Define la variable de entorno o complétala en .env."
        )
    return genai.Client(api_key=settings.gemini_api_key)
