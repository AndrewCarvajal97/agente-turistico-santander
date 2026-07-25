"""Análisis de imágenes (visión) con Gemini vía LangChain.

Como se vio en el curso, **no todos los LLM soportan imágenes**: en esta
configuración solo **Gemini** hace visión (Groq/Cohere son de texto). Por eso este
módulo usa específicamente `ChatGoogleGenerativeAI`.

La imagen se codifica en **base64** (como `encode_image` del curso) y se envía en
un mensaje **multimodal** (`HumanMessage` con un bloque de texto y uno de imagen).
"""
from __future__ import annotations

import base64
from pathlib import Path

from .config import settings

SYSTEM_VISION = (
    "Eres un guía turístico experto en el departamento de Santander, Colombia. "
    "Describe la imagen de forma clara y amable, en español. Si reconoces un lugar, "
    "un plato típico o una actividad relacionada con el turismo de Santander, "
    "identifícalo y añade un dato útil. Si no tiene relación con Santander, descríbela "
    "igual con honestidad."
)


def encode_image(ruta: str | Path) -> str:
    """Lee una imagen y la devuelve codificada en base64 (texto)."""
    with open(ruta, "rb") as archivo:
        return base64.b64encode(archivo.read()).decode("utf-8")


def analizar_imagen(imagen_b64: str, mime: str = "image/jpeg", pregunta: str = "") -> str:
    """Envía una imagen (base64) a Gemini visión y devuelve su descripción."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.gemini_api_key:
        raise ValueError("Falta GEMINI_API_KEY para el análisis de imágenes.")

    chat = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
        max_output_tokens=settings.max_output_tokens,
    )
    texto = pregunta.strip() or (
        "¿Qué muestra esta imagen? Si es un lugar, plato o actividad de Santander, "
        "identifícalo."
    )
    mensaje = HumanMessage(
        content=[
            {"type": "text", "text": texto},
            {"type": "image_url", "image_url": f"data:{mime};base64,{imagen_b64}"},
        ]
    )
    respuesta = chat.invoke([SystemMessage(content=SYSTEM_VISION), mensaje])
    return _extraer_texto(respuesta.content)


def _extraer_texto(contenido) -> str:
    """Extrae el texto de la respuesta, que puede venir como str o lista de bloques."""
    if isinstance(contenido, str):
        return contenido.strip()
    if isinstance(contenido, list):
        partes = [
            bloque.get("text", "") if isinstance(bloque, dict) else str(bloque)
            for bloque in contenido
        ]
        return "".join(partes).strip()
    return str(contenido or "").strip()
