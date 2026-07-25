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
    """Envía una imagen (base64) a Gemini visión y devuelve su descripción.

    Usa una plantilla multimodal (`ChatPromptTemplate`) donde la imagen y la
    pregunta son variables, y una cadena LCEL con `StrOutputParser` para asegurar
    que la salida sea siempre un string (más robusto y fácil de integrar).
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
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

    # El data-URI completo va como una sola variable ({imagen}): LangChain solo
    # permite una variable de formato por plantilla de imagen.
    template_analisis = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_VISION),
            (
                "user",
                [
                    {"type": "text", "text": "{pregunta}"},
                    {"type": "image_url", "image_url": "{imagen}"},
                ],
            ),
        ]
    )
    cadena_analisis = template_analisis | chat | StrOutputParser()
    salida = cadena_analisis.invoke(
        {"pregunta": texto, "imagen": f"data:{mime};base64,{imagen_b64}"}
    )
    return salida.strip()
