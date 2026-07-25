"""Análisis de imágenes (visión) con Gemini vía LangChain — salida estructurada.

Aplica LCEL: una cadena `plantilla | modelo | parser`. Aquí el parser es un
`JsonOutputParser` validado con un modelo **Pydantic**, de modo que la salida sea
un **JSON estructurado** (descripción + etiquetas + relación con Santander), más
fácil de integrar en una app que un texto libre.

Como se vio en el curso, **solo Gemini** hace visión en esta configuración
(Groq/Cohere son de texto), así que se usa `ChatGoogleGenerativeAI`. La imagen se
codifica en base64 y se envía en un mensaje multimodal.
"""
from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel, Field

from .config import settings

SYSTEM_VISION = (
    "Eres un guía turístico experto en el departamento de Santander, Colombia. "
    "Analiza la imagen de forma objetiva y, si reconoces un lugar, plato típico o "
    "actividad relacionada con el turismo de Santander, indícalo. Responde en español."
)


class AnalisisImagen(BaseModel):
    """Estructura de salida del análisis de una imagen."""

    descripcion: str = Field(description="Descripción clara y objetiva de la imagen, en español")
    etiquetas: list[str] = Field(description="Entre 3 y 5 palabras clave en minúsculas, sin tildes")
    relacion_santander: str = Field(
        description="Relación con el turismo de Santander (lugar, plato o actividad), o 'ninguna'"
    )


def encode_image(ruta: str | Path) -> str:
    """Lee una imagen y la devuelve codificada en base64 (texto)."""
    with open(ruta, "rb") as archivo:
        return base64.b64encode(archivo.read()).decode("utf-8")


def _modelo():
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.gemini_api_key:
        raise ValueError("Falta GEMINI_API_KEY para el análisis de imágenes.")
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
        max_output_tokens=settings.max_output_tokens,
    )


def analizar_imagen(imagen_b64: str, mime: str = "image/jpeg", pregunta: str = "") -> dict:
    """Analiza una imagen y devuelve un dict {descripcion, etiquetas, relacion_santander}."""
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    chat = _modelo()
    data_uri = f"data:{mime};base64,{imagen_b64}"
    texto = pregunta.strip() or "Analiza esta imagen."

    # Cadena LCEL con salida estructurada (JSON validado por el modelo Pydantic).
    parser = JsonOutputParser(pydantic_object=AnalisisImagen)
    template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_VISION + "\n\n# FORMATO DE SALIDA (JSON)\n{formato}"),
            (
                "user",
                [
                    {"type": "text", "text": "{pregunta}"},
                    {"type": "image_url", "image_url": "{imagen}"},
                ],
            ),
        ]
    )
    cadena = template | chat | parser
    try:
        return cadena.invoke(
            {
                "pregunta": texto,
                "imagen": data_uri,
                "formato": parser.get_format_instructions(),
            }
        )
    except Exception as exc:  # noqa: BLE001 - respaldo a texto plano si el JSON falla
        print(f"[vision] salida JSON falló ({exc}); uso texto plano.")
        template_txt = ChatPromptTemplate.from_messages(
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
        texto_plano = (template_txt | chat | StrOutputParser()).invoke(
            {"pregunta": texto, "imagen": data_uri}
        )
        return {"descripcion": texto_plano.strip(), "etiquetas": [], "relacion_santander": ""}
