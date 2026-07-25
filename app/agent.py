"""Orquestación del agente.

Estrategia: **inyección de contexto completo**. Como el documento fuente es
pequeño, en lugar de usar recuperación por embeddings se entrega el texto
completo del PDF como contexto a Gemini en cada pregunta. Esto es más simple,
gratuito y preciso para documentos de este tamaño.

Los módulos `chunker` y `vector_store` se conservan como utilidades para escalar
a corpus grandes (recuperación semántica / RAG clásico).
"""
from __future__ import annotations

import os
import time

from google.genai import types
from google.genai import errors as genai_errors

from .config import settings
from .llm_client import build_client
from .pdf_loader import leer_pdf

# Códigos HTTP transitorios que conviene reintentar (cuota momentánea, servicio
# recién habilitado y aún propagándose, indisponibilidad temporal).
_CODIGOS_REINTENTABLES = {429, 500, 503}

SYSTEM_PROMPT = (
    "Eres un asistente turístico experto en el departamento de Santander, Colombia. "
    "Responde de forma clara, amable y concisa, ÚNICAMENTE con la información del "
    "documento proporcionado. Si la respuesta no está en el documento, indica con "
    "honestidad que no cuentas con esa información en la guía. Responde en español."
)


class TourismAgent:
    """Agente que responde preguntas sobre la guía turística de Santander."""

    def __init__(self) -> None:
        self.contexto: str | None = None
        self.fuente: str = ""

    # ------------------------------------------------------------------ #
    # Carga del documento
    # ------------------------------------------------------------------ #
    def indexar(self, pdf_path: str | None = None, forzar: bool = False) -> int:
        """Carga el texto del documento fuente en memoria.

        Returns:
            Número de caracteres cargados del documento.
        """
        ruta_pdf = pdf_path or settings.pdf_path
        self.contexto = leer_pdf(ruta_pdf)
        self.fuente = os.path.basename(ruta_pdf)
        return len(self.contexto)

    def esta_listo(self) -> bool:
        return bool(self.contexto)

    # ------------------------------------------------------------------ #
    # Consulta
    # ------------------------------------------------------------------ #
    def preguntar(self, pregunta: str) -> dict:
        """Responde una pregunta usando el documento como contexto.

        Returns:
            {"respuesta": str, "fuente": str}
        """
        if not self.esta_listo():
            raise RuntimeError("El documento no está cargado. Llama a indexar() primero.")

        pregunta = (pregunta or "").strip()
        if not pregunta:
            return {"respuesta": "Por favor, escribe una pregunta.", "fuente": self.fuente}

        respuesta = self._generar(pregunta, self.contexto or "")
        return {"respuesta": respuesta, "fuente": self.fuente}

    def _generar(self, pregunta: str, contexto: str, intentos: int = 3) -> str:
        """Llama al modelo de chat de Gemini con el documento como contexto.

        Reintenta ante errores transitorios (cuota momentánea, API recién
        habilitada y propagándose, o indisponibilidad temporal del servicio).
        """
        cliente = build_client()

        mensaje = (
            f"### Documento de referencia:\n{contexto}\n\n"
            f"### Pregunta del usuario:\n{pregunta}"
        )
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=800,
        )

        for intento in range(1, intentos + 1):
            try:
                respuesta = cliente.models.generate_content(
                    model=settings.chat_model, contents=mensaje, config=config
                )
                return (respuesta.text or "").strip()
            except genai_errors.APIError as exc:
                codigo = getattr(exc, "code", None)
                # Un 403 en este contexto suele ser la API recién habilitada en un
                # proyecto nuevo, aún propagándose entre los servidores de Google.
                es_propagacion = codigo == 403
                if (codigo in _CODIGOS_REINTENTABLES or es_propagacion) and intento < intentos:
                    time.sleep(2 * intento)  # backoff simple: 2s, 4s
                    continue
                raise
