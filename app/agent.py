"""Orquestación del agente.

Estrategia: **inyección de contexto completo**. Como el documento fuente es
pequeño, en lugar de usar recuperación por embeddings se entrega el texto
completo del PDF como contexto a Gemini en cada pregunta.

Además, el agente puede recibir el **contexto de conversación** de la sesión
(memoria) para "recordar" a un usuario que ya interactuó antes, y expone un
método `resumir()` que la memoria usa para comprimir el historial y no consumir
demasiados tokens.
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
    "honestidad que no cuentas con esa información en la guía. Si hay memoria de una "
    "conversación previa con el usuario, tenla en cuenta para dar continuidad. "
    "Responde en español. Devuelve ÚNICAMENTE la respuesta final para el usuario, "
    "sin notas de proceso, encabezados internos ni texto de razonamiento."
)

SYSTEM_PROMPT_RESUMEN = (
    "Eres un asistente que resume conversaciones. Genera un resumen breve (máximo "
    "120 palabras), en español, que conserve los datos importantes: qué ha preguntado "
    "el usuario, sus intereses y cualquier detalle útil para dar continuidad. "
    "Integra el resumen previo con los intercambios nuevos en un solo texto."
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
        """Carga el texto del documento fuente en memoria."""
        ruta_pdf = pdf_path or settings.pdf_path
        self.contexto = leer_pdf(ruta_pdf)
        self.fuente = os.path.basename(ruta_pdf)
        return len(self.contexto)

    def esta_listo(self) -> bool:
        return bool(self.contexto)

    # ------------------------------------------------------------------ #
    # Consulta
    # ------------------------------------------------------------------ #
    def preguntar(self, pregunta: str, contexto_conversacion: str = "") -> dict:
        """Responde una pregunta usando el documento (y la memoria) como contexto.

        Args:
            pregunta: la pregunta del usuario.
            contexto_conversacion: memoria de la sesión (resumen + últimos turnos).

        Returns:
            {"respuesta": str, "fuente": str}
        """
        if not self.esta_listo():
            raise RuntimeError("El documento no está cargado. Llama a indexar() primero.")

        pregunta = (pregunta or "").strip()
        if not pregunta:
            return {"respuesta": "Por favor, escribe una pregunta.", "fuente": self.fuente}

        bloques = [f"### Documento de referencia:\n{self.contexto}"]
        if contexto_conversacion:
            bloques.append(f"### Memoria de la conversación:\n{contexto_conversacion}")
        bloques.append(f"### Pregunta del usuario:\n{pregunta}")
        mensaje = "\n\n".join(bloques)

        respuesta = self._llamar_modelo(mensaje, SYSTEM_PROMPT, max_tokens=800)
        return {"respuesta": respuesta, "fuente": self.fuente}

    def resumir(self, texto: str, resumen_previo: str = "") -> str:
        """Comprime un tramo de conversación en un resumen breve (para la memoria)."""
        mensaje = (
            f"### Resumen previo:\n{resumen_previo or '(sin resumen previo)'}\n\n"
            f"### Intercambios nuevos a integrar:\n{texto}\n\n"
            f"### Resumen actualizado:"
        )
        return self._llamar_modelo(mensaje, SYSTEM_PROMPT_RESUMEN, max_tokens=300)

    # ------------------------------------------------------------------ #
    # Llamada al modelo (con reintentos)
    # ------------------------------------------------------------------ #
    def _llamar_modelo(
        self, mensaje: str, system_instruction: str, max_tokens: int = 800, intentos: int = 3
    ) -> str:
        """Llama a Gemini reintentando ante errores transitorios."""
        cliente = build_client()
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=max_tokens,
        )

        for intento in range(1, intentos + 1):
            try:
                respuesta = cliente.models.generate_content(
                    model=settings.chat_model, contents=mensaje, config=config
                )
                return (respuesta.text or "").strip()
            except genai_errors.APIError as exc:
                codigo = getattr(exc, "code", None)
                # Un 403 aquí suele ser la API recién habilitada en un proyecto
                # nuevo, aún propagándose entre los servidores de Google.
                es_propagacion = codigo == 403
                if (codigo in _CODIGOS_REINTENTABLES or es_propagacion) and intento < intentos:
                    time.sleep(2 * intento)  # backoff simple: 2s, 4s
                    continue
                raise
