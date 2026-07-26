"""Orquestación del agente de preguntas y respuestas.

Estrategia: **inyección de contexto completo**. Como el documento fuente es
pequeño, en lugar de usar recuperación por embeddings se entrega el texto
completo del PDF como contexto al LLM en cada pregunta.

Además, el agente puede recibir el **contexto de conversación** de la sesión
(memoria) para "recordar" a un usuario que ya interactuó antes. La llamada al
modelo se delega a `app.llm`, que elige el proveedor y aplica el respaldo.
"""
from __future__ import annotations

import os

from . import llm
from .config import settings
from .pdf_loader import leer_pdf

SYSTEM_PROMPT = (
    "Eres un asistente turístico experto en el departamento de Santander, Colombia. "
    "Responde de forma clara, amable y concisa, ÚNICAMENTE con la información del "
    "documento proporcionado. Si la respuesta no está en el documento, indica con "
    "honestidad que no cuentas con esa información en la guía. Si hay memoria de una "
    "conversación previa con el usuario, tenla en cuenta para dar continuidad. "
    "Responde en español. Devuelve ÚNICAMENTE la respuesta final para el usuario, "
    "sin notas de proceso, encabezados internos ni texto de razonamiento."
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
            contexto_conversacion: memoria de la sesión (últimos intercambios).

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

        respuesta = self._llamar_modelo(
            mensaje, SYSTEM_PROMPT, max_tokens=settings.max_output_tokens
        )
        return {"respuesta": respuesta, "fuente": self.fuente}

    # ------------------------------------------------------------------ #
    # Llamada al modelo (con reintentos)
    # ------------------------------------------------------------------ #
    def _llamar_modelo(
        self, mensaje: str, system_instruction: str, max_tokens: int = 2048
    ) -> str:
        """Genera texto con el proveedor de LLM configurado (Gemini, Groq o Cohere)."""
        return llm.generar_texto(mensaje, system_instruction, max_tokens)
