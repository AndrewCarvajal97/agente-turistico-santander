"""Servicio de lectura y escritura de archivos para la memoria de conversaciones.

Guarda cada interacción (pregunta + respuesta) de forma **persistente** en un
archivo, de modo que trascienda el tiempo de ejecución del programa (igual que
el concepto de persistencia en archivos visto en el curso).

Se usa el formato **JSONL** (un objeto JSON por línea) en vez de texto plano
porque las respuestas del agente pueden tener varias líneas; así cada
interacción queda en un único registro fácil de leer y procesar después.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import settings


class ConversationMemory:
    """Lee y escribe el historial de conversaciones en un archivo JSONL."""

    def __init__(self, ruta: str | Path | None = None) -> None:
        self.ruta = Path(ruta or settings.history_path)

    # ------------------------------------------------------------------ #
    # Escritura (persistencia)
    # ------------------------------------------------------------------ #
    def guardar(self, pregunta: str, respuesta: str, fuente: str = "") -> dict:
        """Agrega una interacción al archivo de historial.

        Abre el archivo en modo "a" (append) para NO sobrescribir lo anterior,
        y escribe una línea nueva por cada conversación.
        """
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        registro = {
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "pregunta": pregunta,
            "respuesta": respuesta,
            "fuente": fuente,
        }
        with open(self.ruta, "a", encoding="utf-8") as archivo:
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
        return registro

    # ------------------------------------------------------------------ #
    # Lectura
    # ------------------------------------------------------------------ #
    def leer(self, limite: int | None = None) -> list[dict]:
        """Devuelve las interacciones guardadas (opcionalmente las últimas N)."""
        if not self.ruta.exists():
            return []
        with open(self.ruta, "r", encoding="utf-8") as archivo:
            registros = [json.loads(linea) for linea in archivo if linea.strip()]
        if limite is not None:
            registros = registros[-limite:]
        return registros

    def total(self) -> int:
        """Cantidad de interacciones guardadas."""
        return len(self.leer())

    def limpiar(self) -> None:
        """Elimina el archivo de historial (útil en pruebas)."""
        if self.ruta.exists():
            self.ruta.unlink()
