"""Servicio de memoria de conversaciones por sesión (persistencia en archivos).

Cada usuario se identifica por un `session_id` (generado en el frontend y
guardado en su navegador, sin necesidad de registro). La memoria de cada sesión
se guarda en su propio archivo JSON dentro de `data/memoria/`.

Para no consumir demasiados tokens de Gemini, la memoria funciona como un
**buffer con resumen**: se conservan los últimos intercambios de forma literal y,
cuando el historial reciente supera un límite de tamaño, los intercambios más
antiguos se comprimen en un "resumen" (usando un resumidor externo inyectado).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import settings

# Firma del resumidor: (texto_a_resumir, resumen_previo) -> nuevo_resumen
Resumidor = Callable[[str, str], str]


def _slug(session_id: str) -> str:
    """Convierte el session_id en un nombre de archivo seguro."""
    limpio = re.sub(r"[^A-Za-z0-9_-]", "", session_id or "")[:64]
    return limpio or "anonimo"


def _turno_a_texto(turno: dict) -> str:
    return f"Usuario: {turno['pregunta']}\nAsistente: {turno['respuesta']}"


class ConversationMemory:
    """Lee y escribe la memoria de conversaciones, una sesión por archivo."""

    def __init__(
        self,
        dir_base: str | Path | None = None,
        max_chars: int | None = None,
        turnos_min: int | None = None,
    ) -> None:
        self.dir_base = Path(dir_base or settings.memory_dir)
        self.max_chars = max_chars or settings.memory_max_chars
        self.turnos_min = turnos_min or settings.memory_min_turns

    # ------------------------------------------------------------------ #
    # Utilidades de archivo
    # ------------------------------------------------------------------ #
    def _ruta(self, session_id: str) -> Path:
        return self.dir_base / f"{_slug(session_id)}.json"

    def cargar(self, session_id: str) -> dict:
        """Devuelve el estado de memoria de la sesión (o una estructura vacía)."""
        ruta = self._ruta(session_id)
        if not ruta.exists():
            return {
                "session_id": session_id,
                "ip": "",
                "creado": "",
                "actualizado": "",
                "resumen": "",
                "turnos": [],
            }
        with open(ruta, "r", encoding="utf-8") as archivo:
            return json.load(archivo)

    def _escribir(self, session_id: str, data: dict) -> None:
        self.dir_base.mkdir(parents=True, exist_ok=True)
        with open(self._ruta(session_id), "w", encoding="utf-8") as archivo:
            json.dump(data, archivo, ensure_ascii=False, indent=2)

    def existe(self, session_id: str) -> bool:
        return self._ruta(session_id).exists()

    # ------------------------------------------------------------------ #
    # Escritura de un turno (con resumen automático)
    # ------------------------------------------------------------------ #
    def guardar_turno(
        self,
        session_id: str,
        pregunta: str,
        respuesta: str,
        ip: str = "",
        resumidor: Resumidor | None = None,
    ) -> dict:
        """Agrega un intercambio y resume la memoria si supera el límite."""
        ahora = datetime.now().isoformat(timespec="seconds")
        data = self.cargar(session_id)
        if not data["creado"]:
            data["creado"] = ahora
        if ip:
            data["ip"] = ip
        data["actualizado"] = ahora
        data["turnos"].append({"fecha": ahora, "pregunta": pregunta, "respuesta": respuesta})

        # Si el bloque reciente pesa demasiado, comprimimos los turnos más
        # antiguos en el resumen y conservamos solo los últimos `turnos_min`.
        texto_reciente = "\n".join(_turno_a_texto(t) for t in data["turnos"])
        if len(texto_reciente) > self.max_chars and len(data["turnos"]) > self.turnos_min:
            a_resumir = data["turnos"][: -self.turnos_min]
            data["turnos"] = data["turnos"][-self.turnos_min :]
            texto_evict = "\n".join(_turno_a_texto(t) for t in a_resumir)
            if resumidor is not None:
                data["resumen"] = resumidor(texto_evict, data.get("resumen", ""))
            else:
                # Sin resumidor: fallback simple para mantener el peso acotado.
                base = (data.get("resumen", "") + " " + texto_evict).strip()
                data["resumen"] = base[-self.max_chars :]

        self._escribir(session_id, data)
        return data

    # ------------------------------------------------------------------ #
    # Lectura / construcción de contexto
    # ------------------------------------------------------------------ #
    def construir_contexto(self, session_id: str) -> str:
        """Arma el texto de memoria (resumen + últimos turnos) para el prompt."""
        if not session_id:
            return ""
        data = self.cargar(session_id)
        partes: list[str] = []
        if data.get("resumen"):
            partes.append("Resumen de la conversación previa con este usuario:\n" + data["resumen"])
        if data.get("turnos"):
            ultimos = "\n".join(_turno_a_texto(t) for t in data["turnos"])
            partes.append("Últimos intercambios con este usuario:\n" + ultimos)
        return "\n\n".join(partes)

    def es_recurrente(self, session_id: str) -> bool:
        """True si la sesión ya tenía interacciones previas guardadas."""
        if not self.existe(session_id):
            return False
        data = self.cargar(session_id)
        return bool(data.get("turnos") or data.get("resumen"))

    def listar_sesiones(self) -> list[dict]:
        """Resumen de todas las sesiones guardadas (para inspección/administración)."""
        if not self.dir_base.exists():
            return []
        sesiones = []
        for ruta in sorted(self.dir_base.glob("*.json")):
            with open(ruta, "r", encoding="utf-8") as archivo:
                data = json.load(archivo)
            sesiones.append(
                {
                    "session_id": data.get("session_id"),
                    "ip": data.get("ip", ""),
                    "actualizado": data.get("actualizado", ""),
                    "turnos_recientes": len(data.get("turnos", [])),
                    "tiene_resumen": bool(data.get("resumen")),
                }
            )
        return sesiones
