"""Capa de proveedor de LLM (multi-proveedor).

Permite responder con **Google Gemini** o con **Groq** (modelos open source como
Llama o Gemma) según la variable de entorno `LLM_PROVIDER`, sin cambiar el resto
del código. Incluye reintentos ante errores transitorios (cuota momentánea,
servicio recién habilitado, indisponibilidad temporal).
"""
from __future__ import annotations

import time

from .config import settings

# Códigos HTTP transitorios que conviene reintentar.
_TRANSIENT = {429, 500, 502, 503}


def _codigo_http(exc: Exception) -> int | None:
    """Intenta extraer un código HTTP del error de cualquier proveedor."""
    for attr in ("code", "status_code"):
        valor = getattr(exc, attr, None)
        if isinstance(valor, int):
            return valor
    resp = getattr(exc, "response", None)
    if resp is not None and isinstance(getattr(resp, "status_code", None), int):
        return resp.status_code
    return None


# --------------------------------------------------------------------- #
# Proveedores concretos
# --------------------------------------------------------------------- #
def _generar_gemini(mensaje: str, system_instruction: str, max_tokens: int) -> str:
    from google.genai import types

    from .llm_client import build_client

    cliente = build_client()
    respuesta = cliente.models.generate_content(
        model=settings.chat_model,
        contents=mensaje,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=max_tokens,
        ),
    )
    return (respuesta.text or "").strip()


def _generar_groq(mensaje: str, system_instruction: str, max_tokens: int) -> str:
    from groq import Groq

    if not settings.groq_api_key:
        raise ValueError(
            "Falta GROQ_API_KEY. Obtén una gratis en https://console.groq.com/keys"
        )
    cliente = Groq(api_key=settings.groq_api_key)
    # Groq no usa "pensamiento" interno; con 8192 tokens sobra y nunca se trunca.
    tope = min(max_tokens, 8192)
    respuesta = cliente.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": mensaje},
        ],
        temperature=0.2,
        max_tokens=tope,
    )
    return (respuesta.choices[0].message.content or "").strip()


_PROVEEDORES = {"gemini": _generar_gemini, "groq": _generar_groq}


# --------------------------------------------------------------------- #
# Punto de entrada con reintentos
# --------------------------------------------------------------------- #
def generar_texto(
    mensaje: str, system_instruction: str, max_tokens: int, intentos: int = 3
) -> str:
    """Genera texto con el proveedor configurado, reintentando si es transitorio."""
    proveedor = settings.llm_provider.lower()
    generar = _PROVEEDORES.get(proveedor)
    if generar is None:
        raise ValueError(
            f"LLM_PROVIDER inválido: '{settings.llm_provider}'. Usa 'gemini' o 'groq'."
        )

    for intento in range(1, intentos + 1):
        try:
            return generar(mensaje, system_instruction, max_tokens)
        except Exception as exc:  # noqa: BLE001 - se clasifica por código HTTP
            codigo = _codigo_http(exc)
            # Un 403 (Gemini) suele ser la API recién habilitada, aún propagándose.
            transitorio = codigo in _TRANSIENT or codigo == 403
            if transitorio and intento < intentos:
                time.sleep(2 * intento)  # backoff simple: 2s, 4s
                continue
            raise
