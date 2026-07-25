"""Capa de proveedor de LLM (multi-proveedor con fallback en cadena).

Permite responder con **Google Gemini** o con **Groq** (modelos open source como
Llama o Gemma) según `LLM_PROVIDER`. Además, aplica una **estrategia de respaldo**:
si un modelo/proveedor se queda sin cupo (error 429) o falla, se intenta con el
siguiente candidato de la cadena, sin romper la experiencia del usuario:

    [modelo Groq configurado] -> [otro modelo Groq] -> [Gemini] -> (sin cupo)

Cada modelo de Groq tiene su propia cuota diaria, así que probar otro modelo suele
resolver el agotamiento de tokens. Si TODOS los candidatos fallan, se lanza
`SinCupoError`, que la aplicación traduce en un mensaje amable.
"""
from __future__ import annotations

import time

from .config import settings

# Códigos HTTP transitorios del servidor que conviene reintentar en el mismo modelo.
_TRANSIENT = {500, 502, 503}


class SinCupoError(Exception):
    """Todos los proveedores/modelos disponibles fallaron o agotaron su cuota."""


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
# Proveedores concretos (reciben el modelo a usar)
# --------------------------------------------------------------------- #
def _generar_gemini(mensaje: str, system_instruction: str, max_tokens: int, modelo: str) -> str:
    from google.genai import types

    from .llm_client import build_client

    cliente = build_client()
    respuesta = cliente.models.generate_content(
        model=modelo,
        contents=mensaje,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=max_tokens,
        ),
    )
    return (respuesta.text or "").strip()


def _generar_groq(mensaje: str, system_instruction: str, max_tokens: int, modelo: str) -> str:
    from groq import Groq

    if not settings.groq_api_key:
        raise ValueError("Falta GROQ_API_KEY.")
    cliente = Groq(api_key=settings.groq_api_key)
    # Groq no usa "pensamiento" interno; con ~1024 tokens sobra y no se trunca.
    # Mantenerlo bajo evita superar el límite de tokens por minuto del free tier.
    tope = min(max_tokens, 1024)
    respuesta = cliente.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": mensaje},
        ],
        temperature=0.2,
        max_tokens=tope,
    )
    return (respuesta.choices[0].message.content or "").strip()


_GENERADORES = {"gemini": _generar_gemini, "groq": _generar_groq}


# --------------------------------------------------------------------- #
# Construcción de la cadena de intentos (estrategia de respaldo)
# --------------------------------------------------------------------- #
def _modelos_groq() -> list[str]:
    """Modelo de Groq configurado + alternativas (cada una con su cuota diaria)."""
    modelos = [settings.groq_model]
    for m in settings.groq_fallback_models.split(","):
        m = m.strip()
        if m and m not in modelos:
            modelos.append(m)
    return modelos


def _cadena_intentos() -> list[tuple[str, str]]:
    """Lista ordenada de (proveedor, modelo) a intentar según el proveedor activo."""
    prov = settings.llm_provider.lower()
    cadena: list[tuple[str, str]] = []

    if prov == "groq":
        cadena += [("groq", m) for m in _modelos_groq()]
        if settings.gemini_api_key:  # respaldo cruzado
            cadena.append(("gemini", settings.chat_model))
    else:  # gemini
        cadena.append(("gemini", settings.chat_model))
        if settings.groq_api_key:  # respaldo cruzado
            cadena += [("groq", m) for m in _modelos_groq()]

    return cadena


# --------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------- #
def _intentar_modelo(generar, mensaje, system_instruction, max_tokens, modelo, reintentos=2):
    """Intenta un modelo, reintentando solo ante errores transitorios del servidor."""
    for intento in range(1, reintentos + 1):
        try:
            return generar(mensaje, system_instruction, max_tokens, modelo)
        except Exception as exc:  # noqa: BLE001
            codigo = _codigo_http(exc)
            if codigo in _TRANSIENT and intento < reintentos:
                time.sleep(1.5 * intento)
                continue
            raise  # 429/4xx u otro: se maneja arriba pasando al siguiente candidato


def generar_texto(mensaje: str, system_instruction: str, max_tokens: int) -> str:
    """Genera texto probando la cadena de proveedores/modelos hasta que uno responda."""
    errores: list[str] = []
    for proveedor, modelo in _cadena_intentos():
        generar = _GENERADORES.get(proveedor)
        if generar is None:
            continue
        try:
            return _intentar_modelo(generar, mensaje, system_instruction, max_tokens, modelo)
        except Exception as exc:  # noqa: BLE001 - se registra y se pasa al siguiente
            codigo = _codigo_http(exc)
            etiqueta = "sin cupo (429)" if codigo == 429 else f"error {codigo or '?'}"
            errores.append(f"{proveedor}/{modelo}: {etiqueta}")
            continue

    raise SinCupoError(" | ".join(errores) or "no hay proveedores configurados")
