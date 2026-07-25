"""API FastAPI del agente turístico de Santander.

Endpoints:
  GET  /            -> interfaz web mínima (chat)
  GET  /health      -> estado del servicio
  POST /ask         -> responde una pregunta usando el documento como contexto
  POST /reload      -> recarga el documento fuente
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import TourismAgent
from .config import settings

agent = TourismAgent()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el documento fuente al iniciar el servidor."""
    try:
        settings.validar()
        n = agent.indexar()
        print(f"[startup] Documento cargado ({n} caracteres).")
    except Exception as exc:  # noqa: BLE001 - se registra para diagnóstico
        print(f"[startup] Advertencia: no se pudo cargar el documento -> {exc}")
    yield


app = FastAPI(
    title="Agente Turístico de Santander",
    description="Agente IA que responde preguntas sobre turismo en Santander, "
    "Colombia, usando Google Gemini con el documento como contexto.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------- Modelos de datos --------------------------- #
class PreguntaIn(BaseModel):
    pregunta: str = Field(..., min_length=1, examples=["¿Dónde puedo practicar rafting?"])


class RespuestaOut(BaseModel):
    respuesta: str
    fuente: str


# ------------------------------- Endpoints ------------------------------ #
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documento_cargado": agent.esta_listo()}


@app.post("/ask", response_model=RespuestaOut)
def ask(entrada: PreguntaIn) -> RespuestaOut:
    if not agent.esta_listo():
        raise HTTPException(
            status_code=503,
            detail="El documento no está cargado. Verifica tu GEMINI_API_KEY y usa /reload.",
        )
    try:
        resultado = agent.preguntar(entrada.pregunta)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error al generar respuesta: {exc}")
    return RespuestaOut(**resultado)


@app.post("/reload")
def reload() -> dict:
    try:
        n = agent.indexar(forzar=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error al cargar el documento: {exc}")
    return {"status": "ok", "caracteres": n}


@app.get("/")
def index():
    archivo = STATIC_DIR / "index.html"
    if archivo.exists():
        return FileResponse(archivo)
    return {"mensaje": "Agente Turístico de Santander. Usa POST /ask."}


# Sirve archivos estáticos (CSS/JS futuros) bajo /static.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
