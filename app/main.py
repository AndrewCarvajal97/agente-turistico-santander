"""API FastAPI del agente turístico de Santander.

Endpoints:
  GET  /            -> interfaz web mínima (chat)
  GET  /health      -> estado del servicio
  POST /ask         -> responde una pregunta (con memoria por sesión)
  GET  /history     -> lista las sesiones guardadas (memoria)
  POST /reload      -> recarga el documento fuente
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import TourismAgent
from .config import settings
from .memory import ConversationMemory

agent = TourismAgent()
memory = ConversationMemory()

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
    # Identificador de sesión generado por el frontend (sin registro). Permite
    # "recordar" a un usuario que ya interactuó antes.
    session_id: str = Field(default="", examples=["sess-1a2b3c"])


class RespuestaOut(BaseModel):
    respuesta: str
    fuente: str
    recurrente: bool = False


# ------------------------------- Endpoints ------------------------------ #
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documento_cargado": agent.esta_listo()}


@app.post("/ask", response_model=RespuestaOut)
def ask(entrada: PreguntaIn, request: Request) -> RespuestaOut:
    if not agent.esta_listo():
        raise HTTPException(
            status_code=503,
            detail="El documento no está cargado. Verifica tu GEMINI_API_KEY y usa /reload.",
        )

    session_id = entrada.session_id.strip()
    ip = request.client.host if request.client else ""

    # Memoria: ¿ya conocíamos a este usuario? y contexto de conversación previo.
    recurrente = memory.es_recurrente(session_id) if session_id else False
    contexto_conv = memory.construir_contexto(session_id) if session_id else ""

    try:
        resultado = agent.preguntar(entrada.pregunta, contexto_conversacion=contexto_conv)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error al generar respuesta: {exc}")

    # Persistimos el intercambio en el CSV de memoria. No rompe la respuesta si
    # el guardado falla.
    if session_id:
        try:
            memory.guardar_turno(session_id, entrada.pregunta, resultado["respuesta"], ip=ip)
        except Exception as exc:  # noqa: BLE001
            print(f"[memory] No se pudo guardar la conversación -> {exc}")

    return RespuestaOut(**resultado, recurrente=recurrente)


@app.get("/history")
def history(session_id: str = "", q: str = "") -> dict:
    """Memoria (CSV):
    - `q`: busca un término en preguntas/respuestas (filtro pandas).
    - `session_id`: devuelve el historial de esa sesión.
    - sin parámetros: lista un resumen de todas las sesiones.
    """
    if q:
        resultados = memory.buscar(q)
        return {"busqueda": q, "coincidencias": len(resultados), "resultados": resultados}
    if session_id:
        turnos = memory.historial_sesion(session_id).to_dict(orient="records")
        return {"session_id": session_id, "turnos": turnos}
    sesiones = memory.listar_sesiones()
    return {"total_sesiones": len(sesiones), "sesiones": sesiones}


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
