"""API FastAPI del agente turístico de Santander.

Endpoints:
  GET  /               -> interfaz web mínima (chat)
  GET  /health         -> estado del servicio
  POST /ask            -> responde una pregunta (contexto completo + memoria)
  POST /rag/ask        -> responde con RAG (embeddings + FAISS, vía paralela)
  POST /vision         -> analiza una imagen (Gemini visión)
  GET  /history        -> lista/busca las conversaciones guardadas
  POST /admin/analisis -> categoriza las preguntas (admin, requiere clave)
  POST /agente         -> agente orquestador ReAct (LangGraph, paralelo)
  POST /grafo/ask      -> agente con grafo de estados (triaje + RAG, paralelo)
  POST /reload         -> recarga el documento fuente
"""
from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import analytics, llm, vision
from .agent import TourismAgent
from .config import settings
from .memory import ConversationMemory

# Tamaño máximo de imagen aceptado en /vision (5 MB).
_MAX_IMAGEN = 5 * 1024 * 1024

agent = TourismAgent()
memory = ConversationMemory()

# Mensajes amables ante fallos, para que el usuario nunca vea un crash.
MSG_SIN_CUPO = (
    "Estoy recibiendo muchas consultas en este momento y alcancé el límite temporal "
    "del servicio de IA. 😅 Por favor, intenta de nuevo en unos segundos."
)
MSG_ERROR = (
    "Ups, tuve un problema para generar la respuesta. Por favor, intenta de nuevo "
    "en un momento. 🙏"
)

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
    description="Agente IA sobre turismo en Santander, Colombia. Usa LangChain con "
    "múltiples proveedores de LLM (Gemini, Groq, Cohere), visión con Gemini y un "
    "agente orquestador con herramientas.",
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


class AdminIn(BaseModel):
    clave: str = Field(..., description="Clave de administrador")


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

    # Manejo de errores (patrón try/except con casos distintos):
    #  - SinCupoError: todos los proveedores/modelos agotaron su cuota.
    #  - Cualquier otro error inesperado.
    # En ambos casos respondemos con un mensaje amable en vez de romper la app.
    ok = True
    try:
        resultado = agent.preguntar(entrada.pregunta, contexto_conversacion=contexto_conv)
    except llm.SinCupoError as exc:
        print(f"[ask] sin cupo en todos los proveedores -> {exc}")
        resultado = {"respuesta": MSG_SIN_CUPO, "fuente": agent.fuente}
        ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"[ask] error inesperado -> {exc}")
        resultado = {"respuesta": MSG_ERROR, "fuente": agent.fuente}
        ok = False

    # Guardamos en la memoria solo las respuestas reales (no los mensajes de error).
    if session_id and ok:
        try:
            memory.guardar_turno(session_id, entrada.pregunta, resultado["respuesta"], ip=ip)
        except Exception as exc:  # noqa: BLE001
            print(f"[memory] No se pudo guardar la conversación -> {exc}")

    return RespuestaOut(
        respuesta=resultado["respuesta"], fuente=resultado["fuente"], recurrente=recurrente
    )


@app.post("/vision")
async def vision_endpoint(file: UploadFile = File(...), pregunta: str = Form("")) -> dict:
    """Analiza una imagen con Gemini visión (identifica lugares/platos de Santander)."""
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503, detail="Análisis de imágenes no disponible: falta GEMINI_API_KEY."
        )
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    datos = await file.read()
    if len(datos) > _MAX_IMAGEN:
        raise HTTPException(status_code=413, detail="La imagen supera el tamaño máximo (5 MB).")

    try:
        b64 = base64.b64encode(datos).decode("utf-8")
        return vision.analizar_imagen(b64, file.content_type, pregunta)
    except Exception as exc:  # noqa: BLE001 - degradación amable (visión solo en Gemini)
        print(f"[vision] error -> {exc}")
        return {
            "titulo": "",
            "descripcion": (
                "No pude analizar la imagen en este momento (la visión usa Gemini y "
                "quizá se alcanzó su límite). Intenta de nuevo en un rato. 🙏"
            ),
            "etiquetas": [],
            "tipo": "otro",
            "relacion_santander": "",
        }


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


@app.post("/admin/analisis")
def admin_analisis(entrada: AdminIn) -> dict:
    """Acción de administrador: categoriza las preguntas guardadas (pandas + LLM + JSON)."""
    if not settings.admin_key:
        raise HTTPException(status_code=503, detail="Análisis no disponible: falta ADMIN_KEY.")
    if entrada.clave != settings.admin_key:
        raise HTTPException(status_code=401, detail="Clave de administrador incorrecta.")
    try:
        return analytics.analizar(memory)
    except llm.SinCupoError:
        raise HTTPException(status_code=503, detail=MSG_SIN_CUPO)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error al analizar: {exc}")


@app.post("/rag/ask")
def rag_ask(entrada: PreguntaIn) -> dict:
    """RAG real: recupera los chunks más relevantes (embeddings + FAISS) y responde.

    Vía PARALELA a /ask (que usa contexto completo). Útil para documentos grandes.
    Usa embeddings de Cohere, así que requiere COHERE_API_KEY.
    """
    try:
        from .rag import rag  # import perezoso (solo si se usa el RAG)

        return rag.preguntar(entrada.pregunta)
    except llm.SinCupoError:
        raise HTTPException(status_code=503, detail=MSG_SIN_CUPO)
    except Exception as exc:  # noqa: BLE001
        print(f"[rag] error -> {exc}")
        raise HTTPException(
            status_code=503,
            detail="El RAG no está disponible ahora (falta COHERE_API_KEY o límite de cuota).",
        )


@app.post("/agente")
def agente_endpoint(entrada: PreguntaIn) -> dict:
    """Agente orquestador ReAct (LangGraph): decide qué herramienta usar.

    Implementación PARALELA a /ask y /vision. Consume más tokens (razona + actúa),
    así que está pensada para demostración y para crecer con más herramientas.
    """
    try:
        from . import orchestrator  # import perezoso (solo si se usa el agente)

        return orchestrator.responder(entrada.pregunta)
    except Exception as exc:  # noqa: BLE001
        print(f"[agente] error -> {exc}")
        raise HTTPException(
            status_code=503,
            detail="El agente orquestador no está disponible ahora (posible límite de "
            "cuota del LLM). Intenta más tarde.",
        )


@app.post("/grafo/ask")
def grafo_ask(entrada: PreguntaIn) -> dict:
    """Agente con grafo de estados (LangGraph): triaje → RAG / pedir info / ticket.

    Vía PARALELA. El triaje clasifica la consulta y el grafo la enruta al nodo
    adecuado. Devuelve {respuesta, decision, accion_final, citaciones}.
    """
    try:
        from . import graph  # import perezoso (solo si se usa el grafo)

        return graph.responder(entrada.pregunta)
    except llm.SinCupoError:
        raise HTTPException(status_code=503, detail=MSG_SIN_CUPO)
    except Exception as exc:  # noqa: BLE001
        print(f"[grafo] error -> {exc}")
        raise HTTPException(
            status_code=503,
            detail="El agente con grafo no está disponible ahora (límite de cuota o config).",
        )


@app.get("/grafo/diagrama")
def grafo_diagrama() -> dict:
    """Devuelve el grafo de estados en sintaxis Mermaid (para visualizarlo)."""
    try:
        from . import graph

        return {"mermaid": graph.diagrama_mermaid()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"No se pudo generar el diagrama: {exc}")


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
