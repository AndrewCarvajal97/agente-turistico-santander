"""Tests unitarios que NO requieren conexión a la API de Gemini.

Validan las piezas locales: lectura del PDF y la carga del documento en el agente.

Ejecuta con:  py -m pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import TourismAgent  # noqa: E402
from app.memory import ConversationMemory  # noqa: E402
from app.pdf_loader import leer_pdf  # noqa: E402

PDF = Path(__file__).resolve().parent.parent / "data" / "guia_turistica_santander.pdf"


def test_lectura_pdf_extrae_texto():
    texto = leer_pdf(PDF)
    assert len(texto) > 500
    assert "Santander" in texto


def test_agente_carga_documento():
    agente = TourismAgent()
    assert agente.esta_listo() is False

    n = agente.indexar(pdf_path=str(PDF))
    assert n > 500
    assert agente.esta_listo() is True
    assert agente.fuente == "guia_turistica_santander.pdf"


def test_pregunta_vacia_no_llama_al_modelo():
    agente = TourismAgent()
    agente.indexar(pdf_path=str(PDF))
    # Una pregunta vacía se responde localmente, sin llamar a la API.
    resultado = agente.preguntar("   ")
    assert "respuesta" in resultado and "fuente" in resultado


def test_memoria_por_sesion_guarda_y_recuerda(tmp_path):
    memoria = ConversationMemory(dir_base=tmp_path, max_chars=10_000, turnos_min=3)
    sid = "sess-test"
    assert memoria.es_recurrente(sid) is False

    memoria.guardar_turno(sid, "¿Capital?", "Bucaramanga", ip="1.2.3.4")
    memoria.guardar_turno(sid, "¿Rafting?", "En San Gil,\nrío Fonce")  # multilínea

    assert memoria.es_recurrente(sid) is True
    data = memoria.cargar(sid)
    assert data["ip"] == "1.2.3.4"
    assert len(data["turnos"]) == 2

    contexto = memoria.construir_contexto(sid)
    assert "Bucaramanga" in contexto and "San Gil" in contexto

    # Otra sesión distinta no comparte memoria.
    assert memoria.es_recurrente("otra-sesion") is False


def test_memoria_resume_al_superar_limite(tmp_path):
    # Límite muy bajo para forzar el resumen; resumidor simulado (sin Gemini).
    memoria = ConversationMemory(dir_base=tmp_path, max_chars=50, turnos_min=1)
    sid = "sess-resumen"
    llamadas = {"n": 0}

    def resumidor_fake(texto, resumen_previo):
        llamadas["n"] += 1
        return "RESUMEN"

    for i in range(4):
        memoria.guardar_turno(sid, f"pregunta {i}", f"respuesta larga numero {i}",
                              resumidor=resumidor_fake)

    data = memoria.cargar(sid)
    assert data["resumen"] == "RESUMEN"       # se generó el resumen
    assert llamadas["n"] >= 1                  # se llamó al resumidor
    assert len(data["turnos"]) <= 2            # el bloque reciente quedó acotado
