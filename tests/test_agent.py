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


def test_memoria_guarda_y_lee(tmp_path):
    ruta = tmp_path / "historial.jsonl"
    memoria = ConversationMemory(ruta=ruta)
    assert memoria.leer() == []

    memoria.guardar("¿Capital?", "Bucaramanga", "guia.pdf")
    memoria.guardar("¿Rafting?", "En San Gil,\nrío Fonce", "guia.pdf")  # respuesta multilínea

    registros = memoria.leer()
    assert len(registros) == 2
    assert memoria.total() == 2
    assert registros[0]["pregunta"] == "¿Capital?"
    assert registros[1]["respuesta"] == "En San Gil,\nrío Fonce"
    assert "fecha" in registros[0]
