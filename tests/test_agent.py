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


def test_memoria_csv_guarda_filtra_y_recuerda(tmp_path):
    csv = tmp_path / "historial.csv"
    memoria = ConversationMemory(ruta_csv=csv, max_turnos=6)
    sid = "sess-test"
    assert memoria.es_recurrente(sid) is False

    memoria.guardar_turno(sid, "¿Capital?", "Bucaramanga", ip="1.2.3.4")
    memoria.guardar_turno(sid, "¿Rafting?", "En San Gil, río Fonce")
    memoria.guardar_turno("otra-sesion", "¿Comida?", "Hormigas culonas")

    # El CSV se creó y contiene las 3 filas.
    assert csv.exists()

    # Filtro por session_id: solo los turnos de esa sesión.
    sesion = memoria.historial_sesion(sid)
    assert sesion.shape[0] == 2
    assert memoria.es_recurrente(sid) is True

    # El contexto incluye la memoria de esa sesión.
    contexto = memoria.construir_contexto(sid)
    assert "Bucaramanga" in contexto and "San Gil" in contexto

    # Sesiones distintas no comparten memoria.
    assert memoria.historial_sesion("otra-sesion").shape[0] == 1
    assert memoria.es_recurrente("no-existe") is False


def test_memoria_busqueda_por_termino(tmp_path):
    memoria = ConversationMemory(ruta_csv=tmp_path / "historial.csv")
    memoria.guardar_turno("s1", "¿Dónde hago rafting?", "En San Gil, río Fonce")
    memoria.guardar_turno("s1", "¿Qué comer?", "Hormigas culonas")

    # Búsqueda por término (filtro str.contains, sin distinguir mayúsculas).
    resultados = memoria.buscar("rafting")
    assert len(resultados) == 1
    assert resultados[0]["respuesta"] == "En San Gil, río Fonce"

    assert memoria.buscar("SAN GIL")  # case-insensitive
    assert memoria.buscar("inexistente") == []


def test_memoria_listar_sesiones(tmp_path):
    memoria = ConversationMemory(ruta_csv=tmp_path / "historial.csv")
    memoria.guardar_turno("s1", "p1", "r1")
    memoria.guardar_turno("s1", "p2", "r2")
    memoria.guardar_turno("s2", "p3", "r3")

    sesiones = {s["session_id"]: s for s in memoria.listar_sesiones()}
    assert sesiones["s1"]["turnos"] == 2
    assert sesiones["s2"]["turnos"] == 1
