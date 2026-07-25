"""Tests unitarios que NO requieren conexión a OCI.

Validan las piezas locales del pipeline: lectura de PDF, fragmentación y
búsqueda vectorial (con embeddings simulados).

Ejecuta con:  py -m pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunker import dividir_en_chunks  # noqa: E402
from app.pdf_loader import leer_pdf  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402

PDF = Path(__file__).resolve().parent.parent / "data" / "guia_turistica_santander.pdf"


def test_lectura_pdf_extrae_texto():
    texto = leer_pdf(PDF)
    assert len(texto) > 500
    assert "Santander" in texto


def test_chunker_respeta_tamano_y_solapamiento():
    texto = "Frase de prueba. " * 200
    chunks = dividir_en_chunks(texto, tamano=300, solapamiento=50)
    assert len(chunks) > 1
    assert all(len(c) <= 320 for c in chunks)  # margen por corte natural


def test_chunker_texto_vacio():
    assert dividir_en_chunks("") == []


def test_vector_store_recupera_el_mas_similar():
    textos = ["rafting en el río Fonce", "gastronomía de Santander", "clima templado"]
    # Embeddings simulados (3 dimensiones) claramente separados.
    embeddings = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    store = VectorStore(textos, embeddings)

    resultado = store.buscar([0.9, 0.1, 0.0], k=1)
    assert resultado[0]["texto"] == "rafting en el río Fonce"


def test_vector_store_persistencia(tmp_path):
    store = VectorStore(["a", "b"], [[1, 0], [0, 1]])
    ruta = tmp_path / "idx.npz"
    store.guardar(ruta)
    assert VectorStore.existe(ruta)

    recargado = VectorStore.cargar(ruta)
    assert recargado.textos == ["a", "b"]
