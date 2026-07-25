"""Procesa un lote de preguntas y genera un CSV con las respuestas del agente.

Aplica el flujo de manejo de archivos del curso (leer/escribir con Python + pandas),
pero reutilizando el **agente del proyecto** (`TourismAgent`), de modo que las
respuestas se basan en el documento fuente (el PDF de la guía).

Etapas:
  1. Leer las preguntas desde un archivo de texto (una por línea).
  2. Obtener la respuesta del agente para cada pregunta.
  3. Guardar una lista de diccionarios {pregunta, respuesta, fuente}.
  4. Exportar todo a un archivo CSV usando pandas.

Uso:
    python scripts/generar_respuestas.py
    python scripts/generar_respuestas.py data/preguntas.txt data/respuestas.csv

Requiere el proveedor de LLM configurado (Gemini o Groq) en el archivo .env.
Para lotes grandes se recomienda LLM_PROVIDER=groq (free tier más generoso).
"""
import sys
from pathlib import Path

# Permite ejecutar el script desde la raíz del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from app.agent import TourismAgent  # noqa: E402
from app.config import settings  # noqa: E402


def leer_preguntas(ruta: str) -> list[str]:
    """Etapa 1: lee las preguntas del archivo de texto y las devuelve en una lista."""
    with open(ruta, "r", encoding="utf-8") as archivo:
        return [linea.strip() for linea in archivo if linea.strip()]


def main() -> None:
    entrada = sys.argv[1] if len(sys.argv) > 1 else "data/preguntas.txt"
    salida = sys.argv[2] if len(sys.argv) > 2 else "data/respuestas.csv"

    settings.validar()
    preguntas = leer_preguntas(entrada)
    print(f"Leídas {len(preguntas)} preguntas de '{entrada}'.")

    agente = TourismAgent()
    agente.indexar()

    # Etapas 2 y 3: obtener respuestas y armar la lista de diccionarios.
    filas: list[dict] = []
    for i, pregunta in enumerate(preguntas, start=1):
        resultado = agente.preguntar(pregunta)
        filas.append(
            {
                "pregunta": pregunta,
                "respuesta": resultado["respuesta"],
                "fuente": resultado["fuente"],
            }
        )
        print(f"  [{i}/{len(preguntas)}] {pregunta}")

    # Etapas 4 y 5: exportar a CSV con pandas.
    df = pd.DataFrame(filas)
    df.to_csv(salida, index=False, encoding="utf-8")
    print(f"\n✅ CSV generado en '{salida}' con {len(df)} filas.")


if __name__ == "__main__":
    main()
