"""Construye el índice vectorial del PDF por línea de comandos.

Uso:
    py scripts/build_index.py            # usa el PDF definido en .env
    py scripts/build_index.py otro.pdf   # usa otro PDF

Requiere que las credenciales de OCI estén configuradas (ver README).
"""
import sys
from pathlib import Path

# Permite ejecutar el script desde la raíz del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import TourismAgent  # noqa: E402
from app.config import settings  # noqa: E402


def main() -> None:
    settings.validar()
    pdf = sys.argv[1] if len(sys.argv) > 1 else settings.pdf_path
    print(f"Indexando: {pdf}")
    agente = TourismAgent()
    n = agente.indexar(pdf_path=pdf, forzar=True)
    print(f"✅ Índice creado con {n} fragmentos en {settings.index_path}")


if __name__ == "__main__":
    main()
