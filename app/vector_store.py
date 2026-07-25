"""Almacén vectorial en memoria basado en NumPy con similitud del coseno.

Para un único documento (una guía turística) no se necesita una base de datos
vectorial pesada: un índice NumPy es rápido, sin dependencias externas y fácil de
desplegar. El índice se persiste en disco como un archivo .npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorStore:
    """Guarda fragmentos de texto y sus embeddings, y busca los más similares."""

    def __init__(self, textos: list[str], embeddings: list[list[float]]):
        self.textos: list[str] = list(textos)
        matriz = np.asarray(embeddings, dtype=np.float32)
        # Normaliza para que el producto punto equivalga a la similitud del coseno.
        normas = np.linalg.norm(matriz, axis=1, keepdims=True)
        normas[normas == 0] = 1e-8
        self.matriz: np.ndarray = matriz / normas

    def buscar(self, embedding_consulta: list[float], k: int = 4) -> list[dict]:
        """Devuelve los `k` fragmentos más similares a la consulta.

        Returns:
            Lista de dicts {"texto": str, "score": float} ordenada de mayor a menor.
        """
        if not self.textos:
            return []

        q = np.asarray(embedding_consulta, dtype=np.float32)
        norma = np.linalg.norm(q)
        q = q / (norma if norma else 1e-8)

        similitudes = self.matriz @ q  # producto punto = coseno (vectores normalizados)
        k = min(k, len(self.textos))
        indices = np.argsort(-similitudes)[:k]

        return [
            {"texto": self.textos[i], "score": float(similitudes[i])}
            for i in indices
        ]

    # ------------------------------------------------------------------ #
    # Persistencia en disco
    # ------------------------------------------------------------------ #
    def guardar(self, ruta: str | Path) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ruta,
            textos=np.array(self.textos, dtype=object),
            matriz=self.matriz,
        )

    @classmethod
    def cargar(cls, ruta: str | Path) -> "VectorStore":
        datos = np.load(Path(ruta), allow_pickle=True)
        instancia = cls.__new__(cls)
        instancia.textos = list(datos["textos"])
        instancia.matriz = datos["matriz"].astype(np.float32)
        return instancia

    @staticmethod
    def existe(ruta: str | Path) -> bool:
        return Path(ruta).exists()
