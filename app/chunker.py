"""Divide un texto largo en fragmentos (chunks) con solapamiento.

El solapamiento evita perder contexto en los límites de cada fragmento, lo que
mejora la calidad de la recuperación semántica en el RAG.
"""
from __future__ import annotations


def dividir_en_chunks(
    texto: str,
    tamano: int = 900,
    solapamiento: int = 150,
) -> list[str]:
    """Parte el texto en fragmentos de ~`tamano` caracteres.

    Intenta cortar en un límite de párrafo o frase cercano para no partir ideas
    a la mitad.

    Args:
        texto: texto completo a fragmentar.
        tamano: tamaño objetivo de cada fragmento (en caracteres).
        solapamiento: caracteres compartidos entre fragmentos consecutivos.

    Returns:
        Lista de fragmentos de texto.
    """
    texto = texto.strip()
    if not texto:
        return []
    if solapamiento >= tamano:
        raise ValueError("El solapamiento debe ser menor que el tamaño del chunk.")

    chunks: list[str] = []
    inicio = 0
    largo = len(texto)

    while inicio < largo:
        fin = min(inicio + tamano, largo)

        # Si no es el final del texto, busca un corte "natural" hacia atrás
        # (fin de párrafo, punto o espacio) dentro de una ventana razonable.
        if fin < largo:
            ventana = texto[inicio:fin]
            corte = max(
                ventana.rfind("\n\n"),
                ventana.rfind(". "),
                ventana.rfind("\n"),
            )
            if corte > tamano * 0.5:  # solo si el corte no deja un chunk diminuto
                fin = inicio + corte + 1

        fragmento = texto[inicio:fin].strip()
        if fragmento:
            chunks.append(fragmento)

        if fin >= largo:
            break
        inicio = max(fin - solapamiento, inicio + 1)

    return chunks
