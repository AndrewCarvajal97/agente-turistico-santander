"""Carga y centraliza la configuración del proyecto desde variables de entorno."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Carga las variables definidas en el archivo .env (si existe).
load_dotenv()


def _get(nombre: str, por_defecto: str = "") -> str:
    return os.getenv(nombre, por_defecto).strip()


@dataclass(frozen=True)
class Settings:
    """Configuración inmutable de la aplicación."""

    # Autenticación OCI
    oci_auth: str = _get("OCI_AUTH", "config_file")
    oci_config_profile: str = _get("OCI_CONFIG_PROFILE", "DEFAULT")
    compartment_id: str = _get("OCI_COMPARTMENT_ID")
    genai_endpoint: str = _get(
        "OCI_GENAI_ENDPOINT",
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    )

    # Modelos
    embed_model: str = _get("OCI_EMBED_MODEL", "cohere.embed-multilingual-v3.0")
    chat_model: str = _get("OCI_CHAT_MODEL", "cohere.command-r-08-2024")

    # Documento y parámetros de RAG
    pdf_path: str = _get("PDF_PATH", "data/guia_turistica_santander.pdf")
    index_path: str = _get("INDEX_PATH", "data/index.npz")
    top_k: int = int(_get("TOP_K", "4") or 4)
    chunk_size: int = int(_get("CHUNK_SIZE", "900") or 900)
    chunk_overlap: int = int(_get("CHUNK_OVERLAP", "150") or 150)

    def validar(self) -> None:
        """Lanza un error claro si falta configuración esencial."""
        if not self.compartment_id:
            raise ValueError(
                "Falta OCI_COMPARTMENT_ID. Copia .env.example a .env y complétalo."
            )


settings = Settings()
