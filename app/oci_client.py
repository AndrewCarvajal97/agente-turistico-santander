"""Construye el cliente de OCI Generative AI Inference.

Soporta dos métodos de autenticación:
  - "config_file": usa ~/.oci/config (ideal para desarrollo local).
  - "instance_principal": usa el rol de la instancia de OCI (recomendado en la VM,
    así no se guardan credenciales en el servidor).
"""
from __future__ import annotations

import oci

from .config import settings


def build_genai_client() -> "oci.generative_ai_inference.GenerativeAiInferenceClient":
    """Crea y devuelve un cliente autenticado de Generative AI Inference."""
    if settings.oci_auth == "instance_principal":
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return oci.generative_ai_inference.GenerativeAiInferenceClient(
            config={},
            signer=signer,
            service_endpoint=settings.genai_endpoint,
            retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
            timeout=(10, 240),
        )

    # Autenticación por archivo de configuración (~/.oci/config).
    config = oci.config.from_file(profile_name=settings.oci_config_profile)
    return oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=settings.genai_endpoint,
        retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
        timeout=(10, 240),
    )
