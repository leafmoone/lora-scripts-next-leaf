"""Shared model download-source selection for HuggingFace."""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from config import get_download_mirror

logger = logging.getLogger(__name__)

HF_OFFICIAL_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

_INITIAL_HF_ENDPOINT = str(os.environ.get("HF_ENDPOINT", "") or "").strip().rstrip("/")


def _dedupe_endpoints(endpoints: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for endpoint in endpoints:
        normalized = str(endpoint or "").strip().rstrip("/")
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def get_hf_endpoint_order(
    *, mirror: Optional[str] = None, model_name: str = ""
) -> List[str]:
    """Return HuggingFace-compatible endpoints in the order they should be tried."""
    selected = str(mirror or get_download_mirror() or "auto").strip().lower()
    if selected not in {"auto", "hf-mirror", "modelscope"}:
        selected = "auto"

    env_endpoint = _INITIAL_HF_ENDPOINT

    if selected == "hf-mirror":
        return _dedupe_endpoints([HF_MIRROR_ENDPOINT, env_endpoint, HF_OFFICIAL_ENDPOINT])

    if selected == "modelscope":
        if model_name:
            logger.info(
                "Download Source is ModelScope, but %s is HuggingFace-hosted; "
                "using hf-mirror fallback for this model.",
                model_name,
            )
        return _dedupe_endpoints([HF_MIRROR_ENDPOINT, env_endpoint, HF_OFFICIAL_ENDPOINT])

    return _dedupe_endpoints([env_endpoint, HF_OFFICIAL_ENDPOINT, HF_MIRROR_ENDPOINT])


def apply_hf_endpoint(endpoint: str, *, purpose: str = "") -> str:
    """Make a HuggingFace endpoint visible to libraries that read globals/env."""
    normalized = (
        str(endpoint or HF_OFFICIAL_ENDPOINT).strip().rstrip("/") or HF_OFFICIAL_ENDPOINT
    )
    os.environ["HF_ENDPOINT"] = normalized

    try:
        import huggingface_hub.constants as constants

        constants.ENDPOINT = normalized
        constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            normalized + "/{repo_id}/resolve/{revision}/{filename}"
        )
    except Exception as exc:
        logger.debug(
            "Could not patch huggingface_hub endpoint for %s: %s",
            purpose or normalized, exc,
        )

    if purpose:
        logger.info("Using HuggingFace endpoint %s for %s", normalized, purpose)
    return normalized


def endpoint_label(endpoint: str) -> str:
    normalized = str(endpoint or "").strip().rstrip("/")
    if normalized == HF_OFFICIAL_ENDPOINT:
        return "huggingface.co"
    if normalized == HF_MIRROR_ENDPOINT:
        return "hf-mirror.com"
    return normalized or "huggingface.co"
