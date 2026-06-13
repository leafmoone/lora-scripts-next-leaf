"""OpenAI-compatible vLLM HTTP client for Anima Train two-step VLM chain."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import requests
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MAX_PIXELS = 1_000_000
DEFAULT_REQUEST_TIMEOUT = 300.0
GEMMA_MODEL_KEYS = frozenset({"gemma-4-e4b", "gemma", "spawner-gemma-4-e4b-it"})


class GemmaVllmUnavailableError(RuntimeError):
    """Raised when a Gemma vLLM server is reachable but cannot generate usable text."""


def gemma_vllm_unavailable_message(api_url: str, model_name: str) -> str:
    return (
        "Gemma vLLM backend is not usable: the server is reachable but the generation "
        f"probe returned empty/pad output (url={api_url}, model={model_name}). "
        "On this machine NVIDIA driver 570 / CUDA 12.8 cannot run the vLLM 0.22.x "
        "CUDA 13 custom kernels; disabling custom ops lets the server start but breaks "
        "Gemma generation. Use gemma_vlm_backend=auto/transformers, or move the vLLM "
        "backend to a CUDA-13-capable driver/instance and enable custom ops."
    )


def build_data_url_for_image_path(image_path: str, *, max_pixels: float = DEFAULT_MAX_PIXELS) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        pixels = img.width * img.height
        if max_pixels > 0 and pixels > max_pixels:
            scale = (max_pixels / float(pixels)) ** 0.5
            new_width = max(512, int(img.width * scale))
            new_height = max(512, int(img.height * scale))
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            img = img.resize((new_width, new_height), resampling)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_user_message_content(image_path: str, user_prompt: str, *, max_pixels: float = DEFAULT_MAX_PIXELS) -> list[dict[str, Any]]:
    return [
        {
            "type": "image_url",
            "image_url": {"url": build_data_url_for_image_path(image_path, max_pixels=max_pixels)},
        },
        {"type": "text", "text": str(user_prompt or "").strip()},
    ]


class VlmClient:
    """Call a remote vLLM server via OpenAI-compatible chat completions."""

    def __init__(
        self,
        *,
        api_url: str,
        model_name: str,
        api_key: str = "not-needed",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        max_pixels: float = DEFAULT_MAX_PIXELS,
    ) -> None:
        self.api_url = str(api_url or "").strip()
        self.model_name = str(model_name or "").strip()
        self.api_key = str(api_key or "not-needed")
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.request_timeout = float(request_timeout)
        self.max_pixels = float(max_pixels)

    def complete(
        self,
        *,
        image_path: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        messages: list[dict[str, Any]] = []
        system_text = str(system_prompt or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append(
            {
                "role": "user",
                "content": build_user_message_content(
                    image_path,
                    user_prompt,
                    max_pixels=self.max_pixels,
                ),
            }
        )
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            content = "\n".join(parts)
        return str(content or "").strip()


def is_gemma_vlm_model(vlm_model: str) -> bool:
    return str(vlm_model or "").strip().lower() in GEMMA_MODEL_KEYS


def is_broken_vllm_output(content: str, token_ids: list[int] | None = None) -> bool:
    text = str(content or "").strip()
    if text:
        lowered = text.lower()
        compact = lowered.replace(" ", "").replace("\n", "")
        if lowered in {"<pad>", "<mask>"} or compact in {"<pad><pad>", "<mask><mask>"}:
            return True
        if compact and compact == "<pad>" * compact.count("<pad>"):
            return True
        if not any(ch.isalnum() for ch in text):
            return True
        return False

    if not token_ids:
        return True

    non_pad = [token for token in token_ids if token not in (0, 1)]
    return len(non_pad) == 0


def probe_vllm_generation(
    *,
    api_url: str,
    model_name: str,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> bool:
    """Return True when vLLM returns non-empty, non-garbage text."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "return_token_ids": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed",
    }
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=min(request_timeout, 120.0),
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            return False
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        token_ids = choices[0].get("token_ids")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            content = "\n".join(parts)
        return not is_broken_vllm_output(str(content or ""), token_ids)
    except Exception as exc:
        logger.warning("vLLM probe failed for %s: %s", api_url, exc)
        return False


def create_vlm_client(
    *,
    vlm_model: str,
    api_url: str,
    model_name: str,
    local_model_dir: str | Path | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    gemma_vlm_backend: str = "auto",
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
):
    """Create HTTP or local Gemma client for the Anima Train VLM chain."""
    from .gemma_local_client import LocalGemmaVlmClient
    from .model_resolver import normalize_gemma_vlm_backend

    model_key = str(vlm_model or "").strip().lower()
    backend = normalize_gemma_vlm_backend(gemma_vlm_backend)
    if is_gemma_vlm_model(model_key) and local_model_dir:
        use_local = backend == "transformers"
        if backend == "auto" and api_url:
            use_local = not probe_vllm_generation(
                api_url=api_url,
                model_name=model_name,
                request_timeout=request_timeout,
            )
        elif backend == "vllm" and api_url:
            if not probe_vllm_generation(
                api_url=api_url,
                model_name=model_name,
                request_timeout=request_timeout,
            ):
                raise GemmaVllmUnavailableError(
                    gemma_vllm_unavailable_message(api_url, model_name)
                )
        if use_local:
            if backend == "auto":
                logger.warning(
                    "Gemma vLLM output unusable at %s; falling back to local transformers inference.",
                    api_url,
                )
            else:
                logger.info("Using local transformers Gemma backend (%s).", backend)
            return LocalGemmaVlmClient(
                model_dir=local_model_dir,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    return VlmClient(
        api_url=api_url,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        request_timeout=request_timeout,
    )
