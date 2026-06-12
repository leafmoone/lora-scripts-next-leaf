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
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": str(system_prompt or "").strip()}],
            },
            {
                "role": "user",
                "content": build_user_message_content(
                    image_path,
                    user_prompt,
                    max_pixels=self.max_pixels,
                ),
            },
        ]
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
