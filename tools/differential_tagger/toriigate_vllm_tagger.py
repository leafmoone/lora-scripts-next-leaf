"""
ToriiGate 0.5 VLM backend via vLLM OpenAI-compatible HTTP API.

Mirrors the official distributed caption script pattern (caption_distributed.py):
  - encode image as base64 data URL
  - POST /v1/chat/completions
  - parallel requests via thread pool (concurrency = preferred_batch_size)

Does not load PyTorch/transformers locally; expects a running vLLM server.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from config import (
    TORIIGATE_VLLM_API_KEY,
    TORIIGATE_VLLM_API_URL,
    TORIIGATE_VLLM_MAX_PIXELS_MP,
    TORIIGATE_VLLM_MAX_TOKENS,
    TORIIGATE_VLLM_MODEL,
    TORIIGATE_VLLM_REQUEST_TIMEOUT,
    TORIIGATE_VLLM_TEMPERATURE,
)
from toriigate_tagger import TORIIGATE_MAX_IMAGE_PIXELS, TORIIGATE_SYSTEM_PROMPT, ToriiGateTagger

logger = logging.getLogger(__name__)


def normalize_vlm_backend(value: Optional[str]) -> str:
    """Map UI/CLI backend names to canonical values."""
    key = str(value or "transformers").strip().lower()
    if key in {"toriigate", "transformers", "local", "hf"}:
        return "transformers"
    if key in {"vllm", "openai", "api"}:
        return "vllm"
    return key


class ToriiGateVllmTagger:
    """Call a remote vLLM server for ToriiGate captions."""

    def __init__(
        self,
        *,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
        max_pixels_mp: Optional[float] = None,
        use_gpu: bool = True,
    ) -> None:
        del use_gpu  # remote backend; local GPU flag is ignored
        self.api_url = str(api_url or TORIIGATE_VLLM_API_URL).strip()
        self.api_key = str(api_key if api_key is not None else TORIIGATE_VLLM_API_KEY)
        self.model_name = str(model_name or TORIIGATE_VLLM_MODEL).strip()
        self.max_new_tokens = int(max_new_tokens or TORIIGATE_VLLM_MAX_TOKENS)
        self.temperature = float(
            temperature if temperature is not None else TORIIGATE_VLLM_TEMPERATURE
        )
        self.request_timeout = float(
            request_timeout if request_timeout is not None else TORIIGATE_VLLM_REQUEST_TIMEOUT
        )
        if max_pixels_mp is not None:
            self.max_pixels = max(0.25, float(max_pixels_mp)) * 1_000_000
        else:
            self.max_pixels = float(TORIIGATE_VLLM_MAX_PIXELS_MP or 1.0) * 1_000_000
            if self.max_pixels <= 0:
                self.max_pixels = float(TORIIGATE_MAX_IMAGE_PIXELS)
        self._loaded = False
        self.use_gpu = False

    def _encode_image_base64(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            pixels = img.width * img.height
            if pixels > self.max_pixels:
                scale = (self.max_pixels / float(pixels)) ** 0.5
                new_width = max(512, int(img.width * scale))
                new_height = max(512, int(img.height * scale))
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                img = img.resize((new_width, new_height), resampling)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_messages(self, image_path: str, user_prompt: Optional[str]) -> List[Dict[str, Any]]:
        prompt_text = str(user_prompt or "").strip()
        if not prompt_text:
            from toriigate_tagger import TORIIGATE_SHORT_QUERY

            prompt_text = TORIIGATE_SHORT_QUERY
        image_data = self._encode_image_base64(image_path)
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": TORIIGATE_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

    def _call_api(self, messages: List[Dict[str, Any]]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        result = response.json()
        return str(result["choices"][0]["message"]["content"] or "").strip()

    def load(self) -> None:
        if self._loaded:
            return
        try:
            models_url = self.api_url.rsplit("/chat/completions", 1)[0] + "/models"
            response = requests.get(models_url, timeout=min(15.0, self.request_timeout))
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "vLLM health check via %s failed (%s); will retry on first caption request",
                self.api_url,
                exc,
            )
        self._loaded = True
        logger.info(
            "ToriiGate vLLM backend ready: url=%s model=%s",
            self.api_url,
            self.model_name,
        )

    def _generate_text(self, image_path: str, user_prompt: Optional[str] = None) -> str:
        if not self._loaded:
            self.load()
        messages = self._build_messages(image_path, user_prompt)
        return self._call_api(messages)

    def tag(self, image_path: str, user_prompt: Optional[str] = None) -> Dict[str, Any]:
        try:
            return ToriiGateTagger._build_result(
                self._generate_text(image_path, user_prompt=user_prompt)
            )
        except Exception as exc:
            logger.error("ToriiGate vLLM failed on %s: %s", image_path, exc)
            return {
                "general_tags": [],
                "character_tags": [],
                "rating": "unknown",
                "rating_confidences": {},
                "all_tags": [],
                "error": str(exc),
            }

    def tag_batch(
        self,
        image_paths: List[str],
        *,
        user_prompt: Optional[str] = None,
        user_prompts: Optional[List[str]] = None,
        preferred_batch_size: Optional[int] = None,
        min_batch_size: int = 1,
        return_runtime_info: bool = False,
    ) -> Any:
        del min_batch_size
        if user_prompts is not None and user_prompt is not None:
            raise ValueError("Pass either user_prompt or user_prompts, not both.")
        if user_prompts is not None and len(user_prompts) != len(image_paths):
            raise ValueError(
                f"user_prompts length ({len(user_prompts)}) must match image_paths ({len(image_paths)})"
            )

        if not image_paths:
            empty: List[Dict[str, Any]] = []
            if return_runtime_info:
                return empty, {
                    "initial_chunk_size": 0,
                    "final_chunk_size": 0,
                    "backoff_steps": [],
                    "used_cpu_fallback": False,
                    "attempted_gpu_backoff": False,
                    "backend": "vllm",
                }
            return empty

        if not self._loaded:
            self.load()

        workers = max(1, int(preferred_batch_size or 8))
        results: List[Optional[Dict[str, Any]]] = [None] * len(image_paths)

        def _one(index: int, path: str, prompt: Optional[str]) -> tuple[int, Dict[str, Any]]:
            return index, self.tag(path, user_prompt=prompt)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _one,
                    idx,
                    path,
                    user_prompts[idx] if user_prompts is not None else user_prompt,
                ): idx
                for idx, path in enumerate(image_paths)
            }
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as exc:
                    idx = futures[future]
                    logger.error("ToriiGate vLLM task failed for %s: %s", image_paths[idx], exc)
                    results[idx] = {
                        "general_tags": [],
                        "character_tags": [],
                        "rating": "unknown",
                        "rating_confidences": {},
                        "all_tags": [],
                        "error": str(exc),
                    }

        filled = [item if item is not None else ToriiGateTagger._build_result("") for item in results]
        runtime_info = {
            "initial_chunk_size": workers,
            "final_chunk_size": workers,
            "backoff_steps": [],
            "used_cpu_fallback": False,
            "attempted_gpu_backoff": False,
            "backend": "vllm",
        }
        if return_runtime_info:
            return filled, runtime_info
        return filled


_vllm_tagger = None
_vllm_settings: Dict[str, Any] = {}
_vllm_lock = threading.Lock()


def get_toriigate_vllm_tagger(
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    use_gpu: bool = True,
    force_reload: bool = False,
    **_: Any,
) -> ToriiGateVllmTagger:
    """Get or create the ToriiGate vLLM singleton."""
    global _vllm_tagger, _vllm_settings

    with _vllm_lock:
        new_settings = {
            "api_url": api_url or TORIIGATE_VLLM_API_URL,
            "api_key": api_key if api_key is not None else TORIIGATE_VLLM_API_KEY,
            "model_name": model_name or TORIIGATE_VLLM_MODEL,
            "use_gpu": use_gpu,
        }
        if force_reload or _vllm_tagger is None or new_settings != _vllm_settings:
            _vllm_tagger = ToriiGateVllmTagger(
                api_url=new_settings["api_url"],
                api_key=new_settings["api_key"],
                model_name=new_settings["model_name"],
                use_gpu=use_gpu,
            )
            _vllm_settings = new_settings
        return _vllm_tagger
