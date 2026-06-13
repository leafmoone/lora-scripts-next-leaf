"""Local Gemma 4 inference via transformers (fallback when vLLM output is broken)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODEL = None
_PROCESSOR = None
_MODEL_DIR: Path | None = None
_LOAD_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()


def _load_model(model_dir: Path):
    global _MODEL, _PROCESSOR, _MODEL_DIR
    model_dir = model_dir.resolve()
    with _LOAD_LOCK:
        if _MODEL is not None and _MODEL_DIR == model_dir:
            return _MODEL, _PROCESSOR

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        logger.info("Loading local Gemma model from %s", model_dir)
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _MODEL = model
        _PROCESSOR = processor
        _MODEL_DIR = model_dir
        return _MODEL, _PROCESSOR


class LocalGemmaVlmClient:
    """Run Gemma 4 multimodal inference in-process with transformers."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def complete(
        self,
        *,
        image_path: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        model, processor = _load_model(self.model_dir)
        messages: list[dict[str, Any]] = []
        system_text = str(system_prompt or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path)},
                    {"type": "text", "text": str(user_prompt or "").strip()},
                ],
            }
        )

        with _INFER_LOCK:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
            device_inputs = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            input_len = device_inputs["input_ids"].shape[-1]
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_tokens,
            }
            if self.temperature <= 0:
                generate_kwargs["do_sample"] = False
            else:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = self.temperature

            outputs = model.generate(**device_inputs, **generate_kwargs)
            response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
            parsed = processor.parse_response(response)
            content = parsed.get("content") if isinstance(parsed, dict) else None
            if content is not None:
                return str(content).strip()
            return str(response or "").strip()
