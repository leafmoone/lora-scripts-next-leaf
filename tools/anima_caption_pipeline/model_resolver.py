"""Resolve Gemma local weights and vLLM endpoints for Anima Train."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import (
    GEMMA_LOCAL_DIR,
    GEMMA_MODELSCOPE_ID,
    GEMMA_SERVED_NAME,
    GEMMA_VLLM_URL,
    TORIIGATE_SERVED_NAME,
    TORIIGATE_VLLM_URL,
    project_root_from,
    resolve_vlm_endpoint,
)

logger = logging.getLogger(__name__)

HF_WEIGHT_MARKERS = ("config.json", "model.safetensors", "pytorch_model.bin")
GEMMA_VLM_BACKENDS = frozenset({"auto", "vllm", "transformers"})


def normalize_gemma_vlm_backend(value: str | None, *, default: str = "auto") -> str:
    backend = str(value or default).strip().lower()
    if backend not in GEMMA_VLM_BACKENDS:
        raise ValueError(f"Unsupported gemma_vlm_backend: {backend!r} (expected auto/vllm/transformers)")
    return backend


def should_start_vllm_for_gemma(gemma_vlm_backend: str) -> bool:
    return normalize_gemma_vlm_backend(gemma_vlm_backend) != "transformers"


def is_valid_hf_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "config.json").is_file():
        return True
    return any((path / marker).is_file() for marker in HF_WEIGHT_MARKERS[1:])


def _modelscope_download_cmd(model_id: str, local_dir: Path) -> list[str]:
    venv_bin = os.path.dirname(os.path.abspath(sys.executable))
    modelscope_bin = shutil.which("modelscope", path=venv_bin)
    if modelscope_bin:
        return [modelscope_bin, "download", model_id, "--local_dir", str(local_dir)]
    # modelscope>=1.37 dropped `python -m modelscope`; CLI lives in modelscope.cli.cli
    return [
        sys.executable,
        "-m",
        "modelscope.cli.cli",
        "download",
        model_id,
        "--local_dir",
        str(local_dir),
    ]


def ensure_gemma_model(project_root: str | Path | None = None, auto_download: bool = True) -> Path:
    root = project_root_from(project_root)
    local_dir = root / GEMMA_LOCAL_DIR
    if is_valid_hf_model_dir(local_dir):
        return local_dir

    if not auto_download:
        raise FileNotFoundError(
            f"Gemma model not found at {local_dir}. "
            f"Run: modelscope download {GEMMA_MODELSCOPE_ID} --local_dir ./{GEMMA_LOCAL_DIR}"
        )

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = _modelscope_download_cmd(GEMMA_MODELSCOPE_ID, local_dir)
    logger.info("Downloading Gemma model: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"modelscope download failed (exit {proc.returncode}): {stderr or 'no output'}"
        )
    if not is_valid_hf_model_dir(local_dir):
        raise FileNotFoundError(f"Download finished but model dir is invalid: {local_dir}")
    return local_dir


def resolve_vlm_runtime(
    vlm_model: str,
    *,
    user_url: str = "",
    user_served_name: str = "",
    project_root: str | Path | None = None,
    auto_download_gemma: bool = True,
    gemma_vlm_backend: str = "",
    preset: dict | None = None,
) -> dict[str, str | Path | None]:
    model_key = str(vlm_model or "toriigate-0.5").strip().lower()
    api_url, served_name = resolve_vlm_endpoint(model_key, user_url, user_served_name)
    local_dir: Path | None = None
    backend = "auto"

    if model_key in {"gemma-4-e4b", "gemma", "spawner-gemma-4-e4b-it"}:
        local_dir = ensure_gemma_model(project_root, auto_download=auto_download_gemma)
        if not user_url:
            api_url = GEMMA_VLLM_URL
        if not user_served_name:
            served_name = GEMMA_SERVED_NAME
        preset_backend = ""
        if preset:
            preset_backend = str(preset.get("gemma_vlm_backend", "")).strip()
        backend = normalize_gemma_vlm_backend(gemma_vlm_backend or preset_backend or "auto")
    elif model_key in {"toriigate-0.5", "toriigate", "toriigate-0.5-vllm"}:
        if not user_url:
            api_url = TORIIGATE_VLLM_URL
        if not user_served_name:
            served_name = TORIIGATE_SERVED_NAME

    return {
        "vlm_model": model_key,
        "api_url": api_url,
        "served_name": served_name,
        "local_model_dir": local_dir,
        "gemma_vlm_backend": backend,
    }
