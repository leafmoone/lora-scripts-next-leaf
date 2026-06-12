"""Configuration for Anima Train caption pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

GEMMA_LOCAL_DIR = "models/gemma-4-E3B-it"
GEMMA_MODELSCOPE_ID = "spawner/spawner-gemma-4-E4B-it"
GEMMA_SERVED_NAME = "spawner-gemma-4-e4b-it"
GEMMA_VLLM_PORT = 9002
GEMMA_VLLM_URL = f"http://127.0.0.1:{GEMMA_VLLM_PORT}/v1/chat/completions"

TORIIGATE_SERVED_NAME = "toriigate-0.5"
TORIIGATE_VLLM_PORT = 18901
TORIIGATE_VLLM_URL = f"http://127.0.0.1:{TORIIGATE_VLLM_PORT}/v1/chat/completions"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

VLM_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "toriigate-0.5": {
        "display_name": "ToriiGate 0.5",
        "default_api_url": TORIIGATE_VLLM_URL,
        "default_served_name": TORIIGATE_SERVED_NAME,
        "local_model_dir": None,
    },
    "gemma-4-e4b": {
        "display_name": "Gemma-4-E4B (vLLM)",
        "default_api_url": GEMMA_VLLM_URL,
        "default_served_name": GEMMA_SERVED_NAME,
        "local_model_dir": GEMMA_LOCAL_DIR,
        "modelscope_id": GEMMA_MODELSCOPE_ID,
    },
}


def project_root_from(cwd: str | Path | None = None) -> Path:
    if cwd:
        return Path(cwd).resolve()
    return PROJECT_ROOT


def gemma_local_path(root: Path | None = None) -> Path:
    return project_root_from(root) / GEMMA_LOCAL_DIR


def resolve_vlm_endpoint(
    vlm_model: str,
    user_url: str = "",
    user_served_name: str = "",
) -> tuple[str, str]:
    preset = VLM_MODEL_PRESETS.get(str(vlm_model or "").strip(), VLM_MODEL_PRESETS["toriigate-0.5"])
    api_url = str(user_url or preset["default_api_url"]).strip()
    served_name = str(user_served_name or preset["default_served_name"]).strip()
    return api_url, served_name
