"""
Configuration for standalone AI image tagger.
All configurable values with environment variable support.
"""
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_user_path(path: Optional[str]) -> str:
    """Normalize user-entered paths."""
    text = str(path or "").strip()
    return text

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

PACKAGE_ROOT: Path = Path(__file__).parent.resolve()

DATA_DIR: Path = Path(
    os.environ.get("STANDALONE_TAGGER_DATA_DIR", str(PACKAGE_ROOT / "data"))
).expanduser()

MODELS_DIR: Path = Path(
    os.environ.get("STANDALONE_TAGGER_MODELS_DIR", str(DATA_DIR))
).expanduser()

TEMP_DIR: Path = Path(
    os.environ.get("STANDALONE_TAGGER_TMP_DIR", str(DATA_DIR / "tmp"))
).expanduser()

WD14_MODEL_DIR: Path = Path(
    os.environ.get("STANDALONE_TAGGER_WD14_MODEL_DIR", str(MODELS_DIR / "wd14-tagger"))
).expanduser()

TORIIGATE_MODEL_DIR: Path = Path(
    os.environ.get("STANDALONE_TAGGER_TORIIGATE_MODEL_DIR", str(MODELS_DIR / "toriigate"))
).expanduser()

OPPAI_ORACLE_MODEL_DIR: Path = Path(
    os.environ.get("STANDALONE_TAGGER_OPPAI_ORACLE_MODEL_DIR", str(MODELS_DIR / "oppai-oracle"))
).expanduser()


def read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: expected integer, got {raw!r}") from exc


def read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: expected number, got {raw!r}") from exc


def get_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_temp_dir() -> str:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return str(TEMP_DIR)


def get_wd14_model_dir() -> str:
    WD14_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return str(WD14_MODEL_DIR)


def get_toriigate_model_dir() -> str:
    TORIIGATE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return str(TORIIGATE_MODEL_DIR)


def get_oppai_oracle_model_dir() -> str:
    OPPAI_ORACLE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return str(OPPAI_ORACLE_MODEL_DIR)


# ---------------------------------------------------------------------------
# Download source
# ---------------------------------------------------------------------------

DOWNLOAD_MIRROR_CONFIG_PATH: Path = DATA_DIR / "config" / "download-mirror.json"


def get_download_mirror() -> str:
    try:
        if DOWNLOAD_MIRROR_CONFIG_PATH.exists():
            data = json.loads(DOWNLOAD_MIRROR_CONFIG_PATH.read_text(encoding="utf-8"))
            return str(data.get("mirror", "auto") or "auto").strip().lower()
    except Exception:
        pass
    return os.environ.get("STANDALONE_TAGGER_DOWNLOAD_MIRROR", "auto").strip().lower()


def save_download_mirror(mirror: str) -> None:
    DOWNLOAD_MIRROR_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_MIRROR_CONFIG_PATH.write_text(
        json.dumps({"mirror": mirror}, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Tagger settings
# ---------------------------------------------------------------------------

DEFAULT_TAGGER_MODEL: str = os.environ.get(
    "STANDALONE_TAGGER_DEFAULT_MODEL", "wd-swinv2-tagger-v3"
)

TAGGER_GENERAL_THRESHOLD: float = read_float_env(
    "STANDALONE_TAGGER_GENERAL_THRESHOLD", 0.35
)
TAGGER_CHARACTER_THRESHOLD: float = read_float_env(
    "STANDALONE_TAGGER_CHARACTER_THRESHOLD", 0.85
)
TAGGER_USE_GPU: bool = os.environ.get(
    "STANDALONE_TAGGER_USE_GPU", "true"
).lower() in ("true", "1", "yes")

RATING_CATEGORIES: list = ["general", "sensitive", "questionable", "explicit"]

ALLOWED_IMAGE_EXTENSIONS: set = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"
}

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

TAGGER_MODELS: dict = {
    "wd-eva02-large-tagger-v3": {
        "repo_id": "SmilingWolf/wd-eva02-large-tagger-v3",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "runtime_safety_tier": "heavy",
        "default_threshold": 0.35,
        "default_character_threshold": 0.85,
        "default_copyright_threshold": 0.35,
        "default_max_tags_per_image": 60,
    },
    "wd-swinv2-tagger-v3": {
        "repo_id": "SmilingWolf/wd-swinv2-tagger-v3",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "runtime_safety_tier": "balanced",
        "default_threshold": 0.35,
        "default_character_threshold": 0.85,
        "default_copyright_threshold": 0.35,
        "default_max_tags_per_image": 50,
    },
    "wd-convnext-tagger-v3": {
        "repo_id": "SmilingWolf/wd-convnext-tagger-v3",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "runtime_safety_tier": "balanced",
        "default_threshold": 0.35,
        "default_character_threshold": 0.85,
        "default_copyright_threshold": 0.35,
        "default_max_tags_per_image": 50,
    },
    "wd-vit-tagger-v3": {
        "repo_id": "SmilingWolf/wd-vit-tagger-v3",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "runtime_safety_tier": "light",
        "default_threshold": 0.35,
        "default_character_threshold": 0.85,
        "default_copyright_threshold": 0.35,
        "default_max_tags_per_image": 40,
    },
    "wd-vit-large-tagger-v3": {
        "repo_id": "SmilingWolf/wd-vit-large-tagger-v3",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "runtime_safety_tier": "balanced",
        "default_threshold": 0.35,
        "default_character_threshold": 0.85,
        "default_copyright_threshold": 0.35,
        "default_max_tags_per_image": 55,
    },
    "camie-tagger-v2": {
        "repo_id": "Camais03/camie-tagger-v2",
        "model_file": "camie-tagger-v2.onnx",
        "tags_file": "camie-tagger-v2-metadata.json",
        "runtime_safety_tier": "heavy",
        "metadata_format": "camie_v2",
        "input_layout": "nchw",
        "input_normalization": "imagenet",
        "output_activation": "sigmoid",
        "pad_color": [124, 116, 104],
        "default_threshold": 0.62,
        "default_character_threshold": 0.78,
        "default_copyright_threshold": 0.62,
        "default_max_tags_per_image": 65,
        "supports_rating": True,
    },
    "pixai-tagger-v0.9": {
        "repo_id": "deepghs/pixai-tagger-v0.9-onnx",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "runtime_safety_tier": "heavy",
        "input_layout": "nchw",
        "input_normalization": "minus_one_to_one",
        "resize_mode": "stretch",
        "output_index": 2,
        "output_activation": "identity",
        "default_threshold": 0.45,
        "default_character_threshold": 0.85,
        "default_copyright_threshold": 0.45,
        "default_max_tags_per_image": 65,
        "supports_rating": False,
        "rating_fallback_mode": "derive_from_tags",
    },
    "toriigate-0.5": {
        "repo_id": "Minthy/ToriiGate-0.5",
        "model_file": "config.json",
        "tags_file": "",
        "runtime_backend": "toriigate",
        "runtime_safety_tier": "vlm",
        "minimum_total_ram_gb": 16,
        "minimum_available_ram_gb": 4,
        "minimum_gpu_vram_mb": 16384,
        "minimum_gpu_available_vram_mb": 14000,
        "minimum_cpu_total_ram_gb": 32,
        "minimum_cpu_available_ram_gb": 20,
        "default_threshold": 1.0,
        "default_character_threshold": 1.0,
        "default_copyright_threshold": 1.0,
        "default_max_tags_per_image": 120,
        "supports_rating": True,
    },
    "oppai-oracle-v1.1": {
        "repo_id": "Grio43/OppaiOracle",
        "repo_subfolder": "V1.1_onnx",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "extra_files": ["preprocessing.json", "pr_thresholds.json", "config.json"],
        "runtime_backend": "oppai-oracle",
        "runtime_safety_tier": "heavy",
        "input_layout": "nchw",
        "input_normalization": "minus_one_to_one",
        "resize_mode": "letterbox",
        "pad_color": [114, 114, 114],
        "image_size": 448,
        "output_activation": "identity",
        "supports_rating": True,
        "default_threshold": 0.7927,
        "default_copyright_threshold": 0.7927,
        "default_max_tags_per_image": 60,
        "default_character_threshold": 1.0,
    },
}
