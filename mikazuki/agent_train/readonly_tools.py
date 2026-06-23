from __future__ import annotations

from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def inspect_dataset(path: str, *, recursive: bool = False) -> dict[str, Any]:
    root = Path(str(path or "")).expanduser()
    if not root.exists():
        return {
            "tool": "inspect_dataset",
            "status": "warning",
            "path": str(root),
            "exists": False,
            "is_dir": False,
            "image_count": 0,
            "caption_count": 0,
            "missing_caption_count": 0,
            "message": "路径不存在或当前环境不可访问。",
        }
    if not root.is_dir():
        return {
            "tool": "inspect_dataset",
            "status": "fail",
            "path": str(root),
            "exists": True,
            "is_dir": False,
            "image_count": 0,
            "caption_count": 0,
            "missing_caption_count": 0,
            "message": "路径不是目录。",
        }

    files = root.rglob("*") if recursive else root.iterdir()
    image_files = [item for item in files if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS]
    caption_count = sum(1 for image in image_files if image.with_suffix(".txt").is_file())
    return {
        "tool": "inspect_dataset",
        "status": "success",
        "path": str(root),
        "exists": True,
        "is_dir": True,
        "recursive": recursive,
        "image_count": len(image_files),
        "caption_count": caption_count,
        "missing_caption_count": len(image_files) - caption_count,
        "message": "目录检查完成。",
    }


def inspect_model_file(path: str) -> dict[str, Any]:
    target = Path(str(path or "")).expanduser()
    exists = target.exists()
    is_file = target.is_file()
    status = "success" if exists and is_file else "warning"
    return {
        "tool": "inspect_model_file",
        "status": status,
        "path": str(target),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": target.stat().st_size if exists and is_file else 0,
        "message": "模型文件检查完成。" if exists and is_file else "模型文件不存在或当前环境不可访问。",
    }


def inspect_payload(skill_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if skill_id == "tagger_leaf_batch_caption":
        checks.append(inspect_dataset(str(payload.get("input_dir") or ""), recursive=bool(payload.get("recursive"))))
        return checks

    if skill_id == "differential_lora_train":
        checks.append(inspect_dataset(str(payload.get("folder_a") or "")))
        checks.append(inspect_dataset(str(payload.get("folder_b") or "")))
    elif skill_id in {"anima_lora_train", "anima_fast_train"}:
        checks.append(inspect_dataset(str(payload.get("train_data_dir") or "")))

    for key in ("pretrained_model_name_or_path", "vae", "qwen3"):
        value = str(payload.get(key) or "").strip()
        if value:
            checks.append(inspect_model_file(value))
    return checks
