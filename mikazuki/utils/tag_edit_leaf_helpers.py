"""Shared helpers for Tag-Edit-Leaf API mode and subprocess progress parsing."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

MAX_API_IMAGE_BYTES = 4 * 1024 * 1024
MAX_API_IMAGE_EDGE = 1536
DEFAULT_API_WORKERS = 2
MAX_API_WORKERS = 8
DEFAULT_API_MAX_RETRIES = 2
DEFAULT_API_TIMEOUT = 60


def prepare_image_for_api(
    img_path: Path,
    *,
    max_bytes: int = MAX_API_IMAGE_BYTES,
    max_edge: int = MAX_API_IMAGE_EDGE,
) -> tuple[bytes, str]:
    """Return (image bytes, mime subtype) for API upload, resizing if needed."""
    from PIL import Image

    raw = img_path.read_bytes()
    ext = img_path.suffix.lower().lstrip(".")
    if len(raw) <= max_bytes and ext in {"jpg", "jpeg", "png", "webp"}:
        return raw, "jpeg" if ext in {"jpg", "jpeg"} else ext

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(1.0, max_edge / max(width, height))
        if scale < 1.0:
            img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)

        quality = 85
        while quality >= 40:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                return data, "jpeg"
            quality -= 10

        while max(img.size) > 256:
            img = img.resize((max(1, img.size[0] // 2), max(1, img.size[1] // 2)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70, optimize=True)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                return data, "jpeg"

        return data, "jpeg"


def parse_progress_line(stripped: str, state: dict) -> bool:
    """Update task state from a subprocess stdout line. Returns True if handled."""
    if stripped.startswith("{") and '"type"' in stripped:
        try:
            evt = json.loads(stripped)
        except json.JSONDecodeError:
            evt = None
        if isinstance(evt, dict) and evt.get("type") == "progress":
            current = int(evt.get("current") or 0)
            total = int(evt.get("total") or 0)
            if total > 0:
                state["progress"] = round(current / total * 100)
            if evt.get("phase"):
                state["phase"] = str(evt["phase"])
            if evt.get("message"):
                state["message"] = str(evt["message"])
            return True

    match = re.search(r"\[(\d+)\s*/\s*(\d+)\]", stripped)
    if match:
        try:
            current, total = int(match.group(1)), int(match.group(2))
            if total > 0:
                state["progress"] = round(current / total * 100)
        except (ValueError, IndexError):
            pass
        return True

    match_pct = re.search(r"(?:^|\r)\s*(\d+)%", stripped)
    if match_pct:
        try:
            state["progress"] = min(int(match_pct.group(1)), 99)
        except (ValueError, IndexError):
            pass
        return True

    return False
