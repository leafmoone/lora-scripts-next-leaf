"""JSON parsing and tag text utilities for Anima Train pipeline."""

from __future__ import annotations

import json
import re
from typing import Any


def strip_markdown_fences(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def looks_like_jsonish_output(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if stripped.startswith("{") or stripped.startswith("```json") or stripped.startswith("```"):
        return True
    markers = (
        '"normalized_tags_en"',
        '"expanded_tags_en"',
        '"caption_long_en"',
        '"natural_language_en"',
        '"quality_tags_en"',
    )
    return any(marker in stripped for marker in markers)


def extract_first_json_object(text: str) -> dict[str, Any]:
    stripped = strip_markdown_fences(text)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object start found in model output.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed

    raise ValueError("No complete JSON object found in model output.")


def split_tag_like_text(text: str) -> list[str]:
    normalized = (
        str(text or "")
        .replace("\r", "\n")
        .replace("，", ",")
        .replace("、", ",")
        .replace("；", ",")
        .replace(";", ",")
        .replace("|", ",")
    )
    pieces: list[str] = []
    for chunk in normalized.splitlines():
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith(("-", "*", "•")):
            chunk = chunk[1:].strip()
        for part in chunk.split(","):
            cleaned = part.strip().strip(".")
            if cleaned:
                pieces.append(cleaned)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in pieces:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(item)
    return deduped


def dedupe_tags(tags: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags or []:
        text = str(tag or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def compact_text(text: str, limit: int = 320) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def fallback_json_result(task_type: str, raw_text: str, inputs: dict[str, Any], parse_error: str) -> dict[str, Any]:
    note = f"模型未按 JSON 输出，已使用纯文本回退解析。parse_error={parse_error}"
    if task_type in ("extract_tags_from_image", "vision_tagging", "image_captioning", "refine_wd14_tags"):
        source_tags = split_tag_like_text(inputs.get("raw_tags", "") or raw_text)
        if looks_like_jsonish_output(raw_text):
            merged_tags = dedupe_tags(source_tags)
        else:
            merged_tags = dedupe_tags(source_tags + split_tag_like_text(raw_text))
        caption = compact_text(", ".join(merged_tags[:18]), 220)
        return {
            "normalized_tags_en": source_tags[:32],
            "expanded_tags_en": merged_tags[:48],
            "character_tags_en": inputs.get("resolved_character_tag_strings", []),
            "appearance_tags_en": merged_tags[:20],
            "outfit_tags_en": [],
            "expression_tags_en": [],
            "pose_tags_en": [],
            "camera_tags_en": [],
            "style_tags_en": [],
            "quality_tags_en": ["masterpiece", "best quality"],
            "negative_tags_en": ["low quality", "blurry", "bad anatomy"],
            "caption_short_en": caption,
            "caption_long_en": caption,
            "natural_language_en": caption,
            "notes_cn": note,
        }
    if task_type == "generate_natural_caption":
        source_tags = split_tag_like_text(inputs.get("raw_tags", "") or inputs.get("raw_text", ""))
        if looks_like_jsonish_output(raw_text):
            base_subject = ", ".join(source_tags[:12])
            caption = compact_text(f"anime illustration featuring {base_subject}", 220) if base_subject else ""
        else:
            caption = raw_text.strip() or compact_text(", ".join(source_tags[:20]), 220)
        return {
            "caption_short_en": compact_text(caption, 160),
            "caption_long_en": caption,
            "natural_language_en": caption,
            "notes_cn": note,
        }
    return {"fallback_text": raw_text.strip(), "notes_cn": note}
