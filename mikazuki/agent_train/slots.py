from __future__ import annotations

import re
from typing import Any


PATH_RE = re.compile(r"(/[^ \n\r\t，。；;,]+)")
ASSIGNMENT_RE = re.compile(r"([A-Za-z0-9_./-]+)\s*(?:是|=|:|：)\s*([^ \n\r\t，。；;,]+)")
TRAILING_ACTION_WORDS = ("打标", "标注", "训练")


SLOT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "tagger_leaf_batch_caption": {
        "required": ("path",),
        "labels": {"path": "图片目录"},
    },
    "differential_lora_train": {
        "required": ("folder_a", "folder_b", "pretrained_model_name_or_path", "vae", "qwen3"),
        "labels": {
            "folder_a": "原风格图片目录 folder_a",
            "folder_b": "目标风格图片目录 folder_b",
            "pretrained_model_name_or_path": "Anima 底模路径",
            "vae": "VAE 路径",
            "qwen3": "Qwen3 文本模型路径",
        },
    },
    "anima_lora_train": {
        "required": ("path", "pretrained_model_name_or_path", "vae", "qwen3"),
        "labels": {
            "path": "训练图片目录",
            "pretrained_model_name_or_path": "Anima 底模路径",
            "vae": "VAE 路径",
            "qwen3": "Qwen3 文本模型路径",
        },
    },
    "anima_fast_train": {
        "required": ("path", "pretrained_model_name_or_path", "vae", "qwen3"),
        "labels": {
            "path": "训练图片目录",
            "pretrained_model_name_or_path": "Anima 底模路径",
            "vae": "VAE 路径",
            "qwen3": "Qwen3 文本模型路径",
        },
    },
}

SLOT_ALIASES = {
    "path": "path",
    "train_data_dir": "path",
    "input_dir": "path",
    "图片目录": "path",
    "训练目录": "path",
    "folder_a": "folder_a",
    "foldera": "folder_a",
    "a": "folder_a",
    "folder_b": "folder_b",
    "folderb": "folder_b",
    "b": "folder_b",
    "base": "pretrained_model_name_or_path",
    "model": "pretrained_model_name_or_path",
    "底模": "pretrained_model_name_or_path",
    "pretrained_model_name_or_path": "pretrained_model_name_or_path",
    "vae": "vae",
    "qwen3": "qwen3",
}


def _clean_value(value: str) -> str:
    cleaned = value.strip().strip("\"'，。；;,")
    for word in TRAILING_ACTION_WORDS:
        if cleaned.endswith(word) and len(cleaned) > len(word):
            cleaned = cleaned[: -len(word)]
    return cleaned


def _extract_assignments(message: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    for key, value in ASSIGNMENT_RE.findall(message):
        slot = SLOT_ALIASES.get(key.strip().lower()) or SLOT_ALIASES.get(key.strip())
        if slot:
            slots[slot] = _clean_value(value)
    return slots


def _extract_first_path(message: str) -> str:
    match = PATH_RE.search(message)
    return _clean_value(match.group(1)) if match else ""


def extract_slots(skill_id: str, message: str, existing_slots: dict[str, Any] | None = None) -> dict[str, str]:
    slots = {k: str(v) for k, v in (existing_slots or {}).items() if v not in (None, "")}
    found = _extract_assignments(message)
    slots.update(found)

    path = _extract_first_path(message)
    if path:
        if skill_id == "differential_lora_train":
            if "folder_b" in found:
                slots["folder_b"] = found["folder_b"]
            elif "folder_a" not in slots:
                slots["folder_a"] = path
        elif "path" not in slots and path not in found.values():
            slots["path"] = path
    return slots


def missing_slots(skill_id: str, slots: dict[str, Any]) -> list[str]:
    required = SLOT_DEFINITIONS.get(skill_id, {}).get("required", ())
    return [slot for slot in required if not str(slots.get(slot) or "").strip()]


def missing_slot_message(skill_id: str, missing: list[str]) -> str:
    labels = SLOT_DEFINITIONS.get(skill_id, {}).get("labels", {})
    names = [labels.get(slot, slot) for slot in missing]
    return "还需要补充这些关键参数后才能生成可执行计划：" + "、".join(names) + "。"


def build_slot_result(skill_id: str, message: str, existing_slots: dict[str, Any] | None = None) -> dict[str, Any]:
    slots = extract_slots(skill_id, message, existing_slots)
    missing = missing_slots(skill_id, slots)
    return {
        "slots": slots,
        "missing_slots": missing,
        "can_execute": not missing,
        "assistant_message": "" if not missing else missing_slot_message(skill_id, missing),
    }


def apply_slots_to_payload(skill_id: str, payload: dict[str, Any], slots: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    path = str(slots.get("path") or "").strip()
    if skill_id == "tagger_leaf_batch_caption" and path:
        merged["input_dir"] = path
        merged["output_dir"] = path
    elif skill_id == "differential_lora_train":
        for key in ("folder_a", "folder_b", "pretrained_model_name_or_path", "vae", "qwen3"):
            if slots.get(key):
                merged[key] = slots[key]
        if slots.get("folder_a") and not slots.get("tag_dir"):
            merged["tag_dir"] = slots["folder_a"]
    elif skill_id in {"anima_lora_train", "anima_fast_train"}:
        if path:
            merged["train_data_dir"] = path
            if skill_id == "anima_fast_train":
                merged["source_image_dir"] = path
        for key in ("pretrained_model_name_or_path", "vae", "qwen3"):
            if slots.get(key):
                merged[key] = slots[key]
    return merged
