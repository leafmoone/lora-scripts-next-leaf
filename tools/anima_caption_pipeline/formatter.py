"""Format VLM JSON results into anima_train_v1 training captions."""

from __future__ import annotations

from typing import Any

from .parser import dedupe_tags, split_tag_like_text


def normalize_tag_for_anima(tag: str) -> str:
    text = str(tag or "").strip().lower()
    if not text:
        return ""
    if text.startswith("score_") or "_" in text:
        return text.replace(" ", "_")
    return text.replace("_", " ")


def build_training_two_line_text(tag_items: list[str], caption_text: str) -> str:
    first_line = ", ".join(str(item).strip() for item in tag_items if str(item).strip()).strip()
    second_line = str(caption_text or "").strip()
    if first_line and second_line:
        return first_line + "\n\n" + second_line
    return first_line or second_line


def get_training_base_tags(json_result: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    raw_tags = inputs.get("wd14_raw_tags_en", [])
    if isinstance(raw_tags, list) and raw_tags:
        return dedupe_tags([str(item).strip() for item in raw_tags if str(item).strip()])

    base_tags = (
        json_result.get("extended_tags_en", [])
        or json_result.get("canonical_tags_en", [])
        or json_result.get("expanded_tags_en", [])
        or json_result.get("normalized_tags_en", [])
    )
    return dedupe_tags([str(item).strip() for item in base_tags if str(item).strip()])


def format_anima_train_v1(json_result: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    tag_source = get_training_base_tags(json_result, inputs)
    training_tags = [normalize_tag_for_anima(tag) for tag in tag_source if normalize_tag_for_anima(tag)]
    caption_long_en = str(json_result.get("caption_long_en", "")).strip()
    natural_language_en = str(json_result.get("natural_language_en", "")).strip()
    caption_short_en = str(json_result.get("caption_short_en", "")).strip()
    training_caption = caption_long_en or natural_language_en or caption_short_en
    training_text = build_training_two_line_text(training_tags, training_caption)
    return {
        "target_profile": "anima_train_v1",
        "formatted_prompt": training_text,
        "formatted_negative_prompt": "",
        "formatted_prompt_tags": ", ".join(training_tags),
        "formatted_prompt_caption": training_caption,
        "formatted_training_text": training_text,
        "training_base_tags_en": training_tags,
        "profile_notes_cn": "Anima 训练标注格式：第一段为 tags，空一行后第二段为自然语言；不添加质量词和负面词。",
    }


def enrich_json_result(task_type: str, json_result: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(json_result)
    if inputs.get("wd14_raw_tags_en"):
        enriched["wd14_raw_tags_en"] = list(inputs.get("wd14_raw_tags_en", []))
    enriched.update(format_anima_train_v1(enriched, inputs))
    enriched["_task_agent_chain"] = inputs.get("_task_agent_chain", [])
    return enriched


def preprocess_task_inputs(task_type: str, inputs: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(inputs)
    original_raw_tags = str(prepared.get("raw_tags", "")).strip()
    if original_raw_tags:
        prepared["wd14_raw_tags_en"] = split_tag_like_text(original_raw_tags)
    return prepared
