"""Single-image two-step Anima Train VLM pipeline."""

from __future__ import annotations

import logging
from typing import Any

from .alias_index import AliasIndex
from .formatter import enrich_json_result, preprocess_task_inputs
from .parser import extract_first_json_object, fallback_json_result
from .prompts import build_natural_caption_prompts, build_refine_wd14_prompts, build_visual_tagging_prompts
from .vlm_client import VlmClient

logger = logging.getLogger(__name__)

DEFAULT_ANIMA_TRAIN_STYLE_HINT = "training caption, keep reliable wd14 tags, add concise natural-language description line"
DEFAULT_ANIMA_TRAIN_PURPOSE = "保留 WD14 可用标签，只让 LLM 做补充、校正和自然语言训练描述。输出不要加入绘图质量词。"


def run_vlm_task(
    client: VlmClient,
    task_type: str,
    image_path: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    if task_type == "extract_tags_from_image":
        system_prompt, user_prompt = build_visual_tagging_prompts(inputs)
    elif task_type == "refine_wd14_tags":
        system_prompt, user_prompt = build_refine_wd14_prompts(inputs)
    elif task_type == "generate_natural_caption":
        system_prompt, user_prompt = build_natural_caption_prompts(inputs)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    raw_text = client.complete(
        image_path=image_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    try:
        json_result = extract_first_json_object(raw_text)
    except Exception as exc:
        logger.warning("JSON parse failed for %s (%s): %s", image_path, task_type, exc)
        json_result = fallback_json_result(task_type, raw_text, inputs, str(exc))
    return json_result


def run_single_image_pipeline(
    client: VlmClient,
    image_path: str,
    *,
    raw_tags: str,
    purpose: str = DEFAULT_ANIMA_TRAIN_PURPOSE,
    style_hint: str = DEFAULT_ANIMA_TRAIN_STYLE_HINT,
    trigger: str = "",
    alias_index: AliasIndex | None = None,
) -> dict[str, Any]:
    base_inputs: dict[str, Any] = {
        "raw_tags": raw_tags,
        "purpose": purpose,
        "style_hint": style_hint,
        "trigger": trigger,
        "target_profile": "generic_tag_model",
        "caption_mode": "training_tag_plus_nl",
        "image_path": image_path,
        "enable_image_input": True,
        "_task_agent_chain": [],
    }
    if alias_index:
        base_inputs = alias_index.preprocess_inputs(base_inputs)
    first_task = "refine_wd14_tags"
    base_inputs = preprocess_task_inputs(first_task, base_inputs)
    original_wd14_tags = list(base_inputs.get("wd14_raw_tags_en", []))
    if original_wd14_tags:
        base_inputs["original_wd14_raw_tags_en"] = original_wd14_tags
        base_inputs["training_base_tags_en"] = original_wd14_tags

    step1 = run_vlm_task(client, first_task, image_path, base_inputs)
    chain = list(base_inputs.get("_task_agent_chain", []))
    chain.append(first_task)

    step2_inputs = dict(base_inputs)
    step1_tags = step1.get("expanded_tags_en", []) or step1.get("normalized_tags_en", [])
    if isinstance(step1_tags, list) and step1_tags:
        step2_inputs["raw_tags"] = ", ".join(str(item).strip() for item in step1_tags if str(item).strip())
    if original_wd14_tags:
        step2_inputs["original_wd14_raw_tags_en"] = original_wd14_tags
        step2_inputs["training_base_tags_en"] = original_wd14_tags
    step2_inputs["target_profile"] = "anima_train_v1"
    step2_inputs["raw_text"] = (
        step1.get("natural_language_en", "")
        or step1.get("caption_long_en", "")
        or step1.get("caption_short_en", "")
    )
    if alias_index:
        step2_inputs = alias_index.preprocess_inputs(step2_inputs)
    step2_inputs = preprocess_task_inputs("generate_natural_caption", step2_inputs)
    step2_inputs["_task_agent_chain"] = chain

    step2 = run_vlm_task(client, "generate_natural_caption", image_path, step2_inputs)
    chain.append("generate_natural_caption")

    merged_inputs = dict(step2_inputs)
    merged_inputs["_task_agent_chain"] = chain
    enriched = enrich_json_result("generate_natural_caption", step2, merged_inputs)
    enriched["refine_tags_result"] = step1
    enriched["extract_tags_result"] = step1
    enriched["generate_caption_result"] = step2
    return enriched
