"""Single-image two-step Anima Train VLM pipeline."""

from __future__ import annotations

import logging
from typing import Any

from .alias_index import AliasIndex
from .formatter import enrich_json_result, preprocess_task_inputs
from .parser import extract_first_json_object, fallback_json_result
from .prompts import build_natural_caption_prompts, build_visual_tagging_prompts
from .vlm_client import VlmClient

logger = logging.getLogger(__name__)


def run_vlm_task(
    client: VlmClient,
    task_type: str,
    image_path: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    if task_type == "extract_tags_from_image":
        system_prompt, user_prompt = build_visual_tagging_prompts(inputs)
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
    purpose: str = "character",
    style_hint: str = "",
    trigger: str = "",
    alias_index: AliasIndex | None = None,
) -> dict[str, Any]:
    base_inputs: dict[str, Any] = {
        "raw_tags": raw_tags,
        "purpose": purpose,
        "style_hint": style_hint,
        "trigger": trigger,
        "target_profile": "anima_train_v1",
        "_task_agent_chain": [],
    }
    if alias_index:
        base_inputs = alias_index.preprocess_inputs(base_inputs)
    base_inputs = preprocess_task_inputs("extract_tags_from_image", base_inputs)

    step1 = run_vlm_task(client, "extract_tags_from_image", image_path, base_inputs)
    chain = list(base_inputs.get("_task_agent_chain", []))
    chain.append("extract_tags_from_image")

    step2_inputs = dict(base_inputs)
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
    enriched["extract_tags_result"] = step1
    enriched["generate_caption_result"] = step2
    return enriched
