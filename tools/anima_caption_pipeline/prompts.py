"""Prompt builders for Anima Train two-step VLM chain."""

from __future__ import annotations

from typing import Any


def build_alias_reference_text(alias_hits: list[dict[str, Any]]) -> str:
    if not alias_hits:
        return ""
    lines = ["角色规范 tag 参考："]
    for hit in alias_hits:
        lines.append(f'- "{hit["matched_alias"]}" -> "{hit["canonical_tag"]}"')
    lines.append("如果涉及这些角色，必须直接使用右侧规范 Danbooru tag，不要自行翻译角色名。")
    return "\n".join(lines)


def build_visual_tagging_prompts(inputs: dict[str, Any]) -> tuple[str, str]:
    wd14_tags = str(inputs.get("raw_tags", "")).strip()
    style_hint = str(inputs.get("style_hint", "")).strip()
    purpose = str(inputs.get("purpose", "")).strip()
    target_profile = str(inputs.get("target_profile", "generic_tag_model")).strip()
    alias_reference = build_alias_reference_text(inputs.get("resolved_character_tags", []))
    system_prompt = (
        "You are a precise anime image tagging assistant. "
        "Inspect the provided image directly. When WD14 tags are also provided, treat them as noisy hints rather than ground truth. "
        "Return one strict JSON object only. No markdown, no explanations, no prose outside JSON."
    )
    user_prompt = f"""
请直接观察这张二次元图片并输出结构化打标 JSON。

如果同时给了 WD14 tags，请把它们当作参考输入，不要盲目照抄。

WD14 / 参考 tags：
{wd14_tags or "无"}

风格提示：
{style_hint}

用途：
{purpose}

目标输出范式：
{target_profile}

角色规范 tag 参考：
{alias_reference or "无"}

请输出一个 JSON 对象，字段至少包含：
{{
  "normalized_tags_en": ["string"],
  "expanded_tags_en": ["string"],
  "character_tags_en": ["string"],
  "appearance_tags_en": ["string"],
  "outfit_tags_en": ["string"],
  "expression_tags_en": ["string"],
  "pose_tags_en": ["string"],
  "camera_tags_en": ["string"],
  "style_tags_en": ["string"],
  "quality_tags_en": ["string"],
  "negative_tags_en": ["string"],
  "caption_short_en": "string",
  "caption_long_en": "string",
  "natural_language_en": "string",
  "notes_cn": "string"
}}

要求：
- 基于图像内容判断，不要仅复述 WD14
- tag 以 Danbooru / anime drawing tags 为主
- natural_language_en / caption_long_en 要比 tag 更自然，但不要写成摄影散文
- 如果角色规范 tag 参考不为空，涉及这些角色时优先使用 canonical Danbooru tag
- 输出必须是一个 JSON 对象，首字符是 {{，尾字符是 }}
""".strip()
    return system_prompt, user_prompt


def build_refine_wd14_prompts(inputs: dict[str, Any]) -> tuple[str, str]:
    wd14_tags = str(inputs.get("raw_tags", "")).strip()
    style_hint = str(inputs.get("style_hint", "")).strip()
    purpose = str(inputs.get("purpose", "")).strip()
    target_profile = str(inputs.get("target_profile", "generic_tag_model")).strip()
    alias_reference = build_alias_reference_text(inputs.get("resolved_character_tags", []))
    system_prompt = (
        "You are an anime tag cleanup assistant specialized in WD14-assisted labeling. "
        "Use the provided image and WD14 tags together. "
        "Resolve obvious noise, repair canonical Danbooru tags, and add concise natural-language support text. "
        "Output one JSON object only."
    )
    user_prompt = f"""
请把这组 WD14 打标结果整理成更适合二次元生产和训练的结构化 JSON。

WD14 原始 tags：
{wd14_tags}

风格提示：
{style_hint}

用途：
{purpose}

目标输出范式：
{target_profile}

角色规范 tag 参考：
{alias_reference or "无"}

要求：
- 以 WD14 原始标签为主，不要大幅重写成另一套标签体系
- 优先保留 WD14 已经可用的标签，只做少量校正、去重、补充
- 去掉明显噪声、重复、错误归类
- 同时补一段简洁自然语言，便于 Anima / caption / 训练说明
- 如果图片能帮助判断，请优先依据图片修正 WD14

请输出一个 JSON 对象，字段至少包含：
{{
  "wd14_tags_raw": ["string"],
  "normalized_tags_en": ["string"],
  "expanded_tags_en": ["string"],
  "character_tags_en": ["string"],
  "appearance_tags_en": ["string"],
  "outfit_tags_en": ["string"],
  "expression_tags_en": ["string"],
  "pose_tags_en": ["string"],
  "camera_tags_en": ["string"],
  "style_tags_en": ["string"],
  "quality_tags_en": ["string"],
  "negative_tags_en": ["string"],
  "caption_short_en": "string",
  "caption_long_en": "string",
  "natural_language_en": "string",
  "notes_cn": "string"
}}
""".strip()
    return system_prompt, user_prompt


def build_natural_caption_prompts(inputs: dict[str, Any]) -> tuple[str, str]:
    raw_tags = str(inputs.get("raw_tags", "")).strip()
    raw_text = str(inputs.get("raw_text", "")).strip()
    purpose = str(inputs.get("purpose", "")).strip()
    target_profile = str(inputs.get("target_profile", "flux_natural_language_v1")).strip()
    system_prompt = (
        "You are an anime image caption writer. "
        "Use tags and the image together, then write clean generation-friendly natural language. "
        "Return one strict JSON object only."
    )
    user_prompt = f"""
请为这张图生成更自然的二次元英文描述，并严格输出 JSON。

参考 tags：
{raw_tags or raw_text or "无"}

用途：
{purpose}

目标输出范式：
{target_profile}

请输出一个 JSON 对象，字段至少包含：
{{
  "caption_short_en": "string",
  "caption_long_en": "string",
  "natural_language_en": "string",
  "notes_cn": "string"
}}

要求：
- 以现有 tags 为主要事实依据来写自然语言
- 不要重新发明一套和 tags 不一致的内容
- caption_short_en 用于简短说明
- caption_long_en 用于 Flux / 纯自然语言工作流
- natural_language_en 可以与 caption_long_en 接近，但要保持清晰、稳定、便于模型理解
- 不要写成小说，不要使用过度修辞
- 不要重复输出完整 tags 列表
- 只输出一个很短的 JSON，不要在 JSON 之外添加任何内容
- 输出必须是 JSON
""".strip()
    return system_prompt, user_prompt
