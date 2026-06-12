"""
Smart Tag pipeline: WD14/OppaiOracle + ToriiGate VLM + noise-strip + trigger inject.

Pipeline stages:
    1. Local booru tagging (WD14 / OppaiOracle / Camie)
    2. Noise tag stripping (quality / score / safety / meta / time)
    3. ToriiGate VLM natural-language caption
    4. Final caption assembly: [trigger] [character_tags] [general_tags], [NL_text]

Supports multi-tagger consensus (T-power-PR2) for booru tagging.
"""
from __future__ import annotations

import gc
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from config import ALLOWED_IMAGE_EXTENSIONS, DEFAULT_TAGGER_MODEL, TAGGER_MODELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Noise-tag vocabularies (danbooru / Pony-style training conventions)
# ---------------------------------------------------------------------------

QUALITY_NOISE_TAGS: frozenset = frozenset({
    "masterpiece", "best quality", "good quality", "normal quality",
    "low quality", "worst quality", "high quality", "high_quality",
    "best_quality", "lowres", "highres", "absurdres",
})

SCORE_NOISE_TAGS: frozenset = frozenset(
    {f"score_{i}" for i in range(1, 10)}
    | {"score_9_up", "score_8_up", "score_7_up", "score_6_up"}
    | {f"score {i}" for i in range(1, 10)}
)

SAFETY_NOISE_TAGS: frozenset = frozenset({
    "safe", "sensitive", "questionable", "nsfw", "explicit",
    "rating:general", "rating:sensitive", "rating:questionable", "rating:explicit",
})

META_NOISE_TAGS: frozenset = frozenset({
    "anime", "illustration", "anime screenshot", "anime_screenshot",
    "jpeg artifacts", "jpeg_artifacts", "official art", "official_art",
    "sketch", "monochrome", "greyscale", "grayscale",
})

TIME_NOISE_TAGS: frozenset = frozenset({
    "newest", "recent", "mid", "early", "old",
})

DEFAULT_NOISE_TAGS: frozenset = (
    QUALITY_NOISE_TAGS | SCORE_NOISE_TAGS | SAFETY_NOISE_TAGS
    | META_NOISE_TAGS | TIME_NOISE_TAGS
)

_SCORE_RE: re.Pattern = re.compile(r"^score[\s_]\d+(_up)?$", re.IGNORECASE)
_YEAR_RE: re.Pattern = re.compile(r"^(?:year\s*)?\d{4}$", re.IGNORECASE)
_SYMBOLIC_TAG_RE: re.Pattern = re.compile(
    r"^(?:[:;=][a-z0-9]?|[xX][dDpP3]|[<>^@!;:=_\\/-]{2,}|[<>^@!;:=_\\/-]+[a-z0-9])$"
)
SYMBOL_NOISE_TAGS: frozenset = frozenset({
    ":3", ":d", ":o", ":p", ":q", ":t", ":i", ";3", ";d", ";p",
    ">_<", "<_<", ">_>", "-_-", "^_^", "^^^", "@_@", "=_=", "!?",
})


def is_noise_tag(tag: str, noise_set: Iterable[str] = DEFAULT_NOISE_TAGS) -> bool:
    """Return True if ``tag`` should be stripped before VLM / final caption."""
    lowered = (tag or "").strip().lower()
    if not lowered:
        return True
    if lowered in noise_set:
        return True
    if _SCORE_RE.match(lowered) or _YEAR_RE.match(lowered):
        return True
    if lowered in SYMBOL_NOISE_TAGS or _SYMBOLIC_TAG_RE.match(lowered):
        return True
    return False


def filter_noise_tags(
    tags: List[str], noise_set: Iterable[str] = DEFAULT_NOISE_TAGS
) -> Tuple[List[str], int]:
    """Return ``(kept_tags, stripped_count)`` with noise entries dropped."""
    noise_lower = {n.lower() for n in noise_set}
    kept: List[str] = []
    stripped = 0
    for t in tags:
        if is_noise_tag(t, noise_lower):
            stripped += 1
        else:
            kept.append(t)
    return kept, stripped


def compute_consensus_tags(
    per_tagger_outputs: List[Dict[str, Any]],
    *,
    consensus_min: int = 2,
    skip_categories: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Fuse outputs of N taggers via weighted voting + per-category bypass.

    Each ``per_tagger_outputs`` entry::

        {
            "model": str,
            "weight": float,            # 0.0-1.0, defaults to 1.0
            "general_tags":   [{tag, confidence, category}, ...],
            "character_tags": [...],
            "rating": {label, score} | str,
        }

    Voting rule per tag:
      - sum of weights from taggers that produced it (above their own
        threshold) is >= ``consensus_min``
      - OR the tag's category is in ``skip_categories`` (default
        ``{'character', 'copyright'}``) -- OR semantics for character/copyright.
    """
    skip = set(
        s.lower() for s in (
            skip_categories if skip_categories is not None
            else {"character", "copyright"}
        )
    )
    consensus_min = max(1, int(consensus_min or 1))

    accum: Dict[str, Dict[str, Any]] = {}

    for output in per_tagger_outputs or []:
        weight = float(output.get("weight") or 1.0)
        for category_key, category_label in (
            ("general_tags", "general"),
            ("copyright_tags", "copyright"),
            ("character_tags", "character"),
        ):
            for tag_row in (output.get(category_key) or []):
                if isinstance(tag_row, dict):
                    name = str(tag_row.get("tag") or "").strip()
                    conf = float(tag_row.get("confidence") or 0.0)
                    cat = str(tag_row.get("category") or category_label).lower()
                else:
                    name = str(tag_row or "").strip()
                    conf = 1.0
                    cat = category_label
                if not name:
                    continue
                key = name.lower()
                slot = accum.setdefault(key, {
                    "tag": name,
                    "category": cat,
                    "votes": 0,
                    "weight_sum": 0.0,
                    "max_conf": 0.0,
                    "first_category": category_label,
                })
                slot["votes"] += 1
                slot["weight_sum"] += weight
                if conf > slot["max_conf"]:
                    slot["max_conf"] = conf

    general: List[Dict[str, Any]] = []
    copyright: List[Dict[str, Any]] = []
    character: List[Dict[str, Any]] = []

    for slot in accum.values():
        category = slot["first_category"]
        bypass = category in skip
        if not bypass and slot["weight_sum"] < float(consensus_min):
            continue
        rendered = {
            "tag": slot["tag"],
            "confidence": round(slot["max_conf"], 4) if slot["max_conf"] else 1.0,
            "category": category,
            "votes": slot["votes"],
        }
        if category == "character":
            character.append(rendered)
        elif category == "copyright":
            copyright.append(rendered)
        else:
            general.append(rendered)

    # Rating: pick the one with the highest score across all taggers.
    best_rating = ""
    best_rating_score = -1.0
    for output in per_tagger_outputs or []:
        rating = output.get("rating")
        if not rating:
            continue
        if isinstance(rating, dict):
            label = str(rating.get("label") or "").strip()
            score = float(rating.get("score") or 0.0)
        else:
            label = str(rating).strip()
            score = 1.0
        if label and score > best_rating_score:
            best_rating = label
            best_rating_score = score

    return {
        "general_tags": general,
        "copyright_tags": copyright,
        "character_tags": character,
        "rating": best_rating,
    }


# ---------------------------------------------------------------------------
# VLM prompt presets (training-purpose specific)
# ---------------------------------------------------------------------------

PROMPT_PRESETS: Dict[str, str] = {
    "style": (
        "Task: produce the natural-language portion of a LoRA training "
        "caption that targets STYLE. The text encoder must learn the visual "
        "style of this image, not its specific subject.\n\n"
        "Output 2-3 plain English sentences that cover:\n"
        "  - rendering medium and technique (linework weight, shading style, "
        "screentone, painterly vs vector, palette saturation and temperature)\n"
        "  - lighting and color mood (golden hour, neon, dramatic rim, overcast, etc.)\n"
        "  - composition and framing (close portrait, full body, low angle, dynamic crop)\n"
        "  - subject only at a high level (single figure in motion, group scene); "
        "do not list clothing pieces, accessories, or character-specific traits\n\n"
        "Rules: no leading trigger word or label, no headers, no empty praise like "
        "\"stunning\" or \"gorgeous\".\n\n"
        "WD14 tags for grounding (do not parrot them back literally): {tags}"
    ),
    "character": (
        "Task: produce the natural-language portion of a LoRA training "
        "caption that targets a CHARACTER. The character's fixed identity is "
        "learned from the trained weights, so duplicating it in captions hurts "
        "training. Write only about what changes across images.\n\n"
        "Output 2-3 plain English sentences focused on:\n"
        "  - pose, action, and facial expression of the moment\n"
        "  - position and orientation within the frame\n"
        "  - background, setting, time of day\n"
        "  - shot framing (close-up, full body, over the shoulder, from behind)\n"
        "  - lighting and overall mood\n\n"
        "Do not describe: hair color, eye color, hair style or length, the "
        "character's signature outfit, or any other fixed identity feature. "
        "No leading trigger word, no headers, no labels.\n\n"
        "WD14 tags for grounding: {tags}"
    ),
    "general": (
        "Task: write 2-3 plain English sentences describing this image for use as "
        "the natural-language portion of a LoRA training caption. Cover the visible "
        "subject, the pose or action, clothing, background, lighting, and overall "
        "composition. No headers, no labels, no trigger word.\n\n"
        "WD14 tags for grounding: {tags}"
    ),
    "concept": (
        "Task: write 2-3 plain English sentences for a CONCEPT LoRA caption. "
        "Center the description on the concept being trained (the object, action, "
        "setting, or visual effect that varies across the dataset) and how it "
        "appears in this specific image. Cover composition, lighting, and just "
        "enough subject context to anchor the concept. No headers, no labels, no "
        "trigger word.\n\n"
        "WD14 tags for grounding: {tags}"
    ),
}

TRAINING_PURPOSE_ALIASES: Dict[str, str] = {
    "style": "style", "style_lora": "style", "art": "style", "art_style": "style",
    "character": "character", "character_lora": "character", "char": "character",
    "general": "general",
    "concept": "concept", "concept_lora": "concept",
    "nsfw": "general", "nsfw_lora": "general",
}


def normalize_training_purpose(value: Optional[str]) -> str:
    """Map a user-provided training purpose to a canonical preset key."""
    if not value:
        return "general"
    key = str(value).strip().lower().replace("-", "_")
    return TRAINING_PURPOSE_ALIASES.get(key, "general")


def build_vlm_prompt(
    training_purpose: str,
    wd14_tags: List[str],
    *,
    include_tags: bool = True,
) -> str:
    """Render the per-image VLM prompt for the given training purpose."""
    canonical = normalize_training_purpose(training_purpose)
    template = PROMPT_PRESETS.get(canonical) or PROMPT_PRESETS["general"]
    if include_tags:
        cleaned, _stripped = filter_noise_tags(wd14_tags)
    else:
        cleaned = []
    return template.replace("{tags}", ", ".join(cleaned))


def build_vlm_user_prompt(
    *,
    vlm_prompt_mode: str,
    training_purpose: str,
    wd14_tags: list[str],
    inject_wd14_tags: bool = True,
) -> str:
    """Build ToriiGate user prompt from frontend/CLI smart-tag options."""
    mode = str(vlm_prompt_mode or "lora").strip().lower()
    if mode == "lora":
        return build_vlm_prompt(
            training_purpose,
            wd14_tags,
            include_tags=inject_wd14_tags,
        )
    if mode.startswith("official_"):
        from toriigate_prompts import build_official_user_query, resolve_official_caption_type

        return build_official_user_query(
            resolve_official_caption_type(mode),
            wd14_tags,
            inject_wd14_tags=inject_wd14_tags,
        )
    from toriigate_prompts import build_official_user_query

    return build_official_user_query("short", wd14_tags, inject_wd14_tags=inject_wd14_tags)


# ---------------------------------------------------------------------------
# Caption assembly + trigger injection
# ---------------------------------------------------------------------------


def _normalize_tag(tag: str) -> str:
    """Normalize a single tag: strip, lowercase, swap underscores to spaces."""
    stripped = (tag or "").strip()
    if not stripped:
        return ""
    if _SCORE_RE.match(stripped.lower()):
        return stripped.lower()
    return stripped.replace("_", " ").lower()


def _dedupe_preserving_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def assemble_caption(
    *,
    rating: Optional[str],
    general_tags: List[str],
    character_tags: List[str],
    nl_text: str,
    trigger_word: Optional[str],
    auto_strip_noise: bool,
    include_rating_prefix: bool = False,
) -> str:
    """Assemble the final training caption.

    Layout:
        [trigger] [character_tags] [general_tags], [NL_text]
    """
    pieces: List[str] = []
    nl = (nl_text or "").strip()

    general_norm = [_normalize_tag(t) for t in (general_tags or []) if t]
    character_norm = [_normalize_tag(t) for t in (character_tags or []) if t]

    if auto_strip_noise:
        general_norm, _ = filter_noise_tags(general_norm)
        character_norm, _ = filter_noise_tags(character_norm)

    general_norm = _dedupe_preserving_order(general_norm)
    character_norm = _dedupe_preserving_order(character_norm)

    trigger_clean = (trigger_word or "").strip().lower()
    if trigger_clean:
        already_present = any(
            t.strip().lower() == trigger_clean
            for t in general_norm + character_norm
        )
        if not already_present:
            pieces.append(trigger_clean)

    if include_rating_prefix and rating:
        rating_norm = str(rating).strip().lower()
        if rating_norm and rating_norm != "unknown":
            pieces.append(rating_norm)

    pieces.extend(character_norm)
    pieces.extend(general_norm)

    tag_section = ", ".join(_dedupe_preserving_order(pieces))
    if nl and tag_section:
        return f"{tag_section}, {nl}"
    if nl:
        return nl
    return tag_section


# ---------------------------------------------------------------------------
# Helper data class for tracking
# ---------------------------------------------------------------------------


@dataclass
class SmartTagResult:
    """Result for a single processed image."""
    image_path: str
    caption: str = ""
    general_tags: List[str] = field(default_factory=list)
    copyright_tags: List[str] = field(default_factory=list)
    character_tags: List[str] = field(default_factory=list)
    rating: Optional[str] = None
    nl_text: str = ""
    noise_stripped_count: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _tagger_defaults(model_name: str) -> Dict[str, Any]:
    name = str(model_name or "").strip().lower()
    if not name:
        name = DEFAULT_TAGGER_MODEL
    if name == "oppai-oracle":
        name = "oppai-oracle-v1.1"
    config = TAGGER_MODELS.get(name, {})
    general = float(config.get("default_threshold", 0.35))
    character = float(config.get("default_character_threshold", 0.85))
    copyright = float(config.get("default_copyright_threshold", general))
    max_tags = int(config.get("default_max_tags_per_image", 0) or 0)
    return {
        "general_threshold": general,
        "character_threshold": character,
        "copyright_threshold": copyright,
        "max_tags_per_image": max_tags,
    }


def _flatten_tag_names(items: List[Any]) -> List[str]:
    out: List[str] = []
    for item in items or []:
        if isinstance(item, dict):
            tag = item.get("tag")
            if tag:
                out.append(str(tag))
        elif isinstance(item, str):
            out.append(item)
    return out


def _apply_blacklist_names(names: List[str], blacklist: Optional[List[str]]) -> List[str]:
    if not blacklist:
        return names
    blocked = {str(t).strip().lower() for t in blacklist if str(t).strip()}
    if not blocked:
        return names
    return [t for t in names if str(t).strip().lower() not in blocked]


def _normalize_tag_rows(items: List[Any], category: str) -> List[Dict[str, Any]]:
    """Keep model confidence rows intact while accepting legacy string tags."""
    rows: List[Dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, dict):
            tag = str(item.get("tag") or "").strip()
            if not tag:
                continue
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            row = dict(item)
            row["tag"] = tag
            row["confidence"] = confidence
            row["category"] = str(row.get("category") or category)
            rows.append(row)
        elif item:
            tag = str(item).strip()
            if tag:
                rows.append({"tag": tag, "confidence": 1.0, "category": category})
    return rows


def _strip_noise_tag_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    stripped = 0
    for row in rows:
        if is_noise_tag(str(row.get("tag") or "")):
            stripped += 1
        else:
            kept.append(row)
    return kept, stripped


def _top_tag_rows(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if not limit or limit <= 0 or len(rows) <= limit:
        return rows
    return sorted(rows, key=lambda row: -float(row.get("confidence") or 0.0))[:limit]


def _prepare_smart_tag_rows(
    general_rows: List[Dict[str, Any]],
    copyright_rows: List[Dict[str, Any]],
    character_rows: List[Dict[str, Any]],
    *,
    auto_strip_noise: bool,
    max_tags_per_image: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int]:
    noise_stripped = 0
    if auto_strip_noise:
        general_rows, g_stripped = _strip_noise_tag_rows(general_rows)
        copyright_rows, c_stripped = _strip_noise_tag_rows(copyright_rows)
        character_rows, ch_stripped = _strip_noise_tag_rows(character_rows)
        noise_stripped = g_stripped + c_stripped + ch_stripped

    max_tags = int(max_tags_per_image or 0)
    if max_tags <= 0:
        return general_rows, copyright_rows, character_rows, noise_stripped

    reserved_count = len(character_rows) + len(copyright_rows)
    if reserved_count >= max_tags:
        kept_reserved = _top_tag_rows(character_rows + copyright_rows, max_tags)
        return [], [
            row for row in kept_reserved
            if str(row.get("category") or "").lower() == "copyright"
        ], [
            row for row in kept_reserved
            if str(row.get("category") or "").lower() == "character"
        ], noise_stripped

    general_budget = max_tags - reserved_count
    return _top_tag_rows(general_rows, general_budget), copyright_rows, character_rows, noise_stripped


def _resolve_tagger_by_model(
    model_name: str,
    *,
    general_threshold: float,
    character_threshold: float,
    copyright_threshold: float,
    use_gpu: bool,
):
    """Factory that returns a tagger for a specific model name."""
    name = (model_name or "").strip().lower()
    if name.startswith("toriigate"):
        raise ValueError("ToriiGate cannot be used as a booru consensus tagger.")
    if name.startswith("oppai-oracle"):
        from oppai_oracle_tagger import get_oppai_oracle_tagger
        return get_oppai_oracle_tagger(
            model_name=model_name,
            threshold=general_threshold,
            character_threshold=character_threshold,
            use_gpu=use_gpu,
            force_reload=False,
        )
    from tagger import get_tagger
    return get_tagger(
        model_name=model_name or None,
        threshold=general_threshold,
        character_threshold=character_threshold,
        copyright_threshold=copyright_threshold,
        use_gpu=use_gpu,
        force_reload=False,
    )


def _tag_image_with_thresholds(tagger, image_path: str, **kwargs) -> Dict[str, Any]:
    """Call tagger.tag with the richest threshold set it supports."""
    import inspect

    tag_kwargs: Dict[str, Any] = {
        k: v for k, v in kwargs.items()
        if k in ("threshold", "character_threshold", "copyright_threshold")
    }
    try:
        params = inspect.signature(tagger.tag).parameters
        if "copyright_threshold" not in params:
            tag_kwargs.pop("copyright_threshold", None)
    except (TypeError, ValueError):
        pass
    try:
        return tagger.tag(image_path, **tag_kwargs)
    except TypeError:
        tag_kwargs.pop("copyright_threshold", None)
        return tagger.tag(image_path, **tag_kwargs)


def _tag_batch_with_thresholds(
    tagger,
    image_paths: List[str],
    *,
    preferred_batch_size: int,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Call tagger.tag_batch when available, preserving output order."""
    if not image_paths:
        return []
    if not hasattr(tagger, "tag_batch"):
        return [_tag_image_with_thresholds(tagger, path, **kwargs) for path in image_paths]

    import inspect

    batch_kwargs: Dict[str, Any] = {
        k: v for k, v in kwargs.items()
        if k in ("threshold", "character_threshold", "copyright_threshold")
    }
    batch_kwargs["preferred_batch_size"] = max(1, int(preferred_batch_size or 1))

    try:
        params = inspect.signature(tagger.tag_batch).parameters
        for key in list(batch_kwargs):
            if key not in params:
                batch_kwargs.pop(key, None)
    except (TypeError, ValueError):
        pass

    try:
        outputs = tagger.tag_batch(image_paths, **batch_kwargs)
    except TypeError:
        batch_kwargs.pop("copyright_threshold", None)
        outputs = tagger.tag_batch(image_paths, **batch_kwargs)

    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if not isinstance(outputs, list) or len(outputs) != len(image_paths):
        logger.warning(
            "Batch tagger returned %s result(s) for %s image(s); falling back to single-image tagging.",
            len(outputs) if isinstance(outputs, list) else "non-list",
            len(image_paths),
        )
        return [_tag_image_with_thresholds(tagger, path, **kwargs) for path in image_paths]
    return outputs


def _empty_wd14_prepared() -> Dict[str, Any]:
    return {
        "general_names": [],
        "copyright_names": [],
        "character_names": [],
        "rating": None,
        "noise_stripped_count": 0,
    }


def _finalize_wd14_field_names(
    general_rows: List[Dict[str, Any]],
    copyright_rows: List[Dict[str, Any]],
    character_rows: List[Dict[str, Any]],
    rating: Optional[str],
    req: SmartTagRequest,
) -> Dict[str, Any]:
    general_rows, copyright_rows, character_rows, noise_stripped = _prepare_smart_tag_rows(
        general_rows,
        copyright_rows,
        character_rows,
        auto_strip_noise=req.auto_strip_noise,
        max_tags_per_image=req.max_tags_per_image,
    )
    general_names = _flatten_tag_names(general_rows)
    copyright_names = _flatten_tag_names(copyright_rows)
    character_names = _flatten_tag_names(character_rows)
    if req.blacklist:
        general_names = _apply_blacklist_names(general_names, req.blacklist)
        copyright_names = _apply_blacklist_names(copyright_names, req.blacklist)
        character_names = _apply_blacklist_names(character_names, req.blacklist)
    return {
        "general_names": general_names,
        "copyright_names": copyright_names,
        "character_names": character_names,
        "rating": rating,
        "noise_stripped_count": noise_stripped,
    }


def _prepare_wd14_fields_from_raw(result: Dict[str, Any], req: SmartTagRequest) -> Dict[str, Any]:
    general_rows = _normalize_tag_rows(result.get("general_tags") or [], "general")
    copyright_rows = _normalize_tag_rows(result.get("copyright_tags") or [], "copyright")
    character_rows = _normalize_tag_rows(result.get("character_tags") or [], "character")
    rating = result.get("rating") or None
    return _finalize_wd14_field_names(
        general_rows, copyright_rows, character_rows, rating, req
    )


def _prepare_wd14_fields_from_consensus(
    precomputed_tagger_outputs: List[Dict[str, Any]],
    req: SmartTagRequest,
) -> Dict[str, Any]:
    fused = compute_consensus_tags(
        precomputed_tagger_outputs,
        consensus_min=req.consensus_min,
        skip_categories=req.consensus_skip_categories,
    )
    general_rows = _normalize_tag_rows(fused.get("general_tags") or [], "general")
    copyright_rows = _normalize_tag_rows(fused.get("copyright_tags") or [], "copyright")
    character_rows = _normalize_tag_rows(fused.get("character_tags") or [], "character")
    rating = fused.get("rating") or None
    return _finalize_wd14_field_names(
        general_rows, copyright_rows, character_rows, rating, req
    )


def _extract_nl_text(out: Dict[str, Any]) -> str:
    nl_text = str(out.get("raw_text") or "").strip()
    if nl_text:
        return nl_text
    return ", ".join(_flatten_tag_names(out.get("general_tags") or []))


def _finalize_smart_result(
    image_path: str,
    wd14_fields: Dict[str, Any],
    nl_text: str,
    req: SmartTagRequest,
) -> SmartTagResult:
    caption = assemble_caption(
        rating=wd14_fields.get("rating"),
        general_tags=wd14_fields["general_names"] + wd14_fields["copyright_names"],
        character_tags=wd14_fields["character_names"],
        nl_text=nl_text,
        trigger_word=req.trigger_word,
        auto_strip_noise=req.auto_strip_noise,
    )
    return SmartTagResult(
        image_path=image_path,
        caption=caption,
        general_tags=wd14_fields["general_names"],
        copyright_tags=wd14_fields["copyright_names"],
        character_tags=wd14_fields["character_names"],
        rating=wd14_fields.get("rating"),
        nl_text=nl_text,
        noise_stripped_count=int(wd14_fields.get("noise_stripped_count") or 0),
    )


def _release_ai_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_nl_tagger(
    *,
    use_gpu: bool,
    vlm_backend: str,
    vllm_api_url: str,
    vllm_model: str,
    progress_callback,
    total: int,
):
    from toriigate_vllm_tagger import get_toriigate_vllm_tagger, normalize_vlm_backend

    backend = normalize_vlm_backend(vlm_backend)
    if backend == "vllm":
        if progress_callback:
            progress_callback(0, total, "Connecting to ToriiGate vLLM server...")
        nl_tagger = get_toriigate_vllm_tagger(
            api_url=vllm_api_url or None,
            model_name=vllm_model or None,
            use_gpu=use_gpu,
            force_reload=False,
        )
    else:
        if progress_callback:
            progress_callback(0, total, "Loading ToriiGate natural-language model...")
        from toriigate_tagger import get_toriigate_tagger

        nl_tagger = get_toriigate_tagger(
            model_name="toriigate-0.5",
            use_gpu=use_gpu,
            force_reload=False,
        )
    if hasattr(nl_tagger, "load"):
        nl_tagger.load()
    return nl_tagger


def _run_wd14_single_tagger_phase(
    image_paths: List[str],
    tagger,
    req: SmartTagRequest,
    wd14_batch_size: int,
    progress_callback,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    total = len(image_paths)
    progress_chunk = min(2000, max(256, int(wd14_batch_size) * 16))

    for chunk_start in range(0, total, progress_chunk):
        chunk_end = min(total, chunk_start + progress_chunk)
        chunk_paths = image_paths[chunk_start:chunk_end]
        try:
            batch_outputs = _tag_batch_with_thresholds(
                tagger,
                chunk_paths,
                preferred_batch_size=wd14_batch_size,
                threshold=req.general_threshold,
                character_threshold=req.character_threshold,
                copyright_threshold=req.copyright_threshold,
            )
        except Exception as exc:
            logger.warning(
                "WD14 batch failed for images %s-%s: %s",
                chunk_start + 1,
                chunk_end,
                exc,
            )
            batch_outputs = []

        if len(batch_outputs) != len(chunk_paths):
            batch_outputs = [
                _tag_image_with_thresholds(
                    tagger,
                    path,
                    threshold=req.general_threshold,
                    character_threshold=req.character_threshold,
                    copyright_threshold=req.copyright_threshold,
                )
                for path in chunk_paths
            ]

        for out in batch_outputs:
            prepared.append(_prepare_wd14_fields_from_raw(out or {}, req))

        if progress_callback:
            progress_callback(chunk_end, total, f"WD14 {chunk_end}/{total}")

    return prepared


def _wd14_grounding_tag_names(wd14_fields: Dict[str, Any]) -> List[str]:
    return (
        list(wd14_fields.get("general_names") or [])
        + list(wd14_fields.get("copyright_names") or [])
        + list(wd14_fields.get("character_names") or [])
    )


def _build_vlm_prompt_for_prep(wd14_fields: Dict[str, Any], req: SmartTagRequest) -> str:
    grounding = _wd14_grounding_tag_names(wd14_fields) if req.inject_wd14_tags else []
    return build_vlm_user_prompt(
        vlm_prompt_mode=req.vlm_prompt_mode,
        training_purpose=req.training_purpose,
        wd14_tags=grounding,
        inject_wd14_tags=req.inject_wd14_tags,
    )


def _run_vlm_batch_phase(
    image_paths: List[str],
    wd14_prepared: List[Dict[str, Any]],
    nl_tagger,
    req: SmartTagRequest,
    vlm_batch_size: int,
    progress_callback,
    *,
    phase_label: str = "VLM",
) -> List[str]:
    total = len(image_paths)
    nl_texts = [""] * total
    if nl_tagger is None or not req.enable_vlm:
        return nl_texts

    use_per_image_prompts = bool(req.inject_wd14_tags)
    shared_prompt = ""
    if not use_per_image_prompts:
        shared_prompt = _build_vlm_prompt_for_prep(_empty_wd14_prepared(), req)

    batch_size = max(1, int(vlm_batch_size or 1))

    for start in range(0, total, batch_size):
        chunk_end = min(total, start + batch_size)
        chunk_paths = image_paths[start:chunk_end]
        chunk_prompts: Optional[List[str]] = None
        if use_per_image_prompts:
            chunk_prompts = [
                _build_vlm_prompt_for_prep(wd14_prepared[start + offset], req)
                for offset in range(len(chunk_paths))
            ]
        try:
            batch_kwargs: Dict[str, Any] = {"preferred_batch_size": batch_size}
            if chunk_prompts is not None:
                batch_kwargs["user_prompts"] = chunk_prompts
            else:
                batch_kwargs["user_prompt"] = shared_prompt

            if hasattr(nl_tagger, "tag_batch"):
                outputs = nl_tagger.tag_batch(chunk_paths, **batch_kwargs)
            elif chunk_prompts is not None:
                outputs = [
                    nl_tagger.tag(path, user_prompt=chunk_prompts[offset])
                    for offset, path in enumerate(chunk_paths)
                ]
            else:
                outputs = [
                    nl_tagger.tag(path, user_prompt=shared_prompt)
                    for path in chunk_paths
                ]
            for offset, out in enumerate(outputs):
                nl_texts[start + offset] = _extract_nl_text(out or {})
        except Exception as exc:
            logger.warning(
                "VLM batch failed for images %s-%s: %s",
                start + 1,
                chunk_end,
                exc,
            )

        if progress_callback:
            progress_callback(
                chunk_end,
                total,
                f"{phase_label} {chunk_end}/{total}",
            )

    return nl_texts


def _process_one_image(
    *,
    image_path: str,
    req: SmartTagRequest,
    tagger=None,
    nl_tagger=None,
    precomputed_tagger_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the full per-image pipeline. Returns a dict with caption + tags."""

    # ------- Stage 1: Local booru tagging --------
    general_rows: List[Dict[str, Any]] = []
    copyright_rows: List[Dict[str, Any]] = []
    character_rows: List[Dict[str, Any]] = []
    rating: Optional[str] = None

    if precomputed_tagger_outputs is not None:
        fused = compute_consensus_tags(
            precomputed_tagger_outputs,
            consensus_min=req.consensus_min,
            skip_categories=req.consensus_skip_categories,
        )
        general_rows = _normalize_tag_rows(fused.get("general_tags") or [], "general")
        copyright_rows = _normalize_tag_rows(fused.get("copyright_tags") or [], "copyright")
        character_rows = _normalize_tag_rows(fused.get("character_tags") or [], "character")
        rating = fused.get("rating") or None
    elif req.enable_wd14 and tagger is not None:
        result = _tag_image_with_thresholds(
            tagger,
            image_path,
            threshold=req.general_threshold,
            character_threshold=req.character_threshold,
            copyright_threshold=req.copyright_threshold,
        )
        general_rows = _normalize_tag_rows(result.get("general_tags") or [], "general")
        copyright_rows = _normalize_tag_rows(result.get("copyright_tags") or [], "copyright")
        character_rows = _normalize_tag_rows(result.get("character_tags") or [], "character")
        rating = result.get("rating") or None

    general_rows, copyright_rows, character_rows, noise_stripped = _prepare_smart_tag_rows(
        general_rows, copyright_rows, character_rows,
        auto_strip_noise=req.auto_strip_noise,
        max_tags_per_image=req.max_tags_per_image,
    )
    general_names = _flatten_tag_names(general_rows)
    copyright_names = _flatten_tag_names(copyright_rows)
    character_names = _flatten_tag_names(character_rows)
    if req.blacklist:
        general_names = _apply_blacklist_names(general_names, req.blacklist)
        copyright_names = _apply_blacklist_names(copyright_names, req.blacklist)
        character_names = _apply_blacklist_names(character_names, req.blacklist)

    # ------- Stage 2: ToriiGate VLM caption --------
    nl_text = ""
    if req.enable_vlm and nl_tagger is not None:
        try:
            grounding_tags = general_names + copyright_names + character_names
            user_prompt = build_vlm_user_prompt(
                vlm_prompt_mode=req.vlm_prompt_mode,
                training_purpose=req.training_purpose,
                wd14_tags=grounding_tags,
                inject_wd14_tags=req.inject_wd14_tags,
            )
            out = nl_tagger.tag(image_path, user_prompt=user_prompt)
            nl_text = str(out.get("raw_text") or "").strip()
            if not nl_text:
                nl_text = ", ".join(_flatten_tag_names(out.get("general_tags") or []))
        except Exception as exc:
            logger.warning("ToriiGate caption failed for %s: %s", image_path, exc)
            nl_text = ""

    # ------- Stage 3: Caption assembly --------
    caption = assemble_caption(
        rating=rating,
        general_tags=general_names + copyright_names,
        character_tags=character_names,
        nl_text=nl_text,
        trigger_word=req.trigger_word,
        auto_strip_noise=req.auto_strip_noise,
    )

    return {
        "caption": caption,
        "general_tags": general_names,
        "copyright_tags": copyright_names,
        "character_tags": character_names,
        "general_tag_rows": general_rows,
        "copyright_tag_rows": copyright_rows,
        "character_tag_rows": character_rows,
        "rating": rating,
        "nl_text": nl_text,
        "noise_stripped_count": noise_stripped,
    }


@dataclass
class SmartTagRequest:
    """Input for the Smart Tag pipeline."""
    image_paths: List[str] = field(default_factory=list)
    training_purpose: str = "general"
    trigger_word: str = ""
    auto_strip_noise: bool = True
    enable_wd14: bool = True
    enable_vlm: bool = True
    tagger_model: str = ""
    use_gpu: bool = True
    general_threshold: float = 0.35
    character_threshold: float = 0.85
    copyright_threshold: float = 0.35
    max_tags_per_image: int = 0
    # Multi-tagger consensus
    taggers: List[Dict[str, Any]] = field(default_factory=list)
    consensus_min: int = 2
    consensus_skip_categories: List[str] = field(
        default_factory=lambda: ["character", "copyright"]
    )
    vlm_prompt_mode: str = "lora"
    inject_wd14_tags: bool = True
    vlm_backend: str = "transformers"
    vllm_api_url: str = ""
    vllm_model: str = ""
    blacklist: List[str] = field(default_factory=list)


def discover_images(paths: List[str], recursive: bool = False) -> List[str]:
    """Discover all image files from a list of paths (dirs + files)."""
    discovered: List[str] = []
    seen: Set[str] = set()

    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            s = str(p)
            if s not in seen:
                seen.add(s)
                discovered.append(s)
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in sorted(p.glob(pattern)):
                if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                    s = str(f)
                    if s not in seen:
                        seen.add(s)
                        discovered.append(s)

    return discovered


def run_smart_tag_pipeline(
    image_paths: List[str],
    *,
    training_purpose: str = "general",
    trigger_word: str = "",
    auto_strip_noise: bool = True,
    enable_wd14: bool = True,
    enable_vlm: bool = True,
    tagger_model: str = "",
    use_gpu: bool = True,
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
    copyright_threshold: float = 0.35,
    max_tags_per_image: int = 0,
    taggers: Optional[List[Dict[str, Any]]] = None,
    consensus_min: int = 2,
    consensus_skip_categories: Optional[List[str]] = None,
    progress_callback=None,
    wd14_batch_size: int = 8,
    vlm_batch_size: int = 4,
    vlm_backend: str = "transformers",
    vllm_api_url: str = "",
    vllm_model: str = "",
    vlm_prompt_mode: str = "lora",
    inject_wd14_tags: bool = True,
    blacklist: Optional[List[str]] = None,
) -> List[SmartTagResult]:
    """Run the full Smart Tag pipeline on a list of image paths.

    Args:
        image_paths: List of absolute paths to image files.
        training_purpose: One of: style, character, general, concept.
        trigger_word: Optional trigger word to inject at the start of captions.
        auto_strip_noise: Strip quality/score/safety/meta/time noise tags.
        enable_wd14: Enable local booru tagging stage.
        enable_vlm: Enable ToriiGate VLM caption stage.
        vlm_prompt_mode: ``lora`` or ``official_*`` ToriiGate caption template.
        inject_wd14_tags: Inject WD14 tag list into each image's VLM user prompt when enabled.
        wd14_batch_size: Batch size for WD14 inference (default 8).
        vlm_batch_size: Batch size / concurrency for ToriiGate VLM (default 4).
        vlm_backend: ``transformers`` (local HF) or ``vllm`` (OpenAI-compatible server).
        tagger_model: WD14/OppaiOracle model name for single-tagger mode.
        use_gpu: Use GPU acceleration when available.
        general_threshold: Confidence threshold for general tags.
        character_threshold: Confidence threshold for character tags.
        copyright_threshold: Confidence threshold for copyright tags.
        max_tags_per_image: Max tags per image (0 = unlimited).
        taggers: Multi-tagger config list for consensus mode.
        consensus_min: Minimum tagger weight sum for a tag to survive.
        consensus_skip_categories: Categories with OR semantics (default: character, copyright).
        progress_callback: Optional callable(current, total, message) for progress.

    Returns:
        List of SmartTagResult, one per input image.
    """
    if not image_paths:
        return []

    if not enable_wd14 and not enable_vlm:
        raise ValueError("Smart mode requires at least WD14 tagging or ToriiGate VLM.")

    req = SmartTagRequest(
        image_paths=image_paths,
        training_purpose=normalize_training_purpose(training_purpose),
        trigger_word=trigger_word,
        auto_strip_noise=auto_strip_noise,
        enable_wd14=enable_wd14,
        enable_vlm=enable_vlm,
        tagger_model=tagger_model,
        use_gpu=use_gpu,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        copyright_threshold=copyright_threshold,
        max_tags_per_image=max_tags_per_image,
        taggers=taggers or [],
        consensus_min=consensus_min,
        consensus_skip_categories=consensus_skip_categories or ["character", "copyright"],
        vlm_prompt_mode=str(vlm_prompt_mode or "lora"),
        inject_wd14_tags=bool(inject_wd14_tags),
        vlm_backend=str(vlm_backend or "transformers"),
        vllm_api_url=str(vllm_api_url or "").strip(),
        vllm_model=str(vllm_model or "").strip(),
        blacklist=list(blacklist or []),
    )

    from toriigate_vllm_tagger import normalize_vlm_backend

    vlm_backend_norm = normalize_vlm_backend(req.vlm_backend)

    results: List[SmartTagResult] = []
    total = len(image_paths)
    wd14_prepared: List[Dict[str, Any]] = []
    tagger = None

    # ── Phase 1: WD14 (all images) ─────────────────────────────
    if enable_wd14 and req.taggers:
        all_sources = list(enumerate(image_paths))
        per_image_outputs: List[List[Dict[str, Any]]] = [[] for _ in all_sources]
        tagger_count = len(req.taggers)

        for tagger_idx, entry in enumerate(req.taggers):
            model_name = str(entry.get("model") or "").strip()
            if not model_name:
                continue
            weight = float(entry.get("weight") or 1.0)
            gen_th = float(entry.get("general_threshold") or general_threshold)
            char_th = float(entry.get("character_threshold") or character_threshold)
            copy_th = float(entry.get("copyright_threshold") or copyright_threshold or gen_th)

            if progress_callback:
                progress_callback(
                    0, len(all_sources),
                    f"Loading tagger {tagger_idx + 1}/{tagger_count}: {model_name}..."
                )

            try:
                one_tagger = _resolve_tagger_by_model(
                    model_name,
                    general_threshold=gen_th,
                    character_threshold=char_th,
                    copyright_threshold=copy_th,
                    use_gpu=use_gpu,
                )
                if hasattr(one_tagger, "load"):
                    one_tagger.load()
            except Exception as exc:
                logger.warning("Failed to load tagger %s: %s", model_name, exc)
                continue

            progress_chunk = min(2000, max(256, int(wd14_batch_size) * 16))
            for chunk_start in range(0, len(all_sources), progress_chunk):
                chunk_end = min(len(all_sources), chunk_start + progress_chunk)
                chunk_paths = [path for _, path in all_sources[chunk_start:chunk_end]]
                try:
                    batch_outputs = _tag_batch_with_thresholds(
                        one_tagger,
                        chunk_paths,
                        preferred_batch_size=wd14_batch_size,
                        threshold=gen_th,
                        character_threshold=char_th,
                        copyright_threshold=copy_th,
                    )
                except Exception as exc:
                    logger.warning(
                        "Consensus tagger %s batch failed for images %s-%s: %s",
                        model_name,
                        chunk_start + 1,
                        chunk_end,
                        exc,
                    )
                    batch_outputs = []

                if len(batch_outputs) != len(chunk_paths):
                    batch_outputs = [
                        _tag_image_with_thresholds(
                            one_tagger,
                            path,
                            threshold=gen_th,
                            character_threshold=char_th,
                            copyright_threshold=copy_th,
                        )
                        for path in chunk_paths
                    ]

                for offset, out in enumerate(batch_outputs):
                    img_idx = chunk_start + offset
                    if not out:
                        continue
                    per_image_outputs[img_idx].append({
                        "model": model_name,
                        "weight": weight,
                        "general_tags": out.get("general_tags") or [],
                        "copyright_tags": out.get("copyright_tags") or [],
                        "character_tags": out.get("character_tags") or [],
                        "rating": out.get("rating"),
                    })

                if progress_callback:
                    progress_callback(
                        chunk_end, len(all_sources),
                        f"WD14 ({model_name}) {chunk_end}/{len(all_sources)}"
                    )

        wd14_prepared = [
            _prepare_wd14_fields_from_consensus(per_image_outputs[idx] or [], req)
            for idx in range(total)
        ]
    elif enable_wd14:
        if progress_callback:
            progress_callback(0, total, "Loading local booru tagger...")
        if tagger_model.startswith("oppai-oracle"):
            from oppai_oracle_tagger import get_oppai_oracle_tagger

            tagger = get_oppai_oracle_tagger(
                model_name=tagger_model,
                threshold=general_threshold,
                character_threshold=character_threshold,
                use_gpu=use_gpu,
                force_reload=False,
            )
        else:
            from tagger import get_tagger

            tagger = get_tagger(
                model_name=tagger_model or None,
                threshold=general_threshold,
                character_threshold=character_threshold,
                copyright_threshold=copyright_threshold,
                use_gpu=use_gpu,
                force_reload=False,
            )
        if hasattr(tagger, "load"):
            tagger.load()

        wd14_prepared = _run_wd14_single_tagger_phase(
            image_paths,
            tagger,
            req,
            wd14_batch_size,
            progress_callback,
        )
    else:
        wd14_prepared = [_empty_wd14_prepared() for _ in image_paths]

    if enable_wd14 and enable_vlm and vlm_backend_norm == "transformers":
        tagger = None
        _release_ai_memory()

    # ── Phase 2: VLM batch (shared prompt template) ────────────
    nl_tagger = None
    nl_texts = [""] * total
    if enable_vlm:
        nl_tagger = _load_nl_tagger(
            use_gpu=use_gpu,
            vlm_backend=req.vlm_backend,
            vllm_api_url=req.vllm_api_url,
            vllm_model=req.vllm_model,
            progress_callback=progress_callback,
            total=total,
        )
        vlm_label = "VLM" if vlm_backend_norm == "transformers" else "vLLM"
        nl_texts = _run_vlm_batch_phase(
            image_paths,
            wd14_prepared,
            nl_tagger,
            req,
            vlm_batch_size,
            progress_callback,
            phase_label=vlm_label,
        )

    # ── Phase 3: Assemble captions ───────────────────────────────
    for path, prep, nl_text in zip(image_paths, wd14_prepared, nl_texts):
        try:
            results.append(_finalize_smart_result(path, prep, nl_text, req))
        except Exception as exc:
            logger.error("Smart tag failed on %s: %s", path, exc)
            results.append(SmartTagResult(image_path=path, error=str(exc)))

    return results
