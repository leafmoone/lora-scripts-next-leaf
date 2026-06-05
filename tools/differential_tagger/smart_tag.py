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

    # ------- Stage 2: ToriiGate VLM caption --------
    nl_text = ""
    if req.enable_vlm and nl_tagger is not None:
        try:
            out = nl_tagger.tag(image_path)
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
) -> List[SmartTagResult]:
    """Run the full Smart Tag pipeline on a list of image paths.

    Args:
        image_paths: List of absolute paths to image files.
        training_purpose: One of: style, character, general, concept.
        trigger_word: Optional trigger word to inject at the start of captions.
        auto_strip_noise: Strip quality/score/safety/meta/time noise tags.
        enable_wd14: Enable local booru tagging stage.
        enable_vlm: Enable ToriiGate VLM caption stage.
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
    )

    results: List[SmartTagResult] = []
    total = len(image_paths)

    # --- Load models ---
    tagger = None
    nl_tagger = None

    if enable_wd14:
        if req.taggers:
            # Multi-tagger mode: load each one lazily during processing
            pass
        else:
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

    if enable_vlm:
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

    # --- Multi-tagger mode: process all images per-tagger to avoid model thrashing ---
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

            for img_idx, (_, path) in enumerate(all_sources):
                try:
                    out = _tag_image_with_thresholds(
                        one_tagger, path,
                        threshold=gen_th,
                        character_threshold=char_th,
                        copyright_threshold=copy_th,
                    )
                    per_image_outputs[img_idx].append({
                        "model": model_name,
                        "weight": weight,
                        "general_tags": out.get("general_tags") or [],
                        "copyright_tags": out.get("copyright_tags") or [],
                        "character_tags": out.get("character_tags") or [],
                        "rating": out.get("rating"),
                    })
                except Exception as exc:
                    logger.warning(
                        "Consensus tagger %s failed on %s: %s", model_name, path, exc
                    )
                if progress_callback:
                    progress_callback(
                        img_idx + 1, len(all_sources),
                        f"Tagging ({model_name}) {img_idx + 1}/{len(all_sources)}"
                    )

        # Consensus + VLM per image
        for img_idx, (original_idx, path) in enumerate(all_sources):
            if progress_callback:
                progress_callback(
                    img_idx + 1, len(all_sources),
                    f"Processing {img_idx + 1}/{len(all_sources)}"
                )
            try:
                raw = _process_one_image(
                    image_path=path,
                    req=req,
                    tagger=None,
                    nl_tagger=nl_tagger,
                    precomputed_tagger_outputs=per_image_outputs[img_idx] or None,
                )
                results.append(SmartTagResult(
                    image_path=path,
                    caption=raw.get("caption", ""),
                    general_tags=raw.get("general_tags", []),
                    copyright_tags=raw.get("copyright_tags", []),
                    character_tags=raw.get("character_tags", []),
                    rating=raw.get("rating"),
                    nl_text=raw.get("nl_text", ""),
                    noise_stripped_count=raw.get("noise_stripped_count", 0),
                ))
            except Exception as exc:
                logger.error("Smart tag failed on %s: %s", path, exc)
                results.append(SmartTagResult(
                    image_path=path,
                    error=str(exc),
                ))
    else:
        # Single-tagger or no-tagger mode
        for idx, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(idx + 1, total, f"Processing {idx + 1}/{total}")
            try:
                raw = _process_one_image(
                    image_path=path,
                    req=req,
                    tagger=tagger,
                    nl_tagger=nl_tagger,
                )
                results.append(SmartTagResult(
                    image_path=path,
                    caption=raw.get("caption", ""),
                    general_tags=raw.get("general_tags", []),
                    copyright_tags=raw.get("copyright_tags", []),
                    character_tags=raw.get("character_tags", []),
                    rating=raw.get("rating"),
                    nl_text=raw.get("nl_text", ""),
                    noise_stripped_count=raw.get("noise_stripped_count", 0),
                ))
            except Exception as exc:
                logger.error("Smart tag failed on %s: %s", path, exc)
                results.append(SmartTagResult(
                    image_path=path,
                    error=str(exc),
                ))

    return results
