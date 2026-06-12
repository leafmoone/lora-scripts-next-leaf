"""ToriiGate official user-query templates (from Minthy/ToriiGate-0.5 scripts/prompts.py)."""

from __future__ import annotations

from typing import Iterable, Optional

OFFICIAL_CAPTION_BODIES: dict[str, str] = {
    "short": (
        "The caption for image should be quite short without long purple prose and slop. "
        "Cover main objects and details."
    ),
    "long": (
        "Make a caption for given image with natural text. Use 2 to 5 paragraphs. "
        "Make your description long and vivid, mentioning all the details."
    ),
    "min_structured_md": (
        "Your answer must contain 3 parts:\n\n"
        "# 1. Thoughts about characters\n"
        "You need to think here and compare peoples/creatures that you see on the picture "
        "with given popular tags, or descriptions, or your memories for each characters "
        "to determine who is who.\n"
        'If no characters are listed in input - just write here "No named characters"\n'
        "# 2. Key details\n"
        "Here you need to write about the key details on image, prefere using regular text.\n"
        "# 3. Structured description\n"
        "## General\n"
        "Write about general composition, content of image, background and all things "
        "that are not related to characters directly.\n"
        "## Character name 1 (put here the name if any)\n"
        "Write about datails and content related to specific character, including features, "
        "poses, look, used objects, interactions, and other things.\n"
        "## Character name 2 (put here the name if any)\n"
        "Same for each character.\n"
        "## Image effects\n"
        "Mention image effect, style, camera angle\n\n"
        "In general stick to shorter descriptions."
    ),
}

VLM_PROMPT_MODES = (
    "lora",
    "official_short",
    "official_long",
    "official_min_structured_md",
)


def _clean_tags(tags: Iterable[str]) -> list[str]:
    return [str(t).strip() for t in tags if str(t).strip()]


def resolve_official_caption_type(vlm_prompt_mode: str) -> str:
    mode = str(vlm_prompt_mode or "official_short").strip().lower()
    if mode.startswith("official_"):
        return mode.replace("official_", "", 1)
    return "short"


def build_official_user_query(
    caption_type: str,
    wd14_tags: Optional[Iterable[str]] = None,
    *,
    inject_wd14_tags: bool = True,
) -> str:
    """Build ToriiGate official-style user text (HF scripts/prompts.py make_user_query)."""
    body = OFFICIAL_CAPTION_BODIES.get(caption_type) or OFFICIAL_CAPTION_BODIES["short"]
    parts = ["# Captioning format:", body, ""]

    tags = _clean_tags(wd14_tags or [])
    if inject_wd14_tags and tags:
        parts.extend(["# Booru tags for the image", f"[{', '.join(tags)}]", ""])

    parts.extend(["# Characters on picture:", "Avoid to guess names for characters.", ""])
    return "\n".join(parts).strip() + "\n"
