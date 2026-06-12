"""Tests for ToriiGate / Smart Tag VLM prompt wiring."""

from __future__ import annotations

import sys
from pathlib import Path

TAGGER_DIR = Path(__file__).resolve().parents[1] / "tools" / "differential_tagger"
if str(TAGGER_DIR) not in sys.path:
    sys.path.insert(0, str(TAGGER_DIR))

from smart_tag import build_vlm_prompt, build_vlm_user_prompt  # noqa: E402
from toriigate_prompts import build_official_user_query  # noqa: E402


def test_build_vlm_user_prompt_lora_includes_tags():
    prompt = build_vlm_user_prompt(
        vlm_prompt_mode="lora",
        training_purpose="character",
        wd14_tags=["1girl", "smile"],
        inject_wd14_tags=True,
    )
    assert "character" in prompt.lower()
    assert "1girl" in prompt
    assert "smile" in prompt


def test_build_vlm_user_prompt_lora_without_tag_injection():
    prompt = build_vlm_user_prompt(
        vlm_prompt_mode="lora",
        training_purpose="style",
        wd14_tags=["1girl"],
        inject_wd14_tags=False,
    )
    assert "1girl" not in prompt
    assert "style" in prompt.lower()


def test_build_vlm_user_prompt_official_short_with_tags():
    prompt = build_vlm_user_prompt(
        vlm_prompt_mode="official_short",
        training_purpose="general",
        wd14_tags=["solo", "standing"],
        inject_wd14_tags=True,
    )
    assert "# Captioning format:" in prompt
    assert "quite short" in prompt
    assert "# Booru tags for the image" in prompt
    assert "solo" in prompt


def test_build_official_user_query_matches_hf_structure():
    prompt = build_official_user_query("short", ["tag_a"], inject_wd14_tags=True)
    assert prompt.startswith("# Captioning format:")
    assert "Avoid to guess names for characters." in prompt
