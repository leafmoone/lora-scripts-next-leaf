"""Tests for WD14 tag injection into VLM prompts in Smart batch phase."""

import sys
from pathlib import Path

TAGGER_DIR = Path(__file__).resolve().parents[1] / "tools" / "differential_tagger"
sys.path.insert(0, str(TAGGER_DIR))

from smart_tag import (  # noqa: E402
    SmartTagRequest,
    _build_vlm_prompt_for_prep,
    _run_vlm_batch_phase,
)


def test_build_vlm_prompt_includes_wd14_tags_when_enabled():
    req = SmartTagRequest(
        training_purpose="character",
        vlm_prompt_mode="lora",
        inject_wd14_tags=True,
        enable_vlm=True,
    )
    prep = {
        "general_names": ["1girl", "solo"],
        "copyright_names": [],
        "character_names": ["hatsune_miku"],
    }
    prompt = _build_vlm_prompt_for_prep(prep, req)
    assert "1girl" in prompt
    assert "hatsune_miku" in prompt


def test_build_vlm_prompt_omits_wd14_tags_when_disabled():
    req = SmartTagRequest(
        training_purpose="character",
        vlm_prompt_mode="lora",
        inject_wd14_tags=False,
        enable_vlm=True,
    )
    prep = {
        "general_names": ["1girl", "solo"],
        "copyright_names": [],
        "character_names": ["hatsune_miku"],
    }
    prompt = _build_vlm_prompt_for_prep(prep, req)
    assert "1girl" not in prompt
    assert "hatsune_miku" not in prompt


def test_run_vlm_batch_phase_passes_per_image_prompts():
    captured: list[list[str] | None] = []

    class FakeTagger:
        def tag_batch(self, image_paths, **kwargs):
            captured.append(kwargs.get("user_prompts"))
            return [{"raw_text": f"caption-{i}"} for i in range(len(image_paths))]

    req = SmartTagRequest(enable_vlm=True, inject_wd14_tags=True, vlm_prompt_mode="lora")
    wd14_prepared = [
        {"general_names": ["tag_a"], "copyright_names": [], "character_names": []},
        {"general_names": ["tag_b"], "copyright_names": [], "character_names": []},
    ]
    nl_texts = _run_vlm_batch_phase(
        ["a.jpg", "b.jpg"],
        wd14_prepared,
        FakeTagger(),
        req,
        vlm_batch_size=2,
        progress_callback=None,
    )
    assert nl_texts == ["caption-0", "caption-1"]
    assert captured[0] is not None
    assert len(captured[0]) == 2
    assert "tag_a" in captured[0][0]
    assert "tag_b" in captured[0][1]


def test_run_vlm_batch_phase_uses_shared_prompt_when_inject_disabled():
    captured: list[str | None] = []

    class FakeTagger:
        def tag_batch(self, image_paths, **kwargs):
            captured.append(kwargs.get("user_prompt"))
            return [{"raw_text": "ok"} for _ in image_paths]

    req = SmartTagRequest(enable_vlm=True, inject_wd14_tags=False, vlm_prompt_mode="lora")
    wd14_prepared = [
        {"general_names": ["tag_a"], "copyright_names": [], "character_names": []},
    ]
    _run_vlm_batch_phase(
        ["a.jpg"],
        wd14_prepared,
        FakeTagger(),
        req,
        vlm_batch_size=1,
        progress_callback=None,
    )
    assert captured[0] is not None
    assert "tag_a" not in captured[0]
