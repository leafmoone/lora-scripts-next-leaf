"""Unit tests for Anima Train caption pipeline (no GPU / no live vLLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PROJECT_ROOT / "tools"
sys.path.insert(0, str(PIPELINE_ROOT))

from anima_caption_pipeline.alias_index import AliasIndex, build_sqlite_index  # noqa: E402
from anima_caption_pipeline.config import resolve_vlm_endpoint  # noqa: E402
from anima_caption_pipeline.formatter import format_anima_train_v1  # noqa: E402
from anima_caption_pipeline.model_resolver import is_valid_hf_model_dir  # noqa: E402
from anima_caption_pipeline.parser import (  # noqa: E402
    extract_first_json_object,
    fallback_json_result,
    split_tag_like_text,
)
from anima_caption_pipeline.pipeline import run_single_image_pipeline  # noqa: E402
from anima_caption_pipeline.vlm_client import (  # noqa: E402
    GemmaVllmUnavailableError,
    VlmClient,
    create_vlm_client,
    is_broken_vllm_output,
    probe_vllm_generation,
)


def test_split_tag_like_text_dedupes_and_normalizes():
    text = "1girl, blue hair\nred eyes, blue hair"
    assert split_tag_like_text(text) == ["1girl", "blue hair", "red eyes"]


def test_extract_first_json_object_with_markdown_fence():
    raw = '```json\n{"caption_long_en": "an anime girl"}\n```'
    parsed = extract_first_json_object(raw)
    assert parsed["caption_long_en"] == "an anime girl"


def test_fallback_json_result_for_extract_tags():
    result = fallback_json_result(
        "extract_tags_from_image",
        "1girl, solo, smile",
        {"raw_tags": "1girl, solo"},
        "parse error",
    )
    assert "normalized_tags_en" in result
    assert result["normalized_tags_en"]


def test_format_anima_train_v1_prefers_wd14_tags():
    json_result = {
        "caption_long_en": "A cheerful anime girl with blue hair.",
        "expanded_tags_en": ["wrong_tag"],
    }
    inputs = {"wd14_raw_tags_en": ["1girl", "blue_hair", "smile"]}
    formatted = format_anima_train_v1(json_result, inputs)
    assert formatted["formatted_training_text"].startswith("1girl, blue_hair, smile")
    assert "\n\nA cheerful anime girl with blue hair." in formatted["formatted_training_text"]
    assert formatted["formatted_negative_prompt"] == ""


def test_resolve_vlm_endpoint_defaults():
    url, name = resolve_vlm_endpoint("gemma-4-e4b", "", "")
    assert "9002" in url
    assert name == "spawner-gemma-4-e4b-it"


def test_alias_index_detects_manual_alias():
    index = AliasIndex(enabled=True)
    hits = index.detect_in_tag_list("1girl, 初音未来, solo")
    assert any(hit["canonical_tag"] == "hatsune_miku" for hit in hits)


def test_build_sqlite_index_roundtrip(tmp_path: Path):
    json_path = PIPELINE_ROOT / "anima_caption_pipeline" / "resources" / "danbooru_character_aliases.json"
    db_path = tmp_path / "aliases.sqlite"
    count = build_sqlite_index(json_path, db_path)
    assert count > 0
    index = AliasIndex(db_path=db_path, enabled=True)
    hits = index.detect_in_tag_list("rem, 1girl")
    assert any(hit["canonical_tag"] == "rem_(re:zero)" for hit in hits)


def test_is_valid_hf_model_dir(tmp_path: Path):
    model_dir = tmp_path / "gemma"
    model_dir.mkdir()
    assert not is_valid_hf_model_dir(model_dir)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    assert is_valid_hf_model_dir(model_dir)


def test_is_broken_vllm_output_detects_pad_tokens():
    assert is_broken_vllm_output("", [0, 0, 0]) is True
    assert is_broken_vllm_output("hello", [1, 2, 3]) is False
    assert is_broken_vllm_output("<pad><pad>", [0, 0]) is True


def test_create_vlm_client_falls_back_to_local_gemma(tmp_path: Path):
    model_dir = tmp_path / "gemma"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"gemma4"}', encoding="utf-8")

    with patch("anima_caption_pipeline.gemma_local_client.LocalGemmaVlmClient") as local_cls:
        local_cls.return_value = MagicMock()
        client = create_vlm_client(
            vlm_model="gemma-4-e4b",
            api_url="http://127.0.0.1:9002/v1/chat/completions",
            model_name="spawner-gemma-4-e4b-it",
            local_model_dir=model_dir,
            gemma_vlm_backend="transformers",
        )
    local_cls.assert_called_once()
    assert client is local_cls.return_value


def test_create_vlm_client_forced_gemma_vllm_requires_probe(tmp_path: Path):
    model_dir = tmp_path / "gemma"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"gemma4"}', encoding="utf-8")

    with patch("anima_caption_pipeline.vlm_client.probe_vllm_generation", return_value=False):
        with pytest.raises(GemmaVllmUnavailableError, match="generation probe"):
            create_vlm_client(
                vlm_model="gemma-4-e4b",
                api_url="http://127.0.0.1:9002/v1/chat/completions",
                model_name="spawner-gemma-4-e4b-it",
                local_model_dir=model_dir,
                gemma_vlm_backend="vllm",
            )


def test_run_single_image_pipeline_mock_vlm(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    step1 = {
        "normalized_tags_en": ["1girl"],
        "caption_long_en": "step1 caption",
        "natural_language_en": "step1 natural",
    }
    step2 = {
        "caption_long_en": "A cute anime girl standing in soft light.",
        "natural_language_en": "A cute anime girl standing in soft light.",
    }

    client = MagicMock(spec=VlmClient)
    client.complete.side_effect = [json.dumps(step1), json.dumps(step2)]

    result = run_single_image_pipeline(
        client,
        str(image_path),
        raw_tags="1girl, solo, smile",
        purpose="character",
        alias_index=AliasIndex(enabled=False),
    )
    assert client.complete.call_count == 2
    assert "1girl, solo, smile" in result["formatted_training_text"]
    assert "A cute anime girl standing in soft light." in result["formatted_training_text"]


def test_ensure_gemma_model_without_download(tmp_path: Path):
    from anima_caption_pipeline.model_resolver import ensure_gemma_model

    model_dir = tmp_path / "models" / "gemma-4-E3B-it"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    resolved = ensure_gemma_model(tmp_path, auto_download=False)
    assert resolved == model_dir


def test_ensure_gemma_model_download_failure(tmp_path: Path):
    from anima_caption_pipeline.model_resolver import ensure_gemma_model

    with patch("anima_caption_pipeline.model_resolver.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=1, stdout="", stderr="download failed")
        with pytest.raises(RuntimeError, match="modelscope download failed"):
            ensure_gemma_model(tmp_path, auto_download=True)
