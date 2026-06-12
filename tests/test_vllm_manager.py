"""Tests for Tag-Edit-Leaf vLLM manager helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mikazuki.utils.vllm_manager import (
    get_vlm_preset,
    load_vlm_presets,
    models_endpoint,
    parse_port,
)


def test_load_vlm_presets_contains_gemma_and_toriigate():
    presets = load_vlm_presets()
    assert "gemma-4-e4b" in presets
    assert "toriigate-0.5" in presets
    assert presets["gemma-4-e4b"]["port"] == 9002
    assert presets["toriigate-0.5"]["local_model_dir"]


def test_get_vlm_preset_aliases():
    preset = get_vlm_preset("gemma")
    assert preset["default_served_name"] == "spawner-gemma-4-e4b-it"


def test_parse_port_and_models_endpoint():
    assert parse_port("http://127.0.0.1:9002/v1/chat/completions") == 9002
    assert models_endpoint("http://127.0.0.1:9002/v1/chat/completions") == "http://127.0.0.1:9002/v1/models"


def test_get_vllm_status_stopped_when_no_server():
    from mikazuki.utils.vllm_manager import get_vllm_status

    with patch("mikazuki.utils.vllm_manager.is_port_open", return_value=False):
        with patch("mikazuki.utils.vllm_manager.check_vllm_health", return_value={"ready": False, "message": "down"}):
            status = get_vllm_status("gemma-4-e4b")
    assert status["ready"] is False
    assert status["port"] == 9002


def test_tag_edit_leaf_api_has_vllm_routes():
    api_path = Path(__file__).resolve().parents[1] / "mikazuki" / "app" / "tag_edit_leaf_api.py"
    source = api_path.read_text(encoding="utf-8")
    assert '/vllm/status' in source
    assert '/vllm/start' in source
    assert 'auto_start_vllm' in source
