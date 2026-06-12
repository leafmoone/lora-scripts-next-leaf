"""Tag-Edit-Leaf API tests for anima_train mode (no full app import)."""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_PATH = PROJECT_ROOT / "mikazuki" / "app" / "tag_edit_leaf_api.py"
CONFIG_PATH = PROJECT_ROOT / "config" / "anima_caption_models.json"


def test_anima_caption_models_json_structure():
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    models = data.get("vlm_models", {})
    assert "toriigate-0.5" in models
    assert "gemma-4-e4b" in models
    gemma = models["gemma-4-e4b"]
    assert gemma["local_model_dir"] == "models/gemma-4-E3B-it"
    assert gemma["modelscope_id"] == "spawner/spawner-gemma-4-E4B-it"
    assert "9002" in gemma["default_api_url"]


def test_tag_edit_leaf_api_has_anima_train_branch():
    source = API_PATH.read_text(encoding="utf-8")
    assert 'mode == "anima_train"' in source
    assert "_run_anima_train_tagging" in source
    assert '"anima_train"' in source
    assert "_load_anima_train_vlm_models" in source


def test_anima_train_request_fields_documented_in_api():
    source = API_PATH.read_text(encoding="utf-8")
    expected_fields = [
        "vlm_model",
        "vllm_api_url",
        "vllm_model",
        "vlm_workers",
        "auto_download_gemma",
        "use_alias_index",
        "style_hint",
    ]
    for field in expected_fields:
        assert field in source


def test_frontend_has_anima_train_mode_option():
    html_path = PROJECT_ROOT / "frontend" / "dist" / "tag-edit-leaf.html"
    html = html_path.read_text(encoding="utf-8")
    assert 'value="anima_train"' in html
    assert "animaVlmModel" in html
    assert "gemma-4-e4b" in html
    assert "animaAutoStartVllm" in html
    assert "ensureAnimaVllmReady" in html
    assert "auto_start_vllm" in html
