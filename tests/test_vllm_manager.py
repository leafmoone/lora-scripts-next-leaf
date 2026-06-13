"""Tests for Tag-Edit-Leaf vLLM manager helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import sys
import types

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


def test_start_vllm_existing_gemma_server_requires_generation_probe():
    from mikazuki.utils.vllm_manager import start_vllm

    with patch(
        "mikazuki.utils.vllm_manager.check_vllm_health",
        return_value={"ready": True, "message": "vLLM 已就绪"},
    ):
        with patch(
            "mikazuki.utils.vllm_manager._probe_gemma_vllm_or_raise",
            side_effect=RuntimeError("bad generation"),
        ) as probe:
            try:
                start_vllm("gemma-4-e4b")
            except RuntimeError as exc:
                assert "bad generation" in str(exc)
            else:
                raise AssertionError("expected RuntimeError for bad Gemma vLLM generation")
    probe.assert_called_once()


def test_check_vllm_runtime_compat_blocks_gemma4_on_old_vllm():
    from mikazuki.utils.vllm_manager import _check_vllm_runtime_compat

    model_dir = Path(__file__).resolve().parents[1] / "models" / "gemma-4-E3B-it"
    if not (model_dir / "config.json").is_file():
        return
    fake_vllm = types.SimpleNamespace(__version__="0.9.2")
    with patch.dict(sys.modules, {"vllm": fake_vllm}):
        try:
            _check_vllm_runtime_compat("gemma-4-e4b", model_dir)
        except RuntimeError as exc:
            assert "Gemma 4" in str(exc)
            assert "0.9.2" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for gemma4 on vllm 0.9.2")


def test_cuda_library_paths_prefers_project_venv():
    from mikazuki.utils.vllm_manager import PROJECT_ROOT, _cuda_library_paths

    paths = _cuda_library_paths()
    venv_cu13 = (
        PROJECT_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "lib"
    )
    if not venv_cu13.is_dir():
        return
    assert str(venv_cu13.resolve()) in paths
    assert (venv_cu13 / "libcudart.so.13").is_file()


def test_start_vllm_cmd_disables_custom_ops_for_cuda128():
    from mikazuki.utils.vllm_manager import PROJECT_ROOT

    model_dir = PROJECT_ROOT / "models" / "gemma-4-E3B-it"
    if not (model_dir / "config.json").is_file():
        return

    with patch("mikazuki.utils.vllm_manager.shutil.which", return_value="/usr/bin/vllm"):
        with patch("mikazuki.utils.vllm_manager._validate_model_dir", return_value=model_dir):
            with patch("mikazuki.utils.vllm_manager._check_vllm_runtime_deps"):
                with patch("mikazuki.utils.vllm_manager._check_vllm_runtime_compat"):
                    with patch("mikazuki.utils.vllm_manager.check_vllm_health", return_value={"ready": False}):
                        with patch("mikazuki.utils.vllm_manager.is_port_open", return_value=False):
                            with patch("mikazuki.utils.vllm_manager.subprocess.Popen") as popen:
                                popen.return_value.poll.return_value = None
                                popen.return_value.stdout = iter([])
                                from mikazuki.utils.vllm_manager import start_vllm

                                start_vllm("gemma-4-e4b", wait_ready=False)
                                cmd = popen.call_args.args[0]
    assert "-cc.custom_ops" in cmd
    assert '["none"]' in cmd
    assert "--limit-mm-per-prompt" in cmd
    assert '{"image": 1}' in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "--generation-config" in cmd
    assert "vllm" in cmd


def test_tag_edit_leaf_api_has_vllm_routes():
    api_path = Path(__file__).resolve().parents[1] / "mikazuki" / "app" / "tag_edit_leaf_api.py"
    source = api_path.read_text(encoding="utf-8")
    assert '/vllm/status' in source
    assert '/vllm/start' in source
    assert 'auto_start_vllm' in source
