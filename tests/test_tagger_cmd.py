"""Tests for shared differential tagger command builder."""

from mikazuki.utils.tagger_cmd import build_tagger_cmd


def test_build_tagger_cmd_smart_defaults():
    cmd = build_tagger_cmd(
        {
            "input_dir": "/data/images",
            "output_dir": "/data/out",
            "mode": "smart",
            "model": "wd-eva02-large-tagger-v3",
            "purpose": "character",
            "use_vlm": True,
            "use_wd14": True,
            "vlm_prompt_mode": "lora",
            "inject_wd14_tags": True,
            "resume": True,
            "wd14_batch": 16,
            "vlm_batch": 2,
        },
        python_executable="/usr/bin/python3",
        tagger_dir="/proj/tools/differential_tagger",
    )
    assert "--vlm" in cmd
    assert "--vlm-prompt-mode" in cmd and "lora" in cmd
    assert "--no-wd14" not in cmd
    assert "--no-inject-wd14-tags" not in cmd
    assert cmd[0] == "/usr/bin/python3"
    assert cmd[1].endswith("main.py")
    assert "--smart" in cmd
    assert "--resume" in cmd
    assert "--wd14-batch" in cmd and "16" in cmd
    assert "--vlm-batch" in cmd and "2" in cmd
    assert "--vlm-workers" not in cmd
    assert "--purpose" in cmd and "character" in cmd
    assert "--vlm" in cmd


def test_build_tagger_cmd_consensus_taggers():
    cmd = build_tagger_cmd(
        {
            "input_dir": "/in",
            "mode": "smart",
            "taggers": ["wd-swinv2-tagger-v3", "wd-vit-tagger-v3"],
            "consensus": 2,
        },
        python_executable="python",
        tagger_dir="/tagger",
    )
    assert "--taggers" in cmd
    assert "wd-swinv2-tagger-v3" in cmd
    assert "wd-vit-tagger-v3" in cmd
    assert "--consensus" in cmd and "2" in cmd


def test_build_tagger_cmd_simple_cpu():
    cmd = build_tagger_cmd(
        {
            "input_dir": "/in",
            "mode": "simple",
            "use_cpu": True,
            "blacklist": "watermark, signature",
        },
        python_executable="python",
        tagger_dir="/tagger",
    )
    assert "--simple" in cmd
    assert "--cpu" in cmd
    assert "--blacklist" in cmd
    assert "watermark" in cmd
    assert "signature" in cmd


def test_frontend_smart_config_full_chain():
    """Mirror tag-edit-leaf.html collectConfig() defaults → CLI argv."""
    frontend = {
        "input_dir": "/data/images",
        "output_dir": "/data/images",
        "mode": "smart",
        "model": "wd-eva02-large-tagger-v3",
        "threshold": 0.35,
        "char_threshold": 0.85,
        "max_tags": 0,
        "use_cpu": False,
        "recursive": False,
        "save_captions": True,
        "resume": False,
        "purpose": "character",
        "trigger": "my_char",
        "use_wd14": True,
        "use_vlm": True,
        "vlm_backend": "transformers",
        "vlm_prompt_mode": "lora",
        "inject_wd14_tags": True,
        "taggers": [],
        "consensus": 2,
        "blacklist": ["watermark"],
        "data_dir": "",
        "wd14_batch": 8,
        "vlm_batch": 4,
    }
    cmd = build_tagger_cmd(frontend, python_executable="python3", tagger_dir="/tagger")
    assert "--smart" in cmd
    assert "--vlm" in cmd
    assert "--no-wd14" not in cmd
    assert "--purpose" in cmd and "character" in cmd
    assert "--trigger" in cmd and "my_char" in cmd
    assert "--vlm-prompt-mode" in cmd and "lora" in cmd
    assert "--no-inject-wd14-tags" not in cmd
    assert "--blacklist" in cmd and "watermark" in cmd
    assert "--save-captions" in cmd
    assert "--wd14-batch" not in cmd
    assert "--vlm-batch" not in cmd
    assert "--vlm-workers" not in cmd


def test_build_tagger_cmd_vllm_backend():
    cmd = build_tagger_cmd(
        {
            "input_dir": "/in",
            "mode": "smart",
            "use_vlm": True,
            "vlm_backend": "vllm",
            "vlm_batch": 16,
            "vllm_api_url": "http://127.0.0.1:18901/v1/chat/completions",
            "vllm_model": "toriigate-0.5",
        },
        python_executable="python3",
        tagger_dir="/tagger",
    )
    assert "--vlm-backend" in cmd and "vllm" in cmd
    assert "--vlm-batch" in cmd and "16" in cmd
    assert "--vllm-api-url" in cmd
    assert "--vllm-model" in cmd and "toriigate-0.5" in cmd


def test_frontend_vlm_only_chain():
    cmd = build_tagger_cmd(
        {
            "input_dir": "/in",
            "mode": "smart",
            "use_wd14": False,
            "use_vlm": True,
            "vlm_prompt_mode": "official_short",
            "inject_wd14_tags": False,
        },
        python_executable="python3",
        tagger_dir="/tagger",
    )
    assert "--no-wd14" in cmd
    assert "--vlm" in cmd
    assert "--vlm-prompt-mode" in cmd and "official_short" in cmd
    assert "--no-inject-wd14-tags" in cmd
