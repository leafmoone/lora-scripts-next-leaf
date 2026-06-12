"""Shared command builder for the standalone differential tagger."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def default_tagger_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "tools" / "differential_tagger")


def default_tagger_data_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "models")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_list(value: Any, *, comma: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(",", " ").split() if comma else value.split()
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        return []
    return [str(item).strip() for item in parts if str(item).strip()]


def build_tagger_cmd(
    config: dict[str, Any],
    *,
    python_executable: str | None = None,
    tagger_dir: str | None = None,
) -> list[str]:
    """Build argv for tools/differential_tagger/main.py from a shared config."""
    tagger_dir = os.path.abspath(tagger_dir or default_tagger_dir())
    main_py = os.path.join(tagger_dir, "main.py")
    py = python_executable or sys.executable

    input_dir = str(config.get("input_dir") or config.get("input") or "").strip()
    output_dir = str(config.get("output_dir") or config.get("output") or input_dir).strip()
    data_dir = str(config.get("data_dir") or default_tagger_data_dir()).strip()
    mode = str(config.get("mode") or "smart").strip().lower()
    mode = "simple" if mode == "simple" else "smart"

    cmd = [
        py,
        main_py,
        "--data-dir",
        os.path.abspath(data_dir),
        "--input",
        input_dir,
        "--output",
        output_dir,
        f"--{mode}",
        "--model",
        str(config.get("model") or "wd-eva02-large-tagger-v3"),
        "--threshold",
        str(config.get("threshold", 0.35)),
        "--character-threshold",
        str(config.get("char_threshold", config.get("character_threshold", 0.85))),
    ]

    if _as_bool(config.get("use_cpu"), False):
        cmd.append("--cpu")
    if _as_bool(config.get("recursive"), False):
        cmd.append("--recursive")
    if _as_bool(config.get("save_captions"), True):
        cmd.append("--save-captions")
    if _as_bool(config.get("resume"), False):
        cmd.append("--resume")
    if _as_bool(config.get("verbose"), False):
        cmd.append("--verbose")

    trigger = str(config.get("trigger") or "").strip()
    if trigger:
        cmd.extend(["--trigger", trigger])

    wd14_batch = _as_int(config.get("wd14_batch"), 8)
    if wd14_batch != 8:
        cmd.extend(["--wd14-batch", str(max(1, wd14_batch))])
    vlm_batch = _as_int(config.get("vlm_batch"), 4)
    if vlm_batch != 4:
        cmd.extend(["--vlm-batch", str(max(1, vlm_batch))])

    if mode == "smart":
        cmd.extend(["--purpose", str(config.get("purpose") or "character")])
        use_vlm = _as_bool(config.get("use_vlm"), True)
        use_wd14 = _as_bool(config.get("use_wd14"), True)
        cmd.append("--vlm" if use_vlm else "--no-vlm")
        if not use_wd14:
            cmd.append("--no-wd14")
        vlm_prompt_mode = str(config.get("vlm_prompt_mode") or "lora").strip()
        if vlm_prompt_mode:
            cmd.extend(["--vlm-prompt-mode", vlm_prompt_mode])
        if not _as_bool(config.get("inject_wd14_tags"), True):
            cmd.append("--no-inject-wd14-tags")
        vlm_backend = str(config.get("vlm_backend") or "transformers").strip().lower()
        if vlm_backend in {"vllm", "openai", "api"}:
            cmd.extend(["--vlm-backend", "vllm"])
        elif vlm_backend not in {"transformers", "toriigate", "local", "hf", ""}:
            cmd.extend(["--vlm-backend", vlm_backend])
        vllm_api_url = str(config.get("vllm_api_url") or "").strip()
        if vllm_api_url:
            cmd.extend(["--vllm-api-url", vllm_api_url])
        vllm_model = str(config.get("vllm_model") or "").strip()
        if vllm_model:
            cmd.extend(["--vllm-model", vllm_model])
        taggers = _clean_list(config.get("taggers"))
        if len(taggers) >= 2:
            cmd.extend(["--taggers", *taggers, "--consensus", str(_as_int(config.get("consensus"), 2))])

    max_tags = _as_int(config.get("max_tags"), 0)
    if max_tags > 0:
        cmd.extend(["--max-tags", str(max_tags)])

    blacklist = _clean_list(config.get("blacklist"), comma=True)
    if blacklist:
        cmd.extend(["--blacklist", *blacklist])

    return cmd
