#!/usr/bin/env python3
"""Compatibility shim — implementation in scripts/autodl/apply_lora_next_anima_defaults.py"""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "scripts" / "autodl" / "apply_lora_next_anima_defaults.py"),
    run_name="__main__",
)
