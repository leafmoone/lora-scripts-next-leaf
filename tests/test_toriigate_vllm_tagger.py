"""Tests for ToriiGate vLLM backend helpers."""

import sys
from pathlib import Path

TAGGER_DIR = Path(__file__).resolve().parents[1] / "tools" / "differential_tagger"
sys.path.insert(0, str(TAGGER_DIR))

from toriigate_vllm_tagger import normalize_vlm_backend  # noqa: E402


def test_normalize_vlm_backend_aliases():
    assert normalize_vlm_backend("toriigate") == "transformers"
    assert normalize_vlm_backend("transformers") == "transformers"
    assert normalize_vlm_backend("vllm") == "vllm"
    assert normalize_vlm_backend("openai") == "vllm"
