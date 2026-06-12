"""Tests for IP-Adapter sidecar resume path resolution and loading."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ip_adapter.anima_ip_train import AnimaIPAdapterTrainer, _make_projection


def test_resolve_sidecar_direct_path():
    sidecar = PROJECT_ROOT / "output/ipa_char/ipa_char-step00005000.ipadapter.safetensors"
    if not sidecar.is_file():
        pytest.skip("sidecar checkpoint not available in workspace")

    resolved = AnimaIPAdapterTrainer._resolve_ip_adapter_weights_path(str(sidecar))
    assert resolved == str(sidecar.resolve())


def test_resolve_sidecar_from_lora_stub_path():
    sidecar = PROJECT_ROOT / "output/ipa_char/ipa_char-step00005000.ipadapter.safetensors"
    if not sidecar.is_file():
        pytest.skip("sidecar checkpoint not available in workspace")

    lora_stub = sidecar.with_suffix(".safetensors")
    resolved = AnimaIPAdapterTrainer._resolve_ip_adapter_weights_path(str(lora_stub))
    assert resolved == str(sidecar.resolve())


def test_resolve_sidecar_missing_raises(tmp_path):
    missing = tmp_path / "missing.ipadapter.safetensors"
    with pytest.raises(FileNotFoundError):
        AnimaIPAdapterTrainer._resolve_ip_adapter_weights_path(str(missing))


def test_load_ip_adapter_weights_roundtrip(tmp_path):
    """Save then load sidecar into a freshly built trainer."""
    source = PROJECT_ROOT / "output/ipa_char/ipa_char-step00005000.ipadapter.safetensors"
    if not source.is_file():
        pytest.skip("sidecar checkpoint not available in workspace")

    import argparse
    from safetensors.torch import load_file

    args = argparse.Namespace(
        aux_encoders="clip_ccip",
        ipa_mode="simple",
        adapter_type="mlp",
        num_ip_tokens=4,
        num_ip_tokens_clip=4,
        num_ip_tokens_ccip=8,
        num_ip_tokens_lsnet=4,
        ip_scale=1.0,
        ip_adapter_weights="",
    )

    trainer = AnimaIPAdapterTrainer()
    trainer._aux_encoders = ("ccip",)
    trainer._ipa_mode = "simple"
    trainer._adapter_type = "mlp"
    trainer.clip_proj = _make_projection()
    trainer.ccip_proj = _make_projection()
    from ip_adapter.anima_ip_image_proj import MLPImageProjModel, MultiStreamProj

    trainer.image_proj = MultiStreamProj.from_modules([
        MLPImageProjModel(feature_dim=1024, cross_attention_dim=1024, num_tokens=4),
        MLPImageProjModel(feature_dim=1024, cross_attention_dim=1024, num_tokens=8),
    ])

    trainer.load_ip_adapter_weights(str(source), args=args)

    expected = load_file(str(source))
    actual = trainer._ip_adapter_state_dict()
    assert set(expected.keys()) == set(actual.keys())
    for key in expected:
        diff = (expected[key].float() - actual[key].float()).abs().max().item()
        assert diff < 1e-5, f"tensor mismatch for {key}: max diff {diff}"
