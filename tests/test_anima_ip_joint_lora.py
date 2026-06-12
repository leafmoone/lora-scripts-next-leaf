"""Tests for IP-Adapter + LoRA joint training configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ip_adapter.anima_ip_train import AnimaIPAdapterTrainer, _configure_joint_lora_args, _make_projection
except RuntimeError as exc:
    pytest.skip(f"anima_ip_train import unavailable in this environment: {exc}", allow_module_level=True)


def _base_args(**overrides):
    args = argparse.Namespace(
        train_joint_lora=False,
        network_module="networks.lora_anima",
        network_dim=None,
        network_alpha=None,
        network_train_unet_only=False,
        learning_rate=1e-4,
        ip_adapter_lr=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_configure_joint_lora_off_forces_zero_dim():
    args = _base_args(train_joint_lora=False, network_dim=128)
    _configure_joint_lora_args(args)
    assert args.network_dim == 0
    assert args.network_module == "networks.lora_anima"


def test_configure_joint_lora_on_requires_dim():
    args = _base_args(train_joint_lora=True, network_dim=None)
    with pytest.raises(ValueError, match="network_dim"):
        _configure_joint_lora_args(args)


def test_configure_joint_lora_on_sets_defaults():
    args = _base_args(train_joint_lora=True, network_dim=64, network_alpha=None)
    _configure_joint_lora_args(args)
    assert args.network_dim == 64
    assert args.network_alpha == 64.0
    assert args.network_train_unet_only is True


def test_patched_prepare_appends_ip_params():
    trainer = AnimaIPAdapterTrainer()
    trainer.args = _base_args(train_joint_lora=True, network_dim=8, ip_adapter_lr=5e-4)
    trainer.clip_proj = _make_projection()
    ip_groups = trainer.get_trainable_params()
    assert len(ip_groups) >= 1
    assert all("params" in g and "lr" in g for g in ip_groups)


def test_get_params_to_clip_includes_ip_params():
    trainer = AnimaIPAdapterTrainer()
    trainer.args = _base_args(train_joint_lora=True, network_dim=8, ip_adapter_lr=5e-4)
    trainer.clip_proj = _make_projection()

    from ip_adapter.anima_ip_image_proj import ImageProjModel

    trainer.image_proj = ImageProjModel(
        cross_attention_dim=1024,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
    )

    network = MagicMock()
    lora_param = trainer.clip_proj.weight
    network.get_trainable_params = MagicMock(return_value=[lora_param])

    accelerator = MagicMock()
    accelerator.unwrap_model = MagicMock(return_value=network)

    params = trainer.get_params_to_clip(accelerator, network)
    assert lora_param in params
    assert len(params) > 1
