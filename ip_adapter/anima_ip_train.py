"""
Anima IP-Adapter Training Script with Multi-Encoder Support

Extends ``AnimaNetworkTrainer`` to support IP-Adapter training with
optional auxiliary vision encoders (CCIP for character identity,
LSNet for artist style) alongside CLIP for content conditioning.

Modes (controlled by --aux_encoders):
  ``clip_only``     — CLIP only (original behavior)
  ``clip_ccip``     — CLIP + CCIP dual-stream
  ``clip_lsnet``    — CLIP + LSNet dual-stream
  ``clip_ccip_lsnet`` — CLIP + CCIP + LSNet triple-stream

Fusion is concat-mode: each stream's global feature is projected (768→1024,
trainable) then expanded into N IP tokens that are concatenated onto the text
context before the DiT's frozen cross-attention.

Usage (standalone):
  accelerate launch ip_adapter/anima_ip_train.py \
    --pretrained_model_name_or_path anima.safetensors \
    --vae qwen_image_vae.safetensors \
    --qwen3 qwen3.safetensors \
    --clip_model openai/clip-vit-large-patch14 \
    --aux_encoders clip_ccip_lsnet \
    --ccip_ckpt /path/to/ccip-caformer_b36-24.ckpt \
    --lsnet_ckpt /path/to/best_checkpoint.pth \
    --train_data_dir ./train/anima_ip_dataset \
    --output_dir ./output/ipa \
    --num_ip_tokens 4 \
    --ip_scale 1.0 \
    --learning_rate 1e-4 \
    --max_train_epochs 10
"""

from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
import threading
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator

# Ensure project root and vendor/sd-scripts are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_vendor_root = _PROJECT_ROOT / "vendor" / "sd-scripts"
if str(_vendor_root) not in sys.path:
    sys.path.insert(0, str(_vendor_root))

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    strategy_anima,
    train_util,
)
from anima_train_network import AnimaNetworkTrainer, setup_parser as anima_setup_parser
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from ip_adapter.anima_ip_converter import AnimaIPAConverter
from ip_adapter.anima_ip_image_proj import ImageProjModel, MultiStreamProj, Resampler
from ip_adapter.ccip_encoder import load_ccip_encoder, DEFAULT_CKPT as DEFAULT_CCIP_CKPT
from ip_adapter.lsnet_encoder import load_lsnet_encoder

# CLIP image normalization (OpenAI CLIP mean/std), applied to [0,1] pixels.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _clip_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize [0,1] pixels with CLIP mean/std. x: (B, 3, H, W)."""
    mean = torch.tensor(_CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _make_projection(in_dim: int = 768, out_dim: int = 1024) -> nn.Module:
    """Trainable 768→1024 projection (fp32) used per encoder stream."""
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim)).float()

# ── Mode parser ──────────────────────────────────────────────────


def _configure_joint_lora_args(args) -> None:
    """Apply LoRA joint-training settings before Kohya ``train()`` runs."""
    joint = bool(getattr(args, "train_joint_lora", False))
    module = getattr(args, "network_module", None) or "networks.lora_anima"
    args.network_module = module

    if not joint:
        args.network_dim = 0
        logger.info("IP-Adapter only: train_joint_lora=false, network_dim=0 (no LoRA modules)")
        return

    dim = getattr(args, "network_dim", None)
    if dim is None or int(dim) < 1:
        raise ValueError("--train_joint_lora requires --network_dim >= 1")

    alpha = getattr(args, "network_alpha", None)
    if alpha is None or (isinstance(alpha, (int, float)) and float(alpha) <= 0):
        args.network_alpha = float(args.network_dim)
        logger.info("train_joint_lora: defaulting network_alpha to network_dim (%s)", args.network_alpha)

    if not getattr(args, "network_train_unet_only", False):
        args.network_train_unet_only = True
        logger.info("train_joint_lora: enabled network_train_unet_only")

    logger.info(
        "LoRA joint training: module=%s dim=%s alpha=%s",
        args.network_module,
        args.network_dim,
        args.network_alpha,
    )


def _parse_aux_encoders(mode: str) -> tuple[str, ...]:
    """Parse --aux_encoders into a tuple of auxiliary encoder names."""
    if not mode or mode in ("none", "clip_only", "0"):
        return ()
    parts = [p.strip().lower() for p in mode.split("_")]
    encoders = tuple(p for p in parts if p in ("ccip", "lsnet"))
    if "clip" in parts:
        pass  # ignore "clip" — it's always present
    return encoders


# ── Trainer ──────────────────────────────────────────────────────


class AnimaIPAdapterTrainer(AnimaNetworkTrainer):
    """Anima NetworkTrainer with multi-encoder IP-Adapter support."""

    def __init__(self):
        super().__init__()
        self.clip_image_encoder: Optional[nn.Module] = None
        self.clip_proj: Optional[nn.Module] = None  # 768→1024 projection (trainable)
        self.ccip_proj: Optional[nn.Module] = None  # 768→1024 projection (trainable)
        self.lsnet_proj: Optional[nn.Module] = None  # 768→1024 projection (trainable)
        self.ccip_encoder: Optional[nn.Module] = None
        self.lsnet_encoder: Optional[nn.Module] = None
        self.image_proj: Optional[ImageProjModel | MultiStreamProj] = None
        self.image_proj_resampler: Optional[MultiStreamProj] = None  # double-mode resampler
        self._ipa_mode: str = "simple"
        self._adapter_type: str = "linear"
        self.ip_adapters: dict[str, Any] = {}
        self._aux_encoders: tuple[str, ...] = ()
        self._identity_index: dict[int, dict[str, list[str]]] = {}
        self._precompute_lock = threading.Lock()
        self._precomputed_dataset_ids: set[int] = set()
        self._precompute_pending_datasets: list[Any] = []
        self._precomputed_cache_complete = False

    @staticmethod
    def _make_stream_projectors(num_streams: int, mode: str, num_queries: int | list[int],
                                 use_omni: bool = False) -> MultiStreamProj:
        from ip_adapter.anima_ip_image_proj import Resampler
        if isinstance(num_queries, int):
            num_queries = [num_queries] * num_streams
        if mode == "resampler":
            if use_omni:
                return MultiStreamProj.from_modules([
                    _build_omni_stream(nq, dim=1024, num_heads=16)
                    for nq in num_queries
                ])
            return MultiStreamProj.from_modules([
                Resampler(dim=1024, depth=4, dim_head=64, heads=16,
                          num_queries=nq, output_dim=1024)
                for nq in num_queries
            ])
        if mode == "mlp_simple":
            from ip_adapter.anima_ip_image_proj import MLPImageProjModel
            return MultiStreamProj.from_modules([
                MLPImageProjModel(
                    feature_dim=1024,
                    cross_attention_dim=1024,
                    num_tokens=num_queries[i],
                )
                for i in range(num_streams)
            ])
        return MultiStreamProj(
            num_streams=num_streams, cross_attention_dim=1024,
            embed_dim=1024, tokens_per_stream=num_queries,
        )

    # ── model loading ──────────────────────────────────────────

    def load_target_model(self, args, weight_dtype, accelerator):
        model_type, text_encoders, vae, unet = super().load_target_model(
            args, weight_dtype, accelerator
        )

        # Parse auxiliary encoders
        self._aux_encoders = _parse_aux_encoders(
            getattr(args, "aux_encoders", "") or ""
        )
        self._ipa_mode = getattr(args, "ipa_mode", "simple")
        logger.info(f"IP-Adapter auxiliary encoders: {self._aux_encoders or 'none'}")

        # Projection layers are trainable and are required even when frozen
        # vision encoder features are fully served from disk cache.
        self.clip_proj = _make_projection()
        if "ccip" in self._aux_encoders:
            self.ccip_proj = _make_projection()
        if "lsnet" in self._aux_encoders:
            self.lsnet_proj = _make_projection()

        sample_ref = (getattr(args, "sample_reference_image", "") or "").strip()
        sample_ref_abs = (
            os.path.abspath(sample_ref) if sample_ref and os.path.isfile(sample_ref) else ""
        )
        if sample_ref and not sample_ref_abs:
            logger.warning(
                "Sample reference image not found, sampling will run without IP injection: %s",
                sample_ref,
            )

        dataset_cache_complete = bool(
            self._precomputed_dir
            and self._precompute_pending_datasets
            and self._is_precomputed_cache_complete(self._precompute_pending_datasets)
        )
        sample_ref_cached = (not sample_ref_abs) or self._is_path_cached(sample_ref_abs)

        if dataset_cache_complete and sample_ref_abs and not sample_ref_cached:
            logger.info(
                "Precomputing sample reference image embedding (one-shot): %s",
                sample_ref_abs,
            )
            self._load_frozen_encoders(args, weight_dtype)
            self._precompute_image_path(sample_ref_abs)
            self._unload_frozen_encoders()
            sample_ref_cached = self._is_path_cached(sample_ref_abs)

        self._precomputed_cache_complete = dataset_cache_complete and sample_ref_cached
        if self._precomputed_cache_complete:
            logger.info(
                "Complete precomputed IP embedding cache found; skipping frozen "
                "CLIP/CCIP/LSNet encoder loading."
            )
            if sample_ref_abs:
                logger.info(
                    "Sample reference image will use precomputed cache: %s",
                    sample_ref_abs,
                )
            self._precompute_pending_datasets.clear()
            return model_type, text_encoders, vae, unet

        self._load_frozen_encoders(args, weight_dtype)

        for dataset in self._precompute_pending_datasets:
            self._ensure_precomputed_for_dataset(dataset)
        self._precompute_pending_datasets.clear()

        if sample_ref_abs and not self._is_path_cached(sample_ref_abs):
            self._precompute_image_path(sample_ref_abs)

        return model_type, text_encoders, vae, unet

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        dit, text_encoders = super().load_unet_lazily(
            args, weight_dtype, accelerator, text_encoders
        )

        # Inject IP-Adapter layers
        self.ip_adapters = AnimaIPAConverter.create(dit)
        ip_scale = float(getattr(args, "ip_scale", 1.0) or 1.0)
        for attn in self.ip_adapters.values():
            attn.ip_scale = ip_scale
        num_streams = 1 + len(self._aux_encoders)
        self._ipa_mode = getattr(args, "ipa_mode", "simple")

        # Per-stream token counts (CLIP, then aux in order)
        _nt = args.num_ip_tokens
        _nt_clip = getattr(args, "num_ip_tokens_clip", None) or _nt
        _nt_ccip = getattr(args, "num_ip_tokens_ccip", None) or _nt
        _nt_lsnet = getattr(args, "num_ip_tokens_lsnet", None) or _nt
        _adapter_type = getattr(args, "adapter_type", "linear") or "linear"
        self._adapter_type = _adapter_type

        _tokens = [_nt_clip]
        if "ccip" in self._aux_encoders:
            _tokens.append(_nt_ccip)
        if "lsnet" in self._aux_encoders:
            _tokens.append(_nt_lsnet)

        if self._ipa_mode == "simple":
            if num_streams > 1:
                if _adapter_type == "mlp":
                    from ip_adapter.anima_ip_image_proj import MLPImageProjModel
                    modules = [MLPImageProjModel(feature_dim=1024, cross_attention_dim=1024,
                                                  num_tokens=_tokens[i])
                               for i in range(num_streams)]
                    self.image_proj = MultiStreamProj.from_modules(modules)
                else:
                    self.image_proj = MultiStreamProj(
                        num_streams=num_streams,
                        cross_attention_dim=1024, embed_dim=1024,
                        tokens_per_stream=_tokens,
                    )
            else:
                if _adapter_type == "mlp":
                    from ip_adapter.anima_ip_image_proj import MLPImageProjModel
                    self.image_proj = MLPImageProjModel(feature_dim=1024, cross_attention_dim=1024,
                                                         num_tokens=_nt_clip)
                else:
                    self.image_proj = ImageProjModel(
                        cross_attention_dim=1024, clip_embeddings_dim=1024,
                        clip_extra_context_tokens=_nt_clip,
                    )

        elif self._ipa_mode == "resampler":
            self.image_proj = self._make_stream_projectors(
                num_streams, "resampler", _tokens,
                use_omni=_adapter_type == "omni",
            )

        elif self._ipa_mode == "double":
            self.image_proj = self._make_stream_projectors(num_streams, "simple", _tokens) if _adapter_type != "mlp" else self._make_stream_projectors(num_streams, "mlp_simple", _tokens)
            _tokens_double = [max(n, 8) for n in _tokens]
            self.image_proj_resampler = self._make_stream_projectors(
                num_streams, "resampler", _tokens_double,
                use_omni=_adapter_type == "omni",
            )

        else:
            raise ValueError(f"Unknown ipa_mode: {self._ipa_mode}")

        weights_path = (getattr(args, "ip_adapter_weights", None) or "").strip()
        if weights_path:
            resolved = self._resolve_ip_adapter_weights_path(weights_path)
            self.load_ip_adapter_weights(resolved, args=args)

        return dit, text_encoders

    # ── training ───────────────────────────────────────────────

    def train(self, args):
        """Override train() to inject ImageProjModel params into the LoRA network
        param collection, since IP-Adapter trains ImageProjModel instead of LoRA.
        """
        self.args = args
        _configure_joint_lora_args(args)

        # ── Precomputed embedding cache ────────────────────────
        precomputed_dir = getattr(args, "ip_adapter_precomputed_emb_dir", "") or ""
        self._precomputed_cache: dict[str, torch.Tensor] = {}
        if precomputed_dir:
            import glob as _glob
            os.makedirs(precomputed_dir, exist_ok=True)
            _loaded = 0
            for _pt in _glob.glob(os.path.join(precomputed_dir, "*.pt")):
                try:
                    self._precomputed_cache[os.path.basename(_pt)] = torch.load(_pt, map_location="cpu", weights_only=True)
                    _loaded += 1
                except Exception:
                    pass
            if _loaded > 0:
                logger.info(f"Loaded {_loaded} precomputed IP embeddings from {precomputed_dir}")
            self._precomputed_dir = precomputed_dir
        else:
            self._precomputed_dir = ""

        # IP-Adapter needs a raw conditioning image for the vision encoders every
        # step, decoupled from the VAE target (batch["images"]). We patch the
        # dataset to add a dedicated `ip_reference_images` field, loaded directly
        # from disk (downscaled) so it works regardless of latent caching, and so
        # the reference can be a *different* image than the target (paired mode).
        import library.train_util as _tu

        _BaseDataset = _tu.BaseDataset
        _orig_getitem = _BaseDataset.__getitem__
        _orig_make_buckets = _BaseDataset.make_buckets
        _trainer_ref = self
        skip_invalid_images = bool(getattr(args, "skip_invalid_images", True))

        def _patched_make_buckets(ds_self):
            def _valid_size(size):
                try:
                    width, height = size
                    return int(width) > 0 and int(height) > 0
                except (TypeError, ValueError):
                    return False

            bad_images = []
            for image_key, info in list(ds_self.image_data.items()):
                if info.image_size is None:
                    info.image_size = ds_self.get_image_size(info.absolute_path)
                if not _valid_size(info.image_size):
                    bad_images.append((image_key, info.absolute_path, info.image_size))

            if bad_images:
                shown = "\n".join(
                    f"  - {path} (size={size})"
                    for _key, path, size in bad_images[:20]
                )
                remaining = len(bad_images) - 20
                if remaining > 0:
                    shown += f"\n  ... and {remaining} more"
                if not skip_invalid_images:
                    raise ValueError(
                        "IP-Adapter dataset contains image(s) with invalid dimensions. "
                        "Remove or re-save these files before training:\n"
                        f"{shown}"
                    )
                for image_key, _path, _size in bad_images:
                    ds_self.image_data.pop(image_key, None)
                    ds_self.image_to_subset.pop(image_key, None)
                logger.warning(
                    "Skipped %s invalid image(s) before bucket creation:\n%s",
                    len(bad_images),
                    shown,
                )
                if not ds_self.image_data:
                    raise ValueError(
                        "IP-Adapter dataset has no valid images after filtering "
                        "invalid image files."
                    )

            result = _orig_make_buckets(ds_self)
            if _trainer_ref._precomputed_dir:
                if _trainer_ref.clip_image_encoder is not None:
                    _trainer_ref._ensure_precomputed_for_dataset(ds_self)
                else:
                    _trainer_ref._precompute_pending_datasets.append(ds_self)
            return result

        def _patched_getitem(ds_self, index):
            example = _orig_getitem(ds_self, index)
            if _trainer_ref.clip_image_encoder is not None:
                # Always load reference images from disk (resized to ip_cond_size).
                images, ref_paths = _trainer_ref._load_reference_images_for_index(ds_self, index)
                example["ip_reference_images"] = images
                example["_image_paths"] = ref_paths
            elif _trainer_ref._precomputed_cache_complete:
                example["_image_paths"] = _trainer_ref._get_reference_paths_for_index(ds_self, index)
            return example

        _BaseDataset.__getitem__ = _patched_getitem
        _BaseDataset.make_buckets = _patched_make_buckets

        network_module = importlib.import_module(args.network_module)
        cls = network_module.LoRANetwork
        if hasattr(cls, "prepare_optimizer_params_with_multiple_te_lrs"):
            _prepare_attr = "prepare_optimizer_params_with_multiple_te_lrs"
        elif hasattr(cls, "prepare_optimizer_params"):
            _prepare_attr = "prepare_optimizer_params"
        else:
            raise AttributeError(
                f"{args.network_module}.LoRANetwork has no prepare_optimizer_params* method; "
                "cannot inject IP-Adapter optimizer groups."
            )
        _orig_prepare = getattr(cls, _prepare_attr)
        _orig_save = cls.save_weights

        def _patched_prepare(self_network, *a, **kw):
            results = _orig_prepare(self_network, *a, **kw)
            if isinstance(results, tuple):
                params_list, lr_descs = results if len(results) == 2 else (results[0], None)
            else:
                params_list, lr_descs = results, None
            added = _trainer_ref.get_trainable_params()
            params_list = list(params_list) + added
            # Keep lr_descriptions aligned with the optimizer param groups; otherwise
            # train_network.generate_step_logs indexes lr_descriptions[i] out of range
            # (one entry is required per param group / per lr_scheduler.get_last_lr()).
            if lr_descs is not None:
                lr_descs = list(lr_descs) + [f"ip_proj_{i}" for i in range(len(added))]
            return (params_list, lr_descs) if lr_descs is not None else params_list

        def _patched_save(self_network, file, dtype, metadata, *a, **kw):
            if len(self_network.state_dict()) == 0:
                _trainer_ref._save_ip_adapter_weights(file, dtype)
                if os.path.isfile(file) and os.path.getsize(file) == 0:
                    _orig_remove(file)
                    logger.info(f"Removed empty LoRA stub checkpoint: {file}")
                return None
            result = _orig_save(self_network, file, dtype, metadata, *a, **kw)
            _trainer_ref._save_ip_adapter_weights(file, dtype)
            return result

        # ── Clean up stale sidecar when Kohya removes old checkpoints ──
        _orig_remove = os.remove
        import functools

        @functools.wraps(_orig_remove)
        def _patched_remove(path, **kw):
            result = _orig_remove(path, **kw)
            sidecar = os.path.splitext(path)[0] + ".ipadapter.safetensors"
            if os.path.isfile(sidecar):
                try:
                    _orig_remove(sidecar)
                    logger.info(f"Removed stale IP-Adapter sidecar: {sidecar}")
                except OSError:
                    pass
            return result

        os.remove = _patched_remove

        setattr(cls, _prepare_attr, _patched_prepare)
        cls.save_weights = _patched_save
        os.remove = _patched_remove
        try:
            super().train(args)
        finally:
            setattr(cls, _prepare_attr, _orig_prepare)
            cls.save_weights = _orig_save
            os.remove = _orig_remove
            _BaseDataset.__getitem__ = _orig_getitem
            _BaseDataset.make_buckets = _orig_make_buckets

    def _ensure_precomputed_for_dataset(self, dataset) -> None:
        """Populate the embedding cache once the Kohya dataset is available."""
        if not self._precomputed_dir:
            return
        key = id(dataset)
        if key in self._precomputed_dataset_ids:
            return
        with self._precompute_lock:
            if key in self._precomputed_dataset_ids:
                return
            self._precompute_ip_embeddings(dataset)
            self._precomputed_dataset_ids.add(key)

    @staticmethod
    def _cache_key_for_path(path: str) -> str:
        import hashlib

        return hashlib.md5(path.encode()).hexdigest() + ".pt"

    def _load_frozen_encoders(self, args, weight_dtype) -> None:
        """Load frozen CLIP/CCIP/LSNet encoders (kept on CPU until a forward pass)."""
        if self.clip_image_encoder is not None:
            return
        self.clip_image_encoder = load_clip_vision_model(
            args.clip_model, device="cpu", dtype=weight_dtype
        )
        if "ccip" in self._aux_encoders:
            ccip_ckpt = getattr(args, "ccip_ckpt", DEFAULT_CCIP_CKPT)
            logger.info(f"Loading CCIP encoder from: {ccip_ckpt}")
            self.ccip_encoder = load_ccip_encoder(
                ckpt_path=ccip_ckpt,
                device="cpu",
                dtype=weight_dtype,
            )
        if "lsnet" in self._aux_encoders:
            lsnet_ckpt = getattr(args, "lsnet_ckpt", "")
            if not lsnet_ckpt:
                raise ValueError("--lsnet_ckpt is required when aux_encoders includes 'lsnet'")
            logger.info(f"Loading LSNet encoder from: {lsnet_ckpt}")
            self.lsnet_encoder = load_lsnet_encoder(
                ckpt_path=lsnet_ckpt,
                device="cpu",
                dtype=weight_dtype,
            )

    def _unload_frozen_encoders(self) -> None:
        """Drop frozen encoders and free GPU memory after one-shot precompute."""
        self.clip_image_encoder = None
        self.ccip_encoder = None
        self.lsnet_encoder = None
        if torch.cuda.is_available():
            clean_memory_on_device("cuda")

    def _lookup_cached_features(self, path: str) -> Optional[dict[str, Any]]:
        """Return cached raw encoder features for ``path``, loading from disk if needed."""
        key = self._cache_key_for_path(path)
        feats = self._precomputed_cache.get(key)
        if feats is not None:
            return feats
        if not self._precomputed_dir:
            return None
        cache_file = os.path.join(self._precomputed_dir, key)
        if not os.path.isfile(cache_file):
            return None
        try:
            feats = torch.load(cache_file, map_location="cpu", weights_only=True)
        except Exception:
            return None
        self._precomputed_cache[key] = feats
        return feats

    def _is_path_cached(self, path: str) -> bool:
        """Return True if ``path`` has all required fields in the embedding cache."""
        path = os.path.abspath(path)
        feats = self._lookup_cached_features(path)
        if feats is None:
            return False
        required = self._required_cache_fields()
        return all(feats.get(field) is not None for field in required)

    def _required_cache_fields(self) -> list[str]:
        fields = ["clip"]
        if "ccip" in self._aux_encoders:
            fields.append("ccip")
        if "lsnet" in self._aux_encoders:
            fields.append("lsnet")
        if self._ipa_mode != "simple":
            fields.append("clip_patches")
            if "ccip" in self._aux_encoders:
                fields.append("ccip_patches")
            if "lsnet" in self._aux_encoders:
                fields.append("lsnet_patches")
        return fields

    def _is_precomputed_cache_complete(self, datasets: list[Any]) -> bool:
        """Return True if all dataset images have all required cached features."""
        required = self._required_cache_fields()
        missing = []
        incomplete = []

        for dataset in datasets:
            for info in dataset.image_data.values():
                key = self._cache_key_for_path(info.absolute_path)
                feats = self._precomputed_cache.get(key)
                if feats is None:
                    missing.append(info.absolute_path)
                    if len(missing) >= 5:
                        break
                    continue
                absent = [field for field in required if feats.get(field) is None]
                if absent:
                    incomplete.append((info.absolute_path, absent))
                    if len(incomplete) >= 5:
                        break
            if len(missing) >= 5 or len(incomplete) >= 5:
                break

        if not missing and not incomplete:
            return True

        if missing:
            logger.info(
                "Precomputed IP cache is incomplete: %s missing file(s), first: %s",
                len(missing),
                ", ".join(missing[:3]),
            )
        if incomplete:
            logger.info(
                "Precomputed IP cache has incomplete feature file(s), first: %s",
                "; ".join(f"{path} missing {fields}" for path, fields in incomplete[:3]),
            )
        return False

    def _ensure_identity_index(self, dataset):
        """Build (and cache) {identity -> [image paths]} for paired sampling.

        Identity = the image's immediate parent directory, i.e. each kohya
        concept folder (``N_name``) is treated as one identity. This matches the
        recommended layout of "one folder per character/style".
        """
        key = id(dataset)
        cached = self._identity_index.get(key)
        if cached is not None:
            return cached
        groups: dict[str, list[str]] = {}
        for _k, info in dataset.image_data.items():
            path = info.absolute_path
            groups.setdefault(os.path.dirname(path), []).append(path)
        self._identity_index[key] = groups
        return groups

    def _choose_reference_path(self, dataset, target_path, pair_by):
        """Pick the reference image path for a given target image.

        ``self``   → the target image itself (copy-prone; ok for quick tests).
        ``folder`` → a *different* image from the same identity folder, which
                     forces the encoder to extract identity/style rather than
                     copy pixels. Falls back to the target if the folder has
                     only one image.
        """
        if pair_by != "folder":
            return target_path
        groups = self._ensure_identity_index(dataset)
        candidates = groups.get(os.path.dirname(target_path))
        if not candidates or len(candidates) <= 1:
            return target_path
        for _ in range(8):  # a few tries to avoid drawing the target itself
            ref = random.choice(candidates)
            if ref != target_path:
                return ref
        return target_path

    def _get_reference_paths_for_index(self, dataset, index):
        """Return IP reference image paths for a bucket batch without reading pixels."""
        pair_by = getattr(self.args, "ip_pair_by", "folder") or "folder"

        bi = dataset.buckets_indices[index]
        bucket = dataset.bucket_manager.buckets[bi.bucket_index]
        bbs = bi.bucket_batch_size
        start = bi.batch_index * bbs
        keys = bucket[start : start + bbs]

        ref_paths = []
        for image_key in keys:
            target_path = dataset.image_data[image_key].absolute_path
            ref_paths.append(self._choose_reference_path(dataset, target_path, pair_by))
        return ref_paths

    def _load_reference_images_for_index(self, dataset, index):
        """Load the IP reference images for the batch at ``index``.

        Loaded directly from disk and resized to a fixed ``ip_cond_size`` square
        (the encoders downsample to 224/384/448 anyway), so references of any
        source size stack cleanly. Order matches the targets produced by
        ``__getitem__``. With ``--ip_pair_by folder`` each reference is a
        different same-identity image (true paired training).

        Returns ``(tensor, paths)`` where ``tensor`` is (B, 3, H, W) in [-1,1]
        and ``paths`` is a list of the absolute image paths used for reference.
        """
        from PIL import Image

        cond_size = int(getattr(self.args, "ip_cond_size", 512) or 512)
        tensors = []
        ref_paths = self._get_reference_paths_for_index(dataset, index)
        for ref_path in ref_paths:
            with Image.open(ref_path) as img:
                img = img.convert("RGB").resize((cond_size, cond_size), Image.BILINEAR)
            arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)  # H,W,3 in [0,1]
            arr = arr.permute(2, 0, 1).contiguous() * 2.0 - 1.0  # 3,H,W in [-1,1]
            tensors.append(arr)

        return torch.stack(tensors, dim=0), ref_paths

    def _ip_adapter_state_dict(self):
        """Collect trainable IP-Adapter weights (projections + image_projs)."""
        sd = {}
        for prefix, mod in (
            ("clip_proj", self.clip_proj),
            ("ccip_proj", self.ccip_proj),
            ("lsnet_proj", self.lsnet_proj),
            ("image_proj", self.image_proj),
            ("image_proj_resampler", self.image_proj_resampler),
        ):
            if mod is not None:
                for k, v in mod.state_dict().items():
                    sd[f"{prefix}.{k}"] = v
        return sd

    def _ip_adapter_sidecar_path(self, network_file: str) -> str:
        base, _ = os.path.splitext(network_file)
        return base + ".ipadapter.safetensors"

    def _save_ip_adapter_weights(self, network_file: str, dtype) -> None:
        """Save clip_proj + image_proj next to the LoRA checkpoint.

        The base ``save_weights`` only persists the (empty) LoRA network, so the
        actual IP-Adapter parameters must be written separately.
        """
        from safetensors.torch import save_file

        sd = self._ip_adapter_state_dict()
        if not sd:
            return
        save_dtype = dtype if dtype is not None else torch.float32
        cpu_sd = {k: v.detach().to(device="cpu", dtype=save_dtype).contiguous()
                  for k, v in sd.items()}
        metadata = {
            "ipa_aux_encoders": ",".join(self._aux_encoders),
            "ipa_num_streams": str(1 + len(self._aux_encoders)),
            "ipa_num_ip_tokens": str(int(getattr(self.args, "num_ip_tokens", 4))),
            "ipa_num_ip_tokens_clip": str(int(getattr(self.args, "num_ip_tokens_clip", None) or 4)),
            "ipa_num_ip_tokens_ccip": str(int(getattr(self.args, "num_ip_tokens_ccip", None) or 4)),
            "ipa_num_ip_tokens_lsnet": str(int(getattr(self.args, "num_ip_tokens_lsnet", None) or 4)),
            "ipa_cross_attention_dim": "1024",
            "ipa_mode": self._ipa_mode,
            "ipa_adapter_type": self._adapter_type,
            "ipa_resampler_type": "omni" if self._adapter_type == "omni" else "resampler",
            "ipa_format": "anima-ipadapter-v1",
        }
        if self._ipa_mode in ("resampler", "double"):
            metadata["ipa_resampler_depth"] = str(4)
            metadata["ipa_resampler_heads"] = str(16)
            metadata["ipa_resampler_dim_head"] = str(64)
            # Per-stream num_queries for resampler (may differ: max(nt, 8) per stream)
            proj = self.image_proj_resampler if self.image_proj_resampler is not None else self.image_proj
            nq = [proj.projs[i].num_queries for i in range(min(len(proj.projs), 3))]
            metadata["ipa_num_queries_clip"] = str(nq[0])
            if len(nq) > 1:
                metadata["ipa_num_queries_ccip"] = str(nq[1])
            if len(nq) > 2:
                metadata["ipa_num_queries_lsnet"] = str(nq[2])
            # Keep compat field
            metadata["ipa_num_queries"] = str(nq[0])
        out_path = self._ip_adapter_sidecar_path(network_file)
        save_file(cpu_sd, out_path, metadata=metadata)
        logger.info(f"Saved IP-Adapter weights ({len(cpu_sd)} tensors) to: {out_path}")

    @staticmethod
    def _resolve_ip_adapter_weights_path(path: str) -> str:
        """Resolve a user path to an existing ``*.ipadapter.safetensors`` sidecar."""
        path = os.path.abspath(path.strip())
        if not path:
            raise ValueError("ip_adapter_weights is empty")

        if path.endswith(".ipadapter.safetensors"):
            if os.path.isfile(path):
                return path
            raise FileNotFoundError(f"IP-Adapter sidecar not found: {path}")

        if path.endswith(".safetensors"):
            sidecar = os.path.splitext(path)[0] + ".ipadapter.safetensors"
            if os.path.isfile(sidecar):
                logger.info(
                    "Resolved ip_adapter_weights LoRA checkpoint to sidecar: %s",
                    sidecar,
                )
                return sidecar
            raise FileNotFoundError(
                f"IP-Adapter sidecar not found for checkpoint: {sidecar}"
            )

        candidates = [
            path + ".ipadapter.safetensors",
            path,
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(
            f"IP-Adapter sidecar not found: {path} "
            "(expected *.ipadapter.safetensors)"
        )

    def _warn_sidecar_metadata_mismatch(self, metadata: dict, args) -> None:
        """Log warnings when saved sidecar config differs from current training args."""
        if not metadata:
            return

        checks = (
            ("ipa_aux_encoders", ",".join(self._aux_encoders)),
            ("ipa_mode", getattr(args, "ipa_mode", "simple") or "simple"),
            ("ipa_adapter_type", getattr(args, "adapter_type", "linear") or "linear"),
        )
        for key, current in checks:
            saved = metadata.get(key)
            if saved is None or str(saved) == str(current):
                continue
            logger.warning(
                "Sidecar metadata %s=%r differs from current training config %r; "
                "loading may fail or produce unexpected results",
                key,
                saved,
                current,
            )

    def load_ip_adapter_weights(
        self,
        sidecar_path: str,
        *,
        args: Optional[argparse.Namespace] = None,
    ) -> None:
        """Load clip_proj + image_proj from a saved sidecar (resume / warm-start)."""
        from safetensors import safe_open

        sidecar_path = os.path.abspath(sidecar_path)
        if not os.path.isfile(sidecar_path):
            raise FileNotFoundError(f"IP-Adapter sidecar not found: {sidecar_path}")

        with safe_open(sidecar_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
            state_dict = {k: f.get_tensor(k) for k in f.keys()}

        if args is not None:
            self._warn_sidecar_metadata_mismatch(metadata, args)

        required_prefixes: list[str] = ["clip_proj", "image_proj"]
        if "ccip" in self._aux_encoders:
            required_prefixes.append("ccip_proj")
        if "lsnet" in self._aux_encoders:
            required_prefixes.append("lsnet_proj")
        if self.image_proj_resampler is not None:
            required_prefixes.append("image_proj_resampler")

        loaded_prefixes: list[str] = []
        for prefix, mod in (
            ("clip_proj", self.clip_proj),
            ("ccip_proj", self.ccip_proj),
            ("lsnet_proj", self.lsnet_proj),
            ("image_proj", self.image_proj),
            ("image_proj_resampler", self.image_proj_resampler),
        ):
            if mod is None:
                continue
            sub = {
                k[len(prefix) + 1:]: v
                for k, v in state_dict.items()
                if k.startswith(prefix + ".")
            }
            if not sub:
                if prefix in required_prefixes:
                    raise RuntimeError(
                        f"Sidecar is missing required weights for '{prefix}' in {sidecar_path}"
                    )
                continue
            mod.load_state_dict(sub, strict=True)
            loaded_prefixes.append(prefix)

        missing = [p for p in required_prefixes if p not in loaded_prefixes]
        if missing:
            raise RuntimeError(
                f"Failed to load required IP-Adapter modules {missing} from {sidecar_path}"
            )

        step = metadata.get("ss_step", metadata.get("step", ""))
        extra = f", step={step}" if step else ""
        logger.info(
            "Loaded IP-Adapter weights (%s modules%s) from: %s",
            ", ".join(loaded_prefixes),
            extra,
            sidecar_path,
        )

    # ── training setup ─────────────────────────────────────────

    def get_trainable_params(self):
        lr = float(getattr(self.args, "learning_rate", 1e-4))
        ip_lr = float(getattr(self.args, "ip_adapter_lr", lr * 5.0))
        params = []

        for proj in (self.clip_proj, self.ccip_proj, self.lsnet_proj):
            if proj is not None:
                params.append({"params": list(proj.parameters()), "lr": ip_lr})

        if isinstance(self.image_proj, MultiStreamProj):
            for proj in self.image_proj.projs:
                params.append({"params": list(proj.parameters()), "lr": ip_lr})
        elif self.image_proj is not None:
            params.append({"params": list(self.image_proj.parameters()), "lr": ip_lr})

        if self.image_proj_resampler is not None:
            for proj in self.image_proj_resampler.projs:
                params.append({"params": list(proj.parameters()), "lr": ip_lr})

        return params

    def _collect_ip_parameter_list(self):
        params = []
        for group in self.get_trainable_params():
            params.extend(group["params"])
        return params

    def get_params_to_clip(self, accelerator, network):
        params = list(accelerator.unwrap_model(network).get_trainable_params())
        if getattr(self, "args", None) is not None:
            params.extend(self._collect_ip_parameter_list())
        return params

    # ── forward: multi-encoder IP tokens ────────────────────────

    def _precompute_image_path(self, path: str) -> bool:
        """Precompute and persist encoder features for a single image path."""
        if not self._precomputed_dir:
            return False
        path = os.path.abspath(path)
        if self._is_path_cached(path):
            return True
        if self.clip_image_encoder is None:
            logger.warning(
                "Cannot precompute IP embedding without frozen encoders: %s", path
            )
            return False

        key = self._cache_key_for_path(path)
        cache_file = os.path.join(self._precomputed_dir, key)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        need_patches = self._ipa_mode != "simple"
        self._ensure_encoders_on_device(device)
        try:
            from PIL import Image
            from torch.nn.functional import interpolate

            img = Image.open(path).convert("RGB")
            arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
            images01 = arr.unsqueeze(0).to(device=device, dtype=dtype).clamp(0, 1)

            feats: dict[str, torch.Tensor] = {}
            clip_in = interpolate(images01, size=(224, 224), mode="bilinear", align_corners=False)
            clip_in = _clip_normalize(clip_in)
            with torch.no_grad():
                clip_out = self.clip_image_encoder(clip_in, output_hidden_states=need_patches)
            feats["clip"] = clip_out.image_embeds.detach().cpu()
            if need_patches:
                feats["clip_patches"] = clip_out.hidden_states[-1][:, 1:, :].detach().cpu()

            if self.ccip_encoder is not None:
                ccip_in = interpolate(images01, size=(384, 384), mode="bilinear", align_corners=False)
                with torch.no_grad():
                    result = self.ccip_encoder(ccip_in, return_patches=need_patches)
                if need_patches:
                    ccip_feat, ccip_patches = result
                    feats["ccip"] = ccip_feat.detach().cpu()
                    feats["ccip_patches"] = ccip_patches.detach().cpu()
                else:
                    feats["ccip"] = result.detach().cpu()

            if self.lsnet_encoder is not None:
                lsnet_in = interpolate(images01, size=(448, 448), mode="bilinear", align_corners=False)
                with torch.no_grad():
                    result = self.lsnet_encoder(lsnet_in, return_patches=need_patches)
                if need_patches:
                    lsnet_feat, lsnet_patches = result
                    feats["lsnet"] = lsnet_feat.detach().cpu()
                    feats["lsnet_patches"] = lsnet_patches.detach().cpu()
                else:
                    feats["lsnet"] = result.detach().cpu()

            torch.save(feats, cache_file)
            self._precomputed_cache[key] = feats
            logger.info(f"Precomputed IP embedding → {cache_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to precompute embedding for {path}: {e}")
            return False

    def _precompute_ip_embeddings(self, dataset):
        """Precompute encoder features for all training images and save to disk.

        Each image path gets a ``<hash>.pt`` file containing clip/ccip/lsnet
        raw features.  Subsequent training steps load from cache instead of
        re-running the frozen encoders.
        """
        if not self._precomputed_dir:
            return
        logger.info(f"Precomputing IP embeddings for {len(dataset.image_data)} images…")
        count = 0
        for _k, info in dataset.image_data.items():
            path = info.absolute_path
            if self._is_path_cached(path):
                count += 1
                continue
            if self._precompute_image_path(path):
                count += 1
                if count % 1000 == 0:
                    logger.info(f"  Precomputed {count}/{len(dataset.image_data)} embeddings…")
        logger.info(f"Precomputed {count} IP embeddings → {self._precomputed_dir}")

    def _encode_images_to_ip_tokens(self, images01, device, weight_dtype, cache_keys=None):
        """Encode [0,1] images → IP tokens for every stream.

        If ``cache_keys`` is provided, looks up precomputed cache in
        ``self._precomputed_cache`` to skip the frozen encoder forwards.
        Returns ``(ip_clip, ip_fine, ip_ccip, ip_lsnet)``.
        """
        from torch.nn.functional import interpolate

        mode = self._ipa_mode
        need_patches = mode != "simple"

        def _project_aux_patches(patches, proj):
            if patches is None:
                return None
            B, L, D = patches.shape
            if D == 1024:
                return patches
            return proj(patches.reshape(-1, D)).reshape(B, L, -1)

        def _split_stream_tokens(ip_list):
            ip_tokens = ip_tokens_ccip = ip_tokens_lsnet = None
            ip_tokens = ip_list[0]
            for i, name in enumerate(self._aux_encoders):
                if name == "ccip":
                    ip_tokens_ccip = ip_list[1 + i]
                elif name == "lsnet":
                    ip_tokens_lsnet = ip_list[1 + i]
            return ip_tokens, ip_tokens_ccip, ip_tokens_lsnet

        def _project_tokens(
            clip_embeds,
            ccip_embeds,
            lsnet_embeds,
            clip_patches,
            ccip_patches,
            lsnet_patches,
        ):
            ip_tokens = ip_tokens_fine = ip_tokens_ccip = ip_tokens_lsnet = None

            if mode == "resampler":
                patches_list = [clip_patches]
                if ccip_patches is not None:
                    patches_list.append(_project_aux_patches(ccip_patches, self.ccip_proj))
                if lsnet_patches is not None:
                    patches_list.append(_project_aux_patches(lsnet_patches, self.lsnet_proj))
                ip_tokens, ip_tokens_ccip, ip_tokens_lsnet = _split_stream_tokens(self.image_proj(patches_list))
            else:
                if isinstance(self.image_proj, MultiStreamProj):
                    embeds_list = [clip_embeds]
                    if ccip_embeds is not None:
                        embeds_list.append(ccip_embeds)
                    if lsnet_embeds is not None:
                        embeds_list.append(lsnet_embeds)
                    ip_tokens, ip_tokens_ccip, ip_tokens_lsnet = _split_stream_tokens(self.image_proj(embeds_list))
                else:
                    ip_tokens = self.image_proj(clip_embeds)

                if self.image_proj_resampler is not None and clip_patches is not None:
                    patches_list = [clip_patches]
                    if ccip_patches is not None:
                        patches_list.append(_project_aux_patches(ccip_patches, self.ccip_proj))
                    if lsnet_patches is not None:
                        patches_list.append(_project_aux_patches(lsnet_patches, self.lsnet_proj))
                    p_list = self.image_proj_resampler(patches_list)
                    ip_tokens_fine = torch.cat(p_list, dim=1) if len(p_list) > 1 else p_list[0]

            def _cast(t):
                return t.to(dtype=weight_dtype) if t is not None else None

            return _cast(ip_tokens), _cast(ip_tokens_fine), _cast(ip_tokens_ccip), _cast(ip_tokens_lsnet)

        # ── Cache lookup ─────────────────────────────────────
        if cache_keys and self._precomputed_cache:
            all_feats = [self._precomputed_cache.get(k) for k in cache_keys]
            has_required_patches = True
            if need_patches:
                required_patch_keys = ["clip_patches"]
                if "ccip" in self._aux_encoders:
                    required_patch_keys.append("ccip_patches")
                if "lsnet" in self._aux_encoders:
                    required_patch_keys.append("lsnet_patches")
                has_required_patches = all(
                    f is not None and all(f.get(k) is not None for k in required_patch_keys)
                    for f in all_feats
                )
            if all(f is not None for f in all_feats) and has_required_patches:
                self._ensure_projectors_on_device(device)
                clip_feats = torch.cat([f["clip"] for f in all_feats], dim=0).to(device=device).float()
                clip_embeds = self.clip_proj(clip_feats)
                clip_patches = (
                    torch.cat([f["clip_patches"] for f in all_feats], dim=0).to(device=device).float()
                    if need_patches else None
                )

                ccip_embeds = None
                ccip_patches = None
                if "ccip" in self._aux_encoders and self.ccip_proj is not None:
                    ccip_feats = torch.cat([f["ccip"] for f in all_feats], dim=0).to(device=device).float()
                    ccip_embeds = self.ccip_proj(ccip_feats)
                    if need_patches and all(f.get("ccip_patches") is not None for f in all_feats):
                        ccip_patches = torch.cat([f["ccip_patches"] for f in all_feats], dim=0).to(device=device).float()

                lsnet_embeds = None
                lsnet_patches = None
                if "lsnet" in self._aux_encoders and self.lsnet_proj is not None:
                    lsnet_feats = torch.cat([f["lsnet"] for f in all_feats], dim=0).to(device=device).float()
                    lsnet_embeds = self.lsnet_proj(lsnet_feats)
                    if need_patches and all(f.get("lsnet_patches") is not None for f in all_feats):
                        lsnet_patches = torch.cat([f["lsnet_patches"] for f in all_feats], dim=0).to(device=device).float()
                return _project_tokens(
                    clip_embeds,
                    ccip_embeds,
                    lsnet_embeds,
                    clip_patches,
                    ccip_patches,
                    lsnet_patches,
                )

        # ── Real encoding (fallback) ─────────────────────────
        if images01 is None:
            raise RuntimeError(
                "IP-Adapter precomputed cache miss and no reference image tensor "
                "is available for fallback encoding. Rebuild the cache or disable "
                "complete-cache encoder skipping."
            )
        self._ensure_encoders_on_device(device)
        images01 = images01.to(device=device, dtype=weight_dtype).clamp(0, 1)
        mode = self._ipa_mode
        need_patches = mode != "simple"

        # ── CLIP ─────────────────────────────────────────────
        clip_input = interpolate(images01, size=(224, 224), mode="bilinear", align_corners=False)
        clip_input = _clip_normalize(clip_input)
        with torch.no_grad():
            clip_out = self.clip_image_encoder(clip_input, output_hidden_states=need_patches)
        clip_embeds = self.clip_proj(clip_out.image_embeds.detach().float())
        clip_patches = clip_out.hidden_states[-1][:, 1:, :].detach().float() if need_patches else None

        # ── CCIP ─────────────────────────────────────────────
        ccip_embeds = ccip_patches = None
        if self.ccip_encoder is not None:
            ccip_input = interpolate(images01, size=(384, 384), mode="bilinear", align_corners=False)
            with torch.no_grad():
                result = self.ccip_encoder(ccip_input, return_patches=need_patches)
            feat = result if not need_patches else result[0]
            ccip_embeds = self.ccip_proj(feat.float())
            ccip_patches = result[1].float() if need_patches else None

        # ── LSNet ────────────────────────────────────────────
        lsnet_embeds = lsnet_patches = None
        if self.lsnet_encoder is not None:
            lsnet_input = interpolate(images01, size=(448, 448), mode="bilinear", align_corners=False)
            with torch.no_grad():
                result = self.lsnet_encoder(lsnet_input, return_patches=need_patches)
            feat = result if not need_patches else result[0]
            lsnet_embeds = self.lsnet_proj(feat.float())
            lsnet_patches = result[1].float() if need_patches else None

        return _project_tokens(
            clip_embeds,
            ccip_embeds,
            lsnet_embeds,
            clip_patches,
            ccip_patches,
            lsnet_patches,
        )

    def _training_weight_dtype(self) -> torch.dtype:
        mp = getattr(self.args, "mixed_precision", None) if hasattr(self, "args") else None
        if mp == "bf16":
            return torch.bfloat16
        if mp == "fp16":
            return torch.float16
        return torch.float32

    def _encode_reference_image(self, args, accelerator, weight_dtype):
        """Load and encode the sample reference image into cached IP tokens.

        Uses the same ``ipa_emb`` cache as training images when available.

        Returns:
            ``(ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet)``
            or ``(None,)*4`` if no reference image is configured.
        """
        ref_path = getattr(args, "sample_reference_image", "") or ""
        if not ref_path:
            return None, None, None, None

        ref_path = os.path.abspath(ref_path)
        if not os.path.isfile(ref_path):
            logger.warning(f"Sample reference image not found, skipping IP injection: {ref_path}")
            return None, None, None, None

        device = accelerator.device
        cache_key = self._cache_key_for_path(ref_path)

        if self._precomputed_dir and self._is_path_cached(ref_path):
            return self._encode_images_to_ip_tokens(
                None, device, weight_dtype, cache_keys=[cache_key]
            )

        if self.clip_image_encoder is None:
            logger.warning(
                "Sample reference image is not in precomputed cache and frozen encoders "
                "are not loaded; skipping IP injection: %s",
                ref_path,
            )
            return None, None, None, None

        from PIL import Image
        from torchvision.transforms import functional as tvf

        pil_img = Image.open(ref_path).convert("RGB")
        images01 = tvf.to_tensor(pil_img).unsqueeze(0)
        return self._encode_images_to_ip_tokens(images01, device, weight_dtype)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        """Override to inject IP tokens during genuine samples only.
        
        Encoding is deferred to on_prompt_start so we don't waste
        encoder forward time when sample_images returns early.
        """
        import os

        ref_path = getattr(args, "sample_reference_image", "") or ""
        has_ref = bool(ref_path and os.path.isfile(ref_path))
        if not has_ref:
            super().sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)
            return

        _trainer_ref = self
        _ip_cached = {}

        def _on_prompt_start(prompt_dict, accel):
            if not _ip_cached:
                tokens, fine_tokens, t_ccip, t_lsnet = _trainer_ref._encode_reference_image(
                    args, accel, _trainer_ref._training_weight_dtype(),
                )
                _ip_cached.update(ip=tokens, fine=fine_tokens, ccip=t_ccip, lsnet=t_lsnet)
                if tokens is not None:
                    logger.info(
                        f"IP-Adapter sample: injecting IP tokens "
                        f"(shape={tokens.shape}"
                        f"{', +fine ' + str(fine_tokens.shape) if fine_tokens is not None else ''}"
                        f"{', +ccip' if t_ccip is not None else ''}"
                        f"{', +lsnet' if t_lsnet is not None else ''})"
                    )
            _trainer_ref._stash_ip_tokens(
                _ip_cached.get("ip"), _ip_cached.get("fine"),
                _ip_cached.get("ccip"), _ip_cached.get("lsnet"),
            )

        def _on_prompt_end(prompt_dict):
            _trainer_ref._stash_ip_tokens(None, None, None, None)

        from library import anima_train_utils, strategy_base
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()

        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            qwen3_te,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
            on_prompt_start=_on_prompt_start,
            on_prompt_end=_on_prompt_end,
        )

    def _ensure_encoders_on_device(self, device):
        """Lazily move auxiliary encoders to the training device."""
        if self.clip_image_encoder is not None:
            clip_dev = next(self.clip_image_encoder.parameters()).device
            if clip_dev != device:
                self.clip_image_encoder.to(device=device)
        self._ensure_projectors_on_device(device)
        if self.ccip_encoder is not None:
            ccip_dev = next(self.ccip_encoder.parameters()).device
            if ccip_dev != device:
                self.ccip_encoder.to(device=device)
        if self.lsnet_encoder is not None:
            lsnet_dev = next(self.lsnet_encoder.parameters()).device
            if lsnet_dev != device:
                self.lsnet_encoder.to(device=device)

    def _ensure_projectors_on_device(self, device):
        """Move only trainable IP projection/resampler modules to the device."""
        for proj in (self.clip_proj, self.ccip_proj, self.lsnet_proj):
            if proj is not None and next(proj.parameters()).device != device:
                proj.to(device=device)
        if self.image_proj is not None:
            proj_dev = next(self.image_proj.parameters()).device
            if proj_dev != device:
                self.image_proj.to(device=device)
        if self.image_proj_resampler is not None:
            resamp_dev = next(self.image_proj_resampler.parameters()).device
            if resamp_dev != device:
                self.image_proj_resampler.to(device=device)

    def _stash_ip_tokens(self, ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet):
        """Stash IP tokens on every AnimaIPCrossAttention module before DiT forward.

        Anima DiT Blocks call ``self.cross_attn(x, attn_params, context=...)``
        with a fixed signature.  We pass IP tokens via instance attributes.
        """
        for attn in self.ip_adapters.values():
            attn._ip_tokens = ip_tokens
            attn._ip_tokens_fine = ip_tokens_fine
            attn._ip_tokens_ccip = ip_tokens_ccip
            attn._ip_tokens_lsnet = ip_tokens_lsnet

    def get_noise_pred_and_target(
        self,
        args, accelerator, noise_scheduler,
        latents, batch, text_encoder_conds,
        unet, network, weight_dtype,
        train_unet=True, is_train=True,
    ):
        # Prefer the dedicated IP reference (may be a paired, same-identity image);
        # fall back to the VAE-target image if the reference field is absent.
        images = batch.get("ip_reference_images")
        if images is None:
            images = batch.get("images")  # (B, C, H, W), normalized to [-1, 1]

        ip_tokens = None
        ip_tokens_fine = None
        ip_tokens_ccip = None
        ip_tokens_lsnet = None

        device = accelerator.device
        paths = batch.get("_image_paths", [])
        cache_keys = [self._cache_key_for_path(p) for p in paths] if self._precomputed_cache and paths else None

        if images is None and not cache_keys and (self.clip_image_encoder is not None or self._precomputed_cache_complete):
            raise RuntimeError(
                "IP-Adapter: no conditioning image or cache key — both "
                "batch['ip_reference_images'] and batch['_image_paths'] are missing. "
                "The trainer patches the dataset to load reference images or cache "
                "keys; if you see this, the __getitem__ patch failed to apply."
            )

        if (images is not None or cache_keys) and (self.clip_image_encoder is not None or self._precomputed_cache):
            images01 = None
            if images is not None:
                # Reference images are normalized to [-1, 1]; all encoders expect
                # [0, 1] before their own normalization.
                images01 = images.float() * 0.5 + 0.5
            ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet = self._encode_images_to_ip_tokens(
                images01, device, weight_dtype, cache_keys=cache_keys
            )

        # Stash IP tokens before DiT forward
        self._stash_ip_tokens(ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet)
        clear_after_forward = not (is_train and getattr(args, "gradient_checkpointing", False))

        try:
            return super().get_noise_pred_and_target(
                args, accelerator, noise_scheduler,
                latents, batch, text_encoder_conds,
                unet, network, weight_dtype,
                train_unet=train_unet, is_train=is_train,
            )
        finally:
            if clear_after_forward:
                self._stash_ip_tokens(None, None, None, None)


def _build_omni_stream(num_queries: int, dim: int = 1024, num_heads: int = 16) -> nn.Module:
    """Build an Omni-Adapter stream: RMSNorm -> proj -> expand -> refiner -> out."""
    from ip_adapter._adapter_modules import RMSNormNoAffine, OmniRefinerBlock

    class OmniStream(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_queries = num_queries
            self.output_dim = dim
            self.norm = RMSNormNoAffine(dim)
            self.proj = nn.Linear(dim, dim)
            self.expand = nn.Linear(dim, dim * num_queries)
            self.refiner = nn.ModuleList([OmniRefinerBlock(dim, num_heads) for _ in range(2)])
            self.out_proj = nn.Linear(dim, dim, bias=False)
            self.out_norm = nn.Identity()

        def forward(self, x):
            _batch, length, width = x.shape
            normed = self.norm(x).to(self.proj.weight.dtype)
            if length == 1:
                tokens = self.expand(normed[:, 0]).reshape(_batch, self.num_queries, width)
            else:
                tokens = self.proj(normed)
            for block in self.refiner:
                tokens = block(tokens)
            return self.out_norm(self.out_proj(tokens))

    return OmniStream()


# ── CLIP helper ──────────────────────────────────────────────────


def load_clip_vision_model(
    model_id: str = "openai/clip-vit-large-patch14",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Load CLIP vision model with projection."""
    from transformers import CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained(
        model_id, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ── CLI ──────────────────────────────────────────────────────────


def setup_parser() -> argparse.ArgumentParser:
    parser = anima_setup_parser()
    parser.set_defaults(network_module="networks.lora_anima")

    parser.add_argument(
        "--train_joint_lora",
        action="store_true",
        help="Train LoRA (networks.lora_anima) jointly with IP-Adapter sidecar. "
             "When unset, network_dim is forced to 0 (IP-Adapter only).",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace model ID for CLIP vision encoder",
    )
    parser.add_argument(
        "--aux_encoders",
        type=str,
        default="",
        help=(
            "Auxiliary encoders: 'clip_only' (default), 'clip_ccip', "
            "'clip_lsnet', 'clip_ccip_lsnet'"
        ),
    )
    parser.add_argument(
        "--ccip_ckpt",
        type=str,
        default=DEFAULT_CCIP_CKPT,
        help="Path to CCIP .ckpt file (when aux_encoders includes 'ccip')",
    )
    parser.add_argument(
        "--lsnet_ckpt",
        type=str,
        default="",
        help="Path to LSNet best_checkpoint.pth (when aux_encoders includes 'lsnet')",
    )
    parser.add_argument(
        "--num_ip_tokens",
        type=int,
        default=4,
        help="Number of IP tokens per encoder stream (shared default)",
    )
    parser.add_argument(
        "--num_ip_tokens_clip",
        type=int,
        default=None,
        help="IP tokens for CLIP stream (defaults to num_ip_tokens)",
    )
    parser.add_argument(
        "--num_ip_tokens_ccip",
        type=int,
        default=None,
        help="IP tokens for CCIP stream (defaults to num_ip_tokens)",
    )
    parser.add_argument(
        "--num_ip_tokens_lsnet",
        type=int,
        default=None,
        help="IP tokens for LSNet stream (defaults to num_ip_tokens)",
    )
    parser.add_argument(
        "--ip_scale",
        type=float,
        default=1.0,
        help="IP cross-attention output multiplier",
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        default="linear",
        choices=["linear", "mlp", "omni"],
        help="ImageProj adapter: linear, mlp, or omni (experimental resampler stream)",
    )
    parser.add_argument(
        "--ip_adapter_lr",
        type=float,
        default=None,
        help="Learning rate for IP-Adapter projection layers only "
             "(defaults to learning_rate * 5.0). Set lower than main lr "
             "if IP path overpowers text.",
    )
    parser.add_argument(
        "--ip_adapter_precomputed_emb_dir",
        type=str,
        default="",
        help="Directory for precomputed IP encoder features (.pt cache). "
             "When set, training reads CLIP/CCIP/LSNet features from cache "
             "instead of running the frozen encoders every step — ~2x speedup.",
    )
    parser.add_argument(
        "--skip_invalid_images",
        action="store_true",
        default=True,
        help="Skip unreadable images or images with invalid dimensions before "
             "bucket creation. Enabled by default for large IPA datasets.",
    )
    parser.add_argument(
        "--no_skip_invalid_images",
        dest="skip_invalid_images",
        action="store_false",
        help="Fail fast instead of skipping unreadable or invalid images.",
    )
    parser.add_argument(
        "--ip_cond_size",
        type=int,
        default=512,
        help="Square resize (px) for the reference image fed to the IP vision "
             "encoders. Encoders downsample to 224/384/448 anyway, so a small "
             "value keeps per-step IO cheap.",
    )
    parser.add_argument(
        "--ip_pair_by",
        type=str,
        default="folder",
        choices=["self", "folder"],
        help="IP reference selection. 'self' = use the training image itself as "
             "the reference (copy-prone; behaves LoRA-like on single-identity "
             "data). 'folder' = sample a DIFFERENT image from the same identity "
             "folder (N_name) as the reference — required to learn a transferable "
             "character/style adapter. Organize the dataset as one folder per "
             "identity for this to work.",
    )
    parser.add_argument(
        "--ipa_mode",
        type=str,
        default="simple",
        choices=["simple", "resampler", "double"],
        help="IP-Adapter mode: simple=global CLIP, resampler=perceiver, double=both",
    )
    parser.add_argument(
        "--sample_reference_image",
        type=str,
        default="",
        help="Path to a reference image for IP-Adapter sampling. "
             "Uses ip_adapter_precomputed_emb_dir cache when available "
             "(same <md5>.pt scheme as training images); otherwise encodes "
             "once via frozen encoders. Does not require keeping encoders "
             "loaded for the whole run when the cache entry exists.",
    )
    parser.add_argument(
        "--ip_adapter_weights",
        type=str,
        default="",
        help="Path to an existing IP-Adapter sidecar (*.ipadapter.safetensors) "
             "to resume or warm-start training. A LoRA checkpoint path "
             "(*.safetensors) is also accepted and resolves to the sibling sidecar.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    if hasattr(args, "attn_mode") and args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    _configure_joint_lora_args(args)
    trainer = AnimaIPAdapterTrainer()
    trainer.train(args)
