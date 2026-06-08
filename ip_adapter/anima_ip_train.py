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
import os
import random
import sys
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

from library.device_utils import init_ipex

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

from ip_adapter import (
    AnimaIPAConverter,
    ImageProjModel,
    Resampler,
    MultiStreamProj,
    AnimaIPAdapter,
)
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
        self.ip_adapters: dict[str, Any] = {}
        self._aux_encoders: tuple[str, ...] = ()
        self._identity_index: dict[int, dict[str, list[str]]] = {}

    @staticmethod
    def _make_stream_projectors(num_streams: int, mode: str, num_queries: int | list[int]) -> MultiStreamProj:
        from ip_adapter.anima_ip_image_proj import Resampler
        if isinstance(num_queries, int):
            num_queries = [num_queries] * num_streams
        if mode == "resampler":
            return MultiStreamProj.from_modules([
                Resampler(dim=1024, depth=4, dim_head=64, heads=16,
                          num_queries=nq, output_dim=1024)
                for nq in num_queries
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

        # CLIP encoder (always loaded). Backbone runs in weight_dtype (frozen);
        # the trainable projection is kept in fp32 for stable optimization.
        self.clip_image_encoder = load_clip_vision_model(
            args.clip_model, device="cpu", dtype=weight_dtype
        )
        self.clip_proj = _make_projection()

        # Parse auxiliary encoders
        self._aux_encoders = _parse_aux_encoders(
            getattr(args, "aux_encoders", "") or ""
        )
        logger.info(f"IP-Adapter auxiliary encoders: {self._aux_encoders or 'none'}")

        # CCIP encoder (if enabled)
        if "ccip" in self._aux_encoders:
            ccip_ckpt = getattr(args, "ccip_ckpt", DEFAULT_CCIP_CKPT)
            logger.info(f"Loading CCIP encoder from: {ccip_ckpt}")
            self.ccip_encoder = load_ccip_encoder(
                ckpt_path=ccip_ckpt,
                device="cpu",
                dtype=weight_dtype,
            )
            self.ccip_proj = _make_projection()

        # LSNet encoder (if enabled)
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
            self.lsnet_proj = _make_projection()

        return model_type, text_encoders, vae, unet

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        dit, text_encoders = super().load_unet_lazily(
            args, weight_dtype, accelerator, text_encoders
        )

        # Inject IP-Adapter layers
        self.ip_adapters = AnimaIPAConverter.create(dit)
        num_streams = 1 + len(self._aux_encoders)
        self._ipa_mode = getattr(args, "ipa_mode", "simple")

        # Per-stream token counts (CLIP, then aux in order)
        _nt = args.num_ip_tokens
        _nt_clip = getattr(args, "num_ip_tokens_clip", None) or _nt
        _nt_ccip = getattr(args, "num_ip_tokens_ccip", None) or _nt
        _nt_lsnet = getattr(args, "num_ip_tokens_lsnet", None) or _nt
        _tokens = [_nt_clip]
        if "ccip" in self._aux_encoders:
            _tokens.append(_nt_ccip)
        if "lsnet" in self._aux_encoders:
            _tokens.append(_nt_lsnet)

        if self._ipa_mode == "simple":
            if num_streams > 1:
                self.image_proj = MultiStreamProj(
                    num_streams=num_streams,
                    cross_attention_dim=1024, embed_dim=1024,
                    tokens_per_stream=_tokens,
                )
            else:
                self.image_proj = ImageProjModel(
                    cross_attention_dim=1024, clip_embeddings_dim=1024,
                    clip_extra_context_tokens=_nt_clip,
                )

        elif self._ipa_mode == "resampler":
            self.image_proj = self._make_stream_projectors(num_streams, "resampler", _tokens)

        elif self._ipa_mode == "double":
            self.image_proj = self._make_stream_projectors(num_streams, "simple", _tokens)
            _tokens_double = [max(n, 8) for n in _tokens]
            self.image_proj_resampler = self._make_stream_projectors(
                num_streams, "resampler", _tokens_double
            )

        else:
            raise ValueError(f"Unknown ipa_mode: {self._ipa_mode}")

        return dit, text_encoders

    # ── training ───────────────────────────────────────────────

    def train(self, args):
        """Override train() to inject ImageProjModel params into the LoRA network
        param collection, since IP-Adapter trains ImageProjModel instead of LoRA.
        """
        self.args = args

        # IP-Adapter needs a raw conditioning image for the vision encoders every
        # step, decoupled from the VAE target (batch["images"]). We patch the
        # dataset to add a dedicated `ip_reference_images` field, loaded directly
        # from disk (downscaled) so it works regardless of latent caching, and so
        # the reference can be a *different* image than the target (paired mode).
        import library.train_util as _tu

        _BaseDataset = _tu.BaseDataset
        _orig_getitem = _BaseDataset.__getitem__
        _trainer_ref = self

        def _patched_getitem(ds_self, index):
            example = _orig_getitem(ds_self, index)
            if (
                _trainer_ref.clip_image_encoder is not None
                and getattr(ds_self, "caching_mode", None) is None
            ):
                example["ip_reference_images"] = _trainer_ref._load_reference_images_for_index(
                    ds_self, index
                )
            return example

        _BaseDataset.__getitem__ = _patched_getitem

        import networks.lora

        cls = networks.lora.LoRANetwork
        _orig_prepare = cls.prepare_optimizer_params
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
            # IP-Adapter trains no LoRA modules, so the network state dict is
            # empty. Skip writing that empty placeholder checkpoint and only
            # persist the real IP-Adapter sidecar. (save_model only calls
            # save_weights + optional HF upload, and remove_model checks
            # existence, so skipping the file is safe.)
            if len(self_network.state_dict()) == 0:
                _trainer_ref._save_ip_adapter_weights(file, dtype)
                return None
            result = _orig_save(self_network, file, dtype, metadata, *a, **kw)
            _trainer_ref._save_ip_adapter_weights(file, dtype)
            return result

        cls.prepare_optimizer_params = _patched_prepare
        cls.save_weights = _patched_save
        try:
            super().train(args)
        finally:
            cls.prepare_optimizer_params = _orig_prepare
            cls.save_weights = _orig_save
            _BaseDataset.__getitem__ = _orig_getitem

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

    def _load_reference_images_for_index(self, dataset, index):
        """Load the IP reference images for the batch at ``index``.

        Loaded directly from disk and resized to a fixed ``ip_cond_size`` square
        (the encoders downsample to 224/384/448 anyway), so references of any
        source size stack cleanly. Order matches the targets produced by
        ``__getitem__``. With ``--ip_pair_by folder`` each reference is a
        different same-identity image (true paired training).
        """
        from PIL import Image

        cond_size = int(getattr(self.args, "ip_cond_size", 512) or 512)
        pair_by = getattr(self.args, "ip_pair_by", "self") or "self"

        bi = dataset.buckets_indices[index]
        bucket = dataset.bucket_manager.buckets[bi.bucket_index]
        bbs = bi.bucket_batch_size
        start = bi.batch_index * bbs
        keys = bucket[start : start + bbs]

        tensors = []
        for image_key in keys:
            target_path = dataset.image_data[image_key].absolute_path
            ref_path = self._choose_reference_path(dataset, target_path, pair_by)
            with Image.open(ref_path) as img:
                img = img.convert("RGB").resize((cond_size, cond_size), Image.BILINEAR)
            arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)  # H,W,3 in [0,1]
            arr = arr.permute(2, 0, 1).contiguous() * 2.0 - 1.0  # 3,H,W in [-1,1]
            tensors.append(arr)

        return torch.stack(tensors, dim=0)

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
            "ipa_cross_attention_dim": "1024",
            "ipa_mode": self._ipa_mode,
            "ipa_format": "anima-ipadapter-v1",
        }
        if self._ipa_mode in ("resampler", "double"):
            metadata["ipa_resampler_depth"] = str(4)
            metadata["ipa_resampler_heads"] = str(16)
            metadata["ipa_resampler_dim_head"] = str(64)
            metadata["ipa_num_queries"] = str(
                self.image_proj_resampler.projs[0].num_queries
                if self.image_proj_resampler is not None
                else getattr(self.image_proj.projs[0], "num_queries", getattr(self.args, "num_ip_tokens", 16))
            )
        out_path = self._ip_adapter_sidecar_path(network_file)
        save_file(cpu_sd, out_path, metadata=metadata)
        logger.info(f"Saved IP-Adapter weights ({len(cpu_sd)} tensors) to: {out_path}")

    def load_ip_adapter_weights(self, sidecar_path: str) -> None:
        """Load clip_proj + image_proj from a saved sidecar (for resume/inference)."""
        from safetensors.torch import load_file

        sd = load_file(sidecar_path)
        for prefix, mod in (
            ("clip_proj", self.clip_proj),
            ("ccip_proj", self.ccip_proj),
            ("lsnet_proj", self.lsnet_proj),
            ("image_proj", self.image_proj),
            ("image_proj_resampler", self.image_proj_resampler),
        ):
            if mod is None:
                continue
            sub = {k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix + ".")}
            if sub:
                mod.load_state_dict(sub)
        logger.info(f"Loaded IP-Adapter weights from: {sidecar_path}")

    # ── training setup ─────────────────────────────────────────

    def get_trainable_params(self):
        lr = float(getattr(self.args, "learning_rate", 1e-4))
        params = []

        for proj in (self.clip_proj, self.ccip_proj, self.lsnet_proj):
            if proj is not None:
                params.append({"params": list(proj.parameters()), "lr": lr})

        if isinstance(self.image_proj, MultiStreamProj):
            for proj in self.image_proj.projs:
                params.append({"params": list(proj.parameters()), "lr": lr})
        elif self.image_proj is not None:
            params.append({"params": list(self.image_proj.parameters()), "lr": lr})

        if self.image_proj_resampler is not None:
            for proj in self.image_proj_resampler.projs:
                params.append({"params": list(proj.parameters()), "lr": lr})

        return params

    # ── forward: multi-encoder IP tokens ────────────────────────

    def _encode_images_to_ip_tokens(self, images01, device, weight_dtype):
        """Encode [0,1] images → IP tokens for every stream.

        Returns ``(ip_clip, ip_fine, ip_ccip, ip_lsnet)``. ``ip_fine``
        is only non-None in ``double`` mode (resampler patch-level tokens).
        """
        from torch.nn.functional import interpolate

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

        # ── Global IP tokens (all modes) ─────────────────────
        ip_tokens = ip_tokens_ccip = ip_tokens_lsnet = None
        if isinstance(self.image_proj, MultiStreamProj):
            embeds_list = [clip_embeds]
            if ccip_embeds is not None: embeds_list.append(ccip_embeds)
            if lsnet_embeds is not None: embeds_list.append(lsnet_embeds)
            ip_list = self.image_proj(embeds_list)
            ip_tokens = ip_list[0]
            for i, name in enumerate(self._aux_encoders):
                if name == "ccip":
                    ip_tokens_ccip = ip_list[1 + i]
                elif name == "lsnet":
                    ip_tokens_lsnet = ip_list[1 + i]
        else:
            ip_tokens = self.image_proj(clip_embeds)

        # ── Fine IP tokens (double mode) ─────────────────────
        ip_tokens_fine = None
        if self.image_proj_resampler is not None and clip_patches is not None:
            patches_list = [clip_patches]
            # Project 768-dim patches → 1024-dim via the encoder proj
            if ccip_patches is not None:
                B, L, D = ccip_patches.shape
                ccip_patches = self.ccip_proj(ccip_patches.reshape(-1, D)).reshape(B, L, -1)
                patches_list.append(ccip_patches)
            if lsnet_patches is not None:
                B, L, D = lsnet_patches.shape
                lsnet_patches = self.lsnet_proj(lsnet_patches.reshape(-1, D)).reshape(B, L, -1)
                patches_list.append(lsnet_patches)
            p_list = self.image_proj_resampler(patches_list)
            ip_tokens_fine = p_list[0]

        def _cast(t):
            return t.to(dtype=weight_dtype) if t is not None else None

        return _cast(ip_tokens), _cast(ip_tokens_fine), _cast(ip_tokens_ccip), _cast(ip_tokens_lsnet)

    def _encode_reference_image(self, args, accelerator, weight_dtype):
        """Load and encode the sample reference image into cached IP tokens.

        Returns:
            ``(ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet)``
            or ``(None,)*4`` if no reference image is configured.
        """
        ref_path = getattr(args, "sample_reference_image", "") or ""
        if not ref_path:
            return None, None, None, None

        import os
        if not os.path.isfile(ref_path):
            logger.warning(f"Sample reference image not found, skipping IP injection: {ref_path}")
            return None, None, None, None

        from PIL import Image
        from torchvision.transforms import functional as tvf

        device = accelerator.device

        pil_img = Image.open(ref_path).convert("RGB")
        # to_tensor → [0,1]; same range the training path feeds the encoders.
        images01 = tvf.to_tensor(pil_img).unsqueeze(0)

        ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet = self._encode_images_to_ip_tokens(
            images01, device, weight_dtype
        )
        return ip_tokens, None, ip_tokens_ccip, ip_tokens_lsnet

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
                tokens, _, t_ccip, t_lsnet = _trainer_ref._encode_reference_image(
                    args, accel,
                    _trainer_ref.clip_image_encoder.dtype if _trainer_ref.clip_image_encoder else torch.bfloat16,
                )
                _ip_cached.update(ip=tokens, ccip=t_ccip, lsnet=t_lsnet)
                if tokens is not None:
                    logger.info(
                        f"IP-Adapter sample: injecting IP tokens "
                        f"(shape={tokens.shape}"
                        f"{', +ccip' if t_ccip is not None else ''}"
                        f"{', +lsnet' if t_lsnet is not None else ''})"
                    )
            _trainer_ref._stash_ip_tokens(
                _ip_cached.get("ip"), None,
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
        for proj in (self.clip_proj, self.ccip_proj, self.lsnet_proj):
            if proj is not None and next(proj.parameters()).device != device:
                proj.to(device=device)
        if self.ccip_encoder is not None:
            ccip_dev = next(self.ccip_encoder.parameters()).device
            if ccip_dev != device:
                self.ccip_encoder.to(device=device)
        if self.lsnet_encoder is not None:
            lsnet_dev = next(self.lsnet_encoder.parameters()).device
            if lsnet_dev != device:
                self.lsnet_encoder.to(device=device)
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

        if self.clip_image_encoder is not None and images is None:
            raise RuntimeError(
                "IP-Adapter: no conditioning image — both batch['ip_reference_images'] "
                "and batch['images'] are None. The trainer patches the dataset to load "
                "reference images; if you see this, the __getitem__ patch failed to apply "
                "(check that the dataset subclasses library.train_util.BaseDataset)."
            )

        if images is not None and self.clip_image_encoder is not None:
            device = accelerator.device
            # Reference images are normalized to [-1, 1]; all encoders expect
            # [0, 1] before their own normalization.
            images01 = images.float() * 0.5 + 0.5
            ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet = self._encode_images_to_ip_tokens(
                images01, device, weight_dtype
            )

        # Stash IP tokens before DiT forward
        self._stash_ip_tokens(ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet)

        return super().get_noise_pred_and_target(
            args, accelerator, noise_scheduler,
            latents, batch, text_encoder_conds,
            unet, network, weight_dtype,
            train_unet=train_unet, is_train=is_train,
        )


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
    parser.set_defaults(network_module="networks.lora")  # IP-Adapter doesn't use LoRA, but train() expects this

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
        default="self",
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
             "Encodes the image with CLIP (+CCIP/+LSNet if enabled) "
             "and injects IP tokens during sample generation.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    if hasattr(args, "attn_mode") and args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    trainer = AnimaIPAdapterTrainer()
    trainer.train(args)
