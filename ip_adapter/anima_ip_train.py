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

Each auxiliary encoder has a learnable scalar gate initialized at 0.1
so the model starts CLIP-dominant and gradually activates auxiliary signals.

Usage (standalone):
  accelerate launch ip_adapter/anima_ip_train.py \
    --pretrained_model_name_or_path anima.safetensors \
    --vae qwen_image_vae.safetensors \
    --qwen3 qwen3.safetensors \
    --clip_model openai/clip-vit-large-patch14 \
    --aux_encoders clip_ccip_lsnet \
    --ccip_model ccip-caformer-24-randaug-pruned \
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
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from library.device_utils import init_ipex

init_ipex()

from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    strategy_anima,
    train_util,
)
from vendor.sd-scripts.anima_train_network import AnimaNetworkTrainer
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
        self.ccip_encoder: Optional[nn.Module] = None
        self.lsnet_encoder: Optional[nn.Module] = None
        self.image_proj: Optional[ImageProjModel | MultiStreamProj] = None
        self.image_proj_fine: Optional[Resampler] = None
        self.ip_adapters: dict[str, Any] = {}
        self._aux_encoders: tuple[str, ...] = ()

    # ── model loading ──────────────────────────────────────────

    def load_target_model(self, args, weight_dtype, accelerator):
        model_type, text_encoders, vae, unet = super().load_target_model(
            args, weight_dtype, accelerator
        )

        # CLIP encoder (always loaded)
        self.clip_image_encoder = load_clip_vision_model(
            args.clip_model, device="cpu", dtype=weight_dtype
        )

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

        return model_type, text_encoders, vae, unet

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        dit, text_encoders = super().load_unet_lazily(
            args, weight_dtype, accelerator, text_encoders
        )

        ipa_mode = getattr(args, "ipa_mode", "simple")

        # Inject IP-Adapter layers with auxiliary encoder awareness
        self.ip_adapters = AnimaIPAConverter.create(
            dit,
            ip_scale=args.ip_scale,
            ipa_mode=ipa_mode,
            aux_encoders=self._aux_encoders,
        )
        logger.info(
            f"IP-Adapter injected into {len(self.ip_adapters)} Block(s), "
            f"mode={ipa_mode}, aux_encoders={self._aux_encoders}"
        )

        # Determine number of streams
        num_streams = 1 + len(self._aux_encoders)  # CLIP + aux

        if num_streams > 1:
            # Multi-stream: all encoders share the same num_ip_tokens
            self.image_proj = MultiStreamProj(
                num_streams=num_streams,
                cross_attention_dim=1024,
                embed_dim=1024,
                tokens_per_stream=[args.num_ip_tokens] * num_streams,
            )
        else:
            # Single-stream (original behavior)
            if ipa_mode == "simple":
                self.image_proj = ImageProjModel(
                    cross_attention_dim=1024,
                    clip_embeddings_dim=1024,
                    clip_extra_context_tokens=args.num_ip_tokens,
                )
            elif ipa_mode == "resampler":
                self.image_proj = Resampler(
                    dim=1024, depth=4, dim_head=64, heads=16,
                    num_queries=args.num_ip_tokens, output_dim=1024,
                )
            elif ipa_mode == "double":
                self.image_proj = ImageProjModel(
                    cross_attention_dim=1024,
                    clip_embeddings_dim=1024,
                    clip_extra_context_tokens=args.num_ip_tokens,
                )
                self.image_proj_fine = Resampler(
                    dim=1024, depth=4, dim_head=64, heads=16,
                    num_queries=max(args.num_ip_tokens, 8), output_dim=1024,
                )
            else:
                self.image_proj = ImageProjModel(
                    cross_attention_dim=1024,
                    clip_embeddings_dim=1024,
                    clip_extra_context_tokens=args.num_ip_tokens,
                )

        return dit, text_encoders

    # ── training setup ─────────────────────────────────────────

    def get_trainable_params(self):
        params = []

        if isinstance(self.image_proj, MultiStreamProj):
            for proj in self.image_proj.projs:
                params.append({
                    "params": list(proj.parameters()),
                    "lr": float(getattr(args, "learning_rate", 1e-4)),
                })
        else:
            params.append({
                "params": list(self.image_proj.parameters()),
                "lr": float(getattr(args, "learning_rate", 1e-4)),
            })

        if self.image_proj_fine is not None:
            params.append({
                "params": list(self.image_proj_fine.parameters()),
                "lr": float(getattr(args, "learning_rate", 1e-4)),
            })

        for attn in self.ip_adapters.values():
            params.append({
                "params": list(attn.trainable_parameters()),
                "lr": float(getattr(args, "learning_rate", 1e-4)),
            })

        return params

    # ── forward: multi-encoder IP tokens ────────────────────────

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
        from torch.nn.functional import interpolate
        from torchvision.transforms import functional as tvf

        device = accelerator.device
        self._ensure_encoders_on_device(device)

        pil_img = Image.open(ref_path).convert("RGB")

        # ── CLIP encoding (224x224) ─────────────────────────
        clip_tensor = tvf.to_tensor(pil_img).unsqueeze(0).to(device=device, dtype=weight_dtype)
        clip_tensor = interpolate(clip_tensor, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            clip_outputs = self.clip_image_encoder(clip_tensor, output_hidden_states=True)
        clip_embeds = clip_outputs.image_embeds  # (1, 1024)

        # ── CCIP encoding (384x384) ──────────────────────────
        ccip_embeds = None
        if self.ccip_encoder is not None:
            ccip_tensor = tvf.to_tensor(pil_img).unsqueeze(0).to(device=device, dtype=weight_dtype)
            ccip_tensor = interpolate(ccip_tensor, size=(384, 384), mode="bilinear", align_corners=False)
            with torch.no_grad():
                ccip_embeds = self.ccip_encoder(ccip_tensor)

        # ── LSNet encoding (448x448) ─────────────────────────
        lsnet_embeds = None
        if self.lsnet_encoder is not None:
            lsnet_tensor = tvf.to_tensor(pil_img).unsqueeze(0).to(device=device, dtype=weight_dtype)
            lsnet_tensor = interpolate(lsnet_tensor, size=(448, 448), mode="bilinear", align_corners=False)
            with torch.no_grad():
                lsnet_embeds = self.lsnet_encoder(lsnet_tensor)

        ipa_mode = getattr(args, "ipa_mode", "simple")

        ip_tokens = None
        ip_tokens_fine = None
        ip_tokens_ccip = None
        ip_tokens_lsnet = None

        if isinstance(self.image_proj, MultiStreamProj):
            embeds_list = [clip_embeds]
            if ccip_embeds is not None:
                embeds_list.append(ccip_embeds)
            if lsnet_embeds is not None:
                embeds_list.append(lsnet_embeds)
            ip_list = self.image_proj(embeds_list)
            ip_tokens = ip_list[0]
            aux = list(self._aux_encoders)
            for i, name in enumerate(aux):
                if name == "ccip":
                    ip_tokens_ccip = ip_list[1 + i]
                elif name == "lsnet":
                    ip_tokens_lsnet = ip_list[1 + i]
        elif ipa_mode == "double":
            ip_tokens = self.image_proj(clip_embeds)
            patch_feats = clip_outputs.hidden_states[-2][:, 1:, :]
            ip_tokens_fine = self.image_proj_fine(patch_feats) if self.image_proj_fine is not None else None
        elif ipa_mode == "resampler":
            patch_feats = clip_outputs.hidden_states[-2][:, 1:, :]
            ip_tokens = self.image_proj(patch_feats)
        else:
            ip_tokens = self.image_proj(clip_embeds)

        return ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        """Override to inject IP tokens via callbacks during sampling.

        Encodes the reference image once and stashes IP tokens before
        each prompt's DiT forward.  Tokens are cleared after each prompt.
        """
        import functools

        ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet = (
            self._encode_reference_image(args, accelerator, 
                                         self.clip_image_encoder.dtype 
                                         if self.clip_image_encoder is not None 
                                         else torch.bfloat16)
        )

        has_ip = ip_tokens is not None

        def _on_prompt_start(prompt_dict, accel):
            if has_ip:
                self._stash_ip_tokens(ip_tokens, ip_tokens_fine, ip_tokens_ccip, ip_tokens_lsnet)

        def _on_prompt_end(prompt_dict):
            if has_ip:
                self._stash_ip_tokens(None, None, None, None)

        from library import anima_train_utils, strategy_base
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()

        if has_ip:
            logger.info(
                f"IP-Adapter sample: injecting IP tokens ("
                f"shape={ip_tokens.shape if ip_tokens is not None else 'none'}"
                f"{', +ccip' if ip_tokens_ccip is not None else ''}"
                f"{', +lsnet' if ip_tokens_lsnet is not None else ''}"
                f")"
            )

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
            on_prompt_start=_on_prompt_start if has_ip else None,
            on_prompt_end=_on_prompt_end if has_ip else None,
        )

    def _ensure_encoders_on_device(self, device):
        """Lazily move auxiliary encoders to the training device."""
        if self.ccip_encoder is not None:
            ccip_dev = next(self.ccip_encoder.parameters()).device
            if ccip_dev != device:
                self.ccip_encoder.to(device=device)
        if self.lsnet_encoder is not None:
            lsnet_dev = next(self.lsnet_encoder.parameters()).device
            if lsnet_dev != device:
                self.lsnet_encoder.to(device=device)

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
        from torch.nn.functional import interpolate

        images = batch.get("images")  # (B, C, H, W) raw pixels

        ip_tokens = None
        ip_tokens_fine = None
        ip_tokens_ccip = None
        ip_tokens_lsnet = None

        if images is not None and self.clip_image_encoder is not None:
            device = accelerator.device
            self._ensure_encoders_on_device(device)

            # ── CLIP encoding ────────────────────────────────
            clip_input = interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
            clip_input = clip_input.to(device=device, dtype=weight_dtype)
            with torch.no_grad():
                clip_outputs = self.clip_image_encoder(clip_input, output_hidden_states=True)
            clip_embeds = clip_outputs.image_embeds  # (B, 1024)

            # ── CCIP encoding (GPU-native, 384×384) ─────────
            ccip_embeds = None
            if self.ccip_encoder is not None:
                ccip_input = interpolate(images, size=(384, 384), mode="bilinear", align_corners=False)
                ccip_input = ccip_input.to(device=device, dtype=weight_dtype)
                with torch.no_grad():
                    ccip_embeds = self.ccip_encoder(ccip_input)

            # ── LSNet encoding (if enabled) ──────────────────
            lsnet_embeds = None
            if self.lsnet_encoder is not None:
                lsnet_input = interpolate(images, size=(448, 448), mode="bilinear", align_corners=False)
                lsnet_input = lsnet_input.to(device=device, dtype=weight_dtype)
                with torch.no_grad():
                    lsnet_embeds = self.lsnet_encoder(lsnet_input)

            # ── Project to IP tokens ─────────────────────────
            ipa_mode = getattr(args, "ipa_mode", "simple")

            if isinstance(self.image_proj, MultiStreamProj):
                embeds_list = [clip_embeds]
                if ccip_embeds is not None:
                    embeds_list.append(ccip_embeds)
                if lsnet_embeds is not None:
                    embeds_list.append(lsnet_embeds)
                ip_list = self.image_proj(embeds_list)
                # ip_list = [ip_tokens_clip, ip_tokens_ccip?, ip_tokens_lsnet?]
                ip_tokens = ip_list[0]
                aux = list(self._aux_encoders)
                for i, name in enumerate(aux):
                    if name == "ccip":
                        ip_tokens_ccip = ip_list[1 + i]
                    elif name == "lsnet":
                        ip_tokens_lsnet = ip_list[1 + i]
            elif ipa_mode == "double":
                ip_tokens = self.image_proj(clip_embeds)
                patch_feats = clip_outputs.hidden_states[-2][:, 1:, :]
                ip_tokens_fine = self.image_proj_fine(patch_feats)
            elif ipa_mode == "resampler":
                patch_feats = clip_outputs.hidden_states[-2][:, 1:, :]
                ip_tokens = self.image_proj(patch_feats)
            else:
                ip_tokens = self.image_proj(clip_embeds)

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
    parser = AnimaNetworkTrainer.setup_parser()

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
        help="Number of IP tokens per encoder stream (shared across CLIP and aux)",
    )
    parser.add_argument(
        "--ip_scale",
        type=float,
        default=1.0,
        help="IP cross-attention output multiplier",
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
