"""
Anima IP-Adapter Training Script

Extends ``AnimaNetworkTrainer`` to support IP-Adapter training on
top of Anima DiT models.  Injects trainable ``ip_k_proj`` / ``ip_v_proj``
layers into every Block's cross-attention while keeping the base model frozen.

Usage (standalone):
  accelerate launch ip_adapter/anima_ip_train.py \\
    --pretrained_model_name_or_path anima.safetensors \\
    --vae qwen_image_vae.safetensors \\
    --qwen3 qwen3.safetensors \\
    --clip_model openai/clip-vit-large-patch14 \\
    --train_data_dir ./train/anima_ip_dataset \\
    --output_dir ./output/ipa \\
    --num_ip_tokens 4 \\
    --ip_scale 1.0 \\
    --learning_rate 1e-4 \\
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

# Import internal IP-Adapter modules (namespace-safe, no vendor coupling)
from ip_adapter import AnimaIPAConverter, ImageProjModel, Resampler, AnimaIPAdapter


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AnimaIPAdapterTrainer(AnimaNetworkTrainer):
    """Anima NetworkTrainer extended with IP-Adapter support.

    Injects trainable ``ip_k_proj`` / ``ip_v_proj`` into every
    Block.cross_attn.  Only the IP layers + image_proj are trained;
    the base DiT stays frozen.
    """

    def __init__(self):
        super().__init__()
        self.clip_image_encoder: Optional[nn.Module] = None
        self.image_proj: Optional[ImageProjModel] = None
        self.ip_adapters: dict[str, Any] = {}

    # ── model loading ──────────────────────────────────────────
    def load_target_model(self, args, weight_dtype, accelerator):
        model_type, text_encoders, vae, unet = super().load_target_model(
            args, weight_dtype, accelerator
        )
        # Load CLIP image encoder on CPU (will be moved during training)
        self.clip_image_encoder = load_clip_vision_model(
            args.clip_model, device="cpu", dtype=weight_dtype
        )
        return model_type, text_encoders, vae, unet

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        dit, text_encoders = super().load_unet_lazily(
            args, weight_dtype, accelerator, text_encoders
        )
        # Inject IP-Adapter layers into DiT
        self.ip_adapters = AnimaIPAConverter.create(
            dit, ip_scale=args.ip_scale,
            ipa_mode=getattr(args, "ipa_mode", "simple"),
        )
        logger.info(
            f"IP-Adapter injected into {len(self.ip_adapters)} Block(s), mode={mode}"
        )

        # Image projection model — choose by ipa_mode
        mode = getattr(args, "ipa_mode", "simple")
        if mode == "simple":
            self.image_proj = ImageProjModel(
                cross_attention_dim=1024,
                clip_embeddings_dim=1024,
                clip_extra_context_tokens=args.num_ip_tokens,
            )
        elif mode == "resampler":
            self.image_proj = Resampler(
                dim=1024, depth=4, dim_head=64, heads=16,
                num_queries=args.num_ip_tokens, output_dim=1024,
            )
        elif mode == "double":
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
        """Collect parameters for optimizer.

        Returns:
            List of param groups.  Only IP layers + image_proj are
            marked as trainable; the base DiT stays frozen.
        """
        params = []
        params.append({
            "params": list(self.image_proj.parameters()),
            "lr": float(getattr(args, "learning_rate", 1e-4)),
        })
        for attn in self.ip_adapters.values():
            params.append({
                "params": list(attn.trainable_parameters()),
                "lr": float(getattr(args, "learning_rate", 1e-4)),
            })
        return params

    # ── forward: inject IP tokens ──────────────────────────────
    def get_noise_pred_and_target(
        self,
        args, accelerator, noise_scheduler,
        latents, batch, text_encoder_conds,
        unet, network, weight_dtype,
        train_unet=True, is_train=True,
    ):
        """Override to compute & inject IP tokens before the DiT forward."""
        from torch.nn.functional import interpolate

        # Get raw training images from Kohya loader (standard format)
        images = batch.get("images")  # (B, C, H, W) raw pixels, already normalized
            if images is not None:
                # Resize to CLIP input size (224×224)
                clip_input = interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
                clip_input = clip_input.to(device=accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    clip_outputs = self.clip_image_encoder(clip_input, output_hidden_states=True)

                mode = getattr(args, "ipa_mode", "simple")
                if mode == "double":
                    # Global stream
                    ip_global = self.image_proj(clip_outputs.image_embeds)  # (B, N, 1024)
                    # Fine stream — patch features from penultimate layer
                    patch_feats = clip_outputs.hidden_states[-2][:, 1:, :]  # (B, 256, 1280)
                    ip_fine = self.image_proj_fine(patch_feats)              # (B, N_fine, 1024)
                    self._ip_tokens = ip_global
                    self._ip_tokens_fine = ip_fine
                elif mode == "resampler":
                    patch_feats = clip_outputs.hidden_states[-2][:, 1:, :]
                    self._ip_tokens = self.image_proj(patch_feats)
                else:
                    self._ip_tokens = self.image_proj(clip_outputs.image_embeds)
        else:
            self._ip_tokens = None

        return super().get_noise_pred_and_target(
            args, accelerator, noise_scheduler,
            latents, batch, text_encoder_conds,
            unet, network, weight_dtype,
            train_unet=train_unet, is_train=is_train,
        )


# ---------------------------------------------------------------------------
# CLIP loading helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_parser() -> argparse.ArgumentParser:
    parser = AnimaNetworkTrainer.setup_parser()
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace model ID for CLIP vision encoder",
    )
    parser.add_argument(
        "--num_ip_tokens",
        type=int,
        default=4,
        help="Number of IP tokens (4 recommended for MVP)",
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
        help="IP-Adapter mode: simple=global CLIP feature, resampler=perceiver over patches, double=both",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    if hasattr(args, 'attn_mode') and args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    trainer = AnimaIPAdapterTrainer()
    trainer.train(args)
