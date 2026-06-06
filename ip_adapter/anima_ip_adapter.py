"""
Anima IP-Adapter Inference Wrapper

Loads a pre-trained IP-Adapter checkpoint, handles CLIP image encoding,
and injects IP tokens into the DiT model at inference time.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .anima_ip_attention import AnimaIPCrossAttention
from .anima_ip_converter import AnimaIPAConverter
from .anima_ip_image_proj import ImageProjModel


class AnimaIPAdapter:
    """Load and apply IP-Adapter weights to an Anima DiT model."""

    def __init__(
        self,
        dit: nn.Module,
        clip_image_encoder: nn.Module,
        image_proj: ImageProjModel,
        ip_adapters: Dict[str, AnimaIPCrossAttention],
        ip_scale: float = 1.0,
    ):
        self.dit = dit
        self.clip_image_encoder = clip_image_encoder
        self.image_proj = image_proj
        self.ip_adapters = ip_adapters
        self.ip_scale = ip_scale

        # Freeze CLIP
        for p in self.clip_image_encoder.parameters():
            p.requires_grad = False
        self.clip_image_encoder.eval()

    @property
    def num_ip_tokens(self) -> int:
        return self.image_proj.clip_extra_context_tokens

    def get_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """CLIP encode + project → IP tokens (B, N_ip, cross_attention_dim)."""
        with torch.no_grad():
            clip_outputs = self.clip_image_encoder(pixel_values, output_hidden_states=True)
        # Use pooler output (already projected to 1024)
        clip_embeds = clip_outputs.image_embeds  # (B, 1024)
        return self.image_proj(clip_embeds)

    def set_scale(self, scale: float):
        self.ip_scale = scale
        for attn in self.ip_adapters.values():
            attn.ip_scale = scale

    def state_dict(self):
        """Collect trainable weights: ip_k_proj, ip_v_proj, image_proj."""
        sd = {}
        for name, attn in self.ip_adapters.items():
            for k, v in attn.trainable_state_dict().items():
                sd[f"{name}.{k}"] = v
        for k, v in self.image_proj.state_dict().items():
            sd[f"image_proj.{k}"] = v
        return sd

    def load_state_dict(self, sd: dict):
        for name, attn in self.ip_adapters.items():
            prefix = f"{name}."
            layer_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if layer_sd:
                attn.load_trainable_state_dict(layer_sd)
        proj_sd = {k[len("image_proj."):]: v for k, v in sd.items() if k.startswith("image_proj.")}
        if proj_sd:
            self.image_proj.load_state_dict(proj_sd, strict=False)
