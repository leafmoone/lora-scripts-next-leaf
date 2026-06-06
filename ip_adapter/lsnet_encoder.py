"""
LSNet Artist Style Encoder for IP-Adapter

Builds the LSNet-XL-448 backbone from the source model definition,
loads the ``best_checkpoint.pth`` checkpoint, strips the classification
head, and projects the 768-dim style backbone features to 1024-dim
to match CLIP's embedding space for multi-stream fusion.

Checkpoint reference:
  ``lsnet_xl_artist_448``: embed_dim=[192,384,576,768], depth=[8,12,16,20],
  num_heads=[6,6,6,6], img_size=448, num_classes=39261
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


LSNET_FEATURE_DIM = 768  # embed_dim[-1] of lsnet_xl_artist_448


def _build_lsnet_backbone(num_classes: int = 0) -> nn.Module:
    """Build LSNet-XL-448 backbone without classification head."""
    _vendor_dir = Path(__file__).resolve().parent
    if str(_vendor_dir) not in sys.path:
        sys.path.insert(0, str(_vendor_dir))

    from _lsnet_model import LSNet

    backbone = LSNet(
        img_size=448,
        patch_size=8,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=[192, 384, 576, 768],
        key_dim=[16, 16, 16, 16],
        depth=[8, 12, 16, 20],
        num_heads=[6, 6, 6, 6],
        distillation=False,
    )
    return backbone


class LSNetStyleEncoder(nn.Module):
    """Extract artist style features via LSNet backbone and project to 1024.

    Parameters
    ----------
    ckpt_path : str
        Path to ``best_checkpoint.pth``.
    feature_dim : int
        Backbone output dimension (768 for XL-448).
    output_dim : int
        Output dimension matching CLIP (default 1024).
    freeze_proj : bool
        If True, keep the projection layer frozen.
    """

    def __init__(
        self,
        ckpt_path: str,
        feature_dim: int = LSNET_FEATURE_DIM,
        output_dim: int = 1024,
        freeze_proj: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self.backbone = _build_lsnet_backbone(num_classes=0)
        self._load_ckpt(ckpt_path)

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        if freeze_proj:
            for p in self.proj.parameters():
                p.requires_grad = False

        self.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _load_ckpt(self, ckpt_path: str) -> None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"LSNet checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"]

        our_state = self.backbone.state_dict()
        loaded = {}
        for our_key in our_state:
            if our_key in state_dict:
                loaded[our_key] = state_dict[our_key]

        self.backbone.load_state_dict(loaded, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 448, 448) float32 images in [0, 1]

        Returns:
            (B, 1024) projected artist style embeddings
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            x = self.backbone.patch_embed(x)
            x = self.backbone.blocks1(x)
            x = self.backbone.blocks2(x)
            x = self.backbone.blocks3(x)
            x = self.backbone.blocks4(x)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        x = self.proj(x)
        return x


def load_lsnet_encoder(
    ckpt_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    freeze_proj: bool = True,
) -> LSNetStyleEncoder:
    """Load LSNet style encoder with dimension projection.

    Args:
        ckpt_path: Path to ``best_checkpoint.pth``.
        device: Device to place the model on.
        dtype: Data type.
        freeze_proj: Keep projection frozen (style features only).
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"LSNet checkpoint not found: {ckpt_path}")

    encoder = LSNetStyleEncoder(
        ckpt_path=ckpt_path,
        freeze_proj=freeze_proj,
    )
    encoder.to(device=device, dtype=dtype)
    encoder.eval()
    return encoder
