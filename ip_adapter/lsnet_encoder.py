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
    """Frozen LSNet backbone returning the raw 768-dim artist-style feature.

    The 768→1024 projection lives in the trainer (``lsnet_proj``) so it is
    trainable and saved alongside the other IP-Adapter projections.

    Parameters
    ----------
    ckpt_path : str
        Path to ``best_checkpoint.pth``.
    feature_dim : int
        Backbone output dimension (768 for XL-448).
    """

    def __init__(
        self,
        ckpt_path: str,
        feature_dim: int = LSNET_FEATURE_DIM,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        self.backbone = _build_lsnet_backbone(num_classes=0)
        self._load_ckpt(ckpt_path)

        self.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _load_ckpt(self, ckpt_path: str) -> None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"LSNet checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt

        our_state = self.backbone.state_dict()
        missing = [k for k in our_state if k not in state_dict]
        if missing:
            raise RuntimeError(
                f"LSNet checkpoint is missing {len(missing)}/{len(our_state)} backbone "
                f"keys (architecture mismatch). First few: {missing[:5]}"
            )
        self.backbone.load_state_dict({k: state_dict[k] for k in our_state}, strict=True)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Attention.ab is an ad-hoc tensor (not a Parameter) pinned to
        # whatever device `eval()` ran on — delete it so the next forward
        # re-materialises from attention_biases on the correct device.
        for m in self.backbone.modules():
            if hasattr(m, "ab"):
                del m.ab
        return self

    def forward(self, x: torch.Tensor, return_patches: bool = False):
        """
        Args:
            x: (B, 3, 448, 448) float images in [0, 1]

        Returns:
            (B, 768) raw artist-style backbone feature, or (feat, patches) if
            return_patches=True (patches: B, 196, 768).
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            x = self.backbone.patch_embed(x)
            x = self.backbone.blocks1(x)
            x = self.backbone.blocks2(x)
            x = self.backbone.blocks3(x)
            x = self.backbone.blocks4(x)                    # (B, 768, 14, 14)
            feat = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, 768)
        if return_patches:
            patches = x.flatten(2).transpose(1, 2).contiguous()  # (B, 196, 768)
            return feat, patches
        return feat


def load_lsnet_encoder(
    ckpt_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LSNetStyleEncoder:
    """Load the frozen LSNet style backbone (raw 768-dim feature).

    Args:
        ckpt_path: Path to ``best_checkpoint.pth``.
        device: Device to place the model on.
        dtype: Data type.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"LSNet checkpoint not found: {ckpt_path}")

    encoder = LSNetStyleEncoder(ckpt_path=ckpt_path)
    encoder.to(device=device, dtype=dtype)
    encoder.eval()
    return encoder
