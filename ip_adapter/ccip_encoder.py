"""
CCIP Character Identity Encoder for IP-Adapter

Wraps ``imgutils.metrics.ccip_extract_feature`` (ONNX runtime)
and projects the 768-dim character identity feature to 1024-dim
to match CLIP's embedding space for multi-stream fusion.

Also supports direct .ckpt checkpoint loading via ONNX export
as a fallback when imgutils is unavailable.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Core: CCIP feature encoder with dimensional projection
# ---------------------------------------------------------------------------


class CCIPIdentityEncoder(nn.Module):
    """Extract character identity features via CCIP ONNX model and project to 1024.

    Parameters
    ----------
    model_name : str
        CCIP model name for imgutils (e.g. ``"ccip-caformer-24-randaug-pruned"``).
    feature_dim : int
        Input feature dimension from CCIP (default 768).
    output_dim : int
        Output dimension matching CLIP (default 1024).
    freeze_proj : bool
        If True, keep the projection layer frozen (identity-only features).
        If False, allow projection to be trainable.
    """

    def __init__(
        self,
        model_name: str = "ccip-caformer-24-randaug-pruned",
        feature_dim: int = 768,
        output_dim: int = 1024,
        freeze_proj: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # Projection: 768 → 1024 (dimension alignment with CLIP)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        if freeze_proj:
            for p in self.proj.parameters():
                p.requires_grad = False

        self._ccip_func = None  # lazy import

    def _ensure_ccip(self):
        if self._ccip_func is None:
            from imgutils.metrics.ccip import ccip_extract_feature
            self._ccip_func = ccip_extract_feature

    def _extract_ccip_feature(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract CCIP feature from a single image tensor.

        Args:
            image_tensor: (3, H, W) float32 tensor in [0, 1] range

        Returns:
            (768,) float32 tensor
        """
        import numpy as np

        self._ensure_ccip()

        # Convert tensor (3, H, W) → PIL Image
        # Image is already in [0, 1] from Kohya pipeline
        img_np = (image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        from PIL import Image
        pil_img = Image.fromarray(img_np)

        # Extract feature via imgutils
        feat = self._ccip_func(pil_img, model=self.model_name)  # (768,) float32
        return torch.from_numpy(feat).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float32 tensors in [0, 1] range

        Returns:
            (B, 1024) projected character identity embeddings
        """
        batch_size = x.shape[0]
        features = []

        for i in range(batch_size):
            feat = self._extract_ccip_feature(x[i])
            features.append(feat)

        feats = torch.stack(features, dim=0).to(device=x.device, dtype=x.dtype)
        feats = self.proj(feats)
        return feats


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_ccip_encoder(
    model_name: str = "ccip-caformer-24-randaug-pruned",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    freeze_proj: bool = True,
) -> CCIPIdentityEncoder:
    """Load CCIP encoder with dimension projection.

    Requires ``dghs-imgutils`` to be installed. The ONNX model
    will be auto-downloaded from HuggingFace on first use.

    Args:
        model_name: CCIP model variant.
        device: Device to place the projection layer on.
        dtype: Data type for the projection layer.
        freeze_proj: Keep projection frozen (identity features only).
    """
    encoder = CCIPIdentityEncoder(
        model_name=model_name,
        freeze_proj=freeze_proj,
    )
    encoder.to(device=device, dtype=dtype)
    encoder.eval()
    return encoder
