"""
CCIP CaFormer Character Identity Encoder (Pure PyTorch, GPU-native)

Faithful re-implementation of the **CAFormer-b36** backbone used by the
``ccip-caformer_b36-24.ckpt`` checkpoint (dghs-imgutils CCIP).

Architecture (CAFormer-b36, 384x384 input, channels-last B,H,W,C):
  downsample_layers.0: Conv2d(3->128, k=7, s=4, p=2) + post_norm  ->  96x96
  stages.0: 3x  MetaFormerBlock(dim=128,  token_mixer=SepConv)
  downsample_layers.1: pre_norm + Conv2d(128->256, k=3, s=2)      ->  48x48
  stages.1: 12x MetaFormerBlock(dim=256,  token_mixer=SepConv)
  downsample_layers.2: pre_norm + Conv2d(256->512, k=3, s=2)      ->  24x24
  stages.2: 18x MetaFormerBlock(dim=512,  token_mixer=Attention, res_scale)
  downsample_layers.3: pre_norm + Conv2d(512->768, k=3, s=2)      ->  12x12
  stages.3: 3x  MetaFormerBlock(dim=768,  token_mixer=Attention, res_scale)
  norm: LayerNorm(768)
  feature = norm(global_avg_pool(stages output))  -> (B, 768)

Token mixers (CAFormer = [Conv, Conv, Attn, Attn]):
  SepConv   : pwconv1(Linear) -> StarReLU -> dwconv(7x7, depthwise) -> pwconv2(Linear)
  Attention : qkv(Linear, bias=False) -> SDPA(head_dim=32) -> proj(Linear, bias=False)

The 768-dim pooled+normed feature is the semantic "character identity"
representation before the 2-way similarity head (head.* keys are not loaded).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTN_HEAD_DIM = 32  # CAFormer Attention default head_dim
DEFAULT_CKPT = "/root/lanyun-tmp/workspace/ccip/ccip-caformer_b36-24.ckpt"


# ── primitives ────────────────────────────────────────────────────


class ScaleNorm(nn.Module):
    """LayerNormGeneral over the last (channel) dim, scale-only (bias=False)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x - x.mean(-1, keepdim=True)
        s = c.pow(2).mean(-1, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        return x * self.weight


class StarReLU(nn.Module):
    """scale * relu(x)^2 + bias  (learnable scalar scale/bias)."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.relu(x) ** 2 + self.bias


class Scale(nn.Module):
    """Per-channel learnable residual scale."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


# ── token mixers ──────────────────────────────────────────────────


class SepConv(nn.Module):
    """Inverted separable conv token mixer (ConvFormer). Channels-last I/O."""

    def __init__(self, dim: int, expansion_ratio: float = 2.0, kernel_size: int = 7):
        super().__init__()
        med = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med, bias=False)
        self.act1 = StarReLU()
        self.dwconv = nn.Conv2d(
            med, med, kernel_size=kernel_size, padding=kernel_size // 2,
            groups=med, bias=False,
        )
        self.pwconv2 = nn.Linear(med, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W, C)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(x)
        return x


class Attention(nn.Module):
    """Vanilla MetaFormer self-attention (SDPA). Channels-last I/O."""

    def __init__(self, dim: int, head_dim: int = ATTN_HEAD_DIM):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        attn_dim = self.num_heads * head_dim
        self.qkv = nn.Linear(dim, attn_dim * 3, bias=False)
        self.proj = nn.Linear(attn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W, C)
        B, H, W, C = x.shape
        N = H * W
        x = x.reshape(B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        return out.reshape(B, H, W, C)


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.act = StarReLU()
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ── block / downsample ────────────────────────────────────────────


class MetaFormerBlock(nn.Module):
    def __init__(self, dim: int, token_mixer: nn.Module, use_res_scale: bool):
        super().__init__()
        self.norm1 = ScaleNorm(dim)
        self.token_mixer = token_mixer
        self.res_scale1 = Scale(dim) if use_res_scale else nn.Identity()
        self.norm2 = ScaleNorm(dim)
        self.mlp = Mlp(dim)
        self.res_scale2 = Scale(dim) if use_res_scale else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_scale1(x) + self.token_mixer(self.norm1(x))
        x = self.res_scale2(x) + self.mlp(self.norm2(x))
        return x


class Downsampling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                 padding: int, pre_norm: bool, post_norm: bool, pre_permute: bool):
        super().__init__()
        self.pre_norm = ScaleNorm(in_ch) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = ScaleNorm(out_ch) if post_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        if self.pre_permute:
            x = x.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
        x = self.post_norm(x)
        return x


# ── backbone ──────────────────────────────────────────────────────


class CaFormerBackbone(nn.Module):
    """CAFormer-b36 backbone returning the 768-dim pooled+normed feature."""

    DEPTHS = (3, 12, 18, 3)
    DIMS = (128, 256, 512, 768)
    # CAFormer: SepConv in stage0/1, Attention in stage2/3; res_scale in stage2/3
    TOKEN_MIXERS = ("conv", "conv", "attn", "attn")

    def __init__(self):
        super().__init__()
        dims = self.DIMS

        self.downsample_layers = nn.ModuleList([
            Downsampling(3, dims[0], kernel_size=7, stride=4, padding=2,
                         pre_norm=False, post_norm=True, pre_permute=False),
            Downsampling(dims[0], dims[1], kernel_size=3, stride=2, padding=1,
                         pre_norm=True, post_norm=False, pre_permute=True),
            Downsampling(dims[1], dims[2], kernel_size=3, stride=2, padding=1,
                         pre_norm=True, post_norm=False, pre_permute=True),
            Downsampling(dims[2], dims[3], kernel_size=3, stride=2, padding=1,
                         pre_norm=True, post_norm=False, pre_permute=True),
        ])

        self.stages = nn.ModuleList()
        for i in range(4):
            use_res_scale = self.TOKEN_MIXERS[i] == "attn"
            blocks = []
            for _ in range(self.DEPTHS[i]):
                if self.TOKEN_MIXERS[i] == "conv":
                    mixer: nn.Module = SepConv(dims[i])
                else:
                    mixer = Attention(dims[i])
                blocks.append(MetaFormerBlock(dims[i], mixer, use_res_scale))
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

    def forward(self, x: torch.Tensor, return_patches: bool = False):
        # x: (B, 3, H, W) channels-first input
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # x: (B, 768, 12, 12) — pre-pool feature map
        feat = x.mean(dim=(1, 2))  # global average pool → (B, 768)
        feat = self.norm(feat)
        if return_patches:
            patches = x.flatten(2).transpose(1, 2).contiguous()  # (B, 144, 768)
            return feat, patches
        return feat


class CCIPIdentityEncoder(nn.Module):
    """Frozen CCIP backbone returning the raw 768-dim identity feature.

    The 768→1024 projection lives in the trainer (``ccip_proj``) so it is
    trainable and saved alongside the other IP-Adapter projections.
    """

    feature_dim = 768

    def __init__(self, ckpt_path: str = DEFAULT_CKPT):
        super().__init__()
        self.backbone = CaFormerBackbone()
        self._load_ckpt(ckpt_path)
        self.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _load_ckpt(self, ckpt_path: str) -> None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"CCIP checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        prefix = "module._orig_mod.feature.backbone.caformer."
        remapped = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}

        our_state = self.backbone.state_dict()
        missing = [k for k in our_state if k not in remapped]
        if missing:
            raise RuntimeError(
                f"CCIP checkpoint is missing {len(missing)}/{len(our_state)} backbone "
                f"keys (architecture mismatch). First few: {missing[:5]}"
            )
        self.backbone.load_state_dict({k: remapped[k] for k in our_state}, strict=True)

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """CLIP normalization. Expects input in [0, 1]."""
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, x: torch.Tensor, return_patches: bool = False):
        """(B, 3, 384, 384) in [0,1] → (B, 768) raw identity feature.

        If ``return_patches=True``, also returns pre-pool spatial features
        (B, 144, 768) for Resampler projection.
        """
        x = self.normalize(x)
        with torch.no_grad():
            result = self.backbone(x, return_patches=return_patches)
        if return_patches:
            feat, patches = result
            return feat, patches
        return result


def load_ccip_encoder(ckpt_path: str = DEFAULT_CKPT, device: str = "cpu",
                      dtype: torch.dtype = torch.float32) -> CCIPIdentityEncoder:
    encoder = CCIPIdentityEncoder(ckpt_path=ckpt_path)
    encoder.to(device=device, dtype=dtype)
    encoder.eval()
    return encoder
