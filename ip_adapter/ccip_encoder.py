"""
CCIP CaFormer Character Identity Encoder (Pure PyTorch, GPU-native)

Directly loads ``ccip-caformer_b36-24.ckpt`` and runs CaFormer
backbone + attention pool entirely on CUDA with batched inference.

Architecture (ccip-caformer_b36-24, 384x384 input):
  ds0: Conv2d(3->128, k=7, s=4)  ->  96x96
  stage0: 3x Block(dim=128, n_heads=2)  -> ResScale + MHSA + MLP
  ds1: Conv2d(128->256, k=3, s=2) ->  48x48
  stage1: 12x Block(dim=256, n_heads=4)
  ds2: Conv2d(256->512, k=3, s=2) ->  24x24
  stage2: 18x Block(dim=512, n_heads=8)
  ds3: Conv2d(512->768, k=3, s=2) ->  12x12
  stage3: 3x Block(dim=768, n_heads=12)
  attnpool: AttentionPool2d(spatial=12x12, dim=768, heads=12) -> (B, 768)

Each Block: x + res_scale1 * MHSA(norm1(x)) + res_scale2 * MLP(norm2(x))

Used layer: attnpool output (768-dim CLS token from attention pooling
over all spatial patches). This is the semantic "character identity"
feature before any classification head.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_DIM = 64
DEFAULT_CKPT = "/root/lanyun-tmp/workspace/ccip/ccip-caformer_b36-24.ckpt"


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, eps=1e-6)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = HEAD_DIM
        self.scale = HEAD_DIM ** -0.5
        inner_dim = n_heads * HEAD_DIM
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.proj = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class CaBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.token_mixer = Attention(dim, n_heads)
        self.res_scale1 = nn.Parameter(torch.ones(dim))
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.res_scale2 = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.res_scale1 * self.token_mixer(self.norm1(x))
        x = x + self.res_scale2 * self.mlp(self.norm2(x))
        return x


class AttentionPool2d(nn.Module):
    def __init__(self, spatial_dim: int, embed_dim: int, n_heads: int):
        super().__init__()
        n_patches = spatial_dim * spatial_dim
        self.positional_embedding = nn.Parameter(
            torch.randn(n_patches + 1, embed_dim) / embed_dim ** 0.5
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding
        q = self.q_proj(cls)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(B, 1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        return self.c_proj(out).squeeze(1)


class CaFormerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.ds0 = nn.Conv2d(3, 128, kernel_size=7, stride=4, padding=3, bias=True)
        self.ds0_norm = LayerNorm(128)
        self.stage0 = nn.ModuleList([CaBlock(128, max(128 // HEAD_DIM, 1)) for _ in range(3)])
        self.ds1_norm = LayerNorm(128)
        self.ds1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True)
        self.stage1 = nn.ModuleList([CaBlock(256, max(256 // HEAD_DIM, 1)) for _ in range(12)])
        self.ds2_norm = LayerNorm(256)
        self.ds2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True)
        self.stage2 = nn.ModuleList([CaBlock(512, max(512 // HEAD_DIM, 1)) for _ in range(18)])
        self.ds3_norm = LayerNorm(512)
        self.ds3 = nn.Conv2d(512, 768, kernel_size=3, stride=2, padding=1, bias=True)
        self.stage3 = nn.ModuleList([CaBlock(768, max(768 // HEAD_DIM, 1)) for _ in range(3)])
        self.attnpool = AttentionPool2d(12, 768, max(768 // HEAD_DIM, 1))

    def _run_stage(self, x, blocks, ds=None, ds_norm=None, prev_perm=None):
        if ds is not None and ds_norm is not None:
            x_nchw = prev_perm.permute(0, 3, 1, 2) if prev_perm is not None else x
            x_nchw = ds(ds_norm(x_nchw.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            x = x_nchw.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        for blk in blocks:
            x = blk(x)
        x = x.reshape(B, H, W, C)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ds0 + stage0
        x = self.ds0(x)
        x_nhwc = x.permute(0, 2, 3, 1)
        x_nhwc = self.ds0_norm(x_nhwc)
        B, H, W, C = x_nhwc.shape
        x_nhwc = x_nhwc.reshape(B, H * W, C)
        for blk in self.stage0:
            x_nhwc = blk(x_nhwc)
        x0 = x_nhwc.reshape(B, H, W, C)

        # ds1 + stage1
        x1 = self._run_stage(None, self.stage1, self.ds1, self.ds1_norm, x0)
        # ds2 + stage2
        x2 = self._run_stage(None, self.stage2, self.ds2, self.ds2_norm, x1)
        # ds3 + stage3
        x3 = self._run_stage(None, self.stage3, self.ds3, self.ds3_norm, x2)

        x_out = x3.permute(0, 3, 1, 2)
        x_out = self.attnpool(x_out)
        return x_out


class CCIPIdentityEncoder(nn.Module):
    def __init__(self, ckpt_path: str = DEFAULT_CKPT, output_dim: int = 1024, freeze_proj: bool = False):
        super().__init__()
        self.backbone = CaFormerBackbone()
        self._load_ckpt(ckpt_path)
        self.proj = nn.Sequential(nn.Linear(768, output_dim), nn.LayerNorm(output_dim))
        if freeze_proj:
            for p in self.proj.parameters():
                p.requires_grad = False
        self.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _load_ckpt(self, ckpt_path: str) -> None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"CCIP checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        prefix = "module._orig_mod.feature.backbone.caformer."
        remapped = {}
        for key, value in ckpt.items():
            if key.startswith(prefix):
                remapped[key[len(prefix):]] = value
        our_state = self.backbone.state_dict()
        loaded = {k: remapped[k] for k in our_state if k in remapped}
        self.backbone.load_state_dict(loaded, strict=False)

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        with torch.no_grad():
            feat = self.backbone(x)
        feat = self.proj(feat)
        return feat


def load_ccip_encoder(ckpt_path: str = DEFAULT_CKPT, device: str = "cpu",
                      dtype: torch.dtype = torch.float32, freeze_proj: bool = False) -> CCIPIdentityEncoder:
    encoder = CCIPIdentityEncoder(ckpt_path=ckpt_path, freeze_proj=freeze_proj)
    encoder.to(device=device, dtype=dtype)
    encoder.eval()
    return encoder
