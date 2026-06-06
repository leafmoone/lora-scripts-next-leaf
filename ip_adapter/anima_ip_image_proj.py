"""
Anima IP-Adapter Image Projection Model

Converts CLIP / SigLIP / DINOv2 image embeddings into cross-attention
compatible IP tokens.

Two modes:
  ``ImageProjModel``  — coarse: 1 global CLS embedding → N ip tokens
  ``Resampler``       — fine:   N learnable queries cross-attend over patch features
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageProjModel(nn.Module):
    """Project a single global image embedding into multiple IP tokens."""

    def __init__(
        self,
        cross_attention_dim: int = 1024,
        clip_embeddings_dim: int = 1024,
        clip_extra_context_tokens: int = 4,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens

        self.proj = nn.Linear(
            clip_embeddings_dim,
            cross_attention_dim * clip_extra_context_tokens,
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """(B, clip_dim) → (B, num_tokens, cross_attention_dim)"""
        x = self.proj(image_embeds)
        x = x.reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class PerceiverAttention(nn.Module):
    """Cross-attention: latent queries attend to key-value pairs."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        latents = self.norm(latents)
        q = self.to_q(x)
        kv = self.to_kv(latents)
        k, v = kv.chunk(2, dim=-1)
        q = q.reshape(q.shape[0], q.shape[1], self.heads, -1).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self.heads, -1).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.heads, -1).transpose(1, 2)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


class Resampler(nn.Module):
    """Perceiver Resampler for fine-grained IP-Adapter control."""

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 16,
        ff_mult: int = 4,
        output_dim: int = 1024,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.output_dim = output_dim

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim, dim_head=dim_head, heads=heads),
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, dim * ff_mult),
                        nn.GELU(),
                        nn.Linear(dim * ff_mult, dim),
                    ),
                ])
            )

        self.proj_out = nn.Linear(dim, output_dim) if output_dim != dim else nn.Identity()
        self.norm_out = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, embed_dim) patch features → (B, num_queries, output_dim) IP tokens."""
        latents = self.latents.expand(x.shape[0], -1, -1)
        for attn, ff in self.layers:
            latents = attn(latents, x) + latents
            latents = ff(latents) + latents
        return self.norm_out(self.proj_out(latents))
