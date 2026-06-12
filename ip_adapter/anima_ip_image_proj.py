"""
Anima IP-Adapter Image Projection Model

Converts CLIP / SigLIP / DINOv2 image embeddings into cross-attention
compatible IP tokens.

Two modes:
  ``ImageProjModel``  — coarse: 1 global CLS embedding → N ip tokens
  ``Resampler``       — fine:   N learnable queries cross-attend over patch features
"""

from __future__ import annotations

import math

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
    """Cross-attention: learnable latents (queries) attend to patch features (key/value).

    Standard Perceiver Resampler pattern:
      q = to_q(latents)    ← learnable queries read from image patches
      kv = to_kv(patches)  ← patch features provide key/value
    """

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.norm_latents = nn.LayerNorm(dim)
        self.norm_patches = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, latents: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        latents = self.norm_latents(latents)
        patches = self.norm_patches(patches)
        q = self.to_q(latents)                     # (B, N_q, inner_dim)
        kv = self.to_kv(patches)                   # (B, N_p, inner_dim * 2)
        k, v = kv.chunk(2, dim=-1)
        q = q.reshape(q.shape[0], q.shape[1], self.heads, -1).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self.heads, -1).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.heads, -1).transpose(1, 2)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(latents.shape[0], latents.shape[1], -1)
        return self.to_out(out)


class Resampler(nn.Module):
    """Perceiver Resampler using LLMAdapterTransformerBlock (upgraded from anima-edit).

    Uses 2-layer cross-attention + MLP with learnable queries and RoPE.
    More stable than the original hand-written PerceiverAttention.
    """

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

        from ._adapter_modules import LLMAdapterTransformerBlock, AdapterRotaryEmbedding, LLMAdapterRMSNorm

        actual_depth = 2  # LLMAdapterTransformerBlock is heavier, 2 layers equivalent to 4 PerceiverAttention
        self.feature_norm = LLMAdapterRMSNorm(dim)
        self.source_proj = nn.Linear(dim, dim)
        self.latents = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.rotary = AdapterRotaryEmbedding(dim // heads)

        self.layers = nn.ModuleList([
            LLMAdapterTransformerBlock(source_dim=dim, model_dim=dim, num_heads=heads, self_attn=True)
            for _ in range(actual_depth)
        ])
        self.out_norm = LLMAdapterRMSNorm(dim)
        self.proj_out = nn.Linear(dim, output_dim) if output_dim != dim else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(1024)
        torch.nn.init.trunc_normal_(self.source_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.source_proj.bias)
        torch.nn.init.trunc_normal_(self.latents, std=std, a=-3 * std, b=3 * std)
        for layer in self.layers:
            layer.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, embed_dim) patch features → (B, num_queries, output_dim) IP tokens."""
        B = x.shape[0]
        source = self.source_proj(self.feature_norm(x))
        tokens = self.latents.unsqueeze(0).expand(B, -1, -1).to(dtype=source.dtype, device=source.device)
        pos_q = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        pos_s = torch.arange(source.shape[1], device=source.device).unsqueeze(0)
        pe_q = self.rotary(tokens, pos_q)
        pe_s = self.rotary(source, pos_s)
        for layer in self.layers:
            tokens = layer(tokens, source, position_embeddings=pe_q, position_embeddings_context=pe_s)
        return self.proj_out(self.out_norm(tokens))


# ---------------------------------------------------------------------------
# Multi-Stream Projection for auxiliary encoder fusion
# ---------------------------------------------------------------------------

class MultiStreamProj(nn.Module):
    """Manages independent ImageProjModel instances for each enabled encoder.

    Encoder order convention:
      index 0 = CLIP  (always present)
      index 1 = CCIP  (character identity, optional)
      index 2 = LSNet (artist style, optional)

    Each encoder's global embedding (B, 1024) is projected to
    (B, num_tokens, 1024) independently.
    """

    def __init__(
        self,
        num_streams: int = 3,
        cross_attention_dim: int = 1024,
        embed_dim: int = 1024,
        tokens_per_stream: list[int] | None = None,
    ):
        super().__init__()
        self.num_streams = num_streams

        if tokens_per_stream is None:
            tokens_per_stream = [4] * num_streams

        assert len(tokens_per_stream) == num_streams

        self.projs = nn.ModuleList([
            ImageProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=embed_dim,
                clip_extra_context_tokens=tokens_per_stream[i],
            )
            for i in range(num_streams)
        ])

    @classmethod
    def from_modules(cls, modules: list[nn.Module]) -> "MultiStreamProj":
        """Construct from a list of pre-built modules (e.g. Resamplers)."""
        inst = cls.__new__(cls)
        nn.Module.__init__(inst)
        inst.num_streams = len(modules)
        inst.projs = nn.ModuleList(modules)
        return inst

    def forward(self, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        """Project each encoder's embedding to IP tokens.

        Args:
            embeddings: list of either (B, 1024) global embeds or
                       (B, L, 1024) patch features, one per enabled stream.

        Returns:
            list of (B, N_i, 1024) IP token tensors, one per stream.
        """
        assert len(embeddings) <= self.num_streams
        return [self.projs[i](embeddings[i]) for i in range(len(embeddings))]

# ---------------------------------------------------------------------------
# MLP Image Projection (FaceID-style, from anima-edit)
# ---------------------------------------------------------------------------

class MLPImageProjModel(nn.Module):
    """Non-linear 2-layer MLP projection — more expressive than plain Linear.

    pool(features) → Linear(768→1536) → GELU → Linear(1536→1024×N) → LayerNorm
    """

    def __init__(self, feature_dim: int = 768, cross_attention_dim: int = 1024, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """(B, feat_dim) → mean_pool → MLP → (B, N, cross_attn_dim)"""
        x = self.proj(image_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Upgraded Resampler (LLMAdapterTransformerBlock, from anima-edit)
# ---------------------------------------------------------------------------

class LLMResampler(nn.Module):
    """Perceiver Resampler using LLMAdapterTransformerBlock.

    Uses 2-layer cross-attention + MLP with learnable queries and RoPE.
    More stable and industrial-grade compared to hand-written PerceiverAttention.
    """

    def __init__(self, dim: int = 1024, depth: int = 2, num_heads: int = 16, num_queries: int = 16):
        super().__init__()
        self.num_queries = num_queries
        from ._adapter_modules import LLMAdapterTransformerBlock, AdapterRotaryEmbedding, LLMAdapterRMSNorm

        self.feature_norm = LLMAdapterRMSNorm(dim)
        self.source_proj = nn.Linear(dim, dim)
        self.latents = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.rotary = AdapterRotaryEmbedding(dim // num_heads)

        self.layers = nn.ModuleList([
            LLMAdapterTransformerBlock(source_dim=dim, model_dim=dim, num_heads=num_heads, self_attn=True)
            for _ in range(depth)
        ])
        self.out_norm = LLMAdapterRMSNorm(dim)

        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(1024)
        torch.nn.init.trunc_normal_(self.source_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.source_proj.bias)
        torch.nn.init.trunc_normal_(self.latents, std=std, a=-3 * std, b=3 * std)
        for layer in self.layers:
            layer.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, dim) patch features → (B, num_queries, dim) IP tokens."""
        import math as _m
        B = x.shape[0]
        source = self.source_proj(self.feature_norm(x))
        tokens = self.latents.unsqueeze(0).expand(B, -1, -1).to(dtype=source.dtype, device=source.device)
        pos_q = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        pos_s = torch.arange(source.shape[1], device=source.device).unsqueeze(0)
        pe_q = self.rotary(tokens, pos_q)
        pe_s = self.rotary(source, pos_s)
        for layer in self.layers:
            tokens = layer(tokens, source, position_embeddings=pe_q, position_embeddings_context=pe_s)
        return self.out_norm(tokens)


# ---------------------------------------------------------------------------
# Multi-Stream Projection for auxiliary encoder fusion
# ---------------------------------------------------------------------------
