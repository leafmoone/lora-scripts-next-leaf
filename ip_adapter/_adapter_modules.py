"""
Reusable building blocks for visual condition adapters — transplanted from
anima-edit/library/anima_models.py.

Modules:
  LLMAdapterRMSNorm        — T5-style RMS norm (no mean subtraction)
  LLMAdapterAttention      — QK-norm attention with separate RoPE
  OmniFeedForward          — SwiGLU FFN (SiLU-gated)
  OmniRefinerBlock         — Self-attention + SwiGLU refiner
  LLMAdapterTransformerBlock — Cross-attention + MLP (Perceiver Resampler core)
  AdapterRotaryEmbedding   — 1D RoPE
  AdapterImageRotaryEmbedding — 2D RoPE for known grid sizes
  RMSNormNoAffine          — RMS norm without affine params
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Rotary embeddings ─────────────────────────────────────────


def _adapter_rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _adapter_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_adapter_rotate_half(x) * sin)


class AdapterRotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class AdapterImageRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float = 256.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        half_dim = head_dim // 2
        if half_dim % 2 != 0:
            raise ValueError(f"AdapterImageRotaryEmbedding needs head_dim/2 even, got {head_dim}")
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int], num_images: int = 1):
        height, width = grid_hw
        device = x.device
        rows, cols = torch.arange(height, device=device), torch.arange(width, device=device)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        pos_h, pos_w = yy.reshape(-1).repeat(num_images), xx.reshape(-1).repeat(num_images)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs_h, freqs_w = torch.outer(pos_h.float(), self.inv_freq.float()), torch.outer(pos_w.float(), self.inv_freq.float())
            emb_h = torch.cat((freqs_h, freqs_h), dim=-1)
            emb_w = torch.cat((freqs_w, freqs_w), dim=-1)
            emb = torch.cat((emb_h, emb_w), dim=-1)
            cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ── Normalisation ──────────────────────────────────────────────


class LLMAdapterRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


class RMSNormNoAffine(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def reset_parameters(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


# ── Attention + feed-forward blocks ────────────────────────────


class LLMAdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim, norm_eps=1e-6):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = LLMAdapterRMSNorm(self.head_dim, eps=norm_eps)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = LLMAdapterRMSNorm(self.head_dim, eps=norm_eps)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        v = self.v_proj(context).view(kv_shape).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = _adapter_apply_rotary_pos_emb(q, cos, sin)
            cos_c, sin_c = position_embeddings_context
            k = _adapter_apply_rotary_pos_emb(k, cos_c, sin_c)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(out)


class OmniFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class OmniRefinerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 8.0 / 3.0, norm_eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"OmniRefinerBlock dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.attention_norm1 = LLMAdapterRMSNorm(dim, eps=norm_eps)
        self.attention = LLMAdapterAttention(query_dim=dim, context_dim=dim, n_heads=num_heads, head_dim=dim // num_heads, norm_eps=norm_eps)
        self.attention_norm2 = LLMAdapterRMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = LLMAdapterRMSNorm(dim, eps=norm_eps)
        self.feed_forward = OmniFeedForward(dim=dim, hidden_dim=int(dim * mlp_ratio))
        self.ffn_norm2 = LLMAdapterRMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, position_embeddings=None, attention_mask=None) -> torch.Tensor:
        attn_out = self.attention(self.attention_norm1(x), mask=attention_mask,
                                   position_embeddings=position_embeddings,
                                   position_embeddings_context=position_embeddings)
        x = x + self.attention_norm2(attn_out)
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x

    def init_weights(self) -> None:
        dim = self.attention.q_proj.in_features
        hidden_dim = self.feed_forward.w1.out_features
        std = 1.0 / math.sqrt(dim)
        torch.nn.init.trunc_normal_(self.attention.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.attention.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.attention.v_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.feed_forward.w1.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std)
        out_std = 1.0 / math.sqrt(hidden_dim)
        torch.nn.init.trunc_normal_(self.attention.o_proj.weight, std=out_std, a=-3 * out_std, b=3 * out_std)
        torch.nn.init.trunc_normal_(self.feed_forward.w2.weight, std=out_std, a=-3 * out_std, b=3 * out_std)


class LLMAdapterTransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, self_attn=False, layer_norm=False):
        super().__init__()
        self.has_self_attn = self_attn
        if self.has_self_attn:
            self.norm_self_attn = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
            self.self_attn = LLMAdapterAttention(query_dim=model_dim, context_dim=model_dim, n_heads=num_heads, head_dim=model_dim // num_heads)
        self.norm_cross_attn = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        self.cross_attn = LLMAdapterAttention(query_dim=model_dim, context_dim=source_dim, n_heads=num_heads, head_dim=model_dim // num_heads)
        self.norm_mlp = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        self.mlp = nn.Sequential(nn.Linear(model_dim, int(model_dim * mlp_ratio)), nn.GELU(), nn.Linear(int(model_dim * mlp_ratio), model_dim))

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None,
                position_embeddings=None, position_embeddings_context=None):
        if self.has_self_attn:
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(normed, mask=target_attention_mask,
                                       position_embeddings=position_embeddings,
                                       position_embeddings_context=position_embeddings)
            x = x + attn_out
        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(normed, mask=source_attention_mask, context=context,
                                    position_embeddings=position_embeddings,
                                    position_embeddings_context=position_embeddings_context)
        x = x + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)
