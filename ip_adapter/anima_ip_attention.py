"""
Anima IP-Adapter Cross-Attention Layer

Wraps Anima's native ``Attention`` cross-attention module, adding
trainable ``ip_k_proj`` and ``ip_v_proj`` layers that project
CLIP image features into the DiT's cross-attention space.

Design:
  Original cross-attn (frozen):
    Q = q_proj(latent),  K = k_proj(text),  V = v_proj(text)
  IP cross-attn (trainable):
    K_ip = ip_k_proj(image_tokens),  V_ip = ip_v_proj(image_tokens)

  output = text_out + ip_scale * ip_out
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from library.attention import attention as attention_fn
from library.attention import AttentionParams


class AnimaIPCrossAttention(nn.Module):
    """Drop-in replacement for Anima ``Block.cross_attn``.

    Preserves the frozen original attention for **text** conditioning
    and adds trainable ``ip_k_proj`` / ``ip_v_proj`` for **image** conditioning.
    """

    def __init__(
        self,
        original_attn: nn.Module,  # library.anima_models.Attention
        ip_scale: float = 1.0,
    ):
        super().__init__()
        # ── original cross-attn (frozen) ──────────────────────
        self.orig = original_attn
        for p in self.orig.parameters():
            p.requires_grad = False

        ctx_dim = self.orig._context_dim       # 1024 = crossattn_emb_channels
        inner = self.orig._inner_dim           # n_heads * head_dim

        # ── IP projection layers (trainable) ──────────────────
        self.ip_k_proj = nn.Linear(ctx_dim, inner, bias=False)
        self.ip_v_proj = nn.Linear(ctx_dim, inner, bias=False)
        self.ip_k_norm = nn.Identity()  # QK-norm handled inside orig.forward

        # fine-grained stream (double-stream mode, lazy-init)
        self.ip_k_proj_fine: Optional[nn.Linear] = None
        self.ip_v_proj_fine: Optional[nn.Linear] = None

        # weight init: copy from original k_proj / v_proj
        self.ip_k_proj.weight.data.copy_(self.orig.k_proj.weight.data)
        self.ip_v_proj.weight.data.copy_(self.orig.v_proj.weight.data)

        self.ip_scale = ip_scale

    # ------------------------------------------------------------------
    # Fine-grained stream (lazy-init for double-stream mode)
    # ------------------------------------------------------------------
    def ensure_fine_stream(self):
        """Create fine-grained IP projections on first use."""
        if self.ip_k_proj_fine is not None:
            return
        ctx_dim = self.orig._context_dim
        inner = self.orig._inner_dim
        self.ip_k_proj_fine = nn.Linear(ctx_dim, inner, bias=False)
        self.ip_v_proj_fine = nn.Linear(ctx_dim, inner, bias=False)
        self.ip_k_proj_fine.weight.data.copy_(self.orig.k_proj.weight.data)
        self.ip_v_proj_fine.weight.data.copy_(self.orig.v_proj.weight.data)

    # ------------------------------------------------------------------
    # Training-mode parameter access
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        """Return iterator over trainable params (ip_k_proj + ip_v_proj)."""
        yield from self.ip_k_proj.parameters()
        yield from self.ip_v_proj.parameters()
        if self.ip_k_proj_fine is not None:
            yield from self.ip_k_proj_fine.parameters()
            yield from self.ip_v_proj_fine.parameters()

    def trainable_state_dict(self):
        return {
            "ip_k_proj.weight": self.ip_k_proj.weight,
            "ip_v_proj.weight": self.ip_v_proj.weight,
        }

    def load_trainable_state_dict(self, sd: dict):
        self.ip_k_proj.weight.data.copy_(sd["ip_k_proj.weight"])
        self.ip_v_proj.weight.data.copy_(sd["ip_v_proj.weight"])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                    # (B, L, x_dim)  flattened latent tokens
        attn_params: AttentionParams,
        context: Optional[torch.Tensor] = None,  # (B, N_text, context_dim)
        rope_emb: Optional[torch.Tensor] = None,
        ip_tokens: Optional[torch.Tensor] = None,  # (B, N_ip, context_dim)  global stream
        ip_tokens_fine: Optional[torch.Tensor] = None,  # (B, N_fine, context_dim)  fine-grained stream
    ) -> torch.Tensor:
        # ── 1. Text cross-attention (frozen original) ─────
        text_out = self.orig(x, attn_params, context=context, rope_emb=rope_emb)

        ip_contribution = 0.0
        n_heads = self.orig.n_heads
        head_dim = self.orig.head_dim

        # ── 2. Global IP cross-attention ──────────────────
        if ip_tokens is not None and ip_tokens.numel() > 0:
            q = self.orig.q_proj(x)
            k = self.ip_k_proj(ip_tokens)
            v = self.ip_v_proj(ip_tokens)
            q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=n_heads, d=head_dim), (q, k, v))
            q = self.orig.q_norm(q); k = self.orig.q_norm(k); v = self.orig.v_norm(v)
            ip_global = attention_fn([q, k, v], attn_params=attn_params)
            ip_contribution = ip_contribution + self.orig.output_proj(ip_global)

        # ── 3. Fine-grained IP cross-attention (double) ───
        if ip_tokens_fine is not None and ip_tokens_fine.numel() > 0 and self.ip_k_proj_fine is not None:
            q = self.orig.q_proj(x)
            k = self.ip_k_proj_fine(ip_tokens_fine)
            v = self.ip_v_proj_fine(ip_tokens_fine)
            q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=n_heads, d=head_dim), (q, k, v))
            q = self.orig.q_norm(q); k = self.orig.q_norm(k); v = self.orig.v_norm(v)
            ip_fine = attention_fn([q, k, v], attn_params=attn_params)
            ip_contribution = ip_contribution + self.orig.output_proj(ip_fine)

        if isinstance(ip_contribution, (int, float)):
            return text_out
        return text_out + self.ip_scale * ip_contribution
