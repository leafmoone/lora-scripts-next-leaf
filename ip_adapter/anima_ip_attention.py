"""
Anima IP-Adapter Cross-Attention Layer (Multi-Stream Gated Fusion)

Wraps Anima's native ``Attention`` cross-attention module, adding
trainable ``ip_k_proj`` and ``ip_v_proj`` layers for up to three
encoder streams:

  Stream 0: CLIP   — visual content (always present)
  Stream 1: CCIP   — character identity (optional)
  Stream 2: LSNet  — artist style (optional)

Each stream has its own k/v projections and a learnable scalar
gate.  The gates are initialized as [1.0, 0.1, 0.1] so that
CLIP dominates at the start and auxiliary encoders gradually
activate during training.

Design:
  text_out = frozen original cross-attn(text)
  ip_out = gate_clip * ip_attn(ip_tokens_clip)
         + gate_ccip * ip_attn(ip_tokens_ccip)   [if enabled]
         + gate_lsnet * ip_attn(ip_tokens_lsnet)  [if enabled]
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
    and adds trainable per-stream IP projections for multi-encoder
    image conditioning with gated fusion.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        ip_scale: float = 1.0,
        aux_encoders: tuple[str, ...] = (),
    ):
        super().__init__()

        # ── original cross-attn (frozen) ──────────────────────
        self.orig = original_attn
        for p in self.orig.parameters():
            p.requires_grad = False

        ctx_dim = self.orig._context_dim       # 1024
        inner = self.orig._inner_dim           # n_heads * head_dim

        # ── CLIP stream (always present, index 0) ─────────────
        self.ip_k_proj = nn.Linear(ctx_dim, inner, bias=False)
        self.ip_v_proj = nn.Linear(ctx_dim, inner, bias=False)
        self.ip_k_proj.weight.data.copy_(self.orig.k_proj.weight.data)
        self.ip_v_proj.weight.data.copy_(self.orig.v_proj.weight.data)

        # ── CCIP stream (character identity, optional) ────────
        self._has_ccip = "ccip" in aux_encoders
        self.ip_k_proj_ccip: Optional[nn.Linear] = None
        self.ip_v_proj_ccip: Optional[nn.Linear] = None
        if self._has_ccip:
            self.ip_k_proj_ccip = nn.Linear(ctx_dim, inner, bias=False)
            self.ip_v_proj_ccip = nn.Linear(ctx_dim, inner, bias=False)
            self.ip_k_proj_ccip.weight.data.copy_(
                self.orig.k_proj.weight.data * 0.1
            )
            self.ip_v_proj_ccip.weight.data.copy_(
                self.orig.v_proj.weight.data * 0.1
            )

        # ── LSNet stream (artist style, optional) ─────────────
        self._has_lsnet = "lsnet" in aux_encoders
        self.ip_k_proj_lsnet: Optional[nn.Linear] = None
        self.ip_v_proj_lsnet: Optional[nn.Linear] = None
        if self._has_lsnet:
            self.ip_k_proj_lsnet = nn.Linear(ctx_dim, inner, bias=False)
            self.ip_v_proj_lsnet = nn.Linear(ctx_dim, inner, bias=False)
            self.ip_k_proj_lsnet.weight.data.copy_(
                self.orig.k_proj.weight.data * 0.1
            )
            self.ip_v_proj_lsnet.weight.data.copy_(
                self.orig.v_proj.weight.data * 0.1
            )

        # ── Learnable scalar gates ────────────────────────────
        self.gate_clip = nn.Parameter(torch.tensor(1.0))
        if self._has_ccip:
            self.gate_ccip = nn.Parameter(torch.tensor(0.1))
        if self._has_lsnet:
            self.gate_lsnet = nn.Parameter(torch.tensor(0.1))

        self.ip_scale = ip_scale

        # fine-grained double-stream (lazy-init, CLIP only)
        self.ip_k_proj_fine: Optional[nn.Linear] = None
        self.ip_v_proj_fine: Optional[nn.Linear] = None

    # ------------------------------------------------------------------
    # Fine-grained stream
    # ------------------------------------------------------------------
    def ensure_fine_stream(self):
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
        yield from self.ip_k_proj.parameters()
        yield from self.ip_v_proj.parameters()
        yield self.gate_clip
        if self._has_ccip:
            yield from self.ip_k_proj_ccip.parameters()
            yield from self.ip_v_proj_ccip.parameters()
            yield self.gate_ccip
        if self._has_lsnet:
            yield from self.ip_k_proj_lsnet.parameters()
            yield from self.ip_v_proj_lsnet.parameters()
            yield self.gate_lsnet
        if self.ip_k_proj_fine is not None:
            yield from self.ip_k_proj_fine.parameters()
            yield from self.ip_v_proj_fine.parameters()

    def trainable_state_dict(self):
        sd = {
            "ip_k_proj.weight": self.ip_k_proj.weight,
            "ip_v_proj.weight": self.ip_v_proj.weight,
            "gate_clip": self.gate_clip,
        }
        if self._has_ccip:
            sd["ip_k_proj_ccip.weight"] = self.ip_k_proj_ccip.weight
            sd["ip_v_proj_ccip.weight"] = self.ip_v_proj_ccip.weight
            sd["gate_ccip"] = self.gate_ccip
        if self._has_lsnet:
            sd["ip_k_proj_lsnet.weight"] = self.ip_k_proj_lsnet.weight
            sd["ip_v_proj_lsnet.weight"] = self.ip_v_proj_lsnet.weight
            sd["gate_lsnet"] = self.gate_lsnet
        return sd

    def load_trainable_state_dict(self, sd: dict):
        self.ip_k_proj.weight.data.copy_(sd["ip_k_proj.weight"])
        self.ip_v_proj.weight.data.copy_(sd["ip_v_proj.weight"])
        self.gate_clip.data.copy_(sd.get("gate_clip", torch.tensor(1.0)))
        if self._has_ccip and "ip_k_proj_ccip.weight" in sd:
            self.ip_k_proj_ccip.weight.data.copy_(sd["ip_k_proj_ccip.weight"])
            self.ip_v_proj_ccip.weight.data.copy_(sd["ip_v_proj_ccip.weight"])
            self.gate_ccip.data.copy_(sd.get("gate_ccip", torch.tensor(0.1)))
        if self._has_lsnet and "ip_k_proj_lsnet.weight" in sd:
            self.ip_k_proj_lsnet.weight.data.copy_(sd["ip_k_proj_lsnet.weight"])
            self.ip_v_proj_lsnet.weight.data.copy_(sd["ip_v_proj_lsnet.weight"])
            self.gate_lsnet.data.copy_(sd.get("gate_lsnet", torch.tensor(0.1)))

    # ------------------------------------------------------------------
    # Forward (reads IP tokens from instance attributes)
    # ------------------------------------------------------------------
    # Because Anima DiT Blocks call cross_attn(x, attn_params, context=...)
    # with a fixed signature, we pass IP tokens via instance attributes
    # that the trainer sets before each forward pass.

    def forward(
        self,
        x: torch.Tensor,
        attn_params: AttentionParams,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Read IP tokens stashed by the trainer
        ip_tokens = getattr(self, "_ip_tokens", None)
        ip_tokens_fine = getattr(self, "_ip_tokens_fine", None)
        ip_tokens_ccip = getattr(self, "_ip_tokens_ccip", None)
        ip_tokens_lsnet = getattr(self, "_ip_tokens_lsnet", None)

        # ── 1. Text cross-attention (frozen) ──────────────────
        text_out = self.orig(x, attn_params, context=context, rope_emb=rope_emb)

        ip_contribution = 0.0
        n_heads = self.orig.n_heads
        head_dim = self.orig.head_dim

        def _compute_ip(k_proj: nn.Linear, v_proj: nn.Linear, tokens: torch.Tensor) -> torch.Tensor:
            q = self.orig.q_proj(x)
            k = k_proj(tokens)
            v = v_proj(tokens)
            q, k, v = map(
                lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=n_heads, d=head_dim),
                (q, k, v),
            )
            q = self.orig.q_norm(q)
            k = self.orig.q_norm(k)
            v = self.orig.v_norm(v)
            attn_out = attention_fn([q, k, v], attn_params=attn_params)
            return self.orig.output_proj(attn_out)

        # ── 2. CLIP stream (always active) ────────────────────
        if ip_tokens is not None and ip_tokens.numel() > 0:
            ip_contribution = ip_contribution + self.gate_clip * _compute_ip(
                self.ip_k_proj, self.ip_v_proj, ip_tokens
            )

        # ── 3. CCIP stream (character identity) ───────────────
        if (
            self._has_ccip
            and ip_tokens_ccip is not None
            and ip_tokens_ccip.numel() > 0
        ):
            ip_contribution = ip_contribution + self.gate_ccip * _compute_ip(
                self.ip_k_proj_ccip, self.ip_v_proj_ccip, ip_tokens_ccip
            )

        # ── 4. LSNet stream (artist style) ────────────────────
        if (
            self._has_lsnet
            and ip_tokens_lsnet is not None
            and ip_tokens_lsnet.numel() > 0
        ):
            ip_contribution = ip_contribution + self.gate_lsnet * _compute_ip(
                self.ip_k_proj_lsnet, self.ip_v_proj_lsnet, ip_tokens_lsnet
            )

        # ── 5. Fine-grained CLIP stream (double mode) ──────────
        if ip_tokens_fine is not None and ip_tokens_fine.numel() > 0 and self.ip_k_proj_fine is not None:
            ip_contribution = ip_contribution + _compute_ip(
                self.ip_k_proj_fine, self.ip_v_proj_fine, ip_tokens_fine
            )

        if isinstance(ip_contribution, (int, float)):
            return text_out
        return text_out + self.ip_scale * ip_contribution
