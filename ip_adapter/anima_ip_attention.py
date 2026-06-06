"""
Anima IP-Adapter Cross-Attention Layer (Concat Fusion)

Concatenates IP tokens onto the text context before the DiT's
frozen cross-attention, so image and text conditions compete
in the same softmax.  No additional trainable parameters are
introduced in the attention layer — all trainable params live
in ImageProjModel.

Design:
  context_full = concat([text_context, ip_tokens_clip,
                         ip_tokens_ccip?, ip_tokens_lsnet?])
  output = original_attn(x, context=context_full)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from library.attention import AttentionParams


class AnimaIPCrossAttention(nn.Module):

    def __init__(self, original_attn: nn.Module, ip_scale: float = 1.0,
                 aux_encoders: tuple[str, ...] = ()):
        super().__init__()
        self.orig = original_attn
        for p in self.orig.parameters():
            p.requires_grad = False

    def ensure_fine_stream(self):
        pass

    def trainable_parameters(self):
        return iter([])

    def trainable_state_dict(self):
        return {}

    def load_trainable_state_dict(self, sd: dict):
        pass

    def forward(self, x: torch.Tensor, attn_params: AttentionParams,
                context: Optional[torch.Tensor] = None,
                rope_emb: Optional[torch.Tensor] = None) -> torch.Tensor:

        concat_parts = []
        if context is not None and context.numel() > 0:
            concat_parts.append(context)

        for attr in ("_ip_tokens", "_ip_tokens_fine",
                     "_ip_tokens_ccip", "_ip_tokens_lsnet"):
            tk = getattr(self, attr, None)
            if tk is not None and tk.numel() > 0:
                concat_parts.append(tk)

        full_ctx = torch.cat(concat_parts, dim=1) if len(concat_parts) > 1 else concat_parts[0]
        return self.orig(x, attn_params, context=full_ctx, rope_emb=rope_emb)
