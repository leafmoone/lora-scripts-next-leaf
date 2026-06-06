"""
Anima IP-Adapter Converter — Block Wrapper Injector

Walks all ``Block`` instances inside an Anima DiT model and replaces
each ``block.cross_attn`` with an ``AnimaIPCrossAttention`` wrapper.

Used at training start to inject trainable IP projection layers without
modifying ``library/anima_models.py``.
"""

from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .anima_ip_attention import AnimaIPCrossAttention


class AnimaIPAConverter:
    """Inject IP-Adapter layers into an Anima DiT model."""

    @classmethod
    def create(
        cls,
        dit: nn.Module,
        ip_scale: float = 1.0,
        ipa_mode: str = "simple",
        aux_encoders: tuple[str, ...] = (),
    ) -> Dict[str, AnimaIPCrossAttention]:
        """
        Walk the DiT's ``named_modules()``, wrap every ``Block.cross_attn``.

        Args:
            dit: Anima DiT model.
            ip_scale: Multiplier for the IP cross-attention branch.
            ipa_mode: "simple" / "resampler" / "double".
            aux_encoders: Tuple of auxiliary encoder names, e.g. ("ccip",) or ("ccip", "lsnet").

        Returns:
            ``{block_path: AnimaIPCrossAttention}`` mapping.
        """
        ip_adapters: Dict[str, AnimaIPCrossAttention] = {}

        for name, module in dit.named_modules():
            if module.__class__.__name__ == "Block":
                ip_attn = AnimaIPCrossAttention(
                    module.cross_attn,
                    ip_scale=ip_scale,
                    aux_encoders=aux_encoders,
                )
                module.cross_attn = ip_attn
                if ipa_mode == "double":
                    ip_attn.ensure_fine_stream()
                ip_adapters[name] = ip_attn

        return ip_adapters

    @classmethod
    def get_trainable_params(cls, ip_adapters: Dict[str, AnimaIPCrossAttention]):
        """Flatten trainable parameters from all IP adapters."""
        for ip_attn in ip_adapters.values():
            yield from ip_attn.trainable_parameters()
