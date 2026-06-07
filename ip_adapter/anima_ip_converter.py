"""IP-Adapter Block injector (concat mode)."""
from typing import Dict
import torch.nn as nn
from .anima_ip_attention import AnimaIPCrossAttention

class AnimaIPAConverter:
    @classmethod
    def create(cls, dit: nn.Module) -> Dict[str, AnimaIPCrossAttention]:
        adapters = {}
        for name, mod in dit.named_modules():
            if mod.__class__.__name__ == "Block":
                adapters[name] = AnimaIPCrossAttention(mod.cross_attn)
                mod.cross_attn = adapters[name]
        if not adapters:
            raise RuntimeError(
                "AnimaIPAConverter found no DiT 'Block' modules to wrap — the "
                "IP-Adapter would be inert. Check the model architecture."
            )
        return adapters

    @classmethod
    def get_trainable_params(cls, adapters):
        for a in adapters.values():
            yield from a.trainable_parameters()
