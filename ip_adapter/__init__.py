"""
IP-Adapter support for Anima DiT models.

Modules:
  anima_ip_attention    — IP Cross-Attention layer (multi-stream gated fusion)
  anima_ip_converter    — DiT Block wrapper injector
  anima_ip_image_proj   — Image projection (MultiStreamProj)
  anima_ip_adapter      — Inference wrapper
  ccip_encoder          — CCIP character identity encoder
  lsnet_encoder         — LSNet artist style encoder
"""

# Standalone modules (no ``library`` dependency)
from .anima_ip_image_proj import ImageProjModel, Resampler, MultiStreamProj
from .ccip_encoder import CCIPIdentityEncoder, load_ccip_encoder
from .lsnet_encoder import LSNetStyleEncoder, load_lsnet_encoder

# Training-dependent modules (require vendor/sd-scripts)
try:
    from .anima_ip_attention import AnimaIPCrossAttention
    from .anima_ip_converter import AnimaIPAConverter
    from .anima_ip_adapter import AnimaIPAdapter
except ImportError:
    pass
