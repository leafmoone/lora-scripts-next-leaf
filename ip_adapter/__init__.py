"""
IP-Adapter support for Anima DiT models.

Modules:
  anima_ip_attention   — IP Cross-Attention layer
  anima_ip_converter    — DiT Block wrapper injector
  anima_ip_image_proj   — Image projection model
  anima_ip_adapter      — Inference wrapper
"""

from .anima_ip_attention import AnimaIPCrossAttention
from .anima_ip_converter import AnimaIPAConverter
from .anima_ip_image_proj import ImageProjModel, Resampler
from .anima_ip_adapter import AnimaIPAdapter
