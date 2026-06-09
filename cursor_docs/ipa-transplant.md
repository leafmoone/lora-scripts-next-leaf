# IPA 移植文档 — anima-edit → lora-scripts-next

> 日期: 2026-06-09 | 分支: main

本文档记录从 anima-edit 移植的全部模块及使用方式。

---

## 1. 已移植 (4/6)

### 1.1 MLPImageProjModel
FaceID 风格两层 MLP 投影: pool → Linear(768→1536) → GELU → Linear(1536→1024×N) → LayerNorm。比 ImageProjModel 多了非线性，适合大数据集复杂映射。

### 1.2 LLMResampler
基于 LLMAdapterTransformerBlock 的升级版 Perceiver Resampler。RMSNorm + 1D RoPE + QK-norm attention + 残差 MLP。替代手写 PerceiverAttention。

### 1.3 独立 IP 学习率
`--ip_adapter_lr` 参数，默认 = lr × 5。IP 投影层独立收敛。

### 1.4 LLM Adapter 基础模块
`_adapter_modules.py` 包含 8 个基础模块(RMSNorm, Attention, RoPE, Omni 等)。

## 2. 待移植 (2/6)
- 预计算嵌入缓存到磁盘
- Omni Adapter 接线

## 3. 使用

### 独立 IP lr
```toml
learning_rate=1e-4
ip_adapter_lr=5e-4
```

### LLMResampler
```toml
ipa_mode=resampler
```

### MLPImageProjModel
```python
# 在 load_unet_lazily 中替换
from ip_adapter.anima_ip_image_proj import MLPImageProjModel
self.image_proj = MLPImageProjModel(feature_dim=1024, num_tokens=8)
```
