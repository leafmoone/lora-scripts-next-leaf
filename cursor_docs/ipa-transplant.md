# IPA 移植文档 — anima-edit → lora-scripts-next

> 日期: 2026-06-09 | 分支: main

本文档记录从 anima-edit 移植的全部模块及使用方式。

---

## 1. 已移植 (6/6)

### 1.1 MLPImageProjModel
FaceID 风格两层 MLP 投影: pool → Linear(768→1536) → GELU → Linear(1536→1024×N) → LayerNorm。比 ImageProjModel 多了非线性，适合大数据集复杂映射。

### 1.2 LLMResampler
基于 LLMAdapterTransformerBlock 的升级版 Perceiver Resampler。RMSNorm + 1D RoPE + QK-norm attention + 残差 MLP。替代手写 PerceiverAttention。

### 1.3 独立 IP 学习率
`--ip_adapter_lr` 参数，默认 = lr × 5。IP 投影层独立收敛。

### 1.4 LLM Adapter 基础模块
`_adapter_modules.py` 包含 8 个基础模块(RMSNorm, Attention, RoPE, Omni 等)。

### 1.5 预计算嵌入缓存到磁盘
`ip_adapter_precomputed_emb_dir` 已接入训练流程。首次访问 dataset 时会自动为训练图生成 `.pt` cache；后续 step 可直接读取 CLIP/CCIP/LSNet global features。`resampler` / `double` 模式会额外保存 patch features。

### 1.6 Omni Adapter 接线
`adapter_type="omni"` 已接入 CLI、WebUI、sidecar metadata 与推理加载。当前定位为实验性 patch/resampler 流，主要用于 `ipa_mode="resampler"` 或 `ipa_mode="double"`。

## 2. 模式语义

| `ipa_mode` | global tokens | fine / patch tokens | 适用场景 |
|------------|---------------|---------------------|----------|
| `simple` | `ImageProjModel` / MLP global embedding | 无 | 身份/画风语义注入，速度最快 |
| `resampler` | 无单独 global；patch → Resampler/Omni 后作为主 `_ip_tokens` | 无单独 `_ip_tokens_fine` | 需要更强局部/构图参考 |
| `double` | global `ImageProjModel` / MLP | patch → Resampler/Omni 后作为 `_ip_tokens_fine` | 同时保留全局语义与细节参考 |

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

### 预计算 embedding cache
```toml
ip_adapter_precomputed_emb_dir="./cache/ipa_emb"
```

### Omni Adapter
```toml
adapter_type="omni"
ipa_mode="resampler" # 或 "double"
```

### MLPImageProjModel
```python
# 在 load_unet_lazily 中替换
from ip_adapter.anima_ip_image_proj import MLPImageProjModel
self.image_proj = MLPImageProjModel(feature_dim=1024, num_tokens=8)
```
