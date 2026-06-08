# IPA-LSNet 审查记录 — Stage 1-6 全部修复 + Resampler 接线 + 多图平均

> 分支: `ipa-lsnet` | 日期: 2026-06-08 | 配套设计: `ipa-lsnet-design.md`, `ipa_resampler.md`

本文档按时间顺序记录所有审查结论、修复代码和验证结果。每一阶段最后标记 ✅。

---

## Stage 1: PerceiverAttention 命名 + 独立 LayerNorm

**文件**: `ip_adapter/anima_ip_image_proj.py`

### 审查结论

旧代码逻辑正确（q=可学习latents, kv=图像patch），但形参命名误导（`x`/`latents` vs 直觉的 `latents`/`patches`），且共享一个 `LayerNorm`。

### 修复

```python
# 修复前
self.norm = nn.LayerNorm(dim)
def forward(self, x, latents):
    x = self.norm(x); latents = self.norm(latents)
    q = self.to_q(x)         # x == latents 调用方传入顺序

# 修复后
self.norm_latents = nn.LayerNorm(dim)
self.norm_patches = nn.LayerNorm(dim)
def forward(self, latents, patches):
    latents = self.norm_latents(latents)
    patches = self.norm_patches(patches)
    q = self.to_q(latents)   # 语义清晰
```

调用方 `Resampler.forward` 无需改动（`attn(latents, x)` → patches）。

✅ 通过 | 提交: `4868291`


## Stage 2: 编码器输出 patch 特征

**文件**: `ip_adapter/ccip_encoder.py`, `ip_adapter/lsnet_encoder.py`, 调用方

### 审查结论

三个编码器在 simple 模式下只输出全局特征（768-dim），resampler/double 需要 patch-level 特征。

### 修复

- **CCIP CaFormer**: `backbone.forward(x, return_patches=True)` → `(feat_768, patches_B×144×768)`
- **LSNet**: `forward(x, return_patches=True)` → `(feat_768, patches_B×196×768)`
- **CLIP**: 在 `_encode_images_to_ip_tokens` 里按 `ipa_mode` 决定是否请求 `output_hidden_states=True` 和取 `hidden_states[-1][:, 1:, :]`

| 编码器 | 输入尺寸 | Patch 形状 | Patch 数 |
|--------|---------|-----------|---------|
| CLIP  | 224×224 | (B, 256, 1024) | 256 |
| CCIP  | 384×384 | (B, 144, 768) | 144 |
| LSNet | 448×448 | (B, 196, 768) | 196 |

✅ 通过 | 提交: `4868291`


## Stage 3: 训练入口按 ipa_mode 构建 Resampler

**文件**: `ip_adapter/anima_ip_train.py`, `ip_adapter/anima_ip_image_proj.py`

### 审查结论

`load_unet_lazily` 之前只构建 `ImageProjModel / MultiStreamProj`，不区分 mode。Resampler 类存在但从未被实例化。

### 修复

新增 `_make_stream_projectors` 按 mode 构建：
- `simple`: `MultiStreamProj`（每组用 `ImageProjModel`）
- `resampler`: `MultiStreamProj.from_modules([Resampler(...) × num_streams])`
- `double`: 同时构建 `image_proj`(global) + `image_proj_resampler`(patch)

新增 `MultiStreamProj.from_modules(cls, modules)` 工厂方法。

✅ 通过 | 提交: `4868291`


## Stage 4: _encode_images_to_ip_tokens 分支处理

**文件**: `ip_adapter/anima_ip_train.py`

### 审查结论

`_encode_images_to_ip_tokens` / `get_noise_pred_and_target` 需要根据 `self._ipa_mode` 决定：
1. 是否请求 patch 特征
2. 是否同时产出 `ip_tokens_fine`

### 修复

- 按 `ipa_mode` 分支处理 CLIP/CCIP/LSNet 编码路径
- `double` 模式下用 `image_proj_resampler` 产 `ip_tokens_fine`
- 返回值格式统一为 `(ip_clip, ip_fine, ip_ccip, ip_lsnet)`

✅ 通过 | 提交: `4868291`


## Stage 5: 保存/加载 metadata + 推理 wrapper

**文件**: `ip_adapter/anima_ip_train.py`, `ip_adapter/anima_ip_adapter.py`

### 审查结论

sidecar metadata 缺少 `ipa_mode` 和 resampler 超参，推理 wrapper 无法重建。

### 修复

**Trainer 侧**:
- `_ip_adapter_state_dict` 增加 `image_proj_resampler` 参数
- `_save_ip_adapter_weights` metadata 增加 `ipa_mode`, `ipa_resampler_depth`, `ipa_resampler_heads`, `ipa_num_queries`
- `get_trainable_params` 增加 `image_proj_resampler` 参数收集
- `load_ip_adapter_weights` 增加 `image_proj_resampler` 加载

**推理 wrapper 侧**:
- `AnimaIPAdapter.__init__` 接受 `image_proj_resampler`, `ipa_mode`
- `from_pretrained` 从 metadata 读取 `ipa_mode`，按模式重建 `Resampler`
- `encode()` 按 `self.ipa_mode` 决定是否取 patch 特征

✅ 通过 | 提交: `4868291`


## Stage 6: 推理多图平均融合

**文件**: `ip_adapter/anima_ip_adapter.py`

### 审查结论

推理时多张参考图（如同一角色的不同角度）应能融合为一组 IP token。不改训练侧。

### 修复

`encode()` 新增 `clip_images`, `ccip_images`, `lsnet_images` 三个 list 参数：
- 每张图独立编码 → 1024 投影后特征
- `torch.stack(feats).mean(dim=0)` 求平均
- 平均后再过 `image_proj`

平均在 1024 投影后做（特征语义对齐空间），不在 768 原始空间做。

**使用方式**:
```python
adapter.set_reference(
    ccip_images=["face1.png", "face2.png", "face3.png"],
    ccip_scale=1.0,
)
```

✅ 通过 | 提交: `085a5c5`


## 验证清单

- [x] PerceiverAttention 独立 LayerNorm 无冲突
- [x] CCIP/LSNet `return_patches` 向后兼容（默认 False）
- [x] MultiStreamProj.from_modules 工厂方法正确
- [x] ipa_mode=simple 路径完全不变（向后兼容）
- [x] ipa_mode=resampler/double 路径不报错
- [x] sidecar metadata 完整可恢复
- [x] 推理 wrapper 多图平均不污染训练路径
- [ ] ipa_mode=resampler/double 端到端训练验证（需实际跑一次）
- [ ] 采样路径对 resampler/double 的 `_ip_tokens_fine` 注入验证

## 关键风险

1. ipa_mode=resampler/double 未端到端训练过—代码逻辑正确但实际梯度流/收敛需验证
2. 三个编码器的 patch 维度不一致（256/144/196）→ Resampler 的 cross-attn 开销不同，维度对齐正确性已在编码器 forward 中验证
3. 多图平均只在推理用（纯 forward, no_grad），不影响训练 pipeline
