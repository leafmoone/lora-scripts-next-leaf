# IP-Adapter with CCIP + LSNet Auxiliary Encoders — Design Doc

> 分支: `ipa-lsnet` | 日期: 2026-06-07 | 状态: 训练已跑通

---

## 1. 目标

在 Anima DiT 上训练 IP-Adapter，支持**多编码器并行条件注入**：

```
CLIP  (ViT-L/14, 内容条件)  ──┐
CCIP  (CaFormer, 角色身份)  ──┼──→ concat[text, ip_clip, ip_ccip, ip_lsnet] → DiT
LSNet (XL-448, 画风风格)    ──┘
```

四个训练模式: `clip_only` / `clip_ccip` / `clip_lsnet` / `clip_ccip_lsnet`

---

## 2. 数据流

### 2.1 训练 forward

每 step 完整流程:

```
训练图 (B, 3, H, W) in [0,1]
  │
  ├─ CLIP ViT-L/14 (frozen, 224×224)
  │    image_embeds (B, 768) → clip_proj (768→1024, trainable) → (B, 1024)
  │
  ├─ CCIP CaFormer (frozen, 384×384)
  │    backbone (B, 768) → self.proj (768→1024, trainable) → (B, 1024)
  │
  └─ LSNet XL-448 (frozen, 448×448)
       backbone (B, 768) → self.proj (768→1024, trainable) → (B, 1024)

  ↓

MultiStreamProj = [
  ImageProjModel[0]: (B, 1024) → (B, N_clip, 1024)   ← CLIP tokens
  ImageProjModel[1]: (B, 1024) → (B, N_ccip, 1024)   ← CCIP tokens
  ImageProjModel[2]: (B, 1024) → (B, N_lsnet, 1024)   ← LSNet tokens
]

  ↓

AnimaIPCrossAttention.forward(x, context=text):
  context_full = concat([text_ctx, ip_clip, ip_ccip, ip_lsnet])
  return orig_attn(x, context=context_full)
```

### 2.2 维度对齐

所有编码器输出统一到 **1024-dim** 以匹配 Anima DiT 的 `crossattn_emb_channels`:

| 编码器 | 原生维度 | 投影层 | 参数量 |
|--------|---------|--------|--------|
| CLIP (image_embeds) | 768 | `Linear(768,1024) + LayerNorm` | ~79万 |
| CCIP (attnpool CLS) | 768 | `Linear(768,1024) + LayerNorm` | ~79万 |
| LSNet (avg pool)    | 768 | `Linear(768,1024) + LayerNorm` | ~79万 |

三个投影层均可训练。

---

## 3. 融合方案: Concat（原版 IP-Adapter）

IP tokens 直接 concat 到 text context 上，让图像和文本条件共享同一组预训练的 `k_proj` / `v_proj`，在同一 softmax 中竞争注意力权重：

```
Q → cross_attn(
  K = concat([k_proj(text), k_proj(ip_clip), k_proj(ip_ccip), k_proj(ip_lsnet)]),
  V = concat([v_proj(text), v_proj(ip_clip), v_proj(ip_ccip), v_proj(ip_lsnet)])
)
```

`Attention` 模块本身不引入任何新参数——所有可训练参数集中在投影层和 `ImageProjModel`，总计 ~8.4M。

### 为什么选 concat

| | Concat |
|--|--------|
| 每 Block 新增参数 | 0 |
| 总可训练参数 | ~8.4M (仅投影+ImageProj) |
| 过拟合风险 | 低 |
| 泛化能力 | 借力 DiT 预训练知识（k_proj/v_proj 学过几亿图文对） |

### 本质优势

`ImageProjModel` 只需要把图像特征对齐到文本 embedding 空间，原生的 `k_proj`/`v_proj` 在 Anima 预训练时已经见过几亿图文对，自动知道如何处理对齐后的输入。这就是 **最小假设、最大复用** ——用 8.4M 参数撬动 11.3B DiT 的全部知识。

---

## 4. CCIP 编码器

### 4.1 架构

```
CaFormer b36-24 backbone (ccip-caformer_b36-24.ckpt):
  ds0: Conv2d(3→128, k=7, s=4) → 96×96
  stage0: 3× Block(dim=128, n_heads=2)
  ds1: Conv2d(128→256, k=3, s=2) → 48×48
  stage1: 12× Block(dim=256, n_heads=4)
  ds2: Conv2d(256→512, k=3, s=2) → 24×24
  stage2: 18× Block(dim=512, n_heads=8)
  ds3: Conv2d(512→768, k=3, s=2) → 12×12
  stage3: 3× Block(dim=768, n_heads=12)
  attnpool: AttentionPool2d(12×12, dim=768, heads=12) → (B, 768)
```

### 4.2 加载方式

- Pure PyTorch 实现（完整 CaFormer 架构手写），直接从 `.ckpt` 加载权重
- 与 `dghs-imgutils` 的 ONNX 接口**输出一致**（attnpool CLS token, 768-dim）
- 全 GPU 批量推理，无需逐张 PIL 转换
- 96M 参数，输入 384×384，CLIP 归一化

### 4.3 使用层

`attnpool` 输出的 CLS token — 经过 attention pooling 从 12×12 个 spatial patches 中提取的全局角色身份表征。这是分类头之前语义最丰富的层。

---

## 5. LSNet 编码器

### 5.1 架构

```
LSNet-XL-448 backbone (best_checkpoint.pth):
  patch_embed: Conv2d(3→48→96→192), 3-stage downsampling
  blocks1: 8× Block(dim=192, depthwise mixer + SE + FFN)
  blocks2: 14× Block(dim=384, LSConv/RepVGGDW mixer)
  blocks3: 18× Block(dim=576)
  blocks4: 20× Block(dim=768)
  → adaptive_avg_pool2d → (B, 768)
```

### 5.2 加载方式

- 从 `spawner1145/lsnet-test` 获取模型源码
- 构建 `LSNet` 实例（`num_classes=0`，去掉分类头）
- 加载 `best_checkpoint.pth` 的 `model` 部分
- 102M 参数，输入 448×448，ImageNet 归一化

### 5.3 使用层

`blocks4` 输出经 `adaptive_avg_pool2d` 后的全局画风特征（768-dim）。这是 39,260 类艺术风格的分类头之前、画家风格信息最密集的特征层。

---

## 6. 训练配置

### 6.1 可训练参数

| 组件 | 参数量 |
|------|--------|
| CLIP 投影层 (768→1024) | ~79万 |
| CCIP 投影层 (768→1024) | ~79万 |
| LSNet 投影层 (768→1024) | ~79万 |
| ImageProjModel × N (MultiStreamProj) | ~4.2万 × N |
| **总计 (3流)** | **~8.4M** |
| **总计 (单 CLIP)** | **~4.2M** |

冻结部分: CLIP (428M) + CCIP (96M) + LSNet (102M) + Anima DiT (11.3B) + Qwen3 (596M) + VAE

### 6.2 训练速度

RTX 4090, batch_size=4, bf16:

| 配置 | ~s/it |
|------|-------|
| CLIP only (无采样) | ~2.0s |
| CLIP + CCIP (无采样) | ~2.4s |
| CLIP + CCIP + LSNet (无采样) | ~2.8s |

### 6.3 采样

- 参考图路径: `--sample_reference_image`
- 编码延迟到 `on_prompt_start` 回调，非采样 step 无开销
- 采样图片输出到 `{output_dir}/sample/`，PNG 格式

---

## 7. CLI 用法

```bash
# CLIP + CCIP 双流
accelerate launch ip_adapter/anima_ip_train.py \
  --pretrained_model_name_or_path anima.safetensors \
  --vae qwen_image_vae.safetensors \
  --qwen3 qwen3.safetensors \
  --clip_model openai/clip-vit-large-patch14 \
  --aux_encoders clip_ccip \
  --ccip_ckpt /root/lanyun-tmp/workspace/ccip/ccip-caformer_b36-24.ckpt \
  --num_ip_tokens 4 \
  --train_data_dir ./train/ipa_dataset \
  --output_dir ./output/ipa \
  --sample_reference_image ./ref.png \
  --learning_rate 1e-4 --max_train_epochs 100 --train_batch_size 4

# CLIP + CCIP + LSNet 三流
accelerate launch ip_adapter/anima_ip_train.py \
  --aux_encoders clip_ccip_lsnet \
  --ccip_ckpt /root/lanyun-tmp/workspace/ccip/ccip-caformer_b36-24.ckpt \
  --lsnet_ckpt /root/lanyun-tmp/workspace/lsnet/best_checkpoint.pth \
  ...
```

---

## 8. 文件结构

```
ip_adapter/
├── __init__.py                 # 模块导出
├── anima_ip_attention.py       # Concat-mode cross-attention wrapper
├── anima_ip_converter.py       # DiT Block injector
├── anima_ip_image_proj.py      # ImageProjModel + Resampler + MultiStreamProj
├── anima_ip_adapter.py         # Inference wrapper
├── anima_ip_train.py           # Training script (AnimaIPAdapterTrainer)
├── ccip_encoder.py             # CCIP CaFormer PyTorch encoder (96M)
├── lsnet_encoder.py            # LSNet-XL-448 encoder (102M)
├── _lsnet_model.py             # LSNet model definition (from spawner1145/lsnet-test)
└── _ska.py                     # SKA module (LSNet dependency)

frontend/dist/lora/
└── anima-ipa.html              # IP-Adapter training WebUI page

mikazuki/app/
├── api.py                      # trainer_mapping: "anima-ipa" entry point
└── application.py              # (no changes needed)

mikazuki/utils/
└── train_utils.py              # validate_model whitelist: +"anima-ipa"
```

---

## 9. 技术决策记录

### 为什么不在 DiT Block 内部 K/V 拼接

`AnimaIPCrossAttention` 在 Block 外部完成 concat：`self.orig(x, context=concat([text, ip_tokens]))`。不需要修改 `library/anima_models.py`。

### 为什么 IP tokens 通过 instance attributes 传递

DiT Block 内部 `self.cross_attn(x, ...)` 签名固定。Trainer 在调用 DiT forward 前 `attn._ip_tokens = tokens`，内部 `forward` 读 `getattr(self, "_ip_tokens")`。零侵入。

### 为什么用 image_embeds (768) 而非 hidden_states[-1] (1024)

1. IP-Adapter 原论文用 `image_embeds` — CLIP projection head 已经学了图文对齐
2. 不用 `output_hidden_states=True` — 约 30% forward 加速
3. 768→1024 投影层 (~79万参数) 负责维度对齐，和 CCIP/LSNet 设计一致

### 为什么 optimizer 需要 monkey-patch

`train_network.py` 从 LoRA network 获取参数列表，但 IP-Adapter 不用 LoRA。Monkey-patch `LoRANetwork.prepare_optimizer_params` 注入 ImageProjModel 参数。

### 为什么 sampling 延迟编码

`anima_train_utils.sample_images` 有 steps 检查（`sample_every_n_steps`），但回调之前就编码会浪费 ~1.2s/step 编码器 forward。延迟到 `on_prompt_start` 只在真正生成图时才触发。

---

## 10. 已知限制

1. **CCIP 输入需单角色** — CCIP 对多人图效果不确定
2. **LSNet 输入 448×448** — 比 CLIP 慢，但画风特征更丰富
3. **标签要求** — 每张图配 `.txt`，以触发词开头，不要写画风/角色名（以免和 LSNet/CCIP 信号冲突）
4. **`ipa_mode` 简化** — concat 方案下 simple/resampler/double 都走 ImageProjModel，暂不支持 patch-level resampler
