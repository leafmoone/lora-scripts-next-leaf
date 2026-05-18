# Anima LoRA 训练指南

## 进入训练页面

启动 WebUI 后，左侧 sidebar 点 **Anima LoRA** 进入。

> 技术细节：该页面复用了原 SD3 的 URL 槽位（`/lora/sd3.html`），但参数集合与训练脚本已完全是 Anima。

## 模型路径

表单里需要填以下模型路径：

| 字段 | 含义 |
|------|------|
| `pretrained_model_name_or_path` | Anima DiT 主权重，如 `./sd-models/anima/anima-base-v1.0.safetensors` |
| `vae` | Qwen Image VAE 模型路径（必填） |
| `qwen3` | Qwen3 文本模型，可填 `.safetensors` / `.pt` 文件，或完整本地模型目录 |
| `t5` | T5 文本编码器权重 |

## 训练类型

- **LoRA** — 默认训练类型，适合大多数场景
- **LoKr** — 使用 LyCORIS 后端（`lycoris.kohya` + `algo=lokr`），支持 CP 分解、DoRA 等高级参数

## 预览图生成

打开表单里的 **`enable_preview`** 开关后，采样会切到 Anima 推荐参数：
- 分辨率：1024×1024
- CFG：4.5
- 步数：40
- Seed：42
- 自动填入 Anima 风格的正反向提示词

## 训练步数经验值

在同一套数据与分辨率下，**约 1000–3000 次优化步** 往往已能呈现可用的角色外观。实际所需步数随素材量、repeat、网络维度、学习率变化很大，请以验证图为准。

**`num batches per epoch`** × **目标 epoch** ≈ 累计步数（例如每 epoch 510 batch → 第 2 个 epoch 结束约 1020 步）。

## 后端架构

本地入口 [`scripts/dev/anima_train_network.py`](../scripts/dev/anima_train_network.py) 是兼容 wrapper：它适配 GUI 生成的 TOML，并委托给 `vendor/sd-scripts` 中的 kohya-ss 后端执行训练。

配置文件：[`config/anima_backend.toml`](../config/anima_backend.toml)
