# Anima Train · Gemma 4 VLM 后端说明（vLLM / transformers）

本文说明 Tag-Edit-Leaf **Anima Train** 模式下 Gemma-4-E4B 的 VLM 推理后端、当前环境限制，以及如何配置以使用 vLLM 或本地 transformers。

相关入口：

- 前端：`/tag-edit-leaf.html` → 标注模式 **Anima Train**
- 配置文件：`config/anima_caption_models.json`
- 手动启动 vLLM：`scripts/start_gemma_vllm.sh`
- 流水线代码：`tools/anima_caption_pipeline/`

---

## 1. 流程概览

Anima Train 对每张图执行：

```
WD14 打标 → VLM 两步链 → 写入 .txt（tags 行 + 空行 + caption 行）
                │
                ├─ extract_tags_from_image
                └─ generate_natural_caption
```

VLM 阶段可选后端：

| 后端 | 说明 | 适用场景 |
|------|------|----------|
| **vLLM** | HTTP 调用 OpenAI 兼容 API（端口 9002） | 驱动支持 CUDA 13、需要并发加速 |
| **transformers** | 进程内加载 `AutoModelForMultimodalLM` | CUDA 12.8 等 vLLM 不可用环境（当前默认） |
| **auto** | 先探测 vLLM；输出异常则回退 transformers | 不确定 vLLM 是否可用时 |

WD14 与 VLM **串行**占用 GPU：先跑完 WD14 并释放显存，再跑 VLM。若勾选「WD14 完成后自动启动 vLLM」，仅在 `gemma_vlm_backend` 不为 `transformers` 时才会启动 vLLM 服务。

---

## 2. 为什么当前机器默认不用 vLLM

### 2.1 环境事实

| 项目 | 典型 AutoDL 现状 | vLLM 0.22.1 要求 |
|------|------------------|------------------|
| 驱动 | 570.x，最高 **CUDA 12.8** | **CUDA 13** 自定义算子（`nvidia-cutlass-dsl[cu13]`） |
| PyTorch | 2.11+cu128 | 与 cu128 可共存，但 vLLM 扩展仍依赖 cu13 |

### 2.2 两种失败模式

**模式 A：不禁用 custom ops（期望的正常 vLLM）**

```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
```

vLLM 引擎 **无法启动**。

**模式 B：加 `-cc.custom_ops '["none"]'`（当前 workaround）**

vLLM **能启动**，但 Gemma 4 生成结果异常：

- API 返回 `content: ""`
- `token_ids` 多为 `[0, 0, 0, …]`（`<pad>`）
- 流水线报：`JSON parse failed … No JSON object start found`

用 **transformers 本地推理** 同模型、同图片则输出正常 JSON。因此项目在 CUDA 12.8 上默认：

```json
"gemma_vlm_backend": "transformers"
```

---

## 3. 配置文件说明

文件路径：`config/anima_caption_models.json`

### 3.1 当前推荐（CUDA 12.8，稳定可用）

```json
"gemma-4-e4b": {
  "display_name": "Gemma-4-E4B (vLLM)",
  "default_api_url": "http://127.0.0.1:9002/v1/chat/completions",
  "default_served_name": "spawner-gemma-4-e4b-it",
  "local_model_dir": "models/gemma-4-E3B-it",
  "modelscope_id": "spawner/spawner-gemma-4-E4B-it",
  "port": 9002,
  "gemma_vlm_backend": "transformers",
  "vllm_serve": {
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.42,
    "max_num_seqs": 8
  }
}
```

| 字段 | 含义 |
|------|------|
| `gemma_vlm_backend` | `transformers` / `vllm` / `auto` |
| `vllm_serve.max_model_len` | 上下文长度上限 |
| `vllm_serve.gpu_memory_utilization` | vLLM 预占显存比例；同卡训练建议 **0.42**，独占 GPU 可提高到 0.9 |
| `vllm_serve.max_num_seqs` | 最大并发序列数，应 ≥ UI「VLM 并发数」 |
| `vllm_serve.enable_custom_ops` | 见下文；默认 **false**（即仍加 `custom_ops none`） |

**transformers 模式特点：**

- **不会**启动 vLLM
- VLM **串行**推理（UI「VLM 并发数」无效）
- 进度提示：`Gemma 使用本地 transformers，跳过 vLLM 启动`

### 3.2 启用 vLLM（需 CUDA 13  capable 驱动）

换用支持 **CUDA 13** 的实例后（`nvidia-smi` 中 CUDA Version ≥ 13，驱动通常 580+）：

```json
"gemma-4-e4b": {
  ...
  "gemma_vlm_backend": "vllm",
  "vllm_serve": {
    "enable_custom_ops": true,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.42,
    "max_num_seqs": 8
  }
}
```

或使用自动探测：

```json
"gemma_vlm_backend": "auto"
```

| `gemma_vlm_backend` | 行为 |
|---------------------|------|
| `transformers` | 永不使用 vLLM |
| `vllm` | 强制 vLLM；不回退 |
| `auto` | 探测 vLLM；失败则 transformers |

API 请求体可覆盖配置：`"gemma_vlm_backend": "vllm"`。

---

## 4. 手动启动与验证 vLLM

### 4.1 下载模型（若尚未下载）

在项目根目录执行：

```bash
modelscope download spawner/spawner-gemma-4-E4B-it \
  --local_dir ./models/gemma-4-E3B-it
```

### 4.2 启动服务

**CUDA 12.8（仅能验证启动，Gemma 输出仍可能异常）：**

```bash
cd /path/to/lora-scripts-next-leaf
bash scripts/start_gemma_vllm.sh
```

**CUDA 13 环境（正常 vLLM）：**

```bash
export VLLM_ENABLE_CUSTOM_OPS=1
bash scripts/start_gemma_vllm.sh
```

等价于在 `vllm_serve` 中设置 `"enable_custom_ops": true`，由 SD-Trainer 自动启动时生效。

### 4.3 健康检查

```bash
curl -s http://127.0.0.1:9002/v1/models | head -c 500
```

### 4.4 生成测试（必须通过再改 config）

```bash
curl -s http://127.0.0.1:9002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "spawner-gemma-4-e4b-it",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "max_tokens": 16,
    "temperature": 0
  }'
```

**合格标准：** `choices[0].message.content` 含可读英文（非空、非纯 `<pad>`）。

---

## 5. UI 设置（Tag-Edit-Leaf）

| UI 项 | transformers 模式 | vLLM 模式 |
|-------|---------------------|-----------|
| VLM 模型 | Gemma-4-E4B | Gemma-4-E4B |
| WD14 完成后自动启动 vLLM | 无效（跳过） | 建议开启 |
| VLM 并发数 | 无效（串行） | 建议 4–8，不超过 `max_num_seqs` |
| VLM 批量大小 | Anima Train 下通常忽略 | 同左 |

修改 `config/anima_caption_models.json` 后需 **重启 SD-Trainer**。

---

## 6. 显存与并发建议

同一张 GPU 上 WD14 + Gemma vLLM 时：

| 参数 | 建议 |
|------|------|
| `gpu_memory_utilization` | 0.42（与 LoRA 训练共存）；独占标注可 0.7–0.9 |
| `max_num_seqs` | 8–24；越大并发越高，显存越高 |
| `max_model_len` | 4096 足够两步链 JSON；不必盲目加大 |

transformers 模式单次加载约 **15GB+** 权重，与 vLLM 类似；不会与 vLLM 同时驻留（回退时会停掉 vLLM）。

---

## 7. 故障排查

### 7.1 `JSON parse failed … No JSON object start found`

**原因：** vLLM 返回空或乱码（多为 CUDA 12.8 + `custom_ops none`）。

**处理：**

1. 确认 `gemma_vlm_backend` 为 `transformers`；或
2. 升级至 CUDA 13 驱动并设 `enable_custom_ops: true` + `gemma_vlm_backend: vllm`。

### 7.2 `Engine core initialization failed`

**原因：** 未禁用 custom ops，且驱动不支持 CUDA 13 runtime。

**处理：** 保持 `enable_custom_ops: false`（默认），或升级驱动后设 `true`。

### 7.3 WD14 OOM

**原因：** vLLM 在 WD14 之前占用显存。

**处理：** 开启「WD14 完成后自动启动 vLLM」；不要页面加载时手动常驻 vLLM 再跑 WD14。

### 7.4 vLLM 空闲占显存 ~40GB+

**原因：** `gpu_memory_utilization` 过高，KV cache 预分配。

**处理：** 降至 0.42，`max_model_len` 保持 4096。

---

## 8. 与 ToriiGate 的区别

| 模型 | 后端 | 说明 |
|------|------|------|
| **ToriiGate 0.5** | 仅 vLLM（端口 18901） | 不受 Gemma CUDA 13 问题影响 |
| **Gemma-4-E4B** | vLLM 或 transformers | 见本文 |

若仅需稳定 VLM 标注、暂不升级实例，可改用 **ToriiGate 0.5** + vLLM，无需 transformers 回退。

---

## 9. 快速决策表

| 你的情况 | 推荐配置 |
|----------|----------|
| AutoDL CUDA 12.8，要稳定出结果 | `"gemma_vlm_backend": "transformers"` |
| 已换 CUDA 13 实例，vLLM 测试通过 | `"gemma_vlm_backend": "vllm"` + `"enable_custom_ops": true` |
| 不确定 vLLM 是否正常 | `"gemma_vlm_backend": "auto"` |
| 要并发、低延迟 | 满足 CUDA 13 后用 vLLM + 调高 `max_num_seqs` |
| 不想折腾 Gemma | VLM 选 ToriiGate 0.5 |

---

## 10. 相关文件索引

| 文件 | 作用 |
|------|------|
| `config/anima_caption_models.json` | VLM 预设、`gemma_vlm_backend`、`vllm_serve` |
| `mikazuki/utils/vllm_manager.py` | 自动启动/停止 vLLM、serve 参数 |
| `tools/anima_caption_pipeline/vlm_client.py` | HTTP 客户端、vLLM 探测、client 工厂 |
| `tools/anima_caption_pipeline/gemma_local_client.py` | transformers 本地 Gemma |
| `tools/anima_caption_pipeline/runner.py` | WD14 → VLM 批处理编排 |
| `scripts/start_gemma_vllm.sh` | 命令行启动 Gemma vLLM |

环境变量：

| 变量 | 含义 |
|------|------|
| `VLLM_ENABLE_CUSTOM_OPS=1` | 启动脚本中不禁用 custom ops（需 CUDA 13） |

---

*文档版本：与 vLLM 0.22.1 + Gemma 4 Anima Train 流水线同步。若升级 vLLM 或更换 PyTorch/CUDA 栈，请重新执行第 4.4 节验证后再切换 `gemma_vlm_backend`。*
