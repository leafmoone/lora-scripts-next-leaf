# 打标模型目录（tagger-models）

WebUI「数据集打标」会**优先**使用项目根目录下 `tagger-models/` 中的本地文件。若目录内文件齐全，**不会**再访问 Hugging Face 下载。

适用于：整合包用户、在线下载失败、或希望手动管理 ONNX 模型的用户。

## 打标工具选型

项目内有两套打标入口，按场景选择：

| 场景 | 推荐入口 | 路径 |
|------|----------|------|
| 快速 WD14 booru 标签，与 Kohya 训练流程集成 | 内置 Kohya Tagger | `/tagger.html` |
| Smart Tag（WD14 + ToriiGate VLM 自然语言） | DiffSynth Tagger (Tagger-leaf) | `/tag-edit-leaf.html` |
| 多标签器共识投票、黑名单、触发词 | DiffSynth Tagger | `/tag-edit-leaf.html` |
| 无本地 GPU，使用 OpenAI/Anthropic API | DiffSynth Tagger · API 模式 | `/tag-edit-leaf.html` |
| Anima 训练标注（WD14 + 两步 VLM + tags/caption 双行） | DiffSynth Tagger · Anima Train | `/tag-edit-leaf.html` |
| 差分 LoRA 训练前自动打标 | Differential LoRA 页面内嵌 | `/lora/differential-lora.html` |

**Smart 模式组件**（`/tag-edit-leaf.html` → Smart Tag 高级选项）：

| 选项 | 说明 |
|------|------|
| WD14 booru 标签 | 本地 ONNX 打标，写入 caption 的 tag 段 |
| ToriiGate VLM | 自然语言描述；模板可选 LoRA 目的模板或官方 short/long/min_structured_md |
| VLM 后端 | **Transformers**（本机加载）或 **vLLM**（远程 OpenAI 兼容 API） |
| 向 VLM 注入 WD14 标签 | 开启：每张图按 WD14 结果构建 VLM prompt（Transformers / vLLM 均支持 batch）；关闭：共享无 tag 模板 |
| 训练目的 / 触发词 | 仅 VLM 区块；目的仅对 LoRA 模板生效，触发词写入最终 caption |

### ToriiGate VLM 后端

| 后端 | 说明 | 适用场景 |
|------|------|----------|
| **transformers**（默认） | 本机 `Qwen3_5ForConditionalGeneration` + `model.generate()`，支持 GPU 真 batch | 单卡 4090、无独立 vLLM 服务 |
| **vllm** | HTTP 调用已启动的 vLLM（`/v1/chat/completions`），并发多请求 | 已有 vLLM 服务、多卡或大 batch 吞吐 |

**vLLM 环境变量**（也可在 UI / CLI 覆盖）：

| 变量 | 默认 |
|------|------|
| `SD_TORIIGATE_VLLM_API_URL` | `http://127.0.0.1:18901/v1/chat/completions` |
| `SD_TORIIGATE_VLLM_MODEL` | `toriigate-0.5` |
| `SD_TORIIGATE_VLLM_API_KEY` | `not-needed` |
| `SD_TORIIGATE_VLLM_MAX_TOKENS` | `2048` |
| `SD_TORIIGATE_VLLM_MAX_PIXELS_MP` | `1.0`（兆像素上限，与官方脚本一致） |

CLI 示例：

```bash
# 先另开终端启动 vLLM（示例，路径按本机调整）
# vllm serve Minthy/ToriiGate-0.5 --port 18901 --served-model-name toriigate-0.5

python tools/differential_tagger/main.py --smart --input /data/img --output /data/out \
  --vlm --vlm-backend vllm --vlm-batch 16 --vllm-api-url http://127.0.0.1:18901/v1/chat/completions
```

### 与官方 `caption_distributed.py` 的差异

Minthy 仓库提供的 **caption_distributed.py**（vLLM 并行打标示例脚本）是 **纯 VLM、纯 vLLM、纯 HTTP 并发** 的参考实现。本项目 Smart 流水线与之对比如下：

| 维度 | 官方 `caption_distributed.py` | 本项目 Smart Tag |
|------|------------------------------|------------------|
| **推理后端** | 仅 vLLM HTTP API | Transformers 本地 **或** vLLM（可选） |
| **WD14 booru 标签** | 无（可选读同名 `.json` grounding） | 有；Phase 1 全量 WD14 batch，标签进 caption |
| **流水线** | 单阶段：每张图 → HTTP caption → 写 `_lsv2_zs.txt` | 两阶段：WD14 全部完成 → VLM（共享 prompt 模板）→ 组装 caption |
| **并行方式** | `ThreadPoolExecutor(NUM_WORKERS=16)`，每图独立 HTTP 请求 | Transformers：`processor` 多图真 batch；vLLM：`vlm-batch` 并发 HTTP（同官方思路） |
| **Prompt** | `prompts.py` 的 `C_TYPE`（如 `long_thoughts_v2`）+ JSON 元数据 | LoRA 目的模板 / ToriiGate 官方 short·long·min_structured_md |
| **输出** | 仅 VLM 原文 `.txt` | WD14 tags + NL + 触发词 合并 caption |
| **与 WD14 抢 GPU** | 不涉及（假定 vLLM 独占 GPU） | Transformers 模式会释放 WD14 再 load VLM；vLLM 模式 WD14 可与远程 vLLM 共存 |

**何时用哪种：**

- 只要大批量纯 caption、且已跑 vLLM → 官方脚本或本项目的 `--vlm-backend vllm`。
- 要 LoRA 训练 caption（tag + 自然语言 + 触发词）→ Smart Tag；vLLM 仅替换 VLM 阶段。

**简要建议**：只需 Danbooru 风格 tag → 用 `/tagger.html`；需要自然语言 caption 或 VLM → 用 `/tag-edit-leaf.html`。

### Anima Train 模式

`/tag-edit-leaf.html` → 标注模式 **Anima Train**：WD14 batch → 两步 VLM（`extract_tags_from_image` → `generate_natural_caption`）→ `anima_train_v1` 输出（`tags 行` + 空行 + `caption 行`）。训练 tags 行优先使用 **WD14 原始标签**，不添加质量词/负面词。

| 项 | 值 |
|----|-----|
| 流水线代码 | `tools/anima_caption_pipeline/` |
| VLM 后端 | **Gemma**：vLLM 或 transformers（见 [anima-gemma-vllm.md](./anima-gemma-vllm.md)）；**ToriiGate**：vLLM |
| ToriiGate | 默认 `http://127.0.0.1:18901/v1/chat/completions`，served name `toriigate-0.5` |
| Gemma-4-E4B | ModelScope `spawner/spawner-gemma-4-E4B-it`，本地目录 `./models/gemma-4-E3B-it`，端口 9002 |

> **Gemma vLLM / transformers 详细说明**（CUDA 限制、配置项、故障排查）：[docs/anima-gemma-vllm.md](./anima-gemma-vllm.md)

Gemma 权重下载（项目根目录下执行，Anima Train 选 Gemma 且本地目录无效时也会自动触发）：

```bash
pip install modelscope
modelscope download spawner/spawner-gemma-4-E4B-it \
  --local_dir ./models/gemma-4-E3B-it
```

启动 Gemma vLLM（或使用 `scripts/start_gemma_vllm.sh`）：

```bash
vllm serve ./models/gemma-4-E3B-it \
  --served-model-name spawner-gemma-4-e4b-it \
  --port 9002 \
  --limit-mm-per-prompt image=1 \
  --max-model-len 8192 \
  --trust-remote-code
```

同卡部署时 Runner 先完成 WD14 并释放 ONNX，再并发调用 vLLM；可通过 UI 调低 `VLM 并发数` 避免与 Gemma 抢显存。

角色别名（可选）：`tools/anima_caption_pipeline/resources/danbooru_character_aliases.json`；大表可执行 `python scripts/build_alias_index.py` 生成 SQLite。

## 目录结构

```text
<项目根或整合包根>/
  tagger-models/
    wd14/
      wd14-convnextv2-v2/          ← 默认 WD 模型
        model.onnx
        selected_tags.csv
      wd14-convnext-tagger-v3/     ← 其它 WD 模型（示例）
        model.onnx
        selected_tags.csv
    vlm/
      <model-key>/                 ← 预留：VLM / 自然语言描述模型
```

- **WD14 / CL 系列**：放在 `tagger-models/wd14/<model-key>/`
- **VLM 系列**：放在 `tagger-models/vlm/<model-key>/`
- **兼容旧布局**：`tagger-models/<model-key>/`（一层目录）仍可识别

`<model-key>` 与 WebUI 打标页下拉框中的模型 ID 一致（如 `wd14-convnextv2-v2`）。

## 默认模型

| 项 | 值 |
|---|---|
| 模型 ID | `wd14-convnextv2-v2` |
| Hugging Face | `SmilingWolf/wd-v1-4-convnextv2-tagger-v2`（revision `v2.0`） |
| 本地路径 | `tagger-models/wd14/wd14-convnextv2-v2/` |
| 必需文件 | `model.onnx`、`selected_tags.csv` |

**整合包**：新版 7z 会内置上述目录；一般无需再下载。

**源码安装**：`install-cn.ps1` 或 `python scripts/prefetch_default_tagger.py` 会尝试写入 `tagger-models/`；每次 `run_gui.bat` 启动前若缺失也会自动补全。

## 手动放置模型（下载失败时）

1. 从 Hugging Face（或镜像）下载对应模型的 `model.onnx` 与标签 CSV（通常为 `selected_tags.csv`）。
2. 在项目根（整合包为 `run_gui.bat` 同级）创建目录，例如：
   ```text
   tagger-models/wd14/wd14-convnextv2-v2/
   ```
3. 将两个文件放入该目录，**文件名需与 interrogator 要求一致**（默认 WD 为上述两个文件名）。
4. 重启 WebUI 或刷新打标页，选择对应模型后直接「启动」——不应再触发下载。

若文件不完整，程序会回退到 `huggingface/hub/` 缓存或在线下载（可在打标页选择镜像源）。

## 自定义根目录（高级）

设置环境变量 `MIKAZUKI_TAGGER_MODELS_DIR` 指向其它绝对路径，可覆盖默认的 `<项目根>/tagger-models`。

## 一直停在「加载模型」、无进度（排障）

日志若只有 `Loading … from local tagger-models directory` 而无后续 `Loaded …`：

1. 检查 `model.onnx` 体积约 **370–400 MB**（仅几 MB 多为下载不完整，会校验失败并提示重下）。
2. Windows 便携包首次用 CUDA 加载 ONNX 可能挂死，启动前可强制 CPU：
   ```bat
   set MIKAZUKI_TAGGER_ORT_PROVIDERS=cpu
   python gui.py
   ```
3. 可选：`MIKAZUKI_TAGGER_ORT_LOAD_TIMEOUT=120`（秒）避免无限等待。
4. 打标页底部应有进度 dock；若无，确认 `frontend/dist/tagger.html` 已引入 `tagger-progress.js`（见 Issue #78）。

## 相关文档

- 打标 API 与进度：[docs/api/dataset-tagging.md](api/dataset-tagging.md)
- 整合包目录契约：[docs/portable-packaging-git-update.md](portable-packaging-git-update.md)
- 便携包说明：[scripts/portable/README.md](../scripts/portable/README.md)
