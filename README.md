

# Next Trainer

**One-click LoRA & full finetune training GUI for Windows** — supports **Anima** / SD 1.5 / SDXL / Flux  
Extract and run. No environment setup needed. ~12 GB VRAM for Anima LoRA; **Anima full finetune needs ~24 GB**.  
Powered by [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), Akegarasu-style GUI.





**[中文](https://github.com/wochenlong/lora-scripts-next/blob/main/README-zh.md)**

**[Credits](https://github.com/wochenlong/lora-scripts-next/blob/main/NOTICE.md)**

**[Experimental: Anima Edit branch](https://github.com/wochenlong/lora-scripts-next/tree/anima-edit)**

---



Home portal — quick links to training, monitor, and onboarding

---
没招了，这里是2026年6月9日下午六点，虽然我一直知道有佬在做角色参考，但他说ipa不行，我又是从ipa开始让ai参考一点点做的，所以以为是互补甚至不相干，了解都没了解，结果刚刚开源了我刚让ai说明下项目我就感觉不对劲，一对发现基本思路没啥区别，我人傻了，世界上最让人伤心的是，好不容易做出来了个东西，验证跑到一半，别人做完了
## What's New in This Fork

This fork adds **multi-encoder IP-Adapter training**, a **resampler pipeline**, **Differential LoRA**, a **DiffSynth smart tagger**, and a **parallel tagger backend** on top of the upstream v2.7.0 codebase. All features are accessible from the WebUI homepage or via direct URLs.

### 🖼️ Anima IP-Adapter (Multi-Encoder)

Train an IP-Adapter that injects image conditions into Anima DiT cross-attention. Supports **CLIP** (content) + **CCIP** (character identity) + **LSNet** (artist style) as parallel encoder streams with per-stream token control.

- **Page**: Homepage card → IP-Adapter, or `/lora/anima-ipa.html`
- **CLI**: `accelerate launch ip_adapter/anima_ip_train.py --aux_encoders clip_ccip_lsnet ...`
- **Design doc**: `[cursor_docs/ipa-lsnet-design.md](cursor_docs/ipa-lsnet-design.md)`

### 🎯 Three Projection Modes


| Mode          | Mechanism                            | Best for                                      |
| ------------- | ------------------------------------ | --------------------------------------------- |
| **Simple**    | Global embedding → `ImageProjModel`  | Fast, lightweight, general-purpose            |
| **Resampler** | Patch features → Perceiver Resampler | Fine detail (face texture, clothing patterns) |
| **Double**    | Simple + Resampler in parallel       | Maximum quality                               |


Each encoder stream gets its **own token count** (e.g. `--num_ip_tokens_clip=8 --num_ip_tokens_ccip=4`).

### 🖌️ Differential LoRA Training

Character-splitting LoRA that trains a "difference" between two character folders. Two-stage Kohya training:

- **Page**: Homepage card → Differential LoRA, or `/lora/differential-lora.html`
- **Tools**: `tools/merge_lora_to_base.py`, `tools/average_lora.py`, `tools/convert_differential_to_comfyui.py`

### 🏷️ DiffSynth Smart Tagger (Tag-Edit-Leaf)

Two-mode AI image tagger with WD14 booru tags (*simple*) and ToriiGate VLM natural-language captions (*smart*). Supports multi-tagger consensus voting and blacklist filtering.

- **Page**: Homepage card → DiffSynth Tagger, or `/tag-edit-leaf.html`
- **API**: `POST /api/tag-edit-leaf/scan`, `POST /api/tag-edit-leaf/run`
- **Parallelism**: Set `--wd14-batch 8 --vlm-workers 2` for WD14 batched inference + VLM pipeline
- **Design doc**: `[cursor_docs/ipa-lsnet-review.md](cursor_docs/ipa-lsnet-review.md)`

### 🔗 Multi-Stream vs Auxiliary Encoders


| Encoder   | Purpose                                     | Model         | When to enable          |
| --------- | ------------------------------------------- | ------------- | ----------------------- |
| **CLIP**  | Visual content (pose, composition, objects) | ViT-L/14      | Always (required)       |
| **CCIP**  | Character identity (who)                    | CaFormer 96M  | Character training      |
| **LSNet** | Artist style (how it looks)                 | LSNet-XL 102M | Style-transfer training |


CCIP and LSNet are complementary—they provide orthogonal signals that the IP-Adapter learns to fuse alongside CLIP.

### ⚙️ Install & Run (Linux)

```bash
# 1. Clone 本仓库
git clone https://github.com/leafmoone/lora-scripts-next-leaf.git
cd lora-scripts-next-leaf

# 2. 使用 uv 创建环境并安装依赖（推荐 Python 3.12）
uv venv --python 3.12
source .venv/bin/activate
uv sync

uv pip install dghs-imgutils timm safetensors torchvision modelscope

mkdir -p models
pip install -U huggingface_hub
hf download deepghs/ccip --include "ccip-caformer_b36-24.ckpt" --local-dir ./models/ccip
mkdir -p models/lsnet
modelscope download Heathcliff02/Kaloscope-2.0 --include "best_checkpoint.pth" --local_dir ./models/lsnet


uv python gui.py

---

## Get Started in 3 Steps

```

1. Download  →  SD-Trainer-v2.7.0.7z from [Releases](https://github.com/wochenlong/lora-scripts-next/releases), extract
2. Launch    →  Double-click run_gui.bat (auto-installs deps on first run, ~3 GB)
3. Train     →  Open [http://127.0.0.1:28000](http://127.0.0.1:28000), pick a model, set params, start training

```

The portable package ships the default WD tagger **wd14-convnextv2-v2** under **`tagger-models/wd14/`** (~400 MB). If Hugging Face download fails, place `model.onnx` and `selected_tags.csv` there manually — see `[docs/tagger-models.md](docs/tagger-models.md)`.

> **CLI / cloud training:** `train.sh` is the legacy SD/SDXL/Flux entry. For Anima use the dedicated scripts:
> `bash train_anima_by_toml.sh docs/examples/anima-lora-benchmark-kohya.toml` (standard, non-Fast) or
> `bash train_anima_fast_by_toml.sh docs/examples/anima-lora-benchmark-fast.toml` (Fast plugin; run `bash scripts/cli/install_anima_fast.sh` first).

> **Requirements:** Windows 10/11, NVIDIA GPU (RTX 20+), ~7 GB disk.

<details>
<summary><b>Install from source (Linux / advanced users)</b></summary>

```sh
git clone https://github.com/wochenlong/lora-scripts-next.git
cd lora-scripts-next

# Windows
run_gui.bat

# Linux
bash install.bash && bash run_gui.sh

# Optional: install Flash Attention 2 for faster Anima training
# Windows
install_flash_attn.bat
# Linux
bash install_flash_attn.sh
```

Python **3.10** recommended. See [Flash Attention 2 docs](docs/flash-attention.md) for details.



---

## What's Supported


| Mode                   | Model / script                    | Notes                                                                                                                             |
| ---------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Anima LoRA**         | LoRA · LoKr · **T-LoRA**          | Flash Attention 2 / xformers / SDPA · from ~12 GB VRAM                                                                            |
| **Anima Edit**         | Image editing LoRA (experimental) | Maintained on the `[anima-edit](https://github.com/wochenlong/lora-scripts-next/tree/anima-edit)` branch                          |
| **Anima LoRA Fast**    | LoRA only (plugin)                | Optional [anima_lora](https://github.com/sorryhyun/anima_lora) runtime · ~16 GB+ · see `[docs/anima-fast.md](docs/anima-fast.md)` |
| **Anima Finetune**     | Full DiT (`anima_train.py`)       | Sidebar **全量微调 → Anima Finetune** · **~24 GB VRAM** (4090-class)                                                                  |
| SD 1.5 / SDXL LoRA     | LoRA · LoHa · LoKr                | xformers / SDPA                                                                                                                   |
| SD 1.5 / SDXL Finetune | Dreambooth / SDXL finetune        | Sidebar **全量微调 → Stable Diffusion**                                                                                               |
| Flux                   | LoRA                              | xformers / SDPA                                                                                                                   |




Anima LoRA — sidebar, model & dataset form, config preview on the right



Anima LoRA Fast — optional plugin path under **标准模式 / Fast 模式**; install runtime from the page before training



Anima Finetune — full DiT weights under **全量微调** in the sidebar

---

## Train Monitor

Automatically opens a monitor page (port 6008) when training starts — GPU stats, training parameters, Loss curves, preview samples, and logs all in one dashboard.



GPU load & VRAM, total steps, training params at a glance



Preview samples and TensorBoard-backed Loss / LR curves



Real-time training logs with auto-scroll

---

**VRAM Reference (Anima, 1024 resolution, RTX 4090 benchmarked)**

**Anima LoRA**


| VRAM    | Configuration                                      | Notes           |
| ------- | -------------------------------------------------- | --------------- |
| ≥ 24 GB | Default settings                                   | Easiest         |
| ≥ 16 GB | `gradient_checkpointing`                           | Recommended     |
| ≥ 12 GB | Gradient checkpointing                             | Stable          |
| ≥ 10 GB | Gradient checkpointing + `blocks_to_swap=16`       | Slightly slower |
| ≥ 8 GB  | Gradient checkpointing + swap 24 + cache TE + LoKr | Tight           |


**Anima full finetune** (updates full DiT weights — use **Anima Finetune** in the WebUI, not LoRA)


| VRAM    | Configuration              | Notes                                                            |
| ------- | -------------------------- | ---------------------------------------------------------------- |
| ≥ 24 GB | Default + latents/TE cache | **~23–24 GB dedicated VRAM** in practice; 4090-class recommended |




**Documentation**


| Topic                                     | Link                                                                             |
| ----------------------------------------- | -------------------------------------------------------------------------------- |
| Anima LoRA Training Guide                 | [docs/anima-training.md](docs/anima-training.md)                                 |
| **Anima Fast Mode (optional plugin)**     | [docs/anima-fast.md](docs/anima-fast.md)                                         |
| Open-source notices                       | [NOTICE.md](NOTICE.md)                                                           |
| Anima backend (LoRA + full finetune)      | [docs/anima-backend.md](docs/anima-backend.md)                                   |
| Anima full finetune example TOML          | [docs/examples/anima-full-finetune.toml](docs/examples/anima-full-finetune.toml) |
| Flash Attention 2                         | [docs/flash-attention.md](docs/flash-attention.md)                               |
| Train Monitor & SSE API                   | [docs/train-monitor.md](docs/train-monitor.md)                                   |
| Tagger model directory (`tagger-models/`) | [docs/tagger-models.md](docs/tagger-models.md)                                   |
| Docker Deployment                         | [docs/docker.md](docs/docker.md)                                                 |
| CLI Arguments                             | [docs/cli-args.md](docs/cli-args.md)                                             |




**Changelog**


| Date       | Version                                                                                                                                                                                   |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-05-28 | **v2.7.0** — **Anima LoRA Fast mode** (optional `anima_lora` plugin): WebUI entry, one-click install, train monitor sync, benchmark docs · see `[docs/anima-fast.md](docs/anima-fast.md)` |
| 2026-05-28 | **v2.6.0** — **Anima full finetune** WebUI (`anima-finetune`), `anima_train.py` wrapper, 全量微调 nav, train monitor label fix; ~24 GB VRAM reference                                         |
| 2026-05-27 | **v2.5.3** — Portable dependency health check, sidebar version chip ([#54](https://github.com/wochenlong/lora-scripts-next/issues/54))                                                    |
| 2026-05-21 | **v2.5.0** — UI refresh: new sidebar navigation, home portal page, training monitor dashboard with GPU metrics; CSS cleanup                                                               |
| 2026-05-21 | **v2.4.0** — Training stability: env isolation, NaN filter, sample guard, attn_mode fallback, path normalization; Portable tkinter fix                                                    |
| 2026-05-20 | **v2.3.0** — Train Monitor: TensorBoard-backed curves, parameter checks, log sync                                                                                                         |
| 2026-05-19 | **v2.2.0** — Portable flash-attn fix, crash logging, cross-drive monitor                                                                                                                  |
| 2026-05-19 | **v2.1.0** — Flash Attention 2 prebuilt wheels, save-by-steps                                                                                                                             |
| 2026-05-18 | **v2.0.0** — First portable release, AMD detection, bf16 fix                                                                                                                              |


Full details in [CHANGELOG.md](CHANGELOG.md).



**Credits**

[Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) · [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) · [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) · [T-LoRA](https://github.com/ControlGenAI/T-LoRA) — Full attribution in [NOTICE.md](NOTICE.md)



---

Maintainer: **[@wochenlong](https://github.com/wochenlong)** · [Contributors](CONTRIBUTORS.md)