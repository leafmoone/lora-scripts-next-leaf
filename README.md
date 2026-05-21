<p align="center">
  <img src="assets/readme/anima-cover.png" alt="lora-scripts-next · Anima Trainer" width="880" />
</p>

<h1 align="center">lora-scripts-next</h1>

<p align="center">
  <strong>SD-Trainer</strong> — One-click LoRA training GUI for SD / SDXL / Flux / <b>Anima</b><br/>
  <sub>Powered by <a href="https://github.com/kohya-ss/sd-scripts">kohya-ss/sd-scripts</a>, with the familiar Akegarasu GUI experience.</sub>
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next"><img src="https://img.shields.io/github/stars/wochenlong/lora-scripts-next?style=flat-square&label=stars&logo=github&color=8b5cf6" alt="stars"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next"><img src="https://img.shields.io/github/forks/wochenlong/lora-scripts-next?style=flat-square&label=forks&color=06b6d4" alt="forks"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/LICENSE"><img src="https://img.shields.io/github/license/wochenlong/lora-scripts-next?style=flat-square&color=ec4899" alt="license"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next/releases"><img src="https://img.shields.io/github/v/release/wochenlong/lora-scripts-next?include_prereleases&style=flat-square&color=a78bfa" alt="release"/></a>
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next/releases"><b>Download</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/README-zh.md"><b>中文文档</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/NOTICE.md"><b>Credits & License</b></a>
</p>

---

## Features

- **Multi-model** — SD 1.5 / SDXL / Flux / **Anima** out of the box
- **Anima LoRA training** — One-click sidebar, supports LoRA / LoKr (LyCORIS) / **T-LoRA**
- **Attention backends** — Source/venv: Flash Attention 2 when available; Portable: xformers / PyTorch SDPA ([details](docs/flash-attention.md))
- **T-LoRA** — Timestep-Dependent dynamic-rank LoRA with orthogonal init ([paper](https://github.com/ControlGenAI/T-LoRA))
- **Train Monitor** — TensorBoard-backed Loss/LR cards, key parameter checks, real-time progress & preview samples
- **Built-in TensorBoard** — Accessible from the sidebar
- **GPU detection** — NVIDIA / AMD detection on first run; AMD users get ROCm guidance
- **AutoDL ready** — `start_autodl.sh` startup script

---

## Quick Start

### Windows Portable Package (recommended for beginners)

Download **`SD-Trainer-v2.4.0.7z`** (~21 MB) from [Releases](https://github.com/wochenlong/lora-scripts-next/releases), extract, and double-click `run_gui.bat`.

First launch auto-installs PyTorch + CUDA + all dependencies (~3 GB). Chinese users get mirror acceleration.

| File | Purpose |
|------|---------|
| `run_gui.bat` | Entrypoint |
| `Update-SD-Trainer.bat` | Pull latest code |
| `Download-Anima-Model.bat` | Download Anima base model |

> **Requirements:** Windows 10/11 64-bit, NVIDIA GPU (RTX 20+), ~7 GB disk.

> **Flash Attention 2:** Not supported in portable package — training uses xformers / SDPA. See [docs/flash-attention.md](docs/flash-attention.md).

#### Anima LoRA VRAM Reference (1024 resolution, RTX 4090 benchmarked)

| VRAM | Configuration | Notes |
|------|---------------|-------|
| ≥ 24 GB | Default settings | Easiest |
| ≥ 16 GB | `gradient_checkpointing` | Recommended |
| ≥ 12 GB | Gradient checkpointing | Stable |
| ≥ 10 GB | Gradient checkpointing + `blocks_to_swap=16` | Slightly slower |
| ≥ 8 GB | Gradient checkpointing + swap 24 + cache TE + LoKr | Tight |

### Install from Source

```sh
git clone https://github.com/wochenlong/lora-scripts-next.git
cd lora-scripts-next
```

| OS | Action |
|----|--------|
| Windows | Double-click `run_gui.bat` |
| Linux | `bash install.bash && bash run_gui.sh` |

Browser auto-opens **http://127.0.0.1:28000**. Python **3.10** recommended (3.11–3.12 mostly OK, 3.13+ unsupported).

Pick browser: `python gui.py --browser chrome`

> **Flash Attention 2 (source users):** See [docs/flash-attention.md](docs/flash-attention.md).

---

## Interface Preview

<p align="center">
  <img src="assets/readme/train-monitor-loss.png" alt="Train Monitor Loss Curves" width="920" />
</p>

<p align="center"><sub>TensorBoard-backed Loss / LR scalar cards</sub></p>

<p align="center">
  <img src="assets/readme/train-monitor-samples.png" alt="Train Monitor Preview Samples" width="920" />
</p>

<p align="center"><sub>Preview samples update in the monitor page</sub></p>

<p align="center">
  <img src="assets/readme/train-monitor-logs.png" alt="Train Monitor Log Viewer" width="920" />
</p>

<p align="center"><sub>Logs shown in both CMD and monitor page</sub></p>

---

## Documentation

| Topic | Link |
|-------|------|
| Anima LoRA Training Guide | [docs/anima-training.md](docs/anima-training.md) |
| Flash Attention 2 Setup | [docs/flash-attention.md](docs/flash-attention.md) |
| Train Monitor & SSE API | [docs/train-monitor.md](docs/train-monitor.md) |
| Frontend Customization | [docs/frontend-customization.md](docs/frontend-customization.md) |
| Docker Deployment | [docs/docker.md](docs/docker.md) |
| CLI Arguments | [docs/cli-args.md](docs/cli-args.md) |

---

<details>
<summary><b>Changelog</b></summary>

| Date | Update |
|------|--------|
| 2026-05-21 | **v2.4.0** — Training stability: env isolation, NaN filter, sample_prompts guard, attn_mode fallback, path `\` → `/` normalization; Portable: tkinter, `install_xformers.bat` |
| 2026-05-20 | **v2.3.0** — Train Monitor: TensorBoard-backed curves, parameter checks, port fallback, log echo |
| 2026-05-19 | **v2.2.0** — Portable flash-attn fix, run_gui.bat policy + crash logging, cross-drive monitor, branding |
| 2026-05-19 | **v2.1.0** — Flash Attention 2 prebuilt wheels, save-by-steps, LoKr undefined fix |
| 2026-05-18 | **v2.0.0** — Portable package, Flash Attention 2, AMD detection, bf16 fix, `--browser`, vendor sd-scripts |
| 2026-05-18 | T-LoRA, interactive Loss chart, LoKr standardization, Windows portable, AutoDL |
| 2026-05-17 | Anima backend migrated to kohya-ss/sd-scripts |

Full details in [CHANGELOG.md](CHANGELOG.md).

</details>

<details>
<summary><b>Credits & Upstream</b></summary>

| Project | Role |
|---------|------|
| [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) | GUI framework |
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | Training backend |
| [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) | LoKr / LoHa |
| [ControlGenAI/T-LoRA](https://github.com/ControlGenAI/T-LoRA) | Timestep LoRA |

Full attribution in [`NOTICE.md`](NOTICE.md).

</details>

---

## Contributors

See [**CONTRIBUTORS.md**](CONTRIBUTORS.md).

---

<p align="center"><sub>Maintainer: <b>@wochenlong</b></sub></p>
