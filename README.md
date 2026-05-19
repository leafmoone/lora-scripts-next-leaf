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

## Quick Start

### Windows Portable Package (recommended for beginners)

Download **`SD-Trainer-v2.2.0.7z`** (~15 MB) from [Releases](https://github.com/wochenlong/lora-scripts-next/releases), extract, and double-click `run_gui.bat`.

First launch auto-installs PyTorch + CUDA + all dependencies (~3 GB download). Chinese users get mirror acceleration automatically.

| File | Purpose |
|------|---------|
| `run_gui.bat` | Launch training GUI (http://127.0.0.1:28000) |
| `Update-SD-Trainer.bat` | Pull latest code from GitHub |
| `Download-Anima-Model.bat` | Download Anima base model from ModelScope |

> **Requirements:** Windows 10/11 64-bit, NVIDIA GPU (RTX 20+), ~7 GB disk space.

### Install from Source

```sh
git clone https://github.com/wochenlong/lora-scripts-next.git
cd lora-scripts-next
```

| OS | Action |
|----|--------|
| Windows | Double-click **`run_gui.bat`** (auto-installs on first run, then launches) |
| Linux | `bash install.bash && bash run_gui.sh` |

The browser auto-opens **http://127.0.0.1:28000** on launch.

> **Python version:** 3.10 recommended (full compatibility). 3.11–3.12 mostly works. 3.13+ is not supported.

#### Choose Browser

By default the system default browser opens. Use `--browser` to pick one:

```sh
python gui.py --browser chrome
python gui.py --browser edge
```

#### Flash Attention 2 (existing installs)

New installs get Flash Attention 2 automatically via prebuilt wheels. If you already have an environment and `run_gui.bat` didn't install it, grab the matching wheel manually:

```sh
# Python 3.10 + PyTorch 2.7 + CUDA 12.8 (portable package default)
pip install https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl

# Linux / has C++ compiler — build from source also works
pip install flash-attn --no-build-isolation
```

---

## Features

- **Multi-model** — SD 1.5 / SDXL / Flux / **Anima** all work out of the box
- **Anima LoRA training** — One-click sidebar entry, supports LoRA / LoKr (LyCORIS) / **T-LoRA**
- **Flash Attention 2 acceleration** — Auto-detected and enabled when available; falls back to xformers or PyTorch SDPA. Windows uses prebuilt wheels (no C++ compiler needed); portable package installs automatically on first run
- **T-LoRA** — Timestep-Dependent LoRA with dynamic rank and orthogonal init ([paper](https://github.com/ControlGenAI/T-LoRA))
- **Train Monitor** — Auto-opens with GUI, interactive ECharts Loss chart (zoom / pan / restore), real-time progress and preview samples
- **Built-in TensorBoard** — Accessible from the sidebar, no extra setup
- **GPU detection** — Detects NVIDIA / AMD GPUs on first run; AMD users get a friendly notice with ROCm guidance
- **AutoDL ready** — Dedicated startup script `start_autodl.sh`

---

## Interface Preview

<p align="center">
  <img src="assets/readme/screenshot-webui.png" alt="Training WebUI" width="920" />
</p>

<p align="center">
  <img src="assets/readme/shot-train-monitor.png" alt="Train Monitor" width="920" />
</p>

<p align="center"><sub>Top: Training GUI &nbsp;|&nbsp; Bottom: Train Monitor (port 6008, auto-opens)</sub></p>

---

## Documentation

| Topic | Link |
|-------|------|
| Anima LoRA Training Guide | [docs/anima-training.md](docs/anima-training.md) |
| Train Monitor & SSE API | [docs/train-monitor.md](docs/train-monitor.md) |
| Frontend Customization | [docs/frontend-customization.md](docs/frontend-customization.md) |
| Docker Deployment | [docs/docker.md](docs/docker.md) |
| CLI Arguments | [docs/cli-args.md](docs/cli-args.md) |

---

<details>
<summary><b>Changelog</b></summary>

| Date | Update |
|------|--------|
| 2026-05-19 | **v2.2.0** — Portable package startup fixes (triton-windows), run_gui.bat crash logging, cross-drive train monitor, branding/logo update, CONTRIBUTORS.md |
| 2026-05-19 | **v2.1.0** — Flash Attention 2 prebuilt wheels for Windows (no C++ compiler needed), save-by-steps option, fix LoKr conv_dim/conv_alpha undefined bug |
| 2026-05-18 | **v2.0.0** — Portable package, Flash Attention 2 auto-acceleration, AMD GPU detection, auto bf16/fp16 fix, `--browser chrome/edge`, vendor sd-scripts, update check |
| 2026-05-18 | T-LoRA support, interactive Loss chart, LoKr standardization, Windows portable package, AutoDL script |
| 2026-05-17 | Anima training backend fully migrated to kohya-ss/sd-scripts |
| 2026-05-06 | Train monitor rebuild: real-time Loss cards + sticky scroll |

</details>

<details>
<summary><b>Credits & Upstream</b></summary>

| Project | Role |
|---------|------|
| [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) | GUI framework & one-click training UX |
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | Core training backend |
| [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) | LoKr / LoHa network modules (Apache-2.0) |
| [ControlGenAI/T-LoRA](https://github.com/ControlGenAI/T-LoRA) | Timestep-Dependent LoRA (MIT, AIRI) |
| [bluvoll/Akegarasu-lora-scripts-RF](https://github.com/bluvoll/Akegarasu-lora-scripts-RF) | SDXL Rectified Flow reference |

Full attribution in [`NOTICE.md`](NOTICE.md).

</details>

---

## Contributors

See [**CONTRIBUTORS.md**](CONTRIBUTORS.md) for the full list of contributors and upstream credits.

---

<p align="center"><sub>Maintainer: <b>@wochenlong</b></sub></p>
