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

Download the portable package from [Releases](https://github.com/wochenlong/lora-scripts-next/releases), extract, and double-click `run_gui.bat`.

### Install from Source

```sh
git clone --recurse-submodules https://github.com/wochenlong/lora-scripts-next.git
cd lora-scripts-next
```

| OS | Action |
|----|--------|
| Windows | Double-click **`run_gui.bat`** (auto-installs on first run, then launches) |
| Linux | `bash install.bash && bash run_gui.sh` |

The browser auto-opens **http://127.0.0.1:28000** on launch.

> **Python version:** 3.10 recommended (full compatibility). 3.11–3.12 mostly works. 3.13+ is not supported.

---

## Features

- **Multi-model** — SD 1.5 / SDXL / Flux / **Anima** all work out of the box
- **Anima LoRA training** — One-click sidebar entry, supports LoRA / LoKr (LyCORIS)
- **Train Monitor** — Auto-opens with GUI, real-time Loss, progress, and preview samples
- **Built-in TensorBoard** — Accessible from the sidebar, no extra setup
- **AutoDL ready** — Dedicated startup script `start_autodl.sh`

---

## Interface Preview

<p align="center">
  <img src="assets/readme/screenshot-webui.png" alt="Training WebUI" width="920" />
</p>

<p align="center">
  <img src="assets/readme/screenshot-train-monitor.png" alt="Train Monitor" width="920" />
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
| 2026-05-18 | Anima LoKr training standardized (lycoris.kohya backend) |
| 2026-05-18 | Train monitor auto-starts with GUI; added AutoDL startup script |
| 2026-05-18 | One-click Windows portable package build scripts |
| 2026-05-17 | Anima training backend fully migrated to kohya-ss/sd-scripts |
| 2026-05-06 | Train monitor rebuild: real-time Loss cards + sticky scroll |

</details>

<details>
<summary><b>Credits & Upstream</b></summary>

| Project | Role |
|---------|------|
| [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) | GUI framework & one-click training UX |
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | Core training backend |
| [bluvoll/Akegarasu-lora-scripts-RF](https://github.com/bluvoll/Akegarasu-lora-scripts-RF) | SDXL Rectified Flow reference |

Full attribution in [`NOTICE.md`](NOTICE.md).

</details>

---

<p align="center"><sub>Maintainer: <b>@wochenlong</b></sub></p>
