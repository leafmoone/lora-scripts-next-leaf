<p align="center">
  <img src="assets/readme/logo.svg" alt="lora-scripts-next" width="140" height="140" />
</p>

<h1 align="center">lora-scripts-next</h1>

<p align="center">
  <strong>SD-Trainer</strong> — LoRA · Dreambooth · one-click training shell around <a href="https://github.com/kohya-ss/sd-scripts">kohya-ss/sd-scripts</a><br/>
  <sub><em>A personal fork: ship faster experiments, keep the familiar GUI.</em></sub>
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next"><img src="https://img.shields.io/github/stars/wochenlong/lora-scripts-next?style=flat-square&label=stars&logo=github&color=8b5cf6" alt="stars"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next"><img src="https://img.shields.io/github/forks/wochenlong/lora-scripts-next?style=flat-square&label=forks&color=06b6d4" alt="forks"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/LICENSE"><img src="https://img.shields.io/github/license/wochenlong/lora-scripts-next?style=flat-square&color=ec4899" alt="license"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next/releases"><img src="https://img.shields.io/github/v/release/wochenlong/lora-scripts-next?include_prereleases&style=flat-square&color=a78bfa" alt="release"/></a>
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next/releases"><b>Releases</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/README-zh.md"><b>中文 README</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/NOTICE.md"><b>NOTICE</b></a>
</p>

---

<p align="center">
  <sub>Maintainer: <b>@wochenlong</b> — this repo is where I wire <b>Anima</b>, <b>Rectified Flow</b>, and my own training habits into the classic “秋叶式” stack.</sub>
</p>

<br/>

## At a glance

| | |
|:---|:---|
| **Train WebUI** | Single dashboard: presets, tensorboard hook, tagger, tag editor — open **`http://127.0.0.1:28000`** after `run_gui.ps1` / `run_gui.sh`. |
| **Back end** | [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts); SDXL RF ideas from [bluvoll/Akegarasu-lora-scripts-RF](https://github.com/bluvoll/Akegarasu-lora-scripts-RF); Anima path from [WhitecrowAurora/lora-rescripts](https://github.com/WhitecrowAurora/lora-rescripts) (**SD-reScripts**). |
| **Docs** | Full attribution & licenses in [`NOTICE.md`](NOTICE.md). |

---

## Interface preview

<p align="center">
  <img src="assets/readme/screenshot-webui.png" alt="SD-Trainer WebUI screenshot" width="920" />
</p>

<p align="center"><sub>TensorBoard, WD 1.4 Tagger, and Tag Editor open inside the same WebUI.</sub></p>

---

<details>
<summary><b>Lineage & upstream (click to expand)</b></summary>

This fork lives at **[wochenlong/lora-scripts-next](https://github.com/wochenlong/lora-scripts-next)** and inherits the **Akegarasu SD-Trainer / 秋叶一键训练包** UX from **[Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts)**. Training scripts come from **[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)**. SDXL **Rectified Flow** follows **[bluvoll/Akegarasu-lora-scripts-RF](https://github.com/bluvoll/Akegarasu-lora-scripts-RF)**. **Anima** support is adapted from **[WhitecrowAurora/lora-rescripts](https://github.com/WhitecrowAurora/lora-rescripts)** — **SD-reScripts**, described upstream as a maintained fork / continuation of the LoRA-scripts line.

</details>

---

# Usage

### Dependencies

Python **3.10** and **Git**.

### Clone (with submodules)

```sh
git clone --recurse-submodules https://github.com/wochenlong/lora-scripts-next.git
cd lora-scripts-next
```

## SD-Trainer GUI

### Windows

**Install:** run `install.ps1` (mainland China: `install-cn.ps1`).  
**Train:** run `run_gui.ps1` → opens **[http://127.0.0.1:28000](http://127.0.0.1:28000)**.

### Linux

**Install:** `install.bash`  
**Train:** `bash run_gui.sh` → same URL as above.

### Frontend Static Files

The training GUI backend serves `frontend/dist` by default. The current `frontend` directory is a prebuilt static-file submodule, not the frontend source tree. If that submodule points to an older dist build, the running GUI will show that older UI, including pages such as SD3.

To use a custom or next UI build, build the frontend source first and point the backend at the generated `dist` directory:

```bash
MIKAZUKI_FRONTEND_DIST=/path/to/frontend/dist python gui.py --listen
```

Alternatively, update the `frontend` submodule to the dist repository/commit that contains the desired UI. The backend does not build frontend source automatically.

## Legacy: script-only training

### Windows

Install with `install.ps1`, then edit and run `train.ps1`.

### Linux

Activate venv first (`source venv/bin/activate`), edit `train.sh`, run it.

### TensorBoard

`tensorboard.ps1` → [http://localhost:6006/](http://localhost:6006/)

### Anima single-subject LoRA: step-count rule of thumb

In checkpoint sweeps, **~1k–3k optimizer steps** (same “step” as `total optimization steps` in the log header) is often enough for a usable character; beyond that is mostly polish. Depends on data, repeats, buckets, rank, LR, and taste—trust your samples.

**`num batches per epoch` × epoch** ≈ cumulative steps at epoch end (e.g. 510 batches/epoch → ~1020 steps after epoch 2).

## Program arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--host` | str | `127.0.0.1` | Server host |
| `--port` | int | `28000` | Server port |
| `--listen` | bool | `false` | Listen on all interfaces |
| `--skip-prepare-environment` | bool | `false` | Skip env prep |
| `--disable-tensorboard` | bool | `false` | Disable TensorBoard |
| `--disable-tageditor` | bool | `false` | Disable tag editor |
| `--tensorboard-host` | str | `127.0.0.1` | TensorBoard host |
| `--tensorboard-port` | int | `6006` | TensorBoard port |
| `--localization` | str | | UI locale |
| `--dev` | bool | `false` | Developer mode |
