<div align="center">

<img src="https://github.com/Akegarasu/lora-scripts/assets/36563862/3b177f4a-d92a-4da4-85c8-a0d163061a40" width="200" height="200" alt="SD-Trainer" style="border-radius: 25px">

# SD-Trainer

_✨ Enjoy Stable Diffusion Train！ ✨_

</div>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next" style="margin: 2px;">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/wochenlong/lora-scripts-next">
  </a>
  <a href="https://github.com/wochenlong/lora-scripts-next" style="margin: 2px;">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/wochenlong/lora-scripts-next">
  </a>
  <a href="https://raw.githubusercontent.com/wochenlong/lora-scripts-next/main/LICENSE" style="margin: 2px;">
    <img src="https://img.shields.io/github/license/wochenlong/lora-scripts-next" alt="license">
  </a>
  <a href="https://github.com/wochenlong/lora-scripts-next/releases" style="margin: 2px;">
    <img src="https://img.shields.io/github/v/release/wochenlong/lora-scripts-next?color=blueviolet&include_prereleases" alt="release">
  </a>
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next/releases">Download</a>
  ·
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/README.md">Documents</a>
  ·
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/README-zh.md">中文README</a>
</p>

LoRA-scripts (a.k.a SD-Trainer)

LoRA & Dreambooth training GUI & scripts preset & one key training environment for [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts.git)

### About this repository

Maintained at **[wochenlong/lora-scripts-next](https://github.com/wochenlong/lora-scripts-next)**. It extends the **Akegarasu SD-Trainer** stack (widely known as the **秋叶** training bundle): **[Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts)**. Training backend: **[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)**. SDXL **Rectified Flow** LoRA support follows **[bluvoll/Akegarasu-lora-scripts-RF](https://github.com/bluvoll/Akegarasu-lora-scripts-RF)** (RF branch lineage). **Anima** LoRA support is adapted from **[WhitecrowAurora/lora-rescripts](https://github.com/WhitecrowAurora/lora-rescripts)** — **SD-reScripts**, described upstream as a maintained fork / continuation of the LoRA-scripts line. See `NOTICE.md` for attribution and licenses.

*Screenshots below match the upstream Akegarasu (秋叶) GUI layout.*

## ✨NEW: Train WebUI

The **REAL** Stable Diffusion Training Studio. Everything in one WebUI.

Follow the installation guide below to install the GUI, then run `run_gui.ps1`(windows) or `run_gui.sh`(linux) to start the GUI.

![image](https://github.com/Akegarasu/lora-scripts/assets/36563862/d3fcf5ad-fb8f-4e1d-81f9-c903376c19c6)

| Tensorboard | WD 1.4 Tagger | Tag Editor |
| ------------ | ------------ | ------------ |
| ![image](https://github.com/Akegarasu/lora-scripts/assets/36563862/b2ac5c36-3edf-43a6-9719-cb00b757fc76) | ![image](https://github.com/Akegarasu/lora-scripts/assets/36563862/9504fad1-7d77-46a7-a68f-91fbbdbc7407) | ![image](https://github.com/Akegarasu/lora-scripts/assets/36563862/4597917b-caa8-4e90-b950-8b01738996f2) |


# Usage

### Required Dependencies

Python 3.10 and Git

### Clone repo with submodules

```sh
git clone --recurse-submodules https://github.com/wochenlong/lora-scripts-next.git
```

## ✨ SD-Trainer GUI

### Windows

#### Installation

Run `install.ps1` will automatically create a venv for you and install necessary deps. 
If you are in China mainland, please use `install-cn.ps1`

#### Train

run `run_gui.ps1`, then program will open [http://127.0.0.1:28000](http://127.0.0.1:28000) automanticlly

### Linux

#### Installation

Run `install.bash` will create a venv and install necessary deps. 

#### Train

run `bash run_gui.sh`, then program will open [http://127.0.0.1:28000](http://127.0.0.1:28000) automanticlly

## Legacy training through run script manually

### Windows

#### Installation

Run `install.ps1` will automatically create a venv for you and install necessary deps.

#### Train

Edit `train.ps1`, and run it.

### Linux

#### Installation

Run `install.bash` will create a venv and install necessary deps.

#### Train

Training script `train.sh` **will not** activate venv for you. You should activate venv first.

```sh
source venv/bin/activate
```

Edit `train.sh`, and run it.

#### TensorBoard

Run `tensorboard.ps1` will start TensorBoard at http://localhost:6006/

### Anima single-subject LoRA: step-count rule of thumb

In our side-by-side checkpoint tests, **roughly 1k–3k optimizer steps** (the same “step” as `total optimization steps` in the training header) is often enough for a usable character likeness; longer runs mostly refine details and stability. Your mileage varies with dataset size/quality, repeats, buckets, network rank, LR, and how strict you are about “good enough”—always judge from validation samples.

At startup, **`num batches per epoch` × epoch number** ≈ cumulative steps at the end of that epoch (e.g. 510 batches/epoch → ~1020 steps after epoch 2).

## Program arguments

| Parameter Name                | Type  | Default Value | Description                                      |
|-------------------------------|-------|---------------|--------------------------------------------------|
| `--host`                      | str   | "127.0.0.1"   | Hostname for the server                          |
| `--port`                      | int   | 28000         | Port to run the server                           |
| `--listen`                    | bool  | false         | Enable listening mode for the server             |
| `--skip-prepare-environment`  | bool  | false         | Skip the environment preparation step            |
| `--disable-tensorboard`       | bool  | false         | Disable TensorBoard                              |
| `--disable-tageditor`         | bool  | false         | Disable tag editor                               |
| `--tensorboard-host`          | str   | "127.0.0.1"   | Host to run TensorBoard                          |
| `--tensorboard-port`          | int   | 6006          | Port to run TensorBoard                          |
| `--localization`              | str   |               | Localization settings for the interface          |
| `--dev`                       | bool  | false         | Developer mode to disale some checks             |
