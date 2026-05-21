<p align="center">
  <img src="assets/readme/anima-cover.png" alt="lora-scripts-next · Anima Trainer" width="880" />
</p>

<h1 align="center">SD-Trainer</h1>

<p align="center">
  <b>Windows 一键 LoRA 训练工具</b> — 支持 <b>Anima</b> / SD 1.5 / SDXL / Flux<br/>
  解压即用，无需配环境。基于 <a href="https://github.com/kohya-ss/sd-scripts">kohya-ss/sd-scripts</a>，秋叶系 GUI 体验。
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next/releases"><img src="https://img.shields.io/github/v/release/wochenlong/lora-scripts-next?include_prereleases&style=for-the-badge&color=a78bfa&label=%E4%B8%8B%E8%BD%BD%E6%95%B4%E5%90%88%E5%8C%85" alt="下载整合包"/></a>
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next"><img src="https://img.shields.io/github/stars/wochenlong/lora-scripts-next?style=flat-square&label=stars&logo=github&color=8b5cf6" alt="stars"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/LICENSE"><img src="https://img.shields.io/github/license/wochenlong/lora-scripts-next?style=flat-square&color=ec4899" alt="license"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/README-zh.md"><b>中文</b></a> · <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/NOTICE.md"><b>Credits</b></a>
</p>

---

<p align="center">
  <img src="assets/readme/screenshot-webui.png" alt="SD-Trainer GUI" width="920" />
</p>

<p align="center"><sub>Anima LoRA 训练界面 — 左侧参数配置，右侧实时预览</sub></p>

---

## 三步开始训练

```
1. 下载  →  从 Releases 下载 SD-Trainer-v2.4.0.7z（~21 MB），解压
2. 启动  →  双击 run_gui.bat（首次自动安装依赖 ~3 GB）
3. 训练  →  浏览器打开 http://127.0.0.1:28000，选模型、填参数、开练
```

> **要求：** Windows 10/11，NVIDIA 显卡（RTX 20+），~7 GB 磁盘。

<details>
<summary><b>从源码安装（Linux / 高级用户）</b></summary>

```sh
git clone https://github.com/wochenlong/lora-scripts-next.git
cd lora-scripts-next

# Windows
run_gui.bat

# Linux
bash install.bash && bash run_gui.sh
```

推荐 Python **3.10**。源码用户可启用 [Flash Attention 2](docs/flash-attention.md) 加速。

</details>

---

## 支持什么

| 模型 | 网络类型 | 注意力后端 |
|------|----------|------------|
| **Anima** / SD3 | LoRA · LoKr · **T-LoRA** | Flash Attention 2 / xformers / SDPA |
| SD 1.5 / SDXL | LoRA · LoHa · LoKr | xformers / SDPA |
| Flux | LoRA | xformers / SDPA |

---

## 训练监控

训练启动后自动打开监控页（端口 6008），实时查看 Loss 曲线、预览图、训练日志。

<p align="center">
  <img src="assets/readme/train-monitor-loss.png" alt="Loss 曲线" width="920" />
</p>

<p align="center"><sub>TensorBoard 同源 Loss / LR 四宫格</sub></p>

<p align="center">
  <img src="assets/readme/train-monitor-samples.png" alt="预览图" width="920" />
</p>

<p align="center"><sub>训练预览图实时同步</sub></p>

---

## 显存参考

Anima LoRA, 1024 分辨率, batch=1, bf16（RTX 4090 实测）：

| 显存 | 配置 | 备注 |
|------|------|------|
| ≥ 24 GB | 默认参数 | 最省心 |
| ≥ 16 GB | `gradient_checkpointing` | 推荐日常 |
| ≥ 12 GB | 梯度检查点 | 稳定 |
| ≥ 10 GB | 梯度检查点 + `blocks_to_swap=16` | 速度略降 |
| ≥ 8 GB | 梯度检查点 + swap 24 + 缓存 TE + LoKr | 极限 |

---

## 文档

| 主题 | 链接 |
|------|------|
| Anima LoRA 训练指南 | [docs/anima-training.md](docs/anima-training.md) |
| Flash Attention 2 | [docs/flash-attention.md](docs/flash-attention.md) |
| 训练监控 & SSE 接口 | [docs/train-monitor.md](docs/train-monitor.md) |
| Docker 部署 | [docs/docker.md](docs/docker.md) |
| CLI 参数 | [docs/cli-args.md](docs/cli-args.md) |

---

<details>
<summary><b>更新日志</b></summary>

| 日期 | 版本 |
|------|------|
| 2026-05-21 | **v2.4.0** — 训练稳定性：环境隔离、NaN 过滤、采样保护、attn_mode 降级、路径规范化；整合包 tkinter 修复 |
| 2026-05-20 | **v2.3.0** — 训练监控升级：TensorBoard 同源曲线、参数速查、日志同步 |
| 2026-05-19 | **v2.2.0** — 整合包 flash-attn 治本、闪退日志、跨盘监控 |
| 2026-05-19 | **v2.1.0** — Flash Attention 2 预编译 wheel、按步数保存 |
| 2026-05-18 | **v2.0.0** — 整合包首发、AMD 检测、bf16 修复 |

详见 [CHANGELOG.md](CHANGELOG.md)。

</details>

<details>
<summary><b>致谢</b></summary>

[Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) · [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) · [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) · [T-LoRA](https://github.com/ControlGenAI/T-LoRA) — 完整归属见 [NOTICE.md](NOTICE.md)

</details>

---

<p align="center"><sub>维护者：<b><a href="https://github.com/wochenlong">@wochenlong</a></b> · <a href="CONTRIBUTORS.md">贡献者</a></sub></p>
