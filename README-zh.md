<p align="center">
  <img src="assets/readme/anima-cover.png" alt="lora-scripts-next · Anima Trainer" width="880" />
</p>

<h1 align="center">lora-scripts-next</h1>

<p align="center">
  <strong>SD-Trainer</strong> — LoRA 一键训练 GUI，支持 SD / SDXL / Flux / <b>Anima</b><br/>
  <sub>基于 <a href="https://github.com/kohya-ss/sd-scripts">kohya-ss/sd-scripts</a> 后端，秋叶系界面体验。</sub>
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next"><img src="https://img.shields.io/github/stars/wochenlong/lora-scripts-next?style=flat-square&label=stars&logo=github&color=8b5cf6" alt="stars"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next"><img src="https://img.shields.io/github/forks/wochenlong/lora-scripts-next?style=flat-square&label=forks&color=06b6d4" alt="forks"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/LICENSE"><img src="https://img.shields.io/github/license/wochenlong/lora-scripts-next?style=flat-square&color=ec4899" alt="license"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next/releases"><img src="https://img.shields.io/github/v/release/wochenlong/lora-scripts-next?include_prereleases&style=flat-square&color=a78bfa" alt="release"/></a>
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next/releases"><b>下载整合包</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/README.md"><b>English</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/NOTICE.md"><b>致谢 & 许可</b></a>
</p>

---

## 功能亮点

- **多模型支持** — SD 1.5 / SDXL / Flux / **Anima** 全部开箱即用
- **Anima LoRA 训练** — 侧边栏一键进入，支持 LoRA / LoKr（LyCORIS）/ **T-LoRA**
- **Attention 加速** — 源码/venv 优先 Flash Attention 2；整合包使用 xformers / PyTorch SDPA（[详情](docs/flash-attention.md)）
- **T-LoRA** — 时间步动态 Rank LoRA，正交初始化，防过拟合（[论文](https://github.com/ControlGenAI/T-LoRA)）
- **训练监控页** — TensorBoard 同源 Loss/LR 曲线、关键参数速查、实时进度与预览图
- **内置 TensorBoard** — 侧边栏直接查看
- **显卡检测** — 首次安装自动检测 NVIDIA/AMD，AMD 用户提供 ROCm 指引
- **AutoDL 适配** — 提供 `start_autodl.sh` 启动脚本

---

## 快速开始

### Windows 整合包（推荐新手）

从 [Releases](https://github.com/wochenlong/lora-scripts-next/releases) 下载 **`SD-Trainer-v2.4.0.7z`**（~21 MB），解压后双击 `run_gui.bat` 即可。

首次启动自动安装 PyTorch + CUDA + 所有依赖（~3 GB），国内自动镜像加速。

| 文件 | 用途 |
|------|------|
| `run_gui.bat` | 启动入口 |
| `Update-SD-Trainer.bat` | 拉取最新代码 |
| `Download-Anima-Model.bat` | 下载 Anima 基础模型 |

> **要求：** Windows 10/11 64 位，NVIDIA 显卡（RTX 20+），~7 GB 磁盘。

> **Flash Attention 2：** 整合包暂不支持，训练使用 xformers / SDPA。详见 [docs/flash-attention.md](docs/flash-attention.md)。

#### 显存参考（Anima LoRA, 1024 分辨率, RTX 4090 实测）

| 显存 | 配置 | 备注 |
|------|------|------|
| ≥ 24 GB | 默认参数 | 最省心 |
| ≥ 16 GB | 开 `gradient_checkpointing` | 推荐日常 |
| ≥ 12 GB | 梯度检查点 | 稳定 |
| ≥ 10 GB | 梯度检查点 + `blocks_to_swap=16` | 速度略降 |
| ≥ 8 GB | 梯度检查点 + swap 24 + 缓存 TE + LoKr | 极限 |

### 从源码安装

```sh
git clone https://github.com/wochenlong/lora-scripts-next.git
cd lora-scripts-next
```

| 系统 | 操作 |
|------|------|
| Windows | 双击 `run_gui.bat` |
| Linux | `bash install.bash && bash run_gui.sh` |

浏览器自动打开 **http://127.0.0.1:28000**。推荐 Python **3.10**（3.11–3.12 基本可用，3.13+ 不支持）。

指定浏览器：`python gui.py --browser chrome`

> **Flash Attention 2（源码用户）：** 详见 [docs/flash-attention.md](docs/flash-attention.md)。

---

## 界面预览

<p align="center">
  <img src="assets/readme/train-monitor-loss.png" alt="训练监控 Loss 曲线" width="920" />
</p>

<p align="center"><sub>训练监控页 Loss / LR 四宫格曲线</sub></p>

<p align="center">
  <img src="assets/readme/train-monitor-samples.png" alt="训练监控预览图" width="920" />
</p>

<p align="center"><sub>训练预览图同步到监控页</sub></p>

<p align="center">
  <img src="assets/readme/train-monitor-logs.png" alt="训练日志查看" width="920" />
</p>

<p align="center"><sub>训练日志同步到 CMD 终端与监控页</sub></p>

---

## 详细文档

| 主题 | 链接 |
|------|------|
| Anima LoRA 训练指南 | [docs/anima-training.md](docs/anima-training.md) |
| Flash Attention 2 配置 | [docs/flash-attention.md](docs/flash-attention.md) |
| 训练监控 & SSE 接口 | [docs/train-monitor.md](docs/train-monitor.md) |
| 前端定制 | [docs/frontend-customization.md](docs/frontend-customization.md) |
| Docker 部署 | [docs/docker.md](docs/docker.md) |
| 程序参数一览 | [docs/cli-args.md](docs/cli-args.md) |

---

## 常见问题

<details>
<summary><b>无法运行 run_gui.ps1 / 未数字签名</b></summary>

推荐直接双击 `run_gui.bat`。如果一定要运行 `.ps1`：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gui.ps1
```

或长期放宽：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

</details>

<details>
<summary><b>解压后路径嵌套两层</b></summary>

若路径出现 `...\lora-scripts-next-2.4.0\lora-scripts-next-2.4.0\`，请进入内层含 `run_gui.bat` 的目录。

</details>

---

<details>
<summary><b>更新日志</b></summary>

| 日期 | 内容 |
|------|------|
| 2026-05-21 | **v2.4.0** — 训练稳定性：环境隔离、NaN 过滤、采样保护、attn_mode 降级、路径 `\` → `/` 规范化；整合包：tkinter、`install_xformers.bat` |
| 2026-05-20 | **v2.3.0** — 训练监控升级：TensorBoard 同源曲线、参数速查、端口回退、日志同步 |
| 2026-05-19 | **v2.2.0** — 整合包 flash-attn 治本、run_gui.bat 执行策略与闪退日志、跨盘监控、品牌 |
| 2026-05-19 | **v2.1.0** — Flash Attention 2 Windows 预编译 wheel、按步数保存、修复 LoKr undefined |
| 2026-05-18 | **v2.0.0** — 整合包发布、Flash Attention 2 自动加速、AMD 检测、bf16 修复、`--browser`、移除子模块 |
| 2026-05-18 | T-LoRA、交互 Loss 图表、LoKr 标准化、Windows 便携包、AutoDL |
| 2026-05-17 | Anima 后端迁移至 kohya-ss/sd-scripts |

详细变更见 [CHANGELOG.md](CHANGELOG.md)。

</details>

<details>
<summary><b>致谢 & 上游</b></summary>

| 项目 | 角色 |
|------|------|
| [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) | GUI 框架 |
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | 训练后端 |
| [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) | LoKr / LoHa |
| [ControlGenAI/T-LoRA](https://github.com/ControlGenAI/T-LoRA) | 时间步 LoRA |

完整归属见 [`NOTICE.md`](NOTICE.md)。

</details>

---

## 贡献者

详见 [**CONTRIBUTORS.md**](CONTRIBUTORS.md)。

---

<p align="center"><sub>维护者：<b>@wochenlong</b></sub></p>
