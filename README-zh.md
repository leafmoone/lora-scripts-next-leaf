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

## 快速开始

### Windows 整合包（推荐小白用户）

从 [Releases](https://github.com/wochenlong/lora-scripts-next/releases) 下载整合包，解压后双击 `run_gui.bat` 即可启动。

### 从源码安装

```sh
git clone --recurse-submodules https://github.com/wochenlong/lora-scripts-next.git
cd lora-scripts-next
```

| 系统 | 操作 |
|------|------|
| Windows | 双击 **`run_gui.bat`**（首次自动安装依赖，之后直接启动） |
| Linux | `bash install.bash && bash run_gui.sh` |

启动后浏览器自动打开 **http://127.0.0.1:28000**。

> **Python 版本：** 推荐 **3.10**（所有依赖完美兼容）。3.11–3.12 基本可用，3.13+ 不支持。

---

## 功能亮点

- **多模型支持** — SD 1.5 / SDXL / Flux / **Anima** 全部开箱即用
- **Anima LoRA 训练** — 侧边栏一键进入，支持 LoRA / LoKr（LyCORIS）/ **T-LoRA**
- **T-LoRA** — 基于扩散时间步的动态 Rank LoRA，正交初始化，防止过拟合（[论文](https://github.com/ControlGenAI/T-LoRA)）
- **训练监控页** — 随 GUI 自动启动，ECharts 交互式 Loss 图表（滚轮缩放 / 拖拽平移 / 一键复位），实时进度和预览图
- **TensorBoard 内置** — 侧边栏直接查看，无需额外操作
- **AutoDL 适配** — 提供专用启动脚本 `start_autodl.sh`

---

## 界面预览

<p align="center">
  <img src="assets/readme/screenshot-webui.png" alt="训练 WebUI" width="920" />
</p>

<p align="center">
  <img src="assets/readme/shot-train-monitor.png" alt="训练监控页" width="920" />
</p>

<p align="center"><sub>上：训练 GUI 主界面 &nbsp;|&nbsp; 下：训练监控页（端口 6008，自动打开）</sub></p>

---

## 详细文档

| 主题 | 链接 |
|------|------|
| Anima LoRA 训练指南 | [docs/anima-training.md](docs/anima-training.md) |
| 训练监控 & SSE 接口 | [docs/train-monitor.md](docs/train-monitor.md) |
| 前端定制 | [docs/frontend-customization.md](docs/frontend-customization.md) |
| Docker 部署 | [docs/docker.md](docs/docker.md) |
| 程序参数一览 | [docs/cli-args.md](docs/cli-args.md) |

---

<details>
<summary><b>更新日志</b></summary>

| 日期 | 内容 |
|------|------|
| 2026-05-18 | T-LoRA（时间步动态 LoRA）Anima 训练支持；ECharts 交互式 Loss 图表（缩放/拖拽/复位） |
| 2026-05-18 | 修复 LoRA/LoHa/LoKr/TLoRA 等训练类型在 network_args 出现 undefined 时导致训练报错（自定义参数可正确覆盖） |
| 2026-05-18 | Anima LoKr 训练支持标准化（lycoris.kohya 后端） |
| 2026-05-18 | 训练监控页随 GUI 自动启动，新增 AutoDL 专用启动脚本 |
| 2026-05-18 | 新增 Windows 便携包一键构建脚本 |
| 2026-05-17 | Anima 训练后端完全迁移至 kohya-ss/sd-scripts |
| 2026-05-06 | 训练监控页重构：实时 Loss 卡片 + 粘性滚动 |

</details>

<details>
<summary><b>致谢 & 上游</b></summary>

| 项目 | 角色 |
|------|------|
| [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts) | GUI 框架与一键训练体验（"秋叶式"） |
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | 核心训练后端 |
| [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) | LoKr / LoHa 网络模块（Apache-2.0） |
| [ControlGenAI/T-LoRA](https://github.com/ControlGenAI/T-LoRA) | 时间步动态 LoRA（MIT, AIRI） |
| [bluvoll/Akegarasu-lora-scripts-RF](https://github.com/bluvoll/Akegarasu-lora-scripts-RF) | SDXL Rectified Flow 参考 |

完整归属见 [`NOTICE.md`](NOTICE.md)。

</details>

---

<p align="center"><sub>维护者：<b>@wochenlong</b></sub></p>
