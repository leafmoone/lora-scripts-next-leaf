<p align="center">
  <img src="assets/readme/logo.svg" alt="lora-scripts-next" width="140" height="140" />
</p>

<h1 align="center">lora-scripts-next</h1>

<p align="center">
  <strong>SD-Trainer</strong> — LoRA · Dreambooth · 围绕 <a href="https://github.com/kohya-ss/sd-scripts">kohya-ss/sd-scripts</a> 的一站式训练封装<br/>
  <sub><em>个人维护分支：在熟悉的秋叶系体验上，把 Anima、RF 和自用工作流接到一起。</em></sub>
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next"><img src="https://img.shields.io/github/stars/wochenlong/lora-scripts-next?style=flat-square&label=星标&logo=github&color=8b5cf6" alt="stars"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next"><img src="https://img.shields.io/github/forks/wochenlong/lora-scripts-next?style=flat-square&label=分支&color=06b6d4" alt="forks"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/LICENSE"><img src="https://img.shields.io/github/license/wochenlong/lora-scripts-next?style=flat-square&color=ec4899" alt="license"/></a>
  <a href="https://github.com/wochenlong/lora-scripts-next/releases"><img src="https://img.shields.io/github/v/release/wochenlong/lora-scripts-next?include_prereleases&style=flat-square&color=a78bfa" alt="release"/></a>
</p>

<p align="center">
  <a href="https://github.com/wochenlong/lora-scripts-next/releases"><b>下载发布</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/README.md"><b>English README</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/wochenlong/lora-scripts-next/blob/main/NOTICE.md"><b>NOTICE</b></a>
</p>

---

<p align="center">
  <sub>维护者：<b>@wochenlong</b> — 本仓库用于个人迭代；README 头图与界面截图为 <code>assets/readme/</code> 下随仓资源。</sub>
</p>

<br/>

## 一览

| | |
|:---|:---|
| **训练 WebUI** | 预设、TensorBoard、WD 标签器、标签编辑器同一入口；运行 `run_gui.ps1` / `run_gui.sh` 后打开 **`http://127.0.0.1:28000`**。 |
| **后端** | [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)；SDXL RF 脉络来自 [bluvoll/Akegarasu-lora-scripts-RF](https://github.com/bluvoll/Akegarasu-lora-scripts-RF)；Anima 来自 [WhitecrowAurora/lora-rescripts](https://github.com/WhitecrowAurora/lora-rescripts)（<b>SD-reScripts</b>）。 |
| **许可证与致谢** | 详见 [`NOTICE.md`](NOTICE.md)。 |

---

## 界面预览

<p align="center">
  <img src="assets/readme/screenshot-webui.png" alt="SD-Trainer 训练 WebUI 截图" width="920" />
</p>

<p align="center"><sub>同一 WebUI 内可切到 TensorBoard、WD 1.4 标签器、标签编辑器等工具.</sub></p>

---

<details>
<summary><b>上游与血缘（展开）</b></summary>

当前仓库：**[wochenlong/lora-scripts-next](https://github.com/wochenlong/lora-scripts-next)**。界面与打包体验源自 **秋叶一键训练包 / [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts)**，训练后端为 **[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)**。SDXL **Rectified Flow** 参考 **[bluvoll/Akegarasu-lora-scripts-RF](https://github.com/bluvoll/Akegarasu-lora-scripts-RF)**。**Anima** 实现参考 **[WhitecrowAurora/lora-rescripts](https://github.com/WhitecrowAurora/lora-rescripts)**（**SD-reScripts**：对 LoRA-scripts 的维护型分支 / 延续开发）。

</details>

---

# 使用方法

### 必要依赖

Python **3.10** 与 **Git**。

### 克隆（含子模块）

```sh
git clone --recurse-submodules https://github.com/wochenlong/lora-scripts-next.git
cd lora-scripts-next
```

## SD-Trainer GUI

### Windows

**安装：** 运行 `install-cn.ps1`（或 `install.ps1`）。  
**训练：** 运行 `run_gui.ps1`，浏览器打开 **[http://127.0.0.1:28000](http://127.0.0.1:28000)**。

### Linux

**安装：** `install.bash`  
**训练：** `bash run_gui.sh`，同上地址。

### 前端静态文件

训练 GUI 后端默认加载 `frontend/dist`。当前 `frontend` 是预构建静态文件子模块，而不是前端源码目录；如果该子模块指向旧版 dist，启动后看到的也会是旧 UI（例如仍包含 SD3 页面）。

如果要使用自定义或 next 版前端，请先构建前端源码得到 `dist`，然后用环境变量指定静态文件目录：

```bash
MIKAZUKI_FRONTEND_DIST=/path/to/frontend/dist python gui.py --listen
```

也可以更新 `frontend` 子模块到对应的 dist 仓库/commit。后端不会自动从前端源码构建 UI。

### Docker

#### 编译镜像

```bash
docker build -t lora-scripts-next:latest -f Dockerfile-for-Mainland-China .
```

#### 使用镜像（示例）

```bash
docker run --gpus all -p 28000:28000 -p 6006:6006 registry.cn-hangzhou.aliyuncs.com/go-to-mirror/akegarasu_lora-scripts:latest
```

亦可配合仓库内 `docker-compose.yaml`。镜像体积较大，拉取请耐心等待。GPU 与驱动问题请自行查阅文档。

## 通过脚本的传统训练方式

### Windows

运行 `install.ps1` 安装依赖后，编辑并运行 `train.ps1`。

### Linux

先激活虚拟环境：

```sh
source venv/bin/activate
```

编辑 `train.sh` 并运行。

### TensorBoard

`tensorboard.ps1` → [http://localhost:6006/](http://localhost:6006/)

### Anima 单角色 LoRA：训练步数参考（经验值）

在同一套数据与分辨率下对比 checkpoint 时，**约 1000～3000 次优化步**（`total optimization steps` 含义下的 step）往往已能呈现可用的角色外观；再往后更多是在细节与稳定性上微调。实际所需步数随**素材量与质量、repeat、bucket、网络维度、学习率与主观「够不够好」**变化很大，请以验证图为准。

训练启动日志中的 **`num batches per epoch`** × **目标 epoch** ≈ 到该 epoch 结束时的累计步数；例如每 epoch 510 batch、第 2 个 epoch 结束约 **1020** 步。

## 程序参数

| 参数名称 | 类型 | 默认值 | 描述 |
|----------|------|--------|------|
| `--host` | str | `127.0.0.1` | 服务器主机名 |
| `--port` | int | `28000` | 服务端口 |
| `--listen` | bool | `false` | 监听所有网卡 |
| `--skip-prepare-environment` | bool | `false` | 跳过环境准备 |
| `--disable-tensorboard` | bool | `false` | 禁用 TensorBoard |
| `--disable-tageditor` | bool | `false` | 禁用标签编辑器 |
| `--tensorboard-host` | str | `127.0.0.1` | TensorBoard 主机 |
| `--tensorboard-port` | int | `6006` | TensorBoard 端口 |
| `--localization` | str | | 界面语言 |
| `--dev` | bool | `false` | 开发者模式 |
