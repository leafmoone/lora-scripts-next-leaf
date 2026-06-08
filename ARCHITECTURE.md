# lora-scripts-next (Next Trainer v2.7.0) — 项目架构分析

## 1. 项目概述

**Next Trainer** 是一个一键式 LoRA/微调训练 GUI，支持 **Anima**、**SD 1.5**、**SDXL** 和 **Flux** 模型。基于 kohya-ss/sd-scripts 构建，是对 Akegarasu/lora-scripts 的社区演进分支。

- 版本: **v2.7.0**
- 主入口: `gui.py`
- 后端: FastAPI + Python
- 前端: VuePress 2.0 (SPA)
- 训练引擎: kohya-ss/sd-scripts (标准 Kohya) + sorryhyun/anima_lora (Anima Fast 插件)

---

## 2. 顶层目录结构

| 目录/文件 | 用途 |
|-----------|------|
| `gui.py` | **主入口** — 解析 CLI 参数，启动子服务（TensorBoard、训练监控、标签编辑器），启动 FastAPI |
| `mikazuki/` | **核心后端** Python 包 — 应用的"心脏" |
| `frontend/dist/` | **VuePress 前端** 制品目录 — Vue 3 + Element Plus SPA |
| `vendor/sd-scripts/` | **kohya-ss/sd-scripts** 分支 — 增加 Anima 支持 |
| `train_monitor/` | 独立的训练监控 HTTP 服务器 + 前端 |
| `scripts/` | CLI 脚本、便携启动器、训练脚本 (stable/dev/cli) |
| `build-scripts/` | 便携包构建器 (PowerShell) |
| `config/` | TOML 配置文件 |
| `docs/` | 用户与开发者文档 |
| `tests/` | 28 个 Python 测试文件 |
| `extensions/` | Anima Fast 插件运行目录 |
| `legacy/` | 上游遗留的标注器/笔记本工具 |

---

## 3. 后端架构 (`gui.py` + `mikazuki/`)

### 3.1 入口: `gui.py`

`gui.py` 是**总编排器**，负责:
1. 解析 CLI 参数 (`--host` 默认 `0.0.0.0`, `--port` 默认 `12345` 等)
2. 环境准备 (`prepare_environment()`)
3. 嵌入式 Python 依赖清理
4. 端口扫描与冲突保护
5. 启动子进程: TensorBoard (12348)、训练监控 (12347)、标签编辑器 (28001)
6. 启动 FastAPI `uvicorn.run("mikazuki.app:app", ...)`

### 3.2 `mikazuki/app/` — FastAPI 应用

| 文件 | 用途 |
|------|------|
| `application.py` | FastAPI 应用初始化: SPA 静态文件、CORS、缓存中间件、启动生命周期 |
| `api.py` | **核心 API 路由 (~1030 行)** — 所有训练/标注/插件端点 |
| `models.py` | Pydantic 数据模型 |
| `proxy.py` | TensorBoard 和旧版标注编辑器的反向代理 |

### 3.3 核心模块

| 文件 | 用途 |
|------|------|
| `tasks.py` | **TaskManager** — 训练任务进程管理 (创建/运行/完成/终止/失败) |
| `train_log_hub.py` | **TrainLogHub** — 线程安全环形缓冲区 (15K 行/任务)，用于 SSE 日志流 |
| `process.py` | 训练进程启动器 — 构建 `accelerate launch` 命令，管理 GPU |
| `dataset_editor.py` | **原生数据集编辑器 API** — 扫描/读写/批量编辑/撤销重做 |
| `launch_utils.py` | 环境准备和验证 |
| `portable_utils.py` | 嵌入式 Python 工具 |

### 3.4 API 端点概览

| 端点 | 功能 |
|------|------|
| `POST /api/run` | **核心: 启动训练** — 接收 JSON 配置，路由到对应训练器 |
| `GET /api/train/log/stream/{id}` | SSE 实时训练日志流 |
| `GET /api/tasks` | 任务列表 |
| `POST /api/tasks/terminate/{id}` | 终止任务 |
| `GET /api/schemas/all` | 加载 TypeScript 表单 Schema |
| `POST /api/plugins/anima-lora/install` | 安装 Anima Fast 插件 |
| `POST /api/dataset-editor/scan` | 扫描数据集 |
| `POST /api/interrogate` | 批量图像标注 |

### 3.5 训练器类型映射

```python
trainer_mapping = {
    "sd-lora":        "./scripts/stable/train_network.py",
    "sdxl-lora":      "./scripts/stable/sdxl_train_network.py",
    "sd-dreambooth":  "./scripts/stable/train_db.py",
    "sdxl-finetune":  "./scripts/stable/sdxl_train.py",
    "sd3-lora":       "./scripts/dev/anima_train_network.py",
    "anima-lora":     "./scripts/dev/anima_train_network.py",
    "anima-finetune": "./scripts/dev/anima_train.py",
    "flux-lora":      "./scripts/dev/flux_train_network.py",
    "flux-finetune":  "./scripts/dev/flux_train.py",
}
```

---

## 4. Anima Fast 插件系统 (`mikazuki/anima_fast_backend/`)

最复杂的子系统。Anima Fast 插件是一个**完全独立的 Python 3.13 运行环境**，拥有自己的 venv，运行 `sorryhyun/anima_lora` 训练代码。**不使用** `accelerate`，直接使用自己的 `train.py`。

### 4.1 关键文件

| 文件 | 用途 |
|------|------|
| `settings.py` | 运行时配置发现 — 读取 `config/anima_fast_backend.toml` |
| `adapter.py` | 将 UI 配置转换为 Anima Fast TOML 格式 |
| `installer.py` | 扩展安装 — 构建安装计划、复制源码、删除扩展 |
| `environment.py` | **环境安装 (~650 行)** — 通过 `uv` 安装 Python 3.13，创建 venv，安装依赖 |
| `extension_state.py` | **状态机** — 追踪扩展生命周期 (未安装→安装中→审计中→就绪/损坏) |
| `launcher.py` | 构建 `[python, train.py, --config_file, toml]` 启动指令 |
| `preflight.py` | **预检验证 (~347 行)** — 验证模型文件、图片、参数冲突 |
| `preview.py` | 生成 Anima 训练的采样提示词 |
| `preprocess.py` | **数据集预处理** — 自动调整图片尺寸 |
| `progress.py` | **进度解析** — 从 `progress.jsonl` 提取 loss/epoch/step |
| `source_root.py` | 源码定位 — 多优先级链发现 `sorryhyun/anima_lora` 源码 |

### 4.2 Anima Fast 数据流

```
用户点击"安装插件" → POST /api/plugins/anima-lora/install
  → source_root 发现源码
  → environment 安装 Python 3.13 + venv + 依赖 (uv pip install)
  → 环境审计
  → 状态变为 "ready"

用户提交训练 → POST /api/run (model_train_type: "anima-lora-fast")
  → 就绪检查 (插件已安装且审计通过)
  → 数据集预处理
  → 配置适配 (UI → TOML)
  → 预检验证
  → extensions/anima_lora/.venv/bin/python train.py
```

---

## 5. 训练监控 (`train_monitor/`)

独立的 HTTP 服务器 (`http.server.ThreadingHTTPServer`)，运行在端口 **12347**。

| 文件 | 用途 |
|------|------|
| `server.py` | ~1200 行 — 轮询 GUI API，解析标准输出 (tqdm 进度条, loss)，读取 TensorBoard 数据，GPU 信息，服务预览图 |
| `index.html` | 监控前端 (轻量 SPA) |
| `monitor.js` | ~32KB 监控 UI 逻辑 |
| `monitor.css` | ~25KB 样式 |

---

## 6. 标注系统 (`mikazuki/tagger/`)

基于 ONNX 的批量图像标注器，支持 10 种模型。

### 6.1 关键文件

| 文件 | 用途 |
|------|------|
| `interrogator.py` | **主逻辑** — 10 种标注器注册，后处理 (阈值、去重、JSON 输出) |
| `interrogators/wd14.py` | WaifuDiffusion 标注器 — ConvNeXt/SwinV2/ViT/MoaT/EVA-02 |
| `interrogators/cl.py` | CL-CLIP 标注器 |
| `jobs.py` | 后台任务执行器 — 预下载 + 标注 |
| `progress.py` | 进度追踪 (闲置/下载中/标注中/完成/出错/取消) |
| `ort_session.py` | ONNX Runtime 会话管理 |

### 6.2 可用标注器

`wd14-convnextv2-v2` (默认), `wd-convnext-v3`, `wd-swinv2-v3`, `wd-vit-v3`, `wd14-swinv2-v2`, `wd14-vit-v2`, `wd14-moat-v2`, `wd-eva02-large-tagger-v3`, `wd-vit-large-tagger-v3`, `cl_tagger_1_01`

---

## 7. 前端 (`frontend/dist/`)

- **框架**: VuePress 2.0.0-beta.49 (SSR + 水合 SPA)
- 来源: `hanamizuki-ai/lora-gui-dist` (从 `Akegarasu/lora-scripts-frontend` 构建)
- 制品直接到入库中 (~1.6 MB)

### 主要页面

| 页面 | 路径 | 功能 |
|------|------|------|
| 首页 | `/index.html` | 快速入口 |
| Anima LoRA | `/lora/sd3.html` | Anima LoRA 专家模式 (Kohya) |
| Anima Fast | `/lora/anima-fast.html` | Anima Fast 插件模式 |
| Anima 微调 | `/lora/anima-finetune.html` | Anima 全量微调 |
| Flux | `/lora/flux.html` | Flux LoRA |
| SD 统一 | `/lora/master.html` | SD1.5/SDXL 统一入口 |
| 标注器 | `/tagger.html` | 批量图像标注 |
| 数据集编辑器 | `/dataset-editor.html` | 原生数据集浏览器 |
| 任务监控 | `/task.html` | 训练任务管理 |
| TensorBoard | `/tensorboard.html` | 嵌入 TensorBoard |

### 前端-后端连接

- 表单生成: 前端动态加载 `/api/schemas/all` 中的 JSON Schema
- 训练提交: POST `/api/run`
- 日志流: SSE 连接 `/api/train/log/stream/{task_id}`
- 代理: TensorBoard 通过 `/proxy/tensorboard/`，旧版标注编辑器通过 `/proxy/tageditor/`

---

## 8. Schema 系统 (`mikazuki/schema/`)

TypeScript Schema 文件定义训练参数，前端动态评估构建表单:

| 文件 | 内容 |
|------|------|
| `shared.ts` | **共享 Schema**: 数据集、标注、精度/缓存/批次、网络选项、保存/日志/优化器 (28 种)、预览图、数据增强等 |
| `anima-lora-fast.ts` | **Anima Fast 专属**: 模型字段、Fast 参数、专属数据集、Fast 优化器约束 |

---

## 9. 便携包系统 (`build-scripts/` + `scripts/portable/`)

### 构建流程

```
01-prepare-python.ps1  → 下载嵌入版 Python
02-install-dependencies.ps1 → 安装 pip 包
03-copy-project.ps1 → 复制项目文件
04-create-launchers.ps1 → 生成 .bat 启动器
05-create-zip.ps1 → 打包 7z
```

### 便携包结构

```
<PortableRoot>/
  run_gui.bat
  python_embeded/  → 嵌入版 Python 3.10
  SD-Trainer/      → 项目文件
  tagger-models/wd14/ → 预置标注模型 (~400 MB)
```

---

## 10. CLI 入口

| 脚本 | 用途 |
|------|------|
| `train.sh` | 旧版 SD/SDXL/Flux LoRA CLI |
| `train_anima_by_toml.sh` | Anima 标准 (Kohya) TOML 训练 |
| `train_anima_fast_by_toml.sh` | Anima Fast TOML 训练 |
| `run_gui.sh` | 启动 GUI 服务器 |
| `run_gui_cn.sh` | 启动 GUI (中国镜像) |

---

## 11. 测试 (`tests/`)

28 个测试文件覆盖:

- **Anima Fast**: 配置适配、状态机、环境安装、预检、进度解析、源码定位、API 端点、特性开关
- **Anima Kohya**: 后端适配、训练默认值、微调适配器
- **训练监控**: 状态聚合、模型类型推断、进度合并、错误检测
- **数据集编辑器**: 扫描、读写、批量编辑、撤销重做
- **打包**: 便携包脚本
- **其他**: CLI 入口、混合精度、采样提示词规范化、标注器

---

## 12. 整体架构图

```
┌─────────────────────────────────────────┐
│         浏览器                           │
│   VuePress SPA (frontend/dist/)         │
│   Vue 3 + Element Plus                   │
└────────────────┬────────────────────────┘
                 │ HTTP / SSE
                 ▼
┌─────────────────────────────────────────┐
│      FastAPI (gui.py)                    │
│  ┌───────────────────────────────────┐  │
│  │ mikazuki/app/ (FastAPI 核心)      │  │
│  │  - api.py     (REST 端点)         │  │
│  │  - proxy.py   (代理)              │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │ mikazuki/tasks.py     任务管理    │  │
│  │ mikazuki/process.py   进程启动    │  │
│  │ mikazuki/tagger/      标注系统    │  │
│  │ mikazuki/dataset_editor.py 编辑器 │  │
│  │ mikazuki/anima_fast_backend/      │  │
│  │ mikazuki/anima_backend/           │  │
│  └───────────────────────────────────┘  │
└────────────────┬────────────────────────┘
                 │ 子进程
                 ▼
┌─────────────────────────────────────────┐
│            训练后端                       │
│  ┌───────────────────────────────────┐  │
│  │ 标准 Kohya:                        │  │
│  │ accelerate launch scripts/*.py     │  │
│  │ 使用 vendor/sd-scripts/            │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │ Anima Fast:                        │  │
│  │ extensions/anima_lora/.venv/       │  │
│  │ python train.py                    │  │
│  │ 独立 Python 3.13 环境             │  │
│  │ 管理: uv pip install               │  │
│  └───────────────────────────────────┘  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│        辅助服务 (独立进程)               │
│  train_monitor/server.py  (12347)       │
│  tensorboard              (12348)       │
│  dataset-tag-editor       (28001)       │
└─────────────────────────────────────────┘
```

---

## 13. 依赖

### 主依赖
- **PyTorch** 2.x (CUDA), **accelerate** 0.33.0
- **transformers** 4.51.3, **diffusers** 0.33.1
- **fastapi** 0.95.1, **uvicorn** 0.22.0
- **tensorboard** 2.10.1, **gradio** 3.44.2
- **lycoris-lora** 3.3.0, **safetensors** 0.4.4
- **bitsandbytes** 0.46.0
- 20+ 优化器包

### Anima Fast 依赖 (通过 uv 管理)
- PyTorch 2.11.0+cu130, **transformers** 5.9.0
- **diffusers** 0.37.1, **flash-attn** 2.8.3+cu130
- 独立 Python **3.13** venv

---

## 14. 训练数据流

```
前端表单 (Schema 动态生成)
    │
    ▼
POST /api/run (JSON 配置)
    │
    ▼
后端路由: model_train_type 决定训练器
    │
    ├── 标准 Kohya → accelerate launch scripts/...
    │   ├── 验证模型/数据集
    │   ├── 规范化采样提示词
    │   ├── 写入 TOML
    │   └── 启动训练
    │
    └── Anima Fast → extensions/anima_lora/.venv/bin/python
        ├── 检查插件就绪
        ├── 数据集预处理 (调整图片尺寸)
        ├── 配置适配 (UI → Fast TOML)
        ├── 预检验证
        └── 启动训练
                │
                ▼
stdout → TrainLogHub → SSE → 前端实时日志
progress.jsonl → 监控器 → 状态页面
```
