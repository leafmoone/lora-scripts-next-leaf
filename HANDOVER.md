# lora-scripts-next 交接文档

## 项目当前状态
- **项目定义**：基于 WebUI 的 SD/Anima 模型训练工具（lora-scripts 的下一代）。
- **当前核心焦点**：Windows 轻量整合包制作、国内网络环境适配。
- **仓库**：https://github.com/wochenlong/lora-scripts-next

## 已完成的核心工作

### Anima 后端
- 全面接入 `vendor/sd-scripts`，移除旧版 `anima_train_network_legacy.py`。
- 支持 6 种适配器：**LoRA / LoKr / T-LoRA / LoRA-FA / VeRA / LoHa**。
- `mikazuki/anima_backend/adapter.py` 负责前端配置→sd-scripts 参数的转换：
  - `LYCORIS_NETWORK_ARG_MAP`：把 LoKr 高级参数（`use_cp`、`decompose_both`、`full_matrix`、`dora_wd` 等）正确注入 `network_args`。
  - `TLORA_NETWORK_ARG_FIELDS`：把 T-LoRA 参数（`tlora_min_rank`、`tlora_rank_schedule`、`tlora_orthogonal_init`）注入 `network_args`。
- LyCORIS 预设注入：`config/lycoris_anima_preset.toml` 解决 Anima 模型 0 modules 问题。

### 训练监控页
- 独立服务 `train_status_server.py`，端口 `6008`，随 GUI 自动启动。
- ECharts Loss 图表：滚轮缩放、拖拽平移、双击复位、dataZoom 滑动条。
- 自动识别训练类型（T-LoRA / LoKr / LoHa 等）。
- 训练预览图画廊。

### UI / 产品优化
- `lora_type` 选择器提升到页面顶部（用户进入即选适配器类型）。
- `model_train_type` 改为 hidden（后端路由不变，UI 不占位）。
- Schema 定义：`mikazuki/schema/sd3-lora.ts`。

### 文档
- `README.md` / `README-zh.md`：快速开始导向。
- `docs/anima-training.md`：T-LoRA 教程 + LoKr 参数参考 + `full_matrix` 说明。
- 截图已更新为高清 PNG。

### Windows 便携环境
- 修复 `joblib` 依赖缺失。
- 端口统一为 `28000`。
- `run_gui.bat` 一键安装/启动。

## Windows 轻量整合包（已实现）

### 架构设计
- **方案**：轻量底座 + 首次启动在线安装。
- **底座**（7z 压缩约 16 MB）：Python 3.10 Embeddable + 项目代码 + get-pip.py。
- **首次启动**：`setup_environment.py` 自动检测网络环境，安装 PyTorch 2.7+cu128 和全部依赖。
- **CUDA 版本**：统一 `cu128`，兼容 RTX 20/30/40/50 全系列。

### 核心文件
- `build-scripts/build_portable.ps1`：一键构建便携包，输出 `build/SD-Trainer-Portable/`。
- `setup_environment.py`：首次启动安装向导（网络检测→镜像配置→pip→torch→requirements）。
- 生成的 `run_gui.bat`：检测 torch 是否已安装 → 触发 setup → 启动 GUI。
- 生成的 `python310._pth`：路径隔离，`-s` 标志排除用户级 site-packages。

### 镜像策略
- 国内自动检测（探测 google.com）：
  - PyTorch：阿里云 `mirrors.aliyun.com/pytorch-wheels/cu128/`
  - PyPI：清华 `pypi.tuna.tsinghua.edu.cn/simple`
  - HuggingFace：`hf-mirror.com`

### 构建命令
```powershell
.\build-scripts\build_portable.ps1 -Version "2.0.0" -Clean
```

### 下一步
- 完整版整合包（网盘分发）：包含所有依赖 + 基础模型。

## 重要变更：sd-scripts 从子模块改为 vendored 目录

- `vendor/sd-scripts` 不再是 git 子模块，代码直接提交到仓库中。
- **原因**：子模块指向了上游不存在的自定义 commit，导致 `git clone --recurse-submodules` 失败。
- **影响**：`git clone` 不再需要 `--recurse-submodules`，所有代码一次下载完成。
- `verify_pinned_commit` 在非 git 目录时自动跳过检查，commit 不匹配时仅警告不崩溃。
- 更新 sd-scripts 时手动下载覆盖 `vendor/sd-scripts/` 即可。

## 关键技术上下文
- **端口约定**：GUI `28000`，训练监控 `6008`，TensorBoard `6006`。
- **环境**：Python 3.10。`bitsandbytes`、`xformers`、`triton` 在 Windows 下兼容性脆弱，已有回退机制。
- **目标用户**：大量"小白用户"，安装流程和报错提示需要"防呆"。
- **前端**：VuePress 预构建 dist，schema 驱动表单。Playwright 无法渲染 schema 表单（`Shared schema not found`），截图需手动或真实浏览器。
- **测试**：`tests/test_anima_backend_adapter.py` 覆盖 adapter 参数转换逻辑，用 `python -m unittest` 运行（项目未装 pytest）。
