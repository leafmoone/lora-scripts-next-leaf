# Flash Attention 2 配置指南 / Flash Attention 2 Setup Guide

[中文](#中文) | [English](#english)

---

## 中文

### 整合包用户

**Windows 整合包**（`SD-Trainer-v*.7z`）**不安装 Flash Attention 2**，训练使用 **xformers** 或 **PyTorch SDPA**。

| 原因 | 说明 |
|------|------|
| flash-attn 依赖 triton | 预编译 wheel 能装，但运行时大量算子通过 `flash_attn.ops.triton` 调用 Triton 生成的 CUDA kernel |
| 嵌入式 Python 跑不好 triton | 整合包的 Python Embeddable 缺少完整编译链，`triton` / `triton-windows` 常在 JIT 时失败 |
| 不能只装 flash-attn | 缺 triton 会报 `No module named 'triton'`；`transformers` 等库探测到 flash_attn 也会尝试走 flash 路径 |
| 整合包策略 | 首次安装跳过 flash-attn；若手动安装了不完整组合，启动时自动卸载并设 `TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa` |

> 需要 Flash Attention 2 请使用**源码 + venv** 安装。

### 源码 / venv 用户

适用于 `git clone` 后使用 **venv**，已安装 **PyTorch 2.7.0 + CUDA 12.8**。

#### 作用范围

| 训练类型 | Flash Attention 2 |
|----------|---------------------|
| **Anima / SD3 LoRA** | 自检通过后自动设 `attn_mode=flash` |
| **SD 1.5 / SDXL / Flux** | 使用 xformers，不依赖 flash-attn |

后端优先级（Anima）：`flash` → `xformers` → `torch`（PyTorch SDPA）。

#### 环境要求

- **Python 3.10**（推荐）
- 64 位 venv，**不要**使用整合包的 `python_embeded`
- `torch==2.7.0+cu128`、`torchvision==0.22.0+cu128`
- Windows 须同时安装 `triton-windows` 和 `flash-attn`

#### 方式一：自动安装（推荐）

首次运行 `run_gui.bat` 或 `run_gui_source.bat` 即可（install 脚本自动尝试安装）。

国内用户：
```powershell
powershell -ExecutionPolicy Bypass -File .\install-cn.ps1
```

国际用户：
```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

#### 方式二：手动安装（Windows）

```powershell
.\venv\Scripts\activate

# 1. PyTorch
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 2. Triton（Windows 必装，先于 flash-attn）
pip install "triton-windows<3.4"

# 3. Flash Attention 2 预编译 wheel（cp310 改为 cp311/cp312 对应你的 Python 版本）
pip install https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
# 国内镜像
pip install https://hf-mirror.com/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
```

#### 方式三：Linux / WSL / AutoDL

```bash
bash install.bash
bash run_gui.sh
```

从源码编译 flash-attn 需要 CUDA 工具链与 C++ 编译器；失败时使用 xformers / SDPA。

#### 验证

```bash
python -c "import triton; import flash_attn; from flash_attn.ops.triton.rotary import apply_rotary; print('Flash Attention 2 OK')"
```

#### 常见问题

| 现象 | 处理 |
|------|------|
| `No module named 'triton'` | 先装 `triton-windows<3.4`，再装 flash-attn wheel |
| wheel 装了但训练仍用 xformers | 运行验证命令，triton 与 flash-attn 须配对 |
| 编译很久或失败 | Windows 用预编译 wheel，不要源码编译 |
| PyTorch 不是 2.7+cu128 | 对齐 torch 版本后再装 |
| 装在整合包 `python_embeded` 里 | **不支持**，改用源码 + venv |

---

## English

### Portable Package Users

The Windows portable package does **not** install Flash Attention 2. Training uses **xformers** or **PyTorch SDPA**.

For Flash Attention 2, use **source + venv** install. See the Chinese section above for detailed instructions (commands are language-independent).

### Source / venv Users

Priority for Anima: `flash` → `xformers` → `torch` (SDPA).

See the Chinese section for step-by-step commands — they work identically in English environments.
