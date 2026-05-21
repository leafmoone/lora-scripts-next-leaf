# Flash Attention 2 配置指南

[中文](#中文) | [English](#english)

---

## 中文

### 概述

Flash Attention 2 是**可选加速**组件，**不默认安装**。安装后 Anima LoRA 训练会自动使用。

未安装时训练正常运行，使用 xformers 或 PyTorch SDPA 作为注意力后端。

运行时优先级：`flash` → `xformers` → `torch`（SDPA），由 `api.py` 自动选择，无需手动配置。

### 整合包用户

**整合包不支持 Flash Attention 2**，训练使用 xformers / PyTorch SDPA。

原因：嵌入式 Python 缺少编译工具链，flash-attn 依赖的 triton 无法在其中运行。

不要对 `python_embeded` 手动安装 flash-attn。

### 源码 / venv 用户

#### 一键安装

| 系统 | 命令 |
|------|------|
| **Windows** | 双击 `install_flash_attn.bat` |
| **Linux / WSL** | `bash install_flash_attn.sh` |

脚本会自动安装 triton（Windows）和 flash-attn 预编译 wheel，安装后启动训练即可自动启用。

#### 环境要求

- Python **3.10**（推荐，3.11–3.12 若有对应 wheel 也可）
- `torch==2.7.0+cu128`、`torchvision==0.22.0+cu128`
- 64 位 venv，**不是**整合包的 `python_embeded`

#### 手动安装（Windows）

```powershell
.\venv\Scripts\activate

# Triton（Windows 必装，先于 flash-attn）
pip install "triton-windows<3.4"

# Flash Attention 2 预编译 wheel（cp310 改为 cp311/cp312 对应你的 Python 版本）
pip install https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl

# 国内镜像
pip install https://hf-mirror.com/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
```

#### 手动安装（Linux）

```bash
pip install flash-attn --no-build-isolation
```

需要 CUDA 工具链和 C++ 编译器。

#### 验证

```bash
python -c "import triton; import flash_attn; from flash_attn.ops.triton.rotary import apply_rotary; print('Flash Attention 2 OK')"
```

启动训练后日志应显示 `attn_mode` 为 `flash`。

#### 常见问题

| 现象 | 处理 |
|------|------|
| `No module named 'triton'` | Windows 先装 `triton-windows<3.4` |
| 装了但训练仍用 xformers | 运行验证命令，triton 和 flash-attn 须配对 |
| 编译很久或失败 | Windows 用预编译 wheel |
| PyTorch 不是 2.7+cu128 | 对齐 torch 版本后再装 |
| 装在整合包 `python_embeded` 里 | 不支持，改用源码 + venv |

---

## English

### Overview

Flash Attention 2 is an **optional** acceleration component, **not installed by default**. When installed, Anima LoRA training uses it automatically.

Without it, training runs normally using xformers or PyTorch SDPA.

### Portable Package Users

The portable package does **not** support Flash Attention 2. Do not install flash-attn into `python_embeded`.

### Source / venv Users

Run `install_flash_attn.bat` (Windows) or `bash install_flash_attn.sh` (Linux) to install.

See the Chinese section above for detailed manual install commands — they are language-independent.
