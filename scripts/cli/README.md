# 命令行训练脚本

## Anima Fast（推荐 CLI 路径）

不打开 WebUI、手写 TOML 的进阶流程：

| 文件 | 说明 |
|------|------|
| `install_anima_fast.py` | 安装 Fast 插件（`extensions/anima_lora/` + cu130 venv） |
| `install_anima_fast.bat` / `.sh` | Windows / Linux 包装 |
| `train_anima_fast_by_toml.bat` / `.sh` | 用 TOML 直接调用 `anima_lora` 的 `train.py` |

```powershell
# 1. 安装插件（需 uv、NVIDIA GPU、数 GB 下载）
scripts\cli\install_anima_fast.bat

# 2. 编辑 TOML 后开训（示例配置见 docs/examples/）
scripts\cli\install_anima_fast.bat --dry-run   # 仅查看计划
scripts\cli\train_anima_fast_by_toml.bat docs\examples\anima-lora-benchmark-fast.toml
```

```bash
bash scripts/cli/install_anima_fast.sh
bash scripts/cli/train_anima_fast_by_toml.sh docs/examples/anima-lora-benchmark-fast.toml
```

依赖与 WebUI「开启插件」相同（`uv`、上游 `sorryhyun/anima_lora`）。无本地克隆时会自动浅克隆到 `.cache/anima_fast/upstream/`。

环境变量：`ANIMA_LORA_ROOT`（已有上游仓库路径）、`LORA_ENABLE_ANIMA_FAST=0`（维护者关闭 Fast）。

---

## 秋叶遗留（SD1.5 / SDXL / Flux）

**Anima 标准 Kohya 路径请用 WebUI：** `run_gui.bat` 或 `python gui.py`。

| 文件 | 说明 |
|------|------|
| `train.ps1` / `train.sh` | 顶部改变量，拼 CLI 参数 |
| `train_by_toml.*` | 读 `config/default.toml`（已弃用标记，路径指向 `vendor/sd-scripts`） |

仓库根目录同名文件为**转发器**，避免旧教程链接失效。新脚本请放在本目录，**不要**再往根目录堆 bat/sh。
