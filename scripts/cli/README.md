# 命令行训练脚本（秋叶遗留）

**Anima / T-LoRA 请用 WebUI：** `run_gui.bat` 或 `python gui.py`。

本目录为编辑变量后命令行开训的旧流程（SD1.5 / SDXL / Flux），与 WebUI 使用的 `vendor/sd-scripts` 并行，**不是**主推路径。

| 文件 | 说明 |
|------|------|
| `train.ps1` / `train.sh` | 顶部改变量，拼 CLI 参数 |
| `train_by_toml.*` | 读 `config/default.toml`（已弃用标记，路径指向 `vendor/sd-scripts`） |

仓库根目录同名文件为**转发器**，避免旧教程链接失效。
