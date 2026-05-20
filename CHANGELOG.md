# 更新日志

本文件记录 **wochenlong/lora-scripts-next** 面向镜像与 AutoDL 的发行说明；上游 kohya-ss/sd-scripts 的变更请见其仓库。

---

## v2.3.0 — 2026-05-20

### 训练监控体验升级

- **TensorBoard 同源 Loss 曲线**：6008 训练监控页改为读取 TensorBoard event scalar，默认展示 `loss/average`、`loss/current`、`loss/epoch_average` 与 `lr/unet` 四宫格曲线，避免相对 Loss 曲线含义不清。
- **曲线交互优化**：移除小图底部滑动条，改为 `全部 / 最近 50% / 最近 20% / 最近 10% / 恢复最新` 视野按钮；保留滚轮缩放与拖拽平移。
- **训练参数速查**：监控页顶部显示学习率、优化器、总步数、分辨率、保存频率、精度、seed 等关键参数，方便启动后快速核对配置。
- **终端日志同步**：训练日志同时输出到 CMD 终端与 6008 监控页；终端 echo 失败不会影响监控页日志采集。
- **后台更干净**：静默 6008 监控页正常轮询的 `200/304` access log，只保留真实异常与训练输出。

### 启动稳定性

- **端口冲突回退**：GUI、TensorBoard 与训练监控启动前会严格检测端口；当 6008 被占用时自动切换到可用端口，并避免多个子服务 fallback 到同一个端口。
- **清理测试入口**：移除测试用 `run_gui_anima.bat`，正式包统一使用 `run_gui.bat` 启动。

---

## v2.2.0 — 2026-05-19

### 整合包与启动

- **flash-attn / triton（治本）**：便携包不再安装 `flash-attn`；启动时自动卸载已装但不可用的 `flash-attn` / `triton`；训练使用 **xformers** 或 **PyTorch SDPA**；子进程设置 `TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa`，避免 `No module named 'triton'`（[#14](https://github.com/wochenlong/lora-scripts-next/issues/14) 相关）。
- **triton-windows**：便携包嵌入式 Python 不再安装/保留 `triton-windows`，修复因 triton 编译失败导致的崩溃。
- **run_gui.bat**：纯 cmd 启动（不依赖 `run_gui.ps1`，避免 PowerShell 执行策略报错）；增加 `sd-trainer-log.txt` 启动日志；失败时明确提示日志路径。
- **requirements.txt**：修复 PEP 508 环境标记在 `launch_utils` 中的解析（[#13](https://github.com/wochenlong/lora-scripts-next/issues/13)）。

### 训练监控与 UI

- **跨盘 output_dir**：监控页（6008）在输出目录位于其他盘符时不再断联（[#12](https://github.com/wochenlong/lora-scripts-next/issues/12)）。
- **品牌**：前端作者/链接改为本项目；临时 logo 与 favicon；监控页页头显示 logo。
- **CONTRIBUTORS.md**：贡献者单独文档；README 精简致谢链接。

---

## v2.1 — 2026-05-09

### 训练监控页（端口 6008）

- **Loss 趋势图**：参考 Weights & Biases 风格——16:10 比例、`preserveAspectRatio` 保持比例、网格与坐标轴刻度、100% 基线强调、曲线末端数值标注。
- **指标侧栏**：当前 / 最低（含 step 提示）/ 初始 / 累计下降 / 最近 Δ（着色）/ 趋势 pill；与曲线底部对齐，宽屏下主体仍为左侧曲线。
- **响应式**：窄屏（约 820px 以下）单列堆叠。
