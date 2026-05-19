# 更新日志

本文件记录 **wochenlong/lora-scripts-next** 面向镜像与 AutoDL 的发行说明；上游 kohya-ss/sd-scripts 的变更请见其仓库。

---

## v2.2.1 — 2026-05-19

### 整合包（治本）

- **便携包不再安装 flash-attn**：嵌入式 Python 无法可靠运行 `triton`，预编译 `flash-attn` 仍会在 import 时依赖 `triton`，导致 `transformers` 加载 CLIP 失败。
- **启动时自动卸载**已装但不可用的 `flash-attn` / `triton`；训练使用 **xformers** 或 **PyTorch SDPA**。
- 训练子进程设置 `TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa`，避免 `transformers` 探测 flash 路径。

---

## v2.2.0 — 2026-05-19

### 整合包与启动

- **triton-windows**：便携包嵌入式 Python 不再安装/保留 `triton-windows`，修复整合包启动或训练时因 triton 编译失败导致的崩溃（[#14](https://github.com/wochenlong/lora-scripts-next/issues/14)）。
- **run_gui.bat**：增加 `sd-trainer-log.txt` 启动日志；失败时明确提示日志路径，避免闪退后无法排查。
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
