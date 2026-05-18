# lora-scripts-next 交接文档

## 项目当前状态
- **项目定义**：基于 WebUI 的 SD/Anima 模型训练工具（lora-scripts 的下一代）。
- **当前核心焦点**：Anima 模型训练支持、Windows 小白用户体验优化、国内网络环境适配。

## 最近完成的核心工作
1. **Anima 后端支持**：
   - 移除了旧版 `anima_train_network_legacy.py`，全面接入 `vendor/sd-scripts`。
   - 修复了 LyCORIS `kohya` 模块在 Anima 模型上找不到模块（0 modules）导致 `IndexError` 的问题。解决方案：在 `mikazuki/anima_backend/adapter.py` 中动态注入了针对 Anima 的专属预设文件 `config/lycoris_anima_preset.toml`。
   - 修复了 `network_args` 中出现 `undefined` 导致训练崩溃的问题。
   - 修复了 LoKr 高级参数（`use_cp`、`decompose_both`、`full_matrix` 等）未通过 `network_args` 传递给 LyCORIS 的 bug——此前这些参数作为 TOML 顶层 key 输出，被 sd-scripts 静默忽略。
   - 新增 `full_matrix` 参数支持（LyCORIS 全矩阵 Kronecker 模式）。
2. **训练监控页 (Train Monitor)**：
   - 独立服务 `train_status_server.py`，默认端口 `6008`。
   - 随 GUI (端口 `28000`) 自动启动，并在启动后自动在浏览器中打开。
   - 修复了与 TensorBoard 端口冲突的问题（TensorBoard 默认开启，不影响监控页）。
   - Loss 图表使用 ECharts 渲染，支持滚轮缩放、拖拽平移、双击复位，底部有 dataZoom 滑动条，手动操作后显示「恢复最新」按钮一键回到实时跟随。
   - 自动识别训练类型（Anima T-LoRA / LoKr / LoHa / LoRA-FA / VeRA 等），在监控页顶部展示。
   ![训练监控页](assets/readme/shot-train-monitor.png)
3. **文档与 UI 优化**：
   - 重写了 `README.md` 和 `README-zh.md`，突出“快速开始”，将长篇技术文档移入 `docs/` 目录。
   - 增加了 `docs/anima-training.md`，包含 T-LoRA 教程和 LoKr 参数参考。
4. **Windows 便携环境适配**：
   - 修复了便携包缺少 `joblib` 依赖的问题。
   - 统一了所有脚本和文档中的端口号为 `28000`（之前有 `30000` 混用的情况）。
   - 新增了 `run_gui.bat`，为 Windows 小白用户提供一键安装/启动体验。

## 待办事项 (Pending Tasks)
1. **整合包分发策略**：
   - **需求**：GitHub Releases 有大小限制，且国内下载慢。
   - **计划**：
     - **轻量版整合包 (Lightweight Package)**：放置在 GitHub Releases，控制在 1.5G 以下。需要决定包含/剔除哪些组件（例如：不带大模型权重，精简环境）。
     - **完整版整合包 (Full Package)**：放置在网盘（如夸克、百度网盘等），包含所有依赖和基础模型。
2. **国内网络优化自动配置**：
   - **需求**：国内用户下载依赖和模型经常失败。
   - **计划**：在 `install-cn.ps1` 和 `run_gui.bat` 等安装脚本中，自动配置 pip 国内镜像源（清华/阿里）以及 HuggingFace 镜像源 (`HF_ENDPOINT=https://hf-mirror.com`)。

## 关键技术上下文
- **端口约定**：GUI 主服务 `28000`，训练监控页 `6008`，AutoDL 默认使用 `6006` 和 `6008`。
- **环境依赖**：推荐 Python 3.10。`bitsandbytes`、`xformers` 和 `triton` 在 Windows 下的兼容性较脆弱，当前已通过特定版本和回退机制（如 `xformers` 缺失时回退到 SDPA）处理。
- **目标用户**：大量为“小白用户”，因此任何报错提示、安装流程、一键脚本都需要尽可能做到“防呆”和自动化。
