# Handover: sorryhyun/anima_lora 技术集成

> 本文档用于新会话上下文同步。上一个会话完成了对比分析，本文档汇总决策和技术要点。

## 背景

用户发现 [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora)（韩国开发者）的 Anima 训练速度远超本项目（lora-scripts-next），希望借鉴其优化技术。

## 对比结论

| 维度 | sorryhyun/anima_lora | lora-scripts-next |
|------|---------------------|-------------------|
| 定位 | Anima 专用研究级引擎 | 通用多模型训练平台（SD1.5/SDXL/Flux/Anima） |
| 速度 | **1.1 s/step**（RTX 5060 Ti, rank=32, 1MP） | 估计 5-8 s/step |
| 核心加速 | torch.compile + CUDAGraph + 恒定 token bucketing | 标准 eager 模式 + Flash Attention 2 |
| LoRA 类型 | LoRA, OrthoLoRA, T-LoRA, HydraLoRA | LoRA, LoKr, T-LoRA |
| 高级方法 | Spectrum, DCW, ReFT, IP-Adapter, EasyControl, GRAFT | 无 |
| 模型合并 | 无损 SVD 导出 → ComfyUI 直接加载 | 标准 safetensors |
| PyTorch | 2.12 nightly + CUDA 13.2 | 2.7 + CUDA 12.8 |
| Python | 3.13 | 3.10 |
| GUI | 有（config editor + dataset browser + monitor） | 有（完整 Web GUI，更丰富） |
| 便携包 | 无（CLI + Makefile + uv） | Windows 一键整合包 |
| 门槛 | 高（CLI/uv/make） | 低（双击 bat） |
| 协议 | MIT | AGPL-3.0（继承 sd-scripts） |

## 速度差距的关键技术

1. **torch.compile + CUDAGraph**：最大单一加速因素，把 28 个 DiT block 编译为一张静态计算图，每 step 零 kernel 启动开销
2. **恒定 token bucketing**：所有 bucket 目标 ~4096 patches + 零填充 → 固定 shape → 无重编译
3. **编译友好代码路径**：einops→手动 unflatten/permute，autocast→直接 .to(dtype)，dict loops hoisted
4. **最新 PyTorch/CUDA**：torch 2.12 + CUDA 13.2 vs 我们的 2.7 + 12.8

**注意**：torch.compile + CUDAGraph 与 gradient_checkpointing / blocks_to_swap **互斥**，意味着需要更多 VRAM。

## 可借鉴的方向（按可行性排序）

### 短期可行
- OrthoLoRA（SVD 参数化 + 正交正则化，导出为标准 LoRA）
- 无损 SVD 合并导出

### 中期
- Anima 训练路径加入 torch.compile 支持（可选，VRAM 充足时启用）
- Per-block compile 作为默认，full compile + CUDAGraph 作为高级选项

### 长期
- 恒定 token bucketing 策略（compile 加速的前提）
- Spectrum 推理加速（~3.75x，可做 ComfyUI 节点）
- HydraLoRA（多风格训练）

## 当前项目状态

### 本会话已完成的工作
- 贡献者文档：`CONTRIBUTORS.md` 创建（ageless-h 后端、SupermarKleet UI 设计）
- Bug 修复：跨盘 output_dir 导致监控页断联（#12）
- Bug 修复：PEP 508 环境标记导致 run_gui.bat 崩溃（#13）
- 前端更新日志替换为 lora-scripts-next 版本

### 内存占用问题（用户反馈 86GB）
- 不是 bug，是配置问题：`cache_latents_to_disk` 未开启时 latent 全部常驻内存
- 高分辨率（2048）+ 大数据集 + blocks_to_swap + Qwen3/VAE 在 CPU = 轻松 80GB+
- 建议确认 TOML 中四个 cache 开关都设为 true

## 项目文件结构（关键路径）

```
lora-scripts-next/
├── vendor/sd-scripts/           # 上游训练后端（submodule）
├── scripts/dev/anima_train_network.py  # Anima 训练入口 wrapper
├── mikazuki/
│   ├── app/api.py               # FastAPI 后端路由
│   ├── anima_backend/adapter.py # UI → sd-scripts 配置适配
│   └── schema/sd3-lora.ts       # Anima 前端 schema
├── train_status_server.py       # 训练监控（port 6008）
├── frontend/dist/               # 编译后的前端资源
├── config/                      # 配置文件和预设
└── build-scripts/               # 整合包构建脚本
```

## 下一步行动

在新窗口中根据用户选择的优先级开始实施。建议从 OrthoLoRA 或 torch.compile 可选支持开始。
