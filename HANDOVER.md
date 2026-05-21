# 交接文档：训练监控页

> **状态（2026-05-21）**：UI 重构已验收并合入 `train_monitor/`。下文保留架构与 API 说明；第 6–7 节的设计探索流程仅供后续改版参考。

---

## 1. 项目概况

**SD-Trainer / lora-scripts-next** — 一个 AI 图像模型 LoRA 训练工具，包含完整 Web GUI、训练后端、和独立的训练监控页面。

- **项目路径**：`d:\ai\lora-scripts-next`
- **训练监控页**：`http://localhost:6008`，独立 Python HTTP 服务，每 2 秒轮询更新
- **用户群体**：AI 绘画/训练爱好者，技术水平中等偏上
- **设计风格**：暗色主题（deep navy），科技感，数据密集型仪表盘

---

## 2. 训练监控页文件结构

```
train_monitor/
├── index.html      (~100 行)  HTML 页面模板，纯静态结构
├── monitor.css     (~1230 行)  Design tokens + 组件样式
├── monitor.js      (~720 行)  轮询、渲染、ECharts、结果区/预览
└── server.py       (~1060 行) Python 后端（API、数据采集、静态文件服务）
```

静态资源带查询参数缓存破坏：`monitor.css?v=20260521-ui6`、`monitor.js?v=20260521-ui6`（改版时同步 bump）。

### 入口与启动

- `gui.py` 第 55 行：`subprocess.Popen([sys.executable, "train_monitor/server.py"])`
- 环境变量 `TRAIN_MONITOR_PORT`（默认 6008）控制端口
- 静态文件通过 `/static/monitor.css` 和 `/static/monitor.js` 加载
- 数据通过 `/api/status` JSON 端点提供

### 单独启动监控页（给 Agent）

**项目根目录**：`d:\ai\lora-scripts-next`（以下命令均在此目录执行）

| 场景 | 命令 |
|------|------|
| 只开监控页 | `python train_monitor/server.py` |
| 自定义端口 | PowerShell: `$env:TRAIN_MONITOR_PORT="6009"; python train_monitor/server.py` |
| GUI 非默认端口 | `$env:MIKAZUKI_PORT="28000"; python train_monitor/server.py`（默认即 28000） |

**访问**：浏览器打开 `http://127.0.0.1:6008`（或你设置的端口）

**依赖说明**：

- 监控服务可独立进程运行，不依赖 `gui.py` 同进程启动。
- **训练状态 / 日志 / 任务列表** 来自主 GUI API：`http://127.0.0.1:{MIKAZUKI_PORT}/api`（默认 `MIKAZUKI_PORT=28000`）。
- 仅启动监控、未启动主 GUI 时：页面能打开，通常显示「空闲 / GUI 离线」，无实时训练数据。
- 要验收真实训练 UI：需同时运行 `python gui.py`（或等价启动脚本），且端口与 `MIKAZUKI_PORT` 一致。
- GPU 条等硬件信息由 `server.py` 本机读取（`pynvml`），不依赖 GUI。

**与 GUI 一起启动**：`python gui.py` 会自动拉起监控；禁用：`python gui.py --disable-train-monitor`

**实现入口**：`train_monitor/server.py` 末尾 `if __name__ == "__main__": main()`，监听 `0.0.0.0:PORT`。

### 技术栈

- **无框架**：纯 HTML + CSS + Vanilla JS（无 React/Vue/Tailwind）
- **图表**：ECharts 5（CDN 加载）
- **后端**：Python `http.server`（标准库 ThreadingHTTPServer）
- **GPU 监控**：pynvml（nvidia-ml-py）

---

## 3. 当前页面区块布局（从上到下）

```
┌──────────────────────────────────────────────┐
│ Header: Logo + "训练监控" + 更新时间           │
├──────────────────────────────────────────────┤
│ Error Box (条件显示)                          │
├──────────────────────────────────────────────┤
│ Hero Section:                                │
│   标题 + 状态描述 + 进度百分比                 │
│   [======== 训练进度条 ========]              │
│   Loss 摘要（当前值 + 趋势标签）               │
│   [==== 采样进度条 ====]                      │
├──────────────────────────────────────────────┤
│ Cards Grid (7 张):                           │
│   模型类型 | 状态 | 进度 | Epoch              │
│   耗时 | 剩余 | Loss                          │
├──────────────────────────────────────────────┤
│ Param Row (三列):                            │
│   [GPU 面板]  [总步数 Hero]  [训练参数]        │
│    GPU名称     大数字展示     学习率/优化器      │
│    Load 条     计算公式       Rank/Alpha 等     │
│    VRAM 条                                    │
│    温度/功耗                                   │
├──────────────────────────────────────────────┤
│ 训练预览图 (默认关闭，隐私保护)                 │
│   网格展示采样图，标注 Step/Epoch               │
├──────────────────────────────────────────────┤
│ Loss 趋势 (ECharts):                         │
│   范围按钮：全部 | 50% | 20% | 10% | 最新     │
│   双图表网格：loss/average + loss/current 等   │
│   支持滚轮缩放、拖拽平移                       │
├──────────────────────────────────────────────┤
│ 训练日志 (折叠)                               │
│   自动跟随 / 暂停回看                          │
│   最近 180 行，等宽字体                        │
├──────────────────────────────────────────────┤
│ 训练结果                                      │
│   最新模型名称/路径/大小                        │
│   历史 checkpoint 折叠                         │
└──────────────────────────────────────────────┘
```

---

## 4. 当前设计系统（CSS 变量）

```css
:root {
  color-scheme: dark;
  --bg: #0b1020;        /* 主背景 */
  --panel: #121a2e;     /* 面板背景 */
  --line: #26324d;      /* 边框/分割线 */
  --text: #e5edf8;      /* 主文字 */
  --muted: #91a0b8;     /* 次要文字 */
  --ok: #34d399;        /* 正常/成功 */
  --warn: #fbbf24;      /* 警告 */
  --err: #fb7185;       /* 错误 */
}
```

### 色彩使用

- 进度条渐变：`#38bdf8 → #34d399`（蓝→绿）
- GPU Load 条：`#38bdf8 → #34d399`
- VRAM 条：`#a78bfa → #ec4899`（紫→粉）
- 采样进度条：`#fbbf24 → #fb7185`（黄→红）
- Loss 曲线：`#16bac5`（青色）
- 代码/路径：`#bfdbfe`（浅蓝）
- 卡片 accent：`#a5b4fc`（总步数大数字）

### 字体

- 正文：`system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
- 代码：`ui-monospace, SFMono-Regular, Consolas, monospace`

---

## 5. 数据 API 格式

`GET /api/status` 返回 JSON，关键字段：

```json
{
  "time": "2026-05-21 15:20:00",
  "state": "训练中|已结束|空闲|GUI 离线",
  "model_type": "Flux LoRA|SDXL LoKr|Anima T-LoRA|...",
  "metrics": {
    "step": 150, "total_steps": 600, "percent": 25.0,
    "epoch": "3/10", "loss": "0.0832", "lr": "1e-04",
    "eta": "12分30秒", "elapsed": "4分15秒",
    "sampling": { "active": true, "percent": 40, "step": 8, "total_steps": 20 },
    "loss_trend": "稳定下降|上升波动|小幅波动",
    "has_error": false, "needs_attention": false, "progress_stalled": false
  },
  "gpu_info": {
    "name": "NVIDIA GeForce RTX 4090",
    "vram_used_mb": 18432, "vram_total_mb": 24576,
    "gpu_load": 95, "mem_load": 78,
    "temperature": 72, "power_w": 320.5, "power_limit_w": 450.0
  },
  "train_params": [
    {"label": "总步数", "value": "1200（20图 × 10r ÷ BS2 × 12ep）"},
    {"label": "学习率", "value": "1.00e-04"},
    {"label": "优化器", "value": "AdamW8bit"},
    ...
  ],
  "tensorboard_loss": [
    {"tag": "loss/average", "points": [{"step":1,"value":0.12},...], "latest": 0.08, "min": 0.06},
    {"tag": "lr/unet", "points": [...], "latest": 0.0001}
  ],
  "previews": [
    {"name": "img_e000003.png", "url": "/preview-image?path=...", "role": "最新图", "epoch": 3}
  ],
  "output_scope": "output/520",
  "outputs_primary": [{"name": "...", "path": "...", "size": "...", "mtime": "...", "mtime_ts": 0, "folder": "...", "ext": ".safetensors", "epoch": 10}],
  "outputs_other": [],
  "outputs": [],
  "log_lines": ["line1", "line2", ...]
}
```

---

## 6. 已落地的 UI 改动（验收版）

- **Design tokens**：`monitor.css` `:root` 语义色板（`--color-bg-*`、`--color-accent-primary` 等），保留 `--bg`/`--panel` 等别名兼容旧逻辑
- **Hero**：训练中顶栏高亮、`monitor-training` body 类、进度条 active 态
- **指标卡**：训练中「状态 / 进度 / Loss」accent + 状态 live 圆点
- **硬件与参数**：合并为单 section「硬件与训练参数」；GPU / 总步数 hero / 参数摘要三列
- **预览**：开关式 `preview-toggle` + `localStorage`；默认不加载图片
- **Loss**：ECharts 主题读 CSS 变量；范围按钮；`lr` 缺失时占位卡
- **训练结果**：最新模型卡片 + 复制路径；`outputs_primary` / `outputs_other`；相对时间 `mtime_ts`
- **空闲态**：`model_type` 为 `null` 时前端显示「—」，Hero 标题为「当前空闲」

### 设计重构关注点（后续改版参考）

### 保持不变
- 暗色主题方向
- 数据密集型仪表盘定位
- 所有现有功能区块（不删减功能）
- 纯 Vanilla JS + CSS 技术栈（不引入框架）
- ECharts 图表库

### 希望改善
- **视觉层次**：当前所有面板视觉权重接近，缺少明确的信息主次
- **色彩体系**：渐变和强调色较多且不统一，可以更克制
- **排版节奏**：间距和字号层级可以更精致
- **组件一致性**：卡片、面板、按钮的设计语言可以更统一
- **响应式**：目前有基础 `@media` 但可以做得更好
- **GPU 面板 + 参数面板的信息密度**：当前三列布局可以优化
- **动效**：进度条有 `transition`，但整体缺乏有节奏的微交互

### 约束
- **单页应用**：只有一个页面，无路由
- **无构建步骤**：不能用 Sass/PostCSS/打包工具，必须是纯 CSS
- **CDN 依赖**：ECharts 5 从 CDN 加载，这是唯一的外部依赖
- **文件数量**：保持 `index.html` + `monitor.css` + `monitor.js` 三文件结构

---

## 7. 建议的设计 Agent 工作流

在 `e:\claude` 工作区打开新 Cursor 窗口，参照 `前端设计-Agent使用说明.md`：

```
1. @ui-designer  — 重新设计 design tokens、组件规范、布局线框
2. @ux-architect — 确定信息层级、交互流、组件树
3. 回到本项目应用 CSS + HTML 改动
4. @evidence-collector — 验收
```

### 设计时需要参考的文件

| 文件 | 位置 | 用途 |
|------|------|------|
| `monitor.css` | `d:\ai\lora-scripts-next\train_monitor\monitor.css` | 当前所有样式 |
| `index.html` | `d:\ai\lora-scripts-next\train_monitor\index.html` | 当前 HTML 结构 |
| `monitor.js` | `d:\ai\lora-scripts-next\train_monitor\monitor.js` | 了解动态渲染逻辑 |
| 本文档 | `d:\ai\lora-scripts-next\HANDOVER.md` | 上下文总览 |

---

## 8. 项目其他状态（供参考）

- **最近完成**：训练监控页 UI 重构（tokens、结果区、预览开关、训练态动效）、模型输出分 primary/other、Flash Attention 安装脚本化
- **Git 分支**：`main`，直推
- **远程仓库**：`https://github.com/wochenlong/lora-scripts-next.git`
