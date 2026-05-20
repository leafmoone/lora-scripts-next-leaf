# Loss 曲线交接文档

## 需求

在 6008 监控页面的 Loss 趋势图表中，添加**平滑功能**：

1. 在图表区域提供一个「平滑」滑块控件（范围 0~99）
2. 滑块为 0 时，显示原始曲线（当前默认行为）
3. 滑块 > 0 时：
   - 原始 Loss 曲线变为**浅色半透明**（作为背景参考）
   - 在原始曲线之上叠加一条**EMA 平滑后的曲线**（颜色更醒目、线更粗）
   - 相对 Loss（绿色）和原始 Loss（蓝色）两条线各自都需要有独立的原始 + 平滑对
4. 拖动滑块时**实时更新**图表，无需等待下一次数据轮询

---

## 涉及文件

整个监控页面只有 **一个文件**：

```
train_status_server.py    （项目根目录）
```

这是一个独立的 Python HTTP 服务，所有 HTML/CSS/JS 都以 f-string 内嵌在 `render_page()` 函数中（约 800 行 HTML+JS）。没有单独的前端文件。

### 不需要动的文件

- `mikazuki/app/api.py` — GUI 后端 API，Loss 数据从这里获取，但数据格式不需要改
- `gui.py` — 启动入口，只负责启动 `train_status_server.py` 的子进程
- `frontend/` — 这是 28000 端口的 Vue 前端，跟 6008 监控页无关

---

## 当前架构

### 数据流

```
Kohya sd-scripts 训练进程
    │  (stdout/stderr 输出训练日志)
    ▼
gui.py (FastAPI, 端口 28000)
    │  /api/train/log/tail/{task_id}  → 返回日志行
    ▼
train_status_server.py (端口 6008)
    │  1. collect_status() 调用 28000 API 拉日志
    │  2. parse_log() 从日志行中正则提取 loss_points
    │  3. 计算 relative_loss_points（以初始 Loss 为 100%）
    │  4. /api/status 返回 JSON
    ▼
浏览器 (ECharts 图表)
    │  每 3 秒轮询 /api/status
    │  renderLossChart(metrics) 更新图表
```

### 关键数据结构

`/api/status` 返回的 JSON 中，与 Loss 相关的字段在 `metrics` 对象里：

```json
{
  "metrics": {
    "loss": "0.0823",
    "loss_points": [
      {"step": 1, "loss": 0.1234},
      {"step": 2, "loss": 0.1198},
      ...
    ],
    "relative_loss_points": [
      {"step": 1, "relative": 100.0},
      {"step": 2, "relative": 97.08},
      ...
    ],
    "loss_baseline": 0.1234,
    "loss_current": 0.0823,
    "loss_drop_percent": 33.31,
    "loss_trend": "稳定下降",
    "loss_delta": -0.002
  }
}
```

- `loss_points`：原始 Loss 值，用于右 Y 轴的蓝色曲线
- `relative_loss_points`：以初始 Loss 为 100% 的相对值，用于左 Y 轴的绿色曲线
- 这两个数组的长度和 step 顺序是一致的

---

## 图表实现细节

### 图表库

使用 **ECharts 5**，CDN 引入：

```html
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
```

### 当前双轴结构

图表有两个 Y 轴：
- **左 Y 轴 (yAxisIndex: 0)**：相对 Loss (%)，绿色 `#34d399`
- **右 Y 轴 (yAxisIndex: 1)**：原始 Loss 值，蓝色 `#60a5fa`

### 当前 series 定义（2 条线）

```javascript
series: [
  {
    name: "相对 Loss",
    type: "line",
    data: relValues,        // relative_loss_points 的 relative 值数组
    yAxisIndex: 0,
    smooth: 0.15,
    lineStyle: { color: "#34d399", width: 2 },
    areaStyle: { /* 渐变填充 */ },
    markLine: { /* 100% 基准线 */ }
  },
  {
    name: "原始 Loss",
    type: "line",
    data: rawLoss,           // loss_points 的 loss 值数组
    yAxisIndex: 1,
    smooth: 0.15,
    lineStyle: { color: "#60a5fa", width: 1.8, opacity: 0.85 }
  }
]
```

### 已有的交互功能

- **滚轮缩放**：`dataZoom[0]` type `"inside"`，`zoomOnMouseWheel: true`
- **底部滑块**：`dataZoom[1]` type `"slider"`
- **双击复位**：`lossChart.getZr().on("dblclick", ...)`
- **恢复按钮**：缩放后显示「恢复最新」按钮

### 重要注意事项

1. **f-string 转义**：整个 HTML 是 Python f-string，所有 JS 中的 `{` `}` 必须写成 `{{` `}}`
2. **chart 实例复用**：`lossChart` 只初始化一次（`echarts.init`），后续用 `setOption` 更新
3. **replaceMerge**：当前用 `{ replaceMerge: ["xAxis", "series"] }` 来完整替换 series 数组，如果 series 数量变化（开启/关闭平滑）必须用这个
4. **跟随模式**：`lossChartFollowing` 变量控制是否自动滚动到最新数据

---

## 实现建议

### 推荐的 EMA 平滑算法

```javascript
function emaSmooth(data, weight) {
  // weight: 0~1，越大越平滑
  if (!weight || weight <= 0) return data.slice();
  var result = [];
  var last = data[0];
  for (var i = 0; i < data.length; i++) {
    var v = data[i];
    if (!Number.isFinite(v)) { result.push(v); continue; }
    if (!Number.isFinite(last)) { last = v; result.push(v); continue; }
    last = last * weight + v * (1 - weight);
    result.push(last);
  }
  return result;
}
```

### 需要改动的位置

#### 1. HTML 部分（`render_page` 函数内）

在 Loss 趋势面板的标题栏（搜索 `lossChartReset`）旁边添加滑块控件。

#### 2. JS 部分（`render_page` 函数内）

- 添加 `emaSmooth()` 函数
- 添加平滑状态变量 `lossSmoothWeight`
- 监听滑块 `input` 事件，更新 `lossSmoothWeight` 并重新 setOption
- 修改 `renderLossChart()` 中的 series 构建逻辑：
  - 平滑关闭时：保持当前 2 条 series 不变
  - 平滑开启时：变为 4 条 series（2 条浅色原始 + 2 条粗实线平滑）
- 可以考虑抽取一个 `buildLossSeries(relValues, rawLoss, smoothing)` 函数

#### 3. 不需要改 Python 后端

数据源（`loss_points`, `relative_loss_points`）已经足够，平滑计算完全在前端 JS 侧完成。

### 调试方法

```bash
# 启动 GUI（会自动启动 6008 监控服务）
python gui.py

# 或者用 venv
.venv-anima-test/Scripts/python.exe gui.py

# 浏览器打开
http://127.0.0.1:6008

# 查看 API 数据
http://127.0.0.1:6008/api/status
```

修改 `train_status_server.py` 后需要重启 GUI 才能生效（6008 服务是 GUI 的子进程）。

---

## 文件定位速查

| 要找什么 | 搜索关键词 |
|---------|-----------|
| Loss 数据解析（Python） | `def parse_log` |
| Loss 数据点提取 | `loss_points` |
| 相对 Loss 计算 | `relative_loss_points` |
| 图表 HTML 容器 | `lossTrendChart` |
| 图表初始化 | `echarts.init` |
| 图表渲染函数 | `function renderLossChart` |
| 图表 series 定义 | `series: [` （在 renderLossChart 内） |
| 缩放交互 | `dataZoom` |
| 数据轮询入口 | `renderStatus` |
| 恢复按钮 | `lossChartReset` |
