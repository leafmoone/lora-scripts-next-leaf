# 训练监控页

## 概述

训练监控页是一个独立的实时状态面板（默认端口 6008），随 GUI 自动启动并在浏览器中打开。

## 功能

- 实时训练状态卡片（状态 / 进度 / Epoch / ETA / Loss / LR / it/s）
- Loss 趋势图
- 训练预览图展示
- `./output` 下最新输出文件列表
- 粘性底部滚动（上翻看历史不会被新日志拖回去）

## 访问方式

- **自动打开** — GUI 启动后自动在浏览器打开 `http://127.0.0.1:6008`
- **手动访问** — 浏览器打开 `http://127.0.0.1:6008`
- **指定任务** — URL 后加 `?task_id=<uuid>`
- **嵌入** — `<iframe src="http://127.0.0.1:6008?task_id=…" />`

## SSE 原始流接口

WebUI 启动训练时，后端会通过 Server-Sent Events 转发训练日志：

- **监控页面** — `http://127.0.0.1:28000/train-log`（自动发现当前运行任务）
- **原始流** — `GET /api/train/log/stream/{task_id}` 返回 `text/event-stream`

`task_id` 来自 `POST /api/run` 的返回值。如果 URL 里没带 `task_id`，监控页会自动查询 `GET /api/train/tasks` 选取最新任务。

## 配置

| 环境变量 / 参数 | 默认值 | 说明 |
|----------------|--------|------|
| `--train-monitor-port` | 6008 | 监控页端口 |
| `--disable-train-monitor` | false | 禁用训练监控 |
