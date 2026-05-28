# 数据集打标页 · 模型下载与打标进度 UI 设计

> 范围：`/tagger.html`（侧栏「工具与调试 → 数据集打标」）  
> 关联 Issue：[#40](https://github.com/wochenlong/lora-scripts-next/issues/40) Phase 1  
> 约束：不改三栏栅格；右栏底部 **操作坞** 融合进度与「启动 / 全部重置」

---

## 1. 放置位置（底部操作坞）

右栏改为 **上说明、下操作** 两段式，避免中间「浮岛卡片」：

```
┌─ right-container（flex 列）────────────┐
│  ▲ 可滚动：Tagger 说明 + 推荐参数       │
│  │                                   │
│  ▼                                   │
├─ sd-tagger-dock（贴底，顶部分割线）────┤
│  [空闲] 一行状态文案                    │
│  （进行中才展开）模型 / 打标 细进度条     │
│  [空闲徽章] 提示文案 ………… 预下载（链接）   │
│  [  启 动（主，flex 拉满） ] [ 全部重置 ] │  ← 同一行，各保留一枚（去 Vue 重复挂载）
└──────────────────────────────────────┘
```

原则：

- **进度与按钮同一视觉块**：`border-top` + 轻阴影，像固定底栏，不再单独套一层圆角卡片。
- **空闲时极简**：只显示阶段徽章 + 一行提示，**不展示空进度条**（避免「丑且空」）。
- **进行中再展开**：`downloading` 显示模型条；`tagging` 显示打标条。
- **启动为主操作**：坞内第一个 `el-button` 实心主题色；预下载降为右上角文字链接。

---

## 2. 组件结构

```html
<footer class="sd-tagger-dock" id="sd-tagger-dock">
  <div class="sd-tagger-dock__status" id="sd-tagger-status" aria-live="polite">
    <div class="sd-tagger-dock__status-line">
      <span class="sd-tagger-dock__phase" data-phase>空闲</span>
      <span class="sd-tagger-dock__message" data-status-message>…</span>
    </div>
    <div class="sd-tagger-dock__meters" data-meters><!-- 仅 busy 时 is-visible --></div>
  </div>
  <div class="sd-tagger-dock__toolbar">
    <button type="button" class="sd-tagger-dock__link" data-prefetch-btn>预下载所选模型</button>
  </div>
  <div class="sd-tagger-dock__buttons"><!-- Vue 启动 / 全部重置 迁入 --></div>
</footer>
```

由 `tagger-progress.js` 创建 dock；**启动/重置为 dock 自管按钮**（直接调 API），Vue 原按钮隐藏（DOM 搬迁会破坏 Vue 点击）。
点击后：即时状态文案 + 进度条脉冲 + Element 消息 + 高频轮询 status。

---

## 3. 视觉规格

| 元素 | 规格 |
|------|------|
| 操作坞 | 贴底 `margin-top: auto`；顶边 `1px` 分隔；`box-shadow: 0 -4px 16px rgba(15,23,42,.04)` |
| 状态行 | 左：阶段 pill；右：单行 message（0.75rem 灰字，busy 时加深） |
| 进度条 | 仅 `is-visible` 时显示；轨道高 `5px`；下载条浅色、打标条实心 primary |
| 预下载 | 胶囊 ghost 按钮，嵌在状态行右侧 |
| 完成态 | 与训练器一致用 **主题紫**（非 Element 绿）；浅紫圆点 + 紫字文案 |
| 启动 | 对齐 `.color-btn.el-button--primary`：浅紫底、紫边、紫字、字重 650 |
| 重置 | 对齐 `.max-btn` 灰底次要按钮 |
| 进度条 | 纯色 `var(--el-color-primary)`，不用蓝紫渐变 |

---

## 4. 交互与 API

| 用户操作 | UI 行为 | API |
|----------|---------|-----|
| 进入页面 | 轮询 `GET /api/tagger/status`（1.2s） | 初始 `phase: idle` |

**启动打标流水线**：`interrogator_assets_ready` → 否则 `downloading` → `tagging` → `done`（与预下载共用 `model_fetch.py`）。
| 点击「预下载所选模型」 | 读中栏 `interrogator_model`；禁用按钮；下载条动画 | `POST /api/tagger/prefetch` |
| 点击「启动」 | 模型未缓存时先 `downloading`（按文件 1/N），完成后自动 `tagging` | `POST /api/interrogate` + `ensure_interrogator_assets` |
| 完成 / 失败 | 消息区展示 `message`；3s 后下载条可归零 | `phase: done` / `error` |

`interrogator_model` 从表单读取：中栏 `.schema-container` 内第一个 `select` / `.el-select` 的当前值（与 layout 提交字段一致）。

---

## 5. 实现落点

| 层 | 路径 |
|----|------|
| 设计（本文） | `docs/design/tagger-progress-ui.md` |
| 后端状态 | `mikazuki/tagger/progress.py`、`model_fetch.py` |
| API | `GET /api/tagger/status`、`POST /api/tagger/prefetch` |
| 样式 | `frontend/dist/assets/sd-trainer-ui-polish.css`（`.sd-tagger-dock`） |
| 全局版本号 | 侧栏「Next Trainer」下 `vX.Y.Z`（`/api/version` + `sd-trainer-brand.js`） |
| 右栏 HTML | 无需 SSR 插卡；`tagger.html` 仅引入 `tagger-progress.js` |
| 轮询脚本 | `frontend/dist/assets/tagger-progress.js` 构建底部 dock |
| 维护脚本 | `scripts/patch-tagger-progress-ui.py` |

---

## 6. 验收

- [ ] 进度与「启动 / 全部重置」在同一底部操作坞，无中间浮岛空白
- [ ] 预下载 CL / WD 非默认模型时，下载条有 1/N 文件阶段反馈
- [ ] 打标 100+ 张时，打标条与 `current/total` 同步
- [ ] 与浅色训练页、新手上路卡片风格协调（圆角、浅边框、非亮蓝链接色）

---

## 7. 2026-05-28 补充（UI 抛光 + 加速源）

- 底部 dock 按钮升级为更圆润风格，启动/重置保持同一视觉层级。
- 进度条默认折叠，仅在 `downloading` / `tagging` / `pending` / `cancelling` 时展开。
- 下载阶段在无法拿到精确字节百分比时，展示 `indeterminate` 动画条，避免“看起来卡死”。
- 新增 `download_endpoint`（默认 / `https://hf-mirror.com` / `https://modelscope.cn`），采用 schema 原生下拉渲染。
- 已移除页面底部兜底行逻辑，避免 `download_endpoint` 重复显示。
- `tagger.html` 对 `tagger-progress.js` 与 `style.874872ce.css` 增加版本+时间戳参数，降低首次刷新命中旧缓存概率。
