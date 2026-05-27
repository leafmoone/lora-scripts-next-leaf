# Issue #41 草案 — 数据集工作区（共享层 + 标签编辑 Backend + UI）

> **用途**：在 GitHub **新建 Issue**，标题建议：`[Dataset] 数据集工作区 — 共享 API / 标签编辑 / 合一 UI`  
> **负责人**：@wochenlong  
> **依赖**：#40 打标 API 稳定（可并行开发只读 scan，打标 Tab 可先接旧 API）  
> **设计 Demo**：Cursor 画布 `dataset-workspace-demo.canvas.tsx`

---

## 背景

将侧栏 **「数据集打标」+「标签编辑」** 合并为 **「数据集」** 一项，页内 Tab：

| Tab | 职责 | 主要依赖 |
|-----|------|----------|
| 自动打标 | 批量生成 caption | #40 `POST /interrogate` |
| 标签编辑 | 单张/批量精修 tag | 本 Issue 的 dataset API |

Gradio `dataset-tag-editor`（`:28001`）为 **过渡方案**，本 Issue 完成后可 deprecate。

---

## 功能范围

### A. 共享层（两 Tab 共用）

| 功能 | 说明 |
|------|------|
| 数据集路径 | 文件夹选择、递归、`.txt` 扩展名可配 |
| 图库索引 | 扫描图片列表、缩略图（可选 URL 或 base64） |
| 统计 | 总张数、无 caption 数、未保存数 |
| 筛选 | 全部 / 无 caption / 含某 tag / 已修改 |
| 选择 | 单选、多选、全选当前筛选结果 |

### B. 标签编辑 Tab — 单张

| 功能 | 说明 |
|------|------|
| 预览 | 大图 + 上一张/下一张 |
| 读写在 | `GET/PUT /api/dataset/caption` |
| Tag 展示 | 逗号分隔解析为 tag 列表（芯片 UI） |
| 增删 | 单 tag 添加/删除 |
| 排序 | tag 顺序调整（写回 caption 字符串） |
| 去重 | 单张 caption 内 tag 去重 |
| 保存 | 单张保存；未保存标记 |

### C. 标签编辑 Tab — 批量

| 功能 | 说明 |
|------|------|
| 批量添加 | 前缀/后缀 append tag |
| 批量删除 | 按 tag 名删除（支持多 tag） |
| 批量替换 | 查找替换（Phase 1 字符串；Phase 2 regex） |
| 批量去重 | 每张 caption 内去重 |
| 批量排序 | 字母序 / 保持原序去重 |
| 预览 | 应用前 dry-run，返回影响张数与样例 |
| 保存 | 批量写回 `.txt` |

### D. 自动打标 Tab — UI（接 #40）

| 功能 | 说明 |
|------|------|
| 布局 | **方案 A**：左参数、右进度（见 Demo） |
| 参数 | `mode` tag/NL、`engine` local/API、scope、冲突策略 |
| 进度 | 复用 `/api/tagger/status` + download-status |
| 衔接 | 完成后按钮「进入编辑」→ 切 Tab 并保留 path/筛选 |

### E. 导航与页面

| 功能 | 说明 |
|------|------|
| 侧栏 | 「工具与调试」下合并为 **「数据集」** 一项 |
| 路由 | 新页 `/dataset.md`（或保留 `/tagger.md` redirect） |
| 过渡 | 旧 `/tageditor.md` 可保留一版 redirect 或说明 |

---

## Phase 划分

### Phase 1（MVP）

- [ ] `GET /api/dataset/scan`
- [ ] `GET/PUT /api/dataset/caption`（单张）
- [ ] `POST /api/dataset/tags/batch`（add/remove/replace/dedupe/sort 最小集）
- [ ] 编辑 Tab UI：三栏（队列 / 预览 / 编辑器）+ 单张保存
- [ ] 打标 Tab：复用现有 tagger 表单 + 进度（或最小 patch）
- [ ] 共享顶栏：path + 统计

### Phase 2

- [ ] 批量工具 dry-run 预览
- [ ] regex 替换
- [ ] 缩略图缓存策略
- [ ] 打标 Tab 完整 `mode/engine/scope` 表单（对齐 #40）
- [ ]  deprecate Gradio DTE / 移除 `:28001` 启动（可选）

### Phase 3

- [ ] 编辑 Tab「表格批量模式」（caption2 风格子视图）
- [ ] 对选中图「重打标」快捷动作
- [ ] Kohya metadata json（若训练需要）

---

## API 契约（Phase 1）

### `GET /api/dataset/scan`

Query: `path`, `recursive`, `caption_extension`（默认 `.txt`）

```json
{
  "path": "D:/dataset/my_lora",
  "total": 128,
  "missing_caption": 12,
  "items": [
    {
      "image_path": "D:/dataset/my_lora/001.png",
      "caption_path": "D:/dataset/my_lora/001.txt",
      "has_caption": true,
      "caption_preview": "1girl, solo, long hair"
    }
  ]
}
```

### `GET /api/dataset/caption`

Query: `image_path`

```json
{
  "image_path": "...",
  "caption_path": "...",
  "caption": "1girl, solo",
  "tags": ["1girl", "solo"]
}
```

### `PUT /api/dataset/caption`

```json
{
  "image_path": "...",
  "caption": "1girl, solo, outdoors"
}
```

### `POST /api/dataset/tags/batch`

```json
{
  "image_paths": ["...", "..."],
  "operations": [
    { "op": "add", "tags": ["outdoors"], "position": "end" },
    { "op": "remove", "tags": ["lowres"] },
    { "op": "replace", "search": "1girl", "replace": "1woman" },
    { "op": "dedupe" },
    { "op": "sort", "order": "alpha" }
  ],
  "dry_run": false
}
```

Response:

```json
{
  "affected": 42,
  "samples": [{ "image_path": "...", "before": "...", "after": "..." }]
}
```

---

## 可动代码范围 ✅

| 区域 | 路径 |
|------|------|
| 新 backend 模块 | `mikazuki/dataset/`（建议新建：scan、caption、batch_tags） |
| API 路由 | `mikazuki/app/api.py` |
| 模型 | `mikazuki/app/models.py` |
| Schema | `mikazuki/schema/dataset.ts`（新） |
| 前端 | `frontend/dist/` + `scripts/patch-dataset-*.py`（遵循 `frontend/VENDOR.md`） |
| 侧栏 | `scripts/patch-sidebar-nav.py` |
| 测试 | `tests/test_dataset_*.py` |
| 文档 | `docs/api/dataset-workspace.md` |

---

## 尽量少碰 ⚠️

| 路径 | 说明 |
|------|------|
| `mikazuki/tagger/` 核心逻辑 | 打标行为属 #40；本 Issue 只 **调用** API |
| `vendor/sd-scripts/` | 训练引擎 |
| 整合包契约路径 | 同 #40 |
| #40 进行中的 `models.py` 大改 | 与 @niangao2331 协调合并顺序 |

---

## 交付物

### Backend（Phase 1）

- [ ] `mikazuki/dataset/` 模块
- [ ] 上述 4 个 API endpoint
- [ ] `tests/test_dataset_api.py`
- [ ] `docs/api/dataset-workspace.md`

### Frontend（Phase 1）

- [ ] `/dataset.md` 页面（或 patch 现有 tagger 页为 Tab 壳）
- [ ] 共享顶栏 + 编辑 Tab 三栏
- [ ] 侧栏「数据集」入口
- [ ] 打标 Tab 最小可用（可暂链旧 interrogate）

### 文档

- [ ] `README-zh.md`：新入口说明
- [ ] 帮助页「新手上路」更新链接

---

## 验收标准

1. 选一个测试目录：scan → 编辑单张 caption → 批量 add/remove → 磁盘 `.txt` 正确
2. 打标 Tab 能启动任务并看进度（#40 合并后测全引擎）
3. Tab 切换不丢 path
4. pytest 通过
5. 不破坏现有训练页与其他工具入口

---

## UI 参考（负责人已选倾向）

| 区域 | 方案 |
|------|------|
| 打标 Tab | A：左参数右进度 |
| 编辑 Tab | A：三栏工作台；B 表格作后续「批量模式」 |
| 参考项目 | [BooruDatasetTagManager](https://github.com/starik222/BooruDatasetTagManager)、[sd-image-sorter Caption Editor](https://github.com/peter119lee/sd-image-sorter) |

---

## 与 #40 的协作边界

```text
#40 @niangao2331          #41 @wochenlong
mikazuki/tagger/*    →    只 POST /interrogate
docs/api/dataset-tagging  →  UI 打标 Tab 对接
不改 frontend               frontend/dist + 侧栏
```

合并顺序建议：**#40 PR-1（API 模型扩展）先 merge** → #41 打标 Tab 再接新字段。

---

## 新建 GitHub Issue 模板（复制标题+正文）

**Title:** `[Dataset] 数据集工作区 — 共享 API / 标签编辑 / 合一 UI`

**Body:** 粘贴本文件「背景」至「验收标准」全文，并加：

```markdown
Labels: enhancement
Assignees: wochenlong
Depends on: #40
```

**Labels 建议：** `enhancement`  
**Assignee：** @wochenlong  
**Related：** #40
