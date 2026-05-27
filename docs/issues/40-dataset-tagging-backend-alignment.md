# Issue #40 对齐说明 — 数据集打标（Backend Only）

> **用途**：粘贴到 [Issue #40](https://github.com/wochenlong/lora-scripts-next/issues/40) 作为负责人确认评论，或合并进 Issue 正文「协作约定」一节。  
> **负责人**：@wochenlong  
> **执行人**：@niangao2331  
> **关联**：数据集合一界面 / 标签编辑见 **Issue #41（待开）**

---

## 产品背景（中长期，不阻塞 #40）

侧栏最终将收成 **「数据集」** 一项，页内 Tab：

- **自动打标** ← #40 交付 backend，UI 暂用现有 `/tagger.md`
- **标签编辑** ← #41，负责人另线

#40 **只做打标 backend + API 文档 + 测试**，**不写** `frontend/dist`、不改侧栏。

---

## #40 功能范围（Must Have）

### 1. Danbooru / WD 本地打标（延续 + 加固）

| 项 | 要求 |
|----|------|
| 默认模型 | 维持或优化 `wd14-convnextv2-v2`，对齐 `mikazuki/tagger/defaults.py` |
| 输出 | 图片同目录 `.txt` sidecar，与 Kohya `caption_extension` 习惯一致 |
| 冲突策略 | `ignore` / `copy`（overwrite）/ `prepend`，行为与现网一致且可测 |
| 递归 | `batch_input_recursive` |
| 阈值等 | 沿用现有 `TaggerInterrogateRequest` 字段，向后兼容 |

### 2. 请求模型扩展（为 Tag / NL / API 统一入口）

在 **不破坏现有调用** 前提下，扩展 `TaggerInterrogateRequest`（或新增 `DatasetTagRequest`，由 `POST /interrogate` 接受 union）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `mode` | `tag` \| `natural_language` | 输出形态：逗号分隔 tag vs 自然语言 caption |
| `engine` | `local_wd` \| `local_nl` \| `api_gemini` | 执行引擎 |
| `scope` | `all` \| `missing_caption` \| `selected` | 处理范围；`selected` 需 `image_paths: string[]` |
| `image_paths` | `string[]?` | `scope=selected` 时必填 |
| `prompt_preset` | `string?` | NL/API 用，如 `generic` / `anima_detailed` / `short` |
| `caption_extension` | `string` | 默认 `.txt` |

**Phase 1 最低实现**：`mode=tag` + `engine=local_wd`（等价现状）；另至少实现一种 `natural_language` 或 `api_gemini` 之一。

### 3. 自然语言本地打标（NL）

- 至少接入 **一种** HF 本地 caption 模型（BLIP-2 / Florence / 其他 ONNX 友好模型，执行人选型）
- 与 WD tag 在 API 层区分：`mode` + `engine=local_nl`
- 输出写入 `.txt`；README 说明适用场景（非 Danbooru 标签空间）

### 4. API 打标（Gemini 多模态）

| 项 | 要求 |
|----|------|
| Provider | Google Gemini Vision（首期） |
| Key | 环境变量或 `config/` 本地文件，**禁止**进 git |
| 行为 | 超时、重试（≥2）、429 退避、单图失败不拖死整批（记录 failed） |
| 输出 | `.txt`；可选在 status 里返回最近几条 preview |
| 文档 | README-zh 增加配置说明与费用提示 |

### 5. 模型下载进度

- 新增 `GET /api/tagger/download-status`（或扩展现有 status 的 `download` 字段）
- 首次拉 HF 模型时可见：当前文件、已下字节/总字节或阶段文案
- 实现方式自选：`huggingface_hub` 回调、线程安全状态 dict
- **不得** 阻塞训练相关 API 主线程

### 6. 打标任务进度（延续）

- `GET /api/tagger/status`：current/total、message、phase（`idle` / `downloading` / `tagging` / `done` / `error`）
- 与 `tagger_progress.py` 对齐，Gemini/NL 路径也要更新同一进度对象

---

## #40 明确不做（Out of Scope）

- ❌ `frontend/dist/*`、`scripts/patch-*.py`（侧栏 / layout）
- ❌ `mikazuki/dataset-tag-editor/` 及 `:28001` Gradio 子服务
- ❌ 标签编辑 UI、批量 tag 替换/排序/去重（属 #41）
- ❌ `GET /api/dataset/scan` 完整图库索引（属 #41；#40 可预留字段，不强制）
- ❌ 侧栏「数据集」合一 Tab 壳子（负责人 #41）
- ❌ 改 `run_gui.bat`、`python_embeded/` 目录名、`SD-Trainer/` 整合包契约路径
- ❌ 改 `vendor/sd-scripts/` 训练引擎

---

## 可动代码范围 ✅

| 区域 | 路径 | 说明 |
|------|------|------|
| 打标核心 | `mikazuki/tagger/` | interrogator、interrogators/*、format、progress、defaults |
| NL 引擎 | `mikazuki/tagger/interrogators/` | 新增模块，如 `blip.py` / `gemini_api.py` |
| API | `mikazuki/app/api.py` | 注册新路由；**改前在 #40 comment 说一声** |
| 请求模型 | `mikazuki/app/models.py` | 扩展 Request |
| Schema | `mikazuki/schema/tagger.ts` | **仅字段定义**（供将来 UI；#40 可不 patch dist） |
| 预下载 | `scripts/prefetch_default_tagger.py` | 默认模型列表 |
| 整合包构建 | `build-scripts/build_portable.ps1` | 预下载列表、体积说明 |
| 安装/启动 | `install-cn.ps1`、`scripts/portable/launch_portable.bat` | 预取逻辑 |
| 测试 | `tests/test_tagger*.py` | **必须新增** |
| API 文档 | `docs/api/dataset-tagging.md` | **必须新增** |
| 用户文档 | `README-zh.md` 打标 FAQ 段落 | 简短即可 |

---

## 交付物清单（PR 必须包含）

### 代码

- [ ] 扩展打标 API（`mode` / `engine` / `scope` 至少文档化，Phase 1 实现见上）
- [ ] WD 默认链路与现网兼容
- [ ] NL 或 Gemini 至少一种非 WD 路径跑通
- [ ] `GET /api/tagger/download-status`（或等价）
- [ ] `tagger_progress` 覆盖 download + tag + error

### 文档

- [ ] `docs/api/dataset-tagging.md`：每个 endpoint 的请求/响应 JSON 示例
- [ ] `README-zh.md`：Gemini Key 配置、默认模型、离线说明

### 测试

- [ ] `tests/test_tagger_api.py`（或同类）：mock 文件系统 + mock Gemini
- [ ] 冲突策略、`.txt` 写入、非法 path 拒绝

### PR 描述模板

```markdown
## 关联
Fixes #40（或 Part of #40）

## 自测
- 环境：Windows / 源码 install
- 数据集路径：`...`
- curl 命令 + 输出摘要
- pytest 结果

## 整合包
- [ ] 是否增大 huggingface/ 预置
- [ ] build_portable 是否需改

## 截图/日志
（download-status、tagger/status 各一张）
```

---

## 验收标准（负责人 merge 前）

1. **向后兼容**：不传 `mode`/`engine` 时，行为与当前 main 打标一致
2. **curl 可测**：负责人在无 UI 情况下对测试目录跑通 WD + 至少一种 NL/API
3. **进度可见**：下载与打标至少一种在 `status` 或 `download-status` 可见
4. **测试绿**：`python -m pytest tests/test_tagger_api.py -q` 通过
5. **禁区未动**：`dataset-tag-editor`、`frontend/dist`、整合包契约路径
6. **文档齐**：`docs/api/dataset-tagging.md` 可供 #41 前端直接对接

---

## 建议 PR 拆分（避免巨型 PR）

| PR | 内容 |
|----|------|
| PR-1 | `mode`/`engine`/`scope` 模型扩展 + WD 兼容 + 测试基架 |
| PR-2 | 下载进度 API + prefetch 对齐 |
| PR-3 | `local_nl` 或 `api_gemini`（二选一先上） |
| PR-4 | 另一种引擎 + README |

---

## 自测命令（复制给执行人）

```powershell
# 启动
python gui.py --listen

# WD 打标（兼容旧 body）
curl -X POST http://127.0.0.1:28000/api/interrogate ^
  -H "Content-Type: application/json" ^
  -d "{\"path\":\"D:/path/to/dataset\",\"interrogator_model\":\"wd14-convnextv2-v2\",\"batch_input_recursive\":true,\"batch_output_action_on_conflict\":\"copy\"}"

# 进度
curl http://127.0.0.1:28000/api/tagger/status
curl http://127.0.0.1:28000/api/tagger/download-status

# 测试
python -m pytest tests/test_tagger_api.py -q
```

---

## 分工确认（请 @niangao2331 回复勾选）

- [ ] 我认领 #40 全部 backend 范围
- [ ] 我认领 PR-1 ~ PR-4 中的：___
- [ ] 预计第一 PR 时间：___
- [ ] 默认 NL 模型选型倾向：___
- [ ] Gemini Key 本地存储方案：___

---

## 给 Issue #40 的一键评论（短版）

可直接粘贴：

```markdown
## 负责人对齐（2026-05-24）

**#40 = 仅 backend**，完整说明见仓库 `docs/issues/40-dataset-tagging-backend-alignment.md`。

**要做**：WD 打标加固、`mode/engine/scope` API 扩展、下载进度、NL 或 Gemini 至少一种、pytest + `docs/api/dataset-tagging.md`。

**不做**：frontend/dist、dataset-tag-editor、标签编辑、侧栏合一 UI（见将开的 #41）。

**合并标准**：curl 可测、测试绿、向后兼容、禁区未动。

@niangao2331 请回复认领项与第一 PR 范围。
```
