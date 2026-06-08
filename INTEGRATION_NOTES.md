# 新增功能集成方案与前端限制说明

本文档记录在 `lora-scripts-next` 项目中集成 **Differential LoRA** 和 **DiffSynth Tagger (Tag-Edit-Leaf)** 两个新功能时遇到的前端架构限制，以及最终的解决方案。

---

## 1. 新增功能

### 1.1 Differential LoRA（差分 LoRA 训练）

- **后端**: `mikazuki/differential_lora/` — 核心编排器、TOML 适配器、数据集预处理
- **API**: `mikazuki/app/differential_lora_api.py` — `/api/differential-lora/run`, `/pairs`, `/status`
- **工具**: `tools/merge_lora_to_base.py`, `tools/average_lora.py`, `tools/convert_differential_to_comfyui.py`
- **前端**: `frontend/dist/lora/differential-lora.html` — 独立训练页面
- **配置**: `config/differential_lora.toml`

### 1.2 DiffSynth Tagger（AI 智能打标）

- **后端**: `mikazuki/app/tag_edit_leaf_api.py` — `/api/tag-edit-leaf/run`, `/scan`, `/status`
- **标注引擎**: `tools/differential_tagger/` — DiffSynth 标注器完整代码（14 个文件）
- **前端**: `frontend/dist/tag-edit-leaf.html` — 独立标注页面

### 1.3 Schema

- `mikazuki/schema/differential-lora.ts` — 前端表单 Schema（已定义但独立页面未使用，留作后续 VuePress 集成用）

---

## 2. 核心问题：VuePress SSR 水合（Hydration）限制

### 2.1 架构背景

`lora-scripts-next` 的前端是 **VuePress 2.0.0-beta.49** SPA（单页应用），其页面通过 SSR（服务端渲染）预生成 HTML，浏览器加载后由 Vue 运行时执行 **水合（hydration）** — 将静态 DOM 与 Vue 虚拟 DOM 对齐，激活交互。

```text
浏览器请求 /lora/sd3.html
  → FastAPI 返回静态 HTML（VuePress 编译产物）
  → 浏览器加载 app.547295de.js（VuePress 运行时）
  → Vue 水合: 逐节点比对 DOM 和 VNode 树
  → 激活侧边栏交互、Schema 表单渲染、路由
```

### 2.2 水合崩溃的根本原因

VuePress SSR 生成的 HTML 在侧边栏区域使用了特殊的 **虚拟节点锚点**：

```html
<ul class="sidebar-item-children">
  <!--[--><!--]-->     ← Vue SSR 锚点（数量必须匹配 VNode）
  <li><a href="...">LoRA训练</a></li>
  <li><a href="...">全量微调</a></li>
  <!--[--><!--]-->     ← Vue SSR 锚点
</ul>
```

水合时，Vue 逐节点比对 DOM 和编译时的 VNode 树。**插入新节点会导致 `nextSibling` 返回 `null`**，触发 JavaScript 错误：

```javascript
TypeError: Cannot read properties of null (reading 'nextSibling')
    at nextSibling (app.547295de.js:1:62910)
    at f (app.547295de.js:1:41608)
    // ... Vue 水合链路崩溃
```

此错误导致整个页面渲染失败，侧边栏和其他 Vue 组件全部停止工作。

### 2.3 尝试过的方案

| 方案 | 结果 | 原因 |
|------|------|------|
| 1. HTML 中静态插入 `<li>` 侧边栏条目 | ❌ 崩溃 | SSR 锚点数与 VNode 不匹配 |
| 2. 添加 `target="_self"` 绕过 Vue Router | ❌ 无效 | 水合发生在路由之前，DOM 不匹配先崩溃 |
| 3. 修改 `sd-nav-i18n.js`，在 Vue 水合后 JS 注入 | ❌ 崩溃 | 脚本在 `defer` 阶段运行，仍有竞争条件 |
| 4. 创建独立 JS `sd-dl-extras.js`，延迟注入 | ❌ 不稳定 | 时机不可控，部分页面成功部分崩溃 |
| 5. 只加门户卡片（`sd-home-portal`），不碰侧边栏 | ✅ 可行 | 内容区 DOM 不是 SSR 关键路径 |

### 2.4 结论

**侧边栏无法安全修改** — 这是 VuePress 2 SSR 架构的固有限制。任何对侧边栏 DOM 的增删都会导致水合崩溃。唯一可行的入口是**首页门户卡片**（内容区不参与关键水合）和**直接 URL 访问**。

---

## 3. 最终方案

### 3.1 入口分布

| 入口 | Differential LoRA | DiffSynth Tagger |
|------|-------------------|------------------|
| 首页门户卡片 | ✅ LoRA 训练区 | ✅ 标注工具区 |
| 直接 URL | `/lora/differential-lora.html` | `/tag-edit-leaf.html` |
| 侧边栏 | ❌ 无法添加 | ❌ 无法添加 |

### 3.2 页面类型

两个新增页面均为**独立 HTML 页面**（不引用 VuePress 运行时），因此：
- 不参与 VuePress SPA 路由
- 门户卡片使用 `target="_self"` 触发浏览器原生导航
- 可通过 `curl`、收藏夹、外部链接直接访问

---

## 4. 文件清单

### 4.1 新增文件

```
tools/
├── merge_lora_to_base.py              # Kohya LoRA → 底模融合工具
├── average_lora.py                    # SVD/朴素 LoRA 合并
├── convert_differential_to_comfyui.py # ComfyUI 格式转换
└── differential_tagger/               # DiffSynth 标注器 (14 files)
    ├── main.py                        # CLI 入口
    ├── run.sh                         # 一键启动脚本
    ├── tagger.py                      # WD14 ONNX 标注
    ├── smart_tag.py                   # Smart Tag 流水线
    ├── toriigate_tagger.py            # ToriiGate VLM
    ├── config.py                      # 模型注册表
    ├── download.py                    # HuggingFace 下载
    ├── runtime.py                     # 环境检测
    ├── ai_runtime_guard.py            # VRAM 安全检查
    ├── oppai_oracle_tagger.py         # Oppai Oracle
    └── run_smart_tag.sh

mikazuki/
├── differential_lora/
│   ├── __init__.py                    # 模块导出
│   ├── task_runner.py                 # 核心编排器 (434 行)
│   ├── adapter.py                     # UI 配置 → TOML 转换
│   └── preprocess.py                  # 图片配对、数据集准备
├── app/
│   ├── differential_lora_api.py       # 差分 LoRA REST API
│   └── tag_edit_leaf_api.py           # DiffSynth Tagger REST API
└── schema/
    └── differential-lora.ts           # 前端 Schema（供后续使用）

frontend/dist/
├── lora/
│   └── differential-lora.html         # 差分 LoRA 独立训练页面
├── tag-edit-leaf.html                 # DiffSynth Tagger 独立标注页面
└── assets/
    └── sd-dl-extras.js                # 侧边栏注入脚本（实验性，未启用）

config/
└── differential_lora.toml             # 默认训练配置
```

### 4.2 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `mikazuki/app/application.py` | 注册 `differential_lora_api` 和 `tag_edit_leaf_api` 路由 |
| `frontend/dist/index.html` | 首页新增 Differential LoRA 和 DiffSynth Tagger 门户卡片 |
| `gui.py` | 默认 host 改为 `0.0.0.0`，port 改为 `12345`（本地自定义） |
| `run_gui.sh` | 前台运行 + trap 清理端口 |

---

## 5. API 端点

### 5.1 Differential LoRA

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/differential-lora/run` | 启动差分训练 |
| POST | `/api/differential-lora/pairs` | 预览图片配对 |
| GET | `/api/differential-lora/status` | 查询训练进度 |

### 5.2 DiffSynth Tagger

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/tag-edit-leaf/run` | 启动标注任务 |
| POST | `/api/tag-edit-leaf/scan` | 扫描目录预览图片数 |
| GET | `/api/tag-edit-leaf/status` | 查询标注进度 |
| GET | `/api/tag-edit-leaf/models` | 列出可用标注模型 |

---

## 6. 启动方式

```bash
cd /root/lanyun-tmp/workspace/lora-scripts-next
bash run_gui.sh
# 输出: Frontend: http://0.0.0.0:12346
# Ctrl+C 停止，自动释放端口
```

---

## 7. 后续可能方向

1. **完整 VuePress 集成**：在 `lora-scripts-frontend` 源码仓库中新建 `.md` 源文件和 Vue SFC 组件，执行 `vuepress build` 重新生成整个 `dist/`，这样侧边栏和新页面都能完美集成
2. **内置标注器对接**：将 `tools/differential_tagger/` 的标注逻辑直接在后端 Python 中调用（而非通过 bash 子进程），实现更紧密的集成
3. **Schema 表单迁移**：将独立 HTML 页面的手工表单替换为由 `differential-lora.ts` Schema 驱动的动态表单，与现有训练页面风格统一
