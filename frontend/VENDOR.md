# `frontend/dist/` 来源说明

本目录原本是 git submodule，现已 vendored 进主仓库。这份说明用于追溯当前文件从哪来、改了什么、以后怎么同步上游。

## 上游来源

- **仓库**：[`hanamizuki-ai/lora-gui-dist`](https://github.com/hanamizuki-ai/lora-gui-dist)
- **基线 commit**：`20513393bfdd9ee897c538cf68d478c95fcde6c1`（2025-09-08，标题 "update"，目前是 master 的最新且唯一 tip）
- **上游所有者**：花水木 AI（hanamizuki-ai org）—— 秋叶（@Akegarasu）生态下负责 dist 中转的姊妹组织
- **上游来源源码**：秋叶团队的前端源码仓库 `Akegarasu/lora-scripts-frontend`，build 后的产物 push 到 `lora-gui-dist`
- **本仓库写权限**：无。所以本 fork 选择 vendor 而不是继续用 submodule

## 为什么 vendor 而不是 submodule

- 上游 dist 仓库已 7 个月未更新（截至 vendor 时），近似冻结
- 本 fork 需要修改若干 UI 文字以匹配 Anima LoRA 训练流程，没有写权限就只能 vendor
- 主仓库改成 vendored 后，`git clone` 不再需要 `--recurse-submodules`
- dist 总体积 ~1.6 MB，对主仓库无明显负担

## 本仓库对 dist 的 patch 列表

所有 patch 都在 commit [`23337dd`](../../../commit/23337dd)，目的是把上游写死的 SD3 字面量替换为 Anima LoRA，让 UI 文字与 `mikazuki/schema/sd3-lora.ts` 已经改写成的 Anima schema 对齐。

| 文件 | 改前 | 改后 |
|---|---|---|
| `assets/app.547295de.js`（sidebar 配置） | `{"text":"SD3.5","link":"/lora/sd3.md"}` | `{"text":"Anima LoRA","link":"/lora/sd3.md"}` |
| `assets/sd3.html.1a4bf31e.js`（页面 H1） | `SD3 训练 专家模式` | `Anima LoRA 训练 专家模式` |
| `assets/sd3.html.1a4bf31e.js`（副标题） | `SD3 模型 LoRA 训练 专家模式` | `Anima DiT 模型 LoRA 训练 专家模式` |
| `assets/sd3.html.1a4bf31e.js`（介绍段） | `支持 SD3.5 模型的 LoRA 训练` | `Anima DiT 训练入口，使用 Qwen3 + T5 + Anima 专用参数` |
| `assets/sd3.html.1a4bf31e.js`（多余段落） | `别问为什么新手模式不行，问就是你都用 SD3 了还想当新手？` | 删除该段，并把渲染数组 `l=[c,n,d,r]` 同步缩成 `l=[c,n,d]` |
| `assets/sd3.html.eaeb05e1.js`（页面 metadata） | `"title":"SD3 训练 专家模式"` | `"title":"Anima LoRA 训练 专家模式"` |

### 故意不动的字面量

| 字面量 | 位置 | 原因 |
|---|---|---|
| `/lora/sd3.html` | URL 路由 | SPA 路由 key，改了会让所有指向 sd3 页面的链接 404 |
| `"trainType":"sd3-lora"` | `sd3.html.eaeb05e1.js` frontmatter | 后端 schema 路由 key，对应 `mikazuki/schema/sd3-lora.ts` |
| `id:"sd3-训练-专家模式"` 这类锚点 id | sd3.html chunk | 不影响显示，保留以兼容历史 anchor 链接 |

## 怎么重新 vendor 上游（如果上游恢复更新）

```powershell
# 1. 备份当前 dist
Copy-Item frontend\dist _dist_backup -Recurse

# 2. 拉新版 dist
git clone --depth 1 https://github.com/hanamizuki-ai/lora-gui-dist /tmp/new-dist

# 3. 用新版 dist 覆盖（保留 VENDOR.md）
Move-Item frontend\VENDOR.md _vendor_md_backup
Remove-Item frontend\dist -Recurse -Force
Copy-Item /tmp/new-dist frontend\dist -Recurse
Move-Item _vendor_md_backup frontend\VENDOR.md

# 4. 重新应用上面表格里的 6 处文字 patch（手工或脚本）

# 5. 更新本文件的"基线 commit"和"patch 列表"

# 6. 提交
git add frontend/
git commit -m "vendor: refresh frontend/dist from hanamizuki-ai/lora-gui-dist <NEW_SHA>"
```

## 致谢

`frontend/dist/` 中的全部静态资源版权与许可归原作者所有，详见上游 [`hanamizuki-ai/lora-gui-dist`](https://github.com/hanamizuki-ai/lora-gui-dist) 与 [`Akegarasu/lora-scripts-frontend`](https://github.com/Akegarasu/lora-scripts-frontend)。本仓库仅做最小 patch（见上表）以适配 Anima LoRA 训练入口的命名。
