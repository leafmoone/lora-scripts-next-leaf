# 前端定制

## 默认行为

训练 GUI 后端默认加载 `frontend/dist`。该目录是 `hanamizuki-ai/lora-gui-dist` 这个预编译 submodule，并不是前端源码——本仓库内没有 `package.json` 也没有前端构建步骤。

"Anima LoRA" 页面并不在 dist 里，而是由 `mikazuki/schema/sd3-lora.ts` 渲染出来的：本 fork 把这份 schema 改写成了 Anima 配置，后端把它喂给原版 UI，表单就跟着重新渲染了。

## 自定义前端

如果想接入自己另外打包的前端 dist，设置环境变量即可：

```bash
MIKAZUKI_FRONTEND_DIST=/path/to/your/dist python gui.py --listen
```

也可以把 `frontend` submodule 的 URL 改到自己的 dist 仓库。后端不会自动从前端源码构建 UI。
