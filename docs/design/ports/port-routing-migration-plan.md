# 当前项目端口全面改造施工计划书

## 依据

本施工计划以 [`port-interface-standard.md`](port-interface-standard.md) 为唯一规范依据。

本文档只面向 lora-scripts-next 当前仓库的端口改造落地，目标是把现有 WebUI、TensorBoard、训练监控页、Dataset Tag Editor、Anima daemon 等服务纳入统一路径路由、服务注册表、健康检查和自动化验收体系。

## 改造目标

完成后必须满足：

```text
用户默认入口：
http://127.0.0.1:28000/

正式服务路径：
/              WebUI
/api/          API
/tensorboard/  TensorBoard
/monitor/      训练监控页
/tagger/       Dataset Tag Editor
```

核心目标：

- 用户无需理解 `6006`、`6008`、`28001` 等子服务端口。
- `28000` 默认被占用时，本地模式自动 fallback，并展示最终入口。
- 显式指定公共入口端口时，端口被占用默认失败，避免 AutoDL 映射错位。
- TensorBoard、训练监控、Tag Editor 的内部端口由主进程统一分配。
- 所有服务写入 `.runtime/services.json`。
- 前端和后端统一读取注册表或统一 URL 生成函数。
- 训练监控不会再连接 `6006 GUI API`。
- `/monitor/` 不会打开 TensorBoard。
- `/tensorboard/` 不会打开训练监控。
- 源码启动和整合包启动都使用同一套端口契约。

## 当前问题定位

### 当前端口来源

| 服务 | 当前默认 | 当前来源 | 问题 |
|---|---:|---|---|
| WebUI / API | `28000` | `gui.py --port`、`MIKAZUKI_PORT` | 默认入口端口，但 fallback 后公共 URL 分散 |
| TensorBoard | `6006` | `gui.py --tensorboard-port`、`MIKAZUKI_TENSORBOARD_PORT` | 可能占用训练监控端口或被旧进程污染 |
| 训练监控 | `6008` | `gui.py --train-monitor-port`、`TRAIN_MONITOR_PORT` | 可能连接错误 GUI API |
| Tag Editor | `28001` | `gui.py run_tag_editor()` 硬编码、`MIKAZUKI_TAGEDITOR_PORT` 读取侧不完整 | WebSocket 地址硬编码 |
| Anima daemon | `8765` | `anima_lora/scripts/daemon/config.py` | 暂未纳入主注册表 |

### 关键硬编码点

必须改造的硬编码点：

| 文件 | 当前问题 | 改造方向 |
|---|---|---|
| `gui.py` | 分散分配 `--port`、`--tensorboard-port`、`--train-monitor-port`，Tag Editor 硬编码 `28001` | 引入统一端口分配和服务注册 |
| `mikazuki/app/proxy.py` | TensorBoard / Tag Editor 读取旧 env，WebSocket 写死 `ws://127.0.0.1:28001/queue/join` | 改为读取服务注册表 |
| `mikazuki/app/application.py` | 浏览器自动打开 `MIKAZUKI_HOST:MIKAZUKI_PORT` 和 `127.0.0.1:{TRAIN_MONITOR_PORT}` | 改为打开公共 URL 和 `/monitor/` |
| `train_monitor/server.py` | `GUI_API = http://127.0.0.1:{MIKAZUKI_PORT}/api`，错误信息误写 `6006 GUI API` | 改为读取 `api.internal_url` |
| `scripts/patch-home-portals.py` | 硬编码 `http://127.0.0.1:6008` | 改为路径或运行时配置 |
| `docs/*.md` | 多处旧端口入口 | 迁移为路径入口 |
| `build-scripts/*` / `scripts/portable/*` | 整合包入口仍假设 `28000` | 读取最终公共入口，日志展示实际端口 |

## 总体设计

### 新增运行时组件

建议新增模块：

```text
mikazuki/app/services.py
```

职责：

- 定义服务 ID、默认路径、默认端口池。
- 分配公共入口端口。
- 分配内部端口。
- 生成公共 URL 和内部 URL。
- 写入和读取 `.runtime/services.json`。
- 提供服务身份健康检查。
- 提供统一 URL 生成函数。

建议新增测试：

```text
tests/test_service_registry.py
tests/test_port_allocator.py
tests/test_service_health.py
tests/test_proxy_routes.py
tests/test_launch_port_contract.py
```

### 服务注册表位置

运行时生成：

```text
.runtime/services.json
```

该目录应加入 `.gitignore`。

注册表最小字段：

```json
{
  "schema_version": 1,
  "gateway": {
    "host": "127.0.0.1",
    "port": 28000,
    "public_base_url": "http://127.0.0.1:28000",
    "port_source": "default"
  },
  "services": {
    "api": {
      "public_path": "/api/",
      "public_url": "http://127.0.0.1:28000/api/",
      "internal_url": "http://127.0.0.1:28100",
      "port": 28100,
      "status": "ok"
    }
  }
}
```

### 启动流程

`gui.py` 改造后的启动顺序：

```text
1. 解析参数和环境变量
2. 判断公共入口端口是否显式指定
3. 分配公共入口端口
4. 分配子服务内部端口
5. 写入初始注册表，状态为 starting
6. 启动 Tag Editor、TensorBoard、训练监控等子服务
7. 等待并执行健康检查
8. 更新注册表服务状态为 ok / failed
9. 启动 FastAPI 公共入口
10. FastAPI 代理路由读取注册表
11. 浏览器打开 public_base_url 和 /monitor/
```

注意：如果 FastAPI 主应用仍然直接绑定公共入口端口，`webui` / `api` 的 internal_url 可以等于公共入口本身或标记为 `in_process`。长期目标是网关和 webui 可分离，但第一阶段不强制拆进程。

## 改造阶段

## 阶段 0：准备和保护

### 修改项

1. 新增 `.runtime/` 到 `.gitignore`。
2. 保留旧参数兼容：
   - `--port`
   - `--host`
   - `--tensorboard-port`
   - `--train-monitor-port`
3. 新增标准参数：
   - `--gateway-host`
   - `--gateway-port`
   - `--auto-public-port`
   - `--public-base-url`
   - `--tag-editor-port`

### 验收

- 旧启动命令仍可执行。
- 新参数可被 argparse 识别。
- 未实际改代理前，行为不应退化。

## 阶段 1：端口分配器

### 设计

新增端口分配器：

```text
allocate_public_port(requested, explicit, auto_public_port)
allocate_internal_port(service_id, requested, pool)
```

规则：

- 未显式指定公共端口时，`28000` 被占用可 fallback 到 `28000-28020`。
- 显式指定公共端口时，被占用默认失败。
- 内部端口只在服务自己的池内 fallback。
- 分配结果必须带 `source`：
  - `default`
  - `default-fallback`
  - `explicit`
  - `explicit-auto-fallback`

### 具体修改

| 文件 | 修改 |
|---|---|
| `mikazuki/app/services.py` | 新增端口池常量、端口检测、分配函数 |
| `gui.py` | 调用端口分配器，替换现有 `ensure_port_available` 的职责 |
| `tests/test_port_allocator.py` | 覆盖默认、显式、fallback、池隔离 |

### 验收

- `28000` 被占用且未显式指定时，返回 `28001-28020` 中可用端口。
- `28000` 被占用且显式指定时，抛出明确错误。
- TensorBoard 不会 fallback 到 `28120-28129`。
- 训练监控不会 fallback 到 `28110-28119`。

## 阶段 2：服务注册表

### 设计

新增服务注册表读写：

```text
write_service_registry(registry)
read_service_registry()
get_service(service_id)
public_url(service_id)
internal_url(service_id)
```

### 具体修改

| 文件 | 修改 |
|---|---|
| `mikazuki/app/services.py` | 新增注册表 dataclass / dict 构造、读写函数 |
| `gui.py` | 启动时生成 `.runtime/services.json` |
| `mikazuki/app/application.py` | 可通过 API 暴露前端服务配置 |
| `tests/test_service_registry.py` | 覆盖 schema、路径、URL、状态 |

建议新增 API：

```text
GET /api/services
```

返回：

```json
{
  "publicBaseUrl": "http://127.0.0.1:28000",
  "services": {
    "tensorboard": "/tensorboard/",
    "trainMonitor": "/monitor/",
    "tagEditor": "/tagger/"
  }
}
```

### 验收

- 启动后存在 `.runtime/services.json`。
- 注册表包含 `webui`、`api`、`tensorboard`、`train-monitor`、`tag-editor`。
- 公共 URL 使用最终公共端口。
- 内部 URL 使用实际分配端口。

## 阶段 3：服务身份健康检查

### 设计

新增统一健康检查：

```text
GET /__lora_service_health
```

自有服务返回服务 ID。

第三方服务适配：

- TensorBoard：请求内部首页或已知接口，识别 TensorBoard 特征。
- Tag Editor：识别 Gradio / Tag Editor 特征。
- Anima daemon：提供或适配状态接口。

### 具体修改

| 文件 | 修改 |
|---|---|
| `mikazuki/app/application.py` | 主应用增加健康检查，返回 `webui` / `api` 信息 |
| `train_monitor/server.py` | 增加 `/__lora_service_health` |
| `mikazuki/app/services.py` | 增加健康检查函数 |
| `tests/test_service_health.py` | 覆盖服务 ID 匹配、不匹配、超时、错服务 |

### 验收

- 训练监控返回 `service=train-monitor`。
- 端口上是 TensorBoard 时，不得注册为 `train-monitor`。
- 健康检查失败时服务状态为 `failed`，公共路径返回明确诊断。

## 阶段 4：路径代理改造

### 设计

正式路径：

```text
/tensorboard/
/monitor/
/tagger/
```

兼容路径：

```text
/proxy/tensorboard/
/proxy/tageditor/
```

兼容路径保留一个小版本，可转发或 `302`。

### 具体修改

| 文件 | 修改 |
|---|---|
| `mikazuki/app/proxy.py` | 从注册表读取上游，新增 `/tensorboard/{path:path}`、`/monitor/{path:path}`、`/tagger/{path:path}` |
| `mikazuki/app/proxy.py` | WebSocket 地址改为根据 `tag-editor.internal_url` 生成 |
| `gui.py` | Tag Editor 启动参数改为 `--root-path /tagger` |
| `train_monitor/server.py` | 如果作为内部服务，支持路径前缀或由代理剥离前缀 |
| `tests/test_proxy_routes.py` | 覆盖普通 HTTP、尾斜杠、WebSocket、SSE |

### TensorBoard 处理

优先方案：

- 使用 TensorBoard 官方路径前缀参数。
- 内部端口为 `28110-28119`。
- 公共路径为 `/tensorboard/`。

如果官方路径前缀不稳定：

- 代理层重写静态资源和重定向。
- 必须加集成测试证明 `/tensorboard/` 可用。

### 验收

- `/tensorboard/` 打开 TensorBoard。
- `/monitor/` 打开训练监控。
- `/tagger/` 打开 Dataset Tag Editor。
- `/monitor/` 不会打开 TensorBoard。
- `/tagger/` WebSocket 可连接。

## 阶段 5：训练监控 API 改造

### 当前问题

`train_monitor/server.py` 当前通过：

```text
MIKAZUKI_PORT -> http://127.0.0.1:<port>/api
```

构造 GUI API，且错误信息可能误写为 `6006 GUI API`。

### 目标

训练监控必须读取：

```text
services["api"].internal_url
```

### 具体修改

| 文件 | 修改 |
|---|---|
| `train_monitor/server.py` | 读取 `.runtime/services.json`，构造 `GUI_API` |
| `train_monitor/server.py` | 增加注册表不可用 fallback，但必须诊断清楚 |
| `tests/test_train_monitor_api_target.py` | 覆盖连接正确 API、禁止连接 TensorBoard 端口 |

### 验收

- 注册表中 API 为 `28100` 时，训练监控请求 `28100/api`。
- TensorBoard 占用 `6006` 时，训练监控不会访问 `6006/api`。
- 错误信息不再出现“连接 6006 GUI API”。

## 阶段 6：浏览器自动打开和前端入口改造

### 具体修改

| 文件 | 修改 |
|---|---|
| `mikazuki/app/application.py` | `_start_url()` 使用注册表 `webui.public_url` |
| `mikazuki/app/application.py` | 自动打开训练监控使用 `train-monitor.public_url` |
| `scripts/patch-home-portals.py` | 移除 `http://127.0.0.1:6008`，改为 `/monitor/` |
| 前端 dist 生成脚本 | 不再写入历史端口链接 |

### 验收

- 默认启动打开最终公共 URL。
- `28000` fallback 到 `28002` 时浏览器打开 `28002`。
- 训练监控入口为 `/monitor/`。
- 前端构建产物不包含 `127.0.0.1:6008`、`127.0.0.1:6006`、`127.0.0.1:28001` 作为用户入口。

## 阶段 7：文档和启动脚本同步

### 具体修改

| 文件 | 修改 |
|---|---|
| `docs/cli-args.md` | 增加新参数，标注旧参数兼容 |
| `docs/train-monitor.md` | 入口改为 `/monitor/` |
| `docs/docker.md` | 只要求映射公共入口端口 |
| `docs/autodl-deploy.md` | 用 `--public-base-url` 和公共入口端口 |
| `README.md` / `README-zh.md` | 用户入口改为单入口 + 路径 |
| `start_autodl.sh` / `scripts/autodl/start_lora_next.sh` | 显式公共入口和公共 URL 规则 |
| `scripts/portable/launch_portable.bat` | 保留兼容参数，但日志展示最终公共入口 |

### 验收

- 用户文档不再把 `6006`、`6008`、`28001` 作为普通入口。
- AutoDL 文档明确只映射公共入口端口。
- 整合包文档说明端口 fallback 后看启动日志。

## 阶段 8：自动化验收

### 测试矩阵

| 编号 | 场景 | 验收 |
|---|---|---|
| PRT-001 | 默认启动，端口均空闲 | `/`、`/monitor/`、`/tensorboard/`、`/tagger/` 均可用 |
| PRT-002 | `28000` 被占用，未显式指定端口 | 自动 fallback，注册表和浏览器 URL 一致 |
| PRT-003 | `28000` 被占用，显式指定 `--gateway-port 28000` | 启动失败且诊断明确 |
| PRT-004 | `6006` 被占用 | TensorBoard 使用 `28110-28119`，`/tensorboard/` 可用 |
| PRT-005 | `6008` 被占用 | 训练监控使用 `28120-28129`，`/monitor/` 可用 |
| PRT-006 | `28001` 被占用 | Tag Editor 使用 `28130-28139`，`/tagger/` 和 WebSocket 可用 |
| PRT-007 | TensorBoard 占用训练监控候选端口 | 健康检查识别错服务，`/monitor/` 不打开 TensorBoard |
| PRT-008 | 训练监控连接 API | 使用 `api.internal_url`，不访问 `6006/api` |
| PRT-009 | `/tensorboard` 无尾斜杠 | `302` 到 `/tensorboard/` |
| PRT-010 | `/monitor` 无尾斜杠 | `302` 到 `/monitor/` |
| PRT-011 | `/tagger` 无尾斜杠 | `302` 到 `/tagger/` |
| PRT-012 | `/tagger/` WebSocket | Gradio queue WebSocket 通过公共入口连接 |
| PRT-013 | `/api/train/log/stream/{task_id}` | SSE 可持续读取 |
| PRT-014 | AutoDL 显式公共 URL | 启动日志和前端配置使用 `LORA_PUBLIC_BASE_URL` |
| PRT-015 | 子服务启动失败 | 公共路径返回明确 502，不伪装成其他服务 |
| PRT-016 | 源码启动 | 生成注册表并通过路径入口访问 |
| PRT-017 | 整合包启动 | 使用同一注册表契约，浏览器打开最终公共 URL |

### 自动化建议

1. 单元测试端口分配器，不启动真实服务。
2. 单元测试注册表读写和 URL 拼接。
3. 使用轻量假服务模拟 TensorBoard、训练监控、Tag Editor。
4. 使用 Playwright 或 httpx 做端到端路由测试。
5. CI 中避免依赖 GPU，只测端口契约和代理。

## 阶段 9：实测清单

### 本地 Windows 源码启动

命令：

```powershell
python gui.py --skip-prepare-environment
```

检查：

- 控制台输出最终公共入口。
- `.runtime/services.json` 存在。
- 浏览器打开 `/`。
- `/monitor/` 可用。
- `/tensorboard/` 可用。
- `/tagger/` 可用。

### 本地端口占用实测

占用端口后启动：

```powershell
# 示例：占用 28000
python -m http.server 28000
```

另一个终端启动：

```powershell
python gui.py --skip-prepare-environment
```

检查：

- 公共入口 fallback 到 `28001-28020`。
- 启动日志、注册表、浏览器打开地址一致。

### 显式端口严格模式实测

```powershell
python gui.py --gateway-port 28000 --skip-prepare-environment
```

当 `28000` 被占用时应失败，并给出明确错误。

### AutoDL 实测

示例：

```bash
python gui.py \
  --gateway-host 0.0.0.0 \
  --gateway-port 6006 \
  --public-base-url "$AUTODL_PUBLIC_URL" \
  --skip-prepare-environment
```

检查：

- 日志展示平台 URL。
- 只需要映射公共入口端口。
- `/monitor/`、`/tensorboard/`、`/tagger/` 均通过平台 URL 访问。

### 整合包实测

双击启动脚本后检查：

- 不闪退。
- 日志文件包含最终公共入口。
- 浏览器打开最终公共入口。
- 子服务失败时日志可诊断。

## 发布门禁

发布前必须满足：

- PRT-001 到 PRT-017 全部通过，无法自动化的项有人工实测记录。
- `rg "127\\.0\\.0\\.1:6008|127\\.0\\.0\\.1:6006|127\\.0\\.0\\.1:28001" frontend docs scripts mikazuki train_monitor` 没有普通用户入口残留。
- `.runtime/services.json` 不进入 git。
- 新参数已写入 CLI 文档。
- README 和 AutoDL 文档已更新。
- Windows 源码启动和整合包启动均完成实测。

## 回滚策略

如果某阶段引入严重回归：

1. 保留服务注册表和端口分配器代码，但临时关闭新路径代理。
2. 恢复 `/proxy/tensorboard/`、`/proxy/tageditor/` 旧路径。
3. 保留旧端口诊断入口。
4. 不回滚公共入口 fallback 和注册表日志，因为它们对排查有价值。
5. 标记失败阶段，补齐测试后再继续。

## 风险和对策

| 风险 | 对策 |
|---|---|
| TensorBoard 路径前缀不稳定 | 优先官方参数；不稳定时使用代理重写并增加集成测试 |
| Gradio WebSocket 代理失败 | 单独测试 `/tagger/` queue WebSocket；保留旧路径兼容一版 |
| 旧环境变量污染启动 | 启动日志输出所有最终生效值和来源 |
| AutoDL 固定端口被静默 fallback | 显式端口默认严格失败 |
| 子服务慢启动导致误判失败 | 健康检查重试有限次数，状态先标记 `starting` |
| 前端构建产物仍有旧端口 | 发布门禁使用 `rg` 扫描 |
| 训练监控错连 API | 强制使用 `api.internal_url`，测试禁止访问 `6006/api` |

## 最终交付物

完成改造后应交付：

- `mikazuki/app/services.py` 或等价服务注册模块。
- `.runtime/services.json` 运行时注册表。
- `/api/services` 前端配置接口。
- `/tensorboard/`、`/monitor/`、`/tagger/` 正式路径。
- `/__lora_service_health` 自有服务健康检查。
- PRT-001 到 PRT-017 自动化或实测记录。
- 更新后的 README、CLI、AutoDL、Docker、训练监控文档。
