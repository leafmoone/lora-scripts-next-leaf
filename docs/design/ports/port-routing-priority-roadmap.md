# 端口治理优先级路线图

## 目的

本文档用于把 [`port-routing-migration-plan.md`](port-routing-migration-plan.md) 拆成可执行优先级，避免一次性改造范围过大。

优先级定义：

| 优先级 | 目标 | 发布要求 |
|---|---|---|
| P0 | 关闭当前用户可见端口错连、跳错、无法访问问题 | 必须在最近一次端口治理迭代完成 |
| P1 | 建立稳定、可验证、源码和整合包一致的端口治理体系 | 应在下一阶段版本完成 |
| P2 | 面向未来插件、云端、多实例、安全和观测能力扩展 | 可分批持续推进 |

## P0：关闭当前核心故障

### 目标

P0 只解决当前 `issues.txt` 中影响训练体验的核心问题：

- 训练监控入口不得打开 TensorBoard。
- TensorBoard 入口不得打开训练监控。
- 训练监控不得再连接 `6006 GUI API`。
- `28000`、`6006`、`6008`、`28001` 被占用时，核心训练流程仍能启动。
- 用户通过主页路径入口访问监控、TensorBoard、Dataset Editor，不需要理解多个端口。

### 必做改动

| 模块 | 必做内容 |
|---|---|
| 端口分配 | 实现公共入口默认 fallback；实现 TensorBoard、训练监控、Tag Editor 独立内部端口池 |
| 注册表 | 生成 `.runtime/services.json`，至少包含 `api`、`tensorboard`、`train-monitor`、`tag-editor` |
| 训练监控 | `train_monitor/server.py` 读取 `api.internal_url`，禁止继续从 `6006` 推断 API |
| 路径入口 | 提供 `/monitor/`、`/tensorboard/`、`/tagger/` |
| 代理 | `mikazuki/app/proxy.py` 从注册表读取上游；Tag Editor WebSocket 不再写死 `28001` |
| 浏览器打开 | 自动打开公共入口和 `/monitor/`，不打开 `6008` |
| 前端入口 | 主页训练监控链接改为 `/monitor/`，TensorBoard 改为 `/tensorboard/` |

### P0 验收

必须通过：

| 编号 | 场景 | 期望 |
|---|---|---|
| PRT-001 | 默认启动，端口均空闲 | `/`、`/monitor/`、`/tensorboard/`、`/tagger/` 均可用 |
| PRT-002 | `28000` 被占用，未显式指定端口 | 自动 fallback，注册表和浏览器 URL 一致 |
| PRT-004 | `6006` 被占用 | TensorBoard 使用 `28110-28119`，`/tensorboard/` 可用 |
| PRT-005 | `6008` 被占用 | 训练监控使用 `28120-28129`，`/monitor/` 可用 |
| PRT-006 | `28001` 被占用 | Tag Editor 使用 `28130-28139`，`/tagger/` 和 WebSocket 可用 |
| PRT-007 | TensorBoard 占用训练监控候选端口 | `/monitor/` 不得打开 TensorBoard |
| PRT-008 | 训练监控连接 API | 使用 `api.internal_url`，不访问 `6006/api` |

### P0 非目标

P0 不要求一次性完成：

- 插件完整治理。
- Anima daemon 全量迁移。
- 完整权限认证体系。
- 云端多实例隔离。
- 所有历史文档全面清理。

## P1：建立稳定端口契约

### 目标

P1 把 P0 的热区修复升级为统一工程能力：

- 所有启动路径共用同一端口分配器。
- 源码启动和整合包启动都写入同一服务注册表。
- 自有服务提供 `/__lora_service_health`。
- 路径代理覆盖 HTTP、SSE、WebSocket。
- 文档、CLI、AutoDL、Docker、整合包脚本同步更新。
- 自动化测试覆盖端口契约。

### 必做改动

| 模块 | 必做内容 |
|---|---|
| 服务模块 | 新增 `mikazuki/app/services.py` 或等价模块 |
| 健康检查 | WebUI / API、训练监控实现 `/__lora_service_health` |
| 第三方适配 | TensorBoard、Gradio 实现身份适配检查 |
| CLI | 增加标准参数并保留旧参数兼容 |
| 文档 | 更新 README、CLI、Docker、AutoDL、训练监控文档 |
| 测试 | 新增端口分配器、注册表、健康检查、代理路由测试 |
| 发布门禁 | 扫描前端和文档中旧端口普通入口残留 |

### P1 验收

必须通过：

| 编号 | 场景 | 期望 |
|---|---|---|
| PRT-003 | `28000` 被占用，显式指定 `--gateway-port 28000` | 启动失败且诊断明确 |
| PRT-009 | `/tensorboard` 无尾斜杠 | `302` 到 `/tensorboard/` |
| PRT-010 | `/monitor` 无尾斜杠 | `302` 到 `/monitor/` |
| PRT-011 | `/tagger` 无尾斜杠 | `302` 到 `/tagger/` |
| PRT-012 | `/tagger/` WebSocket | Gradio queue WebSocket 可通过公共入口连接 |
| PRT-013 | `/api/train/log/stream/{task_id}` | SSE 可持续读取 |
| PRT-015 | 子服务启动失败 | 公共路径返回明确 502，不伪装成其他服务 |
| PRT-016 | 源码启动 | 生成注册表并通过路径入口访问 |
| PRT-017 | 整合包启动 | 使用同一注册表契约，浏览器打开最终公共 URL |

### P1 发布门禁

发布前必须满足：

```text
rg "127\\.0\\.0\\.1:6008|127\\.0\\.0\\.1:6006|127\\.0\\.0\\.1:28001" frontend docs scripts mikazuki train_monitor
```

扫描结果不得包含普通用户入口残留。允许存在于迁移说明、禁止示例、测试 fixture 中，但必须有明确语义。

## P2：面向长期扩展

### 目标

P2 面向未来新功能、新插件、云端部署和运维能力：

- 插件服务注册和端口池申请流程。
- API 版本治理。
- 多实例和任务级路径隔离。
- 权限认证和跨域策略。
- 统一诊断页面、日志、指标和追踪。
- 完整 deprecation 生命周期。

### 建议能力

| 能力 | 内容 |
|---|---|
| 插件注册 | 插件声明服务 ID、路径、端口池、健康检查、能力清单 |
| API 版本 | 新公共 API 使用 `/api/v1/`；旧 API 保留兼容层 |
| 多实例 | 支持 `/instances/{id}/...` 或任务级路径，但不得引入子域名要求 |
| 权限 | 公网部署保护 `/api/`、`/monitor/`、`/tagger/`、`/daemon/` |
| 观测 | 请求 ID、服务状态、端口来源、健康检查结果统一日志 |
| 诊断 | `/api/services`、`/api/diagnostics/ports` 等诊断接口 |
| 废弃 | 历史路径和旧变量必须标注版本、替代方案、移除时间 |

### P2 验收

P2 不要求一次性交付，但每个新能力合入时必须满足：

- 符合 [`port-interface-standard.md`](port-interface-standard.md)。
- 有服务注册记录。
- 有健康检查。
- 有端口冲突测试。
- 有路径代理测试。
- 有用户文档或开发者文档。

## 推荐执行顺序

```text
P0-1  端口分配器
P0-2  服务注册表最小实现
P0-3  训练监控读取 api.internal_url
P0-4  /monitor/、/tensorboard/、/tagger/ 路径入口
P0-5  WebSocket 和主页链接修正
P0-6  PRT-001/002/004/005/006/007/008 验收

P1-1  健康检查标准端点
P1-2  第三方服务身份适配
P1-3  CLI / 文档 / 整合包脚本同步
P1-4  自动化验收矩阵补齐
P1-5  发布门禁

P2-1  插件注册规范落地
P2-2  API 版本和诊断接口
P2-3  云端安全和观测增强
```

## 完成定义

### P0 Done

当前用户可见端口问题关闭，核心训练流程不再受 `6006/6008/28001` 错连影响。

### P1 Done

端口契约成为源码启动和整合包启动的共同基础，并由自动化测试保护。

### P2 Done

未来新功能和插件不再需要重新讨论端口策略，只需按标准注册服务、路径和健康检查。
