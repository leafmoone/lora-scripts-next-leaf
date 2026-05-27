# 端口与路径路由（设计文档）

面向维护者、贡献者与 Agent 的**目标架构**说明，不列入用户文档索引，README 主页亦不放传送门。

> **状态**：三份正文由 [@MikumikuDAIFans](https://github.com/MikumikuDAIFans) 整理的**方案草案**；**执行顺序以 [Discussion #53](https://github.com/wochenlong/lora-scripts-next/discussions/53) 团队回复为准**（2026-05-26），全项目排期见 [`docs/team/backlog-priorities.md`](../../team/backlog-priorities.md)。

**实现以当前代码与** [`docs/train-monitor.md`](../../train-monitor.md)、[`docs/cli-args.md`](../../cli-args.md) **为准**；本目录为规划中的统一端口契约（如 `/monitor/`、服务注册表等），可能与 main 不一致。

## Discussion #53 已对齐的执行顺序（摘要）

[@MikumikuDAIFans](https://github.com/MikumikuDAIFans) 在 [#53](https://github.com/wochenlong/lora-scripts-next/discussions/53) 表态：

- **待确定事项 1～6**：认同（单公共入口 + 路径路由、最小注册表、AutoDL 严格端口等）。
- **待确定事项 7**（`rg` 发布门禁）：仅作 **P0 全部完成前** 的临时策略。
- **顺序**：**注册表先行（P0-6、P0-7）→ 唯一外部入口稳定（P0-8、P0-9）→ 再逐步内化子服务端口（P0-10～13，建议整体作为 P1）**。
- **P0-7 前**：训练监控仍主要靠 env / 探测 / 默认端口；**P0-7 后**：优先从注册表读 API，端口可查找、可定位。

| 文档 | 内容 |
|------|------|
| [port-interface-standard.md](port-interface-standard.md) | 长期规范：路径、服务 ID、注册表、健康检查 |
| [port-routing-migration-plan.md](port-routing-migration-plan.md) | 当前仓库改造施工计划 |
| [port-routing-priority-roadmap.md](port-routing-priority-roadmap.md) | 草案中的 P0 / P1 / P2；**实施时以 #53 与 backlog 为准** |

Agent 当前硬规则见 `.cursor/rules/embedded-service-ports.mdc`（以已实现行为为准）。
