# TODO

更新时间：2026-05-06

本文是人工和 AI 共用的任务队列。`Order` 用于人工控制执行顺序，数值越小越靠前；`Priority` 表示重要程度。

## 状态约定

| Status | 含义 |
| --- | --- |
| `todo` | 尚未开始 |
| `doing` | 正在处理 |
| `blocked` | 被外部条件阻塞 |
| `done` | 已完成 |
| `dropped` | 已放弃或不再适用 |

## 任务队列

| Order | ID | Status | Priority | Owner | Area | Task | Acceptance |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 10 | T-001 | todo | P0 | human | docs | 确认 `docs/guide/` 与 `docs/agent/` 的最终文档结构 | 根 README、guide 和 agent 入口职责清楚 |
| 20 | T-002 | todo | P1 | both | docs | 按实际运行经验补充 troubleshooting | 常见硬件、相机、Quest、采集和推理问题都有排查入口 |
| 30 | T-003 | todo | P1 | both | inference | 梳理 `openpi_remote` 的运行边界和失败处理 | `06-inference.md` 写清启动、配置、连接失败和动作执行边界 |
| 40 | T-004 | todo | P2 | both | control | 评估推理执行是否需要独立 ROS 节点 | 有明确结论、迁移步骤或保持现状的理由 |
| 50 | T-005 | todo | P2 | human | devices | 根据实际硬件补齐设备接线和网络说明 | `07-devices.md` 覆盖当前实验台常用接线和 IP 约定 |

## 使用方式

- 人工可以直接调整 `Order` 控制执行顺序。
- 新功能、修复、文档任务都可以放在同一张表里。
- AI 开始任务前应先确认对应 `Acceptance`。
- 完成后把 `Status` 改为 `done`，必要时在 `03-work-log.md` 记录摘要。
