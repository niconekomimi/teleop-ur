# 交接说明

更新时间：2026-05-06

本文用于下一位维护者或 AI 快速接手 `teleop_control`。

## 先读这些

1. `docs/guide/00-overview.md`
2. `docs/guide/01-architecture.md`
3. `docs/guide/02-runtime-boundaries.md`
4. `docs/guide/04-control-flow.md`
5. `docs/agent/00-current-state.md`
6. `docs/agent/01-todo.md`

## 当前关注点

- 文档结构已经切成 `guide/` 和 `agent/`。
- 项目名保持 `teleop_control`，文档按完整 workspace 说明。
- README 保持简洁，不承载详细维护说明。
- TODO 由人工和 AI 共用，人工可通过 `Order` 控制任务顺序。

## 改动前确认

修改代码或文档前，先确认改动属于哪个区域：

- 控制仲裁：`docs/guide/04-control-flow.md`
- 运行角色：`docs/guide/02-runtime-boundaries.md`
- 采集：`docs/guide/05-data-collection.md`
- 推理：`docs/guide/06-inference.md`
- 设备和输入：`docs/guide/07-devices.md`
- 配置：`docs/guide/08-configuration.md`

## 保持一致

- 改 ROS 节点、launch 参数、topic、service 或 controller 时，同步更新 `docs/guide/`。
- 新增任务、阶段性目标或待办时，同步更新 `docs/agent/01-todo.md`。
- 做出会影响长期维护方式的选择时，同步更新 `docs/agent/02-decisions.md`。
- 工作完成后，在 `docs/agent/03-work-log.md` 记录摘要。
