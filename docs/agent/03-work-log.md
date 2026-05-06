# 工作记录

更新时间：2026-05-06

本文记录维护摘要，不记录聊天流水账。每条记录保留背景、改动和后续事项。

## 2026-05-06 文档结构重组

背景：

- README 和 docs 原本更像单包说明，不能完整覆盖 `teleop_control` workspace。
- 正式项目文档和 AI 维护上下文混在一起，不利于长期维护。

改动：

- 将正式文档迁移到 `docs/guide/`。
- 新增 `docs/agent/`，用于当前状态、TODO、决策和交接。
- 将 Quest 专篇扩展为 `07-devices.md`，覆盖机械臂、夹爪、相机和输入。
- 新增 `10-troubleshooting.md` 作为排查入口。
- 根 README 改为 workspace 级项目入口。

后续：

- 根据实际运行经验补充 troubleshooting。
- 继续核对 `guide/` 与代码实现是否完全一致。
- 按 `01-todo.md` 维护后续任务顺序。
