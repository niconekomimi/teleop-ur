# 当前状态

更新时间：2026-05-06

本文给维护者和 AI 快速了解 `teleop_control` 当前状态。正式架构和运行说明以 `docs/guide/` 为准。

## 项目范围

`teleop_control` 作为完整 workspace 维护，不只包含 `teleop_control_py` 主控包。

当前维护范围包括：

- `src/teleop_control_py/` 主控 ROS 2 包
- `src/Universal_Robots_ROS2_Driver/` UR driver 与 MoveIt 相关包
- `src/robotiq_2f_gripper_ros2/` Robotiq 夹爪包
- `src/qbsofthand_control/` qbSoftHand 控制包
- `Real_IL/` 本地推理依赖
- `openpi/` 远端推理依赖
- `scripts/` workspace 工具脚本
- `data/`、`models/`、`udev/` 等运行资产和设备配置

## 已形成的能力

- GUI 是主要入口，负责启动机器人驱动、遥操作、采集和推理流程。
- `ControlCoordinator + SystemOrchestrator + ActionMux` 已经承接 teleop 和 GUI 推理执行链路中的控制规则。
- `data_collector_node` 负责相机采样、状态同步和 HDF5 写盘。
- `robot_commander_node` 负责 Home、Home Zone 和 controller 切换。
- Quest 3 已接入为正式输入后端。
- 推理支持本地 `Real_IL` 和远端 `openpi_remote` 路径。
- 预览录屏与 HDF5 数据集录制是两条独立链路。

## 需要注意的边界

- 推理执行动作下发目前仍在 GUI 侧 `ROS2Worker` 内闭环，不是独立 ROS 节点。
- `teleop_control_node`、`robot_commander_node`、`data_collector_node` 和 GUI 推理执行分布在不同运行时。
- `robot_profiles.yaml` 已是底层 topic、service、controller、Home 和 gripper 默认值的主要配置来源，但 launch 级外部包组合仍在 launch 文件中。
- `ESTOP` 覆盖项目内部状态机和推理执行链路，不等同于完整硬件安全系统。
- 相机链路按采集、MediaPipe、推理和 GUI 预览分开维护。

## 文档结构

- `docs/guide/`：给人看的正式项目文档
- `docs/agent/`：给 AI 和开发连续性看的维护上下文
- 根目录 `README.md`：项目主入口，只保留概览、快速开始和关键文档链接

## 接手建议

接手项目前先读：

1. `docs/guide/00-overview.md`
2. `docs/guide/01-architecture.md`
3. `docs/guide/02-runtime-boundaries.md`
4. `docs/agent/01-todo.md`
5. `docs/agent/04-handoff.md`
