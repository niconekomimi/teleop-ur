# 项目概览

更新时间：2026-05-06

本文是正式项目文档的入口。它从整个 `teleop_control` 工作区说明项目组成、文档阅读顺序和维护边界。

## 项目定位

`teleop_control` 面向真实机器人遥操作、示教数据采集和在线推理执行。日常使用以 GUI 为主入口，底层运行依赖 ROS 2 launch、UR driver、MoveIt Servo、夹爪驱动、相机链路和模型推理仓库。

`src/teleop_control_py/` 是主控包，但不是项目的全部。维护项目时需要同时关注主控包、外部 ROS 包、本地模型仓库、工具脚本、设备规则和本地数据目录。

## 工作区组成

| 路径 | 作用 | 维护方式 |
| --- | --- | --- |
| `src/teleop_control_py/` | 主控 ROS 2 包，包含 GUI、控制、采集、推理桥接和配置 | 日常主要维护 |
| `src/Universal_Robots_ROS2_Driver/` | UR driver、controller、dashboard、MoveIt 配置 | 优先使用上游能力，通过本项目配置和 launch 适配 |
| `src/robotiq_2f_gripper_ros2/` | Robotiq 2F 夹爪驱动和消息包 | 优先少量适配 |
| `src/qbsofthand_control/` | qbSoftHand 控制包 | 按硬件需求维护 |
| `Real_IL/` | 本地模仿学习推理仓库 | 作为模型依赖维护 |
| `openpi/` | 远端 openpi 推理相关仓库 | 作为模型依赖维护 |
| `scripts/` | workspace 级工具脚本 | 与运行和维护流程同步更新 |
| `data/` | 本地数据集、预览录屏、推理日志 | 运行产物，不作为正式源码文档来源 |
| `models/` | 本地模型和权重目录 | 运行资产 |
| `udev/` | 设备规则 | 跟随硬件接入维护 |

## 文档分区

`docs/guide/` 是给维护者和使用者看的正式文档。语气保持稳定，内容以当前代码和可运行流程为准。

`docs/agent/` 是给 AI 和开发连续性看的维护上下文。它记录当前状态、任务队列、决策和交接信息，不替代正式项目文档。

## 阅读顺序

第一次接触项目时，建议按下面顺序阅读：

1. [00-overview.md](00-overview.md)：workspace 组成和文档入口
2. [01-architecture.md](01-architecture.md)：系统由哪些进程、节点和模块组成
3. [02-runtime-boundaries.md](02-runtime-boundaries.md)：每个运行角色负责什么，功能边界在哪里
4. [03-operation.md](03-operation.md)：如何通过 GUI 或命令行启动系统
5. [04-control-flow.md](04-control-flow.md)：控制权、状态机、Home / Home Zone 和推理执行规则
6. [05-data-collection.md](05-data-collection.md)：HDF5 数据集、采集节点和预览链路
7. [06-inference.md](06-inference.md)：本地 `Real_IL` 和远端 `openpi` 推理后端
8. [07-devices.md](07-devices.md)：机械臂、夹爪、相机和输入设备
9. [08-configuration.md](08-configuration.md)：配置文件如何分工，哪些值应该改在哪里
10. [09-maintenance.md](09-maintenance.md)：系统边界、维护约定和文档同步规则
11. [10-troubleshooting.md](10-troubleshooting.md)：常见问题排查入口

AI 或维护交接时，先读：

1. [../agent/00-current-state.md](../agent/00-current-state.md)
2. [../agent/01-todo.md](../agent/01-todo.md)
3. [../agent/04-handoff.md](../agent/04-handoff.md)

## 文档命名约定

正式文档文件名使用小写英文和连字符，例如 `04-control-flow.md`。文件名不带版本号，不带 `current` 或 `analysis` 这类阶段性词汇；文档内部用“更新时间”和正文说明表达状态。

如果后续需要保留历史版本，优先使用 git 历史，不在文件名里追加 `v0.2`、`final`、`new` 之类后缀。

## 维护约定

- 根目录 `README.md` 只回答“这是什么、怎么开始、去哪里看细节”。
- `docs/guide/` 记录架构、运行流程、配置职责和维护说明。
- `docs/agent/` 记录开发连续性、任务顺序和交接状态。
- 修改 launch 参数、ROS topic / service、配置文件职责、HDF5 schema、GUI 启动流程或设备接入方式时，同步更新对应文档。
- 文档以可运行代码为准。写不确定内容时，说明适用范围、限制或待收敛事项。
