# 系统边界与维护说明

更新时间：2026-05-06

本文记录项目已经形成的结构、仍需注意的运行边界，以及修改功能时的文档同步规则。它用于帮助维护者快速判断改动影响范围。

## 已形成的结构

- `robot_profiles.yaml` 已成为机械臂、夹爪、ROS 接口、controller、Home 和 Home Zone 默认值的主配置。
- `ControlCoordinator + SystemOrchestrator + ActionMux` 已经在 teleop 和 GUI 推理执行链路中使用。
- 采集侧复用 `SyncHub + RecorderService`，数据集写盘由采集链路负责。
- 推理 worker 生命周期由 `core/inference_service.py` 管理。
- Home 点运行时覆盖从 `gui_params.yaml` 迁到了 `home_overrides.yaml`。
- `robot_commander_params.yaml` 已从启动链路移除。
- Quest 3 已经接入为正式输入后端，并进入 launch / teleop 主链路。
- 预览刷新频率与推理动作频率解耦，预览录屏独立成本地输出链路。
- 推理后端增加了 `openpi_remote`，但本地仍负责相机、机器人状态和动作执行。

## 运行边界

### 控制运行时仍分布在多个位置

控制职责仍分布在多个位置：

- `teleop_control_node`：人工遥操作闭环
- `robot_commander_node`：Home / Home Zone
- `ROS2Worker`：推理动作执行闭环
- `data_collector_node`：采样和写盘闭环

项目已经有统一控制规则，但目前尚未由单一 controller node 承载所有控制动作。

### `robot_profiles.yaml` 未完全接管 launch 级驱动定义

`robot_profiles.yaml` 已接管 topic、service、controller、Home、Home Zone 和 gripper 默认值，但 UR driver、MoveIt 和上游 gripper launch 的部分参数仍在 `control_system.launch.py` 中装配。

配置建议：

- 底层接口默认值放到 `robot_profiles.yaml`
- 启动组合和外部包 include 逻辑留在 launch 文件
- GUI 默认偏好不作为底层接口的默认来源

### 相机驱动仍有多条运行路径

相机相关链路包括：

- `data_collector_node` 主动拉取相机帧用于 HDF5
- MediaPipe 输入可以使用 SDK 相机或 ROS 图像 topic
- 推理 worker 直接读取相机用于模型输入和预览
- GUI 预览优先读 collector Preview API，其次读推理 preview signal

调整相机逻辑时，先确认改动影响的是采集、MediaPipe 输入、推理还是 GUI 预览。

### Estop 覆盖范围

`SystemOrchestrator` 有 `ESTOP` 相位，推理执行也有急停动作。这个范围覆盖项目内部的推理执行链和状态机，不等同于完整的机器人安全系统。

文档和代码注释中，应避免把它描述为：

- 全局硬件急停
- controller-manager 级统一安全状态
- UR driver 级保护停恢复机制

它主要覆盖推理执行链和本项目内部状态。

### Quest 侧体验仍依赖浏览器和网络环境

Quest 3 bridge 基于 WebXR / Vuer / websocket。它已经是正式输入后端，但使用体验仍受以下因素影响：

- Quest Browser 权限
- TLS / 自签名证书
- 局域网 IP 选择
- 电脑多网卡环境
- WebXR 事件稳定性

排查时优先看 bridge 日志中打印的访问 URL、证书路径和 connected topic。

## 修改功能时的文档同步表

| 修改内容 | 需要同步的文档 |
| --- | --- |
| 新增或改名 ROS 节点 | `01-architecture.md`, `02-runtime-boundaries.md`, `03-operation.md` |
| 修改 launch 参数 | `03-operation.md`, `08-configuration.md` |
| 修改 topic / service / controller 默认值 | `08-configuration.md`, 对应功能文档 |
| 修改状态机或动作优先级 | `04-control-flow.md` |
| 修改 Home / Home Zone | `04-control-flow.md`, `08-configuration.md` |
| 修改 HDF5 schema | `05-data-collection.md`, `06-inference.md` |
| 修改相机或预览链路 | `05-data-collection.md`, `06-inference.md` |
| 修改推理后端 | `06-inference.md`, `02-runtime-boundaries.md` |
| 修改设备、输入、Quest bridge 或映射 | `07-devices.md`, `08-configuration.md` |

## 写作风格

维护文档面向未来的自己和协作者，语气保持清楚、稳定、面向使用和维护。

推荐表达：

- “负责……”
- “执行流程是……”
- “调整时注意……”
- “适用范围是……”
- “这个能力目前覆盖……”

表达限制时，直接说明覆盖范围、依赖条件或待收敛事项。
