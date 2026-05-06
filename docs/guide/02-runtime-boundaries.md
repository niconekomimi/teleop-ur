# 运行边界与职责

更新时间：2026-05-06

本文按运行角色说明项目职责分工和功能边界。它回答“某个功能适合放在哪一层、由哪个模块负责、与哪些模块协作”。

## GUI / Task Layer

主要文件：

- `teleop_control_py/gui/main_window.py`
- `teleop_control_py/gui/app_service.py`
- `teleop_control_py/gui/intent_controller.py`
- `teleop_control_py/gui/runtime_facade.py`
- `teleop_control_py/gui/ros_worker.py`
- `teleop_control_py/gui/http_preview_worker.py`
- `teleop_control_py/gui/preview_recording_worker.py`
- `teleop_control_py/gui/widgets/`

职责：

- `main_window.py` 负责控件、布局、用户操作和状态渲染。
- `GuiAppService` 负责 GUI 侧运行编排，包括启动驱动、遥操作、采集、Home、录制和推理执行。
- `GuiIntentController` 在 GUI 层复用协调器规则，用于提前拦截冲突操作。
- `GuiRuntimeFacade` 汇总进程状态、硬件探测、相机可用性和 GUI 可消费的运行时快照。
- `ROS2Worker` 是 GUI 的 ROS 桥，负责订阅状态、调用服务，并在推理执行期间创建本地控制后端。
- `HttpPreviewWorker` 轮询采集节点 Preview API，把 HTTP/JPEG 预览转给 GUI。
- `PreviewRecordingWorker` 从当前预览源写本地 `mp4`，不参与 HDF5 数据集录制。

适合由服务层、节点或 core 承担的职责：

- 长期运行的 ROS 控制逻辑
- HDF5 数据集写盘
- 底层 topic、service、controller 默认来源管理
- 机器人动作下发的协调与仲裁

## Core Layer

主要文件：

- `teleop_control_py/core/models.py`
- `teleop_control_py/core/orchestrator.py`
- `teleop_control_py/core/mux.py`
- `teleop_control_py/core/control_coordinator.py`
- `teleop_control_py/core/sync_hub.py`
- `teleop_control_py/core/recorder.py`
- `teleop_control_py/core/inference_service.py`
- `teleop_control_py/core/inference_worker.py`
- `teleop_control_py/core/openpi_remote_worker.py`

职责：

- `models.py` 定义公共数据结构，例如 `ActionCommand`、`ControlSource`、`RobotStateSnapshot`、`ObservationSnapshot`。
- `SystemOrchestrator` 管理系统相位和请求约束。
- `ActionMux` 管理动作源优先级和动作分发。
- `ControlCoordinator` 把状态机和动作仲裁封装成统一接口。
- `SyncHub` 负责采集节点中的“相机帧 + 机器人状态 + 动作”同步快照。
- `RecorderService` 管理 HDF5 writer 线程和 demo 生命周期。
- `InferenceService` 管理推理 worker 生命周期，不直接下发 ROS 控制命令。

Core 层保持与具体 UI 解耦。涉及 UI 展示、弹窗和按钮状态的逻辑放在 GUI 层更合适。

## ROS Nodes

### `teleop_control_node`

负责人工遥操作闭环：

- 读取 `joy`、`mediapipe` 或 `quest3` 输入
- 把输入标准化为 `ActionCommand`
- 做速度和加速度限幅
- 通过 `ControlCoordinator` 下发遥操作动作
- 在 Home 或 Home Zone 期间暂停输出

输入：

- `joy`：`/joy`
- `mediapipe`：SDK 相机或图像 topic
- `quest3`：`/quest3/right_controller/pose`、`/quest3/left_controller/pose`、`/quest3/input/joy`

输出：

- arm：向 MoveIt Servo twist topic 发布速度命令
- gripper：调用 Robotiq 或 qbSoftHand 控制接口

### `robot_commander_node`

负责机器人能力动作：

- `/commander/go_home`
- `/commander/go_home_zone`
- `/commander/cancel_home_zone`
- controller 切换
- Home 轨迹发布
- Home Zone 目标采样和运动执行

状态发布：

- `/commander/home_zone_active`
- `/commander/homing_active`
- `/commander/last_motion_result`

### `data_collector_node`

负责数据集采集：

- 初始化双相机客户端
- 缓存 joint、pose、action、gripper 状态
- 通过 `SyncHub` 同步快照
- 通过 `RecorderService` 写 HDF5
- 暴露录制开始、停止、弃用最近 demo 服务
- 暴露采集侧 Preview API

它的职责保持在数据采集链路内，机器人控制和模型推理由对应模块承担。

### `joy_driver_node`

负责手柄设备接入：

- 自动发现 `/dev/input` 设备
- 读取 evdev 事件
- 发布 `/joy`
- 支持 profile、deadzone、publish rate 和自动重连

### `quest3_webxr_bridge_node`

负责 Quest Browser WebXR 数据接入：

- 通过 Vuer / websocket 接收控制器数据
- 发布左右手 pose / matrix
- 发布 `/quest3/input/joy`
- 发布 Quest 连接状态

## GUI Workers

### `ROS2Worker`

`ROS2Worker` 位于 GUI 进程内，但它会创建 ROS node，因此承担一部分运行时桥接职责。

它负责：

- 订阅 joint / pose / action / gripper / commander / collector 状态
- 调用 `/data_collector/*` 服务
- 调用 `/commander/*` 服务
- 更新 commander 的 Home 参数
- 在推理执行期间创建 `ServoArmBackend`、`GripperBackend` 和 `ControlCoordinator`

因此，推理执行链路目前在 GUI 进程内闭环，并不是独立 ROS 节点。

### `InferenceWorker` 和 `OpenPiRemoteWorker`

推理 worker 负责产生动作和预览：

- `InferenceWorker` 本地加载 `Real_IL`
- `OpenPiRemoteWorker` 通过 websocket 请求远端 openpi

它们保持“产出动作和预览”的职责。动作通过 signal 回到 GUI，再交给 `ROS2Worker` 执行，ROS 控制 topic 由执行链路统一处理。

### `PreviewRecordingWorker`

负责预览录屏：

- 输入来自当前活动预览源
- 输出本地 `mp4`
- HDF5 写盘由 `data_collector_node` 负责
- `data_collector_node` 的启停由 GUI 或 ROS service 协调

## 外部驱动与依赖

主要外部组件：

- `ur_robot_driver`
- `ur_moveit_config` / MoveIt Servo
- `robotiq_2f_gripper_hardware`
- `qbsofthand_control`
- `pyrealsense2`
- `depthai`
- `vuer`
- `Real_IL`
- `openpi-client`

维护外部依赖相关逻辑时，优先通过 launch 参数、remap、临时运行时配置或本项目 wrapper 适配，避免直接修改上游包源码。

## 功能归属速查

| 功能 | 归属 |
| --- | --- |
| 启动机器人驱动 | GUI `GuiAppService` / `control_system.launch.py` |
| 人工遥操作 | `teleop_control_node` |
| Home / Home Zone | `robot_commander_node` |
| 控制权仲裁 | `core/orchestrator.py`, `core/mux.py`, `core/control_coordinator.py` |
| HDF5 数据集写盘 | `data_collector_node`, `RecorderService`, `HDF5WriterThread` |
| 采集侧预览 | `data_collector_node`, `PreviewApiServer` |
| GUI 预览窗口 | `HttpPreviewWorker`, 推理 preview signal |
| 预览录屏 | `PreviewRecordingWorker` |
| 本地推理 | `InferenceWorker` |
| 远端 openpi 推理 | `OpenPiRemoteWorker` |
| 推理动作执行 | GUI 侧 `ROS2Worker` |
| 底层 topic / service / controller 默认来源 | `robot_profiles.yaml` |
