# PROJECT_ANALYSIS

更新时间：2026-03-22

## 1. 文档目的

这份文档描述的是当前工作区代码已经实现的真实结构，不是目标态设计稿。

重点回答三个问题：

1. 当前系统由哪些进程、节点和核心模块组成。
2. 当前控制、录制、推理、Home / Home Zone 分别怎么跑。
3. 当前配置分层已经做到什么程度，还差什么。

---

## 2. 一句话结论

当前 `teleop_control_py` 已经从“GUI 直接持有大量业务逻辑”的旧结构，收敛成了：

- GUI 负责界面、意图发起和状态展示。
- `core` 负责状态机、控制仲裁、录制与推理生命周期。
- ROS 节点分别承载遥操作、Home / Home Zone、采集。
- `robot_profiles.yaml` 已经成为机械臂/夹爪接口默认值的主真源。

但它还不是一个“所有控制都归一个中央运行时进程”的最终系统。当前控制职责仍然分散在：

- `teleop_control_node`
- `robot_commander_node`
- `data_collector_node`
- GUI 侧 `ROS2Worker`

因此，当前状态更准确地说是：**架构骨架已经成型，但还处在“多节点协同 + GUI 桥接”的过渡态。**

---

## 3. 当前真实架构分层

## 3.1 GUI / Task Layer

主要文件：

- `teleop_control_py/gui/main_window.py`
- `teleop_control_py/gui/app_service.py`
- `teleop_control_py/gui/intent_controller.py`
- `teleop_control_py/gui/runtime_facade.py`
- `teleop_control_py/gui/ros_worker.py`

职责边界：

- `main_window.py`
  - 负责控件、布局、用户操作、状态渲染。
  - 不再直接自己拼 ROS service / topic 细节。
- `GuiAppService`
  - 管理子进程启动与停止。
  - 管理 `ROS2Worker` 生命周期。
  - 把“启动驱动 / 启动遥操作 / 启动采集 / 调 Home / 开启推理执行”收成统一调用入口。
- `GuiIntentController`
  - 在 GUI 侧复用协调器规则，提前拦截冲突操作。
- `GuiRuntimeFacade`
  - 把进程状态、硬件探测、相机可用性汇总成 GUI 可直接消费的运行时快照。
- `ROS2Worker`
  - 是 GUI 的 ROS 桥。
  - 负责订阅机器人状态、调用 `data_collector` / `commander` 服务。
  - 在“推理执行”场景下，它自己持有一套 `ControlCoordinator + ServoArmBackend + GripperBackend` 来发动作。

当前评价：

- GUI 主窗口已经明显变“更像纯 UI”。
- 但 `ROS2Worker` 仍承担了部分运行时控制职责，尤其是推理执行链。

## 3.2 Core Layer

主要文件：

- `teleop_control_py/core/models.py`
- `teleop_control_py/core/mux.py`
- `teleop_control_py/core/orchestrator.py`
- `teleop_control_py/core/control_coordinator.py`
- `teleop_control_py/core/sync_hub.py`
- `teleop_control_py/core/recorder.py`
- `teleop_control_py/core/inference_service.py`

关键作用：

- `models.py`
  - 定义公共数据结构，如 `ActionCommand`、`CameraFrameSet`、`RobotStateSnapshot`、`ObservationSnapshot`。
- `ActionMux`
  - 负责动作源仲裁。
  - 当前优先级是：
    - `SAFETY = 100`
    - `COMMANDER = 80`
    - `INFERENCE = 20`
    - `TELEOP = 10`
    - `NONE = 0`
- `SystemOrchestrator`
  - 负责上层状态机。
  - 当前相位是：
    - `IDLE`
    - `TELEOP`
    - `HOMING`
    - `HOME_ZONE`
    - `INFERENCE_READY`
    - `INFERENCE_EXECUTING`
    - `ESTOP`
- `ControlCoordinator`
  - 把 `SystemOrchestrator + ActionMux` 封成一套统一接口。
  - `teleop_control_node` 和 `ROS2Worker` 都在复用它。
- `SyncHub`
  - 在采集节点里负责“相机帧 + 机器人状态 + 动作”的同步快照拼装。
- `RecorderService`
  - 包装 HDF5 writer 线程，提供 `start_demo / stop_demo / discard_last_demo / enqueue_sample`。
- `InferenceService`
  - 只负责 `InferenceWorker` 生命周期，不绑定具体 UI。

当前评价：

- `core` 层已经不是空壳，而是当前运行链路中的真实公共骨架。
- 这部分是整个项目目前最接近“可复用平台层”的区域。

## 3.3 Node / Backend Layer

主要文件：

- `teleop_control_py/nodes/teleop_control_node.py`
- `teleop_control_py/nodes/robot_commander_node.py`
- `teleop_control_py/nodes/data_collector_node.py`
- `teleop_control_py/nodes/joy_driver_node.py`
- `teleop_control_py/device_manager/*`
- `teleop_control_py/hardware/*`

职责：

- `teleop_control_node`
  - 读取 `joy` 或 `mediapipe` 输入。
  - 做速度/加速度限幅。
  - 通过 `ControlCoordinator` 下发遥操作动作。
- `robot_commander_node`
  - 提供 `/commander/go_home`
  - 提供 `/commander/go_home_zone`
  - 提供 `/commander/cancel_home_zone`
  - 负责 controller 切换、Home 轨迹发布、Home Zone servo 控制。
- `data_collector_node`
  - 通过 SDK 相机拉帧。
  - 读取 ROS 机器人状态缓存。
  - 用 `SyncHub + RecorderService` 录制 HDF5。
- `joy_driver_node`
  - 独立负责手柄设备接入，发布 `/joy`。
- `device_manager`
  - 提供 `ServoArmBackend / ControllerGripperBackend / SharedMemoryCameraBackend / InputHandlerBackend`
  - 提供 `robot_profile` 加载与默认值装配。

当前评价：

- 当前系统不是一个大而全的单节点。
- 它是“多个专职 ROS 节点 + 一层共享 core 模块”的结构。

## 3.4 External Drivers / Dependencies

当前主要依赖：

- `ur_robot_driver`
- `ur_moveit_config` / MoveIt Servo
- `robotiq_2f_gripper_hardware`
- `qbsofthand_control`
- `pyrealsense2`
- `depthai`
- `Real_IL` 推理模型及其依赖

控制系统 launch 入口仍然是：

- `teleop_control_py/launch/control_system.launch.py`

这个 launch 当前负责：

- 引入 UR driver
- 可选引入 MoveIt / Servo
- 按 `gripper_type` 引入 Robotiq 或 qbSoftHand 驱动
- 创建 `robot_commander_node`
- 按需创建 `teleop_control_node`
- 按需创建 `data_collector_node`
- 当 `input_type == joy` 时引入 `joy_driver.launch.py`

---

## 4. 当前关键运行链路

## 4.1 仅启动机械臂驱动

GUI 的“启动机械臂驱动”不是只拉起 UR 驱动本体，而是通过：

```text
GuiAppService.start_robot_driver()
-> build_robot_driver_command()
-> ros2 launch teleop_control_py control_system.launch.py launch_teleop_node:=false
```

实际效果：

- 启动 UR driver
- 启动 MoveIt / Servo
- 启动夹爪驱动
- 启动 `robot_commander_node`
- 不启动 `teleop_control_node`

因此它更准确地说是“底层控制底座 + commander 启动”，不是“裸驱动”。

## 4.2 遥操作链路

```text
Joy / MediaPipe
-> InputHandlerBackend
-> teleop_control_node
-> ControlCoordinator
-> ActionMux
-> ServoArmBackend / ControllerGripperBackend
-> MoveIt Servo / gripper driver
```

当前特征：

- `teleop_control_node` 加载 `robot_profile`，用 profile 提供机械臂/夹爪接口默认值。
- `input_type` 当前支持：
  - `joy`
  - `mediapipe`
- `mediapipe` 默认走 SDK 相机直连，深度流默认关闭，只有显式开启时才启用。
- 遥操作循环中始终执行速度/加速度限幅。
- `home_zone_active == true` 时，遥操作输出被暂停。
- **当前实现中，新的人工输入不会取消 Home Zone。**

## 4.3 Home / Home Zone 链路

```text
GUI / ROS2Worker
-> /commander/go_home or /commander/go_home_zone
-> robot_commander_node
-> controller switch
-> trajectory or servo motion
```

### Go Home

实际行为：

1. 如果 GUI 当前正在执行 inference，`ROS2Worker` 会先停止 inference execution。
2. `robot_commander_node` 校验当前不在 `homing` / `home_zone` 中。
3. 切换到 trajectory controller。
4. 向 `home_joint_trajectory_topic` 发布 `JointTrajectory`。
5. 等待 `home_duration_sec + 0.5s`。
6. 切回 teleop controller。
7. 发布 `/commander/homing_active=false` 和 `/commander/last_motion_result`。

这就是当前系统里“速度控制切到位置控制再切回”的真实实现。

### Go Home Zone

实际行为：

1. 如果 GUI 当前正在执行 inference，`ROS2Worker` 会先停止 inference execution。
2. 校验当前不在 `homing` / `home_zone` 中。
3. 标记 `home_zone_active=true`。
4. 用 Home 关节角做 FK，得到 Home 末端位姿。
5. 在 Home 位姿附近采样 Home Zone 偏移目标。
6. 对目标位姿做 IK，得到目标关节角。
7. trajectory controller 接管。
8. 发布一条单点 `JointTrajectory`，从当前姿态直接移动到该目标关节角。
9. 结束后恢复 teleop controller，并发布 `home_zone_active=false`。

取消方式：

- 当前只支持显式服务取消：`/commander/cancel_home_zone`
- 节点关闭时也会触发取消
- **不再经过 Cartesian servo 第二段**

## 4.4 录制链路

```text
GuiAppService.start_data_collector()
-> ros2 run teleop_control_py data_collector_node
-> SDK camera pull + ROS state cache
-> SyncHub.capture_snapshot()
-> RecorderService
-> HDF5WriterThread
```

当前特征：

- `data_collector_node` 不依赖相机 ROS topic 做主采样，而是直接用 SDK 相机客户端拉图。
- 支持 `global_camera_source` 和 `wrist_camera_source` 双相机。
- 支持按相机 source / serial / depth 开关创建相机实例。
- 采集时会检查：
  - joint 新鲜度
  - pose 新鲜度
  - action 新鲜度
  - gripper 新鲜度
  - 双相机时间偏差
- 图像会中心裁剪并强制写成 `224 x 224`。
- `RecorderService` 负责 demo 序号递增和弃用最近 demo。

服务接口：

- `/data_collector/start`
- `/data_collector/stop`
- `/data_collector/discard_last_demo`

## 4.5 推理链路

```text
GUI
-> InferenceService
-> InferenceWorker
-> Qt signal
-> ROS2Worker
-> ControlCoordinator
-> ServoArmBackend / ControllerGripperBackend
```

当前特征：

- `InferenceService` 只管理 worker 生命周期。
- `InferenceWorker` 不直接发布 ROS 动作，它通过信号把动作发回 GUI。
- `ROS2Worker` 在本地创建一套 inference 用的 backend 和 `ControlCoordinator`。
- 只有当协调器当前 `active_source == INFERENCE` 时，`ROS2Worker` 才真正下发推理动作。
- 夹爪命令在 `ROS2Worker` 内被离散化为二值开合。

这说明当前“推理执行”仍是 GUI 驱动的控制路径，而不是独立 ROS 推理节点。

---

## 5. 当前状态机与仲裁规则

## 5.1 SystemPhase

当前公共状态机以 `SystemOrchestrator` 为准：

- `IDLE`
- `TELEOP`
- `HOMING`
- `HOME_ZONE`
- `INFERENCE_READY`
- `INFERENCE_EXECUTING`
- `ESTOP`

其中：

- `recording_active` 是正交标志，不单独占一个 phase。
- `home_zone_active` 和 `homing_active` 的优先级高于 teleop / inference。

## 5.2 控制源优先级

`ActionMux` 当前优先级：

| Source | Priority | 用途 |
| --- | ---: | --- |
| `SAFETY` | 100 | 零速/急停保持 |
| `COMMANDER` | 80 | Home Zone / commander 运动 |
| `INFERENCE` | 20 | 推理执行 |
| `TELEOP` | 10 | 人工遥操作 |
| `NONE` | 0 | 空闲 |

规则：

- 高优先级源存在时，低优先级动作会被拒绝。
- `hold` 打开时，除 `SAFETY` 以外的动作全部被拒绝。

## 5.3 GUI / 协调器层的请求约束

当前 `SystemOrchestrator` 的关键约束：

- 启动 teleop 会拒绝：
  - `estopped`
  - `inference_executing`
  - `homing_active`
  - `home_zone_active`
- 使能 inference execution 会拒绝：
  - `estopped`
  - `teleop_running`
  - `homing_active`
  - `home_zone_active`
  - `inference_not_ready`
- `go_home` / `go_home_zone` 会拒绝：
  - `estopped`
  - `homing_active`
  - `home_zone_active`
- `start_recording` 只在 `estopped` 时被拒绝

这套规则现在同时影响：

- GUI 按钮可否发起请求
- `teleop_control_node` 内动作能否通过 mux
- `ROS2Worker` 的推理执行能否实际下发

---

## 6. 当前配置分层

## 6.1 `robot_profiles.yaml`

定位：底层接口真源。

当前负责：

- 机械臂画像名
- 关节名
- ROS topics
- ROS services
- controller 名称
- 默认 Home
- 默认 Home Zone
- 默认夹爪类型与夹爪接口参数

当前已经接入：

- `teleop_control_node`
- `robot_commander_node`
- `data_collector_node`
- `ROS2Worker` inference backend 参数装配

## 6.2 `teleop_params.yaml`

定位：遥操作行为配置。

当前负责：

- `input_type`
- `gripper_type`
- 遥操作频率
- 速度/加速度限幅
- joy 映射
- mediapipe 算法参数

它不再承担机械臂/夹爪接口真值。

## 6.3 `data_collector_params.yaml`

定位：采集行为配置。

当前负责：

- 输出路径
- 录制频率
- 相机来源、序列号、depth 开关
- 时间新鲜度阈值
- preview / writer 行为
- 采集侧 gripper 选择策略

## 6.4 `gui_params.yaml`

定位：GUI 偏好。

当前负责：

- 默认 IP
- 默认输入方式
- 默认相机模型 / serial 偏好
- 默认录制目录和文件名
- GUI 侧 depth 偏好

注意：

- 它仍保存了“相机型号 / 序列号偏好”这类半硬件选择信息。
- 这些值属于“用户偏好”，不是底层控制真值。

## 6.5 `home_overrides.yaml`

定位：运行期 Home 覆盖。

当前负责：

- GUI 点击“设当前姿态为 Home”后，把 6 轴关节角按 `robot_profile` 写入这里。

运行链：

```text
GUI 当前关节角
-> home_overrides.yaml
-> ROS2Worker set_parameters(/commander)
-> commander 运行时使用新 Home
```

## 6.6 `joy_driver_params.yaml`

定位：手柄驱动层配置。

当前负责：

- 设备发现
- publish 频率
- deadzone
- 自动重连

## 6.7 已删除的配置

`robot_commander_params.yaml` 已经不再是启动链路的一部分。

当前 `robot_commander_node` 的关键运行参数来自：

- `robot_profile`
- launch 里的少量显式覆盖（例如 `commander_pose_max_age_sec`）

---

## 7. 当前已经完成的结构性收敛

1. `robot_profile` 已成为机械臂/夹爪接口默认值主来源。
2. `ControlCoordinator + SystemOrchestrator + ActionMux` 已经在真实运行链路中生效。
3. 录制侧已经复用 `SyncHub + RecorderService`，不是 GUI 自己攒数据。
4. 推理 worker 生命周期已经下沉到 `core/inference_service.py`。
5. Home 点持久化已经从 `gui_params.yaml` 迁到独立 `home_overrides.yaml`。
6. `robot_commander_params.yaml` 已从启动链路移除。

---

## 8. 当前仍然存在的边界与缺口

## 8.1 还不是单一中央运行时

当前控制仍分布在多个位置：

- `teleop_control_node` 负责人工遥操作闭环
- `robot_commander_node` 负责 Home / Home Zone
- `ROS2Worker` 负责推理执行闭环
- `data_collector_node` 负责采样闭环

这意味着：

- 已经有统一控制规则
- 但还没有一个“唯一控制节点”承载所有控制动作

## 8.2 `robot_profiles.yaml` 还没有接管 launch 级驱动定义

目前 `robot_profiles.yaml` 已经接管 topics / services / controllers / home / gripper defaults，
但还没有接管：

- UR driver bringup 方式
- MoveIt bringup 方式
- 夹爪驱动 launch 选择

因此当前只能做到：

- “ROS 接口变化尽量只改 `robot_profiles.yaml`”

还不能完全做到：

- “底层硬件和驱动换了，只改 `robot_profiles.yaml` 即可”

## 8.3 相机驱动管理残留仍在 runtime 层

当前主链路已经偏向：

- 录制使用 SDK 相机直连
- MediaPipe 也支持 SDK 相机直连

但 `GuiRuntimeFacade / ResourceManager / ProcessManager` 里仍保留了：

- camera driver subprocess 管理
- camera driver 占用检查

这说明相机管理链路还存在部分历史残留。

## 8.4 当前“急停”不是统一机器人级安全停

当前显式急停主要体现在：

- `ROS2Worker.emergency_stop_inference()`
- `ControlCoordinator.notify_estop(True)`
- `publish_zero(source=SAFETY)`

它当前主要覆盖的是“推理执行链”。

这不是一个物理硬件级、全系统统一广播的安全停实现。

---

## 9. 当前最合理的架构判断

如果按“目标态”和“当前可运行代码”做区分，当前项目更适合这样描述：

- 已经有清晰的 UI 层、核心层、设备层边界。
- 已经建立统一状态机、动作格式、仲裁规则和 profile 真源。
- 仍然保持多 ROS 节点协同，而不是单控制中心收口。
- 已经适合继续做“安全精简”和“职责继续收敛”，不适合再大规模推翻重写。

一句话总结：

**当前项目不是旧版杂糅 GUI 了，也还不是最终统一控制平台；它现在处在一个已经能稳定演进的中间态。**
