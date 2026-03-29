中文 | [English](current_control_behavior_spec_v0.1_EN.md)

# 当前控制行为规格 v0.1

更新时间：2026-03-29

这份文档只描述当前代码真实行为，不描述未来目标架构。

---

## 1. 适用范围

本文覆盖：

- 遥操作
- Home / Home Zone
- 录制
- 推理执行
- GUI 侧意图拦截
- 当前控制权与仲裁规则

本文不覆盖：

- 完整 UI 视觉布局
- 离线 HDF5 编辑器细节
- 未来多机械臂后端设计

---

## 2. 当前运行角色

## 2.1 `teleop_control_node`

职责：

- 读取 `joy`、`mediapipe` 或 `quest3` 输入
- 把输入标准化为 `ActionCommand`
- 做速度和加速度限幅
- 通过 `ControlCoordinator` 下发遥操作动作
- 在 `Home Zone` 期间暂停自身输出

输入：

- `joy`：`/joy`
- `mediapipe`：SDK 相机或图像 topic
- `quest3`：`/quest3/right_controller/pose`、`/quest3/left_controller/pose`、`/quest3/input/joy`

输出：

- 通过 `ServoArmBackend` 向 `servo_twist_topic` 发机械臂速度命令
- 通过 `ControllerGripperBackend` 发夹爪命令

## 2.2 `robot_commander_node`

职责：

- 提供 `go_home`
- 提供 `go_home_zone`
- 提供 `cancel_home_zone`
- 负责 controller 切换
- 负责 Home 轨迹发布
- 负责 Home Zone 的 servo 驱动

服务：

- `/commander/go_home`
- `/commander/go_home_zone`
- `/commander/cancel_home_zone`

状态发布：

- `/commander/home_zone_active`
- `/commander/homing_active`
- `/commander/last_motion_result`

## 2.3 `data_collector_node`

职责：

- 初始化双相机 SDK 客户端
- 缓存机器人状态
- 通过 `SyncHub` 同步快照
- 通过 `RecorderService` 写盘
- 暴露录制开始/停止/弃用接口

服务：

- `/data_collector/start`
- `/data_collector/stop`
- `/data_collector/discard_last_demo`

状态发布：

- `/data_collector/record_stats`

## 2.4 `ROS2Worker`

职责：

- 是 GUI 的 ROS 桥
- 订阅 joint / pose / action / gripper / commander 状态
- 调用 commander 与 data_collector 服务
- 在推理执行期间创建本地控制 backend 并下发动作

注意：

- 当前推理执行不是独立 ROS 节点闭环，而是 GUI 侧 `ROS2Worker` 闭环。

## 2.5 `InferenceWorker`

职责：

- 加载模型
- 读取推理所需图像与机器人状态
- 输出 7 维动作

它本身不直接发 ROS 控制命令。

## 2.6 `ProcessManager`

职责：

- 启动和关闭长期运行的 ROS / Python 子进程

GUI 通过它管理：

- 机械臂驱动进程
- 遥操作进程
- 采集进程

---

## 3. 当前公共状态与仲裁规则

## 3.1 状态机相位

当前公共相位来自 `SystemOrchestrator`：

- `IDLE`
- `TELEOP`
- `HOMING`
- `HOME_ZONE`
- `INFERENCE_READY`
- `INFERENCE_EXECUTING`
- `ESTOP`

正交标志：

- `recording_active`
- `teleop_running`
- `inference_ready`
- `inference_executing`
- `home_zone_active`
- `homing_active`
- `estopped`

相位优先级：

1. `ESTOP`
2. `HOME_ZONE`
3. `HOMING`
4. `INFERENCE_EXECUTING`
5. `TELEOP`
6. `INFERENCE_READY`
7. `IDLE`

## 3.2 动作源优先级

当前 `ActionMux` 优先级：

| Source | Priority | 说明 |
| --- | ---: | --- |
| `SAFETY` | 100 | 零速 / 急停 |
| `COMMANDER` | 80 | Home Zone / commander |
| `INFERENCE` | 20 | 模型执行 |
| `TELEOP` | 10 | 人工输入 |
| `NONE` | 0 | 空闲 |

行为规则：

- 高优先级源活动时，低优先级源动作被拒绝。
- `hold` 生效时，仅 `SAFETY` 可以通过。

## 3.3 GUI 侧意图拦截规则

GUI 使用 `GuiIntentController` 复用 `ControlCoordinator` 规则。

### 启动 teleop 会被拒绝的情况

- `estopped`
- `inference_executing`
- `homing_active`
- `home_zone_active`

### 启动 inference execution 会被拒绝的情况

- `estopped`
- `teleop_running`
- `homing_active`
- `home_zone_active`
- `inference_not_ready`

### 发起 Home / Home Zone 会被拒绝的情况

- `estopped`
- `homing_active`
- `home_zone_active`

补充：

- 当前 `go_home / go_home_zone` 不会因为 `inference_executing` 被协调器直接拒绝。
- GUI 路径下，如果推理执行正在运行，`ROS2Worker` 会先停掉 inference execution，再调用 commander 服务。

### 发起录制会被拒绝的情况

- `estopped`

结论：

- `teleop` 与 `inference execution` 当前是互斥的。
- `Home / Home Zone` 高于 teleop；对 inference execution 的处理是“先停推理，再调用 commander”。
- `recording` 不是独立控制源，原则上可与 teleop / inference 并行，只受采集链自身状态影响。

---

## 4. 场景规格

## 4.1 启动机械臂驱动

GUI 调用：

```text
GuiAppService.start_robot_driver()
-> build_robot_driver_command()
-> ros2 launch teleop_control_py control_system.launch.py launch_teleop_node:=false
```

实际启动内容：

- UR driver
- MoveIt / Servo
- 夹爪驱动
- `robot_commander_node`

不会启动：

- `teleop_control_node`

结果：

- 底层控制底座可用
- GUI 可以通过 `ROS2Worker` 读取 joint / pose / commander 状态

## 4.2 启动遥操作系统

GUI 调用：

```text
GuiAppService.start_teleop()
-> build_teleop_command()
-> ros2 launch teleop_control_py control_system.launch.py
```

实际行为：

- 启动 UR driver
- 启动 MoveIt / Servo
- 启动夹爪驱动
- 启动 `robot_commander_node`
- 启动 `teleop_control_node`
- 若 `input_type == joy`，引入 `joy_driver.launch.py`
- 若 `input_type == quest3` 且 `launch_quest3_bridge` 解析为启用，则自动引入 `quest3_webxr_bridge_node`

参数来源：

- 行为参数来自 `teleop_params.yaml`
- 机械臂/夹爪接口默认值来自 `robot_profile`

### `input_type = joy`

流程：

1. `joy_driver_node` 发布 `/joy`
2. `JoyInputHandler` 读取 `/joy`
3. 输出平移、转动、夹爪开合命令

### `input_type = mediapipe`

流程：

1. `MediaPipeInputHandler` 从 SDK 相机或图像 topic 取图
2. 做手部关键点跟踪
3. 根据配置生成目标位姿或速度命令
4. 可选使用深度恢复手部三维位置

注意：

- SDK 深度默认关闭
- 只有启动 hand / mediapipe 且显式要求深度时才开启

### `input_type = quest3`

流程：

1. `quest3_webxr_bridge_node` 发布 `/quest3/*` 控制器话题
2. `Quest3InputHandler` 读取控制器 pose 与 `/quest3/input/joy`
3. 默认使用 `relative pose + clutch + hand_relative orientation`
4. 默认支持 Quest2ROS 风格的相对 frame 重置

当前默认语义：

- `active_hand = left`
- 左手 `squeeze / grip` 作为 clutch
- 左手 `trigger` 作为夹爪
- 输入层默认关闭低通平滑
- `frame reset` 默认只作用于 `active_hand`

## 4.3 遥操作持续循环

`teleop_control_node._control_loop()` 当前行为：

1. 从 `InputHandlerBackend` 取 `ActionCommand`
2. 计算本周期 `dt`
3. 取出 6 维 twist 目标
4. 如果 `home_zone_active == true`
   - 把上次 twist 清零
   - 本周期直接返回
5. 对非零轴和回零轴分别应用不同加速度上限
6. 调用 `apply_velocity_limits()`
7. 通过 `ControlCoordinator.dispatch()` 发命令

重要行为：

- `Home Zone` 期间遥操作不会输出动作
- 当前代码中不会因为新的人工输入而调用取消 Home Zone

## 4.4 遥操作中点击 Go Home

流程：

1. GUI 通过 `GuiIntentController` 检查是否允许。
2. 如果推理执行正在运行，`ROS2Worker` 会先停止 inference execution。
3. `ROS2Worker` 调 `/commander/go_home`。
4. `robot_commander_node` 如果发现 `homing` 或 `home_zone` 已在运行，则拒绝。
5. `commander` 标记 `homing_active=true`。
6. 切换 controller：
   - 激活 trajectory controller
   - 关闭 teleop controller
7. 向 `home_joint_trajectory_topic` 发布 `JointTrajectory`。
8. 等待 `home_duration_sec + 0.5s`。
9. 切回 controller：
   - 激活 teleop controller
   - 关闭 trajectory controller
10. 发布结果到 `/commander/last_motion_result`。

结论：

- 当前确实存在“速度控制切到位置控制再切回”的行为。

## 4.5 遥操作中点击 Go Home Zone

流程：

1. GUI 通过 `GuiIntentController` 检查是否允许。
2. 如果推理执行正在运行，`ROS2Worker` 会先停止 inference execution。
3. `ROS2Worker` 调 `/commander/go_home_zone`。
4. `robot_commander_node` 标记：
   - `home_zone_active=true`
   - `homing_active=true`
5. 读取配置中的 Home 关节角，并用 FK 计算 Home 末端位姿。
6. 在 Home 位姿附近采样平移和旋转偏移。
7. 对偏移后的目标位姿做 IK，求出目标关节角。
8. 切 trajectory controller。
9. 发布一条单点 `JointTrajectory`，从当前姿态直接移动到该目标关节角。
10. 成功或取消后：
   - 恢复 teleop controller
   - `home_zone_active=false`
   - 发布 motion result

取消渠道：

- `/commander/cancel_home_zone`
- 节点关闭时自动取消

当前说明：

- 不再经过 Cartesian servo 第二段。

## 4.6 录制流程

启动：

```text
GuiAppService.start_data_collector()
-> ros2 run teleop_control_py data_collector_node --ros-args --params-file ...
```

启动时 GUI 会覆盖：

- `robot_profile`
- `output_path`
- `global_camera_source`
- `wrist_camera_source`
- `global_camera_serial_number`
- `wrist_camera_serial_number`
- `global_camera_enable_depth`
- `wrist_camera_enable_depth`
- `end_effector_type`

### `data_collector_node` 就绪后

它会：

- 初始化双相机客户端
- 订阅 joint / pose / twist / gripper
- 启动可选 preview API

### 调用 `/data_collector/start`

行为：

1. 检查当前不在录制中。
2. 检查双相机客户端已可用。
3. `RecorderService.start_demo()`
4. 创建定时器，频率为 `record_fps`

### 每次采样

1. `SyncHub.capture_snapshot()`
2. 校验 joint / pose / action / gripper 新鲜度
3. 校验双相机时间偏差
4. 裁剪并缩放两路 RGB 到 `224 x 224`
5. 组装 `Sample`
6. `RecorderService.enqueue_sample()`

### 调用 `/data_collector/stop`

行为：

1. 关闭采样定时器
2. `RecorderService.stop_demo()`
3. 结束当前 demo

### 调用 `/data_collector/discard_last_demo`

行为：

1. 必须当前不在录制中
2. `RecorderService.discard_last_demo()`
3. 丢弃最近一个 demo，并回退 demo 序号

## 4.7 推理启动与执行

### 启动推理

GUI 调用：

```text
InferenceService.start_inference()
-> 创建 InferenceWorker
```

`InferenceLaunchConfig` 当前包含：

- checkpoint 目录
- task name
- embedding 路径
- global / wrist 相机 source
- loop_hz
- device
- state_provider
- global / wrist serial
- global / wrist depth 开关

结果状态：

- `InferenceWorker` 开始运行
- `ControlCoordinator` 可被标记为 `INFERENCE_READY`

### 使能推理执行

GUI 调用：

```text
GuiAppService.enable_inference_execution()
-> ROS2Worker.set_inference_execution_enabled(True)
```

实际行为：

1. `ROS2Worker` 确保本地 inference backend 已建立。
2. 清除 inference estop。
3. `ControlCoordinator.notify_inference_ready(True)`
4. `ControlCoordinator.notify_inference_execution(True)`
5. 开始按固定频率尝试发送模型动作。

### 每次推理执行周期

1. 读取最近一条模型动作。
2. 转成 `ActionCommand`。
3. 夹爪命令做二值化。
4. 若当前 `active_source != INFERENCE`，直接跳过。
5. 若当前 `active_source == INFERENCE`，通过协调器下发。

### 停止推理执行

行为：

1. `notify_inference_execution(False)`
2. 发布零速

### 推理急停

行为：

1. `self._inference_estopped = True`
2. `self._inference_control_enabled = False`
3. `notify_estop(True)`
4. 发布零速

重要说明：

- 当前“急停”主要作用在 GUI 侧 inference execution 链。
- 它不是底层硬件级急停，也不是一个独立全系统安全节点。

---

## 5. 当前 Home 点行为

当前 Home 点来源顺序：

1. `home_overrides.yaml` 中该 `robot_profile` 的覆盖值
2. `robot_profiles.yaml` 中该 profile 的默认 Home

GUI 点击“设当前姿态为 Home”后的行为：

1. 从 `ROS2Worker` 当前 joint 状态读取 6 轴角度
2. 写入 `config/home_overrides.yaml`
3. 通过 `/commander/set_parameters` 同步到 `robot_commander_node`

结果：

- 当前运行中的 commander 立刻使用新 Home
- 下次重启 GUI / 节点也会继续使用该覆盖值

---

## 6. 当前相机相关真实行为

## 6.1 录制链主路径

当前录制主路径是：

- SDK 相机直连

不是：

- ROS 图像 topic 主采样

## 6.2 MediaPipe 相机

当前 MediaPipe 可以：

- 使用 SDK 相机
- 使用图像 topic

但当前主配置偏向：

- `mediapipe_use_sdk_camera = true`

## 6.3 深度图

当前默认行为：

- SDK 深度默认关闭
- 开启深度时，记录链会显式启用 depth
- 对于需要 depth 的路径，会走对齐后的 RGBD 逻辑

## 6.4 残留链路

`GuiRuntimeFacade` 中仍保留 camera driver subprocess 管理接口，但这不是当前录制和 MediaPipe 的主工作链路。

---

## 7. 当前系统边界

1. 当前已经有统一状态机和动作仲裁规则，但还不是单一中央控制节点。
2. 当前 `teleop`、`commander`、`collector`、`GUI inference execution` 仍分属不同运行体。
3. 当前 `robot_profile` 已经是接口默认值真源，但 launch 级驱动选择还没有完全 profile 化。
4. 当前 `recording` 是采集状态，不是控制权来源。
5. 当前 `Home / Home Zone` 可以在 GUI 路径下抢占 inference execution，但实现方式是“先停止 inference execution，再调用 commander”。
6. 当前 `Home Zone` 不能被新的人工输入取消，只能通过显式取消服务或流程结束退出。

---

## 8. 行为摘要

可以直接把当前系统理解成下面这几条：

- 人工遥操作和推理执行互斥。
- Home / Home Zone 高于 teleop；对 inference execution 的处理是先停推理再执行 commander 流程。
- Home 通过轨迹控制器执行。
- Home Zone 先回 Home，再切回 Servo 控制做位姿扰动。
- 录制通过 SDK 相机 + ROS 状态缓存主动采样。
- 推理执行通过 GUI 侧 `ROS2Worker` 闭环下发，不是独立 ROS 推理节点。
- Home 点可以在运行中重新教点，并持久化到 `home_overrides.yaml`。
