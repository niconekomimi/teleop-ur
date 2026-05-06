# 控制流程

更新时间：2026-05-06

本文记录控制权、状态机、动作仲裁和几个关键场景。数据采集细节见 [05-data-collection.md](05-data-collection.md)，推理后端见 [06-inference.md](06-inference.md)。

## 核心模块

控制规则主要由三个 `core` 模块承载：

- `SystemOrchestrator`：维护系统相位和请求约束
- `ActionMux`：按动作源优先级决定是否下发动作
- `ControlCoordinator`：把状态机和动作仲裁封装成统一接口

`teleop_control_node` 和 GUI 侧 `ROS2Worker` 都会复用这些模块。

## 系统相位

`SystemOrchestrator` 定义的相位：

| Phase | 含义 |
| --- | --- |
| `IDLE` | 空闲 |
| `TELEOP` | 人工遥操作运行中 |
| `HOMING` | Home 运动中 |
| `HOME_ZONE` | Home Zone 运动中 |
| `INFERENCE_READY` | 推理 worker 已就绪，但未执行动作 |
| `INFERENCE_EXECUTING` | 推理动作执行中 |
| `ESTOP` | 安全停状态 |

相位优先级由状态重算逻辑确定：`ESTOP` 高于 `HOME_ZONE`，`HOME_ZONE` 高于 `HOMING`，然后是 `INFERENCE_EXECUTING`、`TELEOP`、`INFERENCE_READY`、`IDLE`。

## 动作源优先级

`ActionMux` 的默认优先级：

| Source | Priority | 用途 |
| --- | ---: | --- |
| `SAFETY` | 100 | 零速、安全停 |
| `COMMANDER` | 80 | Home / Home Zone |
| `INFERENCE` | 20 | 模型推理执行 |
| `TELEOP` | 10 | 人工输入 |
| `NONE` | 0 | 空闲 |

当高优先级动作源处于活动状态时，低优先级动作会被拒绝。`publish_zero()` 默认以 `SAFETY` 源发布零速动作。

## 请求约束

状态机层面的主要约束：

- estop 状态下，teleop、推理执行、Home、Home Zone 和录制请求会被拒绝
- teleop 运行时，推理执行请求会被拒绝
- 推理执行中，teleop 请求会被拒绝
- Home 或 Home Zone 运行中，teleop 和推理执行请求会被拒绝
- Home 和 Home Zone 不允许重入

GUI 侧还会提前拦截部分冲突操作，避免用户操作已经明显不满足状态约束时再落到 ROS 服务层。

## GUI 侧意图拦截

GUI 会在用户操作进入 ROS 服务或子进程前做一层意图检查。主要规则：

启动 teleop 会被拒绝的情况：

- 系统处于 estop
- 推理执行正在运行
- Home 正在运行
- Home Zone 正在运行

启动推理执行会被拒绝的情况：

- 系统处于 estop
- teleop 正在运行
- Home 正在运行
- Home Zone 正在运行
- 推理 worker 尚未 ready

发起 Home / Home Zone 会被拒绝的情况：

- 系统处于 estop
- Home 正在运行
- Home Zone 正在运行

发起录制会被拒绝的情况：

- 系统处于 estop

这些规则只覆盖 GUI 层的操作约束。完整的机器人安全仍依赖外部硬件和驱动层。

## Teleop

`teleop_control_node` 负责人工遥操作闭环：

1. 根据 `input_type` 创建 `JoyInputHandler`、`MediaPipeInputHandler` 或 `Quest3InputHandler`
2. 根据 `gripper_type` 创建 Robotiq 或 qbSoftHand 控制器
3. 通过 `ServoArmBackend` 和 `ControllerGripperBackend` 下发动作
4. 每个控制周期读取输入，应用速度和加速度限制
5. 将 `ActionCommand` 交给 `ControlCoordinator`

当 `/commander/home_zone_active` 或 `/commander/homing_active` 为 true 时，teleop 节点暂停输出并重置输入运行态。

### 遥操作持续循环

每个控制周期执行：

1. 读取输入后端的目标动作
2. 将目标动作转换为 6 维 twist 和 1 维 gripper 命令
3. 对 twist 应用速度和加速度限制
4. 如果输入回零，按 `zero_return_accel_scale` 调整回零速度
5. 将限幅后的动作交给 `ControlCoordinator`
6. 由 `ActionMux` 判断动作源优先级
7. 如果动作被接受，发送到 arm backend 和 gripper backend

如果 Home 或 Home Zone 正在运行，teleop 会清空输入运行态并跳过输出。

## Home

`robot_commander_node` 提供 `/commander/go_home` 服务。

执行流程：

1. 检查当前没有 Home 或 Home Zone 在运行
2. 读取 `home_joint_positions`
3. 发布 `/commander/homing_active`
4. 切换到 trajectory controller
5. 向 `home_joint_trajectory_topic` 发布 1 个 waypoint 的 `JointTrajectory`
6. 等待关节目标收敛
7. 切回 teleop controller
8. 发布 `/commander/last_motion_result`

Home 点默认来自 `robot_profiles.yaml`，GUI 设置当前姿态为 Home 时会写入 `home_overrides.yaml` 并更新 commander 参数。

### 遥操作中点击 Go Home

当 teleop 运行中点击 `Go Home`：

1. GUI 侧先检查当前是否允许 Home
2. 如果推理执行正在运行，GUI 会先停止推理执行
3. `ROS2Worker` 调用 `/commander/go_home`
4. commander 发布 `/commander/homing_active`
5. `teleop_control_node` 收到 homing 状态后暂停输出
6. commander 切换 controller 并执行 Home 轨迹
7. 结束后恢复 teleop controller，teleop 输出恢复

## Home Zone

`robot_commander_node` 提供 `/commander/go_home_zone` 和 `/commander/cancel_home_zone`。

执行流程：

1. 检查当前没有 Home 或 Home Zone 在运行
2. 根据 Home 关节角计算 Home 位姿
3. 在 `home_zone_translation_*` 和 `home_zone_rotation_*` 范围内采样目标偏移
4. 求解目标关节角
5. 发布 `/commander/home_zone_active`
6. 切换到 trajectory controller，移动到采样出的目标关节位姿
7. 切回 teleop controller，发布结果并清零 twist

新的人工输入不会自动取消 Home Zone。需要取消时调用 `/commander/cancel_home_zone`。

### 遥操作中点击 Go Home Zone

当 teleop 运行中点击 `Go Home Zone`：

1. GUI 侧检查 Home Zone 是否允许启动
2. `ROS2Worker` 调用 `/commander/go_home_zone`
3. commander 发布 `/commander/home_zone_active`
4. `teleop_control_node` 暂停人工输入输出
5. commander 根据 Home 点采样目标位姿，并求解目标关节角
6. commander 切换到 trajectory controller 并执行目标关节轨迹
7. 结束后清零 twist、恢复 teleop controller、发布运动结果

Home Zone 不会因为新的人工输入自动取消。如果需要中断，调用 `/commander/cancel_home_zone`。

## 推理执行

推理 worker 负责产出动作，实际动作下发由 GUI 侧 `ROS2Worker` 完成。

执行链路：

```text
InferenceService
-> InferenceWorker 或 OpenPiRemoteWorker
-> action_signal
-> MainWindow
-> GuiAppService.update_inference_action_command()
-> ROS2Worker
-> ControlCoordinator / ActionMux
-> ServoArmBackend / GripperBackend
```

推理 worker 启动后只是进入“推理就绪”状态。只有用户显式启用推理执行后，`ROS2Worker` 才会把模型动作发给机器人。

### 推理急停

GUI 的推理急停会：

1. 停止推理执行开关
2. 让 `ROS2Worker` 尝试发布安全零速动作
3. 更新 GUI 状态

这个动作主要用于停止本项目内的推理动作执行链，不等同于机器人级硬件急停。

## 维护注意事项

- 新增动作源时，先更新 `ControlSource` 和 `ActionMux` 优先级，再更新本文。
- 修改 Home / Home Zone 服务、topic 或 controller 行为时，同步更新 [03-operation.md](03-operation.md) 和 [08-configuration.md](08-configuration.md)。
- 推理动作下发仍在 GUI 侧 `ROS2Worker`，文档和代码注释中应保持这个职责描述。
