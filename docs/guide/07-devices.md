# 设备与输入

更新时间：2026-05-06

本文记录工作区涉及的主要硬件、输入后端和设备接入方式。控制流程见 [04-control-flow.md](04-control-flow.md)，配置来源见 [08-configuration.md](08-configuration.md)。

## 机械臂

当前主链路围绕 UR5 + MoveIt Servo 组织：

- UR driver、dashboard、controller 和 MoveIt 配置来自 `src/Universal_Robots_ROS2_Driver/`
- 本项目通过 `control_system.launch.py` 组合 UR driver、MoveIt Servo、夹爪驱动和 commander
- 底层 topic、controller、Home 和 Home Zone 默认值优先来自 `robot_profiles.yaml`

维护机械臂相关逻辑时，先确认改动属于上游 driver / MoveIt 配置，还是本项目的 launch、profile、backend 或 commander 行为。

## 夹爪

支持的夹爪类型：

| 类型 | 相关路径 | 说明 |
| --- | --- | --- |
| `robotiq` | `src/robotiq_2f_gripper_ros2/` | Robotiq 2F 夹爪驱动和消息包 |
| `qbsofthand` | `src/qbsofthand_control/` | qbSoftHand 控制包 |

主控包通过 gripper backend 适配不同夹爪。默认接口、topic、service 和参数来源见 `robot_profiles.yaml` 与 [08-configuration.md](08-configuration.md)。

## 相机

相机链路按用途分开：

- `data_collector_node` 主动拉取双相机帧，用于 HDF5 录制
- MediaPipe 输入可以使用 SDK 相机或 ROS 图像 topic
- 推理 worker 读取相机帧用于模型输入和推理预览
- GUI 预览优先读取 collector Preview API，其次读取推理 preview signal

相机配置涉及：

- `data_collector_params.yaml`
- `gui_params.yaml`
- 推理启动配置
- GUI 中选择的 global / wrist camera source 和 serial number

调整相机逻辑时，需要明确影响的是采集、MediaPipe 输入、推理还是 GUI 预览。

## 输入后端

`teleop_control_node` 支持三类输入：

| 输入 | 主要组件 | 说明 |
| --- | --- | --- |
| `joy` | `joy_driver_node`, `JoyInputHandler` | 读取 evdev 手柄事件并发布 `/joy` |
| `mediapipe` | `MediaPipeInputHandler` | 基于手部追踪生成遥操作动作 |
| `quest3` | `quest3_webxr_bridge_node`, `Quest3InputHandler` | 通过 Quest Browser WebXR 输入控制器位姿和按钮 |

输入行为参数主要位于 `teleop_params.yaml`。GUI 负责选择输入类型，也可以通过“遥操作设置”弹窗调整常用速度、加速度、轴/按钮映射、clutch 和滤波参数；自定义方案保存在 `gui_params.yaml` 的 `teleop_settings` 中，应用后的实际运行参数写回 `teleop_params.yaml`。

## Quest 3 WebXR

Quest 3 输入由 `quest3_webxr_bridge_node` 和 `Quest3InputHandler` 组成。bridge 负责把 Quest Browser 中的 WebXR 控制器数据转成 ROS topic，teleop 节点负责把这些 topic 转成机器人动作。

### 启动方式

集成启动：

```bash
ros2 launch teleop_control_py control_system.launch.py input_type:=quest3 gripper_type:=robotiq
```

`launch_quest3_bridge:=auto` 是默认值。当 `input_type` 解析为 `quest3` 时，`control_system.launch.py` 会自动启动 `quest3_webxr_bridge_node`。

单独启动 bridge：

```bash
ros2 launch teleop_control_py quest3_webxr_bridge.launch.py
```

如果电脑有多个网卡，建议显式指定 Quest 需要访问的局域网 IP：

```bash
ros2 launch teleop_control_py control_system.launch.py \
    input_type:=quest3 \
    quest3_advertised_host:=192.168.137.227
```

### ROS topics

bridge 发布：

- `/quest3/left_controller/pose`
- `/quest3/right_controller/pose`
- `/quest3/left_controller/matrix`
- `/quest3/right_controller/matrix`
- `/quest3/input/joy`
- `/quest3/connected`

teleop 节点订阅的 topic 名称可通过 `teleop_params.yaml` 的 `quest3_*_topic` 参数调整。

### 连接前提

Quest 侧通常需要：

- 电脑和 Quest 在可互通的局域网内
- bridge 使用 Quest 可访问的 `advertised_host`
- WebXR 页面通过 HTTPS / WSS 访问
- Quest Browser 允许 WebXR 和控制器输入

如果使用自签名证书，首次访问时需要在 Quest Browser 中接受证书。多网卡环境下，优先显式设置 `quest3_advertised_host`。

### Bridge 参数

常用参数：

- `vuer_host`
- `vuer_port`
- `vuer_client_domain`
- `advertised_host`
- `auto_generate_self_signed_cert`
- `tls_workdir`
- `public_wss_url`
- `stream_key`
- `publish_rate_hz`
- `stale_timeout_sec`

集成启动默认从 `teleop_params.yaml` 的 `quest3_webxr_bridge_node` 参数块读取。单独启动时也可以传入 `quest3_webxr_bridge_params.yaml`：

```bash
ros2 launch teleop_control_py quest3_webxr_bridge.launch.py \
    params_file:=src/teleop_control_py/config/quest3_webxr_bridge_params.yaml
```

### Teleop 参数

Quest 3 遥操作参数位于 `teleop_params.yaml`：

- `quest3_active_hand`
- `quest3_motion_mode`
- `quest3_require_connected`
- `quest3_pose_timeout_sec`
- `quest3_linear_axis_mapping`
- `quest3_angular_axis_mapping`
- `quest3_orientation_mode`
- `quest3_clutch_*`
- `quest3_frame_reset_*`
- `quest3_gripper_*`

默认控制方式是相对位姿加 clutch。按住 clutch 后，系统记录控制器和机器人起点，然后用控制器相对位姿驱动目标 TCP 位姿。

### 默认输入语义

常用默认语义：

| 参数 | 默认含义 |
| --- | --- |
| `quest3_active_hand` | 当前控制手，默认按配置选择 |
| `quest3_motion_mode` | 相对位姿目标模式 |
| `quest3_orientation_mode` | `hand_relative` |
| `quest3_gripper_requires_clutch` | 夹爪命令需要 clutch |
| `quest3_enable_input_smoothing` | 默认关闭输入低通 |
| `quest3_frame_reset_scope` | 默认只允许 active hand 重置 |

按钮和轴映射来自 `/quest3/input/joy`，具体数值以 `teleop_params.yaml` 为准。

### Frame Reset

Quest 3 支持类似 Quest2ROS 的相对 frame reset，但默认关闭。当前默认 `target_pose + clutch`
模式下，每次按住 clutch 都会重新锚定相对位姿起点，因此 frame reset 通常不是必需操作。

- `quest3_frame_reset_enabled`
- `quest3_frame_reset_scope`
- `quest3_frame_reset_hold_sec`
- `quest3_frame_reset_rotate_position`
- `quest3_left_frame_reset_buttons`
- `quest3_right_frame_reset_buttons`

启用后，默认 `quest3_frame_reset_scope` 为 `active_hand`，只允许当前控制手触发 reset。

## 维护注意事项

- 修改机械臂、夹爪 topic、service 或 controller 默认值时，同步更新 `robot_profiles.yaml` 和 [08-configuration.md](08-configuration.md)。
- 修改输入映射时，同步检查对应 input handler、`teleop_params.yaml` 和本文。
- 修改 Quest bridge topic 时，同时更新 `teleop_params.yaml` 和本文。
- WebXR 访问问题通常和 TLS、局域网 IP 或浏览器权限有关，优先检查 bridge 日志中打印的 Quest 访问 URL。
