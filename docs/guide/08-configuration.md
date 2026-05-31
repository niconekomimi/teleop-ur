# 配置说明

更新时间：2026-05-06

项目配置按职责拆分。调整配置前，先判断该值属于硬件接口、运行行为、GUI 偏好，还是启动覆盖，再决定修改哪个文件。

## 配置文件分工

| 文件 | 职责 |
| --- | --- |
| `src/teleop_control_py/config/robot_profiles.yaml` | 机器人画像。机械臂模型、关节名、ROS topic / service、controller、Home、Home Zone、gripper 接口默认来源 |
| `src/teleop_control_py/config/teleop_params.yaml` | 遥操作行为。输入后端、控制频率、速度/加速度限制、joy / mediapipe / quest3 映射、MoveIt Servo 覆盖、Robotiq 启动参数 |
| `src/teleop_control_py/config/data_collector_params.yaml` | 采集行为。输出路径、采样频率、相机来源、时间新鲜度阈值、写盘队列、Preview API |
| `src/teleop_control_py/config/gui_params.yaml` | GUI 默认值和用户偏好。IP、输入选择、相机偏好、HDF5 文件名、推理后端和 openpi 默认连接信息 |
| `src/teleop_control_py/config/home_overrides.yaml` | GUI 写入的 Home 点覆盖 |
| `src/teleop_control_py/config/joy_driver_params.yaml` | 手柄设备发现、发布频率、deadzone 和自动重连 |
| `src/teleop_control_py/config/quest3_webxr_bridge_params.yaml` | 单独启动 Quest bridge 时可使用的参数文件 |

## `robot_profiles.yaml`

这是底层接口默认值的统一配置来源。新增机器人或更换控制链路时，优先在这里扩展 profile。

当前 `ur_servo_ur5` 定义了：

- `backend` 和 `arm_model`
- `target_frame_id`
- 6 轴 `joint_names`
- 关键 topic：`joint_states`、`tool_pose`、`servo_twist`、`home_joint_trajectory`
- 关键 service：`start_servo`、`controller_manager_ns`
- controller：`teleop`、`trajectory`
- gripper 默认类型和 Robotiq / qbSoftHand 接口参数
- Home 关节角和 Home Zone 采样范围

配置建议：

- topic、service、controller 名称集中放在 profile 中维护。
- 新增机器人 profile 后，同步检查 `gui_params.yaml` 中 GUI 默认项是否需要更新。
- Home 点属于机器人 profile 的默认值；运行时通过 GUI 设置的 Home 点写入 `home_overrides.yaml`。

## `teleop_params.yaml`

这个文件描述“如何遥操作”，不作为硬件接口的统一来源。

主要内容：

- `input_type`：`joy`、`mediapipe` 或 `quest3`
- `gripper_type`：`robotiq` 或 `qbsofthand`
- `control_hz`
- 速度和加速度限制
- joy 轴、按钮、deadzone 和曲线
- Quest 3 clutch、frame reset、坐标映射和 gripper 控制
- MediaPipe 图像输入、深度、deadman 和手势映射
- `moveit_servo` 覆盖项
- `robotiq_gripper` 启动参数
- 集成启动用的 `quest3_webxr_bridge_node` 参数块

`control_system.launch.py` 会从这里读取输入、夹爪、MoveIt Servo、Robotiq 和 Quest bridge 的默认值。

GUI 的“遥操作设置”弹窗会把应用后的行为参数写回源码树里的 `teleop_params.yaml`。GUI 启动 teleop / robot driver 时会显式传入这个源码参数文件作为 `params_file`，因此调参后不需要重新 build；已经运行中的 `teleop_control_node` 不做热更新，重启遥操作系统后生效。

## `data_collector_params.yaml`

这个文件描述“如何采集和写盘”。

主要内容：

- `output_path`
- `record_fps`
- 全局相机和腕部相机来源
- 相机 serial number 和 depth 开关
- joint / pose / command / gripper / image 的新鲜度阈值
- `camera_pair_max_skew_sec`
- `obs_image_size`
- `image_compression`
- writer 队列和 batch 设置
- Preview API host、port、JPEG 质量

采集节点运行时还会从 `robot_profile` 读取默认 topic、joint names 和 gripper state topic。

## `gui_params.yaml`

这是 GUI 偏好文件。GUI 会读取它作为默认显示值，也会在用户修改 GUI 设置时持久化部分字段。

它适合保存：

- 默认机器人 IP 和 reverse IP
- GUI 默认输入后端、夹爪、joy profile
- 相机型号和 serial number 偏好
- HDF5 输出目录和默认文件名
- 本地 `Real_IL` 推理默认路径
- `openpi_remote` 的 host、port 和 prompt 默认值
- `teleop_settings`：GUI 遥操作设置弹窗中的 joy / mediapipe / quest3 自定义方案、当前选中方案和默认方案

`teleop_settings` 只保存 GUI 方案和选择状态；真正影响运行的速度、加速度、键位和滤波值仍以应用后写入的 `teleop_params.yaml` 为准。

下面这些内容更适合放在机器人画像或行为配置中：

- ROS topic / service 默认来源
- controller 名称
- Home Zone 控制参数
- gripper 硬件接口定义

这些值应放在 `robot_profiles.yaml` 或对应行为配置中。

## Launch 覆盖规则

`control_system.launch.py` 的常见解析顺序：

- 显式 launch argument 优先
- 然后读取参数文件
- 最后使用代码默认值

例如：

- `input_type` 为空时，会从 `teleop_params.yaml` 读取；仍为空则回退到 `joy`
- `gripper_type` 为空时，会从 `teleop_params.yaml` 读取；仍为空则回退到 `robotiq`
- `launch_quest3_bridge:=auto` 时，只有 `input_type` 解析为 `quest3` 才启动 bridge
- `robot_profile` 默认由 `ur_type` 拼成 `ur_servo_<ur_type>`

调整 launch 参数时，同步更新 [03-operation.md](03-operation.md)。

## 已移除或不再作为主入口的配置

`robot_commander_params.yaml` 不再是启动链路的一部分。`robot_commander_node` 的关键参数来自：

- `robot_profiles.yaml`
- launch 中少量显式覆盖，例如 `commander_pose_max_age_sec`
- GUI 写入的 `home_overrides.yaml`

调整 commander 行为时，优先沿用现有配置入口，避免重新引入分散的 commander 参数文件。

## 配置放置建议

建议避免以下做法：

- 把 ROS topic、service 或 controller 名称写进 GUI 偏好文件
- 在 launch 文件里长期维护 gripper topic 默认来源
- 在 `teleop_params.yaml` 中重复维护 `robot_profiles.yaml` 已经定义的底层接口值
- 把采集输出路径和机器人接口定义放在同一个配置层
- 让 README 成为参数手册

如果一个参数同时影响多个节点，优先判断它是否应该进入 `robot_profiles.yaml`。
