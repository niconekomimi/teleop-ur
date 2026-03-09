# teleop-ur

这是一个面向 UR 机械臂的遥操作与数据采集工作区，当前主使用方式是通过图形界面统一启动相机、机械臂、遥操作链路和数据采集。

项目当前默认组合为：

- 输入设备：手柄 joy
- 夹爪类型：robotiq
- 机械臂控制：MoveIt Servo
- 数据采集格式：HDF5
- 主入口：GUI

## 主要入口

本项目最重要、最推荐的程序入口是 GUI。

启动前先完成编译并加载环境：

```bash
colcon build --packages-select teleop_control_py
source install/setup.bash
```

然后启动 GUI：

```bash
ros2 run teleop_control_py teleop_gui
```

如果你在源码态调试，也可以使用：

```bash
python3 scripts/teleop_gui.py
```

GUI 负责统一管理以下内容：

- 启动和停止相机驱动
- 启动和停止机械臂驱动
- 启动整套遥操作系统
- 启动和停止数据采集
- 录制轨迹、停止录制、回 Home、设置当前 Home
- 查看实时状态、相机预览和 HDF5 数据
- 输入 `ur_type`、IP、采集路径等常用参数

如果你只是正常使用系统，优先用 GUI，不需要手动分别启动各个节点。

## 系统说明

这个项目不是单一脚本，而是一套可组合的遥操作与采集系统，支持：

- 多输入后端：`joy`、`mediapipe`
- 多末端执行器：`robotiq`、`qbsofthand`
- 多控制器协同：`forward_position_controller`、`scaled_joint_trajectory_controller`
- 统一数据采集：全局相机、腕部相机、关节状态、末端位姿、夹爪状态、动作

当前主链路如下：

1. 输入设备产生控制信号。
2. `teleop_control_node` 负责做输入解析、限幅、平滑和夹爪控制。
3. MoveIt Servo 接收速度命令并驱动 UR 机械臂。
4. `data_collector_node` 采集图像和机器人状态并写入 HDF5。
5. GUI 负责启动、监控和管理整套流程。

## 目录说明

工作区里当前最关键的路径如下：

- `src/teleop_control_py/launch/control_system.launch.py`：整套系统启动文件
- `src/teleop_control_py/launch/teleop_control.launch.py`：仅遥操作相关启动文件
- `src/teleop_control_py/config/teleop_params.yaml`：遥操作参数
- `src/teleop_control_py/config/data_collector_params.yaml`：数据采集参数
- `src/teleop_control_py/config/gui_params.yaml`：GUI 默认参数
- `src/teleop_control_py/teleop_control_py/gui/app.py`：GUI 包入口
- `src/teleop_control_py/teleop_control_py/data_collector_node.py`：数据采集节点
- `src/teleop_control_py/teleop_control_py/hdf5_writer.py`：HDF5 写入线程

## 快速开始

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 编译工作区

```bash
colcon build --packages-select teleop_control_py
source install/setup.bash
```

### 3. 启动 GUI

```bash
ros2 run teleop_control_py teleop_gui
```

### 4. 在 GUI 中完成常见操作

推荐使用顺序：

1. 选择机械臂型号、IP、输入模式和夹爪类型。
2. 启动相机驱动。（如需手势遥操）
3. 启动机械臂驱动。(可以不启动，遥操作系统已包括)
4. 启动遥操作系统。
5. 启动采集节点。
6. 开始录制数据。
7. 录制完成后停止录制并保存 HDF5。

## 手动启动方式

虽然 GUI 是主入口，但如果你需要调试，也可以手动启动。

### 启动整套系统

```bash
ros2 launch teleop_control_py control_system.launch.py
```

这个启动文件会按参数自动组合：

- UR 驱动
- MoveIt / Servo
- 输入设备链路
- 夹爪驱动
- `teleop_control_node`
- 可选的 `data_collector_node`

### 常见启动示例

手柄 + Robotiq：

```bash
ros2 launch teleop_control_py control_system.launch.py \
    input_type:=joy \
    gripper_type:=robotiq \
    robotiq_serial_port:=/dev/robotiq_gripper
```

手柄 + qbSoftHand：

```bash
ros2 launch teleop_control_py control_system.launch.py \
    input_type:=joy \
    gripper_type:=qbsofthand
```

MediaPipe + Robotiq：

```bash
ros2 launch teleop_control_py control_system.launch.py \
    input_type:=mediapipe \
    gripper_type:=robotiq
```

整套系统 + 数据采集：

```bash
ros2 launch teleop_control_py control_system.launch.py \
    input_type:=joy \
    gripper_type:=robotiq \
    enable_data_collector:=true
```

### 只启动遥操作部分

如果 UR 驱动、MoveIt、夹爪驱动已经单独启动：

```bash
ros2 launch teleop_control_py teleop_control.launch.py
```

### 单独启动数据采集节点

```bash
ros2 run teleop_control_py data_collector_node \
    --ros-args \
    --params-file src/teleop_control_py/config/data_collector_params.yaml
```

常用服务：

```bash
ros2 service call /data_collector/start std_srvs/srv/Trigger {}
ros2 service call /data_collector/stop std_srvs/srv/Trigger {}
ros2 service call /data_collector/go_home std_srvs/srv/Trigger {}
```

## 当前数据采集格式

当前数据集写入 HDF5，按 `data/demo_编号` 组织。

典型内容包括：

- `obs/agentview_rgb`
- `obs/eye_in_hand_rgb`
- `obs/robot0_joint_pos`
- `obs/robot0_gripper_qpos`
- `obs/robot0_eef_pos`
- `obs/robot0_eef_quat`
- `actions`

当前 `actions` 的语义是命令动作，而不是机械臂执行后的绝对位姿。格式为：

```text
[vx, vy, vz, wx, wy, wz, gripper]
```

其中：

- 前 3 维是末端线速度命令
- 中间 3 维是末端角速度命令
- 最后 1 维是夹爪命令

这套动作定义更适合后续做模仿学习。

## 常用配置文件

### 遥操作参数

文件：`src/teleop_control_py/config/teleop_params.yaml`

常改项：

- `input_type`
- `gripper_type`
- `joy_deadzone`
- `joy_curve`
- `input_watchdog_timeout_sec`
- `max_linear_vel`
- `max_angular_vel`
- `max_linear_accel`
- `max_angular_accel`
- `teleop_controller`
- `trajectory_controller`

### 采集参数

文件：`src/teleop_control_py/config/data_collector_params.yaml`

常改项：

- `output_path`
- `global_camera_source`
- `wrist_camera_source`
- `home_joint_positions`
- `servo_twist_topic`
- `pose_topic`
- `gripper_state_topic`

### GUI 参数

文件：`src/teleop_control_py/config/gui_params.yaml`

主要用于保存：

- 默认机械臂型号
- 默认 IP
- 默认输入模式
- 默认夹爪类型
- 默认 HDF5 输出目录和文件名
- 预览图像话题
- Home 关节位置

## 输入与夹爪支持

### 输入后端

- `joy`：当前默认方案，适合低延迟人工遥操作
- `mediapipe`：基于图像输入的手势控制模式

当前默认手柄映射大致为：

- 左摇杆：平移
- 右摇杆：旋转
- `A / B`：Z 方向运动
- `X / Y`：绕 Z 轴旋转
- `LB / RB`：夹爪开合

### 末端执行器

- `robotiq`
- `qbsofthand`

系统已经对这两类夹爪的控制接口做了统一封装。

## 依赖说明

至少需要具备以下环境：

- ROS 2 Humble
- UR 机械臂驱动
- MoveIt Servo
- 相机相关依赖
- 对应夹爪驱动
- Python 依赖见 `requirements.txt`

## 使用建议

- 正常使用时，优先通过 GUI 操作整套系统。
- 如果修改了 `teleop_control_py` 源码，记得重新编译并重新 `source install/setup.bash`。
- 如果打开旧的 HDF5 文件，GUI 会尽量兼容旧数据结构，但新采集数据建议统一使用当前格式。
