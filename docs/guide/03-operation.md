# 运行指南

更新时间：2026-05-06

本文记录项目的日常启动方式。架构说明见 [01-architecture.md](01-architecture.md)，配置文件职责见 [08-configuration.md](08-configuration.md)。

## 环境

建议环境：

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- UR driver、MoveIt Servo 和对应夹爪驱动已安装

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

如果使用本地 `Real_IL` 推理：

```bash
git clone https://github.com/niconekomimi/Real_IL.git Real_IL
pip install -r Real_IL/requirements.txt
```

编译工作区：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select teleop_control_py
source install/setup.bash
```

## GUI 启动

推荐入口：

```bash
ros2 run teleop_control_py teleop_gui
```

推荐操作顺序：

1. 选择 `ur_type`、机器人 IP、输入后端和夹爪类型
2. 启动机械臂驱动
3. 启动遥操作系统
4. 启动采集节点
5. 使用录制、停止录制、弃用最近 demo、Home、Home Zone 等功能
6. 需要模型控制时，先启动推理，再单独打开推理执行

GUI 的进程管理逻辑在 `gui/app_service.py` 和 `gui/support.py` 中。GUI 保存的默认值来自 `config/gui_params.yaml`。

## GUI 按钮与启动命令

GUI 主窗口负责交互入口，主要运行操作会进入 `GuiAppService`：

| GUI 操作 | 主要调用 | 运行效果 |
| --- | --- | --- |
| 启动机械臂驱动 | `start_robot_driver()` | 启动 `control_system.launch.py`，但设置 `launch_teleop_node:=false` |
| 停止机械臂驱动 | `stop_robot_driver()` | 关闭 `robot_driver` 子进程 |
| 启动遥操作系统 | `start_teleop()` | 启动 `control_system.launch.py`，按 GUI 选择启用 `joy`、`mediapipe` 或 `quest3` |
| 停止遥操作系统 | `stop_teleop()` | 关闭 `teleop` 子进程 |
| 启动采集节点 | `start_data_collector()` | 单独启动 `data_collector_node` 并传入相机、输出路径和末端类型 |
| 开始 / 停止录制 | `start_record()` / `stop_record()` | 通过 `ROS2Worker` 调用 `/data_collector/start` 或 `/data_collector/stop` |
| Go Home / Go Home Zone | `go_home()` / `go_home_zone()` | 通过 `ROS2Worker` 调用 `/commander/*` 服务 |
| 启动推理 | `InferenceService.start_inference()` | 启动本地或远端推理 worker |
| 启用推理执行 | `enable_inference_execution()` | 让 `ROS2Worker` 把推理动作下发到机器人 |

这层分工可以让 GUI 主窗口保持清晰：主窗口负责交互，`GuiAppService` 负责运行编排，ROS 控制和采集由对应节点承担。

## 命令行启动

启动整套控制系统：

```bash
ros2 launch teleop_control_py control_system.launch.py
```

常用组合：

```bash
ros2 launch teleop_control_py control_system.launch.py input_type:=joy gripper_type:=robotiq
ros2 launch teleop_control_py control_system.launch.py input_type:=joy gripper_type:=qbsofthand
ros2 launch teleop_control_py control_system.launch.py input_type:=mediapipe gripper_type:=robotiq
ros2 launch teleop_control_py control_system.launch.py input_type:=quest3 gripper_type:=robotiq
```

同时启动采集节点：

```bash
ros2 launch teleop_control_py control_system.launch.py \
    input_type:=joy \
    gripper_type:=robotiq \
    enable_data_collector:=true
```

只启动机器人驱动和 commander，不启动 teleop 节点：

```bash
ros2 launch teleop_control_py control_system.launch.py \
    launch_teleop_node:=false \
    gripper_type:=robotiq
```

单独启动采集节点：

```bash
ros2 run teleop_control_py data_collector_node \
    --ros-args \
    --params-file src/teleop_control_py/config/data_collector_params.yaml
```

查看 launch 参数：

```bash
ros2 launch teleop_control_py control_system.launch.py --show-args
ros2 launch teleop_control_py teleop_control.launch.py --show-args
```

## `control_system.launch.py` 做什么

`control_system.launch.py` 是主启动入口，会按参数组合启动：

- UR driver：`ur_robot_driver/ur_control.launch.py`
- MoveIt / Servo：`ur_moveit_config/ur_moveit.launch.py`
- 夹爪驱动：Robotiq 或 qbSoftHand
- `robot_commander_node`
- 可选 `teleop_control_node`
- 可选 `joy_driver_node`
- 可选 `quest3_webxr_bridge_node`
- 可选 `data_collector_node`

常用控制参数：

| 参数 | 说明 |
| --- | --- |
| `ur_type` | UR 机器人型号，例如 `ur5` |
| `robot_profile` | 后端机器人画像，默认 `ur_servo_<ur_type>` |
| `robot_ip` | UR 控制器 IP |
| `reverse_ip` | 机器人反向连接到本机的 IP |
| `input_type` | `joy`、`mediapipe` 或 `quest3` |
| `gripper_type` | `robotiq` 或 `qbsofthand` |
| `enable_moveit` | 是否启动 MoveIt / Servo |
| `enable_data_collector` | 是否随控制系统启动采集节点 |
| `launch_teleop_node` | 是否启动遥操作节点 |
| `launch_quest3_bridge` | `auto`、`true` 或 `false` |

## 输入后端入口

`joy`：

- `control_system.launch.py` 会包含 `joy_driver.launch.py`
- `joy_driver_node` 发布 `/joy`
- `teleop_control_node` 读取 `/joy` 并生成 `ActionCommand`

`mediapipe`：

- `teleop_control_node` 可直接使用 SDK 相机或订阅 ROS 图像 topic
- 相关参数在 `teleop_params.yaml` 的 `mediapipe_*` 段
- MediaPipe 只负责人工输入，不负责数据集采集

`quest3`：

- `launch_quest3_bridge:=auto` 时会自动启动 bridge
- bridge 发布 `/quest3/*` topic
- `teleop_control_node` 通过 `Quest3InputHandler` 转成机器人动作

更细的设备和 Quest 说明见 [07-devices.md](07-devices.md)。

## 常用服务

采集服务：

```bash
ros2 service call /data_collector/start std_srvs/srv/Trigger {}
ros2 service call /data_collector/stop std_srvs/srv/Trigger {}
ros2 service call /data_collector/discard_last_demo std_srvs/srv/Trigger {}
```

机器人能力服务：

```bash
ros2 service call /commander/go_home std_srvs/srv/Trigger {}
ros2 service call /commander/go_home_zone std_srvs/srv/Trigger {}
ros2 service call /commander/cancel_home_zone std_srvs/srv/Trigger {}
```

## 运行提示

- `input_type:=quest3` 且 `launch_quest3_bridge:=auto` 时，`control_system.launch.py` 会自动启动 Quest 3 bridge。
- `control_mode` 和 `end_effector` 仍作为旧参数别名保留，但新文档和新脚本应使用 `input_type` 和 `gripper_type`。
- `robot_profile` 默认按 `ur_type` 生成，例如 `ur_type:=ur5` 对应 `ur_servo_ur5`。
- 数据采集依赖双相机和机器人状态都可用；录制失败时优先看 `data_collector_node` 日志。
