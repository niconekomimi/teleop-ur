中文 | [English](QUEST3_WEBXR_VUER_EN.md)

# Quest 3 WebXR / Teleop 集成说明

这份文档描述当前仓库里已经落地的 Quest 3 支持，包括：

- `quest3_webxr_bridge_node` 如何把 Quest Browser 的 WebXR 控制器流转成 ROS2 话题
- `Quest3InputHandler` 如何把控制器位姿/按键接入现有 teleop 架构
- 当前推荐的启动方式、Quest 入口 URL、调试方式和用户操作习惯

## 1. 当前已实现能力

当前仓库里，Quest 3 已经不是“仅桥接原型”，而是正式输入后端：

- bridge 节点：
  - `teleop_control_py/nodes/quest3_webxr_bridge_node.py`
- teleop 输入后端：
  - `teleop_control_py/hardware/input/quest3_input_handler.py`
- launch 支持：
  - `control_system.launch.py` 支持 `input_type:=quest3`
  - `teleop_control.launch.py` 支持 `input_type:=quest3`
  - `quest3_webxr_bridge.launch.py` 支持单独启动 bridge
- GUI：
  - 输入源下拉框已经提供 `quest3 (VR 控制器)`

当前 Quest 相关 ROS 话题：

- `/quest3/left_controller/pose`
- `/quest3/right_controller/pose`
- `/quest3/left_controller/matrix`
- `/quest3/right_controller/matrix`
- `/quest3/input/joy`
- `/quest3/connected`

## 2. 推荐控制方法

当前 Quest 遥操作默认使用：

- `target_pose`
- `hand_relative`
- `clutch / deadman`
- 输入层默认关闭低通平滑

也就是：

1. 按住 `clutch`
2. 记录当前控制器位姿和当前机器人 TCP 位姿
3. 用控制器相对起始位姿变化去驱动 TCP 目标位姿
4. 松开 `clutch` 后停止输出

默认交互语义：

- `active_hand = right`
- 右手 `grip / squeeze`：clutch
- 右手 `trigger`：夹爪
- 默认 `orientation_mode = hand_relative`

这套思路更接近 Quest2ROS 一类成熟 VR controller teleop，而不是视觉手势滤波链路。

## 3. Quest2ROS 风格的相对 frame 重置

当前实现里已经加入相对参考系重置。

默认参数：

- `quest3_frame_reset_enabled: true`
- `quest3_frame_reset_scope: "active_hand"`
- `quest3_frame_reset_hold_sec: 0.75`

默认按钮：

- 左手：`X + Y`，也就是 `buttons [4, 5]`
- 右手：`A + B`，也就是 `buttons [10, 11]`

行为语义：

- 长按组合键约 `0.75s`
- 把当前该手控制器姿态记为新的相对参考系
- 之后的 `clutch / pose delta / hand_relative orientation` 都基于这个参考系解释

这和 `clutch` 不是一回事：

- `frame reset` 解决“坐标语义是否自然”
- `clutch` 解决“何时开始拖动机器人”

## 4. 依赖与环境

`vuer` 不是 ROS2 系统包，需要额外安装。

如果你使用当前项目常见的 `clds` 环境：

```bash
source /home/rvl/clds/bin/activate
pip install "vuer>=0.1.4,<0.2"
```

然后编译：

```bash
source /opt/ros/humble/setup.bash
python -m colcon build --packages-select teleop_control_py
source install/setup.bash
```

## 5. 网络与入口 URL

Quest 3 最重要的要求是：

- Quest 和 PC 在同一局域网
- Quest 使用 PC 的 Wi-Fi IPv4，不要误用 UR5 的有线网卡 IP

当前项目推荐入口：

```text
https://vuer.ai?ws=wss://<pc_wifi_ip>:8012
```

例如：

```text
https://vuer.ai?ws=wss://192.168.137.227:8012
```

说明：

- `https://<pc_ip>:8012` 直连页有时能打开
- 但当前代码和实测都更推荐 `vuer.ai?ws=wss://...` 这个入口
- 第一次访问通常需要在 Quest Browser 手动接受一次证书

## 6. 推荐启动方式

### 6.1 一体化启动

如果你想让 `control_system.launch.py` 自动把 Quest bridge 一起带起来：

```bash
ros2 launch teleop_control_py control_system.launch.py \
  input_type:=quest3
```

当前行为：

- `input_type:=quest3`
- `launch_quest3_bridge:=auto`

时，`control_system.launch.py` 会自动启动 `quest3_webxr_bridge_node`。

### 6.2 推荐的分离调试方式

开发期更推荐把 bridge 和 teleop 分开。

终端 1：bridge 常驻

```bash
ros2 launch teleop_control_py quest3_webxr_bridge.launch.py
```

终端 2：teleop 单独重启

```bash
ros2 launch teleop_control_py teleop_control.launch.py input_type:=quest3
```

这样可以：

- 不反复重启 Quest 页面
- 不反复重新建立 WebXR 控制器连接
- 只重启 teleop 节点来调整参数、映射和跟手度

## 7. Quest 端操作步骤

1. 在 PC 上启动 bridge
2. 戴上 Quest，打开 `Quest Browser`
3. 访问：

```text
https://vuer.ai?ws=wss://<pc_wifi_ip>:8012
```

4. 如果出现证书警告，手动继续访问
5. 进入网页后点击 `Enter VR` / `Enter XR`
6. 使用控制器进入 VR 模式

如果你通过 GUI 启动 `quest3` 模式，GUI 里现在也会明确提示：

- 请在 Quest 中进入对应网页
- 请点击进入 VR 模式

## 8. ROS2 话题验证

连接成功后，可先验证：

```bash
ros2 topic echo /quest3/connected --once
```

正常会看到：

```text
data: true
```

再验证控制器数据：

```bash
ros2 topic echo /quest3/input/joy --once
ros2 topic echo /quest3/right_controller/pose --once
```

## 9. `/quest3/input/joy` 映射

`axes` 顺序：

1. `left.triggerValue`
2. `left.squeezeValue`
3. `left.touchpadX`
4. `left.touchpadY`
5. `left.thumbstickX`
6. `left.thumbstickY`
7. `right.triggerValue`
8. `right.squeezeValue`
9. `right.touchpadX`
10. `right.touchpadY`
11. `right.thumbstickX`
12. `right.thumbstickY`

`buttons` 顺序：

1. `left.trigger`
2. `left.squeeze`
3. `left.touchpad`
4. `left.thumbstick`
5. `left.xButton` / `left.aButton`
6. `left.yButton` / `left.bButton`
7. `right.trigger`
8. `right.squeeze`
9. `right.touchpad`
10. `right.thumbstick`
11. `right.aButton`
12. `right.bButton`

## 10. 当前参数入口

Quest 遥操作主要参数都在：

- `src/teleop_control_py/config/teleop_params.yaml`

其中包括：

- bridge 默认参数
- Quest teleop 映射参数
- `frame reset` 参数
- smoothing / clutch / trigger / active hand 参数

当前比较关键的 Quest 参数：

- `quest3_motion_mode`
- `quest3_orientation_mode`
- `quest3_enable_input_smoothing`
- `quest3_frame_reset_enabled`
- `quest3_frame_reset_scope`
- `quest3_left_frame_reset_buttons`
- `quest3_right_frame_reset_buttons`

## 11. 当前限制

当前 Quest 页面主要解决：

- 控制器输入
- ROS2 遥操作接入

还没有在 Quest 页面内完整集成：

- 相机悬浮预览窗
- 控制器触觉反馈
- 更完整的 Quest 端状态 UI

这些都可以在当前 bridge + WebXR 结构上继续往上叠。

## 12. 参考

- Quest2ROS 项目主页：
  - https://quest2ros.github.io/
- Quest2ROS 论文 PDF：
  - https://quest2ros.github.io/files/Quest2ROS.pdf
- OpenXR reference space：
  - https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html
