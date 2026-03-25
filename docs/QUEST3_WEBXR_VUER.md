# Quest 3 WebXR Bridge

这个桥接层先只解决 Quest 3 到 ROS2 的数据传输，不直接接入机械臂控制。

## 1. 设计目标

- Quest 3 不需要侧载原生 App
- 直接用 Quest Browser + WebXR
- Python 侧通过 `vuer` 收控制器流
- ROS2 侧统一发布：
  - `/quest3/left_controller/pose`
  - `/quest3/right_controller/pose`
  - `/quest3/left_controller/matrix`
  - `/quest3/right_controller/matrix`
  - `/quest3/input/joy`
  - `/quest3/connected`

## 2. 安装依赖

`vuer` 不是 ROS2 系统包，需额外安装：

```bash
pip install "vuer>=0.1.4,<0.2"
```

## 3. 启动 bridge

```bash
ros2 launch teleop_control_py quest3_webxr_bridge.launch.py
```

或：

```bash
ros2 run teleop_control_py quest3_webxr_bridge_node \
  --ros-args \
  --params-file src/teleop_control_py/config/quest3_webxr_bridge_params.yaml
```

## 4. Quest 3 访问方式

当前推荐配置默认关闭自签名 TLS，直接走本机 `http/ws`。
前提是你已经在 Quest Browser 的实验/flags 中把该局域网地址加入了安全豁免。

推荐方式：

1. 在 PC 本地运行 bridge
2. 在 Quest Browser 直接打开：

```text
http://<pc_lan_ip>:8012
```

如果你不想走本机 `http/ws`，也仍然可以走外部 TLS 反代：

1. 用 ngrok / localtunnel / 自己的 TLS 反代把本地 websocket 暴露成 `wss://...`
2. 在参数 `public_wss_url` 中填入这个公网或局域网可访问的 `wss://...`
3. 在 Quest Browser 打开：

```text
https://vuer.ai?ws=<public_wss_url>
```

注意：

- 如果没有给局域网地址做浏览器安全豁免，普通 `http://<pc-ip>` 可能无法稳定工作
- WebXR 在头显浏览器中通常要求安全上下文
- 因此至少 websocket 侧应是 `wss://`

## 5. Vuer 控制器事件

bridge 当前按 `vuer` Motion Controller 流处理 `CONTROLLER_MOVE` 事件。

接收到的数据包含：

- `left` / `right`：左右控制器 `Matrix4`
- `leftState` / `rightState`：扳机、握把、触摸板、摇杆和按钮状态

bridge 会把 `Matrix4` 原样转成 `Float32MultiArray`，同时转换成 `PoseStamped`。

## 6. `/quest3/input/joy` 映射

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

## 7. 后续接 teleop 的建议

先不要直接把 Quest 3 映射到现有 `joy` 模式。

更合适的下一步是：

1. 新增 `Quest3InputHandler`
2. 订阅 `/quest3/right_controller/pose` 与 `/quest3/input/joy`
3. 复用现有 `mediapipe target_pose` 的“相对起始位姿 -> TCP 目标位姿”控制逻辑
