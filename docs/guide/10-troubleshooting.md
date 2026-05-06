# 常见问题排查

更新时间：2026-05-06

本文记录运行时常见问题的排查入口。它不替代具体日志分析，主要用于快速定位应该先看哪条链路。

## UR driver 连接不上

优先检查：

- 机器人 IP 是否和 GUI / launch 参数一致
- 电脑有线网卡是否在机器人同网段
- UR 控制器是否处于 remote control 模式
- `control_system.launch.py` 中的 `ur_type`、`robot_ip`、`launch_rviz` 等参数
- UR driver 和 dashboard 日志

相关文档：

- [03-operation.md](03-operation.md)
- [07-devices.md](07-devices.md)
- [08-configuration.md](08-configuration.md)

## MoveIt Servo 或 controller 没起来

优先检查：

- `ros2 control list_controllers`
- trajectory controller 和 servo controller 是否按预期切换
- `robot_profiles.yaml` 中 controller 名称和 topic
- Home / Home Zone 是否正在占用控制链路

相关文档：

- [04-control-flow.md](04-control-flow.md)
- [08-configuration.md](08-configuration.md)

## 夹爪没有响应

优先检查：

- GUI 中选择的 `gripper_type`
- Robotiq 串口或 qbSoftHand 连接是否可见
- `robot_profiles.yaml` 中 gripper topic / service / command 参数
- 对应 gripper driver 日志

相关文档：

- [07-devices.md](07-devices.md)
- [08-configuration.md](08-configuration.md)

## Quest 页面连不上

优先检查：

- Quest 和电脑是否在同一可访问局域网内
- `quest3_advertised_host` 是否是 Quest 能访问的电脑 IP
- 访问 URL 是否使用 bridge 日志打印的 HTTPS / WSS 地址
- Quest Browser 是否已经接受自签名证书
- `/quest3/connected` topic 是否发布连接状态

相关文档：

- [07-devices.md](07-devices.md)

## 相机画面或采集不可用

优先检查：

- GUI 中选择的 global / wrist camera source
- camera serial number 是否和实际设备一致
- `data_collector_params.yaml` 中的相机配置
- collector Preview API 是否运行
- 推理 preview 是否被 collector preview 接管

相关文档：

- [05-data-collection.md](05-data-collection.md)
- [06-inference.md](06-inference.md)
- [07-devices.md](07-devices.md)

## HDF5 没有写入

优先检查：

- `data_collector_node` 是否已经启动
- `/data_collector/start` 服务是否成功返回
- 双相机客户端是否初始化完成
- joint、pose、action、gripper 状态是否满足同步新鲜度要求
- 输出路径是否指向预期数据目录

相关文档：

- [05-data-collection.md](05-data-collection.md)

## 推理 worker ready 但没有动作输出

优先检查：

- 推理是否只是启动，还是已经启用 inference execution
- 当前是否处于 estop、teleop、Home 或 Home Zone 冲突状态
- `ROS2Worker` 是否已经创建 arm / gripper backend
- 模型输出动作维度和 gripper 语义是否符合当前执行链路
- `ActionMux` 当前 active source 是否允许 `INFERENCE`

相关文档：

- [04-control-flow.md](04-control-flow.md)
- [06-inference.md](06-inference.md)
