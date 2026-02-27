# multi_joy_driver

可扩展 ROS2 手柄驱动节点，统一发布 `/joy` (`sensor_msgs/msg/Joy`)。

## 特性

- 支持 `auto` 自动识别手柄类型
- 内置 `xbox` / `ps5` / `generic` profile
- 可通过 profile 机制扩展更多手柄
- 支持断开重连

## 运行

```bash
colcon build --packages-select multi_joy_driver
source install/setup.bash
ros2 launch multi_joy_driver joy_driver.launch.py
```

## 常用参数

- `profile`: `auto | xbox | ps5 | generic`
- `device_path`: 指定 `/dev/input/eventX`
- `device_name`: 按设备名关键字匹配
- `deadzone`: 摇杆死区
- `publish_rate_hz`: 发布频率
- `autoreconnect`: 是否自动重连

## 扩展新手柄

在 `multi_joy_driver/device_profiles.py` 中新增 profile 函数并注册到 `build_profiles()` 即可。
