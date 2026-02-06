# `main.cpp` 的 API 使用说明

本文档概述了 `qbSHR_API_example` 项目中 `main.cpp` 文件使用的 API 函数。每个 API 函数的用途和使用方法均有详细说明。

---

## 1. `open(const std::string &serial_port)`
- **用途**: 打开用于通信的串口。
- **使用方法**:
  - 使用正则表达式验证串口名称。
  - 调用 `communication_handler_->openSerialPort()` 打开串口。
- **返回值**:
  - 成功返回 `0`。
  - 如果串口名称无效或无法打开，返回 `-1`。

---

## 2. `scanForDevices(const int &max_repeats)`
- **用途**: 扫描可用串口上的 qbrobotics 设备。
- **使用方法**:
  - 遍历所有串口并尝试打开。
  - 调用 `communication_handler_->listConnectedDevices()` 获取设备 ID。
  - 过滤无效的设备 ID（例如 `120` 或 `0`）。
  - 为有效设备创建 `qbSoftHandLegacyResearch` 对象。
- **返回值**:
  - 找到的 qbrobotics 设备数量。

---

## 3. `getInfo(INFO_ALL, info_string)`
- **用途**: 获取设备的详细信息。
- **使用方法**:
  - 将设备信息填充到 `info_string` 中。

---

## 4. `getControlReferences(std::vector<int16_t> &control_references)`
- **用途**: 获取设备的控制参考值。
- **使用方法**:
  - 将当前控制参考值填充到 `control_references` 中。

---

## 5. `getCurrents(std::vector<int16_t> &currents)`
- **用途**: 获取设备的电流值。
- **使用方法**:
  - 将电流值填充到 `currents` 中。

---

## 6. `getCurrentsAndPositions(std::vector<int16_t> &currents, std::vector<int16_t> &positions)`
- **用途**: 同时获取设备的电流值和位置值。
- **使用方法**:
  - 分别将电流值和位置值填充到 `currents` 和 `positions` 中。

---

## 7. `getPositions(std::vector<int16_t> &positions)`
- **用途**: 获取设备的位置值。
- **使用方法**:
  - 将当前的位置值填充到 `positions` 中。

---

## 8. `getVelocities(std::vector<int16_t> &velocities)`
- **用途**: 获取设备的速度值。
- **使用方法**:
  - 将速度值填充到 `velocities` 中。

---

## 9. `getAccelerations(std::vector<int16_t> &accelerations)`
- **用途**: 获取设备的加速度值。
- **使用方法**:
  - 将加速度值填充到 `accelerations` 中。

---

## 10. `setMotorStates(bool activate)`
- **用途**: 激活或停用设备的电机。
- **使用方法**:
  - 传入 `true` 激活电机，传入 `false` 停用电机。
  - 成功返回 `0`，失败返回非零值。

---

## 11. `setControlReferences(const std::vector<int16_t> &control_references)`
- **用途**: 设置设备的控制参考值。
- **使用方法**:
  - 传入控制参考值的向量。

---

## 12. `getMotorStates(bool &activate)`
- **用途**: 获取电机的激活状态。
- **使用方法**:
  - 如果电机处于激活状态，将 `activate` 设置为 `true`，否则为 `false`。

---

## 13. `getParamId(uint8_t &device_id)`
- **用途**: 获取设备 ID 参数。
- **使用方法**:
  - 将当前设备 ID 设置到 `device_id` 中。

---

## 14. `getParamPositionPID(std::vector<float> &PID)`
- **用途**: 获取位置 PID 参数。
- **使用方法**:
  - 将位置 PID 值填充到 `PID` 中。

---

## 15. `getParamCurrentPID(std::vector<float> &PID)`
- **用途**: 获取电流 PID 参数。
- **使用方法**:
  - 将电流 PID 值填充到 `PID` 中。

---

## 16. `getParamStartupActivation(uint8_t &activation)`
- **用途**: 获取启动激活参数。
- **使用方法**:
  - 将启动激活值设置到 `activation` 中。

---

## 17. `getParamInputMode(uint8_t &input_mode)`
- **用途**: 获取输入模式参数。
- **使用方法**:
  - 将当前输入模式设置到 `input_mode` 中。

---

## 18. `getParamControlMode(uint8_t &control_mode)`
- **用途**: 获取控制模式参数。
- **使用方法**:
  - 将当前控制模式设置到 `control_mode` 中。

---

## 19. `getParamEncoderResolutions(std::vector<uint8_t> &encoder_resolutions)`
- **用途**: 获取编码器分辨率。
- **使用方法**:
  - 将分辨率值填充到 `encoder_resolutions` 中。

---

## 20. `getParamEncoderOffsets(std::vector<int16_t> &encoder_offsets)`
- **用途**: 获取编码器偏移值。
- **使用方法**:
  - 将偏移值填充到 `encoder_offsets` 中。

---

## 21. `getParamEncoderMultipliers(std::vector<float> &encoder_multipliers)`
- **用途**: 获取编码器倍增器。
- **使用方法**:
  - 将倍增器值填充到 `encoder_multipliers` 中。

---

## 22. `getParamUsePositionLimits(uint8_t &use_position_limits)`
- **用途**: 获取位置限制使用参数。
- **使用方法**:
  - 设置 `use_position_limits` 表示是否使用位置限制。

---

## 23. `getParamPositionLimits(std::vector<int32_t> &position_limits)`
- **用途**: 获取位置限制值。
- **使用方法**:
  - 将限制值填充到 `position_limits` 中。

---

## 24. `getParamPositionMaxSteps(std::vector<int32_t> &position_max_steps)`
- **用途**: 获取最大位置步数。
- **使用方法**:
  - 将最大步数值填充到 `position_max_steps` 中。

---

## 25. `getParamCurrentLimit(int16_t &current_limit)`
- **用途**: 获取电流限制参数。
- **使用方法**:
  - 将电流限制值设置到 `current_limit` 中。

---

## 26. `closeSerialPort(const std::string &serial_port)`
- **用途**: 关闭指定的串口。
- **使用方法**:
  - 调用 `communication_handler_->closeSerialPort()` 关闭串口。
  - 打印消息指示串口已关闭。

---

# ROS 2 控制节点用法（本包新增）

本包现在提供一个 ROS 2 节点 `qbsofthand_control_node`，用 service 控制 SoftHand：

## 1) 启动节点

- 直接扫描 `/dev/ttyUSB*` 并连接第一个找到的设备：
  - `ros2 run qbsofthand_control qbsofthand_control_node`

- 指定串口和设备 ID：
  - `ros2 run qbsofthand_control qbsofthand_control_node --ros-args -p serial_port:=/dev/ttyUSB0 -p device_id:=1`

常用参数：
- `open_reference`：张开参考值（默认 0）
- `close_reference`：闭合参考值（默认 19000；若设备开启 position limits，会尝试用上限自动覆盖）
- `command_rate_hz`：下发控制参考的频率（默认 50Hz）
- `auto_activate`：启动时是否自动使能电机（默认 true）
- `max_step_per_tick`：速度模式下每个周期最大步进（默认 300；实际步进=该值*speed_ratio）

## 2) 控制闭合程度/持续时间/速度比例

service：`/qbsofthand_control_node/set_closure`

- 方式 A（按持续时间控制）：`duration_sec > 0`，节点会在指定时间内插值到目标闭合度（此时速度由 `duration_sec` 决定，`speed_ratio` 不参与）。
  - 示例：2 秒内闭合到 70%：
    - `ros2 service call /qbsofthand_control_node/set_closure qbsofthand_control/srv/SetClosure "{closure: 0.7, duration_sec: 2.0, speed_ratio: 1.0}"`

- 方式 B（按速度比例控制）：`duration_sec == 0`，节点使用步进方式逐步逼近目标；速度由 `speed_ratio` 控制（结合参数 `max_step_per_tick`）。
  - 示例：用速度比例 0.3 慢速闭合到 70%：
    - `ros2 service call /qbsofthand_control_node/set_closure qbsofthand_control/srv/SetClosure "{closure: 0.7, duration_sec: 0.0, speed_ratio: 0.3}"`

说明：两种方式是分开的；如果你希望“到位后保持多久再自动回到张开”，我可以在现有接口上再加一个 `hold_sec` 字段。

## 3) 电机使能/失能

service：`/qbsofthand_control_node/activate`

- 使能：`ros2 service call /qbsofthand_control_node/activate std_srvs/srv/SetBool "{data: true}"`
- 失能：`ros2 service call /qbsofthand_control_node/activate std_srvs/srv/SetBool "{data: false}"`
