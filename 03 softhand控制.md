# QB Soft Hand 控制指南

## 背景

QB Soft Hand（灵巧手）的官方驱动基于 ROS1，不兼容 ROS2 和 Ubuntu 22.04 及以上版本。本文档介绍三种解决方案及推荐方案（方案 3）的使用方法。

---

## 三种解决方案概览

| 方案 | 方式 | 难度 | 说明 |
|------|------|------|------|
| 方案 1 | Docker + ROS1/ROS2 桥接 | 中 | 隔离环境，适合保留旧代码 |
| 方案 2 | 自己移植 ROS1 到 ROS2 | 很高 | 工程量极大，不推荐 |
| **方案 3** | **使用 API 直接控制** | **低** | **推荐**，通过 ROS2 服务自己实现控制 |

---

## 推荐方案：直接 API 控制（方案 3）

本方案通过串口直接访问灵巧手硬件 API，用 ROS2 服务包装功能。

### 前置条件

将当前用户添加到 `dialout` 用户组，以便访问串口设备。

**执行命令：**
```bash
sudo usermod -aG dialout $USER
```

> ⚠️ **重要**：执行后需要重新登录或重启工作站，权限才会生效。

---

## 灵巧手控制

### 1. 初始化与设备发现

灵巧手通过 USB 串口连接。在启动前，需要找到对应的设备。

#### 方式 A：自动扫描（推荐）

自动扫描 `/dev/ttyUSB*` 并连接第一个找到的设备：

```bash
ros2 run qbsofthand_control qbsofthand_control_node
```

#### 方式 B：指定设备（多灵巧手场景）

如果系统中有多个灵巧手，需要指定串口和设备 ID：

```bash
ros2 run qbsofthand_control qbsofthand_control_node \
  --ros-args \
  -p serial_port:=/dev/ttyUSB0 \
  -p device_id:=1
```

**参数说明：**
- `serial_port`: 灵巧手所在的串口（可用 `ls /dev/ttyUSB*` 查看）
- `device_id`: 灵巧手的设备 ID（通常为 1，多个灵巧手时各不相同）

---

### 2. 控制灵巧手闭合

灵巧手通过 ROS2 服务 `/qbsofthand_control_node/set_closure` 进行控制。

**服务类型：**  
`qbsofthand_control/srv/SetClosure`

**可用参数：**
- `closure`: 闭合度（范围 0.0 ~ 1.0，其中 0.0 = 完全打开，1.0 = 完全闭合）
- `duration_sec`: 持续时间（秒）
- `speed_ratio`: 速度比例（范围 0.0 ~ 1.0）

#### 模式 A：按持续时间控制（**推荐用于精确位置控制**）

当 `duration_sec > 0` 时，灵巧手会在指定时间内插值到目标闭合度。此时 `speed_ratio` 参数被忽略，速度由 `duration_sec` 自动决定。

**示例：2 秒内闭合到 0.7**
```bash
ros2 service call /qbsofthand_control_node/set_closure \
  qbsofthand_control/srv/SetClosure \
  "{closure: 0.7, duration_sec: 2.0, speed_ratio: 1.0}"
```

**用途：**
- 精确控制闭合位置
- 避免突然加速或减速
- 抓取物体时实现平稳过程

#### 模式 B：按速度比例控制（**推荐用于连续动作**）

当 `duration_sec == 0` 时，灵巧手使用步进方式逐步逼近目标，速度由 `speed_ratio` 控制（结合参数 `max_step_per_tick`）。

**示例：以 30% 速度闭合到 0.7**
```bash
ros2 service call /qbsofthand_control_node/set_closure \
  qbsofthand_control/srv/SetClosure \
  "{closure: 0.7, duration_sec: 0.0, speed_ratio: 0.3}"
```

**用途：**
- 快速响应（无需等待预设时间）
- 实时调整速度
- 适合频繁发送控制指令

---

## 快速参考

### 常用命令

| 操作 | 命令 |
|------|------|
| 添加用户到 dialout 组 | `sudo usermod -aG dialout $USER` |
| 启动灵巧手驱动 | `ros2 run qbsofthand_control qbsofthand_control_node` |
| 指定设备启动 | `ros2 run qbsofthand_control qbsofthand_control_node --ros-args -p serial_port:=/dev/ttyUSB0 -p device_id:=1` |
| 查看串口设备 | `ls /dev/ttyUSB*` |
| 快速闭合（2 秒到 70%） | `ros2 service call /qbsofthand_control_node/set_closure qbsofthand_control/srv/SetClosure "{closure: 0.7, duration_sec: 2.0, speed_ratio: 1.0}"` |
| 缓慢闭合（30% 速度） | `ros2 service call /qbsofthand_control_node/set_closure qbsofthand_control/srv/SetClosure "{closure: 0.7, duration_sec: 0.0, speed_ratio: 0.3}"` |

### 控制参数速查表

| 参数 | 范围 | 说明 | 使用场景 |
|------|------|------|---------|
| `closure` | 0.0 ~ 1.0 | 0 = 打开，1 = 闭合 | 所有场景 |
| `duration_sec` | > 0 | 持续时间 | 精确位置、平稳运动 |
| `duration_sec` | == 0 | 忽略（即时响应） | 速度控制、快速响应 |
| `speed_ratio` | 0.0 ~ 1.0 | 仅在 `duration_sec == 0` 时生效 | 速度控制模式 |

