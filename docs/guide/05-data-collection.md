# 数据采集

更新时间：2026-05-06

数据采集由 `data_collector_node` 负责。它缓存机器人状态，主动从双相机拉帧，通过 `SyncHub` 拼装同步样本，并由 `RecorderService` 写入 HDF5。

## 启动

通过 GUI 启动时，`GuiAppService.start_data_collector()` 会执行：

```text
ros2 run teleop_control_py data_collector_node --ros-args --params-file ...
```

并覆盖以下参数：

- `robot_profile`
- `output_path`
- `global_camera_source`
- `wrist_camera_source`
- `global_camera_serial_number`
- `wrist_camera_serial_number`
- `global_camera_enable_depth`
- `wrist_camera_enable_depth`
- `end_effector_type`

命令行启动：

```bash
ros2 run teleop_control_py data_collector_node \
    --ros-args \
    --params-file src/teleop_control_py/config/data_collector_params.yaml
```

## ROS 接口

服务：

- `/data_collector/start`
- `/data_collector/stop`
- `/data_collector/discard_last_demo`

状态：

- `/data_collector/record_stats`

订阅：

- joint states
- TCP pose
- servo twist command
- gripper state

具体 topic 默认来自 `robot_profiles.yaml`，采集行为参数来自 `data_collector_params.yaml`。

## 采样流程

启动录制：

1. 检查当前不在录制中
2. 检查全局相机和腕部相机都可用
3. `RecorderService.start_demo()`
4. 按 `record_fps` 创建采样定时器

每次采样：

1. 主动拉取全局相机和腕部相机帧
2. 读取最近的 joint、pose、action、gripper 缓存
3. 检查各类时间新鲜度和双相机时间偏差
4. 中心裁剪并缩放 RGB 到 `224 x 224`
5. 组装 `Sample`
6. 入队给 `HDF5WriterThread`

停止录制：

1. 取消采样定时器
2. `RecorderService.stop_demo()`
3. 更新 demo 统计

弃用最近 demo：

1. 必须当前不在录制中
2. 调用 `RecorderService.discard_last_demo()`
3. 删除最近 demo 并回退 demo 序号

## HDF5 schema

HDF5 文件按 `data/demo_N` 组织。每个 demo 包含：

| 路径 | shape | dtype |
| --- | --- | --- |
| `actions` | `(N, 7)` | `float32` |
| `obs/agentview_rgb` | `(N, 224, 224, 3)` | `uint8` |
| `obs/eye_in_hand_rgb` | `(N, 224, 224, 3)` | `uint8` |
| `obs/robot0_joint_pos` | `(N, 6)` | `float32` |
| `obs/robot0_gripper_qpos` | `(N, 1)` | `float32` |
| `obs/robot0_eef_pos` | `(N, 3)` | `float32` |
| `obs/robot0_eef_quat` | `(N, 4)` | `float32` |

`actions` 当前使用 7 维命令动作：

```text
[vx, vy, vz, wx, wy, wz, gripper]
```

## Preview API

采集节点可以启动轻量 HTTP/JPEG 预览服务：

- `GET /preview/global.jpg`
- `GET /preview/wrist.jpg`

相关参数：

- `preview_api_enabled`
- `preview_api_host`
- `preview_api_port`
- `preview_api_jpeg_quality`
- `preview_fps`

GUI 预览窗口优先使用采集节点 Preview API；如果采集节点不可用，但推理 worker 可提供预览，则显示推理侧预览。

## 预览源优先级

GUI 预览窗口的图像源优先级：

1. 采集节点 Preview API
2. 推理 worker 直连预览
3. 无活动图像源

也就是说，采集节点运行时，GUI 优先显示采集侧画面；采集节点未运行但推理 worker 运行时，GUI 显示推理侧预览。

## 预览录屏

GUI 的预览录屏由 `PreviewRecordingWorker` 完成，它和 HDF5 数据集采集不是同一条链路。

输出目录：

- `data/preview_recordings/<timestamp>/global.mp4`
- `data/preview_recordings/<timestamp>/wrist.mp4`

预览录屏只保存当前预览画面，不会触发 `/data_collector/start` 或 `/data_collector/stop`。

## 相机链路边界

项目里有多条相机链路，维护时要分清：

- 数据集采集：`data_collector_node` 主动从 SDK 相机拉帧，并写入 HDF5
- 采集侧预览：`data_collector_node` 独立 preview timer 拉帧，并通过 HTTP/JPEG 暴露
- MediaPipe 输入：`teleop_control_node` 可以使用 SDK 相机或 ROS 图像 topic
- 推理输入：推理 worker 直接读取相机帧
- GUI 预览：优先读采集侧 Preview API，其次读推理 preview signal

录制主链路不以 ROS image topic 为主要采样路径。

## 深度图

数据集 schema 当前只写 RGB、机器人状态和动作。相机参数里保留 depth 开关，用于启用对齐 RGBD 管道或供输入链路使用；HDF5 writer 当前不写 depth dataset。

如果未来要写 depth，需要同步修改：

- `data_collector_node`
- `data/hdf5_writer.py`
- 数据集重建脚本
- 本文 HDF5 schema

## 常见检查点

- 录制开始失败：检查双相机是否初始化成功
- 样本持续丢弃：检查 joint、pose、action、gripper 和 image 的时间戳阈值
- 图像为空：检查相机来源、serial number 和 depth 开关
- HDF5 写入慢：检查 `image_compression`、队列大小和磁盘性能
