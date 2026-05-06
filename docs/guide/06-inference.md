# 推理

更新时间：2026-05-06

推理功能由 GUI 发起，`InferenceService` 管理 worker 生命周期。模型动作只有在用户显式启用“推理执行”后才会通过 `ROS2Worker` 下发到机器人。

## 推理后端

GUI 目前支持两个后端：

| 后端 | Worker | 说明 |
| --- | --- | --- |
| `real_il` | `InferenceWorker` | 本地加载 `Real_IL` 模型、任务 embedding 和双相机图像 |
| `openpi_remote` | `OpenPiRemoteWorker` | 通过 websocket 连接远端 openpi policy server，本地仍负责相机、机器人状态、预览和动作下发 |

默认后端来自 `config/gui_params.yaml` 的 `default_inference_backend`。

## 本地 `Real_IL`

本地推理需要：

- `Real_IL/` 仓库在工作区根目录
- checkpoint 目录
- task name
- task embedding 路径
- 全局相机和腕部相机来源
- 推理频率和设备，例如 `cuda`

启动后，worker 会读取相机帧和机器人状态，输出 7 维动作。

## 远端 `openpi`

`openpi_remote` 后端需要：

- openpi websocket policy server 可访问
- host 和 port，默认 `127.0.0.1:18000`
- prompt
- 本地双相机和机器人状态可用

worker 会构造如下观测：

- `observation/image`
- `observation/wrist_image`
- `observation/joint_position`
- `observation/gripper_position`
- `prompt`

远端返回 `actions` 后，本地取前 7 维作为动作块，并按当前推理频率逐步发送。

调试远端服务可以使用：

```bash
python3 scripts/openpi_remote_probe.py --host 127.0.0.1 --port 18000 --infer --env ur5
```

## 推理执行链路

```text
InferenceService.start_inference()
-> InferenceWorker / OpenPiRemoteWorker
-> action_signal
-> MainWindow
-> GuiAppService.update_inference_action_command()
-> ROS2Worker
-> ControlCoordinator
-> ActionMux
-> ServoArmBackend / GripperBackend
```

推理 worker 启动后只表示“模型正在产出动作”。要让动作进入机器人控制链，需要在 GUI 中再启用推理执行。

## 启动与执行状态

推理分成两个阶段：

1. 启动推理 worker
2. 启用推理执行

启动推理 worker 后，系统可以显示预览并持续产生动作，但动作不会自动进入机器人控制链。只有推理执行开关打开后，`ROS2Worker` 才会把最近的模型动作交给 `ControlCoordinator`。

停止推理执行只关闭动作下发；停止推理 worker 才会释放模型、相机和远端连接资源。

## 推理侧预览

推理 worker 会通过 `preview_signal` 把全局相机和腕部相机画面发回 GUI。GUI 预览窗口在采集节点不可用时会使用推理侧预览。

推理侧预览不写 HDF5，也不等价于预览录屏。

## 动作日志

GUI 支持记录推理动作日志，相关实现：

- `gui/inference_action_logger.py`
- `GuiAppService.start_inference_action_logging()`
- `GuiAppService.record_inference_action_sample()`

日志默认保存到：

```text
data/inference_action_logs/
```

## 维护注意事项

- 新增推理后端时，优先扩展 `InferenceLaunchConfig` 和 `InferenceService.start_inference()`。
- 新后端需要发出与现有 GUI 契约一致的 `action_signal`、`preview_signal`、`status_signal`、`log_signal` 和 `error_signal`。
- 动作下发仍走 `ROS2Worker`，worker 内保持“产出动作和预览”的职责即可。
- 修改动作维度或语义时，同步更新 [05-data-collection.md](05-data-collection.md) 中的 HDF5 schema。
