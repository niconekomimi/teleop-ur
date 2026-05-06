# teleop_control

[中文](README.md) | English

[![Imitation-learning robot architecture demo](https://i1.hdslb.com/bfs/archive/1d983e799dbbca97152156ce0755d5a8ef2fb6ce.jpg)](https://www.bilibili.com/video/BV13iQuB3E5L/)

> Demo video: click the cover image to open Bilibili.

This is a ROS 2 workspace for real-robot teleoperation, dataset collection, and online inference. The project is organized around a GUI-driven workflow for bringing up the robot driver, teleop stack, HDF5 collector, preview pipeline, and model execution.

## Features

- GUI-managed robot driver, teleop, collector, and inference lifecycle
- `joy`, `mediapipe`, and `quest3` input backends
- `robotiq` and `qbsofthand` gripper configurations
- Synchronized HDF5 demo recording
- `Home`, `Home Zone`, controller switching, and recording services
- Model loading, inference preview, and inference action execution

## Workspace Layout

| Path | Role |
| --- | --- |
| `src/teleop_control_py/` | Main ROS 2 package for the GUI, control, collection, inference bridge, and configuration |
| `src/Universal_Robots_ROS2_Driver/` | UR ROS 2 driver, controllers, and MoveIt configuration |
| `src/robotiq_2f_gripper_ros2/` | Robotiq 2F gripper ROS 2 packages |
| `src/qbsofthand_control/` | qbSoftHand control package |
| `Real_IL/` | Local imitation-learning inference repository |
| `openpi/` | Remote openpi inference repository |
| `scripts/` | Workspace-level utility scripts |
| `data/` | Local datasets, preview recordings, and inference logs |
| `models/` | Local models and checkpoints |
| `udev/` | Device rules |

## Architecture Overview

The workspace is organized into entry, orchestration, core runtime, capability, and device/asset layers:

![teleop_control framework](docs/assets/teleop-control-framework.png)

Main responsibilities:

- The GUI owns workflow orchestration, status display, and user-facing controls.
- The `core` package provides the state machine, action arbitration, synchronized capture, recording, and inference lifecycle services.
- `teleop_control_node` consumes manual input and sends teleop actions.
- `robot_commander_node` provides `Home`, `Home Zone`, and controller switching.
- `data_collector_node` owns camera capture, robot-state synchronization, and HDF5 writing.
- `ROS2Worker` bridges the GUI to ROS and also carries the inference execution path.

For the full architecture and runtime boundaries, see [docs/guide/01-architecture.md](docs/guide/01-architecture.md).

## Quick Start

### Prerequisites

Recommended environment:

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- Installed UR driver, MoveIt Servo, and the matching gripper driver

`requirements.txt` only covers Python packages for this workspace. Install ROS 2 system packages through ROS 2 / apt.

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you want to use `Real_IL` inference, clone it into the workspace root and keep the directory name as `Real_IL`:

```bash
git clone https://github.com/niconekomimi/Real_IL.git Real_IL
pip install -r Real_IL/requirements.txt
```

### Build

```bash
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

When changing only the main package, building just `teleop_control_py` is also useful:

```bash
colcon build --packages-select teleop_control_py
```

### Launch The GUI

Recommended entry:

```bash
ros2 run teleop_control_py teleop_gui
```

![GUI Preview](docs/assets/gui-preview.png)

## GUI Workflow

Typical order:

1. Select `ur_type`, robot IP, input backend, and gripper type
2. If `quest3` is selected, make sure the Quest bridge is running, then open the Quest webpage and click `Enter VR` / `Enter XR`
3. Start the robot driver
4. Start the teleop system
5. Start the collector node
6. Start / stop recording, or discard the last demo
7. Use `Go Home`, `Go Home Zone`, or `Set Current Pose as Home`
8. Start inference and then explicitly enable inference execution if needed

## CLI Entrypoints

Launch the full control system:

```bash
ros2 launch teleop_control_py control_system.launch.py
```

Common parameter combinations:

```bash
ros2 launch teleop_control_py control_system.launch.py input_type:=joy gripper_type:=robotiq
ros2 launch teleop_control_py control_system.launch.py input_type:=joy gripper_type:=qbsofthand
ros2 launch teleop_control_py control_system.launch.py input_type:=mediapipe gripper_type:=robotiq
ros2 launch teleop_control_py control_system.launch.py input_type:=quest3 gripper_type:=robotiq
```

Start the collector with the full system:

```bash
ros2 launch teleop_control_py control_system.launch.py \
    input_type:=joy \
    gripper_type:=robotiq \
    enable_data_collector:=true
```

For split-launch workflows, parameter tuning, and Quest3-specific notes, see:

- [docs/guide/03-operation.md](docs/guide/03-operation.md)
- [docs/guide/07-devices.md](docs/guide/07-devices.md)

Launch only the collector node:

```bash
ros2 run teleop_control_py data_collector_node \
    --ros-args \
    --params-file src/teleop_control_py/config/data_collector_params.yaml
```

## Useful Services

```bash
ros2 service call /data_collector/start std_srvs/srv/Trigger {}
ros2 service call /data_collector/stop std_srvs/srv/Trigger {}
ros2 service call /data_collector/discard_last_demo std_srvs/srv/Trigger {}
ros2 service call /commander/go_home std_srvs/srv/Trigger {}
ros2 service call /commander/go_home_zone std_srvs/srv/Trigger {}
```

## Reproducibility Notes

This is a real-hardware robotics workspace. The code, launch files, configuration layout, and documentation are readable and reusable, while a full run requires matching hardware and local setup.

Parts that require real hardware or a lab environment:

- UR robot, UR driver networking, and MoveIt Servo control
- Robotiq or qbSoftHand gripper
- RealSense / OAK-D cameras
- Quest 3 WebXR input
- Local model checkpoints, demo datasets, robot IPs, and device serial numbers

Parts that can be reviewed or reused without hardware:

- GUI, launch files, ROS nodes, and module boundaries
- `SystemOrchestrator`, `ControlCoordinator`, and `ActionMux` control-rule design
- HDF5 collection pipeline and dataset structure notes
- `Real_IL` / `openpi` inference backend integration
- Architecture, operation, and configuration docs under `docs/guide/`

Third-party dependency notes:

- `Real_IL/` and `openpi/` are external model repositories and should be fetched locally; they are not committed into this repository.
- `data/`, `models/`, `build/`, `install/`, and `log/` are local runtime artifacts, not source release content.
- README commands target a real-hardware environment. Before running them, adjust robot IPs, camera serial numbers, gripper devices, and model paths for your setup.

## Documentation

Detailed maintenance docs are under `docs/`. They are currently maintained in Chinese:

- [docs/guide/00-overview.md](docs/guide/00-overview.md): workspace overview and reading order
- [docs/guide/01-architecture.md](docs/guide/01-architecture.md): architecture layers and runtime boundaries
- [docs/guide/03-operation.md](docs/guide/03-operation.md): GUI and CLI startup workflows
- [docs/guide/08-configuration.md](docs/guide/08-configuration.md): configuration ownership and override rules
- [docs/agent/00-current-state.md](docs/agent/00-current-state.md): current maintenance state
- [docs/agent/01-todo.md](docs/agent/01-todo.md): shared human/AI task queue

## Key Paths

- `src/teleop_control_py/launch/control_system.launch.py`
- `src/teleop_control_py/launch/teleop_control.launch.py`
- `src/teleop_control_py/config/`
- `src/teleop_control_py/teleop_control_py/core/`
- `src/teleop_control_py/teleop_control_py/gui/`
- `src/teleop_control_py/teleop_control_py/nodes/`
- `src/Universal_Robots_ROS2_Driver/`
- `src/robotiq_2f_gripper_ros2/`
- `src/qbsofthand_control/`
- `scripts/`
- `Real_IL/`
- `openpi/`
