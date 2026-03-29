[中文](current_control_behavior_spec_v0.1.md) | English

# Current Control Behavior Spec v0.1

Last updated: 2026-03-29

This document describes only the actual current runtime behavior. It does not describe the future target architecture.

---

## 1. Scope

This document covers:

- teleoperation
- Home / Home Zone
- recording
- inference execution
- GUI-side intent blocking
- current control ownership and arbitration rules

This document does not cover:

- full UI visual layout
- offline HDF5 editor details
- future multi-arm backend design

---

## 2. Current Runtime Roles

## 2.1 `teleop_control_node`

Responsibilities:

- Reads `joy`, `mediapipe`, or `quest3` input
- Normalizes input into `ActionCommand`
- Applies velocity and acceleration limits
- Sends teleop actions through `ControlCoordinator`
- Pauses its own output during `Home Zone`

Inputs:

- `joy`: `/joy`
- `mediapipe`: SDK camera or image topics
- `quest3`: `/quest3/right_controller/pose`, `/quest3/left_controller/pose`, `/quest3/input/joy`

Outputs:

- Sends robot velocity commands to `servo_twist_topic` through `ServoArmBackend`
- Sends gripper commands through `ControllerGripperBackend`

## 2.2 `robot_commander_node`

Responsibilities:

- Provides `go_home`
- Provides `go_home_zone`
- Provides `cancel_home_zone`
- Owns controller switching
- Publishes Home trajectories
- Drives servo-side Home Zone behavior

Services:

- `/commander/go_home`
- `/commander/go_home_zone`
- `/commander/cancel_home_zone`

Published state:

- `/commander/home_zone_active`
- `/commander/homing_active`
- `/commander/last_motion_result`

## 2.3 `data_collector_node`

Responsibilities:

- Initializes dual-camera SDK clients
- Caches robot state
- Builds synchronized snapshots through `SyncHub`
- Writes data through `RecorderService`
- Exposes start / stop / discard interfaces for recording

Services:

- `/data_collector/start`
- `/data_collector/stop`
- `/data_collector/discard_last_demo`

Published state:

- `/data_collector/record_stats`

## 2.4 `ROS2Worker`

Responsibilities:

- Acts as the GUI's ROS bridge
- Subscribes to joint / pose / action / gripper / commander state
- Calls commander and collector services
- Creates a local control backend and sends actions during inference execution

Note:

- Inference execution is currently not a dedicated ROS-node loop; it is a GUI-side `ROS2Worker` loop.

## 2.5 `InferenceWorker`

Responsibilities:

- Loads the model
- Reads the required images and robot state
- Outputs 7D actions

It does not publish ROS control commands directly.

## 2.6 `ProcessManager`

Responsibilities:

- Starts and stops long-running ROS / Python subprocesses

The GUI uses it to manage:

- robot driver processes
- teleop processes
- collection processes

---

## 3. Shared State and Arbitration Rules

## 3.1 State-Machine Phases

The current shared public phase comes from `SystemOrchestrator`:

- `IDLE`
- `TELEOP`
- `HOMING`
- `HOME_ZONE`
- `INFERENCE_READY`
- `INFERENCE_EXECUTING`
- `ESTOP`

Orthogonal flags:

- `recording_active`
- `teleop_running`
- `inference_ready`
- `inference_executing`
- `home_zone_active`
- `homing_active`
- `estopped`

Phase priority:

1. `ESTOP`
2. `HOME_ZONE`
3. `HOMING`
4. `INFERENCE_EXECUTING`
5. `TELEOP`
6. `INFERENCE_READY`
7. `IDLE`

## 3.2 Action Source Priority

Current `ActionMux` priority:

| Source | Priority | Meaning |
| --- | ---: | --- |
| `SAFETY` | 100 | zero velocity / estop |
| `COMMANDER` | 80 | Home Zone / commander |
| `INFERENCE` | 20 | model execution |
| `TELEOP` | 10 | manual input |
| `NONE` | 0 | idle |

Behavior rules:

- When a higher-priority source is active, lower-priority actions are rejected.
- When `hold` is active, only `SAFETY` is allowed through.

## 3.3 GUI-Side Intent Blocking Rules

The GUI uses `GuiIntentController` to reuse `ControlCoordinator` rules.

### Starting teleop is rejected when

- `estopped`
- `inference_executing`
- `homing_active`
- `home_zone_active`

### Starting inference execution is rejected when

- `estopped`
- `teleop_running`
- `homing_active`
- `home_zone_active`
- `inference_not_ready`

### Triggering Home / Home Zone is rejected when

- `estopped`
- `homing_active`
- `home_zone_active`

Additional note:

- `go_home / go_home_zone` are not directly rejected by the coordinator just because `inference_executing` is active.
- On the GUI path, if inference execution is active, `ROS2Worker` will stop inference execution first and then call the commander service.

### Starting recording is rejected when

- `estopped`

Conclusion:

- `teleop` and `inference execution` are currently mutually exclusive.
- `Home / Home Zone` has higher priority than teleop; for inference execution the behavior is "stop inference first, then call commander".
- `recording` is not an independent control source; in principle it can coexist with teleop / inference and is mainly constrained by the collector path itself.

---

## 4. Scenario Specs

## 4.1 Starting the Robot Driver

GUI call:

```text
GuiAppService.start_robot_driver()
-> build_robot_driver_command()
-> ros2 launch teleop_control_py control_system.launch.py launch_teleop_node:=false
```

What actually starts:

- UR driver
- MoveIt / Servo
- gripper driver
- `robot_commander_node`

What does not start:

- `teleop_control_node`

Result:

- the low-level control base is available
- the GUI can read joint / pose / commander state through `ROS2Worker`

## 4.2 Starting the Teleop System

GUI call:

```text
GuiAppService.start_teleop()
-> build_teleop_command()
-> ros2 launch teleop_control_py control_system.launch.py
```

Actual behavior:

- starts the UR driver
- starts MoveIt / Servo
- starts the gripper driver
- starts `robot_commander_node`
- starts `teleop_control_node`
- includes `joy_driver.launch.py` if `input_type == joy`
- includes `quest3_webxr_bridge_node` if `input_type == quest3` and `launch_quest3_bridge` resolves to enabled

Parameter sources:

- behavior parameters come from `teleop_params.yaml`
- robot and gripper interface defaults come from `robot_profile`

### `input_type = joy`

Flow:

1. `joy_driver_node` publishes `/joy`
2. `JoyInputHandler` reads `/joy`
3. It outputs translation, rotation, and gripper commands

### `input_type = mediapipe`

Flow:

1. `MediaPipeInputHandler` reads images from an SDK camera or image topics
2. It performs hand keypoint tracking
3. It generates target poses or velocity commands based on configuration
4. It can optionally use depth to recover 3D hand position

Notes:

- SDK depth is disabled by default
- depth is only enabled when hand / mediapipe is started and depth is explicitly requested

### `input_type = quest3`

Flow:

1. `quest3_webxr_bridge_node` publishes `/quest3/*` controller topics
2. `Quest3InputHandler` reads controller poses and `/quest3/input/joy`
3. It uses `relative pose + clutch + hand_relative orientation` by default
4. It supports Quest2ROS-style relative frame reset by default

Current default semantics:

- `active_hand = right`
- right-hand `squeeze / grip` acts as clutch
- right-hand `trigger` acts as gripper
- input-side low-pass smoothing is disabled by default
- `frame reset` is scoped to `active_hand` by default

## 4.3 Teleop Continuous Loop

Current `teleop_control_node._control_loop()` behavior:

1. Pulls an `ActionCommand` from `InputHandlerBackend`
2. Computes `dt` for this cycle
3. Extracts the 6D twist target
4. If `home_zone_active == true`
   - clears the previous twist
   - returns immediately for this cycle
5. Applies separate acceleration limits for non-zero axes and return-to-zero axes
6. Calls `apply_velocity_limits()`
7. Sends the command through `ControlCoordinator.dispatch()`

Important behavior:

- teleop does not output actions during `Home Zone`
- the current code does not cancel `Home Zone` when new manual input appears

## 4.4 Clicking Go Home During Teleop

Flow:

1. The GUI checks permission through `GuiIntentController`.
2. If inference execution is running, `ROS2Worker` stops it first.
3. `ROS2Worker` calls `/commander/go_home`.
4. `robot_commander_node` rejects the request if `homing` or `home_zone` is already active.
5. `commander` marks `homing_active=true`.
6. Switches controllers:
   - activates trajectory controller
   - deactivates teleop controller
7. Publishes `JointTrajectory` to `home_joint_trajectory_topic`.
8. Waits for `home_duration_sec + 0.5s`.
9. Switches controllers back:
   - activates teleop controller
   - deactivates trajectory controller
10. Publishes the result to `/commander/last_motion_result`.

Conclusion:

- the system does indeed switch from velocity control to position control and then back.

## 4.5 Clicking Go Home Zone During Teleop

Flow:

1. The GUI checks permission through `GuiIntentController`.
2. If inference execution is running, `ROS2Worker` stops it first.
3. `ROS2Worker` calls `/commander/go_home_zone`.
4. `robot_commander_node` marks:
   - `home_zone_active=true`
   - `homing_active=true`
5. Reads the configured Home joint state and computes the Home TCP pose by FK.
6. Samples translation and rotation offsets around the Home pose.
7. Solves IK for the offset target pose.
8. Switches to the trajectory controller.
9. Publishes a single-point `JointTrajectory` to move directly from the current pose to the target joint state.
10. On success or cancellation:
   - restores the teleop controller
   - sets `home_zone_active=false`
   - publishes the motion result

Cancellation paths:

- `/commander/cancel_home_zone`
- automatic cancellation on node shutdown

Current note:

- there is no longer a second Cartesian servo stage.

## 4.6 Recording Flow

Startup:

```text
GuiAppService.start_data_collector()
-> ros2 run teleop_control_py data_collector_node --ros-args --params-file ...
```

At startup the GUI overrides:

- `robot_profile`
- `output_path`
- `global_camera_source`
- `wrist_camera_source`
- `global_camera_serial_number`
- `wrist_camera_serial_number`
- `global_camera_enable_depth`
- `wrist_camera_enable_depth`
- `end_effector_type`

### After `data_collector_node` is ready

It will:

- initialize dual-camera clients
- subscribe to joint / pose / twist / gripper
- optionally start the preview API

### Calling `/data_collector/start`

Behavior:

1. Verifies that recording is not already active.
2. Verifies that dual-camera clients are available.
3. Calls `RecorderService.start_demo()`.
4. Creates a timer at `record_fps`.

### Each sample cycle

1. `SyncHub.capture_snapshot()`
2. Validates freshness of joint / pose / action / gripper
3. Validates timestamp skew between the two cameras
4. Crops and rescales both RGB streams to `224 x 224`
5. Builds a `Sample`
6. Calls `RecorderService.enqueue_sample()`

### Calling `/data_collector/stop`

Behavior:

1. Stops the sampling timer
2. Calls `RecorderService.stop_demo()`
3. Finalizes the current demo

### Calling `/data_collector/discard_last_demo`

Behavior:

1. Requires that recording is not active
2. Calls `RecorderService.discard_last_demo()`
3. Discards the latest demo and rolls back the demo index

## 4.7 Inference Startup and Execution

### Starting inference

GUI call:

```text
InferenceService.start_inference()
-> creates InferenceWorker
```

`InferenceLaunchConfig` currently includes:

- checkpoint directory
- task name
- embedding path
- global / wrist camera source
- loop_hz
- device
- state_provider
- global / wrist serial
- global / wrist depth flags

Resulting state:

- `InferenceWorker` starts running
- `ControlCoordinator` can be marked as `INFERENCE_READY`

### Enabling inference execution

GUI call:

```text
GuiAppService.enable_inference_execution()
-> ROS2Worker.set_inference_execution_enabled(True)
```

Actual behavior:

1. `ROS2Worker` ensures the local inference backend exists.
2. Clears inference estop.
3. Calls `ControlCoordinator.notify_inference_ready(True)`.
4. Calls `ControlCoordinator.notify_inference_execution(True)`.
5. Starts attempting to send model actions at a fixed rate.

### Each inference execution cycle

1. Reads the latest model action.
2. Converts it into an `ActionCommand`.
3. Binarizes the gripper command.
4. If `active_source != INFERENCE`, skips immediately.
5. If `active_source == INFERENCE`, dispatches through the coordinator.

### Stopping inference execution

Behavior:

1. `notify_inference_execution(False)`
2. publishes zero velocity

### Inference estop

Behavior:

1. `self._inference_estopped = True`
2. `self._inference_control_enabled = False`
3. `notify_estop(True)`
4. publishes zero velocity

Important note:

- the current "estop" mainly affects the GUI-side inference execution path
- it is not a hardware-level estop, and not an independent system-wide safety node

---

## 5. Current Home Behavior

Current Home source order:

1. override value for the active `robot_profile` in `home_overrides.yaml`
2. default Home value for that profile in `robot_profiles.yaml`

Behavior after the GUI action "Set Current Pose as Home":

1. Read the current 6-joint state from `ROS2Worker`
2. Write it into `config/home_overrides.yaml`
3. Sync it to `robot_commander_node` through `/commander/set_parameters`

Result:

- the currently running commander immediately uses the new Home
- the same override remains active after GUI / node restart

---

## 6. Current Camera Behavior

## 6.1 Main Recording Path

The main recording path is currently:

- SDK-direct camera access

not:

- ROS image topics as the primary sampling path

## 6.2 MediaPipe Camera Path

MediaPipe can currently:

- use SDK cameras
- use image topics

But the main configuration is biased toward:

- `mediapipe_use_sdk_camera = true`

## 6.3 Depth Images

Current default behavior:

- SDK depth is disabled by default
- when depth is enabled, the recording path explicitly enables it
- paths that need depth use aligned RGBD logic

## 6.4 Residual Path

`GuiRuntimeFacade` still keeps camera-driver subprocess management interfaces, but that is not the main runtime path for current recording or MediaPipe.

---

## 7. Current System Boundaries

1. The system already has a unified state machine and action arbitration, but it is still not a single central controller node.
2. `teleop`, `commander`, `collector`, and GUI-side inference execution still live in different runtimes.
3. `robot_profile` is already the source of truth for interface defaults, but launch-level driver selection is not yet fully profile-driven.
4. `recording` is a collection state, not a control ownership source.
5. On the GUI path, `Home / Home Zone` can preempt inference execution, but the implementation is still "stop inference first, then call commander".
6. `Home Zone` cannot be canceled by new manual input; it exits only through the explicit cancel service or normal completion.

---

## 8. Behavior Summary

The current system can be understood in these terms:

- Manual teleop and inference execution are mutually exclusive.
- Home / Home Zone has higher priority than teleop; for inference execution the behavior is to stop inference first and then run the commander path.
- Home is executed through the trajectory controller.
- Home Zone returns Home first, then goes back to Servo control for local pose variation.
- Recording actively samples through SDK cameras plus ROS state caches.
- Inference execution is dispatched through GUI-side `ROS2Worker`, not an independent ROS inference node.
- Home can be re-taught at runtime and persisted into `home_overrides.yaml`.
