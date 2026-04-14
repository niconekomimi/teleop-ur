[中文](PROJECT_ANALYSIS.md) | English

# PROJECT_ANALYSIS

Last updated: 2026-04-09

## 1. Purpose

This document describes the structure that is actually implemented in the current workspace. It is not a target-state architecture draft.

It mainly answers three questions:

1. Which processes, nodes, and core modules currently exist.
2. How teleop, recording, inference, and Home / Home Zone actually run today.
3. How far configuration layering has already landed, and what is still missing.

---

## 2. One-line Conclusion

`teleop_control_py` has already moved away from the old structure where the GUI directly owned a large amount of business logic, and has converged into:

- The GUI handling UI, user intents, and state display.
- `core` handling state transitions, control arbitration, recording, and inference lifecycle.
- ROS nodes handling teleop, Home / Home Zone, and data collection.
- `robot_profiles.yaml` becoming the main source of truth for low-level robot and gripper interface defaults.

However, it is still not the final form where all control is owned by one central runtime process. Control responsibilities are still split across:

- `teleop_control_node`
- `robot_commander_node`
- `data_collector_node`
- GUI-side `ROS2Worker`

So the most accurate description is: **the architectural skeleton is already in place, but the system is still in a transitional stage of multi-node coordination plus a GUI bridge.**

---

## 3. Current Architecture Layers

## 3.1 GUI / Task Layer

Main files:

- `teleop_control_py/gui/main_window.py`
- `teleop_control_py/gui/app_service.py`
- `teleop_control_py/gui/intent_controller.py`
- `teleop_control_py/gui/runtime_facade.py`
- `teleop_control_py/gui/ros_worker.py`
- `teleop_control_py/gui/http_preview_worker.py`
- `teleop_control_py/gui/preview_recording_worker.py`

Responsibility boundaries:

- `main_window.py`
  - Owns widgets, layout, user actions, and state rendering.
  - No longer assembles raw ROS service / topic details directly.
  - Also owns preview-source switching, preview-window state, and preview-recording UI orchestration.
- `GuiAppService`
  - Manages child-process startup and shutdown.
  - Manages `ROS2Worker` lifecycle.
  - Consolidates actions such as starting drivers, teleop, collection, Home, and inference execution into unified calls.
- `GuiIntentController`
  - Reuses coordinator rules at the GUI layer and blocks conflicting actions early.
- `GuiRuntimeFacade`
  - Aggregates process state, hardware detection, and camera availability into snapshots the GUI can consume directly.
- `ROS2Worker`
  - Acts as the GUI's ROS bridge.
  - Subscribes to robot state and calls `data_collector` / `commander` services.
  - During inference execution, it owns a local `ControlCoordinator + ServoArmBackend + GripperBackend` chain to send commands.
- `HttpPreviewWorker`
  - Polls the collector Preview API.
  - Converts collector-side `HTTP/JPEG` preview frames back into GUI updates.
- `PreviewRecordingWorker`
  - Reuses the currently active preview source.
  - Writes local `mp4` files independently of HDF5 dataset collection.

Current assessment:

- The main GUI window has become much more UI-oriented.
- But `ROS2Worker` still owns part of the runtime control responsibility, especially for inference execution.
- The preview path is still mainly orchestrated at the GUI layer as well: collector API polling, inference direct rendering, and preview recording are all closed there.

## 3.2 Core Layer

Main files:

- `teleop_control_py/core/models.py`
- `teleop_control_py/core/mux.py`
- `teleop_control_py/core/orchestrator.py`
- `teleop_control_py/core/control_coordinator.py`
- `teleop_control_py/core/sync_hub.py`
- `teleop_control_py/core/recorder.py`
- `teleop_control_py/core/inference_service.py`

Key roles:

- `models.py`
  - Defines shared data structures such as `ActionCommand`, `CameraFrameSet`, `RobotStateSnapshot`, and `ObservationSnapshot`.
- `ActionMux`
  - Arbitrates between action sources.
  - Current priority order:
    - `SAFETY = 100`
    - `COMMANDER = 80`
    - `INFERENCE = 20`
    - `TELEOP = 10`
    - `NONE = 0`
- `SystemOrchestrator`
  - Owns the high-level state machine.
  - Current phases:
    - `IDLE`
    - `TELEOP`
    - `HOMING`
    - `HOME_ZONE`
    - `INFERENCE_READY`
    - `INFERENCE_EXECUTING`
    - `ESTOP`
- `ControlCoordinator`
  - Wraps `SystemOrchestrator + ActionMux` behind a unified interface.
  - Both `teleop_control_node` and `ROS2Worker` reuse it.
- `SyncHub`
  - Builds synchronized snapshots of camera frames, robot state, and actions in the collector node.
- `RecorderService`
  - Wraps the HDF5 writer thread and provides `start_demo / stop_demo / discard_last_demo / enqueue_sample`.
- `InferenceService`
  - Only manages `InferenceWorker` lifecycle and is not bound to a specific UI.

Current assessment:

- The `core` layer is no longer an empty shell; it is already part of the live runtime.
- This is currently the part of the project that is closest to a reusable platform layer.

## 3.3 Node / Backend Layer

Main files:

- `teleop_control_py/nodes/teleop_control_node.py`
- `teleop_control_py/nodes/robot_commander_node.py`
- `teleop_control_py/nodes/data_collector_node.py`
- `teleop_control_py/nodes/joy_driver_node.py`
- `teleop_control_py/nodes/quest3_webxr_bridge_node.py`
- `teleop_control_py/data/preview_api.py`
- `teleop_control_py/device_manager/*`
- `teleop_control_py/hardware/*`

Responsibilities:

- `teleop_control_node`
  - Reads `joy`, `mediapipe`, or `quest3` input.
  - Applies velocity and acceleration limits.
  - Sends teleop actions through `ControlCoordinator`.
- `robot_commander_node`
  - Provides `/commander/go_home`
  - Provides `/commander/go_home_zone`
  - Provides `/commander/cancel_home_zone`
  - Owns controller switching, Home trajectory publication, and Home Zone servo behavior.
- `data_collector_node`
  - Pulls frames through SDK camera clients.
  - Reads cached robot ROS state.
  - Records HDF5 through `SyncHub + RecorderService`.
- `joy_driver_node`
  - Owns physical joystick device access and publishes `/joy`.
- `quest3_webxr_bridge_node`
  - Owns Quest Browser WebXR controller ingestion and publishes `/quest3/*`.
- `PreviewApiServer`
  - Exposes `/preview/global.jpg` and `/preview/wrist.jpg` from collector-side cached frames
  - Provides a lightweight monitoring interface for the GUI preview window
- `device_manager`
  - Provides `ServoArmBackend / ControllerGripperBackend / SharedMemoryCameraBackend / InputHandlerBackend`
  - Provides robot profile loading and default-value assembly.

Current assessment:

- The current system is not a single monolithic node.
- It is a structure of multiple specialized ROS nodes plus a shared core layer.
- Collector preview is also not shared-memory direct-to-GUI; it is a collector-side HTTP helper plus GUI polling.

## 3.4 External Drivers / Dependencies

Main external dependencies:

- `ur_robot_driver`
- `ur_moveit_config` / MoveIt Servo
- `robotiq_2f_gripper_hardware`
- `qbsofthand_control`
- `pyrealsense2`
- `depthai`
- `vuer`
- `Real_IL` and its inference dependencies

The main control-system launch entry is still:

- `teleop_control_py/launch/control_system.launch.py`

This launch currently:

- Includes the UR driver
- Optionally includes MoveIt / Servo
- Includes either Robotiq or qbSoftHand drivers based on `gripper_type`
- Creates `robot_commander_node`
- Creates `teleop_control_node` when needed
- Creates `data_collector_node` when needed
- Includes `joy_driver.launch.py` when `input_type == joy`
- Automatically includes `quest3_webxr_bridge_node` when `input_type == quest3` and the bridge switch is enabled

---

## 4. Key Runtime Flows

## 4.1 Robot Driver Bringup Only

The GUI action "start robot driver" does not launch only the raw UR driver. It goes through:

```text
GuiAppService.start_robot_driver()
-> build_robot_driver_command()
-> ros2 launch teleop_control_py control_system.launch.py launch_teleop_node:=false
```

Actual effect:

- Starts the UR driver
- Starts MoveIt / Servo
- Starts the gripper driver
- Starts `robot_commander_node`
- Does not start `teleop_control_node`

So the more accurate description is "low-level control base plus commander bringup", not "bare driver bringup".

## 4.2 Teleop Flow

```text
Joy / MediaPipe
-> InputHandlerBackend
-> teleop_control_node
-> ControlCoordinator
-> ActionMux
-> ServoArmBackend / ControllerGripperBackend
-> MoveIt Servo / gripper driver
```

Current characteristics:

- `teleop_control_node` loads `robot_profile`, and the profile provides default robot and gripper interface values.
- `input_type` currently supports:
  - `joy`
  - `mediapipe`
  - `quest3`
- `mediapipe` defaults to direct SDK camera access, with depth disabled by default and only enabled explicitly.
- `quest3` has a formal `Quest3InputHandler`, using `relative pose + clutch + hand_relative orientation` by default.
- `quest3` supports Quest2ROS-style relative frame reset, scoped to `active_hand` by default.
- The teleop loop always applies velocity and acceleration limits.
- When `home_zone_active == true`, teleop output is paused.
- **In the current implementation, new manual input does not cancel Home Zone.**

## 4.3 Home / Home Zone Flow

```text
GUI / ROS2Worker
-> /commander/go_home or /commander/go_home_zone
-> robot_commander_node
-> controller switch
-> trajectory or servo motion
```

### Go Home

Actual behavior:

1. If GUI-side inference is currently executing, `ROS2Worker` stops inference execution first.
2. `robot_commander_node` verifies that `homing` and `home_zone` are not already active.
3. Switches to the trajectory controller.
4. Publishes a `JointTrajectory` to `home_joint_trajectory_topic`.
5. Waits for `home_duration_sec + 0.5s`.
6. Switches back to the teleop controller.
7. Publishes `/commander/homing_active=false` and `/commander/last_motion_result`.

This is the current real implementation of "switch from velocity control to position control, then switch back".

### Go Home Zone

Actual behavior:

1. If GUI-side inference is currently executing, `ROS2Worker` stops inference execution first.
2. Verifies that `homing` and `home_zone` are not already active.
3. Marks `home_zone_active=true`.
4. Computes the Home end-effector pose using FK from the configured Home joint state.
5. Samples a Home Zone offset around that Home pose.
6. Solves IK for the offset target pose.
7. Hands control to the trajectory controller.
8. Publishes a single-point `JointTrajectory` to move directly from the current pose to the target joint state.
9. Restores the teleop controller and publishes `home_zone_active=false` when finished.

Cancellation:

- Currently only supported via the explicit `/commander/cancel_home_zone` service
- Node shutdown also triggers cancellation
- **There is no longer a second Cartesian servo stage**

## 4.4 Recording Flow

```text
GuiAppService.start_data_collector()
-> ros2 run teleop_control_py data_collector_node
-> SDK camera pull + ROS state cache
-> SyncHub.capture_snapshot()
-> RecorderService
-> HDF5WriterThread
```

Current characteristics:

- `data_collector_node` does not use ROS image topics as the primary capture path; it pulls directly from SDK camera clients.
- Supports both `global_camera_source` and `wrist_camera_source`.
- Supports per-camera source / serial / depth switches when creating camera instances.
- During capture it checks:
  - joint freshness
  - pose freshness
  - action freshness
  - gripper freshness
  - timestamp drift between the two cameras
- Images are center-cropped and always written as `224 x 224`.
- `RecorderService` owns demo index incrementing and last-demo discard behavior.
- the collector can also expose cached preview frames, but that is a monitoring helper path rather than the primary sampling path

Service interfaces:

- `/data_collector/start`
- `/data_collector/stop`
- `/data_collector/discard_last_demo`

## 4.5 Inference Flow

```text
GUI
-> InferenceService
-> InferenceWorker
-> Qt signal
-> ROS2Worker
-> ControlCoordinator
-> ServoArmBackend / ControllerGripperBackend
```

Current characteristics:

- `InferenceService` only manages worker lifecycle.
- `InferenceWorker` does not publish ROS actions directly; it sends actions back to the GUI by signal.
- `ROS2Worker` creates a local inference backend chain and `ControlCoordinator`.
- `ROS2Worker` only sends model actions when the coordinator's `active_source == INFERENCE`.
- Gripper commands are discretized into binary open/close commands inside `ROS2Worker`.

This means inference execution is still a GUI-driven control path, not an independent ROS inference node.

## 4.6 Preview / Preview Recording Flow

```text
preview window
-> collector Preview API or direct inference preview
-> optional PreviewRecordingWorker
-> local mp4
```

Current characteristics:

- The source priority is: collector Preview API > direct inference preview > no active image source.
- The collector-side preview path is: `data_collector_node` timed frame pull -> `PreviewApiServer` exposes `/preview/*.jpg` -> GUI `HttpPreviewWorker` polls and renders.
- Collector preview targets `30 Hz` by default and uses `HTTP/JPEG`, which is better suited for monitoring than high-quality recording.
- Inference preview has already been decoupled from the policy loop, using a separate `30 Hz` preview thread pushed on demand.
- Inference preview is enabled only when the preview window is open, inference is running, and the collector is not owning the preview path.
- Preview recording sits on top of whichever preview source is currently active and writes to `data/preview_recordings/<timestamp>/*.mp4`.
- Preview recording is not HDF5 collection and does not drive `data_collector_node` start/stop.

Current assessment:

- Preview cadence is now separate from policy-action cadence instead of being tightly coupled to it.
- This path is still a GUI-orchestrated monitoring path, not part of the low-level robot control loop.

## 4.7 Quest 3 / WebXR Flow

```text
Quest Browser
-> vuer.ai?ws=wss://<pc_wifi_ip>:8012
-> quest3_webxr_bridge_node
-> /quest3/*
-> Quest3InputHandler
-> teleop_control_node
-> ControlCoordinator
```

Current characteristics:

- Quest input is no longer a standalone prototype; it is now a formal `input_type`.
- `control_system.launch.py` can automatically bring up `quest3_webxr_bridge_node` when `input_type:=quest3` and the bridge switch allows it.
- A split debug workflow is also supported through `quest3_webxr_bridge.launch.py` plus `teleop_control.launch.py input_type:=quest3`.
- The bridge uses `Vuer` to provide a secure `https/wss` WebXR entry, and the recommended Quest URL is `https://vuer.ai?ws=wss://<pc_wifi_ip>:8012`.
- The first visit usually requires one manual certificate acceptance in Quest Browser.
- `Quest3InputHandler` defaults to `relative pose + clutch + hand_relative orientation`.
- Quest2ROS-style frame reset is already supported.

Current assessment:

- The Quest input path is now genuinely integrated into the existing teleop architecture.
- But the Quest-side page is still mostly focused on controller-stream ingestion and does not yet provide a full status UI / haptics / floating-panel experience.

---

## 5. Current State Machine and Arbitration Rules

## 5.1 SystemPhase

The shared public state machine is defined by `SystemOrchestrator`:

- `IDLE`
- `TELEOP`
- `HOMING`
- `HOME_ZONE`
- `INFERENCE_READY`
- `INFERENCE_EXECUTING`
- `ESTOP`

Notes:

- `recording_active` is an orthogonal flag, not its own phase.
- `home_zone_active` and `homing_active` have higher priority than teleop / inference.

## 5.2 Control Source Priority

Current `ActionMux` priority:

| Source | Priority | Purpose |
| --- | ---: | --- |
| `SAFETY` | 100 | zero-velocity / estop hold |
| `COMMANDER` | 80 | Home Zone / commander motion |
| `INFERENCE` | 20 | inference execution |
| `TELEOP` | 10 | manual teleop |
| `NONE` | 0 | idle |

Rules:

- When a higher-priority source is active, lower-priority actions are rejected.
- When `hold` is enabled, everything except `SAFETY` is rejected.

## 5.3 GUI / Coordinator Request Constraints

Key constraints in the current `SystemOrchestrator`:

- Starting teleop is rejected when:
  - `estopped`
  - `inference_executing`
  - `homing_active`
  - `home_zone_active`
- Enabling inference execution is rejected when:
  - `estopped`
  - `teleop_running`
  - `homing_active`
  - `home_zone_active`
  - `inference_not_ready`
- `go_home` / `go_home_zone` are rejected when:
  - `estopped`
  - `homing_active`
  - `home_zone_active`
- `start_recording` is only rejected when:
  - `estopped`

These rules currently affect:

- Whether GUI buttons are allowed to issue requests
- Whether actions inside `teleop_control_node` can pass through the mux
- Whether GUI-side inference execution in `ROS2Worker` can actually send commands

---

## 6. Current Configuration Layering

## 6.1 `robot_profiles.yaml`

Role: low-level interface source of truth.

Currently responsible for:

- robot profile names
- joint names
- ROS topics
- ROS services
- controller names
- default Home
- default Home Zone
- default gripper type and gripper interface parameters

Currently consumed by:

- `teleop_control_node`
- `robot_commander_node`
- `data_collector_node`
- `ROS2Worker` inference backend assembly

## 6.2 `teleop_params.yaml`

Role: teleop behavior configuration.

Currently responsible for:

- `input_type`
- `gripper_type`
- teleop rate
- velocity / acceleration limits
- joy mapping
- Quest 3 mapping, clutch, orientation, and frame-reset parameters
- MediaPipe algorithm parameters

Additional note:

- Default `quest3_webxr_bridge_node` settings such as `advertised_host / vuer_port / topics` also currently live in the same YAML.

It no longer owns the source of truth for robot and gripper interfaces.

## 6.3 `data_collector_params.yaml`

Role: collection behavior configuration.

Currently responsible for:

- output path
- recording rate
- camera source, serial number, and depth switches
- freshness thresholds
- `preview_fps`
- `preview_api_host / port / jpeg_quality`
- preview / writer behavior
- collector-side gripper selection strategy

## 6.4 `gui_params.yaml`

Role: GUI preferences.

Currently responsible for:

- default IP
- default input method
- default camera model / serial preferences
- default recording directory and file name
- GUI-side depth preference

Note:

- It still stores semi-hardware selection preferences such as camera model and serial.
- These are user preferences, not low-level control truth.

## 6.5 `home_overrides.yaml`

Role: runtime Home override storage.

Currently responsible for:

- persisting the 6-joint Home pose for each `robot_profile` when the GUI action "Set Current Pose as Home" is used.

Runtime flow:

```text
GUI current joint state
-> home_overrides.yaml
-> ROS2Worker set_parameters(/commander)
-> commander uses the new Home at runtime
```

## 6.6 `joy_driver_params.yaml`

Role: joystick driver-layer configuration.

Currently responsible for:

- device discovery
- publish rate
- deadzone
- auto reconnect

## 6.7 Removed Configuration

`robot_commander_params.yaml` is no longer part of the launch chain.

The key runtime parameters of `robot_commander_node` now come from:

- `robot_profile`
- a small number of explicit launch overrides such as `commander_pose_max_age_sec`

---

## 7. Structural Convergence Already Completed

1. `robot_profile` has become the main source of default robot and gripper interface values.
2. `ControlCoordinator + SystemOrchestrator + ActionMux` are already active in the live runtime.
3. The recording path already reuses `SyncHub + RecorderService` instead of building samples in the GUI.
4. Inference worker lifecycle has been moved into `core/inference_service.py`.
5. Home persistence has moved from `gui_params.yaml` to a dedicated `home_overrides.yaml`.
6. `robot_commander_params.yaml` has been removed from the startup chain.
7. Quest 3 has converged from a bridge prototype into a formal input backend wired into the main launch / teleop path.
8. Preview cadence has been decoupled from inference-action cadence, and preview recording is now its own local-output path.

---

## 8. Current Boundaries and Gaps

## 8.1 Still Not a Single Central Runtime

Control is still distributed across multiple places:

- `teleop_control_node` for manual teleop
- `robot_commander_node` for Home / Home Zone
- `ROS2Worker` for inference execution
- `data_collector_node` for sampling

This means:

- there is already a unified control rule set
- but there is still no single controller node that owns all control actions

## 8.2 `robot_profiles.yaml` Does Not Yet Own Launch-Level Driver Definitions

`robot_profiles.yaml` already owns topics, services, controllers, Home, and gripper defaults,
but it does not yet own:

- UR driver bringup selection
- MoveIt bringup selection
- gripper driver launch selection

So currently the project can say:

- "If ROS interfaces change, try to update only `robot_profiles.yaml`"

But it cannot yet fully say:

- "If hardware and driver stacks change, only `robot_profiles.yaml` needs to change"

## 8.3 Camera Driver Management Still Has Runtime-Layer Residue

The current main path is already biased toward:

- SDK-direct cameras for recording
- SDK-direct cameras for MediaPipe as well

But `GuiRuntimeFacade / ResourceManager / ProcessManager` still keep:

- camera driver subprocess management
- camera driver occupancy checks

This indicates some legacy camera-management behavior still remains.

## 8.4 Current "Estop" Is Not a Unified Robot-Level Safety Stop

Current explicit emergency stop behavior mainly appears in:

- `ROS2Worker.emergency_stop_inference()`
- `ControlCoordinator.notify_estop(True)`
- `publish_zero(source=SAFETY)`

At the moment it mainly covers the GUI-side inference execution chain.

This is not yet a hardware-level, system-wide, unified emergency stop implementation.

## 8.5 Quest-Side Experience Is Still Lightweight

The current Quest page mainly solves:

- controller input
- ROS2 teleop integration

It does not yet fully integrate:

- floating camera preview panels
- controller haptics
- richer Quest-side status UI

---

## 9. Most Accurate Architecture Judgment Today

If we distinguish between the target architecture and the currently running code, the project is best described as:

- It already has clear boundaries between UI, core, and device layers.
- It already has a unified state machine, action format, arbitration rules, and profile source of truth.
- It still relies on multiple ROS nodes rather than a single central controller.
- It is already in a good state for continued cleanup and responsibility convergence, not for another large rewrite.

One-line summary:

**The project is no longer the old GUI-heavy mixed system, but it is also not yet the final unified control platform. It is now in a stable intermediate state that can keep evolving cleanly.**
