[中文](QUEST3_WEBXR_VUER.md) | English

# Quest 3 WebXR / Teleop Integration Notes

This document describes the Quest 3 support that is already implemented in this repository, including:

- how `quest3_webxr_bridge_node` converts Quest Browser WebXR controller streams into ROS2 topics
- how `Quest3InputHandler` integrates controller pose and buttons into the existing teleop architecture
- the recommended launch flow, Quest entry URL, debug workflow, and expected operator behavior

## 1. Implemented Capabilities

Quest 3 is no longer just a bridge prototype in this repository; it is now a formal input backend:

- bridge node:
  - `teleop_control_py/nodes/quest3_webxr_bridge_node.py`
- teleop input backend:
  - `teleop_control_py/hardware/input/quest3_input_handler.py`
- launch support:
  - `control_system.launch.py` supports `input_type:=quest3`
  - `teleop_control.launch.py` supports `input_type:=quest3`
  - `quest3_webxr_bridge.launch.py` supports standalone bridge bringup
- GUI:
  - the input source dropdown now includes `quest3 (VR controller)`

Current Quest-related ROS topics:

- `/quest3/left_controller/pose`
- `/quest3/right_controller/pose`
- `/quest3/left_controller/matrix`
- `/quest3/right_controller/matrix`
- `/quest3/input/joy`
- `/quest3/connected`

## 2. Recommended Control Method

Quest teleop currently defaults to:

- `target_pose`
- `hand_relative`
- `clutch / deadman`
- input-side low-pass smoothing disabled by default

That means:

1. Hold `clutch`
2. Record the current controller pose and current robot TCP pose
3. Drive the TCP target pose using controller pose deltas relative to that start point
4. Stop output when `clutch` is released

Default interaction semantics:

- `active_hand = left`
- left-hand `grip / squeeze`: clutch
- left-hand `trigger`: gripper
- default `orientation_mode = hand_relative`

This is closer to mature VR-controller teleop systems such as Quest2ROS than to a vision-hand-filtering pipeline.

## 3. Quest2ROS-Style Relative Frame Reset

The current implementation already includes relative reference-frame reset.

Default parameters:

- `quest3_frame_reset_enabled: true`
- `quest3_frame_reset_scope: "active_hand"`
- `quest3_frame_reset_hold_sec: 0.75`
- `quest3_frame_reset_rotate_position: false`

Default button combos:

- left hand: `X + Y`, i.e. `buttons [4, 5]`
- right hand: `A + B`, i.e. `buttons [10, 11]`

Behavior semantics:

- hold the combo for about `0.75s`
- the current controller pose for that hand becomes the new relative reference frame
- by default, only the translation origin and orientation zero-point are reset; the translation axes are not additionally rotated by that orientation reset
- later `clutch / pose delta / hand_relative orientation` behavior is interpreted inside that new frame

Notes:

- `quest3_frame_reset_rotate_position: false`
  is the better fit for this project. It avoids the parasitic coupling where resetting while the wrist is yawed makes later pure translation drift diagonally.
- `quest3_frame_reset_rotate_position: true`
  is closer to Quest2ROS-style full-frame remapping, where both translation and orientation are moved into the new frame.

This is not the same as `clutch`:

- `frame reset` solves whether the control-space semantics feel natural
- `clutch` solves when dragging the robot should begin

## 4. Dependencies and Environment

`vuer` is not a ROS2 system package and must be installed separately.

If you are using the project's common `clds` environment:

```bash
source /home/rvl/clds/bin/activate
pip install "vuer>=0.1.4,<0.2"
```

Then build:

```bash
source /opt/ros/humble/setup.bash
python -m colcon build --packages-select teleop_control_py
source install/setup.bash
```

## 5. Network and Entry URL

The two most important Quest 3 requirements are:

- Quest and PC must be on the same LAN
- Quest must use the PC's Wi-Fi IPv4 address, not the UR5 wired NIC address

Recommended project entry URL:

```text
https://vuer.ai?ws=wss://<pc_wifi_ip>:8012
```

For example:

```text
https://vuer.ai?ws=wss://192.168.137.227:8012
```

Notes:

- the direct page `https://<pc_ip>:8012` may open in some cases
- but current code and real tests both favor `vuer.ai?ws=wss://...`
- the first visit usually requires manual certificate acceptance in Quest Browser

## 6. Recommended Launch Modes

### 6.1 Unified Launch

If you want `control_system.launch.py` to bring up the Quest bridge automatically:

```bash
ros2 launch teleop_control_py control_system.launch.py \
  input_type:=quest3
```

Current behavior:

- `input_type:=quest3`
- `launch_quest3_bridge:=auto`

When those conditions hold, `control_system.launch.py` automatically starts `quest3_webxr_bridge_node`.

### 6.2 Recommended Split Debug Workflow

During development it is better to separate the bridge from teleop.

Terminal 1: keep the bridge alive

```bash
ros2 launch teleop_control_py quest3_webxr_bridge.launch.py
```

Terminal 2: restart teleop independently

```bash
ros2 launch teleop_control_py teleop_control.launch.py input_type:=quest3
```

This lets you:

- avoid reopening the Quest page repeatedly
- avoid re-establishing the WebXR controller connection every time
- restart only teleop while tuning parameters, mappings, and follow-feel

## 7. Quest-Side Operation Steps

1. Start the bridge on the PC
2. Put on the Quest and open `Quest Browser`
3. Visit:

```text
https://vuer.ai?ws=wss://<pc_wifi_ip>:8012
```

4. If a certificate warning appears, continue manually
5. After the page opens, click `Enter VR` / `Enter XR`
6. Enter VR mode using the controllers

If you start `quest3` mode from the GUI, the GUI now also explicitly reminds you to:

- open the corresponding webpage inside Quest
- click to enter VR mode

## 8. ROS2 Topic Verification

After connection succeeds, first verify:

```bash
ros2 topic echo /quest3/connected --once
```

Expected result:

```text
data: true
```

Then verify controller data:

```bash
ros2 topic echo /quest3/input/joy --once
ros2 topic echo /quest3/right_controller/pose --once
```

## 9. `/quest3/input/joy` Mapping

`axes` order:

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

`buttons` order:

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

## 10. Main Parameter Entry Points

Quest teleop parameters mainly live in:

- `src/teleop_control_py/config/teleop_params.yaml`

That file now includes:

- default bridge parameters
- Quest teleop mapping parameters
- `frame reset` parameters
- smoothing / clutch / trigger / active-hand parameters

Important Quest parameters:

- `quest3_motion_mode`
- `quest3_orientation_mode`
- `quest3_enable_input_smoothing`
- `quest3_frame_reset_enabled`
- `quest3_frame_reset_scope`
- `quest3_frame_reset_rotate_position`
- `quest3_left_frame_reset_buttons`
- `quest3_right_frame_reset_buttons`

## 11. Current Limitations

The current Quest page mainly solves:

- controller input
- ROS2 teleop integration

It does not yet fully integrate:

- floating camera preview panels
- controller haptics
- richer Quest-side status UI

All of those can still be built on top of the current bridge + WebXR structure.

## 12. References

- Quest2ROS project page:
  - https://quest2ros.github.io/
- Quest2ROS paper PDF:
  - https://quest2ros.github.io/files/Quest2ROS.pdf
- OpenXR reference space:
  - https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html
