#!/usr/bin/env python3
"""Quest 3 WebXR input handler."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from .input_handlers import InputHandlerBase, _LowPassFilter, _zero_twist
from ...utils.home_zone_utils import compute_pose_error
from ...utils.transform_utils import (
    _clamp,
    quat_conjugate_xyzw,
    quat_multiply_xyzw,
    quat_normalize_xyzw,
    quat_to_rotmat_xyzw,
    rotvec_to_quat_xyzw,
)


@dataclass
class _QuestControllerState:
    pose_position: Optional[np.ndarray] = None
    pose_quat: Optional[np.ndarray] = None
    pose_stamp_monotonic: float = 0.0
    trigger_value: float = 0.0
    squeeze_value: float = 0.0
    thumbstick_xy: tuple[float, float] = (0.0, 0.0)
    buttons: tuple[int, ...] = ()

    def pose_ready(self, stale_timeout_sec: float) -> bool:
        return (
            self.pose_position is not None
            and self.pose_quat is not None
            and (time.monotonic() - self.pose_stamp_monotonic) <= stale_timeout_sec
        )


class Quest3InputHandler(InputHandlerBase):
    """Quest 3 controller strategy with single-arm output and dual-arm-ready state."""

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self.node = node

        self._connected_topic = str(node.get_parameter("quest3_connected_topic").value).strip() or "/quest3/connected"
        self._left_pose_topic = (
            str(node.get_parameter("quest3_left_pose_topic").value).strip() or "/quest3/left_controller/pose"
        )
        self._right_pose_topic = (
            str(node.get_parameter("quest3_right_pose_topic").value).strip() or "/quest3/right_controller/pose"
        )
        self._joy_topic = str(node.get_parameter("quest3_joy_topic").value).strip() or "/quest3/input/joy"
        self._tool_pose_topic = str(node.get_parameter("tool_pose_topic").value).strip()

        self._active_hand = str(node.get_parameter("quest3_active_hand").value).strip().lower()
        if self._active_hand not in {"left", "right"}:
            self._active_hand = "right"

        self._require_connected = bool(node.get_parameter("quest3_require_connected").value)
        self._pose_timeout_sec = max(0.05, float(node.get_parameter("quest3_pose_timeout_sec").value))
        self._motion_mode = str(node.get_parameter("quest3_motion_mode").value).strip().lower()
        if self._motion_mode not in {"velocity", "target_pose"}:
            self._motion_mode = "target_pose"

        self._deadzone = float(node.get_parameter("quest3_deadzone").value)
        self._linear_scale = float(node.get_parameter("quest3_linear_scale").value)
        self._angular_scale = float(node.get_parameter("quest3_angular_scale").value)
        self._position_linear_gain = float(node.get_parameter("quest3_position_linear_gain").value)
        self._position_angular_gain = float(node.get_parameter("quest3_position_angular_gain").value)
        self._linear_mapping = self._vector3_param("quest3_linear_axis_mapping", [0, 1, 2], cast=int)
        self._angular_mapping = self._vector3_param("quest3_angular_axis_mapping", [0, 1, 2], cast=int)
        self._linear_sign = self._vector3_param("quest3_linear_axis_sign", [1.0, 1.0, 1.0], cast=float)
        self._angular_sign = self._vector3_param("quest3_angular_axis_sign", [1.0, 1.0, 1.0], cast=float)

        self._orientation_mode = str(node.get_parameter("quest3_orientation_mode").value).strip().lower()
        if self._orientation_mode not in {"lock", "hand_relative"}:
            self._orientation_mode = "hand_relative"
        self._orientation_mapping = self._vector3_param("quest3_orientation_axis_mapping", [0, 1, 2], cast=int)
        self._orientation_sign = self._vector3_param("quest3_orientation_axis_sign", [1.0, 1.0, 1.0], cast=float)
        self._enable_input_smoothing = bool(node.get_parameter("quest3_enable_input_smoothing").value)
        smoothing_alpha = float(node.get_parameter("quest3_smoothing_alpha").value)
        self._frame_reset_enabled = bool(node.get_parameter("quest3_frame_reset_enabled").value)
        self._frame_reset_scope = str(node.get_parameter("quest3_frame_reset_scope").value).strip().lower()
        if self._frame_reset_scope not in {"active_hand", "both"}:
            self._frame_reset_scope = "active_hand"
        self._frame_reset_hold_sec = max(0.0, float(node.get_parameter("quest3_frame_reset_hold_sec").value))
        self._frame_reset_buttons = {
            "left": self._index_list_param("quest3_left_frame_reset_buttons", [4, 5]),
            "right": self._index_list_param("quest3_right_frame_reset_buttons", [10, 11]),
        }

        self._clutch_axis_threshold = max(0.0, float(node.get_parameter("quest3_clutch_axis_threshold").value))
        self._clutch_filter_enabled = bool(node.get_parameter("quest3_clutch_filter_enabled").value)
        self._clutch_engage_confirm_sec = float(node.get_parameter("quest3_clutch_engage_confirm_sec").value)
        self._clutch_release_confirm_sec = float(node.get_parameter("quest3_clutch_release_confirm_sec").value)
        self._clutch_axis_index = {
            "left": int(node.get_parameter("quest3_left_clutch_axis").value),
            "right": int(node.get_parameter("quest3_right_clutch_axis").value),
        }
        self._clutch_button_index = {
            "left": int(node.get_parameter("quest3_left_clutch_button").value),
            "right": int(node.get_parameter("quest3_right_clutch_button").value),
        }
        self._trigger_axis_index = {
            "left": int(node.get_parameter("quest3_left_trigger_axis").value),
            "right": int(node.get_parameter("quest3_right_trigger_axis").value),
        }
        self._trigger_button_index = {
            "left": int(node.get_parameter("quest3_left_trigger_button").value),
            "right": int(node.get_parameter("quest3_right_trigger_button").value),
        }
        self._gripper_requires_clutch = bool(node.get_parameter("quest3_gripper_requires_clutch").value)
        self._gripper_axis_inverted = bool(node.get_parameter("quest3_gripper_axis_inverted").value)

        self._controllers = {
            "left": _QuestControllerState(),
            "right": _QuestControllerState(),
        }
        self._connected = False
        self._latest_tool_pos: Optional[np.ndarray] = None
        self._latest_tool_quat: Optional[np.ndarray] = None

        self._linear_filter = _LowPassFilter(alpha=smoothing_alpha)
        self._angular_filter = _LowPassFilter(alpha=smoothing_alpha)
        self._clutch_filtered = False
        self._clutch_candidate: Optional[bool] = None
        self._clutch_candidate_since_ns = 0

        self._teleop_active = False
        self._initial_input_pos: Optional[np.ndarray] = None
        self._initial_input_quat: Optional[np.ndarray] = None
        self._initial_robot_pos: Optional[np.ndarray] = None
        self._initial_robot_quat: Optional[np.ndarray] = None
        self._frame_origin_pos: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._frame_origin_quat: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._frame_reset_hold_since: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._frame_reset_latched: dict[str, bool] = {"left": False, "right": False}

        node.create_subscription(Bool, self._connected_topic, self._connected_callback, 10)
        node.create_subscription(PoseStamped, self._left_pose_topic, self._left_pose_callback, qos_profile_sensor_data)
        node.create_subscription(PoseStamped, self._right_pose_topic, self._right_pose_callback, qos_profile_sensor_data)
        node.create_subscription(Joy, self._joy_topic, self._joy_callback, qos_profile_sensor_data)
        if self._tool_pose_topic:
            node.create_subscription(PoseStamped, self._tool_pose_topic, self._tool_pose_callback, 10)

        node.get_logger().info(
            "Quest3InputHandler ready. "
            f"active_hand={self._active_hand}, motion_mode={self._motion_mode}, "
            f"connected_topic={self._connected_topic}, joy_topic={self._joy_topic}, "
            f"input_smoothing={'on' if self._enable_input_smoothing else 'off'}, "
            f"frame_reset={'on' if self._frame_reset_enabled else 'off'}, "
            f"frame_reset_scope={self._frame_reset_scope}"
        )

    def _vector3_param(self, name: str, default: Sequence[float], *, cast):
        raw = list(self.node.get_parameter(name).value)
        if len(raw) != 3:
            return [cast(v) for v in default]
        try:
            return [cast(v) for v in raw]
        except Exception:
            return [cast(v) for v in default]

    def _index_list_param(self, name: str, default: Sequence[int]) -> tuple[int, ...]:
        raw = list(self.node.get_parameter(name).value)
        if not raw:
            return tuple(int(v) for v in default)
        try:
            return tuple(int(v) for v in raw)
        except Exception:
            return tuple(int(v) for v in default)

    def _safe_axis(self, values: Sequence[float], index: int) -> float:
        if index < 0 or index >= len(values):
            return 0.0
        value = float(values[index])
        if not math.isfinite(value):
            return 0.0
        return float(_clamp(value, -1.0, 1.0))

    def _safe_button(self, values: Sequence[int], index: int) -> bool:
        return 0 <= index < len(values) and bool(values[index])

    def _connected_callback(self, msg: Bool) -> None:
        self._connected = bool(msg.data)
        self._refresh_cached_command()

    def _left_pose_callback(self, msg: PoseStamped) -> None:
        self._pose_callback("left", msg)

    def _right_pose_callback(self, msg: PoseStamped) -> None:
        self._pose_callback("right", msg)

    def _pose_callback(self, side: str, msg: PoseStamped) -> None:
        state = self._controllers[side]
        state.pose_position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        state.pose_quat = quat_normalize_xyzw(
            np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=np.float64,
            )
        )
        state.pose_stamp_monotonic = time.monotonic()
        self._refresh_cached_command()

    def _joy_callback(self, msg: Joy) -> None:
        axes = list(msg.axes)
        buttons = list(msg.buttons)

        left = self._controllers["left"]
        right = self._controllers["right"]

        left.trigger_value = max(0.0, self._safe_axis(axes, self._trigger_axis_index["left"]))
        left.squeeze_value = max(0.0, self._safe_axis(axes, self._clutch_axis_index["left"]))
        left.thumbstick_xy = (
            self._safe_axis(axes, 4),
            self._safe_axis(axes, 5),
        )
        left.buttons = tuple(buttons)

        right.trigger_value = max(0.0, self._safe_axis(axes, self._trigger_axis_index["right"]))
        right.squeeze_value = max(0.0, self._safe_axis(axes, self._clutch_axis_index["right"]))
        right.thumbstick_xy = (
            self._safe_axis(axes, 10),
            self._safe_axis(axes, 11),
        )
        right.buttons = tuple(buttons)

        self._maybe_handle_frame_reset(time.monotonic())
        self._refresh_cached_command()

    def _tool_pose_callback(self, msg: PoseStamped) -> None:
        self._latest_tool_pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        self._latest_tool_quat = quat_normalize_xyzw(
            np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=np.float64,
            )
        )

    def _get_current_tool_pose(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self._latest_tool_pos is None or self._latest_tool_quat is None:
            return None
        return self._latest_tool_pos.copy(), self._latest_tool_quat.copy()

    def _reset_control_state(self) -> None:
        self._teleop_active = False
        self._initial_input_pos = None
        self._initial_input_quat = None
        self._initial_robot_pos = None
        self._initial_robot_quat = None
        self._linear_filter.reset(np.zeros(3, dtype=np.float64))
        self._angular_filter.reset(np.zeros(3, dtype=np.float64))

    def _apply_deadzone(self, value: float) -> float:
        raw = float(value)
        threshold = abs(self._deadzone)
        if threshold <= 0.0 or abs(raw) <= threshold:
            return 0.0 if threshold > 0.0 else raw
        return raw - math.copysign(threshold, raw)

    def _clutch_filter(self, desired: bool, now_ns: int) -> bool:
        if not self._clutch_filter_enabled:
            self._clutch_filtered = desired
            self._clutch_candidate = None
            return desired

        if self._clutch_candidate is None or desired != self._clutch_candidate:
            self._clutch_candidate = desired
            self._clutch_candidate_since_ns = now_ns
            return self._clutch_filtered

        if desired == self._clutch_filtered:
            return self._clutch_filtered

        confirm_sec = self._clutch_engage_confirm_sec if desired else self._clutch_release_confirm_sec
        confirm_ns = int(max(0.0, confirm_sec) * 1e9)
        if now_ns - self._clutch_candidate_since_ns >= confirm_ns:
            self._clutch_filtered = desired
            self._clutch_candidate = None
        return self._clutch_filtered

    def _select_axis(self, values: np.ndarray, mapping: list[int], signs: list[float], scale: float) -> np.ndarray:
        out = np.zeros(3, dtype=np.float64)
        for i in range(3):
            source_index = int(mapping[i])
            raw = float(values[source_index]) if 0 <= source_index < len(values) else 0.0
            out[i] = self._apply_deadzone(raw) * float(scale) * float(signs[i])
        return out

    def _build_twist(self, linear_values: np.ndarray, angular_values: np.ndarray) -> Twist:
        twist = _zero_twist()
        twist.linear.x, twist.linear.y, twist.linear.z = [float(v) for v in linear_values]
        twist.angular.x, twist.angular.y, twist.angular.z = [float(v) for v in angular_values]
        return twist

    def _apply_linear_smoothing(self, values: np.ndarray) -> np.ndarray:
        if not self._enable_input_smoothing:
            return np.asarray(values, dtype=np.float64)
        return self._linear_filter.apply(values)

    def _apply_angular_smoothing(self, values: np.ndarray) -> np.ndarray:
        if not self._enable_input_smoothing:
            return np.asarray(values, dtype=np.float64)
        return self._angular_filter.apply(values)

    def _combo_pressed(self, buttons: Sequence[int], combo: Sequence[int]) -> bool:
        valid_indices = [int(index) for index in combo if int(index) >= 0]
        if not valid_indices:
            return False
        return all(self._safe_button(buttons, index) for index in valid_indices)

    def _frame_reset_targets(self) -> tuple[str, ...]:
        if self._frame_reset_scope == "both":
            return ("left", "right")
        return (self._active_hand,)

    def _recenter_controller_frame(self, side: str) -> bool:
        state = self._controllers[side]
        if not state.pose_ready(self._pose_timeout_sec) or state.pose_position is None or state.pose_quat is None:
            self.node.get_logger().warn(f"Quest3 frame reset ignored for {side}: controller pose unavailable.")
            return False

        self._frame_origin_pos[side] = state.pose_position.copy()
        self._frame_origin_quat[side] = state.pose_quat.copy()
        if side == self._active_hand:
            if self._teleop_active:
                self.node.get_logger().info("Quest3 frame reset applied; teleop reference re-anchored.")
            self._reset_control_state()
        self.node.get_logger().info(
            "Quest3 frame reset complete: "
            f"hand={side}, active_hand={self._active_hand}, scope={self._frame_reset_scope}, "
            f"buttons={list(self._frame_reset_buttons[side])}."
        )
        return True

    def _maybe_handle_frame_reset(self, now_monotonic: float) -> None:
        if not self._frame_reset_enabled:
            return

        target_sides = self._frame_reset_targets()
        for side in target_sides:
            state = self._controllers[side]
            combo_pressed = self._combo_pressed(state.buttons, self._frame_reset_buttons[side])
            if not combo_pressed:
                self._frame_reset_hold_since[side] = 0.0
                self._frame_reset_latched[side] = False
                continue

            if self._frame_reset_latched[side]:
                continue

            hold_since = self._frame_reset_hold_since[side]
            if hold_since <= 0.0:
                self._frame_reset_hold_since[side] = now_monotonic
                continue

            if (now_monotonic - hold_since) < self._frame_reset_hold_sec:
                continue

            if self._recenter_controller_frame(side):
                self._frame_reset_latched[side] = True

    def _calibrated_controller_pose(self, side: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
        state = self._controllers[side]
        if not state.pose_ready(self._pose_timeout_sec) or state.pose_position is None or state.pose_quat is None:
            return None

        raw_pos = state.pose_position.copy()
        raw_quat = state.pose_quat.copy()
        origin_pos = self._frame_origin_pos[side]
        origin_quat = self._frame_origin_quat[side]
        if origin_pos is None or origin_quat is None:
            return raw_pos, raw_quat

        origin_inv = quat_conjugate_xyzw(origin_quat)
        rel_pos = quat_to_rotmat_xyzw(origin_inv) @ (raw_pos - origin_pos)
        rel_quat = quat_normalize_xyzw(quat_multiply_xyzw(origin_inv, raw_quat))
        return rel_pos, rel_quat

    def _compute_angular_delta(self, current_input_quat: Optional[np.ndarray]) -> np.ndarray:
        if (
            self._orientation_mode != "hand_relative"
            or self._initial_input_quat is None
            or current_input_quat is None
        ):
            return np.zeros(3, dtype=np.float64)

        q_curr = np.asarray(current_input_quat, dtype=np.float64)
        q_start = np.asarray(self._initial_input_quat, dtype=np.float64)
        x1, y1, z1, w1 = q_curr
        x2, y2, z2, w2 = np.array([-q_start[0], -q_start[1], -q_start[2], q_start[3]], dtype=np.float64)
        q_delta = np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )
        q_norm = float(np.linalg.norm(q_delta))
        if q_norm > 1e-12:
            q_delta = q_delta / q_norm
        angle = 2.0 * math.acos(_clamp(float(q_delta[3]), -1.0, 1.0))
        s = math.sqrt(max(0.0, 1.0 - float(q_delta[3]) * float(q_delta[3])))
        if s < 1e-8 or angle < 1e-8:
            rotvec = np.zeros(3, dtype=np.float64)
        else:
            axis = np.array([q_delta[0] / s, q_delta[1] / s, q_delta[2] / s], dtype=np.float64)
            rotvec = axis * angle
        mapped = self._select_axis(rotvec, self._orientation_mapping, self._orientation_sign, self._angular_scale)
        return self._apply_angular_smoothing(mapped)

    def _build_pose_target_twist(
        self,
        controller_pos: np.ndarray,
        controller_quat: Optional[np.ndarray],
    ) -> Optional[Twist]:
        current_pose = self._get_current_tool_pose()
        if (
            current_pose is None
            or self._initial_input_pos is None
            or self._initial_robot_pos is None
            or self._initial_robot_quat is None
        ):
            return None

        current_pos, current_quat = current_pose
        delta_input = controller_pos - self._initial_input_pos
        target_offset = self._select_axis(delta_input, self._linear_mapping, self._linear_sign, self._linear_scale)
        target_offset = self._apply_linear_smoothing(target_offset)
        target_pos = self._initial_robot_pos + target_offset

        target_quat = self._initial_robot_quat.copy()
        if self._orientation_mode == "hand_relative":
            target_rotvec = self._compute_angular_delta(controller_quat)
            target_quat = quat_multiply_xyzw(target_quat, rotvec_to_quat_xyzw(target_rotvec))
            target_quat = quat_normalize_xyzw(target_quat)

        pos_error, rot_error = compute_pose_error(current_pos, current_quat, target_pos, target_quat)
        linear_values = pos_error * self._position_linear_gain
        angular_values = rot_error * self._position_angular_gain
        return self._build_twist(linear_values, angular_values)

    def _controller_clutch_desired(self, side: str, state: _QuestControllerState) -> bool:
        axis_active = float(state.squeeze_value) >= self._clutch_axis_threshold
        button_active = self._safe_button(state.buttons, self._clutch_button_index[side])
        return bool(axis_active or button_active)

    def _controller_gripper_value(self, side: str, state: _QuestControllerState) -> float:
        axis_value = float(_clamp(state.trigger_value, 0.0, 1.0))
        button_active = self._safe_button(state.buttons, self._trigger_button_index[side])
        value = max(axis_value, 1.0 if button_active else 0.0)
        if self._gripper_axis_inverted:
            value = 1.0 - value
        return float(_clamp(value, 0.0, 1.0))

    def _refresh_cached_command(self) -> None:
        twist = _zero_twist()
        gripper = self._current_gripper()
        now_ns = self.node.get_clock().now().nanoseconds
        state = self._controllers[self._active_hand]
        calibrated_pose = self._calibrated_controller_pose(self._active_hand)

        if self._require_connected and not self._connected:
            if self._teleop_active:
                self.node.get_logger().warn("Quest3 stream disconnected; stopping motion.")
                if self._gripper_requires_clutch:
                    self._request_gripper_cancel()
            self._reset_control_state()
            self._cache_command(twist, gripper)
            return

        clutch = self._clutch_filter(self._controller_clutch_desired(self._active_hand, state), now_ns)
        if not self._gripper_requires_clutch or clutch:
            gripper = self._controller_gripper_value(self._active_hand, state)

        pose_ready = calibrated_pose is not None
        tool_pose_ready = self._motion_mode != "target_pose" or self._get_current_tool_pose() is not None
        if clutch and pose_ready and tool_pose_ready and calibrated_pose is not None:
            controller_pos, controller_quat = calibrated_pose
            if not self._teleop_active:
                self._initial_input_pos = controller_pos.copy()
                self._initial_input_quat = controller_quat.copy()
                current_pose = self._get_current_tool_pose()
                if current_pose is not None:
                    self._initial_robot_pos = current_pose[0].copy()
                    self._initial_robot_quat = current_pose[1].copy()
                self._linear_filter.reset(np.zeros(3, dtype=np.float64))
                self._angular_filter.reset(np.zeros(3, dtype=np.float64))
                self._teleop_active = True
                self.node.get_logger().info(
                    f"Quest3 clutch engaged; teleop active ({self._motion_mode}, hand={self._active_hand})"
                )

            if self._motion_mode == "target_pose":
                pose_twist = self._build_pose_target_twist(controller_pos, controller_quat)
                if pose_twist is not None:
                    twist = pose_twist
            elif self._initial_input_pos is not None:
                delta_input = controller_pos - self._initial_input_pos
                linear_values = self._select_axis(
                    delta_input,
                    self._linear_mapping,
                    self._linear_sign,
                    self._linear_scale,
                )
                linear_values = self._apply_linear_smoothing(linear_values)
                angular_values = self._compute_angular_delta(controller_quat)
                twist = self._build_twist(linear_values, angular_values)
        else:
            if self._teleop_active:
                reason = "clutch released"
                if clutch and not pose_ready:
                    reason = "controller pose stale"
                elif clutch and not tool_pose_ready:
                    reason = "robot TCP unavailable"
                self.node.get_logger().info(f"Quest3 teleop idle ({reason}).")
                if self._gripper_requires_clutch:
                    self._request_gripper_cancel()
            self._reset_control_state()

        self._cache_command(twist, gripper)

    def get_command(self) -> tuple[Twist, float]:
        return self._get_cached_command()
