#!/usr/bin/env python3
"""输入策略层：不同输入源统一输出标准 Twist 与夹爪值。"""

from __future__ import annotations

import math
import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Joy

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - 运行环境未安装时保持 joy 路径可导入
    mp = None

from ...utils.home_zone_utils import compute_pose_error
from ...utils.transform_utils import (
    _clamp,
    map_axis_linear,
    map_axis_nonlinear,
    quat_multiply_xyzw,
    quat_normalize_xyzw,
    rotvec_to_quat_xyzw,
)


def _zero_twist() -> Twist:
    twist = Twist()
    twist.linear.x = 0.0
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0
    return twist


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.zeros_like(vec, dtype=np.float64)
    return (vec / norm).astype(np.float64)


def _rotmat_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    m00, m01, m02 = float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2])
    m10, m11, m12 = float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2])
    m20, m21, m22 = float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2])

    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif (m00 > m11) and (m00 > m22):
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


class _LowPassFilter:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.state: Optional[np.ndarray] = None

    def reset(self, value: Optional[np.ndarray] = None) -> None:
        self.state = None if value is None else np.asarray(value, dtype=np.float64)

    def apply(self, value: np.ndarray) -> np.ndarray:
        incoming = np.asarray(value, dtype=np.float64)
        if self.state is None:
            self.state = incoming
            return incoming
        self.state = self.alpha * incoming + (1.0 - self.alpha) * self.state
        return self.state


class InputHandlerBase(ABC):
    """输入策略统一接口。"""

    def __init__(self, node: Node) -> None:
        self.node = node
        self._lock = threading.Lock()
        self._latest_twist = _zero_twist()
        self._latest_gripper = 0.0
        self._watchdog_timeout_sec = max(0.0, float(node.get_parameter("input_watchdog_timeout_sec").value))
        self._last_msg_time = time.monotonic()

    def _cache_command(self, twist: Twist, gripper: float) -> None:
        with self._lock:
            self._latest_twist = twist
            self._latest_gripper = float(_clamp(gripper, 0.0, 1.0))
            self._last_msg_time = time.monotonic()

    def _get_cached_command(self) -> tuple[Twist, float]:
        with self._lock:
            timed_out = self._watchdog_timeout_sec > 0.0 and (
                time.monotonic() - self._last_msg_time
            ) > self._watchdog_timeout_sec

            twist = _zero_twist() if timed_out else Twist()
            if not timed_out:
                twist.linear.x = self._latest_twist.linear.x
                twist.linear.y = self._latest_twist.linear.y
                twist.linear.z = self._latest_twist.linear.z
                twist.angular.x = self._latest_twist.angular.x
                twist.angular.y = self._latest_twist.angular.y
                twist.angular.z = self._latest_twist.angular.z

            gripper = 0.0 if timed_out else float(self._latest_gripper)
            return twist, gripper

    @abstractmethod
    def get_command(self) -> tuple[Twist, float]:
        """返回标准化控制命令：(twist, gripper_value)。"""

    def stop(self) -> None:
        """释放输入策略持有的资源。"""

    def _request_gripper_cancel(self) -> None:
        cancel_fn = getattr(self.node, "cancel_gripper_motion", None)
        if callable(cancel_fn):
            try:
                cancel_fn()
            except Exception:
                pass


class JoyInputHandler(InputHandlerBase):
    """Joy 手柄输入策略，内部完成映射和缓存。"""

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self._deadzone = float(node.get_parameter("joy_deadzone").value)
        self._curve = str(node.get_parameter("joy_curve").value).strip().lower()
        self._joy_topic = str(node.get_parameter("joy_topic").value)

        self._max_linear = float(node.get_parameter("max_linear_vel").value)
        self._max_angular = float(node.get_parameter("max_angular_vel").value)

        self._linear_axes = [
            int(node.get_parameter("linear_x_axis").value),
            int(node.get_parameter("linear_y_axis").value),
            int(node.get_parameter("linear_z_axis").value),
        ]
        self._linear_z_up_axis = int(node.get_parameter("linear_z_up_axis").value)
        self._linear_z_down_axis = int(node.get_parameter("linear_z_down_axis").value)
        self._angular_axes = [
            int(node.get_parameter("angular_x_axis").value),
            int(node.get_parameter("angular_y_axis").value),
            int(node.get_parameter("angular_z_axis").value),
        ]
        self._angular_z_positive_button = int(node.get_parameter("angular_z_positive_button").value)
        self._angular_z_negative_button = int(node.get_parameter("angular_z_negative_button").value)
        self._linear_sign = [float(value) for value in node.get_parameter("linear_axis_sign").value]
        self._angular_sign = [float(value) for value in node.get_parameter("angular_axis_sign").value]

        self._deadman_button = int(node.get_parameter("deadman_button").value)
        self._deadman_axis = int(node.get_parameter("deadman_axis").value)
        self._deadman_axis_threshold = float(node.get_parameter("deadman_axis_threshold").value)
        self._deadman_enabled = bool(node.get_parameter("joy_deadman_enabled").value)

        self._gripper_axis = int(node.get_parameter("gripper_axis").value)
        self._gripper_close_button = int(node.get_parameter("gripper_close_button").value)
        self._gripper_open_button = int(node.get_parameter("gripper_open_button").value)
        self._gripper_axis_inverted = bool(node.get_parameter("gripper_axis_inverted").value)
        self._last_deadman_active = False

        node.create_subscription(Joy, self._joy_topic, self._joy_callback, qos_profile_sensor_data)

    def _axis_value(self, values: Sequence[float], index: int) -> float:
        if index < 0 or index >= len(values):
            return 0.0
        raw = float(values[index])
        if self._curve == "cubic":
            return map_axis_nonlinear(raw, deadzone=self._deadzone, exponent=3.0, scale=1.0)
        return map_axis_linear(raw, deadzone=self._deadzone, scale=1.0)

    def _button_value(self, values: Sequence[int], index: int) -> bool:
        return 0 <= index < len(values) and bool(values[index])

    def _button_axis(self, buttons: Sequence[int], positive_button: int, negative_button: int) -> float:
        positive = self._button_value(buttons, positive_button)
        negative = self._button_value(buttons, negative_button)
        if positive and not negative:
            return 1.0
        if negative and not positive:
            return -1.0
        return 0.0

    def _trigger_axis(self, axes: Sequence[float], positive_axis: int, negative_axis: int) -> float:
        positive = self._axis_value(axes, positive_axis)
        negative = self._axis_value(axes, negative_axis)
        return float(_clamp(positive - negative, -1.0, 1.0))

    def _deadman_active(self, axes: Sequence[float], buttons: Sequence[int]) -> bool:
        if not self._deadman_enabled:
            return True
        if self._deadman_button >= 0 and self._button_value(buttons, self._deadman_button):
            return True
        if self._deadman_axis >= 0 and self._deadman_axis < len(axes):
            return float(axes[self._deadman_axis]) >= self._deadman_axis_threshold
        return False

    def _joy_callback(self, msg: Joy) -> None:
        axes = list(msg.axes)
        buttons = list(msg.buttons)
        twist = _zero_twist()
        deadman_active = self._deadman_active(axes, buttons)

        if self._last_deadman_active and not deadman_active:
            self._request_gripper_cancel()

        if deadman_active:
            linear_z = self._axis_value(axes, self._linear_axes[2])
            if self._linear_axes[2] < 0:
                linear_z = self._trigger_axis(axes, self._linear_z_up_axis, self._linear_z_down_axis)

            angular_z = self._axis_value(axes, self._angular_axes[2])
            if self._angular_axes[2] < 0:
                angular_z = self._button_axis(
                    buttons,
                    self._angular_z_positive_button,
                    self._angular_z_negative_button,
                )

            linear = [
                self._axis_value(axes, self._linear_axes[0]) * self._linear_sign[0] * self._max_linear,
                self._axis_value(axes, self._linear_axes[1]) * self._linear_sign[1] * self._max_linear,
                linear_z * self._linear_sign[2] * self._max_linear,
            ]
            angular = [
                self._axis_value(axes, self._angular_axes[0]) * self._angular_sign[0] * self._max_angular,
                self._axis_value(axes, self._angular_axes[1]) * self._angular_sign[1] * self._max_angular,
                angular_z * self._angular_sign[2] * self._max_angular,
            ]
            twist.linear.x, twist.linear.y, twist.linear.z = linear
            twist.angular.x, twist.angular.y, twist.angular.z = angular

        gripper = float(self._latest_gripper)
        with self._lock:
            gripper = float(self._latest_gripper)

        if deadman_active:
            if self._gripper_axis >= 0 and self._gripper_axis < len(axes):
                axis_val = self._axis_value(axes, self._gripper_axis)
                gripper = axis_val if axis_val >= 0.0 else 0.5 * (axis_val + 1.0)
                if self._gripper_axis_inverted:
                    gripper = 1.0 - gripper
            else:
                if self._button_value(buttons, self._gripper_close_button):
                    gripper = 1.0
                elif self._button_value(buttons, self._gripper_open_button):
                    gripper = 0.0

        self._cache_command(twist, gripper)
        self._last_deadman_active = deadman_active

    def get_command(self) -> tuple[Twist, float]:
        return self._get_cached_command()


class MediaPipeInputHandler(InputHandlerBase):
    """MediaPipe 输入策略，直接订阅图像话题并在本地完成手势识别。"""

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        if mp is None:
            raise ImportError("MediaPipeInputHandler requires the 'mediapipe' package to be installed.")
        self._motion_mode = str(node.get_parameter("mediapipe_motion_mode").value).strip().lower()
        self._deadzone = float(node.get_parameter("mediapipe_deadzone").value)
        self._linear_scale = float(node.get_parameter("mediapipe_linear_scale").value)
        self._angular_scale = float(node.get_parameter("mediapipe_angular_scale").value)
        self._position_linear_gain = float(node.get_parameter("mediapipe_position_linear_gain").value)
        self._position_angular_gain = float(node.get_parameter("mediapipe_position_angular_gain").value)
        self._linear_mapping = [int(v) for v in node.get_parameter("mediapipe_linear_axis_mapping").value]
        self._angular_mapping = [int(v) for v in node.get_parameter("mediapipe_angular_axis_mapping").value]
        self._linear_sign = [float(v) for v in node.get_parameter("mediapipe_linear_axis_sign").value]
        self._angular_sign = [float(v) for v in node.get_parameter("mediapipe_angular_axis_sign").value]

        configured_input_topic = str(node.get_parameter("mediapipe_input_topic").value).strip()
        legacy_topic = str(node.get_parameter("mediapipe_topic").value).strip()
        self._image_topic = configured_input_topic or legacy_topic or "/camera/camera/color/image_raw"
        self._depth_topic = str(node.get_parameter("mediapipe_depth_topic").value).strip()
        self._camera_info_topic = str(node.get_parameter("mediapipe_camera_info_topic").value).strip()
        self._tool_pose_topic = str(node.get_parameter("tool_pose_topic").value).strip()
        self._hand_position_source = str(node.get_parameter("mediapipe_hand_position_source").value).strip().lower()
        self._orientation_mode = str(node.get_parameter("mediapipe_orientation_mode").value).strip().lower()
        self._orientation_mapping = [int(v) for v in node.get_parameter("mediapipe_orientation_axis_mapping").value]
        self._orientation_sign = [float(v) for v in node.get_parameter("mediapipe_orientation_axis_sign").value]
        self._depth_min_m = float(node.get_parameter("mediapipe_depth_min_m").value)
        self._depth_max_m = float(node.get_parameter("mediapipe_depth_max_m").value)
        self._depth_unit_scale = float(node.get_parameter("mediapipe_depth_unit_scale").value)
        self._gripper_open_dist_px = float(node.get_parameter("mediapipe_gripper_open_dist_px").value)
        self._gripper_close_dist_px = float(node.get_parameter("mediapipe_gripper_close_dist_px").value)
        self._gripper_open_dist_m = float(node.get_parameter("mediapipe_gripper_open_dist_m").value)
        self._gripper_close_dist_m = float(node.get_parameter("mediapipe_gripper_close_dist_m").value)
        self._gripper_metric_hold_sec = float(node.get_parameter("mediapipe_gripper_metric_hold_sec").value)
        self._gripper_requires_deadman = bool(node.get_parameter("mediapipe_gripper_requires_deadman").value)
        self._deadman_filter_enabled = bool(node.get_parameter("mediapipe_deadman_filter_enabled").value)
        self._deadman_engage_confirm_sec = float(node.get_parameter("mediapipe_deadman_engage_confirm_sec").value)
        self._deadman_release_confirm_sec = float(node.get_parameter("mediapipe_deadman_release_confirm_sec").value)
        self._space_deadman_backend = str(node.get_parameter("mediapipe_space_deadman_backend").value).strip().lower()
        self._space_deadman_hold_sec = float(node.get_parameter("mediapipe_space_deadman_hold_sec").value)
        self._show_debug_window = bool(node.get_parameter("mediapipe_show_debug_window").value)
        self._mediapipe_camera_driver = str(node.get_parameter("mediapipe_camera_driver").value).strip().lower()
        self._mediapipe_camera_serial_number = str(node.get_parameter("mediapipe_camera_serial_number").value).strip()
        self._mediapipe_enable_depth = bool(node.get_parameter("mediapipe_enable_depth").value)
        self._mediapipe_use_sdk_camera = bool(node.get_parameter("mediapipe_use_sdk_camera").value)
        smoothing_alpha = float(node.get_parameter("mediapipe_smoothing_alpha").value)

        if self._motion_mode not in {"velocity", "target_pose"}:
            self._motion_mode = "target_pose"
        if self._hand_position_source not in {"depth", "normalized", "hybrid"}:
            self._hand_position_source = "hybrid"
        if self._orientation_mode not in {"lock", "hand_relative"}:
            self._orientation_mode = "lock"
        if self._space_deadman_backend not in {"opencv", "pynput"}:
            self._space_deadman_backend = "pynput"

        self._bridge = CvBridge()
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._mp_drawing = mp.solutions.drawing_utils
        self._linear_filter = _LowPassFilter(alpha=smoothing_alpha)
        self._angular_filter = _LowPassFilter(alpha=smoothing_alpha)

        self._latest_depth_image: Optional[np.ndarray] = None
        self._latest_depth_info: Optional[CameraInfo] = None
        self._depth_fx: Optional[float] = None
        self._depth_fy: Optional[float] = None
        self._depth_cx: Optional[float] = None
        self._depth_cy: Optional[float] = None
        self._latest_tool_pos: Optional[np.ndarray] = None
        self._latest_tool_quat: Optional[np.ndarray] = None

        self._initial_hand_pos: Optional[np.ndarray] = None
        self._initial_hand_quat: Optional[np.ndarray] = None
        self._initial_robot_pos: Optional[np.ndarray] = None
        self._initial_robot_quat: Optional[np.ndarray] = None
        self._deadman_active = False
        self._space_deadman_until_ns = 0
        self._space_down = False
        self._keyboard_listener = None
        self._deadman_filtered = False
        self._deadman_candidate: Optional[bool] = None
        self._deadman_candidate_since_ns = 0
        self._last_metric_gripper_dist_m: Optional[float] = None
        self._last_metric_gripper_dist_ns = 0
        self._window_available = self._show_debug_window or self._space_deadman_backend == "opencv"
        self._sdk_camera = None
        self._using_sdk_camera = False
        self._last_sdk_warn_ns = 0
        self._window_size_synced = False
        self._window_size = (0, 0)

        if self._show_debug_window:
            try:
                window_flags = int(getattr(cv2, "WINDOW_NORMAL", 0)) | int(getattr(cv2, "WINDOW_GUI_NORMAL", 0))
                cv2.namedWindow("Teleop", window_flags)
            except Exception as exc:
                self.node.get_logger().warn(
                    f"MediaPipe debug window init failed, disabling OpenCV UI. Err: {exc}"
                )
                self._window_available = False

        self._using_sdk_camera = self._maybe_start_sdk_camera()
        if not self._using_sdk_camera:
            node.create_subscription(Image, self._image_topic, self._image_callback, qos_profile_sensor_data)
            if self._depth_topic:
                node.create_subscription(Image, self._depth_topic, self._depth_callback, qos_profile_sensor_data)
            if self._camera_info_topic:
                node.create_subscription(CameraInfo, self._camera_info_topic, self._camera_info_callback, qos_profile_sensor_data)
        if self._tool_pose_topic:
            node.create_subscription(PoseStamped, self._tool_pose_topic, self._tool_pose_callback, 10)

        if self._space_deadman_backend == "pynput":
            self._start_keyboard_listener()

        node.get_logger().info(
            f"MediaPipeInputHandler ready. motion_mode={self._motion_mode}, "
            f"camera_source={'SDK' if self._using_sdk_camera else 'ROS_TOPIC'}, "
            f"camera_driver={self._normalized_camera_driver()}, "
            f"camera_serial={'auto' if not self._mediapipe_camera_serial_number else self._mediapipe_camera_serial_number}, "
            f"sdk_depth={'on' if self._mediapipe_enable_depth else 'off'}, "
            f"image_topic={self._image_topic}, depth_topic={self._depth_topic or '-'}, "
            f"camera_info_topic={self._camera_info_topic or '-'}, tool_pose_topic={self._tool_pose_topic or '-'}"
        )

    def _normalized_camera_driver(self) -> str:
        value = str(self._mediapipe_camera_driver).strip().lower()
        if value in {"oakd"}:
            return "oakd"
        return "realsense"

    def _maybe_start_sdk_camera(self) -> bool:
        if not self._mediapipe_use_sdk_camera:
            return False
        try:
            from ..cameras.camera_client import create_sdk_camera
        except Exception as exc:
            self.node.get_logger().warn(
                f"MediaPipe SDK camera import failed, fallback to ROS topic input. Err: {exc}"
            )
            return False

        camera_driver = self._normalized_camera_driver()
        try:
            self._sdk_camera = create_sdk_camera(
                camera_driver,
                serial_number=self._mediapipe_camera_serial_number,
                logger=self.node.get_logger(),
                enable_depth=self._mediapipe_enable_depth,
            )
            self._sdk_camera.start()
            self.node.get_logger().info(
                "MediaPipe camera initialized from SDK: "
                f"type={camera_driver}, serial={'auto' if not self._mediapipe_camera_serial_number else self._mediapipe_camera_serial_number}, "
                f"depth={'on' if self._mediapipe_enable_depth else 'off'}"
            )
            return True
        except Exception as exc:
            self._sdk_camera = None
            self.node.get_logger().warn(
                f"MediaPipe SDK camera startup failed, fallback to ROS topic input. Err: {exc}"
            )
            return False

    def _sync_depth_intrinsics_from_sdk(self, intrinsics) -> None:
        if intrinsics is None:
            return
        fx = float(getattr(intrinsics, "fx", 0.0))
        fy = float(getattr(intrinsics, "fy", 0.0))
        cx = float(getattr(intrinsics, "cx", 0.0))
        cy = float(getattr(intrinsics, "cy", 0.0))
        if fx <= 0.0 or fy <= 0.0:
            return
        self._depth_fx = fx
        self._depth_fy = fy
        self._depth_cx = cx
        self._depth_cy = cy

        camera_info = self._latest_depth_info or CameraInfo()
        camera_info.width = int(getattr(intrinsics, "width", 0))
        camera_info.height = int(getattr(intrinsics, "height", 0))
        camera_info.k = [
            self._depth_fx,
            0.0,
            self._depth_cx,
            0.0,
            self._depth_fy,
            self._depth_cy,
            0.0,
            0.0,
            1.0,
        ]
        self._latest_depth_info = camera_info

    def _depth_ready(self) -> bool:
        return (
            self._latest_depth_image is not None
            and self._latest_depth_info is not None
            and self._depth_fx is not None
            and self._depth_fy is not None
        )

    def _build_depth_preview(self, preview_width: int, preview_height: int) -> Optional[np.ndarray]:
        if preview_width <= 0 or preview_height <= 0:
            return None
        depth = self._latest_depth_image
        if depth is None:
            return None

        depth_values = np.asarray(depth)
        if depth_values.size == 0:
            return None
        if depth_values.dtype == np.uint16:
            depth_m = depth_values.astype(np.float32) * self._depth_unit_scale
        else:
            depth_m = depth_values.astype(np.float32)

        valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
        if not np.any(valid_mask):
            return None

        valid_depth = depth_m[valid_mask]
        depth_min = self._depth_min_m if self._depth_min_m > 0.0 else float(np.percentile(valid_depth, 5.0))
        depth_max = self._depth_max_m if self._depth_max_m > depth_min else float(np.percentile(valid_depth, 95.0))
        if depth_max <= depth_min:
            depth_max = depth_min + 1e-3

        depth_norm = np.zeros_like(depth_m, dtype=np.uint8)
        clipped = np.clip((depth_m - depth_min) / (depth_max - depth_min), 0.0, 1.0)
        depth_norm[valid_mask] = (255.0 * (1.0 - clipped[valid_mask])).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        depth_color[~valid_mask] = (0, 0, 0)
        return cv2.resize(depth_color, (int(preview_width), int(preview_height)), interpolation=cv2.INTER_LINEAR)

    def _poll_sdk_camera(self) -> None:
        if self._sdk_camera is None:
            return
        try:
            frame = self._sdk_camera.get_frame()
        except Exception as exc:
            now_ns = self.node.get_clock().now().nanoseconds
            if now_ns - self._last_sdk_warn_ns > int(2.0 * 1e9):
                self.node.get_logger().warn(f"MediaPipe SDK camera read failed: {exc}")
                self._last_sdk_warn_ns = now_ns
            return

        if frame is None:
            return

        depth_frame = getattr(frame, "depth", None)
        if depth_frame is not None:
            self._latest_depth_image = np.asarray(depth_frame, dtype=np.float32)
        self._sync_depth_intrinsics_from_sdk(getattr(frame, "intrinsics", None))

        bgr = getattr(frame, "bgr", None)
        if bgr is None:
            return
        try:
            image_msg = self._bridge.cv2_to_imgmsg(np.ascontiguousarray(bgr), encoding="bgr8")
        except Exception as exc:
            now_ns = self.node.get_clock().now().nanoseconds
            if now_ns - self._last_sdk_warn_ns > int(2.0 * 1e9):
                self.node.get_logger().warn(f"MediaPipe SDK frame conversion failed: {exc}")
                self._last_sdk_warn_ns = now_ns
            return
        self._image_callback(image_msg)

    def _start_keyboard_listener(self) -> None:
        try:
            from pynput import keyboard  # type: ignore

            def on_press(key):
                try:
                    if key == keyboard.Key.space:
                        self._space_down = True
                except Exception:
                    pass

            def on_release(key):
                try:
                    if key == keyboard.Key.space:
                        self._space_down = False
                except Exception:
                    pass

            self._keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._keyboard_listener.daemon = True
            self._keyboard_listener.start()
        except Exception as exc:
            self.node.get_logger().warn(
                f"Failed to start pynput keyboard listener; falling back to opencv. Err: {exc}"
            )
            self._space_deadman_backend = "opencv"

    def _apply_deadzone(self, value: float) -> float:
        raw = float(value)
        threshold = abs(self._deadzone)
        if threshold <= 0.0 or abs(raw) <= threshold:
            return 0.0 if threshold > 0.0 else raw
        return raw - math.copysign(threshold, raw)

    def _deadman_filter(self, desired: bool, now_ns: int) -> bool:
        if not self._deadman_filter_enabled:
            self._deadman_filtered = desired
            self._deadman_candidate = None
            return desired

        if self._deadman_candidate is None or desired != self._deadman_candidate:
            self._deadman_candidate = desired
            self._deadman_candidate_since_ns = now_ns
            return self._deadman_filtered

        if desired == self._deadman_filtered:
            return self._deadman_filtered

        confirm_sec = self._deadman_engage_confirm_sec if desired else self._deadman_release_confirm_sec
        confirm_ns = int(max(0.0, confirm_sec) * 1e9)
        if now_ns - self._deadman_candidate_since_ns >= confirm_ns:
            self._deadman_filtered = desired
            self._deadman_candidate = None
        return self._deadman_filtered

    def _current_gripper(self) -> float:
        with self._lock:
            return float(self._latest_gripper)

    def _tool_pose_callback(self, msg: PoseStamped) -> None:
        position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        )
        quat = quat_normalize_xyzw(
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
        self._latest_tool_pos = position
        self._latest_tool_quat = quat

    def _get_current_tool_pose(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        tool_pos = getattr(self, "_latest_tool_pos", None)
        tool_quat = getattr(self, "_latest_tool_quat", None)
        if tool_pos is None or tool_quat is None:
            return None
        return np.asarray(tool_pos, dtype=np.float64).copy(), np.asarray(tool_quat, dtype=np.float64).copy()

    def _reset_deadman_state(self) -> None:
        self._deadman_active = False
        self._initial_hand_pos = None
        self._initial_hand_quat = None
        self._initial_robot_pos = None
        self._initial_robot_quat = None
        self._linear_filter.reset(np.zeros(3, dtype=np.float64))
        self._angular_filter.reset(np.zeros(3, dtype=np.float64))

    def _landmark_px(self, landmark, width: int, height: int) -> tuple[int, int]:
        return int(landmark.x * width), int(landmark.y * height)

    def _get_depth_m(self, u: int, v: int) -> Optional[float]:
        if self._latest_depth_image is None:
            return None
        height, width = self._latest_depth_image.shape[:2]
        if u < 0 or v < 0 or u >= width or v >= height:
            return None
        window = self._latest_depth_image[max(v - 2, 0) : min(v + 3, height), max(u - 2, 0) : min(u + 3, width)]
        values = window.flatten()
        if values.dtype == np.uint16:
            scaled = values.astype(np.float32) * self._depth_unit_scale
        else:
            scaled = values.astype(np.float32)
        scaled = scaled[np.isfinite(scaled) & (scaled > 0.0)]
        if scaled.size == 0:
            return None
        depth = float(np.median(scaled))
        if self._depth_min_m > 0.0 and depth < self._depth_min_m:
            return None
        if self._depth_max_m > 0.0 and depth > self._depth_max_m:
            return None
        return depth

    def _deproject(self, uv: tuple[int, int], depth_m: float) -> np.ndarray:
        if self._depth_fx is None or self._depth_fy is None or self._depth_cx is None or self._depth_cy is None:
            raise RuntimeError("Camera intrinsics are unavailable")
        u, v = uv
        x = (u - self._depth_cx) * depth_m / self._depth_fx
        y = (v - self._depth_cy) * depth_m / self._depth_fy
        z = depth_m
        return np.array([x, y, z], dtype=np.float64)

    def _landmark_3d_m(self, landmark, width: int, height: int) -> Optional[np.ndarray]:
        if self._latest_depth_image is None or self._latest_depth_info is None or self._depth_fx is None:
            return None
        uv = self._landmark_px(landmark, width, height)
        depth_m = self._get_depth_m(*uv)
        if depth_m is None:
            return None
        return self._deproject(uv, depth_m)

    def _get_hand_position(self, wrist_landmark, width: int, height: int) -> tuple[Optional[np.ndarray], str]:
        if self._hand_position_source in {"depth", "hybrid"}:
            depth_position = self._landmark_3d_m(wrist_landmark, width, height)
            if depth_position is not None:
                return depth_position.astype(np.float64), "DEPTH"
            if self._hand_position_source == "depth":
                return None, "DEPTH_NONE"

        return np.array([wrist_landmark.x, wrist_landmark.y, wrist_landmark.z], dtype=np.float64), "NORM"

    def _physical_distance_m(self, p1, p2, width: int, height: int) -> Optional[float]:
        if self._latest_depth_image is None or self._latest_depth_info is None or self._depth_fx is None:
            return None
        uv1 = self._landmark_px(p1, width, height)
        uv2 = self._landmark_px(p2, width, height)
        depth1 = self._get_depth_m(*uv1)
        depth2 = self._get_depth_m(*uv2)
        if depth1 is None and depth2 is None:
            return None
        if depth1 is None:
            shared_depth = float(depth2)
        elif depth2 is None:
            shared_depth = float(depth1)
        else:
            shared_depth = 0.5 * (float(depth1) + float(depth2))
        point1 = self._deproject(uv1, shared_depth)
        point2 = self._deproject(uv2, shared_depth)
        return float(np.linalg.norm(point2 - point1))

    def _distance_px(self, p1, p2, width: int, height: int) -> float:
        x1, y1 = p1.x * width, p1.y * height
        x2, y2 = p2.x * width, p2.y * height
        return math.hypot(x2 - x1, y2 - y1)

    def _gripper_from_distance(self, thumb_tip, index_tip, width: int, height: int) -> float:
        dist_m = self._physical_distance_m(thumb_tip, index_tip, width, height)
        if dist_m is None and self._last_metric_gripper_dist_m is not None:
            now_ns = self.node.get_clock().now().nanoseconds
            hold_ns = int(max(0.0, self._gripper_metric_hold_sec) * 1e9)
            if now_ns - self._last_metric_gripper_dist_ns <= hold_ns:
                dist_m = float(self._last_metric_gripper_dist_m)

        if dist_m is not None:
            self._last_metric_gripper_dist_m = float(dist_m)
            self._last_metric_gripper_dist_ns = self.node.get_clock().now().nanoseconds
            openness = _clamp(
                (dist_m - self._gripper_close_dist_m)
                / max(1e-9, self._gripper_open_dist_m - self._gripper_close_dist_m),
                0.0,
                1.0,
            )
        else:
            dist_px = self._distance_px(thumb_tip, index_tip, width, height)
            openness = _clamp(
                (dist_px - self._gripper_close_dist_px)
                / max(1e-9, self._gripper_open_dist_px - self._gripper_close_dist_px),
                0.0,
                1.0,
            )
        return 1.0 - openness

    def _hand_quat_from_points(
        self,
        wrist_point: np.ndarray,
        index_point: np.ndarray,
        pinky_point: np.ndarray,
    ) -> Optional[np.ndarray]:
        x_axis = _normalize_vector(index_point - wrist_point)
        y_hint = _normalize_vector(pinky_point - wrist_point)
        z_axis = np.cross(x_axis, y_hint)
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < 1e-9:
            return None
        z_axis /= z_norm
        y_axis = np.cross(z_axis, x_axis)
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm < 1e-9:
            return None
        y_axis /= y_norm
        x_axis = _normalize_vector(x_axis)
        rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        return _rotmat_to_quat_xyzw(rotation)

    def _get_hand_orientation_quat(self, results, wrist_landmark, index_mcp, pinky_mcp, width: int, height: int) -> tuple[Optional[np.ndarray], str]:
        try:
            if getattr(results, "multi_hand_world_landmarks", None):
                landmarks = results.multi_hand_world_landmarks[0].landmark
                wrist_point = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z], dtype=np.float64)
                index_point = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z], dtype=np.float64)
                pinky_point = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z], dtype=np.float64)
                quat = self._hand_quat_from_points(wrist_point, index_point, pinky_point)
                return (quat, "WLD") if quat is not None else (None, "WLD_NONE")
        except Exception:
            pass

        wrist_point = self._landmark_3d_m(wrist_landmark, width, height)
        index_point = self._landmark_3d_m(index_mcp, width, height)
        pinky_point = self._landmark_3d_m(pinky_mcp, width, height)
        if wrist_point is not None and index_point is not None and pinky_point is not None:
            quat = self._hand_quat_from_points(wrist_point, index_point, pinky_point)
            return (quat, "DEP") if quat is not None else (None, "DEP_NONE")
        return None, "ORI_NONE"

    def _select_axis(self, values: np.ndarray, mapping: list[int], signs: list[float], scale: float) -> np.ndarray:
        out = np.zeros(3, dtype=np.float64)
        for i in range(3):
            source_index = mapping[i]
            raw = float(values[source_index]) if 0 <= source_index < len(values) else 0.0
            out[i] = self._apply_deadzone(raw) * scale * float(signs[i])
        return out

    def _build_twist(self, linear_values: np.ndarray, angular_values: np.ndarray) -> Twist:
        twist = _zero_twist()
        twist.linear.x, twist.linear.y, twist.linear.z = [float(v) for v in linear_values]
        twist.angular.x, twist.angular.y, twist.angular.z = [float(v) for v in angular_values]
        return twist

    def _compute_angular_delta(self, current_hand_quat: Optional[np.ndarray]) -> np.ndarray:
        if (
            self._orientation_mode != "hand_relative"
            or self._initial_hand_quat is None
            or current_hand_quat is None
        ):
            return np.zeros(3, dtype=np.float64)

        q_curr = np.asarray(current_hand_quat, dtype=np.float64)
        q_start = np.asarray(self._initial_hand_quat, dtype=np.float64)
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
        w = _clamp(float(q_delta[3]), -1.0, 1.0)
        angle = 2.0 * math.acos(w)
        s = math.sqrt(max(0.0, 1.0 - w * w))
        if s < 1e-8 or angle < 1e-8:
            rotvec = np.zeros(3, dtype=np.float64)
        else:
            axis = np.array([q_delta[0] / s, q_delta[1] / s, q_delta[2] / s], dtype=np.float64)
            rotvec = axis * angle
        mapped = self._select_axis(rotvec, self._orientation_mapping, self._orientation_sign, self._angular_scale)
        return self._angular_filter.apply(mapped)

    def _build_pose_target_twist(
        self,
        wrist_pos: np.ndarray,
        current_hand_quat: Optional[np.ndarray],
    ) -> Optional[Twist]:
        current_pose = self._get_current_tool_pose()
        if (
            current_pose is None
            or self._initial_hand_pos is None
            or self._initial_robot_pos is None
            or self._initial_robot_quat is None
        ):
            return None

        current_pos, current_quat = current_pose
        delta_hand = wrist_pos - self._initial_hand_pos
        target_offset = self._select_axis(delta_hand, self._linear_mapping, self._linear_sign, self._linear_scale)
        target_offset = self._linear_filter.apply(target_offset)
        target_pos = self._initial_robot_pos + target_offset

        target_quat = self._initial_robot_quat.copy()
        if self._orientation_mode == "hand_relative":
            target_rotvec = self._compute_angular_delta(current_hand_quat)
            target_quat = quat_multiply_xyzw(target_quat, rotvec_to_quat_xyzw(target_rotvec))
            target_quat = quat_normalize_xyzw(target_quat)

        pos_error, rot_error = compute_pose_error(current_pos, current_quat, target_pos, target_quat)
        linear_values = pos_error * self._position_linear_gain
        angular_values = rot_error * self._position_angular_gain
        return self._build_twist(linear_values, angular_values)

    def _image_callback(self, msg: Image) -> None:
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.node.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        height, width, _ = cv_image.shape
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)

        twist = _zero_twist()
        cached_gripper = self._current_gripper()
        status_text = "IDLE"
        status_color = (0, 0, 255)
        wrist_source = "-"
        hand_ori_source = "-"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist = hand_landmarks.landmark[0]
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            index_mcp = hand_landmarks.landmark[5]
            pinky_mcp = hand_landmarks.landmark[17]

            if self._show_debug_window:
                self._mp_drawing.draw_landmarks(cv_image, hand_landmarks, self._mp_hands.HAND_CONNECTIONS)

            wrist_pos, wrist_source = self._get_hand_position(wrist, width, height)
            hand_quat, hand_ori_source = self._get_hand_orientation_quat(results, wrist, index_mcp, pinky_mcp, width, height)
            now_ns = self.node.get_clock().now().nanoseconds

            if self._space_deadman_backend == "pynput":
                key_deadman = self._space_down
            else:
                key_deadman = now_ns < self._space_deadman_until_ns
            deadman = self._deadman_filter(bool(key_deadman), now_ns)

            gripper_cmd = self._gripper_from_distance(thumb_tip, index_tip, width, height)
            if not self._gripper_requires_deadman or deadman:
                cached_gripper = gripper_cmd

            if deadman and wrist_pos is not None:
                pose_ready = self._motion_mode != "target_pose" or self._get_current_tool_pose() is not None
                if not pose_ready:
                    self._reset_deadman_state()
                    status_text = "WAITING_TCP"
                    status_color = (0, 165, 255)
                else:
                    if not self._deadman_active:
                        self._initial_hand_pos = wrist_pos.copy()
                        self._initial_hand_quat = hand_quat.copy() if hand_quat is not None else None
                        if self._motion_mode == "target_pose":
                            current_pose = self._get_current_tool_pose()
                            if current_pose is not None:
                                self._initial_robot_pos = current_pose[0].copy()
                                self._initial_robot_quat = current_pose[1].copy()
                        self._linear_filter.reset(np.zeros(3, dtype=np.float64))
                        self._angular_filter.reset(np.zeros(3, dtype=np.float64))
                        self._deadman_active = True
                        self.node.get_logger().info(
                            f"Deadman engaged; gesture teleop active ({self._motion_mode})"
                        )

                    if self._motion_mode == "target_pose":
                        pose_twist = self._build_pose_target_twist(wrist_pos, hand_quat)
                        if pose_twist is not None:
                            twist = pose_twist
                            status_text = "POSE_CTRL"
                            status_color = (0, 200, 0)
                        else:
                            status_text = "WAITING_TCP"
                            status_color = (0, 165, 255)
                    else:
                        delta_hand = wrist_pos - self._initial_hand_pos
                        linear_values = self._select_axis(delta_hand, self._linear_mapping, self._linear_sign, self._linear_scale)
                        linear_values = self._linear_filter.apply(linear_values)
                        angular_values = self._compute_angular_delta(hand_quat)
                        twist = self._build_twist(linear_values, angular_values)
                        status_text = "CONTROLLING"
                        status_color = (0, 200, 0)
            else:
                if self._deadman_active:
                    self.node.get_logger().info("Deadman released; gesture teleop idle")
                    if self._gripper_requires_deadman:
                        self._request_gripper_cancel()
                self._reset_deadman_state()
                if wrist_pos is None and deadman:
                    status_text = "WAITING_DEPTH" if self._hand_position_source != "normalized" else "WAITING_HAND"
                    status_color = (0, 165, 255)

            if self._show_debug_window:
                thumb_px = (int(thumb_tip.x * width), int(thumb_tip.y * height))
                index_px = (int(index_tip.x * width), int(index_tip.y * height))
                cv2.line(cv_image, thumb_px, index_px, (255, 255, 0), 2)
                cv2.circle(cv_image, thumb_px, 4, (255, 0, 255), -1)
                cv2.circle(cv_image, index_px, 4, (0, 255, 255), -1)
        else:
            if self._deadman_active:
                self.node.get_logger().info("Hand lost; stopping motion")
                if self._gripper_requires_deadman:
                    self._request_gripper_cancel()
            self._reset_deadman_state()

        self._cache_command(twist, cached_gripper)

        if self._window_available:
            try:
                if self._show_debug_window:
                    if (not self._window_size_synced) or self._window_size != (width, height):
                        try:
                            cv2.resizeWindow("Teleop", int(width), int(height))
                            self._window_size = (width, height)
                            self._window_size_synced = True
                        except Exception:
                            pass
                    depth_required = self._hand_position_source in {"depth", "hybrid"}
                    depth_ready = self._depth_ready()
                    depth_text = "DEPTH:READY" if depth_ready else ("DEPTH:WAIT" if depth_required else "DEPTH:OFF")
                    depth_color = (0, 200, 0) if depth_ready else ((0, 165, 255) if depth_required else (200, 200, 200))
                    camera_source = "SDK" if self._using_sdk_camera else "ROS"
                    overlay = f"{status_text} ({wrist_source},{hand_ori_source})"
                    cv2.putText(
                        cv_image,
                        overlay,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        status_color,
                        3,
                    )
                    cv2.putText(
                        cv_image,
                        f"{depth_text} [{camera_source}]",
                        (20, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        depth_color,
                        2,
                    )

                    if depth_ready:
                        preview_width = max(220, int(width * 0.32))
                        preview_width = min(preview_width, max(220, width - 40))
                        preview_height = max(160, int(height * 0.32))
                        preview_height = min(preview_height, max(160, height - 60))
                        depth_preview = self._build_depth_preview(preview_width, preview_height)
                        if depth_preview is not None:
                            x1 = max(20, width - preview_width - 20)
                            y1 = 20
                            x2 = min(width, x1 + preview_width)
                            y2 = min(height, y1 + preview_height)
                            if x2 > x1 and y2 > y1:
                                cv_image[y1:y2, x1:x2] = depth_preview[: y2 - y1, : x2 - x1]
                                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                                cv2.putText(
                                    cv_image,
                                    "DEPTH",
                                    (x1 + 10, y1 + 24),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 255),
                                    2,
                                )
                    cv2.imshow("Teleop", cv_image)
                key = cv2.waitKey(1) & 0xFF
                if self._space_deadman_backend == "opencv" and key == 32:
                    hold_ns = int(max(0.0, self._space_deadman_hold_sec) * 1e9)
                    self._space_deadman_until_ns = self.node.get_clock().now().nanoseconds + hold_ns
            except Exception as exc:
                self.node.get_logger().warn(f"MediaPipe debug window unavailable, disabling OpenCV UI. Err: {exc}")
                self._window_available = False

    def _depth_callback(self, msg: Image) -> None:
        try:
            self._latest_depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.node.get_logger().warn(f"Depth conversion failed: {exc}")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._latest_depth_info = msg
        if len(msg.k) == 9:
            self._depth_fx = float(msg.k[0])
            self._depth_fy = float(msg.k[4])
            self._depth_cx = float(msg.k[2])
            self._depth_cy = float(msg.k[5])

    def get_command(self) -> tuple[Twist, float]:
        if self._using_sdk_camera:
            self._poll_sdk_camera()
        return self._get_cached_command()

    def stop(self) -> None:
        try:
            if self._sdk_camera is not None:
                self._sdk_camera.stop()
        except Exception:
            pass
        finally:
            self._sdk_camera = None
        try:
            if self._keyboard_listener is not None:
                self._keyboard_listener.stop()
        except Exception:
            pass
        try:
            self._hands.close()
        except Exception:
            pass
        if self._show_debug_window:
            try:
                cv2.destroyWindow("Teleop")
            except Exception:
                pass
