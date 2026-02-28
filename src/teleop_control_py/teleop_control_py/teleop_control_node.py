#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import cv2
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from qbsofthand_control.srv import SetClosure
from rclpy.node import Node
from std_msgs.msg import Float32

from teleop_control_py.input_handlers import BaseInputHandler, HandInputHandler, JoyInputHandler


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class _NullInputHandler(BaseInputHandler):
    """Fallback strategy when an input backend is not implemented."""

    def get_command(self):
        return None

    def get_gripper_state(self):
        return None

    def is_active(self) -> bool:
        return False


class TeleopControlNode(Node):
    """Teleoperation bridge that delegates input processing via Strategy Pattern."""

    def __init__(self) -> None:
        super().__init__("teleop_control_node")

        # Parameters for easy runtime tuning
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("robot_pose_topic", "/tcp_pose_broadcaster/pose")
        self.declare_parameter("target_pose_topic", "/target_pose")
        self.declare_parameter("target_frame_id", "base")
        self.declare_parameter("gripper_cmd_topic", "/gripper/cmd")
        self.declare_parameter("scale_factor", 0.5)
        self.declare_parameter("axis_mapping", [0, 1, 2])
        self.declare_parameter("smoothing_alpha", 0.2)
        self.declare_parameter("gripper_open_dist_px", 100.0)
        self.declare_parameter("gripper_close_dist_px", 20.0)
        self.declare_parameter("gripper_open_dist_m", 0.12)
        self.declare_parameter("gripper_close_dist_m", 0.03)
        self.declare_parameter("depth_unit_scale", 0.001)  # 16UC1 in mm -> m
        self.declare_parameter("hand_position_source", "hybrid")  # depth|normalized|hybrid
        # lock: keep robot orientation fixed
        # hand_relative: apply hand wrist delta rotation onto robot initial orientation
        self.declare_parameter("orientation_mode", "lock")  # lock|hand_relative
        # When using hand_relative, map the hand delta-rotation vector components into robot delta.
        # This helps fix axis mixing/inversion caused by camera/hand coordinate conventions.
        self.declare_parameter("orientation_axis_mapping", [0, 1, 2])
        self.declare_parameter("orientation_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("depth_min_m", 0.1)
        self.declare_parameter("depth_max_m", 2.0)
        # Keyboard deadman input backend:
        # - opencv: uses cv2.waitKey() events (can miss key-up; not ideal for true "hold")
        # - pynput: tracks press/release reliably (recommended for immediate stop)
        self.declare_parameter("space_deadman_backend", "opencv")  # opencv|pynput
        # opencv backend uses a latch window extended by key repeats
        self.declare_parameter("space_deadman_hold_sec", 0.3)

        # Deadman filtering: avoid flicker without adding noticeable stop latency.
        # Engage can be slightly delayed to avoid false triggers; release should be quick.
        self.declare_parameter("deadman_filter_enabled", True)
        self.declare_parameter("deadman_engage_confirm_sec", 0.10)
        self.declare_parameter("deadman_release_confirm_sec", 0.03)
        self.declare_parameter("gripper_requires_deadman", True)
        self.declare_parameter("gripper_step", 0.2)
        self.declare_parameter("gripper_duration_sec", 0.5)
        self.declare_parameter("gripper_speed_ratio", 1.0)
        # When depth-based metric gripper distance becomes temporarily unavailable,
        # hold the last valid metric distance for a short time to prevent oscillation.
        self.declare_parameter("gripper_metric_hold_sec", 0.25)
        self.declare_parameter("control_mode", "hand")  # hand|xbox
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("xbox_loop_hz", 60.0)
        self.declare_parameter("xbox_deadzone", 0.12)
        self.declare_parameter("xbox_linear_speed", 0.15)  # m/s
        self.declare_parameter("xbox_angular_speed", 0.8)  # rad/s
        self.declare_parameter("xbox_linear_axis", [0, 1, 4])
        self.declare_parameter("xbox_linear_sign", [1.0, -1.0, -1.0])
        self.declare_parameter("xbox_angular_axis", [7, 6, 3])
        self.declare_parameter("xbox_angular_sign", [1.0, 1.0, -1.0])
        self.declare_parameter("xbox_deadman_button", 5)
        self.declare_parameter("xbox_gripper_close_button", 0)
        self.declare_parameter("xbox_gripper_open_button", 1)

        gripper_cmd_topic = self.get_parameter("gripper_cmd_topic").get_parameter_value().string_value
        self.control_mode = self.get_parameter("control_mode").get_parameter_value().string_value.strip().lower()
        if self.control_mode not in {"hand", "xbox"}:
            self.get_logger().warn("control_mode must be 'hand' or 'xbox'; falling back to 'hand'")
            self.control_mode = "hand"

        self.target_frame_id = self.get_parameter("target_frame_id").get_parameter_value().string_value

        self.gripper_step = float(self.get_parameter("gripper_step").get_parameter_value().double_value)
        self.gripper_duration_sec = float(self.get_parameter("gripper_duration_sec").get_parameter_value().double_value)
        self.gripper_speed_ratio = float(self.get_parameter("gripper_speed_ratio").get_parameter_value().double_value)

        # Publishers (fixed topics as a central dispatcher)
        self.pose_pub = self.create_publisher(PoseStamped, "/pose_target_cmds", 10)
        self.twist_pub = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.gripper_pub = self.create_publisher(Float32, gripper_cmd_topic, 10)
        self.gripper_client = self.create_client(SetClosure, "/qbsofthand_control_node/set_closure")
        if not self.gripper_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("SoftHand service not available; will retry on demand")
        self._last_gripper_cmd: Optional[float] = None
        self._last_gripper_call_ns = 0
        self._gripper_delta_threshold = 0.02
        self._gripper_min_interval_ns = int(0.1 * 1e9)
        self.input_handler: BaseInputHandler
        if self.control_mode == "hand":
            self.input_handler = HandInputHandler(self)
            self.get_logger().info("TeleopControlNode mode=hand (HandInputHandler)")
        else:
            self.input_handler = JoyInputHandler(self)
            self.get_logger().info("TeleopControlNode mode=xbox (JoyInputHandler)")

        self._tick_timer = self.create_timer(1.0 / 60.0, self._tick)

    def destroy_node(self):  # type: ignore[override]
        try:
            if isinstance(self.input_handler, HandInputHandler):
                self.input_handler.destroy()
        except Exception:
            pass
        return super().destroy_node()

    def _tick(self) -> None:
        # Deadman first: immediate stop if not active
        if not self.input_handler.is_active():
            stop = TwistStamped()
            stop.header.stamp = self.get_clock().now().to_msg()
            stop.header.frame_id = self.target_frame_id
            stop.twist.linear.x = 0.0
            stop.twist.linear.y = 0.0
            stop.twist.linear.z = 0.0
            stop.twist.angular.x = 0.0
            stop.twist.angular.y = 0.0
            stop.twist.angular.z = 0.0
            self.twist_pub.publish(stop)
            return

        cmd = self.input_handler.get_command()
        if isinstance(cmd, PoseStamped):
            self.pose_pub.publish(cmd)
        elif isinstance(cmd, TwistStamped):
            self.twist_pub.publish(cmd)

        gripper = self.input_handler.get_gripper_state()
        if gripper is not None:
            self._publish_gripper(float(gripper))

    def _publish_gripper(self, cmd: float) -> None:
        cmd_clamped = float(clamp(cmd, 0.0, 1.0))
        step = self.gripper_step if self.gripper_step > 0 else 0.2
        quantized = round(cmd_clamped / step) * step
        quantized = float(clamp(quantized, 0.0, 1.0))
        msg = Float32()
        msg.data = quantized
        self.gripper_pub.publish(msg)
        self._call_gripper_service(quantized)

    def _call_gripper_service(self, closure: float) -> None:
        closure = float(clamp(closure, 0.0, 1.0))
        now_ns = self.get_clock().now().nanoseconds
        if (
            self._last_gripper_cmd is not None
            and abs(closure - self._last_gripper_cmd) < self._gripper_delta_threshold
            and now_ns - self._last_gripper_call_ns < self._gripper_min_interval_ns
        ):
            return
        if not self.gripper_client.service_is_ready():
            if not self.gripper_client.wait_for_service(timeout_sec=0.0):
                return
        request = SetClosure.Request()
        request.closure = closure
        request.duration_sec = self.gripper_duration_sec
        request.speed_ratio = self.gripper_speed_ratio
        try:
            self.gripper_client.call_async(request)
            self._last_gripper_cmd = closure
            self._last_gripper_call_ns = now_ns
        except Exception as exc:  # pragma: no cover - best effort
            self.get_logger().warn(f"SoftHand call failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
