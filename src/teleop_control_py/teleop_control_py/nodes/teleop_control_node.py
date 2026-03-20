#!/usr/bin/env python3
"""遥操作主节点：只负责参数装配、策略实例化和高频控制循环。"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from ..hardware.gripper_controllers import QbSoftHandController, RobotiqController
from ..hardware.input_handlers import JoyInputHandler, MediaPipeInputHandler
from ..hardware.servo_pose_follower import ServoPoseFollower
from ..utils.transform_utils import apply_velocity_limits


class TeleopControlNode(Node):
    """主调度节点，遵循 DIP：只依赖抽象协议，不依赖具体设备细节。"""

    def __init__(self) -> None:
        super().__init__("teleop_control_node")
        self._declare_parameters()

        self._input_type = str(self.get_parameter("input_type").value).strip().lower()
        self._gripper_type = str(self.get_parameter("gripper_type").value).strip().lower()
        self._control_hz = max(1.0, float(self.get_parameter("control_hz").value))

        linear_vel_param = "max_linear_vel"
        angular_vel_param = "max_angular_vel"
        linear_accel_param = "max_linear_accel"
        angular_accel_param = "max_angular_accel"
        if self._input_type == "mediapipe":
            linear_vel_param = "mediapipe_max_linear_vel"
            angular_vel_param = "mediapipe_max_angular_vel"
            linear_accel_param = "mediapipe_max_linear_accel"
            angular_accel_param = "mediapipe_max_angular_accel"

        self._max_velocity = np.array(
            [
                float(self.get_parameter(linear_vel_param).value),
                float(self.get_parameter(linear_vel_param).value),
                float(self.get_parameter(linear_vel_param).value),
                float(self.get_parameter(angular_vel_param).value),
                float(self.get_parameter(angular_vel_param).value),
                float(self.get_parameter(angular_vel_param).value),
            ],
            dtype=np.float64,
        )
        self._max_acceleration = np.array(
            [
                float(self.get_parameter(linear_accel_param).value),
                float(self.get_parameter(linear_accel_param).value),
                float(self.get_parameter(linear_accel_param).value),
                float(self.get_parameter(angular_accel_param).value),
                float(self.get_parameter(angular_accel_param).value),
                float(self.get_parameter(angular_accel_param).value),
            ],
            dtype=np.float64,
        )
        self._zero_return_accel_scale = max(0.01, float(self.get_parameter("zero_return_accel_scale").value))
        self._last_twist_vec = np.zeros(6, dtype=np.float64)
        self._last_loop_time = time.monotonic()
        self._home_zone_active = False
        self._home_zone_cancel_inflight = False
        self._last_home_zone_cancel_monotonic = 0.0

        self.input_handler = self._build_input_handler(self._input_type)
        self.gripper_ctrl = self._build_gripper_controller(self._gripper_type)
        self.arm_ctrl = ServoPoseFollower(self)
        self._home_zone_active_sub = self.create_subscription(
            Bool,
            "/commander/home_zone_active",
            self._on_home_zone_active,
            10,
        )
        self._home_zone_cancel_client = self.create_client(Trigger, "/commander/cancel_home_zone")

        self._timer = self.create_timer(1.0 / self._control_hz, self._control_loop)
        self.get_logger().info(
            f"TeleopControlNode ready. input_type={self._input_type}, gripper_type={self._gripper_type}, "
            f"control_hz={self._control_hz:.1f}, zero_return_accel_scale={self._zero_return_accel_scale:.2f}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_type", "joy")
        self.declare_parameter("gripper_type", "robotiq")
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("target_frame_id", "base")
        self.declare_parameter("servo_twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("tool_pose_topic", "/tcp_pose_broadcaster/pose")
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("auto_switch_controllers", True)
        self.declare_parameter("controller_manager_ns", "/controller_manager")
        self.declare_parameter("teleop_controller", "forward_position_controller")
        self.declare_parameter("trajectory_controller", "scaled_joint_trajectory_controller")
        self.declare_parameter("startup_retry_period_sec", 1.0)
        self.declare_parameter("max_linear_vel", 1.5)
        self.declare_parameter("max_angular_vel", 3.0)
        self.declare_parameter("max_linear_accel", 4.0)
        self.declare_parameter("max_angular_accel", 8.0)
        self.declare_parameter("zero_return_accel_scale", 0.35)

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("input_watchdog_timeout_sec", 0.2)
        self.declare_parameter("joy_deadzone", 0.05)
        self.declare_parameter("joy_curve", "linear")
        self.declare_parameter("joy_deadman_enabled", False)
        self.declare_parameter("deadman_button", -1)
        self.declare_parameter("deadman_axis", 4)
        self.declare_parameter("deadman_axis_threshold", 0.5)
        self.declare_parameter("linear_x_axis", 0)
        self.declare_parameter("linear_y_axis", 1)
        self.declare_parameter("linear_z_axis", -1)
        self.declare_parameter("linear_z_up_axis", 4)
        self.declare_parameter("linear_z_down_axis", 5)
        self.declare_parameter("linear_z_up_button", 1)
        self.declare_parameter("linear_z_down_button", 0)
        self.declare_parameter("angular_x_axis", 3)
        self.declare_parameter("angular_y_axis", 2)
        self.declare_parameter("angular_z_axis", -1)
        self.declare_parameter("angular_z_positive_button", 1)
        self.declare_parameter("angular_z_negative_button", 0)
        self.declare_parameter("linear_axis_sign", [-1.0, -1.0, 1.0])
        self.declare_parameter("angular_axis_sign", [-1.0, 1.0, 1.0])
        self.declare_parameter("gripper_close_button", 5)
        self.declare_parameter("gripper_open_button", 4)
        self.declare_parameter("gripper_axis", -1)
        self.declare_parameter("gripper_axis_inverted", False)

        self.declare_parameter("mediapipe_input_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("mediapipe_topic", "")
        self.declare_parameter("mediapipe_depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("mediapipe_camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("mediapipe_motion_mode", "target_pose")
        self.declare_parameter("mediapipe_deadzone", 0.02)
        self.declare_parameter("mediapipe_linear_scale", 2.8)
        self.declare_parameter("mediapipe_angular_scale", 1.0)
        self.declare_parameter("mediapipe_position_linear_gain", 7.0)
        self.declare_parameter("mediapipe_position_angular_gain", 5.0)
        self.declare_parameter("mediapipe_max_linear_vel", 2.8)
        self.declare_parameter("mediapipe_max_angular_vel", 5.0)
        self.declare_parameter("mediapipe_max_linear_accel", 10.0)
        self.declare_parameter("mediapipe_max_angular_accel", 18.0)
        self.declare_parameter("mediapipe_linear_axis_mapping", [0, 1, 2])
        self.declare_parameter("mediapipe_angular_axis_mapping", [0, 1, 2])
        self.declare_parameter("mediapipe_linear_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("mediapipe_angular_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("mediapipe_hand_position_source", "depth")
        self.declare_parameter("mediapipe_orientation_mode", "lock")
        self.declare_parameter("mediapipe_orientation_axis_mapping", [0, 1, 2])
        self.declare_parameter("mediapipe_orientation_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("mediapipe_depth_min_m", 0.1)
        self.declare_parameter("mediapipe_depth_max_m", 2.0)
        self.declare_parameter("mediapipe_depth_unit_scale", 0.001)
        self.declare_parameter("mediapipe_smoothing_alpha", 0.45)
        self.declare_parameter("mediapipe_gripper_open_dist_px", 100.0)
        self.declare_parameter("mediapipe_gripper_close_dist_px", 20.0)
        self.declare_parameter("mediapipe_gripper_open_dist_m", 0.10)
        self.declare_parameter("mediapipe_gripper_close_dist_m", 0.03)
        self.declare_parameter("mediapipe_gripper_metric_hold_sec", 0.25)
        self.declare_parameter("mediapipe_gripper_requires_deadman", True)
        self.declare_parameter("mediapipe_deadman_filter_enabled", True)
        self.declare_parameter("mediapipe_deadman_engage_confirm_sec", 0.10)
        self.declare_parameter("mediapipe_deadman_release_confirm_sec", 0.03)
        self.declare_parameter("mediapipe_space_deadman_backend", "pynput")
        self.declare_parameter("mediapipe_space_deadman_hold_sec", 0.3)
        self.declare_parameter("mediapipe_show_debug_window", True)

        self.declare_parameter("gripper_cmd_topic", "/gripper/cmd")
        self.declare_parameter("gripper_command_delta", 0.01)
        self.declare_parameter("gripper_quantization_levels", 10)
        self.declare_parameter("robotiq_command_interface", "position_action")
        self.declare_parameter("robotiq_confidence_topic", "/robotiq_2f_gripper/confidence_command")
        self.declare_parameter("robotiq_binary_topic", "/robotiq_2f_gripper/binary_command")
        self.declare_parameter("robotiq_action_name", "/robotiq_2f_gripper_action")
        self.declare_parameter("robotiq_binary_threshold", 0.5)
        self.declare_parameter("robotiq_open_ratio", 0.9)
        self.declare_parameter("robotiq_max_open_position_m", 0.142)
        self.declare_parameter("robotiq_target_speed", 1.0)
        self.declare_parameter("robotiq_target_force", 0.5)
        self.declare_parameter("qbsofthand_service_name", "/qbsofthand_control_node/set_closure")
        self.declare_parameter("qbsofthand_duration_sec", 0.3)
        self.declare_parameter("qbsofthand_speed_ratio", 1.0)

    def _build_input_handler(self, input_type: str):
        strategies = {
            "joy": JoyInputHandler,
            "mediapipe": MediaPipeInputHandler,
        }
        handler_cls = strategies.get(input_type)
        if handler_cls is None:
            self.get_logger().warn(f"未知 input_type '{input_type}'，回退到 joy。")
            handler_cls = JoyInputHandler
        return handler_cls(self)

    def _build_gripper_controller(self, gripper_type: str):
        strategies = {
            "robotiq": RobotiqController,
            "qbsofthand": QbSoftHandController,
        }
        controller_cls = strategies.get(gripper_type)
        if controller_cls is None:
            self.get_logger().warn(f"未知 gripper_type '{gripper_type}'，回退到 robotiq。")
            controller_cls = RobotiqController
        return controller_cls(self)

    def _twist_to_vector(self, twist: Twist) -> np.ndarray:
        return np.array(
            [
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ],
            dtype=np.float64,
        )

    def _vector_to_twist(self, values: np.ndarray) -> Twist:
        twist = Twist()
        twist.linear.x = float(values[0])
        twist.linear.y = float(values[1])
        twist.linear.z = float(values[2])
        twist.angular.x = float(values[3])
        twist.angular.y = float(values[4])
        twist.angular.z = float(values[5])
        return twist

    def _on_home_zone_active(self, msg: Bool) -> None:
        active = bool(msg.data)
        if active == self._home_zone_active:
            return
        self._home_zone_active = active
        if active:
            self._last_twist_vec = np.zeros(6, dtype=np.float64)
            self.get_logger().info("Home Zone active. Teleop output paused until Home Zone finishes or is canceled.")
        else:
            self.get_logger().info("Home Zone inactive. Teleop output resumed.")

    def _on_home_zone_cancel_done(self, future) -> None:
        self._home_zone_cancel_inflight = False
        try:
            future.result()
        except Exception:
            pass

    def cancel_gripper_motion(self) -> None:
        try:
            self.gripper_ctrl.cancel_motion()
        except Exception:
            pass

    def _maybe_cancel_home_zone_for_operator_input(self, target_vec: np.ndarray) -> None:
        if not self._home_zone_active:
            return
        if float(np.max(np.abs(target_vec))) <= 1e-3:
            return
        if self._home_zone_cancel_inflight:
            return
        now = time.monotonic()
        if (now - self._last_home_zone_cancel_monotonic) < 0.5:
            return
        if not self._home_zone_cancel_client.wait_for_service(timeout_sec=0.0):
            return
        self._last_home_zone_cancel_monotonic = now
        self._home_zone_cancel_inflight = True
        future = self._home_zone_cancel_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_home_zone_cancel_done)

    def _control_loop(self) -> None:
        twist, gripper_val = self.input_handler.get_command()
        now = time.monotonic()
        dt = max(1e-4, now - self._last_loop_time)
        self._last_loop_time = now

        target_vec = self._twist_to_vector(twist)
        if self._home_zone_active:
            self._last_twist_vec = np.zeros(6, dtype=np.float64)
            self._maybe_cancel_home_zone_for_operator_input(target_vec)
            return
        effective_acceleration = self._max_acceleration.copy()
        zero_axes = np.isclose(target_vec, 0.0, atol=1e-6)
        effective_acceleration[zero_axes] *= self._zero_return_accel_scale
        limited_vec = apply_velocity_limits(
            target=target_vec,
            previous=self._last_twist_vec,
            max_velocity=self._max_velocity,
            max_acceleration=effective_acceleration,
            dt=dt,
        )
        self._last_twist_vec = limited_vec

        limited_twist = self._vector_to_twist(limited_vec)
        self.arm_ctrl.send_twist(limited_twist)
        self.gripper_ctrl.set_gripper(gripper_val)

    def destroy_node(self) -> bool:  # type: ignore[override]
        try:
            self.input_handler.stop()
        except Exception:
            pass
        try:
            self.arm_ctrl.stop()
        except Exception:
            pass
        try:
            self.gripper_ctrl.stop()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
