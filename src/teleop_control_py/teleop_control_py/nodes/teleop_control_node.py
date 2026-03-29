#!/usr/bin/env python3
"""遥操作主节点：只负责参数装配、策略实例化和高频控制循环。"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from ..core import ControlCoordinator, ControlSource
from ..device_manager import (
    ControllerGripperBackend,
    DEFAULT_ROBOT_PROFILE_NAME,
    InputHandlerBackend,
    ServoArmBackend,
    default_robot_profiles_path,
    load_robot_profile,
)
from ..hardware.control.gripper_controllers import QbSoftHandController, RobotiqController
from ..hardware.control.servo_pose_follower import ServoPoseFollower
from ..hardware.input import JoyInputHandler, MediaPipeInputHandler, Quest3InputHandler
from ..utils.transform_utils import apply_velocity_limits


class TeleopControlNode(Node):
    """主调度节点，逐步迁移到新的核心骨架。"""

    def __init__(self) -> None:
        super().__init__("teleop_control_node")
        self.declare_parameter("robot_profile", DEFAULT_ROBOT_PROFILE_NAME)
        self.declare_parameter("robot_profiles_file", str(default_robot_profiles_path()))
        self._robot_profile_name = str(self.get_parameter("robot_profile").value).strip() or DEFAULT_ROBOT_PROFILE_NAME
        self._robot_profiles_file = str(self.get_parameter("robot_profiles_file").value).strip()
        self._robot_profile = load_robot_profile(self._robot_profile_name, self._robot_profiles_file)

        self._declare_parameters(self._robot_profile)

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
        elif self._input_type == "quest3":
            linear_vel_param = "quest3_max_linear_vel"
            angular_vel_param = "quest3_max_angular_vel"
            linear_accel_param = "quest3_max_linear_accel"
            angular_accel_param = "quest3_max_angular_accel"

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
        self._linear_z_zero_return_accel_scale = max(
            0.01, float(self.get_parameter("linear_z_zero_return_accel_scale").value)
        )
        self._last_twist_vec = np.zeros(6, dtype=np.float64)
        self._last_loop_time = time.monotonic()
        self._home_zone_active = False

        self.input_handler = self._build_input_handler(self._input_type)
        self.gripper_ctrl = self._build_gripper_controller(self._gripper_type)
        self.arm_ctrl = ServoPoseFollower(self)

        self._input_backend = InputHandlerBackend(self.input_handler, source=ControlSource.TELEOP)
        self._gripper_backend = ControllerGripperBackend(self.gripper_ctrl)
        self._arm_backend = ServoArmBackend(self.arm_ctrl)
        self._control_coordinator = ControlCoordinator(
            self._arm_backend,
            self._gripper_backend,
            active_source=ControlSource.TELEOP,
            logger=self.get_logger(),
        )
        self._control_coordinator.notify_teleop_started()

        self._home_zone_active_sub = self.create_subscription(
            Bool,
            "/commander/home_zone_active",
            self._on_home_zone_active,
            10,
        )

        self._timer = self.create_timer(1.0 / self._control_hz, self._control_loop)
        self.get_logger().info(
            f"TeleopControlNode ready. robot_profile={self._robot_profile.name}, input_type={self._input_type}, "
            f"gripper_type={self._gripper_type}, control_hz={self._control_hz:.1f}, "
            f"zero_return_accel_scale={self._zero_return_accel_scale:.2f}, "
            f"linear_z_zero_return_accel_scale={self._linear_z_zero_return_accel_scale:.2f}"
        )

    def _declare_parameters(self, robot_profile) -> None:
        grippers = robot_profile.grippers
        robotiq = grippers.robotiq
        qbsofthand = grippers.qbsofthand
        self.declare_parameter("input_type", "joy")
        self.declare_parameter("gripper_type", grippers.default_type)
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("target_frame_id", robot_profile.target_frame_id)
        self.declare_parameter("servo_twist_topic", robot_profile.topics.servo_twist)
        self.declare_parameter("tool_pose_topic", robot_profile.topics.tool_pose)
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", robot_profile.services.start_servo)
        self.declare_parameter("auto_switch_controllers", True)
        self.declare_parameter("controller_manager_ns", robot_profile.services.controller_manager_ns)
        self.declare_parameter("teleop_controller", robot_profile.controllers.teleop)
        self.declare_parameter("trajectory_controller", robot_profile.controllers.trajectory)
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
        self.declare_parameter("linear_z_scale", 1.0)
        self.declare_parameter("linear_z_trigger_deadzone", 0.05)
        self.declare_parameter("linear_z_trigger_release_deadzone", 0.05)
        self.declare_parameter("linear_z_trigger_snap_release_threshold", 0.80)
        self.declare_parameter("linear_z_trigger_snap_release_drop", 0.08)
        self.declare_parameter("linear_z_zero_return_accel_scale", 1.0)
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

        self.declare_parameter("quest3_connected_topic", "/quest3/connected")
        self.declare_parameter("quest3_left_pose_topic", "/quest3/left_controller/pose")
        self.declare_parameter("quest3_right_pose_topic", "/quest3/right_controller/pose")
        self.declare_parameter("quest3_joy_topic", "/quest3/input/joy")
        self.declare_parameter("quest3_require_connected", True)
        self.declare_parameter("quest3_pose_timeout_sec", 0.3)
        self.declare_parameter("quest3_active_hand", "left")
        self.declare_parameter("quest3_motion_mode", "target_pose")
        self.declare_parameter("quest3_deadzone", 0.02)
        self.declare_parameter("quest3_linear_scale", 1.0)
        self.declare_parameter("quest3_angular_scale", 1.0)
        self.declare_parameter("quest3_position_linear_gain", 7.0)
        self.declare_parameter("quest3_position_angular_gain", 5.0)
        self.declare_parameter("quest3_max_linear_vel", 2.8)
        self.declare_parameter("quest3_max_angular_vel", 5.0)
        self.declare_parameter("quest3_max_linear_accel", 10.0)
        self.declare_parameter("quest3_max_angular_accel", 18.0)
        self.declare_parameter("quest3_linear_axis_mapping", [0, 1, 2])
        self.declare_parameter("quest3_angular_axis_mapping", [0, 1, 2])
        self.declare_parameter("quest3_linear_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("quest3_angular_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("quest3_orientation_mode", "hand_relative")
        self.declare_parameter("quest3_orientation_axis_mapping", [0, 1, 2])
        self.declare_parameter("quest3_orientation_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("quest3_enable_input_smoothing", False)
        self.declare_parameter("quest3_smoothing_alpha", 0.45)
        self.declare_parameter("quest3_frame_reset_enabled", True)
        self.declare_parameter("quest3_frame_reset_scope", "active_hand")
        self.declare_parameter("quest3_frame_reset_hold_sec", 0.75)
        self.declare_parameter("quest3_frame_reset_rotate_position", False)
        self.declare_parameter("quest3_left_frame_reset_buttons", [4, 5])
        self.declare_parameter("quest3_right_frame_reset_buttons", [10, 11])
        self.declare_parameter("quest3_clutch_filter_enabled", True)
        self.declare_parameter("quest3_clutch_engage_confirm_sec", 0.02)
        self.declare_parameter("quest3_clutch_release_confirm_sec", 0.02)
        self.declare_parameter("quest3_clutch_axis_threshold", 0.15)
        self.declare_parameter("quest3_left_clutch_axis", 1)
        self.declare_parameter("quest3_right_clutch_axis", 7)
        self.declare_parameter("quest3_left_clutch_button", 1)
        self.declare_parameter("quest3_right_clutch_button", 7)
        self.declare_parameter("quest3_left_trigger_axis", 0)
        self.declare_parameter("quest3_right_trigger_axis", 6)
        self.declare_parameter("quest3_left_trigger_button", 0)
        self.declare_parameter("quest3_right_trigger_button", 6)
        self.declare_parameter("quest3_gripper_requires_clutch", True)
        self.declare_parameter("quest3_gripper_axis_inverted", False)

        self.declare_parameter("mediapipe_input_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("mediapipe_topic", "")
        self.declare_parameter("mediapipe_depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("mediapipe_camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("mediapipe_camera_driver", "realsense")
        self.declare_parameter("mediapipe_camera_serial_number", "")
        self.declare_parameter("mediapipe_enable_depth", False)
        self.declare_parameter("mediapipe_use_sdk_camera", True)
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

        self.declare_parameter("gripper_cmd_topic", grippers.command_topic)
        self.declare_parameter("gripper_command_delta", grippers.command_delta)
        self.declare_parameter("gripper_quantization_levels", grippers.quantization_levels)
        self.declare_parameter("robotiq_command_interface", robotiq.command_interface)
        self.declare_parameter("robotiq_confidence_topic", robotiq.confidence_topic)
        self.declare_parameter("robotiq_binary_topic", robotiq.binary_topic)
        self.declare_parameter("robotiq_action_name", robotiq.action_name)
        self.declare_parameter("robotiq_binary_threshold", robotiq.binary_threshold)
        self.declare_parameter("robotiq_open_ratio", robotiq.open_ratio)
        self.declare_parameter("robotiq_max_open_position_m", robotiq.max_open_position_m)
        self.declare_parameter("robotiq_target_speed", robotiq.target_speed)
        self.declare_parameter("robotiq_target_force", robotiq.target_force)
        self.declare_parameter("qbsofthand_service_name", qbsofthand.service_name)
        self.declare_parameter("qbsofthand_duration_sec", qbsofthand.duration_sec)
        self.declare_parameter("qbsofthand_speed_ratio", qbsofthand.speed_ratio)

    def _build_input_handler(self, input_type: str):
        strategies = {
            "joy": JoyInputHandler,
            "mediapipe": MediaPipeInputHandler,
            "quest3": Quest3InputHandler,
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

    def _on_home_zone_active(self, msg: Bool) -> None:
        active = bool(msg.data)
        if active == self._home_zone_active:
            return

        self._home_zone_active = active
        self._last_twist_vec = np.zeros(6, dtype=np.float64)
        self._control_coordinator.notify_home_zone(active)

        if active:
            self.get_logger().info("Home Zone active. Teleop output paused until Home Zone finishes.")
        else:
            self.get_logger().info("Home Zone inactive. Teleop output resumed.")

    def cancel_gripper_motion(self) -> None:
        cancel_fn = getattr(self._gripper_backend, "cancel_motion", None)
        if callable(cancel_fn):
            try:
                cancel_fn()
            except Exception:
                pass

    def _control_loop(self) -> None:
        command = self._input_backend.get_action_command()
        now = time.monotonic()
        dt = max(1e-4, now - self._last_loop_time)
        self._last_loop_time = now

        target_vec = command.twist_vector()
        if self._home_zone_active:
            self._last_twist_vec = np.zeros(6, dtype=np.float64)
            return

        effective_acceleration = self._max_acceleration.copy()
        zero_axes = np.isclose(target_vec, 0.0, atol=1e-6)
        effective_acceleration[zero_axes] *= self._zero_return_accel_scale
        if zero_axes[2]:
            effective_acceleration[2] *= self._linear_z_zero_return_accel_scale
        limited_vec = apply_velocity_limits(
            target=target_vec,
            previous=self._last_twist_vec,
            max_velocity=self._max_velocity,
            max_acceleration=effective_acceleration,
            dt=dt,
        )
        self._last_twist_vec = limited_vec

        dispatch_result = self._control_coordinator.dispatch(command.with_twist_vector(limited_vec))
        if not dispatch_result.accepted:
            self.get_logger().debug(f"Teleop action dropped by mux: {dispatch_result.reason}")

    def destroy_node(self) -> bool:  # type: ignore[override]
        for backend in (self._input_backend, self._arm_backend, self._gripper_backend):
            try:
                backend.stop()
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
