#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
	DeclareLaunchArgument,
	GroupAction,
	IncludeLaunchDescription,
	LogInfo,
	OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import SetRemap


def _default_python_executable() -> str:
	for prefix_var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
		prefix = os.environ.get(prefix_var)
		if not prefix:
			continue
		candidate = os.path.join(prefix, "bin", "python3")
		if os.path.exists(candidate):
			return candidate
	return "python3"


def _load_teleop_params(params_file: str) -> Dict[str, Any]:
	try:
		import yaml  # type: ignore
	except Exception:
		return {}

	try:
		with open(params_file, "r", encoding="utf-8") as f:
			data = yaml.safe_load(f)
	except Exception:
		return {}

	if not isinstance(data, dict):
		return {}

	for key in ("teleop_control_node", "/teleop_control_node"):
		block = data.get(key)
		if not isinstance(block, dict):
			continue
		params = block.get("ros__parameters")
		if not isinstance(params, dict):
			continue
		return params
	return {}


def _coerce_input_type(value: str) -> str:
	normalized = value.strip().lower()
	if normalized == "xbox":
		return "joy"
	if normalized == "hand":
		return "mediapipe"
	if normalized in ("joy", "mediapipe"):
		return normalized
	return ""


def _coerce_gripper_type(value: str) -> str:
	normalized = value.strip().lower()
	if normalized == "auto":
		return "qbsofthand"
	if normalized in ("qbsofthand", "robotiq"):
		return normalized
	return ""


def _resolve_input_type(context) -> str:
	params_file = LaunchConfiguration("params_file").perform(context)
	resolved = _coerce_input_type(LaunchConfiguration("input_type").perform(context))
	if resolved:
		return resolved

	resolved = _coerce_input_type(LaunchConfiguration("control_mode").perform(context))
	if resolved:
		return resolved

	params = _load_teleop_params(params_file)
	resolved = _coerce_input_type(str(params.get("input_type", "")))
	if resolved:
		return resolved

	resolved = _coerce_input_type(str(params.get("control_mode", "")))
	if resolved:
		return resolved

	return "joy"


def _resolve_gripper_type(context) -> str:
	params_file = LaunchConfiguration("params_file").perform(context)
	resolved = _coerce_gripper_type(LaunchConfiguration("gripper_type").perform(context))
	if resolved:
		return resolved

	resolved = _coerce_gripper_type(LaunchConfiguration("end_effector").perform(context))
	if resolved:
		return resolved

	params = _load_teleop_params(params_file)
	resolved = _coerce_gripper_type(str(params.get("gripper_type", "")))
	if resolved:
		return resolved

	resolved = _coerce_gripper_type(str(params.get("end_effector", "")))
	if resolved:
		return resolved

	return "robotiq"


def _teleop_node_enabled(context) -> bool:
	value = LaunchConfiguration("launch_teleop_node").perform(context).strip().lower()
	return value in ("1", "true", "yes", "on")


def _maybe_include_end_effector_driver(context, *args, **kwargs):
	ee = _resolve_gripper_type(context)
	actions = [LogInfo(msg=f"[control_system] resolved gripper_type: {ee}")]

	if ee == "robotiq":
		robotiq_share = get_package_share_directory("robotiq_2f_gripper_hardware")
		robotiq_launch_path = os.path.join(robotiq_share, "launch", "robotiq_2f_gripper_launch.py")
		actions.append(
			IncludeLaunchDescription(
				PythonLaunchDescriptionSource(robotiq_launch_path),
				launch_arguments={
					"namespace": LaunchConfiguration("robotiq_namespace"),
					"serial_port": LaunchConfiguration("robotiq_serial_port"),
					"fake_hardware": LaunchConfiguration("robotiq_fake_hardware"),
					"config_file": LaunchConfiguration("robotiq_config_file"),
					"rviz2": LaunchConfiguration("robotiq_rviz2"),
				}.items(),
			)
		)
		actions.append(LogInfo(msg="[control_system] Included robotiq_2f_gripper_hardware/robotiq_2f_gripper_launch.py"))
		return actions

	# Default (qbsofthand): keep previous behavior to avoid breaking a known-good setup.
	actions.append(
		Node(
			package="qbsofthand_control",
			executable="qbsofthand_control_node",
			name="qbsofthand_control_node",
			output="screen",
		)
	)
	return actions


def _maybe_include_joy_driver(context, *args, **kwargs):
	input_type = _resolve_input_type(context)
	teleop_enabled = _teleop_node_enabled(context)

	actions = [LogInfo(msg=f"[control_system] resolved input_type: {input_type}, launch_teleop_node={teleop_enabled}")]
	if not teleop_enabled:
		actions.append(LogInfo(msg="[control_system] Skip joy driver because teleop node is disabled"))
		return actions
	if input_type != "joy":
		return actions

	joy_share = get_package_share_directory("multi_joy_driver")
	joy_launch_path = os.path.join(joy_share, "launch", "joy_driver.launch.py")
	actions.append(
		IncludeLaunchDescription(
			PythonLaunchDescriptionSource(joy_launch_path),
			launch_arguments={
				"python_executable": LaunchConfiguration("python_executable"),
				"profile": LaunchConfiguration("joy_profile"),
				"device_path": LaunchConfiguration("joy_device_path"),
			}.items(),
		)
	)
	actions.append(LogInfo(msg="[control_system] Included multi_joy_driver/joy_driver.launch.py"))
	return actions


def _maybe_include_moveit_servo(context, *args, **kwargs):
	enable_moveit_raw = LaunchConfiguration("enable_moveit").perform(context).strip().lower()
	enable_moveit = enable_moveit_raw in ("1", "true", "yes", "on")

	actions = [
		LogInfo(
			msg=(
				f"[control_system] enable_moveit={enable_moveit_raw}"
			)
		)
	]
	if not enable_moveit:
		return actions

	moveit_share = get_package_share_directory("ur_moveit_config")
	moveit_launch_path = os.path.join(moveit_share, "launch", "ur_moveit.launch.py")
	actions.append(
		GroupAction(
			actions=[
				# Do NOT modify upstream UR packages. Instead, remap Servo's private input topic
				# so our teleop publisher `/servo_node/delta_twist_cmds` is always consumed.
				SetRemap(src="~/delta_twist_cmds", dst="/servo_node/delta_twist_cmds"),
				IncludeLaunchDescription(
					PythonLaunchDescriptionSource(moveit_launch_path),
					launch_arguments={
						"ur_type": LaunchConfiguration("ur_type"),
						"launch_rviz": LaunchConfiguration("launch_moveit_rviz"),
						"launch_servo": LaunchConfiguration("launch_servo"),
						"use_sim_time": "false",
					}.items(),
				),
			]
		)
	)
	actions.append(LogInfo(msg="[control_system] Included ur_moveit_config/ur_moveit.launch.py"))
	return actions


def _maybe_include_realsense(context, *args, **kwargs):
	input_type = _resolve_input_type(context)

	enable_camera_raw = LaunchConfiguration("enable_camera").perform(context).strip().lower()
	enable_camera = enable_camera_raw in ("1", "true", "yes", "on")

	actions = [
		LogInfo(
			msg=(
				f"[control_system] enable_camera={enable_camera_raw} "
				f"input_type={input_type}"
			)
		)
	]

	if input_type != "mediapipe":
		actions.append(LogInfo(msg="[control_system] Skip RealSense because input_type is not mediapipe"))
		return actions

	if not enable_camera:
		actions.append(LogInfo(msg="[control_system] RealSense disabled by launch arg"))
		return actions

	realsense_share = get_package_share_directory("realsense2_camera")
	realsense_launch_path = os.path.join(realsense_share, "launch", "rs_launch.py")
	actions.append(
		IncludeLaunchDescription(
			PythonLaunchDescriptionSource(realsense_launch_path),
		)
	)
	actions.append(LogInfo(msg="[control_system] Included realsense2_camera/rs_launch.py"))
	return actions


def _collector_end_effector_type(gripper_type: str) -> str:
	return "qbsofthand" if gripper_type == "qbsofthand" else "robotic_gripper"


def _maybe_include_data_collector(context, *args, **kwargs):
	enable_raw = LaunchConfiguration("enable_data_collector").perform(context).strip().lower()
	enable = enable_raw in ("1", "true", "yes", "on")
	actions = [LogInfo(msg=f"[control_system] enable_data_collector={enable_raw}")]
	if not enable:
		return actions

	gripper_type = _resolve_gripper_type(context)
	collector_ee = _collector_end_effector_type(gripper_type)
	actions.append(
		Node(
			package="teleop_control_py",
			executable="data_collector_node",
			name="data_collector",
			output="screen",
			parameters=[LaunchConfiguration("data_collector_params_file")],
			arguments=[
				"--ros-args",
				"-p",
				f"end_effector_type:={collector_ee}",
			],
		)
	)
	actions.append(
		LogInfo(
			msg=(
				"[control_system] Included teleop_control_py/data_collector_node "
				f"with end_effector_type={collector_ee}"
			)
		)
	)
	return actions


def generate_launch_description() -> LaunchDescription:
	teleop_share = get_package_share_directory("teleop_control_py")
	default_params = os.path.join(teleop_share, "config", "teleop_params.yaml")

	params_file_arg = DeclareLaunchArgument(
		"params_file",
		default_value=default_params,
		description="Path to teleop parameter file",
	)
	python_executable_arg = DeclareLaunchArgument(
		"python_executable",
		default_value=_default_python_executable(),
		description="Python executable used to run python-based nodes",
	)
	control_mode_arg = DeclareLaunchArgument(
		"control_mode",
		default_value="",
		description="Deprecated alias for input_type (hand->mediapipe, xbox->joy).",
	)
	end_effector_arg = DeclareLaunchArgument(
		"end_effector",
		default_value="",
		description="Deprecated alias for gripper_type (auto->qbsofthand).",
	)
	input_type_arg = DeclareLaunchArgument(
		"input_type",
		default_value="",
		description="Optional input backend override (joy|mediapipe). Empty means read from params_file.",
	)
	gripper_type_arg = DeclareLaunchArgument(
		"gripper_type",
		default_value="",
		description="Optional gripper backend override (robotiq|qbsofthand). Empty means read from params_file.",
	)
	joy_profile_arg = DeclareLaunchArgument(
		"joy_profile",
		default_value="auto",
		description="Joystick profile passed to multi_joy_driver (auto|xbox|ps5|generic).",
	)
	joy_device_path_arg = DeclareLaunchArgument(
		"joy_device_path",
		default_value="",
		description="Optional joystick event device path for multi_joy_driver.",
	)
	mediapipe_input_topic_arg = DeclareLaunchArgument(
		"mediapipe_input_topic",
		default_value="",
		description="Optional MediaPipe image topic override.",
	)

	robotiq_namespace_arg = DeclareLaunchArgument(
		"robotiq_namespace",
		default_value="",
		description="Namespace for Robotiq gripper (usually empty)",
	)
	robotiq_serial_port_arg = DeclareLaunchArgument(
		"robotiq_serial_port",
		default_value="/dev/robotiq_gripper",
		description="Serial port for Robotiq gripper (recommended to use a udev symlink, e.g. /dev/robotiq_gripper)",
	)
	robotiq_fake_hw_arg = DeclareLaunchArgument(
		"robotiq_fake_hardware",
		default_value="False",
		description="Use fake Robotiq hardware",
	)
	robotiq_config_file_arg = DeclareLaunchArgument(
		"robotiq_config_file",
		default_value="",
		description="Optional Robotiq config YAML path",
	)
	robotiq_rviz2_arg = DeclareLaunchArgument(
		"robotiq_rviz2",
		default_value="False",
		description="Launch RViz2 for Robotiq visualization",
	)

	ur_type_arg = DeclareLaunchArgument(
		"ur_type",
		default_value="ur5",
		description="UR robot type (ur5, ur10, etc.)",
	)
	robot_ip_arg = DeclareLaunchArgument(
		"robot_ip",
		default_value="192.168.1.211",
		description="UR robot IP address",
	)
	reverse_ip_arg = DeclareLaunchArgument(
		"reverse_ip",
		default_value="192.168.1.10",
		description="Reverse connection IP address",
	)
	launch_rviz_arg = DeclareLaunchArgument(
		"launch_rviz",
		default_value="false",
		description="Launch RViz for visualization",
	)
	launch_moveit_rviz_arg = DeclareLaunchArgument(
		"launch_moveit_rviz",
		default_value="false",
		description="Launch MoveIt RViz (ur_moveit_config)",
	)
	launch_servo_arg = DeclareLaunchArgument(
		"launch_servo",
		default_value="true",
		description="Launch MoveIt Servo (ur_moveit_config)",
	)
	initial_joint_controller_arg = DeclareLaunchArgument(
		"initial_joint_controller",
		default_value="forward_position_controller",
		description="Initial UR joint controller for teleop-first bringup.",
	)
	enable_moveit_arg = DeclareLaunchArgument(
		"enable_moveit",
		default_value="true",
		description="Enable MoveIt bringup (move_group + optional servo).",
	)
	enable_camera_arg = DeclareLaunchArgument(
		"enable_camera",
		default_value="true",
		description="Enable RealSense camera (effective only when input_type=mediapipe)",
	)
	enable_data_collector_arg = DeclareLaunchArgument(
		"enable_data_collector",
		default_value="false",
		description="Enable data_collector_node bringup as part of the full control system.",
	)
	launch_teleop_node_arg = DeclareLaunchArgument(
		"launch_teleop_node",
		default_value="true",
		description="Launch teleop_control_node input/output loop.",
	)
	data_collector_params_file_arg = DeclareLaunchArgument(
		"data_collector_params_file",
		default_value=os.path.join(teleop_share, "config", "data_collector_params.yaml"),
		description="Path to data collector parameter file",
	)

	ur_driver_share = get_package_share_directory("ur_robot_driver")
	ur_launch = IncludeLaunchDescription(
		PythonLaunchDescriptionSource(os.path.join(ur_driver_share, "launch", "ur_control.launch.py")),
		launch_arguments={
			"ur_type": LaunchConfiguration("ur_type"),
			"robot_ip": LaunchConfiguration("robot_ip"),
			"reverse_ip": LaunchConfiguration("reverse_ip"),
			"initial_joint_controller": LaunchConfiguration("initial_joint_controller"),
			"launch_rviz": LaunchConfiguration("launch_rviz"),
		}.items(),
	)

	teleop_launch = IncludeLaunchDescription(
		PythonLaunchDescriptionSource(os.path.join(teleop_share, "launch", "teleop_control.launch.py")),
		launch_arguments={
			"params_file": LaunchConfiguration("params_file"),
			"python_executable": LaunchConfiguration("python_executable"),
			"input_type": LaunchConfiguration("input_type"),
			"gripper_type": LaunchConfiguration("gripper_type"),
			"mediapipe_input_topic": LaunchConfiguration("mediapipe_input_topic"),
			"control_mode": LaunchConfiguration("control_mode"),
			"end_effector": LaunchConfiguration("end_effector"),
		}.items(),
		condition=IfCondition(LaunchConfiguration("launch_teleop_node")),
	)

	return LaunchDescription(
		[
			params_file_arg,
			python_executable_arg,
			input_type_arg,
			gripper_type_arg,
			joy_profile_arg,
			joy_device_path_arg,
			mediapipe_input_topic_arg,
			control_mode_arg,
			end_effector_arg,
			robotiq_namespace_arg,
			robotiq_serial_port_arg,
			robotiq_fake_hw_arg,
			robotiq_config_file_arg,
			robotiq_rviz2_arg,
			ur_type_arg,
			robot_ip_arg,
			reverse_ip_arg,
			launch_rviz_arg,
			launch_moveit_rviz_arg,
			launch_servo_arg,
			initial_joint_controller_arg,
			enable_moveit_arg,
			enable_camera_arg,
			enable_data_collector_arg,
			launch_teleop_node_arg,
			data_collector_params_file_arg,
			OpaqueFunction(function=_maybe_include_joy_driver),
			OpaqueFunction(function=_maybe_include_moveit_servo),
			OpaqueFunction(function=_maybe_include_realsense),
			OpaqueFunction(function=_maybe_include_end_effector_driver),
			OpaqueFunction(function=_maybe_include_data_collector),
			ur_launch,
			teleop_launch,
		]
	)
