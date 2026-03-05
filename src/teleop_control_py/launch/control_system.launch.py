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


def _read_control_mode_from_params(params_file: str) -> str:
	mode = "hand"
	try:
		import yaml  # type: ignore
	except Exception:
		return mode

	try:
		with open(params_file, "r", encoding="utf-8") as f:
			data = yaml.safe_load(f)
	except Exception:
		return mode

	if not isinstance(data, dict):
		return mode

	for key in ("teleop_control_node", "/teleop_control_node"):
		block = data.get(key)
		if not isinstance(block, dict):
			continue
		params = block.get("ros__parameters")
		if not isinstance(params, dict):
			continue
		cm = params.get("control_mode")
		if isinstance(cm, str) and cm.strip():
			return cm.strip().lower()
	return mode


def _read_end_effector_from_params(params_file: str) -> str:
	ee = "robotiq"
	try:
		import yaml  # type: ignore
	except Exception:
		return ee

	try:
		with open(params_file, "r", encoding="utf-8") as f:
			data = yaml.safe_load(f)
	except Exception:
		return ee

	if not isinstance(data, dict):
		return ee

	for key in ("teleop_control_node", "/teleop_control_node"):
		block = data.get(key)
		if not isinstance(block, dict):
			continue
		params = block.get("ros__parameters")
		if not isinstance(params, dict):
			continue
		raw = params.get("end_effector")
		if isinstance(raw, str) and raw.strip():
			return raw.strip().lower()
	return ee


def _resolve_control_mode(context) -> str:
	override = LaunchConfiguration("control_mode").perform(context).strip().lower()
	if override in ("hand", "xbox"):
		return override
	params_file = LaunchConfiguration("params_file").perform(context)
	return _read_control_mode_from_params(params_file)


def _resolve_end_effector(context) -> str:
	override = LaunchConfiguration("end_effector").perform(context).strip().lower()
	if override in ("auto", "qbsofthand", "robotiq"):
		return override
	params_file = LaunchConfiguration("params_file").perform(context)
	return _read_end_effector_from_params(params_file)


def _maybe_include_end_effector_driver(context, *args, **kwargs):
	ee = _resolve_end_effector(context)
	actions = [LogInfo(msg=f"[control_system] resolved end_effector: {ee}")]

	# NOTE: auto-detection removed. Keep backward compatibility by treating 'auto' as qbsofthand.
	if ee == "auto":
		actions.append(LogInfo(msg="[control_system] end_effector=auto is deprecated; using qbsofthand (manual selection only)"))
		ee = "qbsofthand"

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
	control_mode = _resolve_control_mode(context)

	actions = [LogInfo(msg=f"[control_system] resolved control_mode: {control_mode}")]
	if control_mode != "xbox":
		return actions

	joy_share = get_package_share_directory("multi_joy_driver")
	joy_launch_path = os.path.join(joy_share, "launch", "joy_driver.launch.py")
	actions.append(
		IncludeLaunchDescription(
			PythonLaunchDescriptionSource(joy_launch_path),
			launch_arguments={
				"python_executable": LaunchConfiguration("python_executable"),
			}.items(),
		)
	)
	actions.append(LogInfo(msg="[control_system] Included multi_joy_driver/joy_driver.launch.py"))
	return actions


def _maybe_include_moveit_servo(context, *args, **kwargs):
	control_mode = _resolve_control_mode(context)

	enable_moveit_raw = LaunchConfiguration("enable_moveit").perform(context).strip().lower()
	enable_moveit = enable_moveit_raw in ("1", "true", "yes", "on")

	actions = [
		LogInfo(
			msg=(
				f"[control_system] enable_moveit={enable_moveit_raw} "
				f"control_mode={control_mode}"
			)
		)
	]
	if (control_mode != "xbox") or (not enable_moveit):
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
	control_mode = _resolve_control_mode(context)

	enable_camera_raw = LaunchConfiguration("enable_camera").perform(context).strip().lower()
	enable_camera = enable_camera_raw in ("1", "true", "yes", "on")

	actions = [
		LogInfo(
			msg=(
				f"[control_system] enable_camera={enable_camera_raw} "
				f"control_mode={control_mode}"
			)
		)
	]

	if control_mode == "xbox":
		actions.append(LogInfo(msg="[control_system] Skip RealSense in xbox mode"))
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
		description="Optional control mode override (hand|xbox). Empty means read from params_file.",
	)
	end_effector_arg = DeclareLaunchArgument(
		"end_effector",
		default_value="",
		description="Optional end effector override (qbsofthand|robotiq; 'auto' is a deprecated alias for qbsofthand). Empty means read from params_file.",
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
	enable_moveit_arg = DeclareLaunchArgument(
		"enable_moveit",
		default_value="true",
		description="Enable MoveIt bringup (move_group + optional servo). Auto-included for xbox mode.",
	)
	enable_camera_arg = DeclareLaunchArgument(
		"enable_camera",
		default_value="true",
		description="Enable RealSense camera (effective only in hand mode)",
	)

	ur_driver_share = get_package_share_directory("ur_robot_driver")
	ur_launch = IncludeLaunchDescription(
		PythonLaunchDescriptionSource(os.path.join(ur_driver_share, "launch", "ur_control.launch.py")),
		launch_arguments={
			"ur_type": LaunchConfiguration("ur_type"),
			"robot_ip": LaunchConfiguration("robot_ip"),
			"reverse_ip": LaunchConfiguration("reverse_ip"),
			"launch_rviz": LaunchConfiguration("launch_rviz"),
		}.items(),
	)

	teleop_launch = IncludeLaunchDescription(
		PythonLaunchDescriptionSource(os.path.join(teleop_share, "launch", "teleop_control.launch.py")),
		launch_arguments={
			"params_file": LaunchConfiguration("params_file"),
			"python_executable": LaunchConfiguration("python_executable"),
			"control_mode": LaunchConfiguration("control_mode"),
			"end_effector": LaunchConfiguration("end_effector"),
		}.items(),
	)

	return LaunchDescription(
		[
			params_file_arg,
			python_executable_arg,
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
			enable_moveit_arg,
			enable_camera_arg,
			OpaqueFunction(function=_maybe_include_joy_driver),
			OpaqueFunction(function=_maybe_include_moveit_servo),
			OpaqueFunction(function=_maybe_include_realsense),
			OpaqueFunction(function=_maybe_include_end_effector_driver),
			ur_launch,
			teleop_launch,
		]
	)
