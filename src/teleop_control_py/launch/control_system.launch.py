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


def _maybe_include_joy_driver(context, *args, **kwargs):
	params_file = LaunchConfiguration("params_file").perform(context)
	control_mode = _read_control_mode_from_params(params_file)

	actions = [LogInfo(msg=f"[control_system] control_mode from params: {control_mode}")]
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
	params_file = LaunchConfiguration("params_file").perform(context)
	control_mode = _read_control_mode_from_params(params_file)

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
		description="Enable RealSense camera (set to false if USB issues persist)",
	)

	softhand_node = Node(
		package="qbsofthand_control",
		executable="qbsofthand_control_node",
		name="qbsofthand_control_node",
		output="screen",
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

	realsense_share = get_package_share_directory("realsense2_camera")
	realsense_launch = IncludeLaunchDescription(
		PythonLaunchDescriptionSource(os.path.join(realsense_share, "launch", "rs_launch.py")),
		launch_arguments={
			"align_depth.enable": "true",
		}.items(),
		condition=IfCondition(LaunchConfiguration("enable_camera")),
	)

	teleop_launch = IncludeLaunchDescription(
		PythonLaunchDescriptionSource(os.path.join(teleop_share, "launch", "teleop_control.launch.py")),
		launch_arguments={
			"params_file": LaunchConfiguration("params_file"),
			"python_executable": LaunchConfiguration("python_executable"),
		}.items(),
	)

	return LaunchDescription(
		[
			params_file_arg,
			python_executable_arg,
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
			softhand_node,
			ur_launch,
			realsense_launch,
			teleop_launch,
		]
	)
