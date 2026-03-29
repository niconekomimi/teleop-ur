#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import tempfile
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
from launch_ros.actions import SetParameter
from launch_ros.actions import SetRemap
from launch_ros.parameter_descriptions import ParameterValue


def _python_supports_modules(candidate: str, modules: tuple[str, ...]) -> bool:
	try:
		result = subprocess.run(
			[
				candidate,
				"-c",
				(
					"import importlib.util, sys; "
					"mods = sys.argv[1:]; "
					"missing = [m for m in mods if importlib.util.find_spec(m) is None]; "
					"raise SystemExit(0 if not missing else 1)"
				),
				*modules,
			],
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
			timeout=2.0,
			check=False,
		)
	except Exception:
		return False
	return result.returncode == 0


def _default_python_executable() -> str:
	candidates = []
	for prefix_var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
		prefix = os.environ.get(prefix_var)
		if not prefix:
			continue
		candidate = os.path.join(prefix, "bin", "python3")
		if os.path.exists(candidate):
			candidates.append(candidate)
	home_candidate = os.path.expanduser("~/clds/bin/python3")
	if os.path.exists(home_candidate):
		candidates.append(home_candidate)
	candidates.append("python3")

	for candidate in candidates:
		if _python_supports_modules(candidate, ("mediapipe", "pynput")):
			return candidate

	for candidate in candidates:
		if candidate == "python3" or os.path.exists(candidate):
			return candidate
	return "python3"


def _load_yaml_dict(params_file: str) -> Dict[str, Any]:
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
	return data


def _load_teleop_params(params_file: str) -> Dict[str, Any]:
	data = _load_yaml_dict(params_file)
	if not data:
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


def _load_named_params(params_file: str, section_name: str) -> Dict[str, Any]:
	data = _load_yaml_dict(params_file)
	if not data:
		return {}

	block = data.get(section_name)
	if not isinstance(block, dict):
		return {}

	params = block.get("ros__parameters")
	if isinstance(params, dict):
		return params
	return block


def _resolve_launch_or_config(context, arg_name: str, config: Dict[str, Any], key: str, default: str = "") -> str:
	raw = LaunchConfiguration(arg_name).perform(context).strip()
	if raw != "":
		return raw
	if key in config and config[key] is not None:
		return str(config[key])
	return default


def _materialize_robotiq_config(config: Dict[str, Any]) -> str:
	try:
		import yaml  # type: ignore
	except Exception:
		return ""

	config_path = os.path.join(tempfile.gettempdir(), "teleop_control_py_robotiq_config.yaml")
	payload = {
		"open_threshold": float(config.get("open_threshold", 0.3)),
		"close_threshold": float(config.get("close_threshold", -0.3)),
	}
	try:
		with open(config_path, "w", encoding="utf-8") as f:
			yaml.safe_dump(payload, f, sort_keys=False)
	except Exception:
		return ""
	return config_path


def _coerce_input_type(value: str) -> str:
	normalized = value.strip().lower()
	if normalized == "xbox":
		return "joy"
	if normalized == "hand":
		return "mediapipe"
	if normalized in ("quest3", "quest", "vr", "webxr"):
		return "quest3"
	if normalized in ("joy", "mediapipe", "quest3"):
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


def _quest3_bridge_enabled(context, resolved_input_type: Optional[str] = None) -> bool:
	value = LaunchConfiguration("launch_quest3_bridge").perform(context).strip().lower()
	if value in ("1", "true", "yes", "on"):
		return True
	if value in ("0", "false", "no", "off"):
		return False
	if resolved_input_type is None:
		resolved_input_type = _resolve_input_type(context)
	return resolved_input_type == "quest3"


def _maybe_include_quest3_bridge(context, *args, **kwargs):
	input_type = _resolve_input_type(context)
	enabled_raw = LaunchConfiguration("launch_quest3_bridge").perform(context).strip() or "auto"
	enabled = _quest3_bridge_enabled(context, input_type)
	params_file = LaunchConfiguration("quest3_bridge_params_file").perform(context).strip()
	resolved_params_file = params_file or LaunchConfiguration("params_file").perform(context)
	advertised_host = LaunchConfiguration("quest3_advertised_host").perform(context).strip()

	actions = [
		LogInfo(
			msg=(
				f"[control_system] launch_quest3_bridge={enabled_raw}, "
				f"resolved input_type={input_type}, bridge_params_file={resolved_params_file}"
			)
		)
	]
	if not enabled:
		actions.append(LogInfo(msg="[control_system] Skip Quest3 bridge"))
		return actions

	parameters = [resolved_params_file]
	if advertised_host:
		parameters.append({"advertised_host": advertised_host})

	actions.append(
		Node(
			package="teleop_control_py",
			executable="quest3_webxr_bridge_node",
			name="quest3_webxr_bridge_node",
			output="screen",
			parameters=parameters,
		)
	)
	return actions


def _maybe_include_end_effector_driver(context, *args, **kwargs):
	ee = _resolve_gripper_type(context)
	actions = [LogInfo(msg=f"[control_system] resolved gripper_type: {ee}")]

	if ee == "robotiq":
		params_file = LaunchConfiguration("params_file").perform(context)
		robotiq_cfg = _load_named_params(params_file, "robotiq_gripper")
		robotiq_config_file = _resolve_launch_or_config(context, "robotiq_config_file", {}, "config_file", "")
		if robotiq_config_file == "" and robotiq_cfg:
			robotiq_config_file = _materialize_robotiq_config(robotiq_cfg)

		robotiq_share = get_package_share_directory("robotiq_2f_gripper_hardware")
		robotiq_launch_path = os.path.join(robotiq_share, "launch", "robotiq_2f_gripper_launch.py")
		actions.append(
			IncludeLaunchDescription(
				PythonLaunchDescriptionSource(robotiq_launch_path),
				launch_arguments={
					"namespace": LaunchConfiguration("robotiq_namespace"),
					"serial_port": _resolve_launch_or_config(context, "robotiq_serial_port", robotiq_cfg, "serial_port", "/dev/robotiq_gripper"),
					"baudrate": _resolve_launch_or_config(context, "robotiq_baudrate", robotiq_cfg, "baudrate", "115200"),
					"timeout": _resolve_launch_or_config(context, "robotiq_timeout", robotiq_cfg, "timeout", "1.0"),
					"action_timeout": _resolve_launch_or_config(context, "robotiq_action_timeout", robotiq_cfg, "action_timeout", "20"),
					"slave_address": _resolve_launch_or_config(context, "robotiq_slave_address", robotiq_cfg, "slave_address", "9"),
					"fake_hardware": _resolve_launch_or_config(context, "robotiq_fake_hardware", robotiq_cfg, "fake_hardware", "False"),
					"config_file": robotiq_config_file,
					"rviz2": _resolve_launch_or_config(context, "robotiq_rviz2", robotiq_cfg, "rviz2", "False"),
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

	teleop_share = get_package_share_directory("teleop_control_py")
	joy_launch_path = os.path.join(teleop_share, "launch", "joy_driver.launch.py")
	joy_params_file = os.path.join(teleop_share, "config", "joy_driver_params.yaml")
	actions.append(
		IncludeLaunchDescription(
			PythonLaunchDescriptionSource(joy_launch_path),
			launch_arguments={
				"params_file": joy_params_file,
				"python_executable": LaunchConfiguration("python_executable"),
				"profile": LaunchConfiguration("joy_profile"),
				"device_path": LaunchConfiguration("joy_device_path"),
			}.items(),
		)
	)
	actions.append(LogInfo(msg="[control_system] Included teleop_control_py joy_driver.launch.py"))
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

	params_file = LaunchConfiguration("params_file").perform(context)
	servo_override_params = _load_named_params(params_file, "moveit_servo")
	moveit_share = get_package_share_directory("ur_moveit_config")
	moveit_launch_path = os.path.join(moveit_share, "launch", "ur_moveit.launch.py")
	actions.append(
		GroupAction(
			actions=[
				# Do NOT modify upstream UR packages. Instead, remap Servo's private input topic
				# so our teleop publisher `/servo_node/delta_twist_cmds` is always consumed.
				SetRemap(src="~/delta_twist_cmds", dst="/servo_node/delta_twist_cmds"),
				*[
					SetParameter(name=f"moveit_servo.{key}", value=ParameterValue(value))
					for key, value in servo_override_params.items()
				],
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
	if servo_override_params:
		actions.append(
			LogInfo(
				msg=(
					f"[control_system] Applied MoveIt Servo overrides from params_file: "
					f"{', '.join(sorted(servo_override_params.keys()))}"
				)
			)
		)
	actions.append(LogInfo(msg="[control_system] Included ur_moveit_config/ur_moveit.launch.py"))
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
	resolved_robot_profile = LaunchConfiguration("robot_profile").perform(context).strip()
	actions.append(
		Node(
			package="teleop_control_py",
			executable="data_collector_node",
			name="data_collector",
			output="screen",
			parameters=[
				LaunchConfiguration("data_collector_params_file"),
				{
					"end_effector_type": collector_ee,
					"robot_profile": LaunchConfiguration("robot_profile"),
				},
			],
		)
	)
	actions.append(
		LogInfo(
			msg=(
				"[control_system] Included teleop_control_py.nodes.data_collector_node "
				f"with end_effector_type={collector_ee}, robot_profile={resolved_robot_profile}"
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
		description="Optional input backend override (joy|mediapipe|quest3). Empty means read from params_file.",
	)
	gripper_type_arg = DeclareLaunchArgument(
		"gripper_type",
		default_value="",
		description="Optional gripper backend override (robotiq|qbsofthand). Empty means read from params_file.",
	)
	joy_profile_arg = DeclareLaunchArgument(
		"joy_profile",
		default_value="auto",
		description="Joystick profile for joy_driver_node (auto|xbox|ps5|generic).",
	)
	joy_device_path_arg = DeclareLaunchArgument(
		"joy_device_path",
		default_value="",
		description="Optional joystick event device path for joy_driver_node.",
	)
	mediapipe_input_topic_arg = DeclareLaunchArgument(
		"mediapipe_input_topic",
		default_value="",
		description="Optional MediaPipe image topic override.",
	)
	mediapipe_depth_topic_arg = DeclareLaunchArgument(
		"mediapipe_depth_topic",
		default_value="",
		description="Optional MediaPipe depth topic override.",
	)
	mediapipe_camera_info_topic_arg = DeclareLaunchArgument(
		"mediapipe_camera_info_topic",
		default_value="",
		description="Optional MediaPipe aligned camera_info topic override.",
	)
	mediapipe_camera_driver_arg = DeclareLaunchArgument(
		"mediapipe_camera_driver",
		default_value="realsense",
		description="MediaPipe camera driver type (realsense|oakd).",
	)
	mediapipe_camera_serial_number_arg = DeclareLaunchArgument(
		"mediapipe_camera_serial_number",
		default_value="",
		description="Optional serial number for auto-started MediaPipe camera driver.",
	)
	mediapipe_enable_depth_arg = DeclareLaunchArgument(
		"mediapipe_enable_depth",
		default_value="false",
		description="Enable SDK depth stream for MediaPipe input (true|false).",
	)
	mediapipe_show_debug_window_arg = DeclareLaunchArgument(
		"mediapipe_show_debug_window",
		default_value="true",
		description="Enable MediaPipe OpenCV debug window in teleop node.",
	)
	mediapipe_use_sdk_camera_arg = DeclareLaunchArgument(
		"mediapipe_use_sdk_camera",
		default_value="true",
		description="Use SDK camera directly for MediaPipe input (true|false).",
	)

	robotiq_namespace_arg = DeclareLaunchArgument(
		"robotiq_namespace",
		default_value="",
		description="Namespace for Robotiq gripper (usually empty)",
	)
	robotiq_serial_port_arg = DeclareLaunchArgument(
		"robotiq_serial_port",
		default_value="",
		description="Optional serial port override for Robotiq gripper. Empty means read from params_file.",
	)
	robotiq_baudrate_arg = DeclareLaunchArgument(
		"robotiq_baudrate",
		default_value="",
		description="Optional baudrate override for Robotiq gripper. Empty means read from params_file.",
	)
	robotiq_timeout_arg = DeclareLaunchArgument(
		"robotiq_timeout",
		default_value="",
		description="Optional serial timeout override for Robotiq gripper. Empty means read from params_file.",
	)
	robotiq_action_timeout_arg = DeclareLaunchArgument(
		"robotiq_action_timeout",
		default_value="",
		description="Optional action timeout override for Robotiq gripper. Empty means read from params_file.",
	)
	robotiq_slave_address_arg = DeclareLaunchArgument(
		"robotiq_slave_address",
		default_value="",
		description="Optional Modbus slave address override for Robotiq gripper. Empty means read from params_file.",
	)
	robotiq_fake_hw_arg = DeclareLaunchArgument(
		"robotiq_fake_hardware",
		default_value="",
		description="Optional fake hardware override for Robotiq gripper. Empty means read from params_file.",
	)
	robotiq_config_file_arg = DeclareLaunchArgument(
		"robotiq_config_file",
		default_value="",
		description="Optional Robotiq config YAML path override. Empty means materialize from params_file.",
	)
	robotiq_rviz2_arg = DeclareLaunchArgument(
		"robotiq_rviz2",
		default_value="",
		description="Optional RViz2 override for Robotiq visualization. Empty means read from params_file.",
	)

	ur_type_arg = DeclareLaunchArgument(
		"ur_type",
		default_value="ur5",
		description="UR robot type (ur5, ur10, etc.)",
	)
	robot_profile_arg = DeclareLaunchArgument(
		"robot_profile",
		default_value=["ur_servo_", LaunchConfiguration("ur_type")],
		description="Backend-level robot profile name used by teleop/commander/collector.",
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
	launch_quest3_bridge_arg = DeclareLaunchArgument(
		"launch_quest3_bridge",
		default_value="auto",
		description="Launch Quest3 WebXR bridge (auto|true|false). auto launches it when input_type resolves to quest3.",
	)
	quest3_bridge_params_file_arg = DeclareLaunchArgument(
		"quest3_bridge_params_file",
		default_value="",
		description="Optional params file override for quest3_webxr_bridge_node. Empty means reuse params_file.",
	)
	quest3_advertised_host_arg = DeclareLaunchArgument(
		"quest3_advertised_host",
		default_value="",
		description="Optional Quest bridge LAN host/IP override. Empty means read advertised_host from params file.",
	)
	data_collector_params_file_arg = DeclareLaunchArgument(
		"data_collector_params_file",
		default_value=os.path.join(teleop_share, "config", "data_collector_params.yaml"),
		description="Path to data collector parameter file",
	)
	commander_pose_max_age_sec_arg = DeclareLaunchArgument(
		"commander_pose_max_age_sec",
		default_value="0.05",
		description="Commander pose freshness threshold in seconds.",
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

	robot_commander_node = Node(
		package="teleop_control_py",
		executable="robot_commander_node",
		name="commander",
		output="screen",
		parameters=[
			{
				"ur_type": LaunchConfiguration("ur_type"),
				"robot_profile": LaunchConfiguration("robot_profile"),
				"pose_max_age_sec": ParameterValue(
					LaunchConfiguration("commander_pose_max_age_sec"),
					value_type=float,
				),
			},
		],
	)

	teleop_launch = IncludeLaunchDescription(
		PythonLaunchDescriptionSource(os.path.join(teleop_share, "launch", "teleop_control.launch.py")),
		launch_arguments={
			"params_file": LaunchConfiguration("params_file"),
			"python_executable": LaunchConfiguration("python_executable"),
			"input_type": LaunchConfiguration("input_type"),
			"gripper_type": LaunchConfiguration("gripper_type"),
			"mediapipe_input_topic": LaunchConfiguration("mediapipe_input_topic"),
			"mediapipe_depth_topic": LaunchConfiguration("mediapipe_depth_topic"),
			"mediapipe_camera_info_topic": LaunchConfiguration("mediapipe_camera_info_topic"),
			"mediapipe_camera_driver": LaunchConfiguration("mediapipe_camera_driver"),
			"mediapipe_camera_serial_number": LaunchConfiguration("mediapipe_camera_serial_number"),
			"mediapipe_enable_depth": LaunchConfiguration("mediapipe_enable_depth"),
			"mediapipe_show_debug_window": LaunchConfiguration("mediapipe_show_debug_window"),
			"mediapipe_use_sdk_camera": LaunchConfiguration("mediapipe_use_sdk_camera"),
			"control_mode": LaunchConfiguration("control_mode"),
			"end_effector": LaunchConfiguration("end_effector"),
			"robot_profile": LaunchConfiguration("robot_profile"),
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
			mediapipe_depth_topic_arg,
			mediapipe_camera_info_topic_arg,
			mediapipe_camera_driver_arg,
			mediapipe_camera_serial_number_arg,
			mediapipe_enable_depth_arg,
			mediapipe_show_debug_window_arg,
			mediapipe_use_sdk_camera_arg,
			control_mode_arg,
			end_effector_arg,
			robotiq_namespace_arg,
			robotiq_serial_port_arg,
			robotiq_baudrate_arg,
			robotiq_timeout_arg,
			robotiq_action_timeout_arg,
			robotiq_slave_address_arg,
			robotiq_fake_hw_arg,
			robotiq_config_file_arg,
			robotiq_rviz2_arg,
			ur_type_arg,
			robot_profile_arg,
			robot_ip_arg,
			reverse_ip_arg,
			launch_rviz_arg,
			launch_moveit_rviz_arg,
			launch_servo_arg,
			initial_joint_controller_arg,
				enable_moveit_arg,
				enable_data_collector_arg,
				launch_teleop_node_arg,
				launch_quest3_bridge_arg,
				quest3_bridge_params_file_arg,
				quest3_advertised_host_arg,
				data_collector_params_file_arg,
				commander_pose_max_age_sec_arg,
				OpaqueFunction(function=_maybe_include_quest3_bridge),
				OpaqueFunction(function=_maybe_include_joy_driver),
				OpaqueFunction(function=_maybe_include_moveit_servo),
				OpaqueFunction(function=_maybe_include_end_effector_driver),
			OpaqueFunction(function=_maybe_include_data_collector),
			ur_launch,
			robot_commander_node,
			teleop_launch,
		]
	)
