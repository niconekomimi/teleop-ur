#!/usr/bin/env python3
import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import LogInfo
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration


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
    # Prefer an activated venv/conda env if present; otherwise fall back to PATH lookup.
    candidates: list[str] = []
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


def _load_teleop_params(params_file: str) -> dict:
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
    if normalized in ("quest3", "quest", "vr", "webxr"):
        return "quest3"
    if normalized in ("joy", "mediapipe", "quest3"):
        return normalized
    return ""


def _coerce_gripper_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "qbsofthand"
    if normalized in ("robotiq", "qbsofthand"):
        return normalized
    return ""


def _resolve_input_type(params_file: str, input_type_override: str, control_mode_override: str) -> str:
    resolved = _coerce_input_type(input_type_override)
    if resolved:
        return resolved

    resolved = _coerce_input_type(control_mode_override)
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


def _resolve_gripper_type(params_file: str, gripper_type_override: str, end_effector_override: str) -> str:
    resolved = _coerce_gripper_type(gripper_type_override)
    if resolved:
        return resolved

    resolved = _coerce_gripper_type(end_effector_override)
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


def _resolve_param_string(params_file: str, key: str, override: str, default: str) -> str:
    value = str(override).strip()
    if value:
        return value

    params = _load_teleop_params(params_file)
    loaded = str(params.get(key, "")).strip()
    if loaded:
        return loaded

    return default


def _as_bool(value: str, default: bool = False) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _ros_string_literal(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _resolve_mediapipe_input_topic(params_file: str, override: str) -> str:
    value = str(override).strip()
    if value:
        return value

    params = _load_teleop_params(params_file)
    for key in ("mediapipe_input_topic", "image_topic", "mediapipe_topic"):
        loaded = str(params.get(key, "")).strip()
        if loaded:
            return loaded

    return "/camera/camera/color/image_raw"


def _launch_teleop_node(context, *args, **kwargs):
    resolved_robot_profile = str(LaunchConfiguration("robot_profile").perform(context)).strip() or "ur_servo_ur5"
    params_file = LaunchConfiguration("params_file").perform(context)
    resolved_input = _resolve_input_type(
        params_file,
        LaunchConfiguration("input_type").perform(context),
        LaunchConfiguration("control_mode").perform(context),
    )
    resolved_gripper = _resolve_gripper_type(
        params_file,
        LaunchConfiguration("gripper_type").perform(context),
        LaunchConfiguration("end_effector").perform(context),
    )
    resolved_mediapipe_input_topic = _resolve_mediapipe_input_topic(
        params_file,
        LaunchConfiguration("mediapipe_input_topic").perform(context),
    )
    resolved_mediapipe_depth_topic = _resolve_param_string(
        params_file,
        "mediapipe_depth_topic",
        LaunchConfiguration("mediapipe_depth_topic").perform(context),
        "/camera/camera/aligned_depth_to_color/image_raw",
    )
    resolved_mediapipe_camera_info_topic = _resolve_param_string(
        params_file,
        "mediapipe_camera_info_topic",
        LaunchConfiguration("mediapipe_camera_info_topic").perform(context),
        "/camera/camera/aligned_depth_to_color/camera_info",
    )
    resolved_mediapipe_camera_driver = _resolve_param_string(
        params_file,
        "mediapipe_camera_driver",
        LaunchConfiguration("mediapipe_camera_driver").perform(context),
        "realsense",
    )
    resolved_mediapipe_camera_serial_number = _resolve_param_string(
        params_file,
        "mediapipe_camera_serial_number",
        LaunchConfiguration("mediapipe_camera_serial_number").perform(context),
        "",
    )
    resolved_mediapipe_show_debug_window = _as_bool(
        _resolve_param_string(
            params_file,
            "mediapipe_show_debug_window",
            LaunchConfiguration("mediapipe_show_debug_window").perform(context),
            "true",
        ),
        default=True,
    )
    resolved_mediapipe_enable_depth = _as_bool(
        _resolve_param_string(
            params_file,
            "mediapipe_enable_depth",
            LaunchConfiguration("mediapipe_enable_depth").perform(context),
            "false",
        ),
        default=False,
    )
    resolved_mediapipe_use_sdk_camera = _as_bool(
        _resolve_param_string(
            params_file,
            "mediapipe_use_sdk_camera",
            LaunchConfiguration("mediapipe_use_sdk_camera").perform(context),
            "true",
        ),
        default=True,
    )

    return [
        LogInfo(msg=f"[teleop_control.launch] resolved input_type: {resolved_input}"),
        LogInfo(msg=f"[teleop_control.launch] resolved gripper_type: {resolved_gripper}"),
        LogInfo(msg=f"[teleop_control.launch] resolved mediapipe_input_topic: {resolved_mediapipe_input_topic}"),
        LogInfo(msg=f"[teleop_control.launch] resolved mediapipe_depth_topic: {resolved_mediapipe_depth_topic}"),
        LogInfo(msg=f"[teleop_control.launch] resolved mediapipe_camera_info_topic: {resolved_mediapipe_camera_info_topic}"),
        LogInfo(msg=f"[teleop_control.launch] resolved mediapipe_camera_driver: {resolved_mediapipe_camera_driver}"),
        LogInfo(
            msg=(
                "[teleop_control.launch] resolved mediapipe_camera_serial_number: "
                f"{resolved_mediapipe_camera_serial_number or 'auto'}"
            )
        ),
        LogInfo(msg=f"[teleop_control.launch] resolved mediapipe_enable_depth: {resolved_mediapipe_enable_depth}"),
        LogInfo(msg=f"[teleop_control.launch] resolved mediapipe_show_debug_window: {resolved_mediapipe_show_debug_window}"),
        LogInfo(msg=f"[teleop_control.launch] resolved mediapipe_use_sdk_camera: {resolved_mediapipe_use_sdk_camera}"),
        LogInfo(msg=f"[teleop_control.launch] resolved robot_profile: {resolved_robot_profile}"),
        ExecuteProcess(
            name="teleop_control_node",
            cmd=[
                LaunchConfiguration("python_executable"),
                "-m",
                "teleop_control_py.nodes.teleop_control_node",
                "--ros-args",
                "-r",
                "__node:=teleop_control_node",
                "--params-file",
                LaunchConfiguration("params_file"),
                "-p",
                f"input_type:={resolved_input}",
                "-p",
                f"gripper_type:={resolved_gripper}",
                "-p",
                f"robot_profile:={resolved_robot_profile}",
                "-p",
                f"mediapipe_input_topic:={resolved_mediapipe_input_topic}",
                "-p",
                f"mediapipe_depth_topic:={resolved_mediapipe_depth_topic}",
                "-p",
                f"mediapipe_camera_info_topic:={resolved_mediapipe_camera_info_topic}",
                "-p",
                f"mediapipe_camera_driver:={resolved_mediapipe_camera_driver}",
                "-p",
                f"mediapipe_camera_serial_number:={_ros_string_literal(resolved_mediapipe_camera_serial_number)}",
                "-p",
                f"mediapipe_enable_depth:={str(bool(resolved_mediapipe_enable_depth)).lower()}",
                "-p",
                f"mediapipe_show_debug_window:={str(bool(resolved_mediapipe_show_debug_window)).lower()}",
                "-p",
                f"mediapipe_use_sdk_camera:={str(bool(resolved_mediapipe_use_sdk_camera)).lower()}",
            ],
            output="screen",
        ),
    ]

def generate_launch_description():
    # 获取包路径
    pkg_share = get_package_share_directory("teleop_control_py")
    default_params = os.path.join(pkg_share, "config", "teleop_params.yaml")

    # 声明参数文件路径参数
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to teleop parameter file",
    )

    python_executable_arg = DeclareLaunchArgument(
        "python_executable",
        default_value=_default_python_executable(),
        description="Python executable used to run teleop_control_node (needs mediapipe)",
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

    mediapipe_input_topic_arg = DeclareLaunchArgument(
        "mediapipe_input_topic",
        default_value="",
        description="Optional MediaPipe image topic override.",
    )

    mediapipe_depth_topic_arg = DeclareLaunchArgument(
        "mediapipe_depth_topic",
        default_value="",
        description="Optional MediaPipe depth image topic override.",
    )

    mediapipe_camera_info_topic_arg = DeclareLaunchArgument(
        "mediapipe_camera_info_topic",
        default_value="",
        description="Optional MediaPipe aligned camera_info topic override.",
    )

    mediapipe_camera_driver_arg = DeclareLaunchArgument(
        "mediapipe_camera_driver",
        default_value="",
        description="Optional MediaPipe camera driver override (realsense|oakd).",
    )

    mediapipe_camera_serial_number_arg = DeclareLaunchArgument(
        "mediapipe_camera_serial_number",
        default_value="",
        description="Optional MediaPipe camera serial number override.",
    )

    mediapipe_show_debug_window_arg = DeclareLaunchArgument(
        "mediapipe_show_debug_window",
        default_value="",
        description="Optional MediaPipe debug window override (true|false).",
    )

    mediapipe_enable_depth_arg = DeclareLaunchArgument(
        "mediapipe_enable_depth",
        default_value="",
        description="Optional MediaPipe SDK depth override (true|false).",
    )

    mediapipe_use_sdk_camera_arg = DeclareLaunchArgument(
        "mediapipe_use_sdk_camera",
        default_value="true",
        description="Use SDK camera directly for MediaPipe input (true|false).",
    )

    robot_profile_arg = DeclareLaunchArgument(
        "robot_profile",
        default_value="ur_servo_ur5",
        description="Backend-level robot profile name for teleop defaults.",
    )

    return LaunchDescription(
        [
            params_file_arg,
            python_executable_arg,
            input_type_arg,
            gripper_type_arg,
            control_mode_arg,
            end_effector_arg,
            mediapipe_input_topic_arg,
            mediapipe_depth_topic_arg,
            mediapipe_camera_info_topic_arg,
            mediapipe_camera_driver_arg,
            mediapipe_camera_serial_number_arg,
            mediapipe_show_debug_window_arg,
            mediapipe_enable_depth_arg,
            mediapipe_use_sdk_camera_arg,
            robot_profile_arg,
            OpaqueFunction(function=_launch_teleop_node),
        ]
    )
