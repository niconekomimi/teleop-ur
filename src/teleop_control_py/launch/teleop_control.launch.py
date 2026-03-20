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
    if normalized in ("joy", "mediapipe"):
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

    return [
        LogInfo(msg=f"[teleop_control.launch] resolved input_type: {resolved_input}"),
        LogInfo(msg=f"[teleop_control.launch] resolved gripper_type: {resolved_gripper}"),
        LogInfo(msg=f"[teleop_control.launch] resolved mediapipe_input_topic: {resolved_mediapipe_input_topic}"),
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
                f"mediapipe_input_topic:={resolved_mediapipe_input_topic}",
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
        description="Optional input backend override (joy|mediapipe). Empty means read from params_file.",
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

    return LaunchDescription(
        [
            params_file_arg,
            python_executable_arg,
            input_type_arg,
            gripper_type_arg,
            control_mode_arg,
            end_effector_arg,
            mediapipe_input_topic_arg,
            OpaqueFunction(function=_launch_teleop_node),
        ]
    )
