#!/usr/bin/env python3
"""GUI shared helpers for launch commands, defaults, and runtime status probes."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
from pathlib import Path
import struct
import socket
from typing import Dict, List, Optional

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


@dataclass(frozen=True)
class GuiSettings:
    default_robot_ip: str
    default_reverse_ip: str
    ur_type: str
    ur_type_options: List[str]
    default_input_type: str
    default_gripper_type: str
    default_joy_profile: str
    joy_profiles: List[str]
    default_mediapipe_input_topic: str
    default_preview_global_topic: str
    default_preview_wrist_topic: str
    camera_driver_options: List[str]
    default_camera_driver: str
    default_global_camera_source: str
    default_wrist_camera_source: str
    default_hdf5_output_dir: str
    default_hdf5_filename: str

    home_joint_positions: List[float]

def _workspace_root_from_file(current_file: str | Path) -> Path:
    current_path = Path(current_file).resolve()
    for candidate in [current_path] + list(current_path.parents):
        config_path = candidate / "src" / "teleop_control_py" / "config" / "gui_params.yaml"
        if config_path.exists():
            return candidate

    try:
        share_dir = Path(get_package_share_directory("teleop_control_py"))
        return share_dir.parents[2]
    except PackageNotFoundError:
        pass

    return current_path.parent


def get_repo_gui_config_path(current_file: str | Path) -> Path:
    workspace_root = _workspace_root_from_file(current_file)
    return workspace_root / "src" / "teleop_control_py" / "config" / "gui_params.yaml"


def get_installed_gui_config_path() -> Optional[Path]:
    try:
        share_dir = Path(get_package_share_directory("teleop_control_py"))
    except PackageNotFoundError:
        return None
    return share_dir / "config" / "gui_params.yaml"


def load_gui_settings(current_file: str | Path) -> GuiSettings:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    candidate_paths = [get_repo_gui_config_path(current_file)]
    installed = get_installed_gui_config_path()
    if installed is not None:
        candidate_paths.append(installed)

    raw: Dict[str, object] = {}
    if yaml is not None:
        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    content = yaml.safe_load(handle) or {}
            except Exception:
                continue
            block = content.get("teleop_gui", {}) if isinstance(content, dict) else {}
            params = block.get("ros__parameters", {}) if isinstance(block, dict) else {}
            if isinstance(params, dict):
                raw = params
                break

    workspace_root = _workspace_root_from_file(current_file)
    output_dir_raw = str(raw.get("default_hdf5_output_dir", "data")).strip() or "data"
    output_dir = Path(output_dir_raw).expanduser()
    if not output_dir.is_absolute():
        output_dir = workspace_root / output_dir

    return GuiSettings(
        default_robot_ip=str(raw.get("default_robot_ip", "192.168.1.211")),
        default_reverse_ip=str(raw.get("default_reverse_ip", "192.168.1.10")),
        ur_type=str(raw.get("ur_type", "ur5")),
        ur_type_options=[str(v) for v in raw.get("ur_type_options", ["ur3", "ur5", "ur10", "ur16e", "ur20", "ur30e"])],
        default_input_type=str(raw.get("default_input_type", "joy")),
        default_gripper_type=str(raw.get("default_gripper_type", "robotiq")),
        default_joy_profile=str(raw.get("default_joy_profile", "auto")),
        joy_profiles=[str(v) for v in raw.get("joy_profiles", ["auto", "xbox", "ps5", "generic"])],
        default_mediapipe_input_topic=str(raw.get("default_mediapipe_input_topic", "/camera/camera/color/image_raw")),
        default_preview_global_topic=str(raw.get("default_preview_global_topic", "/data_collector/preview/global/image_raw")),
        default_preview_wrist_topic=str(raw.get("default_preview_wrist_topic", "/data_collector/preview/wrist/image_raw")),
        camera_driver_options=[str(v) for v in raw.get("camera_driver_options", ["realsense", "oakd"])],
        default_camera_driver=str(raw.get("default_camera_driver", "realsense")),
        default_global_camera_source=str(raw.get("default_global_camera_source", "realsense")),
        default_wrist_camera_source=str(raw.get("default_wrist_camera_source", "oakd")),
        default_hdf5_output_dir=str(output_dir),
        default_hdf5_filename=str(raw.get("default_hdf5_filename", "libero_demos.hdf5")),
        home_joint_positions=[float(v) for v in raw.get("home_joint_positions", [])],
    )


def save_gui_settings_overrides(current_file: str | Path, updates: Dict[str, object]) -> Path:
    import yaml  # type: ignore

    config_path = get_repo_gui_config_path(current_file)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
    else:
        content = {}

    if not isinstance(content, dict):
        content = {}

    teleop_gui = content.setdefault("teleop_gui", {})
    if not isinstance(teleop_gui, dict):
        teleop_gui = {}
        content["teleop_gui"] = teleop_gui

    params = teleop_gui.setdefault("ros__parameters", {})
    if not isinstance(params, dict):
        params = {}
        teleop_gui["ros__parameters"] = params

    for key, value in updates.items():
        params[key] = value

    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False, allow_unicode=False)

    return config_path


def _interface_ipv4_address(interface_name: str) -> Optional[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed_name = struct.pack("256s", interface_name[:15].encode("utf-8"))
        result = fcntl.ioctl(sock.fileno(), 0x8915, packed_name)
        return socket.inet_ntoa(result[20:24])
    except OSError:
        return None
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _linux_preferred_interface_ip() -> Optional[str]:
    wired_prefixes = ("en", "eth")
    wifi_prefixes = ("wl",)
    candidates: List[tuple[int, str]] = []

    try:
        interface_names = [name for _, name in socket.if_nameindex()]
    except OSError:
        return None

    for name in interface_names:
        if name == "lo":
            continue

        ip = _interface_ipv4_address(name)
        if not ip or ip.startswith("127."):
            continue

        priority = 2
        if name.startswith(wired_prefixes):
            priority = 0
        elif name.startswith(wifi_prefixes):
            priority = 1

        candidates.append((priority, ip))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def get_local_ip() -> str:
    preferred_ip = _linux_preferred_interface_ip()
    if preferred_ip:
        return preferred_ip

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "unknown"
    finally:
        try:
            sock.close()
        except Exception:
            pass


def build_camera_driver_command(camera_driver: str) -> List[str]:
    key = camera_driver.strip().lower()
    if key == "realsense":
        return ["ros2", "launch", "realsense2_camera", "rs_launch.py"]
    if key == "oakd":
        return ["ros2", "launch", "depthai_examples", "rgb_stereo_node.launch.py"]
    raise ValueError(f"Unsupported camera driver: {camera_driver}")


def build_robot_driver_command(robot_ip: str, reverse_ip: str, ur_type: str) -> List[str]:
    return [
        "ros2",
        "launch",
        "ur_robot_driver",
        "ur_control.launch.py",
        f"ur_type:={ur_type}",
        f"robot_ip:={robot_ip}",
        f"reverse_ip:={reverse_ip}",
        "launch_rviz:=false",
        "initial_joint_controller:=forward_position_controller",
    ]


def build_teleop_command(
    robot_ip: str,
    reverse_ip: str,
    ur_type: str,
    input_type: str,
    gripper_type: str,
    joy_profile: str,
    mediapipe_input_topic: str,
) -> List[str]:
    cmd = [
        "ros2",
        "launch",
        "teleop_control_py",
        "control_system.launch.py",
        f"ur_type:={ur_type}",
        f"robot_ip:={robot_ip}",
        f"reverse_ip:={reverse_ip}",
        f"input_type:={input_type}",
        f"gripper_type:={gripper_type}",
        f"joy_profile:={joy_profile}",
        f"mediapipe_input_topic:={mediapipe_input_topic}",
        "enable_camera:=false",
    ]
    return cmd


def detect_joystick_devices() -> List[str]:
    candidates = list(Path("/dev/input/by-id").glob("*-event-joystick"))
    names = [path.name.replace("-event-joystick", "") for path in sorted(candidates)]
    return names


def _normalize_camera_source(source: Optional[str]) -> str:
    value = (source or "").strip().lower()
    if value == "oakd":
        return "oakd"
    if value in {"realsense", "rs"}:
        return "realsense"
    return value


def hardware_conflicts_for_collector(
    camera_process: Optional[str],
    collector_running: bool,
    global_camera_source: str,
    wrist_camera_source: str,
) -> List[str]:
    conflicts: List[str] = []
    occupied_sources = {
        _normalize_camera_source(global_camera_source),
        _normalize_camera_source(wrist_camera_source),
    }
    normalized_process = _normalize_camera_source(camera_process)
    if collector_running and normalized_process in occupied_sources:
        conflicts.append(f"采集节点正在占用 {normalized_process}")
    return conflicts


def collector_camera_occupancy(global_camera_source: str, wrist_camera_source: str) -> Dict[str, bool]:
    sources = {
        _normalize_camera_source(global_camera_source),
        _normalize_camera_source(wrist_camera_source),
    }
    return {
        "realsense": "realsense" in sources,
        "oakd": "oakd" in sources,
    }
