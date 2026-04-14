#!/usr/bin/env python3
"""GUI shared helpers for launch commands, defaults, and runtime status probes."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
from pathlib import Path
import re
import struct
import socket
from typing import Dict, List, Optional

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from ..device_manager import robot_profile_name_from_ur_type


@dataclass(frozen=True)
class GuiSettings:
    default_robot_ip: str
    default_reverse_ip: str
    ur_type: str
    default_input_type: str
    default_gripper_type: str
    default_joy_profile: str
    joy_profiles: List[str]
    default_mediapipe_input_topic: str
    default_mediapipe_camera: str
    default_mediapipe_camera_serial_number: str
    mediapipe_camera_options: List[str]
    realsense_d435_serial_number: str
    realsense_d455_serial_number: str
    default_collector_global_camera_model: str
    default_collector_global_camera_serial_number: str
    default_collector_wrist_camera_model: str
    default_collector_wrist_camera_serial_number: str
    default_inference_global_camera_source: str
    default_inference_global_camera_model: str
    default_inference_global_camera_serial_number: str
    default_inference_wrist_camera_source: str
    default_inference_wrist_camera_model: str
    default_inference_wrist_camera_serial_number: str
    default_inference_model_dir: str
    default_inference_env: str
    default_inference_task: str
    default_inference_embedding_path: str
    default_inference_device: str
    default_inference_hz: float
    collect_inference_action_logs: bool
    camera_driver_options: List[str]
    default_camera_driver: str
    default_global_camera_source: str
    default_wrist_camera_source: str
    realsense_enable_depth: bool
    oakd_enable_depth: bool
    default_hdf5_output_dir: str
    default_hdf5_filename: str

    home_joint_positions: List[float]


@dataclass(frozen=True)
class SdkCameraDevice:
    source: str
    model: str
    serial_number: str
    display_name: str

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


def get_repo_home_override_path(current_file: str | Path) -> Path:
    workspace_root = _workspace_root_from_file(current_file)
    return workspace_root / "src" / "teleop_control_py" / "config" / "home_overrides.yaml"


def get_installed_home_override_path() -> Optional[Path]:
    try:
        share_dir = Path(get_package_share_directory("teleop_control_py"))
    except PackageNotFoundError:
        return None
    return share_dir / "config" / "home_overrides.yaml"


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

    def _as_bool(value: object, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

    def _camera_enable_depth(key: str, default: bool = False) -> bool:
        if key in raw:
            return _as_bool(raw.get(key), default)
        return default

    default_global_camera_source = str(raw.get("default_global_camera_source", "realsense")).strip().lower() or "realsense"
    default_wrist_camera_source = str(raw.get("default_wrist_camera_source", "oakd")).strip().lower() or "oakd"
    default_mediapipe_camera = str(raw.get("default_mediapipe_camera", "d435")).strip().lower() or "d435"

    default_global_camera_model = str(
        raw.get(
            "default_collector_global_camera_model",
            "d455" if default_global_camera_source == "realsense" else default_global_camera_source,
        )
    ).strip().lower()
    if not default_global_camera_model:
        default_global_camera_model = "d455" if default_global_camera_source == "realsense" else default_global_camera_source or "d455"

    default_wrist_camera_model = str(
        raw.get(
            "default_collector_wrist_camera_model",
            "oakd" if default_wrist_camera_source == "oakd" else "d435",
        )
    ).strip().lower()
    if not default_wrist_camera_model:
        default_wrist_camera_model = "oakd" if default_wrist_camera_source == "oakd" else "d435"

    default_inference_global_camera_source = str(
        raw.get("default_inference_global_camera_source", default_global_camera_source)
    ).strip().lower() or default_global_camera_source
    default_inference_wrist_camera_source = str(
        raw.get("default_inference_wrist_camera_source", default_wrist_camera_source)
    ).strip().lower() or default_wrist_camera_source

    default_inference_global_camera_model = str(
        raw.get(
            "default_inference_global_camera_model",
            raw.get("default_collector_global_camera_model", default_global_camera_model),
        )
    ).strip().lower()
    if not default_inference_global_camera_model:
        default_inference_global_camera_model = (
            "d455"
            if default_inference_global_camera_source == "realsense"
            else default_inference_global_camera_source or default_global_camera_model
        )

    default_inference_wrist_camera_model = str(
        raw.get(
            "default_inference_wrist_camera_model",
            raw.get("default_collector_wrist_camera_model", default_wrist_camera_model),
        )
    ).strip().lower()
    if not default_inference_wrist_camera_model:
        default_inference_wrist_camera_model = (
            "oakd"
            if default_inference_wrist_camera_source == "oakd"
            else "d435"
        )

    return GuiSettings(
        default_robot_ip=str(raw.get("default_robot_ip", "192.168.1.211")),
        default_reverse_ip=str(raw.get("default_reverse_ip", "192.168.1.10")),
        ur_type=str(raw.get("ur_type", "ur5")),
        default_input_type=str(raw.get("default_input_type", "joy")),
        default_gripper_type=str(raw.get("default_gripper_type", "robotiq")),
        default_joy_profile=str(raw.get("default_joy_profile", "auto")),
        joy_profiles=[str(v) for v in raw.get("joy_profiles", ["auto", "xbox", "ps5", "generic"])],
        default_mediapipe_input_topic=str(raw.get("default_mediapipe_input_topic", "/d435/camera/color/image_raw")),
        default_mediapipe_camera=default_mediapipe_camera,
        default_mediapipe_camera_serial_number=str(
            raw.get(
                "default_mediapipe_camera_serial_number",
                raw.get("realsense_d435_serial_number", raw.get("realsense_d455_serial_number", "")),
            )
        ),
        mediapipe_camera_options=[str(v) for v in raw.get("mediapipe_camera_options", ["d435", "d455", "oakd", "camera"])],
        realsense_d435_serial_number=str(raw.get("realsense_d435_serial_number", "")),
        realsense_d455_serial_number=str(raw.get("realsense_d455_serial_number", "")),
        default_collector_global_camera_model=default_global_camera_model,
        default_collector_global_camera_serial_number=str(
            raw.get(
                "default_collector_global_camera_serial_number",
                raw.get("realsense_d455_serial_number", ""),
            )
        ),
        default_collector_wrist_camera_model=default_wrist_camera_model,
        default_collector_wrist_camera_serial_number=str(
            raw.get("default_collector_wrist_camera_serial_number", "")
        ),
        default_inference_global_camera_source=default_inference_global_camera_source,
        default_inference_global_camera_model=default_inference_global_camera_model,
        default_inference_global_camera_serial_number=str(
            raw.get(
                "default_inference_global_camera_serial_number",
                raw.get(
                    "default_collector_global_camera_serial_number",
                    raw.get("realsense_d455_serial_number", ""),
                ),
            )
        ),
        default_inference_wrist_camera_source=default_inference_wrist_camera_source,
        default_inference_wrist_camera_model=default_inference_wrist_camera_model,
        default_inference_wrist_camera_serial_number=str(
            raw.get(
                "default_inference_wrist_camera_serial_number",
                raw.get("default_collector_wrist_camera_serial_number", ""),
            )
        ),
        default_inference_model_dir=str(raw.get("default_inference_model_dir", "")),
        default_inference_env=str(raw.get("default_inference_env", "")),
        default_inference_task=str(raw.get("default_inference_task", "")),
        default_inference_embedding_path=str(raw.get("default_inference_embedding_path", "")),
        default_inference_device=str(raw.get("default_inference_device", "cuda")).strip().lower() or "cuda",
        default_inference_hz=float(raw.get("default_inference_hz", 10.0)),
        collect_inference_action_logs=_as_bool(raw.get("collect_inference_action_logs", False), False),
        camera_driver_options=[str(v) for v in raw.get("camera_driver_options", ["realsense", "oakd"])],
        default_camera_driver=str(raw.get("default_camera_driver", "realsense")),
        default_global_camera_source=default_global_camera_source,
        default_wrist_camera_source=default_wrist_camera_source,
        realsense_enable_depth=_camera_enable_depth("realsense_enable_depth"),
        oakd_enable_depth=_camera_enable_depth("oakd_enable_depth"),
        default_hdf5_output_dir=str(output_dir),
        default_hdf5_filename=str(raw.get("default_hdf5_filename", "libero_demos.hdf5")),
        home_joint_positions=[float(v) for v in raw.get("home_joint_positions", [])],
    )


def camera_enable_depth(settings: GuiSettings, camera_source: str) -> bool:
    normalized = str(camera_source).strip().lower()
    if normalized in {"realsense", "rs"}:
        return bool(settings.realsense_enable_depth)
    if normalized == "oakd":
        return bool(settings.oakd_enable_depth)
    return False


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


def load_home_override(current_file: str | Path, robot_profile: str) -> List[float]:
    try:
        import yaml  # type: ignore
    except Exception:
        return []

    normalized_profile = str(robot_profile).strip()
    if not normalized_profile:
        return []

    candidate_paths = [get_repo_home_override_path(current_file)]
    installed = get_installed_home_override_path()
    if installed is not None:
        candidate_paths.append(installed)

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content = yaml.safe_load(handle) or {}
        except Exception:
            continue
        if not isinstance(content, dict):
            continue
        block = content.get("home_overrides", {})
        if not isinstance(block, dict):
            continue
        profiles = block.get("profiles", {})
        if not isinstance(profiles, dict):
            continue
        entry = profiles.get(normalized_profile, {})
        if not isinstance(entry, dict):
            continue
        try:
            joints = [float(value) for value in list(entry.get("home_joint_positions", []))[:6]]
        except Exception:
            joints = []
        if len(joints) == 6:
            return joints

    return []


def save_home_override(current_file: str | Path, robot_profile: str, joints: List[float]) -> Path:
    import yaml  # type: ignore

    normalized_profile = str(robot_profile).strip()
    if not normalized_profile:
        raise ValueError("robot_profile is required")

    normalized_joints = [float(value) for value in list(joints)[:6]]
    if len(normalized_joints) != 6:
        raise ValueError("home_joint_positions must contain 6 values")

    config_path = get_repo_home_override_path(current_file)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
    else:
        content = {}

    if not isinstance(content, dict):
        content = {}

    home_overrides = content.setdefault("home_overrides", {})
    if not isinstance(home_overrides, dict):
        home_overrides = {}
        content["home_overrides"] = home_overrides

    profiles = home_overrides.setdefault("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
        home_overrides["profiles"] = profiles

    profiles[normalized_profile] = {
        "home_joint_positions": normalized_joints,
    }

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

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "unknown"
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


def _extract_camera_model(*texts: str, fallback: str = "camera") -> str:
    combined = " ".join(str(text or "") for text in texts).lower()
    match = re.search(r"(d\d{3})", combined)
    if match:
        return match.group(1)
    if "l515" in combined:
        return "l515"
    if "oak" in combined:
        return "oakd"
    return fallback


def _discover_realsense_sdk_cameras() -> List[SdkCameraDevice]:
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception:
        return []

    devices: List[SdkCameraDevice] = []
    try:
        context = rs.context()
        queried = context.query_devices()
    except Exception:
        return []

    for device in queried:
        def _device_info(key) -> str:
            try:
                value = device.get_info(key)
                return str(value).strip()
            except Exception:
                return ""

        name = _device_info(rs.camera_info.name)
        serial = _device_info(rs.camera_info.serial_number)
        product_line = _device_info(rs.camera_info.product_line)
        model = _extract_camera_model(name, product_line, serial, fallback="realsense")
        pretty_name = (name or "RealSense").strip()
        display = f"{pretty_name} (SN: {serial})" if serial else f"{pretty_name} (SN: auto)"
        devices.append(
            SdkCameraDevice(
                source="realsense",
                model=model,
                serial_number=serial,
                display_name=display,
            )
        )
    return devices


def _discover_oak_sdk_cameras() -> List[SdkCameraDevice]:
    try:
        import depthai as dai  # type: ignore
    except Exception:
        return []

    try:
        getter = getattr(dai.Device, "getAllAvailableDevices", None)
        if not callable(getter):
            return []
        queried = list(getter())
    except Exception:
        return []

    devices: List[SdkCameraDevice] = []
    for info in queried:
        serial = ""
        for attr in ("getMxId", "mxid", "mx_id", "serial"):
            value = getattr(info, attr, None)
            try:
                if callable(value):
                    value = value()
            except Exception:
                value = None
            if value:
                serial = str(value).strip()
                break

        name = ""
        for attr in ("name", "getName"):
            value = getattr(info, attr, None)
            try:
                if callable(value):
                    value = value()
            except Exception:
                value = None
            if value:
                name = str(value).strip()
                break
        if not name:
            name = "OAK-D"

        model = _extract_camera_model(name, serial, fallback="oakd")
        display = f"{name} (SN: {serial})" if serial else f"{name} (SN: auto)"
        devices.append(
            SdkCameraDevice(
                source="oakd",
                model=model,
                serial_number=serial,
                display_name=display,
            )
        )
    return devices


def discover_sdk_cameras() -> List[SdkCameraDevice]:
    candidates = _discover_realsense_sdk_cameras() + _discover_oak_sdk_cameras()
    deduped: List[SdkCameraDevice] = []
    seen: set[tuple[str, str, str]] = set()
    for camera in candidates:
        key = (
            str(camera.source).strip().lower(),
            str(camera.serial_number).strip(),
            str(camera.model).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(camera)
    deduped.sort(key=lambda item: (item.source, item.model, item.serial_number))
    return deduped


def build_camera_driver_command(camera_driver: str, *, enable_depth: bool = False) -> List[str]:
    key = camera_driver.strip().lower()
    if key == "realsense":
        align_value = "true" if enable_depth else "false"
        enable_depth_value = "true" if enable_depth else "false"
        return [
            "ros2",
            "launch",
            "realsense2_camera",
            "rs_launch.py",
            f"enable_depth:={enable_depth_value}",
            f"align_depth.enable:={align_value}",
        ]
    if key == "oakd":
        if not enable_depth:
            return [
                "ros2",
                "launch",
                "depthai_examples",
                "rgb_stereo_node.launch.py",
                "useDepth:=false",
                "useVideo:=true",
                "usePreview:=false",
            ]
        return [
            "ros2",
            "launch",
            "depthai_examples",
            "stereo_inertial_node.launch.py",
            "depth_aligned:=true",
            "enableRviz:=false",
            "enableSpatialDetection:=false",
            "syncNN:=false",
        ]
    raise ValueError(f"Unsupported camera driver: {camera_driver}")


def build_robot_driver_command(
    robot_ip: str,
    reverse_ip: str,
    ur_type: str,
    gripper_type: str,
    robot_profile: Optional[str] = None,
) -> List[str]:
    resolved_profile = str(robot_profile).strip() if robot_profile is not None else ""
    if not resolved_profile:
        resolved_profile = robot_profile_name_from_ur_type(ur_type)
    return [
        "ros2",
        "launch",
        "teleop_control_py",
        "control_system.launch.py",
        f"ur_type:={ur_type}",
        f"robot_profile:={resolved_profile}",
        f"robot_ip:={robot_ip}",
        f"reverse_ip:={reverse_ip}",
        f"gripper_type:={gripper_type}",
        "launch_rviz:=false",
        "launch_moveit_rviz:=false",
        "launch_servo:=true",
        "enable_moveit:=true",
        "launch_teleop_node:=false",
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
    mediapipe_depth_topic: str,
    mediapipe_camera_info_topic: str,
    mediapipe_camera_driver: str,
    mediapipe_camera_serial_number: str,
    mediapipe_enable_depth: bool,
    mediapipe_show_debug_window: bool,
    mediapipe_use_sdk_camera: bool,
    robot_profile: Optional[str] = None,
) -> List[str]:
    def _append_optional_launch_arg(items: List[str], name: str, value: str) -> None:
        normalized = str(value).strip()
        if not normalized:
            return
        items.append(f"{name}:={normalized}")

    resolved_profile = str(robot_profile).strip() if robot_profile is not None else ""
    if not resolved_profile:
        resolved_profile = robot_profile_name_from_ur_type(ur_type)
    resolved_input_type = str(input_type).strip() or "joy"
    resolved_gripper_type = str(gripper_type).strip() or "robotiq"
    resolved_joy_profile = str(joy_profile).strip() or "auto"
    resolved_mediapipe_input_topic = str(mediapipe_input_topic).strip() or "/camera/camera/color/image_raw"
    cmd = [
        "ros2",
        "launch",
        "teleop_control_py",
        "control_system.launch.py",
        f"ur_type:={ur_type}",
        f"robot_profile:={resolved_profile}",
        f"robot_ip:={robot_ip}",
        f"reverse_ip:={reverse_ip}",
        f"input_type:={resolved_input_type}",
        f"gripper_type:={resolved_gripper_type}",
        f"joy_profile:={resolved_joy_profile}",
        f"mediapipe_input_topic:={resolved_mediapipe_input_topic}",
        f"mediapipe_enable_depth:={'true' if mediapipe_enable_depth else 'false'}",
        f"mediapipe_show_debug_window:={'true' if mediapipe_show_debug_window else 'false'}",
        f"mediapipe_use_sdk_camera:={'true' if mediapipe_use_sdk_camera else 'false'}",
    ]
    _append_optional_launch_arg(cmd, "mediapipe_depth_topic", mediapipe_depth_topic)
    _append_optional_launch_arg(cmd, "mediapipe_camera_info_topic", mediapipe_camera_info_topic)
    _append_optional_launch_arg(cmd, "mediapipe_camera_driver", mediapipe_camera_driver)
    _append_optional_launch_arg(cmd, "mediapipe_camera_serial_number", mediapipe_camera_serial_number)
    return cmd


def detect_joystick_devices() -> List[str]:
    candidates = list(Path("/dev/input/by-id").glob("*-event-joystick"))
    names = [path.name.replace("-event-joystick", "") for path in sorted(candidates)]
    return names
