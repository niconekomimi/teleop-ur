"""Robot profile loading utilities for backend configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
except Exception:  # noqa: BLE001
    PackageNotFoundError = Exception

    def get_package_share_directory(_package_name: str) -> str:
        raise PackageNotFoundError()


DEFAULT_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
DEFAULT_HOME_JOINT_POSITIONS = (
    1.524178,
    -2.100060,
    1.864580,
    -1.345048,
    -1.575888,
    1.528195,
)
DEFAULT_ROBOT_PROFILE_NAME = "ur_servo_ur5"


@dataclass(frozen=True)
class RobotTopics:
    joint_states: str = "/joint_states"
    tool_pose: str = "/tcp_pose_broadcaster/pose"
    servo_twist: str = "/servo_node/delta_twist_cmds"
    gripper_state: str = "/gripper/state"
    home_joint_trajectory: str = "/scaled_joint_trajectory_controller/joint_trajectory"


@dataclass(frozen=True)
class RobotServices:
    start_servo: str = "/servo_node/start_servo"
    controller_manager_ns: str = "/controller_manager"


@dataclass(frozen=True)
class RobotControllers:
    teleop: str = "forward_position_controller"
    trajectory: str = "scaled_joint_trajectory_controller"


@dataclass(frozen=True)
class RobotHome:
    duration_sec: float = 3.0
    joint_positions: tuple[float, ...] = DEFAULT_HOME_JOINT_POSITIONS


@dataclass(frozen=True)
class RobotHomeZone:
    translation_min_m: tuple[float, float, float] = (0.04, 0.04, 0.04)
    translation_max_m: tuple[float, float, float] = (0.08, 0.08, 0.08)
    rotation_min_deg: tuple[float, float, float] = (5.0, 5.0, 5.0)
    rotation_max_deg: tuple[float, float, float] = (10.0, 10.0, 10.0)
    timeout_sec: float = 30.0
    rate_hz: float = 60.0
    max_linear_vel: float = 0.50
    max_angular_vel: float = 1.5
    linear_gain: float = 2.4
    angular_gain: float = 2.4
    position_tolerance_m: float = 0.005
    rotation_tolerance_deg: float = 6.0


@dataclass(frozen=True)
class RobotRobotiqGripper:
    state_topic: str = "/gripper/state"
    command_interface: str = "position_action"
    confidence_topic: str = "/robotiq_2f_gripper/confidence_command"
    binary_topic: str = "/robotiq_2f_gripper/binary_command"
    action_name: str = "/robotiq_2f_gripper_action"
    binary_threshold: float = 0.5
    open_ratio: float = 0.9
    max_open_position_m: float = 0.142
    target_speed: float = 1.0
    target_force: float = 0.5


@dataclass(frozen=True)
class RobotQbSoftHandGripper:
    state_topic: str = "/gripper/cmd"
    service_name: str = "/qbsofthand_control_node/set_closure"
    duration_sec: float = 0.3
    speed_ratio: float = 1.0


@dataclass(frozen=True)
class RobotGrippers:
    default_type: str = "robotiq"
    command_topic: str = "/gripper/cmd"
    command_delta: float = 0.01
    quantization_levels: int = 10
    robotiq: RobotRobotiqGripper = RobotRobotiqGripper()
    qbsofthand: RobotQbSoftHandGripper = RobotQbSoftHandGripper()


@dataclass(frozen=True)
class RobotProfile:
    name: str
    backend: str = "ur_servo"
    arm_model: str = "ur5"
    target_frame_id: str = "base"
    joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES
    topics: RobotTopics = RobotTopics()
    services: RobotServices = RobotServices()
    controllers: RobotControllers = RobotControllers()
    grippers: RobotGrippers = RobotGrippers()
    home: RobotHome = RobotHome()
    home_zone: RobotHomeZone = RobotHomeZone()


DEFAULT_ROBOT_PROFILE = RobotProfile(name=DEFAULT_ROBOT_PROFILE_NAME)


def _workspace_root_from_file(current_file: str | Path) -> Path:
    current_path = Path(current_file).resolve()
    for candidate in [current_path] + list(current_path.parents):
        config_path = candidate / "src" / "teleop_control_py" / "config" / "robot_profiles.yaml"
        if config_path.exists():
            return candidate

    try:
        share_dir = Path(get_package_share_directory("teleop_control_py"))
        return share_dir.parents[2]
    except PackageNotFoundError:
        pass

    return current_path.parent


def default_robot_profiles_path(current_file: str | Path | None = None) -> Path:
    if current_file is None:
        current_file = __file__
    workspace_root = _workspace_root_from_file(current_file)
    return workspace_root / "src" / "teleop_control_py" / "config" / "robot_profiles.yaml"


def robot_profile_name_from_ur_type(ur_type: str) -> str:
    normalized = str(ur_type).strip().lower()
    if not normalized:
        return DEFAULT_ROBOT_PROFILE_NAME
    return f"ur_servo_{normalized}"


def _optional_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    return yaml


def _load_profiles_file(profiles_file: str | Path | None = None) -> tuple[str, dict[str, Any]]:
    yaml = _optional_yaml()
    if yaml is None:
        return DEFAULT_ROBOT_PROFILE_NAME, {}

    path = Path(profiles_file) if profiles_file else default_robot_profiles_path()
    if not path.exists():
        return DEFAULT_ROBOT_PROFILE_NAME, {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return DEFAULT_ROBOT_PROFILE_NAME, {}

    if not isinstance(data, dict):
        return DEFAULT_ROBOT_PROFILE_NAME, {}

    default_name = str(data.get("default", DEFAULT_ROBOT_PROFILE_NAME)).strip() or DEFAULT_ROBOT_PROFILE_NAME
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
    return default_name, profiles


def _str_value(raw: dict[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    return str(value).strip() or default


def _float_value(raw: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(raw.get(key, default))
    except Exception:
        return float(default)


def _int_value(raw: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(raw.get(key, default))
    except Exception:
        return int(default)


def _float_tuple(values: Any, default: tuple[float, ...], size: int | None = None) -> tuple[float, ...]:
    try:
        normalized = tuple(float(value) for value in list(values))
    except Exception:
        return tuple(default)
    if size is not None and len(normalized) != size:
        return tuple(default)
    if not normalized:
        return tuple(default)
    return normalized


def _str_tuple(values: Any, default: tuple[str, ...], size: int | None = None) -> tuple[str, ...]:
    try:
        normalized = tuple(str(value).strip() for value in list(values))
    except Exception:
        return tuple(default)
    if size is not None and len(normalized) != size:
        return tuple(default)
    if not normalized:
        return tuple(default)
    return normalized


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalized_gripper_type(value: str, default: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"robotiq", "qbsofthand"}:
        return normalized
    return default


def _build_profile(name: str, raw: dict[str, Any]) -> RobotProfile:
    base = DEFAULT_ROBOT_PROFILE
    topics_raw = _mapping(raw.get("topics"))
    services_raw = _mapping(raw.get("services"))
    controllers_raw = _mapping(raw.get("controllers"))
    grippers_raw = _mapping(raw.get("grippers"))
    robotiq_raw = _mapping(grippers_raw.get("robotiq"))
    qbsofthand_raw = _mapping(grippers_raw.get("qbsofthand"))
    home_raw = _mapping(raw.get("home"))
    home_zone_raw = _mapping(raw.get("home_zone"))
    resolved_topics = RobotTopics(
        joint_states=_str_value(topics_raw, "joint_states", base.topics.joint_states),
        tool_pose=_str_value(topics_raw, "tool_pose", base.topics.tool_pose),
        servo_twist=_str_value(topics_raw, "servo_twist", base.topics.servo_twist),
        gripper_state=_str_value(topics_raw, "gripper_state", base.topics.gripper_state),
        home_joint_trajectory=_str_value(
            topics_raw,
            "home_joint_trajectory",
            base.topics.home_joint_trajectory,
        ),
    )
    resolved_grippers = RobotGrippers(
        default_type=_normalized_gripper_type(
            grippers_raw.get("default_type", base.grippers.default_type),
            base.grippers.default_type,
        ),
        command_topic=_str_value(grippers_raw, "command_topic", base.grippers.command_topic),
        command_delta=_float_value(grippers_raw, "command_delta", base.grippers.command_delta),
        quantization_levels=_int_value(
            grippers_raw,
            "quantization_levels",
            base.grippers.quantization_levels,
        ),
        robotiq=RobotRobotiqGripper(
            state_topic=_str_value(robotiq_raw, "state_topic", resolved_topics.gripper_state),
            command_interface=_str_value(
                robotiq_raw,
                "command_interface",
                base.grippers.robotiq.command_interface,
            ),
            confidence_topic=_str_value(
                robotiq_raw,
                "confidence_topic",
                base.grippers.robotiq.confidence_topic,
            ),
            binary_topic=_str_value(
                robotiq_raw,
                "binary_topic",
                base.grippers.robotiq.binary_topic,
            ),
            action_name=_str_value(
                robotiq_raw,
                "action_name",
                base.grippers.robotiq.action_name,
            ),
            binary_threshold=_float_value(
                robotiq_raw,
                "binary_threshold",
                base.grippers.robotiq.binary_threshold,
            ),
            open_ratio=_float_value(
                robotiq_raw,
                "open_ratio",
                base.grippers.robotiq.open_ratio,
            ),
            max_open_position_m=_float_value(
                robotiq_raw,
                "max_open_position_m",
                base.grippers.robotiq.max_open_position_m,
            ),
            target_speed=_float_value(
                robotiq_raw,
                "target_speed",
                base.grippers.robotiq.target_speed,
            ),
            target_force=_float_value(
                robotiq_raw,
                "target_force",
                base.grippers.robotiq.target_force,
            ),
        ),
        qbsofthand=RobotQbSoftHandGripper(
            state_topic=_str_value(
                qbsofthand_raw,
                "state_topic",
                base.grippers.qbsofthand.state_topic,
            ),
            service_name=_str_value(
                qbsofthand_raw,
                "service_name",
                base.grippers.qbsofthand.service_name,
            ),
            duration_sec=_float_value(
                qbsofthand_raw,
                "duration_sec",
                base.grippers.qbsofthand.duration_sec,
            ),
            speed_ratio=_float_value(
                qbsofthand_raw,
                "speed_ratio",
                base.grippers.qbsofthand.speed_ratio,
            ),
        ),
    )

    return RobotProfile(
        name=name,
        backend=_str_value(raw, "backend", base.backend),
        arm_model=_str_value(raw, "arm_model", base.arm_model),
        target_frame_id=_str_value(raw, "target_frame_id", base.target_frame_id),
        joint_names=_str_tuple(raw.get("joint_names"), base.joint_names),
        topics=resolved_topics,
        services=RobotServices(
            start_servo=_str_value(services_raw, "start_servo", base.services.start_servo),
            controller_manager_ns=_str_value(
                services_raw,
                "controller_manager_ns",
                base.services.controller_manager_ns,
            ),
        ),
        controllers=RobotControllers(
            teleop=_str_value(controllers_raw, "teleop", base.controllers.teleop),
            trajectory=_str_value(controllers_raw, "trajectory", base.controllers.trajectory),
        ),
        grippers=resolved_grippers,
        home=RobotHome(
            duration_sec=_float_value(home_raw, "duration_sec", base.home.duration_sec),
            joint_positions=_float_tuple(home_raw.get("joint_positions"), base.home.joint_positions),
        ),
        home_zone=RobotHomeZone(
            translation_min_m=_float_tuple(
                home_zone_raw.get("translation_min_m"),
                base.home_zone.translation_min_m,
                size=3,
            ),
            translation_max_m=_float_tuple(
                home_zone_raw.get("translation_max_m"),
                base.home_zone.translation_max_m,
                size=3,
            ),
            rotation_min_deg=_float_tuple(
                home_zone_raw.get("rotation_min_deg"),
                base.home_zone.rotation_min_deg,
                size=3,
            ),
            rotation_max_deg=_float_tuple(
                home_zone_raw.get("rotation_max_deg"),
                base.home_zone.rotation_max_deg,
                size=3,
            ),
            timeout_sec=_float_value(home_zone_raw, "timeout_sec", base.home_zone.timeout_sec),
            rate_hz=_float_value(home_zone_raw, "rate_hz", base.home_zone.rate_hz),
            max_linear_vel=_float_value(
                home_zone_raw,
                "max_linear_vel",
                base.home_zone.max_linear_vel,
            ),
            max_angular_vel=_float_value(
                home_zone_raw,
                "max_angular_vel",
                base.home_zone.max_angular_vel,
            ),
            linear_gain=_float_value(home_zone_raw, "linear_gain", base.home_zone.linear_gain),
            angular_gain=_float_value(home_zone_raw, "angular_gain", base.home_zone.angular_gain),
            position_tolerance_m=_float_value(
                home_zone_raw,
                "position_tolerance_m",
                base.home_zone.position_tolerance_m,
            ),
            rotation_tolerance_deg=_float_value(
                home_zone_raw,
                "rotation_tolerance_deg",
                base.home_zone.rotation_tolerance_deg,
            ),
        ),
    )


def available_robot_profiles(profiles_file: str | Path | None = None) -> list[str]:
    default_name, profiles = _load_profiles_file(profiles_file)
    names = sorted(str(name).strip() for name in profiles.keys() if str(name).strip())
    if DEFAULT_ROBOT_PROFILE_NAME not in names:
        names.insert(0, DEFAULT_ROBOT_PROFILE_NAME)
    if default_name not in names:
        names.insert(0, default_name)
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def load_robot_profile(
    profile_name: str | None = None,
    profiles_file: str | Path | None = None,
) -> RobotProfile:
    default_name, profiles = _load_profiles_file(profiles_file)
    resolved_name = str(profile_name or "").strip() or default_name or DEFAULT_ROBOT_PROFILE_NAME
    raw = profiles.get(resolved_name)
    if not isinstance(raw, dict):
        if resolved_name == DEFAULT_ROBOT_PROFILE_NAME:
            return DEFAULT_ROBOT_PROFILE
        fallback_raw = profiles.get(default_name)
        if isinstance(fallback_raw, dict):
            return _build_profile(default_name, fallback_raw)
        return DEFAULT_ROBOT_PROFILE
    return _build_profile(resolved_name, raw)
