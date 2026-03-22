"""Thin GUI application service for process and ROS worker orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from teleop_control_py.core.inference_worker import build_robot_state_vector
from teleop_control_py.gui.runtime import ProcessManager
from teleop_control_py.gui.support import build_robot_driver_command, build_teleop_command

from .ros_worker import ROS2Worker


@dataclass(frozen=True)
class RosWorkerCallbacks:
    log: Callable[[str], None]
    demo_status: Callable[[str], None]
    record_stats: Callable[[int, str, float], None]
    recording_state: Callable[[bool], None]
    home_zone_active: Callable[[bool], None]
    homing_active: Callable[[bool], None]
    commander_result: Callable[[str], None]


@dataclass(frozen=True)
class RosWorkerConfig:
    robot_profile: str
    gripper_topic: str
    home_joint_positions: Sequence[float]
    home_duration_sec: float
    joint_names: Sequence[str]
    trajectory_topic: str
    teleop_controller: str
    trajectory_controller: str
    home_zone_translation_min_m: Sequence[float]
    home_zone_translation_max_m: Sequence[float]
    home_zone_rotation_min_deg: Sequence[float]
    home_zone_rotation_max_deg: Sequence[float]
    inference_gripper_type: str
    inference_control_hz: float = 50.0


@dataclass(frozen=True)
class RobotDriverLaunchConfig:
    robot_ip: str
    reverse_ip: str
    ur_type: str
    gripper_type: str
    robot_profile: str


@dataclass(frozen=True)
class TeleopLaunchConfig:
    robot_ip: str
    reverse_ip: str
    ur_type: str
    input_type: str
    gripper_type: str
    joy_profile: str
    mediapipe_input_topic: str
    mediapipe_depth_topic: str
    mediapipe_camera_info_topic: str
    mediapipe_camera_driver: str
    mediapipe_camera_serial_number: str
    mediapipe_enable_depth: bool
    mediapipe_show_debug_window: bool
    mediapipe_use_sdk_camera: bool
    robot_profile: str


@dataclass(frozen=True)
class CollectorLaunchConfig:
    robot_profile: str
    output_path: str
    global_camera_source: str
    wrist_camera_source: str
    global_camera_serial_number: str
    wrist_camera_serial_number: str
    global_camera_enable_depth: bool
    wrist_camera_enable_depth: bool
    end_effector_type: str


class GuiAppService:
    """Owns GUI-side runtime helpers that are not UI rendering concerns."""

    def __init__(self, process_manager: ProcessManager | Any, collector_gripper_topic: str) -> None:
        self._process_manager = getattr(process_manager, "process_manager", process_manager)
        self._collector_gripper_topic = str(collector_gripper_topic).strip()
        self._ros_worker: Optional[ROS2Worker] = None
        self._preview_robot_state_callback: Optional[Callable[[str], None]] = None
        self._preview_record_stats_callback: Optional[Callable[[int, str, float], None]] = None
        self._preview_bound_worker: Optional[ROS2Worker] = None

    @property
    def ros_worker(self) -> Optional[ROS2Worker]:
        return self._ros_worker

    def has_ros_worker(self) -> bool:
        return self._ros_worker is not None

    def _bind_preview_callbacks(self, worker: ROS2Worker) -> None:
        if self._preview_robot_state_callback is None or self._preview_record_stats_callback is None:
            return
        if self._preview_bound_worker is worker:
            return
        worker.robot_state_str_signal.connect(self._preview_robot_state_callback)
        worker.record_stats_signal.connect(self._preview_record_stats_callback)
        self._preview_bound_worker = worker

    def attach_preview_callbacks(
        self,
        robot_state_callback: Callable[[str], None],
        record_stats_callback: Callable[[int, str, float], None],
    ) -> None:
        if (
            self._preview_robot_state_callback is not robot_state_callback
            or self._preview_record_stats_callback is not record_stats_callback
        ):
            self._preview_bound_worker = None
        self._preview_robot_state_callback = robot_state_callback
        self._preview_record_stats_callback = record_stats_callback
        worker = self._ros_worker
        if worker is not None:
            self._bind_preview_callbacks(worker)

    def current_robot_state(self) -> Optional[dict]:
        worker = self._ros_worker
        if worker is None:
            return None
        state = getattr(worker, "robot_state", None)
        if not isinstance(state, dict):
            return None
        snapshot = {}
        for key, value in state.items():
            if isinstance(value, list):
                snapshot[key] = list(value)
            else:
                snapshot[key] = value
        return snapshot

    def current_robot_state_vector(self):
        return build_robot_state_vector(self.current_robot_state())

    def current_joint_positions(self) -> list[float]:
        state = self.current_robot_state() or {}
        joints = state.get("joints", [])
        return [float(value) for value in joints] if isinstance(joints, list) else []

    def is_recording(self) -> bool:
        worker = self._ros_worker
        return bool(worker is not None and getattr(worker, "is_recording", False))

    def ensure_ros_worker(self, config: RosWorkerConfig, callbacks: RosWorkerCallbacks) -> ROS2Worker:
        worker = self._ros_worker
        if worker is None:
            worker = ROS2Worker(
                gripper_topic=config.gripper_topic or self._collector_gripper_topic,
                robot_profile=config.robot_profile,
            )
            worker.log_signal.connect(callbacks.log)
            worker.demo_status_signal.connect(callbacks.demo_status)
            worker.record_stats_signal.connect(callbacks.record_stats)
            worker.recording_state_signal.connect(callbacks.recording_state)
            worker.home_zone_active_signal.connect(callbacks.home_zone_active)
            worker.homing_active_signal.connect(callbacks.homing_active)
            worker.commander_result_signal.connect(callbacks.commander_result)
            self._bind_preview_callbacks(worker)
            worker.start()
            self._ros_worker = worker

        self.apply_ros_worker_config(config)
        self._bind_preview_callbacks(worker)
        return worker

    def apply_ros_worker_config(self, config: RosWorkerConfig) -> Optional[ROS2Worker]:
        worker = self._ros_worker
        if worker is None:
            return None

        worker.set_robot_profile(config.robot_profile)
        worker.set_home_config(
            joint_positions=list(config.home_joint_positions),
            duration_sec=float(config.home_duration_sec),
            joint_names=list(config.joint_names),
            trajectory_topic=config.trajectory_topic,
            teleop_controller=config.teleop_controller,
            trajectory_controller=config.trajectory_controller,
            home_zone_translation_min_m=list(config.home_zone_translation_min_m),
            home_zone_translation_max_m=list(config.home_zone_translation_max_m),
            home_zone_rotation_min_deg=list(config.home_zone_rotation_min_deg),
            home_zone_rotation_max_deg=list(config.home_zone_rotation_max_deg),
        )
        worker.set_inference_control_config(
            gripper_type=config.inference_gripper_type,
            control_hz=float(config.inference_control_hz),
        )
        return worker

    def stop_ros_worker(self) -> None:
        worker = self._ros_worker
        self._ros_worker = None
        self._preview_bound_worker = None
        if worker is None:
            return
        worker.stop()

    def cancel_home_zone(self) -> bool:
        worker = self._ros_worker
        if worker is None:
            return False
        worker.call_cancel_home_zone()
        return True

    def go_home(self) -> bool:
        worker = self._ros_worker
        if worker is None:
            return False
        worker.call_go_home()
        return True

    def go_home_zone(self) -> bool:
        worker = self._ros_worker
        if worker is None:
            return False
        worker.call_go_home_zone()
        return True

    def start_record(self) -> bool:
        worker = self._ros_worker
        if worker is None:
            return False
        worker.call_start_record()
        return True

    def stop_record(self) -> bool:
        worker = self._ros_worker
        if worker is None:
            return False
        worker.call_stop_record()
        return True

    def discard_last_demo(self) -> bool:
        worker = self._ros_worker
        if worker is None:
            return False
        worker.call_discard_last_demo()
        return True

    def set_home_from_current(self) -> Optional[list[float]]:
        worker = self._ros_worker
        if worker is None:
            return None
        joints = self.current_joint_positions()
        if len(joints) != 6:
            return None
        worker.call_set_home_from_current()
        return joints

    def inference_execution_enabled(self) -> bool:
        worker = self._ros_worker
        return bool(worker is not None and worker.inference_execution_enabled())

    def enable_inference_execution(
        self,
        *,
        ros_worker_config: RosWorkerConfig,
        ros_worker_callbacks: RosWorkerCallbacks,
    ) -> bool:
        worker = self.ensure_ros_worker(ros_worker_config, ros_worker_callbacks)
        worker.set_inference_execution_enabled(True)
        return True

    def disable_inference_execution(self) -> bool:
        worker = self._ros_worker
        if worker is None:
            return False
        worker.set_inference_execution_enabled(False)
        return True

    def emergency_stop_inference_execution(self) -> bool:
        worker = self._ros_worker
        if worker is None:
            return False
        worker.emergency_stop_inference()
        return True

    def update_inference_action_command(self, action) -> bool:
        worker = self._ros_worker
        if worker is None:
            return False
        worker.update_inference_action_command(action)
        return True

    def ros_worker_required(self, *, preview_running: bool, inference_running: bool) -> bool:
        return bool(
            preview_running
            or inference_running
            or self._process_manager.is_running("data_collector")
            or self._process_manager.is_running("robot_driver")
            or self._process_manager.is_running("teleop")
        )

    def stop_ros_worker_if_unused(self, *, preview_running: bool, inference_running: bool) -> bool:
        if self._ros_worker is None:
            return False
        if self.ros_worker_required(
            preview_running=preview_running,
            inference_running=inference_running,
        ):
            return False
        self.stop_ros_worker()
        return True

    def start_robot_driver(
        self,
        config: RobotDriverLaunchConfig,
        *,
        ros_worker_config: Optional[RosWorkerConfig] = None,
        ros_worker_callbacks: Optional[RosWorkerCallbacks] = None,
    ) -> bool:
        cmd = build_robot_driver_command(
            robot_ip=config.robot_ip,
            reverse_ip=config.reverse_ip,
            ur_type=config.ur_type,
            gripper_type=config.gripper_type,
            robot_profile=config.robot_profile,
        )
        if not self._process_manager.run_subprocess("robot_driver", cmd):
            return False
        if ros_worker_config is not None and ros_worker_callbacks is not None:
            self.ensure_ros_worker(ros_worker_config, ros_worker_callbacks)
        return True

    def stop_robot_driver(self) -> None:
        self._process_manager.kill_subprocess("robot_driver")

    def start_teleop(
        self,
        config: TeleopLaunchConfig,
        *,
        ros_worker_config: Optional[RosWorkerConfig] = None,
        ros_worker_callbacks: Optional[RosWorkerCallbacks] = None,
    ) -> bool:
        cmd = build_teleop_command(
            robot_ip=config.robot_ip,
            reverse_ip=config.reverse_ip,
            ur_type=config.ur_type,
            input_type=config.input_type,
            gripper_type=config.gripper_type,
            joy_profile=config.joy_profile,
            mediapipe_input_topic=config.mediapipe_input_topic,
            mediapipe_depth_topic=config.mediapipe_depth_topic,
            mediapipe_camera_info_topic=config.mediapipe_camera_info_topic,
            mediapipe_camera_driver=config.mediapipe_camera_driver,
            mediapipe_camera_serial_number=config.mediapipe_camera_serial_number,
            mediapipe_enable_depth=bool(config.mediapipe_enable_depth),
            mediapipe_show_debug_window=bool(config.mediapipe_show_debug_window),
            mediapipe_use_sdk_camera=bool(config.mediapipe_use_sdk_camera),
            robot_profile=config.robot_profile,
        )
        if not self._process_manager.run_subprocess("teleop", cmd):
            return False
        if ros_worker_config is not None and ros_worker_callbacks is not None:
            self.ensure_ros_worker(ros_worker_config, ros_worker_callbacks)
        return True

    def stop_teleop(self) -> None:
        self._process_manager.kill_subprocess("teleop")

    def start_data_collector(
        self,
        config: CollectorLaunchConfig,
        *,
        ros_worker_config: Optional[RosWorkerConfig] = None,
        ros_worker_callbacks: Optional[RosWorkerCallbacks] = None,
    ) -> bool:
        yaml_args = []
        yaml_path = self._process_manager.find_package_share_file(
            "teleop_control_py",
            "config/data_collector_params.yaml",
        )
        if yaml_path is not None:
            yaml_args = ["--params-file", str(yaml_path)]

        cmd = ["ros2", "run", "teleop_control_py", "data_collector_node", "--ros-args"]
        if yaml_args:
            cmd.extend(yaml_args)
        cmd.extend(
            [
                "-p", f"robot_profile:={config.robot_profile}",
                "-p", f"output_path:={config.output_path}",
                "-p", f"global_camera_source:={config.global_camera_source}",
                "-p", f"wrist_camera_source:={config.wrist_camera_source}",
                "-p", f"global_camera_enable_depth:={'true' if config.global_camera_enable_depth else 'false'}",
                "-p", f"wrist_camera_enable_depth:={'true' if config.wrist_camera_enable_depth else 'false'}",
                "-p", f"end_effector_type:={config.end_effector_type}",
            ]
        )
        global_serial = str(config.global_camera_serial_number).strip()
        wrist_serial = str(config.wrist_camera_serial_number).strip()
        def _ros_string_literal(value: str) -> str:
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if global_serial:
            cmd.extend(["-p", f"global_camera_serial_number:={_ros_string_literal(global_serial)}"])
        if wrist_serial:
            cmd.extend(["-p", f"wrist_camera_serial_number:={_ros_string_literal(wrist_serial)}"])

        if not self._process_manager.run_subprocess("data_collector", cmd):
            return False
        if ros_worker_config is not None and ros_worker_callbacks is not None:
            self.ensure_ros_worker(ros_worker_config, ros_worker_callbacks)
        return True

    def stop_data_collector(self) -> None:
        self._process_manager.kill_subprocess("data_collector")
