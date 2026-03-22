#!/usr/bin/env python3
"""基于定时器主动抓帧的数据采集节点。"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
import h5py
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float32MultiArray
from std_srvs.srv import Trigger

from ..core import CameraFrameSet, RecorderService, RobotStateSnapshot, SyncHub
from ..data.hdf5_writer import Sample
from ..device_manager import (
    DEFAULT_ROBOT_PROFILE_NAME,
    SharedMemoryCameraBackend,
    default_robot_profiles_path,
    load_robot_profile,
)
from ..data.preview_api import PreviewApiServer
from ..utils.transform_utils import _clamp, center_crop_square_and_resize_rgb, compose_eef_action


@dataclass(frozen=True)
class CameraFramePair:
    global_bgr: np.ndarray
    wrist_bgr: np.ndarray
    global_time: Time
    wrist_time: Time
    ref_time: Time
    skew_sec: float


class DataCollectorNode(Node):
    """负责 ROS 状态缓存、相机调度和 HDF5 采样入队。"""

    def __init__(self) -> None:
        super().__init__("data_collector")

        self.declare_parameter("robot_profile", DEFAULT_ROBOT_PROFILE_NAME)
        self.declare_parameter("robot_profiles_file", str(default_robot_profiles_path()))
        self._robot_profile_name = str(self.get_parameter("robot_profile").value).strip() or DEFAULT_ROBOT_PROFILE_NAME
        self._robot_profiles_file = str(self.get_parameter("robot_profiles_file").value).strip()
        self._robot_profile = load_robot_profile(self._robot_profile_name, self._robot_profiles_file)
        default_end_effector_type = (
            "qbsofthand" if self._robot_profile.grippers.default_type == "qbsofthand" else "robotic_gripper"
        )

        self.declare_parameter("output_path", os.path.join(os.getcwd(), "data", "libero_demos.hdf5"))
        self.declare_parameter("record_fps", 10.0)
        self.declare_parameter("global_camera_source", "realsense")
        self.declare_parameter("wrist_camera_source", "oakd")
        self.declare_parameter("global_camera_serial_number", "")
        self.declare_parameter("wrist_camera_serial_number", "")
        self.declare_parameter("global_camera_enable_depth", False)
        self.declare_parameter("wrist_camera_enable_depth", False)
        self.declare_parameter("joint_states_topic", self._robot_profile.topics.joint_states)
        self.declare_parameter("tool_pose_topic", self._robot_profile.topics.tool_pose)
        self.declare_parameter("servo_twist_topic", self._robot_profile.topics.servo_twist)
        self.declare_parameter("require_gripper", True)
        self.declare_parameter("end_effector_type", default_end_effector_type)
        self.declare_parameter("gripper_state_topic", "")
        self.declare_parameter("robotic_gripper_state_topic", self._robot_profile.grippers.robotiq.state_topic)
        self.declare_parameter("qbsofthand_state_topic", self._robot_profile.grippers.qbsofthand.state_topic)
        self.declare_parameter("obs_image_size", 224)
        self.declare_parameter(
            "joint_names",
            list(self._robot_profile.joint_names),
        )
        self.declare_parameter("pose_max_age_sec", 0.2)
        self.declare_parameter("joint_max_age_sec", 0.2)
        self.declare_parameter("command_max_age_sec", 0.2)
        self.declare_parameter("gripper_max_age_sec", 0.5)
        self.declare_parameter("image_max_age_sec", 0.10)
        self.declare_parameter("camera_pair_max_skew_sec", 0.15)
        self.declare_parameter("pose_stamp_zero_is_ref", True)
        self.declare_parameter("pose_use_received_time_fallback", True)
        self.declare_parameter("stats_period_sec", 2.0)
        self.declare_parameter("queue_maxsize", 400)
        self.declare_parameter("writer_batch_size", 32)
        self.declare_parameter("writer_flush_every_n", 200)
        self.declare_parameter("image_compression", "lzf")
        self.declare_parameter("preview_fps", 30.0)
        self.declare_parameter("preview_api_enabled", True)
        self.declare_parameter("preview_api_host", "127.0.0.1")
        self.declare_parameter("preview_api_port", 8765)
        self.declare_parameter("preview_api_jpeg_quality", 80)
        self.declare_parameter("enable_keyboard", False)

        self._output_path = str(self.get_parameter("output_path").value)
        self._record_fps = max(0.1, float(self.get_parameter("record_fps").value))
        self._joint_names = list(self.get_parameter("joint_names").value)
        self._pose_max_age = float(self.get_parameter("pose_max_age_sec").value)
        self._joint_max_age = float(self.get_parameter("joint_max_age_sec").value)
        self._command_max_age = float(self.get_parameter("command_max_age_sec").value)
        self._gripper_max_age = float(self.get_parameter("gripper_max_age_sec").value)
        self._image_max_age = float(self.get_parameter("image_max_age_sec").value)
        self._camera_pair_max_skew = float(self.get_parameter("camera_pair_max_skew_sec").value)
        self._pose_stamp_zero_is_ref = bool(self.get_parameter("pose_stamp_zero_is_ref").value)
        self._pose_use_received_time_fallback = bool(self.get_parameter("pose_use_received_time_fallback").value)
        self._require_gripper = bool(self.get_parameter("require_gripper").value)

        self._obs_image_size = int(self.get_parameter("obs_image_size").value)
        if self._obs_image_size != 224:
            self.get_logger().warn(
                f"obs_image_size={self._obs_image_size} 不是 LIBERO 标准值 224，将强制改为 224。"
            )
            self._obs_image_size = 224

        compression_value = self.get_parameter("image_compression").value
        if compression_value is None:
            self._image_compression: Optional[str] = None
        else:
            compression = str(compression_value).strip().lower()
            self._image_compression = None if compression in {"", "none", "null"} else compression

        self._record_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._camera_pull_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._warn_last_monotonic: Dict[str, float] = {}

        self._recording = False
        self._demo_index = self._discover_next_demo_index(self._output_path)
        self._current_demo_name: Optional[str] = None
        self._capture_timer = None
        self._current_demo_frames = 0
        self._recent_frame_times = deque()

        self._latest_joint_pos: Optional[np.ndarray] = None
        self._latest_joint_time: Optional[Time] = None
        self._latest_pose_pos: Optional[np.ndarray] = None
        self._latest_pose_quat: Optional[np.ndarray] = None
        self._latest_pose_time: Optional[Time] = None
        self._latest_pose_receive_time: Optional[Time] = None
        self._latest_twist_linear: Optional[np.ndarray] = None
        self._latest_twist_angular: Optional[np.ndarray] = None
        self._latest_twist_time: Optional[Time] = None
        self._latest_gripper: Optional[float] = None
        self._latest_gripper_time: Optional[Time] = None
        self._latest_global_bgr: Optional[np.ndarray] = None
        self._latest_wrist_bgr: Optional[np.ndarray] = None
        self._latest_global_frame_time: Optional[Time] = None
        self._latest_wrist_frame_time: Optional[Time] = None
        self._latest_frame_ref_time: Optional[Time] = None
        self._latest_frame_skew_sec: Optional[float] = None

        self._stats: Dict[str, int] = {}
        self._keyboard_thread: Optional[threading.Thread] = None
        self._keyboard_stop_evt = threading.Event()
        self._preview_api_enabled = bool(self.get_parameter("preview_api_enabled").value)
        self._preview_api_host = str(self.get_parameter("preview_api_host").value).strip() or "127.0.0.1"
        self._preview_api_port = int(self.get_parameter("preview_api_port").value)
        self._preview_api_jpeg_quality = int(self.get_parameter("preview_api_jpeg_quality").value)
        self._preview_timer = None
        self._preview_api_server: Optional[PreviewApiServer] = None

        self._recorder = RecorderService(
            output_path=self._output_path,
            compression=self._image_compression,
            queue_maxsize=int(self.get_parameter("queue_maxsize").value),
            batch_size=int(self.get_parameter("writer_batch_size").value),
            flush_every_n=int(self.get_parameter("writer_flush_every_n").value),
            logger=self.get_logger(),
        )

        self._camera_instances: Dict[tuple[str, str, bool], object] = {}
        global_source = self._normalize_camera_source(self.get_parameter("global_camera_source").value)
        wrist_source = self._normalize_camera_source(self.get_parameter("wrist_camera_source").value)
        self._global_camera_serial_number = str(self.get_parameter("global_camera_serial_number").value).strip()
        self._wrist_camera_serial_number = str(self.get_parameter("wrist_camera_serial_number").value).strip()
        self._global_camera_enable_depth = bool(self.get_parameter("global_camera_enable_depth").value)
        self._wrist_camera_enable_depth = bool(self.get_parameter("wrist_camera_enable_depth").value)
        self.global_cam = self._get_or_create_camera(
            global_source,
            self._global_camera_serial_number,
            enable_depth=self._global_camera_enable_depth,
        )
        self.wrist_cam = self._get_or_create_camera(
            wrist_source,
            self._wrist_camera_serial_number,
            enable_depth=self._wrist_camera_enable_depth,
        )
        self._global_camera_source = global_source
        self._wrist_camera_source = wrist_source
        self._sync_hub = SyncHub(
            camera_provider=self._select_record_camera_frame_set,
            state_provider=self._sync_state_provider,
            action_provider=self._sync_action_provider,
        )

        joint_topic = str(self.get_parameter("joint_states_topic").value)
        pose_topic = str(self.get_parameter("tool_pose_topic").value)
        twist_topic = str(self.get_parameter("servo_twist_topic").value)
        gripper_topic = self._resolve_gripper_topic()

        self.create_subscription(JointState, joint_topic, self._on_joint_state, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, pose_topic, self._on_tool_pose, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, twist_topic, self._on_servo_twist, qos_profile_sensor_data)
        self.create_subscription(Float32, gripper_topic, self._on_gripper, qos_profile_sensor_data)

        self._record_stats_pub = self.create_publisher(Float32MultiArray, "~/record_stats", 10)

        if self._preview_api_enabled:
            preview_fps = max(0.1, float(self.get_parameter("preview_fps").value))
            self._preview_timer = self.create_timer(1.0 / preview_fps, self._preview_step)

        if self._preview_api_enabled:
            try:
                self._preview_api_server = PreviewApiServer(
                    host=self._preview_api_host,
                    port=self._preview_api_port,
                    frame_provider=self._get_preview_frame_for_api,
                    jpeg_quality=self._preview_api_jpeg_quality,
                    logger=self.get_logger(),
                )
                self._preview_api_server.start()
            except Exception as exc:  # noqa: BLE001
                self._preview_api_server = None
                self.get_logger().warn(f"启动 Preview API 失败，预览窗口将无法拉取采集画面: {exc!r}")

        self._srv_start = self.create_service(Trigger, "~/start", self._srv_start_cb)
        self._srv_stop = self.create_service(Trigger, "~/stop", self._srv_stop_cb)
        self._srv_discard_last_demo = self.create_service(Trigger, "~/discard_last_demo", self._srv_discard_last_demo_cb)

        if bool(self.get_parameter("enable_keyboard").value):
            self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            self._keyboard_thread.start()

        period = float(self.get_parameter("stats_period_sec").value)
        if period > 0.0:
            self._stats_timer = self.create_timer(period, self._log_stats)

        self.get_logger().info(
            "DataCollectorNode ready. Services: ~/start, ~/stop, ~/discard_last_demo. "
            f"robot_profile={self._robot_profile.name}, Global camera={self._global_camera_source}, "
            f"Wrist camera={self._wrist_camera_source}, Gripper={gripper_topic}, Output={self._output_path}, "
            f"FPS={self._record_fps:.2f}, Global SN={self._global_camera_serial_number or 'auto'}, "
            f"Wrist SN={self._wrist_camera_serial_number or 'auto'}, "
            f"Global depth={self._global_camera_enable_depth}, "
            f"Wrist depth={self._wrist_camera_enable_depth}"
        )

        if self._recorder.next_demo_index > 0:
            self.get_logger().info(
                f"Detected existing demos in HDF5. Next demo index: {self._recorder.next_demo_index}"
            )

        if self._preview_api_enabled:
            self.get_logger().info(
                "Preview API enabled. "
                f"Host={self._preview_api_host}, Port={self._preview_api_port}, JPEG={self._preview_api_jpeg_quality}"
            )

    def _normalize_camera_source(self, value: object) -> str:
        source = str(value).strip().lower()
        if source in {"realsense", "oakd"}:
            return source
        self.get_logger().warn(f"未知相机来源 '{source}'，回退到 realsense。")
        return "realsense"

    def _get_or_create_camera(
        self,
        source: str,
        serial_number: str = "",
        *,
        enable_depth: bool = False,
    ) -> Optional[object]:
        cache_key = (source, str(serial_number).strip(), bool(enable_depth))
        if cache_key in self._camera_instances:
            return self._camera_instances[cache_key]

        try:
            camera = SharedMemoryCameraBackend.create(
                source,
                serial_number=serial_number,
                enable_depth=enable_depth,
                logger=self.get_logger(),
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                "初始化 "
                f"{source} 相机失败 (serial={serial_number or 'auto'}, depth={enable_depth}): {exc!r}"
            )
            camera = None

        self._camera_instances[cache_key] = camera
        return camera

    def _resolve_gripper_topic(self) -> str:
        override = str(self.get_parameter("gripper_state_topic").value).strip()
        if override:
            return override

        ee_type = str(self.get_parameter("end_effector_type").value).strip().lower()
        if ee_type == "robotic_gripper":
            return str(self.get_parameter("robotic_gripper_state_topic").value)
        if ee_type == "qbsofthand":
            return str(self.get_parameter("qbsofthand_state_topic").value)

        self.get_logger().warn(f"未知末端执行器类型 '{ee_type}'，回退到 robotic_gripper。")
        return str(self.get_parameter("robotic_gripper_state_topic").value)

    def _inc_stat(self, key: str, n: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] = int(self._stats.get(key, 0)) + int(n)

    def _discover_next_demo_index(self, output_path: str) -> int:
        if not output_path or not os.path.exists(output_path):
            return 0

        try:
            with h5py.File(output_path, "r") as h5_file:
                data_group = h5_file.get("data")
                if not isinstance(data_group, h5py.Group):
                    return 0

                next_index = 0
                for name in data_group.keys():
                    if not name.startswith("demo_"):
                        continue
                    suffix = name[5:]
                    if not suffix.isdigit():
                        continue
                    next_index = max(next_index, int(suffix) + 1)
                return next_index
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"读取已有 HDF5 demo 索引失败，将从 demo_0 开始: {exc!r}")
            return 0

    def _reset_stats(self) -> None:
        with self._stats_lock:
            self._stats = {}

    def _log_stats(self) -> None:
        with self._record_lock:
            recording = self._recording
        if not recording:
            return

        with self._stats_lock:
            stats = dict(self._stats)
            self._stats = {}

        if not stats:
            return

        qsize = self._recorder.queue_size()

        ordered = ", ".join(f"{key}={value}" for key, value in sorted(stats.items()))
        self.get_logger().info(f"Recorder stats: {ordered} | queue={qsize}")

    def _warn_throttled(self, key: str, msg: str, period_sec: float = 2.0) -> None:
        now = time.monotonic()
        last = float(self._warn_last_monotonic.get(key, 0.0))
        if (now - last) < period_sec:
            return
        self._warn_last_monotonic[key] = now
        self.get_logger().warn(msg)

    def _publish_record_stats(self) -> None:
        msg = Float32MultiArray()
        now = time.monotonic()

        with self._record_lock:
            frames = int(self._current_demo_frames)
            while self._recent_frame_times and (now - float(self._recent_frame_times[0])) > 1.0:
                self._recent_frame_times.popleft()

            if len(self._recent_frame_times) >= 2:
                duration = float(self._recent_frame_times[-1] - self._recent_frame_times[0])
                realtime_fps = 0.0 if duration <= 1e-6 else (len(self._recent_frame_times) - 1) / duration
            else:
                realtime_fps = 0.0

        msg.data = [float(frames), float(realtime_fps)]
        self._record_stats_pub.publish(msg)

    def _on_joint_state(self, msg: JointState) -> None:
        joint_pos = self._map_joint_positions(msg)
        if joint_pos is None:
            self._inc_stat("joint_map_fail")
            return

        joint_time = Time.from_msg(msg.header.stamp)
        if joint_time.nanoseconds == 0:
            joint_time = self.get_clock().now()

        with self._cache_lock:
            self._latest_joint_pos = joint_pos
            self._latest_joint_time = joint_time

    def _on_tool_pose(self, msg: PoseStamped) -> None:
        pose_time = Time.from_msg(msg.header.stamp)
        receive_time = self.get_clock().now()
        with self._cache_lock:
            self._latest_pose_pos = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=np.float32,
            )
            self._latest_pose_quat = np.array(
                [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w],
                dtype=np.float32,
            )
            self._latest_pose_time = pose_time
            self._latest_pose_receive_time = receive_time

    def _on_gripper(self, msg: Float32) -> None:
        with self._cache_lock:
            self._latest_gripper = float(_clamp(float(msg.data), 0.0, 1.0))
            self._latest_gripper_time = self.get_clock().now()

    def _on_servo_twist(self, msg: TwistStamped) -> None:
        twist_time = Time.from_msg(msg.header.stamp)
        if twist_time.nanoseconds == 0:
            twist_time = self.get_clock().now()

        with self._cache_lock:
            self._latest_twist_linear = np.array(
                [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
                dtype=np.float32,
            )
            self._latest_twist_angular = np.array(
                [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z],
                dtype=np.float32,
            )
            self._latest_twist_time = twist_time

    def _map_joint_positions(self, msg: JointState) -> Optional[np.ndarray]:
        if not msg.name or not msg.position:
            return None

        name_to_idx = {name: index for index, name in enumerate(msg.name)}
        joint_count = min(6, len(self._joint_names))
        out = np.zeros(joint_count, dtype=np.float32)
        for index, joint_name in enumerate(self._joint_names[:joint_count]):
            msg_index = name_to_idx.get(joint_name)
            if msg_index is None or msg_index >= len(msg.position):
                missing = [name for name in self._joint_names[:joint_count] if name not in name_to_idx]
                self._warn_throttled(
                    "joint_map",
                    "JointState 映射失败，缺少关节: "
                    + ", ".join(missing)
                    + " | 当前示例 name: "
                    + ", ".join(list(msg.name)[:8]),
                )
                return None
            out[index] = float(msg.position[msg_index])
        return out

    def _get_cached_state(
        self,
        ref_time: Time,
    ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]], Optional[str], Optional[str]]:
        with self._cache_lock:
            joint_pos = None if self._latest_joint_pos is None else self._latest_joint_pos.copy()
            joint_time = self._latest_joint_time
            pose_pos = None if self._latest_pose_pos is None else self._latest_pose_pos.copy()
            pose_quat = None if self._latest_pose_quat is None else self._latest_pose_quat.copy()
            pose_time = self._latest_pose_time
            pose_receive_time = self._latest_pose_receive_time
            gripper = self._latest_gripper
            gripper_time = self._latest_gripper_time

        if joint_pos is None:
            return None, "no_joint", None
        if joint_time is None:
            return None, "joint_stale", "joint_states 缺少有效时间戳。"
        if self._joint_max_age > 0.0:
            joint_age = abs((ref_time - joint_time).nanoseconds) * 1e-9
            if joint_age > self._joint_max_age:
                return (
                    None,
                    "joint_stale",
                    f"joint_states 过旧，topic={self.get_parameter('joint_states_topic').value}, age={joint_age:.3f}s。",
                )
        if pose_pos is None or pose_quat is None or pose_time is None:
            return None, "no_pose", None

        pose_ref_time = ref_time if self._pose_stamp_zero_is_ref and pose_time.nanoseconds == 0 else pose_time
        if self._pose_max_age > 0.0:
            pose_msg_age = abs((ref_time - pose_ref_time).nanoseconds) * 1e-9
            pose_receive_age = None
            if pose_receive_time is not None:
                pose_receive_age = abs((ref_time - pose_receive_time).nanoseconds) * 1e-9

            if pose_msg_age > self._pose_max_age:
                if (
                    self._pose_use_received_time_fallback
                    and pose_receive_age is not None
                    and pose_receive_age <= self._pose_max_age
                ):
                    self._inc_stat("pose_stamp_fallback")
                else:
                    detail = (
                        f"末端位姿过旧，topic={self.get_parameter('tool_pose_topic').value}, "
                        f"msg_age={pose_msg_age:.3f}s"
                    )
                    if pose_receive_age is not None:
                        detail += f", recv_age={pose_receive_age:.3f}s"
                    detail += "。建议检查发布频率、header.stamp 和时钟来源。"
                    return None, "pose_stale", detail

        if gripper is None or gripper_time is None:
            if self._require_gripper:
                return None, "no_gripper", None
            gripper = 0.0
        elif self._gripper_max_age > 0.0:
            if abs((ref_time - gripper_time).nanoseconds) * 1e-9 > self._gripper_max_age:
                if self._require_gripper:
                    return None, "gripper_stale", None

        return (joint_pos, pose_pos, pose_quat, float(gripper)), None, None

    def _get_cached_action(self, ref_time: Time, gripper: float) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
        with self._cache_lock:
            linear = None if self._latest_twist_linear is None else self._latest_twist_linear.copy()
            angular = None if self._latest_twist_angular is None else self._latest_twist_angular.copy()
            twist_time = self._latest_twist_time

        if linear is None or angular is None or twist_time is None:
            return None, "no_action_cmd", "尚未收到有效 servo twist 命令，当前帧跳过。"

        if self._command_max_age > 0.0:
            cmd_age = abs((ref_time - twist_time).nanoseconds) * 1e-9
            if cmd_age > self._command_max_age:
                return (
                    None,
                    "action_cmd_stale",
                    f"servo twist 命令过旧，topic={self.get_parameter('servo_twist_topic').value}, age={cmd_age:.3f}s。",
                )
        return compose_eef_action(linear, angular, gripper), None, None

    def _pull_camera_frames(self) -> Optional[CameraFramePair]:
        if self.global_cam is None or self.wrist_cam is None:
            return None

        with self._camera_pull_lock:
            global_start = self.get_clock().now()
            global_bgr = self.global_cam.get_bgr_frame()
            global_end = self.get_clock().now()
            wrist_start = self.get_clock().now()
            wrist_bgr = self.wrist_cam.get_bgr_frame()
            wrist_end = self.get_clock().now()

        if global_bgr is None or wrist_bgr is None:
            return None

        global_time = global_end if global_end.nanoseconds >= global_start.nanoseconds else global_start
        wrist_time = wrist_end if wrist_end.nanoseconds >= wrist_start.nanoseconds else wrist_start
        ref_time = wrist_time if wrist_time.nanoseconds >= global_time.nanoseconds else global_time
        skew_sec = abs((wrist_time - global_time).nanoseconds) * 1e-9
        return CameraFramePair(
            global_bgr=np.ascontiguousarray(global_bgr),
            wrist_bgr=np.ascontiguousarray(wrist_bgr),
            global_time=global_time,
            wrist_time=wrist_time,
            ref_time=ref_time,
            skew_sec=skew_sec,
        )

    def _cache_camera_frames(self, frame_pair: Optional[CameraFramePair]) -> None:
        if frame_pair is None:
            return

        with self._cache_lock:
            self._latest_global_bgr = frame_pair.global_bgr.copy()
            self._latest_wrist_bgr = frame_pair.wrist_bgr.copy()
            self._latest_global_frame_time = frame_pair.global_time
            self._latest_wrist_frame_time = frame_pair.wrist_time
            self._latest_frame_ref_time = frame_pair.ref_time
            self._latest_frame_skew_sec = float(frame_pair.skew_sec)

    def _get_cached_camera_frames(self) -> Optional[CameraFramePair]:
        with self._cache_lock:
            if (
                self._latest_global_bgr is None
                or self._latest_wrist_bgr is None
                or self._latest_global_frame_time is None
                or self._latest_wrist_frame_time is None
                or self._latest_frame_ref_time is None
                or self._latest_frame_skew_sec is None
            ):
                return None
            return CameraFramePair(
                global_bgr=self._latest_global_bgr.copy(),
                wrist_bgr=self._latest_wrist_bgr.copy(),
                global_time=self._latest_global_frame_time,
                wrist_time=self._latest_wrist_frame_time,
                ref_time=self._latest_frame_ref_time,
                skew_sec=float(self._latest_frame_skew_sec),
            )

    def _get_preview_frame_for_api(self, camera_name: str) -> Optional[np.ndarray]:
        frame_pair = self._get_cached_camera_frames()
        if frame_pair is None:
            return None
        if camera_name == "global":
            return frame_pair.global_bgr
        if camera_name == "wrist":
            return frame_pair.wrist_bgr
        return None

    def _pull_and_cache_camera_frames(self) -> Optional[CameraFramePair]:
        frame_pair = self._pull_camera_frames()
        self._cache_camera_frames(frame_pair)
        return frame_pair

    def _preview_step(self) -> None:
        if self.global_cam is None or self.wrist_cam is None:
            return

        try:
            frame_pair = self._pull_and_cache_camera_frames()
        except Exception as exc:  # noqa: BLE001
            self._warn_throttled("preview_pull", f"拉取预览图像失败: {exc!r}")
            return

        if frame_pair is None:
            return

    def _select_record_camera_frames(self) -> Optional[CameraFramePair]:
        frame_pair = self._get_cached_camera_frames()
        if frame_pair is not None and self._image_max_age > 0.0:
            image_age = abs((self.get_clock().now() - frame_pair.ref_time).nanoseconds) * 1e-9
            if image_age > self._image_max_age:
                frame_pair = None

        if frame_pair is None:
            frame_pair = self._pull_and_cache_camera_frames()

        return frame_pair

    def _camera_frame_pair_to_frame_set(self, frame_pair: CameraFramePair) -> CameraFrameSet:
        return CameraFrameSet(
            global_bgr=frame_pair.global_bgr,
            wrist_bgr=frame_pair.wrist_bgr,
            global_time_ns=frame_pair.global_time.nanoseconds,
            wrist_time_ns=frame_pair.wrist_time.nanoseconds,
            ref_time_ns=frame_pair.ref_time.nanoseconds,
            skew_sec=float(frame_pair.skew_sec),
            ref_context=frame_pair.ref_time,
        )

    def _select_record_camera_frame_set(self) -> Optional[CameraFrameSet]:
        frame_pair = self._select_record_camera_frames()
        if frame_pair is None:
            return None
        return self._camera_frame_pair_to_frame_set(frame_pair)

    def _sync_state_provider(
        self,
        frame_set: CameraFrameSet,
    ) -> Tuple[Optional[RobotStateSnapshot], Optional[str], Optional[str]]:
        ref_time = frame_set.ref_context
        if not isinstance(ref_time, Time):
            return None, "invalid_ref_time", "采样参考时间无效。"

        cached, reason, detail = self._get_cached_state(ref_time)
        if cached is None:
            return None, reason, detail

        joint_pos, eef_pos, eef_quat, gripper = cached
        with self._cache_lock:
            twist_linear = None if self._latest_twist_linear is None else self._latest_twist_linear.copy()
            twist_angular = None if self._latest_twist_angular is None else self._latest_twist_angular.copy()

        return (
            RobotStateSnapshot(
                joint_pos=joint_pos,
                eef_pos=eef_pos,
                eef_quat=eef_quat,
                gripper=gripper,
                twist_linear=twist_linear,
                twist_angular=twist_angular,
                ref_context={"ref_time": ref_time},
            ),
            None,
            None,
        )

    def _sync_action_provider(
        self,
        frame_set: CameraFrameSet,
        state: RobotStateSnapshot,
    ) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
        ref_time = frame_set.ref_context
        if not isinstance(ref_time, Time):
            return None, "invalid_ref_time", "采样参考时间无效。"
        return self._get_cached_action(ref_time, state.gripper)

    def _capture_step(self) -> None:
        with self._record_lock:
            if not self._recording or self._current_demo_name is None:
                self._inc_stat("not_recording")
                return
            demo_name = self._current_demo_name

        if self.global_cam is None or self.wrist_cam is None:
            self._warn_throttled("camera_missing", "相机客户端未成功初始化，当前帧跳过。")
            self._inc_stat("camera_missing")
            return

        try:
            snapshot, reason, detail = self._sync_hub.capture_snapshot()
        except Exception as exc:  # noqa: BLE001
            self._warn_throttled("snapshot_fail", f"采样快照失败: {exc!r}")
            self._inc_stat("snapshot_fail")
            return

        if snapshot is None:
            reason = reason or "snapshot_missing"
            self._inc_stat(reason)
            if reason == "camera_empty":
                self._warn_throttled("camera_empty", "尚未获取到双相机有效图像，当前帧跳过。")
            elif reason == "no_joint":
                self._warn_throttled("no_joint", "尚未收到有效 /joint_states，当前帧跳过。")
            elif reason == "joint_stale":
                self._warn_throttled("joint_stale", detail or "joint_states 过旧，当前帧跳过。")
            elif reason == "no_pose":
                self._warn_throttled("no_pose", "尚未收到有效末端位姿，当前帧跳过。")
            elif reason == "pose_stale":
                self._warn_throttled("pose_stale", detail or "末端位姿过旧，建议检查 tool_pose_topic 和时延。")
            elif reason == "no_gripper":
                self._warn_throttled("no_gripper", "尚未收到夹爪状态，当前帧跳过。")
            elif reason == "gripper_stale":
                self._warn_throttled("gripper_stale", "夹爪状态过旧，当前帧跳过。")
            elif reason in {"no_action_cmd", "action_cmd_stale"}:
                self._warn_throttled(reason, detail or "动作命令缺失，当前帧跳过。")
            else:
                self._warn_throttled(reason, detail or f"采样快照失败: {reason}")
            return

        frame_set = snapshot.camera_frames
        if self._camera_pair_max_skew > 0.0 and frame_set.skew_sec > self._camera_pair_max_skew:
            self._inc_stat("camera_skew")
            self._warn_throttled(
                "camera_skew",
                f"双相机时间偏差过大，当前帧跳过: skew={frame_set.skew_sec:.3f}s。",
            )
            return

        try:
            agentview_rgb = center_crop_square_and_resize_rgb(frame_set.global_bgr, self._obs_image_size)
            eye_in_hand_rgb = center_crop_square_and_resize_rgb(frame_set.wrist_bgr, self._obs_image_size)
        except Exception as exc:  # noqa: BLE001
            self._warn_throttled("image_fail", f"图像预处理失败: {exc!r}")
            self._inc_stat("image_fail")
            return

        if snapshot.action_vector is None:
            self._warn_throttled("action_missing", "动作命令缺失，当前帧跳过。")
            self._inc_stat("action_missing")
            return

        robot_state = snapshot.robot_state
        sample = Sample(
            demo_name=demo_name,
            agentview_rgb=agentview_rgb,
            eye_in_hand_rgb=eye_in_hand_rgb,
            robot0_joint_pos=robot_state.joint_pos.astype(np.float32, copy=False),
            robot0_gripper_qpos=np.array([robot_state.gripper], dtype=np.float32),
            robot0_eef_pos=robot_state.eef_pos.astype(np.float32, copy=False),
            robot0_eef_quat=robot_state.eef_quat.astype(np.float32, copy=False),
            actions=snapshot.action_vector.astype(np.float32, copy=False),
        )

        if not self._recorder.enqueue_sample(sample):
            self._warn_throttled("queue_full", "写盘队列已满，当前样本被丢弃。")
            self._inc_stat("queue_full")
            return

        self._inc_stat("enqueued")
        with self._record_lock:
            self._current_demo_frames += 1
            now = time.monotonic()
            self._recent_frame_times.append(now)
            while self._recent_frame_times and (now - float(self._recent_frame_times[0])) > 1.0:
                self._recent_frame_times.popleft()
        self._publish_record_stats()

    def _srv_start_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        with self._record_lock:
            if self._recording:
                res.success = False
                res.message = "Already recording"
                return res

            if self.global_cam is None or self.wrist_cam is None:
                res.success = False
                res.message = "Camera client is not available"
                return res

            success, message, demo_name = self._recorder.start_demo()
            if not success or demo_name is None:
                res.success = False
                res.message = message
                return res

            self._reset_stats()
            self._current_demo_name = demo_name
            self._recording = True
            self._current_demo_frames = 0
            self._recent_frame_times.clear()

            period = 1.0 / self._record_fps
            if self._capture_timer is not None:
                try:
                    self._capture_timer.cancel()
                except Exception:
                    pass
            self._capture_timer = self.create_timer(period, self._capture_step)

        self._publish_record_stats()

        res.success = True
        res.message = f"{message} at {self._record_fps:.2f} Hz"
        self.get_logger().info(res.message)
        return res

    def _srv_stop_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        with self._record_lock:
            if not self._recording or self._current_demo_name is None:
                res.success = False
                res.message = "Not recording"
                return res

            self._recording = False
            self._current_demo_name = None

            if self._capture_timer is not None:
                try:
                    self._capture_timer.cancel()
                except Exception:
                    pass
                self._capture_timer = None

        success, message, _demo_name = self._recorder.stop_demo()
        if not success:
            res.success = False
            res.message = message
            return res

        with self._record_lock:
            self._recent_frame_times.clear()
        self._publish_record_stats()

        res.success = True
        res.message = message
        self.get_logger().info(res.message)
        return res

    def _srv_discard_last_demo_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        with self._record_lock:
            if self._recording:
                res.success = False
                res.message = "Cannot discard demo while recording"
                return res

        success, message, _demo_name = self._recorder.discard_last_demo()
        if not success:
            res.success = False
            res.message = message
            return res

        with self._record_lock:
            self._recent_frame_times.clear()
            self._current_demo_frames = 0
        self._publish_record_stats()

        res.success = True
        res.message = message
        self.get_logger().info(res.message)
        return res

    def _keyboard_loop(self) -> None:
        self.get_logger().info("Keyboard control enabled: r=start, s=stop, q=quit (press Enter)")
        while not self._keyboard_stop_evt.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
                continue

            if not line:
                time.sleep(0.1)
                continue

            cmd = line.strip().lower()
            if cmd == "r":
                self._srv_start_cb(Trigger.Request(), Trigger.Response())
            elif cmd == "s":
                self._srv_stop_cb(Trigger.Request(), Trigger.Response())
            elif cmd == "q":
                self.get_logger().info("Keyboard quit requested")
                if rclpy.ok():
                    rclpy.shutdown()
                break

    def destroy_node(self) -> bool:
        self._keyboard_stop_evt.set()

        with self._record_lock:
            if self._capture_timer is not None:
                try:
                    self._capture_timer.cancel()
                except Exception:
                    pass
                self._capture_timer = None

            if self._preview_timer is not None:
                try:
                    self._preview_timer.cancel()
                except Exception:
                    pass
                self._preview_timer = None

            active_demo = self._current_demo_name
            self._recording = False
            self._current_demo_name = None

        if self._preview_api_server is not None:
            try:
                self._preview_api_server.stop()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"停止 Preview API 失败: {exc!r}")
            self._preview_api_server = None

        for camera in {id(camera): camera for camera in self._camera_instances.values() if camera is not None}.values():
            try:
                camera.stop()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"关闭相机失败: {exc!r}")

        try:
            self._recorder.close()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"关闭 HDF5 写线程失败: {exc!r}")

        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
