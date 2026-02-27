#!/usr/bin/env python3
"""ROS 2 (Humble) demonstration data recorder for LIBERO / robomimic.

This node records episodes ("demos") into a single HDF5 file following the
dataset structure used by LIBERO / robomimic:

  /data/demo_0/actions                 (N, 7) float32  [x,y,z, rx,ry,rz, gripper]
  /data/demo_0/obs/agentview_rgb       (N,224,224,3) uint8   (global camera)
  /data/demo_0/obs/eye_in_hand_rgb     (N,224,224,3) uint8   (wrist camera)
  /data/demo_0/obs/robot0_joint_pos    (N, 6) float32
  /data/demo_0/obs/robot0_eef_pos      (N, 3) float32
  /data/demo_0/obs/robot0_eef_quat     (N, 4) float32  (x,y,z,w)

Key design points:
- Non-blocking: ROS callbacks only preprocess + enqueue samples.
- Background writer thread writes to HDF5 in batches.
- ApproximateTimeSynchronizer aligns two images + joint states (slop=0.1s).
- Tool pose + gripper are read from most-recent messages (time-checked).

Control:
- Services (recommended for reliability):
    /data_collector/start  (std_srvs/Trigger) -> begin a new demo
    /data_collector/stop   (std_srvs/Trigger) -> stop current demo
  You can call them with:
    ros2 service call /data_collector/start std_srvs/srv/Trigger {}
    ros2 service call /data_collector/stop  std_srvs/srv/Trigger {}
- Optional keyboard thread (enable_keyboard=true):
    r: start new demo
    s: stop
    q: shutdown
"""

from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import h5py
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg

import message_filters


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return (q / norm).astype(np.float64)


def _quat_to_rotvec_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> rotation vector (axis * angle)."""
    qn = _quat_normalize_xyzw(q_xyzw)
    x, y, z, w = (float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3]))
    w = _clamp(w, -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x / s, y / s, z / s], dtype=np.float64)
    return (axis * angle).astype(np.float32)


def _center_crop_square_and_resize_rgb(bgr: np.ndarray, output_size: int) -> np.ndarray:
    """Center crop to square (use min(H,W)), then resize to output_size x output_size.

    Output format: RGB uint8 contiguous array.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected BGR HxWx3 image")
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError("output_size must be > 0")
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = bgr[y0 : y0 + side, x0 : x0 + side]
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


@dataclass(frozen=True)
class Sample:
    demo_name: str
    agentview_rgb: np.ndarray  # (224,224,3) uint8
    eye_in_hand_rgb: np.ndarray  # (224,224,3) uint8
    robot0_joint_pos: np.ndarray  # (6,) float32
    robot0_eef_pos: np.ndarray  # (3,) float32
    robot0_eef_quat: np.ndarray  # (4,) float32 (x,y,z,w)
    actions: np.ndarray  # (7,) float32 [x,y,z, rx,ry,rz, gripper]


@dataclass(frozen=True)
class Command:
    kind: str  # start_demo | stop_demo | close
    demo_name: Optional[str] = None


class HDF5WriterThread(threading.Thread):
    """Background writer that appends samples into LIBERO/robomimic HDF5 structure."""

    def __init__(
        self,
        output_path: str,
        item_queue: "queue.Queue[object]",
        compression: Optional[str] = "lzf",
        batch_size: int = 32,
        flush_every_n: int = 200,
        logger: Optional[object] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._output_path = output_path
        self._queue = item_queue
        self._compression = compression
        self._batch_size = max(1, int(batch_size))
        self._flush_every_n = max(1, int(flush_every_n))
        self._logger = logger

        self._stop_evt = threading.Event()

        self._h5: Optional[h5py.File] = None
        self._data_group: Optional[h5py.Group] = None
        self._current_demo: Optional[str] = None
        self._demo_handles: Dict[str, Dict[str, object]] = {}
        self._demo_counts: Dict[str, int] = {}

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self._queue.put_nowait(Command(kind="close"))
        except queue.Full:
            pass

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        fn = getattr(self._logger, level, None)
        if callable(fn):
            fn(msg)

    def _ensure_file_open(self) -> None:
        if self._h5 is not None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._output_path)), exist_ok=True)
        self._h5 = h5py.File(self._output_path, "a")
        self._data_group = self._h5.require_group("data")

    def _create_demo_if_needed(self, demo_name: str) -> None:
        assert self._data_group is not None
        if demo_name in self._demo_handles:
            return

        demo_group = self._data_group.require_group(demo_name)
        obs_group = demo_group.require_group("obs")

        def _ds(
            group: h5py.Group,
            name: str,
            shape_tail: Tuple[int, ...],
            dtype: np.dtype,
            compression: Optional[str],
        ) -> h5py.Dataset:
            maxshape = (None,) + shape_tail
            chunks = (1,) + shape_tail
            return group.require_dataset(
                name,
                shape=(0,) + shape_tail,
                maxshape=maxshape,
                chunks=chunks,
                dtype=dtype,
                compression=compression,
            )

        handles: Dict[str, object] = {
            "demo_group": demo_group,
            "actions": _ds(demo_group, "actions", (7,), np.float32, None),
            "agentview_rgb": _ds(obs_group, "agentview_rgb", (224, 224, 3), np.uint8, self._compression),
            "eye_in_hand_rgb": _ds(obs_group, "eye_in_hand_rgb", (224, 224, 3), np.uint8, self._compression),
            "robot0_joint_pos": _ds(obs_group, "robot0_joint_pos", (6,), np.float32, None),
            "robot0_eef_pos": _ds(obs_group, "robot0_eef_pos", (3,), np.float32, None),
            "robot0_eef_quat": _ds(obs_group, "robot0_eef_quat", (4,), np.float32, None),
        }

        demo_group.attrs["num_samples"] = 0
        self._demo_handles[demo_name] = handles
        self._demo_counts[demo_name] = 0
        self._log("info", f"HDF5: created group data/{demo_name}")

    def _finalize_demo(self, demo_name: str) -> None:
        handles = self._demo_handles.get(demo_name)
        if not handles:
            return
        demo_group: h5py.Group = handles["demo_group"]  # type: ignore[assignment]
        demo_group.attrs["num_samples"] = int(self._demo_counts.get(demo_name, 0))

    def _append_batch(self, demo_name: str, batch: "list[Sample]") -> None:
        self._ensure_file_open()
        assert self._h5 is not None
        assert self._data_group is not None
        self._create_demo_if_needed(demo_name)
        handles = self._demo_handles[demo_name]

        n0 = int(self._demo_counts[demo_name])
        b = len(batch)
        n1 = n0 + b

        def _resize_and_write(ds: h5py.Dataset, data: np.ndarray) -> None:
            ds.resize((n1,) + ds.shape[1:])
            ds[n0:n1] = data

        _resize_and_write(handles["actions"], np.stack([s.actions for s in batch], axis=0))  # type: ignore[arg-type]
        _resize_and_write(handles["agentview_rgb"], np.stack([s.agentview_rgb for s in batch], axis=0))  # type: ignore[arg-type]
        _resize_and_write(handles["eye_in_hand_rgb"], np.stack([s.eye_in_hand_rgb for s in batch], axis=0))  # type: ignore[arg-type]
        _resize_and_write(handles["robot0_joint_pos"], np.stack([s.robot0_joint_pos for s in batch], axis=0))  # type: ignore[arg-type]
        _resize_and_write(handles["robot0_eef_pos"], np.stack([s.robot0_eef_pos for s in batch], axis=0))  # type: ignore[arg-type]
        _resize_and_write(handles["robot0_eef_quat"], np.stack([s.robot0_eef_quat for s in batch], axis=0))  # type: ignore[arg-type]

        self._demo_counts[demo_name] = n1

        if (n1 % self._flush_every_n) == 0:
            self._h5.flush()

    def run(self) -> None:
        pending: "list[Sample]" = []
        pending_demo: Optional[str] = None

        try:
            while not self._stop_evt.is_set():
                try:
                    item = self._queue.get(timeout=0.25)
                except queue.Empty:
                    item = None

                if item is None:
                    if pending_demo is not None and pending:
                        self._append_batch(pending_demo, pending)
                        pending = []
                    continue

                if isinstance(item, Command):
                    if item.kind == "start_demo":
                        if item.demo_name is None:
                            continue
                        if pending_demo is not None and pending:
                            self._append_batch(pending_demo, pending)
                            pending = []
                        pending_demo = item.demo_name
                        self._ensure_file_open()
                        self._create_demo_if_needed(item.demo_name)
                        self._current_demo = item.demo_name
                    elif item.kind == "stop_demo":
                        if pending_demo is not None and pending:
                            self._append_batch(pending_demo, pending)
                            pending = []
                        if item.demo_name is not None:
                            self._finalize_demo(item.demo_name)
                        self._current_demo = None
                        pending_demo = None
                        if self._h5 is not None:
                            self._h5.flush()
                    elif item.kind == "close":
                        break
                    continue

                if isinstance(item, Sample):
                    if pending_demo is None:
                        pending_demo = item.demo_name
                    if item.demo_name != pending_demo:
                        if pending:
                            self._append_batch(pending_demo, pending)
                            pending = []
                        pending_demo = item.demo_name
                    pending.append(item)
                    if len(pending) >= self._batch_size:
                        assert pending_demo is not None
                        self._append_batch(pending_demo, pending)
                        pending = []

            if pending_demo is not None and pending:
                self._append_batch(pending_demo, pending)
                pending = []

            if self._current_demo is not None:
                self._finalize_demo(self._current_demo)

        except Exception as exc:  # noqa: BLE001
            self._log("error", f"HDF5 writer thread crashed: {exc!r}")
        finally:
            try:
                if self._h5 is not None:
                    for demo_name in list(self._demo_handles.keys()):
                        self._finalize_demo(demo_name)
                    self._h5.flush()
                    self._h5.close()
            except Exception as exc:  # noqa: BLE001
                self._log("error", f"HDF5 writer thread close failed: {exc!r}")


class DataCollectorNode(Node):
    def __init__(self) -> None:
        super().__init__("data_collector")

        # --- Parameters ---
        self.declare_parameter("output_path", os.path.join(os.getcwd(), "data", "libero_demos.hdf5"))

        # Camera selection via YAML
        # - Prefer defining camera roles (global/wrist) as sources (realsense/oakd)
        # - Optionally override with explicit topics (global_image_topic, wrist_image_topic)
        self.declare_parameter("realsense_color_topic", "/camera/color/image_raw")
        self.declare_parameter("oakd_rgb_topic", "/oak/rgb/image_raw")
        self.declare_parameter("global_camera_source", "realsense")  # realsense|oakd
        self.declare_parameter("wrist_camera_source", "oakd")  # realsense|oakd
        self.declare_parameter("global_image_topic", "")  # optional explicit override
        self.declare_parameter("wrist_image_topic", "")  # optional explicit override

        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("tool_pose_topic", "/ur5/tool_pose")
        # If true, missing/stale gripper messages will drop frames.
        # If false, gripper defaults to 0.0 until messages arrive.
        self.declare_parameter("require_gripper", True)

        # End-effector selection
        # Dataset always stores a scalar gripper state in [0, 1].
        # - robotic_gripper: typically publishes state on /gripper/state (Float32)
        # - qbsofthand: this workspace's qbsofthand_control node is service-driven (no state topic).
        #   In practice, using the last commanded closure (e.g. /gripper/cmd Float32) as a proxy
        #   works well for dataset labels.
        self.declare_parameter("end_effector_type", "robotic_gripper")  # robotic_gripper|qbsofthand
        self.declare_parameter("gripper_state_topic", "")  # optional explicit override
        self.declare_parameter("robotic_gripper_state_topic", "/gripper/state")
        self.declare_parameter("qbsofthand_state_topic", "/gripper/cmd")

        # Image output settings (LIBERO standard expects 224)
        self.declare_parameter("obs_image_size", 224)

        self.declare_parameter(
            "joint_names",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )

        self.declare_parameter("sync_slop_sec", 0.1)
        self.declare_parameter("sync_queue_size", 30)
        self.declare_parameter("pose_max_age_sec", 0.2)
        self.declare_parameter("gripper_max_age_sec", 0.5)
        self.declare_parameter("pose_stamp_zero_is_ref", True)

        # Diagnostics
        self.declare_parameter("stats_period_sec", 2.0)

        self.declare_parameter("queue_maxsize", 400)
        self.declare_parameter("writer_batch_size", 32)
        self.declare_parameter("writer_flush_every_n", 200)
        self.declare_parameter("image_compression", "lzf")  # lzf|gzip|None
        self.declare_parameter("enable_keyboard", False)

        # Go-home settings (service ~/go_home)
        self.declare_parameter("home_joint_positions", [0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.declare_parameter("home_duration_sec", 3.0)
        self.declare_parameter(
            "home_joint_trajectory_topic",
            "/scaled_joint_trajectory_controller/joint_trajectory",
        )

        self._output_path = str(self.get_parameter("output_path").value)
        self._joint_names = list(self.get_parameter("joint_names").value)
        self._pose_max_age = float(self.get_parameter("pose_max_age_sec").value)
        self._gripper_max_age = float(self.get_parameter("gripper_max_age_sec").value)
        self._pose_stamp_zero_is_ref = bool(self.get_parameter("pose_stamp_zero_is_ref").value)
        self._require_gripper = bool(self.get_parameter("require_gripper").value)

        # Enforce LIBERO image size unless you explicitly know your downstream can handle changes.
        self._obs_image_size = int(self.get_parameter("obs_image_size").value)
        if self._obs_image_size != 224:
            self.get_logger().warn(
                f"obs_image_size={self._obs_image_size} is not LIBERO-standard (224). For compatibility, forcing 224."
            )
            self._obs_image_size = 224

        comp_val = self.get_parameter("image_compression").value
        self._image_compression: Optional[str]
        if comp_val is None:
            self._image_compression = None
        else:
            comp = str(comp_val).strip().lower()
            self._image_compression = None if comp in {"", "none", "null"} else comp

        # --- State ---
        self._bridge = CvBridge()
        self._recording = False
        self._demo_index = 0
        self._current_demo_name: Optional[str] = None

        self._latest_tool_pose: Optional[PoseStamped] = None
        self._latest_tool_pose_time: Optional[Time] = None
        self._latest_gripper: Optional[Float32] = None
        self._latest_gripper_time: Optional[Time] = None
        self._cache_lock = threading.Lock()

        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = {}
        self._last_stats_wall = time.time()

        self._warn_last_monotonic: Dict[str, float] = {}

        # --- Queue + writer thread ---
        qmax = int(self.get_parameter("queue_maxsize").value)
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=max(1, qmax))
        self._writer = HDF5WriterThread(
            output_path=self._output_path,
            item_queue=self._queue,
            compression=self._image_compression,
            batch_size=int(self.get_parameter("writer_batch_size").value),
            flush_every_n=int(self.get_parameter("writer_flush_every_n").value),
            logger=self.get_logger(),
        )
        self._writer.start()

        # --- Subscribers ---
        global_topic, wrist_topic = self._resolve_camera_topics()
        joint_topic = str(self.get_parameter("joint_states_topic").value)
        pose_topic = str(self.get_parameter("tool_pose_topic").value)
        gripper_topic = self._resolve_gripper_topic()

        # Go-home publisher
        home_topic = str(self.get_parameter("home_joint_trajectory_topic").value)
        self._home_pub = self.create_publisher(JointTrajectory, home_topic, 10)

        # Separate subscriptions for cached topics
        self.create_subscription(PoseStamped, pose_topic, self._on_tool_pose, 30)
        self.create_subscription(Float32, gripper_topic, self._on_gripper, 30)

        # Time sync for: global image, wrist image, joint states
        self._sub_global = message_filters.Subscriber(self, Image, global_topic)
        self._sub_wrist = message_filters.Subscriber(self, Image, wrist_topic)
        self._sub_joint = message_filters.Subscriber(self, JointState, joint_topic)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_global, self._sub_wrist, self._sub_joint],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_sec").value),
            allow_headerless=False,
        )
        self._sync.registerCallback(self._on_synced)

        # --- Services ---
        self._srv_start = self.create_service(Trigger, "~/start", self._srv_start_cb)
        self._srv_stop = self.create_service(Trigger, "~/stop", self._srv_stop_cb)
        self._srv_go_home = self.create_service(Trigger, "~/go_home", self._srv_go_home_cb)

        # --- Optional keyboard thread ---
        self._keyboard_enabled = bool(self.get_parameter("enable_keyboard").value)
        self._keyboard_thread: Optional[threading.Thread] = None
        self._keyboard_stop_evt = threading.Event()
        if self._keyboard_enabled:
            self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            self._keyboard_thread.start()

        self.get_logger().info(
            "DataCollectorNode ready. Services: ~/start, ~/stop, ~/go_home. "
            f"Global cam: {global_topic} | Wrist cam: {wrist_topic} | Gripper: {gripper_topic} | Output: {self._output_path}"
        )

        period = float(self.get_parameter("stats_period_sec").value)
        if period > 0.0:
            self._stats_timer = self.create_timer(period, self._log_stats)

    def _inc_stat(self, key: str, n: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] = int(self._stats.get(key, 0)) + int(n)

    def _reset_stats(self) -> None:
        with self._stats_lock:
            self._stats = {}
            self._last_stats_wall = time.time()

    def _log_stats(self) -> None:
        if not self._recording:
            return
        with self._stats_lock:
            stats = dict(self._stats)
            self._stats = {}
        if not stats:
            return
        qsize = -1
        try:
            qsize = self._queue.qsize()
        except Exception:
            pass
        ordered = ", ".join([f"{k}={v}" for k, v in sorted(stats.items(), key=lambda kv: kv[0])])
        self.get_logger().info(f"Recorder stats: {ordered} | queue={qsize}")

    def _warn_throttled(self, key: str, msg: str, period_sec: float = 2.0) -> None:
        now = time.monotonic()
        last = float(self._warn_last_monotonic.get(key, 0.0))
        if (now - last) < period_sec:
            return
        self._warn_last_monotonic[key] = now
        self.get_logger().warn(msg)

    def _resolve_camera_topics(self) -> Tuple[str, str]:
        """Resolve global/wrist image topics based on role mapping parameters."""
        global_override = str(self.get_parameter("global_image_topic").value).strip()
        wrist_override = str(self.get_parameter("wrist_image_topic").value).strip()
        if global_override and wrist_override:
            return global_override, wrist_override

        realsense_topic = str(self.get_parameter("realsense_color_topic").value).strip()
        oakd_topic = str(self.get_parameter("oakd_rgb_topic").value).strip()
        global_src = str(self.get_parameter("global_camera_source").value).strip().lower()
        wrist_src = str(self.get_parameter("wrist_camera_source").value).strip().lower()

        def pick(src: str) -> str:
            if src == "realsense":
                return realsense_topic
            if src == "oakd":
                return oakd_topic
            self.get_logger().warn(f"Unknown camera source '{src}', falling back to realsense")
            return realsense_topic

        global_topic = global_override if global_override else pick(global_src)
        wrist_topic = wrist_override if wrist_override else pick(wrist_src)
        return global_topic, wrist_topic

    def _resolve_gripper_topic(self) -> str:
        override = str(self.get_parameter("gripper_state_topic").value).strip()
        if override:
            return override

        ee = str(self.get_parameter("end_effector_type").value).strip().lower()
        if ee == "robotic_gripper":
            return str(self.get_parameter("robotic_gripper_state_topic").value)
        if ee == "qbsofthand":
            return str(self.get_parameter("qbsofthand_state_topic").value)

        self.get_logger().warn(f"Unknown end_effector_type '{ee}', falling back to robotic_gripper")
        return str(self.get_parameter("robotic_gripper_state_topic").value)

    # ----------------- Cached topics -----------------
    def _on_tool_pose(self, msg: PoseStamped) -> None:
        with self._cache_lock:
            self._latest_tool_pose = msg
            self._latest_tool_pose_time = Time.from_msg(msg.header.stamp)

    def _on_gripper(self, msg: Float32) -> None:
        now = self.get_clock().now()
        with self._cache_lock:
            self._latest_gripper = msg
            self._latest_gripper_time = now

    def _get_cached_pose_and_gripper(
        self, ref_time: Time
    ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, float]], Optional[str]]:
        """Return ((eef_pos, eef_quat, gripper), None) if ok; otherwise (None, reason_key)."""
        with self._cache_lock:
            pose_msg = self._latest_tool_pose
            pose_t = self._latest_tool_pose_time
            grip_msg = self._latest_gripper
            grip_t = self._latest_gripper_time

        if pose_msg is None or pose_t is None:
            return None, "no_pose"

        # Some broadcasters may publish a zero stamp; optionally treat it as ref_time.
        if self._pose_stamp_zero_is_ref and pose_t.nanoseconds == 0:
            pose_t = ref_time

        # If pose_max_age_sec <= 0, disable pose age check.
        if self._pose_max_age > 0.0:
            if abs((ref_time - pose_t).nanoseconds) * 1e-9 > self._pose_max_age:
                return None, "pose_stale"

        if grip_msg is None or grip_t is None:
            if self._require_gripper:
                return None, "no_gripper"
            gripper = 0.0
        else:
            # If gripper_max_age_sec <= 0, disable gripper age check.
            if self._gripper_max_age > 0.0:
                if abs((self.get_clock().now() - grip_t).nanoseconds) * 1e-9 > self._gripper_max_age:
                    if self._require_gripper:
                        return None, "gripper_stale"
            gripper = float(_clamp(float(grip_msg.data), 0.0, 1.0))

        p = pose_msg.pose.position
        q = pose_msg.pose.orientation

        eef_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        eef_quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
        return (eef_pos, eef_quat, gripper), None

    # ----------------- Joint mapping -----------------
    def _map_joint_positions(self, msg: JointState) -> Optional[np.ndarray]:
        if not msg.name or not msg.position:
            return None
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        out = np.zeros(6, dtype=np.float32)
        for j, joint_name in enumerate(self._joint_names[:6]):
            idx = name_to_idx.get(joint_name)
            if idx is None or idx >= len(msg.position):
                missing = [jn for jn in self._joint_names[:6] if jn not in name_to_idx]
                self._warn_throttled(
                    "joint_map",
                    "Joint mapping failed. Missing joints: "
                    + ", ".join(missing)
                    + " | Example joint_states.name: "
                    + ", ".join(list(msg.name)[:8]),
                    period_sec=2.0,
                )
                return None
            out[j] = float(msg.position[idx])
        return out

    # ----------------- Sync callback -----------------
    def _on_synced(self, global_img: Image, wrist_img: Image, joint_msg: JointState) -> None:
        if not self._recording or self._current_demo_name is None:
            self._inc_stat("not_recording")
            return

        self._inc_stat("synced_cb")

        ref_time = Time.from_msg(global_img.header.stamp)
        cached, reason = self._get_cached_pose_and_gripper(ref_time)
        if cached is None:
            reason = reason or "missing_pose_or_gripper"
            self._inc_stat(reason)
            if reason == "no_pose":
                self._warn_throttled("no_pose", "No tool pose received yet. Check tool_pose_topic in YAML.")
            elif reason == "pose_stale":
                self._warn_throttled(
                    "pose_stale",
                    "Tool pose too old vs image timestamp. Check time sync / pose_max_age_sec.",
                )
            elif reason == "no_gripper":
                self._warn_throttled(
                    "no_gripper",
                    "No gripper message yet. Check gripper_state_topic or set require_gripper:=false.",
                )
            elif reason == "gripper_stale":
                self._warn_throttled(
                    "gripper_stale",
                    "Gripper message too old. Increase gripper_max_age_sec or disable require_gripper.",
                )
            return
        eef_pos, eef_quat, gripper = cached

        joint_pos = self._map_joint_positions(joint_msg)
        if joint_pos is None:
            self._inc_stat("joint_map_fail")
            return

        try:
            global_bgr = self._bridge.imgmsg_to_cv2(global_img, desired_encoding="bgr8")
            wrist_bgr = self._bridge.imgmsg_to_cv2(wrist_img, desired_encoding="bgr8")
            agentview_rgb = _center_crop_square_and_resize_rgb(global_bgr, self._obs_image_size)
            eye_in_hand_rgb = _center_crop_square_and_resize_rgb(wrist_bgr, self._obs_image_size)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Image preprocess failed: {exc!r}")
            self._inc_stat("image_fail")
            return

        rotvec = _quat_to_rotvec_xyzw(eef_quat)
        action = np.array(
            [
                float(eef_pos[0]),
                float(eef_pos[1]),
                float(eef_pos[2]),
                float(rotvec[0]),
                float(rotvec[1]),
                float(rotvec[2]),
                float(gripper),
            ],
            dtype=np.float32,
        )

        sample = Sample(
            demo_name=self._current_demo_name,
            agentview_rgb=agentview_rgb,
            eye_in_hand_rgb=eye_in_hand_rgb,
            robot0_joint_pos=joint_pos.astype(np.float32, copy=False),
            robot0_eef_pos=eef_pos.astype(np.float32, copy=False),
            robot0_eef_quat=eef_quat.astype(np.float32, copy=False),
            actions=action,
        )

        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            # Backpressure: drop sample rather than blocking callbacks.
            self.get_logger().warn("Writer queue full; dropping sample")
            self._inc_stat("queue_full")
        else:
            self._inc_stat("enqueued")

    # ----------------- Services / control -----------------
    def _srv_start_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if self._recording:
            res.success = False
            res.message = "Already recording"
            return res

        self._reset_stats()

        demo_name = f"demo_{self._demo_index}"
        self._demo_index += 1
        self._current_demo_name = demo_name
        self._recording = True

        try:
            self._queue.put_nowait(Command(kind="start_demo", demo_name=demo_name))
        except queue.Full:
            self._recording = False
            self._current_demo_name = None
            res.success = False
            res.message = "Queue full; cannot start demo"
            return res

        res.success = True
        res.message = f"Started recording {demo_name}"
        self.get_logger().info(res.message)
        return res

    def _srv_stop_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if not self._recording or self._current_demo_name is None:
            res.success = False
            res.message = "Not recording"
            return res

        demo_name = self._current_demo_name
        self._recording = False
        self._current_demo_name = None

        try:
            self._queue.put_nowait(Command(kind="stop_demo", demo_name=demo_name))
        except queue.Full:
            # Worst case: writer thread will finalize on close.
            pass

        res.success = True
        res.message = f"Stopped recording {demo_name}"
        self.get_logger().info(res.message)
        return res

    def _srv_go_home_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        """Publish a JointTrajectory to move the robot back to the configured home joint pose."""
        home_positions = list(self.get_parameter("home_joint_positions").value)
        if len(home_positions) < 6:
            res.success = False
            res.message = "home_joint_positions must have 6 elements"
            return res
        home_positions = [float(x) for x in home_positions[:6]]

        duration = float(self.get_parameter("home_duration_sec").value)
        if duration <= 0.0:
            duration = 3.0

        jt = JointTrajectory()
        jt.joint_names = [str(n) for n in self._joint_names[:6]]

        pt = JointTrajectoryPoint()
        pt.positions = home_positions
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        pt.time_from_start = DurationMsg(sec=sec, nanosec=nsec)
        jt.points = [pt]

        self._home_pub.publish(jt)
        res.success = True
        res.message = f"Published home trajectory ({duration:.2f}s)"
        self.get_logger().info(res.message)
        return res

    # ----------------- Keyboard (optional) -----------------
    def _keyboard_loop(self) -> None:
        # Very small, robust keyboard loop: reads single lines.
        # Use: r + Enter, s + Enter, q + Enter
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
                rclpy.shutdown()
                break

    # ----------------- Shutdown -----------------
    def destroy_node(self) -> bool:
        try:
            self._keyboard_stop_evt.set()
        except Exception:
            pass
        try:
            self._writer.stop()
            self._writer.join(timeout=3.0)
        except Exception:
            pass
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
