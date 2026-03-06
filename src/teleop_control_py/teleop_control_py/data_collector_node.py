#!/usr/bin/env python3
"""基于定时器主动抓帧的数据采集节点。"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
import h5py
from builtin_interfaces.msg import Duration as DurationMsg
from controller_manager_msgs.srv import SwitchController
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32, Float32MultiArray
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .camera_client import OAKClient, RealSenseClient
from .hdf5_writer import Command, HDF5WriterThread, Sample
from .transform_utils import _clamp, center_crop_square_and_resize_rgb, compose_eef_action


class DataCollectorNode(Node):
    """负责 ROS 状态缓存、相机调度和 HDF5 采样入队。"""

    def __init__(self) -> None:
        super().__init__("data_collector")

        self.declare_parameter("output_path", os.path.join(os.getcwd(), "data", "libero_demos.hdf5"))
        self.declare_parameter("record_fps", 10.0)
        self.declare_parameter("global_camera_source", "realsense")
        self.declare_parameter("wrist_camera_source", "oakd")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("tool_pose_topic", "/tcp_pose_broadcaster/pose")
        self.declare_parameter("require_gripper", True)
        self.declare_parameter("end_effector_type", "robotic_gripper")
        self.declare_parameter("gripper_state_topic", "")
        self.declare_parameter("robotic_gripper_state_topic", "/gripper/state")
        self.declare_parameter("qbsofthand_state_topic", "/gripper/cmd")
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
        self.declare_parameter("pose_max_age_sec", 0.2)
        self.declare_parameter("gripper_max_age_sec", 0.5)
        self.declare_parameter("pose_stamp_zero_is_ref", True)
        self.declare_parameter("pose_use_received_time_fallback", True)
        self.declare_parameter("stats_period_sec", 2.0)
        self.declare_parameter("queue_maxsize", 400)
        self.declare_parameter("writer_batch_size", 32)
        self.declare_parameter("writer_flush_every_n", 200)
        self.declare_parameter("image_compression", "lzf")
        self.declare_parameter("preview_enabled", True)
        self.declare_parameter("preview_fps", 30.0)
        self.declare_parameter("preview_global_topic", "/data_collector/preview/global/image_raw")
        self.declare_parameter("preview_wrist_topic", "/data_collector/preview/wrist/image_raw")
        self.declare_parameter("enable_keyboard", False)
        self.declare_parameter(
            "home_joint_positions",
            [1.524178, -2.100060, 1.864580, -1.345048, -1.575888, 1.528195],
        )
        self.declare_parameter("home_duration_sec", 3.0)
        self.declare_parameter(
            "home_joint_trajectory_topic",
            "/scaled_joint_trajectory_controller/joint_trajectory",
        )
        self.declare_parameter("teleop_controller", "forward_position_controller")
        self.declare_parameter("trajectory_controller", "scaled_joint_trajectory_controller")

        self._output_path = str(self.get_parameter("output_path").value)
        self._record_fps = max(0.1, float(self.get_parameter("record_fps").value))
        self._joint_names = list(self.get_parameter("joint_names").value)
        self._pose_max_age = float(self.get_parameter("pose_max_age_sec").value)
        self._gripper_max_age = float(self.get_parameter("gripper_max_age_sec").value)
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
        self._latest_pose_pos: Optional[np.ndarray] = None
        self._latest_pose_quat: Optional[np.ndarray] = None
        self._latest_pose_time: Optional[Time] = None
        self._latest_pose_receive_time: Optional[Time] = None
        self._latest_gripper: Optional[float] = None
        self._latest_gripper_time: Optional[Time] = None
        self._latest_global_bgr: Optional[np.ndarray] = None
        self._latest_wrist_bgr: Optional[np.ndarray] = None
        self._latest_frame_time: Optional[float] = None

        self._stats: Dict[str, int] = {}
        self._homing_in_progress = False
        self._keyboard_thread: Optional[threading.Thread] = None
        self._keyboard_stop_evt = threading.Event()
        self._bridge = CvBridge()
        self._preview_enabled = bool(self.get_parameter("preview_enabled").value)
        self._preview_global_topic = str(self.get_parameter("preview_global_topic").value)
        self._preview_wrist_topic = str(self.get_parameter("preview_wrist_topic").value)
        self._preview_global_pub = None
        self._preview_wrist_pub = None
        self._preview_timer = None

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

        self._camera_instances: Dict[str, object] = {}
        global_source = self._normalize_camera_source(self.get_parameter("global_camera_source").value)
        wrist_source = self._normalize_camera_source(self.get_parameter("wrist_camera_source").value)
        self.global_cam = self._get_or_create_camera(global_source)
        self.wrist_cam = self._get_or_create_camera(wrist_source)
        self._global_camera_source = global_source
        self._wrist_camera_source = wrist_source

        joint_topic = str(self.get_parameter("joint_states_topic").value)
        pose_topic = str(self.get_parameter("tool_pose_topic").value)
        gripper_topic = self._resolve_gripper_topic()

        self.create_subscription(JointState, joint_topic, self._on_joint_state, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, pose_topic, self._on_tool_pose, qos_profile_sensor_data)
        self.create_subscription(Float32, gripper_topic, self._on_gripper, qos_profile_sensor_data)

        home_topic = str(self.get_parameter("home_joint_trajectory_topic").value)
        self._home_pub = self.create_publisher(JointTrajectory, home_topic, 10)
        self._switch_ctrl_client = self.create_client(SwitchController, "/controller_manager/switch_controller")
        self._record_stats_pub = self.create_publisher(Float32MultiArray, "~/record_stats", 10)

        if self._preview_enabled:
            self._preview_global_pub = self.create_publisher(Image, self._preview_global_topic, qos_profile_sensor_data)
            self._preview_wrist_pub = self.create_publisher(Image, self._preview_wrist_topic, qos_profile_sensor_data)
            preview_fps = max(0.1, float(self.get_parameter("preview_fps").value))
            self._preview_timer = self.create_timer(1.0 / preview_fps, self._preview_step)

        self._srv_start = self.create_service(Trigger, "~/start", self._srv_start_cb)
        self._srv_stop = self.create_service(Trigger, "~/stop", self._srv_stop_cb)
        self._srv_go_home = self.create_service(Trigger, "~/go_home", self._srv_go_home_cb)

        if bool(self.get_parameter("enable_keyboard").value):
            self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            self._keyboard_thread.start()

        period = float(self.get_parameter("stats_period_sec").value)
        if period > 0.0:
            self._stats_timer = self.create_timer(period, self._log_stats)

        self.get_logger().info(
            "DataCollectorNode ready. Services: ~/start, ~/stop, ~/go_home. "
            f"Global camera={self._global_camera_source}, Wrist camera={self._wrist_camera_source}, "
            f"Gripper={gripper_topic}, Output={self._output_path}, FPS={self._record_fps:.2f}"
        )

        if self._demo_index > 0:
            self.get_logger().info(f"Detected existing demos in HDF5. Next demo index: {self._demo_index}")

        if self._preview_enabled:
            self.get_logger().info(
                "Preview topics enabled. "
                f"Global={self._preview_global_topic}, Wrist={self._preview_wrist_topic}"
            )

    def _normalize_camera_source(self, value: object) -> str:
        source = str(value).strip().lower()
        if source in {"realsense", "oakd"}:
            return source
        self.get_logger().warn(f"未知相机来源 '{source}'，回退到 realsense。")
        return "realsense"

    def _get_or_create_camera(self, source: str) -> Optional[object]:
        if source in self._camera_instances:
            return self._camera_instances[source]

        try:
            if source == "realsense":
                camera = RealSenseClient(logger=self.get_logger())
            elif source == "oakd":
                camera = OAKClient(logger=self.get_logger())
            else:
                raise ValueError(f"Unsupported camera source: {source}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"初始化 {source} 相机失败: {exc!r}")
            camera = None

        self._camera_instances[source] = camera
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

        try:
            qsize = self._queue.qsize()
        except Exception:
            qsize = -1

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

        with self._cache_lock:
            self._latest_joint_pos = joint_pos

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
            pose_pos = None if self._latest_pose_pos is None else self._latest_pose_pos.copy()
            pose_quat = None if self._latest_pose_quat is None else self._latest_pose_quat.copy()
            pose_time = self._latest_pose_time
            pose_receive_time = self._latest_pose_receive_time
            gripper = self._latest_gripper
            gripper_time = self._latest_gripper_time

        if joint_pos is None:
            return None, "no_joint", None
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
            if abs((self.get_clock().now() - gripper_time).nanoseconds) * 1e-9 > self._gripper_max_age:
                if self._require_gripper:
                    return None, "gripper_stale", None

        return (joint_pos, pose_pos, pose_quat, float(gripper)), None, None

    def _pull_camera_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.global_cam is None or self.wrist_cam is None:
            return None, None

        with self._camera_pull_lock:
            global_bgr = self.global_cam.get_bgr_frame()
            wrist_bgr = self.wrist_cam.get_bgr_frame()
        return global_bgr, wrist_bgr

    def _cache_camera_frames(self, global_bgr: Optional[np.ndarray], wrist_bgr: Optional[np.ndarray]) -> None:
        if global_bgr is None or wrist_bgr is None:
            return

        with self._cache_lock:
            self._latest_global_bgr = np.ascontiguousarray(global_bgr)
            self._latest_wrist_bgr = np.ascontiguousarray(wrist_bgr)
            self._latest_frame_time = time.monotonic()

    def _get_cached_camera_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._cache_lock:
            global_bgr = None if self._latest_global_bgr is None else self._latest_global_bgr.copy()
            wrist_bgr = None if self._latest_wrist_bgr is None else self._latest_wrist_bgr.copy()
        return global_bgr, wrist_bgr

    def _pull_and_cache_camera_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        global_bgr, wrist_bgr = self._pull_camera_frames()
        self._cache_camera_frames(global_bgr, wrist_bgr)
        return global_bgr, wrist_bgr

    def _publish_preview_frames(self, global_bgr: Optional[np.ndarray], wrist_bgr: Optional[np.ndarray]) -> None:
        if not self._preview_enabled:
            return

        stamp = self.get_clock().now().to_msg()

        if global_bgr is not None and self._preview_global_pub is not None:
            try:
                msg = self._bridge.cv2_to_imgmsg(global_bgr, encoding="bgr8")
                msg.header.stamp = stamp
                msg.header.frame_id = "data_collector_global_preview"
                self._preview_global_pub.publish(msg)
            except Exception as exc:  # noqa: BLE001
                self._warn_throttled("preview_global_pub", f"发布全局预览图失败: {exc!r}")

        if wrist_bgr is not None and self._preview_wrist_pub is not None:
            try:
                msg = self._bridge.cv2_to_imgmsg(wrist_bgr, encoding="bgr8")
                msg.header.stamp = stamp
                msg.header.frame_id = "data_collector_wrist_preview"
                self._preview_wrist_pub.publish(msg)
            except Exception as exc:  # noqa: BLE001
                self._warn_throttled("preview_wrist_pub", f"发布手部预览图失败: {exc!r}")

    def _preview_step(self) -> None:
        if self.global_cam is None or self.wrist_cam is None:
            return

        try:
            global_bgr, wrist_bgr = self._pull_and_cache_camera_frames()
        except Exception as exc:  # noqa: BLE001
            self._warn_throttled("preview_pull", f"拉取预览图像失败: {exc!r}")
            return

        if global_bgr is None or wrist_bgr is None:
            return

        self._publish_preview_frames(global_bgr, wrist_bgr)

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
            global_bgr, wrist_bgr = self._get_cached_camera_frames()
            if global_bgr is None or wrist_bgr is None:
                global_bgr, wrist_bgr = self._pull_and_cache_camera_frames()
        except Exception as exc:  # noqa: BLE001
            self._warn_throttled("camera_pull", f"主动拉取相机图像失败: {exc!r}")
            self._inc_stat("camera_pull_fail")
            return

        if global_bgr is None or wrist_bgr is None:
            self._inc_stat("camera_empty")
            return

        ref_time = self.get_clock().now()
        cached, reason, detail = self._get_cached_state(ref_time)
        if cached is None:
            reason = reason or "missing_state"
            self._inc_stat(reason)
            if reason == "no_joint":
                self._warn_throttled("no_joint", "尚未收到有效 /joint_states，当前帧跳过。")
            elif reason == "no_pose":
                self._warn_throttled("no_pose", "尚未收到有效末端位姿，当前帧跳过。")
            elif reason == "pose_stale":
                self._warn_throttled("pose_stale", detail or "末端位姿过旧，建议检查 tool_pose_topic 和时延。")
            elif reason == "no_gripper":
                self._warn_throttled("no_gripper", "尚未收到夹爪状态，当前帧跳过。")
            elif reason == "gripper_stale":
                self._warn_throttled("gripper_stale", "夹爪状态过旧，当前帧跳过。")
            return

        joint_pos, eef_pos, eef_quat, gripper = cached

        try:
            agentview_rgb = center_crop_square_and_resize_rgb(global_bgr, self._obs_image_size)
            eye_in_hand_rgb = center_crop_square_and_resize_rgb(wrist_bgr, self._obs_image_size)
        except Exception as exc:  # noqa: BLE001
            self._warn_throttled("image_fail", f"图像预处理失败: {exc!r}")
            self._inc_stat("image_fail")
            return

        # Record the realized robot-side action, so any teleop-side deadzone/limiting is already reflected here.
        action = compose_eef_action(eef_pos, eef_quat, gripper)

        # 严格按照相机角色写入数据集字段，避免 agentview / wrist 对调。
        sample = Sample(
            demo_name=demo_name,
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

            demo_name = f"demo_{self._demo_index}"
            self._demo_index += 1

            try:
                self._queue.put_nowait(Command(kind="start_demo", demo_name=demo_name))
            except queue.Full:
                res.success = False
                res.message = "Queue full; cannot start demo"
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
        res.message = f"Started recording {demo_name} at {self._record_fps:.2f} Hz"
        self.get_logger().info(res.message)
        return res

    def _srv_stop_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        with self._record_lock:
            if not self._recording or self._current_demo_name is None:
                res.success = False
                res.message = "Not recording"
                return res

            demo_name = self._current_demo_name
            self._recording = False
            self._current_demo_name = None

            if self._capture_timer is not None:
                try:
                    self._capture_timer.cancel()
                except Exception:
                    pass
                self._capture_timer = None

        try:
            self._queue.put_nowait(Command(kind="stop_demo", demo_name=demo_name))
        except queue.Full:
            self.get_logger().warn("停止录制时队列已满，最终关闭时会尝试补齐元数据。")

        with self._record_lock:
            self._recent_frame_times.clear()
        self._publish_record_stats()

        res.success = True
        res.message = f"Stopped recording {demo_name}"
        self.get_logger().info(res.message)
        return res

    def _srv_go_home_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if self._homing_in_progress:
            res.success = False
            res.message = "Homing sequence is already in progress!"
            return res

        home_positions = list(self.get_parameter("home_joint_positions").value)
        if len(home_positions) < 6:
            res.success = False
            res.message = "home_joint_positions must have 6 elements"
            return res

        self._homing_in_progress = True
        threading.Thread(target=self._execute_go_home_sequence, args=(home_positions,), daemon=True).start()

        res.success = True
        res.message = "Homing sequence started (controllers will switch automatically)"
        self.get_logger().info(res.message)
        return res

    def _execute_go_home_sequence(self, home_positions: list) -> None:
        try:
            home_positions = [float(value) for value in home_positions[:6]]
            duration = float(self.get_parameter("home_duration_sec").value)
            if duration <= 0.0:
                duration = 3.0

            teleop_ctrl = str(self.get_parameter("teleop_controller").value)
            traj_ctrl = str(self.get_parameter("trajectory_controller").value)

            req_to_traj = SwitchController.Request()
            req_to_traj.activate_controllers = [traj_ctrl]
            req_to_traj.deactivate_controllers = [teleop_ctrl]
            req_to_traj.strictness = SwitchController.Request.BEST_EFFORT

            if self._switch_ctrl_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Switching to {traj_ctrl}...")
                self._switch_ctrl_client.call_async(req_to_traj)
                time.sleep(0.5)
            else:
                self.get_logger().warn("Controller manager not available, attempting to publish anyway.")

            trajectory = JointTrajectory()
            trajectory.joint_names = [str(name) for name in self._joint_names[:6]]

            point = JointTrajectoryPoint()
            point.positions = home_positions
            seconds = int(duration)
            nanoseconds = int((duration - seconds) * 1e9)
            point.time_from_start = DurationMsg(sec=seconds, nanosec=nanoseconds)
            trajectory.points = [point]

            self._home_pub.publish(trajectory)
            self.get_logger().info(f"Published home trajectory, waiting {duration:.2f}s...")

            time.sleep(duration + 0.5)

            req_to_teleop = SwitchController.Request()
            req_to_teleop.activate_controllers = [teleop_ctrl]
            req_to_teleop.deactivate_controllers = [traj_ctrl]
            req_to_teleop.strictness = SwitchController.Request.BEST_EFFORT

            if self._switch_ctrl_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Restoring {teleop_ctrl}...")
                self._switch_ctrl_client.call_async(req_to_teleop)
                self.get_logger().info("Homing sequence complete. Teleop restored.")
        finally:
            self._homing_in_progress = False

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

        if active_demo is not None:
            try:
                self._queue.put_nowait(Command(kind="stop_demo", demo_name=active_demo))
            except queue.Full:
                pass

        for camera in {id(camera): camera for camera in self._camera_instances.values() if camera is not None}.values():
            try:
                camera.stop()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"关闭相机失败: {exc!r}")

        try:
            self._writer.stop()
            self._writer.join(timeout=3.0)
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
