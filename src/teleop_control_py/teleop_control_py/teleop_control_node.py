#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from qbsofthand_control.srv import SetClosure
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32


class LowPassFilter:
    """Simple exponential moving average for jitter reduction."""

    def __init__(self, alpha: float, initial: Optional[np.ndarray] = None) -> None:
        self.alpha = float(alpha)
        self.state = initial

    def reset(self, value: Optional[np.ndarray] = None) -> None:
        self.state = value

    def apply(self, value: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = value
            return value
        self.state = self.alpha * value + (1.0 - self.alpha) * self.state
        return self.state


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


class TeleopControlNode(Node):
    """
    Teleoperation bridge from MediaPipe hand tracking to UR5 + SoftHand.

    Key ideas:
    - Deadman switch: pinch gesture OR holding space bar keeps motion enabled.
    - Relative control: capture initial hand pose and robot pose, then apply deltas.
    - Low-pass filter on target position for smoother motion.
    - Gripper: thumb-index distance mapped to [0.0, 1.0].

    How to tune:
    - Scale factor: parameter `scale_factor` (default 0.5) multiplies hand deltas before sending.
    - Axis mapping: parameter `axis_mapping` defines which hand axis drives each robot axis, e.g. [0, 2, 1]
      maps hand X->robot X, hand Z->robot Y, hand Y->robot Z.
    """

    def __init__(self) -> None:
        super().__init__("teleop_control_node")

        # Parameters for easy runtime tuning
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("robot_pose_topic", "/tcp_pose_broadcaster/pose")
        self.declare_parameter("target_pose_topic", "/target_pose")
        self.declare_parameter("target_frame_id", "base")
        self.declare_parameter("gripper_cmd_topic", "/gripper/cmd")
        self.declare_parameter("scale_factor", 0.5)
        self.declare_parameter("axis_mapping", [0, 1, 2])
        self.declare_parameter("smoothing_alpha", 0.2)
        self.declare_parameter("pinch_threshold_px", 45.0)
        self.declare_parameter("gripper_open_dist_px", 100.0)
        self.declare_parameter("gripper_close_dist_px", 20.0)
        self.declare_parameter("gripper_open_dist_m", 0.12)
        self.declare_parameter("gripper_close_dist_m", 0.03)
        self.declare_parameter("depth_unit_scale", 0.001)  # 16UC1 in mm -> m
        self.declare_parameter("hand_position_source", "hybrid")  # depth|normalized|hybrid
        self.declare_parameter("depth_min_m", 0.1)
        self.declare_parameter("depth_max_m", 2.0)
        self.declare_parameter("use_pinch_deadman", False)
        self.declare_parameter("space_deadman_hold_sec", 0.3)
        self.declare_parameter("gripper_requires_deadman", True)
        self.declare_parameter("gripper_step", 0.2)
        self.declare_parameter("gripper_duration_sec", 0.5)
        self.declare_parameter("gripper_speed_ratio", 1.0)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        robot_pose_topic = self.get_parameter("robot_pose_topic").get_parameter_value().string_value
        target_pose_topic = self.get_parameter("target_pose_topic").get_parameter_value().string_value
        self.target_frame_id = self.get_parameter("target_frame_id").get_parameter_value().string_value
        gripper_cmd_topic = self.get_parameter("gripper_cmd_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.scale_factor = float(self.get_parameter("scale_factor").get_parameter_value().double_value)
        self.axis_mapping = self._validate_axis_mapping(
            self.get_parameter("axis_mapping").get_parameter_value().integer_array_value
        )
        smoothing_alpha = float(self.get_parameter("smoothing_alpha").get_parameter_value().double_value)
        self.pinch_threshold_px = float(
            self.get_parameter("pinch_threshold_px").get_parameter_value().double_value
        )
        self.gripper_open_dist_px = float(
            self.get_parameter("gripper_open_dist_px").get_parameter_value().double_value
        )
        self.gripper_close_dist_px = float(
            self.get_parameter("gripper_close_dist_px").get_parameter_value().double_value
        )
        self.gripper_open_dist_m = float(
            self.get_parameter("gripper_open_dist_m").get_parameter_value().double_value
        )
        self.gripper_close_dist_m = float(
            self.get_parameter("gripper_close_dist_m").get_parameter_value().double_value
        )
        self.depth_unit_scale = float(
            self.get_parameter("depth_unit_scale").get_parameter_value().double_value
        )
        self.hand_position_source = (
            self.get_parameter("hand_position_source").get_parameter_value().string_value.strip().lower()
        )
        self.depth_min_m = float(self.get_parameter("depth_min_m").get_parameter_value().double_value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").get_parameter_value().double_value)
        self.use_pinch_deadman = bool(
            self.get_parameter("use_pinch_deadman").get_parameter_value().bool_value
        )
        self.space_deadman_hold_sec = float(
            self.get_parameter("space_deadman_hold_sec").get_parameter_value().double_value
        )
        self.gripper_requires_deadman = bool(
            self.get_parameter("gripper_requires_deadman").get_parameter_value().bool_value
        )
        self.gripper_step = float(
            self.get_parameter("gripper_step").get_parameter_value().double_value
        )
        self.gripper_duration_sec = float(
            self.get_parameter("gripper_duration_sec").get_parameter_value().double_value
        )
        self.gripper_speed_ratio = float(
            self.get_parameter("gripper_speed_ratio").get_parameter_value().double_value
        )

        # ROS interfaces
        self.bridge = CvBridge()
        self.create_subscription(Image, image_topic, self._image_callback, 10)
        self.create_subscription(Image, depth_topic, self._depth_callback, 5)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_callback, 1)
        self.create_subscription(PoseStamped, robot_pose_topic, self._pose_callback, 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, target_pose_topic, 10)
        self.gripper_pub = self.create_publisher(Float32, gripper_cmd_topic, 10)
        self.gripper_client = self.create_client(SetClosure, "/qbsofthand_control_node/set_closure")
        if not self.gripper_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("SoftHand service not available; will retry on demand")
        self._last_gripper_cmd: Optional[float] = None
        self._last_gripper_call_ns = 0
        self._gripper_delta_threshold = 0.02
        self._gripper_min_interval_ns = int(0.1 * 1e9)

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # State
        self.latest_robot_pose: Optional[Pose] = None
        self.initial_hand_pos: Optional[np.ndarray] = None
        self.initial_robot_pos: Optional[np.ndarray] = None  # 3-vector
        self.initial_robot_orientation: Optional[np.ndarray] = None  # 4-vector
        self.deadman_active = False
        self._space_deadman_until_ns: int = 0
        self.depth_image: Optional[np.ndarray] = None
        self.depth_stamp_ns: int = 0
        self.depth_encoding: Optional[str] = None
        self.depth_info: Optional[CameraInfo] = None
        self.fx = self.fy = self.cx = self.cy = None

        # Filtering per-axis for the target position
        self.pos_filter = LowPassFilter(alpha=smoothing_alpha)

        self.get_logger().info("TeleopControlNode ready. Hold pinch or space to move.")

    def _validate_axis_mapping(self, mapping: List[int]) -> List[int]:
        if len(mapping) != 3:
            self.get_logger().warn("axis_mapping must have 3 entries; falling back to [0, 1, 2]")
            return [0, 1, 2]
        if sorted(mapping) != [0, 1, 2]:
            self.get_logger().warn("axis_mapping must be a permutation of [0,1,2]; falling back to [0,1,2]")
            return [0, 1, 2]
        return list(mapping)

    def _pose_callback(self, msg: PoseStamped) -> None:
        self.latest_robot_pose = msg.pose

    def _image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # pragma: no cover - defensive for camera errors
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        height, width, _ = cv_image.shape
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        status_text = "IDLE"
        status_color = (0, 0, 255)
        wrist_source = "-"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(
                cv_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

            wrist = hand_landmarks.landmark[0]
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            wrist_pos, wrist_source = self._get_hand_position(wrist, width, height)
            deadman_gesture = self._is_pinch(thumb_tip, index_tip, width, height)
            now_ns = self.get_clock().now().nanoseconds
            key_deadman = now_ns < self._space_deadman_until_ns
            deadman = key_deadman or (self.use_pinch_deadman and deadman_gesture)

            # Gripper command from pinch distance (prefers metric if depth+info available)
            gripper_cmd = self._gripper_from_distance(thumb_tip, index_tip, width, height)
            if not self.gripper_requires_deadman or deadman:
                self._publish_gripper(gripper_cmd)

            if deadman and self.latest_robot_pose is not None and wrist_pos is not None:
                if not self.deadman_active:
                    # Arm was idle; capture anchors
                    self.initial_hand_pos = wrist_pos.copy()
                    self.initial_robot_pos = self._position_to_array(self.latest_robot_pose.position)
                    self.initial_robot_orientation = self._orientation_to_array(
                        self.latest_robot_pose.orientation
                    )
                    self.pos_filter.reset(self.initial_robot_pos.copy())
                    self.deadman_active = True
                    self.get_logger().info("Deadman engaged; starting relative control")

                target_pose = self._compute_target_pose(wrist_pos)
                if target_pose is not None:
                    self._publish_target_pose(target_pose)
                    status_text = "CONTROLLING"
                    status_color = (0, 200, 0)
                else:
                    status_text = "WAITING_POSE"
                    status_color = (0, 165, 255)
            else:
                if self.deadman_active:
                    self.get_logger().info("Deadman released; holding position")
                self.deadman_active = False
                self.initial_hand_pos = None
                if self.latest_robot_pose is not None:
                    # Hold position by re-publishing the latest pose
                    hold_pose = PoseStamped()
                    hold_pose.header = msg.header
                    hold_pose.header.frame_id = self.target_frame_id
                    hold_pose.pose = self.latest_robot_pose
                    self.target_pose_pub.publish(hold_pose)

            if deadman and self.latest_robot_pose is not None and wrist_pos is None:
                status_text = "WAITING_DEPTH" if self.hand_position_source != "normalized" else "WAITING_HAND"
                status_color = (0, 165, 255)

            # Visual aids
            thumb_px = (int(thumb_tip.x * width), int(thumb_tip.y * height))
            index_px = (int(index_tip.x * width), int(index_tip.y * height))
            cv2.line(cv_image, thumb_px, index_px, (255, 255, 0), 2)
            cv2.circle(cv_image, thumb_px, 4, (255, 0, 255), -1)
            cv2.circle(cv_image, index_px, 4, (0, 255, 255), -1)
        else:
            # No hand detected
            if self.deadman_active:
                self.get_logger().info("Hand lost; stopping motion")
            self.deadman_active = False
            self.initial_hand_pos = None

        cv2.putText(
            cv_image,
            f"{status_text} ({wrist_source})" if results.multi_hand_landmarks else status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            status_color,
            3,
        )
        cv2.imshow("Teleop", cv_image)

        # Keyboard handling (space deadman): latch for a short window to emulate "hold"
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # space
            hold_ns = int(max(0.0, self.space_deadman_hold_sec) * 1e9)
            self._space_deadman_until_ns = self.get_clock().now().nanoseconds + hold_ns

    def _get_hand_position(self, wrist_landmark, width: int, height: int) -> Tuple[Optional[np.ndarray], str]:
        """Return hand position vector for relative control.

        - depth: metric 3D (meters) in camera coordinates via depth+intrinsics
        - normalized: mediapipe normalized (x,y in [0,1], z relative)
        - hybrid: prefer depth, fallback to normalized
        """

        source = self.hand_position_source
        if source not in {"depth", "normalized", "hybrid"}:
            source = "hybrid"

        if source in {"depth", "hybrid"}:
            p3d = self._landmark_3d_m(wrist_landmark, width, height)
            if p3d is not None:
                return p3d.astype(np.float32), "DEPTH"
            if source == "depth":
                return None, "DEPTH_NONE"

        return np.array([wrist_landmark.x, wrist_landmark.y, wrist_landmark.z], dtype=np.float32), "NORM"

    def _is_pinch(self, thumb_tip, index_tip, width: int, height: int) -> bool:
        dist = self._distance_px(thumb_tip, index_tip, width, height)
        return dist < self.pinch_threshold_px

    def _gripper_from_distance(self, thumb_tip, index_tip, width: int, height: int) -> float:
        dist_m = self._physical_distance_m(thumb_tip, index_tip, width, height)
        if dist_m is not None:
            u = clamp(
                (dist_m - self.gripper_close_dist_m)
                / (self.gripper_open_dist_m - self.gripper_close_dist_m),
                0.0,
                1.0,
            )
        else:
            dist_px = self._distance_px(thumb_tip, index_tip, width, height)
            u = clamp(
                (dist_px - self.gripper_close_dist_px)
                / (self.gripper_open_dist_px - self.gripper_close_dist_px),
                0.0,
                1.0,
            )
        # u=0 -> closed, u=1 -> open; invert for command (1 closed)
        return 1.0 - u

    def _distance_px(self, p1, p2, width: int, height: int) -> float:
        x1, y1 = p1.x * width, p1.y * height
        x2, y2 = p2.x * width, p2.y * height
        return math.hypot(x2 - x1, y2 - y1)

    def _physical_distance_m(self, p1, p2, width: int, height: int) -> Optional[float]:
        if self.depth_image is None or self.depth_info is None or self.fx is None:
            return None
        uv1 = self._landmark_px(p1, width, height)
        uv2 = self._landmark_px(p2, width, height)
        d1 = self._get_depth_m(*uv1)
        d2 = self._get_depth_m(*uv2)
        if d1 is None or d2 is None:
            return None
        p3d_1 = self._deproject(uv1, d1)
        p3d_2 = self._deproject(uv2, d2)
        return float(np.linalg.norm(p3d_2 - p3d_1))

    def _landmark_px(self, landmark, width: int, height: int) -> Tuple[int, int]:
        u = int(landmark.x * width)
        v = int(landmark.y * height)
        return u, v

    def _get_depth_m(self, u: int, v: int) -> Optional[float]:
        # Use a small window median to reduce noise; ignore zeros/NaNs
        if self.depth_image is None:
            return None
        h, w = self.depth_image.shape[:2]
        if u < 0 or v < 0 or u >= w or v >= h:
            return None
        window = self.depth_image[max(v - 2, 0) : min(v + 3, h), max(u - 2, 0) : min(u + 3, w)]
        values = window.flatten()
        if values.dtype == np.uint16:
            vals = values.astype(np.float32) * self.depth_unit_scale
        else:
            vals = values.astype(np.float32)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size == 0:
            return None
        depth = float(np.median(vals))
        if self.depth_min_m > 0.0 and depth < self.depth_min_m:
            return None
        if self.depth_max_m > 0.0 and depth > self.depth_max_m:
            return None
        return depth

    def _landmark_3d_m(self, landmark, width: int, height: int) -> Optional[np.ndarray]:
        if self.depth_image is None or self.depth_info is None or self.fx is None:
            return None
        uv = self._landmark_px(landmark, width, height)
        depth_m = self._get_depth_m(*uv)
        if depth_m is None:
            return None
        return self._deproject(uv, depth_m)

    def _deproject(self, uv: Tuple[int, int], depth_m: float) -> np.ndarray:
        u, v = uv
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return np.array([x, y, z], dtype=np.float32)

    def _compute_target_pose(self, current_hand_pos: np.ndarray) -> Optional[PoseStamped]:
        if self.initial_hand_pos is None or self.initial_robot_pos is None or self.initial_robot_orientation is None:
            return None

        delta_hand = current_hand_pos - self.initial_hand_pos
        mapped_delta = np.array(
            [delta_hand[self.axis_mapping[0]], delta_hand[self.axis_mapping[1]], delta_hand[self.axis_mapping[2]]],
            dtype=np.float32,
        )
        target_pos = self.initial_robot_pos + mapped_delta * self.scale_factor

        # Filter target position
        filtered_pos = self.pos_filter.apply(target_pos)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.target_frame_id
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = float(filtered_pos[0])
        pose_msg.pose.position.y = float(filtered_pos[1])
        pose_msg.pose.position.z = float(filtered_pos[2])
        pose_msg.pose.orientation.x = float(self.initial_robot_orientation[0])
        pose_msg.pose.orientation.y = float(self.initial_robot_orientation[1])
        pose_msg.pose.orientation.z = float(self.initial_robot_orientation[2])
        pose_msg.pose.orientation.w = float(self.initial_robot_orientation[3])
        return pose_msg

    def _position_to_array(self, position) -> np.ndarray:
        return np.array([position.x, position.y, position.z], dtype=np.float32)

    def _orientation_to_array(self, orientation) -> np.ndarray:
        return np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            dtype=np.float32,
        )

    def _publish_target_pose(self, pose: PoseStamped) -> None:
        self.target_pose_pub.publish(pose)

    def _publish_gripper(self, cmd: float) -> None:
        cmd_clamped = float(clamp(cmd, 0.0, 1.0))
        step = self.gripper_step if self.gripper_step > 0 else 0.2
        quantized = round(cmd_clamped / step) * step
        quantized = float(clamp(quantized, 0.0, 1.0))
        msg = Float32()
        msg.data = quantized
        self.gripper_pub.publish(msg)
        self._call_gripper_service(quantized)

    def _depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_image = depth
            self.depth_encoding = msg.encoding
            self.depth_stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        except Exception as exc:  # pragma: no cover - defensive
            self.get_logger().warn(f"Depth conversion failed: {exc}")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.depth_info = msg
        if len(msg.k) == 9:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def _call_gripper_service(self, closure: float) -> None:
        closure = float(clamp(closure, 0.0, 1.0))
        now_ns = self.get_clock().now().nanoseconds
        if (
            self._last_gripper_cmd is not None
            and abs(closure - self._last_gripper_cmd) < self._gripper_delta_threshold
            and now_ns - self._last_gripper_call_ns < self._gripper_min_interval_ns
        ):
            return
        if not self.gripper_client.service_is_ready():
            if not self.gripper_client.wait_for_service(timeout_sec=0.0):
                return
        request = SetClosure.Request()
        request.closure = closure
        request.duration_sec = self.gripper_duration_sec
        request.speed_ratio = self.gripper_speed_ratio
        try:
            self.gripper_client.call_async(request)
            self._last_gripper_cmd = closure
            self._last_gripper_call_ns = now_ns
        except Exception as exc:  # pragma: no cover - best effort
            self.get_logger().warn(f"SoftHand call failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
