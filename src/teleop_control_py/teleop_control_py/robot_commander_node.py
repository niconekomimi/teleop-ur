#!/usr/bin/env python3
"""机器人核心能力节点：负责 Home / Home Zone / 控制器切换。"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .home_zone_utils import compose_pose_with_rpy_offset, drive_pose_target, pose_to_string, sample_home_zone_pose_offsets
from .ur_kinematics import try_forward_kinematics


class RobotCommanderNode(Node):
    """承接机器人控制核心能力，脱离数据采集职责。"""

    def __init__(self) -> None:
        super().__init__("commander")

        self.declare_parameter("ur_type", "ur5")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("tool_pose_topic", "/tcp_pose_broadcaster/pose")
        self.declare_parameter("servo_twist_topic", "/servo_node/delta_twist_cmds")
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
        self.declare_parameter("pose_stamp_zero_is_ref", True)
        self.declare_parameter("pose_use_received_time_fallback", True)
        self.declare_parameter(
            "home_joint_positions",
            [1.524178, -2.100060, 1.864580, -1.345048, -1.575888, 1.528195],
        )
        self.declare_parameter("home_zone_translation_min_m", [0.04, 0.04, 0.04])
        self.declare_parameter("home_zone_translation_max_m", [0.08, 0.08, 0.08])
        self.declare_parameter("home_zone_rotation_min_deg", [5.0, 5.0, 5.0])
        self.declare_parameter("home_zone_rotation_max_deg", [10.0, 10.0, 10.0])
        self.declare_parameter("home_zone_timeout_sec", 30.0)
        self.declare_parameter("home_zone_rate_hz", 60.0)
        self.declare_parameter("home_zone_max_linear_vel", 0.50)
        self.declare_parameter("home_zone_max_angular_vel", 1.5)
        self.declare_parameter("home_zone_linear_gain", 2.4)
        self.declare_parameter("home_zone_angular_gain", 2.4)
        self.declare_parameter("home_zone_position_tolerance_m", 0.005)
        self.declare_parameter("home_zone_rotation_tolerance_deg", 6.0)
        self.declare_parameter("home_duration_sec", 3.0)
        self.declare_parameter(
            "home_joint_trajectory_topic",
            "/scaled_joint_trajectory_controller/joint_trajectory",
        )
        self.declare_parameter("teleop_controller", "forward_position_controller")
        self.declare_parameter("trajectory_controller", "scaled_joint_trajectory_controller")

        self._ur_type = str(self.get_parameter("ur_type").value).strip().lower() or "ur5"
        self._joint_names = [str(name) for name in list(self.get_parameter("joint_names").value)[:6]]
        self._pose_max_age = float(self.get_parameter("pose_max_age_sec").value)
        self._pose_stamp_zero_is_ref = bool(self.get_parameter("pose_stamp_zero_is_ref").value)
        self._pose_use_received_time_fallback = bool(self.get_parameter("pose_use_received_time_fallback").value)

        self._cache_lock = threading.Lock()
        self._home_zone_lock = threading.Lock()

        self._latest_joint_pos: Optional[np.ndarray] = None
        self._latest_pose_pos: Optional[np.ndarray] = None
        self._latest_pose_quat: Optional[np.ndarray] = None
        self._latest_pose_time: Optional[Time] = None
        self._latest_pose_receive_time: Optional[Time] = None

        self._homing_in_progress = False
        self._home_zone_in_progress = False
        self._home_zone_token = 0

        joint_topic = str(self.get_parameter("joint_states_topic").value)
        pose_topic = str(self.get_parameter("tool_pose_topic").value)
        twist_topic = str(self.get_parameter("servo_twist_topic").value)
        home_topic = str(self.get_parameter("home_joint_trajectory_topic").value)

        self.create_subscription(JointState, joint_topic, self._on_joint_state, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, pose_topic, self._on_tool_pose, qos_profile_sensor_data)

        self._home_pub = self.create_publisher(JointTrajectory, home_topic, 10)
        self._home_zone_twist_pub = self.create_publisher(TwistStamped, twist_topic, 10)
        self._home_zone_active_pub = self.create_publisher(Bool, "~/home_zone_active", 10)

        self._list_ctrl_client = self.create_client(ListControllers, "/controller_manager/list_controllers")
        self._switch_ctrl_client = self.create_client(SwitchController, "/controller_manager/switch_controller")
        self._start_servo_client = self.create_client(Trigger, "/servo_node/start_servo")

        self._srv_go_home = self.create_service(Trigger, "~/go_home", self._srv_go_home_cb)
        self._srv_go_home_zone = self.create_service(Trigger, "~/go_home_zone", self._srv_go_home_zone_cb)
        self._srv_cancel_home_zone = self.create_service(Trigger, "~/cancel_home_zone", self._srv_cancel_home_zone_cb)

        self._publish_home_zone_active(False)
        self.get_logger().info(
            "RobotCommanderNode ready. Services: ~/go_home, ~/go_home_zone, ~/cancel_home_zone. "
            f"Joint topic={joint_topic}, Pose topic={pose_topic}, Home topic={home_topic}"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        joint_pos = self._map_joint_positions(msg)
        if joint_pos is None:
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

    def _map_joint_positions(self, msg: JointState) -> Optional[np.ndarray]:
        if not msg.name or not msg.position:
            return None

        name_to_idx = {name: index for index, name in enumerate(msg.name)}
        out = np.zeros(len(self._joint_names), dtype=np.float32)
        for index, joint_name in enumerate(self._joint_names):
            msg_index = name_to_idx.get(joint_name)
            if msg_index is None or msg_index >= len(msg.position):
                return None
            out[index] = float(msg.position[msg_index])
        return out

    def _srv_go_home_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if self._homing_in_progress or self._home_zone_in_progress:
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

    def _next_home_zone_token(self) -> int:
        with self._home_zone_lock:
            self._home_zone_token += 1
            return self._home_zone_token

    def _current_home_zone_token(self) -> int:
        with self._home_zone_lock:
            return self._home_zone_token

    def _cancel_home_zone(self, reason: str) -> None:
        token = self._next_home_zone_token()
        self._publish_home_zone_active(False)
        for _ in range(5):
            self._publish_home_zone_zero_twist()
            time.sleep(0.01)
        threading.Thread(target=self._restore_home_zone_teleop_control, daemon=True).start()
        self.get_logger().info(f"Home Zone cancel requested ({reason}), token={token}")

    def _home_zone_aborted(self, token: int) -> bool:
        return int(token) != int(self._current_home_zone_token())

    def _srv_go_home_zone_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if self._homing_in_progress or self._home_zone_in_progress:
            res.success = False
            res.message = "Home Zone sequence is already in progress!"
            return res

        home_positions = list(self.get_parameter("home_joint_positions").value)
        if len(home_positions) < 6:
            res.success = False
            res.message = "home_joint_positions must have 6 elements"
            return res

        ranges = [
            list(self.get_parameter("home_zone_translation_min_m").value),
            list(self.get_parameter("home_zone_translation_max_m").value),
            list(self.get_parameter("home_zone_rotation_min_deg").value),
            list(self.get_parameter("home_zone_rotation_max_deg").value),
        ]
        if min(len(values) for values in ranges) < 3:
            res.success = False
            res.message = "home_zone translation/rotation ranges must have 3 elements"
            return res

        token = self._next_home_zone_token()
        self._home_zone_in_progress = True
        self._publish_home_zone_active(True)
        threading.Thread(
            target=self._execute_go_home_zone_sequence,
            args=(token, [float(value) for value in home_positions[:6]]),
            daemon=True,
        ).start()
        res.success = True
        res.message = "Home Zone sequence started"
        self.get_logger().info(res.message)
        return res

    def _srv_cancel_home_zone_cb(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if not self._home_zone_in_progress:
            res.success = True
            res.message = "No Home Zone motion is running"
            return res
        self._cancel_home_zone("service request")
        res.success = True
        res.message = "Home Zone cancellation requested"
        return res

    def _list_controller_states(self) -> Optional[dict[str, str]]:
        if not self._list_ctrl_client.wait_for_service(timeout_sec=0.5):
            return None
        try:
            future = self._list_ctrl_client.call_async(ListControllers.Request())
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not future.done():
                time.sleep(0.02)
            if not future.done():
                return None
            response = future.result()
            return {controller.name: controller.state for controller in response.controller}
        except Exception:
            return None

    def _publish_home_zone_active(self, active: bool) -> None:
        try:
            msg = Bool()
            msg.data = bool(active)
            self._home_zone_active_pub.publish(msg)
        except Exception:
            pass

    def _restore_home_zone_teleop_control(self) -> None:
        teleop_ctrl = str(self.get_parameter("teleop_controller").value)
        traj_ctrl = str(self.get_parameter("trajectory_controller").value)
        if not self._switch_ctrl_client.wait_for_service(timeout_sec=0.5):
            return
        try:
            request = SwitchController.Request()
            request.activate_controllers = [teleop_ctrl]
            request.deactivate_controllers = [traj_ctrl]
            request.strictness = SwitchController.Request.BEST_EFFORT
            future = self._switch_ctrl_client.call_async(request)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not future.done():
                time.sleep(0.02)
            if future.done():
                response = future.result()
                if response is not None and bool(response.ok):
                    self.get_logger().info("Home Zone cancel restored teleop controller.")
            self._wait_for_home_zone_teleop_ready(timeout_sec=1.0)
        except Exception:
            pass

    def _wait_for_home_zone_teleop_ready(self, timeout_sec: float = 2.0) -> bool:
        teleop_ctrl = str(self.get_parameter("teleop_controller").value)
        traj_ctrl = str(self.get_parameter("trajectory_controller").value)
        deadline = time.monotonic() + max(0.2, float(timeout_sec))
        while time.monotonic() < deadline:
            states = self._list_controller_states()
            if states is None:
                time.sleep(0.05)
                continue
            if states.get(teleop_ctrl) == "active" and states.get(traj_ctrl) != "active":
                return True
            time.sleep(0.05)
        return False

    def _wait_for_home_zone_trajectory_ready(self, timeout_sec: float = 2.0) -> bool:
        teleop_ctrl = str(self.get_parameter("teleop_controller").value)
        traj_ctrl = str(self.get_parameter("trajectory_controller").value)
        deadline = time.monotonic() + max(0.2, float(timeout_sec))
        while time.monotonic() < deadline:
            states = self._list_controller_states()
            if states is None:
                time.sleep(0.05)
                continue
            if states.get(traj_ctrl) == "active" and states.get(teleop_ctrl) != "active":
                return True
            time.sleep(0.05)
        return False

    def _switch_home_zone_controllers(
        self,
        activate_controllers: list[str],
        deactivate_controllers: list[str],
        *,
        timeout_sec: float = 2.0,
    ) -> bool:
        states = self._list_controller_states()
        if states is not None:
            activate_needed = [name for name in activate_controllers if states.get(name) != "active"]
            deactivate_needed = [name for name in deactivate_controllers if states.get(name) == "active"]
            if not activate_needed and not deactivate_needed:
                return True
        else:
            activate_needed = list(activate_controllers)
            deactivate_needed = list(deactivate_controllers)

        if not self._switch_ctrl_client.wait_for_service(timeout_sec=0.5):
            return False
        try:
            request = SwitchController.Request()
            request.activate_controllers = activate_needed
            request.deactivate_controllers = deactivate_needed
            request.strictness = SwitchController.Request.BEST_EFFORT
            future = self._switch_ctrl_client.call_async(request)
            deadline = time.monotonic() + max(0.5, float(timeout_sec))
            while time.monotonic() < deadline and not future.done():
                time.sleep(0.02)
            if not future.done():
                return False
            response = future.result()
            if response is not None and not bool(response.ok):
                return False
        except Exception:
            return False

        if activate_controllers == [str(self.get_parameter("teleop_controller").value)]:
            return self._wait_for_home_zone_teleop_ready(timeout_sec=timeout_sec)
        if activate_controllers == [str(self.get_parameter("trajectory_controller").value)]:
            return self._wait_for_home_zone_trajectory_ready(timeout_sec=timeout_sec)
        return True

    def _execute_go_home_for_home_zone_sequence(self, token: int, home_positions: list[float]) -> None:
        home_positions = [float(value) for value in home_positions[:6]]
        duration = float(self.get_parameter("home_duration_sec").value)
        if duration <= 0.0:
            duration = 3.0

        teleop_ctrl = str(self.get_parameter("teleop_controller").value)
        traj_ctrl = str(self.get_parameter("trajectory_controller").value)
        if not self._switch_home_zone_controllers([traj_ctrl], [teleop_ctrl], timeout_sec=2.0):
            raise RuntimeError(f"Home Zone 无法切换到 {traj_ctrl}")

        trajectory = JointTrajectory()
        trajectory.joint_names = [str(name) for name in self._joint_names[:6]]

        point = JointTrajectoryPoint()
        point.positions = home_positions
        seconds = int(duration)
        nanoseconds = int((duration - seconds) * 1e9)
        point.time_from_start = DurationMsg(sec=seconds, nanosec=nanoseconds)
        trajectory.points = [point]

        self._home_pub.publish(trajectory)
        self.get_logger().info(f"Published Home-for-Home-Zone trajectory, waiting {duration:.2f}s...")

        deadline = time.monotonic() + duration + 0.5
        while time.monotonic() < deadline:
            if self._home_zone_aborted(token):
                return
            time.sleep(0.05)

        if self._home_zone_aborted(token):
            return
        if not self._switch_home_zone_controllers([teleop_ctrl], [traj_ctrl], timeout_sec=2.0):
            raise RuntimeError(f"Home Zone 无法恢复 {teleop_ctrl}")
        self.get_logger().info("Home-for-Home-Zone sequence complete. Teleop restored.")

    def _ensure_home_zone_servo_started(self) -> None:
        if not self._start_servo_client.wait_for_service(timeout_sec=0.5):
            return
        try:
            future = self._start_servo_client.call_async(Trigger.Request())
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not future.done():
                time.sleep(0.02)
            if future.done():
                response = future.result()
                if response is not None:
                    self.get_logger().info(
                        f"Home Zone Servo start returned success={bool(response.success)} message={response.message}"
                    )
        except Exception:
            pass

    def _get_fk_pose_snapshot(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._cache_lock:
            joints = None if self._latest_joint_pos is None else self._latest_joint_pos.copy()
        if joints is None or len(joints) < 6:
            return None
        return try_forward_kinematics(self._ur_type, joints[:6])

    def _get_pose_topic_snapshot(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._cache_lock:
            pos = None if self._latest_pose_pos is None else self._latest_pose_pos.copy()
            quat = None if self._latest_pose_quat is None else self._latest_pose_quat.copy()
        if pos is None or quat is None or len(pos) < 3 or len(quat) < 4:
            return None
        return pos[:3].astype(np.float64), quat[:4].astype(np.float64)

    def _get_fresh_pose_topic_snapshot(self, max_age_sec: Optional[float] = None) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._cache_lock:
            pos = None if self._latest_pose_pos is None else self._latest_pose_pos.copy()
            quat = None if self._latest_pose_quat is None else self._latest_pose_quat.copy()
            pose_time = self._latest_pose_time
            pose_receive_time = self._latest_pose_receive_time
        if pos is None or quat is None or pose_time is None or len(pos) < 3 or len(quat) < 4:
            return None
        max_age = float(self._pose_max_age if max_age_sec is None else max_age_sec)
        if max_age > 0.0:
            now = self.get_clock().now()
            receive_age = None
            if pose_receive_time is not None:
                receive_age = abs((now - pose_receive_time).nanoseconds) * 1e-9
                if receive_age <= max_age:
                    return pos[:3].astype(np.float64), quat[:4].astype(np.float64)
            pose_ref = now if self._pose_stamp_zero_is_ref and pose_time.nanoseconds == 0 else pose_time
            pose_age = abs((now - pose_ref).nanoseconds) * 1e-9
            if pose_age > max_age:
                if (
                    self._pose_use_received_time_fallback
                    and receive_age is not None
                    and receive_age <= max_age
                ):
                    return pos[:3].astype(np.float64), quat[:4].astype(np.float64)
                return None
        return pos[:3].astype(np.float64), quat[:4].astype(np.float64)

    def _get_home_zone_pose_snapshot(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        pose = self._get_fresh_pose_topic_snapshot(max_age_sec=max(0.3, float(self._pose_max_age)))
        if pose is not None:
            return pose
        pose = self._get_fk_pose_snapshot()
        if pose is not None:
            return pose
        return self._get_pose_topic_snapshot()

    def _wait_for_home_zone_pose(self, timeout_sec: float = 2.0) -> Optional[tuple[np.ndarray, np.ndarray]]:
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        while time.monotonic() < deadline:
            pose = self._get_home_zone_pose_snapshot()
            if pose is not None:
                return pose
            time.sleep(0.02)
        return None

    def _publish_home_zone_twist(self, linear_xyz, angular_xyz) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.twist.linear.x = float(linear_xyz[0])
        msg.twist.linear.y = float(linear_xyz[1])
        msg.twist.linear.z = float(linear_xyz[2])
        msg.twist.angular.x = float(angular_xyz[0])
        msg.twist.angular.y = float(angular_xyz[1])
        msg.twist.angular.z = float(angular_xyz[2])
        self._home_zone_twist_pub.publish(msg)

    def _publish_home_zone_zero_twist(self) -> None:
        self._publish_home_zone_twist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def _execute_go_home_zone_sequence(self, token: int, home_positions: list[float]) -> None:
        try:
            if self._home_zone_aborted(token):
                return

            self._homing_in_progress = True
            self._execute_go_home_for_home_zone_sequence(token, home_positions)
            self._homing_in_progress = False
            if self._home_zone_aborted(token):
                return

            time.sleep(0.2)
            self._wait_for_home_zone_teleop_ready(timeout_sec=2.0)
            time.sleep(0.1)
            current_pose = self._wait_for_home_zone_pose(timeout_sec=2.0)
            if current_pose is None:
                raise RuntimeError("未拿到当前末端位姿，无法执行 Home Zone")
            current_pos, current_quat = current_pose

            translation_offset, rotation_offset = sample_home_zone_pose_offsets(
                list(self.get_parameter("home_zone_translation_min_m").value)[:3],
                list(self.get_parameter("home_zone_translation_max_m").value)[:3],
                list(self.get_parameter("home_zone_rotation_min_deg").value)[:3],
                list(self.get_parameter("home_zone_rotation_max_deg").value)[:3],
            )
            target_pos, target_quat = compose_pose_with_rpy_offset(
                current_pos,
                current_quat,
                translation_offset,
                rotation_offset,
            )
            self.get_logger().info(
                "Home Zone target sampled from current pose: "
                f"offset_xyz={np.array2string(translation_offset, precision=4)} "
                f"offset_rpy_deg={np.array2string(np.rad2deg(rotation_offset), precision=2)} "
                f"{pose_to_string(target_pos, target_quat)}"
            )

            self._ensure_home_zone_servo_started()
            max_linear_vel = max(1e-3, float(self.get_parameter("home_zone_max_linear_vel").value))
            max_angular_vel = max(1e-3, float(self.get_parameter("home_zone_max_angular_vel").value))
            linear_gain = max(0.1, float(self.get_parameter("home_zone_linear_gain").value))
            angular_gain = max(0.1, float(self.get_parameter("home_zone_angular_gain").value))
            position_tolerance_m = max(1e-3, float(self.get_parameter("home_zone_position_tolerance_m").value))
            rotation_tolerance_deg = max(0.1, float(self.get_parameter("home_zone_rotation_tolerance_deg").value))
            configured_timeout = float(self.get_parameter("home_zone_timeout_sec").value)
            dynamic_timeout = max(
                configured_timeout,
                2.0 + 1.5 * float(np.linalg.norm(translation_offset)) / max_linear_vel,
                1.5 + 1.5 * float(np.linalg.norm(rotation_offset)) / max_angular_vel,
            )
            success, pos_err_m, rot_err_deg = drive_pose_target(
                self._get_home_zone_pose_snapshot,
                self._publish_home_zone_twist,
                self._publish_home_zone_zero_twist,
                target_pos,
                target_quat,
                linear_gain=linear_gain,
                angular_gain=angular_gain,
                max_linear_vel=max_linear_vel,
                max_angular_vel=max_angular_vel,
                position_tolerance_m=position_tolerance_m,
                rotation_tolerance_deg=rotation_tolerance_deg,
                timeout_sec=dynamic_timeout,
                rate_hz=float(self.get_parameter("home_zone_rate_hz").value),
                settle_sec=0.15,
                should_abort_fn=lambda: self._home_zone_aborted(token),
            )
            level = self.get_logger().info if success else self.get_logger().warn
            level(
                f"Home Zone {'reached' if success else 'stopped'} "
                f"residual_pos={pos_err_m:.4f}m residual_rot={rot_err_deg:.2f}deg"
            )
        except Exception as exc:
            self.get_logger().error(f"Home Zone sequence failed: {exc}")
        finally:
            self._publish_home_zone_active(False)
            for _ in range(5):
                self._publish_home_zone_zero_twist()
                time.sleep(0.02)
            self._homing_in_progress = False
            self._home_zone_in_progress = False

    def _execute_go_home_sequence(self, home_positions: list[float]) -> None:
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

    def destroy_node(self) -> bool:
        try:
            self._cancel_home_zone("node shutdown")
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotCommanderNode()
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
