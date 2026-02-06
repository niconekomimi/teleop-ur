#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.duration import Duration
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration as DurationMsg
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformException, TransformListener
from transforms3d.quaternions import (
    qinverse as quaternion_conjugate,
    qmult as quaternion_multiply,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def quat_to_vec_angle(q) -> tuple[float, np.ndarray]:
    # q assumed normalized (x, y, z, w). Returns (angle, axis)
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if norm == 0.0:
        return 0.0, np.array([0.0, 0.0, 0.0], dtype=np.float32)
    qx, qy, qz, qw = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)
    angle = 2.0 * math.acos(clamp(qw, -1.0, 1.0))
    s = math.sqrt(1.0 - qw * qw)
    if s < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = np.array([qx / s, qy / s, qz / s], dtype=np.float32)
    return angle, axis


class ServoPoseFollower(Node):
    """Convert Pose targets to MoveIt Servo Twist commands."""

    def __init__(self) -> None:
        super().__init__("servo_pose_follower")

        # Parameters
        self.declare_parameter("target_pose_topic", "/target_pose")
        self.declare_parameter("servo_twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("planning_frame", "base")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("linear_gain", 1.5)
        self.declare_parameter("angular_gain", 1.5)
        self.declare_parameter("max_linear_speed", 0.4)  # m/s
        self.declare_parameter("max_angular_speed", 1.2)  # rad/s
        self.declare_parameter("position_deadband", 0.002)  # 2mm
        self.declare_parameter("rotation_deadband", 0.01)   # rad
        self.declare_parameter("command_rate_hz", 100.0)

        # Startup helpers (keep everything inside this node)
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("auto_switch_controllers", True)
        self.declare_parameter("controller_manager_ns", "/controller_manager")
        self.declare_parameter("activate_controller", "forward_position_controller")
        self.declare_parameter("deactivate_controllers", ["scaled_joint_trajectory_controller"])
        self.declare_parameter("switch_strictness", "strict")  # strict|best_effort
        self.declare_parameter("startup_retry_period_sec", 1.0)

        # Fallback: if controller switching fails, bridge Servo joint position output to scaled JTC
        self.declare_parameter("fallback_bridge_to_scaled_jtc", True)
        self.declare_parameter("fallback_enable_after_failures", 3)
        self.declare_parameter("servo_joint_command_topic", "/forward_position_controller/commands")
        self.declare_parameter(
            "scaled_jtc_command_topic", "/scaled_joint_trajectory_controller/joint_trajectory"
        )
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
        self.declare_parameter("fallback_point_time_sec", 0.10)

        self.target_pose_topic = (
            self.get_parameter("target_pose_topic").get_parameter_value().string_value
        )
        self.servo_twist_topic = (
            self.get_parameter("servo_twist_topic").get_parameter_value().string_value
        )
        self.planning_frame = (
            self.get_parameter("planning_frame").get_parameter_value().string_value
        )
        self.ee_frame = self.get_parameter("ee_frame").get_parameter_value().string_value
        self.linear_gain = float(self.get_parameter("linear_gain").get_parameter_value().double_value)
        self.angular_gain = float(self.get_parameter("angular_gain").get_parameter_value().double_value)
        self.max_linear_speed = float(
            self.get_parameter("max_linear_speed").get_parameter_value().double_value
        )
        self.max_angular_speed = float(
            self.get_parameter("max_angular_speed").get_parameter_value().double_value
        )
        self.position_deadband = float(
            self.get_parameter("position_deadband").get_parameter_value().double_value
        )
        self.rotation_deadband = float(
            self.get_parameter("rotation_deadband").get_parameter_value().double_value
        )
        self.command_period = 1.0 / float(
            self.get_parameter("command_rate_hz").get_parameter_value().double_value
        )

        self.auto_start_servo = bool(
            self.get_parameter("auto_start_servo").get_parameter_value().bool_value
        )
        self.start_servo_service = (
            self.get_parameter("start_servo_service").get_parameter_value().string_value
        )
        self.auto_switch_controllers = bool(
            self.get_parameter("auto_switch_controllers").get_parameter_value().bool_value
        )
        self.controller_manager_ns = (
            self.get_parameter("controller_manager_ns").get_parameter_value().string_value
        ).rstrip("/")
        self.activate_controller = (
            self.get_parameter("activate_controller").get_parameter_value().string_value
        )
        self.deactivate_controllers = list(
            self.get_parameter("deactivate_controllers").get_parameter_value().string_array_value
        )
        self.switch_strictness = (
            self.get_parameter("switch_strictness").get_parameter_value().string_value.strip().lower()
        )
        self.startup_retry_period_sec = float(
            self.get_parameter("startup_retry_period_sec").get_parameter_value().double_value
        )

        self.fallback_bridge_to_scaled_jtc = bool(
            self.get_parameter("fallback_bridge_to_scaled_jtc").get_parameter_value().bool_value
        )
        self.fallback_enable_after_failures = int(
            self.get_parameter("fallback_enable_after_failures").get_parameter_value().integer_value
        )
        self.servo_joint_command_topic = (
            self.get_parameter("servo_joint_command_topic").get_parameter_value().string_value
        )
        self.scaled_jtc_command_topic = (
            self.get_parameter("scaled_jtc_command_topic").get_parameter_value().string_value
        )
        self.joint_names = list(
            self.get_parameter("joint_names").get_parameter_value().string_array_value
        )
        self.fallback_point_time_sec = float(
            self.get_parameter("fallback_point_time_sec").get_parameter_value().double_value
        )

        # Interfaces
        self.cmd_pub = self.create_publisher(TwistStamped, self.servo_twist_topic, 10)
        self.target_sub = self.create_subscription(
            PoseStamped, self.target_pose_topic, self._target_callback, 10
        )

        # TF buffer/listener
        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # State
        self.latest_target: Optional[PoseStamped] = None

        self._servo_started = False
        self._controller_switched = False
        self._startup_inflight = False
        self._switch_failures = 0
        self._bridge_active = False

        # Service clients (best-effort; will retry)
        self._start_servo_client = self.create_client(Trigger, self.start_servo_service)
        self._switch_ctrl_client = self.create_client(
            SwitchController, f"{self.controller_manager_ns}/switch_controller"
        )
        self._list_ctrl_client = self.create_client(
            ListControllers, f"{self.controller_manager_ns}/list_controllers"
        )

        # Fallback bridge interfaces (created but only used if enabled)
        self._scaled_traj_pub = self.create_publisher(JointTrajectory, self.scaled_jtc_command_topic, 10)
        self._servo_cmd_sub = self.create_subscription(
            Float64MultiArray, self.servo_joint_command_topic, self._servo_joint_cmd_cb, 10
        )

        retry_period = max(0.2, self.startup_retry_period_sec)
        self._startup_timer = self.create_timer(retry_period, self._startup_tick)

        self.create_timer(self.command_period, self._publish_twist)
        self.get_logger().info(
            f"ServoPoseFollower using target '{self.target_pose_topic}' -> twist '{self.servo_twist_topic}'"
        )

    def _startup_tick(self) -> None:
        if self._startup_inflight:
            return

        tasks_done = True
        if self.auto_start_servo and not self._servo_started:
            tasks_done = False
            self._try_start_servo()
            return

        if self.auto_switch_controllers and not self._controller_switched:
            tasks_done = False
            self._try_switch_controllers()
            return

        if tasks_done:
            try:
                self._startup_timer.cancel()
            except Exception:
                pass

    def _try_start_servo(self) -> None:
        if not self._start_servo_client.service_is_ready():
            self._start_servo_client.wait_for_service(timeout_sec=0.0)
            return
        self._startup_inflight = True
        future = self._start_servo_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_start_servo_done)

    def _on_start_servo_done(self, future) -> None:
        self._startup_inflight = False
        try:
            resp = future.result()
        except Exception as exc:  # pragma: no cover
            self.get_logger().warning(f"start_servo call failed: {exc}")
            return
        if getattr(resp, "success", False):
            self._servo_started = True
            self.get_logger().info("MoveIt Servo started")
        else:
            msg = getattr(resp, "message", "")
            self.get_logger().warning(f"MoveIt Servo start failed: {msg}")

    def _try_switch_controllers(self) -> None:
        if not self._switch_ctrl_client.service_is_ready():
            self._switch_ctrl_client.wait_for_service(timeout_sec=0.0)
            return

        self._startup_inflight = True

        def _after_list(list_future):
            self._startup_inflight = False
            try:
                list_resp = list_future.result()
            except Exception as exc:  # pragma: no cover
                self.get_logger().warning(f"list_controllers failed: {exc}")
                return
            active_to_deactivate = set(self.deactivate_controllers)
            # If any other trajectory controller is active, deactivate it too (avoid resource conflict)
            for c in getattr(list_resp, "controller", []):
                name = getattr(c, "name", "")
                state = getattr(c, "state", "")
                if state == "active" and name in {"joint_trajectory_controller", "scaled_joint_trajectory_controller"}:
                    active_to_deactivate.add(name)

            self._call_switch_controller(sorted(active_to_deactivate))

        if self._list_ctrl_client.service_is_ready():
            list_future = self._list_ctrl_client.call_async(ListControllers.Request())
            list_future.add_done_callback(_after_list)
        else:
            # No list service; switch with configured list only
            self._startup_inflight = False
            self._call_switch_controller(self.deactivate_controllers)

    def _call_switch_controller(self, deactivate_list: list[str]) -> None:
        req = SwitchController.Request()
        req.activate_controllers = [self.activate_controller]
        req.deactivate_controllers = list(deactivate_list)
        strict = self.switch_strictness != "best_effort"
        req.strictness = (
            SwitchController.Request.STRICT if strict else SwitchController.Request.BEST_EFFORT
        )
        req.activate_asap = True
        req.timeout = DurationMsg(sec=2, nanosec=0)

        self._startup_inflight = True
        future = self._switch_ctrl_client.call_async(req)
        future.add_done_callback(self._on_switch_done)

    def _on_switch_done(self, future) -> None:
        self._startup_inflight = False
        try:
            resp = future.result()
        except Exception as exc:  # pragma: no cover
            self.get_logger().warning(f"switch_controller call failed: {exc}")
            return
        if getattr(resp, "ok", False):
            self._controller_switched = True
            self.get_logger().info(
                f"Controllers switched: activated '{self.activate_controller}'"
            )
            return

        self._switch_failures += 1

        # If strict failed, fallback to best-effort automatically
        if self.switch_strictness != "best_effort":
            self.get_logger().warning(
                "Controller switch failed (strict). Will retry with best_effort"
            )
            self.switch_strictness = "best_effort"
        else:
            self.get_logger().warning("Controller switch failed (best_effort). Check controller_manager logs")

        if (
            self.fallback_bridge_to_scaled_jtc
            and not self._bridge_active
            and self._switch_failures >= max(0, self.fallback_enable_after_failures)
        ):
            self._bridge_active = True
            self.get_logger().warning(
                "Enabling fallback bridge: /forward_position_controller/commands -> /scaled_joint_trajectory_controller/joint_trajectory"
            )

    def _servo_joint_cmd_cb(self, msg: Float64MultiArray) -> None:
        # Only bridge if controller switch didn't succeed
        if not self._bridge_active:
            return
        if not msg.data:
            return
        if len(self.joint_names) != len(msg.data):
            # Mismatch usually means wrong joint_names (prefix) or wrong robot type
            self.get_logger().warning(
                f"Fallback bridge joint count mismatch: joint_names={len(self.joint_names)} cmd={len(msg.data)}"
            )
            return

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(self.joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = list(msg.data)
        dt = max(0.02, float(self.fallback_point_time_sec))
        sec = int(dt)
        nanosec = int((dt - sec) * 1e9)
        pt.time_from_start = DurationMsg(sec=sec, nanosec=nanosec)
        traj.points = [pt]
        self._scaled_traj_pub.publish(traj)

    def _target_callback(self, msg: PoseStamped) -> None:
        self.latest_target = msg

    def _lookup_ee_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.planning_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as exc:
            self.get_logger().warning(f"TF lookup failed: {exc}")
            return None
        pose = PoseStamped()
        pose.header.frame_id = tf.header.frame_id
        pose.header.stamp = tf.header.stamp
        pose.pose.position.x = tf.transform.translation.x
        pose.pose.position.y = tf.transform.translation.y
        pose.pose.position.z = tf.transform.translation.z
        pose.pose.orientation = tf.transform.rotation
        return pose

    def _publish_twist(self) -> None:
        if self.latest_target is None:
            return
        current = self._lookup_ee_pose()
        if current is None:
            return

        # Transform target into planning_frame if needed (e.g. base -> base_link)
        target = self._transform_target_to_planning(self.latest_target)
        if target is None:
            return

        lin_err = np.array(
            [
                target.pose.position.x - current.pose.position.x,
                target.pose.position.y - current.pose.position.y,
                target.pose.position.z - current.pose.position.z,
            ],
            dtype=np.float32,
        )
        pos_norm = float(np.linalg.norm(lin_err))

        q_curr = np.array(
            [
                current.pose.orientation.x,
                current.pose.orientation.y,
                current.pose.orientation.z,
                current.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        q_tgt = np.array(
            [
                target.pose.orientation.x,
                target.pose.orientation.y,
                target.pose.orientation.z,
                target.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        q_err = quaternion_multiply(q_tgt, quaternion_conjugate(q_curr))
        ang_err, ang_axis = quat_to_vec_angle(q_err)
        ang_vec = ang_axis * ang_err

        # Apply deadbands
        if pos_norm < self.position_deadband:
            lin_err[:] = 0.0
        if abs(ang_err) < self.rotation_deadband:
            ang_vec[:] = 0.0

        # PD-like velocity commands
        lin_cmd = lin_err * self.linear_gain
        ang_cmd = ang_vec * self.angular_gain

        # Clip speeds
        lin_speed = float(np.linalg.norm(lin_cmd))
        if lin_speed > self.max_linear_speed > 0:
            lin_cmd *= self.max_linear_speed / lin_speed
        ang_speed = float(np.linalg.norm(ang_cmd))
        if ang_speed > self.max_angular_speed > 0:
            ang_cmd *= self.max_angular_speed / ang_speed

        msg = TwistStamped()
        msg.header.frame_id = self.planning_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(lin_cmd[0])
        msg.twist.linear.y = float(lin_cmd[1])
        msg.twist.linear.z = float(lin_cmd[2])
        msg.twist.angular.x = float(ang_cmd[0])
        msg.twist.angular.y = float(ang_cmd[1])
        msg.twist.angular.z = float(ang_cmd[2])
        self.cmd_pub.publish(msg)

    def _transform_target_to_planning(self, target: PoseStamped) -> Optional[PoseStamped]:
        if target.header.frame_id in ("", self.planning_frame):
            return target
        try:
            tf = self.tf_buffer.lookup_transform(
                self.planning_frame,
                target.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as exc:
            self.get_logger().warning(f"TF transform target failed: {exc}")
            return None
        out = PoseStamped()
        out.header.frame_id = self.planning_frame
        out.header.stamp = target.header.stamp

        # Transform position
        t = tf.transform.translation
        q = tf.transform.rotation
        q_tf = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        # target point in source frame
        p_src = np.array([target.pose.position.x, target.pose.position.y, target.pose.position.z], dtype=np.float64)
        # rotate + translate
        p_rot = _quat_apply(q_tf, p_src)
        p_dst = p_rot + np.array([t.x, t.y, t.z], dtype=np.float64)
        out.pose.position.x, out.pose.position.y, out.pose.position.z = p_dst.tolist()

        # Transform orientation: q_dst = q_tf * q_target
        q_target = np.array(
            [
                target.pose.orientation.x,
                target.pose.orientation.y,
                target.pose.orientation.z,
                target.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        q_dst = quaternion_multiply(q_tf, q_target)
        out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z, out.pose.orientation.w = q_dst.tolist()
        return out


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Rotate vector v by quaternion q (x,y,z,w)
    q_vec = q[:3]
    qw = q[3]
    t = 2.0 * np.cross(q_vec, v)
    return v + qw * t + np.cross(q_vec, t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ServoPoseFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
