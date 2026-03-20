#!/usr/bin/env python3
"""机械臂输出层：专门负责下发 Twist 指令。"""

from __future__ import annotations

import time

from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import Twist, TwistStamped
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class ServoPoseFollower:
    """UR5 运动控制接口，不关心输入源和夹爪。"""

    def __init__(self, node: Node) -> None:
        self._node = node
        self._topic = str(node.get_parameter("servo_twist_topic").value)
        self._frame_id = str(node.get_parameter("target_frame_id").value)
        self._publisher = node.create_publisher(TwistStamped, self._topic, 10)

        self._auto_start_servo = bool(node.get_parameter("auto_start_servo").value)
        self._auto_switch_controllers = bool(node.get_parameter("auto_switch_controllers").value)
        self._start_servo_service = str(node.get_parameter("start_servo_service").value)
        controller_manager_ns = str(node.get_parameter("controller_manager_ns").value).rstrip("/")
        self._teleop_controller = str(node.get_parameter("teleop_controller").value)
        self._trajectory_controller = str(node.get_parameter("trajectory_controller").value)
        self._startup_retry_period_sec = max(0.2, float(node.get_parameter("startup_retry_period_sec").value))

        self._servo_started = not self._auto_start_servo
        self._controller_switched = not self._auto_switch_controllers
        self._startup_inflight = False
        self._last_wait_log_monotonic = 0.0

        self._start_servo_client = node.create_client(Trigger, self._start_servo_service)
        self._list_ctrl_client = node.create_client(
            ListControllers,
            f"{controller_manager_ns}/list_controllers",
        )
        self._switch_ctrl_client = node.create_client(
            SwitchController,
            f"{controller_manager_ns}/switch_controller",
        )

        self._startup_timer = None
        if self._auto_start_servo or self._auto_switch_controllers:
            self._startup_timer = node.create_timer(self._startup_retry_period_sec, self._startup_tick)

    def _startup_tick(self) -> None:
        if self._startup_inflight:
            return

        if not self._controller_switched:
            if not self._list_ctrl_client.wait_for_service(timeout_sec=0.0):
                return

            self._startup_inflight = True
            future = self._list_ctrl_client.call_async(ListControllers.Request())
            future.add_done_callback(self._on_list_controllers_done)
            return

        if not self._servo_started:
            if not self._start_servo_client.wait_for_service(timeout_sec=0.0):
                return

            self._startup_inflight = True
            future = self._start_servo_client.call_async(Trigger.Request())
            future.add_done_callback(self._on_start_servo_done)
            return

        if self._startup_timer is not None:
            self._startup_timer.cancel()
            self._startup_timer = None

    def _log_wait_once(self, message: str, period_sec: float = 2.0) -> None:
        now = time.monotonic()
        if (now - self._last_wait_log_monotonic) < period_sec:
            return
        self._last_wait_log_monotonic = now
        self._node.get_logger().info(message)

    def _on_list_controllers_done(self, future) -> None:
        self._startup_inflight = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warn(f"读取控制器状态失败: {exc}")
            return

        states = {controller.name: controller.state for controller in response.controller}
        teleop_state = states.get(self._teleop_controller)
        trajectory_state = states.get(self._trajectory_controller)

        if teleop_state == "active" and trajectory_state != "active":
            if not self._controller_switched:
                self._node.get_logger().info(
                    f"Teleop controller ready: activate={self._teleop_controller}, "
                    f"deactivate={self._trajectory_controller}"
                )
            self._controller_switched = True
            return

        if teleop_state is None:
            self._log_wait_once(f"Waiting for controller '{self._teleop_controller}' to be loaded...")
            return

        if not self._switch_ctrl_client.wait_for_service(timeout_sec=0.0):
            self._log_wait_once("Waiting for /controller_manager/switch_controller service...")
            return

        request = SwitchController.Request()
        request.activate_controllers = [self._teleop_controller]
        request.deactivate_controllers = [self._trajectory_controller] if trajectory_state == "active" else []
        request.strictness = SwitchController.Request.BEST_EFFORT

        self._startup_inflight = True
        switch_future = self._switch_ctrl_client.call_async(request)
        switch_future.add_done_callback(self._on_switch_done)

    def _on_switch_done(self, future) -> None:
        self._startup_inflight = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warn(f"切换 Teleop 控制器失败: {exc}")
            return

        if response.ok:
            self._node.get_logger().info("Teleop controller switch request accepted, verifying state...")
        else:
            self._node.get_logger().warn("控制器切换请求被拒绝，稍后重试。")

    def _on_start_servo_done(self, future) -> None:
        self._startup_inflight = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warn(f"启动 Servo 失败: {exc}")
            return

        if response.success:
            self._servo_started = True
            self._node.get_logger().info("MoveIt Servo started.")
        else:
            self._node.get_logger().warn(f"MoveIt Servo start returned false: {response.message}")

    def send_twist(self, twist_msg: Twist) -> None:
        stamped = TwistStamped()
        stamped.header.stamp = self._node.get_clock().now().to_msg()
        stamped.header.frame_id = self._frame_id
        stamped.twist = twist_msg
        self._publisher.publish(stamped)

    def stop(self) -> None:
        if self._startup_timer is not None:
            try:
                self._startup_timer.cancel()
            except Exception:
                pass
        self.send_twist(Twist())


class ServoPoseFollowerNode(Node):
    """独立调试节点：订阅 Twist 并转发到 Servo 输出。"""

    def __init__(self) -> None:
        super().__init__("servo_pose_follower")
        self.declare_parameter("servo_twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("target_frame_id", "base")
        self.declare_parameter("input_twist_topic", "~/input_twist")
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("auto_switch_controllers", True)
        self.declare_parameter("controller_manager_ns", "/controller_manager")
        self.declare_parameter("teleop_controller", "forward_position_controller")
        self.declare_parameter("trajectory_controller", "scaled_joint_trajectory_controller")
        self.declare_parameter("startup_retry_period_sec", 1.0)
        self._follower = ServoPoseFollower(self)
        self.create_subscription(Twist, str(self.get_parameter("input_twist_topic").value), self._cb, 10)

    def _cb(self, msg: Twist) -> None:
        self._follower.send_twist(msg)

    def destroy_node(self) -> bool:  # type: ignore[override]
        self._follower.stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ServoPoseFollowerNode()
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
