#!/usr/bin/env python3
"""夹爪策略层：不同末端执行器统一暴露 set_gripper(value) 接口。"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Optional

from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray

from .transform_utils import _clamp

try:
    from robotiq_2f_gripper_msgs.action import MoveTwoFingerGripper
except Exception:  # pragma: no cover - 运行环境未安装时保持模块可导入
    MoveTwoFingerGripper = None

try:
    from qbsofthand_control.srv import SetClosure
except Exception:  # pragma: no cover - 运行环境未安装时保持模块可导入
    SetClosure = None


class GripperControllerBase(ABC):
    """夹爪控制器统一接口。"""

    def __init__(self, node: Node) -> None:
        self.node = node
        self._lock = threading.Lock()
        self._last_value: Optional[float] = None
        self._suppressed_value: Optional[float] = None
        self._quantization_levels = max(0, int(node.get_parameter("gripper_quantization_levels").value))
        self._state_topic = str(node.get_parameter("gripper_cmd_topic").value)
        self._state_pub = node.create_publisher(Float32, self._state_topic, 10)

    @abstractmethod
    def set_gripper(self, value: float) -> None:
        """设置夹爪开合度，0.0 为全开，1.0 为全闭。"""

    def stop(self) -> None:
        """可选释放资源。"""

    def cancel_motion(self) -> None:
        """可选取消当前夹爪动作。"""

    def _publish_state(self, value: float) -> None:
        msg = Float32()
        msg.data = float(_clamp(value, 0.0, 1.0))
        self._state_pub.publish(msg)

    def _normalize_gripper_value(self, value: float) -> float:
        closure = float(_clamp(value, 0.0, 1.0))
        if self._quantization_levels <= 1:
            return closure

        step_count = self._quantization_levels - 1
        if step_count <= 0:
            return closure
        return float(round(closure * step_count) / step_count)


class RobotiqController(GripperControllerBase):
    """Robotiq 控制器，默认通过 action 发送目标开口宽度。"""

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self._command_interface = str(node.get_parameter("robotiq_command_interface").value).strip().lower()
        self._confidence_topic = str(node.get_parameter("robotiq_confidence_topic").value)
        self._binary_topic = str(node.get_parameter("robotiq_binary_topic").value)
        self._action_name = str(node.get_parameter("robotiq_action_name").value)
        self._binary_threshold = float(node.get_parameter("robotiq_binary_threshold").value)
        self._open_ratio = float(_clamp(float(node.get_parameter("robotiq_open_ratio").value), 0.0, 1.0))
        self._max_open_position_m = max(0.0, float(node.get_parameter("robotiq_max_open_position_m").value))
        self._target_speed = float(_clamp(float(node.get_parameter("robotiq_target_speed").value), 0.0, 1.0))
        self._target_force = float(_clamp(float(node.get_parameter("robotiq_target_force").value), 0.0, 1.0))
        self._min_delta = float(node.get_parameter("gripper_command_delta").value)
        self._confidence_pub = node.create_publisher(Float32MultiArray, self._confidence_topic, 10)
        self._binary_pub = node.create_publisher(Float32MultiArray, self._binary_topic, 10)
        self._action_client = None if MoveTwoFingerGripper is None else ActionClient(node, MoveTwoFingerGripper, self._action_name)
        self._goal_inflight = False
        self._pending_position_m: Optional[float] = None
        self._last_position_m: Optional[float] = None
        self._active_goal_handle = None
        self._cancel_requested = False
        self._warned_action_unavailable = False
        self._warned_action_not_ready = False

    def set_gripper(self, value: float) -> None:
        closure = self._normalize_gripper_value(value)
        with self._lock:
            if self._suppressed_value is not None:
                if abs(closure - self._suppressed_value) < self._min_delta:
                    return
                self._suppressed_value = None
            if self._last_value is not None and abs(closure - self._last_value) < self._min_delta:
                return
            self._last_value = closure

        self._publish_state(closure)

        if self._command_interface in {"position", "position_action", "action"}:
            target_position = self._max_open_position_m * self._open_ratio * (1.0 - closure)
            self._send_position_goal(target_position)
            return

        if self._command_interface in {"binary", "binary_topic"}:
            msg = Float32MultiArray()
            msg.data = [-1.0 if closure >= self._binary_threshold else 1.0]
            self._binary_pub.publish(msg)
            return

        # confidence_command: positive=open, negative=close, 0=neutral
        msg = Float32MultiArray()
        msg.data = [float(1.0 - 2.0 * closure)]
        self._confidence_pub.publish(msg)

    def _send_position_goal(self, target_position: float) -> None:
        action_client = self._action_client
        if action_client is None or MoveTwoFingerGripper is None:
            if not self._warned_action_unavailable:
                self._warned_action_unavailable = True
                self.node.get_logger().warn("Robotiq position_action 不可用，缺少 action 接口依赖。")
            self._allow_retry()
            return

        with self._lock:
            if self._goal_inflight:
                self._pending_position_m = target_position
                return
            self._goal_inflight = True
            self._last_position_m = target_position
            self._pending_position_m = None
            self._cancel_requested = False

        if not action_client.wait_for_server(timeout_sec=0.0):
            with self._lock:
                self._goal_inflight = False
                self._active_goal_handle = None
            if not self._warned_action_not_ready:
                self._warned_action_not_ready = True
                self.node.get_logger().warn(
                    f"Robotiq action server 未就绪: {self._action_name}，当前夹爪命令被跳过。"
                )
            self._allow_retry()
            return

        self._warned_action_not_ready = False
        goal = MoveTwoFingerGripper.Goal()
        goal.target_position = float(target_position)
        goal.target_speed = self._target_speed
        goal.target_force = self._target_force
        send_future = action_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_position_goal_response)

    def _on_position_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"Robotiq action goal 发送失败: {exc!r}")
            self._allow_retry()
            self._complete_position_goal()
            return

        if not goal_handle.accepted:
            self.node.get_logger().warn("Robotiq action goal 被拒绝。")
            self._allow_retry()
            self._complete_position_goal()
            return

        cancel_requested = False
        with self._lock:
            self._active_goal_handle = goal_handle
            cancel_requested = self._cancel_requested

        if cancel_requested:
            self._cancel_goal_handle(goal_handle)

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_position_goal_result)

    def _on_position_goal_result(self, future) -> None:
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"Robotiq action 执行失败: {exc!r}")
            self._allow_retry()
        self._complete_position_goal()

    def _complete_position_goal(self) -> None:
        pending: Optional[float]
        last_position: Optional[float]
        with self._lock:
            self._goal_inflight = False
            pending = self._pending_position_m
            self._pending_position_m = None
            last_position = self._last_position_m
            self._active_goal_handle = None
            self._cancel_requested = False

        if pending is None:
            return
        if last_position is not None and abs(pending - last_position) < 1e-6:
            return
        self._send_position_goal(pending)

    def _cancel_goal_handle(self, goal_handle) -> None:
        try:
            cancel_future = goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self._on_cancel_response)
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"Robotiq action cancel 发送失败: {exc!r}")

    def _on_cancel_response(self, future) -> None:
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"Robotiq action cancel 失败: {exc!r}")

    def _allow_retry(self) -> None:
        with self._lock:
            self._last_value = None

    def cancel_motion(self) -> None:
        if self._command_interface not in {"position", "position_action", "action"}:
            return

        goal_handle = None
        with self._lock:
            self._pending_position_m = None
            self._suppressed_value = self._last_value
            self._last_value = None
            self._cancel_requested = True
            goal_handle = self._active_goal_handle

        if goal_handle is not None:
            self._cancel_goal_handle(goal_handle)


class QbSoftHandController(GripperControllerBase):
    """qbSoftHand 控制器，优先调用服务，失败时退化到话题。"""

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self._service_name = str(node.get_parameter("qbsofthand_service_name").value)
        self._topic_name = str(node.get_parameter("gripper_cmd_topic").value)
        self._duration_sec = float(node.get_parameter("qbsofthand_duration_sec").value)
        self._speed_ratio = float(node.get_parameter("qbsofthand_speed_ratio").value)
        self._min_delta = float(node.get_parameter("gripper_command_delta").value)
        self._topic_pub = node.create_publisher(Float32, self._topic_name, 10)
        self._service_client = None if SetClosure is None else node.create_client(SetClosure, self._service_name)

    def set_gripper(self, value: float) -> None:
        closure = self._normalize_gripper_value(value)
        with self._lock:
            if self._suppressed_value is not None:
                if abs(closure - self._suppressed_value) < self._min_delta:
                    return
                self._suppressed_value = None
            if self._last_value is not None and abs(closure - self._last_value) < self._min_delta:
                return
            self._last_value = closure

        self._publish_state(closure)

        if self._service_client is not None and self._service_client.service_is_ready():
            request = SetClosure.Request()
            request.closure = closure
            request.duration_sec = self._duration_sec
            request.speed_ratio = self._speed_ratio
            self._service_client.call_async(request)
            return

        topic_msg = Float32()
        topic_msg.data = closure
        self._topic_pub.publish(topic_msg)
