import re
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from geometry_msgs.msg import TwistStamped
from PySide6.QtCore import QThread, Signal
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from std_srvs.srv import Trigger

from ..core import ActionCommand, ControlCoordinator, ControlSource
from ..device_manager import (
    ControllerGripperBackend,
    DEFAULT_ROBOT_PROFILE_NAME,
    ServoArmBackend,
    default_robot_profiles_path,
    load_robot_profile,
)
from ..hardware.control.gripper_controllers import QbSoftHandController, RobotiqController


class ROS2Worker(QThread):
    robot_state_str_signal = Signal(str)
    record_stats_signal = Signal(int, str, float)
    log_signal = Signal(str)
    demo_status_signal = Signal(str)
    recording_state_signal = Signal(bool)
    home_zone_active_signal = Signal(bool)
    homing_active_signal = Signal(bool)
    commander_result_signal = Signal(str)

    def __init__(self, pose_topic=None, gripper_topic=None, action_topic=None, robot_profile=DEFAULT_ROBOT_PROFILE_NAME, robot_profiles_file=None):
        super().__init__()
        self.pose_topic = str(pose_topic).strip()
        self.gripper_topic = str(gripper_topic).strip()
        self.action_topic = str(action_topic).strip()
        self.joint_states_topic = ""
        self._robot_profile_name = str(robot_profile).strip() or DEFAULT_ROBOT_PROFILE_NAME
        self._robot_profiles_file = str(robot_profiles_file or default_robot_profiles_path())
        self._robot_profile = load_robot_profile(self._robot_profile_name, self._robot_profiles_file)
        self.node = None
        self._is_running = True

        self._home_joint_positions = []
        self._home_duration_sec = 3.0
        self._home_joint_names = []
        self._home_joint_trajectory_topic = ""
        self._teleop_controller = ""
        self._trajectory_controller = ""
        self._home_zone_translation_min_m = []
        self._home_zone_translation_max_m = []
        self._home_zone_rotation_min_deg = []
        self._home_zone_rotation_max_deg = []
        self._inference_gripper_type = "robotiq"
        self._inference_control_hz = 50.0
        self._inference_control_enabled = False
        self._inference_estopped = False
        self._inference_action_lock = threading.Lock()
        self._latest_inference_action = None
        self._inference_control_timer = None
        self._arm_ctrl = None
        self._gripper_ctrl = None
        self._control_coordinator = None
        self._binary_gripper_state = None
        self.joint_sub = None
        self.pose_sub = None
        self.gripper_sub = None
        self.action_sub = None
        self.home_zone_active_sub = None
        self.homing_active_sub = None
        self.commander_result_sub = None
        self._binary_open_threshold = 0.35
        self._binary_close_threshold = 0.65
        self.start_cli = None
        self.stop_cli = None
        self.discard_cli = None
        self.home_cli = None
        self.home_zone_cli = None
        self.cancel_home_zone_cli = None
        self.set_param_cli = None
        self._pending_go_home = False
        self._pending_go_home_zone = False

        self.is_recording = False
        self.start_time = 0.0
        self.record_fps = 10.0
        self.actual_recorded_frames = 0
        self.realtime_record_fps = 0.0
        self.last_demo_name = "无 (未录制)"

        self.robot_state = {
            "joints": [0.0] * 6,
            "pose": [0.0, 0.0, 0.0],
            "quat": [0.0, 0.0, 0.0, 1.0],
            "gripper": 0.0,
            "action_linear": [0.0, 0.0, 0.0],
            "action_angular": [0.0, 0.0, 0.0],
        }

        self._apply_robot_profile_defaults(self._robot_profile)

    def _apply_robot_profile_defaults(self, robot_profile) -> None:
        self.joint_states_topic = str(robot_profile.topics.joint_states)
        self.pose_topic = str(robot_profile.topics.tool_pose)
        self.action_topic = str(robot_profile.topics.servo_twist)
        if not self.gripper_topic:
            self.gripper_topic = str(robot_profile.topics.gripper_state)
        self._home_joint_positions = [float(value) for value in robot_profile.home.joint_positions[:6]]
        self._home_duration_sec = float(robot_profile.home.duration_sec)
        self._home_joint_names = [str(name) for name in robot_profile.joint_names[:6]]
        self._home_joint_trajectory_topic = str(robot_profile.topics.home_joint_trajectory)
        self._teleop_controller = str(robot_profile.controllers.teleop)
        self._trajectory_controller = str(robot_profile.controllers.trajectory)
        self._home_zone_translation_min_m = [float(value) for value in robot_profile.home_zone.translation_min_m[:3]]
        self._home_zone_translation_max_m = [float(value) for value in robot_profile.home_zone.translation_max_m[:3]]
        self._home_zone_rotation_min_deg = [float(value) for value in robot_profile.home_zone.rotation_min_deg[:3]]
        self._home_zone_rotation_max_deg = [float(value) for value in robot_profile.home_zone.rotation_max_deg[:3]]

    def set_robot_profile(self, profile_name: str, robot_profiles_file: str | None = None) -> None:
        resolved_name = str(profile_name).strip() or DEFAULT_ROBOT_PROFILE_NAME
        resolved_file = str(robot_profiles_file or self._robot_profiles_file or default_robot_profiles_path())
        self._robot_profile_name = resolved_name
        self._robot_profiles_file = resolved_file
        self._robot_profile = load_robot_profile(resolved_name, resolved_file)
        self._apply_robot_profile_defaults(self._robot_profile)
        if self.node is not None:
            self._refresh_runtime_subscriptions()
            self._apply_inference_control_parameters()
            self._sync_commander_home_config(log_failures=True)

    def set_inference_control_config(self, gripper_type: str, control_hz: float = 50.0) -> None:
        normalized = str(gripper_type).strip().lower()
        if normalized not in {"robotiq", "qbsofthand"}:
            normalized = "robotiq"
        self._inference_gripper_type = normalized
        self._inference_control_hz = max(1.0, float(control_hz))

        if self.node is not None:
            self._apply_inference_control_parameters()
            if self._arm_ctrl is not None or self._gripper_ctrl is not None:
                self._setup_inference_control_backend()

    def inference_execution_enabled(self) -> bool:
        return bool(self._inference_control_enabled)

    def set_inference_execution_enabled(self, enabled: bool) -> None:
        self._inference_control_enabled = bool(enabled)
        if enabled:
            self._inference_estopped = False
            if self.node is not None and (self._arm_ctrl is None or self._gripper_ctrl is None or self._control_coordinator is None):
                self._apply_inference_control_parameters()
                self._setup_inference_control_backend()
            if self._control_coordinator is not None:
                self._control_coordinator.clear_estop()
                self._control_coordinator.notify_inference_ready(True)
                self._control_coordinator.notify_inference_execution(True)
            current_gripper = float(self.robot_state.get("gripper", 0.0))
            self._binary_gripper_state = 1.0 if current_gripper >= 0.5 else 0.0
            self.log_signal.emit("推理执行已使能。")
            return

        if self._control_coordinator is not None:
            self._control_coordinator.notify_inference_execution(False)
        self._publish_zero_twist()
        self.log_signal.emit("推理执行已停止。")

    def emergency_stop_inference(self) -> None:
        self._inference_estopped = True
        self._inference_control_enabled = False
        if self._control_coordinator is not None:
            self._control_coordinator.notify_estop(True)
        self._publish_zero_twist()
        self.log_signal.emit("已触发推理急停。")

    def update_inference_action_command(self, action) -> None:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size < 7:
            return
        with self._inference_action_lock:
            self._latest_inference_action = action_array[:7].copy()

    def set_home_config(
        self,
        joint_positions,
        duration_sec: float = 3.0,
        joint_names=None,
        trajectory_topic: str = "/scaled_joint_trajectory_controller/joint_trajectory",
        teleop_controller: str = "forward_position_controller",
        trajectory_controller: str = "scaled_joint_trajectory_controller",
        home_zone_translation_min_m=None,
        home_zone_translation_max_m=None,
        home_zone_rotation_min_deg=None,
        home_zone_rotation_max_deg=None,
    ) -> None:
        try:
            values = [float(value) for value in list(joint_positions)[:6]]
        except Exception:
            values = []

        if len(values) == 6:
            self._home_joint_positions = values
        self._home_duration_sec = max(0.1, float(duration_sec))
        if joint_names is not None:
            names = [str(name) for name in list(joint_names)[:6]]
            if len(names) == 6:
                self._home_joint_names = names
        self._home_joint_trajectory_topic = str(trajectory_topic).strip() or self._home_joint_trajectory_topic
        self._teleop_controller = str(teleop_controller).strip() or self._teleop_controller
        self._trajectory_controller = str(trajectory_controller).strip() or self._trajectory_controller
        self._home_zone_translation_min_m = self._normalize_vector_param(
            home_zone_translation_min_m,
            self._home_zone_translation_min_m,
        )
        self._home_zone_translation_max_m = self._normalize_vector_param(
            home_zone_translation_max_m,
            self._home_zone_translation_max_m,
        )
        self._home_zone_rotation_min_deg = self._normalize_vector_param(
            home_zone_rotation_min_deg,
            self._home_zone_rotation_min_deg,
        )
        self._home_zone_rotation_max_deg = self._normalize_vector_param(
            home_zone_rotation_max_deg,
            self._home_zone_rotation_max_deg,
        )

        self._sync_commander_home_config()

    def _normalize_vector_param(self, values, fallback):
        try:
            normalized = [float(value) for value in list(values)[:3]]
        except Exception:
            normalized = []
        return normalized if len(normalized) == 3 else list(fallback)

    def _build_commander_param_msgs(self):
        params = [
            Parameter("home_joint_positions", Parameter.Type.DOUBLE_ARRAY, list(self._home_joint_positions[:6])),
            Parameter("home_duration_sec", Parameter.Type.DOUBLE, float(self._home_duration_sec)),
            Parameter("joint_names", Parameter.Type.STRING_ARRAY, [str(name) for name in self._home_joint_names[:6]]),
            Parameter("home_joint_trajectory_topic", Parameter.Type.STRING, self._home_joint_trajectory_topic),
            Parameter("teleop_controller", Parameter.Type.STRING, self._teleop_controller),
            Parameter("trajectory_controller", Parameter.Type.STRING, self._trajectory_controller),
            Parameter(
                "home_zone_translation_min_m",
                Parameter.Type.DOUBLE_ARRAY,
                list(self._home_zone_translation_min_m[:3]),
            ),
            Parameter(
                "home_zone_translation_max_m",
                Parameter.Type.DOUBLE_ARRAY,
                list(self._home_zone_translation_max_m[:3]),
            ),
            Parameter(
                "home_zone_rotation_min_deg",
                Parameter.Type.DOUBLE_ARRAY,
                list(self._home_zone_rotation_min_deg[:3]),
            ),
            Parameter(
                "home_zone_rotation_max_deg",
                Parameter.Type.DOUBLE_ARRAY,
                list(self._home_zone_rotation_max_deg[:3]),
            ),
        ]
        return [param.to_parameter_msg() for param in params]

    def _sync_commander_home_config(self, *, log_failures: bool = False) -> bool:
        if self.set_param_cli is None:
            return False
        if not self.set_param_cli.wait_for_service(timeout_sec=0.2):
            if log_failures:
                self.log_signal.emit("未检测到 /commander/set_parameters，无法同步 Home 配置。")
            return False

        request = SetParameters.Request()
        request.parameters = self._build_commander_param_msgs()
        future = self.set_param_cli.call_async(request)
        if log_failures:
            future.add_done_callback(self._on_sync_commander_home_config_done)
        return True

    def _on_sync_commander_home_config_done(self, future) -> None:
        try:
            response = future.result()
            results = getattr(response, "results", None)
            if not results:
                return
            failures = [result.reason for result in results if not bool(getattr(result, "successful", False))]
            if failures:
                self.log_signal.emit(f"同步 commander Home 配置失败: {failures[0]}")
        except Exception as exc:
            self.log_signal.emit(f"同步 commander Home 配置异常: {exc}")

    def _refresh_runtime_subscriptions(self) -> None:
        if self.node is None:
            return

        subscription_specs = [
            ("joint_sub", JointState, self.joint_states_topic, self.joint_callback, 10),
            ("pose_sub", PoseStamped, self.pose_topic, self.pose_callback, 10),
            ("gripper_sub", Float32, self.gripper_topic, self.gripper_callback, 10),
            ("action_sub", TwistStamped, self.action_topic, self.action_callback, 10),
            ("home_zone_active_sub", Bool, "/commander/home_zone_active", self.home_zone_active_callback, 10),
            ("homing_active_sub", Bool, "/commander/homing_active", self.homing_active_callback, 10),
            ("commander_result_sub", String, "/commander/last_motion_result", self.commander_result_callback, 10),
        ]
        for attr_name, msg_type, topic_name, callback, qos in subscription_specs:
            existing = getattr(self, attr_name, None)
            if existing is not None:
                try:
                    self.node.destroy_subscription(existing)
                except Exception:
                    pass
            setattr(self, attr_name, self.node.create_subscription(msg_type, topic_name, callback, qos))

    def _set_or_declare_parameter(self, name: str, value) -> None:
        if self.node is None:
            return
        param = Parameter(name, value=value)
        if self.node.has_parameter(name):
            self.node.set_parameters([param])
            return
        self.node.declare_parameter(name, value)

    def _apply_inference_control_parameters(self) -> None:
        if self.node is None:
            return
        grippers = self._robot_profile.grippers
        robotiq = grippers.robotiq
        qbsofthand = grippers.qbsofthand
        self._set_or_declare_parameter("gripper_type", self._inference_gripper_type)
        self._set_or_declare_parameter("target_frame_id", self._robot_profile.target_frame_id)
        self._set_or_declare_parameter("servo_twist_topic", self.action_topic)
        self._set_or_declare_parameter("auto_start_servo", True)
        self._set_or_declare_parameter("start_servo_service", self._robot_profile.services.start_servo)
        self._set_or_declare_parameter("auto_switch_controllers", True)
        self._set_or_declare_parameter("controller_manager_ns", self._robot_profile.services.controller_manager_ns)
        self._set_or_declare_parameter("teleop_controller", self._teleop_controller)
        self._set_or_declare_parameter("trajectory_controller", self._trajectory_controller)
        self._set_or_declare_parameter("startup_retry_period_sec", 1.0)
        self._set_or_declare_parameter("gripper_cmd_topic", self.gripper_topic or grippers.command_topic)
        self._set_or_declare_parameter("gripper_command_delta", grippers.command_delta)
        self._set_or_declare_parameter("gripper_quantization_levels", grippers.quantization_levels)
        self._set_or_declare_parameter("robotiq_command_interface", robotiq.command_interface)
        self._set_or_declare_parameter("robotiq_confidence_topic", robotiq.confidence_topic)
        self._set_or_declare_parameter("robotiq_binary_topic", robotiq.binary_topic)
        self._set_or_declare_parameter("robotiq_action_name", robotiq.action_name)
        self._set_or_declare_parameter("robotiq_binary_threshold", robotiq.binary_threshold)
        self._set_or_declare_parameter("robotiq_open_ratio", robotiq.open_ratio)
        self._set_or_declare_parameter("robotiq_max_open_position_m", robotiq.max_open_position_m)
        self._set_or_declare_parameter("robotiq_target_speed", robotiq.target_speed)
        self._set_or_declare_parameter("robotiq_target_force", robotiq.target_force)
        self._set_or_declare_parameter("qbsofthand_service_name", qbsofthand.service_name)
        self._set_or_declare_parameter("qbsofthand_duration_sec", qbsofthand.duration_sec)
        self._set_or_declare_parameter("qbsofthand_speed_ratio", qbsofthand.speed_ratio)

    def _build_gripper_controller(self):
        controller_cls = RobotiqController if self._inference_gripper_type == "robotiq" else QbSoftHandController
        return controller_cls(self.node)

    def _setup_inference_control_backend(self) -> None:
        if self.node is None:
            return

        previous_arm = self._arm_ctrl
        previous_gripper = self._gripper_ctrl
        self._arm_ctrl = ServoArmBackend.from_node(self.node)
        self._gripper_ctrl = ControllerGripperBackend(self._build_gripper_controller())
        self._control_coordinator = ControlCoordinator(
            self._arm_ctrl,
            self._gripper_ctrl,
            active_source=ControlSource.NONE,
            logger=self.node.get_logger(),
        )
        self._control_coordinator.notify_inference_ready(True)
        if self._inference_estopped:
            self._control_coordinator.notify_estop(True)
        elif self._inference_control_enabled:
            self._control_coordinator.notify_inference_execution(True)

        for controller in (previous_arm, previous_gripper):
            if controller is None:
                continue
            try:
                controller.stop()
            except Exception:
                pass

    def _publish_zero_twist(self) -> None:
        if self._control_coordinator is not None:
            self._control_coordinator.publish_zero(source=ControlSource.SAFETY)
            return
        if self._arm_ctrl is not None:
            self._arm_ctrl.send_zero_twist()

    def _binary_gripper_command(self, value: float) -> float:
        closure = max(0.0, min(1.0, float(value)))
        if self._binary_gripper_state is None:
            current_gripper = float(self.robot_state.get("gripper", 0.0))
            self._binary_gripper_state = 1.0 if current_gripper >= 0.5 else 0.0

        if closure >= self._binary_close_threshold:
            self._binary_gripper_state = 1.0
        elif closure <= self._binary_open_threshold:
            self._binary_gripper_state = 0.0

        return float(self._binary_gripper_state)

    def _publish_inference_control_step(self) -> None:
        if not self._inference_control_enabled or self._inference_estopped:
            return
        if self._arm_ctrl is None or self._gripper_ctrl is None or self._control_coordinator is None:
            if self.node is not None:
                self._apply_inference_control_parameters()
                self._setup_inference_control_backend()
            if self._arm_ctrl is None or self._gripper_ctrl is None or self._control_coordinator is None:
                return

        if self._control_coordinator.snapshot().active_source != ControlSource.INFERENCE:
            return

        with self._inference_action_lock:
            action = None if self._latest_inference_action is None else self._latest_inference_action.copy()

        if action is None or action.size < 7:
            return

        command = ActionCommand.from_array7(action, source=ControlSource.INFERENCE)
        command = command.with_gripper(self._binary_gripper_command(command.gripper))
        dispatch_result = self._control_coordinator.dispatch(command)
        if not dispatch_result.accepted and self.node is not None:
            self.node.get_logger().debug(f"Inference action dropped by coordinator: {dispatch_result.reason}")

    def run(self):
        rclpy.init()
        self.node = Node("teleop_gui_node")
        self._apply_inference_control_parameters()
        self._inference_control_timer = self.node.create_timer(
            1.0 / max(self._inference_control_hz, 1.0),
            self._publish_inference_control_step,
        )

        self._refresh_runtime_subscriptions()
        self.record_stats_sub = self.node.create_subscription(
            Float32MultiArray,
            "/data_collector/record_stats",
            self.record_stats_callback,
            qos_profile_sensor_data,
        )

        self.start_cli = self.node.create_client(Trigger, "/data_collector/start")
        self.stop_cli = self.node.create_client(Trigger, "/data_collector/stop")
        self.discard_cli = self.node.create_client(Trigger, "/data_collector/discard_last_demo")
        self.home_cli = self.node.create_client(Trigger, "/commander/go_home")
        self.home_zone_cli = self.node.create_client(Trigger, "/commander/go_home_zone")
        self.cancel_home_zone_cli = self.node.create_client(Trigger, "/commander/cancel_home_zone")
        self.set_param_cli = self.node.create_client(SetParameters, "/commander/set_parameters")
        self._sync_commander_home_config()
        if self._pending_go_home:
            self._pending_go_home = False
            self.call_go_home()
        if self._pending_go_home_zone:
            self._pending_go_home_zone = False
            self.call_go_home_zone()

        self.stats_timer = self.node.create_timer(0.1, self.stats_timer_callback)

        while rclpy.ok() and self._is_running:
            rclpy.spin_once(self.node, timeout_sec=0.05)

        self.node.destroy_node()
        rclpy.shutdown()

    def joint_callback(self, msg):
        target_joints = list(self._home_joint_names[:6])
        if not target_joints:
            return
        if msg.name and msg.position:
            name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
            out = []
            for joint_name in target_joints:
                if joint_name in name_to_idx:
                    out.append(msg.position[name_to_idx[joint_name]])

            if len(out) == len(target_joints):
                self.robot_state["joints"] = out
                self._emit_robot_state()

    def pose_callback(self, msg):
        position = msg.pose.position
        quat = msg.pose.orientation
        self.robot_state["pose"] = [position.x, position.y, position.z]
        self.robot_state["quat"] = [quat.x, quat.y, quat.z, quat.w]
        self._emit_robot_state()

    def gripper_callback(self, msg):
        self.robot_state["gripper"] = max(0.0, min(1.0, float(msg.data)))
        self._emit_robot_state()

    def action_callback(self, msg):
        self.robot_state["action_linear"] = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        self.robot_state["action_angular"] = [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
        self._emit_robot_state()

    def home_zone_active_callback(self, msg):
        active = bool(msg.data)
        if self._control_coordinator is not None:
            self._control_coordinator.notify_home_zone(active)
            if active:
                self._publish_zero_twist()
        self.home_zone_active_signal.emit(active)

    def homing_active_callback(self, msg):
        active = bool(msg.data)
        if self._control_coordinator is not None:
            self._control_coordinator.notify_homing(active)
            if active:
                self._publish_zero_twist()
        self.homing_active_signal.emit(active)

    def commander_result_callback(self, msg):
        message = str(getattr(msg, "data", "")).strip()
        if message:
            self.commander_result_signal.emit(message)

    def _emit_robot_state(self):
        joints = np.array(self.robot_state.get("joints", [0.0] * 6))
        pos = np.array(self.robot_state.get("pose", [0.0, 0.0, 0.0]))
        quat = np.array(self.robot_state.get("quat", [0.0, 0.0, 0.0, 1.0]))
        gripper = self.robot_state.get("gripper", 0.0)
        action_linear = np.array(self.robot_state.get("action_linear", [0.0, 0.0, 0.0]))
        action_angular = np.array(self.robot_state.get("action_angular", [0.0, 0.0, 0.0]))

        action = np.array([
            action_linear[0],
            action_linear[1],
            action_linear[2],
            action_angular[0],
            action_angular[1],
            action_angular[2],
            gripper,
        ])

        formatter = {"float_kind": lambda value: f"{value:6.3f}"}
        joints_str = np.array2string(joints, formatter=formatter)
        pos_str = np.array2string(pos, formatter=formatter)
        quat_str = np.array2string(quat, formatter=formatter)
        gripper_str = np.array2string(np.array([gripper]), formatter=formatter)
        action_str = np.array2string(action, formatter=formatter)

        text = "【实时机器人状态 (与HDF5写入对齐)】\n"
        text += "-" * 25 + "\n"
        text += f"► 关节位置 [6]:\n {joints_str}\n\n"
        text += f"► 末端 XYZ [3]:\n {pos_str}\n\n"
        text += f"► 末端 四元数 [4]:\n {quat_str}\n\n"
        text += f"► 夹爪状态 [1]:\n {gripper_str}\n\n"
        text += "-" * 25 + "\n"
        text += f"► 实时 Action [7]:\n {action_str}\n"
        text += "  (VxVyVz, WxWyWz, Gripper)"
        self.robot_state_str_signal.emit(text)

    def stats_timer_callback(self):
        if self.is_recording:
            elapsed = max(0.0, time.time() - self.start_time)
            elapsed_sec = int(elapsed)
            mins = elapsed_sec // 60
            secs = elapsed_sec % 60
            time_str = f"{mins:02d}:{secs:02d}"
            self.record_stats_signal.emit(int(self.actual_recorded_frames), time_str, float(self.realtime_record_fps))

    def record_stats_callback(self, msg):
        try:
            data = list(getattr(msg, "data", []))
            if len(data) >= 1:
                self.actual_recorded_frames = max(0, int(round(float(data[0]))))
            if len(data) >= 2:
                self.realtime_record_fps = max(0.0, float(data[1]))
        except Exception:
            pass

    def call_start_record(self):
        if self.start_cli is None:
            self.log_signal.emit("ROS 监听器初始化中，请稍后再试。")
            return
        if self.start_cli.wait_for_service(timeout_sec=1.0):
            future = self.start_cli.call_async(Trigger.Request())
            future.add_done_callback(self.start_record_done)
        else:
            self.log_signal.emit("错误: 找不到 /data_collector/start 服务")

    def call_stop_record(self):
        if self.stop_cli is None:
            self.log_signal.emit("ROS 监听器初始化中，请稍后再试。")
            return
        if self.stop_cli.wait_for_service(timeout_sec=1.0):
            future = self.stop_cli.call_async(Trigger.Request())
            future.add_done_callback(self.stop_record_done)
        else:
            self.log_signal.emit("错误: 找不到 /data_collector/stop 服务")

    def call_discard_last_demo(self):
        if self.discard_cli is None:
            self.log_signal.emit("ROS 监听器初始化中，请稍后再试。")
            return
        if self.discard_cli.wait_for_service(timeout_sec=1.0):
            future = self.discard_cli.call_async(Trigger.Request())
            future.add_done_callback(self.discard_last_demo_done)
        else:
            self.log_signal.emit("错误: 找不到 /data_collector/discard_last_demo 服务")

    def call_go_home(self):
        if self._inference_control_enabled:
            self.set_inference_execution_enabled(False)
            self.log_signal.emit("回 Home 前已停止推理执行。")

        if self.home_cli is None:
            self._pending_go_home = True
            self.log_signal.emit("ROS 监听器初始化中，Home 请求将在就绪后自动发送。")
            return

        if self.home_cli is not None and self.home_cli.wait_for_service(timeout_sec=1.0):
            future = self.home_cli.call_async(Trigger.Request())
            future.add_done_callback(self.go_home_done)
            return

        self.log_signal.emit("错误: 找不到 /commander/go_home 服务，请先启动机械臂驱动或遥操作系统。")

    def call_go_home_zone(self):
        if self._inference_control_enabled:
            self.set_inference_execution_enabled(False)
            self.log_signal.emit("回 Home Zone 前已停止推理执行。")

        if self.home_zone_cli is None:
            self._pending_go_home_zone = True
            self.log_signal.emit("ROS 监听器初始化中，Home Zone 请求将在就绪后自动发送。")
            return

        if self.home_zone_cli is None or not self.home_zone_cli.wait_for_service(timeout_sec=1.0):
            self.log_signal.emit("错误: 找不到 /commander/go_home_zone 服务，请先启动机械臂驱动或遥操作系统。")
            return

        future = self.home_zone_cli.call_async(Trigger.Request())
        future.add_done_callback(self.go_home_zone_done)

    def call_cancel_home_zone(self) -> None:
        if self.cancel_home_zone_cli is None or not self.cancel_home_zone_cli.wait_for_service(timeout_sec=0.5):
            return
        try:
            future = self.cancel_home_zone_cli.call_async(Trigger.Request())
            future.add_done_callback(self.cancel_home_zone_done)
        except Exception:
            pass

    def call_set_home_from_current(self):
        joints = [float(value) for value in self.robot_state.get("joints", [])]
        if len(joints) != 6:
            self.log_signal.emit("错误: 当前关节状态无效，无法设置 Home 点")
            return

        self._home_joint_positions = list(joints)
        if self.set_param_cli is None or not self.set_param_cli.wait_for_service(timeout_sec=1.0):
            self.log_signal.emit("错误: 找不到 /commander 参数服务")
            return

        req = SetParameters.Request()
        req.parameters = self._build_commander_param_msgs()
        future = self.set_param_cli.call_async(req)
        future.add_done_callback(self.set_home_done)

    def start_record_done(self, future):
        try:
            response = future.result()
            self.log_signal.emit(f"录制服务: {response.message}")
            if response.success:
                self.is_recording = True
                self.start_time = time.time()
                self.actual_recorded_frames = 0
                self.realtime_record_fps = 0.0
                match = re.search(r"at\s+([0-9]+(?:\.[0-9]+)?)\s+Hz", str(response.message))
                if match:
                    try:
                        self.record_fps = max(0.1, float(match.group(1)))
                    except Exception:
                        self.record_fps = 10.0
                demo_match = re.search(r"Started recording\s+(\S+)\s+at\s+[0-9]+(?:\.[0-9]+)?\s+Hz", str(response.message))
                if demo_match:
                    demo_name = demo_match.group(1)
                else:
                    msg_parts = response.message.split()
                    demo_name = msg_parts[2] if len(msg_parts) >= 3 else "未知"
                self.last_demo_name = demo_name
                self.demo_status_signal.emit(demo_name)
                self.recording_state_signal.emit(True)
        except Exception as exc:
            self.log_signal.emit(f"服务调用异常: {exc}")

    def stop_record_done(self, future):
        try:
            response = future.result()
            self.log_signal.emit(f"停止录制服务: {response.message}")
            if response.success:
                self.is_recording = False
                self.realtime_record_fps = 0.0
                demo_match = re.search(r"Stopped recording\s+(\S+)", str(response.message))
                if demo_match:
                    self.last_demo_name = demo_match.group(1)
                self.demo_status_signal.emit(self.last_demo_name)
                self.recording_state_signal.emit(False)
        except Exception as exc:
            self.log_signal.emit(f"服务调用异常: {exc}")

    def discard_last_demo_done(self, future):
        try:
            response = future.result()
            self.log_signal.emit(f"弃用录制服务: {response.message}")
            if response.success:
                self.last_demo_name = "无 (未录制)"
                self.demo_status_signal.emit(self.last_demo_name)
                self.recording_state_signal.emit(False)
        except Exception as exc:
            self.log_signal.emit(f"服务调用异常: {exc}")

    def go_home_done(self, future):
        try:
            response = future.result()
            self.log_signal.emit(f"回Home服务: {response.message}")
        except Exception as exc:
            self.log_signal.emit(f"服务调用异常: {exc}")

    def go_home_zone_done(self, future):
        try:
            response = future.result()
            self.log_signal.emit(f"回Home Zone服务: {response.message}")
        except Exception as exc:
            self.log_signal.emit(f"服务调用异常: {exc}")

    def cancel_home_zone_done(self, future):
        try:
            response = future.result()
            if bool(getattr(response, "success", False)):
                self.log_signal.emit(f"取消Home Zone服务: {response.message}")
        except Exception:
            pass

    def set_home_done(self, future):
        try:
            response = future.result()
            results = getattr(response, "results", None)
            if results and len(results) > 0 and bool(results[0].successful):
                joints = [float(value) for value in self.robot_state.get("joints", [])]
                joints_str = np.array2string(np.array(joints), formatter={"float_kind": lambda value: f"{value:6.3f}"})
                self.log_signal.emit(f"已将当前关节姿态设置为 Home 点(本次运行生效): {joints_str}")
            else:
                reason = getattr(results[0], "reason", None) if results and len(results) > 0 else None
                self.log_signal.emit(f"设置 Home 点失败: {reason or '未知原因'}")
        except Exception as exc:
            self.log_signal.emit(f"设置 Home 点异常: {exc}")

    def stop(self):
        try:
            self.call_cancel_home_zone()
        except Exception:
            pass
        try:
            self.emergency_stop_inference()
        except Exception:
            pass
        self._is_running = False
        self.wait()
