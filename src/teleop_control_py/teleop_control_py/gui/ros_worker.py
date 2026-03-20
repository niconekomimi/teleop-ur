import math
import re
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, Twist
from geometry_msgs.msg import TwistStamped
from PySide6.QtCore import QThread, Signal
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float32MultiArray
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ..gripper_controllers import QbSoftHandController, RobotiqController
from ..servo_pose_follower import ServoPoseFollower


class ROS2Worker(QThread):
    robot_state_str_signal = Signal(str)
    record_stats_signal = Signal(int, str, float)
    log_signal = Signal(str)
    demo_status_signal = Signal(str)

    def __init__(self, pose_topic, gripper_topic, action_topic):
        super().__init__()
        self.pose_topic = str(pose_topic).strip()
        self.gripper_topic = str(gripper_topic).strip()
        self.action_topic = str(action_topic).strip()
        self.node = None
        self._is_running = True

        self._home_joint_positions = [1.524178, -2.100060, 1.864580, -1.345048, -1.575888, 1.528195]
        self._home_duration_sec = 3.0
        self._home_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._home_joint_trajectory_topic = "/scaled_joint_trajectory_controller/joint_trajectory"
        self._teleop_controller = "forward_position_controller"
        self._trajectory_controller = "scaled_joint_trajectory_controller"
        self._home_pub = None
        self._switch_ctrl_client = None
        self._homing_in_progress = False
        self._inference_gripper_type = "robotiq"
        self._inference_control_hz = 50.0
        self._inference_control_enabled = False
        self._inference_estopped = False
        self._inference_action_lock = threading.Lock()
        self._latest_inference_action = None
        self._inference_control_timer = None
        self._arm_ctrl = None
        self._gripper_ctrl = None
        self._binary_gripper_state = None
        self._binary_open_threshold = 0.35
        self._binary_close_threshold = 0.65
        self.start_cli = None
        self.stop_cli = None
        self.home_cli = None
        self.home_zone_cli = None
        self.cancel_home_zone_cli = None
        self.set_param_cli = None

        self.is_recording = False
        self.start_time = 0.0
        self.record_fps = 10.0
        self.actual_recorded_frames = 0
        self.realtime_record_fps = 0.0

        self.robot_state = {
            "joints": [0.0] * 6,
            "pose": [0.0, 0.0, 0.0],
            "quat": [0.0, 0.0, 0.0, 1.0],
            "gripper": 0.0,
            "action_linear": [0.0, 0.0, 0.0],
            "action_angular": [0.0, 0.0, 0.0],
        }

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
            if self.node is not None and (self._arm_ctrl is None or self._gripper_ctrl is None):
                self._apply_inference_control_parameters()
                self._setup_inference_control_backend()
            current_gripper = float(self.robot_state.get("gripper", 0.0))
            self._binary_gripper_state = 1.0 if current_gripper >= 0.5 else 0.0
            self.log_signal.emit("推理执行已使能。")
            return

        self._publish_zero_twist()
        self.log_signal.emit("推理执行已停止。")

    def emergency_stop_inference(self) -> None:
        self._inference_estopped = True
        self._inference_control_enabled = False
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

        if self.node is not None and self._home_pub is not None:
            try:
                self.node.destroy_publisher(self._home_pub)
            except Exception:
                pass
            self._home_pub = self.node.create_publisher(JointTrajectory, self._home_joint_trajectory_topic, 10)

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
        self._set_or_declare_parameter("gripper_type", self._inference_gripper_type)
        self._set_or_declare_parameter("target_frame_id", "base")
        self._set_or_declare_parameter("servo_twist_topic", self.action_topic)
        self._set_or_declare_parameter("auto_start_servo", True)
        self._set_or_declare_parameter("start_servo_service", "/servo_node/start_servo")
        self._set_or_declare_parameter("auto_switch_controllers", True)
        self._set_or_declare_parameter("controller_manager_ns", "/controller_manager")
        self._set_or_declare_parameter("teleop_controller", self._teleop_controller)
        self._set_or_declare_parameter("trajectory_controller", self._trajectory_controller)
        self._set_or_declare_parameter("startup_retry_period_sec", 1.0)
        self._set_or_declare_parameter("gripper_cmd_topic", self.gripper_topic)
        self._set_or_declare_parameter("gripper_command_delta", 0.01)
        self._set_or_declare_parameter("gripper_quantization_levels", 10)
        self._set_or_declare_parameter("robotiq_command_interface", "position_action")
        self._set_or_declare_parameter("robotiq_confidence_topic", "/robotiq_2f_gripper/confidence_command")
        self._set_or_declare_parameter("robotiq_binary_topic", "/robotiq_2f_gripper/binary_command")
        self._set_or_declare_parameter("robotiq_action_name", "/robotiq_2f_gripper_action")
        self._set_or_declare_parameter("robotiq_binary_threshold", 0.5)
        self._set_or_declare_parameter("robotiq_open_ratio", 0.9)
        self._set_or_declare_parameter("robotiq_max_open_position_m", 0.142)
        self._set_or_declare_parameter("robotiq_target_speed", 1.0)
        self._set_or_declare_parameter("robotiq_target_force", 0.5)
        self._set_or_declare_parameter("qbsofthand_service_name", "/qbsofthand_control_node/set_closure")
        self._set_or_declare_parameter("qbsofthand_duration_sec", 0.3)
        self._set_or_declare_parameter("qbsofthand_speed_ratio", 1.0)

    def _build_gripper_controller(self):
        controller_cls = RobotiqController if self._inference_gripper_type == "robotiq" else QbSoftHandController
        return controller_cls(self.node)

    def _setup_inference_control_backend(self) -> None:
        if self.node is None:
            return

        previous_arm = self._arm_ctrl
        previous_gripper = self._gripper_ctrl
        self._arm_ctrl = ServoPoseFollower(self.node)
        self._gripper_ctrl = self._build_gripper_controller()

        for controller in (previous_arm, previous_gripper):
            if controller is None:
                continue
            try:
                controller.stop()
            except Exception:
                pass

    def _publish_zero_twist(self) -> None:
        if self._arm_ctrl is None:
            return
        self._arm_ctrl.send_twist(Twist())

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
        if self._arm_ctrl is None or self._gripper_ctrl is None:
            if self.node is not None:
                self._apply_inference_control_parameters()
                self._setup_inference_control_backend()
            if self._arm_ctrl is None or self._gripper_ctrl is None:
                return

        with self._inference_action_lock:
            action = None if self._latest_inference_action is None else self._latest_inference_action.copy()

        if action is None or action.size < 7:
            return

        twist = Twist()
        twist.linear.x = float(action[0])
        twist.linear.y = float(action[1])
        twist.linear.z = float(action[2])
        twist.angular.x = float(action[3])
        twist.angular.y = float(action[4])
        twist.angular.z = float(action[5])
        self._arm_ctrl.send_twist(twist)
        self._gripper_ctrl.set_gripper(self._binary_gripper_command(float(action[6])))

    def run(self):
        rclpy.init()
        self.node = Node("teleop_gui_node")
        self._apply_inference_control_parameters()
        self._home_pub = self.node.create_publisher(JointTrajectory, self._home_joint_trajectory_topic, 10)
        self._switch_ctrl_client = self.node.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
        )
        self._inference_control_timer = self.node.create_timer(
            1.0 / max(self._inference_control_hz, 1.0),
            self._publish_inference_control_step,
        )

        self.joint_sub = self.node.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.pose_sub = self.node.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, 10)
        self.gripper_sub = self.node.create_subscription(Float32, self.gripper_topic, self.gripper_callback, 10)
        self.action_sub = self.node.create_subscription(TwistStamped, self.action_topic, self.action_callback, 10)
        self.record_stats_sub = self.node.create_subscription(
            Float32MultiArray,
            "/data_collector/record_stats",
            self.record_stats_callback,
            qos_profile_sensor_data,
        )

        self.start_cli = self.node.create_client(Trigger, "/data_collector/start")
        self.stop_cli = self.node.create_client(Trigger, "/data_collector/stop")
        self.home_cli = self.node.create_client(Trigger, "/data_collector/go_home")
        self.home_zone_cli = self.node.create_client(Trigger, "/data_collector/go_home_zone")
        self.cancel_home_zone_cli = self.node.create_client(Trigger, "/data_collector/cancel_home_zone")
        self.set_param_cli = self.node.create_client(SetParameters, "/data_collector/set_parameters")

        self.stats_timer = self.node.create_timer(0.1, self.stats_timer_callback)

        while rclpy.ok() and self._is_running:
            rclpy.spin_once(self.node, timeout_sec=0.05)

        self.node.destroy_node()
        rclpy.shutdown()

    def joint_callback(self, msg):
        target_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        if msg.name and msg.position:
            name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
            out = []
            for joint_name in target_joints:
                if joint_name in name_to_idx:
                    out.append(msg.position[name_to_idx[joint_name]])

            if len(out) == 6:
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
        if self.start_cli.wait_for_service(timeout_sec=1.0):
            future = self.start_cli.call_async(Trigger.Request())
            future.add_done_callback(self.start_record_done)
        else:
            self.log_signal.emit("错误: 找不到 /data_collector/start 服务")

    def call_stop_record(self):
        if self.stop_cli.wait_for_service(timeout_sec=1.0):
            future = self.stop_cli.call_async(Trigger.Request())
            future.add_done_callback(self.stop_record_done)
        else:
            self.log_signal.emit("错误: 找不到 /data_collector/stop 服务")

    def call_go_home(self):
        if self._inference_control_enabled:
            self.set_inference_execution_enabled(False)
            self.log_signal.emit("回 Home 前已停止推理执行。")

        if self.home_cli.wait_for_service(timeout_sec=1.0):
            future = self.home_cli.call_async(Trigger.Request())
            future.add_done_callback(self.go_home_done)
            return

        self.log_signal.emit("未检测到 /data_collector/go_home 服务，改用本地 Home 轨迹。")
        self._start_local_go_home()

    def call_go_home_zone(self):
        if self._inference_control_enabled:
            self.set_inference_execution_enabled(False)
            self.log_signal.emit("回 Home Zone 前已停止推理执行。")

        if self.home_zone_cli is None or not self.home_zone_cli.wait_for_service(timeout_sec=1.0):
            self.log_signal.emit("错误: 找不到 /data_collector/go_home_zone 服务。Home Zone 仅在采集节点内实现。")
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

    def _start_local_go_home(self) -> None:
        if self._homing_in_progress:
            self.log_signal.emit("Home 轨迹已在执行中。")
            return
        if len(self._home_joint_positions) != 6:
            self.log_signal.emit("错误: Home 点配置无效，需要 6 个关节角。")
            return
        self._homing_in_progress = True
        threading.Thread(target=self._execute_local_go_home_sequence, daemon=True).start()

    def _execute_local_go_home_sequence(self) -> None:
        try:
            if self._home_pub is None or self.node is None:
                self.log_signal.emit("错误: ROS2Worker 尚未就绪，无法执行 Home 轨迹。")
                return

            req_to_traj = SwitchController.Request()
            req_to_traj.activate_controllers = [self._trajectory_controller]
            req_to_traj.deactivate_controllers = [self._teleop_controller]
            req_to_traj.strictness = SwitchController.Request.BEST_EFFORT

            if self._switch_ctrl_client is not None and self._switch_ctrl_client.wait_for_service(timeout_sec=1.0):
                self.log_signal.emit(f"切换到 {self._trajectory_controller}，准备执行 Home 轨迹。")
                self._switch_ctrl_client.call_async(req_to_traj)
                time.sleep(0.5)
            else:
                self.log_signal.emit("未检测到 controller_manager，直接尝试发布 Home 轨迹。")

            trajectory = JointTrajectory()
            trajectory.joint_names = [str(name) for name in self._home_joint_names[:6]]

            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in self._home_joint_positions[:6]]
            duration = max(0.1, float(self._home_duration_sec))
            seconds = int(duration)
            nanoseconds = int((duration - seconds) * 1e9)
            point.time_from_start = DurationMsg(sec=seconds, nanosec=nanoseconds)
            trajectory.points = [point]
            self._home_pub.publish(trajectory)
            self.log_signal.emit(f"已发布 Home 轨迹，预计耗时 {duration:.2f}s。")

            time.sleep(duration + 0.5)

            req_to_teleop = SwitchController.Request()
            req_to_teleop.activate_controllers = [self._teleop_controller]
            req_to_teleop.deactivate_controllers = [self._trajectory_controller]
            req_to_teleop.strictness = SwitchController.Request.BEST_EFFORT

            if self._switch_ctrl_client is not None and self._switch_ctrl_client.wait_for_service(timeout_sec=1.0):
                self.log_signal.emit(f"恢复 {self._teleop_controller}。")
                self._switch_ctrl_client.call_async(req_to_teleop)

            self.log_signal.emit("Home 轨迹执行完成。")
        except Exception as exc:
            self.log_signal.emit(f"本地 Home 轨迹执行失败: {exc}")
        finally:
            self._homing_in_progress = False

    def call_set_home_from_current(self):
        joints = [float(value) for value in self.robot_state.get("joints", [])]
        if len(joints) != 6:
            self.log_signal.emit("错误: 当前关节状态无效，无法设置 Home 点")
            return

        if not self.set_param_cli.wait_for_service(timeout_sec=1.0):
            self.log_signal.emit("错误: 找不到 /data_collector 参数服务")
            return

        param = Parameter("home_joint_positions", Parameter.Type.DOUBLE_ARRAY, joints)
        req = SetParameters.Request()
        req.parameters = [param.to_parameter_msg()]
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
                self.demo_status_signal.emit(demo_name)
        except Exception as exc:
            self.log_signal.emit(f"服务调用异常: {exc}")

    def stop_record_done(self, future):
        try:
            response = future.result()
            self.log_signal.emit(f"停止录制服务: {response.message}")
            if response.success:
                self.is_recording = False
                self.realtime_record_fps = 0.0
                self.demo_status_signal.emit("无 (未录制)")
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
