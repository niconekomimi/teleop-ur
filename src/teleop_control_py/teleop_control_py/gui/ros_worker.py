import math
import re
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from PySide6.QtCore import QThread, Signal
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32, Float32MultiArray
from std_srvs.srv import Trigger


class ROS2Worker(QThread):
    global_image_signal = Signal(np.ndarray)
    wrist_image_signal = Signal(np.ndarray)
    robot_state_str_signal = Signal(str)
    record_stats_signal = Signal(int, str, float)
    log_signal = Signal(str)
    demo_status_signal = Signal(str)

    def __init__(self, global_topic, wrist_topic):
        super().__init__()
        self.global_topic = global_topic
        self.wrist_topic = wrist_topic
        self._global_sub_topic = None
        self._wrist_sub_topic = None
        self.node = None
        self.bridge = CvBridge()
        self._is_running = True

        self.global_sub = None
        self.wrist_sub = None
        self.enable_image_processing = False

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
        }

    def _emit_image_topics_log(self) -> None:
        self.log_signal.emit(
            f"ROS 2 图像监听话题已切换:\n - 全局: {self.global_topic} (预览打开时才订阅)"
            f"\n - 手部: {self.wrist_topic} (预览打开时才订阅)"
        )

    def set_image_topics(self, global_topic: str, wrist_topic: str) -> None:
        new_global_topic = str(global_topic).strip()
        new_wrist_topic = str(wrist_topic).strip()
        changed = (new_global_topic != self.global_topic) or (new_wrist_topic != self.wrist_topic)

        self.global_topic = new_global_topic
        self.wrist_topic = new_wrist_topic
        if not changed:
            return

        if self.node is not None:
            self._destroy_image_subscriptions()
        self._emit_image_topics_log()

    def _ensure_image_subscriptions(self):
        if self.node is None:
            return
        if self.global_sub is not None and self._global_sub_topic != self.global_topic:
            try:
                self.node.destroy_subscription(self.global_sub)
            except Exception:
                pass
            self.global_sub = None
            self._global_sub_topic = None
        if self.wrist_sub is not None and self._wrist_sub_topic != self.wrist_topic:
            try:
                self.node.destroy_subscription(self.wrist_sub)
            except Exception:
                pass
            self.wrist_sub = None
            self._wrist_sub_topic = None

        if self.global_sub is None:
            self.global_sub = self.node.create_subscription(
                Image, self.global_topic, self.global_callback, qos_profile_sensor_data
            )
            self._global_sub_topic = self.global_topic
        if self.wrist_sub is None:
            self.wrist_sub = self.node.create_subscription(
                Image, self.wrist_topic, self.wrist_callback, qos_profile_sensor_data
            )
            self._wrist_sub_topic = self.wrist_topic

    def _destroy_image_subscriptions(self):
        if self.node is None:
            return
        if self.global_sub is not None:
            try:
                self.node.destroy_subscription(self.global_sub)
            except Exception:
                pass
            self.global_sub = None
            self._global_sub_topic = None
        if self.wrist_sub is not None:
            try:
                self.node.destroy_subscription(self.wrist_sub)
            except Exception:
                pass
            self.wrist_sub = None
            self._wrist_sub_topic = None

    def _image_subs_timer_callback(self):
        if self.enable_image_processing:
            self._ensure_image_subscriptions()
        else:
            self._destroy_image_subscriptions()

    def run(self):
        rclpy.init()
        self.node = Node("teleop_gui_node")
        self.image_subs_timer = self.node.create_timer(0.2, self._image_subs_timer_callback)

        self.joint_sub = self.node.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.pose_sub = self.node.create_subscription(PoseStamped, "/tcp_pose_broadcaster/pose", self.pose_callback, 10)
        self.gripper_sub = self.node.create_subscription(Float32, "/gripper/cmd", self.gripper_callback, 10)
        self.record_stats_sub = self.node.create_subscription(
            Float32MultiArray,
            "/data_collector/record_stats",
            self.record_stats_callback,
            qos_profile_sensor_data,
        )

        self.start_cli = self.node.create_client(Trigger, "/data_collector/start")
        self.stop_cli = self.node.create_client(Trigger, "/data_collector/stop")
        self.home_cli = self.node.create_client(Trigger, "/data_collector/go_home")
        self.set_param_cli = self.node.create_client(SetParameters, "/data_collector/set_parameters")

        self.stats_timer = self.node.create_timer(0.1, self.stats_timer_callback)

        self._emit_image_topics_log()

        while rclpy.ok() and self._is_running:
            rclpy.spin_once(self.node, timeout_sec=0.05)

        try:
            self._destroy_image_subscriptions()
        except Exception:
            pass

        self.node.destroy_node()
        rclpy.shutdown()

    def global_callback(self, msg):
        try:
            if not self.enable_image_processing:
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.global_image_signal.emit(cv_image)
        except Exception:
            pass

    def wrist_callback(self, msg):
        try:
            if not self.enable_image_processing:
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.wrist_image_signal.emit(cv_image)
        except Exception:
            pass

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

    def _quat_to_rotvec_xyzw(self, q_xyzw):
        quat = np.array(q_xyzw, dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if norm <= 0.0:
            normalized = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            normalized = quat / norm

        x, y, z, w = (float(normalized[0]), float(normalized[1]), float(normalized[2]), float(normalized[3]))
        w = max(-1.0, min(1.0, w))
        angle = 2.0 * math.acos(w)
        sine = math.sqrt(max(0.0, 1.0 - w * w))

        if sine < 1e-8 or angle < 1e-8:
            return np.zeros(3, dtype=np.float32)

        axis = np.array([x / sine, y / sine, z / sine], dtype=np.float64)
        return (axis * angle).astype(np.float32)

    def _emit_robot_state(self):
        joints = np.array(self.robot_state.get("joints", [0.0] * 6))
        pos = np.array(self.robot_state.get("pose", [0.0, 0.0, 0.0]))
        quat = np.array(self.robot_state.get("quat", [0.0, 0.0, 0.0, 1.0]))
        gripper = self.robot_state.get("gripper", 0.0)

        rotvec = self._quat_to_rotvec_xyzw(quat)
        action = np.array([pos[0], pos[1], pos[2], rotvec[0], rotvec[1], rotvec[2], gripper])

        formatter = {"float_kind": lambda value: f"{value:6.3f}"}
        joints_str = np.array2string(joints, formatter=formatter)
        pos_str = np.array2string(pos, formatter=formatter)
        quat_str = np.array2string(quat, formatter=formatter)
        action_str = np.array2string(action, formatter=formatter)

        text = "【实时机器人状态 (与HDF5写入对齐)】\n"
        text += "-" * 25 + "\n"
        text += f"► 关节位置 [6]:\n {joints_str}\n\n"
        text += f"► 末端 XYZ [3]:\n {pos_str}\n\n"
        text += f"► 末端 四元数 [4]:\n {quat_str}\n\n"
        text += "-" * 25 + "\n"
        text += f"► 实时 Action [7]:\n {action_str}\n"
        text += "  (XYZ, RxRyRz, Gripper)"
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
        if self.home_cli.wait_for_service(timeout_sec=1.0):
            future = self.home_cli.call_async(Trigger.Request())
            future.add_done_callback(self.go_home_done)
        else:
            self.log_signal.emit("错误: 找不到 /data_collector/go_home 服务")

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
        self._is_running = False
        self.wait()
