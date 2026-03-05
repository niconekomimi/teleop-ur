import sys
import os
import subprocess
import signal
import cv2
import time
import math
import numpy as np
import h5py
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
    QGridLayout, QMessageBox, QDialog, QSizePolicy, QCheckBox, QSlider,
    QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QTextCursor


# ==========================================
# ROS 2 后台工作线程
# ==========================================
class ROS2Worker(QThread):
    global_image_signal = Signal(np.ndarray)
    wrist_image_signal = Signal(np.ndarray)
    robot_state_str_signal = Signal(str)
    record_stats_signal = Signal(int, str)  # 帧数, 时间字符串
    log_signal = Signal(str)
    demo_status_signal = Signal(str)

    def __init__(self, global_topic, wrist_topic):
        super().__init__()
        self.global_topic = global_topic
        self.wrist_topic = wrist_topic
        self.node = None
        self.bridge = CvBridge()
        self._is_running = True

        self.global_sub = None
        self.wrist_sub = None

        # Only decode/emit images when a preview window is actually consuming them.
        # This avoids heavy cv_bridge + Qt rendering cost during recording.
        self.enable_image_processing = False
        
        self.is_recording = False
        self.start_time = 0.0
        self.recorded_frames = 0
        
        self.robot_state = {
            'joints': [0.0] * 6,
            'pose': [0.0, 0.0, 0.0],
            'quat': [0.0, 0.0, 0.0, 1.0],
            'gripper': 0.0
        }

    def _ensure_image_subscriptions(self):
        if self.node is None:
            return
        if self.global_sub is None:
            self.global_sub = self.node.create_subscription(
                Image, self.global_topic, self.global_callback, qos_profile_sensor_data
            )
        if self.wrist_sub is None:
            self.wrist_sub = self.node.create_subscription(
                Image, self.wrist_topic, self.wrist_callback, qos_profile_sensor_data
            )

    def _destroy_image_subscriptions(self):
        if self.node is None:
            return
        if self.global_sub is not None:
            try:
                self.node.destroy_subscription(self.global_sub)
            except Exception:
                pass
            self.global_sub = None
        if self.wrist_sub is not None:
            try:
                self.node.destroy_subscription(self.wrist_sub)
            except Exception:
                pass
            self.wrist_sub = None

    def _image_subs_timer_callback(self):
        # Subscribing to sensor_msgs/Image still deserializes/copies large buffers into Python.
        # Only subscribe when we truly need preview images.
        if self.enable_image_processing:
            self._ensure_image_subscriptions()
        else:
            self._destroy_image_subscriptions()

    def run(self):
        rclpy.init()
        self.node = Node('teleop_gui_node')

        # Do NOT subscribe to images by default; only when preview is open.
        self.image_subs_timer = self.node.create_timer(0.2, self._image_subs_timer_callback)
            
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.pose_sub = self.node.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self.pose_callback, 10)
        self.gripper_sub = self.node.create_subscription(Float32, '/gripper/cmd', self.gripper_callback, 10)

        # 录制与控制服务客户端
        self.start_cli = self.node.create_client(Trigger, '/data_collector/start')
        self.stop_cli = self.node.create_client(Trigger, '/data_collector/stop')
        self.home_cli = self.node.create_client(Trigger, '/data_collector/go_home')
        self.set_param_cli = self.node.create_client(SetParameters, '/data_collector/set_parameters')

        self.stats_timer = self.node.create_timer(0.1, self.stats_timer_callback)

        self.log_signal.emit(
            f"ROS 2 监听已启动:\n - 全局: {self.global_topic} (预览打开时才订阅)"
            f"\n - 手部: {self.wrist_topic} (预览打开时才订阅)"
        )
        
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
            if self.is_recording:
                self.recorded_frames += 1
            if not self.enable_image_processing:
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.global_image_signal.emit(cv_image)
        except Exception:
            pass

    def wrist_callback(self, msg):
        try:
            if not self.enable_image_processing:
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.wrist_image_signal.emit(cv_image)
        except Exception:
            pass

    def joint_callback(self, msg):
        target_joints = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        if msg.name and msg.position:
            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            out = []
            for jn in target_joints:
                if jn in name_to_idx:
                    out.append(msg.position[name_to_idx[jn]])

            if len(out) == 6:
                self.robot_state['joints'] = out
                self._emit_robot_state()

    def pose_callback(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        self.robot_state['pose'] = [p.x, p.y, p.z]
        self.robot_state['quat'] = [q.x, q.y, q.z, q.w]
        self._emit_robot_state()

    def gripper_callback(self, msg):
        self.robot_state['gripper'] = max(0.0, min(1.0, float(msg.data)))
        self._emit_robot_state()

    def _quat_to_rotvec_xyzw(self, q_xyzw):
        q = np.array(q_xyzw, dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm <= 0.0:
            qn = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            qn = q / norm

        x, y, z, w = float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3])
        w = max(-1.0, min(1.0, w))
        angle = 2.0 * math.acos(w)
        s = math.sqrt(max(0.0, 1.0 - w * w))

        if s < 1e-8 or angle < 1e-8:
            return np.zeros(3, dtype=np.float32)

        axis = np.array([x / s, y / s, z / s], dtype=np.float64)
        return (axis * angle).astype(np.float32)

    def _emit_robot_state(self):
        j = np.array(self.robot_state.get('joints', [0.0] * 6))
        p = np.array(self.robot_state.get('pose', [0.0, 0.0, 0.0]))
        q = np.array(self.robot_state.get('quat', [0.0, 0.0, 0.0, 1.0]))
        gripper = self.robot_state.get('gripper', 0.0)

        rotvec = self._quat_to_rotvec_xyzw(q)
        action = np.array([p[0], p[1], p[2], rotvec[0], rotvec[1], rotvec[2], gripper])

        j_str = np.array2string(j, formatter={'float_kind':lambda x: f"{x:6.3f}"})
        p_str = np.array2string(p, formatter={'float_kind':lambda x: f"{x:6.3f}"})
        q_str = np.array2string(q, formatter={'float_kind':lambda x: f"{x:6.3f}"})
        a_str = np.array2string(action, formatter={'float_kind':lambda x: f"{x:6.3f}"})

        text = "【实时机器人状态 (与HDF5写入对齐)】\n"
        text += "-"*25 + "\n"
        text += f"► 关节位置 [6]:\n {j_str}\n\n"
        text += f"► 末端 XYZ [3]:\n {p_str}\n\n"
        text += f"► 末端 四元数 [4]:\n {q_str}\n\n"
        text += "-"*25 + "\n"
        text += f"► 实时 Action [7]:\n {a_str}\n"
        text += "  (XYZ, RxRyRz, Gripper)"

        self.robot_state_str_signal.emit(text)

    def stats_timer_callback(self):
        if self.is_recording:
            elapsed_sec = int(time.time() - self.start_time)
            mins = elapsed_sec // 60
            secs = elapsed_sec % 60
            time_str = f"{mins:02d}:{secs:02d}"
            frames = self.recorded_frames if (self.global_sub is not None) else -1
            self.record_stats_signal.emit(frames, time_str)

    def call_start_record(self):
        if self.start_cli.wait_for_service(timeout_sec=1.0):
            req = Trigger.Request()
            future = self.start_cli.call_async(req)
            future.add_done_callback(self.start_record_done)
        else:
            self.log_signal.emit("错误: 找不到 /data_collector/start 服务")

    def call_stop_record(self):
        if self.stop_cli.wait_for_service(timeout_sec=1.0):
            req = Trigger.Request()
            future = self.stop_cli.call_async(req)
            future.add_done_callback(self.stop_record_done)
        else:
            self.log_signal.emit("错误: 找不到 /data_collector/stop 服务")

    def call_go_home(self):
        if self.home_cli.wait_for_service(timeout_sec=1.0):
            req = Trigger.Request()
            future = self.home_cli.call_async(req)
            future.add_done_callback(self.go_home_done)
        else:
            self.log_signal.emit("错误: 找不到 /data_collector/go_home 服务")

    def call_set_home_from_current(self):
        joints = [float(v) for v in self.robot_state.get('joints', [])]
        if len(joints) != 6:
            self.log_signal.emit("错误: 当前关节状态无效，无法设置 Home 点")
            return

        if not self.set_param_cli.wait_for_service(timeout_sec=1.0):
            self.log_signal.emit("错误: 找不到 /data_collector 参数服务")
            return

        param = Parameter('home_joint_positions', Parameter.Type.DOUBLE_ARRAY, joints)
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
                self.recorded_frames = 0
                
                msg_parts = response.message.split()
                demo_name = msg_parts[-1] if len(msg_parts) > 0 else "未知"
                self.demo_status_signal.emit(demo_name)
        except Exception as e:
            self.log_signal.emit(f"服务调用异常: {e}")

    def stop_record_done(self, future):
        try:
            response = future.result()
            self.log_signal.emit(f"停止录制服务: {response.message}")
            if response.success:
                self.is_recording = False
                self.demo_status_signal.emit("无 (未录制)")
        except Exception as e:
            self.log_signal.emit(f"服务调用异常: {e}")

    def go_home_done(self, future):
        try:
            response = future.result()
            self.log_signal.emit(f"回Home服务: {response.message}")
        except Exception as e:
            self.log_signal.emit(f"服务调用异常: {e}")

    def set_home_done(self, future):
        try:
            resp = future.result()
            results = getattr(resp, 'results', None)
            if results and len(results) > 0 and bool(results[0].successful):
                joints = [float(v) for v in self.robot_state.get('joints', [])]
                j_str = np.array2string(np.array(joints), formatter={'float_kind':lambda x: f"{x:6.3f}"})
                self.log_signal.emit(f"已将当前关节姿态设置为 Home 点(本次运行生效): {j_str}")
            else:
                reason = None
                if results and len(results) > 0:
                    reason = getattr(results[0], 'reason', None)
                self.log_signal.emit(f"设置 Home 点失败: {reason or '未知原因'}")
        except Exception as e:
            self.log_signal.emit(f"设置 Home 点异常: {e}")

    def stop(self):
        self._is_running = False
        self.wait()


# ==========================================
# 已录制数据集 (HDF5) 回放预览窗口
# ==========================================
class HDF5ViewerDialog(QDialog):
    def __init__(self, initial_hdf5_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDF5 数据集高级回放器")
        self.resize(1150, 750)
        
        self.hdf5_path = initial_hdf5_path
        self.f = None
        self.current_demo_group = None

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.on_play_timeout)
        self.is_playing = False
        self.base_interval_ms = 100 

        self.setup_ui()
        
        if os.path.exists(self.hdf5_path):
            self.open_hdf5_file(self.hdf5_path)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        self.btn_open_file = QPushButton("📂 选择 HDF5 文件")
        self.btn_open_file.setStyleSheet("font-weight: bold; background-color: #d0e8f1;")
        self.btn_open_file.clicked.connect(self.open_file_dialog)
        top_layout.addWidget(self.btn_open_file)
        
        self.lbl_file_path = QLabel(os.path.basename(self.hdf5_path) if self.hdf5_path else "未选择文件")
        self.lbl_file_path.setStyleSheet("color: #555; font-style: italic;")
        top_layout.addWidget(self.lbl_file_path)

        top_layout.addWidget(QLabel("  |  选择录制序列:"))
        self.demo_combo = QComboBox()
        self.demo_combo.currentIndexChanged.connect(self.load_demo)
        top_layout.addWidget(self.demo_combo)
        
        self.lbl_frame_info = QLabel("当前帧: 0 / 0")
        self.lbl_frame_info.setStyleSheet("font-weight: bold; color: blue;")
        top_layout.addWidget(self.lbl_frame_info)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        ctrl_layout = QHBoxLayout()
        self.btn_prev = QPushButton("⏮ 上一帧")
        self.btn_prev.clicked.connect(self.step_prev)
        ctrl_layout.addWidget(self.btn_prev)

        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setStyleSheet("font-weight: bold; color: green; min-width: 80px;")
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.btn_play)

        self.btn_next = QPushButton("⏭ 下一帧")
        self.btn_next.clicked.connect(self.step_next)
        ctrl_layout.addWidget(self.btn_next)

        ctrl_layout.addWidget(QLabel("  倍速:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "2.0x", "4.0x", "8.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        ctrl_layout.addWidget(self.speed_combo)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.sliderPressed.connect(self.pause_playback)
        ctrl_layout.addWidget(self.slider)

        main_layout.addLayout(ctrl_layout)

        content_layout = QHBoxLayout()

        cam_layout = QVBoxLayout()
        global_title = QLabel("【全局相机 (Agent View)】")
        global_title.setAlignment(Qt.AlignCenter)
        self.lbl_agent = QLabel("无画面")
        self.lbl_agent.setAlignment(Qt.AlignCenter)
        self.lbl_agent.setStyleSheet("background-color: black; color: white;")
        self.lbl_agent.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        wrist_title = QLabel("【手部相机 (Eye-in-Hand)】")
        wrist_title.setAlignment(Qt.AlignCenter)
        self.lbl_wrist = QLabel("无画面")
        self.lbl_wrist.setAlignment(Qt.AlignCenter)
        self.lbl_wrist.setStyleSheet("background-color: black; color: white;")
        self.lbl_wrist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        cam_layout.addWidget(global_title)
        cam_layout.addWidget(self.lbl_agent)
        cam_layout.addWidget(wrist_title)
        cam_layout.addWidget(self.lbl_wrist)
        content_layout.addLayout(cam_layout, stretch=2)

        self.text_state = QTextEdit()
        self.text_state.setReadOnly(True)
        self.text_state.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; background-color: #f5f5f5;")
        content_layout.addWidget(self.text_state, stretch=1)

        main_layout.addLayout(content_layout)

    def open_file_dialog(self):
        start_dir = os.path.dirname(self.hdf5_path) if self.hdf5_path else os.getcwd()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择已录制的 HDF5 数据集", 
            start_dir, 
            "HDF5 Files (*.hdf5 *.h5);;All Files (*)"
        )
        if file_path:
            self.open_hdf5_file(file_path)

    def open_hdf5_file(self, path):
        self.pause_playback()
        if self.f is not None:
            self.f.close()
            self.f = None
            
        self.hdf5_path = path
        self.lbl_file_path.setText(os.path.basename(path))
        self.setWindowTitle(f"HDF5 数据集高级回放器 - {os.path.basename(path)}")
        self.demo_combo.clear()
        
        try:
            self.f = h5py.File(path, 'r')
            if 'data' not in self.f:
                raise KeyError("HDF5 文件中未找到 'data' 根组，格式不符合要求。")
                
            demos = list(self.f['data'].keys())
            if not demos:
                raise ValueError("数据集中没有任何 demo 序列。")
                
            demos.sort(key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
            self.demo_combo.addItems(demos)
            
        except Exception as e:
            QMessageBox.critical(self, "HDF5 读取错误", str(e))

    def load_demo(self):
        self.pause_playback()
        demo_name = self.demo_combo.currentText()
        if not demo_name or self.f is None:
            return
            
        self.current_demo_group = self.f['data'][demo_name]
        
        num_samples = self.current_demo_group.attrs.get('num_samples', 0)
        if num_samples == 0 and 'actions' in self.current_demo_group:
            num_samples = self.current_demo_group['actions'].shape[0]
            
        if num_samples > 0:
            self.slider.blockSignals(True)
            self.slider.setMaximum(int(num_samples) - 1)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self.update_frame_display()
        else:
            self.lbl_frame_info.setText("空序列")

    def toggle_play(self):
        if self.current_demo_group is None:
            return
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
            
        self.is_playing = True
        self.btn_play.setText("⏸ 暂停")
        self.btn_play.setStyleSheet("font-weight: bold; color: red; min-width: 80px;")
        self.change_speed() 

    def pause_playback(self):
        self.is_playing = False
        self.playback_timer.stop()
        self.btn_play.setText("▶ 播放")
        self.btn_play.setStyleSheet("font-weight: bold; color: green; min-width: 80px;")

    def change_speed(self):
        text = self.speed_combo.currentText().replace("x", "")
        try:
            factor = float(text)
        except:
            factor = 1.0
            
        interval = int(self.base_interval_ms / factor)
        if self.is_playing:
            self.playback_timer.start(interval)

    def on_play_timeout(self):
        curr = self.slider.value()
        if curr < self.slider.maximum():
            self.slider.setValue(curr + 1)
        else:
            self.pause_playback() 

    def step_prev(self):
        self.pause_playback()
        curr = self.slider.value()
        if curr > 0:
            self.slider.setValue(curr - 1)

    def step_next(self):
        self.pause_playback()
        curr = self.slider.value()
        if curr < self.slider.maximum():
            self.slider.setValue(curr + 1)

    def on_slider_changed(self):
        self.update_frame_display()

    def update_frame_display(self):
        if self.current_demo_group is None:
            return
            
        idx = self.slider.value()
        total = self.slider.maximum() + 1
        self.lbl_frame_info.setText(f"当前帧: {idx + 1} / {total}")
        
        try:
            agent_rgb = self.current_demo_group['obs']['agentview_rgb'][idx]
            wrist_rgb = self.current_demo_group['obs']['eye_in_hand_rgb'][idx]
            
            h, w, ch = agent_rgb.shape
            bp = ch * w
            
            qimg_a = QImage(agent_rgb.data, w, h, bp, QImage.Format_RGB888)
            self.lbl_agent.setPixmap(QPixmap.fromImage(qimg_a).scaled(self.lbl_agent.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            qimg_w = QImage(wrist_rgb.data, w, h, bp, QImage.Format_RGB888)
            self.lbl_wrist.setPixmap(QPixmap.fromImage(qimg_w).scaled(self.lbl_wrist.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            joints = self.current_demo_group['obs']['robot0_joint_pos'][idx]
            pos = self.current_demo_group['obs']['robot0_eef_pos'][idx]
            quat = self.current_demo_group['obs']['robot0_eef_quat'][idx]
            actions = self.current_demo_group['actions'][idx]
            
            j_str = np.array2string(joints, formatter={'float_kind':lambda x: f"{x:6.3f}"})
            p_str = np.array2string(pos, formatter={'float_kind':lambda x: f"{x:6.3f}"})
            q_str = np.array2string(quat, formatter={'float_kind':lambda x: f"{x:6.3f}"})
            a_str = np.array2string(actions, formatter={'float_kind':lambda x: f"{x:6.3f}"})
            
            text = "【录制帧数据】\n"
            text += "-"*25 + "\n"
            text += f"► 关节位置 [6]:\n {j_str}\n\n"
            text += f"► 末端 XYZ [3]:\n {p_str}\n\n"
            text += f"► 末端 四元数 [4]:\n {q_str}\n\n"
            text += "-"*25 + "\n"
            text += f"► 保存的 Action [7]:\n {a_str}\n"
            text += "  (XYZ, RxRyRz, Gripper)"
            
            self.text_state.setText(text)
            
        except Exception as e:
            self.text_state.setText(f"读取帧数据失败: {e}")

    def closeEvent(self, event):
        self.pause_playback()
        if self.f is not None:
            self.f.close()
        event.accept()


# ==========================================
# 实时图像与状态预览窗口
# ==========================================
class CameraPreviewWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("实时预览与状态监视器")
        self.resize(1150, 750)
        
        self.show_cropped_only = True 
        
        main_layout = QVBoxLayout(self)
        
        # --- 顶部：复选框与录制状态 ---
        top_layout = QHBoxLayout()
        self.crop_cb = QCheckBox("仅显示中心裁切区域 (与录入数据集画面完全一致)")
        self.crop_cb.setStyleSheet("font-weight: bold; color: #d32f2f;")
        self.crop_cb.setChecked(True)
        self.crop_cb.toggled.connect(self.on_crop_toggled)
        top_layout.addWidget(self.crop_cb)
        
        top_layout.addStretch()
        
        self.lbl_record_status = QLabel("状态: 未录制 | 时长: 00:00 | 估算帧数: 0")
        self.lbl_record_status.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;")
        top_layout.addWidget(self.lbl_record_status)
        main_layout.addLayout(top_layout)
        
        # --- 中部内容区：左侧相机，右侧状态 ---
        content_layout = QHBoxLayout()
        
        # 1. 左侧相机布局 (上下排列)
        # 如果你希望左侧的两个相机也是左右排列，只需将这里的 QVBoxLayout 改为 QHBoxLayout 即可
        cameras_layout = QVBoxLayout()
        
        global_title = QLabel("【全局相机 (Agent View)】")
        global_title.setAlignment(Qt.AlignCenter)
        self.global_label = QLabel("无画面")
        self.global_label.setAlignment(Qt.AlignCenter)
        self.global_label.setStyleSheet("background-color: black; color: white;")
        self.global_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cameras_layout.addWidget(global_title)
        cameras_layout.addWidget(self.global_label)
        
        wrist_title = QLabel("【手部相机 (Eye-in-Hand)】")
        wrist_title.setAlignment(Qt.AlignCenter)
        self.wrist_label = QLabel("无画面")
        self.wrist_label.setAlignment(Qt.AlignCenter)
        self.wrist_label.setStyleSheet("background-color: black; color: white;")
        self.wrist_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cameras_layout.addWidget(wrist_title)
        cameras_layout.addWidget(self.wrist_label)
        
        # 将相机区域加入主内容区，分配较大权重 (stretch=3)
        content_layout.addLayout(cameras_layout, stretch=3)
        
        # 2. 右侧机器人状态布局
        self.text_robot_state = QTextEdit()
        self.text_robot_state.setReadOnly(True)
        self.text_robot_state.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; background-color: #fcfcfc;")
        self.text_robot_state.setText("等待机器人状态数据...")
        self.text_robot_state.setMinimumWidth(300) # 保证右侧文本框不会被过度挤压
        
        # 将状态文本框加入主内容区，分配较小权重 (stretch=1)
        content_layout.addWidget(self.text_robot_state, stretch=1)
        
        main_layout.addLayout(content_layout)

    def on_crop_toggled(self, checked):
        self.show_cropped_only = checked

    def process_image(self, cv_img):
        if cv_img is None or len(cv_img.shape) < 2:
            return cv_img
            
        h, w = cv_img.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2

        if self.show_cropped_only:
            return cv_img[y0:y0+side, x0:x0+side].copy()

        overlay = cv_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        masked_img = cv2.addWeighted(overlay, 0.5, cv_img, 0.5, 0)
        masked_img[y0:y0+side, x0:x0+side] = cv_img[y0:y0+side, x0:x0+side]
        cv2.rectangle(masked_img, (x0, y0), (x0+side, y0+side), (0, 255, 0), 2)
        
        return masked_img

    def cv2_to_qpixmap(self, cv_img):
        try:
            processed_img = self.process_image(cv_img)
            rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)
        except Exception:
            return QPixmap()

    @Slot(np.ndarray)
    def update_global_image(self, cv_img):
        pixmap = self.cv2_to_qpixmap(cv_img)
        if not pixmap.isNull():
            self.global_label.setPixmap(pixmap.scaled(self.global_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(np.ndarray)
    def update_wrist_image(self, cv_img):
        pixmap = self.cv2_to_qpixmap(cv_img)
        if not pixmap.isNull():
            self.wrist_label.setPixmap(pixmap.scaled(self.wrist_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(str)
    def update_robot_state_str(self, text):
        self.text_robot_state.setText(text)

    @Slot(int, str)
    def update_record_stats(self, frames, time_str):
        frames_str = "N/A" if frames is None or int(frames) < 0 else str(int(frames))
        self.lbl_record_status.setText(f"状态: 🔴录制中 | 时长: {time_str} | 估算帧数: {frames_str}")
        self.lbl_record_status.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")
        
    def reset_record_stats(self):
        self.lbl_record_status.setText("状态: 未录制 | 时长: 00:00 | 估算帧数: 0")
        self.lbl_record_status.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;")


# ==========================================
# 主界面
# ==========================================
class TeleopMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LIBERO Teleop & Data Collection Station")
        self.resize(850, 750)
        
        self.processes = {}
        self.ros_worker = None
        self.preview_window = None

        self.setup_ui()
        self.refresh_topics()

    def _shutdown(self) -> None:
        # Idempotent cleanup for both window-close and Ctrl-C exit.
        try:
            if self.ros_worker:
                self.ros_worker.stop()
        except Exception:
            pass

        for key in list(self.processes.keys()):
            try:
                self.kill_subprocess(key)
            except Exception:
                pass

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # ------------------ 1. 系统配置区 ------------------
        settings_group = QGroupBox("1. 系统配置")
        settings_layout = QGridLayout()
        
        settings_layout.addWidget(QLabel("遥操作模式:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["xbox", "hand"])
        settings_layout.addWidget(self.mode_combo, 0, 1)

        settings_layout.addWidget(QLabel("机器人 IP:"), 0, 2)
        self.ip_input = QLineEdit("192.168.1.211")
        settings_layout.addWidget(self.ip_input, 0, 3)

        settings_layout.addWidget(QLabel("末端执行器:"), 0, 4)
        self.ee_combo = QComboBox()
        self.ee_combo.addItems(["qbsofthand", "robotiq"])
        self.ee_combo.setCurrentText("robotiq")
        settings_layout.addWidget(self.ee_combo, 0, 5)

        settings_layout.addWidget(QLabel("全局相机话题:"), 1, 0)
        self.global_topic_combo = QComboBox()
        self.global_topic_combo.setEditable(True)
        settings_layout.addWidget(self.global_topic_combo, 1, 1)

        settings_layout.addWidget(QLabel("手部相机话题:"), 1, 2)
        self.wrist_topic_combo = QComboBox()
        self.wrist_topic_combo.setEditable(True)
        settings_layout.addWidget(self.wrist_topic_combo, 1, 3)

        self.btn_refresh = QPushButton("🔄 刷新图像话题")
        self.btn_refresh.clicked.connect(self.refresh_topics)
        settings_layout.addWidget(self.btn_refresh, 1, 4)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # ------------------ 2. 硬件驱动区 ------------------
        hw_group = QGroupBox("2. 硬件驱动 (Sensors)")
        hw_layout = QHBoxLayout()

        self.btn_rs = QPushButton("▶ 启动 RealSense")
        self.btn_rs.setCheckable(True)
        self.btn_rs.clicked.connect(self.toggle_realsense)

        self.btn_oak = QPushButton("▶ 启动 OAK")
        self.btn_oak.setCheckable(True)
        self.btn_oak.clicked.connect(self.toggle_oak)

        hw_layout.addWidget(self.btn_rs)
        hw_layout.addWidget(self.btn_oak)
        hw_group.setLayout(hw_layout)
        main_layout.addWidget(hw_group)

        # ------------------ 3. 遥操作系统区 ----------
        sys_group = QGroupBox("3. 遥操作系统 (Teleop System)")
        sys_layout = QHBoxLayout()

        self.btn_teleop = QPushButton("▶ 启动遥操作系统")
        self.btn_teleop.setMinimumHeight(40)
        self.btn_teleop.setStyleSheet("font-weight: bold;")
        self.btn_teleop.setCheckable(True)
        self.btn_teleop.clicked.connect(self.toggle_teleop)
        
        sys_layout.addWidget(self.btn_teleop)
        sys_group.setLayout(sys_layout)
        main_layout.addWidget(sys_group)

        # ------------------ 4. 数据录制区 ------------------
        record_group = QGroupBox("4. 数据录制 (需先启动采集节点)")
        record_layout = QGridLayout()

        # 第 0 行: 路径与回放按钮
        record_layout.addWidget(QLabel("HDF5 保存路径:"), 0, 0)
        self.record_path_input = QLineEdit(f"{os.getcwd()}/data/libero_demos.hdf5")
        record_layout.addWidget(self.record_path_input, 0, 1, 1, 3)
        
        self.btn_preview_hdf5 = QPushButton("📂 预览已录制文件(HDF5)")
        self.btn_preview_hdf5.setStyleSheet("background-color: #e0e0e0; font-weight: bold;")
        self.btn_preview_hdf5.clicked.connect(self.open_hdf5_viewer)
        record_layout.addWidget(self.btn_preview_hdf5, 0, 4)

        # 第 1 行: 控制按钮
        self.btn_collector = QPushButton("▶ 启动采集节点")
        self.btn_collector.setCheckable(True)
        self.btn_collector.clicked.connect(self.toggle_data_collector)
        record_layout.addWidget(self.btn_collector, 1, 0)

        self.btn_start_record = QPushButton("🔴 开始录制")
        self.btn_start_record.setStyleSheet("color: red; font-weight: bold;")
        self.btn_start_record.clicked.connect(self.start_record)
        record_layout.addWidget(self.btn_start_record, 1, 1)

        self.btn_stop_record = QPushButton("⬛ 停止录制")
        self.btn_stop_record.clicked.connect(self.stop_record)
        record_layout.addWidget(self.btn_stop_record, 1, 2)

        self.btn_go_home = QPushButton("🏠 回 Home 点")
        self.btn_go_home.setStyleSheet("font-weight: bold; color: #d35400;")
        self.btn_go_home.clicked.connect(self.go_home)
        record_layout.addWidget(self.btn_go_home, 1, 3)

        self.btn_set_home_current = QPushButton("📌 设当前姿态为 Home")
        self.btn_set_home_current.setStyleSheet("font-weight: bold; color: #1e8449;")
        self.btn_set_home_current.clicked.connect(self.set_home_from_current)
        record_layout.addWidget(self.btn_set_home_current, 1, 4)

        # 第 2 行: 状态信息
        record_layout.addWidget(QLabel("当前录制序列:"), 2, 0)
        self.lbl_demo_status = QLabel("无 (未录制)")
        self.lbl_demo_status.setStyleSheet("color: blue; font-weight: bold;")
        record_layout.addWidget(self.lbl_demo_status, 2, 1)
        
        self.lbl_main_record_stats = QLabel("录制时长: 00:00 | 帧数: 0")
        self.lbl_main_record_stats.setStyleSheet("font-weight: bold; color: #555;")
        record_layout.addWidget(self.lbl_main_record_stats, 2, 2, 1, 3)

        record_group.setLayout(record_layout)
        main_layout.addWidget(record_group)

        # ------------------ 5. 预览与日志区 ------------------
        preview_group = QGroupBox("5. 监视器与日志")
        preview_layout = QVBoxLayout()
        
        self.btn_preview = QPushButton("🖼️ 打开实时预览与状态窗")
        self.btn_preview.clicked.connect(self.open_preview_window)
        preview_layout.addWidget(self.btn_preview)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        preview_layout.addWidget(self.log_output)

        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # 在界面初始化时尝试加载一次已持久化的 Home 覆盖文件
        self._apply_persisted_home_to_ui_log()

    def _home_override_yaml_path(self) -> Path:
        # 放在 scripts 目录下，跟随工程一起移动，且通常可写
        return Path(__file__).resolve().parent / "teleop_gui_home_override.yaml"

    def _write_home_override_yaml(self, joints: List[float]) -> Optional[Path]:
        if len(joints) != 6:
            return None

        path = self._home_override_yaml_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            vals = ", ".join([f"{float(v):.6f}" for v in joints])
            content = (
                "data_collector:\n"
                "  ros__parameters:\n"
                f"    home_joint_positions: [{vals}]\n"
            )
            path.write_text(content, encoding="utf-8")
            return path
        except Exception as e:
            self.log(f"写入 Home 持久化文件失败: {e}")
            return None

    def _apply_persisted_home_to_ui_log(self) -> None:
        path = self._home_override_yaml_path()
        if path.exists():
            self.log(f"检测到已保存的 Home 覆盖文件: {path}")

    # ================= 业务逻辑 =================

    def log(self, message):
        self.log_output.append(message)
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def refresh_topics(self):
        self.btn_refresh.setEnabled(False)
        self.log("正在扫描活跃的图像话题...")
        QApplication.processEvents()
        
        # 定义你指定的默认话题
        global_default = "/camera/camera/color/image_raw"
        # GUI 启动 OAK 使用 depthai_ros_driver，常见 RGB topic 为 /color/video/image
        wrist_default = "/color/video/image"
        
        try:
            res = subprocess.run(
                ['ros2', 'topic', 'list', '-t'], 
                capture_output=True, text=True, timeout=3
            )
            image_topics = []
            for line in res.stdout.splitlines():
                if 'sensor_msgs/msg/Image' in line:
                    topic_name = line.split()[0]
                    image_topics.append(topic_name)
                    
            self.global_topic_combo.clear()
            self.wrist_topic_combo.clear()
            
            if image_topics:
                # 列表去重
                image_topics = sorted(list(set(image_topics)))
                
                self.global_topic_combo.addItems(image_topics)
                self.wrist_topic_combo.addItems(image_topics)
                
                # 如果活跃话题里包含默认的，优先选中它；否则选第一个
                if global_default in image_topics:
                    self.global_topic_combo.setCurrentText(global_default)
                else:
                    self.global_topic_combo.setCurrentText(image_topics[0])
                    
                if wrist_default in image_topics:
                    self.wrist_topic_combo.setCurrentText(wrist_default)
                else:
                    self.wrist_topic_combo.setCurrentText(image_topics[0])
                    
                self.log(f"扫描完成，找到 {len(image_topics)} 个图像话题。")
            else:
                self.log("未检测到活跃图像话题。已填入默认值。")
                self.global_topic_combo.addItems([global_default])
                self.wrist_topic_combo.addItems([wrist_default])
                
        except Exception as e:
            self.log(f"话题扫描失败: {e}")
            self.global_topic_combo.addItems([global_default])
            self.wrist_topic_combo.addItems([wrist_default])
        finally:
            self.btn_refresh.setEnabled(True)

    def run_subprocess(self, key, cmd_list):
        self.log(f"执行指令: {' '.join(cmd_list)}")
        try:
            proc = subprocess.Popen(
                cmd_list, 
                preexec_fn=os.setsid
            )
            self.processes[key] = proc
            return True
        except Exception as e:
            self.log(f"启动 {key} 失败: {e}")
            return False

    def kill_subprocess(self, key):
        if key in self.processes:
            proc = self.processes[key]
            if proc.poll() is None:
                self.log(f"正在终止 {key} (SIGINT)...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    proc.wait(timeout=3)
                    self.log(f"{key} 已正常关闭。")
                except subprocess.TimeoutExpired:
                    self.log(f"超时！正在强制终止 {key} (SIGKILL)...")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        proc.wait(timeout=2)
                    except Exception as e:
                        self.log(f"强制终止失败: {e}")
                except Exception as e:
                    self.log(f"终止进程发生异常: {e}")
            del self.processes[key]

    # ================= 按钮槽函数 =================

    def toggle_realsense(self, checked):
        if checked:
            # Use RealSense launch defaults (do not override any parameters here).
            cmd = ["ros2", "launch", "realsense2_camera", "rs_launch.py"]
            self.run_subprocess("realsense", cmd)
            self.btn_rs.setText("⏹ 停止 RealSense")
            self.btn_rs.setStyleSheet("background-color: lightgreen;")
        else:
            self.kill_subprocess("realsense")
            self.btn_rs.setText("▶ 启动 RealSense")
            self.btn_rs.setStyleSheet("")

    def toggle_oak(self, checked):
        if checked:
            # Use depthai_examples to publish the video stream topic (/color/video/image).
            # This matches data_collector_params.yaml (oakd_rgb_topic) and avoids the /oak/... topic set.
            cmd = [
                "ros2",
                "launch",
                "depthai_examples",
                "rgb_stereo_node.launch.py",
                # "useVideo:=true",
                # "usePreview:=false",
                # "useDepth:=false",
            ]
            self.run_subprocess("oak", cmd)
            self.btn_oak.setText("⏹ 停止 OAK")
            self.btn_oak.setStyleSheet("background-color: lightgreen;")
        else:
            self.kill_subprocess("oak")
            self.btn_oak.setText("▶ 启动 OAK")
            self.btn_oak.setStyleSheet("")

    def toggle_teleop(self, checked):
        if checked:
            mode = self.mode_combo.currentText()
            ip = self.ip_input.text()
            ee = self.ee_combo.currentText().strip().lower() if hasattr(self, "ee_combo") else "auto"
            cmd = [
                "ros2", "launch", "teleop_control_py", "control_system.launch.py",
                f"robot_ip:={ip}",
                f"control_mode:={mode}",
                f"end_effector:={ee}",
                "enable_camera:=false"  # <---- 修复核心：严禁 launch 内部再次启动 RealSense，避免 USB 冲突
            ]
            self.run_subprocess("teleop", cmd)
            self.btn_teleop.setText("⏹ 停止遥操作系统")
            self.btn_teleop.setStyleSheet("background-color: lightgreen; font-weight: bold;")
            
            QMessageBox.information(self, "操作提示", "遥操作系统已启动！\n\n请不要忘记按示教器的【程序运行播放键】。")
        else:
            self.kill_subprocess("teleop")
            self.btn_teleop.setText("▶ 启动遥操作系统")
            self.btn_teleop.setStyleSheet("font-weight: bold;")

    def toggle_data_collector(self, checked):
        if checked:
            out_path = self.record_path_input.text()
            global_topic = self.global_topic_combo.currentText()
            wrist_topic = self.wrist_topic_combo.currentText()
            
            yaml_args = []
            try:
                cmd_pkg = ['ros2', 'pkg', 'prefix', 'teleop_control_py']
                res = subprocess.run(cmd_pkg, capture_output=True, text=True)
                if res.returncode == 0:
                    pkg_path = res.stdout.strip()
                    yaml_path = Path(pkg_path) / "share/teleop_control_py/config/data_collector_params.yaml"
                    if yaml_path.exists():
                        yaml_args = ["--params-file", str(yaml_path)]
            except Exception as e:
                pass

            # 追加 GUI 的 Home 持久化覆盖文件（若存在）
            home_override = self._home_override_yaml_path()
            if home_override.exists():
                yaml_args.extend(["--params-file", str(home_override)])

            cmd = [
                "ros2", "run", "teleop_control_py", "data_collector_node",
                "--ros-args"
            ]
            
            if yaml_args:
                cmd.extend(yaml_args)
                
            cmd.extend([
                "-p", f"output_path:={out_path}",
                "-p", f"global_image_topic:={global_topic}",
                "-p", f"wrist_image_topic:={wrist_topic}",
                "-p", "end_effector_type:=qbsofthand"  # 确保软体手类型
            ])

            self.run_subprocess("data_collector", cmd)
            self.btn_collector.setText("⏹ 停止采集节点")
            self.btn_collector.setStyleSheet("background-color: lightgreen;")
            
            self.start_ros_worker(global_topic, wrist_topic)
        else:
            self.kill_subprocess("data_collector")
            self.btn_collector.setText("▶ 启动采集节点")
            self.btn_collector.setStyleSheet("")
            
            if self.ros_worker:
                self.ros_worker.stop()
                self.ros_worker = None

    def start_ros_worker(self, global_topic, wrist_topic):
        if self.ros_worker is None:
            self.ros_worker = ROS2Worker(global_topic, wrist_topic)
            self.ros_worker.log_signal.connect(self.log)
            self.ros_worker.demo_status_signal.connect(self.update_demo_status)
            self.ros_worker.record_stats_signal.connect(self.update_main_record_stats)
            
            if self.preview_window:
                self.ros_worker.global_image_signal.connect(self.preview_window.update_global_image)
                self.ros_worker.wrist_image_signal.connect(self.preview_window.update_wrist_image)
                self.ros_worker.robot_state_str_signal.connect(self.preview_window.update_robot_state_str)
                self.ros_worker.record_stats_signal.connect(self.preview_window.update_record_stats)
            
            self.ros_worker.start()

    def start_record(self):
        if self.ros_worker:
            self.ros_worker.call_start_record()
        else:
            QMessageBox.warning(self, "警告", "请先启动采集节点！")

    def stop_record(self):
        if self.ros_worker:
            self.ros_worker.call_stop_record()

    def go_home(self):
        if self.ros_worker:
            self.ros_worker.call_go_home()
        else:
            QMessageBox.warning(self, "警告", "请先启动采集节点，因为 Home 服务依赖该节点提供。")

    def set_home_from_current(self):
        if self.ros_worker:
            joints = [float(v) for v in self.ros_worker.robot_state.get('joints', [])]
            if len(joints) != 6:
                QMessageBox.warning(self, "警告", "当前关节状态无效(需要 6 个关节角)，无法设置 Home 点。")
                return

            # 1) 立即对正在运行的 data_collector 生效
            self.ros_worker.call_set_home_from_current()

            # 2) 写入持久化覆盖文件，保证重启采集节点/关闭 GUI 后仍生效
            saved = self._write_home_override_yaml(joints)
            if saved:
                j_str = np.array2string(np.array(joints), formatter={'float_kind':lambda x: f"{x:6.3f}"})
                self.log(f"已持久化 Home 点到: {saved}\nHome joints: {j_str}")
        else:
            QMessageBox.warning(self, "警告", "请先启动采集节点并接收实时关节数据，再设置 Home 点。")

    @Slot(str)
    def update_demo_status(self, demo_name):
        self.lbl_demo_status.setText(demo_name)
        if demo_name != "无 (未录制)":
            self.lbl_demo_status.setStyleSheet("color: red; font-weight: bold;")
            self.lbl_main_record_stats.setStyleSheet("font-weight: bold; color: red;")
        else:
            self.lbl_demo_status.setStyleSheet("color: blue; font-weight: bold;")
            self.lbl_main_record_stats.setStyleSheet("font-weight: bold; color: #555;")
            self.lbl_main_record_stats.setText("录制已停止")
            if self.preview_window:
                self.preview_window.reset_record_stats()

    @Slot(int, str)
    def update_main_record_stats(self, frames, time_str):
        frames_str = "N/A" if frames is None or int(frames) < 0 else str(int(frames))
        self.lbl_main_record_stats.setText(f"录制时长: {time_str} | 估算帧数: {frames_str}")

    @Slot(int)
    def _on_preview_window_finished(self, _result: int):
        # The preview window is closed; stop heavy image decoding.
        if self.ros_worker is not None:
            self.ros_worker.enable_image_processing = False

    def open_preview_window(self):
        if self.ros_worker is None:
            QMessageBox.information(self, "提示", "请先点击【▶ 启动采集节点】以开启 ROS图像与状态监听。")
            return
            
        if self.preview_window is None:
            self.preview_window = CameraPreviewWindow(self)
            self.preview_window.finished.connect(self._on_preview_window_finished)
            self.ros_worker.global_image_signal.connect(self.preview_window.update_global_image)
            self.ros_worker.wrist_image_signal.connect(self.preview_window.update_wrist_image)
            self.ros_worker.robot_state_str_signal.connect(self.preview_window.update_robot_state_str)
            self.ros_worker.record_stats_signal.connect(self.preview_window.update_record_stats)
            
            if not self.ros_worker.is_recording:
                self.preview_window.reset_record_stats()

        # Enable heavy image decoding only when the preview window is used.
        self.ros_worker.enable_image_processing = True
        
        self.preview_window.show()
        self.preview_window.raise_()
        self.preview_window.activateWindow()

    def open_hdf5_viewer(self):
        if self.ros_worker and self.ros_worker.is_recording:
            QMessageBox.warning(self, "警告", "当前正在录制数据，为了防止文件损坏，请在【停止录制】后再进行 HDF5 预览。")
            return
            
        hdf5_path = self.record_path_input.text().strip()
        viewer = HDF5ViewerDialog(hdf5_path, parent=self)
        viewer.exec()

    def closeEvent(self, event):
        self._shutdown()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TeleopMainWindow()
    window.show()
    exit_code = 0
    try:
        # Keep Python's default SIGINT handler (KeyboardInterrupt) so we can cleanup.
        exit_code = app.exec()
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        try:
            window._shutdown()
        except Exception:
            pass
    sys.exit(exit_code)