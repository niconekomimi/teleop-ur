import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from teleop_control_py.gui_support import (
    build_camera_driver_command,
    build_robot_driver_command,
    build_teleop_command,
    collector_camera_occupancy,
    detect_joystick_devices,
    get_local_ip,
    hardware_conflicts_for_collector,
    load_gui_settings,
    save_gui_settings_overrides,
)
from teleop_control_py.model_inference import (
    InferenceWorker,
    MODELS_ROOT,
    TASK_EMBEDDINGS_ROOT,
    build_robot_state_vector,
    discover_checkpoint_dirs,
    discover_task_envs,
    format_action,
    guess_embedding_path,
    load_embedding_keys,
)

from .http_preview_worker import HttpPreviewWorker
from .ros_worker import ROS2Worker
from .widgets import CameraPreviewWindow, HDF5ViewerDialog


class TeleopMainWindow(QMainWindow):
    COLLECTOR_PREVIEW_API_BASE_URL = "http://127.0.0.1:8765"
    COLLECTOR_POSE_TOPIC = "/tcp_pose_broadcaster/pose"
    COLLECTOR_GRIPPER_TOPIC = "/gripper/cmd"
    COLLECTOR_ACTION_TOPIC = "/servo_node/delta_twist_cmds"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Teleop & Data Collection Station")
        self.resize(1240, 860)
        self.setMinimumSize(1240, 860)

        self.gui_settings = load_gui_settings(__file__)
        self.processes = {}
        self.ros_worker = None
        self.inference_worker = None
        self.preview_window = None
        self.preview_api_worker = None
        self._preview_api_active = False
        self._latest_inference_global_bgr = None
        self._latest_inference_wrist_bgr = None
        self.module_status_labels = {}
        self.hardware_status_labels = {}

        self.process_watch_timer = QTimer(self)
        self.process_watch_timer.timeout.connect(self._poll_subprocesses)
        self.process_watch_timer.start(1000)

        self.status_refresh_timer = QTimer(self)
        self.status_refresh_timer.timeout.connect(self._refresh_runtime_status)
        self.status_refresh_timer.start(1000)

        self.setup_ui()
        self._refresh_runtime_status()

    def _shutdown(self) -> None:
        try:
            self.stop_inference()
        except Exception:
            pass

        try:
            self._stop_preview_api_worker()
        except Exception:
            pass

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
        main_layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        section_style = (
            "QGroupBox { font-size: 15px; font-weight: 700; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #1f3a5f; }"
        )

        settings_group = QGroupBox("系统配置")
        settings_group.setStyleSheet(section_style)
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("输入后端:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("joy (手柄)", "joy")
        self.mode_combo.addItem("mediapipe (手势输入)", "mediapipe")
        default_input_index = max(0, self.mode_combo.findData(self.gui_settings.default_input_type))
        self.mode_combo.setCurrentIndex(default_input_index)
        settings_layout.addWidget(self.mode_combo, 0, 1)

        settings_layout.addWidget(QLabel("手柄型号:"), 0, 2)
        self.joy_profile_combo = QComboBox()
        for profile in self.gui_settings.joy_profiles:
            self.joy_profile_combo.addItem(profile, profile)
        joy_profile_index = max(0, self.joy_profile_combo.findData(self.gui_settings.default_joy_profile))
        self.joy_profile_combo.setCurrentIndex(joy_profile_index)
        settings_layout.addWidget(self.joy_profile_combo, 0, 3)

        settings_layout.addWidget(QLabel("UR 类型:"), 0, 4)
        self.ur_type_input = QLineEdit(self.gui_settings.ur_type or "ur5")
        self.ur_type_input.setPlaceholderText("例如: ur5, ur10e, ur16e")
        settings_layout.addWidget(self.ur_type_input, 0, 5, 1, 2)

        settings_layout.addWidget(QLabel("手势识别输入:"), 1, 0)
        self.mediapipe_topic_combo = QComboBox()
        self.mediapipe_topic_combo.setEditable(True)
        self.mediapipe_topic_combo.setCurrentText(self.gui_settings.default_mediapipe_input_topic)
        settings_layout.addWidget(self.mediapipe_topic_combo, 1, 1, 1, 5)

        self.btn_refresh_topics = QPushButton("刷新")
        self.btn_refresh_topics.clicked.connect(self.refresh_mediapipe_topics)
        settings_layout.addWidget(self.btn_refresh_topics, 1, 6)

        settings_layout.addWidget(QLabel("机器人 IP:"), 2, 0)
        self.ip_input = QLineEdit(self.gui_settings.default_robot_ip)
        settings_layout.addWidget(self.ip_input, 2, 1, 1, 2)

        settings_layout.addWidget(QLabel("本机 IP:"), 2, 3)
        self.local_ip_label = QLabel(get_local_ip())
        self.local_ip_label.setStyleSheet("font-weight: bold; color: #0b7285;")
        settings_layout.addWidget(self.local_ip_label, 2, 4, 1, 2)

        settings_layout.addWidget(QLabel("末端执行器:"), 3, 0)
        self.ee_combo = QComboBox()
        self.ee_combo.addItem("robotiq", "robotiq")
        self.ee_combo.addItem("qbsofthand", "qbsofthand")
        ee_index = max(0, self.ee_combo.findData(self.gui_settings.default_gripper_type))
        self.ee_combo.setCurrentIndex(ee_index)
        settings_layout.addWidget(self.ee_combo, 3, 1)

        self.input_hint_label = QLabel()
        self.input_hint_label.setWordWrap(True)
        self.input_hint_label.setStyleSheet("color: #555; font-size: 12px;")
        settings_layout.addWidget(self.input_hint_label, 3, 2, 1, 5)

        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)

        startup_group = QGroupBox("启动设置")
        startup_group.setStyleSheet(section_style)
        startup_layout = QGridLayout()

        startup_layout.addWidget(QLabel("相机 ROS2 驱动:"), 0, 0)
        self.camera_driver_combo = QComboBox()
        for option in self.gui_settings.camera_driver_options:
            self.camera_driver_combo.addItem(option, option)
        camera_driver_index = max(0, self.camera_driver_combo.findData(self.gui_settings.default_camera_driver))
        self.camera_driver_combo.setCurrentIndex(camera_driver_index)
        startup_layout.addWidget(self.camera_driver_combo, 0, 1)

        self.btn_camera_driver = QPushButton("启动相机驱动")
        self.btn_camera_driver.setCheckable(True)
        self.btn_camera_driver.clicked.connect(self.toggle_camera_driver)
        startup_layout.addWidget(self.btn_camera_driver, 0, 2)

        startup_layout.addWidget(QLabel("机械臂 ROS2 驱动:"), 1, 0)
        self.btn_robot_driver = QPushButton("启动机械臂驱动")
        self.btn_robot_driver.setCheckable(True)
        self.btn_robot_driver.clicked.connect(self.toggle_robot_driver)
        startup_layout.addWidget(self.btn_robot_driver, 1, 1, 1, 2)

        startup_layout.addWidget(QLabel("遥操作系统:"), 2, 0)
        self.btn_teleop = QPushButton("启动遥操作系统")
        self.btn_teleop.setMinimumHeight(40)
        self.btn_teleop.setStyleSheet("font-weight: bold;")
        self.btn_teleop.setCheckable(True)
        self.btn_teleop.clicked.connect(self.toggle_teleop)
        startup_layout.addWidget(self.btn_teleop, 2, 1, 1, 2)

        self.startup_hint_label = QLabel("当遥操作系统启动时，会接管机械臂驱动；GUI 会显示机械臂驱动为运行中，但不允许单独关闭。")
        self.startup_hint_label.setWordWrap(True)
        self.startup_hint_label.setStyleSheet("color: #555; font-size: 12px;")
        startup_layout.addWidget(self.startup_hint_label, 3, 0, 1, 3)

        startup_group.setLayout(startup_layout)
        left_layout.addWidget(startup_group)

        status_group = QGroupBox("状态总览")
        status_group.setStyleSheet(section_style)
        status_container_layout = QVBoxLayout()

        module_group = QGroupBox("模块情况")
        module_group.setStyleSheet(section_style)
        module_layout = QGridLayout()
        module_layout.addWidget(QLabel("模块"), 0, 0)
        module_layout.addWidget(QLabel("状态"), 0, 1)

        hardware_group = QGroupBox("硬件情况")
        hardware_group.setStyleSheet(section_style)
        hardware_layout = QGridLayout()
        hardware_layout.addWidget(QLabel("硬件"), 0, 0)
        hardware_layout.addWidget(QLabel("状态"), 0, 1)

        module_names = [
            ("camera_driver", "相机 ROS2 驱动"),
            ("robot_driver", "机械臂 ROS2 驱动"),
            ("teleop", "遥操作系统"),
            ("data_collector", "采集节点"),
            ("inference", "模型推理"),
            ("preview", "实时预览"),
        ]
        hardware_names = [
            ("joystick", "手柄设备"),
            ("realsense", "RealSense"),
            ("oakd", "OAK-D"),
            ("robot", "UR 机械臂"),
            ("gripper", "末端执行器"),
        ]

        for row, (key, title) in enumerate(module_names, start=1):
            module_layout.addWidget(QLabel(title), row, 0)
            label = QLabel("未知")
            self.module_status_labels[key] = label
            module_layout.addWidget(label, row, 1)

        for row, (key, title) in enumerate(hardware_names, start=1):
            hardware_layout.addWidget(QLabel(title), row, 0)
            label = QLabel("未知")
            self.hardware_status_labels[key] = label
            hardware_layout.addWidget(label, row, 1)

        module_group.setLayout(module_layout)
        hardware_group.setLayout(hardware_layout)
        status_container_layout.addWidget(module_group)
        status_container_layout.addWidget(hardware_group)
        status_group.setLayout(status_container_layout)
        right_layout.addWidget(status_group)

        inference_group = QGroupBox("模型推理")
        inference_group.setStyleSheet(section_style)
        inference_layout = QGridLayout()

        inference_layout.addWidget(QLabel("模型文件夹:"), 0, 0)
        self.inference_model_dir_input = QLineEdit()
        self.inference_model_dir_input.setPlaceholderText("例如: models/ddim_dec_transformer")
        self.inference_model_dir_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        inference_layout.addWidget(self.inference_model_dir_input, 0, 1, 1, 3)

        self.btn_browse_inference_model = QPushButton("选择")
        self.btn_browse_inference_model.clicked.connect(self.choose_inference_model_dir)
        inference_layout.addWidget(self.btn_browse_inference_model, 0, 4)

        self.btn_refresh_inference_options = QPushButton("刷新")
        self.btn_refresh_inference_options.clicked.connect(self.refresh_inference_options)
        inference_layout.addWidget(self.btn_refresh_inference_options, 0, 5)

        inference_layout.addWidget(QLabel("任务环境:"), 1, 0)
        self.inference_env_combo = QComboBox()
        inference_layout.addWidget(self.inference_env_combo, 1, 1, 1, 2)

        inference_layout.addWidget(QLabel("任务名称:"), 1, 3)
        self.inference_task_combo = QComboBox()
        inference_layout.addWidget(self.inference_task_combo, 1, 4, 1, 2)

        inference_layout.addWidget(QLabel("Embeddings:"), 2, 0)
        self.inference_embedding_input = QLineEdit()
        self.inference_embedding_input.setReadOnly(True)
        self.inference_embedding_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        inference_layout.addWidget(self.inference_embedding_input, 2, 1, 1, 3)

        self.btn_browse_inference_embedding = QPushButton("手动选择")
        self.btn_browse_inference_embedding.clicked.connect(self.choose_inference_embedding_path)
        inference_layout.addWidget(self.btn_browse_inference_embedding, 2, 4)

        self.btn_auto_match_embedding = QPushButton("自动匹配")
        self.btn_auto_match_embedding.clicked.connect(self.update_inference_embedding_path)
        inference_layout.addWidget(self.btn_auto_match_embedding, 2, 5)

        inference_layout.addWidget(QLabel("全局相机:"), 3, 0)
        self.inference_global_camera_combo = QComboBox()
        self.inference_global_camera_combo.addItem("realsense", "realsense")
        self.inference_global_camera_combo.addItem("oakd", "oakd")
        global_infer_index = max(
            0,
            self.inference_global_camera_combo.findData(self.gui_settings.default_global_camera_source),
        )
        self.inference_global_camera_combo.setCurrentIndex(global_infer_index)
        inference_layout.addWidget(self.inference_global_camera_combo, 3, 1)

        inference_layout.addWidget(QLabel("手部相机:"), 3, 2)
        self.inference_wrist_camera_combo = QComboBox()
        self.inference_wrist_camera_combo.addItem("oakd", "oakd")
        self.inference_wrist_camera_combo.addItem("realsense", "realsense")
        wrist_infer_index = max(
            0,
            self.inference_wrist_camera_combo.findData(self.gui_settings.default_wrist_camera_source),
        )
        self.inference_wrist_camera_combo.setCurrentIndex(wrist_infer_index)
        inference_layout.addWidget(self.inference_wrist_camera_combo, 3, 3)

        inference_layout.addWidget(QLabel("运行设备:"), 3, 4)
        self.inference_device_combo = QComboBox()
        self.inference_device_combo.addItem("auto", "auto")
        self.inference_device_combo.addItem("cuda", "cuda")
        self.inference_device_combo.addItem("cpu", "cpu")
        self.inference_device_combo.setCurrentIndex(max(0, self.inference_device_combo.findData("cuda")))
        inference_layout.addWidget(self.inference_device_combo, 3, 5)

        inference_layout.addWidget(QLabel("推理频率(Hz):"), 4, 0)
        self.inference_hz_spin = QDoubleSpinBox()
        self.inference_hz_spin.setRange(0.2, 30.0)
        self.inference_hz_spin.setDecimals(1)
        self.inference_hz_spin.setSingleStep(0.5)
        self.inference_hz_spin.setValue(10.0)
        inference_layout.addWidget(self.inference_hz_spin, 4, 1)

        inference_layout.addWidget(QLabel("状态:"), 4, 2)
        self.lbl_inference_status = QLabel("未启动")
        self.lbl_inference_status.setWordWrap(True)
        self.lbl_inference_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_inference_status.setMaximumWidth(260)
        self.lbl_inference_status.setStyleSheet("font-weight: bold; color: #6c757d;")
        inference_layout.addWidget(self.lbl_inference_status, 4, 3, 1, 3)

        self.btn_inference = QPushButton("启动推理")
        self.btn_inference.setCheckable(True)
        self.btn_inference.clicked.connect(self.toggle_inference)
        inference_layout.addWidget(self.btn_inference, 5, 0)

        self.btn_execute_inference = QPushButton("开始执行任务")
        self.btn_execute_inference.setCheckable(True)
        self.btn_execute_inference.setEnabled(False)
        self.btn_execute_inference.clicked.connect(self.toggle_inference_execution)
        inference_layout.addWidget(self.btn_execute_inference, 5, 1)

        self.btn_inference_estop = QPushButton("急停")
        self.btn_inference_estop.setEnabled(False)
        self.btn_inference_estop.setStyleSheet("background-color: #ffdddd; color: #c92a2a; font-weight: bold;")
        self.btn_inference_estop.clicked.connect(self.emergency_stop_inference_execution)
        inference_layout.addWidget(self.btn_inference_estop, 5, 2)

        inference_layout.addWidget(QLabel("执行:"), 5, 3)
        self.lbl_inference_execute_status = QLabel("未使能")
        self.lbl_inference_execute_status.setStyleSheet("font-weight: bold; color: #6c757d;")
        inference_layout.addWidget(self.lbl_inference_execute_status, 5, 4, 1, 2)

        self.inference_hint_label = QLabel("说明: 直接调用 Real_IL 的 RealRobotPolicy, UI 负责相机采集、预览发布、推理调度和动作下发。")
        self.inference_hint_label.setWordWrap(True)
        self.inference_hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.inference_hint_label.setStyleSheet("color: #555; font-size: 12px;")
        inference_layout.addWidget(self.inference_hint_label, 6, 0, 1, 6)

        inference_layout.addWidget(QLabel("动作输出:"), 7, 0)
        self.inference_action_output = QTextEdit()
        self.inference_action_output.setReadOnly(True)
        self.inference_action_output.setMinimumHeight(120)
        self.inference_action_output.setMaximumHeight(140)
        inference_layout.addWidget(self.inference_action_output, 8, 0, 1, 6)

        inference_group.setLayout(inference_layout)
        right_layout.addWidget(inference_group)

        record_group = QGroupBox("数据录制")
        record_group.setStyleSheet(section_style)
        record_layout = QGridLayout()
        record_layout.addWidget(QLabel("HDF5 保存目录:"), 0, 0)
        self.record_dir_input = QLineEdit(self.gui_settings.default_hdf5_output_dir)
        self.record_dir_input.setToolTip(self.gui_settings.default_hdf5_output_dir)
        self.record_dir_input.setCursorPosition(0)
        record_layout.addWidget(self.record_dir_input, 0, 1, 1, 3)

        self.btn_choose_record_dir = QPushButton("选择目录")
        self.btn_choose_record_dir.clicked.connect(self.choose_record_output_dir)
        record_layout.addWidget(self.btn_choose_record_dir, 0, 4)

        record_layout.addWidget(QLabel("HDF5 文件名:"), 1, 0)
        self.record_name_input = QLineEdit(self.gui_settings.default_hdf5_filename)
        self.record_name_input.setToolTip(self.gui_settings.default_hdf5_filename)
        self.record_name_input.setPlaceholderText("例如: libero_demos.hdf5")
        record_layout.addWidget(self.record_name_input, 1, 1, 1, 3)

        self.btn_preview_hdf5 = QPushButton("预览已录制文件(HDF5)")
        self.btn_preview_hdf5.setStyleSheet("background-color: #e0e0e0; font-weight: bold;")
        self.btn_preview_hdf5.clicked.connect(self.open_hdf5_viewer)
        record_layout.addWidget(self.btn_preview_hdf5, 1, 4)

        self.btn_collector = QPushButton("启动采集节点")
        self.btn_collector.setCheckable(True)
        self.btn_collector.clicked.connect(self.toggle_data_collector)
        record_layout.addWidget(self.btn_collector, 2, 0)

        self.btn_start_record = QPushButton("开始录制")
        self.btn_start_record.setStyleSheet("color: red; font-weight: bold;")
        self.btn_start_record.clicked.connect(self.start_record)
        record_layout.addWidget(self.btn_start_record, 2, 1)

        self.btn_stop_record = QPushButton("停止录制")
        self.btn_stop_record.clicked.connect(self.stop_record)
        record_layout.addWidget(self.btn_stop_record, 2, 2)

        self.btn_go_home = QPushButton("回 Home 点")
        self.btn_go_home.setStyleSheet("font-weight: bold; color: #d35400;")
        self.btn_go_home.clicked.connect(self.go_home)
        record_layout.addWidget(self.btn_go_home, 2, 3)

        self.btn_go_home_zone = QPushButton("回 Home Zone")
        self.btn_go_home_zone.setStyleSheet("font-weight: bold; color: #8e44ad;")
        self.btn_go_home_zone.clicked.connect(self.go_home_zone)
        record_layout.addWidget(self.btn_go_home_zone, 2, 4)

        self.btn_set_home_current = QPushButton("设当前姿态为 Home")
        self.btn_set_home_current.setStyleSheet("font-weight: bold; color: #1e8449;")
        self.btn_set_home_current.clicked.connect(self.set_home_from_current)
        record_layout.addWidget(self.btn_set_home_current, 3, 4)

        record_layout.addWidget(QLabel("录制全局相机源:"), 3, 0)
        self.global_camera_source_combo = QComboBox()
        self.global_camera_source_combo.addItem("realsense", "realsense")
        self.global_camera_source_combo.addItem("oakd", "oakd")
        global_camera_index = max(0, self.global_camera_source_combo.findData(self.gui_settings.default_global_camera_source))
        self.global_camera_source_combo.setCurrentIndex(global_camera_index)
        record_layout.addWidget(self.global_camera_source_combo, 3, 1)

        record_layout.addWidget(QLabel("录制手部相机源:"), 3, 2)
        self.wrist_camera_source_combo = QComboBox()
        self.wrist_camera_source_combo.addItem("oakd", "oakd")
        self.wrist_camera_source_combo.addItem("realsense", "realsense")
        wrist_camera_index = max(0, self.wrist_camera_source_combo.findData(self.gui_settings.default_wrist_camera_source))
        self.wrist_camera_source_combo.setCurrentIndex(wrist_camera_index)
        record_layout.addWidget(self.wrist_camera_source_combo, 3, 3)

        record_layout.addWidget(QLabel("当前录制序列:"), 4, 0)
        self.lbl_demo_status = QLabel("无 (未录制)")
        self.lbl_demo_status.setStyleSheet("color: blue; font-weight: bold;")
        record_layout.addWidget(self.lbl_demo_status, 4, 1)

        self.lbl_main_record_stats = QLabel("录制时长: 00:00 | 帧数: 0")
        self.lbl_main_record_stats.setStyleSheet("font-weight: bold; color: #555;")
        record_layout.addWidget(self.lbl_main_record_stats, 4, 2, 1, 3)

        record_group.setLayout(record_layout)
        left_layout.addWidget(record_group)

        preview_group = QGroupBox("监视器与日志")
        preview_group.setStyleSheet(section_style)
        preview_layout = QVBoxLayout()
        self.btn_preview = QPushButton("打开实时预览与状态窗")
        self.btn_preview.clicked.connect(self.open_preview_window)
        preview_layout.addWidget(self.btn_preview)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        preview_layout.addWidget(self.log_output)
        preview_group.setLayout(preview_layout)
        left_layout.addWidget(preview_group, 1)
        right_layout.addStretch(1)

        self._apply_persisted_home_to_ui_log()
        self.refresh_mediapipe_topics(log_result=False)
        self.mode_combo.currentIndexChanged.connect(self._update_input_hint)
        self.mode_combo.currentIndexChanged.connect(self._update_input_mode_widgets)
        self.mode_combo.currentIndexChanged.connect(self._refresh_runtime_status)
        self.joy_profile_combo.currentIndexChanged.connect(self._update_input_hint)
        self.joy_profile_combo.currentIndexChanged.connect(self._refresh_runtime_status)
        self.mediapipe_topic_combo.currentTextChanged.connect(self._update_input_hint)
        self.mediapipe_topic_combo.currentTextChanged.connect(self._refresh_runtime_status)
        self.camera_driver_combo.currentIndexChanged.connect(self._refresh_runtime_status)
        self.global_camera_source_combo.currentIndexChanged.connect(self._refresh_runtime_status)
        self.wrist_camera_source_combo.currentIndexChanged.connect(self._refresh_runtime_status)
        self.inference_env_combo.currentIndexChanged.connect(self.refresh_inference_task_names)
        self.inference_task_combo.currentTextChanged.connect(self._notify_running_inference_goal_changed)
        self.inference_model_dir_input.textChanged.connect(self._sync_inference_model_dir_tooltip)
        self.ee_combo.currentIndexChanged.connect(self._refresh_runtime_status)
        self.ee_combo.currentIndexChanged.connect(self._sync_ros_worker_inference_control_config)
        self.ur_type_input.textChanged.connect(self._refresh_runtime_status)
        self.refresh_inference_options()
        self._update_input_hint()
        self._update_input_mode_widgets()

    def _save_home_to_gui_params(self, joints: List[float]) -> Optional[Path]:
        if len(joints) != 6:
            return None

        try:
            path = save_gui_settings_overrides(
                __file__,
                {
                    "home_joint_positions": [float(value) for value in joints],
                    "ur_type": self._selected_ur_type(),
                },
            )
            self.gui_settings = load_gui_settings(__file__)
            self._sync_ros_worker_home_config()
            return path
        except Exception as exc:
            self.log(f"写入 GUI 配置失败: {exc}")
            return None

    def _apply_persisted_home_to_ui_log(self) -> None:
        if len(self.gui_settings.home_joint_positions) == 6:
            joints_str = np.array2string(
                np.array(self.gui_settings.home_joint_positions),
                formatter={"float_kind": lambda value: f"{value:6.3f}"},
            )
            self.log(f"已加载 GUI 配置中的 Home 点: {joints_str}")

    def _sync_ros_worker_home_config(self) -> None:
        if self.ros_worker is None:
            return
        self.ros_worker.set_home_config(
            joint_positions=self.gui_settings.home_joint_positions,
            duration_sec=3.0,
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            trajectory_topic="/scaled_joint_trajectory_controller/joint_trajectory",
            teleop_controller="forward_position_controller",
            trajectory_controller="scaled_joint_trajectory_controller",
        )

    def _sync_ros_worker_inference_control_config(self) -> None:
        if self.ros_worker is None:
            return
        self.ros_worker.set_inference_control_config(
            gripper_type=self._selected_gripper_type(),
            control_hz=50.0,
        )

    def _selected_input_type(self) -> str:
        value = self.mode_combo.currentData()
        return str(value).strip().lower() if value is not None else "joy"

    def _selected_joy_profile(self) -> str:
        value = self.joy_profile_combo.currentData()
        return str(value).strip().lower() if value is not None else self.gui_settings.default_joy_profile

    def _selected_ur_type(self) -> str:
        return self.ur_type_input.text().strip() or self.gui_settings.ur_type or "ur5"

    def _selected_mediapipe_topic(self) -> str:
        return self.mediapipe_topic_combo.currentText().strip() or self.gui_settings.default_mediapipe_input_topic

    def _selected_record_output_dir(self) -> str:
        return self.record_dir_input.text().strip() or self.gui_settings.default_hdf5_output_dir

    def _selected_record_filename(self) -> str:
        filename = self.record_name_input.text().strip() or self.gui_settings.default_hdf5_filename
        if "." not in Path(filename).name:
            filename = f"{filename}.hdf5"
        return filename

    def _selected_record_output_path(self) -> str:
        output_dir = Path(self._selected_record_output_dir()).expanduser()
        filename = self._selected_record_filename()
        return str((output_dir / filename).resolve())

    def _default_mediapipe_topics(self) -> List[str]:
        return [
            self.gui_settings.default_mediapipe_input_topic,
            "/camera/camera/color/image_raw",
            "/camera/color/image_raw",
            "/color/video/image",
        ]

    def _set_combo_items_unique(self, combo: QComboBox, values: List[str], preferred: str) -> None:
        seen = set()
        unique_values = []
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique_values.append(text)

        combo.blockSignals(True)
        combo.clear()
        for value in unique_values:
            combo.addItem(value, value)
        combo.setCurrentText(preferred if preferred.strip() else (unique_values[0] if unique_values else ""))
        combo.blockSignals(False)

    def refresh_mediapipe_topics(self, log_result: bool = True) -> None:
        current_value = self._selected_mediapipe_topic()
        topics = list(self._default_mediapipe_topics())

        try:
            result = subprocess.run(
                ["ros2", "topic", "list", "-t"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if "sensor_msgs/msg/Image" not in line:
                        continue
                    topic_name = line.split()[0]
                    topics.append(topic_name)
        except Exception as exc:
            if log_result:
                self.log(f"刷新手势识别输入话题失败: {exc}")

        preferred = current_value or self.gui_settings.default_mediapipe_input_topic
        self._set_combo_items_unique(self.mediapipe_topic_combo, topics + [preferred], preferred)

        if log_result:
            self.log(f"已刷新手势识别输入话题，共 {self.mediapipe_topic_combo.count()} 个候选项。")
        self._update_input_hint()

    def _preview_window_visible(self) -> bool:
        return bool(self.preview_window is not None and self.preview_window.isVisible())

    def _collector_preview_api_should_run(self) -> bool:
        return self._preview_window_visible() and self._process_running("data_collector")

    def _inference_preview_should_render(self) -> bool:
        return self._preview_window_visible() and self._inference_running() and not self._process_running("data_collector")

    def _selected_reverse_ip(self) -> str:
        configured_ip = self.gui_settings.default_reverse_ip.strip()
        if configured_ip and configured_ip.lower() not in {"auto", "unknown"}:
            return configured_ip

        detected_ip = self.local_ip_label.text().strip()
        if detected_ip and detected_ip.lower() != "unknown":
            return detected_ip

        return "192.168.1.10"

    def _selected_gripper_type(self) -> str:
        value = self.ee_combo.currentData()
        return str(value).strip().lower() if value is not None else "robotiq"

    def _selected_collector_end_effector_type(self) -> str:
        return "qbsofthand" if self._selected_gripper_type() == "qbsofthand" else "robotic_gripper"

    def _selected_camera_source(self, combo: QComboBox, fallback: str) -> str:
        value = combo.currentData()
        return str(value).strip().lower() if value is not None else fallback

    def _selected_camera_driver(self) -> str:
        value = self.camera_driver_combo.currentData()
        return str(value).strip().lower() if value is not None else self.gui_settings.default_camera_driver

    def _selected_inference_model_dir(self) -> str:
        return self.inference_model_dir_input.text().strip()

    def _selected_inference_env(self) -> str:
        value = self.inference_env_combo.currentData()
        if value is not None:
            return str(value).strip()
        return self.inference_env_combo.currentText().strip()

    def _selected_inference_task_name(self) -> str:
        value = self.inference_task_combo.currentData()
        if value is None:
            value = self.inference_task_combo.currentText()
        return str(value).strip()

    def _selected_inference_device(self) -> Optional[str]:
        value = self.inference_device_combo.currentData()
        normalized = str(value).strip().lower() if value is not None else "auto"
        return None if normalized == "auto" else normalized

    def _selected_inference_camera_source(self, combo: QComboBox, fallback: str) -> str:
        value = combo.currentData()
        return str(value).strip().lower() if value is not None else fallback

    def _sync_inference_model_dir_tooltip(self) -> None:
        model_dir = self._selected_inference_model_dir()
        self.inference_model_dir_input.setToolTip(model_dir)
        self.inference_model_dir_input.setCursorPosition(0)

    def choose_inference_model_dir(self) -> None:
        current_dir = self._selected_inference_model_dir() or str(MODELS_ROOT.resolve())
        selected_dir = QFileDialog.getExistingDirectory(self, "选择模型文件夹", current_dir)
        if not selected_dir:
            return
        normalized_dir = str(Path(selected_dir).expanduser().resolve())
        self.inference_model_dir_input.setText(normalized_dir)
        self._sync_inference_model_dir_tooltip()
        self.log(f"推理模型文件夹已更新为: {normalized_dir}")

    def choose_inference_embedding_path(self) -> None:
        start_dir = self.inference_embedding_input.text().strip() or str(TASK_EMBEDDINGS_ROOT.resolve())
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 task embedding 文件",
            start_dir,
            "Pickle Files (*.pkl)",
        )
        if not selected_path:
            return
        normalized_path = str(Path(selected_path).expanduser().resolve())
        self.inference_embedding_input.setText(normalized_path)
        self.inference_embedding_input.setToolTip(normalized_path)
        self.inference_embedding_input.setCursorPosition(0)
        self._populate_inference_tasks_from_embedding(Path(normalized_path))
        self._notify_running_inference_goal_changed()
        self.log(f"推理 embedding 文件已手动指定为: {normalized_path}")

    def refresh_inference_options(self) -> None:
        checkpoint_dirs = discover_checkpoint_dirs()
        current_model_dir = self._selected_inference_model_dir()
        preferred_model_dir = current_model_dir
        if not preferred_model_dir and checkpoint_dirs:
            preferred_model_dir = str(checkpoint_dirs[0])
        if preferred_model_dir:
            self.inference_model_dir_input.setText(preferred_model_dir)
            self._sync_inference_model_dir_tooltip()

        envs = discover_task_envs()
        preferred_env = self._selected_inference_env() if self.inference_env_combo.count() > 0 else None
        if not preferred_env:
            preferred_env = envs[0] if envs else ""
        self._set_combo_items_unique(self.inference_env_combo, envs, preferred_env)
        self.refresh_inference_task_names()

        if checkpoint_dirs:
            self.log(
                "已扫描模型目录: "
                + ", ".join(str(path) for path in checkpoint_dirs)
            )
        else:
            self.log("未扫描到可用模型目录，请手动选择包含 last_model.pth 的文件夹。")

    def _populate_inference_tasks_from_embedding(self, embedding_path: Optional[Path]) -> None:
        task_names = sorted(load_embedding_keys(embedding_path)) if embedding_path is not None else []
        current_task = self._selected_inference_task_name()
        preferred_task = current_task if current_task in task_names else (task_names[0] if task_names else "")

        self.inference_task_combo.blockSignals(True)
        self.inference_task_combo.clear()
        for task_name in task_names:
            self.inference_task_combo.addItem(task_name, task_name)
        if preferred_task:
            self.inference_task_combo.setCurrentText(preferred_task)
        self.inference_task_combo.blockSignals(False)

    def refresh_inference_task_names(self) -> None:
        env_name = self._selected_inference_env()
        embedding_path = guess_embedding_path(env_name=env_name) if env_name else None
        text = str(embedding_path) if embedding_path is not None else ""
        self.inference_embedding_input.setText(text)
        self.inference_embedding_input.setToolTip(text)
        self.inference_embedding_input.setCursorPosition(0)
        self._populate_inference_tasks_from_embedding(embedding_path)
        self._notify_running_inference_goal_changed()

    def update_inference_embedding_path(self) -> None:
        env_name = self._selected_inference_env()
        embedding_path = guess_embedding_path(env_name=env_name) if env_name else None
        text = str(embedding_path) if embedding_path is not None else ""
        self.inference_embedding_input.setText(text)
        self.inference_embedding_input.setToolTip(text)
        self.inference_embedding_input.setCursorPosition(0)
        self._populate_inference_tasks_from_embedding(embedding_path)
        self._notify_running_inference_goal_changed()

    def _notify_running_inference_goal_changed(self, *_args) -> None:
        worker = self.inference_worker
        if worker is None or not worker.isRunning():
            return

        task_name = self._selected_inference_task_name()
        embedding_path_text = self.inference_embedding_input.text().strip()
        if not task_name or not embedding_path_text:
            return

        embedding_path = Path(embedding_path_text).expanduser().resolve()
        if not embedding_path.is_file():
            self.log(f"推理任务切换已忽略: embedding 文件不存在 {embedding_path}")
            return

        if task_name not in load_embedding_keys(embedding_path):
            self.log(f"推理任务切换已忽略: 任务 `{task_name}` 不在 embedding 文件 {embedding_path} 中")
            return

        worker.request_task_update(
            task_name=task_name,
            task_embedding_path=str(embedding_path),
        )

    def _set_inference_status(self, text: str, color: str) -> None:
        self.lbl_inference_status.setText(text)
        self.lbl_inference_status.setStyleSheet(f"font-weight: bold; color: {color};")

    def _set_inference_button_running(self, running: bool) -> None:
        self._set_button_running(
            self.btn_inference,
            running,
            "启动推理",
            "停止推理",
            "background-color: lightgreen; font-weight: bold;",
        )

    def _set_inference_execute_button_running(self, running: bool) -> None:
        self._set_button_running(
            self.btn_execute_inference,
            running,
            "开始执行任务",
            "停止执行任务",
            "background-color: #d9f2d9; font-weight: bold;",
        )

    def _set_inference_execute_status(self, text: str, color: str) -> None:
        self.lbl_inference_execute_status.setText(text)
        self.lbl_inference_execute_status.setStyleSheet(f"font-weight: bold; color: {color};")

    def _inference_running(self) -> bool:
        return self.inference_worker is not None and self.inference_worker.isRunning()

    def _inference_execution_running(self) -> bool:
        if self.ros_worker is not None and self.ros_worker.inference_execution_enabled():
            return True
        return self.btn_execute_inference.isChecked()

    def _inference_camera_occupancy(self) -> dict[str, bool]:
        if not self._inference_running():
            return {"realsense": False, "oakd": False}

        global_source = self._selected_inference_camera_source(
            self.inference_global_camera_combo,
            self.gui_settings.default_global_camera_source,
        )
        wrist_source = self._selected_inference_camera_source(
            self.inference_wrist_camera_combo,
            self.gui_settings.default_wrist_camera_source,
        )
        selected_sources = {global_source, wrist_source}
        return {
            "realsense": "realsense" in selected_sources,
            "oakd": "oakd" in selected_sources,
        }

    def _ros_worker_required(self) -> bool:
        preview_running = bool(self.preview_window is not None and self.preview_window.isVisible())
        return preview_running or self._process_running("data_collector") or self._inference_running()

    def _stop_ros_worker_if_unused(self) -> None:
        if self.ros_worker is None or self._ros_worker_required():
            return
        self.ros_worker.stop()
        self.ros_worker = None

    def _start_preview_api_worker(self) -> None:
        if not self._collector_preview_api_should_run():
            return
        if self.preview_api_worker is not None:
            self.preview_api_worker.set_base_url(self.COLLECTOR_PREVIEW_API_BASE_URL)
            return

        self.preview_api_worker = HttpPreviewWorker(self.COLLECTOR_PREVIEW_API_BASE_URL, fps=30.0)
        self.preview_api_worker.availability_signal.connect(self._on_preview_api_availability_changed)
        self.preview_api_worker.log_signal.connect(self.log)
        if self.preview_window is not None:
            self.preview_api_worker.global_image_signal.connect(self.preview_window.update_global_image)
            self.preview_api_worker.wrist_image_signal.connect(self.preview_window.update_wrist_image)
        self.preview_api_worker.start()

    def _stop_preview_api_worker(self) -> None:
        worker = self.preview_api_worker
        self.preview_api_worker = None
        self._preview_api_active = False
        if worker is None:
            return
        worker.stop()

    @Slot(bool)
    def _on_preview_api_availability_changed(self, available: bool) -> None:
        self._preview_api_active = bool(available)
        if self.preview_window is not None:
            source = "采集节点 API" if available else "采集节点 API (未就绪)"
            self.preview_window.set_preview_source(source)
            if not available:
                self.preview_window.clear_images()

    def _cache_inference_preview_frames(self, global_bgr, wrist_bgr) -> None:
        self._latest_inference_global_bgr = None if global_bgr is None else np.ascontiguousarray(global_bgr).copy()
        self._latest_inference_wrist_bgr = None if wrist_bgr is None else np.ascontiguousarray(wrist_bgr).copy()

    def _clear_inference_preview_frames(self) -> None:
        self._latest_inference_global_bgr = None
        self._latest_inference_wrist_bgr = None

    def _render_inference_preview_frames(self) -> None:
        if not self._inference_preview_should_render() or self.preview_window is None:
            return
        if self._latest_inference_global_bgr is not None:
            self.preview_window.update_global_image(self._latest_inference_global_bgr)
        if self._latest_inference_wrist_bgr is not None:
            self.preview_window.update_wrist_image(self._latest_inference_wrist_bgr)

    def _ensure_ros_worker_for_inference(self) -> None:
        if self.ros_worker is not None:
            return
        self.start_ros_worker()
        self.log("已启动 ROS 监听器，用于给推理读取机器人状态。")

    def _sync_preview_pipeline(self) -> None:
        if not self._preview_window_visible():
            self._stop_preview_api_worker()
            return

        if self._collector_preview_api_should_run():
            self._start_preview_api_worker()
            if self.preview_window is not None:
                source = "采集节点 API" if self._preview_api_active else "采集节点 API (连接中)"
                self.preview_window.set_preview_source(source)
                if not self._preview_api_active:
                    self.preview_window.clear_images()
            return

        self._stop_preview_api_worker()
        if self._inference_preview_should_render():
            if self.preview_window is not None:
                self.preview_window.set_preview_source("推理直连")
                if self._latest_inference_global_bgr is None and self._latest_inference_wrist_bgr is None:
                    self.preview_window.clear_images()
            self._render_inference_preview_frames()
            return

        if self.preview_window is not None:
            self.preview_window.set_preview_source("无活动图像源")
            self.preview_window.clear_images()

    def _current_robot_state_for_inference(self) -> Optional[np.ndarray]:
        if self.ros_worker is None:
            return None
        return build_robot_state_vector(self.ros_worker.robot_state)

    def _inference_camera_conflicts(self, global_source: str, wrist_source: str) -> List[str]:
        conflicts: List[str] = []
        if global_source == wrist_source:
            conflicts.append("当前推理需要两路不同相机视角，请不要把全局相机和手部相机设置成同一源。")

        selected_sources = {global_source, wrist_source}
        if self._process_running("data_collector"):
            occupancy = collector_camera_occupancy(
                self._selected_camera_source(self.global_camera_source_combo, self.gui_settings.default_global_camera_source),
                self._selected_camera_source(self.wrist_camera_source_combo, self.gui_settings.default_wrist_camera_source),
            )
            for source in sorted(selected_sources):
                if occupancy.get(source, False):
                    conflicts.append(f"采集节点正在占用 {source}")

        for source in sorted(selected_sources):
            if self._camera_driver_running(source):
                conflicts.append(f"ROS2 相机驱动正在占用 {source}")

        return conflicts

    def _process_running(self, key: str) -> bool:
        proc = self.processes.get(key)
        return proc is not None and proc.poll() is None

    def _camera_driver_running(self, camera_name: str) -> bool:
        return self._process_running(f"camera_driver_{camera_name}")

    def _active_camera_drivers(self) -> List[str]:
        active = []
        for camera_name in ("realsense", "oakd"):
            if self._camera_driver_running(camera_name):
                active.append(camera_name)
        return active

    def _set_status_label(self, label: QLabel, text: str, color: str) -> None:
        label.setText(text)
        label.setStyleSheet(f"font-weight: bold; color: {color};")

    def _update_input_mode_widgets(self) -> None:
        is_joy = self._selected_input_type() == "joy"
        self.joy_profile_combo.setEnabled(is_joy)
        self.mediapipe_topic_combo.setEnabled(not is_joy)

    def _refresh_runtime_status(self) -> None:
        self.local_ip_label.setText(get_local_ip())

        teleop_running = self._process_running("teleop")
        robot_driver_running = teleop_running or self._process_running("robot_driver")
        collector_running = self._process_running("data_collector")
        preview_running = bool(self.preview_window is not None and self.preview_window.isVisible())
        active_camera_drivers = self._active_camera_drivers()
        collector_usage = collector_camera_occupancy(
            self._selected_camera_source(self.global_camera_source_combo, self.gui_settings.default_global_camera_source),
            self._selected_camera_source(self.wrist_camera_source_combo, self.gui_settings.default_wrist_camera_source),
        )
        inference_usage = self._inference_camera_occupancy()

        if not active_camera_drivers:
            self._set_status_label(self.module_status_labels["camera_driver"], "未启动", "#6c757d")
        else:
            joined = " / ".join(active_camera_drivers)
            self._set_status_label(self.module_status_labels["camera_driver"], f"运行中 ({joined})", "#2b8a3e")

        if teleop_running:
            self._set_status_label(self.module_status_labels["robot_driver"], "由遥操作系统托管", "#e67700")
        elif self._process_running("robot_driver"):
            self._set_status_label(self.module_status_labels["robot_driver"], "独立运行中", "#2b8a3e")
        else:
            self._set_status_label(self.module_status_labels["robot_driver"], "未启动", "#6c757d")

        self._set_status_label(self.module_status_labels["teleop"], "运行中" if teleop_running else "未启动", "#2b8a3e" if teleop_running else "#6c757d")
        self._set_status_label(self.module_status_labels["data_collector"], "运行中" if collector_running else "未启动", "#2b8a3e" if collector_running else "#6c757d")
        inference_text, inference_color = self._inference_module_summary()
        self._set_status_label(self.module_status_labels["inference"], inference_text, inference_color)
        self._set_status_label(self.module_status_labels["preview"], "打开" if preview_running else "关闭", "#2b8a3e" if preview_running else "#6c757d")

        joy_devices = detect_joystick_devices()
        if joy_devices:
            joy_text = joy_devices[0]
            if teleop_running and self._selected_input_type() == "joy":
                joy_text = f"被遥操作占用: {joy_text}"
            self._set_status_label(self.hardware_status_labels["joystick"], joy_text, "#2b8a3e")
        else:
            self._set_status_label(self.hardware_status_labels["joystick"], "未检测到", "#c92a2a")

        if self._inference_running() and inference_usage["realsense"]:
            self._set_status_label(self.hardware_status_labels["realsense"], "推理模块占用", "#e67700")
        elif collector_running and collector_usage["realsense"]:
            self._set_status_label(self.hardware_status_labels["realsense"], "采集节点占用", "#e67700")
        elif "realsense" in active_camera_drivers:
            self._set_status_label(self.hardware_status_labels["realsense"], "ROS2 驱动占用", "#2b8a3e")
        else:
            self._set_status_label(self.hardware_status_labels["realsense"], "空闲", "#6c757d")

        if self._inference_running() and inference_usage["oakd"]:
            self._set_status_label(self.hardware_status_labels["oakd"], "推理模块占用", "#e67700")
        elif collector_running and collector_usage["oakd"]:
            self._set_status_label(self.hardware_status_labels["oakd"], "采集节点占用", "#e67700")
        elif "oakd" in active_camera_drivers:
            self._set_status_label(self.hardware_status_labels["oakd"], "ROS2 驱动占用", "#2b8a3e")
        else:
            self._set_status_label(self.hardware_status_labels["oakd"], "空闲", "#6c757d")

        if teleop_running:
            self._set_status_label(self.hardware_status_labels["robot"], "遥操作系统占用", "#2b8a3e")
        elif self._inference_execution_running():
            self._set_status_label(self.hardware_status_labels["robot"], "推理执行中", "#e67700")
        elif robot_driver_running:
            self._set_status_label(self.hardware_status_labels["robot"], "机械臂驱动占用", "#2b8a3e")
        else:
            self._set_status_label(self.hardware_status_labels["robot"], "空闲", "#6c757d")

        if teleop_running:
            self._set_status_label(self.hardware_status_labels["gripper"], f"{self._selected_gripper_type()} 驱动运行中", "#2b8a3e")
        elif self._inference_execution_running():
            self._set_status_label(self.hardware_status_labels["gripper"], f"{self._selected_gripper_type()} 推理执行", "#e67700")
        else:
            self._set_status_label(self.hardware_status_labels["gripper"], "空闲", "#6c757d")

        self.btn_robot_driver.setEnabled(not teleop_running)
        self.btn_robot_driver.setToolTip("遥操作系统运行时，机械臂驱动由 teleop 统一托管，不能单独关闭。" if teleop_running else "")
        self.btn_teleop.setEnabled(not self._inference_execution_running())
        self.btn_teleop.setToolTip("推理执行中时不能启动遥操作系统，避免双重控制。" if self._inference_execution_running() else "")

        selected_driver = self._selected_camera_driver()
        if self._camera_driver_running(selected_driver):
            self._set_button_running(
                self.btn_camera_driver,
                True,
                "启动相机驱动",
                "停止所选驱动",
                "background-color: lightgreen;",
            )
            self.btn_camera_driver.setToolTip(f"当前选中的 {selected_driver} 已在运行，点击可停止该驱动。")
        else:
            self._set_button_running(self.btn_camera_driver, False, "启动相机驱动", "停止当前驱动")
            self.btn_camera_driver.setToolTip(f"点击启动所选相机驱动 {selected_driver}。")

        if preview_running:
            self._sync_preview_pipeline()

    def _inference_module_summary(self) -> tuple[str, str]:
        status_text = self.lbl_inference_status.text().strip()
        exec_text = self.lbl_inference_execute_status.text().strip()
        if self._inference_running():
            if exec_text == "已急停":
                return "急停", "#c92a2a"
            if self._inference_execution_running():
                return "执行中", "#2b8a3e"
            if status_text.startswith("运行中"):
                return "运行中", "#2b8a3e"
            return "启动中", "#e67700"
        if exec_text == "已急停":
            return "急停", "#c92a2a"
        if status_text.startswith("错误"):
            return "错误", "#c92a2a"
        if status_text and status_text != "未启动":
            return status_text, "#e67700"
        return "未启动", "#6c757d"

    def _update_input_hint(self) -> None:
        input_type = self._selected_input_type()
        if input_type == "mediapipe":
            self.input_hint_label.setText(
                f"提示: 当前 mediapipe 模式会直接订阅图像输入 `{self._selected_mediapipe_topic()}`，并在 teleop 节点内完成手部识别与夹爪控制。"
            )
            self.input_hint_label.setStyleSheet("color: #555; font-size: 12px;")
            return

        self.input_hint_label.setText(
            f"提示: Joy 模式当前使用手柄配置 `{self._selected_joy_profile()}`，控制链路会直接输出到 MoveIt Servo。"
        )
        self.input_hint_label.setStyleSheet("color: #555; font-size: 12px;")

    def _set_button_running(self, button: QPushButton, running: bool, start_text: str, stop_text: str, style: str = "") -> None:
        button.blockSignals(True)
        button.setChecked(running)
        button.blockSignals(False)
        button.setText(stop_text if running else start_text)
        button.setStyleSheet(style if running else ("font-weight: bold;" if button is self.btn_teleop else ""))

    def _handle_process_exit(self, key: str, returncode: int) -> None:
        self.log(f"进程 {key} 已退出，返回码: {returncode}")
        self.processes.pop(key, None)

        if key in {"camera_driver_realsense", "camera_driver_oakd"}:
            self._refresh_runtime_status()
            return
        if key == "teleop":
            self._set_button_running(self.btn_teleop, False, "启动遥操作系统", "停止遥操作系统")
            self._set_button_running(self.btn_robot_driver, False, "启动机械臂驱动", "停止机械臂驱动")
            self._refresh_runtime_status()
            return
        if key == "robot_driver":
            self._set_button_running(self.btn_robot_driver, False, "启动机械臂驱动", "停止机械臂驱动")
            return
        if key == "data_collector":
            self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
            self._sync_preview_pipeline()
            self._stop_ros_worker_if_unused()

    def _poll_subprocesses(self) -> None:
        for key, proc in list(self.processes.items()):
            returncode = proc.poll()
            if returncode is None:
                continue
            self._handle_process_exit(key, int(returncode))

    def log(self, message):
        self.log_output.append(message)
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def choose_record_output_dir(self):
        current_dir = self._selected_record_output_dir()
        selected_dir = QFileDialog.getExistingDirectory(self, "选择 HDF5 保存目录", current_dir)
        if selected_dir:
            normalized_dir = str(Path(selected_dir).expanduser().resolve())
            self.record_dir_input.setText(normalized_dir)
            self.record_dir_input.setToolTip(normalized_dir)
            self.record_dir_input.setCursorPosition(0)
            try:
                save_gui_settings_overrides(__file__, {"default_hdf5_output_dir": normalized_dir})
                self.gui_settings = load_gui_settings(__file__)
            except Exception as exc:
                self.log(f"保存 HDF5 默认目录失败: {exc}")
            self.log(f"HDF5 保存目录已更新为: {normalized_dir}")

    def run_subprocess(self, key, cmd_list):
        self.log(f"执行指令: {' '.join(cmd_list)}")
        try:
            proc = subprocess.Popen(cmd_list, preexec_fn=os.setsid)
            self.processes[key] = proc
            return True
        except Exception as exc:
            self.log(f"启动 {key} 失败: {exc}")
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
                    except Exception as exc:
                        self.log(f"强制终止失败: {exc}")
                except Exception as exc:
                    self.log(f"终止进程发生异常: {exc}")
            del self.processes[key]

    def toggle_camera_driver(self, checked):
        selected_driver = self._selected_camera_driver()
        process_key = f"camera_driver_{selected_driver}"

        if checked:
            inference_usage = self._inference_camera_occupancy()
            if self._inference_running() and inference_usage.get(selected_driver, False):
                QMessageBox.warning(self, "硬件占用冲突", f"推理模块正在占用 {selected_driver}")
                self._set_button_running(self.btn_camera_driver, False, "启动相机驱动", "停止所选驱动")
                return

            conflicts = hardware_conflicts_for_collector(
                selected_driver,
                self._process_running("data_collector"),
                self._selected_camera_source(self.global_camera_source_combo, self.gui_settings.default_global_camera_source),
                self._selected_camera_source(self.wrist_camera_source_combo, self.gui_settings.default_wrist_camera_source),
            )
            if conflicts:
                QMessageBox.warning(self, "硬件占用冲突", "；".join(conflicts))
                self._set_button_running(self.btn_camera_driver, False, "启动相机驱动", "停止所选驱动")
                return

            if self._camera_driver_running(selected_driver):
                self.log(f"相机驱动 {selected_driver} 已经在运行。")
                self._refresh_runtime_status()
                return

            if self.run_subprocess(process_key, build_camera_driver_command(selected_driver)):
                self._set_button_running(self.btn_camera_driver, True, "启动相机驱动", "停止所选驱动", "background-color: lightgreen;")
            else:
                self._set_button_running(self.btn_camera_driver, False, "启动相机驱动", "停止所选驱动")
        else:
            if self._camera_driver_running(selected_driver):
                self.kill_subprocess(process_key)
            self._set_button_running(self.btn_camera_driver, False, "启动相机驱动", "停止所选驱动")
        self._refresh_runtime_status()

    def toggle_robot_driver(self, checked):
        if self._process_running("teleop"):
            QMessageBox.information(self, "提示", "遥操作系统运行中时，机械臂驱动由 teleop 统一托管，不能单独操作。")
            self._set_button_running(self.btn_robot_driver, True, "启动机械臂驱动", "停止机械臂驱动", "background-color: #ffe8cc;")
            return

        if checked:
            cmd = build_robot_driver_command(
                self.ip_input.text().strip(),
                self._selected_reverse_ip(),
                self._selected_ur_type(),
                self._selected_gripper_type(),
            )
            if self.run_subprocess("robot_driver", cmd):
                self._set_button_running(self.btn_robot_driver, True, "启动机械臂驱动", "停止机械臂驱动", "background-color: lightgreen;")
            else:
                self._set_button_running(self.btn_robot_driver, False, "启动机械臂驱动", "停止机械臂驱动")
        else:
            self.kill_subprocess("robot_driver")
            self._set_button_running(self.btn_robot_driver, False, "启动机械臂驱动", "停止机械臂驱动")

    def toggle_teleop(self, checked):
        if checked:
            if self.ros_worker is not None:
                self.ros_worker.call_cancel_home_zone()
            if self._inference_execution_running():
                QMessageBox.warning(self, "控制冲突", "推理任务正在直接控制机器人，请先停止任务执行。")
                self._set_button_running(self.btn_teleop, False, "启动遥操作系统", "停止遥操作系统")
                return
            input_type = self._selected_input_type()
            ip = self.ip_input.text().strip()
            gripper_type = self._selected_gripper_type()
            joy_profile = self._selected_joy_profile()
            mediapipe_topic = self._selected_mediapipe_topic()

            if self._process_running("robot_driver"):
                self.log("检测到机械臂 ROS2 驱动已独立启动，启动遥操作前先停止独立驱动实例。")
                self.kill_subprocess("robot_driver")

            self.log(
                "准备启动遥操作系统: "
                f"input_type={input_type}, gripper_type={gripper_type}, robot_ip={ip}, "
                f"joy_profile={joy_profile}, mediapipe_input_topic={mediapipe_topic}"
            )
            cmd = build_teleop_command(
                robot_ip=ip,
                reverse_ip=self._selected_reverse_ip(),
                ur_type=self._selected_ur_type(),
                input_type=input_type,
                gripper_type=gripper_type,
                joy_profile=joy_profile,
                mediapipe_input_topic=mediapipe_topic,
            )
            if self.run_subprocess("teleop", cmd):
                self._set_button_running(self.btn_teleop, True, "启动遥操作系统", "停止遥操作系统", "background-color: lightgreen; font-weight: bold;")
            else:
                self._set_button_running(self.btn_teleop, False, "启动遥操作系统", "停止遥操作系统")
                return

            if input_type == "mediapipe":
                QMessageBox.information(self, "MediaPipe 提示", f"当前已选择 mediapipe 输入。\n\nteleop 将直接订阅图像话题 `{mediapipe_topic}` 进行手势识别，请确认对应相机驱动已启动。")

            QMessageBox.information(self, "操作提示", "遥操作系统已启动！\n\n请不要忘记按示教器的【程序运行播放键】。")
        else:
            self.kill_subprocess("teleop")
            self._set_button_running(self.btn_teleop, False, "启动遥操作系统", "停止遥操作系统")
            self._set_button_running(self.btn_robot_driver, False, "启动机械臂驱动", "停止机械臂驱动")
            self._refresh_runtime_status()

    def toggle_data_collector(self, checked):
        if checked:
            out_path = self._selected_record_output_path()
            collector_ee_type = self._selected_collector_end_effector_type()
            global_camera_source = self._selected_camera_source(self.global_camera_source_combo, "realsense")
            wrist_camera_source = self._selected_camera_source(self.wrist_camera_source_combo, self.gui_settings.default_wrist_camera_source)
            try:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                QMessageBox.warning(self, "输出路径无效", f"无法创建 HDF5 输出目录:\n{exc}")
                self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
                return
            active_camera_drivers = self._active_camera_drivers()
            conflicts: List[str] = []
            for active_camera_driver in active_camera_drivers:
                conflicts.extend(
                    hardware_conflicts_for_collector(active_camera_driver, True, global_camera_source, wrist_camera_source)
                )
            inference_usage = self._inference_camera_occupancy()
            if self._inference_running():
                for source in sorted({global_camera_source, wrist_camera_source}):
                    if inference_usage.get(source, False):
                        conflicts.append(f"推理模块正在占用 {source}")
            if conflicts:
                QMessageBox.warning(self, "相机占用冲突", "当前采集将直接占用相机 SDK，不能与同一硬件的 ROS2 驱动同时运行。\n\n" + "；".join(conflicts))
                self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
                return

            self.log(
                "准备启动采集节点: "
                f"end_effector_type={collector_ee_type}, output_path={out_path}, "
                f"global_camera_source={global_camera_source}, wrist_camera_source={wrist_camera_source}"
            )

            yaml_args = []
            try:
                result = subprocess.run(["ros2", "pkg", "prefix", "teleop_control_py"], capture_output=True, text=True)
                if result.returncode == 0:
                    pkg_path = result.stdout.strip()
                    yaml_path = Path(pkg_path) / "share/teleop_control_py/config/data_collector_params.yaml"
                    if yaml_path.exists():
                        yaml_args = ["--params-file", str(yaml_path)]
            except Exception:
                pass

            cmd = ["ros2", "run", "teleop_control_py", "data_collector_node", "--ros-args"]
            if yaml_args:
                cmd.extend(yaml_args)
            cmd.extend([
                "-p", f"output_path:={out_path}",
                "-p", f"global_camera_source:={global_camera_source}",
                "-p", f"wrist_camera_source:={wrist_camera_source}",
                "-p", f"end_effector_type:={collector_ee_type}",
                "-p", f"ur_type:={self._selected_ur_type()}",
            ])
            if len(self.gui_settings.home_joint_positions) == 6:
                joint_values = ", ".join(f"{float(value):.6f}" for value in self.gui_settings.home_joint_positions)
                cmd.extend(["-p", f"home_joint_positions:=[{joint_values}]"])
            if len(self.gui_settings.home_zone_translation_min_m) == 3:
                values = ", ".join(f"{float(value):.6f}" for value in self.gui_settings.home_zone_translation_min_m)
                cmd.extend(["-p", f"home_zone_translation_min_m:=[{values}]"])
            if len(self.gui_settings.home_zone_translation_max_m) == 3:
                values = ", ".join(f"{float(value):.6f}" for value in self.gui_settings.home_zone_translation_max_m)
                cmd.extend(["-p", f"home_zone_translation_max_m:=[{values}]"])
            if len(self.gui_settings.home_zone_rotation_min_deg) == 3:
                values = ", ".join(f"{float(value):.6f}" for value in self.gui_settings.home_zone_rotation_min_deg)
                cmd.extend(["-p", f"home_zone_rotation_min_deg:=[{values}]"])
            if len(self.gui_settings.home_zone_rotation_max_deg) == 3:
                values = ", ".join(f"{float(value):.6f}" for value in self.gui_settings.home_zone_rotation_max_deg)
                cmd.extend(["-p", f"home_zone_rotation_max_deg:=[{values}]"])

            if not self.run_subprocess("data_collector", cmd):
                self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
                return

            self._set_button_running(self.btn_collector, True, "启动采集节点", "停止采集节点", "background-color: lightgreen;")
            self.start_ros_worker()
            self._sync_preview_pipeline()
        else:
            self.kill_subprocess("data_collector")
            self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
            self._stop_ros_worker_if_unused()

    def start_ros_worker(self):
        if self.ros_worker is None:
            self.ros_worker = ROS2Worker(
                self.COLLECTOR_POSE_TOPIC,
                self.COLLECTOR_GRIPPER_TOPIC,
                self.COLLECTOR_ACTION_TOPIC,
            )
            self.ros_worker.log_signal.connect(self.log)
            self.ros_worker.demo_status_signal.connect(self.update_demo_status)
            self.ros_worker.record_stats_signal.connect(self.update_main_record_stats)
            self._sync_ros_worker_home_config()
            self._sync_ros_worker_inference_control_config()

            if self.preview_window:
                self.ros_worker.robot_state_str_signal.connect(self.preview_window.update_robot_state_str)
                self.ros_worker.record_stats_signal.connect(self.preview_window.update_record_stats)

            self.ros_worker.start()
            return

        self._sync_ros_worker_home_config()
        self._sync_ros_worker_inference_control_config()

    def toggle_inference(self, checked) -> None:
        if checked:
            model_dir = self._selected_inference_model_dir()
            if not model_dir:
                QMessageBox.warning(self, "模型缺失", "请先选择模型文件夹。")
                self._set_inference_button_running(False)
                return

            checkpoint_dir = Path(model_dir).expanduser().resolve()
            if not checkpoint_dir.is_dir() or not (checkpoint_dir / "last_model.pth").exists():
                QMessageBox.warning(self, "模型缺失", "所选目录不包含 last_model.pth。")
                self._set_inference_button_running(False)
                return

            task_name = self._selected_inference_task_name()
            if not task_name:
                QMessageBox.warning(self, "任务缺失", "请先选择任务名称。")
                self._set_inference_button_running(False)
                return

            embedding_path_text = self.inference_embedding_input.text().strip()
            if not embedding_path_text:
                QMessageBox.warning(self, "Embedding 缺失", "当前未匹配到 task embedding 文件。")
                self._set_inference_button_running(False)
                return

            embedding_path = Path(embedding_path_text).expanduser().resolve()
            if not embedding_path.is_file():
                QMessageBox.warning(self, "Embedding 缺失", "当前 embedding 文件不存在。")
                self._set_inference_button_running(False)
                return

            if task_name not in load_embedding_keys(embedding_path):
                QMessageBox.warning(
                    self,
                    "Embedding 不匹配",
                    f"任务 `{task_name}` 不在 embedding 文件中:\n{embedding_path}",
                )
                self._set_inference_button_running(False)
                return

            global_source = self._selected_inference_camera_source(
                self.inference_global_camera_combo,
                self.gui_settings.default_global_camera_source,
            )
            wrist_source = self._selected_inference_camera_source(
                self.inference_wrist_camera_combo,
                self.gui_settings.default_wrist_camera_source,
            )
            conflicts = self._inference_camera_conflicts(global_source, wrist_source)
            if conflicts:
                QMessageBox.warning(self, "相机占用冲突", "；".join(conflicts))
                self._set_inference_button_running(False)
                return

            self._ensure_ros_worker_for_inference()
            self.inference_worker = InferenceWorker(
                checkpoint_dir=str(checkpoint_dir),
                task_name=task_name,
                task_embedding_path=str(embedding_path),
                global_camera_source=global_source,
                wrist_camera_source=wrist_source,
                loop_hz=float(self.inference_hz_spin.value()),
                device=self._selected_inference_device(),
                state_provider=self._current_robot_state_for_inference,
                parent=self,
            )
            self.inference_worker.action_signal.connect(self.update_inference_action)
            self.inference_worker.preview_signal.connect(self.update_inference_preview)
            self.inference_worker.status_signal.connect(self.update_inference_status)
            self.inference_worker.log_signal.connect(self.log)
            self.inference_worker.error_signal.connect(self._on_inference_error)
            self.inference_worker.finished.connect(self._on_inference_finished)
            self._clear_inference_preview_frames()
            self.inference_worker.start()
            self._sync_preview_pipeline()

            self.inference_action_output.setPlainText("")
            self._set_inference_button_running(True)
            self._set_inference_status("启动中", "#e67700")
            self._set_inference_execute_button_running(False)
            self._set_inference_execute_status("未使能", "#6c757d")
            self.btn_execute_inference.setEnabled(False)
            self.btn_inference_estop.setEnabled(False)
            self._refresh_runtime_status()
            return

        self.stop_inference()

    def toggle_inference_execution(self, checked) -> None:
        if checked:
            if self.ros_worker is not None:
                self.ros_worker.call_cancel_home_zone()
            if not self._inference_running():
                QMessageBox.warning(self, "推理未运行", "请先启动推理，再开始执行任务。")
                self._set_inference_execute_button_running(False)
                return
            if self._process_running("teleop"):
                QMessageBox.warning(self, "控制冲突", "遥操作系统正在输出控制命令，请先停止遥操作系统。")
                self._set_inference_execute_button_running(False)
                return
            if not self._process_running("robot_driver"):
                QMessageBox.warning(self, "机械臂未就绪", "请先启动机械臂驱动，再执行推理任务。")
                self._set_inference_execute_button_running(False)
                return

            self._ensure_ros_worker_for_inference()
            self._sync_ros_worker_home_config()
            self._sync_ros_worker_inference_control_config()
            self.ros_worker.set_inference_execution_enabled(True)
            self._set_inference_execute_button_running(True)
            self._set_inference_execute_status("执行中", "#2b8a3e")
            self.btn_inference_estop.setEnabled(True)
            self.log("已开始执行推理任务，动作将直接发送到控制器。")
            self._refresh_runtime_status()
            return

        if self.ros_worker is not None:
            self.ros_worker.set_inference_execution_enabled(False)
        self._set_inference_execute_button_running(False)
        self._set_inference_execute_status("未使能", "#6c757d")
        self.btn_inference_estop.setEnabled(self._inference_running())
        self._refresh_runtime_status()

    def emergency_stop_inference_execution(self) -> None:
        if self.ros_worker is not None:
            self.ros_worker.emergency_stop_inference()
        self._set_inference_execute_button_running(False)
        self._set_inference_execute_status("已急停", "#c92a2a")
        self.btn_inference_estop.setEnabled(self._inference_running())
        self._refresh_runtime_status()

    def stop_inference(self) -> None:
        worker = self.inference_worker
        if worker is None:
            self._set_inference_button_running(False)
            self._set_inference_status("未启动", "#6c757d")
            self._set_inference_execute_button_running(False)
            self._set_inference_execute_status("未使能", "#6c757d")
            self.btn_execute_inference.setEnabled(False)
            self.btn_inference_estop.setEnabled(False)
            return

        self.inference_worker = None
        if self.ros_worker is not None:
            self.ros_worker.set_inference_execution_enabled(False)
        worker.stop()
        if not worker.wait(4000):
            self.log("推理线程未在 4s 内退出，继续等待资源回收。")
        self._clear_inference_preview_frames()
        self._set_inference_button_running(False)
        self._set_inference_status("未启动", "#6c757d")
        self._set_inference_execute_button_running(False)
        self._set_inference_execute_status("未使能", "#6c757d")
        self.btn_execute_inference.setEnabled(False)
        self.btn_inference_estop.setEnabled(False)
        self._refresh_runtime_status()
        self._sync_preview_pipeline()
        self._stop_ros_worker_if_unused()

    @Slot(object)
    def update_inference_action(self, action) -> None:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        self.inference_action_output.setPlainText(format_action(action_array))
        if self.ros_worker is not None:
            self.ros_worker.update_inference_action_command(action_array)

    @Slot(object, object)
    def update_inference_preview(self, global_bgr, wrist_bgr) -> None:
        self._cache_inference_preview_frames(global_bgr, wrist_bgr)
        self._render_inference_preview_frames()

    @Slot(str)
    def update_inference_status(self, status: str) -> None:
        color = "#2b8a3e" if status.startswith("运行中") else "#e67700"
        if status == "未启动":
            color = "#6c757d"
        elif status.startswith("等待"):
            color = "#e67700"
        self._set_inference_status(status, color)
        controls_ready = self._inference_running() and (status.startswith("运行中") or status == "相机已就绪")
        self.btn_execute_inference.setEnabled(controls_ready)
        self.btn_inference_estop.setEnabled(self._inference_running())
        self._sync_preview_pipeline()
        self._refresh_runtime_status()

    @Slot(str)
    def _on_inference_error(self, message: str) -> None:
        self.log(f"推理线程出错: {message}")
        self.inference_action_output.setPlainText("推理失败，详见下方日志输出。")
        if self.ros_worker is not None:
            self.ros_worker.emergency_stop_inference()
        self._clear_inference_preview_frames()
        self._set_inference_status("错误，详见日志", "#c92a2a")
        self._set_inference_execute_button_running(False)
        self._set_inference_execute_status("已急停", "#c92a2a")
        self.btn_execute_inference.setEnabled(False)
        self.btn_inference_estop.setEnabled(False)
        self._refresh_runtime_status()
        self._sync_preview_pipeline()

    @Slot()
    def _on_inference_finished(self) -> None:
        self.inference_worker = None
        if self.ros_worker is not None:
            self.ros_worker.set_inference_execution_enabled(False)
        self._clear_inference_preview_frames()
        self._set_inference_button_running(False)
        self._set_inference_execute_button_running(False)
        if self.lbl_inference_execute_status.text() != "已急停":
            self._set_inference_execute_status("未使能", "#6c757d")
        self.btn_execute_inference.setEnabled(False)
        self.btn_inference_estop.setEnabled(False)
        if self.lbl_inference_status.text() != "错误，详见日志":
            self._set_inference_status("未启动", "#6c757d")
        self._refresh_runtime_status()
        self._sync_preview_pipeline()
        self._stop_ros_worker_if_unused()

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
            if self._inference_execution_running():
                self._set_inference_execute_button_running(False)
                self._set_inference_execute_status("未使能", "#6c757d")
                self.btn_inference_estop.setEnabled(self._inference_running())
                self._refresh_runtime_status()
            self.ros_worker.call_go_home()
        else:
            QMessageBox.warning(self, "警告", "请先启动推理或采集节点，以建立 ROS 控制链后再执行 Home。")

    def go_home_zone(self):
        if self.ros_worker:
            if self._inference_execution_running():
                self._set_inference_execute_button_running(False)
                self._set_inference_execute_status("未使能", "#6c757d")
                self.btn_inference_estop.setEnabled(self._inference_running())
                self._refresh_runtime_status()
            self.ros_worker.call_go_home_zone()
        else:
            QMessageBox.warning(self, "警告", "请先启动采集节点，再执行 Home Zone。")

    def set_home_from_current(self):
        if self.ros_worker:
            joints = [float(value) for value in self.ros_worker.robot_state.get("joints", [])]
            if len(joints) != 6:
                QMessageBox.warning(self, "警告", "当前关节状态无效(需要 6 个关节角)，无法设置 Home 点。")
                return

            joints_str = np.array2string(np.array(joints), formatter={"float_kind": lambda value: f"{value:6.3f}"})
            reply = QMessageBox.question(
                self,
                "确认设置 Home 点",
                "确定将当前机械臂姿态设为新的 Home 点吗？\n\n"
                f"当前 joints:\n{joints_str}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

            self.ros_worker.call_set_home_from_current()
            saved = self._save_home_to_gui_params(joints)
            if saved:
                self.log(f"已持久化 Home 点到 GUI 配置: {saved}\nHome joints: {joints_str}")
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

    @Slot(int, str, float)
    def update_main_record_stats(self, frames, time_str, realtime_fps):
        frames_str = "N/A" if frames is None or int(frames) < 0 else str(int(frames))
        fps_text = f"{float(realtime_fps):.2f} Hz" if realtime_fps is not None else "N/A"
        self.lbl_main_record_stats.setText(f"录制时长: {time_str} | 已录制帧数: {frames_str} | 实时录制帧率: {fps_text}")

    @Slot(int)
    def _on_preview_window_finished(self, _result: int):
        self._stop_preview_api_worker()
        if self.preview_window is not None:
            self.preview_window.set_preview_source("已关闭")
            self.preview_window.clear_images()
        self._stop_ros_worker_if_unused()

    def open_preview_window(self):
        if self.ros_worker is None:
            self.start_ros_worker()
            self.log("已启动独立 ROS 监听器，用于状态与录制信息同步。")

        if self.preview_window is None:
            self.preview_window = CameraPreviewWindow(self)
            self.preview_window.finished.connect(self._on_preview_window_finished)
            self.ros_worker.robot_state_str_signal.connect(self.preview_window.update_robot_state_str)
            self.ros_worker.record_stats_signal.connect(self.preview_window.update_record_stats)
            if not self.ros_worker.is_recording:
                self.preview_window.reset_record_stats()

        self.preview_window.show()
        self.preview_window.raise_()
        self.preview_window.activateWindow()
        self._sync_preview_pipeline()

    def open_hdf5_viewer(self):
        if self.ros_worker and self.ros_worker.is_recording:
            QMessageBox.warning(self, "警告", "当前正在录制数据，为了防止文件损坏，请在【停止录制】后再进行 HDF5 预览。")
            return

        viewer = HDF5ViewerDialog(self._selected_record_output_path(), parent=self)
        viewer.exec()

    def closeEvent(self, event):
        self._shutdown()
        event.accept()
