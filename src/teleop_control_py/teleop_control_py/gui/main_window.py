from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QTimer, Slot, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
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
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from teleop_control_py.core import (
    CameraRuntimeContext,
    ControlCoordinator,
    HardwareConflictError,
    InferenceLaunchConfig,
    InferenceService,
    InferenceWorkerCallbacks,
    SystemPhase,
)
from teleop_control_py.gui.support import (
    camera_enable_depth,
    discover_sdk_cameras,
    get_local_ip,
    load_home_override,
    load_gui_settings,
    save_home_override,
    save_gui_settings_overrides,
)
from teleop_control_py.device_manager import load_robot_profile, robot_profile_name_from_ur_type
from teleop_control_py.core.inference_worker import (
    InferenceActionSample,
    MODELS_ROOT,
    TASK_EMBEDDINGS_ROOT,
    discover_checkpoint_dirs,
    discover_task_envs,
    format_action,
    guess_embedding_path,
    load_embedding_keys,
)

from .app_service import (
    CollectorLaunchConfig,
    GuiAppService,
    RobotDriverLaunchConfig,
    RosWorkerCallbacks,
    RosWorkerConfig,
    TeleopLaunchConfig,
)
from .http_preview_worker import HttpPreviewWorker
from .intent_controller import GuiIntentController, IntentResult
from .runtime_facade import GuiRuntimeFacade
from .widgets import CameraPreviewWindow, HDF5ViewerDialog


class TeleopMainWindow(QMainWindow):
    COLLECTOR_PREVIEW_API_BASE_URL = "http://127.0.0.1:8765"
    COLLECTOR_GRIPPER_TOPIC = "/gripper/cmd"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Teleop & Data Collection Station")
        self.resize(1240, 860)
        self.setMinimumSize(1240, 860)
        self.setDockNestingEnabled(True)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

        self.gui_settings = load_gui_settings(__file__)
        self.runtime_facade = GuiRuntimeFacade(
            self,
            camera_enable_depth={
                "realsense": self.gui_settings.realsense_enable_depth,
                "oakd": self.gui_settings.oakd_enable_depth,
            },
        )
        self.runtime_facade.log_signal.connect(self.log)
        self.runtime_facade.process_exited.connect(self._on_process_exit)
        self.orchestrator = ControlCoordinator()
        self.intent_controller = GuiIntentController(self.orchestrator)
        self.app_service = GuiAppService(self.runtime_facade, self.COLLECTOR_GRIPPER_TOPIC)
        self.app_service.attach_preview_callbacks(
            self._dispatch_robot_state_update,
            self._dispatch_preview_record_stats,
        )
        self.inference_service = InferenceService()
        self.preview_window = None
        self._preview_window_api_connected = False
        self.preview_api_worker = None
        self._preview_api_active = False
        self._latest_inference_global_bgr = None
        self._latest_inference_wrist_bgr = None
        self.module_status_labels = {}
        self.hardware_status_labels = {}
        self.toolbar_runtime_label = None
        self.toolbar_phase_label = None
        self.toolbar_preview_label = None
        self.vision_panel = None
        self._active_teleop_camera_source = ""
        self._active_teleop_camera_serial = ""
        self._sdk_cameras: list[dict[str, str]] = []
        self._camera_options_loaded = False
        self._suspend_gui_settings_persist = False
        self._pending_inference_execution_start = False
        self._pending_inference_action_log_kwargs: Optional[dict[str, object]] = None
        self._gui_settings_persist_connected = False
        self._gui_settings_persist_timer = QTimer(self)
        self._gui_settings_persist_timer.setSingleShot(True)
        self._gui_settings_persist_timer.timeout.connect(self._persist_gui_settings_snapshot)

        self.status_refresh_timer = QTimer(self)
        self.status_refresh_timer.timeout.connect(self._refresh_runtime_status)
        self.status_refresh_timer.start(1000)

        self.setup_ui()
        self._refresh_runtime_status()

    @property
    def inference_worker(self):
        return self.inference_service.worker

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
            self.app_service.stop_ros_worker()
        except Exception:
            pass

        try:
            self.runtime_facade.stop_all_processes()
        except Exception:
            pass

    def setup_ui(self):
        button_height = 30
        emphasis_spin_height = 30

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        self._section_style = (
            "QGroupBox { font-size: 15px; font-weight: 700; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #1f3a5f; }"
        )

        settings_group = QGroupBox("系统配置")
        settings_group.setStyleSheet(self._section_style)
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("输入后端:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("joy (手柄)", "joy")
        self.mode_combo.addItem("mediapipe (手势输入)", "mediapipe")
        self.mode_combo.addItem("quest3 (VR 控制器)", "quest3")
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

        settings_layout.addWidget(QLabel("手势识别输入相机:"), 1, 0)
        self.mediapipe_camera_combo = QComboBox()
        for option in self.gui_settings.mediapipe_camera_options:
            normalized = str(option).strip().lower() or "d435"
            self.mediapipe_camera_combo.addItem(str(option), normalized)
        mediapipe_camera_index = max(0, self.mediapipe_camera_combo.findData(self.gui_settings.default_mediapipe_camera))
        self.mediapipe_camera_combo.setCurrentIndex(mediapipe_camera_index)
        settings_layout.addWidget(self.mediapipe_camera_combo, 1, 1)

        self.mediapipe_topic_combo = QComboBox()
        self.mediapipe_topic_combo.setEditable(True)
        self.mediapipe_topic_combo.setCurrentText(self.gui_settings.default_mediapipe_input_topic)
        self.mediapipe_topic_combo.setVisible(False)
        settings_layout.addWidget(self.mediapipe_topic_combo, 1, 3, 1, 2)

        self.btn_refresh_topics = QPushButton("刷新SDK相机")
        self.btn_refresh_topics.setVisible(True)
        self.btn_refresh_topics.setMinimumHeight(button_height)
        self.btn_refresh_topics.clicked.connect(self.refresh_camera_devices)
        settings_layout.addWidget(self.btn_refresh_topics, 1, 5)

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
        self.input_hint_label.setVisible(False)

        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)

        startup_group = QGroupBox("启动节点")
        startup_group.setStyleSheet(self._section_style)
        startup_layout = QGridLayout()
        startup_layout.setContentsMargins(10, 6, 10, 10)
        startup_layout.setHorizontalSpacing(12)
        startup_layout.setVerticalSpacing(8)

        self.camera_driver_combo = QComboBox()
        for option in self.gui_settings.camera_driver_options:
            self.camera_driver_combo.addItem(option, option)
        camera_driver_index = max(0, self.camera_driver_combo.findData(self.gui_settings.default_camera_driver))
        self.camera_driver_combo.setCurrentIndex(camera_driver_index)
        self.camera_driver_combo.setVisible(False)

        self.btn_camera_driver = QPushButton("相机 ROS2 驱动（已停用）")
        self.btn_camera_driver.setCheckable(True)
        self.btn_camera_driver.setEnabled(False)
        self.btn_camera_driver.setVisible(False)
        # 暂时停用手动相机 ROS2 驱动模块，统一走相机 SDK 管理配置。
        # self.btn_camera_driver.clicked.connect(self.toggle_camera_driver)
        self.camera_module_hint_label = QLabel("相机 ROS2 驱动入口已停用。")
        self.camera_module_hint_label.setWordWrap(True)
        self.camera_module_hint_label.setStyleSheet("color: #555; font-size: 12px;")
        self.camera_module_hint_label.setVisible(False)

        startup_layout.addWidget(QLabel("机械臂 ROS2 驱动:"), 0, 0)
        self.btn_robot_driver = QPushButton("启动机械臂驱动")
        self.btn_robot_driver.setFixedHeight(button_height)
        self.btn_robot_driver.setStyleSheet(
            "QPushButton { background-color: #e8f7ef; color: #1d6f42; font-weight: bold; } "
            "QPushButton:checked { background-color: #cfeedd; }"
        )
        self.btn_robot_driver.setCheckable(True)
        self.btn_robot_driver.clicked.connect(self.toggle_robot_driver)
        startup_layout.addWidget(self.btn_robot_driver, 0, 1, 1, 2)

        startup_layout.addWidget(QLabel("遥操作系统:"), 1, 0)
        self.btn_teleop = QPushButton("启动遥操作系统")
        self.btn_teleop.setFixedHeight(button_height)
        self.btn_teleop.setStyleSheet(
            "QPushButton { background-color: #e7f0ff; color: #154c9c; font-weight: bold; } "
            "QPushButton:checked { background-color: #d4e4ff; }"
        )
        self.btn_teleop.setCheckable(True)
        self.btn_teleop.clicked.connect(self.toggle_teleop)
        startup_layout.addWidget(self.btn_teleop, 1, 1, 1, 2)

        self.startup_hint_label = QLabel("当遥操作系统启动时，会接管机械臂驱动；GUI 会显示机械臂驱动为运行中，但不允许单独关闭。")
        self.startup_hint_label.setWordWrap(True)
        self.startup_hint_label.setStyleSheet("color: #555; font-size: 12px;")
        self.startup_hint_label.setVisible(False)

        startup_group.setLayout(startup_layout)
        left_layout.addWidget(startup_group)

        system_ops_group = QGroupBox("回home操作")
        system_ops_group.setStyleSheet(self._section_style)
        system_ops_layout = QGridLayout()
        system_ops_layout.setContentsMargins(10, 6, 10, 10)
        system_ops_layout.setHorizontalSpacing(12)
        system_ops_layout.setVerticalSpacing(8)
        system_ops_layout.setColumnStretch(0, 1)
        system_ops_layout.setColumnStretch(1, 1)
        system_ops_layout.setColumnStretch(2, 1)

        self.btn_go_home = QPushButton("回 Home 点")
        self.btn_go_home.setFixedHeight(button_height)
        self.btn_go_home.setStyleSheet("font-weight: bold; color: #d35400;")
        self.btn_go_home.clicked.connect(self.go_home)
        system_ops_layout.addWidget(self.btn_go_home, 0, 0)

        self.btn_go_home_zone = QPushButton("回 Home Zone")
        self.btn_go_home_zone.setFixedHeight(button_height)
        self.btn_go_home_zone.setStyleSheet("font-weight: bold; color: #8e44ad;")
        self.btn_go_home_zone.clicked.connect(self.go_home_zone)
        system_ops_layout.addWidget(self.btn_go_home_zone, 0, 1)

        self.btn_set_home_current = QPushButton("设当前姿态为 Home")
        self.btn_set_home_current.setFixedHeight(button_height)
        self.btn_set_home_current.setStyleSheet("font-weight: bold; color: #1e8449;")
        self.btn_set_home_current.clicked.connect(self.set_home_from_current)
        system_ops_layout.addWidget(self.btn_set_home_current, 0, 2)

        system_ops_group.setLayout(system_ops_layout)
        left_layout.addWidget(system_ops_group)

        record_group = QGroupBox("数据录制")
        record_group.setStyleSheet(self._section_style)
        record_layout = QGridLayout()
        record_layout.setContentsMargins(10, 6, 10, 10)
        record_layout.setHorizontalSpacing(12)
        record_layout.setVerticalSpacing(8)
        record_layout.setColumnStretch(1, 1)
        record_layout.setColumnStretch(2, 1)
        record_layout.setColumnStretch(3, 1)
        record_layout.setColumnStretch(4, 1)
        record_layout.addWidget(QLabel("HDF5 保存目录:"), 0, 0)
        self.record_dir_input = QLineEdit(self.gui_settings.default_hdf5_output_dir)
        self.record_dir_input.setToolTip(self.gui_settings.default_hdf5_output_dir)
        self.record_dir_input.setCursorPosition(0)
        record_layout.addWidget(self.record_dir_input, 0, 1, 1, 3)

        self.btn_choose_record_dir = QPushButton("选择目录")
        self.btn_choose_record_dir.setMinimumHeight(button_height)
        self.btn_choose_record_dir.clicked.connect(self.choose_record_output_dir)
        record_layout.addWidget(self.btn_choose_record_dir, 0, 4)

        record_layout.addWidget(QLabel("HDF5 文件名:"), 1, 0)
        self.record_name_input = QLineEdit(self.gui_settings.default_hdf5_filename)
        self.record_name_input.setToolTip(self.gui_settings.default_hdf5_filename)
        self.record_name_input.setPlaceholderText("例如: libero_demos.hdf5")
        record_layout.addWidget(self.record_name_input, 1, 1, 1, 3)

        self.btn_preview_hdf5 = QPushButton("预览HDF5内容")
        self.btn_preview_hdf5.setMinimumHeight(button_height)
        self.btn_preview_hdf5.setStyleSheet(
            "QPushButton { background-color: #fff2d9; color: #9a5b00; font-weight: bold; }"
        )
        self.btn_preview_hdf5.clicked.connect(self.open_hdf5_viewer)
        record_layout.addWidget(self.btn_preview_hdf5, 1, 4)

        camera_row = QWidget()
        camera_row_layout = QHBoxLayout(camera_row)
        camera_row_layout.setContentsMargins(0, 0, 0, 0)
        camera_row_layout.setSpacing(12)

        global_camera_column = QWidget()
        global_camera_layout = QHBoxLayout(global_camera_column)
        global_camera_layout.setContentsMargins(0, 0, 0, 0)
        global_camera_layout.setSpacing(8)
        global_camera_layout.addWidget(QLabel("录制全局相机:"))
        self.global_camera_source_combo = QComboBox()
        global_camera_layout.addWidget(self.global_camera_source_combo, 1)

        wrist_camera_column = QWidget()
        wrist_camera_layout = QHBoxLayout(wrist_camera_column)
        wrist_camera_layout.setContentsMargins(0, 0, 0, 0)
        wrist_camera_layout.setSpacing(8)
        wrist_camera_layout.addWidget(QLabel("录制局部相机:"))
        self.wrist_camera_source_combo = QComboBox()
        wrist_camera_layout.addWidget(self.wrist_camera_source_combo, 1)

        camera_row_layout.addWidget(global_camera_column, 1)
        camera_row_layout.addWidget(wrist_camera_column, 1)
        record_layout.addWidget(camera_row, 2, 0, 1, 5)

        record_actions_row = QWidget()
        record_actions_layout = QHBoxLayout(record_actions_row)
        record_actions_layout.setContentsMargins(0, 0, 0, 0)
        record_actions_layout.setSpacing(12)

        self.btn_collector = QPushButton("启动采集节点")
        self.btn_collector.setFixedHeight(button_height)
        self.btn_collector.setStyleSheet(
            "QPushButton { background-color: #e7f6f4; color: #0f766e; font-weight: bold; } "
            "QPushButton:checked { background-color: #d2efeb; }"
        )
        self.btn_collector.setCheckable(True)
        self.btn_collector.clicked.connect(self.toggle_data_collector)
        record_actions_layout.addWidget(self.btn_collector, 1)

        self.btn_start_record = QPushButton("开始录制")
        self.btn_start_record.setFixedHeight(button_height)
        self.btn_start_record.setStyleSheet("color: red; font-weight: bold;")
        self.btn_start_record.clicked.connect(self.start_record)
        record_actions_layout.addWidget(self.btn_start_record, 1)

        self.btn_stop_record = QPushButton("停止录制")
        self.btn_stop_record.setFixedHeight(button_height)
        self.btn_stop_record.clicked.connect(self.stop_record)
        record_actions_layout.addWidget(self.btn_stop_record, 1)

        self.btn_discard_record = QPushButton("弃用当前 Demo")
        self.btn_discard_record.setFixedHeight(button_height)
        self.btn_discard_record.setStyleSheet("font-weight: bold; color: #b22222;")
        self.btn_discard_record.clicked.connect(self.discard_current_demo)
        record_actions_layout.addWidget(self.btn_discard_record, 1)
        record_layout.addWidget(record_actions_row, 3, 0, 1, 5)

        self.camera_binding_hint_label = QLabel("相机按型号选择，系统会自动绑定对应设备。")
        self.camera_binding_hint_label.setWordWrap(True)
        self.camera_binding_hint_label.setStyleSheet("color: #666; font-size: 12px;")
        self.camera_binding_hint_label.setVisible(False)

        record_layout.addWidget(QLabel("当前录制序列:"), 4, 0)
        self.lbl_demo_status = QLabel("无 (未录制)")
        self.lbl_demo_status.setAlignment(Qt.AlignCenter)
        self.lbl_demo_status.setFixedHeight(34)
        self.lbl_demo_status.setStyleSheet("color: blue; font-weight: bold;")
        record_layout.addWidget(self.lbl_demo_status, 4, 1)

        self.lbl_main_record_stats = QLabel("录制时长: 00:00 | 帧数: 0")
        self.lbl_main_record_stats.setAlignment(Qt.AlignCenter)
        self.lbl_main_record_stats.setFixedHeight(34)
        self.lbl_main_record_stats.setStyleSheet("font-weight: bold; color: #555;")
        record_layout.addWidget(self.lbl_main_record_stats, 4, 2, 1, 3)

        record_group.setLayout(record_layout)
        left_layout.addWidget(record_group)

        preview_group = QGroupBox("监视器与日志")
        preview_group.setStyleSheet(self._section_style)
        preview_layout = QVBoxLayout()
        self.btn_preview = QPushButton("打开实时预览与状态窗")
        self.btn_preview.setMinimumHeight(button_height)
        self.btn_preview.clicked.connect(self.open_preview_window)
        preview_layout.addWidget(self.btn_preview)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        preview_layout.addWidget(self.log_output)
        preview_group.setLayout(preview_layout)
        left_layout.addWidget(preview_group, 1)

        status_group = QGroupBox("状态总览")
        status_group.setStyleSheet(self._section_style)
        status_container_layout = QVBoxLayout()
        status_container_layout.addWidget(
            self._build_status_group(
                "模块情况",
                [
                    ("robot_driver", "机械臂 ROS2 驱动"),
                    ("teleop", "遥操作系统"),
                    ("data_collector", "采集节点"),
                    ("inference", "模型推理"),
                    ("preview", "实时预览"),
                ],
                self.module_status_labels,
            )
        )
        status_container_layout.addWidget(
            self._build_status_group(
                "硬件情况",
                [
                    ("joystick", "手柄设备"),
                    ("camera_1", "相机1"),
                    ("camera_2", "相机2"),
                    ("camera_3", "相机3"),
                    ("robot", "机械臂"),
                    ("gripper", "末端执行器"),
                ],
                self.hardware_status_labels,
            )
        )
        status_group.setLayout(status_container_layout)
        right_layout.addWidget(status_group)

        inference_group = QGroupBox("模型推理")
        inference_group.setStyleSheet(self._section_style)
        inference_layout = QGridLayout()

        inference_layout.addWidget(QLabel("模型文件夹:"), 0, 0)
        self.inference_model_dir_input = QLineEdit()
        self.inference_model_dir_input.setPlaceholderText("例如: models/ddim_dec_transformer")
        self.inference_model_dir_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        inference_layout.addWidget(self.inference_model_dir_input, 0, 1, 1, 3)

        self.btn_browse_inference_model = QPushButton("选择")
        self.btn_browse_inference_model.setMinimumHeight(button_height)
        self.btn_browse_inference_model.clicked.connect(self.choose_inference_model_dir)
        inference_layout.addWidget(self.btn_browse_inference_model, 0, 4)

        self.btn_refresh_inference_options = QPushButton("刷新")
        self.btn_refresh_inference_options.setMinimumHeight(button_height)
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
        self.btn_browse_inference_embedding.setMinimumHeight(button_height)
        self.btn_browse_inference_embedding.clicked.connect(self.choose_inference_embedding_path)
        inference_layout.addWidget(self.btn_browse_inference_embedding, 2, 4)

        self.btn_auto_match_embedding = QPushButton("自动匹配")
        self.btn_auto_match_embedding.setMinimumHeight(button_height)
        self.btn_auto_match_embedding.clicked.connect(self.update_inference_embedding_path)
        inference_layout.addWidget(self.btn_auto_match_embedding, 2, 5)

        inference_layout.addWidget(QLabel("全局相机:"), 3, 0)
        self.inference_global_camera_combo = QComboBox()
        inference_layout.addWidget(self.inference_global_camera_combo, 3, 1)

        inference_layout.addWidget(QLabel("手部相机:"), 3, 2)
        self.inference_wrist_camera_combo = QComboBox()
        inference_layout.addWidget(self.inference_wrist_camera_combo, 3, 3)

        inference_layout.addWidget(QLabel("运行设备:"), 3, 4)
        self.inference_device_combo = QComboBox()
        self.inference_device_combo.addItem("auto", "auto")
        self.inference_device_combo.addItem("cuda", "cuda")
        self.inference_device_combo.addItem("cpu", "cpu")
        self.inference_device_combo.setCurrentIndex(max(0, self.inference_device_combo.findData("cuda")))
        inference_layout.addWidget(self.inference_device_combo, 3, 5)

        inference_layout.addWidget(QLabel("高层动作频率(Hz):"), 4, 0)
        self.inference_hz_spin = QDoubleSpinBox()
        self.inference_hz_spin.setRange(0.2, 30.0)
        self.inference_hz_spin.setDecimals(1)
        self.inference_hz_spin.setSingleStep(0.5)
        self.inference_hz_spin.setValue(10.0)
        self.inference_hz_spin.setToolTip("控制高层动作输出频率；不等同于完整重规划频率。实际重规划频率约为该值 / replan_every。")
        self.inference_hz_spin.setFixedHeight(emphasis_spin_height)
        inference_layout.addWidget(self.inference_hz_spin, 4, 1)

        inference_layout.addWidget(QLabel("状态:"), 4, 2)
        self.lbl_inference_status = QLabel("未启动")
        self.lbl_inference_status.setWordWrap(True)
        self.lbl_inference_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_inference_status.setMaximumWidth(260)
        self.lbl_inference_status.setStyleSheet("font-weight: bold; color: #6c757d;")
        inference_layout.addWidget(self.lbl_inference_status, 4, 3, 1, 3)

        self.btn_inference = QPushButton("启动推理")
        self.btn_inference.setFixedHeight(button_height)
        self.btn_inference.setStyleSheet(
            "QPushButton { background-color: #f1eaff; color: #6b3fa0; font-weight: bold; } "
            "QPushButton:checked { background-color: #e3d8fb; }"
        )
        self.btn_inference.setCheckable(True)
        self.btn_inference.clicked.connect(self.toggle_inference)
        inference_layout.addWidget(self.btn_inference, 5, 0)

        self.btn_execute_inference = QPushButton("开始执行任务")
        self.btn_execute_inference.setFixedHeight(button_height)
        self.btn_execute_inference.setCheckable(True)
        self.btn_execute_inference.setEnabled(False)
        self.btn_execute_inference.clicked.connect(self.toggle_inference_execution)
        inference_layout.addWidget(self.btn_execute_inference, 5, 1)

        self.btn_inference_estop = QPushButton("急停")
        self.btn_inference_estop.setEnabled(False)
        self.btn_inference_estop.setFixedHeight(button_height)
        self.btn_inference_estop.setStyleSheet(
            "QPushButton { background-color: #f8d7da; color: #b42318; font-weight: bold; } "
            "QPushButton:disabled { background-color: #f3d1d4; color: #a45a5f; }"
        )
        self.btn_inference_estop.clicked.connect(self.emergency_stop_inference_execution)
        inference_layout.addWidget(self.btn_inference_estop, 5, 2)

        inference_layout.addWidget(QLabel("执行:"), 5, 3)
        self.lbl_inference_execute_status = QLabel("未使能")
        self.lbl_inference_execute_status.setStyleSheet("font-weight: bold; color: #6c757d;")
        inference_layout.addWidget(self.lbl_inference_execute_status, 5, 4, 1, 2)

        self.chk_collect_inference_logs = QCheckBox("记录执行段动作日志")
        self.chk_collect_inference_logs.setChecked(bool(self.gui_settings.collect_inference_action_logs))
        self.chk_collect_inference_logs.setToolTip("勾选后，仅在点击“开始执行任务”期间保存高层动作日志。")
        self.chk_collect_inference_logs.toggled.connect(self._on_collect_inference_logs_toggled)
        inference_layout.addWidget(self.chk_collect_inference_logs, 6, 0, 1, 3)

        self.inference_hint_label = QLabel(
            "说明: 直接调用 Real_IL 的 RealRobotPolicy。这里的频率表示高层动作输出频率，不等同于完整重规划频率；GUI 负责相机采集、预览发布、推理调度和动作下发。勾选上方开关后，点击开始执行任务会把执行期间的高层动作保存到 data/inference_action_logs。"
        )
        self.inference_hint_label.setWordWrap(True)
        self.inference_hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.inference_hint_label.setStyleSheet("color: #555; font-size: 12px;")
        self.inference_hint_label.setVisible(False)

        inference_layout.addWidget(QLabel("动作输出:"), 7, 0)
        self.inference_action_output = QTextEdit()
        self.inference_action_output.setReadOnly(True)
        self.inference_action_output.setMinimumHeight(120)
        self.inference_action_output.setMaximumHeight(140)
        inference_layout.addWidget(self.inference_action_output, 8, 0, 1, 6)

        inference_group.setLayout(inference_layout)
        right_layout.addWidget(inference_group)
        right_layout.addStretch(1)

        self._apply_persisted_home_to_ui_log()
        self.refresh_camera_devices(log_result=False)
        self.refresh_mediapipe_topics(log_result=False)
        self._sync_mediapipe_topic_from_camera_selection()
        self.mode_combo.currentIndexChanged.connect(self._update_input_hint)
        self.mode_combo.currentIndexChanged.connect(self._update_input_mode_widgets)
        self.mode_combo.currentIndexChanged.connect(self._refresh_runtime_status)
        self.joy_profile_combo.currentIndexChanged.connect(self._update_input_hint)
        self.joy_profile_combo.currentIndexChanged.connect(self._refresh_runtime_status)
        self.mediapipe_camera_combo.currentIndexChanged.connect(self._on_mediapipe_camera_selection_changed)
        self.mediapipe_topic_combo.currentTextChanged.connect(self._on_mediapipe_topic_changed)
        self.global_camera_source_combo.currentIndexChanged.connect(self._on_collector_camera_selection_changed)
        self.wrist_camera_source_combo.currentIndexChanged.connect(self._on_collector_camera_selection_changed)
        self.inference_global_camera_combo.currentIndexChanged.connect(self._on_inference_camera_selection_changed)
        self.inference_wrist_camera_combo.currentIndexChanged.connect(self._on_inference_camera_selection_changed)
        self.inference_env_combo.currentIndexChanged.connect(self.refresh_inference_task_names)
        self.inference_task_combo.currentIndexChanged.connect(self._on_inference_task_selection_changed)
        self.inference_model_dir_input.textChanged.connect(self._sync_inference_model_dir_tooltip)
        self.ee_combo.currentIndexChanged.connect(self._refresh_runtime_status)
        self.ee_combo.currentIndexChanged.connect(self._sync_ros_worker_inference_control_config)
        self.ur_type_input.textChanged.connect(self._refresh_runtime_status)
        self.refresh_inference_options()
        self._restore_persisted_gui_state()
        self._connect_gui_settings_persistence()
        self._update_input_hint()
        self._update_input_mode_widgets()

    def _apply_window_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #f6f8fb;
                color: #223042;
                font-family: \"Microsoft YaHei UI\", \"Noto Sans CJK SC\", \"PingFang SC\", sans-serif;
                font-size: 13px;
            }

            QDockWidget {
                font-size: 13px;
            }

            QDockWidget::title {
                text-align: left;
                background: #f7f9fc;
                color: #17324d;
                padding: 8px 12px;
                border-bottom: 1px solid #e3eaf2;
                font-weight: 700;
            }

            QScrollArea, QTabWidget::pane {
                border: none;
                background: transparent;
            }

            QLabel {
                background: transparent;
            }

            QLineEdit, QComboBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #d3dce8;
                border-radius: 10px;
                min-height: 34px;
                padding: 0 10px;
                selection-background-color: #2f6fed;
            }

            QTextEdit {
                background: #ffffff;
                border: 1px solid #d3dce8;
                border-radius: 10px;
                padding: 8px 10px;
                selection-background-color: #2f6fed;
            }

            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
                border: 1px solid #2f6fed;
            }

            QComboBox::drop-down {
                border: none;
                width: 22px;
            }

            QTabBar::tab {
                background: #eef3f8;
                color: #355070;
                border: 1px solid #d7e1ec;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-weight: 600;
            }

            QTabBar::tab:selected {
                background: #ffffff;
                color: #18324d;
                border-bottom-color: #ffffff;
            }

            QPushButton {
                background: #ffffff;
                color: #223042;
                border: 1px solid #d4dce6;
                border-radius: 10px;
                min-height: 34px;
                padding: 0 14px;
                font-weight: 600;
            }

            QPushButton:hover {
                background: #f5f8fc;
            }

            QPushButton:pressed {
                background: #ebf0f6;
            }

            QPushButton:disabled {
                background: #f2f5f8;
                color: #9aa8b8;
                border-color: #dde5ee;
            }
            """
        )

    @staticmethod
    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        value = str(hex_color).strip().lstrip("#")
        if len(value) != 6:
            return hex_color
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
        return f"rgba({red}, {green}, {blue}, {alpha:.3f})"

    def _badge_style(self, color: str, *, alpha: float = 0.12) -> str:
        background = self._hex_to_rgba(color, alpha)
        border = self._hex_to_rgba(color, min(alpha + 0.12, 0.45))
        return (
            f"color: {color}; font-weight: 700; padding: 5px 10px; "
            f"border-radius: 13px; background: {background}; border: 1px solid {border};"
        )

    def _button_badge_style(self, color: str, *, alpha: float = 0.12) -> str:
        background = self._hex_to_rgba(color, alpha)
        border = self._hex_to_rgba(color, min(alpha + 0.12, 0.45))
        return (
            f"color: {color}; font-weight: 700; padding: 0 14px; "
            f"border-radius: 10px; background: {background}; border: 1px solid {border};"
        )

    @staticmethod
    def _button_style(
        background: str,
        foreground: str,
        border: str | None = None,
        *,
        hover_background: str | None = None,
        pressed_background: str | None = None,
        checked_background: str | None = None,
        checked_foreground: str | None = None,
        checked_border: str | None = None,
    ) -> str:
        resolved_border = border or background
        resolved_hover = hover_background or background
        resolved_pressed = pressed_background or resolved_hover
        resolved_checked_background = checked_background or background
        resolved_checked_foreground = checked_foreground or foreground
        resolved_checked_border = checked_border or resolved_checked_background
        return (
            "QPushButton { "
            f"background-color: {background}; color: {foreground}; "
            f"border: 1px solid {resolved_border}; border-radius: 10px; min-height: 34px; padding: 0 14px; font-weight: 700;"
            "} "
            "QPushButton:hover { "
            f"background-color: {resolved_hover}; border-color: {resolved_border};"
            "} "
            "QPushButton:pressed { "
            f"background-color: {resolved_pressed};"
            "} "
            "QPushButton:checked { "
            f"background-color: {resolved_checked_background}; color: {resolved_checked_foreground}; "
            f"border-color: {resolved_checked_border};"
            "} "
        )

    def _apply_dock_layout_preset(self) -> None:
        self.dock_hardware.setMinimumWidth(280)
        self.dock_state.setMinimumWidth(300)
        self.dock_task.setMinimumHeight(220)

        self.resizeDocks([self.dock_hardware, self.dock_state], [300, 340], Qt.Horizontal)
        self.resizeDocks([self.dock_task], [260], Qt.Vertical)

    def _make_scroll_tab_page(self) -> tuple[QScrollArea, QWidget, QVBoxLayout]:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(content)
        return scroll, content, layout

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Global Control", self)
        toolbar.setMovable(False)
        toolbar.setStyleSheet(
            "QToolBar { padding: 10px 12px; border-bottom: 1px solid #dce5ef; spacing: 10px; background: #fbfdff; }"
        )
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        self.toolbar_runtime_label = QLabel("ROS: 初始化中")
        self.toolbar_runtime_label.setAlignment(Qt.AlignCenter)
        self.toolbar_runtime_label.setStyleSheet(self._badge_style("#1e8449", alpha=0.14))
        toolbar.addWidget(self.toolbar_runtime_label)

        self.toolbar_phase_label = QLabel("模式: IDLE")
        self.toolbar_phase_label.setAlignment(Qt.AlignCenter)
        self.toolbar_phase_label.setStyleSheet(self._badge_style("#1f3a5f", alpha=0.12))
        toolbar.addWidget(self.toolbar_phase_label)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("输入源:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("joy (手柄)", "joy")
        self.mode_combo.addItem("mediapipe (手势输入)", "mediapipe")
        self.mode_combo.addItem("quest3 (VR 控制器)", "quest3")
        default_input_index = max(0, self.mode_combo.findData(self.gui_settings.default_input_type))
        self.mode_combo.setCurrentIndex(default_input_index)
        toolbar.addWidget(self.mode_combo)

        toolbar.addWidget(QLabel("手柄配置:"))
        self.joy_profile_combo = QComboBox()
        for profile in self.gui_settings.joy_profiles:
            self.joy_profile_combo.addItem(profile, profile)
        joy_profile_index = max(0, self.joy_profile_combo.findData(self.gui_settings.default_joy_profile))
        self.joy_profile_combo.setCurrentIndex(joy_profile_index)
        toolbar.addWidget(self.joy_profile_combo)

        toolbar.addWidget(QLabel("相机驱动:"))
        self.camera_driver_combo = QComboBox()
        for option in self.gui_settings.camera_driver_options:
            self.camera_driver_combo.addItem(option, option)
        camera_driver_index = max(0, self.camera_driver_combo.findData(self.gui_settings.default_camera_driver))
        self.camera_driver_combo.setCurrentIndex(camera_driver_index)
        toolbar.addWidget(self.camera_driver_combo)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        self.toolbar_preview_label = QLabel("监视器: 内嵌视觉面板")
        self.toolbar_preview_label.setAlignment(Qt.AlignCenter)
        self.toolbar_preview_label.setStyleSheet(self._badge_style("#4c6a92", alpha=0.10))
        toolbar.addWidget(self.toolbar_preview_label)

    def _build_central_panel(self) -> None:
        self.vision_panel = StudioVisionPanel(self)
        self.setCentralWidget(self.vision_panel)

    def _make_tab_page(self) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        return page, layout

    def _build_left_dock(self) -> None:
        dock = QDockWidget("系统控制台", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        tabs = QTabWidget()
        tabs.addTab(self._build_left_config_tab(), "基础配置")
        tabs.addTab(self._build_left_driver_tab(), "驱动控制")
        tabs.addTab(self._build_left_status_tab(), "模块状态")

        dock.setWidget(tabs)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.dock_hardware = dock

    def _build_left_config_tab(self) -> QWidget:
        page, layout = self._make_tab_page()

        settings_group = QGroupBox("系统级配置")
        settings_group.setStyleSheet(self._section_style)
        settings_layout = QGridLayout(settings_group)
        settings_layout.setContentsMargins(12, 8, 12, 12)
        settings_layout.setSpacing(8)
        settings_layout.setColumnStretch(1, 3)
        settings_layout.setColumnStretch(3, 2)

        settings_layout.addWidget(QLabel("机械臂 IP:"), 0, 0)
        self.ip_input = QLineEdit(self.gui_settings.default_robot_ip)
        settings_layout.addWidget(self.ip_input, 0, 1)

        settings_layout.addWidget(QLabel("本机 IP:"), 0, 2)
        self.local_ip_label = QLabel(get_local_ip())
        self.local_ip_label.setAlignment(Qt.AlignCenter)
        self.local_ip_label.setStyleSheet(self._badge_style("#0b7285", alpha=0.10))
        settings_layout.addWidget(self.local_ip_label, 0, 3)

        settings_layout.addWidget(QLabel("UR 类型:"), 1, 0)
        self.ur_type_input = QLineEdit(self.gui_settings.ur_type or "ur5")
        self.ur_type_input.setPlaceholderText("例如: ur5, ur10e, ur16e")
        settings_layout.addWidget(self.ur_type_input, 1, 1)

        settings_layout.addWidget(QLabel("末端夹爪:"), 1, 2)
        self.ee_combo = QComboBox()
        self.ee_combo.addItem("robotiq", "robotiq")
        self.ee_combo.addItem("qbsofthand", "qbsofthand")
        ee_index = max(0, self.ee_combo.findData(self.gui_settings.default_gripper_type))
        self.ee_combo.setCurrentIndex(ee_index)
        settings_layout.addWidget(self.ee_combo, 1, 3)

        settings_layout.addWidget(QLabel("手势识别输入:"), 2, 0)
        self.mediapipe_topic_combo = QComboBox()
        self.mediapipe_topic_combo.setEditable(True)
        self.mediapipe_topic_combo.setCurrentText(self.gui_settings.default_mediapipe_input_topic)
        settings_layout.addWidget(self.mediapipe_topic_combo, 2, 1, 1, 2)

        self.btn_refresh_topics = QPushButton("刷新话题")
        self.btn_refresh_topics.setStyleSheet(
            self._button_style("#eef3fb", "#355070", "#cad5e2", hover_background="#e4edf9", pressed_background="#dbe6f6")
        )
        self.btn_refresh_topics.clicked.connect(self.refresh_mediapipe_topics)
        settings_layout.addWidget(self.btn_refresh_topics, 2, 3)

        self.input_hint_label = QLabel()
        self.input_hint_label.setWordWrap(True)
        self.input_hint_label.setStyleSheet("color: #5b6777; font-size: 12px; padding-top: 4px;")
        self.input_hint_label.setVisible(False)

        layout.addWidget(settings_group)
        layout.addStretch(1)
        return page

    def _build_left_driver_tab(self) -> QWidget:
        page, layout = self._make_tab_page()

        startup_group = QGroupBox("核心驱动控制")
        startup_group.setStyleSheet(self._section_style)
        startup_layout = QGridLayout(startup_group)
        startup_layout.setContentsMargins(12, 8, 12, 12)
        startup_layout.setSpacing(8)
        startup_layout.setColumnStretch(0, 1)
        startup_layout.setColumnStretch(1, 1)

        self.btn_camera_driver = QPushButton("启动相机驱动")
        self.btn_camera_driver.setStyleSheet(
            self._button_style(
                "#eef3fb",
                "#355070",
                "#c5d4ea",
                hover_background="#e3edf9",
                pressed_background="#d6e4f7",
                checked_background="#2f6fed",
                checked_foreground="#ffffff",
                checked_border="#2a62d1",
            )
        )
        self.btn_camera_driver.setCheckable(True)
        self.btn_camera_driver.clicked.connect(self.toggle_camera_driver)
        startup_layout.addWidget(self.btn_camera_driver, 0, 0, 1, 2)

        self.btn_robot_driver = QPushButton("启动机械臂驱动")
        self.btn_robot_driver.setStyleSheet(
            self._button_style(
                "#ebf8f1",
                "#1d6f42",
                "#b8ddc7",
                hover_background="#dff3e7",
                pressed_background="#d1ecd9",
                checked_background="#18945c",
                checked_foreground="#ffffff",
                checked_border="#14784b",
            )
        )
        self.btn_robot_driver.setCheckable(True)
        self.btn_robot_driver.clicked.connect(self.toggle_robot_driver)
        startup_layout.addWidget(self.btn_robot_driver, 1, 0, 1, 2)

        self.btn_teleop = QPushButton("启动遥操作系统")
        self.btn_teleop.setMinimumHeight(40)
        self.btn_teleop.setStyleSheet(
            self._button_style(
                "#dceaff",
                "#154c9c",
                "#b8d1f5",
                hover_background="#cfe2ff",
                pressed_background="#bdd7fb",
                checked_background="#1d4ed8",
                checked_foreground="#ffffff",
                checked_border="#183fb1",
            )
        )
        self.btn_teleop.setCheckable(True)
        self.btn_teleop.clicked.connect(self.toggle_teleop)
        startup_layout.addWidget(self.btn_teleop, 2, 0, 1, 2)

        self.startup_hint_label = QLabel(
            "遥操作启动后会托管机械臂驱动；GUI 会显示驱动运行中，但不允许单独关闭。"
        )
        self.startup_hint_label.setWordWrap(True)
        self.startup_hint_label.setStyleSheet("color: #5b6777; font-size: 12px; padding-top: 4px;")
        self.startup_hint_label.setVisible(False)

        layout.addWidget(startup_group)
        layout.addStretch(1)
        return page

    def _build_left_status_tab(self) -> QWidget:
        page, layout = self._make_tab_page()

        status_group = QGroupBox("模块在线状态")
        status_group.setStyleSheet(self._section_style)
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(10)

        status_layout.addWidget(
            self._build_status_group(
                "核心模块",
                [
                    ("camera_driver", "相机SDK管理"),
                    ("robot_driver", "机械臂 ROS2 驱动"),
                    ("teleop", "遥操作系统"),
                    ("data_collector", "采集节点"),
                    ("inference", "模型推理"),
                    ("preview", "实时预览"),
                ],
                self.module_status_labels,
            )
        )

        status_layout.addWidget(
            self._build_status_group(
                "硬件资源",
                [
                    ("joystick", "手柄设备"),
                    ("camera_1", "相机1"),
                    ("camera_2", "相机2"),
                    ("camera_3", "相机3"),
                    ("robot", "机械臂"),
                    ("gripper", "末端执行器"),
                ],
                self.hardware_status_labels,
            )
        )

        layout.addWidget(status_group)
        layout.addStretch(1)
        return page

    def _build_right_dock(self) -> None:
        dock = QDockWidget("机器人实时状态", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        tabs = QTabWidget()
        state_tab = QWidget()
        state_layout = QVBoxLayout(state_tab)
        state_layout.setContentsMargins(12, 12, 12, 12)
        state_layout.setSpacing(12)

        summary_group = QGroupBox("状态摘要")
        summary_group.setStyleSheet(self._section_style)
        summary_layout = QGridLayout(summary_group)
        summary_layout.setContentsMargins(12, 8, 12, 12)
        summary_layout.setSpacing(8)
        summary_layout.setColumnStretch(1, 1)

        summary_layout.addWidget(QLabel("推理状态:"), 0, 0)
        self.lbl_inference_status = QLabel("未启动")
        self.lbl_inference_status.setWordWrap(True)
        self.lbl_inference_status.setAlignment(Qt.AlignCenter)
        self.lbl_inference_status.setStyleSheet(self._badge_style("#6c757d", alpha=0.10))
        summary_layout.addWidget(self.lbl_inference_status, 0, 1)

        summary_layout.addWidget(QLabel("执行状态:"), 1, 0)
        self.lbl_inference_execute_status = QLabel("未使能")
        self.lbl_inference_execute_status.setAlignment(Qt.AlignCenter)
        self.lbl_inference_execute_status.setStyleSheet(self._badge_style("#6c757d", alpha=0.10))
        summary_layout.addWidget(self.lbl_inference_execute_status, 1, 1)

        summary_layout.addWidget(QLabel("当前 Episode:"), 2, 0)
        self.lbl_demo_status = QLabel("无 (未录制)")
        self.lbl_demo_status.setAlignment(Qt.AlignCenter)
        self.lbl_demo_status.setStyleSheet(self._button_badge_style("#2f6fed", alpha=0.10))
        summary_layout.addWidget(self.lbl_demo_status, 2, 1)

        self.lbl_main_record_stats = QLabel("录制时长: 00:00 | 帧数: 0")
        self.lbl_main_record_stats.setAlignment(Qt.AlignCenter)
        self.lbl_main_record_stats.setStyleSheet(self._button_badge_style("#4c6a92", alpha=0.08))
        summary_layout.addWidget(self.lbl_main_record_stats, 3, 0, 1, 2)
        state_layout.addWidget(summary_group)

        robot_state_group = QGroupBox("机器人状态流")
        robot_state_group.setStyleSheet(self._section_style)
        robot_state_layout = QVBoxLayout(robot_state_group)
        self.robot_state_output = QTextEdit()
        self.robot_state_output.setReadOnly(True)
        self.robot_state_output.setPlaceholderText("等待 ROS worker 推送机器人状态...")
        self.robot_state_output.setStyleSheet(
            "background-color: #fbfdff; color: #1f2937; border: 1px solid #d9e3ef; "
            "border-radius: 14px; padding: 10px; font-family: Consolas, 'Courier New', monospace;"
        )
        robot_state_layout.addWidget(self.robot_state_output)
        state_layout.addWidget(robot_state_group, 1)

        safety_group = QGroupBox("安全与调试")
        safety_group.setStyleSheet(self._section_style)
        safety_layout = QGridLayout(safety_group)
        safety_layout.setContentsMargins(12, 8, 12, 12)
        safety_layout.setSpacing(8)
        safety_layout.setColumnStretch(0, 1)
        safety_layout.setColumnStretch(1, 1)
        self.btn_go_home = QPushButton("回 Home 点")
        self.btn_go_home.setStyleSheet(
            self._button_style("#fff0e3", "#b3541e", "#f0c39d", hover_background="#ffe7d1", pressed_background="#fbdab8")
        )
        self.btn_go_home.clicked.connect(self.go_home)
        safety_layout.addWidget(self.btn_go_home, 0, 0)

        self.btn_go_home_zone = QPushButton("回 Home Zone")
        self.btn_go_home_zone.setStyleSheet(
            self._button_style("#f3e8ff", "#7b3fa0", "#ddc3f0", hover_background="#ebdcff", pressed_background="#dfc8fb")
        )
        self.btn_go_home_zone.clicked.connect(self.go_home_zone)
        safety_layout.addWidget(self.btn_go_home_zone, 0, 1)

        self.btn_set_home_current = QPushButton("设当前姿态为 Home")
        self.btn_set_home_current.setStyleSheet(
            self._button_style("#e8f7ef", "#1d6f42", "#b8ddc7", hover_background="#dbf0e5", pressed_background="#cce7d8")
        )
        self.btn_set_home_current.clicked.connect(self.set_home_from_current)
        safety_layout.addWidget(self.btn_set_home_current, 1, 0, 1, 2)
        state_layout.addWidget(safety_group)

        state_scroll = QScrollArea()
        state_scroll.setWidgetResizable(True)
        state_scroll.setFrameShape(QScrollArea.NoFrame)
        state_scroll.setWidget(state_tab)

        tabs.addTab(state_scroll, "Primary Arm")
        dock.setWidget(tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.dock_state = dock

    def _build_bottom_dock(self) -> None:
        dock = QDockWidget("任务工作台", self)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)

        tabs = QTabWidget()
        tabs.addTab(self._build_record_tab(), "数据采集")
        tabs.addTab(self._build_inference_tab(), "模型推理")
        tabs.addTab(self._build_log_tab(), "日志与预览")

        dock.setWidget(tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.dock_task = dock

    def _build_record_tab(self) -> QWidget:
        page, content, layout = self._make_scroll_tab_page()

        record_group = QGroupBox("数据采集 Workflow")
        record_group.setStyleSheet(self._section_style)
        record_layout = QGridLayout(record_group)
        record_layout.setContentsMargins(12, 8, 12, 12)
        record_layout.setSpacing(8)

        record_layout.addWidget(QLabel("HDF5 保存目录:"), 0, 0)
        self.record_dir_input = QLineEdit(self.gui_settings.default_hdf5_output_dir)
        self.record_dir_input.setToolTip(self.gui_settings.default_hdf5_output_dir)
        self.record_dir_input.setCursorPosition(0)
        record_layout.addWidget(self.record_dir_input, 0, 1, 1, 3)

        self.btn_choose_record_dir = QPushButton("选择目录")
        self.btn_choose_record_dir.setStyleSheet(
            self._button_style("#eef3fb", "#355070", "#cad5e2", hover_background="#e4edf9", pressed_background="#dbe6f6")
        )
        self.btn_choose_record_dir.clicked.connect(self.choose_record_output_dir)
        record_layout.addWidget(self.btn_choose_record_dir, 0, 4)

        record_layout.addWidget(QLabel("HDF5 文件名:"), 1, 0)
        self.record_name_input = QLineEdit(self.gui_settings.default_hdf5_filename)
        self.record_name_input.setToolTip(self.gui_settings.default_hdf5_filename)
        self.record_name_input.setPlaceholderText("例如: libero_demos.hdf5")
        record_layout.addWidget(self.record_name_input, 1, 1, 1, 3)

        self.btn_preview_hdf5 = QPushButton("打开 HDF5 回放器")
        self.btn_preview_hdf5.setStyleSheet(
            self._button_style("#fff3c9", "#8a5a00", "#efd48b", hover_background="#ffebb0", pressed_background="#f8df94")
        )
        self.btn_preview_hdf5.clicked.connect(self.open_hdf5_viewer)
        record_layout.addWidget(self.btn_preview_hdf5, 1, 4)

        self.btn_collector = QPushButton("启动采集节点")
        self.btn_collector.setStyleSheet(
            self._button_style(
                "#e8f7ef",
                "#1d6f42",
                "#b8ddc7",
                hover_background="#dbf0e5",
                pressed_background="#cce7d8",
                checked_background="#18945c",
                checked_foreground="#ffffff",
                checked_border="#14784b",
            )
        )
        self.btn_collector.setCheckable(True)
        self.btn_collector.clicked.connect(self.toggle_data_collector)
        record_layout.addWidget(self.btn_collector, 2, 0)

        self.btn_start_record = QPushButton("开始录制")
        self.btn_start_record.setStyleSheet(
            self._button_style("#d94841", "#ffffff", "#c43c36", hover_background="#c93f38", pressed_background="#b93630")
        )
        self.btn_start_record.clicked.connect(self.start_record)
        record_layout.addWidget(self.btn_start_record, 2, 1)

        self.btn_stop_record = QPushButton("停止录制")
        self.btn_stop_record.setStyleSheet(
            self._button_style("#fff4f4", "#a63b3b", "#efcccc", hover_background="#ffe8e8", pressed_background="#f9d8d8")
        )
        self.btn_stop_record.clicked.connect(self.stop_record)
        record_layout.addWidget(self.btn_stop_record, 2, 2)

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

        layout.addWidget(record_group)
        layout.addStretch(1)
        return page

    def _build_inference_tab(self) -> QWidget:
        page, content, layout = self._make_scroll_tab_page()

        inference_group = QGroupBox("模型推理 Workflow")
        inference_group.setStyleSheet(self._section_style)
        inference_layout = QGridLayout(inference_group)
        inference_layout.setContentsMargins(12, 8, 12, 12)
        inference_layout.setSpacing(8)

        inference_layout.addWidget(QLabel("模型文件夹:"), 0, 0)
        self.inference_model_dir_input = QLineEdit()
        self.inference_model_dir_input.setPlaceholderText("例如: models/ddim_dec_transformer")
        self.inference_model_dir_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        inference_layout.addWidget(self.inference_model_dir_input, 0, 1, 1, 3)

        self.btn_browse_inference_model = QPushButton("选择")
        self.btn_browse_inference_model.setStyleSheet(
            self._button_style("#eef3fb", "#355070", "#cad5e2", hover_background="#e4edf9", pressed_background="#dbe6f6")
        )
        self.btn_browse_inference_model.clicked.connect(self.choose_inference_model_dir)
        inference_layout.addWidget(self.btn_browse_inference_model, 0, 4)

        self.btn_refresh_inference_options = QPushButton("刷新")
        self.btn_refresh_inference_options.setStyleSheet(
            self._button_style("#eef3fb", "#355070", "#cad5e2", hover_background="#e4edf9", pressed_background="#dbe6f6")
        )
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
        self.btn_browse_inference_embedding.setStyleSheet(
            self._button_style("#eef3fb", "#355070", "#cad5e2", hover_background="#e4edf9", pressed_background="#dbe6f6")
        )
        self.btn_browse_inference_embedding.clicked.connect(self.choose_inference_embedding_path)
        inference_layout.addWidget(self.btn_browse_inference_embedding, 2, 4)

        self.btn_auto_match_embedding = QPushButton("自动匹配")
        self.btn_auto_match_embedding.setStyleSheet(
            self._button_style("#eef3fb", "#355070", "#cad5e2", hover_background="#e4edf9", pressed_background="#dbe6f6")
        )
        self.btn_auto_match_embedding.clicked.connect(self.update_inference_embedding_path)
        inference_layout.addWidget(self.btn_auto_match_embedding, 2, 5)

        inference_layout.addWidget(QLabel("全局相机:"), 3, 0)
        self.inference_global_camera_combo = QComboBox()
        self.inference_global_camera_combo.addItem("realsense", "realsense")
        self.inference_global_camera_combo.addItem("oakd", "oakd")
        global_infer_index = max(0, self.inference_global_camera_combo.findData(self.gui_settings.default_global_camera_source))
        self.inference_global_camera_combo.setCurrentIndex(global_infer_index)
        inference_layout.addWidget(self.inference_global_camera_combo, 3, 1)

        inference_layout.addWidget(QLabel("手部相机:"), 3, 2)
        self.inference_wrist_camera_combo = QComboBox()
        self.inference_wrist_camera_combo.addItem("oakd", "oakd")
        self.inference_wrist_camera_combo.addItem("realsense", "realsense")
        wrist_infer_index = max(0, self.inference_wrist_camera_combo.findData(self.gui_settings.default_wrist_camera_source))
        self.inference_wrist_camera_combo.setCurrentIndex(wrist_infer_index)
        inference_layout.addWidget(self.inference_wrist_camera_combo, 3, 3)

        inference_layout.addWidget(QLabel("运行设备:"), 3, 4)
        self.inference_device_combo = QComboBox()
        self.inference_device_combo.addItem("auto", "auto")
        self.inference_device_combo.addItem("cuda", "cuda")
        self.inference_device_combo.addItem("cpu", "cpu")
        self.inference_device_combo.setCurrentIndex(max(0, self.inference_device_combo.findData("cuda")))
        inference_layout.addWidget(self.inference_device_combo, 3, 5)

        inference_layout.addWidget(QLabel("高层动作频率(Hz):"), 4, 0)
        self.inference_hz_spin = QDoubleSpinBox()
        self.inference_hz_spin.setRange(0.2, 30.0)
        self.inference_hz_spin.setDecimals(1)
        self.inference_hz_spin.setSingleStep(0.5)
        self.inference_hz_spin.setValue(10.0)
        self.inference_hz_spin.setToolTip("控制高层动作输出频率；不等同于完整重规划频率。实际重规划频率约为该值 / replan_every。")
        inference_layout.addWidget(self.inference_hz_spin, 4, 1)

        self.btn_inference = QPushButton("启动推理")
        self.btn_inference.setStyleSheet(
            self._button_style(
                "#e8f7ef",
                "#1d6f42",
                "#b8ddc7",
                hover_background="#dbf0e5",
                pressed_background="#cce7d8",
                checked_background="#0f9f66",
                checked_foreground="#ffffff",
                checked_border="#0c8152",
            )
        )
        self.btn_inference.setCheckable(True)
        self.btn_inference.clicked.connect(self.toggle_inference)
        inference_layout.addWidget(self.btn_inference, 5, 0)

        self.btn_execute_inference = QPushButton("开始执行任务")
        self.btn_execute_inference.setStyleSheet(
            self._button_style(
                "#fff0d6",
                "#a05a00",
                "#f0c98c",
                hover_background="#ffe6bf",
                pressed_background="#ffdb9e",
                checked_background="#f59e0b",
                checked_foreground="#ffffff",
                checked_border="#d88907",
            )
        )
        self.btn_execute_inference.setCheckable(True)
        self.btn_execute_inference.setEnabled(False)
        self.btn_execute_inference.clicked.connect(self.toggle_inference_execution)
        inference_layout.addWidget(self.btn_execute_inference, 5, 1)

        self.btn_inference_estop = QPushButton("急停")
        self.btn_inference_estop.setEnabled(False)
        self.btn_inference_estop.setStyleSheet(
            self._button_style("#7a1f1f", "#ffffff", "#651919", hover_background="#6d1919", pressed_background="#5b1414")
        )
        self.btn_inference_estop.clicked.connect(self.emergency_stop_inference_execution)
        inference_layout.addWidget(self.btn_inference_estop, 5, 2)

        self.chk_collect_inference_logs = QCheckBox("记录执行段动作日志")
        self.chk_collect_inference_logs.setChecked(bool(self.gui_settings.collect_inference_action_logs))
        self.chk_collect_inference_logs.setToolTip("勾选后，仅在点击“开始执行任务”期间保存高层动作日志。")
        self.chk_collect_inference_logs.toggled.connect(self._on_collect_inference_logs_toggled)
        inference_layout.addWidget(self.chk_collect_inference_logs, 6, 0, 1, 3)

        self.inference_hint_label = QLabel(
            "说明: 直接调用 Real_IL 的 RealRobotPolicy。这里的频率表示高层动作输出频率，不等同于完整重规划频率；GUI 负责预览、推理调度和动作下发。勾选上方开关后，点击开始执行任务会把执行期间的高层动作保存到 data/inference_action_logs。"
        )
        self.inference_hint_label.setWordWrap(True)
        self.inference_hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.inference_hint_label.setStyleSheet("color: #5b6777; font-size: 12px;")
        self.inference_hint_label.setVisible(False)

        inference_layout.addWidget(QLabel("动作输出:"), 7, 0)
        self.inference_action_output = QTextEdit()
        self.inference_action_output.setReadOnly(True)
        self.inference_action_output.setMinimumHeight(110)
        self.inference_action_output.setMaximumHeight(160)
        self.inference_action_output.setStyleSheet(
            "background-color: #fbfdff; color: #1f2937; border: 1px solid #d9e3ef; "
            "border-radius: 12px; padding: 10px; font-family: Consolas, 'Courier New', monospace;"
        )
        inference_layout.addWidget(self.inference_action_output, 8, 0, 1, 6)

        layout.addWidget(inference_group)
        layout.addStretch(1)
        return page

    def _build_log_tab(self) -> QWidget:
        page, content, layout = self._make_scroll_tab_page()

        log_group = QGroupBox("系统运行日志")
        log_group.setStyleSheet(self._section_style)
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(12, 8, 12, 12)
        log_layout.setSpacing(10)

        self.btn_preview = QPushButton("弹出详细预览与状态窗")
        self.btn_preview.setStyleSheet(
            self._button_style("#edf2ff", "#4b5cb8", "#c4ccf0", hover_background="#e1e8ff", pressed_background="#d3ddfb")
        )
        self.btn_preview.clicked.connect(self.open_preview_window)
        log_layout.addWidget(self.btn_preview)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color: #17202a; color: #d5e2f0; border: 1px solid #283646; "
            "border-radius: 12px; font-family: Consolas, 'Courier New', monospace;"
        )
        log_layout.addWidget(self.log_output)

        layout.addWidget(log_group)
        return page

        inference_layout.addWidget(QLabel("模型文件夹:"), 0, 0)
        self.inference_model_dir_input = QLineEdit()
        self.inference_model_dir_input.setPlaceholderText("例如: models/ddim_dec_transformer")
        self.inference_model_dir_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        inference_layout.addWidget(self.inference_model_dir_input, 0, 1, 1, 3)

        self.btn_browse_inference_model = QPushButton("选择")
        self.btn_browse_inference_model.setStyleSheet(
            self._button_style("#eef3fb", "#355070", "#cad5e2", hover_background="#e4edf9", pressed_background="#dbe6f6")
        )
        self.btn_browse_inference_model.clicked.connect(self.choose_inference_model_dir)
        inference_layout.addWidget(self.btn_browse_inference_model, 0, 4)

        self.btn_refresh_inference_options = QPushButton("刷新")
        self.btn_refresh_inference_options.setStyleSheet(
            self._button_style("#eef3fb", "#355070", "#cad5e2", hover_background="#e4edf9", pressed_background="#dbe6f6")
        )
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
        self.btn_browse_inference_embedding.setStyleSheet(
            self._button_style("#eef3fb", "#355070", "#cad5e2", hover_background="#e4edf9", pressed_background="#dbe6f6")
        )
        self.btn_browse_inference_embedding.clicked.connect(self.choose_inference_embedding_path)
        inference_layout.addWidget(self.btn_browse_inference_embedding, 2, 4)

        self.btn_auto_match_embedding = QPushButton("自动匹配")
        self.btn_auto_match_embedding.setStyleSheet(
            self._button_style("#eef3fb", "#355070", "#cad5e2", hover_background="#e4edf9", pressed_background="#dbe6f6")
        )
        self.btn_auto_match_embedding.clicked.connect(self.update_inference_embedding_path)
        inference_layout.addWidget(self.btn_auto_match_embedding, 2, 5)

        inference_layout.addWidget(QLabel("全局相机:"), 3, 0)
        self.inference_global_camera_combo = QComboBox()
        self.inference_global_camera_combo.addItem("realsense", "realsense")
        self.inference_global_camera_combo.addItem("oakd", "oakd")
        global_infer_index = max(0, self.inference_global_camera_combo.findData(self.gui_settings.default_global_camera_source))
        self.inference_global_camera_combo.setCurrentIndex(global_infer_index)
        inference_layout.addWidget(self.inference_global_camera_combo, 3, 1)

        inference_layout.addWidget(QLabel("手部相机:"), 3, 2)
        self.inference_wrist_camera_combo = QComboBox()
        self.inference_wrist_camera_combo.addItem("oakd", "oakd")
        self.inference_wrist_camera_combo.addItem("realsense", "realsense")
        wrist_infer_index = max(0, self.inference_wrist_camera_combo.findData(self.gui_settings.default_wrist_camera_source))
        self.inference_wrist_camera_combo.setCurrentIndex(wrist_infer_index)
        inference_layout.addWidget(self.inference_wrist_camera_combo, 3, 3)

        inference_layout.addWidget(QLabel("运行设备:"), 3, 4)
        self.inference_device_combo = QComboBox()
        self.inference_device_combo.addItem("auto", "auto")
        self.inference_device_combo.addItem("cuda", "cuda")
        self.inference_device_combo.addItem("cpu", "cpu")
        self.inference_device_combo.setCurrentIndex(max(0, self.inference_device_combo.findData("cuda")))
        inference_layout.addWidget(self.inference_device_combo, 3, 5)

        inference_layout.addWidget(QLabel("高层动作频率(Hz):"), 4, 0)
        self.inference_hz_spin = QDoubleSpinBox()
        self.inference_hz_spin.setRange(0.2, 30.0)
        self.inference_hz_spin.setDecimals(1)
        self.inference_hz_spin.setSingleStep(0.5)
        self.inference_hz_spin.setValue(10.0)
        self.inference_hz_spin.setToolTip("控制高层动作输出频率；不等同于完整重规划频率。实际重规划频率约为该值 / replan_every。")
        inference_layout.addWidget(self.inference_hz_spin, 4, 1)

        self.btn_inference = QPushButton("启动推理")
        self.btn_inference.setStyleSheet(
            self._button_style(
                "#e8f7ef",
                "#1d6f42",
                "#b8ddc7",
                hover_background="#dbf0e5",
                pressed_background="#cce7d8",
                checked_background="#0f9f66",
                checked_foreground="#ffffff",
                checked_border="#0c8152",
            )
        )
        self.btn_inference.setCheckable(True)
        self.btn_inference.clicked.connect(self.toggle_inference)
        inference_layout.addWidget(self.btn_inference, 5, 0)

        self.btn_execute_inference = QPushButton("开始执行任务")
        self.btn_execute_inference.setStyleSheet(
            self._button_style(
                "#fff0d6",
                "#a05a00",
                "#f0c98c",
                hover_background="#ffe6bf",
                pressed_background="#ffdb9e",
                checked_background="#f59e0b",
                checked_foreground="#ffffff",
                checked_border="#d88907",
            )
        )
        self.btn_execute_inference.setCheckable(True)
        self.btn_execute_inference.setEnabled(False)
        self.btn_execute_inference.clicked.connect(self.toggle_inference_execution)
        inference_layout.addWidget(self.btn_execute_inference, 5, 1)

        self.btn_inference_estop = QPushButton("急停")
        self.btn_inference_estop.setEnabled(False)
        self.btn_inference_estop.setStyleSheet(
            self._button_style("#7a1f1f", "#ffffff", "#651919", hover_background="#6d1919", pressed_background="#5b1414")
        )
        self.btn_inference_estop.clicked.connect(self.emergency_stop_inference_execution)
        inference_layout.addWidget(self.btn_inference_estop, 5, 2)

        self.chk_collect_inference_logs = QCheckBox("记录执行段动作日志")
        self.chk_collect_inference_logs.setChecked(bool(self.gui_settings.collect_inference_action_logs))
        self.chk_collect_inference_logs.setToolTip("勾选后，仅在点击“开始执行任务”期间保存高层动作日志。")
        self.chk_collect_inference_logs.toggled.connect(self._on_collect_inference_logs_toggled)
        inference_layout.addWidget(self.chk_collect_inference_logs, 6, 0, 1, 3)

        self.inference_hint_label = QLabel(
            "说明: 直接调用 Real_IL 的 RealRobotPolicy。这里的频率表示高层动作输出频率，不等同于完整重规划频率；GUI 负责预览、推理调度和动作下发。勾选上方开关后，点击开始执行任务会把执行期间的高层动作保存到 data/inference_action_logs。"
        )
        self.inference_hint_label.setWordWrap(True)
        self.inference_hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.inference_hint_label.setStyleSheet("color: #555; font-size: 12px;")
        self.inference_hint_label.setVisible(False)

        inference_layout.addWidget(QLabel("动作输出:"), 7, 0)
        self.inference_action_output = QTextEdit()
        self.inference_action_output.setReadOnly(True)
        self.inference_action_output.setMinimumHeight(110)
        self.inference_action_output.setMaximumHeight(140)
        self.inference_action_output.setStyleSheet(
            "background-color: #fbfdff; color: #1f2937; border: 1px solid #d9e3ef; "
            "border-radius: 14px; padding: 10px; font-family: Consolas, 'Courier New', monospace;"
        )
        inference_layout.addWidget(self.inference_action_output, 8, 0, 1, 6)
        layout.addWidget(inference_group, 5)

        log_group = QGroupBox("系统运行日志")
        log_group.setStyleSheet(self._section_style)
        log_layout = QVBoxLayout(log_group)
        self.btn_preview = QPushButton("弹出详细预览与状态窗")
        self.btn_preview.setStyleSheet(
            self._button_style("#edf2ff", "#4b5cb8", "#c4ccf0", hover_background="#e1e8ff", pressed_background="#d3ddfb")
        )
        self.btn_preview.clicked.connect(self.open_preview_window)
        log_layout.addWidget(self.btn_preview)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color: #17202a; color: #d5e2f0; border: 1px solid #283646; "
            "border-radius: 12px; font-family: Consolas, 'Courier New', monospace;"
        )
        log_layout.addWidget(self.log_output)
        layout.addWidget(log_group, 4)

        dock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.dock_task = dock

    def _build_status_group(self, title: str, entries: List[tuple[str, str]], target: dict) -> QGroupBox:
        group = QGroupBox(title)
        group.setStyleSheet(self._section_style)
        grid = QGridLayout(group)
        grid.addWidget(QLabel("名称"), 0, 0)
        grid.addWidget(QLabel("状态"), 0, 1)
        for row, (key, label_text) in enumerate(entries, start=1):
            grid.addWidget(QLabel(label_text), row, 0)
            value_label = QLabel("未知")
            target[key] = value_label
            grid.addWidget(value_label, row, 1)
        return group

    @Slot(str)
    def _dispatch_robot_state_update(self, text: str) -> None:
        if hasattr(self, "robot_state_output") and self.robot_state_output is not None:
            self.robot_state_output.setText(text)
        if self.preview_window is not None:
            self.preview_window.update_robot_state_str(text)

    @Slot(int, str, float)
    def _dispatch_preview_record_stats(self, frames, time_str, realtime_fps) -> None:
        if hasattr(self, "vision_panel") and self.vision_panel is not None:
            self.vision_panel.update_record_stats(frames, time_str, realtime_fps)
        if self.preview_window is not None:
            self.preview_window.update_record_stats(frames, time_str, realtime_fps)

    def _persisted_home_joint_positions(self) -> List[float]:
        profile_name = self._selected_robot_profile()
        persisted = load_home_override(__file__, profile_name)
        if len(persisted) == 6:
            return persisted
        # 兼容旧版：若尚未迁移到独立覆盖文件，则回退一次旧 GUI 配置。
        legacy_profile = robot_profile_name_from_ur_type(self.gui_settings.ur_type)
        if profile_name == legacy_profile and len(self.gui_settings.home_joint_positions) == 6:
            return [float(value) for value in self.gui_settings.home_joint_positions]
        return []

    def _save_home_override(self, joints: List[float]) -> Optional[Path]:
        if len(joints) != 6:
            return None

        try:
            path = save_home_override(__file__, self._selected_robot_profile(), joints)
            self._sync_ros_worker_home_config()
            return path
        except Exception as exc:
            self.log(f"写入 Home 覆盖配置失败: {exc}")
            return None

    def _apply_persisted_home_to_ui_log(self) -> None:
        persisted = self._persisted_home_joint_positions()
        if len(persisted) == 6:
            joints_str = np.array2string(
                np.array(persisted),
                formatter={"float_kind": lambda value: f"{value:6.3f}"},
            )
            self.log(f"已加载持久化 Home 点: {joints_str}")

    def _build_ros_worker_callbacks(self) -> RosWorkerCallbacks:
        return RosWorkerCallbacks(
            log=self.log,
            demo_status=self.update_demo_status,
            record_stats=self.update_main_record_stats,
            recording_state=self._on_recording_state_changed,
            home_zone_active=self._on_home_zone_active_changed,
            homing_active=self._on_homing_active_changed,
            commander_result=self._on_commander_result,
        )

    def _current_ros_worker_config(self) -> RosWorkerConfig:
        profile = load_robot_profile(self._selected_robot_profile())
        return RosWorkerConfig(
            robot_profile=profile.name,
            gripper_topic=self.COLLECTOR_GRIPPER_TOPIC,
            home_joint_positions=self._persisted_home_joint_positions() or list(profile.home.joint_positions),
            home_duration_sec=float(profile.home.duration_sec),
            joint_names=list(profile.joint_names),
            trajectory_topic=profile.topics.home_joint_trajectory,
            teleop_controller=profile.controllers.teleop,
            trajectory_controller=profile.controllers.trajectory,
            home_zone_translation_min_m=list(profile.home_zone.translation_min_m),
            home_zone_translation_max_m=list(profile.home_zone.translation_max_m),
            home_zone_rotation_min_deg=list(profile.home_zone.rotation_min_deg),
            home_zone_rotation_max_deg=list(profile.home_zone.rotation_max_deg),
            inference_gripper_type=self._selected_gripper_type(),
            inference_control_hz=50.0,
        )

    def _sync_ros_worker_home_config(self) -> None:
        if not self.app_service.has_ros_worker():
            return
        self.app_service.apply_ros_worker_config(self._current_ros_worker_config())

    def _sync_ros_worker_inference_control_config(self) -> None:
        if not self.app_service.has_ros_worker():
            return
        self.app_service.apply_ros_worker_config(self._current_ros_worker_config())

    def _build_inference_worker_callbacks(self) -> InferenceWorkerCallbacks:
        return InferenceWorkerCallbacks(
            action=self.update_inference_action,
            preview=self.update_inference_preview,
            status=self.update_inference_status,
            log=self.log,
            error=self._on_inference_error,
            finished=self._on_inference_finished,
        )

    def _current_inference_launch_config(
        self,
        *,
        checkpoint_dir: Path,
        task_name: str,
        embedding_path: Path,
        global_source: str,
        wrist_source: str,
    ) -> InferenceLaunchConfig:
        return InferenceLaunchConfig(
            checkpoint_dir=str(checkpoint_dir),
            task_name=task_name,
            task_embedding_path=str(embedding_path),
            global_camera_source=global_source,
            wrist_camera_source=wrist_source,
            loop_hz=float(self.inference_hz_spin.value()),
            device=self._selected_inference_device(),
            state_provider=self._current_robot_state_for_inference,
            global_camera_serial_number=self._selected_inference_camera_serial_numbers()[0],
            wrist_camera_serial_number=self._selected_inference_camera_serial_numbers()[1],
            global_camera_enable_depth=self._camera_enable_depth(global_source),
            wrist_camera_enable_depth=self._camera_enable_depth(wrist_source),
        )

    def _camera_enable_depth(self, camera_source: str) -> bool:
        return camera_enable_depth(self.gui_settings, camera_source)

    def _default_mediapipe_camera_model(self) -> str:
        model = str(self.gui_settings.default_mediapipe_camera).strip().lower()
        if model in {"", "realsense", "rs", "camera"}:
            return "d435"
        return model

    def _default_mediapipe_camera_source(self) -> str:
        model = self._default_mediapipe_camera_model()
        if model == "oakd":
            return "oakd"
        return "realsense"

    @staticmethod
    def _camera_model_label(source: str, model: str) -> str:
        normalized_source = str(source).strip().lower()
        normalized_model = str(model).strip().lower()
        if not normalized_model or normalized_model == "camera":
            normalized_model = "oakd" if normalized_source == "oakd" else "d435"
        if normalized_model == "oakd":
            return "OAK-D"
        return normalized_model.upper()

    @staticmethod
    def _camera_option_data(source: str, model: str, serial: str, *, label: str = "") -> dict[str, str]:
        normalized_source = str(source).strip().lower()
        normalized_model = str(model).strip().lower() or normalized_source
        normalized_serial = str(serial).strip()
        normalized_label = str(label).strip() or TeleopMainWindow._camera_model_label(normalized_source, normalized_model)
        return {
            "source": normalized_source,
            "model": normalized_model,
            "serial": normalized_serial,
            "label": normalized_label,
        }

    def _camera_option_candidates(self) -> list[dict[str, str]]:
        options: list[dict[str, str]] = []
        for sdk_camera in discover_sdk_cameras():
            options.append(
                self._camera_option_data(
                    sdk_camera.source,
                    sdk_camera.model,
                    sdk_camera.serial_number,
                )
            )

        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for option in options:
            key = (
                str(option.get("source", "")).strip().lower(),
                str(option.get("model", "")).strip().lower(),
                str(option.get("serial", "")).strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(option)

        def _priority(item: dict[str, str]) -> tuple[int, int, str, str]:
            source = str(item.get("source", "")).strip().lower()
            model = str(item.get("model", "")).strip().lower()
            serial = str(item.get("serial", "")).strip()
            source_rank = 0 if source == "realsense" else (1 if source == "oakd" else 2)
            model_rank = {"d455": 0, "d435": 1, "oakd": 2}.get(model, 9)
            serial_rank = 1 if serial else 2
            return source_rank, model_rank, serial_rank, serial

        deduped.sort(key=_priority)
        model_counts: dict[str, int] = {}
        for item in deduped:
            label_key = self._camera_model_label(str(item.get("source", "")), str(item.get("model", "")))
            model_counts[label_key] = model_counts.get(label_key, 0) + 1

        model_indices: dict[str, int] = {}
        relabeled: list[dict[str, str]] = []
        for item in deduped:
            source = str(item.get("source", "")).strip().lower()
            model = str(item.get("model", "")).strip().lower()
            serial = str(item.get("serial", "")).strip()
            base_label = self._camera_model_label(source, model)
            if model_counts.get(base_label, 0) > 1:
                current_index = model_indices.get(base_label, 0) + 1
                model_indices[base_label] = current_index
                label = f"{base_label} 相机{current_index}"
            else:
                label = base_label
            relabeled.append(self._camera_option_data(source, model, serial, label=label))
        return relabeled

    @staticmethod
    def _selected_camera_option(
        combo: QComboBox,
        fallback_source: str,
        fallback_model: str = "",
    ) -> dict[str, str]:
        value = combo.currentData()
        if isinstance(value, dict):
            source = str(value.get("source", "")).strip().lower() or str(fallback_source).strip().lower()
            model = str(value.get("model", "")).strip().lower() or str(fallback_model).strip().lower() or source
            serial = str(value.get("serial", "")).strip()
            return {"source": source, "model": model, "serial": serial}
        normalized_value = str(value).strip().lower() if value is not None else ""
        normalized_fallback_source = str(fallback_source).strip().lower()
        normalized_fallback_model = str(fallback_model).strip().lower()

        if normalized_value in {"oakd"}:
            normalized_source = "oakd"
        elif normalized_value in {"realsense", "rs", "d435", "d455", "camera", "l515"}:
            normalized_source = "realsense"
        else:
            normalized_source = normalized_fallback_source or "realsense"

        normalized_model = normalized_value or normalized_fallback_model
        if normalized_model in {"realsense", "rs"}:
            normalized_model = "d435"
        if not normalized_model:
            normalized_model = "oakd" if normalized_source == "oakd" else "d435"
        return {"source": normalized_source, "model": normalized_model, "serial": ""}

    @staticmethod
    def _set_camera_combo_options(
        combo: QComboBox,
        options: list[dict[str, str]],
        *,
        preferred_source: str,
        preferred_model: str = "",
        preferred_serial: str = "",
    ) -> None:
        normalized_source = str(preferred_source).strip().lower()
        normalized_model = str(preferred_model).strip().lower()
        normalized_serial = str(preferred_serial).strip()

        combo.blockSignals(True)
        combo.clear()
        for option in options:
            combo.addItem(str(option.get("label", "")), dict(option))

        selected_index = -1
        for index, option in enumerate(options):
            source = str(option.get("source", "")).strip().lower()
            model = str(option.get("model", "")).strip().lower()
            serial = str(option.get("serial", "")).strip()
            if (
                normalized_source
                and normalized_serial
                and source == normalized_source
                and serial == normalized_serial
            ):
                selected_index = index
                break
            if (
                selected_index < 0
                and normalized_source
                and normalized_model
                and source == normalized_source
                and model == normalized_model
            ):
                selected_index = index
            if selected_index < 0 and normalized_source and source == normalized_source:
                selected_index = index

        if selected_index < 0 and options:
            selected_index = 0
        if selected_index >= 0:
            combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def refresh_camera_devices(self, log_result: bool = True) -> None:
        options = self._camera_option_candidates()
        self._sdk_cameras = [dict(item) for item in options]
        self._camera_options_loaded = True
        inference_global_option = self._selected_camera_option(
            self.inference_global_camera_combo,
            self.gui_settings.default_inference_global_camera_source,
            self.gui_settings.default_inference_global_camera_model,
        )
        inference_wrist_option = self._selected_camera_option(
            self.inference_wrist_camera_combo,
            self.gui_settings.default_inference_wrist_camera_source,
            self.gui_settings.default_inference_wrist_camera_model,
        )

        self._set_camera_combo_options(
            self.mediapipe_camera_combo,
            options,
            preferred_source=self._default_mediapipe_camera_source(),
            preferred_model=self._default_mediapipe_camera_model(),
            preferred_serial=self.gui_settings.default_mediapipe_camera_serial_number,
        )
        self._set_camera_combo_options(
            self.global_camera_source_combo,
            options,
            preferred_source=self.gui_settings.default_global_camera_source,
            preferred_model=self.gui_settings.default_collector_global_camera_model,
            preferred_serial=self.gui_settings.default_collector_global_camera_serial_number,
        )
        self._set_camera_combo_options(
            self.wrist_camera_source_combo,
            options,
            preferred_source=self.gui_settings.default_wrist_camera_source,
            preferred_model=self.gui_settings.default_collector_wrist_camera_model,
            preferred_serial=self.gui_settings.default_collector_wrist_camera_serial_number,
        )
        self._set_camera_combo_options(
            self.inference_global_camera_combo,
            options,
            preferred_source=str(inference_global_option.get("source", "")).strip().lower()
            or self.gui_settings.default_inference_global_camera_source,
            preferred_model=str(inference_global_option.get("model", "")).strip().lower()
            or self.gui_settings.default_inference_global_camera_model,
            preferred_serial=str(inference_global_option.get("serial", "")).strip(),
        )
        self._set_camera_combo_options(
            self.inference_wrist_camera_combo,
            options,
            preferred_source=str(inference_wrist_option.get("source", "")).strip().lower()
            or self.gui_settings.default_inference_wrist_camera_source,
            preferred_model=str(inference_wrist_option.get("model", "")).strip().lower()
            or self.gui_settings.default_inference_wrist_camera_model,
            preferred_serial=str(inference_wrist_option.get("serial", "")).strip(),
        )

        self._sync_mediapipe_topic_from_camera_selection()
        self._update_input_hint()
        self._refresh_runtime_status()

        if log_result:
            self.log(f"已刷新 SDK 相机列表，检测到 {len(options)} 台设备。")

    def _persist_camera_preferences(self) -> None:
        updates = self._camera_preference_updates()
        try:
            save_gui_settings_overrides(__file__, updates)
            self.gui_settings = load_gui_settings(__file__)
        except Exception as exc:
            self.log(f"保存相机默认配置失败: {exc}")

    def _on_mediapipe_camera_selection_changed(self, *_args) -> None:
        self._sync_mediapipe_topic_from_camera_selection()
        self._update_input_hint()
        self._refresh_runtime_status()
        self._persist_camera_preferences()

    def _on_mediapipe_topic_changed(self, *_args) -> None:
        self._update_input_hint()
        self._refresh_runtime_status()
        self._persist_camera_preferences()

    def _on_collector_camera_selection_changed(self, *_args) -> None:
        self._on_collector_camera_settings_changed()

    def _on_collector_camera_settings_changed(self, *_args) -> None:
        self._refresh_runtime_status()
        self._persist_camera_preferences()

    def _on_inference_camera_selection_changed(self, *_args) -> None:
        self._refresh_runtime_status()
        self._persist_camera_preferences()

    def _selected_input_type(self) -> str:
        value = self.mode_combo.currentData()
        return str(value).strip().lower() if value is not None else "joy"

    def _selected_joy_profile(self) -> str:
        value = self.joy_profile_combo.currentData()
        return str(value).strip().lower() if value is not None else self.gui_settings.default_joy_profile

    def _selected_ur_type(self) -> str:
        return self.ur_type_input.text().strip() or self.gui_settings.ur_type or "ur5"

    def _selected_robot_profile(self) -> str:
        return robot_profile_name_from_ur_type(self._selected_ur_type())

    def _selected_mediapipe_camera(self) -> str:
        option = self._selected_camera_option(
            self.mediapipe_camera_combo,
            self._default_mediapipe_camera_source(),
            self._default_mediapipe_camera_model(),
        )
        model = str(option.get("model", "")).strip().lower()
        return model or self._default_mediapipe_camera_model()

    def _selected_mediapipe_camera_serial_number(self) -> str:
        option = self._selected_camera_option(
            self.mediapipe_camera_combo,
            self._default_mediapipe_camera_source(),
            self._default_mediapipe_camera_model(),
        )
        selected_serial = str(option.get("serial", "")).strip()
        if selected_serial:
            return selected_serial
        # SDK 相机列表刷新后若当前项没有序列号，不再回退到历史配置，
        # 以免使用到过期序列号导致拉帧失败（表现为持续 WAITING_DEPTH）。
        if self._camera_options_loaded:
            return ""
        return self.gui_settings.default_mediapipe_camera_serial_number.strip()

    def _mediapipe_camera_profiles(self) -> dict[str, dict[str, str]]:
        d435_serial = self.gui_settings.realsense_d435_serial_number.strip()
        d455_serial = self.gui_settings.realsense_d455_serial_number.strip()
        return {
            "d435": {
                "name": "d435",
                "driver": "realsense",
                "namespace": "d435",
                "camera_name": "camera",
                "serial": d435_serial,
                "input_topic": "/d435/camera/color/image_raw",
                "depth_topic": "/d435/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/d435/camera/aligned_depth_to_color/camera_info",
            },
            "d455": {
                "name": "d455",
                "driver": "realsense",
                "namespace": "d455",
                "camera_name": "camera",
                "serial": d455_serial,
                "input_topic": "/d455/camera/color/image_raw",
                "depth_topic": "/d455/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/d455/camera/aligned_depth_to_color/camera_info",
            },
            "oakd": {
                "name": "oakd",
                "driver": "oakd",
                "namespace": "oakd",
                "camera_name": "camera",
                "serial": "",
                "input_topic": "/oakd/rgb/image_raw",
                "depth_topic": "/oakd/stereo/depth",
                "camera_info_topic": "/oakd/rgb/camera_info",
            },
            "camera": {
                "name": "camera",
                "driver": "realsense",
                "namespace": "camera",
                "camera_name": "camera",
                "serial": "",
                "input_topic": "/camera/camera/color/image_raw",
                "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
            },
        }

    def _selected_mediapipe_camera_profile(self) -> dict[str, str]:
        profiles = self._mediapipe_camera_profiles()
        selected = self._selected_mediapipe_camera()
        default_key = self._default_mediapipe_camera_model()
        fallback = profiles.get(default_key) or profiles.get("d435") or next(iter(profiles.values()))
        profile = dict(profiles.get(selected) or fallback)
        option = self._selected_camera_option(
            self.mediapipe_camera_combo,
            self._default_mediapipe_camera_source(),
            self._default_mediapipe_camera_model(),
        )
        source_override = str(option.get("source", "")).strip().lower()
        if source_override:
            profile["driver"] = source_override
        serial_override = self._selected_mediapipe_camera_serial_number()
        if serial_override:
            profile["serial"] = serial_override
        return profile

    def _sync_mediapipe_topic_from_camera_selection(self) -> None:
        profile = self._selected_mediapipe_camera_profile()
        topic = str(profile.get("input_topic", "")).strip()
        if not topic:
            return
        current = self.mediapipe_topic_combo.currentText().strip()
        if current == topic:
            return
        self.mediapipe_topic_combo.blockSignals(True)
        self.mediapipe_topic_combo.setCurrentText(topic)
        self.mediapipe_topic_combo.blockSignals(False)

    def _selected_mediapipe_topic(self) -> str:
        profile = self._selected_mediapipe_camera_profile()
        topic = self.mediapipe_topic_combo.currentText().strip()
        if topic:
            return topic
        fallback = str(profile.get("input_topic", "")).strip()
        return fallback or self.gui_settings.default_mediapipe_input_topic

    def _selected_mediapipe_depth_topic(self) -> str:
        profile = self._selected_mediapipe_camera_profile()
        return str(profile.get("depth_topic", "")).strip() or "/camera/camera/aligned_depth_to_color/image_raw"

    def _selected_mediapipe_camera_info_topic(self) -> str:
        profile = self._selected_mediapipe_camera_profile()
        return str(profile.get("camera_info_topic", "")).strip() or "/camera/camera/aligned_depth_to_color/camera_info"

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
        profile = self._selected_mediapipe_camera_profile()
        return [
            str(profile.get("input_topic", "")).strip(),
            self.gui_settings.default_mediapipe_input_topic,
            "/d435/camera/color/image_raw",
            "/d455/camera/color/image_raw",
            "/oakd/rgb/image_raw",
            "/camera/camera/color/image_raw",
            "/camera/color/image_raw",
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

        preferred = current_value or self.gui_settings.default_mediapipe_input_topic
        self._set_combo_items_unique(self.mediapipe_topic_combo, topics + [preferred], preferred)

        if log_result:
            self.log(f"已刷新手势识别输入话题，共 {self.mediapipe_topic_combo.count()} 个候选项。")
        self._update_input_hint()

    def _preview_window_visible(self) -> bool:
        return bool(self.preview_window is not None and self.preview_window.isVisible())

    def _collector_preview_api_should_run(self) -> bool:
        return self._preview_window_visible() and self.runtime_facade.is_process_running("data_collector")

    def _inference_preview_should_render(self) -> bool:
        return self._preview_window_visible() and self._inference_running() and not self.runtime_facade.is_process_running("data_collector")

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

    def _selected_robot_display_name(self) -> str:
        name = self._selected_ur_type().strip()
        return name.upper() if name else "机械臂"

    def _selected_gripper_display_name(self) -> str:
        gripper_type = self._selected_gripper_type()
        if gripper_type == "robotiq":
            return "Robotiq"
        if gripper_type == "qbsofthand":
            return "qbSoftHand"
        return gripper_type or "末端执行器"

    def _selected_collector_end_effector_type(self) -> str:
        return "qbsofthand" if self._selected_gripper_type() == "qbsofthand" else "robotic_gripper"

    def _selected_camera_source(self, combo: QComboBox, fallback: str) -> str:
        option = self._selected_camera_option(combo, fallback, fallback)
        source = str(option.get("source", "")).strip().lower()
        return source or fallback

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
        option = self._selected_camera_option(combo, fallback, fallback)
        source = str(option.get("source", "")).strip().lower()
        return source or fallback

    def _selected_collector_camera_sources(self) -> tuple[str, str]:
        return (
            self._selected_camera_source(self.global_camera_source_combo, self.gui_settings.default_global_camera_source),
            self._selected_camera_source(self.wrist_camera_source_combo, self.gui_settings.default_wrist_camera_source),
        )

    def _selected_collector_camera_serial_numbers(self) -> tuple[str, str]:
        global_option = self._selected_camera_option(
            self.global_camera_source_combo,
            self.gui_settings.default_global_camera_source,
            self.gui_settings.default_collector_global_camera_model,
        )
        wrist_option = self._selected_camera_option(
            self.wrist_camera_source_combo,
            self.gui_settings.default_wrist_camera_source,
            self.gui_settings.default_collector_wrist_camera_model,
        )
        global_serial = str(global_option.get("serial", "")).strip()
        wrist_serial = str(wrist_option.get("serial", "")).strip()
        return global_serial, wrist_serial

    def _selected_inference_camera_serial_numbers(self) -> tuple[str, str]:
        global_option = self._selected_camera_option(
            self.inference_global_camera_combo,
            self.gui_settings.default_inference_global_camera_source,
            self.gui_settings.default_inference_global_camera_model,
        )
        wrist_option = self._selected_camera_option(
            self.inference_wrist_camera_combo,
            self.gui_settings.default_inference_wrist_camera_source,
            self.gui_settings.default_inference_wrist_camera_model,
        )
        global_serial = str(global_option.get("serial", "")).strip()
        wrist_serial = str(wrist_option.get("serial", "")).strip()
        return global_serial, wrist_serial

    def _count_sdk_cameras_by_source(self, source: str) -> int:
        normalized = str(source or "").strip().lower()
        if not normalized:
            return 0
        return sum(
            1
            for device in self._sdk_cameras
            if str(device.get("source", "")).strip().lower() == normalized
        )

    @staticmethod
    def _camera_option_matches_device(
        option: dict[str, str],
        *,
        source: str,
        model: str,
        serial: str,
    ) -> bool:
        option_source = str(option.get("source", "")).strip().lower()
        option_model = str(option.get("model", "")).strip().lower()
        option_serial = str(option.get("serial", "")).strip()
        normalized_source = str(source).strip().lower()
        normalized_model = str(model).strip().lower()
        normalized_serial = str(serial).strip()
        if option_source != normalized_source or option_model != normalized_model:
            return False
        if normalized_serial and option_serial:
            return option_serial == normalized_serial
        return True

    def _camera_slot_runtime_status(
        self,
        slot_index: int,
        *,
        teleop_running: bool,
        collector_running: bool,
        inference_running: bool,
    ) -> tuple[str, str]:
        if slot_index < 1 or len(self._sdk_cameras) < slot_index:
            return "未检测到", "#6c757d"

        slot_device = dict(self._sdk_cameras[slot_index - 1])
        label = str(slot_device.get("label", "")).strip() or f"相机{slot_index}"
        source = str(slot_device.get("source", "")).strip().lower()
        model = str(slot_device.get("model", "")).strip().lower()
        serial = str(slot_device.get("serial", "")).strip()

        global_option = self._selected_camera_option(
            self.global_camera_source_combo,
            self.gui_settings.default_global_camera_source,
            self.gui_settings.default_collector_global_camera_model,
        )
        wrist_option = self._selected_camera_option(
            self.wrist_camera_source_combo,
            self.gui_settings.default_wrist_camera_source,
            self.gui_settings.default_collector_wrist_camera_model,
        )
        mediapipe_option = self._selected_camera_option(
            self.mediapipe_camera_combo,
            self._default_mediapipe_camera_source(),
            self._default_mediapipe_camera_model(),
        )
        inference_global_option = self._selected_camera_option(
            self.inference_global_camera_combo,
            self.gui_settings.default_inference_global_camera_source,
            self.gui_settings.default_inference_global_camera_model,
        )
        inference_wrist_option = self._selected_camera_option(
            self.inference_wrist_camera_combo,
            self.gui_settings.default_inference_wrist_camera_source,
            self.gui_settings.default_inference_wrist_camera_model,
        )

        if collector_running and (
            self._camera_option_matches_device(global_option, source=source, model=model, serial=serial)
            or self._camera_option_matches_device(wrist_option, source=source, model=model, serial=serial)
        ):
            return f"{label} 采集占用", "#e67700"
        if inference_running and (
            self._camera_option_matches_device(inference_global_option, source=source, model=model, serial=serial)
            or self._camera_option_matches_device(inference_wrist_option, source=source, model=model, serial=serial)
        ):
            return f"{label} 推理占用", "#e67700"
        if teleop_running and self._selected_input_type() == "mediapipe" and self._camera_option_matches_device(
            mediapipe_option,
            source=source,
            model=model,
            serial=serial,
        ):
            return f"{label} 手势占用", "#e67700"
        return f"{label} 可用", "#2b8a3e"

    def _selected_inference_camera_sources(self) -> tuple[str, str]:
        return (
            self._selected_inference_camera_source(
                self.inference_global_camera_combo,
                self.gui_settings.default_inference_global_camera_source,
            ),
            self._selected_inference_camera_source(
                self.inference_wrist_camera_combo,
                self.gui_settings.default_inference_wrist_camera_source,
            ),
        )

    def _set_active_teleop_camera_binding(self, source: str, serial_number: str) -> None:
        normalized = str(source).strip().lower()
        if normalized not in {"realsense", "oakd"}:
            self._active_teleop_camera_source = ""
            self._active_teleop_camera_serial = ""
            return
        self._active_teleop_camera_source = normalized
        self._active_teleop_camera_serial = str(serial_number).strip()

    def _clear_active_teleop_camera_binding(self) -> None:
        self._active_teleop_camera_source = ""
        self._active_teleop_camera_serial = ""

    def _active_camera_driver_devices(self) -> list[tuple[str, str]]:
        devices: list[tuple[str, str]] = []
        if self.runtime_facade.is_process_running("teleop") and self._active_teleop_camera_source:
            devices.append((self._active_teleop_camera_source, self._active_teleop_camera_serial))
        return devices

    def _camera_runtime_context(self) -> CameraRuntimeContext:
        collector_global_source, collector_wrist_source = self._selected_collector_camera_sources()
        collector_global_serial, collector_wrist_serial = self._selected_collector_camera_serial_numbers()
        inference_global_source, inference_wrist_source = self._selected_inference_camera_sources()
        inference_global_serial, inference_wrist_serial = self._selected_inference_camera_serial_numbers()
        return self.runtime_facade.build_camera_context(
            collector_global_source=collector_global_source,
            collector_wrist_source=collector_wrist_source,
            collector_global_serial=collector_global_serial,
            collector_wrist_serial=collector_wrist_serial,
            inference_global_source=inference_global_source,
            inference_wrist_source=inference_wrist_source,
            inference_global_serial=inference_global_serial,
            inference_wrist_serial=inference_wrist_serial,
            inference_running=self._inference_running(),
            active_camera_driver_devices=self._active_camera_driver_devices(),
        )

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

    def _on_inference_task_selection_changed(self, *_args) -> None:
        self._notify_running_inference_goal_changed(log_request=True)

    def _notify_running_inference_goal_changed(self, *_args, log_request: bool = False) -> None:
        if not self.inference_service.is_running():
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

        if not self.inference_service.request_task_update(
            task_name=task_name,
            task_embedding_path=str(embedding_path),
        ):
            self.log(f"推理任务切换失败: 当前推理线程未运行，任务 `{task_name}` 未下发。")
            return

        self._set_inference_status(f"切换任务中 | {task_name}", "#e67700")
        if log_request:
            self.log(f"已请求实时切换推理任务: {task_name}")

    def _set_inference_status(self, text: str, color: str) -> None:
        self._set_status_label(self.lbl_inference_status, text, color)

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
        self._set_status_label(self.lbl_inference_execute_status, text, color)

    def _show_intent_result(self, result: IntentResult) -> None:
        if result.allowed:
            return
        QMessageBox.warning(self, result.title or "控制冲突", result.message or f"当前请求被状态机拒绝: {result.reason}")

    @Slot(bool)
    def _on_recording_state_changed(self, active: bool) -> None:
        self.orchestrator.notify_recording(bool(active))

    @Slot(bool)
    def _on_home_zone_active_changed(self, active: bool) -> None:
        self.orchestrator.notify_home_zone(bool(active))
        self._refresh_runtime_status()

    @Slot(bool)
    def _on_homing_active_changed(self, active: bool) -> None:
        self.orchestrator.notify_homing(bool(active))
        self._refresh_runtime_status()

    @Slot(str)
    def _on_commander_result(self, message: str) -> None:
        self.log(f"Commander 状态: {message}")

    def _apply_commander_preconditions(self) -> bool:
        if not self.app_service.has_ros_worker():
            self.start_ros_worker()
            self.log("已启动独立 ROS 监听器，用于执行 Home / Home Zone 控制。")
        if not self.app_service.has_ros_worker():
            return False
        if self._inference_execution_running():
            self._clear_pending_inference_execution_start()
            self.app_service.disable_inference_execution()
            self.orchestrator.notify_inference_execution(False)
            self._set_inference_execute_button_running(False)
            self._set_inference_execute_status("未使能", "#6c757d")
            self.btn_inference_estop.setEnabled(self._inference_running())
            self._refresh_runtime_status()
        return True

    def _request_commander_motion(self, motion: str) -> None:
        result = self.intent_controller.check_commander_motion(motion)
        if not result.allowed:
            self._show_intent_result(result)
            return
        if not self._apply_commander_preconditions():
            return
        if motion == "home":
            self.app_service.go_home()
            return
        self.app_service.go_home_zone()

    def _request_recording(self, active: bool) -> None:
        result = self.intent_controller.check_recording(active)
        if not result.allowed:
            self._show_intent_result(result)
            return
        if not self.app_service.has_ros_worker():
            QMessageBox.warning(self, "警告", "请先启动采集节点！")
            return
        if active:
            self.app_service.start_record()
            return
        self.app_service.stop_record()

    def _inference_running(self) -> bool:
        return self.inference_service.is_running()

    def _inference_execution_running(self) -> bool:
        if self.app_service.inference_execution_enabled():
            return True
        return self.btn_execute_inference.isChecked()

    def _clear_pending_inference_execution_start(self) -> None:
        self._pending_inference_execution_start = False
        self._pending_inference_action_log_kwargs = None

    def _build_inference_action_log_kwargs(self) -> Optional[dict[str, object]]:
        if not self._should_collect_inference_action_logs():
            return None
        model_dir = self._selected_inference_model_dir()
        task_name = self._selected_inference_task_name()
        embedding_path = self.inference_embedding_input.text().strip()
        global_source, wrist_source = self._selected_inference_camera_sources()
        return {
            "checkpoint_dir": str(Path(model_dir).expanduser().resolve()) if model_dir else "",
            "task_name": task_name,
            "task_embedding_path": str(Path(embedding_path).expanduser().resolve()) if embedding_path else "",
            "loop_hz": float(self.inference_hz_spin.value()),
            "global_camera_source": global_source,
            "wrist_camera_source": wrist_source,
            "device": str(self._selected_inference_device() or "auto"),
            "control_hz": float(self._current_ros_worker_config().inference_control_hz),
        }

    def _queue_inference_execution_start(self) -> None:
        self._pending_inference_execution_start = True
        self._pending_inference_action_log_kwargs = self._build_inference_action_log_kwargs()
        self._set_inference_execute_button_running(True)
        self._set_inference_execute_status("等待块起点", "#e67700")
        self.log("已请求开始执行，将在下一个动作块第 1 步时使能控制。")
        self._refresh_runtime_status()

    def _activate_pending_inference_execution(self) -> None:
        if not self._pending_inference_execution_start:
            return

        log_kwargs = self._pending_inference_action_log_kwargs
        self._clear_pending_inference_execution_start()
        self.app_service.enable_inference_execution(
            ros_worker_config=self._current_ros_worker_config(),
            ros_worker_callbacks=self._build_ros_worker_callbacks(),
        )
        if log_kwargs is not None:
            try:
                csv_path = self.app_service.start_inference_action_logging(**log_kwargs)
                self.log(f"本次执行段高层推理动作将记录到: {csv_path}")
            except Exception as exc:
                self.log(f"创建执行段高层推理动作日志失败: {exc}")
        self.orchestrator.notify_inference_execution(True)
        self._set_inference_execute_button_running(True)
        self._set_inference_execute_status("执行中", "#2b8a3e")
        self.btn_inference_estop.setEnabled(True)
        self.log("已在新的动作块起点开始执行推理任务，动作将直接发送到控制器。")
        self._refresh_runtime_status()

    def _ros_worker_required(self) -> bool:
        preview_running = bool(self.preview_window is not None and self.preview_window.isVisible())
        return self.app_service.ros_worker_required(
            preview_running=preview_running,
            inference_running=self._inference_running(),
        )

    def _stop_ros_worker_if_unused(self) -> None:
        self.app_service.stop_ros_worker_if_unused(
            preview_running=bool(self.preview_window is not None and self.preview_window.isVisible()),
            inference_running=self._inference_running(),
        )

    def _start_preview_api_worker(self) -> None:
        if not self._collector_preview_api_should_run():
            return
        if self.preview_api_worker is not None:
            self.preview_api_worker.set_base_url(self.COLLECTOR_PREVIEW_API_BASE_URL)
            return

        self.preview_api_worker = HttpPreviewWorker(self.COLLECTOR_PREVIEW_API_BASE_URL, fps=30.0)
        self.preview_api_worker.availability_signal.connect(self._on_preview_api_availability_changed)
        self.preview_api_worker.log_signal.connect(self.log)
        if self.vision_panel is not None:
            self.preview_api_worker.global_image_signal.connect(self.vision_panel.update_global_image)
            self.preview_api_worker.wrist_image_signal.connect(self.vision_panel.update_wrist_image)
        if self.preview_window is not None:
            self.preview_api_worker.global_image_signal.connect(self.preview_window.update_global_image)
            self.preview_api_worker.wrist_image_signal.connect(self.preview_window.update_wrist_image)
            self._preview_window_api_connected = True
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
        if self.vision_panel is not None:
            self.vision_panel.set_preview_source("采集节点 API" if available else "采集节点 API (未就绪)")
            if not available:
                self.vision_panel.clear_images()
        if self.toolbar_preview_label is not None:
            self.toolbar_preview_label.setText(f"监视器: {'采集节点 API' if available else '采集节点 API (未就绪)'}")
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
        if not self._inference_preview_should_render():
            return
        if self._latest_inference_global_bgr is not None and self.vision_panel is not None:
            self.vision_panel.update_global_image(self._latest_inference_global_bgr)
        if self._latest_inference_global_bgr is not None and self.preview_window is not None:
            self.preview_window.update_global_image(self._latest_inference_global_bgr)
        if self._latest_inference_wrist_bgr is not None and self.vision_panel is not None:
            self.vision_panel.update_wrist_image(self._latest_inference_wrist_bgr)
        if self._latest_inference_wrist_bgr is not None and self.preview_window is not None:
            self.preview_window.update_wrist_image(self._latest_inference_wrist_bgr)

    def _ensure_ros_worker_for_inference(self) -> None:
        if self.app_service.has_ros_worker():
            return
        self.start_ros_worker()
        self.log("已启动 ROS 监听器，用于给推理读取机器人状态。")

    def _sync_preview_pipeline(self) -> None:
        if self._collector_preview_api_should_run():
            self._start_preview_api_worker()
            source = "采集节点 API" if self._preview_api_active else "采集节点 API (连接中)"
            if self.toolbar_preview_label is not None:
                self.toolbar_preview_label.setText(f"监视器: {source}")
            if self.vision_panel is not None:
                self.vision_panel.set_preview_source(source)
                if not self._preview_api_active:
                    self.vision_panel.clear_images()
            if self.preview_window is not None:
                self.preview_window.set_preview_source(source)
                if not self._preview_api_active:
                    self.preview_window.clear_images()
            return

        self._stop_preview_api_worker()
        if self._inference_preview_should_render():
            if self.toolbar_preview_label is not None:
                self.toolbar_preview_label.setText("监视器: 推理直连")
            if self.vision_panel is not None:
                self.vision_panel.set_preview_source("推理直连")
                if self._latest_inference_global_bgr is None and self._latest_inference_wrist_bgr is None:
                    self.vision_panel.clear_images()
            if self.preview_window is not None:
                self.preview_window.set_preview_source("推理直连")
                if self._latest_inference_global_bgr is None and self._latest_inference_wrist_bgr is None:
                    self.preview_window.clear_images()
            self._render_inference_preview_frames()
            return

        if self.vision_panel is not None:
            self.vision_panel.set_preview_source("无活动图像源")
            self.vision_panel.clear_images()
        if self.toolbar_preview_label is not None:
            self.toolbar_preview_label.setText("监视器: 无活动图像源")
        if self.preview_window is not None:
            self.preview_window.set_preview_source("无活动图像源")
            self.preview_window.clear_images()

    def _current_robot_state_for_inference(self) -> Optional[np.ndarray]:
        return self.app_service.current_robot_state_vector()

    def _should_collect_inference_action_logs(self) -> bool:
        checkbox = getattr(self, "chk_collect_inference_logs", None)
        if checkbox is None:
            return bool(self.gui_settings.collect_inference_action_logs)
        return bool(checkbox.isChecked())

    @staticmethod
    def _normalize_settings_path(value: str) -> str:
        text = str(value).strip()
        if not text:
            return ""
        try:
            return str(Path(text).expanduser().resolve())
        except Exception:
            return text

    def _restore_persisted_gui_state(self) -> None:
        self._suspend_gui_settings_persist = True
        try:
            model_dir = self._normalize_settings_path(self.gui_settings.default_inference_model_dir)
            if model_dir:
                self.inference_model_dir_input.setText(model_dir)
                self._sync_inference_model_dir_tooltip()

            device_name = str(self.gui_settings.default_inference_device).strip().lower() or "cuda"
            device_index = self.inference_device_combo.findData(device_name)
            if device_index < 0:
                device_index = self.inference_device_combo.findData("cuda")
            if device_index >= 0:
                self.inference_device_combo.setCurrentIndex(device_index)

            hz_value = float(self.gui_settings.default_inference_hz)
            hz_value = min(max(hz_value, self.inference_hz_spin.minimum()), self.inference_hz_spin.maximum())
            self.inference_hz_spin.setValue(hz_value)

            preferred_env = str(self.gui_settings.default_inference_env).strip()
            if preferred_env:
                env_index = self.inference_env_combo.findData(preferred_env)
                if env_index < 0:
                    env_index = self.inference_env_combo.findText(preferred_env)
                if env_index >= 0:
                    self.inference_env_combo.setCurrentIndex(env_index)

            preferred_embedding = self._normalize_settings_path(self.gui_settings.default_inference_embedding_path)
            if preferred_embedding:
                self.inference_embedding_input.setText(preferred_embedding)
                self.inference_embedding_input.setToolTip(preferred_embedding)
                self.inference_embedding_input.setCursorPosition(0)
                embedding_path = Path(preferred_embedding).expanduser()
                if embedding_path.is_file():
                    self._populate_inference_tasks_from_embedding(embedding_path.resolve())

            preferred_task = str(self.gui_settings.default_inference_task).strip()
            if preferred_task:
                task_index = self.inference_task_combo.findData(preferred_task)
                if task_index < 0:
                    task_index = self.inference_task_combo.findText(preferred_task)
                if task_index >= 0:
                    self.inference_task_combo.setCurrentIndex(task_index)
        finally:
            self._suspend_gui_settings_persist = False

    def _connect_gui_settings_persistence(self) -> None:
        if self._gui_settings_persist_connected:
            return

        self.mode_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.joy_profile_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.ur_type_input.textChanged.connect(self._schedule_gui_settings_persist)
        self.mediapipe_camera_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.mediapipe_topic_combo.currentTextChanged.connect(self._schedule_gui_settings_persist)
        self.ip_input.textChanged.connect(self._schedule_gui_settings_persist)
        self.ee_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.camera_driver_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.record_dir_input.textChanged.connect(self._schedule_gui_settings_persist)
        self.record_name_input.textChanged.connect(self._schedule_gui_settings_persist)
        self.global_camera_source_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.wrist_camera_source_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.inference_model_dir_input.textChanged.connect(self._schedule_gui_settings_persist)
        self.inference_env_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.inference_task_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.inference_embedding_input.textChanged.connect(self._schedule_gui_settings_persist)
        self.inference_global_camera_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.inference_wrist_camera_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.inference_device_combo.currentIndexChanged.connect(self._schedule_gui_settings_persist)
        self.inference_hz_spin.valueChanged.connect(self._schedule_gui_settings_persist)
        self.chk_collect_inference_logs.toggled.connect(self._schedule_gui_settings_persist)
        self._gui_settings_persist_connected = True

    def _camera_preference_updates(self) -> dict[str, object]:
        mediapipe_option = self._selected_camera_option(
            self.mediapipe_camera_combo,
            self._default_mediapipe_camera_source(),
            self._default_mediapipe_camera_model(),
        )
        global_option = self._selected_camera_option(
            self.global_camera_source_combo,
            self.gui_settings.default_global_camera_source,
            self.gui_settings.default_collector_global_camera_model,
        )
        wrist_option = self._selected_camera_option(
            self.wrist_camera_source_combo,
            self.gui_settings.default_wrist_camera_source,
            self.gui_settings.default_collector_wrist_camera_model,
        )
        inference_global_option = self._selected_camera_option(
            self.inference_global_camera_combo,
            self.gui_settings.default_inference_global_camera_source,
            self.gui_settings.default_inference_global_camera_model,
        )
        inference_wrist_option = self._selected_camera_option(
            self.inference_wrist_camera_combo,
            self.gui_settings.default_inference_wrist_camera_source,
            self.gui_settings.default_inference_wrist_camera_model,
        )

        mediapipe_serial = str(mediapipe_option.get("serial", "")).strip()
        global_serial = str(global_option.get("serial", "")).strip()
        wrist_serial = str(wrist_option.get("serial", "")).strip()
        inference_global_serial = str(inference_global_option.get("serial", "")).strip()
        inference_wrist_serial = str(inference_wrist_option.get("serial", "")).strip()

        global_source = str(global_option.get("source", "")).strip().lower() or self.gui_settings.default_global_camera_source
        wrist_source = str(wrist_option.get("source", "")).strip().lower() or self.gui_settings.default_wrist_camera_source
        inference_global_source = (
            str(inference_global_option.get("source", "")).strip().lower()
            or self.gui_settings.default_inference_global_camera_source
        )
        inference_wrist_source = (
            str(inference_wrist_option.get("source", "")).strip().lower()
            or self.gui_settings.default_inference_wrist_camera_source
        )

        return {
            "default_mediapipe_camera": str(mediapipe_option.get("model", "")).strip().lower() or self._default_mediapipe_camera_model(),
            "default_mediapipe_camera_serial_number": mediapipe_serial,
            "default_mediapipe_input_topic": self._selected_mediapipe_topic(),
            "default_global_camera_source": global_source,
            "default_wrist_camera_source": wrist_source,
            "default_collector_global_camera_model": str(global_option.get("model", "")).strip().lower() or self.gui_settings.default_collector_global_camera_model,
            "default_collector_wrist_camera_model": str(wrist_option.get("model", "")).strip().lower() or self.gui_settings.default_collector_wrist_camera_model,
            "default_collector_global_camera_serial_number": global_serial,
            "default_collector_wrist_camera_serial_number": wrist_serial,
            "default_inference_global_camera_source": inference_global_source,
            "default_inference_global_camera_model": str(inference_global_option.get("model", "")).strip().lower()
            or self.gui_settings.default_inference_global_camera_model,
            "default_inference_global_camera_serial_number": inference_global_serial,
            "default_inference_wrist_camera_source": inference_wrist_source,
            "default_inference_wrist_camera_model": str(inference_wrist_option.get("model", "")).strip().lower()
            or self.gui_settings.default_inference_wrist_camera_model,
            "default_inference_wrist_camera_serial_number": inference_wrist_serial,
        }

    def _collect_gui_settings_overrides(self) -> dict[str, object]:
        updates: dict[str, object] = {
            "default_robot_ip": self.ip_input.text().strip() or self.gui_settings.default_robot_ip,
            "default_input_type": self._selected_input_type(),
            "default_joy_profile": self._selected_joy_profile(),
            "ur_type": self._selected_ur_type(),
            "default_gripper_type": self._selected_gripper_type(),
            "default_camera_driver": self._selected_camera_driver(),
            "default_hdf5_output_dir": self._normalize_settings_path(self._selected_record_output_dir()),
            "default_hdf5_filename": self._selected_record_filename(),
            "default_inference_model_dir": self._normalize_settings_path(self._selected_inference_model_dir()),
            "default_inference_env": self._selected_inference_env(),
            "default_inference_task": self._selected_inference_task_name(),
            "default_inference_embedding_path": self._normalize_settings_path(self.inference_embedding_input.text().strip()),
            "default_inference_device": str(self._selected_inference_device() or "auto"),
            "default_inference_hz": float(self.inference_hz_spin.value()),
            "collect_inference_action_logs": bool(self._should_collect_inference_action_logs()),
        }
        updates.update(self._camera_preference_updates())
        return updates

    def _schedule_gui_settings_persist(self, *_args) -> None:
        if self._suspend_gui_settings_persist:
            return
        self._gui_settings_persist_timer.start(350)

    def _persist_gui_settings_snapshot(self, *, log_errors: bool = True) -> Optional[Path]:
        if self._suspend_gui_settings_persist:
            return None
        try:
            return save_gui_settings_overrides(__file__, self._collect_gui_settings_overrides())
        except Exception as exc:
            if log_errors:
                self.log(f"保存 GUI 持久化配置失败: {exc}")
            return None

    def _prompt_inference_action_log_outcome(self, csv_path: Path, *, reason_label: str) -> str:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Question)
        dialog.setWindowTitle("记录任务结果")
        dialog.setText("请标注这次执行任务是否成功。")
        dialog.setInformativeText(
            f"日志已保存到:\n{csv_path}\n\n结束原因: {reason_label}\n\n点击【跳过并删除】会删除当前这次记录。"
        )
        dialog.setWindowFlag(Qt.WindowCloseButtonHint, False)
        success_button = dialog.addButton("成功", QMessageBox.AcceptRole)
        failure_button = dialog.addButton("失败", QMessageBox.DestructiveRole)
        skip_button = dialog.addButton("跳过并删除", QMessageBox.RejectRole)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked is success_button:
            return "success"
        if clicked is failure_button:
            return "failure"
        if clicked is skip_button:
            return "skip"
        return "failure"

    def _finalize_inference_action_log(self, *, reason_label: str, stop_reason: str) -> Optional[Path]:
        csv_path = self.app_service.stop_inference_action_logging()
        if csv_path is None:
            return None

        resolved_csv = Path(csv_path).expanduser().resolve()
        outcome = self._prompt_inference_action_log_outcome(resolved_csv, reason_label=reason_label)
        if outcome == "skip":
            deleted_dir = self.app_service.discard_inference_action_log(resolved_csv)
            if deleted_dir is not None:
                self.log(f"已删除高层推理动作日志({reason_label}): {deleted_dir}")
            else:
                self.log(f"跳过标注后删除高层推理动作日志失败: {resolved_csv}")
            return None

        metadata_path = self.app_service.annotate_inference_action_log_result(
            resolved_csv,
            outcome=outcome,
            stop_reason=stop_reason,
        )
        outcome_text = "成功" if outcome == "success" else "失败"
        if metadata_path is not None:
            self.log(f"已保存高层推理动作日志({reason_label})，结果标注为【{outcome_text}】: {metadata_path}")
        else:
            self.log(f"已保存高层推理动作日志({reason_label})，结果标注为【{outcome_text}】: {resolved_csv}")
        return resolved_csv

    @Slot(bool)
    def _on_collect_inference_logs_toggled(self, checked: bool) -> None:
        save_gui_settings_overrides(__file__, {"collect_inference_action_logs": bool(checked)})
        self.gui_settings = load_gui_settings(__file__)
        if not checked:
            self._finalize_inference_action_log(
                reason_label="关闭记录开关",
                stop_reason="toggle_record_off",
            )

    def _set_status_label(self, label: QLabel, text: str, color: str) -> None:
        label.setText(text)
        label.setStyleSheet(f"font-weight: bold; color: {color};")

    def _update_input_mode_widgets(self) -> None:
        input_type = self._selected_input_type()
        is_joy = input_type == "joy"
        is_mediapipe = input_type == "mediapipe"
        self.joy_profile_combo.setEnabled(is_joy)
        self.mediapipe_topic_combo.setEnabled(is_mediapipe)
        self.mediapipe_camera_combo.setEnabled(is_mediapipe)

    def _refresh_runtime_status(self) -> None:
        popup_preview_running = bool(self.preview_window is not None and self.preview_window.isVisible())
        preview_running = self._collector_preview_api_should_run() or self._inference_preview_should_render() or popup_preview_running
        collector_global_source, collector_wrist_source = self._selected_collector_camera_sources()
        collector_global_serial, collector_wrist_serial = self._selected_collector_camera_serial_numbers()
        inference_global_source, inference_wrist_source = self._selected_inference_camera_sources()
        inference_global_serial, inference_wrist_serial = self._selected_inference_camera_serial_numbers()
        runtime_snapshot = self.runtime_facade.runtime_snapshot(
            collector_global_source=collector_global_source,
            collector_wrist_source=collector_wrist_source,
            collector_global_serial=collector_global_serial,
            collector_wrist_serial=collector_wrist_serial,
            inference_global_source=inference_global_source,
            inference_wrist_source=inference_wrist_source,
            inference_global_serial=inference_global_serial,
            inference_wrist_serial=inference_wrist_serial,
            inference_running=self._inference_running(),
            input_type=self._selected_input_type(),
            selected_camera_driver=self._selected_camera_driver(),
            preview_running=popup_preview_running,
            active_camera_driver_devices=self._active_camera_driver_devices(),
        )
        self.local_ip_label.setText(runtime_snapshot.local_ip)

        teleop_running = runtime_snapshot.teleop_running
        robot_driver_running = runtime_snapshot.robot_driver_running
        collector_running = runtime_snapshot.collector_running
        camera_context = runtime_snapshot.camera_context
        detected_count = len(self._sdk_cameras)
        if "camera_driver" in self.module_status_labels:
            if detected_count > 0:
                self._set_status_label(self.module_status_labels["camera_driver"], f"SDK可用 ({detected_count})", "#2b8a3e")
            elif self._camera_options_loaded:
                self._set_status_label(self.module_status_labels["camera_driver"], "SDK未检测到", "#e67700")
            else:
                self._set_status_label(self.module_status_labels["camera_driver"], "初始化中", "#6c757d")

        if teleop_running:
            self._set_status_label(self.module_status_labels["robot_driver"], "由遥操作系统托管", "#e67700")
        elif runtime_snapshot.robot_driver_standalone_running:
            self._set_status_label(self.module_status_labels["robot_driver"], "独立运行中", "#2b8a3e")
        else:
            self._set_status_label(self.module_status_labels["robot_driver"], "未启动", "#6c757d")

        self._set_status_label(self.module_status_labels["teleop"], "运行中" if teleop_running else "未启动", "#2b8a3e" if teleop_running else "#6c757d")
        self._set_status_label(self.module_status_labels["data_collector"], "运行中" if collector_running else "未启动", "#2b8a3e" if collector_running else "#6c757d")
        inference_text, inference_color = self._inference_module_summary()
        self._set_status_label(self.module_status_labels["inference"], inference_text, inference_color)
        self._set_status_label(self.module_status_labels["preview"], "打开" if preview_running else "关闭", "#2b8a3e" if preview_running else "#6c757d")

        if "joystick" in self.hardware_status_labels:
            self._set_status_label(
                self.hardware_status_labels["joystick"],
                runtime_snapshot.joystick_status_text,
                runtime_snapshot.joystick_status_color,
            )

        for slot_index, key in enumerate(("camera_1", "camera_2", "camera_3"), start=1):
            if key not in self.hardware_status_labels:
                continue
            slot_text, slot_color = self._camera_slot_runtime_status(
                slot_index,
                teleop_running=teleop_running,
                collector_running=collector_running,
                inference_running=runtime_snapshot.inference_running,
            )
            self._set_status_label(self.hardware_status_labels[key], slot_text, slot_color)

        runtime_state = self.orchestrator.snapshot()
        phase_text_map = {
            SystemPhase.IDLE: "IDLE",
            SystemPhase.TELEOP: "TELEOP",
            SystemPhase.INFERENCE_READY: "INFERENCE READY",
            SystemPhase.INFERENCE_EXECUTING: "INFERENCE",
            SystemPhase.HOMING: "HOMING",
            SystemPhase.HOME_ZONE: "HOME ZONE",
            SystemPhase.ESTOP: "E-STOP",
        }
        phase_text = phase_text_map.get(runtime_state.phase, runtime_state.phase.name)
        if runtime_state.recording_active and phase_text not in {"HOMING", "HOME ZONE", "E-STOP"}:
            phase_text = f"{phase_text} / RECORDING"
        ros_ready = teleop_running or robot_driver_running or collector_running or self._inference_running()
        if self.toolbar_runtime_label is not None:
            self.toolbar_runtime_label.setText(f"ROS: {'就绪' if ros_ready else '待机'} | 本机 IP: {runtime_snapshot.local_ip}")
        if self.toolbar_phase_label is not None:
            self.toolbar_phase_label.setText(f"模式: {phase_text}")
        commander_busy = runtime_state.phase in {SystemPhase.HOMING, SystemPhase.HOME_ZONE}
        if "robot" in self.hardware_status_labels:
            robot_name = self._selected_robot_display_name()
            if runtime_state.phase == SystemPhase.HOMING:
                self._set_status_label(self.hardware_status_labels["robot"], f"{robot_name} 回 Home 中", "#d9480f")
            elif runtime_state.phase == SystemPhase.HOME_ZONE:
                self._set_status_label(self.hardware_status_labels["robot"], f"{robot_name} Home Zone 中", "#8e44ad")
            elif teleop_running:
                self._set_status_label(self.hardware_status_labels["robot"], f"{robot_name} 遥操作占用", "#2b8a3e")
            elif self._inference_execution_running():
                self._set_status_label(self.hardware_status_labels["robot"], f"{robot_name} 推理执行中", "#e67700")
            elif robot_driver_running:
                self._set_status_label(self.hardware_status_labels["robot"], f"{robot_name} 可用", "#2b8a3e")
            else:
                self._set_status_label(self.hardware_status_labels["robot"], "空闲", "#6c757d")

        if "gripper" in self.hardware_status_labels:
            gripper_name = self._selected_gripper_display_name()
            if runtime_state.phase == SystemPhase.HOMING:
                self._set_status_label(self.hardware_status_labels["gripper"], f"{gripper_name} 跟随 Home 流程", "#d9480f")
            elif runtime_state.phase == SystemPhase.HOME_ZONE:
                self._set_status_label(self.hardware_status_labels["gripper"], f"{gripper_name} 跟随 Home Zone", "#8e44ad")
            elif teleop_running:
                self._set_status_label(self.hardware_status_labels["gripper"], f"{gripper_name} 遥操作占用", "#2b8a3e")
            elif self._inference_execution_running():
                self._set_status_label(self.hardware_status_labels["gripper"], f"{gripper_name} 推理执行", "#e67700")
            elif robot_driver_running:
                self._set_status_label(self.hardware_status_labels["gripper"], f"{gripper_name} 可用", "#2b8a3e")
            else:
                self._set_status_label(self.hardware_status_labels["gripper"], "空闲", "#6c757d")

        self.btn_robot_driver.setEnabled(not teleop_running)
        self.btn_robot_driver.setToolTip("遥操作系统运行时，机械臂驱动由 teleop 统一托管，不能单独关闭。" if teleop_running else "")
        teleop_toggle_allowed = teleop_running or (not self._inference_execution_running() and not commander_busy)
        self.btn_teleop.setEnabled(teleop_toggle_allowed)
        if teleop_running:
            self.btn_teleop.setToolTip("")
        elif self._inference_execution_running():
            self.btn_teleop.setToolTip("推理执行中时不能启动遥操作系统，避免双重控制。")
        elif commander_busy:
            self.btn_teleop.setToolTip("Home / Home Zone 进行中时不能启动新的遥操作流程。")
        else:
            self.btn_teleop.setToolTip("")
        self.btn_go_home.setEnabled(not commander_busy)
        self.btn_go_home_zone.setEnabled(not commander_busy)
        home_tooltip = "当前已有 Home / Home Zone 在执行。" if commander_busy else ""
        self.btn_go_home.setToolTip(home_tooltip)
        self.btn_go_home_zone.setToolTip(home_tooltip)
        if hasattr(self, "btn_discard_record"):
            current_demo = self.lbl_demo_status.text().strip()
            has_discard_candidate = current_demo not in {"", "无 (未录制)"}
            can_discard_demo = collector_running and (not self.app_service.is_recording()) and has_discard_candidate
            self.btn_discard_record.setEnabled(can_discard_demo)
            if not collector_running:
                self.btn_discard_record.setToolTip("请先启动采集节点。")
            elif self.app_service.is_recording():
                self.btn_discard_record.setToolTip("录制进行中，停止录制后才可弃用。")
            elif not has_discard_candidate:
                self.btn_discard_record.setToolTip("当前没有可弃用的 Demo。")
            else:
                self.btn_discard_record.setToolTip(f"将删除最后一个 Demo（当前: {current_demo}）。")

        self.btn_camera_driver.setEnabled(False)
        self.btn_camera_driver.blockSignals(True)
        self.btn_camera_driver.setChecked(False)
        self.btn_camera_driver.blockSignals(False)
        self.btn_camera_driver.setText("相机 ROS2 驱动（已停用）")
        self.btn_camera_driver.setToolTip("相机 ROS2 驱动入口已停用，当前统一使用 SDK 相机管理。")

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
            profile = self._selected_mediapipe_camera_profile()
            camera_name = str(profile.get("name", "d435")).upper()
            self.input_hint_label.setText(
                f"手势识别输入当前使用相机 `{camera_name}`。"
            )
            self.input_hint_label.setStyleSheet("color: #555; font-size: 12px;")
            return

        if input_type == "quest3":
            self.input_hint_label.setText(
                "Quest3 模式使用 WebXR 控制器输入。请先保持 Quest bridge 运行，并在头显中打开对应的 Quest 页面。"
            )
            self.input_hint_label.setStyleSheet("color: #555; font-size: 12px;")
            return

        self.input_hint_label.setText(
            f"Joy 模式当前使用手柄配置 `{self._selected_joy_profile()}`。"
        )
        self.input_hint_label.setStyleSheet("color: #555; font-size: 12px;")

    def _set_button_running(self, button: QPushButton, running: bool, start_text: str, stop_text: str, style: str = "") -> None:
        button.blockSignals(True)
        button.setChecked(running)
        button.blockSignals(False)
        button.setText(stop_text if running else start_text)
        button.update()

    @Slot(str, int)
    def _on_process_exit(self, key: str, _returncode: int) -> None:
        if key in {"camera_driver_realsense", "camera_driver_oakd"}:
            self._refresh_runtime_status()
            return
        if key == "teleop":
            self._clear_active_teleop_camera_binding()
            self.orchestrator.notify_teleop_stopped()
            self._set_button_running(self.btn_teleop, False, "启动遥操作系统", "停止遥操作系统")
            self._set_button_running(self.btn_robot_driver, False, "启动机械臂驱动", "停止机械臂驱动")
            self._stop_ros_worker_if_unused()
            self._refresh_runtime_status()
            return
        if key == "robot_driver":
            self._set_button_running(self.btn_robot_driver, False, "启动机械臂驱动", "停止机械臂驱动")
            self._stop_ros_worker_if_unused()
            self._refresh_runtime_status()
            return
        if key == "data_collector":
            self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
            self._sync_preview_pipeline()
            self._stop_ros_worker_if_unused()
            self._refresh_runtime_status()

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

    def toggle_camera_driver(self, checked):
        _ = checked
        self._set_button_running(self.btn_camera_driver, False, "相机 ROS2 驱动（已停用）", "相机 ROS2 驱动（已停用）")
        self.btn_camera_driver.setEnabled(False)
        self.btn_camera_driver.setToolTip("相机 ROS2 驱动入口已停用。")
        self.log("相机 ROS2 驱动入口已停用，当前统一使用相机 SDK 管理配置。")
        self._refresh_runtime_status()

    def toggle_robot_driver(self, checked):
        if self.runtime_facade.is_process_running("teleop"):
            QMessageBox.information(self, "提示", "遥操作系统运行中时，机械臂驱动由 teleop 统一托管，不能单独操作。")
            self._set_button_running(self.btn_robot_driver, True, "启动机械臂驱动", "停止机械臂驱动", "background-color: #ffe8cc;")
            return

        if checked:
            launch_config = RobotDriverLaunchConfig(
                robot_ip=self.ip_input.text().strip(),
                reverse_ip=self._selected_reverse_ip(),
                ur_type=self._selected_ur_type(),
                gripper_type=self._selected_gripper_type(),
                robot_profile=self._selected_robot_profile(),
            )
            if self.app_service.start_robot_driver(
                launch_config,
                ros_worker_config=self._current_ros_worker_config(),
                ros_worker_callbacks=self._build_ros_worker_callbacks(),
            ):
                self._set_button_running(self.btn_robot_driver, True, "启动机械臂驱动", "停止机械臂驱动", "background-color: lightgreen;")
            else:
                self._set_button_running(self.btn_robot_driver, False, "启动机械臂驱动", "停止机械臂驱动")
        else:
            self.app_service.stop_robot_driver()
            self._set_button_running(self.btn_robot_driver, False, "启动机械臂驱动", "停止机械臂驱动")
            self._stop_ros_worker_if_unused()

    def toggle_teleop(self, checked):
        if checked:
            self.app_service.cancel_home_zone()
            result = self.intent_controller.check_start_teleop()
            if not result.allowed:
                self._show_intent_result(result)
                self._set_button_running(self.btn_teleop, False, "启动遥操作系统", "停止遥操作系统")
                return
            input_type = self._selected_input_type()
            ip = self.ip_input.text().strip()
            gripper_type = self._selected_gripper_type()
            joy_profile = self._selected_joy_profile()
            mediapipe_profile = self._selected_mediapipe_camera_profile()
            mediapipe_topic = self._selected_mediapipe_topic()
            mediapipe_depth_topic = self._selected_mediapipe_depth_topic()
            mediapipe_camera_info_topic = self._selected_mediapipe_camera_info_topic()
            mediapipe_camera_driver = str(mediapipe_profile.get("driver", "realsense")).strip() or "realsense"
            mediapipe_camera_serial = str(mediapipe_profile.get("serial", "")).strip()
            mediapipe_enable_depth = input_type == "mediapipe"
            mediapipe_use_sdk_camera = True

            if input_type == "mediapipe":
                if not self._camera_options_loaded:
                    self.refresh_camera_devices(log_result=False)
                    mediapipe_profile = self._selected_mediapipe_camera_profile()
                    mediapipe_camera_driver = str(mediapipe_profile.get("driver", "realsense")).strip() or "realsense"
                    mediapipe_camera_serial = str(mediapipe_profile.get("serial", "")).strip()
                same_source_count = self._count_sdk_cameras_by_source(mediapipe_camera_driver)
                if same_source_count > 1 and not mediapipe_camera_serial:
                    QMessageBox.warning(
                        self,
                        "手势相机未锁定",
                        (
                            f"当前检测到 {same_source_count} 台 {mediapipe_camera_driver} 相机，"
                            "但手势相机没有绑定到具体序列号。\n"
                            "为避免影响其它相机，请先在“手势识别输入相机”中选择具体设备后再启动。"
                        ),
                    )
                    self._set_button_running(self.btn_teleop, False, "启动遥操作系统", "停止遥操作系统")
                    return

            if input_type == "mediapipe" and self.runtime_facade.is_process_running("data_collector"):
                mediapipe_option = self._selected_camera_option(
                    self.mediapipe_camera_combo,
                    self._default_mediapipe_camera_source(),
                    self._default_mediapipe_camera_model(),
                )
                global_option = self._selected_camera_option(
                    self.global_camera_source_combo,
                    self.gui_settings.default_global_camera_source,
                    self.gui_settings.default_collector_global_camera_model,
                )
                wrist_option = self._selected_camera_option(
                    self.wrist_camera_source_combo,
                    self.gui_settings.default_wrist_camera_source,
                    self.gui_settings.default_collector_wrist_camera_model,
                )
                if self._camera_option_matches_device(
                    mediapipe_option,
                    source=str(global_option.get("source", "")).strip().lower(),
                    model=str(global_option.get("model", "")).strip().lower(),
                    serial=str(global_option.get("serial", "")).strip(),
                ) or self._camera_option_matches_device(
                    mediapipe_option,
                    source=str(wrist_option.get("source", "")).strip().lower(),
                    model=str(wrist_option.get("model", "")).strip().lower(),
                    serial=str(wrist_option.get("serial", "")).strip(),
                ):
                    QMessageBox.warning(
                        self,
                        "相机占用冲突",
                        "手势识别相机与当前采集节点使用的相机冲突。\n请为手势识别切换到未被采集占用的相机后再启动。",
                    )
                    self._set_button_running(self.btn_teleop, False, "启动遥操作系统", "停止遥操作系统")
                    return

            if self.runtime_facade.is_process_running("robot_driver"):
                self.log("检测到机械臂 ROS2 驱动已独立启动，启动遥操作前先停止独立驱动实例。")
                self.app_service.stop_robot_driver()

            self.log(
                "准备启动遥操作系统: "
                f"input_type={input_type}, gripper_type={gripper_type}, robot_ip={ip}, "
                f"joy_profile={joy_profile}, mediapipe_input_topic={mediapipe_topic}, "
                f"mediapipe_depth_topic={mediapipe_depth_topic}, "
                f"mediapipe_camera_info_topic={mediapipe_camera_info_topic}, "
                f"mediapipe_camera_driver={mediapipe_camera_driver}, "
                f"mediapipe_enable_depth={mediapipe_enable_depth}, "
                f"mediapipe_use_sdk_camera={mediapipe_use_sdk_camera}"
            )
            launch_config = TeleopLaunchConfig(
                robot_ip=ip,
                reverse_ip=self._selected_reverse_ip(),
                ur_type=self._selected_ur_type(),
                input_type=input_type,
                gripper_type=gripper_type,
                joy_profile=joy_profile,
                mediapipe_input_topic=mediapipe_topic,
                mediapipe_depth_topic=mediapipe_depth_topic,
                mediapipe_camera_info_topic=mediapipe_camera_info_topic,
                mediapipe_camera_driver=mediapipe_camera_driver,
                mediapipe_camera_serial_number=mediapipe_camera_serial,
                mediapipe_enable_depth=mediapipe_enable_depth,
                mediapipe_show_debug_window=True,
                mediapipe_use_sdk_camera=mediapipe_use_sdk_camera,
                robot_profile=self._selected_robot_profile(),
            )
            if self.app_service.start_teleop(
                launch_config,
                ros_worker_config=self._current_ros_worker_config(),
                ros_worker_callbacks=self._build_ros_worker_callbacks(),
            ):
                self._clear_active_teleop_camera_binding()
                self.orchestrator.notify_teleop_started()
                self._set_button_running(self.btn_teleop, True, "启动遥操作系统", "停止遥操作系统", "background-color: lightgreen; font-weight: bold;")
            else:
                self._clear_active_teleop_camera_binding()
                self._set_button_running(self.btn_teleop, False, "启动遥操作系统", "停止遥操作系统")
                return

            if input_type == "mediapipe":
                camera_name = str(mediapipe_profile.get("name", "d435")).upper()
                QMessageBox.information(
                    self,
                    "MediaPipe 提示",
                    (
                        "当前已选择 mediapipe 输入。\n\n"
                        f"相机: `{camera_name}`\n"
                        "MediaPipe 将直接作为识别算法输入。"
                    ),
                )
            elif input_type == "quest3":
                QMessageBox.information(
                    self,
                    "Quest3 提示",
                    (
                        "当前已选择 Quest3 输入。\n\n"
                        "请确认 Quest bridge 正在运行，并且头显已进入 Quest WebXR 控制页面。"
                    ),
                )

            QMessageBox.information(self, "操作提示", "遥操作系统已启动！\n\n请不要忘记按示教器的【程序运行播放键】。")
        else:
            self.app_service.stop_teleop()
            self.orchestrator.notify_teleop_stopped()
            self._set_button_running(self.btn_teleop, False, "启动遥操作系统", "停止遥操作系统")
            self._set_button_running(self.btn_robot_driver, False, "启动机械臂驱动", "停止机械臂驱动")
            self._refresh_runtime_status()

    def toggle_data_collector(self, checked):
        if checked:
            out_path = self._selected_record_output_path()
            collector_ee_type = self._selected_collector_end_effector_type()
            global_camera_source, wrist_camera_source = self._selected_collector_camera_sources()
            global_camera_serial, wrist_camera_serial = self._selected_collector_camera_serial_numbers()
            if (
                global_camera_source == wrist_camera_source
                and global_camera_serial
                and wrist_camera_serial
                and global_camera_serial == wrist_camera_serial
            ):
                QMessageBox.warning(self, "相机配置冲突", "全局与手部相机选择了同一个物理相机，请为两路视角选择不同设备。")
                self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
                return
            try:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                QMessageBox.warning(self, "输出路径无效", f"无法创建 HDF5 输出目录:\n{exc}")
                self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
                return
            try:
                self.runtime_facade.ensure_camera_availability(
                    requester="collector",
                    requested_sources=[global_camera_source, wrist_camera_source],
                    requested_serial_numbers=[global_camera_serial, wrist_camera_serial],
                    context=self._camera_runtime_context(),
                )
            except HardwareConflictError as exc:
                QMessageBox.warning(self, "相机占用冲突", "当前采集将直接占用相机 SDK，不能与同一硬件的 ROS2 驱动同时运行。\n\n" + str(exc))
                self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
                return

            self.log(
                "准备启动采集节点: "
                f"end_effector_type={collector_ee_type}, output_path={out_path}, "
                f"global_camera_source={global_camera_source}, wrist_camera_source={wrist_camera_source}"
            )

            launch_config = CollectorLaunchConfig(
                robot_profile=self._selected_robot_profile(),
                output_path=out_path,
                global_camera_source=global_camera_source,
                wrist_camera_source=wrist_camera_source,
                global_camera_serial_number=global_camera_serial,
                wrist_camera_serial_number=wrist_camera_serial,
                global_camera_enable_depth=self._camera_enable_depth(global_camera_source),
                wrist_camera_enable_depth=self._camera_enable_depth(wrist_camera_source),
                end_effector_type=collector_ee_type,
            )

            if not self.app_service.start_data_collector(
                launch_config,
                ros_worker_config=self._current_ros_worker_config(),
                ros_worker_callbacks=self._build_ros_worker_callbacks(),
            ):
                self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
                return

            self._set_button_running(self.btn_collector, True, "启动采集节点", "停止采集节点", "background-color: lightgreen;")
            self._sync_preview_pipeline()
        else:
            self.app_service.stop_data_collector()
            self._set_button_running(self.btn_collector, False, "启动采集节点", "停止采集节点")
            self._stop_ros_worker_if_unused()

    def start_ros_worker(self):
        self.app_service.ensure_ros_worker(
            self._current_ros_worker_config(),
            self._build_ros_worker_callbacks(),
        )

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

            global_source, wrist_source = self._selected_inference_camera_sources()
            global_serial, wrist_serial = self._selected_inference_camera_serial_numbers()
            try:
                self.runtime_facade.ensure_camera_availability(
                    requester="inference",
                    requested_sources=[global_source, wrist_source],
                    requested_serial_numbers=[global_serial, wrist_serial],
                    context=self._camera_runtime_context(),
                    require_distinct_views=True,
                )
            except HardwareConflictError as exc:
                QMessageBox.warning(self, "相机占用冲突", str(exc))
                self._set_inference_button_running(False)
                return

            self._ensure_ros_worker_for_inference()
            launch_config = self._current_inference_launch_config(
                checkpoint_dir=checkpoint_dir,
                task_name=task_name,
                embedding_path=embedding_path,
                global_source=global_source,
                wrist_source=wrist_source,
            )
            self._clear_inference_preview_frames()
            self.inference_service.start_inference(
                launch_config,
                self._build_inference_worker_callbacks(),
                parent=self,
            )
            self.orchestrator.clear_estop()
            self.orchestrator.notify_inference_ready(True)
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
            self.app_service.cancel_home_zone()
            if not self._inference_running():
                QMessageBox.warning(self, "推理未运行", "请先启动推理，再开始执行任务。")
                self._set_inference_execute_button_running(False)
                return
            result = self.intent_controller.check_enable_inference_execution()
            if not result.allowed:
                self._show_intent_result(result)
                self._set_inference_execute_button_running(False)
                return
            if not self.runtime_facade.is_process_running("robot_driver"):
                QMessageBox.warning(self, "机械臂未就绪", "请先启动机械臂驱动，再执行推理任务。")
                self._set_inference_execute_button_running(False)
                return

            self.app_service.ensure_ros_worker(
                self._current_ros_worker_config(),
                self._build_ros_worker_callbacks(),
            )
            self._queue_inference_execution_start()
            return

        was_pending_start = self._pending_inference_execution_start
        self._clear_pending_inference_execution_start()
        self.app_service.disable_inference_execution()
        self._finalize_inference_action_log(
            reason_label="停止执行",
            stop_reason="stop_execution",
        )
        if was_pending_start and not self.app_service.inference_execution_enabled():
            self.log("已取消等待开始执行。")
        self.orchestrator.notify_inference_execution(False)
        self._set_inference_execute_button_running(False)
        self._set_inference_execute_status("未使能", "#6c757d")
        self.btn_inference_estop.setEnabled(self._inference_running())
        self._refresh_runtime_status()

    def emergency_stop_inference_execution(self) -> None:
        self._clear_pending_inference_execution_start()
        self.app_service.emergency_stop_inference_execution()
        self._finalize_inference_action_log(
            reason_label="急停",
            stop_reason="estop",
        )
        self.orchestrator.notify_estop(True)
        self._set_inference_execute_button_running(False)
        self._set_inference_execute_status("已急停", "#c92a2a")
        self.btn_inference_estop.setEnabled(self._inference_running())
        self._refresh_runtime_status()

    def stop_inference(self) -> None:
        self._clear_pending_inference_execution_start()
        if self.inference_worker is None:
            self._finalize_inference_action_log(
                reason_label="未启动但清理",
                stop_reason="cleanup_without_worker",
            )
            self._set_inference_button_running(False)
            self._set_inference_status("未启动", "#6c757d")
            self._set_inference_execute_button_running(False)
            self._set_inference_execute_status("未使能", "#6c757d")
            self.btn_execute_inference.setEnabled(False)
            self.btn_inference_estop.setEnabled(False)
            return

        self.app_service.disable_inference_execution()
        self.orchestrator.notify_inference_execution(False)
        self.orchestrator.notify_inference_ready(False)
        self.orchestrator.clear_estop()
        if not self.inference_service.stop_inference(timeout_ms=4000):
            self.log("推理线程未在 4s 内退出，继续等待资源回收。")
        else:
            self._finalize_inference_action_log(
                reason_label="手动停止",
                stop_reason="manual_stop_inference",
            )
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
        timing: Optional[dict[str, object]] = None
        if isinstance(action, InferenceActionSample):
            timing = {
                "cycle_compute_ms": float(action.cycle_compute_ms),
                "camera_fetch_ms": float(action.camera_fetch_ms),
                "preprocess_ms": float(action.preprocess_ms),
                "robot_state_ms": float(action.robot_state_ms),
                "policy_call_ms": float(action.policy_call_ms),
                "is_replan_step": bool(action.is_replan_step),
                "plan_step_idx": int(action.plan_step_idx),
                "replan_every": int(action.replan_every),
            }
            action_array = np.asarray(action.action, dtype=np.float32).reshape(-1)
        else:
            action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        self.inference_action_output.setPlainText(format_action(action_array))
        self.app_service.update_inference_action_command(action_array)
        if self._pending_inference_execution_start:
            should_start_execution = timing is None
            if timing is not None:
                should_start_execution = bool(timing.get("is_replan_step", False))
                if not should_start_execution:
                    try:
                        should_start_execution = int(timing.get("plan_step_idx", -1)) <= 0
                    except Exception:
                        should_start_execution = False
            if should_start_execution:
                self._activate_pending_inference_execution()
        self.app_service.record_inference_action_sample(action_array, timing=timing)

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
        self._clear_pending_inference_execution_start()
        self.log(f"推理线程出错: {message}")
        self._finalize_inference_action_log(
            reason_label="推理报错",
            stop_reason="inference_error",
        )
        self.inference_action_output.setPlainText("推理失败，详见下方日志输出。")
        self.app_service.emergency_stop_inference_execution()
        self.orchestrator.notify_estop(True)
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
        self._clear_pending_inference_execution_start()
        self._finalize_inference_action_log(
            reason_label="线程结束",
            stop_reason="worker_finished",
        )
        self.app_service.disable_inference_execution()
        self.orchestrator.notify_inference_execution(False)
        self.orchestrator.notify_inference_ready(False)
        self.orchestrator.clear_estop()
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
        self._request_recording(True)

    def stop_record(self):
        self._request_recording(False)

    def discard_current_demo(self):
        if not self.runtime_facade.is_process_running("data_collector"):
            QMessageBox.warning(self, "警告", "请先启动采集节点，再执行 Demo 弃用。")
            return
        if self.app_service.is_recording():
            QMessageBox.warning(self, "警告", "当前正在录制，请先停止录制后再弃用 Demo。")
            return

        demo_name = self.lbl_demo_status.text().strip()
        if not demo_name or demo_name == "无 (未录制)":
            QMessageBox.information(self, "提示", "当前没有可弃用的 Demo。")
            return

        reply = QMessageBox.question(
            self,
            "确认弃用 Demo",
            f"将永久删除最近一个 Demo：{demo_name}\n\n该操作不可恢复，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        if not self.app_service.discard_last_demo():
            QMessageBox.warning(self, "弃用失败", "ROS 监听器尚未就绪，无法调用弃用服务。")
            return

        self.log(f"已请求弃用 Demo: {demo_name}")
        self._refresh_runtime_status()

    def go_home(self):
        self._request_commander_motion("home")

    def go_home_zone(self):
        self._request_commander_motion("home_zone")

    def set_home_from_current(self):
        if not self.app_service.has_ros_worker():
            QMessageBox.warning(self, "警告", "请先启动机械臂驱动、遥操作系统或预览窗口，并接收实时关节数据，再设置 Home 点。")
            return

        joints = self.app_service.current_joint_positions()
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

        applied_joints = self.app_service.set_home_from_current()
        if applied_joints is None:
            QMessageBox.warning(self, "警告", "当前关节状态无效(需要 6 个关节角)，无法设置 Home 点。")
            return
        saved = self._save_home_override(applied_joints)
        if saved:
            self.log(f"已持久化 Home 点到独立覆盖文件: {saved}\nHome joints: {joints_str}")

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
            if self.vision_panel is not None:
                self.vision_panel.reset_record_stats()
            if self.preview_window:
                self.preview_window.reset_record_stats()

    @Slot(int, str, float)
    def update_main_record_stats(self, frames, time_str, realtime_fps):
        frames_str = "N/A" if frames is None or int(frames) < 0 else str(int(frames))
        fps_text = f"{float(realtime_fps):.2f} Hz" if realtime_fps is not None else "N/A"
        self.lbl_main_record_stats.setText(f"录制时长: {time_str} | 已录制帧数: {frames_str} | 实时录制帧率: {fps_text}")
        self.lbl_main_record_stats.setStyleSheet("font-weight: bold; color: red;")

    @Slot(int)
    def _on_preview_window_finished(self, _result: int):
        if self.preview_window is not None:
            self.preview_window.set_preview_source("已关闭")
            self.preview_window.clear_images()
        self._stop_ros_worker_if_unused()
        self._sync_preview_pipeline()

    def open_preview_window(self):
        if not self.app_service.has_ros_worker():
            self.start_ros_worker()
            self.log("已启动独立 ROS 监听器，用于状态与录制信息同步。")

        if self.preview_window is None:
            self.preview_window = CameraPreviewWindow(self)
            self.preview_window.finished.connect(self._on_preview_window_finished)
            self._preview_window_api_connected = False
        if not self.app_service.is_recording():
            self.preview_window.reset_record_stats()

        if self.preview_api_worker is not None and not self._preview_window_api_connected:
            self.preview_api_worker.global_image_signal.connect(self.preview_window.update_global_image)
            self.preview_api_worker.wrist_image_signal.connect(self.preview_window.update_wrist_image)
            self._preview_window_api_connected = True

        self.preview_window.show()
        self.preview_window.raise_()
        self.preview_window.activateWindow()
        self._sync_preview_pipeline()

    def open_hdf5_viewer(self):
        if self.app_service.is_recording():
            QMessageBox.warning(self, "警告", "当前正在录制数据，为了防止文件损坏，请在【停止录制】后再进行 HDF5 预览。")
            return

        viewer = HDF5ViewerDialog(self._selected_record_output_path(), parent=self)
        viewer.exec()

    def closeEvent(self, event):
        self._persist_gui_settings_snapshot(log_errors=False)
        self._shutdown()
        event.accept()
