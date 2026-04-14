import cv2
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
)


class CameraPreviewWindow(QDialog):
    preview_record_toggle_requested = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("实时预览与状态监视器")
        self.resize(1150, 750)
        self.show_cropped_only = True

        main_layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        self.crop_cb = QCheckBox("仅显示中心裁切区域 (与录入数据集画面完全一致)")
        self.crop_cb.setStyleSheet("font-weight: bold; color: #d32f2f;")
        self.crop_cb.setChecked(True)
        self.crop_cb.toggled.connect(self.on_crop_toggled)
        top_layout.addWidget(self.crop_cb)
        top_layout.addStretch()

        self.lbl_preview_source = QLabel("预览源: 无活动图像源")
        self.lbl_preview_source.setStyleSheet("font-weight: bold; color: #555;")
        top_layout.addWidget(self.lbl_preview_source)
        main_layout.addLayout(top_layout)

        status_layout = QHBoxLayout()
        self.lbl_dataset_record_status = QLabel("采集状态: 未录制 | 时长: 00:00 | 已录制帧数: 0 | 实时录制帧率: 0.00 Hz")
        self.lbl_dataset_record_status.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;")
        status_layout.addWidget(self.lbl_dataset_record_status, stretch=1)

        self.lbl_preview_record_status = QLabel("预览录屏: 未录制")
        self.lbl_preview_record_status.setStyleSheet("font-weight: bold; font-size: 14px; color: #555;")
        status_layout.addWidget(self.lbl_preview_record_status, stretch=1)
        main_layout.addLayout(status_layout)

        record_controls_layout = QHBoxLayout()
        record_controls_layout.addWidget(QLabel("预览录屏目标:"))
        self.preview_record_target_combo = QComboBox()
        self.preview_record_target_combo.addItem("全局画面", "global")
        self.preview_record_target_combo.addItem("手部画面", "wrist")
        self.preview_record_target_combo.addItem("全局 + 手部", "both")
        self.preview_record_target_combo.setCurrentIndex(2)
        record_controls_layout.addWidget(self.preview_record_target_combo)

        record_controls_layout.addWidget(QLabel("录屏尺寸:"))
        self.preview_record_frame_mode_combo = QComboBox()
        self.preview_record_frame_mode_combo.addItem("源大小", "source")
        self.preview_record_frame_mode_combo.addItem("中心裁切正方形", "square")
        self.preview_record_frame_mode_combo.setCurrentIndex(0)
        record_controls_layout.addWidget(self.preview_record_frame_mode_combo)

        self.btn_preview_record = QPushButton("开始预览录屏")
        self.btn_preview_record.setCheckable(True)
        self.btn_preview_record.toggled.connect(self._emit_preview_record_toggle)
        record_controls_layout.addWidget(self.btn_preview_record)
        record_controls_layout.addStretch()
        main_layout.addLayout(record_controls_layout)

        content_layout = QHBoxLayout()
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
        content_layout.addLayout(cameras_layout, stretch=3)

        self.text_robot_state = QTextEdit()
        self.text_robot_state.setReadOnly(True)
        self.text_robot_state.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; background-color: #fcfcfc;")
        self.text_robot_state.setText("等待机器人状态数据...")
        self.text_robot_state.setMinimumWidth(300)
        content_layout.addWidget(self.text_robot_state, stretch=1)
        main_layout.addLayout(content_layout)

    def on_crop_toggled(self, checked):
        self.show_cropped_only = checked

    def _emit_preview_record_toggle(self, checked: bool) -> None:
        self.preview_record_toggle_requested.emit(bool(checked), self.selected_preview_record_target())

    def selected_preview_record_target(self) -> str:
        value = self.preview_record_target_combo.currentData()
        normalized = str(value).strip().lower() if value is not None else "both"
        return normalized if normalized in {"global", "wrist", "both"} else "both"

    def selected_preview_record_frame_mode(self) -> str:
        value = self.preview_record_frame_mode_combo.currentData()
        normalized = str(value).strip().lower() if value is not None else "source"
        return normalized if normalized in {"source", "square"} else "source"

    def process_image(self, cv_img):
        if cv_img is None or len(cv_img.shape) < 2:
            return cv_img

        height, width = cv_img.shape[:2]
        side = min(height, width)
        x0 = (width - side) // 2
        y0 = (height - side) // 2

        if self.show_cropped_only:
            return cv_img[y0:y0 + side, x0:x0 + side].copy()

        overlay = cv_img.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        masked = cv2.addWeighted(overlay, 0.5, cv_img, 0.5, 0)
        masked[y0:y0 + side, x0:x0 + side] = cv_img[y0:y0 + side, x0:x0 + side]
        cv2.rectangle(masked, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
        return masked

    def cv2_to_qpixmap(self, cv_img):
        try:
            processed = self.process_image(cv_img)
            rgb_img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_img.shape
            bytes_per_line = channels * width
            qimg = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)
        except Exception:
            return QPixmap()

    @Slot(object)
    def update_global_image(self, cv_img):
        pixmap = self.cv2_to_qpixmap(cv_img)
        if not pixmap.isNull():
            self.global_label.setPixmap(pixmap.scaled(self.global_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(object)
    def update_wrist_image(self, cv_img):
        pixmap = self.cv2_to_qpixmap(cv_img)
        if not pixmap.isNull():
            self.wrist_label.setPixmap(pixmap.scaled(self.wrist_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(str)
    def update_robot_state_str(self, text):
        self.text_robot_state.setText(text)

    def set_preview_source(self, source_text: str) -> None:
        text = str(source_text).strip() or "未知"
        self.lbl_preview_source.setText(f"预览源: {text}")

    def clear_images(self) -> None:
        self.global_label.clear()
        self.wrist_label.clear()
        self.global_label.setText("无画面")
        self.wrist_label.setText("无画面")

    @Slot(int, str, float)
    def update_dataset_record_stats(self, frames, time_str, realtime_fps):
        frames_str = "N/A" if frames is None or int(frames) < 0 else str(int(frames))
        fps_text = f"{float(realtime_fps):.2f} Hz" if realtime_fps is not None else "N/A"
        self.lbl_dataset_record_status.setText(f"采集状态: 录制中 | 时长: {time_str} | 已录制帧数: {frames_str} | 实时录制帧率: {fps_text}")
        self.lbl_dataset_record_status.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")

    def reset_dataset_record_stats(self):
        self.lbl_dataset_record_status.setText("采集状态: 未录制 | 时长: 00:00 | 已录制帧数: 0 | 实时录制帧率: 0.00 Hz")
        self.lbl_dataset_record_status.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;")

    @Slot(str)
    def update_preview_recording_status(self, text: str) -> None:
        message = str(text).strip() or "预览录屏: 未录制"
        color = "#555"
        if "录制中" in message:
            color = "#c92a2a"
        elif "已停止" in message:
            color = "#1d6f42"
        self.lbl_preview_record_status.setText(message)
        self.lbl_preview_record_status.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {color};")

    def set_preview_recording_state(self, active: bool, *, output_dir: str = "") -> None:
        self.btn_preview_record.blockSignals(True)
        self.btn_preview_record.setChecked(bool(active))
        self.btn_preview_record.blockSignals(False)
        self.btn_preview_record.setText("停止预览录屏" if active else "开始预览录屏")
        self.preview_record_target_combo.setEnabled(not active)
        self.preview_record_frame_mode_combo.setEnabled(not active)
        tooltip = str(output_dir).strip()
        if tooltip:
            self.btn_preview_record.setToolTip(tooltip)
            self.lbl_preview_record_status.setToolTip(tooltip)
        else:
            self.btn_preview_record.setToolTip("")
            self.lbl_preview_record_status.setToolTip("")
        if active:
            if "录制中" not in self.lbl_preview_record_status.text():
                self.update_preview_recording_status("预览录屏: 录制中 | 等待画面...")
        elif "已停止" not in self.lbl_preview_record_status.text():
            self.update_preview_recording_status("预览录屏: 未录制")

    @Slot(int, str, float)
    def update_record_stats(self, frames, time_str, realtime_fps):
        self.update_dataset_record_stats(frames, time_str, realtime_fps)

    def reset_record_stats(self):
        self.reset_dataset_record_stats()
