import cv2
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class _ImageViewport(QLabel):
    def __init__(self, empty_text: str, parent=None):
        super().__init__(empty_text, parent)
        self._base_text = str(empty_text)
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(260)
        self.setStyleSheet(
            "background-color: #111827; color: #8fb0ff; font-size: 16px; font-weight: 600; "
            "border: 1px solid #2c3b52; border-radius: 16px; padding: 12px;"
        )

    def set_frame_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._refresh()

    def clear_frame(self) -> None:
        self._pixmap = QPixmap()
        self.clear()
        self.setText(self._base_text)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class StudioVisionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._show_cropped_only = True
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("studioVisionPanel")
        self.setStyleSheet(
            "#studioVisionPanel { background: #f6f8fb; }"
            "QWidget#visionTopBar { background: #ffffff; border: 1px solid #d9e3ef; border-radius: 16px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        top_bar_widget = QWidget(self)
        top_bar_widget.setObjectName("visionTopBar")
        top_bar = QHBoxLayout(top_bar_widget)
        top_bar.setContentsMargins(14, 12, 14, 12)
        top_bar.setSpacing(10)
        self.crop_cb = QCheckBox("仅显示中心裁切区域")
        self.crop_cb.setChecked(True)
        self.crop_cb.setStyleSheet(
            "QCheckBox { color: #7a4b00; font-weight: 700; padding: 6px 12px; "
            "background: #fff5df; border: 1px solid #f3d39a; border-radius: 12px; }"
            "QCheckBox::indicator { width: 16px; height: 16px; }"
            "QCheckBox::indicator:unchecked { background: #ffffff; border: 1px solid #c7b68f; border-radius: 4px; }"
            "QCheckBox::indicator:checked { background: #f59e0b; border: 1px solid #d38807; border-radius: 4px; }"
        )
        self.crop_cb.toggled.connect(self.on_crop_toggled)
        top_bar.addWidget(self.crop_cb)

        self.lbl_preview_source = QLabel("预览源: 无活动图像源")
        self.lbl_preview_source.setAlignment(Qt.AlignCenter)
        self.lbl_preview_source.setStyleSheet(self._badge_style("#4c6a92", alpha=0.10))
        top_bar.addWidget(self.lbl_preview_source)
        top_bar.addStretch(1)

        self.lbl_record_status = QLabel("状态: 未录制 | 时长: 00:00 | 已录制帧数: 0 | 实时录制帧率: 0.00 Hz")
        self.lbl_record_status.setAlignment(Qt.AlignCenter)
        self.lbl_record_status.setStyleSheet(self._badge_style("#4c6a92", alpha=0.08))
        top_bar.addWidget(self.lbl_record_status)
        layout.addWidget(top_bar_widget)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        global_group = QGroupBox("【全局相机 (Agent View)】")
        global_layout = QVBoxLayout(global_group)
        global_layout.setContentsMargins(12, 16, 12, 12)
        self.global_label = _ImageViewport("1280x720\n\n等待推流...")
        global_layout.addWidget(self.global_label)

        wrist_group = QGroupBox("【手部相机 (Eye-in-Hand)】")
        wrist_layout = QVBoxLayout(wrist_group)
        wrist_layout.setContentsMargins(12, 16, 12, 12)
        self.wrist_label = _ImageViewport("640x480\n\n等待推流...")
        wrist_layout.addWidget(self.wrist_label)

        group_style = (
            "QGroupBox { font-size: 14px; font-weight: 700; border: 1px solid #d7e1ee; "
            "border-radius: 18px; margin-top: 12px; background: #ffffff; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; color: #18324d; }"
        )
        global_group.setStyleSheet(group_style)
        wrist_group.setStyleSheet(group_style)

        grid.addWidget(global_group, 0, 0)
        grid.addWidget(wrist_group, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, stretch=1)

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
            f"color: {color}; font-weight: 700; padding: 6px 12px; "
            f"border-radius: 13px; background: {background}; border: 1px solid {border};"
        )

    @Slot(bool)
    def on_crop_toggled(self, checked: bool) -> None:
        self._show_cropped_only = bool(checked)

    def process_image(self, cv_img):
        if cv_img is None or len(cv_img.shape) < 2:
            return cv_img

        height, width = cv_img.shape[:2]
        side = min(height, width)
        x0 = (width - side) // 2
        y0 = (height - side) // 2

        if self._show_cropped_only:
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
            return QPixmap.fromImage(qimg.copy())
        except Exception:
            return QPixmap()

    @Slot(object)
    def update_global_image(self, cv_img):
        pixmap = self.cv2_to_qpixmap(cv_img)
        if not pixmap.isNull():
            self.global_label.set_frame_pixmap(pixmap)

    @Slot(object)
    def update_wrist_image(self, cv_img):
        pixmap = self.cv2_to_qpixmap(cv_img)
        if not pixmap.isNull():
            self.wrist_label.set_frame_pixmap(pixmap)

    def set_preview_source(self, source_text: str) -> None:
        text = str(source_text).strip() or "未知"
        self.lbl_preview_source.setText(f"预览源: {text}")
        if "无活动" in text:
            color = "#6c757d"
        elif "未就绪" in text or "连接中" in text:
            color = "#e67700"
        else:
            color = "#2f6fed"
        self.lbl_preview_source.setStyleSheet(self._badge_style(color, alpha=0.10))

    def clear_images(self) -> None:
        self.global_label.clear_frame()
        self.wrist_label.clear_frame()

    @Slot(int, str, float)
    def update_record_stats(self, frames, time_str, realtime_fps):
        frames_str = "N/A" if frames is None or int(frames) < 0 else str(int(frames))
        fps_text = f"{float(realtime_fps):.2f} Hz" if realtime_fps is not None else "N/A"
        self.lbl_record_status.setText(
            f"状态: 录制中 | 时长: {time_str} | 已录制帧数: {frames_str} | 实时录制帧率: {fps_text}"
        )
        self.lbl_record_status.setStyleSheet(self._badge_style("#d94841", alpha=0.12))

    def reset_record_stats(self):
        self.lbl_record_status.setText("状态: 未录制 | 时长: 00:00 | 已录制帧数: 0 | 实时录制帧率: 0.00 Hz")
        self.lbl_record_status.setStyleSheet(self._badge_style("#4c6a92", alpha=0.08))
