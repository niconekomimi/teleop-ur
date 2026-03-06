import os
from pathlib import Path

import h5py
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QMessageBox,
)


class HDF5ViewerDialog(QDialog):
    def __init__(self, initial_hdf5_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDF5 数据集高级回放器")
        self.resize(1150, 750)

        self.hdf5_path = initial_hdf5_path
        self.file_handle = None
        self.current_demo_group = None
        self.current_demo_name = None
        self.demo_names = []
        self.pending_crop_ranges = {}
        self.pending_deleted_demos = set()

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

        self.btn_delete_current = QPushButton("删除当前 demo")
        self.btn_delete_current.clicked.connect(self.delete_current_demo)
        top_layout.addWidget(self.btn_delete_current)

        self.btn_save_changes = QPushButton("保存修改")
        self.btn_save_changes.setStyleSheet("font-weight: bold; background-color: #ffe8a1;")
        self.btn_save_changes.clicked.connect(self.save_pending_changes)
        top_layout.addWidget(self.btn_save_changes)

        self.lbl_pending_state = QLabel("无待保存修改")
        self.lbl_pending_state.setStyleSheet("color: #555;")
        top_layout.addWidget(self.lbl_pending_state)

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
        self.speed_combo.addItems(["0.5x (5 Hz)", "1.0x (10 Hz)", "2.0x (20 Hz)", "4.0x (40 Hz)", "8.0x (80 Hz)"])
        self.speed_combo.setCurrentText("1.0x (10 Hz)")
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        ctrl_layout.addWidget(self.speed_combo)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.sliderPressed.connect(self.pause_playback)
        ctrl_layout.addWidget(self.slider)
        main_layout.addLayout(ctrl_layout)

        edit_layout = QHBoxLayout()
        edit_layout.addWidget(QLabel("裁剪范围(保存后生效):"))
        self.crop_start_spin = QSpinBox()
        self.crop_start_spin.setMinimum(1)
        self.crop_start_spin.setMaximum(1)
        edit_layout.addWidget(self.crop_start_spin)

        edit_layout.addWidget(QLabel("到"))
        self.crop_end_spin = QSpinBox()
        self.crop_end_spin.setMinimum(1)
        self.crop_end_spin.setMaximum(1)
        edit_layout.addWidget(self.crop_end_spin)

        self.btn_apply_crop = QPushButton("设置裁剪")
        self.btn_apply_crop.clicked.connect(self.apply_crop_range)
        edit_layout.addWidget(self.btn_apply_crop)

        self.btn_clear_crop = QPushButton("清除裁剪")
        self.btn_clear_crop.clicked.connect(self.clear_crop_range)
        edit_layout.addWidget(self.btn_clear_crop)

        self.lbl_crop_state = QLabel("当前 demo 无待保存裁剪")
        self.lbl_crop_state.setStyleSheet("color: #555;")
        edit_layout.addWidget(self.lbl_crop_state)
        edit_layout.addStretch()
        main_layout.addLayout(edit_layout)

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
            "HDF5 Files (*.hdf5 *.h5);;All Files (*)",
        )
        if file_path:
            self.open_hdf5_file(file_path)

    def open_hdf5_file(self, path):
        self.pause_playback()
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

        self.hdf5_path = path
        self.lbl_file_path.setText(os.path.basename(path))
        self.setWindowTitle(f"HDF5 数据集高级回放器 - {os.path.basename(path)}")
        self.demo_combo.clear()
        self.demo_names = []
        self.pending_crop_ranges = {}
        self.pending_deleted_demos = set()
        self.current_demo_group = None
        self.current_demo_name = None

        try:
            self.file_handle = h5py.File(path, "r")
            if "data" not in self.file_handle:
                raise KeyError("HDF5 文件中未找到 'data' 根组，格式不符合要求。")

            demos = list(self.file_handle["data"].keys())
            if not demos:
                raise ValueError("数据集中没有任何 demo 序列。")

            self.demo_names = sorted(demos, key=self._demo_sort_key)
            self._refresh_demo_combo()
        except Exception as exc:
            QMessageBox.critical(self, "HDF5 读取错误", str(exc))

    def _demo_sort_key(self, item):
        try:
            return int(str(item).split("_")[1])
        except Exception:
            return 10**9

    def _infer_num_samples(self, demo_group):
        num_samples = int(demo_group.attrs.get("num_samples", 0))
        if num_samples == 0 and "actions" in demo_group:
            num_samples = int(demo_group["actions"].shape[0])
        return num_samples

    def _effective_crop_range(self, demo_name, total=None):
        if total is None:
            if self.file_handle is None:
                return 0, 0
            demo_group = self.file_handle["data"][demo_name]
            total = self._infer_num_samples(demo_group)
        start, end = self.pending_crop_ranges.get(demo_name, (0, total))
        start = max(0, min(int(start), total))
        end = max(start, min(int(end), total))
        return start, end

    def _demo_display_name(self, demo_name):
        suffixes = []
        if demo_name in self.pending_crop_ranges:
            start, end = self.pending_crop_ranges[demo_name]
            suffixes.append(f"裁剪 {start + 1}:{end}")
        if not suffixes:
            return demo_name
        return f"{demo_name} [{' | '.join(suffixes)}]"

    def _refresh_demo_combo(self, selected_demo=None):
        if selected_demo is None:
            selected_demo = self.current_demo_name or self.demo_combo.currentData() or (self.demo_names[0] if self.demo_names else None)

        self.demo_combo.blockSignals(True)
        self.demo_combo.clear()
        for demo_name in self.demo_names:
            self.demo_combo.addItem(self._demo_display_name(demo_name), demo_name)
        if selected_demo is not None:
            index = self.demo_combo.findData(selected_demo)
            if index >= 0:
                self.demo_combo.setCurrentIndex(index)
        self.demo_combo.blockSignals(False)
        self._update_pending_state_label()
        self.load_demo()

    def _update_pending_state_label(self):
        changes = []
        if self.pending_crop_ranges:
            changes.append(f"裁剪 {len(self.pending_crop_ranges)} 个")
        if self.pending_deleted_demos:
            changes.append(f"删除 {len(self.pending_deleted_demos)} 个")
        if not changes:
            self.lbl_pending_state.setText("无待保存修改")
            self.lbl_pending_state.setStyleSheet("color: #555;")
        else:
            self.lbl_pending_state.setText("待保存修改: " + "，".join(changes))
            self.lbl_pending_state.setStyleSheet("color: #b26a00; font-weight: bold;")

        self.btn_delete_current.setEnabled(len(self.demo_names) > 1)

    def _copy_attrs(self, source, target):
        for key, value in source.attrs.items():
            target.attrs[key] = value

    def _copy_group_sliced(self, source_group, target_group, start, end, total):
        self._copy_attrs(source_group, target_group)
        for name, item in source_group.items():
            if isinstance(item, h5py.Group):
                child = target_group.create_group(name)
                self._copy_group_sliced(item, child, start, end, total)
                continue

            kwargs = {
                "dtype": item.dtype,
            }
            if item.compression is not None:
                kwargs["compression"] = item.compression
                if item.compression_opts is not None:
                    kwargs["compression_opts"] = item.compression_opts
            if item.shuffle:
                kwargs["shuffle"] = True
            if item.fletcher32:
                kwargs["fletcher32"] = True

            if item.shape and int(item.shape[0]) == int(total):
                data = item[start:end]
            else:
                data = item[()]

            target_dataset = target_group.create_dataset(name, data=data, **kwargs)
            self._copy_attrs(item, target_dataset)

    def save_pending_changes(self):
        if not self.pending_crop_ranges and not self.pending_deleted_demos:
            QMessageBox.information(self, "提示", "当前没有待保存的裁剪或删除操作。")
            return

        self.pause_playback()
        if not self.demo_names:
            QMessageBox.warning(self, "无法保存", "至少需要保留一个 demo，当前操作会删除全部 demo。")
            return

        selected_demo = self.demo_combo.currentData()
        reply = QMessageBox.question(
            self,
            "确认保存",
            "保存后将实际写回 HDF5 文件，并将剩余 demo 按顺序重新命名为连续编号。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

        source_path = Path(self.hdf5_path)
        temp_path = source_path.with_suffix(source_path.suffix + ".tmp")

        try:
            with h5py.File(source_path, "r") as source_h5, h5py.File(temp_path, "w") as target_h5:
                self._copy_attrs(source_h5, target_h5)
                source_data = source_h5["data"]
                target_data = target_h5.create_group("data")
                self._copy_attrs(source_data, target_data)

                source_demo_names = sorted(source_data.keys(), key=self._demo_sort_key)
                kept_demo_names = [name for name in self.demo_names if name in source_data]
                if not kept_demo_names:
                    raise ValueError("保存结果为空，已中止写回以避免删除全部 demo。")

                unknown_demo_names = [name for name in kept_demo_names if name not in source_demo_names]
                if unknown_demo_names:
                    raise ValueError(f"待保留 demo 不存在于源文件: {unknown_demo_names}")

                for index, source_demo_name in enumerate(kept_demo_names):
                    source_demo = source_data[source_demo_name]
                    new_demo_name = f"demo_{index}"
                    total = self._infer_num_samples(source_demo)
                    start, end = self._effective_crop_range(source_demo_name, total)

                    target_demo = target_data.create_group(new_demo_name)
                    self._copy_group_sliced(source_demo, target_demo, start, end, total)
                    target_demo.attrs["num_samples"] = int(max(0, end - start))

            os.replace(temp_path, source_path)
        except Exception as exc:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            QMessageBox.critical(self, "保存失败", f"保存对 HDF5 的修改失败:\n{exc}")
            self.open_hdf5_file(str(source_path))
            return

        QMessageBox.information(self, "保存完成", "裁剪和删除修改已经写回 HDF5 文件，剩余 demo 已按顺序连续重命名。")
        self.open_hdf5_file(str(source_path))

    def delete_current_demo(self):
        demo_name = self.demo_combo.currentData()
        if not demo_name:
            return

        if len(self.demo_names) <= 1:
            QMessageBox.warning(self, "无法删除", "至少需要保留一个 demo。")
            return

        reply = QMessageBox.question(
            self,
            "确认删除当前 demo",
            f"当前选中的 {demo_name} 会从列表中移除，并在点击“保存修改”后真正写回文件。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        current_index = self.demo_combo.currentIndex()
        self.pending_deleted_demos.add(demo_name)
        self.pending_crop_ranges.pop(demo_name, None)
        self.demo_names = [name for name in self.demo_names if name != demo_name]

        if not self.demo_names:
            self._update_pending_state_label()
            return

        next_index = min(current_index, len(self.demo_names) - 1)
        next_demo = self.demo_names[next_index]
        self._refresh_demo_combo(next_demo)

    def apply_crop_range(self):
        demo_name = self.demo_combo.currentData()
        if not demo_name or self.file_handle is None:
            return

        demo_group = self.file_handle["data"][demo_name]
        total = self._infer_num_samples(demo_group)
        start = int(self.crop_start_spin.value()) - 1
        end = int(self.crop_end_spin.value())
        if start < 0 or end > total or start >= end:
            QMessageBox.warning(self, "裁剪范围无效", "请确认裁剪起止帧范围有效，且起点小于终点。")
            return

        if start == 0 and end == total:
            self.pending_crop_ranges.pop(demo_name, None)
        else:
            self.pending_crop_ranges[demo_name] = (start, end)
        self._refresh_demo_combo(demo_name)

    def clear_crop_range(self):
        demo_name = self.demo_combo.currentData()
        if not demo_name:
            return
        self.pending_crop_ranges.pop(demo_name, None)
        self._refresh_demo_combo(demo_name)

    def load_demo(self):
        self.pause_playback()
        demo_name = self.demo_combo.currentData()
        if not demo_name or self.file_handle is None:
            return

        self.current_demo_name = str(demo_name)
        self.current_demo_group = self.file_handle["data"][demo_name]
        total_num_samples = self._infer_num_samples(self.current_demo_group)
        crop_start, crop_end = self._effective_crop_range(demo_name)
        num_samples = max(0, crop_end - crop_start)

        self.crop_start_spin.blockSignals(True)
        self.crop_end_spin.blockSignals(True)
        self.crop_start_spin.setMaximum(max(1, total_num_samples))
        self.crop_end_spin.setMaximum(max(1, total_num_samples))
        self.crop_start_spin.setValue(crop_start + 1 if total_num_samples > 0 else 1)
        self.crop_end_spin.setValue(crop_end if total_num_samples > 0 else 1)
        self.crop_start_spin.blockSignals(False)
        self.crop_end_spin.blockSignals(False)

        if demo_name in self.pending_crop_ranges:
            self.lbl_crop_state.setText(f"当前 demo 待裁剪: {crop_start + 1} 到 {crop_end} 帧")
            self.lbl_crop_state.setStyleSheet("color: #b26a00; font-weight: bold;")
        else:
            self.lbl_crop_state.setText("当前 demo 无待保存裁剪")
            self.lbl_crop_state.setStyleSheet("color: #555;")

        if num_samples > 0:
            self.slider.blockSignals(True)
            self.slider.setMaximum(int(num_samples) - 1)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self.update_frame_display()
        else:
            self.lbl_frame_info.setText("空序列")
            self.slider.blockSignals(True)
            self.slider.setMaximum(0)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self.lbl_agent.setPixmap(QPixmap())
            self.lbl_agent.setText("无画面")
            self.lbl_wrist.setPixmap(QPixmap())
            self.lbl_wrist.setText("无画面")
            self.text_state.setText("当前 demo 为空，无法预览。")

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
        text = self.speed_combo.currentText().split()[0].replace("x", "")
        try:
            factor = float(text)
        except Exception:
            factor = 1.0

        interval = int(self.base_interval_ms / factor)
        if self.is_playing:
            self.playback_timer.start(interval)

    def on_play_timeout(self):
        current = self.slider.value()
        if current < self.slider.maximum():
            self.slider.setValue(current + 1)
        else:
            self.pause_playback()

    def step_prev(self):
        self.pause_playback()
        current = self.slider.value()
        if current > 0:
            self.slider.setValue(current - 1)

    def step_next(self):
        self.pause_playback()
        current = self.slider.value()
        if current < self.slider.maximum():
            self.slider.setValue(current + 1)

    def on_slider_changed(self):
        self.update_frame_display()

    def update_frame_display(self):
        if self.current_demo_group is None:
            return

        idx = self.slider.value()
        total = self.slider.maximum() + 1
        crop_start, _ = self._effective_crop_range(self.current_demo_name)
        source_idx = crop_start + idx
        self.lbl_frame_info.setText(f"当前帧: {idx + 1} / {total} | 原始帧: {source_idx + 1}")

        try:
            agent_rgb = self.current_demo_group["obs"]["agentview_rgb"][source_idx]
            wrist_rgb = self.current_demo_group["obs"]["eye_in_hand_rgb"][source_idx]
            height, width, channels = agent_rgb.shape
            bytes_per_line = channels * width

            qimg_agent = QImage(agent_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.lbl_agent.setPixmap(QPixmap.fromImage(qimg_agent).scaled(self.lbl_agent.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            qimg_wrist = QImage(wrist_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.lbl_wrist.setPixmap(QPixmap.fromImage(qimg_wrist).scaled(self.lbl_wrist.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            joints = self.current_demo_group["obs"]["robot0_joint_pos"][source_idx]
            pos = self.current_demo_group["obs"]["robot0_eef_pos"][source_idx]
            quat = self.current_demo_group["obs"]["robot0_eef_quat"][source_idx]
            actions = self.current_demo_group["actions"][source_idx]

            formatter = {"float_kind": lambda value: f"{value:6.3f}"}
            joints_str = np.array2string(joints, formatter=formatter)
            pos_str = np.array2string(pos, formatter=formatter)
            quat_str = np.array2string(quat, formatter=formatter)
            actions_str = np.array2string(actions, formatter=formatter)

            text = "【录制帧数据】\n"
            text += "-" * 25 + "\n"
            text += f"► 关节位置 [6]:\n {joints_str}\n\n"
            text += f"► 末端 XYZ [3]:\n {pos_str}\n\n"
            text += f"► 末端 四元数 [4]:\n {quat_str}\n\n"
            text += "-" * 25 + "\n"
            text += f"► 保存的 Action [7]:\n {actions_str}\n"
            text += "  (XYZ, RxRyRz, Gripper)"
            self.text_state.setText(text)
        except Exception as exc:
            self.text_state.setText(f"读取帧数据失败: {exc}")

    def closeEvent(self, event):
        self.pause_playback()
        if self.file_handle is not None:
            self.file_handle.close()
        event.accept()
