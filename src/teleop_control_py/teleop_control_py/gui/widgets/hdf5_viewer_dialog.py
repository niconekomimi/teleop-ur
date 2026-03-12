import os
from pathlib import Path

import h5py
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
    QInputDialog,
    QProgressDialog,
    QWidget,
)

from teleop_control_py.dataset_rebuilder import rebuild_file, quat_to_rotvec_xyzw, sorted_demo_names


class HDF5ViewerDialog(QDialog):
    ZERO_ACTION_ATOL = 1e-6
    ZERO_ACTION_CONTEXT_FRAMES = 5

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
        self._agent_source_pixmap = QPixmap()
        self._wrist_source_pixmap = QPixmap()

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

        self.btn_prev_demo = QPushButton("上一条 demo")
        self.btn_prev_demo.clicked.connect(self.goto_prev_demo)
        top_layout.addWidget(self.btn_prev_demo)

        self.btn_next_demo = QPushButton("下一条 demo")
        self.btn_next_demo.clicked.connect(self.goto_next_demo)
        top_layout.addWidget(self.btn_next_demo)

        self.btn_delete_current = QPushButton("删除当前 demo")
        self.btn_delete_current.clicked.connect(self.delete_current_demo)
        top_layout.addWidget(self.btn_delete_current)

        self.btn_save_changes = QPushButton("保存修改")
        self.btn_save_changes.setStyleSheet("font-weight: bold; background-color: #ffe8a1;")
        self.btn_save_changes.clicked.connect(self.save_pending_changes)
        top_layout.addWidget(self.btn_save_changes)

        self.btn_rebuild_schema = QPushButton("转换为训练格式")
        self.btn_rebuild_schema.setStyleSheet("font-weight: bold; background-color: #d7f4d1;")
        self.btn_rebuild_schema.clicked.connect(self.rebuild_dataset_schema)
        top_layout.addWidget(self.btn_rebuild_schema)

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

        self.btn_auto_trim_zero_actions = QPushButton("自动裁切零动作首尾")
        self.btn_auto_trim_zero_actions.clicked.connect(self.auto_trim_zero_action_ranges)
        edit_layout.addWidget(self.btn_auto_trim_zero_actions)

        self.lbl_crop_state = QLabel("当前 demo 无待保存裁剪")
        self.lbl_crop_state.setStyleSheet("color: #555;")
        edit_layout.addWidget(self.lbl_crop_state)
        edit_layout.addStretch()
        main_layout.addLayout(edit_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        self.cam_panel = QWidget()
        self.cam_panel.setMinimumWidth(620)
        self.cam_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cam_layout = QHBoxLayout(self.cam_panel)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        cam_layout.setSpacing(12)

        agent_panel = QWidget()
        agent_layout = QVBoxLayout(agent_panel)
        agent_layout.setContentsMargins(0, 0, 0, 0)
        agent_layout.setSpacing(6)

        global_title = QLabel("【全局相机 (Agent View)】")
        global_title.setAlignment(Qt.AlignCenter)
        self.lbl_agent = QLabel("无画面")
        self.lbl_agent.setAlignment(Qt.AlignCenter)
        self.lbl_agent.setStyleSheet("background-color: #202020; color: white; border-radius: 4px;")
        self.lbl_agent.setMinimumHeight(220)
        self.lbl_agent.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        agent_layout.addWidget(global_title)
        agent_layout.addWidget(self.lbl_agent)

        wrist_panel = QWidget()
        wrist_layout = QVBoxLayout(wrist_panel)
        wrist_layout.setContentsMargins(0, 0, 0, 0)
        wrist_layout.setSpacing(6)

        wrist_title = QLabel("【手部相机 (Eye-in-Hand)】")
        wrist_title.setAlignment(Qt.AlignCenter)
        self.lbl_wrist = QLabel("无画面")
        self.lbl_wrist.setAlignment(Qt.AlignCenter)
        self.lbl_wrist.setStyleSheet("background-color: #202020; color: white; border-radius: 4px;")
        self.lbl_wrist.setMinimumHeight(220)
        self.lbl_wrist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        wrist_layout.addWidget(wrist_title)
        wrist_layout.addWidget(self.lbl_wrist)

        cam_layout.addWidget(agent_panel, 1)
        cam_layout.addWidget(wrist_panel, 1)
        content_layout.addWidget(self.cam_panel, 3)

        self.text_state = QTextEdit()
        self.text_state.setReadOnly(True)
        self.text_state.setMinimumWidth(280)
        self.text_state.setMaximumWidth(420)
        self.text_state.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.text_state.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; background-color: #f5f5f5;")
        content_layout.addWidget(self.text_state, 1)
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

    def rebuild_dataset_schema(self):
        class RebuildCancelledError(Exception):
            pass

        if self.hdf5_path and os.path.exists(self.hdf5_path):
            input_path = self.hdf5_path
        else:
            start_dir = os.getcwd()
            input_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择要转换的 HDF5 文件",
                start_dir,
                "HDF5 Files (*.hdf5 *.h5);;All Files (*)",
            )
            if not input_path:
                return

        input_file = Path(input_path)
        default_output = input_file.with_name(f"{input_file.stem}_rebuilt{input_file.suffix}")
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择转换后输出文件",
            str(default_output),
            "HDF5 Files (*.hdf5 *.h5);;All Files (*)",
        )
        if not output_path:
            return

        if os.path.abspath(output_path) == os.path.abspath(input_path):
            QMessageBox.warning(self, "输出文件无效", "输出文件不能和输入文件相同，请选择新的 HDF5 文件名。")
            return

        compression_settings = self._prompt_rebuild_compression()
        if compression_settings is None:
            return

        reply = QMessageBox.question(
            self,
            "确认转换",
            "将按训练格式重建一个新的 HDF5 文件。\n\n"
            "转换内容包括：\n"
            "- 保留 actions\n"
            "- 生成 dones、rewards、states\n"
            "- 生成 robot_states = joint_states + gripper_states\n"
            "- 将末端四元数转换为轴角 ee_ori\n"
            "- 保持图像原始尺寸\n"
            f"- 压缩策略: {compression_settings['description']}\n"
            "- 输出 demo 会连续重命名为 demo_0..N\n\n"
            "是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        output_file = Path(output_path)
        temp_output_path = output_file.with_name(f"{output_file.name}.tmp")
        progress_dialog = None

        try:
            with h5py.File(input_path, "r") as source_h5:
                source_data = source_h5.get("data")
                if not isinstance(source_data, h5py.Group):
                    raise RuntimeError("输入 HDF5 中未找到 /data 组。")

                demo_names = [
                    name for name in sorted_demo_names(source_data) if isinstance(source_data.get(name), h5py.Group)
                ]
                if not demo_names:
                    raise RuntimeError("输入 HDF5 中没有可转换的 demo_* 组。")

            datasets_per_demo = 12
            total_steps = len(demo_names) * datasets_per_demo + 1
            progress_dialog = QProgressDialog("正在准备转换...", "取消", 0, total_steps, self)
            progress_dialog.setWindowTitle("转换为训练格式")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setAutoClose(False)
            progress_dialog.setAutoReset(False)
            progress_dialog.setValue(0)
            progress_dialog.show()
            QApplication.processEvents()

            completed_steps = 0

            def update_progress(dataset_path):
                nonlocal completed_steps
                completed_steps += 1
                progress_dialog.setLabelText(f"正在转换训练格式...\n{dataset_path}")
                progress_dialog.setValue(completed_steps)
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    raise RebuildCancelledError()

            results = rebuild_file(
                input_path=input_path,
                output_path=str(temp_output_path),
                include_states=True,
                renumber=True,
                compression=compression_settings["compression"],
                compression_opts=compression_settings["compression_opts"],
                progress_callback=update_progress,
            )
            progress_dialog.setLabelText("正在写入输出文件...")
            progress_dialog.setValue(total_steps)
            QApplication.processEvents()
            if progress_dialog.wasCanceled():
                raise RebuildCancelledError()

            os.replace(temp_output_path, output_file)
        except RebuildCancelledError:
            try:
                if temp_output_path.exists():
                    temp_output_path.unlink()
            except Exception:
                pass
            if progress_dialog is not None:
                progress_dialog.close()
            QMessageBox.information(self, "已取消", "已取消转换，输出文件未被写入。")
            return
        except Exception as exc:
            try:
                if temp_output_path.exists():
                    temp_output_path.unlink()
            except Exception:
                pass
            if progress_dialog is not None:
                progress_dialog.close()
            QMessageBox.critical(self, "转换失败", f"重建数据集失败:\n{exc}")
            return
        finally:
            if progress_dialog is not None:
                progress_dialog.close()

        total_frames = sum(num_samples for _, _, num_samples in results)
        reply = QMessageBox.question(
            self,
            "转换完成",
            f"已完成转换。\n\n输出文件: {output_path}\n"
            f"demo 数量: {len(results)}\n"
            f"总帧数: {total_frames}\n\n"
            "是否立即打开输出文件？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self.open_hdf5_file(output_path)

    def _prompt_rebuild_compression(self):
        options = [
            ("继承源数据集压缩参数（默认）", "inherit"),
            ("统一使用 lzf 压缩", "lzf"),
            ("统一使用 gzip 压缩", "gzip"),
            ("统一不压缩", None),
        ]
        labels = [label for label, _ in options]
        selected_label, ok = QInputDialog.getItem(
            self,
            "选择压缩策略",
            "输出文件的压缩策略：",
            labels,
            0,
            False,
        )
        if not ok:
            return None

        compression_map = {label: value for label, value in options}
        compression = compression_map[selected_label]
        compression_opts = None
        if compression == "gzip":
            compression_opts, ok = QInputDialog.getInt(
                self,
                "设置 gzip 级别",
                "gzip 压缩级别 (0-9)：",
                4,
                0,
                9,
                1,
            )
            if not ok:
                return None

        return {
            "compression": compression,
            "compression_opts": compression_opts,
            "description": self._describe_rebuild_compression(compression, compression_opts),
        }

    def _describe_rebuild_compression(self, compression, compression_opts):
        if compression == "inherit":
            return "继承源数据集压缩参数"
        if compression is None:
            return "统一不压缩"
        if compression == "gzip":
            return f"统一使用 gzip 压缩 (level={compression_opts})"
        return f"统一使用 {compression} 压缩"

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

    def _update_demo_navigation_buttons(self):
        current_index = self.demo_combo.currentIndex()
        has_demos = bool(self.demo_names)
        self.btn_prev_demo.setEnabled(has_demos and current_index > 0)
        self.btn_next_demo.setEnabled(has_demos and 0 <= current_index < len(self.demo_names) - 1)

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
        self._update_demo_navigation_buttons()

    def _copy_attrs(self, source, target):
        for key, value in source.attrs.items():
            target.attrs[key] = value

    def _count_group_datasets(self, source_group):
        count = 0
        for item in source_group.values():
            if isinstance(item, h5py.Group):
                count += self._count_group_datasets(item)
            else:
                count += 1
        return count

    def _copy_group_sliced(self, source_group, target_group, start, end, total, progress_callback=None, path_prefix=""):
        self._copy_attrs(source_group, target_group)
        for name, item in source_group.items():
            current_path = f"{path_prefix}/{name}" if path_prefix else name
            if isinstance(item, h5py.Group):
                child = target_group.create_group(name)
                self._copy_group_sliced(
                    item,
                    child,
                    start,
                    end,
                    total,
                    progress_callback=progress_callback,
                    path_prefix=current_path,
                )
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
            if progress_callback is not None:
                progress_callback(current_path)

    def save_pending_changes(self):
        class SaveCancelledError(Exception):
            pass

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
        progress_dialog = None

        try:
            with h5py.File(source_path, "r") as source_h5:
                source_data = source_h5["data"]
                source_demo_names = sorted(source_data.keys(), key=self._demo_sort_key)
                kept_demo_names = [name for name in self.demo_names if name in source_data]
                if not kept_demo_names:
                    raise ValueError("保存结果为空，已中止写回以避免删除全部 demo。")

                unknown_demo_names = [name for name in kept_demo_names if name not in source_demo_names]
                if unknown_demo_names:
                    raise ValueError(f"待保留 demo 不存在于源文件: {unknown_demo_names}")

                total_steps = 1 + sum(self._count_group_datasets(source_data[name]) for name in kept_demo_names)
                progress_dialog = QProgressDialog("正在准备保存修改...", "取消", 0, total_steps, self)
                progress_dialog.setWindowTitle("保存修改")
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.setMinimumDuration(0)
                progress_dialog.setAutoClose(False)
                progress_dialog.setAutoReset(False)
                progress_dialog.setValue(0)
                progress_dialog.show()
                QApplication.processEvents()

                completed_steps = 0

                with h5py.File(temp_path, "w") as target_h5:
                    self._copy_attrs(source_h5, target_h5)
                    target_data = target_h5.create_group("data")
                    self._copy_attrs(source_data, target_data)

                    for index, source_demo_name in enumerate(kept_demo_names):
                        if progress_dialog.wasCanceled():
                            raise SaveCancelledError()

                        source_demo = source_data[source_demo_name]
                        new_demo_name = f"demo_{index}"
                        total = self._infer_num_samples(source_demo)
                        start, end = self._effective_crop_range(source_demo_name, total)

                        progress_dialog.setLabelText(
                            f"正在保存 {source_demo_name} ({index + 1}/{len(kept_demo_names)})..."
                        )
                        QApplication.processEvents()
                        if progress_dialog.wasCanceled():
                            raise SaveCancelledError()

                        target_demo = target_data.create_group(new_demo_name)

                        def update_progress(dataset_path, demo_name=source_demo_name, demo_index=index):
                            nonlocal completed_steps
                            completed_steps += 1
                            progress_dialog.setLabelText(
                                f"正在保存 {demo_name} ({demo_index + 1}/{len(kept_demo_names)})...\n{dataset_path}"
                            )
                            progress_dialog.setValue(completed_steps)
                            QApplication.processEvents()
                            if progress_dialog.wasCanceled():
                                raise SaveCancelledError()

                        self._copy_group_sliced(
                            source_demo,
                            target_demo,
                            start,
                            end,
                            total,
                            progress_callback=update_progress,
                        )
                        target_demo.attrs["num_samples"] = int(max(0, end - start))

                progress_dialog.setLabelText("正在替换原始 HDF5 文件...")
                progress_dialog.setValue(total_steps)
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    raise SaveCancelledError()

            os.replace(temp_path, source_path)
        except SaveCancelledError:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            if progress_dialog is not None:
                progress_dialog.close()
            QMessageBox.information(self, "已取消", "已取消保存修改，原始 HDF5 文件未被改动。")
            self.open_hdf5_file(str(source_path))
            return
        except Exception as exc:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            if progress_dialog is not None:
                progress_dialog.close()
            QMessageBox.critical(self, "保存失败", f"保存对 HDF5 的修改失败:\n{exc}")
            self.open_hdf5_file(str(source_path))
            return
        finally:
            if progress_dialog is not None:
                progress_dialog.close()

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

    def _compute_zero_action_trim_range(self, demo_group, start, end):
        if "actions" not in demo_group:
            return None

        actions = np.asarray(demo_group["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] <= 0:
            return None

        start = max(0, int(start))
        end = min(int(end), int(actions.shape[0]))
        if start >= end:
            return None

        window = actions[start:end]
        zero_mask = np.all(np.isclose(window, 0.0, atol=self.ZERO_ACTION_ATOL), axis=1)
        nonzero_indices = np.flatnonzero(~zero_mask)
        if nonzero_indices.size == 0:
            return None

        trimmed_start = start + int(nonzero_indices[0])
        trimmed_end = start + int(nonzero_indices[-1]) + 1
        trimmed_start = max(start, trimmed_start - self.ZERO_ACTION_CONTEXT_FRAMES)
        trimmed_end = min(end, trimmed_end + self.ZERO_ACTION_CONTEXT_FRAMES)
        return trimmed_start, trimmed_end

    def auto_trim_zero_action_ranges(self):
        if self.file_handle is None or not self.demo_names:
            return

        updated = 0
        unchanged = 0
        skipped = []
        selected_demo = self.demo_combo.currentData()

        for demo_name in self.demo_names:
            if demo_name not in self.file_handle["data"]:
                continue

            demo_group = self.file_handle["data"][demo_name]
            total = self._infer_num_samples(demo_group)
            start, end = self._effective_crop_range(demo_name, total)
            trim_range = self._compute_zero_action_trim_range(demo_group, start, end)
            if trim_range is None:
                skipped.append(str(demo_name))
                continue

            trimmed_start, trimmed_end = trim_range
            new_range = None if (trimmed_start == 0 and trimmed_end == total) else (trimmed_start, trimmed_end)
            previous_range = self.pending_crop_ranges.get(demo_name)
            if new_range is None:
                if demo_name in self.pending_crop_ranges:
                    self.pending_crop_ranges.pop(demo_name, None)
                    updated += 1
                else:
                    unchanged += 1
                continue

            if previous_range == new_range:
                unchanged += 1
                continue

            self.pending_crop_ranges[demo_name] = new_range
            updated += 1

        self._refresh_demo_combo(selected_demo)

        message = f"自动裁切完成。\n\n已更新: {updated} 个 demo\n无变化: {unchanged} 个 demo"
        if skipped:
            preview = "、".join(skipped[:5])
            if len(skipped) > 5:
                preview += " ..."
            message += f"\n跳过: {len(skipped)} 个 demo（区间内 actions 全为 0 或缺少 actions）\n{preview}"
        QMessageBox.information(self, "自动裁切完成", message)

    def clear_crop_range(self):
        demo_name = self.demo_combo.currentData()
        if not demo_name:
            return
        self.pending_crop_ranges.pop(demo_name, None)
        self._refresh_demo_combo(demo_name)

    def _switch_demo_by_offset(self, offset):
        if not self.demo_names:
            return

        current_index = self.demo_combo.currentIndex()
        if current_index < 0:
            current_index = 0

        next_index = max(0, min(current_index + int(offset), len(self.demo_names) - 1))
        if next_index == current_index:
            return
        self.demo_combo.setCurrentIndex(next_index)

    def goto_prev_demo(self):
        self.pause_playback()
        self._switch_demo_by_offset(-1)

    def goto_next_demo(self):
        self.pause_playback()
        self._switch_demo_by_offset(1)

    def _set_camera_pixmap(self, label, pixmap):
        if pixmap.isNull():
            label.setPixmap(QPixmap())
            return

        target_size = label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            label.setPixmap(pixmap)
            return

        label.setPixmap(pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refresh_camera_pixmaps(self):
        if not self._agent_source_pixmap.isNull():
            self._set_camera_pixmap(self.lbl_agent, self._agent_source_pixmap)
        if not self._wrist_source_pixmap.isNull():
            self._set_camera_pixmap(self.lbl_wrist, self._wrist_source_pixmap)

    def _clear_camera_views(self):
        self._agent_source_pixmap = QPixmap()
        self._wrist_source_pixmap = QPixmap()
        self.lbl_agent.setPixmap(QPixmap())
        self.lbl_agent.setText("无画面")
        self.lbl_wrist.setPixmap(QPixmap())
        self.lbl_wrist.setText("无画面")

    def load_demo(self):
        self.pause_playback()
        demo_name = self.demo_combo.currentData()
        if not demo_name or self.file_handle is None:
            self._update_demo_navigation_buttons()
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
            self._clear_camera_views()
            self.text_state.setText("当前 demo 为空，无法预览。")

        self._update_demo_navigation_buttons()

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

    def _read_obs_dataset(self, obs_group, candidates, source_idx):
        for name in candidates:
            if name in obs_group:
                return np.asarray(obs_group[name][source_idx])
        raise KeyError(f"缺少观测字段，候选项: {', '.join(candidates)}")

    def _get_frame_payload(self, source_idx):
        obs_group = self.current_demo_group["obs"]
        actions = np.asarray(self.current_demo_group["actions"][source_idx], dtype=np.float32)

        joints = self._read_obs_dataset(obs_group, ["joint_states", "robot0_joint_pos"], source_idx).astype(np.float32)
        pos = self._read_obs_dataset(obs_group, ["ee_pos", "robot0_eef_pos"], source_idx).astype(np.float32)

        if "ee_ori" in obs_group:
            ori = np.asarray(obs_group["ee_ori"][source_idx], dtype=np.float32)
            quat = None
            schema_name = "训练格式"
        else:
            quat = np.asarray(self._read_obs_dataset(obs_group, ["robot0_eef_quat"], source_idx), dtype=np.float32)
            ori = quat_to_rotvec_xyzw(quat)
            schema_name = "采集格式"

        if "gripper_states" in obs_group:
            gripper = np.asarray(obs_group["gripper_states"][source_idx], dtype=np.float32)
        elif "robot0_gripper_qpos" in obs_group:
            gripper = np.asarray(obs_group["robot0_gripper_qpos"][source_idx], dtype=np.float32)
        else:
            gripper = np.array([actions[-1]], dtype=np.float32)

        ee_states = np.concatenate([pos, ori], axis=0).astype(np.float32)
        robot_states = None
        if "robot_states" in self.current_demo_group:
            robot_states = np.asarray(self.current_demo_group["robot_states"][source_idx], dtype=np.float32)

        dones = None
        rewards = None
        if "dones" in self.current_demo_group:
            dones = np.asarray(self.current_demo_group["dones"][source_idx])
        if "rewards" in self.current_demo_group:
            rewards = np.asarray(self.current_demo_group["rewards"][source_idx])

        return {
            "actions": actions,
            "joints": joints,
            "gripper": gripper,
            "pos": pos,
            "ori": ori,
            "quat": quat,
            "ee_states": ee_states,
            "robot_states": robot_states,
            "dones": dones,
            "rewards": rewards,
            "schema_name": schema_name,
        }

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
            agent_height, agent_width, agent_channels = agent_rgb.shape
            wrist_height, wrist_width, wrist_channels = wrist_rgb.shape

            agent_bytes_per_line = agent_channels * agent_width
            qimg_agent = QImage(agent_rgb.data, agent_width, agent_height, agent_bytes_per_line, QImage.Format_RGB888)
            self._agent_source_pixmap = QPixmap.fromImage(qimg_agent)
            self.lbl_agent.setText("")
            self._set_camera_pixmap(self.lbl_agent, self._agent_source_pixmap)

            wrist_bytes_per_line = wrist_channels * wrist_width
            qimg_wrist = QImage(wrist_rgb.data, wrist_width, wrist_height, wrist_bytes_per_line, QImage.Format_RGB888)
            self._wrist_source_pixmap = QPixmap.fromImage(qimg_wrist)
            self.lbl_wrist.setText("")
            self._set_camera_pixmap(self.lbl_wrist, self._wrist_source_pixmap)

            payload = self._get_frame_payload(source_idx)
            joints = payload["joints"]
            pos = payload["pos"]
            ori = payload["ori"]
            quat = payload["quat"]
            actions = payload["actions"]
            gripper_qpos = payload["gripper"]
            ee_states = payload["ee_states"]
            robot_states = payload["robot_states"]
            dones = payload["dones"]
            rewards = payload["rewards"]
            schema_name = payload["schema_name"]

            formatter = {"float_kind": lambda value: f"{value:6.3f}"}
            joints_str = np.array2string(joints, formatter=formatter)
            gripper_str = np.array2string(gripper_qpos, formatter=formatter)
            pos_str = np.array2string(pos, formatter=formatter)
            ori_str = np.array2string(ori, formatter=formatter)
            actions_str = np.array2string(actions, formatter=formatter)
            ee_states_str = np.array2string(ee_states, formatter=formatter)

            text = "【录制帧数据】\n"
            text += f"格式: {schema_name}\n"
            text += "-" * 25 + "\n"
            text += f"► 关节位置 [6]:\n {joints_str}\n\n"
            text += f"► 夹爪状态 [1]:\n {gripper_str}\n\n"
            text += f"► 末端 XYZ [3]:\n {pos_str}\n\n"
            text += f"► 末端 轴角 [3]:\n {ori_str}\n\n"
            if quat is not None:
                quat_str = np.array2string(quat, formatter=formatter)
                text += f"► 末端 四元数 [4]:\n {quat_str}\n\n"
            text += f"► 末端状态 [6]:\n {ee_states_str}\n\n"
            if robot_states is not None:
                robot_states_str = np.array2string(robot_states, formatter=formatter)
                text += f"► 机器人状态 [7]:\n {robot_states_str}\n\n"
            if dones is not None:
                text += f"► done: {int(dones)}\n"
            if rewards is not None:
                text += f"► reward: {float(rewards):.3f}\n"
            if dones is not None or rewards is not None:
                text += "\n"
            text += "-" * 25 + "\n"
            text += f"► 保存的 Action [7]:\n {actions_str}\n"
            text += "  (VxVyVz, WxWyWz, Gripper)"
            self.text_state.setText(text)
        except Exception as exc:
            self.text_state.setText(f"读取帧数据失败: {exc}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_camera_pixmaps()

    def closeEvent(self, event):
        self.pause_playback()
        if self.file_handle is not None:
            self.file_handle.close()
        event.accept()
