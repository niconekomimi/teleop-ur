#!/usr/bin/env python3
"""Independent preview video recording worker."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


def _normalize_record_target(target: str) -> str:
    normalized = str(target).strip().lower()
    if normalized in {"global", "wrist", "both"}:
        return normalized
    return "both"


def _normalize_frame_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"source", "square"}:
        return normalized
    return "source"


def _target_label(target: str) -> str:
    normalized = _normalize_record_target(target)
    if normalized == "global":
        return "全局"
    if normalized == "wrist":
        return "手部"
    return "全局+手部"


def _frame_mode_label(mode: str) -> str:
    normalized = _normalize_frame_mode(mode)
    if normalized == "square":
        return "中心正方形"
    return "源大小"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _center_crop_square(frame: np.ndarray) -> np.ndarray:
    if frame.ndim < 2:
        return frame
    frame_height, frame_width = frame.shape[:2]
    side = min(frame_height, frame_width)
    x0 = max(0, (frame_width - side) // 2)
    y0 = max(0, (frame_height - side) // 2)
    return np.ascontiguousarray(frame[y0:y0 + side, x0:x0 + side]).copy()


class PreviewRecordingWorker(QThread):
    status_signal = Signal(str)
    error_signal = Signal(str)
    stopped_signal = Signal(str)

    def __init__(
        self,
        output_dir: str | Path,
        *,
        target: str = "both",
        frame_mode: str = "source",
        fps: float = 30.0,
        stale_frame_age_sec: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.target = _normalize_record_target(target)
        self.frame_mode = _normalize_frame_mode(frame_mode)
        self.fps = max(1.0, float(fps))
        self._interval_sec = 1.0 / self.fps
        self._stale_frame_age_sec = (
            max(self._interval_sec * 3.0, 0.25)
            if stale_frame_age_sec is None
            else max(float(stale_frame_age_sec), self._interval_sec)
        )
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frames: dict[str, Optional[np.ndarray]] = {"global": None, "wrist": None}
        self._latest_frame_times: dict[str, float] = {"global": 0.0, "wrist": 0.0}
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._writer_sizes: dict[str, tuple[int, int]] = {}
        self._frames_written: dict[str, int] = {"global": 0, "wrist": 0}
        self._tick_frames = 0
        self._start_time = 0.0
        self._last_status_emit = 0.0

    def stop(self) -> None:
        self._stop_event.set()
        self.wait(2000)

    def wants_camera(self, camera_name: str) -> bool:
        normalized = str(camera_name).strip().lower()
        if self.target == "both":
            return normalized in {"global", "wrist"}
        return normalized == self.target

    def update_frame(self, camera_name: str, frame: Optional[np.ndarray]) -> None:
        normalized = str(camera_name).strip().lower()
        if not self.wants_camera(normalized) or frame is None:
            return
        array = np.asarray(frame)
        if array.ndim < 2:
            return
        with self._frame_lock:
            self._latest_frames[normalized] = np.ascontiguousarray(array).copy()
            self._latest_frame_times[normalized] = time.monotonic()

    def _prepare_frame_for_recording(self, frame: np.ndarray) -> np.ndarray:
        prepared = np.ascontiguousarray(frame)
        if self.frame_mode == "square":
            prepared = _center_crop_square(prepared)
        return np.ascontiguousarray(prepared).copy()

    def run(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            self.error_signal.emit(f"创建预览录屏目录失败: {exc!r}")
            self.stopped_signal.emit(str(self.output_dir))
            return

        self._start_time = time.monotonic()
        self._last_status_emit = 0.0
        next_cycle = self._start_time

        try:
            self.status_signal.emit(self._format_status(active=True))
            while not self._stop_event.is_set():
                loop_started = time.monotonic()
                frame_batch = self._snapshot_frames(loop_started)
                wrote_any = False
                for camera_name, frame in frame_batch.items():
                    self._write_frame(camera_name, frame)
                    wrote_any = True
                if wrote_any:
                    self._tick_frames += 1

                if (loop_started - self._last_status_emit) >= 0.25:
                    self.status_signal.emit(self._format_status(active=True))
                    self._last_status_emit = loop_started

                next_cycle += self._interval_sec
                remaining = next_cycle - time.monotonic()
                if remaining > 0:
                    self._stop_event.wait(remaining)
                else:
                    next_cycle = time.monotonic()
        except Exception as exc:  # noqa: BLE001
            self.error_signal.emit(f"预览录屏失败: {exc!r}")
        finally:
            self._release_writers()
            self.status_signal.emit(self._format_status(active=False))
            self.stopped_signal.emit(str(self.output_dir))

    def _snapshot_frames(self, now: float) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        with self._frame_lock:
            for camera_name in ("global", "wrist"):
                if not self.wants_camera(camera_name):
                    continue
                frame = self._latest_frames.get(camera_name)
                updated_at = float(self._latest_frame_times.get(camera_name, 0.0))
                if frame is None or updated_at <= 0.0:
                    continue
                if (now - updated_at) > self._stale_frame_age_sec:
                    continue
                frames[camera_name] = frame.copy()
        return frames

    def _write_frame(self, camera_name: str, frame: np.ndarray) -> None:
        prepared_frame = self._prepare_frame_for_recording(frame)
        writer = self._writers.get(camera_name)
        if writer is None:
            writer = self._open_writer(camera_name, prepared_frame)

        expected_width, expected_height = self._writer_sizes[camera_name]
        frame_height, frame_width = prepared_frame.shape[:2]
        if frame_width != expected_width or frame_height != expected_height:
            prepared_frame = cv2.resize(prepared_frame, (expected_width, expected_height), interpolation=cv2.INTER_AREA)

        writer.write(prepared_frame)
        self._frames_written[camera_name] = int(self._frames_written.get(camera_name, 0)) + 1

    def _open_writer(self, camera_name: str, frame: np.ndarray) -> cv2.VideoWriter:
        frame_height, frame_width = frame.shape[:2]
        output_path = self.output_dir / f"{camera_name}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法打开视频写入器: {output_path}")
        self._writers[camera_name] = writer
        self._writer_sizes[camera_name] = (frame_width, frame_height)
        return writer

    def _release_writers(self) -> None:
        for writer in self._writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self._writers.clear()
        self._writer_sizes.clear()

    def _format_status(self, *, active: bool) -> str:
        elapsed = max(0.0, time.monotonic() - self._start_time) if self._start_time > 0.0 else 0.0
        tick_fps = (self._tick_frames / elapsed) if elapsed > 1e-6 else 0.0
        segments = [
            f"预览录屏: {'录制中' if active else '已停止'}",
            f"目标: {_target_label(self.target)}",
            f"尺寸: {_frame_mode_label(self.frame_mode)}",
            f"时长: {_format_duration(elapsed)}",
        ]
        if self.wants_camera("global"):
            segments.append(f"全局: {int(self._frames_written.get('global', 0))} 帧")
        if self.wants_camera("wrist"):
            segments.append(f"手部: {int(self._frames_written.get('wrist', 0))} 帧")
        segments.append(f"实时: {tick_fps:.2f} Hz")
        return " | ".join(segments)
