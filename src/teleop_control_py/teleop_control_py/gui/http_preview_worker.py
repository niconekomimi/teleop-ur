#!/usr/bin/env python3
"""HTTP polling worker for collector preview images."""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class HttpPreviewWorker(QThread):
    global_image_signal = Signal(np.ndarray)
    wrist_image_signal = Signal(np.ndarray)
    availability_signal = Signal(bool)
    log_signal = Signal(str)

    def __init__(self, base_url: str, fps: float = 20.0) -> None:
        super().__init__()
        self._base_url = self._normalize_base_url(base_url)
        self._interval_sec = 1.0 / max(1.0, float(fps))
        self._stop_event = threading.Event()
        self._availability = None
        self._last_health_check = 0.0

    def set_base_url(self, base_url: str) -> None:
        self._base_url = self._normalize_base_url(base_url)

    def stop(self) -> None:
        self._stop_event.set()
        self.wait(1500)

    def run(self) -> None:
        while not self._stop_event.is_set():
            loop_started = time.monotonic()
            available = self._availability if self._availability is not None else False
            if (loop_started - self._last_health_check) >= 1.0 or self._availability is None:
                available = self._ping()
                self._set_availability(available)
                self._last_health_check = loop_started

            if available:
                self._fetch_and_emit("global", self.global_image_signal)
                self._fetch_and_emit("wrist", self.wrist_image_signal)

            elapsed = time.monotonic() - loop_started
            remaining = self._interval_sec - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

        self._set_availability(False)

    def _set_availability(self, available: bool) -> None:
        if self._availability == available:
            return
        self._availability = available
        self.availability_signal.emit(available)

    def _ping(self) -> bool:
        url = f"{self._base_url}/healthz"
        try:
            with urllib.request.urlopen(url, timeout=0.2) as response:
                return int(response.getcode() or 0) == 200
        except Exception:
            return False

    def _fetch_and_emit(self, camera_name: str, signal) -> None:
        frame = self._fetch_frame(camera_name)
        if frame is not None:
            signal.emit(frame)

    def _fetch_frame(self, camera_name: str):
        url = f"{self._base_url}/preview/{camera_name}.jpg"
        try:
            with urllib.request.urlopen(url, timeout=0.3) as response:
                status = int(response.getcode() or 0)
                if status == 204:
                    return None
                if status != 200:
                    return None
                payload = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 204 and self._availability:
                self._set_availability(False)
            return None
        except Exception:
            if self._availability:
                self._set_availability(False)
            return None

        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        return None if frame is None else np.ascontiguousarray(frame)

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        return str(base_url).strip().rstrip("/")
