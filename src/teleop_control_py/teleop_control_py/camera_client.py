#!/usr/bin/env python3
"""相机 SDK 轻量封装。"""

from __future__ import annotations

import os
import threading
import time
import warnings
from typing import Optional

import cv2
import depthai as dai
import numpy as np
import pyrealsense2 as rs


class RealSenseClient:
    """基于 pyrealsense2 的 RGB 主动抓帧客户端。"""

    def __init__(self, logger: Optional[object] = None) -> None:
        self._logger = logger
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self._pipeline.start(self._config)

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        log_fn = getattr(self._logger, level, None)
        if callable(log_fn):
            log_fn(msg)

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=200)
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return None
            rgb_frame = np.asanyarray(color_frame.get_data())
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(bgr_frame)
        except Exception as exc:  # noqa: BLE001
            self._log("warn", f"RealSense 拉帧失败: {exc!r}")
            return None

    def stop(self) -> None:
        try:
            self._pipeline.stop()
        except Exception as exc:  # noqa: BLE001
            self._log("warn", f"关闭 RealSense 失败: {exc!r}")


class OAKClient:
    """基于 depthai 的 OAK-D 主动抓帧客户端。"""

    def __init__(self, logger: Optional[object] = None) -> None:
        self._logger = logger
        os.environ.setdefault("DEPTHAI_SEARCH_TIMEOUT", "3000")

        self._pipeline = dai.Pipeline()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".*ColorCamera node is deprecated.*",
            )
            color = self._pipeline.create(dai.node.ColorCamera)

        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setFps(30)
        color.setInterleaved(False)
        color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self._queue = color.video.createOutputQueue(maxSize=1, blocking=False)

        self._pipeline_thread = threading.Thread(target=self._pipeline.run, daemon=True)
        self._pipeline_thread.start()

        deadline = time.monotonic() + 5.0
        while not self._pipeline.isRunning():
            if time.monotonic() > deadline:
                raise RuntimeError("OAK-D pipeline 未在 5 秒内进入运行状态")
            time.sleep(0.01)

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        log_fn = getattr(self._logger, level, None)
        if callable(log_fn):
            log_fn(msg)

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        try:
            packet = self._queue.tryGet() if hasattr(self._queue, "tryGet") else None
            if packet is None:
                return None
            frame = packet.getCvFrame()
            if frame is None:
                return None
            return np.ascontiguousarray(frame)
        except Exception as exc:  # noqa: BLE001
            self._log("warn", f"OAK-D 拉帧失败: {exc!r}")
            return None

    def stop(self) -> None:
        try:
            self._pipeline.stop()
        except Exception as exc:  # noqa: BLE001
            self._log("warn", f"关闭 OAK-D 失败: {exc!r}")
        try:
            self._pipeline_thread.join(timeout=1.0)
        except Exception:
            pass