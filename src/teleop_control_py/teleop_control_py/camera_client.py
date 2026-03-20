#!/usr/bin/env python3
"""相机 SDK 轻量封装。"""

from __future__ import annotations

import os
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
            # Drain unread frames first so recording consumes the freshest available image.
            frames = None
            if hasattr(self._pipeline, "poll_for_frames"):
                while True:
                    polled = self._pipeline.poll_for_frames()
                    if not polled:
                        break
                    frames = polled

            if frames is None:
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
        self._stopped = False
        os.environ.setdefault("DEPTHAI_SEARCH_TIMEOUT", "3000")

        self._device = dai.Device(dai.UsbSpeed.SUPER)
        self._pipeline = dai.Pipeline(self._device)
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
        self._pipeline.start()

        usb_speed = self._device.getUsbSpeed()
        self._log("info", f"OAK-D USB speed: {usb_speed.name}")
        if not self._pipeline.isRunning():
            self._device.close()
            raise RuntimeError("OAK-D pipeline 启动失败")

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        log_fn = getattr(self._logger, level, None)
        if callable(log_fn):
            log_fn(msg)

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        if self._stopped or getattr(self, "_queue", None) is None:
            return None
        try:
            if hasattr(self._queue, "isClosed") and self._queue.isClosed():
                return None
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
        if self._stopped:
            return
        self._stopped = True

        try:
            if hasattr(self._queue, "close"):
                self._queue.close()
        except Exception as exc:  # noqa: BLE001
            self._log("warn", f"关闭 OAK-D 输出队列失败: {exc!r}")

        self._queue = None
        self._pipeline = None
        self._device = None
