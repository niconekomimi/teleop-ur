#!/usr/bin/env python3
"""SDK-backed camera implementations used only inside the hardware daemon."""

from __future__ import annotations

import os
import warnings
from typing import Optional

import cv2
import depthai as dai
import numpy as np
import pyrealsense2 as rs

from .interfaces import BaseCamera


def _log(logger: Optional[object], level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class RealSenseClient(BaseCamera):
    """pyrealsense2-backed RGB camera bound to an optional serial number."""

    def __init__(self, serial_number: str = "", logger: Optional[object] = None) -> None:
        self._serial_number = str(serial_number).strip()
        self._logger = logger
        self._pipeline: Optional[rs.pipeline] = None
        self._config: Optional[rs.config] = None

    def start(self) -> None:
        if self._pipeline is not None:
            return

        pipeline = rs.pipeline()
        config = rs.config()
        if self._serial_number:
            config.enable_device(self._serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        pipeline.start(config)

        self._pipeline = pipeline
        self._config = config

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        if self._pipeline is None:
            self.start()
        if self._pipeline is None:
            return None

        try:
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
            _log(self._logger, "warn", f"RealSense 拉帧失败: {exc!r}")
            return None

    def stop(self) -> None:
        if self._pipeline is None:
            return
        try:
            self._pipeline.stop()
        except Exception as exc:  # noqa: BLE001
            _log(self._logger, "warn", f"关闭 RealSense 失败: {exc!r}")
        finally:
            self._pipeline = None
            self._config = None


class OAKClient(BaseCamera):
    """DepthAI-backed RGB camera bound to an optional serial number."""

    def __init__(self, serial_number: str = "", logger: Optional[object] = None) -> None:
        self._serial_number = str(serial_number).strip()
        self._logger = logger
        self._stopped = False
        self._device = None
        self._pipeline = None
        self._queue = None

    def start(self) -> None:
        if self._queue is not None and not self._stopped:
            return

        self._stopped = False
        os.environ.setdefault("DEPTHAI_SEARCH_TIMEOUT", "3000")

        device_info = dai.DeviceInfo(self._serial_number) if self._serial_number else None
        if device_info is not None:
            self._device = dai.Device(device_info, dai.UsbSpeed.SUPER)
        else:
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
        _log(self._logger, "info", f"OAK-D USB speed: {usb_speed.name}")
        if not self._pipeline.isRunning():
            self._device.close()
            self._device = None
            self._pipeline = None
            self._queue = None
            raise RuntimeError("OAK-D pipeline 启动失败")

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        if self._queue is None and not self._stopped:
            self.start()
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
            _log(self._logger, "warn", f"OAK-D 拉帧失败: {exc!r}")
            return None

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        try:
            if getattr(self._queue, "close", None) is not None:
                self._queue.close()
        except Exception as exc:  # noqa: BLE001
            _log(self._logger, "warn", f"关闭 OAK-D 输出队列失败: {exc!r}")

        try:
            if getattr(self._device, "close", None) is not None:
                self._device.close()
        except Exception as exc:  # noqa: BLE001
            _log(self._logger, "warn", f"关闭 OAK-D 设备失败: {exc!r}")

        self._queue = None
        self._pipeline = None
        self._device = None


def create_sdk_camera(camera_type: str, *, serial_number: str = "", logger: Optional[object] = None) -> BaseCamera:
    normalized = str(camera_type).strip().lower()
    if normalized == "realsense":
        return RealSenseClient(serial_number=serial_number, logger=logger)
    if normalized == "oakd":
        return OAKClient(serial_number=serial_number, logger=logger)
    raise ValueError(f"Unsupported camera type: {camera_type}")
