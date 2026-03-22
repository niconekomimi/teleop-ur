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

from .interfaces import BaseCamera, CameraFrame, CameraIntrinsics


def _log(logger: Optional[object], level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


def _oak_socket(name: str):
    return getattr(dai.CameraBoardSocket, name, None)


def _oak_rgb_socket():
    return _oak_socket("CAM_A") or _oak_socket("RGB")


def _oak_left_socket():
    return _oak_socket("CAM_B") or _oak_socket("LEFT")


def _oak_right_socket():
    return _oak_socket("CAM_C") or _oak_socket("RIGHT")


def _oak_set_socket(node, socket) -> None:
    if socket is None:
        return
    for attr in ("setBoardSocket", "setCamera"):
        setter = getattr(node, attr, None)
        if callable(setter):
            setter(socket)
            return


def _oak_output_queue(device, stream_name: str, *, max_size: int = 1):
    getter = getattr(device, "getOutputQueue", None)
    if not callable(getter):
        raise RuntimeError("DepthAI device does not expose getOutputQueue")
    return getter(stream_name, max_size, False)


class RealSenseClient(BaseCamera):
    """pyrealsense2-backed RGB camera with optional depth, intrinsics, disparity."""

    def __init__(
        self,
        serial_number: str = "",
        logger: Optional[object] = None,
        *,
        enable_depth: bool = False,
        enable_disparity_offset: bool = False,
        disparity_offset: float = 0.0,
        stereo_baseline_m: float = 0.05,
    ) -> None:
        self._serial_number = str(serial_number).strip()
        self._logger = logger
        self._enable_depth = bool(enable_depth)
        self._enable_disparity_offset = bool(enable_disparity_offset)
        self._disparity_offset = float(disparity_offset)
        self._stereo_baseline_m = max(1e-6, float(stereo_baseline_m))
        self._pipeline: Optional[rs.pipeline] = None
        self._config: Optional[rs.config] = None
        self._align: Optional[rs.align] = None
        self._profile: Optional[rs.pipeline_profile] = None
        self._depth_scale: float = 0.001
        self._stream_ready = False

    def start(self) -> None:
        if self._pipeline is not None:
            return

        pipeline = rs.pipeline()
        config = rs.config()
        if self._serial_number:
            config.enable_device(self._serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        if self._enable_depth:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self._profile = pipeline.start(config)
        self._pipeline = pipeline
        self._config = config
        self._align = rs.align(rs.stream.color) if self._enable_depth else None
        self._stream_ready = False
        if self._enable_depth and self._profile:
            try:
                sensor = self._profile.get_device().first_depth_sensor()
                self._depth_scale = float(sensor.get_depth_scale()) or 0.001
            except Exception:
                self._depth_scale = 0.001

    def _get_frames(self) -> Optional[tuple]:
        if self._pipeline is None:
            self.start()
        if self._pipeline is None:
            return None
        try:
            frames = None
            if self._stream_ready and hasattr(self._pipeline, "poll_for_frames"):
                while True:
                    polled = self._pipeline.poll_for_frames()
                    if not polled:
                        break
                    frames = polled
            if frames is None:
                timeout_ms = 3000 if not self._stream_ready else 200
                frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
            if self._align is not None:
                frames = self._align.process(frames)
            self._stream_ready = True
            return frames
        except Exception as exc:  # noqa: BLE001
            _log(self._logger, "warn", f"RealSense 拉帧失败: {exc!r}")
            return None

    def _intrinsics_from_frame(self, color_frame) -> Optional[CameraIntrinsics]:
        try:
            profile = getattr(color_frame, "profile", None)
            if profile is None:
                return None
            video_profile = None
            as_video_stream_profile = getattr(profile, "as_video_stream_profile", None)
            if callable(as_video_stream_profile):
                video_profile = as_video_stream_profile()
            if video_profile is None:
                try:
                    video_profile = rs.video_stream_profile(profile)
                except Exception:
                    video_profile = None
            if video_profile is None:
                return None
            get_intrinsics = getattr(video_profile, "get_intrinsics", None)
            intr = get_intrinsics() if callable(get_intrinsics) else getattr(video_profile, "intrinsics", None)
            if intr is None:
                return None
            w = intr.width
            h = intr.height
            return CameraIntrinsics(
                fx=float(intr.fx),
                fy=float(intr.fy),
                cx=float(intr.ppx),
                cy=float(intr.ppy),
                width=int(w) if w > 0 else color_frame.get_data().shape[1],
                height=int(h) if h > 0 else color_frame.get_data().shape[0],
            )
        except Exception:
            return None

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        frames = self._get_frames()
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
        rgb_frame = np.asanyarray(color_frame.get_data())
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(bgr_frame)

    def get_frame(self) -> Optional[CameraFrame]:
        frames = self._get_frames()
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
        rgb_frame = np.asanyarray(color_frame.get_data())
        bgr_frame = np.ascontiguousarray(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

        depth = None
        disparity = None
        intrinsics = self._intrinsics_from_frame(color_frame)

        if self._enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                depth = (depth_mm * self._depth_scale).astype(np.float32)
                depth = np.ascontiguousarray(depth)

                if self._enable_disparity_offset and intrinsics is not None and intrinsics.fx > 0:
                    depth_safe = np.where(depth > 1e-6, depth, np.nan)
                    disp = (self._stereo_baseline_m * intrinsics.fx) / depth_safe
                    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
                    disp = (disp + self._disparity_offset).astype(np.float32)
                    disparity = np.ascontiguousarray(np.clip(disp, 0.0, 1e6))

        return CameraFrame(bgr=bgr_frame, depth=depth, intrinsics=intrinsics, disparity=disparity)

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
            self._align = None
            self._profile = None
            self._stream_ready = False


class OAKClient(BaseCamera):
    """DepthAI-backed RGB camera with optional depth, intrinsics, disparity (stereo pipeline)."""

    def __init__(
        self,
        serial_number: str = "",
        logger: Optional[object] = None,
        *,
        enable_depth: bool = False,
        enable_disparity_offset: bool = False,
        disparity_offset: float = 0.0,
    ) -> None:
        self._serial_number = str(serial_number).strip()
        self._logger = logger
        self._enable_depth = bool(enable_depth)
        self._enable_disparity_offset = bool(enable_disparity_offset)
        self._disparity_offset = float(disparity_offset)
        self._stopped = False
        self._device = None
        self._pipeline = None
        self._rgb_queue = None
        self._depth_queue = None
        self._calib = None
        self._intrinsics_cache: Optional[CameraIntrinsics] = None
        self._rgb_resolution = (1920, 1080)

    def start(self) -> None:
        if self._rgb_queue is not None and not self._stopped:
            return

        self._stopped = False
        os.environ.setdefault("DEPTHAI_SEARCH_TIMEOUT", "3000")

        device_info = dai.DeviceInfo(self._serial_number) if self._serial_number else None
        if device_info is not None:
            self._device = dai.Device(device_info, dai.UsbSpeed.SUPER)
        else:
            self._device = dai.Device(dai.UsbSpeed.SUPER)

        try:
            self._calib = self._device.readCalibration()
        except Exception:
            self._calib = None

        self._pipeline = dai.Pipeline(self._device)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".*ColorCamera node is deprecated.*",
            )
            color = self._pipeline.create(dai.node.ColorCamera)

        _oak_set_socket(color, _oak_rgb_socket())
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setFps(30)
        color.setInterleaved(False)
        color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self._rgb_resolution = (1920, 1080)

        if self._enable_depth:
            mono_left = self._pipeline.create(dai.node.MonoCamera)
            mono_right = self._pipeline.create(dai.node.MonoCamera)
            stereo = self._pipeline.create(dai.node.StereoDepth)
            xout_rgb = self._pipeline.create(dai.node.XLinkOut)
            xout_depth = self._pipeline.create(dai.node.XLinkOut)

            _oak_set_socket(mono_left, _oak_left_socket())
            _oak_set_socket(mono_right, _oak_right_socket())
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_left.setFps(30)
            mono_right.setFps(30)

            preset_mode = getattr(getattr(dai.node.StereoDepth, "PresetMode", None), "HIGH_DENSITY", None)
            if preset_mode is not None and hasattr(stereo, "setDefaultProfilePreset"):
                stereo.setDefaultProfilePreset(preset_mode)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            if hasattr(stereo, "setExtendedDisparity"):
                stereo.setExtendedDisparity(False)
            rgb_socket = _oak_rgb_socket()
            if rgb_socket is not None and hasattr(stereo, "setDepthAlign"):
                stereo.setDepthAlign(rgb_socket)
            if hasattr(stereo, "setOutputSize"):
                stereo.setOutputSize(*self._rgb_resolution)

            xout_rgb.setStreamName("rgb")
            xout_depth.setStreamName("depth")
            color.video.link(xout_rgb.input)
            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)
            stereo.depth.link(xout_depth.input)
        else:
            self._rgb_queue = color.video.createOutputQueue(maxSize=1, blocking=False)

        self._pipeline.start()
        if self._enable_depth:
            self._rgb_queue = _oak_output_queue(self._device, "rgb", max_size=1)
            self._depth_queue = _oak_output_queue(self._device, "depth", max_size=1)

        usb_speed = self._device.getUsbSpeed()
        _log(self._logger, "info", f"OAK-D USB speed: {usb_speed.name}")
        if not self._pipeline.isRunning():
            self._device.close()
            self._device = None
            self._pipeline = None
            self._rgb_queue = None
            self._depth_queue = None
            raise RuntimeError("OAK-D pipeline 启动失败")

    def _get_oak_intrinsics(self) -> Optional[CameraIntrinsics]:
        if self._intrinsics_cache is not None:
            return self._intrinsics_cache
        if self._calib is None:
            return None
        try:
            rgb_socket = _oak_rgb_socket()
            intr = self._calib.getCameraIntrinsics(rgb_socket) if rgb_socket is not None else None
            if intr is None or len(intr) < 3:
                return None
            fx, fy = float(intr[0][0]), float(intr[1][1])
            cx, cy = float(intr[0][2]), float(intr[1][2])
            width, height = self._rgb_resolution
            self._intrinsics_cache = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
            return self._intrinsics_cache
        except Exception:
            return None

    @staticmethod
    def _queue_packet(queue):
        if queue is None:
            return None
        if hasattr(queue, "isClosed") and queue.isClosed():
            return None
        if hasattr(queue, "tryGet"):
            return queue.tryGet()
        if hasattr(queue, "get"):
            return queue.get()
        return None

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        if self._rgb_queue is None and not self._stopped:
            self.start()
        if self._stopped or self._rgb_queue is None:
            return None

        try:
            packet = self._queue_packet(self._rgb_queue)
            if packet is None:
                return None
            frame = packet.getCvFrame()
            if frame is None:
                return None
            return np.ascontiguousarray(frame)
        except Exception as exc:  # noqa: BLE001
            _log(self._logger, "warn", f"OAK-D 拉帧失败: {exc!r}")
            return None

    def get_frame(self) -> Optional[CameraFrame]:
        bgr = self.get_bgr_frame()
        if bgr is None:
            return None
        intrinsics = self._get_oak_intrinsics()
        depth = None
        disparity = None

        if self._enable_depth and self._depth_queue is not None:
            try:
                packet = self._queue_packet(self._depth_queue)
                if packet is not None:
                    raw_depth = packet.getFrame() if hasattr(packet, "getFrame") else packet.getCvFrame()
                    if raw_depth is not None:
                        depth = np.asarray(raw_depth, dtype=np.float32)
                        if depth.size > 0:
                            depth = np.ascontiguousarray(depth / 1000.0)
                            if self._enable_disparity_offset and intrinsics is not None and intrinsics.fx > 0:
                                depth_safe = np.where(depth > 1e-6, depth, np.nan)
                                disp = (0.075 * intrinsics.fx) / depth_safe
                                disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
                                disp = (disp + self._disparity_offset).astype(np.float32)
                                disparity = np.ascontiguousarray(np.clip(disp, 0.0, 1e6))
            except Exception as exc:  # noqa: BLE001
                _log(self._logger, "warn", f"OAK-D 深度帧读取失败: {exc!r}")

        return CameraFrame(bgr=bgr, depth=depth, intrinsics=intrinsics, disparity=disparity)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        try:
            if getattr(self._rgb_queue, "close", None) is not None:
                self._rgb_queue.close()
        except Exception:  # noqa: BLE001
            pass

        try:
            if getattr(self._depth_queue, "close", None) is not None:
                self._depth_queue.close()
        except Exception:  # noqa: BLE001
            pass

        try:
            if getattr(self._device, "close", None) is not None:
                self._device.close()
        except Exception:  # noqa: BLE001
            pass

        self._rgb_queue = None
        self._depth_queue = None
        self._pipeline = None
        self._device = None
        self._calib = None
        self._intrinsics_cache = None


def create_sdk_camera(
    camera_type: str,
    *,
    serial_number: str = "",
    logger: Optional[object] = None,
    enable_depth: bool = False,
    enable_disparity_offset: bool = False,
    disparity_offset: float = 0.0,
    stereo_baseline_m: float = 0.05,
) -> BaseCamera:
    normalized = str(camera_type).strip().lower()
    if normalized == "realsense":
        return RealSenseClient(
            serial_number=serial_number,
            logger=logger,
            enable_depth=enable_depth,
            enable_disparity_offset=enable_disparity_offset,
            disparity_offset=disparity_offset,
            stereo_baseline_m=stereo_baseline_m,
        )
    if normalized == "oakd":
        return OAKClient(
            serial_number=serial_number,
            logger=logger,
            enable_depth=enable_depth,
            enable_disparity_offset=enable_disparity_offset,
            disparity_offset=disparity_offset,
        )
    raise ValueError(f"Unsupported camera type: {camera_type}")
