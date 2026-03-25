#!/usr/bin/env python3
import os
import sys
import time
import warnings
import statistics  # 新增：用于计算标准差和抖动

# depthai-core 在无设备时可能会等待很久；设一个默认搜索超时避免脚本“卡死”。
os.environ.setdefault("DEPTHAI_SEARCH_TIMEOUT", "5000")

import depthai as dai


def _oak_socket(name: str):
    return getattr(dai.CameraBoardSocket, name, None)


def _oak_rgb_socket():
    return _oak_socket("CAM_A") or _oak_socket("RGB")


def _print_depthai_diagnostics() -> None:
    print(f"DepthAI version: {getattr(dai, '__version__', 'unknown')}")
    try:
        devices = dai.Device.getAllAvailableDevices()
    except Exception as exc:
        print(f"无法枚举 DepthAI 设备: {exc}")
        return

    if not devices:
        print("DepthAI 枚举到的设备: 0")
        print("可能原因（按优先级）：")
        print("- 设备未插好/线是充电线/USB 口或 Hub 兼容性问题")
        return

    print(f"DepthAI 枚举到的设备: {len(devices)}")
    for idx, dev in enumerate(devices):
        mxid = getattr(dev, "mxid", "unknown")
        state = getattr(dev, "state", "unknown")
        print(f"- [{idx}] mxid={mxid} state={state}")

# 1. 极简管道配置
print("开始构建 DepthAI pipeline...")
try:
    pipeline = dai.Pipeline()
except RuntimeError as exc:
    print(f"创建 Pipeline 失败: {exc}")
    _print_depthai_diagnostics()
    sys.exit(1)
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*ColorCamera node is deprecated.*",
    )
    cam = pipeline.create(dai.node.ColorCamera)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")

# 设为 1080P 和 30FPS
rgb_socket = _oak_rgb_socket()
if rgb_socket is not None:
    for attr in ("setBoardSocket", "setCamera"):
        setter = getattr(cam, attr, None)
        if callable(setter):
            setter(rgb_socket)
            break
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(30)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.video.link(xout.input)

device: dai.Device | None = None
q = None

# 2. 连接设备并测算
print("正在连接 OAK-D... 请稍候")
try:
    def _shutdown() -> None:
        try:
            if device is not None:
                device.close()
        except Exception:
            pass

    device = dai.Device(pipeline, dai.UsbSpeed.SUPER)
    q = device.getOutputQueue("rgb", 4, False)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if device.isPipelineRunning():
            break
        time.sleep(0.02)

    if not device.isPipelineRunning():
        raise RuntimeError("Pipeline 未在超时时间内进入 running 状态")

    print("连接成功！开始输出性能统计报告 (按 Ctrl+C 退出) :\n")

    # --- 统计变量 ---
    count = 0
    window_size = 30  # 每 30 帧（大约 1 秒）输出一次统计报告
    intervals: list[float] = []  # 存放窗口内相邻帧的时间间隔 (Δt, ms)
    window_first_arrival_time: float | None = None
    window_last_arrival_time: float | None = None
    window_prev_arrival_time: float | None = None
    window_frame_count = 0

    while True:
        # 阻塞获取帧
        assert q is not None
        frame = q.get()

        host_arrival_time = time.monotonic()
        timestamp_getter = getattr(frame, "getTimestampDevice", None)
        if not callable(timestamp_getter):
            timestamp_getter = frame.getTimestamp
        hardware_exposure_time = timestamp_getter().total_seconds()
        
        # 1. 计算单帧底层物理延迟
        latency_ms = (host_arrival_time - hardware_exposure_time) * 1000

        # 2. 统计窗口内帧间隔 (Δt)
        if window_frame_count == 0:
            window_first_arrival_time = host_arrival_time
            window_prev_arrival_time = host_arrival_time
            window_last_arrival_time = host_arrival_time
        else:
            assert window_prev_arrival_time is not None
            interval_ms = (host_arrival_time - window_prev_arrival_time) * 1000
            intervals.append(interval_ms)
            window_prev_arrival_time = host_arrival_time
            window_last_arrival_time = host_arrival_time

        window_frame_count += 1
        count += 1

        # 3. 每满一个窗口（30帧），进行一次统计并打印
        if window_frame_count >= window_size:
            if window_first_arrival_time is not None and window_last_arrival_time is not None:
                elapsed = window_last_arrival_time - window_first_arrival_time
            else:
                elapsed = 0.0

            if elapsed > 0 and window_frame_count > 1:
                fps = (window_frame_count - 1) / elapsed
            else:
                fps = 0.0

            if intervals:
                min_int = min(intervals)
                max_int = max(intervals)
                avg_int = sum(intervals) / len(intervals)
                std_dev = statistics.stdev(intervals) if len(intervals) > 1 else 0.0

                print(f"=== 第 {count - window_size + 1:04d} 到 {count:04d} 帧 | 1秒统计报告 ===")
                print(f" 🎯 实际 FPS    : {fps:.2f} Hz")
                print(f" ⏱️ 物理延迟    : {latency_ms:.2f} 毫秒 (当前最新帧)")
                print(f" 📊 帧间隔 (Δt) : 平均 {avg_int:.2f} ms")
                print(f" ⚠️ 抖动/极值   : 最小 {min_int:.2f} ms | 最大 {max_int:.2f} ms | 标准差 {std_dev:.2f} ms")
                print("-" * 55)

            # 重置统计窗口（只清窗口内统计，不影响主循环取帧）
            intervals.clear()
            window_first_arrival_time = None
            window_last_arrival_time = None
            window_prev_arrival_time = None
            window_frame_count = 0

except BrokenPipeError:
    _shutdown()
    sys.exit(0)
except KeyboardInterrupt:
    _shutdown()
    sys.exit(0)
except RuntimeError as exc:
    print(f"连接失败: {exc}")
    _print_depthai_diagnostics()
    _shutdown()
    sys.exit(1)
