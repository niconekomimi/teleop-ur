#!/usr/bin/env python3
import os
import time
import threading
import warnings
import statistics

import cv2
import numpy as np
import depthai as dai
import pyrealsense2 as rs

os.environ.setdefault("DEPTHAI_SEARCH_TIMEOUT", "5000")

# =========================
# 参数
# =========================
OAK_FPS = 30
RS_FPS = 30
WINDOW_SIZE = 30

RS_WIDTH = 1280
RS_HEIGHT = 720

DISPLAY_W = 960
DISPLAY_H = 540

WIN_NAME = "OAK-D + RealSense 30Hz Test"

exit_flag = False


def _oak_socket(name: str):
    return getattr(dai.CameraBoardSocket, name, None)


def _oak_rgb_socket():
    return _oak_socket("CAM_A") or _oak_socket("RGB")

# -------------------------
# OAK 共享状态
# -------------------------
oak_latest_frame = None
oak_frame_lock = threading.Lock()

oak_stats = {
    "fps": 0.0,
    "latency_ms": 0.0,
    "avg_dt_ms": 0.0,
    "min_dt_ms": 0.0,
    "max_dt_ms": 0.0,
    "jitter_ms": 0.0,
}
oak_stats_lock = threading.Lock()

# -------------------------
# RealSense 共享状态
# -------------------------
rs_latest_frame = None
rs_frame_lock = threading.Lock()

rs_stats = {
    "fps": 0.0,
    "latency_ms": 0.0,
    "avg_dt_ms": 0.0,
    "min_dt_ms": 0.0,
    "max_dt_ms": 0.0,
    "jitter_ms": 0.0,
}
rs_stats_lock = threading.Lock()


# =========================
# OAK-D
# =========================
def build_oak_pipeline():
    pipeline = dai.Pipeline()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*ColorCamera node is deprecated.*",
        )
        cam = pipeline.create(dai.node.ColorCamera)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")

    rgb_socket = _oak_rgb_socket()
    if rgb_socket is not None:
        for attr in ("setBoardSocket", "setCamera"):
            setter = getattr(cam, attr, None)
            if callable(setter):
                setter(rgb_socket)
                break
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(OAK_FPS)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.video.link(xout.input)

    return pipeline


def oak_capture_loop(pipeline):
    global oak_latest_frame, exit_flag

    device = None
    try:
        device = dai.Device(pipeline, dai.UsbSpeed.SUPER)
        q = device.getOutputQueue("rgb", 2, False)

        deadline = time.monotonic() + 5.0
        while not device.isPipelineRunning():
            if time.monotonic() > deadline:
                raise RuntimeError("OAK pipeline 未在 5 秒内进入 running 状态")
            time.sleep(0.005)

        print("OAK-D 已连接。")

        frame_count = 0
        total_count = 0
        first_t = None
        prev_t = None
        last_t = None
        intervals_ms = []

        while not exit_flag:
            frame_obj = q.get()
            now = time.monotonic()

            with oak_frame_lock:
                oak_latest_frame = frame_obj

            timestamp_getter = getattr(frame_obj, "getTimestampDevice", None)
            if not callable(timestamp_getter):
                timestamp_getter = frame_obj.getTimestamp
            hw_ts = timestamp_getter().total_seconds()
            latency_ms = (now - hw_ts) * 1000.0

            if frame_count == 0:
                first_t = now
                prev_t = now
                last_t = now
            else:
                dt_ms = (now - prev_t) * 1000.0
                intervals_ms.append(dt_ms)
                prev_t = now
                last_t = now

            frame_count += 1
            total_count += 1

            if frame_count >= WINDOW_SIZE:
                elapsed = (last_t - first_t) if (first_t is not None and last_t is not None) else 0.0
                fps = (frame_count - 1) / elapsed if elapsed > 0 and frame_count > 1 else 0.0

                if intervals_ms:
                    avg_dt = sum(intervals_ms) / len(intervals_ms)
                    min_dt = min(intervals_ms)
                    max_dt = max(intervals_ms)
                    jitter = statistics.stdev(intervals_ms) if len(intervals_ms) > 1 else 0.0
                else:
                    avg_dt = 0.0
                    min_dt = 0.0
                    max_dt = 0.0
                    jitter = 0.0

                with oak_stats_lock:
                    oak_stats["fps"] = fps
                    oak_stats["latency_ms"] = latency_ms
                    oak_stats["avg_dt_ms"] = avg_dt
                    oak_stats["min_dt_ms"] = min_dt
                    oak_stats["max_dt_ms"] = max_dt
                    oak_stats["jitter_ms"] = jitter

                print(
                    f"[OAK] 帧 {total_count - WINDOW_SIZE + 1:04d}-{total_count:04d} | "
                    f"FPS={fps:.2f} Hz | Latency={latency_ms:.2f} ms | "
                    f"Δt(avg/min/max)=({avg_dt:.2f}/{min_dt:.2f}/{max_dt:.2f}) ms | "
                    f"Jitter={jitter:.2f} ms"
                )

                frame_count = 0
                first_t = None
                prev_t = None
                last_t = None
                intervals_ms.clear()

    except Exception as exc:
        print(f"OAK 线程异常退出: {exc}")
        exit_flag = True
    finally:
        exit_flag = True
        try:
            if device is not None:
                device.close()
        except Exception:
            pass


# =========================
# RealSense
# =========================
def realsense_capture_loop():
    global rs_latest_frame, exit_flag

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)

    try:
        profile = pipeline.start(config)
        print("RealSense 已连接。")

        sensor = profile.get_device().first_color_sensor()
        try:
            sensor.set_option(rs.option.enable_auto_exposure, 1)
        except Exception:
            pass

        frame_count = 0
        total_count = 0
        first_t = None
        prev_t = None
        last_t = None
        intervals_ms = []

        while not exit_flag:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            now = time.monotonic()
            img = np.asanyarray(color_frame.get_data())

            with rs_frame_lock:
                rs_latest_frame = img

            # 注意：这个 latency 只能看相对变化，不建议和 OAK 绝对值直接硬比
            try:
                hw_ts_ms = color_frame.get_timestamp()
                latency_ms = now * 1000.0 - hw_ts_ms
            except Exception:
                latency_ms = 0.0

            if frame_count == 0:
                first_t = now
                prev_t = now
                last_t = now
            else:
                dt_ms = (now - prev_t) * 1000.0
                intervals_ms.append(dt_ms)
                prev_t = now
                last_t = now

            frame_count += 1
            total_count += 1

            if frame_count >= WINDOW_SIZE:
                elapsed = (last_t - first_t) if (first_t is not None and last_t is not None) else 0.0
                fps = (frame_count - 1) / elapsed if elapsed > 0 and frame_count > 1 else 0.0

                if intervals_ms:
                    avg_dt = sum(intervals_ms) / len(intervals_ms)
                    min_dt = min(intervals_ms)
                    max_dt = max(intervals_ms)
                    jitter = statistics.stdev(intervals_ms) if len(intervals_ms) > 1 else 0.0
                else:
                    avg_dt = 0.0
                    min_dt = 0.0
                    max_dt = 0.0
                    jitter = 0.0

                with rs_stats_lock:
                    rs_stats["fps"] = fps
                    rs_stats["latency_ms"] = latency_ms
                    rs_stats["avg_dt_ms"] = avg_dt
                    rs_stats["min_dt_ms"] = min_dt
                    rs_stats["max_dt_ms"] = max_dt
                    rs_stats["jitter_ms"] = jitter

                print(
                    f"[RS ] 帧 {total_count - WINDOW_SIZE + 1:04d}-{total_count:04d} | "
                    f"FPS={fps:.2f} Hz | Latency={latency_ms:.2f} ms | "
                    f"Δt(avg/min/max)=({avg_dt:.2f}/{min_dt:.2f}/{max_dt:.2f}) ms | "
                    f"Jitter={jitter:.2f} ms"
                )

                frame_count = 0
                first_t = None
                prev_t = None
                last_t = None
                intervals_ms.clear()

    except Exception as exc:
        print(f"RealSense 线程异常退出: {exc}")
        exit_flag = True
    finally:
        exit_flag = True
        try:
            pipeline.stop()
        except Exception:
            pass


# =========================
# 显示辅助
# =========================
def make_waiting_frame(label, width=DISPLAY_W, height=DISPLAY_H):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        f"{label}: waiting...",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    return img


def normalize_for_display(img):
    return cv2.resize(img, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)


def draw_overlay(img, title, s):
    panel_x, panel_y = 20, 20
    panel_w, panel_h = 430, 205

    cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title,                                (panel_x + 12, panel_y + 28),  font, 0.75, (255, 255, 255), 2)
    cv2.putText(img, f"FPS       : {s['fps']:.2f} Hz",     (panel_x + 12, panel_y + 58),  font, 0.62, (0, 255, 0),   2)
    cv2.putText(img, f"Latency   : {s['latency_ms']:.2f} ms", (panel_x + 12, panel_y + 86),  font, 0.58, (0, 255, 255), 2)
    cv2.putText(img, f"Avg DT    : {s['avg_dt_ms']:.2f} ms",  (panel_x + 12, panel_y + 114), font, 0.58, (255, 255, 0), 2)
    cv2.putText(img, f"Min DT    : {s['min_dt_ms']:.2f} ms",  (panel_x + 12, panel_y + 142), font, 0.58, (255, 200, 0), 2)
    cv2.putText(img, f"Max DT    : {s['max_dt_ms']:.2f} ms",  (panel_x + 12, panel_y + 170), font, 0.58, (0, 200, 255), 2)
    cv2.putText(img, f"Jitter SD : {s['jitter_ms']:.2f} ms",  (panel_x + 12, panel_y + 198), font, 0.58, (255, 0, 255), 2)


# =========================
# 显示
# =========================
def display_loop():
    global exit_flag

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    print("显示窗口已打开，按 q 退出。")

    while not exit_flag:
        with oak_frame_lock:
            oak_frame_obj = oak_latest_frame

        with rs_frame_lock:
            rs_img_raw = rs_latest_frame.copy() if rs_latest_frame is not None else None

        if oak_frame_obj is not None:
            oak_img = oak_frame_obj.getCvFrame()
            oak_img = normalize_for_display(oak_img)
        else:
            oak_img = make_waiting_frame("OAK-D")

        if rs_img_raw is not None:
            rs_img = normalize_for_display(rs_img_raw)
        else:
            rs_img = make_waiting_frame("RealSense")

        with oak_stats_lock:
            oak_s = dict(oak_stats)
        with rs_stats_lock:
            rs_s = dict(rs_stats)

        draw_overlay(oak_img, "OAK-D", oak_s)
        draw_overlay(rs_img, "RealSense", rs_s)

        show = cv2.hconcat([oak_img, rs_img])
        cv2.imshow(WIN_NAME, show)

        if cv2.waitKey(1) == ord("q"):
            exit_flag = True
            break

    cv2.destroyAllWindows()


# =========================
# main
# =========================
def main():
    oak_pipeline = build_oak_pipeline()

    t_oak = threading.Thread(target=oak_capture_loop, args=(oak_pipeline,), daemon=True)
    t_rs = threading.Thread(target=realsense_capture_loop, daemon=True)

    t_oak.start()
    t_rs.start()

    try:
        display_loop()
    except KeyboardInterrupt:
        pass

    global exit_flag
    exit_flag = True
    t_oak.join(timeout=1.0)
    t_rs.join(timeout=1.0)
    print("测试结束。")


if __name__ == "__main__":
    main()
