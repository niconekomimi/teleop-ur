#!/usr/bin/env python3
"""Remote openpi inference worker that fits the local teleop GUI contract."""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from teleop_control_py.device_manager import SharedMemoryCameraBackend

from .inference_worker import DEFAULT_PREVIEW_HZ, InferenceActionSample

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
except Exception as exc:  # noqa: BLE001
    image_tools = None
    websocket_client_policy = None
    _OPENPI_IMPORT_ERROR = exc
else:
    _OPENPI_IMPORT_ERROR = None


def _normalize_camera_source(source: str) -> str:
    value = str(source).strip().lower()
    if value in {"realsense", "rs"}:
        return "realsense"
    if value == "oakd":
        return "oakd"
    return value


def _copy_preview_frame(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if frame is None:
        return None
    return np.ascontiguousarray(frame).copy()


def _timing_ms(payload: object, key: str) -> float:
    if not isinstance(payload, dict):
        return 0.0
    try:
        return float(payload.get(key, 0.0))
    except Exception:
        return 0.0


class OpenPiRemoteWorker(QThread):
    action_signal = Signal(object)
    preview_signal = Signal(object, object)
    status_signal = Signal(str)
    log_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(
        self,
        *,
        openpi_host: str,
        openpi_port: int,
        openpi_api_key: str = "",
        remote_prompt: str = "",
        global_camera_source: str,
        wrist_camera_source: str,
        loop_hz: float,
        state_provider: Optional[Callable[[], object]] = None,
        prompt_provider: Optional[Callable[[], str]] = None,
        global_camera_serial_number: str = "",
        wrist_camera_serial_number: str = "",
        global_camera_enable_depth: bool = False,
        wrist_camera_enable_depth: bool = False,
        preview_hz: float = DEFAULT_PREVIEW_HZ,
        image_size: int = 224,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.openpi_host = str(openpi_host).strip() or "127.0.0.1"
        self.openpi_port = max(1, int(openpi_port))
        self.openpi_api_key = str(openpi_api_key).strip()
        self.remote_prompt = str(remote_prompt).strip()
        self.global_camera_source = _normalize_camera_source(global_camera_source)
        self.wrist_camera_source = _normalize_camera_source(wrist_camera_source)
        self.global_camera_serial_number = str(global_camera_serial_number).strip()
        self.wrist_camera_serial_number = str(wrist_camera_serial_number).strip()
        self.global_camera_enable_depth = bool(global_camera_enable_depth)
        self.wrist_camera_enable_depth = bool(wrist_camera_enable_depth)
        self.loop_hz = max(0.2, float(loop_hz))
        self.preview_hz = max(0.2, float(preview_hz))
        self.image_size = max(32, int(image_size))
        self.state_provider = state_provider
        self.prompt_provider = prompt_provider
        self._running = True
        self._camera_fetch_lock = threading.Lock()
        self._preview_enabled_event = threading.Event()
        self._preview_stop_event = threading.Event()
        self._preview_thread: Optional[threading.Thread] = None
        self._policy_client = None

    def stop(self) -> None:
        self._running = False
        self._preview_enabled_event.set()
        self._preview_stop_event.set()
        policy_client = self._policy_client
        websocket = getattr(policy_client, "_ws", None)
        if websocket is not None:
            try:
                websocket.close()
            except Exception:
                pass

    def set_preview_streaming(self, enabled: bool) -> None:
        if enabled:
            self._preview_enabled_event.set()
            return
        self._preview_enabled_event.clear()

    def request_task_update(self, task_name: str, task_embedding_path: str) -> None:
        del task_name, task_embedding_path

    def _build_camera_client(
        self,
        source: str,
        serial_number: str = "",
        *,
        enable_depth: bool = False,
    ):
        return SharedMemoryCameraBackend.create(
            source,
            serial_number=serial_number,
            enable_depth=enable_depth,
            logger=self,
        )

    def _read_camera_pair(self, global_camera, wrist_camera) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        with self._camera_fetch_lock:
            fetch_start = time.perf_counter()
            global_bgr = global_camera.get_bgr_frame()
            wrist_bgr = wrist_camera.get_bgr_frame()
            fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
        return global_bgr, wrist_bgr, float(fetch_ms)

    def _emit_preview_frame_pair(self, global_bgr, wrist_bgr) -> None:
        if global_bgr is None and wrist_bgr is None:
            return
        self.preview_signal.emit(
            _copy_preview_frame(global_bgr),
            _copy_preview_frame(wrist_bgr),
        )

    def _preview_loop(self, global_camera, wrist_camera) -> None:
        target_period = 1.0 / max(self.preview_hz, 1e-6)
        next_cycle = time.perf_counter()
        while self._running and not self._preview_stop_event.is_set():
            if not self._preview_enabled_event.is_set():
                next_cycle = time.perf_counter()
                if self._preview_stop_event.wait(0.1):
                    break
                continue

            try:
                global_bgr, wrist_bgr, _fetch_ms = self._read_camera_pair(global_camera, wrist_camera)
            except Exception:
                if self._preview_stop_event.wait(min(target_period, 0.1)):
                    break
                next_cycle = time.perf_counter()
                continue

            if global_bgr is not None or wrist_bgr is not None:
                self._emit_preview_frame_pair(global_bgr, wrist_bgr)

            next_cycle += target_period
            sleep_duration = next_cycle - time.perf_counter()
            if sleep_duration > 0:
                if self._preview_stop_event.wait(sleep_duration):
                    break
            else:
                next_cycle = time.perf_counter()

    def _start_preview_thread(self, global_camera, wrist_camera) -> None:
        thread = self._preview_thread
        if thread is not None and thread.is_alive():
            return
        self._preview_stop_event.clear()
        self._preview_thread = threading.Thread(
            target=self._preview_loop,
            args=(global_camera, wrist_camera),
            name="OpenPiInferencePreviewThread",
            daemon=True,
        )
        self._preview_thread.start()

    def _stop_preview_thread(self, timeout_sec: float = 1.0) -> None:
        thread = self._preview_thread
        self._preview_thread = None
        self._preview_enabled_event.set()
        self._preview_stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_sec)

    def _wait_for_initial_frames(self, global_camera, wrist_camera) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        while self._running:
            global_bgr, wrist_bgr, _fetch_ms = self._read_camera_pair(global_camera, wrist_camera)
            if global_bgr is not None and wrist_bgr is not None:
                return global_bgr, wrist_bgr

            self.status_signal.emit("等待首帧")
            time.sleep(0.01)

        return None, None

    def _read_robot_state(self) -> Optional[dict[str, object]]:
        if self.state_provider is None:
            return None
        try:
            state = self.state_provider()
        except Exception as exc:
            self.log_signal.emit(f"读取机器人状态失败: {exc}")
            return None

        if state is None:
            return None

        if isinstance(state, dict):
            snapshot: dict[str, object] = {}
            for key, value in state.items():
                if isinstance(value, list):
                    snapshot[key] = list(value)
                else:
                    snapshot[key] = value
            return snapshot

        try:
            array = np.asarray(state, dtype=np.float32).reshape(-1)
        except Exception:
            return None

        if array.size < 7:
            return None

        return {
            "joints": [float(value) for value in array[:6]],
            "gripper": float(array[6]),
        }

    def _current_prompt(self) -> str:
        prompt = self.remote_prompt
        if self.prompt_provider is not None:
            try:
                candidate = self.prompt_provider()
            except Exception as exc:
                self.log_signal.emit(f"读取 openpi prompt 失败，继续使用当前值: {exc}")
            else:
                candidate_text = str(candidate).strip()
                if candidate_text:
                    prompt = candidate_text
        return str(prompt).strip()

    def _ensure_ready_imports(self) -> None:
        if image_tools is not None and websocket_client_policy is not None:
            return
        raise RuntimeError(
            "openpi-client 未安装，无法启用 openpi 远端后端。"
            f"{f' 原始错误: {_OPENPI_IMPORT_ERROR}' if _OPENPI_IMPORT_ERROR else ''}"
        )

    def _wait_for_remote_policy(self):
        endpoint = f"{self.openpi_host}:{self.openpi_port}"
        last_log_sec = 0.0

        while self._running:
            try:
                if not self.openpi_host.startswith("ws"):
                    with socket.create_connection((self.openpi_host, self.openpi_port), timeout=1.0):
                        pass

                client = websocket_client_policy.WebsocketClientPolicy(
                    host=self.openpi_host,
                    port=self.openpi_port,
                    api_key=self.openpi_api_key or None,
                )
                metadata = client.get_server_metadata() or {}
                self.log_signal.emit(
                    "openpi 远端已连接: "
                    f"endpoint={endpoint}, metadata={metadata}"
                )
                return client
            except Exception as exc:
                now = time.monotonic()
                self.status_signal.emit("等待 openpi 远端服务")
                if now - last_log_sec >= 2.0:
                    self.log_signal.emit(f"等待 openpi 远端服务 {endpoint}: {exc}")
                    last_log_sec = now
                time.sleep(0.2)

        return None

    def _preprocess_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        resized = image_tools.resize_with_pad(rgb, self.image_size, self.image_size)
        resized = image_tools.convert_to_uint8(np.asarray(resized))
        return np.ascontiguousarray(resized, dtype=np.uint8)

    def _build_observation(
        self,
        global_bgr: np.ndarray,
        wrist_bgr: np.ndarray,
        robot_state: dict[str, object],
        prompt: str,
    ) -> dict[str, object]:
        joints_raw = robot_state.get("joints", [])
        gripper_raw = robot_state.get("gripper", None)
        joints = np.asarray(joints_raw, dtype=np.float32).reshape(-1)
        if joints.size != 6 or gripper_raw is None:
            raise RuntimeError("当前机器人状态不完整，缺少 6 维关节角或夹爪状态。")

        gripper = np.asarray([float(gripper_raw)], dtype=np.float32)
        return {
            "observation/image": self._preprocess_image(global_bgr),
            "observation/wrist_image": self._preprocess_image(wrist_bgr),
            "observation/joint_position": joints,
            "observation/gripper_position": gripper,
            "prompt": prompt,
        }

    def _request_action_chunk(self, policy_client, global_camera, wrist_camera):
        global_bgr, wrist_bgr, camera_fetch_ms = self._read_camera_pair(global_camera, wrist_camera)
        if global_bgr is None or wrist_bgr is None:
            self.status_signal.emit("等待相机帧")
            return None

        self._emit_preview_frame_pair(global_bgr, wrist_bgr)

        robot_state_start = time.perf_counter()
        robot_state = self._read_robot_state()
        robot_state_ms = (time.perf_counter() - robot_state_start) * 1000.0
        if not robot_state:
            self.status_signal.emit("等待机器人状态")
            return None

        prompt = self._current_prompt()
        if not prompt:
            raise RuntimeError("openpi 远端推理需要 prompt。")

        preprocess_start = time.perf_counter()
        observation = self._build_observation(global_bgr, wrist_bgr, robot_state, prompt)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        policy_call_start = time.perf_counter()
        result = policy_client.infer(observation)
        policy_call_ms = (time.perf_counter() - policy_call_start) * 1000.0

        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2 or actions.shape[1] < 7:
            raise RuntimeError(f"openpi 返回动作维度异常: {actions.shape}")

        action_chunk = np.ascontiguousarray(actions[:, :7], dtype=np.float32)
        policy_timing = result.get("policy_timing")
        server_timing = result.get("server_timing")
        return {
            "action_chunk": action_chunk,
            "camera_fetch_ms": float(camera_fetch_ms),
            "preprocess_ms": float(preprocess_ms),
            "robot_state_ms": float(robot_state_ms),
            "policy_call_ms": float(policy_call_ms),
            "policy_infer_ms": _timing_ms(policy_timing, "infer_ms"),
            "server_infer_ms": _timing_ms(server_timing, "infer_ms"),
        }

    def run(self) -> None:
        global_camera = None
        wrist_camera = None
        policy_client = None
        target_period = 1.0 / max(self.loop_hz, 1e-6)
        completed_cycles: deque[float] = deque(maxlen=max(5, int(round(self.loop_hz * 2))))
        current_chunk: Optional[np.ndarray] = None
        current_chunk_idx = 0
        current_chunk_len = 1
        last_remote_ms = 0.0

        try:
            self._ensure_ready_imports()
            self.status_signal.emit("连接 openpi 中")
            policy_client = self._wait_for_remote_policy()
            if not self._running or policy_client is None:
                return
            self._policy_client = policy_client

            self.status_signal.emit("打开相机中")
            global_camera = self._build_camera_client(
                self.global_camera_source,
                self.global_camera_serial_number,
                enable_depth=self.global_camera_enable_depth,
            )
            wrist_camera = self._build_camera_client(
                self.wrist_camera_source,
                self.wrist_camera_serial_number,
                enable_depth=self.wrist_camera_enable_depth,
            )

            global_bgr, wrist_bgr = self._wait_for_initial_frames(global_camera, wrist_camera)
            if not self._running:
                return
            if global_bgr is None or wrist_bgr is None:
                raise RuntimeError("相机已打开，但未收到首帧")

            self._emit_preview_frame_pair(global_bgr, wrist_bgr)
            self._start_preview_thread(global_camera, wrist_camera)

            self.log_signal.emit(
                "openpi 远端推理已启动: "
                f"host={self.openpi_host}, port={self.openpi_port}, "
                f"prompt={self._current_prompt() or '<empty>'}, "
                f"cameras=({self.global_camera_source}, {self.wrist_camera_source}), "
                f"serials=({self.global_camera_serial_number or 'auto'}, {self.wrist_camera_serial_number or 'auto'}), "
                f"depth=({self.global_camera_enable_depth}, {self.wrist_camera_enable_depth})"
            )

            next_cycle = time.perf_counter()
            self.status_signal.emit("相机已就绪")
            while self._running:
                cycle_start = time.perf_counter()
                is_replan_step = False
                camera_fetch_ms = 0.0
                preprocess_ms = 0.0
                robot_state_ms = 0.0
                policy_call_ms = 0.0

                if current_chunk is None or current_chunk_idx >= current_chunk.shape[0]:
                    self.status_signal.emit("请求 openpi 动作块")
                    request_result = self._request_action_chunk(policy_client, global_camera, wrist_camera)
                    if request_result is None:
                        time.sleep(min(target_period, 0.05))
                        next_cycle = time.perf_counter()
                        continue

                    current_chunk = request_result["action_chunk"]
                    current_chunk_idx = 0
                    current_chunk_len = max(1, int(current_chunk.shape[0]))
                    is_replan_step = True
                    camera_fetch_ms = float(request_result["camera_fetch_ms"])
                    preprocess_ms = float(request_result["preprocess_ms"])
                    robot_state_ms = float(request_result["robot_state_ms"])
                    policy_call_ms = float(request_result["policy_call_ms"])
                    last_remote_ms = max(
                        float(request_result["server_infer_ms"]),
                        float(request_result["policy_infer_ms"]),
                        policy_call_ms,
                    )

                if current_chunk is None or current_chunk_idx >= current_chunk.shape[0]:
                    time.sleep(min(target_period, 0.05))
                    next_cycle = time.perf_counter()
                    continue

                plan_step_idx = int(current_chunk_idx)
                action = np.asarray(current_chunk[current_chunk_idx], dtype=np.float32).reshape(-1)
                current_chunk_idx += 1

                now = time.perf_counter()
                cycle_compute_ms = (now - cycle_start) * 1000.0
                self.action_signal.emit(
                    InferenceActionSample(
                        action=action,
                        cycle_compute_ms=float(cycle_compute_ms),
                        camera_fetch_ms=float(camera_fetch_ms),
                        preprocess_ms=float(preprocess_ms),
                        robot_state_ms=float(robot_state_ms),
                        policy_call_ms=float(policy_call_ms),
                        is_replan_step=bool(is_replan_step),
                        plan_step_idx=int(plan_step_idx),
                        replan_every=int(current_chunk_len),
                    )
                )

                completed_cycles.append(now)
                if len(completed_cycles) >= 2:
                    window_elapsed = max(completed_cycles[-1] - completed_cycles[0], 1e-6)
                    actual_hz = (len(completed_cycles) - 1) / window_elapsed
                else:
                    actual_hz = 0.0

                phase_label = (
                    f"重规划 {plan_step_idx + 1}/{current_chunk_len}"
                    if is_replan_step
                    else f"缓存步 {plan_step_idx + 1}/{current_chunk_len}"
                )
                self.status_signal.emit(
                    "运行中 | "
                    f"openpi 输出 {actual_hz:.2f} Hz | "
                    f"远端 {last_remote_ms:.1f} ms | "
                    f"{phase_label}"
                )

                next_cycle += target_period
                sleep_duration = next_cycle - time.perf_counter()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                else:
                    next_cycle = time.perf_counter()
        except Exception as exc:
            self.error_signal.emit(str(exc))
        finally:
            self._stop_preview_thread()
            self._policy_client = None
            websocket = getattr(policy_client, "_ws", None)
            if websocket is not None:
                try:
                    websocket.close()
                except Exception:
                    pass

            for camera in (global_camera, wrist_camera):
                if camera is None:
                    continue
                try:
                    camera.stop()
                except Exception:
                    pass

            self.status_signal.emit("未启动")
