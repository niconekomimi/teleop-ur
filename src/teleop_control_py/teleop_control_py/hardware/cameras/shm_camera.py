#!/usr/bin/env python3
"""Shared-memory camera microservice transport."""

from __future__ import annotations

import fcntl
import hashlib
import multiprocessing as mp
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Optional

import numpy as np

from .interfaces import BaseCamera

META_READY = 0
META_HEIGHT = 1
META_WIDTH = 2
META_CHANNELS = 3
META_FRAME_COUNTER = 4
META_ACTIVE_SLOT = 5
META_CLIENT_COUNT = 6
META_STOP_REQUESTED = 7
META_DAEMON_PID = 8
META_RESERVED = 9
META_FIELD_COUNT = 10
META_DTYPE = np.int64
META_NBYTES = META_FIELD_COUNT * np.dtype(META_DTYPE).itemsize
FRAME_SLOT_COUNT = 2
DEFAULT_CONNECT_TIMEOUT_SEC = 10.0
DEFAULT_IDLE_TIMEOUT_SEC = 5.0
SPAWN_CONTEXT = mp.get_context("spawn")


def normalize_camera_type(camera_type: str) -> str:
    normalized = str(camera_type).strip().lower()
    if normalized in {"realsense", "rs"}:
        return "realsense"
    if normalized == "oakd":
        return "oakd"
    raise ValueError(f"Unsupported camera type: {camera_type}")


@dataclass(frozen=True)
class CameraShmSpec:
    camera_type: str
    serial_number: str
    enable_depth: bool
    identifier: str
    meta_name: str
    frame_name: str
    lock_path: str


def build_camera_spec(
    camera_type: str,
    serial_number: str = "",
    *,
    enable_depth: bool = False,
) -> CameraShmSpec:
    normalized_type = normalize_camera_type(camera_type)
    normalized_serial = str(serial_number).strip()
    normalized_enable_depth = bool(enable_depth)
    identity = f"{normalized_type}:{normalized_serial or 'default'}:depth={int(normalized_enable_depth)}"
    digest = hashlib.sha1(identity.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    base_name = f"teleop_cam_{normalized_type}_{digest}"
    return CameraShmSpec(
        camera_type=normalized_type,
        serial_number=normalized_serial,
        enable_depth=normalized_enable_depth,
        identifier=identity,
        meta_name=f"{base_name}_meta",
        frame_name=f"{base_name}_frame",
        lock_path=str(Path("/tmp") / f"{base_name}.lock"),
    )


def _log(logger: Optional[object], level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _unregister_shm_from_tracker(shm: SharedMemory) -> None:
    try:
        resource_tracker.unregister(getattr(shm, "_name", shm.name), "shared_memory")
    except Exception:
        pass


def open_shared_memory(name: str, *, unregister: bool = False) -> SharedMemory:
    shm = SharedMemory(name=name)
    if unregister:
        _unregister_shm_from_tracker(shm)
    return shm


@contextmanager
def camera_lock(lock_path: str):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_meta_snapshot(spec: CameraShmSpec) -> Optional[np.ndarray]:
    try:
        meta_shm = open_shared_memory(spec.meta_name, unregister=True)
    except FileNotFoundError:
        return None

    try:
        meta = np.ndarray((META_FIELD_COUNT,), dtype=META_DTYPE, buffer=meta_shm.buf).copy()
        return meta
    finally:
        meta_shm.close()


class CameraDaemon(SPAWN_CONTEXT.Process):
    """Owns the real SDK camera and publishes frames into shared memory."""

    def __init__(
        self,
        camera_type: str,
        serial_number: str = "",
        *,
        enable_depth: bool = False,
        idle_timeout_sec: float = DEFAULT_IDLE_TIMEOUT_SEC,
    ) -> None:
        spec = build_camera_spec(
            camera_type,
            serial_number,
            enable_depth=enable_depth,
        )
        super().__init__(name=f"camera-daemon[{spec.identifier}]")
        self.spec = spec
        self.idle_timeout_sec = max(1.0, float(idle_timeout_sec))
        self._stop_event = SPAWN_CONTEXT.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _child_log(self, level: str, message: str) -> None:
        print(f"[{self.name}] {level.upper()}: {message}", flush=True)

    def _wait_for_first_frame(self, camera) -> np.ndarray:
        deadline = time.monotonic() + DEFAULT_CONNECT_TIMEOUT_SEC
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            frame = camera.get_bgr_frame()
            if frame is not None:
                return np.ascontiguousarray(frame)
            time.sleep(0.01)
        raise TimeoutError(f"{self.spec.identifier} did not produce an initial frame in time")

    def run(self) -> None:
        camera = None
        meta_shm: Optional[SharedMemory] = None
        frame_shm: Optional[SharedMemory] = None
        meta = None
        frame_slots = None

        try:
            from .camera_client import create_sdk_camera

            camera = create_sdk_camera(
                self.spec.camera_type,
                serial_number=self.spec.serial_number,
                enable_depth=self.spec.enable_depth,
                logger=None,
            )
            camera.start()
            first_frame = self._wait_for_first_frame(camera)
            if first_frame.ndim != 3 or first_frame.dtype != np.uint8:
                raise RuntimeError(
                    f"{self.spec.identifier} produced unsupported frame shape={first_frame.shape} dtype={first_frame.dtype}"
                )

            frame_shape = tuple(int(value) for value in first_frame.shape)
            frame_bytes = int(np.prod(frame_shape, dtype=np.int64))

            meta_shm = SharedMemory(name=self.spec.meta_name, create=True, size=META_NBYTES)
            frame_shm = SharedMemory(
                name=self.spec.frame_name,
                create=True,
                size=FRAME_SLOT_COUNT * frame_bytes,
            )

            meta = np.ndarray((META_FIELD_COUNT,), dtype=META_DTYPE, buffer=meta_shm.buf)
            meta[:] = 0
            frame_slots = np.ndarray((FRAME_SLOT_COUNT, *frame_shape), dtype=np.uint8, buffer=frame_shm.buf)
            np.copyto(frame_slots[0], first_frame)

            meta[META_HEIGHT] = frame_shape[0]
            meta[META_WIDTH] = frame_shape[1]
            meta[META_CHANNELS] = frame_shape[2]
            meta[META_FRAME_COUNTER] = 1
            meta[META_ACTIVE_SLOT] = 0
            meta[META_DAEMON_PID] = os.getpid()
            meta[META_READY] = 1

            last_client_activity = time.monotonic()
            while not self._stop_event.is_set():
                if int(meta[META_STOP_REQUESTED]) == 1:
                    break

                now = time.monotonic()
                if int(meta[META_CLIENT_COUNT]) > 0:
                    last_client_activity = now
                elif (now - last_client_activity) >= self.idle_timeout_sec:
                    break

                frame = camera.get_bgr_frame()
                if frame is None:
                    time.sleep(0.002)
                    continue

                if frame.shape != frame_shape or frame.dtype != np.uint8:
                    self._child_log(
                        "warn",
                        f"Skip frame with unexpected shape={frame.shape} dtype={frame.dtype}, expected={frame_shape}/uint8",
                    )
                    continue

                next_slot = 1 if int(meta[META_ACTIVE_SLOT]) == 0 else 0
                np.copyto(frame_slots[next_slot], frame)
                meta[META_ACTIVE_SLOT] = next_slot
                meta[META_FRAME_COUNTER] = int(meta[META_FRAME_COUNTER]) + 1
        except Exception as exc:  # noqa: BLE001
            self._child_log("error", f"Camera daemon crashed for {self.spec.identifier}: {exc!r}")
        finally:
            if meta is not None:
                try:
                    meta[META_READY] = 0
                    meta[META_STOP_REQUESTED] = 1
                except Exception:
                    pass

            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass

            if frame_shm is not None:
                try:
                    frame_shm.close()
                except Exception:
                    pass
                try:
                    frame_shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass

            if meta_shm is not None:
                try:
                    meta_shm.close()
                except Exception:
                    pass
                try:
                    meta_shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass


class ShmCameraClient(BaseCamera):
    """Read-only shared-memory client that never touches the camera SDK."""

    def __init__(
        self,
        camera_type: str,
        serial_number: str = "",
        *,
        enable_depth: bool = False,
        logger: Optional[object] = None,
        connect_timeout_sec: float = DEFAULT_CONNECT_TIMEOUT_SEC,
    ) -> None:
        self.spec = build_camera_spec(
            camera_type,
            serial_number,
            enable_depth=enable_depth,
        )
        self._logger = logger
        self._connect_timeout_sec = max(0.5, float(connect_timeout_sec))
        self._meta_shm: Optional[SharedMemory] = None
        self._frame_shm: Optional[SharedMemory] = None
        self._meta: Optional[np.ndarray] = None
        self._frame_slots: Optional[np.ndarray] = None
        self._started = False
        self._registered_client = False

    def start(self) -> None:
        if self._started:
            return

        deadline = time.monotonic() + self._connect_timeout_sec
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                if self._meta_shm is None:
                    self._meta_shm = open_shared_memory(self.spec.meta_name, unregister=True)
                    self._meta = np.ndarray((META_FIELD_COUNT,), dtype=META_DTYPE, buffer=self._meta_shm.buf)

                if self._meta is None or int(self._meta[META_READY]) != 1:
                    time.sleep(0.02)
                    continue

                height = int(self._meta[META_HEIGHT])
                width = int(self._meta[META_WIDTH])
                channels = int(self._meta[META_CHANNELS])
                if min(height, width, channels) <= 0:
                    time.sleep(0.02)
                    continue

                if self._frame_shm is None:
                    self._frame_shm = open_shared_memory(self.spec.frame_name, unregister=True)
                    self._frame_slots = np.ndarray(
                        (FRAME_SLOT_COUNT, height, width, channels),
                        dtype=np.uint8,
                        buffer=self._frame_shm.buf,
                    )
                    self._frame_slots.setflags(write=False)

                self._increment_client_count()
                self._started = True
                return
            except FileNotFoundError as exc:
                last_error = exc
                time.sleep(0.02)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                break

        self.stop()
        raise TimeoutError(f"Failed to connect shared camera {self.spec.identifier}: {last_error!r}")

    def _increment_client_count(self) -> None:
        if self._registered_client or self._meta is None:
            return
        with camera_lock(self.spec.lock_path):
            if int(self._meta[META_READY]) != 1:
                raise RuntimeError(f"Shared camera {self.spec.identifier} is not ready")
            self._meta[META_CLIENT_COUNT] = int(self._meta[META_CLIENT_COUNT]) + 1
            self._registered_client = True

    def _decrement_client_count(self) -> None:
        if not self._registered_client:
            return
        if self._meta is None:
            self._registered_client = False
            return
        try:
            with camera_lock(self.spec.lock_path):
                current = int(self._meta[META_CLIENT_COUNT])
                self._meta[META_CLIENT_COUNT] = max(0, current - 1)
        except Exception:
            pass
        finally:
            self._registered_client = False

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        if not self._started:
            self.start()
        if self._meta is None or self._frame_slots is None:
            return None
        if int(self._meta[META_READY]) != 1:
            return None

        for _ in range(4):
            frame_counter_before = int(self._meta[META_FRAME_COUNTER])
            slot_before = int(self._meta[META_ACTIVE_SLOT])
            frame_view = self._frame_slots[slot_before]
            frame_counter_after = int(self._meta[META_FRAME_COUNTER])
            slot_after = int(self._meta[META_ACTIVE_SLOT])
            if (
                frame_counter_before > 0
                and frame_counter_before == frame_counter_after
                and slot_before == slot_after
            ):
                return frame_view

        return self._frame_slots[int(self._meta[META_ACTIVE_SLOT])]

    def stop(self) -> None:
        self._decrement_client_count()

        if self._frame_shm is not None:
            try:
                self._frame_shm.close()
            except Exception:
                pass
        if self._meta_shm is not None:
            try:
                self._meta_shm.close()
            except Exception:
                pass

        self._frame_shm = None
        self._meta_shm = None
        self._frame_slots = None
        self._meta = None
        self._started = False

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
