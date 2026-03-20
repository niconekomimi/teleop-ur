#!/usr/bin/env python3
"""Unified hardware factory for shared-memory backed cameras."""

from __future__ import annotations

import time
from typing import Dict, Optional

from .interfaces import BaseCamera
from .shm_camera import (
    DEFAULT_CONNECT_TIMEOUT_SEC,
    META_DAEMON_PID,
    META_READY,
    CameraDaemon,
    ShmCameraClient,
    build_camera_spec,
    camera_lock,
    open_shared_memory,
    read_meta_snapshot,
)


class HardwareFactory:
    """Creates business-facing hardware clients and bootstraps daemons on demand."""

    _owned_daemons: Dict[str, CameraDaemon] = {}

    @classmethod
    def create_camera(cls, camera_type: str, **kwargs) -> BaseCamera:
        serial_number = str(kwargs.pop("serial_number", "")).strip()
        logger = kwargs.pop("logger", None)
        connect_timeout_sec = float(kwargs.pop("connect_timeout_sec", DEFAULT_CONNECT_TIMEOUT_SEC))
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported camera factory arguments: {unknown}")

        spec = build_camera_spec(camera_type, serial_number)
        cls._ensure_camera_daemon(
            camera_type=spec.camera_type,
            serial_number=spec.serial_number,
            connect_timeout_sec=connect_timeout_sec,
            logger=logger,
        )
        client = ShmCameraClient(
            spec.camera_type,
            serial_number=spec.serial_number,
            logger=logger,
            connect_timeout_sec=connect_timeout_sec,
        )
        client.start()
        return client

    @classmethod
    def _ensure_camera_daemon(
        cls,
        *,
        camera_type: str,
        serial_number: str,
        connect_timeout_sec: float,
        logger: Optional[object],
    ) -> None:
        spec = build_camera_spec(camera_type, serial_number)
        with camera_lock(spec.lock_path):
            if cls._daemon_ready(spec):
                return

            cls._cleanup_stale_shared_memory(spec)
            daemon = CameraDaemon(
                spec.camera_type,
                spec.serial_number,
                idle_timeout_sec=max(3.0, connect_timeout_sec),
            )
            daemon.start()
            cls._owned_daemons[spec.identifier] = daemon

            deadline = time.monotonic() + max(1.0, connect_timeout_sec)
            while time.monotonic() < deadline:
                if cls._daemon_ready(spec):
                    return
                if not daemon.is_alive():
                    break
                time.sleep(0.05)

        raise TimeoutError(f"Camera daemon failed to start for {spec.identifier}")

    @classmethod
    def _daemon_ready(cls, spec) -> bool:
        meta = read_meta_snapshot(spec)
        if meta is None:
            return False
        daemon_pid = int(meta[META_DAEMON_PID])
        if int(meta[META_READY]) != 1 or daemon_pid <= 0:
            return False
        try:
            import os

            os.kill(daemon_pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @classmethod
    def _cleanup_stale_shared_memory(cls, spec) -> None:
        meta = read_meta_snapshot(spec)
        if meta is not None:
            daemon_pid = int(meta[META_DAEMON_PID])
            if int(meta[META_READY]) == 1 and daemon_pid > 0:
                try:
                    import os

                    os.kill(daemon_pid, 0)
                    return
                except ProcessLookupError:
                    pass
                except PermissionError:
                    return

        for shm_name in (spec.frame_name, spec.meta_name):
            try:
                shm = open_shared_memory(shm_name, unregister=True)
            except FileNotFoundError:
                continue
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            finally:
                try:
                    shm.close()
                except Exception:
                    pass
