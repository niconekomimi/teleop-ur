"""Camera backend adapters backed by the shared-memory camera service."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...hardware.cameras.factory import HardwareFactory


class SharedMemoryCameraBackend:
    def __init__(self, camera_client) -> None:
        self._camera_client = camera_client

    @classmethod
    def create(
        cls,
        camera_type: str,
        *,
        serial_number: str = '',
        enable_depth: bool = False,
        logger: Optional[object] = None,
        connect_timeout_sec: float = 10.0,
    ) -> 'SharedMemoryCameraBackend':
        client = HardwareFactory.create_camera(
            camera_type,
            serial_number=serial_number,
            enable_depth=enable_depth,
            logger=logger,
            connect_timeout_sec=connect_timeout_sec,
        )
        return cls(client)

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        return self._camera_client.get_bgr_frame()

    def stop(self) -> None:
        stop_fn = getattr(self._camera_client, 'stop', None)
        if callable(stop_fn):
            stop_fn()
