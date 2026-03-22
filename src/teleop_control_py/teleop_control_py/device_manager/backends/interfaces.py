"""Protocol-style interfaces for the staged architecture migration."""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from ...core.models import ActionCommand


class InputBackend(Protocol):
    def get_action_command(self) -> ActionCommand:
        ...

    def stop(self) -> None:
        ...


class ArmBackend(Protocol):
    def send_delta_twist(self, command: ActionCommand) -> None:
        ...

    def send_zero_twist(self) -> None:
        ...

    def stop(self) -> None:
        ...


class GripperBackend(Protocol):
    def set_gripper(self, value: float) -> None:
        ...

    def stop(self) -> None:
        ...


class CameraBackend(Protocol):
    def get_bgr_frame(self) -> Optional[np.ndarray]:
        ...

    def stop(self) -> None:
        ...
