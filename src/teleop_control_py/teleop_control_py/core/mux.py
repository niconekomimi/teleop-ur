"""Action routing and source arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .models import ActionCommand, ControlSource
from ..device_manager.backends.interfaces import ArmBackend, GripperBackend


DEFAULT_PRIORITY: dict[ControlSource, int] = {
    ControlSource.NONE: 0,
    ControlSource.TELEOP: 10,
    ControlSource.INFERENCE: 20,
    ControlSource.COMMANDER: 80,
    ControlSource.SAFETY: 100,
}


@dataclass(frozen=True)
class MuxDispatchResult:
    accepted: bool
    active_source: ControlSource
    reason: str = ""


class ActionMux:
    def __init__(
        self,
        arm_backend: Optional[ArmBackend],
        gripper_backend: Optional[GripperBackend],
        *,
        active_source: ControlSource = ControlSource.NONE,
        priority_map: Optional[Mapping[ControlSource, int]] = None,
        logger: Optional[object] = None,
    ) -> None:
        self._arm_backend = arm_backend
        self._gripper_backend = gripper_backend
        self._active_source = ControlSource(active_source)
        self._priorities = dict(DEFAULT_PRIORITY)
        if priority_map:
            self._priorities.update({ControlSource(key): int(value) for key, value in priority_map.items()})
        self._hold_active = False
        self._hold_reason = ""
        self._logger = logger

    @property
    def active_source(self) -> ControlSource:
        return self._active_source

    def set_active_source(self, source: ControlSource) -> None:
        self._active_source = ControlSource(source)

    def set_hold(self, active: bool, reason: str = "") -> None:
        self._hold_active = bool(active)
        self._hold_reason = str(reason)

    def publish(self, command: ActionCommand) -> MuxDispatchResult:
        source = ControlSource(command.source)
        if self._hold_active and source != ControlSource.SAFETY:
            return MuxDispatchResult(False, self._active_source, self._hold_reason or 'mux_hold')

        active_priority = self._priorities.get(self._active_source, 0)
        source_priority = self._priorities.get(source, 0)
        if self._active_source not in {ControlSource.NONE, source} and source_priority < active_priority:
            return MuxDispatchResult(False, self._active_source, 'lower_priority_source')

        if source_priority >= active_priority:
            self._active_source = source

        if self._arm_backend is not None:
            self._arm_backend.send_delta_twist(command)
        if self._gripper_backend is not None:
            self._gripper_backend.set_gripper(command.gripper)
        return MuxDispatchResult(True, self._active_source)

    def publish_zero(self, source: ControlSource = ControlSource.SAFETY) -> MuxDispatchResult:
        command = ActionCommand.zero(source=source)
        return self.publish(command)
