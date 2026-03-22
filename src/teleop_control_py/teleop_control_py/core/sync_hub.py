"""Reusable synchronization hub for camera, state, and action snapshots."""

from __future__ import annotations

from typing import Callable, Optional

from .models import CameraFrameSet, ObservationSnapshot, RobotStateSnapshot


CameraProvider = Callable[[], Optional[CameraFrameSet]]
StateProvider = Callable[[CameraFrameSet], tuple[Optional[RobotStateSnapshot], Optional[str], Optional[str]]]
ActionProvider = Callable[[CameraFrameSet, RobotStateSnapshot], tuple[Optional[object], Optional[str], Optional[str]]]


class SyncHub:
    def __init__(
        self,
        camera_provider: CameraProvider,
        state_provider: StateProvider,
        action_provider: Optional[ActionProvider] = None,
    ) -> None:
        self._camera_provider = camera_provider
        self._state_provider = state_provider
        self._action_provider = action_provider

    def capture_snapshot(self) -> tuple[Optional[ObservationSnapshot], Optional[str], Optional[str]]:
        frame_set = self._camera_provider()
        if frame_set is None:
            return None, 'camera_empty', None

        state, reason, detail = self._state_provider(frame_set)
        if state is None:
            return None, reason or 'state_unavailable', detail

        action_vector = None
        if self._action_provider is not None:
            action_vector, action_reason, action_detail = self._action_provider(frame_set, state)
            if action_vector is None:
                return None, action_reason or 'action_unavailable', action_detail

        return ObservationSnapshot(camera_frames=frame_set, robot_state=state, action_vector=action_vector), None, None
