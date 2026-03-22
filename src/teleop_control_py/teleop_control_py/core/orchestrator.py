"""Top-level runtime state machine for the staged architecture migration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import ControlSource


class SystemPhase(str, Enum):
    IDLE = "idle"
    TELEOP = "teleop"
    HOMING = "homing"
    HOME_ZONE = "home_zone"
    INFERENCE_READY = "inference_ready"
    INFERENCE_EXECUTING = "inference_executing"
    ESTOP = "estop"


@dataclass(frozen=True)
class OrchestratorState:
    phase: SystemPhase = SystemPhase.IDLE
    recording_active: bool = False
    teleop_running: bool = False
    inference_ready: bool = False
    inference_executing: bool = False
    home_zone_active: bool = False
    homing_active: bool = False
    estopped: bool = False
    active_source: ControlSource = ControlSource.NONE


@dataclass(frozen=True)
class TransitionDecision:
    allowed: bool
    reason: str
    state: OrchestratorState


class SystemOrchestrator:
    def __init__(self) -> None:
        self._state = OrchestratorState()

    def snapshot(self) -> OrchestratorState:
        return self._state

    def _recompute_state(self, **changes) -> OrchestratorState:
        recording_active = bool(changes.get("recording_active", self._state.recording_active))
        teleop_running = bool(changes.get("teleop_running", self._state.teleop_running))
        inference_ready = bool(changes.get("inference_ready", self._state.inference_ready))
        inference_executing = bool(changes.get("inference_executing", self._state.inference_executing))
        home_zone_active = bool(changes.get("home_zone_active", self._state.home_zone_active))
        homing_active = bool(changes.get("homing_active", self._state.homing_active))
        estopped = bool(changes.get("estopped", self._state.estopped))

        if estopped:
            phase = SystemPhase.ESTOP
            active_source = ControlSource.SAFETY
        elif home_zone_active:
            phase = SystemPhase.HOME_ZONE
            active_source = ControlSource.COMMANDER
        elif homing_active:
            phase = SystemPhase.HOMING
            active_source = ControlSource.COMMANDER
        elif inference_executing:
            phase = SystemPhase.INFERENCE_EXECUTING
            active_source = ControlSource.INFERENCE
        elif teleop_running:
            phase = SystemPhase.TELEOP
            active_source = ControlSource.TELEOP
        elif inference_ready:
            phase = SystemPhase.INFERENCE_READY
            active_source = ControlSource.NONE
        else:
            phase = SystemPhase.IDLE
            active_source = ControlSource.NONE

        self._state = OrchestratorState(
            phase=phase,
            recording_active=recording_active,
            teleop_running=teleop_running,
            inference_ready=inference_ready,
            inference_executing=inference_executing,
            home_zone_active=home_zone_active,
            homing_active=homing_active,
            estopped=estopped,
            active_source=active_source,
        )
        return self._state

    def clear_estop(self) -> OrchestratorState:
        return self._recompute_state(estopped=False)

    def request_start_teleop(self) -> TransitionDecision:
        if self._state.estopped:
            return TransitionDecision(False, "estopped", self._state)
        if self._state.inference_executing:
            return TransitionDecision(False, "inference_executing", self._state)
        if self._state.homing_active:
            return TransitionDecision(False, "homing_active", self._state)
        if self._state.home_zone_active:
            return TransitionDecision(False, "home_zone_active", self._state)
        return TransitionDecision(True, "", self._state)

    def notify_teleop_started(self) -> OrchestratorState:
        return self._recompute_state(teleop_running=True, inference_executing=False)

    def notify_teleop_stopped(self) -> OrchestratorState:
        return self._recompute_state(teleop_running=False, inference_executing=False)

    def notify_recording(self, active: bool) -> OrchestratorState:
        return self._recompute_state(recording_active=bool(active))

    def request_enable_inference_execution(self) -> TransitionDecision:
        if self._state.estopped:
            return TransitionDecision(False, "estopped", self._state)
        if self._state.teleop_running:
            return TransitionDecision(False, "teleop_running", self._state)
        if self._state.homing_active:
            return TransitionDecision(False, "homing_active", self._state)
        if self._state.home_zone_active:
            return TransitionDecision(False, "home_zone_active", self._state)
        if not self._state.inference_ready:
            return TransitionDecision(False, "inference_not_ready", self._state)
        return TransitionDecision(True, "", self._state)

    def request_start_recording(self) -> TransitionDecision:
        if self._state.estopped:
            return TransitionDecision(False, "estopped", self._state)
        return TransitionDecision(True, "", self._state)

    def request_stop_recording(self) -> TransitionDecision:
        return TransitionDecision(True, "", self._state)

    def request_go_home(self) -> TransitionDecision:
        if self._state.estopped:
            return TransitionDecision(False, "estopped", self._state)
        if self._state.homing_active:
            return TransitionDecision(False, "homing_active", self._state)
        if self._state.home_zone_active:
            return TransitionDecision(False, "home_zone_active", self._state)
        return TransitionDecision(True, "", self._state)

    def request_go_home_zone(self) -> TransitionDecision:
        if self._state.estopped:
            return TransitionDecision(False, "estopped", self._state)
        if self._state.homing_active:
            return TransitionDecision(False, "homing_active", self._state)
        if self._state.home_zone_active:
            return TransitionDecision(False, "home_zone_active", self._state)
        return TransitionDecision(True, "", self._state)

    def notify_inference_ready(self, ready: bool) -> OrchestratorState:
        return self._recompute_state(
            inference_ready=bool(ready),
            inference_executing=self._state.inference_executing if ready else False,
        )

    def notify_inference_execution(self, active: bool) -> OrchestratorState:
        return self._recompute_state(inference_executing=bool(active))

    def notify_home_zone(self, active: bool) -> OrchestratorState:
        return self._recompute_state(home_zone_active=bool(active))

    def notify_homing(self, active: bool) -> OrchestratorState:
        return self._recompute_state(homing_active=bool(active))

    def notify_estop(self, active: bool) -> OrchestratorState:
        return self._recompute_state(
            estopped=bool(active),
            inference_executing=False if active else self._state.inference_executing,
            home_zone_active=False if active else self._state.home_zone_active,
            homing_active=False if active else self._state.homing_active,
        )

    def notify_fault(self, key: str) -> OrchestratorState:
        """Hardware component fault: degrade state based on which component failed."""
        return self._recompute_state(
            teleop_running=False if key == "teleop" else self._state.teleop_running,
            home_zone_active=(
                False
                if key in ("teleop", "robot_commander")
                else self._state.home_zone_active
            ),
            homing_active=(
                False
                if key in ("teleop", "robot_commander")
                else self._state.homing_active
            ),
            inference_executing=(
                False if key == "inference_worker" else self._state.inference_executing
            ),
        )
