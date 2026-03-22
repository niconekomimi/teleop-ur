"""Unified control arbitration coordinator built on top of the core state machine."""

from __future__ import annotations

from typing import Mapping, Optional

from ..device_manager.backends.interfaces import ArmBackend, GripperBackend
from .models import ActionCommand, ControlSource
from .mux import ActionMux, MuxDispatchResult
from .orchestrator import OrchestratorState, SystemOrchestrator, TransitionDecision


class ControlCoordinator:
    """Keeps the orchestrator state and command mux aligned behind one interface."""

    def __init__(
        self,
        arm_backend: Optional[ArmBackend] = None,
        gripper_backend: Optional[GripperBackend] = None,
        *,
        active_source: ControlSource = ControlSource.NONE,
        priority_map: Optional[Mapping[ControlSource, int]] = None,
        logger: Optional[object] = None,
    ) -> None:
        self._orchestrator = SystemOrchestrator()
        self._mux = ActionMux(
            arm_backend,
            gripper_backend,
            active_source=active_source,
            priority_map=priority_map,
            logger=logger,
        )
        self._sync_mux(self._orchestrator.snapshot())

    def snapshot(self) -> OrchestratorState:
        return self._orchestrator.snapshot()

    @property
    def active_source(self) -> ControlSource:
        return self._mux.active_source

    def _sync_mux(self, state: OrchestratorState) -> OrchestratorState:
        self._mux.set_active_source(state.active_source)
        self._mux.set_hold(state.estopped, 'estopped' if state.estopped else '')
        return state

    def request_start_teleop(self) -> TransitionDecision:
        return self._orchestrator.request_start_teleop()

    def request_enable_inference_execution(self) -> TransitionDecision:
        return self._orchestrator.request_enable_inference_execution()

    def request_start_recording(self) -> TransitionDecision:
        return self._orchestrator.request_start_recording()

    def request_stop_recording(self) -> TransitionDecision:
        return self._orchestrator.request_stop_recording()

    def request_go_home(self) -> TransitionDecision:
        return self._orchestrator.request_go_home()

    def request_go_home_zone(self) -> TransitionDecision:
        return self._orchestrator.request_go_home_zone()

    def clear_estop(self) -> OrchestratorState:
        return self._sync_mux(self._orchestrator.clear_estop())

    def notify_teleop_started(self) -> OrchestratorState:
        return self._sync_mux(self._orchestrator.notify_teleop_started())

    def notify_teleop_stopped(self) -> OrchestratorState:
        return self._sync_mux(self._orchestrator.notify_teleop_stopped())

    def notify_recording(self, active: bool) -> OrchestratorState:
        return self._sync_mux(self._orchestrator.notify_recording(active))

    def notify_inference_ready(self, ready: bool) -> OrchestratorState:
        return self._sync_mux(self._orchestrator.notify_inference_ready(ready))

    def notify_inference_execution(self, active: bool) -> OrchestratorState:
        return self._sync_mux(self._orchestrator.notify_inference_execution(active))

    def notify_home_zone(self, active: bool) -> OrchestratorState:
        return self._sync_mux(self._orchestrator.notify_home_zone(active))

    def notify_homing(self, active: bool) -> OrchestratorState:
        return self._sync_mux(self._orchestrator.notify_homing(active))

    def notify_estop(self, active: bool) -> OrchestratorState:
        return self._sync_mux(self._orchestrator.notify_estop(active))

    def notify_fault(self, key: str) -> OrchestratorState:
        return self._sync_mux(self._orchestrator.notify_fault(key))

    def dispatch(self, command: ActionCommand) -> MuxDispatchResult:
        state = self._sync_mux(self._orchestrator.snapshot())
        result = self._mux.publish(command)
        self._sync_mux(state)
        return result

    def publish_zero(self, source: ControlSource = ControlSource.SAFETY) -> MuxDispatchResult:
        state = self._sync_mux(self._orchestrator.snapshot())
        result = self._mux.publish_zero(source=source)
        self._sync_mux(state)
        return result
