"""Core services and domain models for teleop_control_py."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    'ActionCommand',
    'ActionMux',
    'CameraFrameSet',
    'CameraRuntimeContext',
    'ControlCoordinator',
    'ControlSource',
    'DaemonWatchdogTarget',
    'HardwareConflictError',
    'InferenceLaunchConfig',
    'InferenceService',
    'InferenceWorkerCallbacks',
    'MuxDispatchResult',
    'ObservationSnapshot',
    'OrchestratorState',
    'ProcessManager',
    'ProcessWatchdogTarget',
    'RecorderService',
    'ResourceManager',
    'RobotStateSnapshot',
    'SHMRegistry',
    'SyncHub',
    'SystemOrchestrator',
    'SystemPhase',
    'ThreadWatchdogTarget',
    'TopicWatchdogTarget',
    'TransitionDecision',
    'Watchdog',
    'WatchdogTarget',
]


def __getattr__(name: str) -> Any:
    if name in {'CameraRuntimeContext', 'HardwareConflictError', 'ResourceManager'}:
        module = import_module('.resource_manager', __name__)
        return getattr(module, name)
    if name in {'DaemonEntry', 'SHMRegistry'}:
        module = import_module('.shm_registry', __name__)
        return getattr(module, name)
    if name in {
        'DaemonWatchdogTarget',
        'ProcessWatchdogTarget',
        'ThreadWatchdogTarget',
        'TopicWatchdogTarget',
        'Watchdog',
        'WatchdogTarget',
    }:
        module = import_module('.watchdog', __name__)
        return getattr(module, name)
    if name == 'ProcessManager':
        module = import_module('teleop_control_py.gui.runtime.process_manager')
        return getattr(module, name)
    if name in {'ActionCommand', 'CameraFrameSet', 'ControlSource', 'ObservationSnapshot', 'RobotStateSnapshot'}:
        module = import_module('.models', __name__)
        return getattr(module, name)
    if name in {'ActionMux', 'MuxDispatchResult'}:
        module = import_module('.mux', __name__)
        return getattr(module, name)
    if name == 'ControlCoordinator':
        module = import_module('.control_coordinator', __name__)
        return getattr(module, name)
    if name == 'SyncHub':
        module = import_module('.sync_hub', __name__)
        return getattr(module, name)
    if name == 'RecorderService':
        module = import_module('.recorder', __name__)
        return getattr(module, name)
    if name in {'OrchestratorState', 'SystemOrchestrator', 'SystemPhase', 'TransitionDecision'}:
        module = import_module('.orchestrator', __name__)
        return getattr(module, name)
    if name in {'InferenceLaunchConfig', 'InferenceService', 'InferenceWorkerCallbacks'}:
        module = import_module('.inference_service', __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
