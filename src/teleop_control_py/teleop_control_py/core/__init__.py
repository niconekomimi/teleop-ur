"""Core services for teleop_control_py GUI."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CameraRuntimeContext",
    "HardwareConflictError",
    "HardwareManager",
    "ProcessManager",
]


def __getattr__(name: str) -> Any:
    if name in {"CameraRuntimeContext", "HardwareConflictError", "HardwareManager"}:
        module = import_module(".hardware_manager", __name__)
        return getattr(module, name)
    if name == "ProcessManager":
        module = import_module(".process_manager", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
