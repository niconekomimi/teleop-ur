"""Gripper backend adapters built on top of the current controller classes."""

from __future__ import annotations


class ControllerGripperBackend:
    def __init__(self, controller) -> None:
        self._controller = controller

    def set_gripper(self, value: float) -> None:
        self._controller.set_gripper(value)

    def stop(self) -> None:
        stop_fn = getattr(self._controller, 'stop', None)
        if callable(stop_fn):
            stop_fn()

    def cancel_motion(self) -> None:
        cancel_fn = getattr(self._controller, 'cancel_motion', None)
        if callable(cancel_fn):
            cancel_fn()
