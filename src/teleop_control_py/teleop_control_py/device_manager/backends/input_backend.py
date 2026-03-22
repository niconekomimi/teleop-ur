"""Adapters that expose the existing input handlers as InputBackends."""

from __future__ import annotations

from ...core.models import ActionCommand, ControlSource


class InputHandlerBackend:
    def __init__(self, handler, *, source: ControlSource = ControlSource.TELEOP) -> None:
        self._handler = handler
        self._source = ControlSource(source)

    def get_action_command(self) -> ActionCommand:
        twist, gripper = self._handler.get_command()
        return ActionCommand.from_twist(twist, gripper=gripper, source=self._source)

    def stop(self) -> None:
        stop_fn = getattr(self._handler, 'stop', None)
        if callable(stop_fn):
            stop_fn()
