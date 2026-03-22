"""Arm backend adapters built on top of the current ServoPoseFollower."""

from __future__ import annotations

from ...core.models import ActionCommand
from ...hardware.control.servo_pose_follower import ServoPoseFollower


class ServoArmBackend:
    def __init__(self, follower: ServoPoseFollower) -> None:
        self._follower = follower

    @classmethod
    def from_node(cls, node):
        return cls(ServoPoseFollower(node))

    def send_delta_twist(self, command: ActionCommand) -> None:
        self._follower.send_twist(command.to_twist())

    def send_zero_twist(self) -> None:
        self._follower.send_zero_twist()

    def stop(self) -> None:
        self._follower.stop()
