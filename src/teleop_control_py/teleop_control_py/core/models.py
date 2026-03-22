"""Core domain models for command routing and synchronized observations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from geometry_msgs.msg import Twist


def _as_readonly_vector(values, size: int, *, dtype=np.float64) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.size != size:
        raise ValueError(f"Expected vector of size {size}, got {array.size}")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


class ControlSource(str, Enum):
    NONE = "none"
    TELEOP = "teleop"
    INFERENCE = "inference"
    COMMANDER = "commander"
    SAFETY = "safety"


@dataclass(frozen=True)
class ActionCommand:
    linear_xyz: np.ndarray
    angular_xyz: np.ndarray
    gripper: float = 0.0
    source: ControlSource = ControlSource.NONE
    stamp_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'linear_xyz', _as_readonly_vector(self.linear_xyz, 3))
        object.__setattr__(self, 'angular_xyz', _as_readonly_vector(self.angular_xyz, 3))
        object.__setattr__(self, 'gripper', float(max(0.0, min(1.0, self.gripper))))
        object.__setattr__(self, 'source', ControlSource(self.source))

    @classmethod
    def zero(cls, source: ControlSource = ControlSource.NONE) -> 'ActionCommand':
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), gripper=0.0, source=source)

    @classmethod
    def from_twist(
        cls,
        twist: Twist,
        gripper: float = 0.0,
        source: ControlSource = ControlSource.NONE,
    ) -> 'ActionCommand':
        return cls(
            (twist.linear.x, twist.linear.y, twist.linear.z),
            (twist.angular.x, twist.angular.y, twist.angular.z),
            gripper=gripper,
            source=source,
        )

    @classmethod
    def from_array7(cls, values, source: ControlSource = ControlSource.NONE) -> 'ActionCommand':
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size < 7:
            raise ValueError(f'Expected action array with at least 7 values, got {array.size}')
        return cls(array[:3], array[3:6], gripper=float(array[6]), source=source)

    def to_twist(self) -> Twist:
        twist = Twist()
        twist.linear.x = float(self.linear_xyz[0])
        twist.linear.y = float(self.linear_xyz[1])
        twist.linear.z = float(self.linear_xyz[2])
        twist.angular.x = float(self.angular_xyz[0])
        twist.angular.y = float(self.angular_xyz[1])
        twist.angular.z = float(self.angular_xyz[2])
        return twist

    def twist_vector(self) -> np.ndarray:
        return np.concatenate((self.linear_xyz, self.angular_xyz)).astype(np.float64, copy=False)

    def action_vector(self) -> np.ndarray:
        return np.concatenate((self.linear_xyz, self.angular_xyz, np.array([self.gripper], dtype=np.float64)))

    def with_twist_vector(self, values) -> 'ActionCommand':
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != 6:
            raise ValueError(f'Expected 6D twist vector, got {vector.size}')
        return ActionCommand(vector[:3], vector[3:6], gripper=self.gripper, source=self.source)

    def with_gripper(self, value: float) -> 'ActionCommand':
        return ActionCommand(self.linear_xyz, self.angular_xyz, gripper=value, source=self.source)


@dataclass(frozen=True)
class CameraFrameSet:
    global_bgr: np.ndarray
    wrist_bgr: np.ndarray
    global_time_ns: int
    wrist_time_ns: int
    ref_time_ns: int
    skew_sec: float
    ref_context: Optional[object] = None

    def __post_init__(self) -> None:
        global_bgr = np.ascontiguousarray(self.global_bgr)
        wrist_bgr = np.ascontiguousarray(self.wrist_bgr)
        global_bgr.setflags(write=False)
        wrist_bgr.setflags(write=False)
        object.__setattr__(self, 'global_bgr', global_bgr)
        object.__setattr__(self, 'wrist_bgr', wrist_bgr)
        object.__setattr__(self, 'global_time_ns', int(self.global_time_ns))
        object.__setattr__(self, 'wrist_time_ns', int(self.wrist_time_ns))
        object.__setattr__(self, 'ref_time_ns', int(self.ref_time_ns))
        object.__setattr__(self, 'skew_sec', float(self.skew_sec))


@dataclass(frozen=True)
class RobotStateSnapshot:
    joint_pos: np.ndarray
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper: float
    twist_linear: Optional[np.ndarray] = None
    twist_angular: Optional[np.ndarray] = None
    ref_context: Optional[object] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'joint_pos', _as_readonly_vector(self.joint_pos, len(np.asarray(self.joint_pos).reshape(-1)), dtype=np.float32))
        object.__setattr__(self, 'eef_pos', _as_readonly_vector(self.eef_pos, 3, dtype=np.float32))
        object.__setattr__(self, 'eef_quat', _as_readonly_vector(self.eef_quat, 4, dtype=np.float32))
        object.__setattr__(self, 'gripper', float(max(0.0, min(1.0, self.gripper))))
        if self.twist_linear is not None:
            object.__setattr__(self, 'twist_linear', _as_readonly_vector(self.twist_linear, 3, dtype=np.float32))
        if self.twist_angular is not None:
            object.__setattr__(self, 'twist_angular', _as_readonly_vector(self.twist_angular, 3, dtype=np.float32))


@dataclass(frozen=True)
class ObservationSnapshot:
    camera_frames: CameraFrameSet
    robot_state: RobotStateSnapshot
    action_vector: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.action_vector is None:
            return
        object.__setattr__(self, 'action_vector', _as_readonly_vector(self.action_vector, 7, dtype=np.float32))
