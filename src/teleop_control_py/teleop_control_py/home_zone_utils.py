#!/usr/bin/env python3
"""Home Zone sampling and bounded Cartesian closed-loop helpers."""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence

import numpy as np

from .transform_utils import (
    euler_to_quat_xyzw,
    quat_conjugate_xyzw,
    quat_multiply_xyzw,
    quat_to_rotvec_xyzw,
)


def sample_signed_ranges(
    min_abs_values: Sequence[float],
    max_abs_values: Sequence[float],
) -> np.ndarray:
    min_vals = np.maximum(0.0, np.asarray(list(min_abs_values), dtype=np.float64).reshape(-1))
    max_vals = np.maximum(min_vals, np.asarray(list(max_abs_values), dtype=np.float64).reshape(-1))
    if min_vals.shape != max_vals.shape or min_vals.size == 0:
        raise ValueError("min/max range shapes must match and be non-empty")
    magnitudes = np.random.uniform(min_vals, max_vals)
    signs = np.where(np.random.rand(min_vals.size) < 0.5, -1.0, 1.0)
    return (magnitudes * signs).astype(np.float64)


def sample_home_zone_pose_offsets(
    translation_min_m: Sequence[float],
    translation_max_m: Sequence[float],
    rotation_min_deg: Sequence[float],
    rotation_max_deg: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    translation = sample_signed_ranges(translation_min_m, translation_max_m)
    rotation_deg = sample_signed_ranges(rotation_min_deg, rotation_max_deg)
    return translation, np.deg2rad(rotation_deg)


def compose_pose_with_rpy_offset(
    base_pos_xyz: Sequence[float],
    base_quat_xyzw: Sequence[float],
    translation_offset_xyz: Sequence[float],
    rotation_offset_rpy_rad: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    base_pos = np.asarray(list(base_pos_xyz), dtype=np.float64).reshape(3)
    base_quat = np.asarray(list(base_quat_xyzw), dtype=np.float64).reshape(4)
    translation_offset = np.asarray(list(translation_offset_xyz), dtype=np.float64).reshape(3)
    rotation_offset = np.asarray(list(rotation_offset_rpy_rad), dtype=np.float64).reshape(3)
    delta_quat = euler_to_quat_xyzw(
        float(rotation_offset[0]),
        float(rotation_offset[1]),
        float(rotation_offset[2]),
    )
    target_quat = quat_multiply_xyzw(base_quat, delta_quat)
    norm = float(np.linalg.norm(target_quat))
    if norm > 1e-12:
        target_quat = target_quat / norm
    return base_pos + translation_offset, target_quat.astype(np.float64)


def compute_pose_error(
    current_pos_xyz: Sequence[float],
    current_quat_xyzw: Sequence[float],
    target_pos_xyz: Sequence[float],
    target_quat_xyzw: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    current_pos = np.asarray(list(current_pos_xyz), dtype=np.float64).reshape(3)
    current_quat = np.asarray(list(current_quat_xyzw), dtype=np.float64).reshape(4)
    target_pos = np.asarray(list(target_pos_xyz), dtype=np.float64).reshape(3)
    target_quat = np.asarray(list(target_quat_xyzw), dtype=np.float64).reshape(4)

    pos_error = target_pos - current_pos
    quat_error = quat_multiply_xyzw(target_quat, quat_conjugate_xyzw(current_quat))
    if float(quat_error[3]) < 0.0:
        quat_error = -quat_error
    rot_error = quat_to_rotvec_xyzw(quat_error).astype(np.float64)
    return pos_error, rot_error


def drive_pose_target(
    get_pose_fn,
    send_twist_fn,
    stop_twist_fn,
    target_pos_xyz: Sequence[float],
    target_quat_xyzw: Sequence[float],
    *,
    linear_gain: float = 1.6,
    angular_gain: float = 1.6,
    max_linear_vel: float = 0.06,
    max_angular_vel: float = 0.6,
    position_tolerance_m: float = 0.005,
    rotation_tolerance_deg: float = 2.0,
    timeout_sec: float = 4.0,
    rate_hz: float = 40.0,
    settle_sec: float = 0.15,
    should_abort_fn=None,
) -> tuple[bool, float, float]:
    interval = 1.0 / max(1.0, float(rate_hz))
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    rot_tol = math.radians(max(0.0, float(rotation_tolerance_deg)))
    last_pos = float("inf")
    last_rot = float("inf")

    while time.monotonic() < deadline:
        if should_abort_fn is not None and bool(should_abort_fn()):
            stop_twist_fn()
            return False, last_pos, math.degrees(last_rot)

        current_pose = get_pose_fn()
        if current_pose is None:
            time.sleep(interval)
            continue

        current_pos, current_quat = current_pose
        pos_error, rot_error = compute_pose_error(current_pos, current_quat, target_pos_xyz, target_quat_xyzw)
        last_pos = float(np.linalg.norm(pos_error))
        last_rot = float(np.linalg.norm(rot_error))
        if last_pos <= position_tolerance_m and last_rot <= rot_tol:
            settle_deadline = time.monotonic() + max(0.0, float(settle_sec))
            while time.monotonic() < settle_deadline:
                stop_twist_fn()
                time.sleep(min(0.02, interval))
            stop_twist_fn()
            return True, last_pos, math.degrees(last_rot)

        linear = np.clip(linear_gain * pos_error, -max_linear_vel, max_linear_vel)
        angular = np.clip(angular_gain * rot_error, -max_angular_vel, max_angular_vel)
        send_twist_fn(linear, angular)
        time.sleep(interval)

    stop_twist_fn()
    return False, last_pos, math.degrees(last_rot)


def pose_to_string(position: Sequence[float], quat_xyzw: Sequence[float]) -> str:
    pos = np.array2string(np.asarray(list(position), dtype=np.float64), precision=4)
    quat = np.array2string(np.asarray(list(quat_xyzw), dtype=np.float64), precision=4)
    return f"pos={pos} quat={quat}"
