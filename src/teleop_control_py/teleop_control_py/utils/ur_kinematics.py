#!/usr/bin/env python3
"""Minimal forward kinematics for common UR robot variants."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .transform_utils import (
    quat_conjugate_xyzw,
    quat_multiply_xyzw,
    quat_to_rotvec_xyzw,
)


_ALPHA = np.array([math.pi / 2.0, 0.0, 0.0, math.pi / 2.0, -math.pi / 2.0, 0.0], dtype=np.float64)

_UR_DH = {
    "ur3": {
        "a": np.array([0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0], dtype=np.float64),
        "d": np.array([0.15185, 0.0, 0.0, 0.11235, 0.08535, 0.0819], dtype=np.float64),
    },
    "ur5": {
        "a": np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0], dtype=np.float64),
        "d": np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823], dtype=np.float64),
    },
    "ur10": {
        "a": np.array([0.0, -0.612, -0.5723, 0.0, 0.0, 0.0], dtype=np.float64),
        "d": np.array([0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922], dtype=np.float64),
    },
    "ur3e": {
        "a": np.array([0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0], dtype=np.float64),
        "d": np.array([0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921], dtype=np.float64),
    },
    "ur5e": {
        "a": np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0], dtype=np.float64),
        "d": np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996], dtype=np.float64),
    },
    "ur10e": {
        "a": np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0], dtype=np.float64),
        "d": np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655], dtype=np.float64),
    },
    "ur16e": {
        "a": np.array([0.0, -0.4784, -0.36, 0.0, 0.0, 0.0], dtype=np.float64),
        "d": np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655], dtype=np.float64),
    },
}


def _quat_from_rotmat(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def normalize_ur_type(ur_type: str) -> str:
    normalized = str(ur_type).strip().lower()
    if normalized in _UR_DH:
        return normalized
    return "ur5" if normalized == "" else normalized


def forward_kinematics(ur_type: str, joint_positions: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(list(joint_positions), dtype=np.float64).reshape(-1)
    if joints.size < 6:
        raise ValueError("joint_positions must contain 6 elements")

    key = normalize_ur_type(ur_type)
    if key not in _UR_DH:
        raise KeyError(f"Unsupported ur_type for FK: {ur_type}")

    params = _UR_DH[key]
    transform = np.eye(4, dtype=np.float64)
    for idx in range(6):
        transform = transform @ _dh_transform(
            float(params["a"][idx]),
            float(_ALPHA[idx]),
            float(params["d"][idx]),
            float(joints[idx]),
        )

    position = transform[:3, 3].astype(np.float64)
    quat_xyzw = _quat_from_rotmat(transform[:3, :3])
    return position, quat_xyzw


def try_forward_kinematics(ur_type: str, joint_positions: Sequence[float]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    try:
        return forward_kinematics(ur_type, joint_positions)
    except Exception:
        return None


def _wrap_joints_near_reference(joints: np.ndarray, reference: np.ndarray) -> np.ndarray:
    wrapped = np.asarray(joints, dtype=np.float64).copy()
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    for idx in range(min(wrapped.size, ref.size)):
        delta = wrapped[idx] - ref[idx]
        wrapped[idx] = ref[idx] + ((delta + math.pi) % (2.0 * math.pi) - math.pi)
    return wrapped.astype(np.float64)


def _compute_pose_error(
    current_pos_xyz: np.ndarray,
    current_quat_xyzw: np.ndarray,
    target_pos_xyz: np.ndarray,
    target_quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos_error = np.asarray(target_pos_xyz, dtype=np.float64).reshape(3) - np.asarray(
        current_pos_xyz,
        dtype=np.float64,
    ).reshape(3)
    quat_error = quat_multiply_xyzw(
        np.asarray(target_quat_xyzw, dtype=np.float64).reshape(4),
        quat_conjugate_xyzw(np.asarray(current_quat_xyzw, dtype=np.float64).reshape(4)),
    )
    if float(quat_error[3]) < 0.0:
        quat_error = -quat_error
    rot_error = quat_to_rotvec_xyzw(quat_error).astype(np.float64)
    return pos_error.astype(np.float64), rot_error.astype(np.float64)


def _numerical_geometric_jacobian(
    ur_type: str,
    joint_positions: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints = np.asarray(joint_positions, dtype=np.float64).reshape(6)
    base_pos, base_quat = forward_kinematics(ur_type, joints)
    jacobian = np.zeros((6, 6), dtype=np.float64)

    for idx in range(6):
        displaced = joints.copy()
        displaced[idx] += float(epsilon)
        pos_eps, quat_eps = forward_kinematics(ur_type, displaced)
        jacobian[:3, idx] = (pos_eps - base_pos) / float(epsilon)

        quat_delta = quat_multiply_xyzw(quat_eps, quat_conjugate_xyzw(base_quat))
        if float(quat_delta[3]) < 0.0:
            quat_delta = -quat_delta
        jacobian[3:, idx] = quat_to_rotvec_xyzw(quat_delta).astype(np.float64) / float(epsilon)

    return jacobian, base_pos.astype(np.float64), base_quat.astype(np.float64)


def solve_inverse_kinematics(
    ur_type: str,
    target_pos_xyz: Sequence[float],
    target_quat_xyzw: Sequence[float],
    seed_joint_positions: Sequence[float],
    *,
    max_iterations: int = 120,
    damping: float = 1e-2,
    max_step_norm: float = 0.25,
    position_tolerance_m: float = 1e-4,
    rotation_tolerance_deg: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    joints = np.asarray(list(seed_joint_positions), dtype=np.float64).reshape(-1)
    if joints.size < 6:
        raise ValueError("seed_joint_positions must contain 6 elements")

    reference = joints[:6].astype(np.float64).copy()
    target_pos = np.asarray(list(target_pos_xyz), dtype=np.float64).reshape(3)
    target_quat = np.asarray(list(target_quat_xyzw), dtype=np.float64).reshape(4)
    rot_tol_rad = math.radians(max(0.0, float(rotation_tolerance_deg)))

    best_joints = reference.copy()
    best_pos_err = float("inf")
    best_rot_err = float("inf")

    for _ in range(max(1, int(max_iterations))):
        jacobian, current_pos, current_quat = _numerical_geometric_jacobian(ur_type, joints[:6])
        pos_error, rot_error = _compute_pose_error(current_pos, current_quat, target_pos, target_quat)
        pos_norm = float(np.linalg.norm(pos_error))
        rot_norm = float(np.linalg.norm(rot_error))

        if pos_norm + rot_norm < best_pos_err + best_rot_err:
            best_joints = joints[:6].copy()
            best_pos_err = pos_norm
            best_rot_err = rot_norm

        if pos_norm <= position_tolerance_m and rot_norm <= rot_tol_rad:
            return _wrap_joints_near_reference(joints[:6], reference), pos_norm, math.degrees(rot_norm)

        error_vec = np.concatenate([pos_error, rot_error], dtype=np.float64)
        lhs = jacobian @ jacobian.T + (float(damping) ** 2) * np.eye(6, dtype=np.float64)
        step = jacobian.T @ np.linalg.solve(lhs, error_vec)

        step_norm = float(np.linalg.norm(step))
        if step_norm > max(1e-9, float(max_step_norm)):
            step *= float(max_step_norm) / step_norm

        joints[:6] = _wrap_joints_near_reference(joints[:6] + step, reference)

    raise RuntimeError(
        "IK did not converge "
        f"(best_pos={best_pos_err:.4f}m best_rot={math.degrees(best_rot_err):.2f}deg)"
    )


def try_inverse_kinematics(
    ur_type: str,
    target_pos_xyz: Sequence[float],
    target_quat_xyzw: Sequence[float],
    seed_joint_positions: Sequence[float],
    **kwargs,
) -> Optional[tuple[np.ndarray, float, float]]:
    try:
        return solve_inverse_kinematics(
            ur_type,
            target_pos_xyz,
            target_quat_xyzw,
            seed_joint_positions,
            **kwargs,
        )
    except Exception:
        return None
