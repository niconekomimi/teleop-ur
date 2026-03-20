#!/usr/bin/env python3
"""Minimal forward kinematics for common UR robot variants."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


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
