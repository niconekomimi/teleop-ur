#!/usr/bin/env python3
"""遥操作与数据采集共用的纯数学工具库。"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    """将标量限制在给定区间内。"""
    return max(low, min(high, value))


clamp = _clamp


def apply_deadzone(value: float, deadzone: float) -> float:
    """对单轴输入施加死区，并保持剩余区间线性缩放。"""
    value = float(value)
    deadzone = abs(float(deadzone))
    if deadzone <= 0.0:
        return value
    if abs(value) <= deadzone:
        return 0.0
    scaled = (abs(value) - deadzone) / max(1e-12, 1.0 - deadzone)
    return math.copysign(scaled, value)


def map_axis_linear(value: float, deadzone: float = 0.0, scale: float = 1.0) -> float:
    """将摇杆值经死区处理后做线性映射。"""
    return float(scale) * apply_deadzone(value, deadzone)


def map_axis_nonlinear(
    value: float,
    deadzone: float = 0.0,
    exponent: float = 2.0,
    scale: float = 1.0,
) -> float:
    """将摇杆值做非线性映射，便于小范围更细腻、大范围更激进。"""
    shaped = apply_deadzone(value, deadzone)
    exponent = max(1.0, float(exponent))
    return float(scale) * math.copysign(abs(shaped) ** exponent, shaped)


def _quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    """归一化 XYZW 四元数。"""
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return (q / norm).astype(np.float64)


quat_normalize_xyzw = _quat_normalize_xyzw


def quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    """四元数共轭。"""
    q = np.asarray(q, dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法，输入输出均为 XYZW。"""
    x1, y1, z1, w1 = (float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3]))
    x2, y2, z2, w2 = (float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3]))
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """欧拉角转 XYZW 四元数。"""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    quat = np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )
    return _quat_normalize_xyzw(quat)


def quat_to_euler_xyzw(q: np.ndarray) -> tuple[float, float, float]:
    """XYZW 四元数转欧拉角。"""
    qn = _quat_normalize_xyzw(q)
    x, y, z, w = (float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3]))

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    """XYZW 四元数转旋转矩阵。"""
    qn = _quat_normalize_xyzw(q)
    x, y, z, w = (float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3]))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


quat_to_rotmat_xyzw = _quat_to_rotmat_xyzw


def _normalize_vec3(v: np.ndarray) -> np.ndarray:
    """归一化三维向量。"""
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return (v / norm).astype(np.float64)


normalize_vec3 = _normalize_vec3


def _quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """计算从 ``v_from`` 旋转到 ``v_to`` 的最小旋转四元数。"""
    a = _normalize_vec3(v_from)
    b = _normalize_vec3(v_to)
    if float(np.linalg.norm(a)) <= 1e-12 or float(np.linalg.norm(b)) <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    dot = _clamp(float(np.dot(a, b)), -1.0, 1.0)
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if dot < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = _normalize_vec3(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)

    cross = np.cross(a, b)
    quat = np.array([cross[0], cross[1], cross[2], 1.0 + dot], dtype=np.float64)
    return _quat_normalize_xyzw(quat)


quat_from_two_vectors = _quat_from_two_vectors


def _quat_to_rotvec_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    """XYZW 四元数转旋转向量。"""
    qn = _quat_normalize_xyzw(q_xyzw)
    x, y, z, w = (float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3]))
    w = _clamp(w, -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x / s, y / s, z / s], dtype=np.float64)
    return (axis * angle).astype(np.float32)


quat_to_rotvec_xyzw = _quat_to_rotvec_xyzw


def rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """旋转向量转 XYZW 四元数。"""
    rv = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rv / angle
    half = angle * 0.5
    s = math.sin(half)
    return _quat_normalize_xyzw(
        np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float64)
    )


def apply_velocity_limits(
    target: Sequence[float],
    previous: Optional[Sequence[float]] = None,
    max_velocity: Optional[Sequence[float]] = None,
    max_acceleration: Optional[Sequence[float]] = None,
    dt: float = 0.02,
) -> np.ndarray:
    """统一对速度向量做速度和加速度双重限幅。"""
    target_arr = np.asarray(target, dtype=np.float64).copy()
    if max_velocity is not None:
        max_vel_arr = np.asarray(max_velocity, dtype=np.float64)
        target_arr = np.clip(target_arr, -max_vel_arr, max_vel_arr)

    if previous is None or max_acceleration is None:
        return target_arr.astype(np.float64)

    # Clamp the integration window so host-side stalls do not create a huge one-shot velocity jump.
    dt = min(max(float(dt), 1e-6), 0.1)
    prev_arr = np.asarray(previous, dtype=np.float64)
    max_acc_arr = np.asarray(max_acceleration, dtype=np.float64)
    delta = target_arr - prev_arr
    max_delta = max_acc_arr * dt
    delta = np.clip(delta, -max_delta, max_delta)
    return (prev_arr + delta).astype(np.float64)


def compose_eef_action(
    linear_xyz: Sequence[float],
    angular_xyz: Sequence[float],
    gripper: float,
) -> np.ndarray:
    """组装当前数据集使用的 7 维动作向量 [vx, vy, vz, wx, wy, wz, gripper]。"""
    linear = np.asarray(linear_xyz, dtype=np.float32)
    angular = np.asarray(angular_xyz, dtype=np.float32)
    return np.array(
        [
            float(linear[0]),
            float(linear[1]),
            float(linear[2]),
            float(angular[0]),
            float(angular[1]),
            float(angular[2]),
            float(_clamp(gripper, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def center_crop_square_and_resize_rgb(bgr: np.ndarray, output_size: int) -> np.ndarray:
    """中心裁剪为正方形后缩放到指定尺寸，输出 RGB。"""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected BGR HxWx3 image")

    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError("output_size must be > 0")

    height, width = int(bgr.shape[0]), int(bgr.shape[1])
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    crop = bgr[y0 : y0 + side, x0 : x0 + side]
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)
