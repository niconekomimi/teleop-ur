#!/usr/bin/env python3
"""数据采集节点使用的数学和图像工具函数。"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    """将数值限制在给定区间内。"""
    return max(low, min(high, value))


def _quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    """归一化四元数，输入输出均为 XYZW 顺序。"""
    q = q.astype(np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return (q / norm).astype(np.float64)


def _quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    """将 XYZW 四元数转换为 3x3 旋转矩阵。"""
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


def _normalize_vec3(v: np.ndarray) -> np.ndarray:
    """归一化 3 维向量；零向量会安全回退。"""
    v = v.astype(np.float64)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return (v / norm).astype(np.float64)


def _quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """计算将 ``v_from`` 旋转到 ``v_to`` 的最小旋转四元数。"""
    a = _normalize_vec3(v_from)
    b = _normalize_vec3(v_to)
    if float(np.linalg.norm(a)) <= 1e-12 or float(np.linalg.norm(b)) <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    dot = float(np.dot(a, b))
    dot = _clamp(dot, -1.0, 1.0)
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


def _quat_to_rotvec_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    """将 XYZW 四元数转换为旋转向量。"""
    qn = _quat_normalize_xyzw(q_xyzw)
    x, y, z, w = (float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3]))
    w = _clamp(w, -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x / s, y / s, z / s], dtype=np.float64)
    return (axis * angle).astype(np.float32)


def center_crop_square_and_resize_rgb(bgr: np.ndarray, output_size: int) -> np.ndarray:
    """中心裁剪成正方形后缩放，输出 RGB uint8 连续数组。"""
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