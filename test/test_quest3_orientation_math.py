#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src/teleop_control_py"))

from teleop_control_py.utils.transform_utils import (  # noqa: E402
    apply_deadzone,
    axis_mapping_sign_to_rotation_matrix,
    clip_rotvec_magnitude,
    finite_difference_body_angular_velocity,
    finite_difference_linear_velocity,
    quat_multiply_xyzw,
    quat_to_shortest_rotvec_xyzw,
    relative_body_quat_delta_xyzw,
    rebase_pose_with_origin_xyzw,
    rotvec_to_quat_xyzw,
)


def test_shortest_rotvec_is_invariant_to_quaternion_sign() -> None:
    rotvec = np.deg2rad(np.array([0.0, 0.0, 20.0], dtype=np.float64))
    quat = rotvec_to_quat_xyzw(rotvec)

    positive = quat_to_shortest_rotvec_xyzw(quat)
    negative = quat_to_shortest_rotvec_xyzw(-quat)

    assert np.allclose(positive, negative, atol=1e-8)
    assert np.allclose(positive, rotvec, atol=1e-6)


def test_shortest_rotvec_limits_rotation_to_shortest_path() -> None:
    quat = rotvec_to_quat_xyzw(np.deg2rad(np.array([0.0, 0.0, 181.0], dtype=np.float64)))
    rotvec = quat_to_shortest_rotvec_xyzw(quat)

    assert np.isclose(float(np.linalg.norm(rotvec)), np.deg2rad(179.0), atol=1e-6)
    assert rotvec[2] < 0.0


def test_relative_body_quat_delta_matches_local_wrist_rotation() -> None:
    start = rotvec_to_quat_xyzw(np.deg2rad(np.array([0.0, 0.0, 90.0], dtype=np.float64)))
    local_roll = rotvec_to_quat_xyzw(np.deg2rad(np.array([30.0, 0.0, 0.0], dtype=np.float64)))
    current = quat_multiply_xyzw(start, local_roll)

    q_delta = relative_body_quat_delta_xyzw(start, current)
    rotvec = quat_to_shortest_rotvec_xyzw(q_delta)

    assert np.allclose(rotvec, np.deg2rad(np.array([30.0, 0.0, 0.0], dtype=np.float64)), atol=1e-6)


def test_apply_deadzone_preserves_full_scale_output() -> None:
    assert np.isclose(apply_deadzone(1.0, 0.2), 1.0)
    assert np.isclose(apply_deadzone(-1.0, 0.2), -1.0)
    assert np.isclose(apply_deadzone(0.2, 0.2), 0.0)


def test_rebase_pose_can_decouple_position_from_orientation() -> None:
    origin_pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    origin_quat = rotvec_to_quat_xyzw(np.deg2rad(np.array([0.0, 0.0, 90.0], dtype=np.float64)))
    raw_pos = origin_pos + np.array([0.1, 0.0, 0.0], dtype=np.float64)
    raw_quat = quat_multiply_xyzw(origin_quat, rotvec_to_quat_xyzw(np.deg2rad(np.array([0.0, 20.0, 0.0]))))

    decoupled_pos, decoupled_quat = rebase_pose_with_origin_xyzw(
        raw_pos,
        raw_quat,
        origin_pos,
        origin_quat,
        rotate_position=False,
    )
    rotated_pos, rotated_quat = rebase_pose_with_origin_xyzw(
        raw_pos,
        raw_quat,
        origin_pos,
        origin_quat,
        rotate_position=True,
    )

    assert np.allclose(decoupled_pos, np.array([0.1, 0.0, 0.0], dtype=np.float64), atol=1e-8)
    assert not np.allclose(rotated_pos, decoupled_pos, atol=1e-8)
    assert np.allclose(quat_to_shortest_rotvec_xyzw(decoupled_quat), quat_to_shortest_rotvec_xyzw(rotated_quat))


def test_finite_difference_linear_velocity_is_zero_when_hand_stops() -> None:
    previous = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    current = previous.copy()
    velocity = finite_difference_linear_velocity(previous, current, 0.01)

    assert np.allclose(velocity, np.zeros(3, dtype=np.float64), atol=1e-8)


def test_finite_difference_body_angular_velocity_uses_neighbor_samples() -> None:
    previous = rotvec_to_quat_xyzw(np.zeros(3, dtype=np.float64))
    current = rotvec_to_quat_xyzw(np.deg2rad(np.array([0.0, 0.0, 9.0], dtype=np.float64)))
    angular_velocity = finite_difference_body_angular_velocity(previous, current, 0.1)

    assert np.allclose(
        angular_velocity,
        np.deg2rad(np.array([0.0, 0.0, 90.0], dtype=np.float64)),
        atol=1e-6,
    )


def test_axis_mapping_sign_to_rotation_matrix_matches_current_quest_linear_mapping() -> None:
    rotation = axis_mapping_sign_to_rotation_matrix([0, 2, 1], [-1.0, 1.0, 1.0])

    basis_x = rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    basis_y = rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    basis_z = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

    assert np.allclose(basis_x, np.array([-1.0, 0.0, 0.0], dtype=np.float64))
    assert np.allclose(basis_y, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    assert np.allclose(basis_z, np.array([0.0, 1.0, 0.0], dtype=np.float64))


def test_clip_rotvec_magnitude_limits_large_single_clutch_rotation() -> None:
    raw = np.deg2rad(np.array([0.0, 0.0, 120.0], dtype=np.float64))
    clipped = clip_rotvec_magnitude(raw, np.deg2rad(90.0))

    assert np.isclose(float(np.linalg.norm(clipped)), np.deg2rad(90.0), atol=1e-6)
    assert clipped[2] > 0.0
