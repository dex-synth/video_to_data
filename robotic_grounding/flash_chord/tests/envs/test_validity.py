# SPDX-License-Identifier: Apache-2.0
"""Numerical bounds detect failures without modifying physics state."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("failure,bit", [("nonfinite", 1), ("position", 2), ("linear", 4),
                                        ("angular", 8), ("joint_position", 16), ("joint_speed", 32)])
def test_state_bounds_are_world_local_and_do_not_clamp(failure, bit):
    import newton
    import warp as wp
    from flash_chord.envs.validity import StateValidity, StateValidityConfig

    with wp.ScopedDevice("cpu"):
        poses = np.array([[0, 0, 0, 0, 0, 0, 1]] * 2, dtype=np.float32)
        model = SimpleNamespace(
            body_count=2, joint_coord_count=2, joint_dof_count=2, joint_count=2,
            body_q=wp.array(poses, dtype=wp.transform),
            joint_type=wp.array([newton.JointType.REVOLUTE] * 2, dtype=wp.int32),
            joint_q_start=wp.array([0, 1], dtype=wp.int32),
            joint_qd_start=wp.array([0, 1], dtype=wp.int32),
            joint_limit_lower=wp.array([-1., -1.]), joint_limit_upper=wp.array([1., 1.]),
        )
        velocities = np.zeros((2, 6), dtype=np.float32)
        q, qd = np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
        if failure == "nonfinite":
            q[0] = np.nan
        if failure == "position":
            poses[0, 0] = 11
        if failure == "linear":
            velocities[0, 0] = 21
        if failure == "angular":
            velocities[0, 3] = 201
        if failure == "joint_position":
            q[0] = 2.1
        if failure == "joint_speed":
            qd[0] = 201
        state = SimpleNamespace(body_q=wp.array(poses, dtype=wp.transform),
                                body_qd=wp.array(velocities, dtype=wp.spatial_vector),
                                joint_q=wp.array(q), joint_qd=wp.array(qd))
        reason = wp.zeros(2, dtype=wp.int32)
        checker = StateValidity(model, 2, StateValidityConfig(), "cpu")
        checker.check(state, reason)
        np.testing.assert_array_equal(reason.numpy(), [bit, 0])
        for name, expected in (("body_q", poses), ("body_qd", velocities), ("joint_q", q), ("joint_qd", qd)):
            np.testing.assert_array_equal(getattr(state, name).numpy(), expected)
