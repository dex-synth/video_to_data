# SPDX-License-Identifier: Apache-2.0
"""Numerical bounds detect failures without modifying physics state."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("failure,bit", [("nonfinite", 1), ("position", 2), ("linear", 4),
                                        ("angular", 8), ("joint_position", 16), ("joint_speed", 32)])
@pytest.mark.parametrize("mode", [None, "ignore", "discard", "terminate"])
def test_state_bounds_are_world_local_and_do_not_clamp(failure, bit, mode):
    import newton
    import warp as wp
    from flash_chord.envs.validity import StateValidity, StateValidityConfig
    from flash_chord.envs.rl import apply_joint_limit_termination, terminate_invalid_states

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
        config = StateValidityConfig() if mode is None else StateValidityConfig(joint_position_limit_mode=mode)
        checker = StateValidity(model, 2, config, "cpu")
        checker.check(state, reason)
        expected_bit = 0 if failure == "joint_position" and mode == "ignore" else bit
        np.testing.assert_array_equal(reason.numpy(), [expected_bit, 0])
        score = wp.array([1., 1.], dtype=wp.float32)
        terminated = wp.zeros(2, dtype=wp.int32)
        valid = wp.zeros(2, dtype=wp.int32)
        if config.joint_position_limit_mode == "terminate":
            cause = wp.zeros(2, dtype=wp.int32)
            wp.launch(apply_joint_limit_termination, dim=2, inputs=[reason, terminated, cause])
        expected_invalid = expected_bit if config.joint_position_limit_mode != "terminate" else expected_bit & ~16
        wp.launch(terminate_invalid_states, dim=2, inputs=[reason, score, terminated, valid])
        np.testing.assert_array_equal(terminated.numpy(), [int(expected_bit != 0), 0])
        np.testing.assert_array_equal(reason.numpy(), [expected_invalid, 0])
        np.testing.assert_array_equal(valid.numpy(), [int(expected_invalid == 0), 1])
        np.testing.assert_array_equal(score.numpy(), [float(expected_invalid == 0), 1.])
        for name, expected in (("body_q", poses), ("body_qd", velocities), ("joint_q", q), ("joint_qd", qd)):
            np.testing.assert_array_equal(getattr(state, name).numpy(), expected)


def test_joint_limit_terminal_reward_matches_task_failure_and_keeps_other_invalid_bits():
    import warp as wp
    from flash_chord.envs.rl import apply_joint_limit_termination, terminate_invalid_states
    from flash_chord.objectives.composition import ObjectiveBuffers
    from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES, ObjectiveConfig

    with wp.ScopedDevice("cpu"):
        # Joint-only failure, object-only failure, both, joint+NaN, joint+speed, healthy.
        reason = wp.array([16, 0, 16, 17, 48, 0], dtype=wp.int32)
        terminated = wp.array([0, 1, 1, 0, 0, 0], dtype=wp.int32)
        cause, valid = wp.zeros(6, dtype=wp.int32), wp.zeros(6, dtype=wp.int32)
        wp.launch(apply_joint_limit_termination, dim=6, inputs=[reason, terminated, cause])
        buffers = ObjectiveBuffers.build(6, 0.05, ObjectiveConfig(), device="cpu")
        zero = wp.zeros(6)
        buffers.compose(terminated, zero, zero)
        wp.launch(terminate_invalid_states, dim=6, inputs=[reason, buffers.score, terminated, valid])
        np.testing.assert_array_equal(valid.numpy(), [1, 1, 1, 0, 0, 1])
        np.testing.assert_array_equal(cause.numpy(), [1, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(reason.numpy(), [0, 0, 0, 1, 32, 0])
        # Default -100 termination weight times 1/20 s; simultaneous causes penalize once.
        np.testing.assert_allclose(buffers.score.numpy(), [-5, -5, -5, 0, 0, 0])
        terms = buffers.terms.numpy().reshape(6, -1)
        np.testing.assert_array_equal(terms[:, OBJECTIVE_TERM_NAMES.index("termination")], [1, 1, 1, 1, 1, 0])
