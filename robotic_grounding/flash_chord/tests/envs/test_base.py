# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the shared environment transition."""

import numpy as np
import pytest


pytestmark = pytest.mark.gpu


def test_finish_transition_distinguishes_termination_truncation_and_active_worlds():
    import warp as wp

    from flash_chord.envs.base import finish_transition

    with wp.ScopedDevice("cuda:0"):
        timestep = wp.array([2, 4, 5, 2], dtype=wp.int32)
        episode_step = wp.array([10, 20, 30, 40], dtype=wp.int32)
        terminated = wp.array([1, 1, 0, 0], dtype=wp.int32)
        truncated = wp.array([0, 0, 1, 0], dtype=wp.int32)
        wp.launch(
            finish_transition,
            dim=4,
            inputs=[
                terminated,
                truncated,
            ],
            outputs=[timestep, episode_step],
        )

    assert terminated.numpy().tolist() == [1, 1, 0, 0]
    assert truncated.numpy().tolist() == [0, 0, 1, 0]
    assert timestep.numpy().tolist() == [2, 4, 5, 3]
    assert episode_step.numpy().tolist() == [10, 20, 30, 41]


def test_base_env_eager_and_captured_transition_agree(h2r_setup):
    import warp as wp

    from flash_chord.envs.base import BaseEnv, BaseEnvConfig
    from flash_chord.lifecycle.termination import Termination, TrackingTermination
    from flash_chord.objectives.composition import Objective
    from flash_chord.objectives.config import (
        ContactSupportObjectiveTermConfig,
        ObjectiveConfig,
        ObjectiveTermConfig,
        ShapedObjectiveTermConfig,
    )
    from flash_chord.runtime.actions import Action
    from flash_chord.h2r.action import H2rActionConfig
    from flash_chord.runtime.delay import TargetDelayConfig
    from flash_chord.runtime.sim import SimConfig

    with wp.ScopedDevice("cuda:0"):
        setup = h2r_setup()
        scene, reference = setup.scene, setup.reference
        action_strategy = H2rActionConfig(delay=TargetDelayConfig(min_steps=2, max_steps=2)).build(scene)
        objective_config = ObjectiveConfig(
            object_keypoints=ShapedObjectiveTermConfig(weight=0.0, enabled=False),
            hand_keypoints=ShapedObjectiveTermConfig(weight=0.0, enabled=False),
            hand_joint_pos=ShapedObjectiveTermConfig(weight=0.0, enabled=False),
            contact_wrench_support=ContactSupportObjectiveTermConfig(weight=0.0, enabled=False),
            missed_contact=ObjectiveTermConfig(weight=0.0, enabled=False),
            unintended_contact=ObjectiveTermConfig(weight=0.0, enabled=False),
            termination=ObjectiveTermConfig(weight=0.0),
            action_rate_l2=ObjectiveTermConfig(weight=0.0),
            action_l2=ObjectiveTermConfig(weight=0.0),
            num_wrench_basis=4,
            num_friction_cone_edges=4,
        )
        env = BaseEnv(
            scene,
            reference,
            config=BaseEnvConfig(sim=SimConfig(fps=30.0, substeps=4, njmax=1024, nconmax=256), objective=objective_config),
            action=action_strategy,
        )
        action = wp.zeros(env.action.action_dim, dtype=wp.float32)

        assert env.action is action_strategy
        assert isinstance(env.action, Action)
        assert isinstance(env.termination, Termination)
        assert isinstance(env.termination, TrackingTermination)
        assert isinstance(env.objective, Objective)
        assert env.contact is not None
        expected_contact_slots = sum(
            env.command.layout.num_bodies * scene.layout.hand(side).contact_link_count
            for side in env.contact.layout.sides
        )
        assert env.contact.layout.slots_per_world == expected_contact_slots

        env.reset(frame_id=0)
        env.step(action)
        eager_joint_q = env.state_0.joint_q.numpy().copy()
        eager_joint_qd = env.state_0.joint_qd.numpy().copy()
        assert env.timestep.numpy().tolist() == [1]
        assert env.episode_step.numpy().tolist() == [1]
        assert env.terminated.numpy().tolist() == [0]
        assert env.truncated.numpy().tolist() == [0]
        assert np.isfinite(eager_joint_q).all()
        assert np.isfinite(eager_joint_qd).all()
        assert np.isfinite(env.contact.contact_pos_b.numpy()).all()
        assert np.isfinite(env.contact.contact_force_direction_b.numpy()).all()
        np.testing.assert_allclose(env.objective.terms.numpy(), 0.0)
        np.testing.assert_allclose(env.objective.score.numpy(), 0.0)

        env.reset(frame_id=0)
        env.capture_transition()
        env.reset(frame_id=0)
        env.step(action)
        captured_joint_q = env.state_0.joint_q.numpy()
        captured_joint_qd = env.state_0.joint_qd.numpy()

    np.testing.assert_allclose(captured_joint_q, eager_joint_q, atol=1.0e-6)
    np.testing.assert_allclose(captured_joint_qd, eager_joint_qd, atol=1.0e-6)
