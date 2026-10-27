# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical replay uses the sealed H2R scene and the training action path."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_h2r_replay_resets_and_steps_the_common_scene(h2r_binding, h2r_setup):
    import warp as wp
    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.h2r.configuration import require_h2r_input
    from flash_chord.runtime.replay import ReplayConfig, ReplayRunner

    cfg = compose_config("replay", [f"task.parquet='{h2r_binding}'"])
    require_h2r_input(cfg)
    with wp.ScopedDevice("cuda:0"):
        setup = h2r_setup()
        runner = ReplayRunner(setup.scene, setup.reference, instantiate_typed(cfg.replay, ReplayConfig))
        runner.reset(0)
        np.testing.assert_allclose(runner.state_0.joint_q.numpy()[:56], setup.scene.robot_reference.joint_q[0])
        assert runner.action.action_dim == 56
        assert runner.voc_scale.numpy().tolist() == [0.0]
        for frame in range(8):
            runner.step(frame)
            for value in (runner.state_0.joint_q, runner.state_0.joint_qd, runner.action.joint_target):
                assert np.isfinite(value.numpy()).all()
            np.testing.assert_array_equal(runner.action.action_l2.numpy(), 0.0)
        assert runner.sim_time == pytest.approx(8 / 30)
        runner.reset(1)
        np.testing.assert_allclose(runner.state_0.joint_q.numpy()[:56], setup.scene.robot_reference.joint_q[1])
        np.testing.assert_array_equal(runner.state_0.joint_qd.numpy(), 0)
        assert runner.sim_time == 0
