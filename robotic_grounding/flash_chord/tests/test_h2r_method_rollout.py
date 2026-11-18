# SPDX-License-Identifier: Apache-2.0
"""The method rollout of a real checkpoint: its training simulator from the evaluation initial state."""

import os
from pathlib import Path

import numpy as np
import pytest

from flash_chord.h2r.controller import METHOD, controller_information, initial_robot
from flash_chord.h2r.h2r_io.method_rollout import CONDITIONS, load_method_rollout
from flash_chord.h2r.inputs import load_prepared, sha256
from flash_chord.h2r.method_rollout import export
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config


@pytest.mark.gpu
def test_method_rollout_is_every_physics_step_of_the_checkpoint_from_the_evaluation_state(tmp_path):
    keys = ("FLASHCHORD_H2R_CHECKPOINT", "FLASHCHORD_H2R_BINDING")
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("set checkpoint and binding paths for the method rollout")
    checkpoint, binding = (Path(os.environ[key]) for key in keys)
    identity = sha256(checkpoint)
    task = load_prepared(binding)[1]

    # The export checks itself that the physics-step states pass through the native control-step record.
    manifest, episodes = load_method_rollout(export(checkpoint, binding.parent, tmp_path / "rollout", worlds=2), task)

    # The same identity and deployed artifact as the evaluation client reports in `start`.
    assert manifest["binding"] == {"demo_id": task.demo.demo_id, "robot_id": task.robot.robot_id, "method": METHOD,
                                   "controller": controller_information(checkpoint)}
    assert manifest["binding"]["controller"]["sha256"] == identity == sha256(checkpoint)  # only read
    assert manifest["conditions"] == CONDITIONS
    sim = checkpoint_config(read_checkpoint_metadata(checkpoint))["env"]["sim"]
    physics_dt = 1 / (sim["fps"] * sim["substeps"])
    assert manifest["simulator"]["physics_dt_s"] == physics_dt
    assert manifest["simulator"]["control_dt_s"] == 1 / sim["fps"] and not manifest["simulator"]["deterministic"]

    proposed, start = initial_robot(binding.parent), task.initial_object_state
    control_steps = int(np.ceil(task.duration_s * sim["fps"] - 1e-10))
    assert len(episodes) == 2
    for states, entry in zip(episodes, manifest["episodes"], strict=True):
        time, verdict = states["time_s"], entry["verdict"]
        np.testing.assert_allclose(time, np.arange(len(time)) * physics_dt, atol=1e-12, rtol=0)
        # The first state is the evaluation initial state, in the simulator's single precision.
        np.testing.assert_allclose(states["robot_q"][0], [proposed[name] for name in manifest["joint_names"]],
                                   atol=1e-6, rtol=0)
        np.testing.assert_allclose(states["object_root_poses"][0],
                                   [start["object_roots"][name] for name in manifest["object_roots"]],
                                   atol=1e-6, rtol=0)
        np.testing.assert_allclose(states["object_q"][0],
                                   [start["object_joints"][name] for name in manifest["object_joints"]],
                                   atol=1e-6, rtol=0)
        if verdict["ended_at_s"] is None:  # the whole task, every physics step of every control step
            assert len(time) == control_steps * sim["substeps"] + 1 and time[-1] >= task.duration_s - 1e-9
        else:  # ended by the training environment itself
            assert not verdict["success"] and verdict["reason"]
            assert verdict["ended_at_s"] == time[-1] < task.duration_s
        # Physics steps are recorded, not control steps held: the robot moves within a control step.
        assert np.abs(np.diff(states["robot_q"][: sim["substeps"] + 1], axis=0)).max(axis=1).all()
        assert set(verdict["metrics"]) == {"max_wrist_position_m", "max_wrist_orientation_rad",
                                           "max_object_position_m", "max_object_orientation_rad"}
