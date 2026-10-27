"""An analytic moving body catches evaluation's one-control-step time shift."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from flash_chord.evaluation import rollout


@pytest.mark.parametrize("start", [0, 2])
def test_samples_include_initial_state_and_exact_reference_endpoint(monkeypatch, start):
    device = "cpu"
    reference = SimpleNamespace(
        num_frames=5,
        object_body_pos_w=lambda: np.array([[[i * 0.1, 0, 0]] for i in range(5)]),
        object_body_quat_w=lambda: np.tile([1, 0, 0, 0], (5, 1, 1)),
        object_body_names=lambda: ("object",),
    )
    layout = SimpleNamespace(num_bodies=1, num_objects=1, body_ids=(0,),
                             voc_object_body_offsets=(0, 1), voc_object_body_ids=(0,))
    state = SimpleNamespace(body_q=wp.array([[start * 0.1, 0, 0, 0, 0, 0, 1]], dtype=wp.transform, device=device))
    termination = SimpleNamespace(cause_names=("object",), error_names=("object_position_m",),
                                  packed_diagnostic_names=("object", "object_position_m"),
                                  packed_diagnostics=wp.zeros(2, dtype=wp.float32, device=device))
    env = SimpleNamespace(
        world_count=1, state_0=state, device=wp.get_device(device), scene=None,
        command=SimpleNamespace(body_ids_w=wp.array([0], dtype=wp.int32, device=device), layout=layout),
        timestep=wp.array([start], dtype=wp.int32, device=device),
        terminated=wp.zeros(1, dtype=wp.int32, device=device),
        truncated=wp.zeros(1, dtype=wp.int32, device=device),
        termination=termination, config=SimpleNamespace(sim=SimpleNamespace(fps=10.0)),
    )
    advances = []
    target = [start * 0.1]

    def gather(phase):
        target[0] = int(phase.numpy()[0]) * 0.1

    env.command.gather = gather

    def step(action):
        frame = int(env.timestep.numpy()[0]) + 1
        advances.append(frame)
        env.timestep.assign([frame])
        state.body_q.assign([[frame * .1, 0, 0, 0, 0, 0, 1]])
        return SimpleNamespace(observation=None)

    def evaluate(body, phase):
        frame = int(phase.numpy()[0])
        # Object termination consumes the command buffer, which otherwise
        # still holds the previous transition's reference after stepping.
        error = abs(float(body.numpy()[0, 0]) - target[0])
        termination.packed_diagnostics.assign([0, error])
        env.truncated.assign([int(frame == 4)])

    termination.evaluate = evaluate
    monkeypatch.setattr(rollout, "sample_object_vertices", lambda _: np.zeros((1, 500, 3)))
    result = rollout.record_object_pose_rollout(
        env, SimpleNamespace(step=step), lambda *args: (None, None), None, None, None,
        reference=reference, step_count=5-start, start_frame=start, progress_interval=0,
    )
    assert advances == list(range(start + 1, 5))
    np.testing.assert_allclose(result.achieved_pose_w[:, 0], result.reference_pose_w, atol=2e-8)
    np.testing.assert_allclose(result.tracking_error, 0, atol=2e-8)
    assert result.completion.completion_step.tolist() == [4-start]
    assert result.completion.truncated.tolist() == [True]
    assert result.completion.reference_progress.tolist() == [1.0]
