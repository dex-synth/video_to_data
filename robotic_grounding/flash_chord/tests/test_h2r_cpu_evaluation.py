"""Optional bitwise comparison with an independently executed bench CPU run."""

import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from flash_chord.h2r.cpu_evaluation import CpuTask, evaluate

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def inputs():
    keys = ("FLASHCHORD_H2R_CPU_TASK", "FLASHCHORD_H2R_CHECKPOINT",
            "FLASHCHORD_H2R_BINDING", "FLASHCHORD_H2R_CPU_BENCH", "FLASHCHORD_H2R_CPU_METRICS")
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("set CPU task, checkpoint, binding, independent bench run and metric specification paths")
    return [Path(os.environ[key]) for key in keys]


def test_cpu_snapshot_preserves_double_precision_solver_state(inputs):
    task, _, binding, _, _ = inputs
    world = CpuTask(task)
    with np.load(binding.parent / "robot_reference.npz") as reference:
        initial = dict(zip(reference["joint_names"].tolist(), reference["joint_q"][0]))
    world.reset(initial)
    world.step(initial)
    assert np.any(world.data.qpos != world.data.qpos.astype(np.float32))
    expected = mujoco.MjData(world.model)
    mujoco.mj_copyData(expected, world.model, world.data)
    world.snapshot(1)
    world.snapshot(1)
    np.testing.assert_array_equal(world.data.qpos, expected.qpos)
    np.testing.assert_array_equal(world.data.qvel, expected.qvel)
    np.testing.assert_array_equal(world.data.qacc_warmstart, expected.qacc_warmstart)
    world.step(initial)
    mujoco.mj_step(world.model, expected)
    np.testing.assert_array_equal(world.data.qpos, expected.qpos)
    np.testing.assert_array_equal(world.data.qvel, expected.qvel)


def test_full_cpu_rollout_matches_independent_bench_bitwise(inputs, tmp_path):
    task, checkpoint, binding, bench, spec = inputs
    run = json.loads((bench / "server/run.json").read_text())
    assert run["simulator"] == "mujoco"
    assert run["versions"]["mujoco"] == mujoco.__version__
    output = evaluate(task, checkpoint, binding.parent, tmp_path / "native", metric_spec=spec)
    with np.load(output / "physical.npz") as actual, np.load(bench / "server/seed-42/physical.npz") as expected:
        assert set(actual.files) == set(expected.files)
        for name in actual.files:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
            assert actual[name].dtype == expected[name].dtype, name
            assert actual[name].tobytes() == expected[name].tobytes(), name
        targets = expected["submitted_target"][1::4].copy()
    controls = [json.loads(line) for line in (bench / "client/controls.jsonl").read_text().splitlines()]
    with np.load(output / "controls.npz") as actual:
        for name in ("observation", "raw_action", "processed_action"):
            expected = np.asarray([row[name] for row in controls if name in row])
            np.testing.assert_array_equal(actual[name], expected, err_msg=name)
            assert actual[name].dtype == expected.dtype, name
            assert actual[name].tobytes() == expected.tobytes(), name
        np.testing.assert_array_equal(actual["target"], targets)
        assert actual["target"].dtype == targets.dtype
        assert actual["target"].tobytes() == targets.tobytes()
        np.testing.assert_array_equal(actual["tick"], np.arange(0, run["end_tick"], 4))
