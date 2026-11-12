# SPDX-License-Identifier: Apache-2.0
"""Optional checks against a prepared common bundle, never a substitute fixture."""

import json
import os
import runpy
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

from flash_chord.configuration import compose_config, instantiate_typed, resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.h2r.configuration import require_h2r_input
from flash_chord.h2r.inputs import human_bundle
from flash_chord.h2r.reference import load_h2r
from flash_chord.runtime.actions.hand_pose import ActionBuffers
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene


@pytest.fixture
def binding():
    path = os.environ.get("FLASHCHORD_H2R_BINDING")
    if not path:
        pytest.skip("set FLASHCHORD_H2R_BINDING to a prepared h2r_binding.json")
    return Path(path)


def test_reference_retains_source_clock_and_object(binding):
    _, metadata, _, arrays = human_bundle(binding.parent / "human")
    reference = load_h2r(binding, control_fps=30)
    assert reference.num_frames == len(arrays["time_s"])
    np.testing.assert_allclose(reference.object_body_pos_w(), arrays["part_pos_m"], atol=0, rtol=0)
    np.testing.assert_allclose(reference.object_body_quat_w(), arrays["part_quat_wxyz"], atol=1e-12, rtol=0)
    if metadata["dataset"] == "oakink2":
        assert reference.object_articulation().shape == (reference.num_frames, 0)
        assert reference.object_body_names() == metadata["object_geometry"]["part_order"]
        assert len(reference.object_assets()) == len(reference.object_body_names())
    else:
        np.testing.assert_allclose(reference.object_articulation()[:, 0], arrays["object_hinge_rad"], atol=0, rtol=0)
    assert (reference.num_frames - 1) / reference.fps == metadata["duration_s"]
    for kwargs in ({"motion_speed": 0.5}, {"control_fps": 25}, {"source_frame_playback": True}):
        with pytest.raises(ValueError):
            load_h2r(binding, **kwargs)
    with pytest.raises(ValueError):
        reference.frame_window(1, -1)


def test_application_guard_accepts_sealed_input_and_rejects_changed_binding(binding, tmp_path):
    cfg = compose_config("train_flash_sac", [f"task.parquet={binding}"])
    expected = require_h2r_input(cfg)
    assert expected["human_bundle_id"]
    moved = tmp_path / "prepared"
    shutil.copytree(binding.parent, moved)
    cfg.task.parquet = str(moved / "h2r_binding.json")
    assert require_h2r_input(cfg, expected_binding=expected) == expected
    with pytest.raises(ValueError, match="same sealed H2R input"):
        require_h2r_input(cfg, expected_binding={**expected, "scene_id": "other"})
    (moved / "human/reference.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input changed"):
        require_h2r_input(cfg)


@pytest.mark.parametrize("application", ["train_flash_sac", "view_policy", "evaluate_policy"])
def test_mixed_native_and_h2r_checkpoint_is_rejected(binding, monkeypatch, application):
    cfg = compose_config("train_flash_sac", [f"task.parquet={binding}"])
    native = resolved_dict(cfg)
    native["embodiment"] = {"_target_": "flash_chord.embodiments.sharpa_hands.SharpaHands"}
    metadata = {"resolved_config_json": json.dumps(native)}
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if application == "evaluate_policy":
        from flash_chord.evaluation import harness

        monkeypatch.setattr(harness, "read_checkpoint_metadata", lambda _: metadata)
        with pytest.raises(ValueError, match="H2R"):
            harness.evaluate_checkpoint("/tmp/native.safetensors", world_count=1)
    else:
        main = runpy.run_path(str(scripts / f"{application}.py"))["main"].__wrapped__
        monkeypatch.setitem(main.__globals__, "read_checkpoint_metadata", lambda _: metadata)
        if application == "train_flash_sac":
            cfg.training.resume = True
            cfg.training.checkpoint_path = "/tmp/native.safetensors"
        else:
            cfg = compose_config("view_policy", ["evaluation.checkpoint=/tmp/native.safetensors"])
        with pytest.raises(ValueError, match="H2R"):
            main(cfg)


def test_prepared_bundle_is_portable_and_detects_changed_inputs(binding, tmp_path):
    moved = tmp_path / "moved"
    shutil.copytree(binding.parent, moved)
    reference = load_h2r(moved / "h2r_binding.json")
    assert all(Path(p).is_relative_to(moved) for p in reference.object_mesh_paths())
    (moved / "human/reference.npz").write_bytes(b"changed input")
    with pytest.raises(ValueError, match="Input changed"):
        load_h2r(moved / "h2r_binding.json")


@pytest.mark.gpu
def test_native_action_semantics_common_fk_and_physical_rollout(binding):
    cfg = compose_config(
        "train_flash_sac",
        [
            "experiment=h2r_sharpa_flash_sac",
            f"task.parquet={binding}",
            "scene.world_count=2",
            "logging.mode=disabled",
            "env.voc_scale=0",
        ],
    )
    with wp.ScopedDevice("cuda:0"):
        config = instantiate_typed(cfg.env, RLEnvConfig)
        setup = setup_scene(
            parquet=binding,
            control_fps=config.sim.fps,
            motion_speed=1.0,
            embodiment=instantiate_typed(cfg.embodiment, Embodiment),
            collision=instantiate_typed(cfg.collision, CollisionPolicy),
            world_count=2,
            include_support=True,
            decompose_objects=False,
        )
        scene = setup.scene
        if setup.reference.metadata.source_dataset == "oakink2":
            assert len(scene.objects) == len(setup.reference.object_body_names())
            q = scene.model.joint_q.numpy().reshape(2, -1)
            for obj in scene.objects:
                index = obj.root.reference_body_id
                np.testing.assert_allclose(q[0, list(obj.root.free_q_ids)[:3]],
                    setup.reference.object_body_pos_w()[0, index], atol=1e-7, rtol=0)
        env = RLEnv(scene, setup.reference, config=config)
        env.reset()
        native = ActionBuffers.build(
            scene.layout,
            scene.robot_reference,
            2,
            config.action,
            num_joint_dof=scene.model.joint_dof_count // 2,
            wrist_dof_ids={hand.side: hand.arm_dof_ids for hand in scene.layout.hands},
        )
        common = mujoco.MjModel.from_xml_path(str(binding.parent / "robot/robot.xml"))
        state = mujoco.MjData(common)
        rng = np.random.default_rng(7)
        # Exercise the native EMA, normalized residual mapping and regularization
        # over successive calls, then verify the decoded targets by COMMON FK.
        for _ in range(5):
            action = wp.array(rng.uniform(-0.8, 0.8, 112), dtype=wp.float32)
            native.process(action, env.timestep, env.state_0)
            env.action.process(action, env.timestep, env.state_0)
            for name in ("processed_target", "action_l2", "action_rate_l2"):
                np.testing.assert_array_equal(getattr(env.action, name).numpy(), getattr(native, name).numpy())
            processed = env.action.processed_target.numpy().reshape(2, 58)
            targets = env.action.joint_target.numpy().reshape(2, -1)
            for world in range(2):
                state.qpos[:] = targets[world, :56]
                mujoco.mj_forward(common, state)
                for side, offset in (("right", 0), ("left", 29)):
                    palm = state.body(f"{side}_hand_C_MC")
                    np.testing.assert_allclose(palm.xpos, processed[world, offset : offset + 3], atol=2e-6, rtol=0)
                    assert abs(np.dot(palm.xquat, processed[world, offset + 3 : offset + 7])) > 1 - 2e-6
            env.control.joint_f.zero_()
            env.action.prepare_control(env.control)
            env.action.apply_control(env.state_0, env.control)
            np.testing.assert_array_equal(env.control.joint_f.numpy(), 0)
        env.reset()
        zero = wp.zeros(112, dtype=wp.float32)
        for _ in range(8):
            observation, reward, _, _ = env.step(zero)
            assert np.isfinite(observation.numpy()).all()
            assert np.isfinite(reward.numpy()).all()
            assert np.isfinite(env.state_0.joint_q.numpy()).all()


def test_control_reference_interpolates_without_changing_task_endpoint(binding):
    from scipy.spatial.transform import Rotation, Slerp

    source = load_h2r(binding, control_fps=30)
    reference = load_h2r(binding, control_fps=20)
    duration = source.sample_times_s[-1]
    times = reference.sample_times_s
    assert reference.fps == 20
    assert times[0] == 0 and times[-1] == duration
    assert times[-2] < duration <= times[-2] + 1 / 20 + 1e-12
    np.testing.assert_allclose(np.diff(times[:-1]), 1 / 20, atol=1e-12)
    for side in ("left", "right"):
        position = np.stack([np.interp(times, source.sample_times_s, col)
                             for col in source.wrist_pos_w(side).T], axis=1)
        np.testing.assert_allclose(reference.wrist_pos_w(side), position, atol=1e-12)
        quaternions = source.wrist_quat_w(side)[:, [1, 2, 3, 0]]
        expected = Slerp(source.sample_times_s, Rotation.from_quat(quaternions))(times)
        actual = Rotation.from_quat(reference.wrist_quat_w(side)[:, [1, 2, 3, 0]])
        np.testing.assert_allclose((actual.inv() * expected).magnitude(), 0, atol=1e-12)
        np.testing.assert_array_equal(reference.finger_joint_pos(side)[[0, -1]],
                                      source.finger_joint_pos(side)[[0, -1]])
    np.testing.assert_allclose(reference.object_body_pos_w()[[0, -1]],
                               source.object_body_pos_w()[[0, -1]], atol=1e-12)


@pytest.mark.parametrize("fps,substeps,message", [
    (30, 4, "control frequency differs"),
    (20, 6, "physics substeps differ"),
])
def test_resume_requires_the_saved_training_clock(binding, monkeypatch, fps, substeps, message):
    cfg = compose_config("train_flash_sac", [f"task.parquet={binding}"])
    saved = resolved_dict(cfg)
    saved["env"]["sim"].update(fps=fps, substeps=substeps)
    metadata = {"resolved_config_json": json.dumps(saved)}
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train_flash_sac.py"))["main"].__wrapped__
    monkeypatch.setitem(main.__globals__, "read_checkpoint_metadata", lambda _: metadata)
    cfg.training.resume = True
    cfg.training.checkpoint_path = "/tmp/policy30.safetensors"
    with pytest.raises(ValueError, match=message):
        main(cfg)


@pytest.mark.parametrize("saved_mode,message", [
    (None, "joint_position_limit_mode is discard"),
    ("ignore", "joint_position_limit_mode is ignore"),
])
def test_resume_requires_the_saved_joint_limit_mode(binding, monkeypatch, saved_mode, message):
    cfg = compose_config("train_flash_sac", [f"task.parquet={binding}"])
    saved = resolved_dict(cfg)
    if saved_mode is None:  # Checkpoint written before the option existed.
        del saved["env"]["state_validity"]["joint_position_limit_mode"]
    else:
        saved["env"]["state_validity"]["joint_position_limit_mode"] = saved_mode
    metadata = {"resolved_config_json": json.dumps(saved)}
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train_flash_sac.py"))["main"].__wrapped__
    monkeypatch.setitem(main.__globals__, "read_checkpoint_metadata", lambda _: metadata)
    cfg.training.resume = True
    cfg.training.checkpoint_path = "/tmp/policy.safetensors"
    with pytest.raises(ValueError, match=message):
        main(cfg)
