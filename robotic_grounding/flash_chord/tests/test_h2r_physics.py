"""Compare actual H2R training dynamics to the task's own assembled MuJoCo scene."""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from flash_chord.configuration import compose_config, instantiate_typed
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.h2r.configuration import require_h2r_input
from flash_chord.h2r.reference import load_h2r
from flash_chord.h2r.robot import H2rSharpaHands
from flash_chord.h2r.scene import build_h2r_scene
from flash_chord.h2r.validation import named_id, task_model, validate_solver
from flash_chord.scene.collision import CollisionPolicy

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def physics():
    binding = os.environ.get("FLASHCHORD_H2R_BINDING")
    if not binding:
        pytest.skip("set FLASHCHORD_H2R_BINDING to a prepared h2r_binding.json for physical parity")
    binding = Path(binding)
    cfg = compose_config("train_flash_sac", [
        "experiment=h2r_sharpa_flash_sac", f"task.parquet='{binding}'",
        "scene.world_count=1", "env.auto_reset=false", "env.voc_scale=0",
        "env.voc.max_force=0", "env.voc.max_torque=0", "logging.mode=disabled",
    ])
    require_h2r_input(cfg)
    with wp.ScopedDevice("cuda:0"):
        reference = load_h2r(binding, control_fps=cfg.env.sim.fps)
        scene = build_h2r_scene(reference, H2rSharpaHands(binding), 1, CollisionPolicy(ground=False))
        env = RLEnv(scene, reference, instantiate_typed(cfg.env, RLEnvConfig))
    # The comparison model is the sealed task's assembled scene, never one derived from the method model.
    return env, task_model(reference.task), reference.task


def test_fixed_support_and_active_solver_guards(physics):
    env, common, task = physics
    solver, gpu = env.solver, env.solver.mjw_model
    assert solver.mj_model.nv == common.nv
    np.testing.assert_allclose(gpu.stat.meaninertia.numpy(), common.stat.meaninertia, rtol=2e-4)
    assert not gpu.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    validate_solver(solver, task)

    # Check the active GPU state, not just the CPU inspection model.
    original = gpu.stat.meaninertia.numpy()
    try:
        gpu.stat.meaninertia.fill_(1e9)
        with pytest.raises(ValueError, match="GPU meaninertia"):
            validate_solver(solver, task)
    finally:
        gpu.stat.meaninertia.assign(original)
    original_flags = gpu.opt.disableflags
    try:
        gpu.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        with pytest.raises(ValueError, match="GPU physics disable flags"):
            validate_solver(solver, task)
    finally:
        gpu.opt.disableflags = original_flags


@pytest.mark.parametrize("lift_m", [0.0, 1.0])
def test_same_initial_state_and_targets_in_common_physics(physics, lift_m):
    """Same state and targets in the method model and in the task model, both stepped by MJWarp.

    The method model may not deviate from the task model by more than the task model deviates from
    itself when its start moves by a few float32 steps: contact-rich starts (hands closed around an
    object, a light part between tongs) amplify rounding on their own. At the reference pose the
    objects rest on their supports and the hands move, so contacts, masses, inertias and joints
    are all exercised; raised into free fall, the objects check gravity and integration alone.
    """
    from copy import deepcopy

    import mujoco_warp as mjwarp
    import newton

    env, common, _ = physics
    # Compare asset import at the training timestep. The sealed evaluation
    # model keeps its own timestep and is not changed by this local copy.
    common = deepcopy(common)
    common.opt.timestep = env.config.sim.sim_dt
    env.reset(frame_id=0)
    if lift_m:
        q = env.state_0.joint_q.numpy()
        for obj in env.scene.objects:
            q[obj.root.free_q_ids[2]] += lift_m
        env.state_0.joint_q.assign(q)
        newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
    data = mujoco.MjData(common)
    layout = env.scene.layout.scalar_joints
    q_ids = [int(common.joint(n).qposadr[0]) for n in layout.names]
    dof_ids = [int(common.joint(n).dofadr[0]) for n in layout.names]
    actuator_ids = [int(np.flatnonzero(common.actuator_trnid[:, 0] == common.joint(n).id)[0])
                    for n in layout.names]
    data.qpos[q_ids] = env.state_0.joint_q.numpy()[list(layout.q_ids)]
    task = env.reference.task
    parts = env.reference.object_body_names()
    hinges = [joint.mjcf_name for obj in task.demo.objects.values() for joint in obj.joints]
    roots = [int(common.joint(obj.root_joint).qposadr[0]) for obj in task.demo.objects.values()]
    # Both runtimes initialize through float32 coordinates; preserve that boundary.
    for obj, root in zip(task.demo.objects.values(), roots):
        index = parts.index(obj.root_part)
        data.qpos[root:root + 3] = env.reference.object_body_pos_w()[0, index].astype(np.float32)
        data.qpos[root + 2] = np.float32(np.float32(data.qpos[root + 2]) + np.float32(lift_m))
        data.qpos[root + 3:root + 7] = env.reference.object_body_quat_w()[0, index].astype(np.float32)
    for column, name in enumerate(hinges):
        data.qpos[int(common.joint(name).qposadr[0])] = np.float32(env.reference.object_articulation()[0, column])
    # The rounding reference: the same task, its start moved by a few float32 steps.
    shifted = deepcopy(data)
    shifted.qpos[q_ids] += 2e-7
    for root in roots:
        shifted.qpos[root:root + 3] += (2e-8, 2e-8, 6e-8)
    worlds = []
    for start in (data, shifted):
        mujoco.mj_forward(common, start)
        # Both worlds use the same physics backend. CPU MuJoCo is used below
        # only to inspect post-integration kinematics, never to advance physics.
        worlds.append(mjwarp.put_data(common, start, nworld=1, nconmax=1024, njmax=4096))
    gpu_model = mjwarp.put_model(common)
    bodies = [task.object_binding["parts"][p] for p in parts]
    contacts_seen = 0
    method_errors, rounding_errors = np.zeros(4), np.zeros(4)  # robot q, robot qd, object position, rotation

    def errors(q, qd, poses, reference):
        position = max(float(np.linalg.norm(a[:3] - reference.body(n).xpos)) for a, n in zip(poses, bodies))
        rotation = max(float((Rotation.from_quat(a[3:]).inv()
                              * Rotation.from_quat(reference.body(n).xquat[[1, 2, 3, 0]])).magnitude())
                       for a, n in zip(poses, bodies))
        return np.array([np.abs(q - reference.qpos[q_ids]).max(), np.abs(qd - reference.qvel[dof_ids]).max(),
                         position, rotation])

    def poses(cpu, gpu):
        cpu.qpos[:], cpu.qvel[:] = gpu.qpos.numpy()[0], gpu.qvel.numpy()[0]
        # Refresh post-integration poses without advancing the world.
        mujoco.mj_kinematics(common, cpu)
        return [np.r_[cpu.body(n).xpos, cpu.body(n).xquat[[1, 2, 3, 0]]] for n in bodies]

    for step in range(4):
        action = wp.array(np.full(env.action.action_dim, 0.1 * (-1) ** step, np.float32), device=env.device)
        env.step(action)
        if step == 0:
            # VOC=0 must also disable the method's implicit servo on every object joint.
            native = env.solver.mj_model
            for name in hinges:
                assist = np.flatnonzero(native.actuator_trnid[:, 0] == named_id(native, "joint", name))
                assert len(assist) > 0
                for field in ("actuator_gainprm", "actuator_biasprm"):
                    np.testing.assert_array_equal(getattr(env.solver.mjw_model, field).numpy()[:, assist], 0)
        data.ctrl[actuator_ids] = env.action.joint_target.numpy()[list(layout.dof_ids)]
        for world in worlds:
            world.ctrl.assign(data.ctrl[None, :].astype(np.float32))
        for _ in range(env.config.sim.substeps):
            for world in worlds:
                mjwarp.step(gpu_model, world)
            contacts_seen += int(worlds[0].nacon.numpy()[0])
        poses(data, worlds[0])
        other = poses(shifted, worlds[1])
        method = env.state_0.body_q.numpy()[list(env.command.layout.body_ids)]
        method_errors = np.maximum(method_errors, errors(
            env.state_0.joint_q.numpy()[list(layout.q_ids)], env.state_0.joint_qd.numpy()[list(layout.dof_ids)],
            method, data))
        rounding_errors = np.maximum(rounding_errors, errors(
            shifted.qpos[q_ids], shifted.qvel[dof_ids], other, data))
    assert contacts_seen > 0 or lift_m
    # Robot q (rad or m), robot qd, object position (m) and rotation (rad). Before flashchord-h2r-v3,
    # Newton's default hull limit on the object meshes changed the face contacts: on the ARCTIC box the
    # rotation differed by 5.8e-3 rad while the rounding reference stayed below 1e-6.
    floors = np.array([2e-5, 2e-3, 1e-5, 1e-5])
    bounds = np.maximum(floors, 3 * rounding_errors)
    assert (method_errors <= bounds).all(), f"method {method_errors}, rounding {rounding_errors}"


def test_training_advances_at_100hz_without_changing_common_timestep(physics):
    env, common, task = physics
    original_common_dt = common.opt.timestep
    assert original_common_dt == task.timestep_s  # the task's nominal step is not the training step
    assert (env.config.sim.fps, env.config.sim.substeps) == (20, 5)
    env.reset(frame_id=0)
    before = env.solver.mjw_data.time.numpy().copy()
    env.step(wp.zeros(env.action.action_dim, dtype=wp.float32, device=env.device))
    np.testing.assert_allclose(env.solver.mjw_data.time.numpy() - before, 0.05, atol=1e-6)
    np.testing.assert_allclose(env.solver.mjw_model.opt.timestep.numpy(), 0.01, atol=1e-8)
    assert common.opt.timestep == original_common_dt


def test_masked_reset_discards_solver_and_contact_history(physics):
    """A captured reset must recover poisoned worlds without resetting peers."""
    from dataclasses import replace

    original, _, task = physics
    binding = original.reference.binding_path
    with wp.ScopedDevice("cuda:0"):
        scene = build_h2r_scene(original.reference, H2rSharpaHands(binding), 2,
                               CollisionPolicy(ground=False))
        env = RLEnv(scene, original.reference, replace(original.config, auto_reset=False))
        env.reset(frame_id=0)
        initial_q = env.state_0.joint_q.numpy().copy()
        initial_obs = env.observation_strategy.compute(env.state_0).numpy().copy()
        data = env.solver.mjw_data
        buffers = (data.qacc_warmstart, data.qfrc_applied, data.xfrc_applied, data.ctrl)
        for buffer in buffers:
            buffer.fill_(float("nan"))
        env.contact.contact_pos_b.fill_(1)
        env.contact.contact_force_direction_b.fill_(1)
        mask = wp.array([1, 0], dtype=wp.int32)
        frame = wp.zeros(2, dtype=wp.int32)
        with wp.ScopedCapture() as capture:
            env.reset(reset_frame=frame, reset_mask=mask)
        wp.capture_launch(capture.graph)
        env.check_reset_validity(mask)
        for buffer in buffers:
            values = buffer.numpy()
            np.testing.assert_array_equal(values[0], 0)
            assert np.isnan(values[1]).all()
        obs = env.observation_strategy.compute(env.state_0).numpy()
        np.testing.assert_array_equal(obs[:env.observation_dim], initial_obs[:env.observation_dim])
        slots = env.contact.layout.slots_per_world
        np.testing.assert_array_equal(env.contact.contact_pos_b.numpy()[:slots], 0)
        np.testing.assert_array_equal(env.contact.contact_pos_b.numpy()[slots:], 1)
        np.testing.assert_array_equal(env.state_0.joint_q.numpy(), initial_q)
        mask.assign([0, 1])
        wp.capture_launch(capture.graph)
        env.check_reset_validity(mask)
        # Poison the reference table so replay must produce a failed reset.
        table = env.reset_table.ref_joint_q.numpy()
        poisoned = table.copy()
        poisoned[0, 0] = float("nan")
        env.reset_table.ref_joint_q.assign(poisoned)
        wp.capture_launch(capture.graph)
        with pytest.raises(RuntimeError, match=r"worlds \[1\]"):
            env.check_reset_validity(mask)
        env.check_reset_validity(wp.array([1, 0], dtype=wp.int32))
        with pytest.raises(RuntimeError, match=r"worlds \[1\]"):
            env.reset(reset_frame=frame, reset_mask=mask)
        env.reset_table.ref_joint_q.assign(table)
        wp.capture_launch(capture.graph)
        env.check_reset_validity(mask)
        env.step(wp.zeros(2 * env.action.action_dim, dtype=wp.float32))
        assert np.isfinite(env.state_0.joint_q.numpy()).all()


def test_immediate_training_reset_matches_deployment_state_and_clock(physics):
    from dataclasses import replace

    original, _, task = physics
    binding = original.reference.binding_path
    config = replace(original.config, reset=replace(
        original.config.reset, immediate_first_frame_probability=1.0))
    with wp.ScopedDevice("cuda:0"):
        scene = build_h2r_scene(original.reference, H2rSharpaHands(binding), 2,
                               CollisionPolicy(ground=False))
        env = RLEnv(scene, original.reference, config)
        env.reset(frame_id=0)
        explicit = [a.numpy().copy() for a in
                    (env.state_0.joint_q, env.state_0.joint_qd, env.observation)]
        env.reset()
        for actual, expected in zip(
            (env.state_0.joint_q, env.state_0.joint_qd, env.observation), explicit
        ):
            np.testing.assert_array_equal(actual.numpy(), expected)
        np.testing.assert_array_equal(env.object_control_scale.numpy(), 0)
        env.capture_step()
        env.reset()
        env.step(wp.zeros(2 * env.action.action_dim, dtype=wp.float32))
        np.testing.assert_array_equal(env.timestep.numpy(), 1)
        np.testing.assert_array_equal(env.object_control_scale.numpy(), 0)


def test_sampled_frame_zero_skips_preamble_in_physical_training_step(physics):
    from dataclasses import replace

    original, _, task = physics
    binding = original.reference.binding_path
    config = replace(original.config, reset=replace(
        original.config.reset, always_reset_to_first_frame=True,
        immediate_first_frame_probability=0.0, skip_first_frame_settling=True))
    with wp.ScopedDevice("cuda:0"):
        scene = build_h2r_scene(original.reference, H2rSharpaHands(binding), 1,
                               CollisionPolicy(ground=False))
        env = RLEnv(scene, original.reference, config)
        env.reset()
        np.testing.assert_array_equal(env.timestep.numpy(), 0)
        np.testing.assert_array_equal(env.object_control_scale.numpy(), 0)
        assert np.all(env.reset_policy.finger_scale.numpy() < 0.7)
        env.capture_step()
        env.reset()
        env.step(wp.zeros(env.action.action_dim, dtype=wp.float32))
        np.testing.assert_array_equal(env.timestep.numpy(), 1)
        np.testing.assert_array_equal(env.object_control_scale.numpy(), 0)
        assert np.isfinite(env.state_0.joint_q.numpy()).all()


def test_rollout_errors_use_recorded_state_timestamp(physics):
    from types import SimpleNamespace
    from flash_chord.evaluation.rollout import record_object_pose_rollout

    env, _, _ = physics
    env.reset(frame_id=0)
    zero = wp.zeros(env.action.action_dim, dtype=wp.float32, device=env.device)

    def step(action):
        env.step(action)
        return SimpleNamespace(observation=None)

    rollout = record_object_pose_rollout(
        env, SimpleNamespace(step=step), lambda *args: (zero, None), None, None, None,
        reference=env.reference, step_count=5, progress_interval=0,
    )
    assert env.timestep.numpy().tolist() == [4]
    expected = np.linalg.norm(
        rollout.achieved_pose_w[:, 0, :, :3] - rollout.reference_pose_w[..., :3], axis=-1
    ).max(axis=1)
    index = rollout.tracking_error_names.index("object_position_m")
    np.testing.assert_allclose(rollout.tracking_error[:, 0, index], expected, atol=1e-7, rtol=0)
