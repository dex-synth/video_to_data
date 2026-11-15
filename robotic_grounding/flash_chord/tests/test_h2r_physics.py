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

    With the objects raised into free fall the two must agree to rounding: masses, inertias,
    joints, gravity and integration are the task's. At the reference pose the objects rest on
    their supports; convex-mesh contact points then depend on the imported mesh data, so only
    a short-horizon bound is checked.
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
    # Both runtimes initialize through float32 coordinates; preserve that boundary.
    for obj in task.demo.objects.values():
        root, index = int(common.joint(obj.root_joint).qposadr[0]), parts.index(obj.root_part)
        data.qpos[root:root + 3] = env.reference.object_body_pos_w()[0, index].astype(np.float32)
        data.qpos[root + 2] = np.float32(np.float32(data.qpos[root + 2]) + np.float32(lift_m))
        data.qpos[root + 3:root + 7] = env.reference.object_body_quat_w()[0, index].astype(np.float32)
    for column, name in enumerate(hinges):
        data.qpos[int(common.joint(name).qposadr[0])] = np.float32(env.reference.object_articulation()[0, column])
    mujoco.mj_forward(common, data)
    # Both worlds use the same physics backend. CPU MuJoCo is used below
    # only to inspect post-integration kinematics, never to advance physics.
    gpu_model = mjwarp.put_model(common)
    gpu_data = mjwarp.put_data(common, data, nworld=1, nconmax=1024, njmax=4096)
    contacts_seen = 0
    position_errors, rotation_errors = [], []
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
        targets = env.action.joint_target.numpy()[list(layout.dof_ids)]
        data.ctrl[actuator_ids] = targets
        gpu_data.ctrl.assign(data.ctrl[None, :].astype(np.float32))
        for _ in range(env.config.sim.substeps):
            mjwarp.step(gpu_model, gpu_data)
            contacts_seen += int(gpu_data.nacon.numpy()[0])
        data.qpos[:] = gpu_data.qpos.numpy()[0]
        data.qvel[:] = gpu_data.qvel.numpy()[0]
        np.testing.assert_allclose(env.state_0.joint_q.numpy()[list(layout.q_ids)], data.qpos[q_ids], atol=2e-5, rtol=0)
        np.testing.assert_allclose(env.state_0.joint_qd.numpy()[list(layout.dof_ids)], data.qvel[dof_ids], atol=2e-3, rtol=0)
        # Refresh post-integration poses without advancing either world.
        mujoco.mj_kinematics(common, data)
        native = env.state_0.body_q.numpy()
        for body_id, name in zip(env.command.layout.body_ids, (task.object_binding["parts"][p] for p in parts)):
            observed = data.body(name)
            position_errors.append(float(np.linalg.norm(native[body_id, :3] - observed.xpos)))
            delta = Rotation.from_quat(native[body_id, 3:]).inv() * Rotation.from_quat(observed.xquat[[1, 2, 3, 0]])
            rotation_errors.append(float(delta.magnitude()))
    if lift_m:
        assert max(position_errors) < 1e-6 and max(rotation_errors) < 1e-6
    else:
        assert contacts_seen > 0
        # The imported float32 mesh/FK model need not choose identical contact
        # manifolds. Bound short-horizon motion independently of the tight servo check.
        assert max(position_errors) < 1e-3  # 1 mm over four control intervals
        assert max(rotation_errors) < 1e-2  # 0.57 degrees


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
