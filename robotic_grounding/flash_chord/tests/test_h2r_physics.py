"""Compare actual H2R dynamics to an independently supplied common MuJoCo scene."""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from flash_chord.configuration import compose_config, instantiate_typed
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.h2r.reference import load_h2r
from flash_chord.h2r.robot import H2rSharpaHands
from flash_chord.h2r.scene import build_h2r_scene
from flash_chord.h2r.validation import validate_solver
from flash_chord.scene.collision import CollisionPolicy

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def physics():
    binding = os.environ.get("FLASHCHORD_H2R_BINDING")
    scene_path = os.environ.get("FLASHCHORD_H2R_SCENE")
    if not binding or not scene_path:
        pytest.skip("set FLASHCHORD_H2R_BINDING and FLASHCHORD_H2R_SCENE for physical parity")
    binding = Path(binding)
    cfg = compose_config("train_flash_sac", [
        "experiment=h2r_sharpa_flash_sac", f"task.parquet={binding}",
        "scene.world_count=1", "env.auto_reset=false", "env.voc_scale=0",
        "env.voc.max_force=0", "env.voc.max_torque=0", "logging.mode=disabled",
    ])
    with wp.ScopedDevice("cuda:0"):
        reference = load_h2r(binding)
        scene = build_h2r_scene(reference, H2rSharpaHands(binding), 1, CollisionPolicy(ground=False))
        env = RLEnv(scene, reference, instantiate_typed(cfg.env, RLEnvConfig))
    return env, mujoco.MjModel.from_xml_path(scene_path), binding


def test_fixed_support_and_active_solver_guards(physics):
    env, common, binding = physics
    solver, gpu = env.solver, env.solver.mjw_model
    assert solver.mj_model.nv == common.nv
    np.testing.assert_allclose(gpu.stat.meaninertia.numpy(), common.stat.meaninertia, rtol=2e-4)
    assert not gpu.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    validate_solver(solver, binding)

    # Check the active GPU state, not just the CPU inspection model.
    original = gpu.stat.meaninertia.numpy()
    try:
        gpu.stat.meaninertia.fill_(1e9)
        with pytest.raises(ValueError, match="GPU meaninertia"):
            validate_solver(solver, binding)
    finally:
        gpu.stat.meaninertia.assign(original)
    original_flags = gpu.opt.disableflags
    try:
        gpu.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        with pytest.raises(ValueError, match="GPU physics disable flags"):
            validate_solver(solver, binding)
    finally:
        gpu.opt.disableflags = original_flags


def test_same_initial_state_and_targets_in_common_physics(physics):
    import mujoco_warp as mjwarp

    env, common, _ = physics
    env.reset(frame_id=0)
    data = mujoco.MjData(common)
    layout = env.scene.layout.scalar_joints
    q_ids = [int(common.joint(n).qposadr[0]) for n in layout.names]
    dof_ids = [int(common.joint(n).dofadr[0]) for n in layout.names]
    actuator_ids = [int(np.flatnonzero(common.actuator_trnid[:, 0] == common.joint(n).id)[0])
                    for n in layout.names]
    data.qpos[q_ids] = env.state_0.joint_q.numpy()[list(layout.q_ids)]
    root = int(common.joint("object_free").qposadr[0])
    # Both runtimes initialize through float32 coordinates; preserve that boundary.
    data.qpos[root:root + 3] = env.reference.object_body_pos_w()[0, 0].astype(np.float32)
    data.qpos[root + 3:root + 7] = env.reference.object_body_quat_w()[0, 0].astype(np.float32)
    data.qpos[int(common.joint("object_hinge").qposadr[0])] = np.float32(env.reference.object_articulation()[0, 0])
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
            from flash_chord.h2r.validation import named_id

            # VOC=0 must also disable the method's implicit hinge servos.
            native = env.solver.mj_model
            hinge = named_id(native, "joint", "object_hinge")
            assist = np.flatnonzero(native.actuator_trnid[:, 0] == hinge)
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
        for body_id, name in zip(env.command.layout.body_ids, ("object_bottom", "object_top")):
            observed = data.body(name)
            position_errors.append(float(np.linalg.norm(native[body_id, :3] - observed.xpos)))
            delta = Rotation.from_quat(native[body_id, 3:]).inv() * Rotation.from_quat(observed.xquat[[1, 2, 3, 0]])
            rotation_errors.append(float(delta.magnitude()))
    assert contacts_seen > 0
    # The imported float32 mesh/FK model need not choose identical contact
    # manifolds. Bound short-horizon motion independently of the tight servo check.
    assert max(position_errors) < 1e-3  # 1 mm over 16 physics steps
    assert max(rotation_errors) < 5e-3  # 0.29 degrees


def test_masked_reset_discards_solver_and_contact_history(physics):
    """A captured reset must recover poisoned worlds without resetting peers."""
    from dataclasses import replace

    original, _, binding = physics
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
        env.step(wp.zeros(2 * env.action.action_dim, dtype=wp.float32))
        assert np.isfinite(env.state_0.joint_q.numpy()).all()
