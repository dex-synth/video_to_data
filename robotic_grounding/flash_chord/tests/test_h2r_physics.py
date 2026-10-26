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
    contacts_seen = 0
    position_errors, rotation_errors = [], []
    for step in range(4):
        action = wp.array(np.full(env.action.action_dim, 0.1 * (-1) ** step, np.float32), device=env.device)
        env.step(action)
        targets = env.action.joint_target.numpy()[list(layout.dof_ids)]
        data.ctrl[actuator_ids] = targets
        for _ in range(env.config.sim.substeps):
            mujoco.mj_step(common, data)
            contacts_seen += data.ncon
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
