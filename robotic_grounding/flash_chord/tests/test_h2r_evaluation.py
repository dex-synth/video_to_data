"""Optional deployment parity against actual native states and measured contacts."""

import json
import os
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from flash_chord.configuration import instantiate_typed
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.h2r.controller import Controller
from flash_chord.h2r.robot import H2rSharpaHands
from flash_chord.h2r.scene import build_h2r_scene
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference


@pytest.fixture(scope="module")
def controller():
    keys = ("FLASHCHORD_H2R_CHECKPOINT", "FLASHCHORD_H2R_BINDING", "FLASHCHORD_H2R_HELLO")
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("set checkpoint, binding and hello paths for deployment parity")
    checkpoint, binding, hello = (Path(os.environ[key]) for key in keys)
    message = json.loads(hello.read_text())
    # Deliberately differ from model/actuator order; the wire is name based.
    message["joint_names"].reverse()
    return Controller(checkpoint, binding.parent, message)


def native_frame(controller, env, tick):
    """Serialize actual Newton state and contacts using the physical wire contract."""
    c = controller
    state = env.state_0
    body = state.body_q.numpy()
    velocity = state.body_qd.numpy()
    # Independent inverse of the deployment conversion: the wire carries
    # body-origin velocity whereas Newton stores COM-referenced velocity.
    offset = Rotation.from_quat(body[:, 3:]).apply(c.model.body_com.numpy())
    velocity[:, :3] -= np.cross(velocity[:, 3:], offset)
    names = c.hello["joint_names"]
    q = dict(zip(c.names, state.joint_q.numpy()[c.q_ids]))
    dq = dict(zip(c.names, state.joint_qd.numpy()[c.dof_ids]))
    records = []
    if tick:
        count = int(env.contacts.rigid_contact_count.numpy()[0])
        shape0, shape1 = (getattr(env.contacts, f"rigid_contact_shape{i}").numpy()[:count] for i in (0, 1))
        point0, point1 = (getattr(env.contacts, f"rigid_contact_point{i}").numpy()[:count] for i in (0, 1))
        forces = env.contacts.force.numpy()[:count, :3]
        shape_bodies = c.model.shape_body.numpy()
        object_ids = set(c.command.layout.body_ids)
        for s0, s1, p0, p1, f0 in zip(shape0, shape1, point0, point1, forces):
            b0, b1 = shape_bodies[s0], shape_bodies[s1]
            if (b0 in object_ids) == (b1 in object_ids) or min(b0, b1) < 0:
                continue
            # Exclude support/object contacts, just as the bench task does.
            if c.scene.support.body_id in (b0, b1):
                continue
            records.append({"body1": c.body_names[b0], "body2": c.body_names[b1],
                            "point1_local": p0.tolist(), "point2_local": p1.tolist(),
                            "force_on_body2_world": (-f0).tolist()})
        # Both contact orders must produce identical method features.
        for record in records[::2]:
            record["body1"], record["body2"] = record["body2"], record["body1"]
            record["point1_local"], record["point2_local"] = record["point2_local"], record["point1_local"]
            record["force_on_body2_world"] = (-np.asarray(record["force_on_body2_world"])).tolist()
    return {"episode_id": c.ids[0], "done": False, "tick": tick, "time_s": tick / 120,
            "robot_q": [float(q[n]) for n in names], "robot_dq": [float(dq[n]) for n in names],
            "bodies": {name: np.r_[body[i, [0, 1, 2, 6, 3, 4, 5]], velocity[i]].tolist()
                       for i, name in c.observed_bodies.items()},
            "contacts": {"solve_tick": tick - 1 if tick else None, "records": records}}


@pytest.mark.gpu
def test_native_state_observation_actor_and_control_parity(controller):
    c = controller
    assert c.count == 1
    cfg = OmegaConf.create(c.saved_config["env"])
    cfg.auto_reset = False
    cfg.voc_scale = 0
    cfg.voc.max_force = cfg.voc.max_torque = 0
    cfg.reset.voc_decay_steps = 0
    cfg.sim.njmax = 1024
    with wp.ScopedDevice("cuda:0"):
        scene = build_h2r_scene(c.reference, H2rSharpaHands(c.reference.binding_path), 1,
                               CollisionPolicy(ground=False))
        env = RLEnv(scene, c.reference, instantiate_typed(cfg, RLEnvConfig))
        gpu = jax.devices("gpu")[0]
        gpu_actor, gpu_policy, gpu_key, _ = load_policy_for_inference(
            os.environ["FLASHCHORD_H2R_CHECKPOINT"],
            instantiate_typed(OmegaConf.create(c.saved_config["training"]), TrainingConfig),
            jax.device_put(np.zeros((1, c.observation.observation_dim), dtype=np.float32), gpu),
            c.action.action_dim, deterministic=True, device=gpu)
        env.reset(frame_id=0)
        max_difference = {"observation": 0., "action": 0., "target": 0.}
        contacts_seen = 0
        for step in range(24):
            frame = native_frame(c, env, step * 4)
            contacts_seen += len(frame["contacts"]["records"])
            native_obs = env.observation_strategy.compute(env.state_0).numpy().reshape(1, -1)
            native_action, _ = gpu_policy(gpu_actor, jax.device_put(native_obs, gpu), gpu_key)
            _, actual = c.step([frame])
            for name, got, expected in (("observation", actual["observation"], native_obs),
                                        ("action", actual["raw_action"], native_action)):
                np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-4)
                max_difference[name] = max(max_difference[name], float(np.max(np.abs(np.asarray(got) - expected))))
            env.step(wp.array(np.asarray(native_action).ravel(), dtype=wp.float32, device=env.device))
            np.testing.assert_allclose(c.action.joint_target.numpy(), env.action.joint_target.numpy(), atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(c.action.processed_target.numpy(), env.action.processed_target.numpy(), atol=1e-5, rtol=1e-4)
            max_difference["target"] = max(max_difference["target"], float(np.max(np.abs(
                c.action.joint_target.numpy() - env.action.joint_target.numpy()))))
            assert int(env.timestep.numpy()[0]) == step + 1
        assert contacts_seen > 0
        with pytest.raises(ValueError, match="tick"):
            c.step([frame])
        frame["done"] = True
        result, _ = c.step([frame])
        assert result["commands"] == {}
        print("deployment parity", max_difference, "raw contacts", contacts_seen)
