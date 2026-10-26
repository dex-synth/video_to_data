"""FlashCHORD control over externally supplied physical state; no physics solver."""

from pathlib import Path
from types import SimpleNamespace
import os

# Set before JAX initializes a backend. Fixed compilation choices matter in a
# contact feedback loop, where even one ULP can grow into a different rollout.
_flags = os.environ.get("XLA_FLAGS", "").split()
for _flag in ("--xla_gpu_deterministic_ops=true", "--xla_gpu_autotune_level=0"):
    _prefix = _flag.split("=", 1)[0] + "="
    if any(value.startswith(_prefix) and value != _flag for value in _flags):
        raise ValueError(f"H2R evaluation requires {_flag}")
    if _flag not in _flags:
        _flags.append(_flag)
os.environ["XLA_FLAGS"] = " ".join(_flags)

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import warp as wp

from flash_chord.configuration import instantiate_typed
from flash_chord.envs.observation import PolicyObservationConfig
from flash_chord.runtime.command import CommandBuffers
from flash_chord.runtime.contact import ContactTracker, ContactTrackerConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import (
    checkpoint_config, checkpoint_policy_schema, policy_schema,
    validate_policy_schema, validate_reference_config,
)
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
from flash_chord.utils.quat import quat_rotate_xyzw

from .action import H2rActionConfig
from .inputs import load_binding
from .reference import load_h2r
from .robot import H2rSharpaHands
from .scene import build_h2r_scene


def finite_array(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Invalid {name}: expected finite {shape}, got {value.shape}")
    return value


class Controller:
    """Reuse native observations, contact aggregation, actor and pose actions."""

    def __init__(self, checkpoint, prepared, hello, *, device="cuda:0"):
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("H2R deployment preserves native CUDA observation/action arithmetic")
        self.hello = hello
        self.ids = [entry["episode_id"] for entry in hello["episodes"]]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Episode IDs must be nonempty and unique")
        self.count = len(self.ids)
        prepared = Path(prepared).resolve()
        binding_path = prepared / "h2r_binding.json"
        self.binding = load_binding(binding_path)
        for key, source in (("bundle_id", "human_bundle_id"),
                            ("robot_manifest_sha256", "robot_manifest_sha256"),
                            ("robot_name", "robot_name"), ("source_frame_range", "source_frame_range")):
            if hello[key] != self.binding[source]:
                raise ValueError(f"Controller input differs from task: {key}")
        obs = hello["observation"]
        if (hello["protocol"] != "h2r-evaluation-v1" or obs.get("contacts") is not True
                or obs.get("contact_schema") != "h2r-body-contacts-v1"
                or obs["quaternion"] != "wxyz" or obs["body_velocity"] != "world_at_body_origin"):
            raise ValueError("Unsupported physical observation contract")
        if (hello["control_ticks"] != 4 or not np.isclose(hello["dt_s"], 1 / 120, rtol=0, atol=1e-12)
                or not np.isclose(hello["end_tick"] * hello["dt_s"], self.binding["duration_s"], atol=1e-12)):
            raise ValueError("Task clock differs from trained H2R reference")
        metadata = read_checkpoint_metadata(checkpoint)
        self.saved_config = checkpoint_config(metadata)
        cfg = OmegaConf.create(self.saved_config)
        trained_binding = load_binding(Path(cfg.embodiment.binding_path))
        if trained_binding != self.binding:
            raise ValueError("Checkpoint preparation differs from deployment input")
        env = cfg.env
        if (env.arm_position_integral.gain_s_inv2 != 0 or env.enforce_joint_velocity_limits
                or env.action.delay.min_steps != 0 or env.action.delay.max_steps != 0
                or env.sim.fps != 30 or env.sim.substeps != 4):
            raise ValueError("Deployment requires the trained 30 Hz, zero-delay position controller")
        self.reference = load_h2r(binding_path)
        validate_reference_config(self.saved_config, self.reference)
        with wp.ScopedDevice(self.device):
            self.scene = build_h2r_scene(self.reference, H2rSharpaHands(binding_path), self.count,
                                         CollisionPolicy(ground=False))
            self.model = self.scene.model
            self.state = self.model.state()
            self.action = instantiate_typed(env.action, H2rActionConfig).build(self.scene, device=self.device)
            self.command = CommandBuffers.build(self.scene, self.reference, device=self.device)
            self.contact = ContactTracker.build(
                self.scene.layout, self.command.layout.body_ids, self.count,
                self.model.body_count // self.count, self.scene.collision_layout.shape_count,
                config=instantiate_typed(env.contact, ContactTrackerConfig), device=self.device)
            self.timestep = wp.zeros(self.count, dtype=wp.int32, device=self.device)
            self.episode_step = wp.zeros(self.count, dtype=wp.int32, device=self.device)
            self.observation = instantiate_typed(env.observation, PolicyObservationConfig).build(
                self.scene, self.action, self.command, self.contact, self.timestep, self.episode_step,
                wp.zeros(self.model.joint_coord_count, device=self.device), device=self.device)
            self.action.reset(wp.ones(self.count, dtype=wp.int32, device=self.device))
        validate_policy_schema(checkpoint_policy_schema(metadata), policy_schema(self.action, self.observation))
        training = instantiate_typed(cfg.training, TrainingConfig)
        # Keep the same GEMM shape regardless of the number of evaluation slots.
        self.policy_device = jax.devices("gpu")[self.device.ordinal]
        with jax.default_device(self.policy_device):
            self.actor, self.policy, self.key, _ = load_policy_for_inference(
                str(checkpoint), training, jnp.zeros((1, self.observation.observation_dim)),
                self.action.action_dim, deterministic=True, device=self.policy_device)
        scalar = self.scene.layout.scalar_joints
        self.names = list(scalar.names)
        if set(self.names) != set(hello["joint_names"]) or len(hello["joint_names"]) != len(self.names):
            raise ValueError("Robot joint names differ")
        self.q_ids, self.dof_ids = list(scalar.q_ids), list(scalar.dof_ids)
        self.columns = [hello["joint_names"].index(name) for name in self.names]
        self.bodies_per_world = self.model.body_count // self.count
        self.body_names = [label.rsplit("/", 1)[-1]
                           for label in self.model.body_label[:self.bodies_per_world]]
        # The method's fixed support is geometry only; it is never used as feedback.
        self.observed_bodies = {i: name for i, name in enumerate(self.body_names)
                                if i != self.scene.support.body_id}
        required = set(self.observed_bodies.values())
        if len(set(self.body_names)) != len(self.body_names) or not required <= set(obs["bodies"]):
            raise ValueError(f"Evaluation is missing model bodies: {required - set(obs['bodies'])}")
        self.body_ids = {name: i for i, name in enumerate(self.body_names)}
        self.body_com = self.model.body_com.numpy()[:self.bodies_per_world]
        self.shape_by_body = {}
        shape_bodies = self.model.shape_body.numpy()[:self.contact.shapes_per_world]
        roles = list(zip(self.contact.layout.shape_hand_ids, self.contact.layout.shape_link_ids))
        for shape, body in enumerate(shape_bodies):
            if body < 0:
                continue
            name = self.body_names[body]
            previous = self.shape_by_body.setdefault(name, shape)
            if roles[shape] != roles[previous]:
                raise ValueError(f"Body-level contacts cannot represent different shape roles: {name}")
        with np.load(prepared / "robot_reference.npz", allow_pickle=False) as data:
            self.initial = dict(zip(data["joint_names"].tolist(), data["joint_q"][0].tolist()))
        self.last_tick = {identity: -4 for identity in self.ids}
        self.finished = set()

    def start(self):
        return {"type": "start", "task_id": self.hello["task_id"],
                "binding": {"bundle_id": self.binding["human_bundle_id"],
                            "robot_manifest_sha256": self.binding["robot_manifest_sha256"],
                            "method": "flashchord", "controller": "flashchord.h2r.evaluation"},
                "requirements": {"bodies": list(self.observed_bodies.values()), "contacts": True, "control_dt_s": 1 / 30},
                "initial_robot": {identity: self.initial for identity in self.ids}}

    def update(self, frames):
        """Populate native state and contacts exclusively from evaluation observations."""
        by_id = {frame["episode_id"]: frame for frame in frames}
        if len(by_id) != len(frames) or set(by_id) != set(self.ids):
            raise ValueError("Frame episode IDs differ")
        q = self.state.joint_q.numpy().reshape(self.count, -1)
        dq = self.state.joint_qd.numpy().reshape(self.count, -1)
        body_q = self.state.body_q.numpy().reshape(self.count, -1, 7)
        body_dq = self.state.body_qd.numpy().reshape(self.count, -1, 6)
        phase = self.timestep.numpy()
        contact_rows = []
        active = []
        for world, identity in enumerate(self.ids):
            frame = by_id[identity]
            if frame["done"]:
                self.finished.add(identity)
                continue
            tick = frame["tick"]
            if (identity in self.finished or type(tick) is not int
                    or tick != self.last_tick[identity] + 4 or tick >= self.hello["end_tick"]):
                raise ValueError("Stale, terminal or out-of-order control tick")
            if not np.isclose(frame["time_s"], tick * self.hello["dt_s"], rtol=0, atol=1e-12):
                raise ValueError("Frame time differs from tick")
            self.last_tick[identity] = tick
            phase[world] = tick // 4
            q[world, self.q_ids] = finite_array(frame["robot_q"], (len(self.names),), "robot_q")[self.columns]
            dq[world, self.dof_ids] = finite_array(frame["robot_dq"], (len(self.names),), "robot_dq")[self.columns]
            for body, name in self.observed_bodies.items():
                value = finite_array(frame["bodies"][name], (13,), name)
                if not np.isclose(np.linalg.norm(value[3:7]), 1, atol=1e-5):
                    raise ValueError(f"Non-unit quaternion for {name}")
                body_q[world, body] = value[[0, 1, 2, 4, 5, 6, 3]]
                body_dq[world, body] = value[7:13]
                # Newton FK publishes COM velocity. Preserve the checkpoint's
                # native observation semantics, including its wrist-frame handling.
                offset = quat_rotate_xyzw(value[[4, 5, 6, 3]], self.body_com[body])
                body_dq[world, body, :3] += np.cross(value[10:13], offset)
            contacts = frame["contacts"]
            if not isinstance(contacts, dict) or contacts["solve_tick"] != (tick - 1 if tick else None):
                raise ValueError("Missing or mistimed real contact observation")
            for contact in contacts["records"]:
                a, b = contact["body1"], contact["body2"]
                if a not in self.shape_by_body or b not in self.shape_by_body:
                    raise ValueError(f"Unknown contact body: {a}/{b}")
                shape_offset = world * self.contact.shapes_per_world
                contact_rows.append((self.shape_by_body[a] + shape_offset, self.shape_by_body[b] + shape_offset,
                                     finite_array(contact["point1_local"], (3,), "point1"),
                                     finite_array(contact["point2_local"], (3,), "point2"),
                                     -finite_array(contact["force_on_body2_world"], (3,), "force")))
            active.append((world, identity, tick))
        self.state.joint_q.assign(q.ravel())
        self.state.joint_qd.assign(dq.ravel())
        self.state.body_q.assign(body_q.reshape(-1, 7))
        self.state.body_qd.assign(body_dq.reshape(-1, 6))
        self.timestep.assign(phase)
        self.episode_step.assign(phase)
        n = len(contact_rows)
        shapes = np.zeros((max(1, n), 2), dtype=np.int32)
        points = np.zeros((max(1, n), 2, 3), dtype=np.float32)
        forces = np.zeros((max(1, n), 6), dtype=np.float32)
        for i, (a, b, p1, p2, force) in enumerate(contact_rows):
            shapes[i], points[i], forces[i, :3] = (a, b), (p1, p2), force
        def array(values, dtype):
            return wp.array(values, dtype=dtype, device=self.device)
        contacts = SimpleNamespace(
            rigid_contact_count=array([n], wp.int32), rigid_contact_max=max(1, n),
            rigid_contact_shape0=array(shapes[:, 0], wp.int32),
            rigid_contact_shape1=array(shapes[:, 1], wp.int32),
            rigid_contact_point0=array(points[:, 0], wp.vec3),
            rigid_contact_point1=array(points[:, 1], wp.vec3), force=array(forces, wp.spatial_vector))
        with wp.ScopedDevice(self.device):
            self.contact.update(self.model, self.state, contacts)
        return active

    def step(self, frames):
        active = self.update(frames)
        if not active:
            return {"type": "step", "commands": {}}, {}
        with wp.ScopedDevice(self.device):
            observation = self.observation.compute(self.state).numpy().reshape(self.count, -1)
            raw = self.infer(observation)
            self.action.process(wp.array(raw.ravel(), dtype=wp.float32, device=self.device),
                                self.timestep, self.state)
        targets = self.action.joint_target.numpy().reshape(self.count, -1)[:, self.dof_ids]
        if not np.isfinite(observation).all() or not np.isfinite(targets).all():
            raise ValueError("Non-finite policy observation or target")
        commands = {identity: {"tick": tick, "joint_targets": dict(zip(self.names, targets[world].tolist()))}
                    for world, identity, tick in active}
        diagnostics = {"observation": observation.tolist(), "raw_action": raw.tolist(),
                       "processed_action": self.action.processed_target.numpy().reshape(self.count, -1).tolist()}
        return {"type": "step", "commands": commands}, diagnostics

    def infer(self, observation):
        result = []
        for row in observation:
            value = jax.device_put(np.asarray(row, dtype=np.float32)[None], self.policy_device)
            action, _ = self.policy(self.actor, value, self.key)
            result.append(np.asarray(action)[0])
        return np.asarray(result)
