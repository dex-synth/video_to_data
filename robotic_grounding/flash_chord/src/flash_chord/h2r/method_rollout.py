# SPDX-License-Identifier: Apache-2.0
"""Hand a checkpoint's rollout in its training simulator to the benchmark as a method rollout.

Next to its independent evaluation, the benchmark reports how a method does in its own simulator: the
same deployed checkpoint with deterministic actions, the simulator it was trained in, the evaluation
initial state, no assistance and no disturbance. This entry rebuilds the training environment from the
checkpoint's own configuration under the native unassisted evaluation composition
(`flash_chord.evaluation.harness`), runs the native rollout recorder once from reference frame 0 and
additionally keeps `joint_q` after every physics step. Each world is written as one episode of an
`h2r-method-rollout-v1` package, which the benchmark imports and scores as `method-sim`.
"""

import argparse
from dataclasses import fields, replace
from pathlib import Path

from h2r_io.method_rollout import state_names, write_method_rollout
import jax.numpy as jnp
import mujoco_warp
import newton
import numpy as np
import warp as wp
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import (
    apply_evaluation_composition,
    reference_tracking_criteria,
    score_reference_tracking,
)
from flash_chord.evaluation.rollout import record_object_pose_rollout
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference

from .configuration import require_h2r_input
from .controller import METHOD, controller_information, initial_robot  # also fixes the deployment's XLA flags
from .inputs import BINDING
from .reference import load_h2r
from .robot import H2rSharpaHands
from .scene import build_h2r_scene

# How this rollout is run. The benchmark accepts a method rollout only as such.
CONDITIONS = {
    "initial_state": "evaluation",  # explicit reset to frame 0: the proposed robot pose, checked below
    "initial_velocity": "zero",  # the reset zeroes every velocity; there is no settling hold
    "assistance": "none",  # no virtual object control (which also drives object joints), no reset
    "disturbance": "none",  # the training environment has none; a randomized action delay is rejected
    "action": "deterministic",  # the actor's mean action
}
WORLDS = 10  # GPU physics is not repeatable bit for bit: ten worlds from the same initial state


@wp.kernel
def keep_joint_q(
    joint_q: wp.array(dtype=wp.float32),
    row: wp.array(dtype=wp.int32),
    rows: wp.array2d(dtype=wp.float32),
):
    if row[0] < rows.shape[0]:
        rows[row[0], wp.tid()] = joint_q[wp.tid()]


@wp.kernel
def next_row(row: wp.array(dtype=wp.int32)):
    row[0] += 1


class RecordingSolver:
    """The environment's solver, which also keeps `joint_q` of all worlds after every physics step.

    The environment calls `step` once per substep inside its captured control step, so the copies join
    the same device graph and the environment's own step path stays as it is in training.
    """

    def __init__(self, solver, steps, width, device):
        self.solver, self.device = solver, device
        self.joint_q = wp.zeros((steps, width), dtype=wp.float32, device=device)
        self.row = wp.zeros(1, dtype=wp.int32, device=device)

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def step(self, state_in, state_out, control, contacts, dt):
        self.solver.step(state_in, state_out, control, contacts, dt)
        wp.launch(keep_joint_q, dim=self.joint_q.shape[1], device=self.device,
                  inputs=[state_out.joint_q, self.row], outputs=[self.joint_q])
        wp.launch(next_row, dim=1, device=self.device, inputs=[self.row])


def build_environment(checkpoint, prepared, worlds):
    """The checkpoint's training environment under the native unassisted evaluation composition.

    Returns the environment, its reference, the training configuration and the native success criteria.
    """
    binding_path = Path(prepared).resolve() / BINDING
    config = checkpoint_config(read_checkpoint_metadata(checkpoint))
    binding = require_h2r_input(config)  # the preparation this checkpoint was trained on
    criteria = reference_tracking_criteria(config)  # read before the composition disables these terms
    # No VOC, settling hold or auto reset; a tracking failure is scored afterwards instead of ending a world.
    apply_evaluation_composition(config, world_count=worlds)
    config["task"]["parquet"] = config["embodiment"]["binding_path"] = str(binding_path)
    cfg = OmegaConf.create(config)
    require_h2r_input(cfg, expected_binding=binding)
    if cfg.env.action.delay.min_steps or cfg.env.action.delay.max_steps:
        raise ValueError("The deployed controller has no action delay; this checkpoint was trained with one")
    reference = load_h2r(binding_path, control_fps=cfg.env.sim.fps)
    scene = build_h2r_scene(reference, H2rSharpaHands(binding_path), worlds,
                            instantiate_typed(cfg.collision, CollisionPolicy))
    env = RLEnv(scene, reference, config=instantiate_typed(cfg.env, RLEnvConfig))
    return env, reference, instantiate_typed(cfg.training, TrainingConfig), criteria


def run(env, reference, policy, actor, key):
    """Roll the policy out once over the whole reference from the evaluation initial state.

    Returns `joint_q` after every physics step `[N, world, q]` with the initial state first, the native
    control-step rollout, and `[control sample, world]` whether the environment itself had terminated
    the world (an invalid state or a joint far outside its range), which stops its reference clock.

    Raises:
        RuntimeError: The rollout did not keep the declared conditions, or the physics-step states do
            not pass through the natively recorded control steps.
    """
    worlds, substeps = env.world_count, env.config.sim.substeps
    steps = (reference.num_frames - 1) * substeps
    recorder = env.solver = RecordingSolver(env.solver, steps, env.state_0.joint_q.shape[0], env.device)
    env.reset()
    env.capture_step()  # as in training and the native evaluation: one control step is one device graph
    env.reset(frame_id=0)  # explicit: the exact reference pose of frame 0, zero velocity, no hold
    recorder.row.zero_()
    initial = env.state_0.joint_q.numpy()
    assisted = env.reset_policy.applied_voc_scale
    if env.state_0.joint_qd.numpy().any() or assisted.numpy().any():
        raise RuntimeError("The reset left a velocity or virtual object control")
    observation = jnp.array(env.observation.numpy().reshape(worlds, env.observation_dim))
    terminated = []

    def act(actor, observation, key):
        terminated.append(env.done.numpy() != 0)  # as left by the previous control step
        return policy(actor, observation, key)

    rollout = record_object_pose_rollout(env, WarpRLEnv(env), act, actor, key, observation,
                                         reference=reference, step_count=reference.num_frames)
    terminated.append(env.done.numpy() != 0)
    if recorder.row.numpy()[0] != steps or assisted.numpy().any():
        raise RuntimeError("The rollout did not run every physics step once, without virtual object control")
    joint_q = np.concatenate([initial[None], recorder.joint_q.numpy()]).reshape(steps + 1, worlds, -1)
    finite = [world for world in range(worlds) if world not in rollout.non_finite_worlds]
    for obj in env.scene.objects:  # a free root's coordinates are its body pose, both xyzw
        start, body = obj.root.free_q_ids[0], obj.root.reference_body_id
        if not np.allclose(joint_q[::substeps, finite, start:start + 7], rollout.achieved_pose_w[:, finite, body],
                           rtol=0, atol=1e-5):
            raise RuntimeError("The physics-step states differ from the native control-step record")
    return joint_q, rollout, np.array(terminated)


def native_outcome(rollout, criteria, world, frames):
    """FlashCHORD's reference-tracking outcome of one world over its first `frames` control samples."""
    completion = replace(rollout.completion, **{
        field.name: getattr(rollout.completion, field.name)[[world]] for field in fields(rollout.completion)})
    single = replace(rollout, tracking_error=rollout.tracking_error[:frames, [world]], completion=completion,
                     non_finite_worlds=(0,) if world in rollout.non_finite_worlds else ())
    return score_reference_tracking(single, criteria)


def export(checkpoint, prepared, output, worlds=WORLDS):
    """Write the method rollout of one checkpoint: every world is one trajectory.

    A world the environment itself terminated ends there (`ended_at_s`); its later states followed a
    stopped reference clock and are not part of the task.

    Raises:
        ValueError: The checkpoint was not trained on this preparation or cannot be deployed as it is.
        RuntimeError: The rollout did not start from the proposed robot pose; see also `run`.
    """
    env, reference, training, criteria = build_environment(checkpoint, prepared, worlds)
    actor, policy, key, _ = load_policy_for_inference(
        str(checkpoint), training, jnp.zeros((worlds, env.observation_dim), dtype=jnp.float32),
        env.action.action_dim, deterministic=True)
    joint_q, rollout, terminated = run(env, reference, policy, actor, key)
    sim, task = env.config.sim, reference.task
    physics_dt, names = 1 / sim.physics_fps, state_names(task)
    scalar = env.scene.layout.scalar_joints
    robot = [scalar.q_ids[scalar.names.index(name)] for name in names["joint_names"]]
    roots = {obj.name: obj.root.free_q_ids for obj in env.scene.objects}
    joints = sorted((joint for obj in env.scene.objects for joint in obj.articulations),
                    key=lambda joint: joint.reference_id)  # the reference columns follow the task's joints
    if len(joints) != len(names["object_joints"]):
        raise ValueError("The scene's object joints differ from the task")
    proposed = initial_robot(reference.binding_path.parent)  # double precision; the simulator holds float32
    if not np.allclose(joint_q[0][:, robot], [proposed[name] for name in names["joint_names"]], rtol=0, atol=1e-6):
        raise RuntimeError("The rollout did not start from the robot pose proposed to the evaluation")
    episodes = []
    for world in range(worlds):
        count = len(joint_q)
        if terminated[:, world].any():
            count = int(np.argmax(terminated[:, world])) * sim.substeps + 1
        finite = np.isfinite(joint_q[:count, world]).all(axis=1)
        if not finite.all():
            count = int(np.argmin(finite))
        ended, frames = count < len(joint_q), (count - 1) // sim.substeps + 1
        native = native_outcome(rollout, criteria, world, frames)
        success = native.success_rate == 1.0 and not ended
        causes = [name for name, fraction in native.failure_cause_fraction.items() if fraction]
        reasons = ["terminated by the training environment (invalid state or joint limit)"] * ended + [
            f"{name} tracking beyond the training termination threshold" for name in causes]
        state = joint_q[:count, world]
        states = {
            "time_s": np.arange(count) * physics_dt,
            "robot_q": state[:, robot],
            # Newton stores xyzw; the package holds wxyz.
            "object_root_poses": np.stack([state[:, [ids[i] for i in (0, 1, 2, 6, 3, 4, 5)]]
                                           for ids in (roots[name] for name in names["object_roots"])], axis=1),
            "object_q": state[:, [joint.q_id for joint in joints]],
        }
        verdict = {
            "success": bool(success),
            "reason": None if success else "; ".join(reasons) or "reference end not reached",
            "ended_at_s": (count - 1) * physics_dt if ended else None,
            "metrics": {f"max_{name}": float(rollout.tracking_error[:frames, world, column].max())
                        for column, name in enumerate(rollout.tracking_error_names)},
        }
        episodes.append({"states": states, "verdict": verdict})
    limits = ", ".join(f"{name} {limit:g}" for name, limit in zip(criteria.error_names, criteria.thresholds)
                       if limit is not None)
    return write_method_rollout(
        output,
        task,
        method=METHOD,
        controller=controller_information(checkpoint),  # as reported to the evaluation (controller.py)
        training_seed=training.seed,
        simulator={
            "name": "newton",
            "version": f"{newton.__version__} (mujoco_warp {mujoco_warp.__version__}, warp {wp.__version__})",
            "physics_dt_s": physics_dt,
            "control_dt_s": 1 / sim.fps,
            "device": env.device.name,
            "deterministic": False,
        },
        conditions=CONDITIONS,
        criteria=(
            "FlashCHORD's reference-tracking success of one world (`chord_sr` of flash_chord.evaluation): it "
            "reaches the end of the reference with finite states and, at every control step, its largest "
            f"tracking errors stay within the training termination thresholds ({limits})."
        ),
        episodes=episodes,
    )


def main():
    """Export the method rollout of one checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "prepared", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.checkpoint, args.prepared, args.output))


if __name__ == "__main__":
    main()
