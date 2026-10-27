# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched object-pose recording for one deterministic evaluation cohort.

Per-step state accumulates into preallocated device buffers and is read back once at the end.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.evaluation.metrics import ADD_VERTEX_COUNT, ADD_VERTEX_SEED
from flash_chord.runtime.command import ObjectLayout
from flash_chord.utils.quat import quat_rotate_xyzw, wxyz_to_xyzw

_POSE_WIDTH = 7
_TRAJECTORY_SCHEMA = "flash_chord_object_trajectories_v1"


@dataclass(frozen=True, slots=True)
class CompletionOutcome:
    """Per-world episode outcome latched at each world's first completion."""

    completion_step: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    reference_progress: np.ndarray


@dataclass(frozen=True, slots=True)
class ObjectPoseRollout:
    """One recorded cohort: object poses, tracking errors, and completion outcome.

    Poses are position followed by an **xyzw** quaternion, ordered by reference body id.
    """

    achieved_pose_w: np.ndarray
    reference_pose_w: np.ndarray
    tracking_error: np.ndarray
    tracking_error_names: tuple[str, ...]
    object_vertices_o: np.ndarray
    body_object_ids: np.ndarray
    object_body_names: tuple[str, ...]
    completion: CompletionOutcome
    control_fps: float
    start_frame: int
    non_finite_worlds: tuple[int, ...]


def _object_trajectory_schema(rollout: ObjectPoseRollout, provenance: Mapping[str, object] | None = None):
    import pyarrow as pa

    steps, worlds, bodies, width = rollout.achieved_pose_w.shape
    metadata = {
        b"flash_chord.schema": _TRAJECTORY_SCHEMA.encode(),
        b"flash_chord.position_unit": b"meter",
        b"flash_chord.quaternion_order": b"xyzw",
        b"flash_chord.row_order": b"step,environment_id,object_body_id",
        b"flash_chord.control_fps": str(rollout.control_fps).encode(),
        b"flash_chord.start_frame": str(rollout.start_frame).encode(),
        b"flash_chord.step_count": str(steps).encode(),
        b"flash_chord.world_count": str(worlds).encode(),
        b"flash_chord.object_body_count": str(bodies).encode(),
        b"flash_chord.pose_width": str(width).encode(),
    }
    for key, value in (provenance or {}).items():
        if value is None or str(value) == "":
            continue
        metadata[f"flash_chord.{key}".encode()] = str(value).encode()
    return pa.schema(
        [
            pa.field("environment_id", pa.int32(), nullable=False),
            pa.field("step", pa.int32(), nullable=False),
            pa.field("reference_frame", pa.int32(), nullable=False),
            pa.field("time_s", pa.float64(), nullable=False),
            pa.field("object_id", pa.int32(), nullable=False),
            pa.field("object_body_id", pa.int32(), nullable=False),
            pa.field("object_body_name", pa.dictionary(pa.int32(), pa.string()), nullable=False),
            pa.field("environment_valid", pa.bool_(), nullable=False),
            pa.field("achieved_position_x", pa.float32(), nullable=False),
            pa.field("achieved_position_y", pa.float32(), nullable=False),
            pa.field("achieved_position_z", pa.float32(), nullable=False),
            pa.field("achieved_quaternion_x", pa.float32(), nullable=False),
            pa.field("achieved_quaternion_y", pa.float32(), nullable=False),
            pa.field("achieved_quaternion_z", pa.float32(), nullable=False),
            pa.field("achieved_quaternion_w", pa.float32(), nullable=False),
            pa.field("reference_position_x", pa.float32(), nullable=False),
            pa.field("reference_position_y", pa.float32(), nullable=False),
            pa.field("reference_position_z", pa.float32(), nullable=False),
            pa.field("reference_quaternion_x", pa.float32(), nullable=False),
            pa.field("reference_quaternion_y", pa.float32(), nullable=False),
            pa.field("reference_quaternion_z", pa.float32(), nullable=False),
            pa.field("reference_quaternion_w", pa.float32(), nullable=False),
        ],
        metadata=metadata,
    )


def _validate_object_trajectory_rollout(rollout: ObjectPoseRollout) -> tuple[int, int, int]:
    achieved = np.asarray(rollout.achieved_pose_w)
    reference = np.asarray(rollout.reference_pose_w)
    if achieved.ndim != 4 or achieved.shape[-1] != _POSE_WIDTH:
        raise ValueError(f"achieved poses must have shape [step, world, body, 7], got {achieved.shape}")
    steps, worlds, bodies, _ = achieved.shape
    if reference.shape != (steps, bodies, _POSE_WIDTH):
        raise ValueError(f"reference poses must have shape {(steps, bodies, _POSE_WIDTH)}, got {reference.shape}")
    if np.asarray(rollout.body_object_ids).shape != (bodies,):
        raise ValueError(
            f"body_object_ids must have shape {(bodies,)}, got {np.asarray(rollout.body_object_ids).shape}"
        )
    if len(rollout.object_body_names) != bodies:
        raise ValueError(f"object_body_names must contain {bodies} names, got {len(rollout.object_body_names)}")
    if rollout.control_fps <= 0.0:
        raise ValueError(f"control_fps must be positive, got {rollout.control_fps}")
    invalid = np.asarray(rollout.non_finite_worlds, dtype=np.int64)
    if invalid.size and (invalid.min() < 0 or invalid.max() >= worlds):
        raise ValueError(f"non_finite_worlds must be within [0, {worlds}), got {rollout.non_finite_worlds}")
    return steps, worlds, bodies


def write_object_trajectories_parquet(
    path: str | Path,
    rollout: ObjectPoseRollout,
    *,
    step_chunk: int = 32,
    provenance: Mapping[str, object] | None = None,
) -> Path:
    """Atomically write step-major object trajectories with Parquet provenance.

    Poses use metres and xyzw quaternions; invalid worlds remain flagged.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if step_chunk <= 0:
        raise ValueError(f"step_chunk must be positive, got {step_chunk}")
    steps, worlds, bodies = _validate_object_trajectory_rollout(rollout)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".parquet", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    schema = _object_trajectory_schema(rollout, provenance)
    body_ids = np.arange(bodies, dtype=np.int32)
    world_ids = np.arange(worlds, dtype=np.int32)
    object_ids = np.asarray(rollout.body_object_ids, dtype=np.int32)
    valid_worlds = np.ones(worlds, dtype=np.bool_)
    if rollout.non_finite_worlds:
        valid_worlds[np.asarray(rollout.non_finite_worlds, dtype=np.int64)] = False
    pose_names = (
        "position_x",
        "position_y",
        "position_z",
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "quaternion_w",
    )
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd", use_dictionary=True) as writer:
            for begin in range(0, steps, step_chunk):
                end = min(begin + step_chunk, steps)
                chunk_steps = end - begin
                row_steps = np.repeat(np.arange(begin, end, dtype=np.int32), worlds * bodies)
                row_worlds = np.tile(np.repeat(world_ids, bodies), chunk_steps)
                row_bodies = np.tile(body_ids, chunk_steps * worlds)
                row_object_ids = np.tile(object_ids, chunk_steps * worlds)
                name_indices = pa.array(row_bodies, type=pa.int32())
                names = pa.DictionaryArray.from_arrays(name_indices, pa.array(rollout.object_body_names))
                achieved = np.asarray(rollout.achieved_pose_w[begin:end], dtype=np.float32).reshape(-1, _POSE_WIDTH)
                reference = np.broadcast_to(
                    np.asarray(rollout.reference_pose_w[begin:end], dtype=np.float32)[:, None, :, :],
                    (chunk_steps, worlds, bodies, _POSE_WIDTH),
                ).reshape(-1, _POSE_WIDTH)
                columns = {
                    "environment_id": pa.array(row_worlds),
                    "step": pa.array(row_steps),
                    "reference_frame": pa.array(row_steps + rollout.start_frame),
                    "time_s": pa.array(row_steps.astype(np.float64) / rollout.control_fps),
                    "object_id": pa.array(row_object_ids),
                    "object_body_id": pa.array(row_bodies),
                    "object_body_name": names,
                    "environment_valid": pa.array(valid_worlds[row_worlds]),
                }
                for index, name in enumerate(pose_names):
                    columns[f"achieved_{name}"] = pa.array(achieved[:, index])
                    columns[f"reference_{name}"] = pa.array(reference[:, index])
                writer.write_table(pa.Table.from_pydict(columns, schema=schema))
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@wp.kernel
def record_object_pose(
    body_q: wp.array(dtype=wp.transform),
    object_body_ids_w: wp.array(dtype=wp.int32),  # [W*B] absolute body ids, reference-body order
    step: int,
    slot_count: int,  # W*B
    pose_out: wp.array(dtype=wp.float32),  # [T*W*B*7] out, position then xyzw quaternion
) -> None:
    """Gather every object body's world transform into one step slice."""
    slot = wp.tid()
    transform = body_q[object_body_ids_w[slot]]
    position = wp.transform_get_translation(transform)
    rotation = wp.transform_get_rotation(transform)
    base = (step * slot_count + slot) * 7
    pose_out[base + 0] = position[0]
    pose_out[base + 1] = position[1]
    pose_out[base + 2] = position[2]
    pose_out[base + 3] = rotation[0]
    pose_out[base + 4] = rotation[1]
    pose_out[base + 5] = rotation[2]
    pose_out[base + 6] = rotation[3]


@wp.kernel
def record_tracking_error(
    packed_diagnostics: wp.array(dtype=wp.float32),  # [W*diagnostic_width], causes then max errors
    diagnostic_width: int,
    cause_count: int,
    error_count: int,
    step: int,
    world_count: int,
    error_out: wp.array(dtype=wp.float32),  # [T*W*error_count] out
) -> None:
    """Copy the per-world maximum tracking errors into one step slice."""
    world = wp.tid()
    source = world * diagnostic_width + cause_count
    base = (step * world_count + world) * error_count
    for term in range(error_count):
        error_out[base + term] = packed_diagnostics[source + term]


@wp.kernel
def latch_completion(
    done: wp.array(dtype=wp.int32),
    truncation: wp.array(dtype=wp.int32),
    reference_frame: wp.array(dtype=wp.int32),
    last_reference_frame: int,
    step: int,
    completion_step: wp.array(dtype=wp.int32),  # out, preset to -1
    completed_terminated: wp.array(dtype=wp.int32),
    completed_truncated: wp.array(dtype=wp.int32),
    completed_progress: wp.array(dtype=wp.float32),
) -> None:
    """Record each world's first completion without ending its rollout."""
    world = wp.tid()
    if completion_step[world] >= 0:
        return
    if done[world] != 0 or truncation[world] != 0:
        completion_step[world] = step
        completed_terminated[world] = done[world]
        completed_truncated[world] = truncation[world]
        completed_progress[world] = float(reference_frame[world]) / float(wp.max(last_reference_frame, 1))


def body_object_ids(layout: ObjectLayout) -> np.ndarray:
    """``(B,)`` owning object index per reference body, joined through the simulation body id."""
    reference_of_simulation = {body_id: reference for reference, body_id in enumerate(layout.body_ids)}
    ids = np.full(layout.num_bodies, -1, dtype=np.int64)
    offsets = layout.voc_object_body_offsets
    for object_index in range(layout.num_objects):
        for slot in range(offsets[object_index], offsets[object_index + 1]):
            ids[reference_of_simulation[layout.voc_object_body_ids[slot]]] = object_index
    if np.any(ids < 0):
        raise ValueError("object layout does not assign every reference body to an object")
    return ids


def reference_object_pose(reference: Reference, start_frame: int, step_count: int) -> np.ndarray:
    """``(T, B, 7)`` reference object poses, position then **xyzw** quaternion."""
    position = np.asarray(reference.object_body_pos_w(), dtype=np.float64)
    rotation = wxyz_to_xyzw(np.asarray(reference.object_body_quat_w(), dtype=np.float64))
    stop = start_frame + step_count
    if stop > position.shape[0]:
        raise ValueError(f"reference has {position.shape[0]} frames; rollout needs {stop}")
    return np.concatenate([position[start_frame:stop], rotation[start_frame:stop]], axis=-1)


def sample_object_vertices(
    scene,
    *,
    count: int = ADD_VERTEX_COUNT,
    seed: int = ADD_VERTEX_SEED,
) -> np.ndarray:
    """``(B, count, 3)`` body-frame vertices sampled from the shapes the simulation itself uses.

    Ordered by reference body id. Geometry is read from the finalized model with the scale and
    shape transform already applied, so it carries the same units and frame as ``body_q``.
    """
    import newton

    model = scene.model
    visible = int(newton.ShapeFlags.VISIBLE)
    mesh_type = int(newton.GeoType.MESH)
    flags = np.asarray(model.shape_flags.numpy())
    shape_body = np.asarray(model.shape_body.numpy())
    shape_type = np.asarray(model.shape_type.numpy())
    shape_scale = np.asarray(model.shape_scale.numpy(), dtype=np.float64)
    shape_transform = np.asarray(model.shape_transform.numpy(), dtype=np.float64)

    by_reference: dict[int, np.ndarray] = {}
    for binding in scene.objects:
        for body in binding.bodies:
            pieces = []
            for shape in range(binding.shapes.start, binding.shapes.stop):
                if shape_body[shape] != body.body_id or shape_type[shape] != mesh_type:
                    continue
                if not flags[shape] & visible:
                    continue
                source = model.shape_source[shape]
                if source is None or not hasattr(source, "vertices"):
                    continue
                scaled = np.asarray(source.vertices, dtype=np.float64) * shape_scale[shape][None, :]
                offset = shape_transform[shape]
                pieces.append(quat_rotate_xyzw(offset[3:7], scaled) + offset[:3][None, :])
            if not pieces:
                raise ValueError(f"object {binding.name!r} body {body.body_id} has no visible mesh shape")
            by_reference[body.reference_body_id] = np.concatenate(pieces)

    missing = set(range(len(by_reference))) - set(by_reference)
    if missing or not by_reference:
        raise ValueError(f"object bodies do not cover reference ids 0..{len(by_reference) - 1}")
    generator = np.random.default_rng(seed)
    sampled = np.empty((len(by_reference), count, 3), dtype=np.float64)
    for reference_id in sorted(by_reference):
        vertices = by_reference[reference_id]
        chosen = generator.choice(vertices.shape[0], size=count, replace=vertices.shape[0] < count)
        sampled[reference_id] = vertices[chosen]
    return sampled


def record_object_pose_rollout(
    env,
    jax_env,
    policy_action: Callable,
    actor_state,
    key,
    observation,
    *,
    reference: Reference,
    step_count: int,
    start_frame: int = 0,
    progress_interval: int = 50,
) -> ObjectPoseRollout:
    """Run one deterministic cohort and return its recorded poses, errors, and outcome.

    ``observation`` is the caller's post-reset observation. The tracking-failure termination
    terms must be disabled so every world keeps advancing through the whole reference.
    """
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    if start_frame + step_count > reference.num_frames:
        raise ValueError("requested rollout extends beyond the reference")
    if not np.all(env.timestep.numpy() == start_frame):
        raise ValueError("rollout initial reference frame differs from start_frame")
    termination = env.termination
    cause_names = tuple(termination.cause_names)
    error_names = tuple(termination.error_names)
    diagnostic_names = tuple(termination.packed_diagnostic_names)
    if diagnostic_names != cause_names + error_names:
        raise ValueError(
            f"termination publishes {diagnostic_names}; expected causes {cause_names} followed by errors {error_names}"
        )
    error_count = len(error_names)
    if not error_count:
        raise ValueError("evaluation requires at least one tracking-error diagnostic")

    world_count = env.world_count
    body_ids_w = env.command.body_ids_w
    layout = env.command.layout
    bodies = layout.num_bodies
    slot_count = world_count * bodies
    device = env.device

    pose = wp.zeros(step_count * slot_count * _POSE_WIDTH, dtype=wp.float32, device=device)
    error = wp.zeros(step_count * world_count * error_count, dtype=wp.float32, device=device)
    completion_step = wp.full(world_count, -1, dtype=wp.int32, device=device)
    completed_terminated = wp.zeros(world_count, dtype=wp.int32, device=device)
    completed_truncated = wp.zeros(world_count, dtype=wp.int32, device=device)
    completed_progress = wp.zeros(world_count, dtype=wp.float32, device=device)

    for step in range(step_count):
        # N reference samples span N-1 control intervals. Record the initial
        # state at t=0, then each post-step state at its actual reference time.
        if step:
            action, key = policy_action(actor_state, observation, key)
            observation = jax_env.step(action).observation
        # Training diagnostics compare the transition to its held command.
        # Evaluation measures the recorded state against its current timestamp.
        env.command.gather(env.timestep)
        termination.evaluate(env.state_0.body_q, env.timestep)
        # Explicit device: a launch off the arrays' device is silently skipped, not raised.
        wp.launch(
            record_object_pose,
            dim=slot_count,
            device=device,
            inputs=[env.state_0.body_q, body_ids_w, step, slot_count],
            outputs=[pose],
        )
        wp.launch(
            record_tracking_error,
            dim=world_count,
            device=device,
            inputs=[
                termination.packed_diagnostics,
                len(diagnostic_names),
                len(cause_names),
                error_count,
                step,
                world_count,
            ],
            outputs=[error],
        )
        wp.launch(
            latch_completion,
            dim=world_count,
            device=device,
            inputs=[env.terminated, env.truncated, env.timestep, reference.num_frames - 1, step],
            outputs=[completion_step, completed_terminated, completed_truncated, completed_progress],
        )
        if progress_interval and (step % progress_interval == 0 or step + 1 == step_count):
            print(f"evaluation step {step + 1}/{step_count}", flush=True)

    achieved = pose.numpy().reshape(step_count, world_count, bodies, _POSE_WIDTH).astype(np.float64)
    finite = np.isfinite(achieved).all(axis=(0, 2, 3))
    non_finite = tuple(int(world) for world in np.flatnonzero(~finite))
    if non_finite:
        achieved[:, ~finite] = 0.0
        print(f"warning: {len(non_finite)} worlds recorded non-finite poses and are excluded", flush=True)
    completion = CompletionOutcome(
        completion_step=completion_step.numpy().astype(np.int32),
        terminated=completed_terminated.numpy().astype(np.bool_),
        truncated=completed_truncated.numpy().astype(np.bool_),
        reference_progress=completed_progress.numpy().astype(np.float64),
    )
    return ObjectPoseRollout(
        achieved_pose_w=achieved,
        reference_pose_w=reference_object_pose(reference, start_frame, step_count),
        tracking_error=error.numpy().reshape(step_count, world_count, error_count).astype(np.float64),
        tracking_error_names=error_names,
        object_vertices_o=sample_object_vertices(env.scene),
        body_object_ids=body_object_ids(layout),
        object_body_names=tuple(reference.object_body_names()),
        completion=completion,
        control_fps=float(env.config.sim.fps),
        start_frame=start_frame,
        non_finite_worlds=non_finite,
    )
