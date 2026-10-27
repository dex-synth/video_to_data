# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-tracking termination strategies and their shared runtime protocols."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import warp as wp

from flash_chord.embodiments.binding import DeviceRobotReference
from flash_chord.embodiments.frames import DeviceBodyFrameMap
from flash_chord.scene.builder import Scene
from flash_chord.utils.quat import quat_geodesic_angle, quat_mul_xyzw


@dataclass(frozen=True)
class ThresholdTerminationTermConfig:
    """Enablement and finite non-negative threshold for one tracking-failure condition."""

    threshold: float
    enabled: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError(f"termination threshold must be finite and non-negative, got {self.threshold}")


@dataclass(frozen=True)
class ReferenceEndTerminationTermConfig:
    """Whether exhausting the reference trajectory truncates the episode."""

    enabled: bool = True


@dataclass(frozen=True)
class TrackingTerminationConfig:
    """Named reference-tracking failures and reference-end truncation."""

    wrist_position: ThresholdTerminationTermConfig = ThresholdTerminationTermConfig(threshold=0.15)
    wrist_orientation: ThresholdTerminationTermConfig = ThresholdTerminationTermConfig(
        threshold=0.0,
        enabled=False,
    )
    object_position: ThresholdTerminationTermConfig = ThresholdTerminationTermConfig(threshold=0.10)
    object_orientation: ThresholdTerminationTermConfig = ThresholdTerminationTermConfig(threshold=0.50)
    reference_end: ReferenceEndTerminationTermConfig = ReferenceEndTerminationTermConfig()

    def build(
        self,
        scene: Scene,
        object_body_ids_w: wp.array,
        object_ref_pos_w: wp.array,
        object_ref_quat_w: wp.array,
        num_objects: int,
        device=None,
        episode_step: wp.array | None = None,
    ) -> Termination:
        """Build the fused reference-tracking termination strategy."""
        return TrackingTermination.build(
            scene,
            config=self,
            object_body_ids_w=object_body_ids_w,
            object_ref_pos_w=object_ref_pos_w,
            object_ref_quat_w=object_ref_quat_w,
            episode_step=episode_step,
            num_objects=num_objects,
            device=device,
        )


@runtime_checkable
class Termination(Protocol):
    """Device termination strategy consumed by :class:`BaseEnv`."""

    terminated: wp.array
    truncated: wp.array

    def evaluate(self, body_q: wp.array, timestep: wp.array) -> None:
        """Write completion flags for the current transition."""
        ...


@runtime_checkable
class TerminationSpec(Protocol):
    """Setup-time termination builder selected by configuration."""

    def build(
        self,
        scene: Scene,
        object_body_ids_w: wp.array,
        object_ref_pos_w: wp.array,
        object_ref_quat_w: wp.array,
        num_objects: int,
        device=None,
        episode_step: wp.array | None = None,
    ) -> Termination:
        """Bind runtime dependencies and return a termination strategy."""
        ...


@runtime_checkable
class TerminationDiagnostics(Protocol):
    """Optional named cause masks and tracking errors for rollout diagnostics."""

    cause_names: tuple[str, ...]
    cause_arrays: tuple[wp.array, ...]
    error_names: tuple[str, ...]
    error_arrays: tuple[wp.array, ...]


@runtime_checkable
class PackedTerminationDiagnostics(Protocol):
    """Optional row-major float diagnostics exposed through one device array.

    Each world occupies one row whose columns follow ``packed_diagnostic_names``.
    """

    packed_diagnostic_names: tuple[str, ...]
    packed_diagnostics: wp.array


_TRACKING_CAUSE_NAMES = ("wrist", "object")
_TRACKING_ERROR_NAMES = (
    "wrist_position_m",
    "wrist_orientation_rad",
    "object_position_m",
    "object_orientation_rad",
)
_TRACKING_DIAGNOSTIC_WIDTH = len(_TRACKING_CAUSE_NAMES) + len(_TRACKING_ERROR_NAMES)


@wp.kernel
def termination_kernel(
    body_q: wp.array(dtype=wp.transform),
    timestep: wp.array(dtype=wp.int32),
    reference_num_frames: int,
    bodies_per_world: int,
    num_hands: int,
    wrist_frame_body_id: wp.array(dtype=wp.int32),  # [num_hands] retained per-world body id
    wrist_body_to_frame_pos: wp.array(dtype=wp.vec3),  # [num_hands] semantic wrist offset
    wrist_body_to_frame_quat: wp.array(dtype=wp.quat),  # [num_hands] semantic wrist rotation
    ref_wrist_pos: wp.array(dtype=wp.vec3),  # [T*num_hands] frame-major
    ref_wrist_quat: wp.array(dtype=wp.quat),  # [T*num_hands] frame-major, xyzw
    num_objects: int,
    object_body_id: wp.array(dtype=wp.int32),  # [W*num_objects] absolute body id, w-major
    ref_object_pos: wp.array(dtype=wp.vec3),  # [W*num_objects]
    ref_object_quat: wp.array(dtype=wp.quat),  # [W*num_objects] xyzw
    enable_wrist_pos: int,
    wrist_pos_thresh: float,
    enable_wrist_ori: int,
    wrist_ori_thresh: float,
    enable_object_pos: int,
    object_pos_thresh: float,
    enable_object_ori: int,
    object_ori_thresh: float,
    diagnostic_width: int,
    wrist_pos_err: wp.array(dtype=wp.float32),  # [W*num_hands] out
    wrist_ori_err: wp.array(dtype=wp.float32),  # out
    object_pos_err: wp.array(dtype=wp.float32),  # [W*num_objects] out
    object_ori_err: wp.array(dtype=wp.float32),  # out
    wrist_done: wp.array(dtype=wp.int32),  # [W] out
    object_done: wp.array(dtype=wp.int32),  # [W] out
    packed_diagnostics: wp.array(dtype=wp.float32),  # [W*6] row-major out: causes, then max errors
) -> None:
    """One thread per world: pose error of each wrist/object vs the reference, and the per-world flags."""
    w = wp.tid()
    base = w * bodies_per_world
    frame = timestep[w]
    if frame < 0:
        frame = 0
    if frame >= reference_num_frames:
        frame = reference_num_frames - 1

    wd = int(0)
    max_wrist_pos_err = float(0.0)
    max_wrist_ori_err = float(0.0)
    for h in range(num_hands):
        i = w * num_hands + h
        reference_i = frame * num_hands + h
        body_xf = body_q[base + wrist_frame_body_id[h]]
        body_quat = wp.transform_get_rotation(body_xf)
        frame_pos = wp.transform_point(body_xf, wrist_body_to_frame_pos[h])
        frame_quat = wp.normalize(quat_mul_xyzw(body_quat, wrist_body_to_frame_quat[h]))
        dp = wp.length(frame_pos - ref_wrist_pos[reference_i])
        do = quat_geodesic_angle(frame_quat, ref_wrist_quat[reference_i])
        wrist_pos_err[i] = dp
        wrist_ori_err[i] = do
        max_wrist_pos_err = wp.max(max_wrist_pos_err, dp)
        max_wrist_ori_err = wp.max(max_wrist_ori_err, do)
        if (enable_wrist_pos != 0 and dp > wrist_pos_thresh) or (enable_wrist_ori != 0 and do > wrist_ori_thresh):
            wd = 1
    wrist_done[w] = wd

    od = int(0)
    max_object_pos_err = float(0.0)
    max_object_ori_err = float(0.0)
    for o in range(num_objects):
        i = w * num_objects + o
        xf = body_q[object_body_id[i]]  # absolute id already carries the world offset
        dp = wp.length(wp.transform_get_translation(xf) - ref_object_pos[i])
        do = quat_geodesic_angle(wp.transform_get_rotation(xf), ref_object_quat[i])
        object_pos_err[i] = dp
        object_ori_err[i] = do
        max_object_pos_err = wp.max(max_object_pos_err, dp)
        max_object_ori_err = wp.max(max_object_ori_err, do)
        if (enable_object_pos != 0 and dp > object_pos_thresh) or (enable_object_ori != 0 and do > object_ori_thresh):
            od = 1
    object_done[w] = od

    diagnostic_base = w * diagnostic_width
    packed_diagnostics[diagnostic_base] = float(wd)
    packed_diagnostics[diagnostic_base + 1] = float(od)
    packed_diagnostics[diagnostic_base + 2] = max_wrist_pos_err
    packed_diagnostics[diagnostic_base + 3] = max_wrist_ori_err
    packed_diagnostics[diagnostic_base + 4] = max_object_pos_err
    packed_diagnostics[diagnostic_base + 5] = max_object_ori_err


@dataclass
class TerminationBuffers:
    """Per-world device outputs and semantic wrist-frame maps, sized to the scene.

    Each wrist frame is represented by one retained body plus a constant body-local
    transform. The kernel therefore remains independent of whether fixed semantic
    links were retained or collapsed during embodiment construction.
    """

    world_count: int
    num_hands: int
    num_objects: int
    bodies_per_world: int
    wrist_frames: DeviceBodyFrameMap
    wrist_pos_err: wp.array  # float32 [W*num_hands]
    wrist_ori_err: wp.array
    object_pos_err: wp.array  # float32 [W*num_objects]
    object_ori_err: wp.array
    wrist_done: wp.array  # int32 [W]
    object_done: wp.array  # int32 [W]
    packed_diagnostics: wp.array  # float32 [W*6], row-major causes followed by max errors

    @classmethod
    def from_scene(
        cls,
        scene: Scene,
        sides: tuple[str, ...] | None = None,
        num_objects: int | None = None,
        device=None,
    ) -> "TerminationBuffers":
        hands = scene.layout.hands if sides is None else tuple(scene.layout.hand(side) for side in sides)
        wrist_frames = [hand.palm_frame for hand in hands]
        n_hands = len(wrist_frames)
        n_obj = len(scene.objects) if num_objects is None else num_objects
        n_worlds = scene.world_count
        bodies_per_world = scene.model.body_count // n_worlds
        f32 = lambda n: wp.zeros(n, dtype=wp.float32, device=device)  # noqa: E731
        return cls(
            world_count=n_worlds,
            num_hands=n_hands,
            num_objects=n_obj,
            bodies_per_world=bodies_per_world,
            wrist_frames=DeviceBodyFrameMap.build(wrist_frames, device=device),
            wrist_pos_err=f32(n_worlds * n_hands),
            wrist_ori_err=f32(n_worlds * n_hands),
            object_pos_err=f32(n_worlds * n_obj),
            object_ori_err=f32(n_worlds * n_obj),
            wrist_done=wp.zeros(n_worlds, dtype=wp.int32, device=device),
            object_done=wp.zeros(n_worlds, dtype=wp.int32, device=device),
            packed_diagnostics=f32(n_worlds * _TRACKING_DIAGNOSTIC_WIDTH),
        )


def evaluate_termination(
    buf: TerminationBuffers,
    body_q: wp.array,
    timestep: wp.array,
    reference: DeviceRobotReference,
    object_body_ids: wp.array,
    ref_object_pos: wp.array,
    ref_object_quat: wp.array,
    config: TrackingTerminationConfig | None = None,
) -> None:
    """Launch the termination kernel into ``buf`` (per-world errors + ``done`` masks on device)."""
    config = config or TrackingTerminationConfig()
    wp.launch(
        termination_kernel,
        dim=buf.world_count,
        inputs=[
            body_q,
            timestep,
            reference.num_frames,
            buf.bodies_per_world,
            buf.num_hands,
            buf.wrist_frames.body_ids,
            buf.wrist_frames.body_to_frame_pos,
            buf.wrist_frames.body_to_frame_quat,
            reference.wrist_pos_w,
            reference.wrist_quat_w,
            buf.num_objects,
            object_body_ids,
            ref_object_pos,
            ref_object_quat,
            int(config.wrist_position.enabled),
            config.wrist_position.threshold,
            int(config.wrist_orientation.enabled),
            config.wrist_orientation.threshold,
            int(config.object_position.enabled),
            config.object_position.threshold,
            int(config.object_orientation.enabled),
            config.object_orientation.threshold,
            _TRACKING_DIAGNOSTIC_WIDTH,
        ],
        outputs=[
            buf.wrist_pos_err,
            buf.wrist_ori_err,
            buf.object_pos_err,
            buf.object_ori_err,
            buf.wrist_done,
            buf.object_done,
            buf.packed_diagnostics,
        ],
    )


@wp.kernel
def publish_tracking_completion(
    wrist_done: wp.array(dtype=wp.int32),
    object_done: wp.array(dtype=wp.int32),
    timestep: wp.array(dtype=wp.int32),
    last_frame: int,
    truncate_at_reference_end: int,
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
) -> None:
    """Aggregate tracking failures and reference exhaustion per world."""
    world = wp.tid()
    terminated[world] = wp.max(wrist_done[world], object_done[world])
    truncated[world] = int(0)
    if truncate_at_reference_end != 0 and timestep[world] >= last_frame:
        truncated[world] = 1


@dataclass
class TrackingTermination:
    """Reference wrist/object tracking termination with bound device targets."""

    buffers: TerminationBuffers
    config: TrackingTerminationConfig
    last_frame: int
    reference: DeviceRobotReference
    object_body_ids_w: wp.array
    object_ref_pos_w: wp.array
    object_ref_quat_w: wp.array
    terminated: wp.array
    truncated: wp.array

    @property
    def cause_names(self) -> tuple[str, ...]:
        return _TRACKING_CAUSE_NAMES

    @property
    def cause_arrays(self) -> tuple[wp.array, ...]:
        return (self.buffers.wrist_done, self.buffers.object_done)

    @property
    def error_names(self) -> tuple[str, ...]:
        return _TRACKING_ERROR_NAMES

    @property
    def error_arrays(self) -> tuple[wp.array, ...]:
        return (
            self.buffers.wrist_pos_err,
            self.buffers.wrist_ori_err,
            self.buffers.object_pos_err,
            self.buffers.object_ori_err,
        )

    @property
    def packed_diagnostic_names(self) -> tuple[str, ...]:
        return self.cause_names + self.error_names

    @property
    def packed_diagnostics(self) -> wp.array:
        return self.buffers.packed_diagnostics

    @classmethod
    def build(
        cls,
        scene: Scene,
        config: TrackingTerminationConfig,
        object_body_ids_w: wp.array,
        object_ref_pos_w: wp.array,
        object_ref_quat_w: wp.array,
        num_objects: int,
        device=None,
        episode_step: wp.array | None = None,
    ) -> "TrackingTermination":
        """Bind tracking inputs and allocate completion outputs."""
        return cls(
            buffers=TerminationBuffers.from_scene(
                scene,
                sides=scene.layout.sides,
                num_objects=num_objects,
                device=device,
            ),
            config=config,
            last_frame=scene.robot_reference.num_frames - 1,
            reference=DeviceRobotReference.build(scene.robot_reference, scene.layout, device=device),
            object_body_ids_w=object_body_ids_w,
            object_ref_pos_w=object_ref_pos_w,
            object_ref_quat_w=object_ref_quat_w,
            terminated=wp.zeros(scene.world_count, dtype=wp.int32, device=device),
            truncated=wp.zeros(scene.world_count, dtype=wp.int32, device=device),
        )

    def evaluate(self, body_q: wp.array, timestep: wp.array) -> None:
        evaluate_termination(
            self.buffers,
            body_q,
            timestep,
            self.reference,
            self.object_body_ids_w,
            self.object_ref_pos_w,
            self.object_ref_quat_w,
            self.config,
        )
        wp.launch(
            publish_tracking_completion,
            dim=self.buffers.world_count,
            inputs=[
                self.buffers.wrist_done,
                self.buffers.object_done,
                timestep,
                self.last_frame,
                int(self.config.reference_end.enabled),
            ],
            outputs=[self.terminated, self.truncated],
        )


@dataclass
class TerminationResult:
    """Host readback of :class:`TerminationBuffers` for the replay verification (peaks + any-done)."""

    max_wrist_pos: float
    max_wrist_ori: float
    max_object_pos: float
    max_object_ori: float
    wrist_done: bool
    object_done: bool

    @property
    def any(self) -> bool:
        return self.wrist_done or self.object_done


def read_termination(buf: TerminationBuffers) -> TerminationResult:
    """Read the device buffers back to the host (verification only — a sync; not for the runtime step)."""
    wpos, wori = buf.wrist_pos_err.numpy(), buf.wrist_ori_err.numpy()
    opos, oori = buf.object_pos_err.numpy(), buf.object_ori_err.numpy()
    peak = lambda a: float(a.max()) if a.size else 0.0  # noqa: E731
    return TerminationResult(
        max_wrist_pos=peak(wpos),
        max_wrist_ori=peak(wori),
        max_object_pos=peak(opos),
        max_object_ori=peak(oori),
        wrist_done=bool(buf.wrist_done.numpy().any()),
        object_done=bool(buf.object_done.numpy().any()),
    )
