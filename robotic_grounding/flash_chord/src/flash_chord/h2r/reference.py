# SPDX-License-Identifier: Apache-2.0
"""A native hand-pose reference bound to one sealed H2R task and its time grid."""

from dataclasses import dataclass, replace
from pathlib import Path

from h2r_io import Task
import numpy as np

from flash_chord.data.mano_sharpa import ManoSharpaReference, load_mano_sharpa
from flash_chord.data.reference import (
    ObjectArticulationSpec,
    ObjectAssetSpec,
    ObjectBodySpec,
    ObjectJointDriveSpec,
    ObjectJointPhysicsSpec,
)

from .inputs import load_prepared

# The native assistance drive of an articulated-object joint (flash_chord.assets.registry); VOC scales it.
JOINT_DRIVE = ObjectJointDriveSpec(kp=50.0, kd=2.0, effort_limit=50.0)


@dataclass
class H2rReference(ManoSharpaReference):
    binding_path: Path
    task: Task
    sample_times_s: np.ndarray

    def frame_window(self, start_frame=0, end_frame=-1):
        if start_frame != 0 or end_frame not in (-1, self.num_frames):
            raise ValueError("Create the desired source window in H2R before method preparation")
        return self


def object_assets(task, scene_xml):
    """Every interaction object of the task as a native asset: parts, free root and joints.

    Articulation columns follow the scene's objects and their joints, as written by preparation.
    """
    compiled = task.compiled["base_scene"]["joints"]
    assets, column = [], 0
    for obj in task.demo.objects.values():
        joints = []
        for joint in obj.joints:
            physics = ObjectJointPhysicsSpec(compiled[joint.name]["armature"], compiled[joint.name]["friction"])
            joints.append(ObjectArticulationSpec(joint.mjcf_name, column, physics, JOINT_DRIVE))
            column += 1
        bodies = tuple(ObjectBodySpec(part, body) for part, body in obj.parts.items())
        assets.append(ObjectAssetSpec(obj.name, str(scene_xml), bodies, obj.root_part, tuple(joints)))
    return tuple(assets)


def load_h2r(path, control_fps=None, motion_speed=1.0, source_frame_playback=False):
    path = Path(path).resolve()
    binding, task = load_prepared(path)
    if motion_speed != 1.0 or control_fps not in (None, 20.0, 30.0) or source_frame_playback:
        raise ValueError("H2R requires 1x speed and 20 Hz control (or the explicit 30 Hz recipe)")
    root = path.parent
    assets = object_assets(task, root / "assets" / "scene.xml")
    source = load_mano_sharpa(str(root / "processed.parquet"), object_assets=assets, sides=task.hand_sides)
    if (source.fps != task.demo.reference_hz
            or not np.isclose((source.num_frames - 1) / source.fps, binding["duration_s"], atol=1e-12, rtol=0)):
        raise ValueError("Processed reference changed the common clock or duration")
    fps = source.fps if control_fps is None else float(control_fps)
    # Keep every control instant before the fixed endpoint and one terminal
    # sample. Only that last sample may represent a partial control interval.
    count = int(np.ceil(binding["duration_s"] * fps - 1e-10)) + 1
    times = np.minimum(np.arange(count) / fps, binding["duration_s"])
    reference = source if fps == source.fps else load_mano_sharpa(
        str(root / "processed.parquet"), control_fps=fps, object_assets=assets, sides=task.hand_sides,
        sample_times_s=times)
    reference._object_mesh_paths = [str((root / p).resolve()) for p in reference.object_mesh_paths()]
    reference._metadata = replace(reference.metadata, source_path=str(path))
    return H2rReference(**vars(reference), binding_path=path, task=task, sample_times_s=times)
