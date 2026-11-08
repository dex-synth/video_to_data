# SPDX-License-Identifier: Apache-2.0
"""A native hand-pose reference with explicit H2R asset and time bindings."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from flash_chord.data.mano_sharpa import ManoSharpaReference, load_mano_sharpa
from flash_chord.data.reference import (
    ObjectArticulationSpec,
    ObjectAssetSpec,
    ObjectBodySpec,
    ObjectJointDriveSpec,
    ObjectJointPhysicsSpec,
)

from .inputs import load_binding, member, read_json


@dataclass
class H2rReference(ManoSharpaReference):
    binding_path: Path
    sample_times_s: np.ndarray

    def frame_window(self, start_frame=0, end_frame=-1):
        if start_frame != 0 or end_frame not in (-1, self.num_frames):
            raise ValueError("Create the desired source window in H2R before method preparation")
        return self


def load_h2r(path, control_fps=None, motion_speed=1.0, source_frame_playback=False):
    path = Path(path).resolve()
    binding = load_binding(path)
    if motion_speed != 1.0 or control_fps not in (None, 20.0, 30.0) or source_frame_playback:
        raise ValueError("H2R requires 1x speed, 20 Hz control (or the historical 30 Hz recipe)")
    fps = 30.0 if control_fps is None else float(control_fps)
    # Keep every control instant before the fixed endpoint and one terminal
    # sample. Only that last sample may represent a partial control interval.
    count = int(np.ceil(binding["duration_s"] * fps - 1e-10)) + 1
    times = np.minimum(np.arange(count) / fps, binding["duration_s"])
    scene = read_json(path.parent / "human" / "scene.json")
    if scene.get("dataset") == "oakink2":
        ref = read_json(path.parent / "human" / "reference.json")
        assets = tuple(ObjectAssetSpec(
            part, str(path.parent / "assets" / "object.xml"),
            (ObjectBodySpec(part, "object_" + part),), part, (),
        ) for part in ref["object_geometry"]["part_order"])
    else:
        joint = scene["joint"]
        assets = (ObjectAssetSpec(
            scene["object"],
            str(path.parent / "assets" / "object.xml"),
            (ObjectBodySpec("bottom", "object_bottom"), ObjectBodySpec("top", "object_top")),
            "bottom",
            (ObjectArticulationSpec(
                "object_hinge", 0,
                ObjectJointPhysicsSpec(joint["armature"], joint["friction"]),
                ObjectJointDriveSpec(50.0, 2.0, 50.0),
            ),),
        ),)
    source = load_mano_sharpa(str(path.parent / "processed.parquet"), object_assets=assets)
    if source.fps != 30 or not np.isclose((source.num_frames - 1) / 30, binding["duration_s"], atol=1e-12, rtol=0):
        raise ValueError("Processed reference changed the common duration")
    reference = source if fps == 30 else load_mano_sharpa(
        str(path.parent / "processed.parquet"), control_fps=fps, object_assets=assets, sample_times_s=times)
    reference._object_mesh_paths = [str(member(path.parent, p)) for p in reference.object_mesh_paths()]
    reference._metadata = replace(reference.metadata, source_path=str(path))
    return H2rReference(**vars(reference), binding_path=path, sample_times_s=times)
