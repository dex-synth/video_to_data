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

    def frame_window(self, start_frame=0, end_frame=-1):
        if start_frame != 0 or end_frame not in (-1, self.num_frames):
            raise ValueError("Create the desired source window in H2R before method preparation")
        return self


def load_h2r(path, control_fps=None, motion_speed=1.0, source_frame_playback=False):
    path = Path(path).resolve()
    binding = load_binding(path)
    if motion_speed != 1.0 or control_fps not in (None, 30.0) or source_frame_playback:
        raise ValueError("H2R Sharpa input requires 1x speed and 30 Hz; no implicit resampling")
    scene = read_json(path.parent / "human" / "scene.json")
    joint = scene["joint"]
    asset = ObjectAssetSpec(
        scene["object"],
        str(path.parent / "assets" / "object.xml"),
        (ObjectBodySpec("bottom", "object_bottom"), ObjectBodySpec("top", "object_top")),
        "bottom",
        (
            ObjectArticulationSpec(
                "object_hinge",
                0,
                ObjectJointPhysicsSpec(joint["armature"], joint["friction"]),
                ObjectJointDriveSpec(50.0, 2.0, 50.0),
            ),
        ),
    )
    reference = load_mano_sharpa(str(path.parent / "processed.parquet"), object_assets=(asset,))
    reference._object_mesh_paths = [str(member(path.parent, p)) for p in reference.object_mesh_paths()]
    if not np.isclose((reference.num_frames - 1) / 30, binding["duration_s"], atol=1e-12, rtol=0):
        raise ValueError("Processed reference changed the common duration")
    reference._metadata = replace(reference.metadata, source_path=str(path))
    return H2rReference(**vars(reference), binding_path=path)
