# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add a sequence's fixed support surfaces from its support USDA.

The support USDA holds inline ``Cylinder`` / ``Cube`` prims with authored transforms and
dimensions. They are attached to the world without dynamic degrees of freedom. Newton
cylinders extend along Z, matching the USDA's ``axis = "Z"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import newton

from flash_chord.scene.collision import ShapeSpan


@dataclass(frozen=True)
class SupportBinding:
    """Resolved fixed body and shapes owned by one support component."""

    body_id: int
    shapes: ShapeSpan

    def __post_init__(self) -> None:
        if self.body_id < 0:
            raise ValueError("support body ID must be nonnegative")
        if not len(self.shapes):
            raise ValueError("support must own at least one shape")


def add_support_surfaces(builder: newton.ModelBuilder, support_usda_path) -> SupportBinding:
    """Add the support USDA's prims on one world-fixed body.

    Returns the exact component topology. An explicitly requested missing/invalid file fails.
    """
    path = Path(support_usda_path)
    if not path.is_file():
        raise FileNotFoundError(f"support surface does not exist: {path}")

    import warp as wp
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"failed to open support surface: {path}")
    shapes = []
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name not in ("Cylinder", "Cube"):
            continue
        translate = prim.GetAttribute("xformOp:translate").Get()
        position = (
            (0.0, 0.0, 0.0) if translate is None else (float(translate[0]), float(translate[1]), float(translate[2]))
        )
        if not all(math.isfinite(value) for value in position):
            raise ValueError(f"support shape {prim.GetPath()} has a non-finite translation")
        if type_name == "Cylinder":
            dimensions = (
                float(prim.GetAttribute("radius").Get()),
                float(prim.GetAttribute("height").Get()),
            )
        else:
            size = prim.GetAttribute("size").Get()
            scale = prim.GetAttribute("xformOp:scale").Get()
            scale = (1.0, 1.0, 1.0) if scale is None else tuple(float(value) for value in scale)
            if len(scale) != 3 or not all(math.isfinite(value) and value != 0.0 for value in scale):
                raise ValueError(f"support shape {prim.GetPath()} must have a finite nonzero scale")
            edge = float(size if size is not None else 2.0)
            dimensions = tuple(edge * abs(value) for value in scale)
        if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
            raise ValueError(f"support shape {prim.GetPath()} must have positive finite dimensions")
        shapes.append((type_name, prim.GetName(), position, dimensions))
    if not shapes:
        raise ValueError(f"support surface contains no Cylinder or Cube collision shapes: {path}")

    body_start = builder.body_count
    joint_start = builder.joint_count
    q_start = builder.joint_coord_count
    dof_start = builder.joint_dof_count
    shape_start = builder.shape_count
    # A kinematic FREE joint gets a 1e10 armature in Newton's MuJoCo backend.
    # That inflates meaninertia and changes convergence for the entire scene.
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        is_kinematic=True,
        label="support",
    )
    joint = builder.add_joint_fixed(parent=-1, child=body, label="support_fixed")
    builder.add_articulation([joint])
    for type_name, name, position, dimensions in shapes:
        xform = wp.transform(wp.vec3(*position), wp.quat_identity())
        if type_name == "Cylinder":
            radius, height = dimensions
            builder.add_shape_cylinder(
                body,
                xform=xform,
                radius=radius,
                half_height=0.5 * height,
                label=name,
            )
        else:  # Cube: USD `size * xformOp:scale` gives the full dimensions
            hx, hy, hz = (0.5 * value for value in dimensions)
            builder.add_shape_box(body, xform=xform, hx=hx, hy=hy, hz=hz, label=name)
    if (
        body != body_start
        or builder.body_count != body_start + 1
        or builder.joint_count != joint_start + 1
        or builder.joint_coord_count != q_start
        or builder.joint_dof_count != dof_start
    ):
        raise RuntimeError("support creation must append one fixed body without dynamic coordinates")
    if (
        builder.joint_type[joint] != newton.JointType.FIXED
        or builder.joint_parent[joint] != -1
        or builder.joint_child[joint] != body
        or builder.joint_q_start[joint] != q_start
        or builder.joint_qd_start[joint] != dof_start
    ):
        raise RuntimeError("support body must have a world-parented fixed joint")
    return SupportBinding(
        body_id=body,
        shapes=ShapeSpan(shape_start, builder.shape_count),
    )
