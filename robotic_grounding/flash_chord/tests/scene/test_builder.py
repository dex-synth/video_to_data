# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scene assembly (embodiment + objects + replication)."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.gpu


def _scene(binding, world_count, **kwargs):
    import warp as wp
    from flash_chord.h2r.robot import H2rSharpaHands
    from flash_chord.scene.setup import setup_scene
    from flash_chord.scene.collision import CollisionPolicy

    with wp.ScopedDevice("cuda:0"):
        return setup_scene(
            parquet=binding, embodiment=H2rSharpaHands(binding), world_count=world_count,
            control_fps=30, motion_speed=1, include_support=True, decompose_objects=False,
            collision=CollisionPolicy(ground=False), **kwargs,
        ).scene


def test_scene_assembles_and_replicates(h2r_binding):
    s1 = _scene(h2r_binding, 1)
    assert len(s1.objects) == 1  # common articulated object
    assert s1.robot_reference.num_frames > 1
    assert s1.robot_reference.fps == 30.0
    assert s1.robot_reference.num_joint_q == s1.layout.num_joint_q
    assert s1.robot_reference.num_joint_dof == s1.layout.num_joint_dof
    assert s1.robot_reference.hand("left").finger_joint_pos.shape == (s1.robot_reference.num_frames, 22)
    # One common robot plus an object with a free root and a hinge.
    assert s1.model.joint_dof_count == s1.layout.num_joint_dof + 7 * len(s1.objects)
    assert s1.model.contacts().force is not None
    np.testing.assert_array_equal(s1.object_scales, 1.0)
    np.testing.assert_array_equal(s1.object_root_position_offsets_w, 0.0)

    base_bodies = s1.model.body_count
    base_dof = s1.model.joint_dof_count
    s2 = _scene(h2r_binding, 2)
    assert s2.model.body_count == 2 * base_bodies  # replication scales bodies + dofs
    assert s2.model.joint_dof_count == 2 * base_dof


def test_common_scene_rejects_extra_object_damping_and_keeps_common_friction(h2r_binding):
    import newton

    with pytest.raises(ValueError, match="extra object damping"):
        _scene(h2r_binding, 1, object_free_joint_damping=0.01)
    scene = _scene(h2r_binding, 1, contact_friction=0.5)
    active = (scene.model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    np.testing.assert_allclose(
        scene.model.shape_material_mu.numpy()[active], scene.model.h2r_settings["contact"]["friction"][0],
    )


def test_common_scene_rejects_object_scale_randomization(h2r_binding):
    with pytest.raises(ValueError, match="geometry and object scale"):
        _scene(h2r_binding, 3, object_scale_min=0.8, object_scale_max=1.2, object_scale_seed=7)


def test_ground_free_scene_requires_explicit_support_before_building():
    from flash_chord.scene.builder import build_scene
    from flash_chord.scene.collision import CollisionPolicy

    with pytest.raises(ValueError, match="ground-free scenes require an explicit support"):
        build_scene(None, None, collision=CollisionPolicy(ground=False))


def test_scene_rejects_semantic_frames_outside_the_replicated_world_layout(h2r_binding):
    from flash_chord.embodiments.base import BodyFrame

    scene = _scene(h2r_binding, 1)
    left = scene.layout.hand("left")
    invalid_body_id = scene.model.body_count
    invalid_left = replace(
        left,
        palm_frame=BodyFrame(left.palm_frame.name, invalid_body_id),
    )
    semantic_frames = tuple(
        invalid_left.palm_frame if frame.name == left.palm_frame.name else frame
        for frame in scene.layout.semantic_frames
    )
    invalid_layout = replace(
        scene.layout,
        hands=tuple(invalid_left if hand.side == "left" else hand for hand in scene.layout.hands),
        semantic_frames=semantic_frames,
    )

    with pytest.raises(ValueError, match="semantic frame body IDs must be below"):
        replace(scene, layout=invalid_layout)


def test_scene_rejects_ambiguous_or_undersized_replicated_joint_topology(h2r_binding):
    scene = _scene(h2r_binding, 2)

    def model_with(**counts):
        return SimpleNamespace(
            body_count=scene.model.body_count,
            shape_count=counts.get("shape_count", scene.model.shape_count),
            joint_coord_count=counts.get("joint_coord_count", scene.model.joint_coord_count),
            joint_dof_count=counts.get("joint_dof_count", scene.model.joint_dof_count),
        )

    with pytest.raises(ValueError, match="scene model has .* shapes; expected"):
        replace(scene, model=model_with(shape_count=scene.model.shape_count - 1))
    with pytest.raises(ValueError, match="joint coordinate count .* not divisible"):
        replace(scene, model=model_with(joint_coord_count=scene.model.joint_coord_count - 1))
    with pytest.raises(ValueError, match="joint DOF count .* not divisible"):
        replace(scene, model=model_with(joint_dof_count=scene.model.joint_dof_count - 1))
    with pytest.raises(ValueError, match="robot layout requires"):
        replace(scene, model=model_with(joint_coord_count=2 * (scene.layout.num_joint_q - 1)))
    with pytest.raises(ValueError, match="robot layout requires"):
        replace(scene, model=model_with(joint_dof_count=2 * (scene.layout.num_joint_dof - 1)))


def test_scene_rejects_object_bindings_outside_model_or_collision_ownership(h2r_binding):
    from flash_chord.scene.collision import ShapeSpan

    scene = _scene(h2r_binding, 1)
    first = scene.objects[0]
    q_per_world = scene.model.joint_coord_count
    invalid_root = replace(first.root, free_q_ids=tuple(range(q_per_world, q_per_world + 7)))
    with pytest.raises(ValueError, match="invalid or overlapping q IDs"):
        replace(scene, objects=[replace(first, root=invalid_root), *scene.objects[1:]])

    assert len(first.shapes) > 1
    incomplete_shapes = ShapeSpan(first.shapes.start + 1, first.shapes.stop)
    with pytest.raises(ValueError, match="cover the scene object shape span exactly"):
        replace(scene, objects=[replace(first, shapes=incomplete_shapes), *scene.objects[1:]])
