# SPDX-License-Identifier: Apache-2.0
"""Compose the common robot, articulated object and finite table in Newton."""

import newton
import numpy as np

from flash_chord.embodiments.frames import capture_body_ids
from flash_chord.scene.builder import Scene
from flash_chord.scene.collision import CollisionPolicy, ShapeSpan, capture_collision_layout
from flash_chord.scene.objects import ObjectBinding, ObjectBodyBinding, ObjectJointBinding, ObjectRootBinding
from flash_chord.scene.support import add_support_surfaces

from .builder import CommonModelBuilder
from .inputs import read_json
from .robot import H2rSharpaHands, joint_ids


def build_h2r_scene(reference, embodiment, world_count, collision):
    if not isinstance(embodiment, H2rSharpaHands) or collision != CollisionPolicy(ground=False):
        raise ValueError("H2R scene requires its Sharpa embodiment and the complete ground-free collision policy")
    root = reference.binding_path.parent
    settings = read_json(root / "human" / "scene.json")["settings"]
    if settings["gravity_m_s2"][:2] != [0, 0]:
        raise ValueError("H2R currently requires Z-axis gravity")
    builder = CommonModelBuilder(gravity=settings["gravity_m_s2"][2])
    layout = embodiment.build(builder)
    robot_reference = embodiment.bind_reference(builder, layout, reference)
    robot_shapes = ShapeSpan(0, builder.shape_count)
    builder.add_mjcf(str(root / "assets" / "object.xml"), parse_mujoco_options=False, collapse_fixed_joints=False)
    object_shapes = ShapeSpan(robot_shapes.stop, builder.shape_count)
    bodies = capture_body_ids(builder, ("object_bottom", "object_top"))
    joints = joint_ids(builder)
    free = joints["object_free"]
    hinge = joints["object_hinge"]
    q, d = builder.joint_q_start[free], builder.joint_qd_start[free]
    hq, hd = builder.joint_q_start[hinge], builder.joint_qd_start[hinge]
    builder.joint_q[q : q + 3] = reference.object_body_pos_w()[0, 0].tolist()
    builder.joint_q[q + 3 : q + 7] = reference.object_body_quat_w()[0, 0, [1, 2, 3, 0]].tolist()
    builder.joint_q[hq] = float(reference.object_articulation()[0, 0])
    drive = reference.object_assets()[0].articulations[0].drive
    builder.joint_target_ke[hd] = drive.kp
    builder.joint_target_kd[hd] = drive.kd
    builder.joint_effort_limit[hd] = drive.effort_limit
    builder.joint_target_mode[hd] = int(newton.JointTargetMode.POSITION_VELOCITY)
    obj = ObjectBinding(
        reference.object_name(),
        (ObjectBodyBinding(bodies["object_bottom"], 0), ObjectBodyBinding(bodies["object_top"], 1)),
        ObjectRootBinding(bodies["object_bottom"], 0, tuple(range(q, q + 7)), tuple(range(d, d + 6))),
        (ObjectJointBinding(hq, hd, 0, drive),),
        object_shapes,
    )
    support = add_support_surfaces(builder, root / "assets" / "support.usda")
    collision_layout = capture_collision_layout(builder, layout, robot_shapes, object_shapes, support.shapes)
    # The common scene's contact override applies to robot pads as well as objects/table.
    contact = settings["contact"]
    timeconst, ratio = contact["solref"]
    if timeconst <= 0 or ratio <= 0:
        raise ValueError("H2R currently supports positive MuJoCo contact timeconst/dampratio")
    for shape in range(builder.shape_count):
        # Newton's defaults need not match the authored common contact envelope.
        builder.shape_margin[shape] = contact["margin_m"]
        builder.shape_gap[shape] = contact["gap_m"]
        for field, value in (
            ("mu", contact["friction"][0]),
            ("mu_torsional", contact["friction"][1]),
            ("mu_rolling", contact["friction"][2]),
            ("ke", 1 / (timeconst * ratio) ** 2),
            ("kd", 2 / timeconst),
        ):
            getattr(builder, "shape_material_" + field)[shape] = value
        builder.custom_attributes["mujoco:condim"].values[shape] = contact["condim"]
        builder.custom_attributes["mujoco:solref"].values[shape] = tuple(contact["solref"])
        builder.custom_attributes["mujoco:solref_mode"].values[shape] = 1  # Raw MuJoCo parameters in Newton 1.5.
        builder.custom_attributes["mujoco:geom_solimp"].values[shape] = tuple(contact["solimp"])
        builder.custom_attributes["mujoco:geom_priority"].values[shape] = 0
        builder.custom_attributes["mujoco:geom_solmix"].values[shape] = 1.0
    builder.request_contact_attributes("force")
    batch = CommonModelBuilder(gravity=settings["gravity_m_s2"][2])
    batch.replicate(builder, world_count)
    model = batch.finalize()
    model.h2r_settings = settings
    model.h2r_binding_path = reference.binding_path
    return Scene(
        model=model,
        layout=layout,
        robot_reference=robot_reference,
        collision_layout=collision_layout,
        collision_policy=collision,
        objects=[obj],
        support=support,
        object_scales=np.ones((world_count, 1)),
        object_root_position_offsets_w=np.zeros((world_count, 1, 3)),
        world_count=world_count,
    )
