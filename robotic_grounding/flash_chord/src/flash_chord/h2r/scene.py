# SPDX-License-Identifier: Apache-2.0
"""Compose the task's robot, interaction objects and static environment in Newton."""

import newton
import numpy as np

from flash_chord.embodiments.frames import capture_body_ids
from flash_chord.scene.builder import Scene
from flash_chord.scene.collision import CollisionPolicy, ShapeSpan, capture_collision_layout
from flash_chord.scene.objects import ObjectBinding, ObjectBodyBinding, ObjectJointBinding, ObjectRootBinding
from flash_chord.scene.support import SupportBinding

from .assets import SUPPORT_BODY
from .builder import CommonModelBuilder
from .robot import H2rSharpaHands, joint_ids


def build_h2r_scene(reference, embodiment, world_count, collision):
    if not isinstance(embodiment, H2rSharpaHands) or collision != CollisionPolicy(ground=False):
        raise ValueError("H2R scene requires its Sharpa embodiment and the complete ground-free collision policy")
    task = reference.task
    settings = task.demo.settings
    if settings["gravity_m_s2"][:2] != [0, 0]:
        raise ValueError("H2R currently requires Z-axis gravity")
    builder = CommonModelBuilder(gravity=settings["gravity_m_s2"][2])
    layout = embodiment.build(builder)
    robot_reference = embodiment.bind_reference(builder, layout, reference)
    robot_shapes = ShapeSpan(0, builder.shape_count)
    builder.add_mjcf(str(reference.binding_path.parent / "assets" / "scene.xml"),
                     parse_mujoco_options=False, collapse_fixed_joints=False)
    support = bind_support(builder)
    object_shapes = ShapeSpan(robot_shapes.stop, support.shapes.start)
    objects = [bind_object(builder, reference, obj, asset, object_shapes)
               for obj, asset in zip(task.demo.objects.values(), reference.object_assets(), strict=True)]
    return finalize_scene(builder, reference, layout, robot_reference, robot_shapes,
                          object_shapes, support, objects, settings, world_count, collision)


def body_shapes(builder, bodies, label):
    """The contiguous shape span owned by the given bodies."""
    shapes = [s for s in range(builder.shape_count) if builder.shape_body[s] in bodies]
    if not shapes or shapes != list(range(shapes[0], shapes[-1] + 1)):
        raise ValueError(f"Expected contiguous common geometry for {label}")
    return ShapeSpan(shapes[0], shapes[-1] + 1)


def bind_support(builder):
    """The static environment: the last body, fixed to the world without coordinates."""
    body = capture_body_ids(builder, (SUPPORT_BODY,))[SUPPORT_BODY]
    joint = builder.joint_child.index(body)
    shapes = body_shapes(builder, {body}, SUPPORT_BODY)
    if (body != builder.body_count - 1 or shapes.stop != builder.shape_count
            or builder.joint_type[joint] != newton.JointType.FIXED or builder.joint_parent[joint] != -1):
        raise ValueError("The static environment must be imported last as one world-fixed body")
    return SupportBinding(body, shapes)


def bind_object(builder, reference, obj, asset, object_shapes):
    """One interaction object: part bodies, free root at the first reference frame, and driven joints."""
    parts = reference.object_body_names()
    bodies = capture_body_ids(builder, tuple(obj.parts.values()))
    joints = joint_ids(builder)
    free = joints[obj.root_joint]
    root = parts.index(obj.root_part)
    q, d = builder.joint_q_start[free], builder.joint_qd_start[free]
    builder.joint_q[q : q + 3] = reference.object_body_pos_w()[0, root].tolist()
    builder.joint_q[q + 3 : q + 7] = reference.object_body_quat_w()[0, root, [1, 2, 3, 0]].tolist()
    articulations = []
    for joint in asset.articulations:
        index = joints[joint.simulation_joint_name]
        jq, jd = builder.joint_q_start[index], builder.joint_qd_start[index]
        builder.joint_q[jq] = float(reference.object_articulation()[0, joint.reference_index])
        # The native assistance drive; virtual object control scales it and VOC=0 removes it.
        builder.joint_target_ke[jd] = joint.drive.kp
        builder.joint_target_kd[jd] = joint.drive.kd
        builder.joint_effort_limit[jd] = joint.drive.effort_limit
        builder.joint_target_mode[jd] = int(newton.JointTargetMode.POSITION_VELOCITY)
        articulations.append(ObjectJointBinding(jq, jd, joint.reference_index, joint.drive))
    shapes = body_shapes(builder, set(bodies.values()), obj.name)
    if shapes.start < object_shapes.start or shapes.stop > object_shapes.stop:
        raise ValueError(f"Geometry of {obj.name} lies outside the object span")
    return ObjectBinding(
        obj.name,
        tuple(ObjectBodyBinding(bodies[body], parts.index(part)) for part, body in obj.parts.items()),
        ObjectRootBinding(bodies[obj.root_body], root, tuple(range(q, q + 7)), tuple(range(d, d + 6))),
        tuple(articulations),
        shapes,
    )


def finalize_scene(builder, reference, layout, robot_reference, robot_shapes,
                   object_shapes, support, objects, settings, world_count, collision):
    collision_layout = capture_collision_layout(builder, layout, robot_shapes, object_shapes, support.shapes)
    # The common scene's contact override applies to robot pads as well as objects and environment.
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
    model.h2r_task = reference.task
    return Scene(
        model=model,
        layout=layout,
        robot_reference=robot_reference,
        collision_layout=collision_layout,
        collision_policy=collision,
        objects=objects,
        support=support,
        object_scales=np.ones((world_count, len(objects))),
        object_root_position_offsets_w=np.zeros((world_count, len(objects), 3)),
        world_count=world_count,
    )
