# SPDX-License-Identifier: Apache-2.0
"""Fail before training when importing the task's assets changes physical semantics."""

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .assets import SUPPORT_BODY


def named_id(model, kind, name):
    matches = [
        i
        for i in range(getattr(model, {"joint": "njnt", "body": "nbody", "geom": "ngeom"}[kind]))
        if getattr(model, kind)(i).name == name or getattr(model, kind)(i).name.endswith("_" + name)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one imported {kind} {name}, found {matches}")
    return matches[0]


def close(actual, expected, label, atol=2e-6, rtol=2e-5):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f"H2R physical mismatch for {label}: imported={actual}, common={expected}")


def inertia(model, body):
    rotation = Rotation.from_quat(model.body_iquat[body][[1, 2, 3, 0]]).as_matrix()
    return rotation @ np.diag(model.body_inertia[body]) @ rotation.T


def solref_coefficients(solref):
    """Compare equivalent positive time/damping and negative stiffness forms."""
    first, second = solref
    if first > 0 and second > 0:
        return np.array([1 / (first * second) ** 2, 2 / first])
    return -np.asarray(solref)


def task_model(task):
    """The task's assembled scene, compiled: the physical model that the evaluation runs."""
    return mujoco.MjModel.from_xml_path(str(task.scene_xml))


def restore_joint_force_limits(solver, task):
    """Newton's generated joint-target drives enable scalar joint-total force limits.

    The common wrists explicitly disable this limit. Restore the flags in both
    the inspection model and the active GPU model; actuator limits stay intact.
    Newton's runtime DOF updates change ranges, but do not change these flags.
    """
    expected = task_model(task)
    for name in task.robot.semantics["joint_names"]:
        aj = named_id(solver.mj_model, "joint", name)
        solver.mj_model.jnt_actfrclimited[aj] = expected.jnt_actfrclimited[expected.joint(name).id]
    solver.mjw_model.jnt_actfrclimited.assign(solver.mj_model.jnt_actfrclimited.astype(bool))


def collision_allowed(model, first, second):
    """Effective MuJoCo mask/exclusion/parent filtering for two colliders."""
    if not (
        model.geom_contype[first] & model.geom_conaffinity[second]
        or model.geom_contype[second] & model.geom_conaffinity[first]
    ):
        return False
    first, second = sorted((int(model.geom_bodyid[first]), int(model.geom_bodyid[second])))
    if (first << 16) + second in model.exclude_signature:
        return False
    first, second = model.body_weldid[first], model.body_weldid[second]
    if first == second:
        return False
    parent_child = (
        first
        and second
        and (
            model.body_weldid[model.body_parentid[first]] == second
            or model.body_weldid[model.body_parentid[second]] == first
        )
    )
    return not parent_child


def mesh_hull(model, mesh):
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    start = model.mesh_graphadr[mesh]
    if start < 0:
        raise ValueError("Common collision mesh has no convex hull")
    count = model.mesh_graph[start]
    return vertices[model.mesh_graph[start + 2 + count : start + 2 + 2 * count]]


CONTACT_FIELDS = ("friction", "solref", "solimp", "condim", "margin", "gap", "priority", "solmix")


def validate_geometry(actual, expected, robot_bodies):
    """Check collision hulls, contact parameters, placement and every effective collision pair.

    `actual` holds its colliding geoms as robot, then interaction objects, then the static
    environment on the support body; the task's geoms are paired in that order.
    """
    for model in (actual, expected):
        if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT:
            raise ValueError("H2R geometry checks require the common parent collision filter")

    def group(geom):
        body = expected.geom_bodyid[geom]
        if expected.body(body).name in robot_bodies:
            return 0
        return 2 if expected.body_weldid[body] == 0 else 1  # welded to the world: static environment

    active = np.flatnonzero(expected.geom_contype | expected.geom_conaffinity)
    order = sorted((int(g) for g in active), key=group)  # stable: keeps the task's order inside each group
    if actual.ngeom != len(order):
        raise ValueError("Newton added or removed common collision geometry")
    support = named_id(actual, "body", SUPPORT_BODY)
    poses = []
    for model in (actual, expected):
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        poses.append(data)
    for ag, g in enumerate(order):
        label = expected.geom(g).name or f"geom {g}"
        static = group(g) == 2
        body = support if static else named_id(actual, "body", expected.body(expected.geom_bodyid[g]).name)
        close(actual.geom_bodyid[ag], body, label + ".body")
        close(actual.geom_type[ag], expected.geom_type[g], label + ".type")
        if static:  # one support body replaces the world and its fixed bodies: compare world placement
            close(poses[0].geom_xpos[ag], poses[1].geom_xpos[g], label + ".world_pos")
            close(poses[0].geom_xmat[ag], poses[1].geom_xmat[g], label + ".world_orientation")
        else:
            close(actual.geom_pos[ag], expected.geom_pos[g], label + ".pos")
            close(abs(np.dot(actual.geom_quat[ag], expected.geom_quat[g])), 1, label + ".orientation")
        # MuJoCo capsules use radius and half-length only. Newton also fills
        # size[2] with the radius; that unused slot does not change the shape.
        size_count = 2 if expected.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 3
        close(actual.geom_size[ag, :size_count], expected.geom_size[g, :size_count], label + ".size")
        for field in CONTACT_FIELDS:
            close(getattr(actual, "geom_" + field)[ag], getattr(expected, "geom_" + field)[g], f"{label}.{field}")
        if expected.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            a = mesh_hull(actual, actual.geom_dataid[ag])
            b = mesh_hull(expected, expected.geom_dataid[g])
            # STL import can merge duplicate vertices. Compare hull geometry,
            # including maxhullvert reduction, independently of vertex ordering.
            error = max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max())
            close(error, 0, label + ".collision_hull_m")
    for i, g in enumerate(order):
        for j in range(i):
            if collision_allowed(actual, i, j) != collision_allowed(expected, g, order[j]):
                raise ValueError(f"Collision filtering changed for {expected.geom(g).name}/"
                                 f"{expected.geom(order[j]).name}")


def validate_solver(solver, task):
    """Check the solver model, not merely the intermediate Newton builder, against the task's scene."""
    actual, expected = solver.mj_model, task_model(task)
    # Fixed support must not introduce DOFs. A kinematic FREE joint is not
    # equivalent: Newton adds a huge armature and changes global convergence.
    close(actual.nv, expected.nv, "dynamic DOF count")
    close(actual.nq, expected.nq, "dynamic coordinate count")
    close(actual.body_jntnum[named_id(actual, "body", SUPPORT_BODY)], 0, "support.dynamic_joints")
    close(actual.opt.disableflags, expected.opt.disableflags, "physics disable flags (including MultiCCD)")
    gpu = solver.mjw_model
    close(gpu.opt.disableflags, expected.opt.disableflags, "GPU physics disable flags")
    close(gpu.dof_armature.numpy(), actual.dof_armature, "GPU armature")
    close(gpu.stat.meaninertia.numpy(), actual.stat.meaninertia, "GPU meaninertia", rtol=2e-4)
    close(actual.stat.meaninertia, expected.stat.meaninertia, "meaninertia", rtol=2e-4)
    for j in range(expected.njnt):
        name = expected.joint(j).name
        aj = named_id(actual, "joint", name)
        for field in ("type", "axis", "pos", "limited", "range", "margin", "solimp", "stiffness"):
            close(getattr(actual, "jnt_" + field)[aj], getattr(expected, "jnt_" + field)[j], name + "." + field)
        close(
            solref_coefficients(actual.jnt_solref[aj]),
            solref_coefficients(expected.jnt_solref[j]),
            name + ".limit_solref",
        )
        if expected.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            ad, ed = actual.jnt_dofadr[aj], expected.jnt_dofadr[j]
            for field in ("damping", "armature", "frictionloss", "solimp", "solref"):
                close(getattr(actual, "dof_" + field)[ad], getattr(expected, "dof_" + field)[ed], name + "." + field)
            close(actual.qpos_spring[actual.jnt_qposadr[aj]], expected.qpos_spring[expected.jnt_qposadr[j]],
                  name + ".springref")
            close(actual.jnt_actgravcomp[aj], expected.jnt_actgravcomp[j], name + ".actgravcomp")
    for b in range(1, expected.nbody):
        if expected.body_mass[b] == 0:
            continue
        name = expected.body(b).name
        ab = named_id(actual, "body", name)
        for field in ("mass", "ipos", "gravcomp"):
            close(getattr(actual, "body_" + field)[ab], getattr(expected, "body_" + field)[b], name + "." + field)
        close(inertia(actual, ab), inertia(expected, b), name + ".inertia", atol=1e-9, rtol=2e-4)
    for a in range(expected.nu):
        j = expected.actuator_trnid[a, 0]
        name = expected.joint(j).name
        aj = named_id(actual, "joint", name)
        matches = np.flatnonzero(actual.actuator_trnid[:, 0] == aj)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one position servo for {name}")
        aa = matches[0]
        for field in ("gainprm", "biasprm", "forcelimited", "ctrllimited"):
            close(
                getattr(actual, "actuator_" + field)[aa],
                getattr(expected, "actuator_" + field)[a],
                name + ".actuator_" + field,
            )
        for limited, field in (("forcelimited", "forcerange"), ("ctrllimited", "ctrlrange")):
            if getattr(expected, "actuator_" + limited)[a]:
                close(
                    getattr(actual, "actuator_" + field)[aa], getattr(expected, "actuator_" + field)[a],
                    name + "." + field,
                )
        close(actual.jnt_actfrclimited[aj], expected.jnt_actfrclimited[j], name + ".actfrclimited")
        if expected.jnt_actfrclimited[j]:
            close(actual.jnt_actfrcrange[aj], expected.jnt_actfrcrange[j], name + ".actfrcrange")
    for field in CONTACT_FIELDS[:6]:
        close(getattr(gpu, "geom_" + field).numpy(), getattr(actual, "geom_" + field), "GPU geom." + field)
    close(actual.opt.gravity, expected.opt.gravity, "gravity")
    validate_geometry(actual, expected, set(task.robot.semantics["body_names"]))
