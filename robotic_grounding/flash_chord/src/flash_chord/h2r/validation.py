# SPDX-License-Identifier: Apache-2.0
"""Fail before training when importing common assets changes physical semantics."""

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .inputs import read_json


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


def restore_joint_force_limits(solver, binding_path):
    """Newton 1.2 unconditionally enables scalar joint-total force limits.

    The common wrists explicitly disable this limit. Restore the flags in both
    the inspection model and the active GPU model; actuator limits stay intact.
    Newton's runtime DOF updates change ranges, but do not change these flags.
    """
    robot = mujoco.MjModel.from_xml_path(str(Path(binding_path).parent / "robot/robot.xml"))
    for j in range(robot.njnt):
        aj = named_id(solver.mj_model, "joint", robot.joint(j).name)
        solver.mj_model.jnt_actfrclimited[aj] = robot.jnt_actfrclimited[j]
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


def validate_geometry(actual, robot, source):
    """Check collision hulls, finite support and every effective collision pair."""

    def active(model):
        return np.flatnonzero(model.geom_contype | model.geom_conaffinity)

    for model in (actual, robot, source):
        if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT:
            raise ValueError("H2R geometry checks require the common parent collision filter")
    table = source.geom("table").id
    expected = (
        [(robot, int(g)) for g in active(robot)]
        + [(source, int(g)) for g in active(source) if g != table]
        + [(source, table)]
    )
    if actual.ngeom != len(expected):
        raise ValueError("Newton added or removed common collision geometry")
    for ag, (model, g) in enumerate(expected):
        label = model.geom(g).name
        expected_body = (
            named_id(actual, "body", model.body(model.geom_bodyid[g]).name)
            if g != table or model is robot
            else named_id(actual, "body", "support")
        )
        close(actual.geom_bodyid[ag], expected_body, label + ".body")
        for field in ("type", "pos", "size"):
            close(getattr(actual, "geom_" + field)[ag], getattr(model, "geom_" + field)[g], label + "." + field)
        close(abs(np.dot(actual.geom_quat[ag], model.geom_quat[g])), 1, label + ".orientation")
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            a = mesh_hull(actual, actual.geom_dataid[ag])
            b = mesh_hull(model, model.geom_dataid[g])
            # STL import can merge duplicate vertices. Compare hull geometry,
            # including maxhullvert reduction, independently of vertex ordering.
            error = max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max())
            close(error, 0, label + ".collision_hull_m")
    for i, (model, g) in enumerate(expected):
        for j in range(i):
            other, h = expected[j]
            common = model is not other or collision_allowed(model, g, h)
            if collision_allowed(actual, i, j) != common:
                raise ValueError(f"Collision filtering changed for {model.geom(g).name}/{other.geom(h).name}")


def validate_solver(solver, binding_path):
    """Check the solver model, not merely the intermediate Newton builder."""
    root = Path(binding_path).parent
    actual = solver.mj_model
    robot = mujoco.MjModel.from_xml_path(str(root / "robot" / "robot.xml"))
    source = mujoco.MjModel.from_xml_path(str(root / "human" / "scene.xml"))
    for expected in (robot, source):
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
                    close(
                        getattr(actual, "dof_" + field)[ad], getattr(expected, "dof_" + field)[ed], name + "." + field
                    )
                close(actual.jnt_actgravcomp[aj], expected.jnt_actgravcomp[j], name + ".actgravcomp")
        for b in range(1, expected.nbody):
            if expected.body_mass[b] == 0:
                continue
            name = expected.body(b).name
            ab = named_id(actual, "body", name)
            for field in ("mass", "ipos", "gravcomp"):
                close(getattr(actual, "body_" + field)[ab], getattr(expected, "body_" + field)[b], name + "." + field)
            close(inertia(actual, ab), inertia(expected, b), name + ".inertia", atol=1e-9, rtol=2e-4)
    for a in range(robot.nu):
        j = robot.actuator_trnid[a, 0]
        name = robot.joint(j).name
        aj = named_id(actual, "joint", name)
        matches = np.flatnonzero(actual.actuator_trnid[:, 0] == aj)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one position servo for {name}")
        aa = matches[0]
        for field in ("gainprm", "biasprm", "forcelimited", "ctrllimited"):
            close(
                getattr(actual, "actuator_" + field)[aa],
                getattr(robot, "actuator_" + field)[a],
                name + ".actuator_" + field,
            )
        for limited, field in (("forcelimited", "forcerange"), ("ctrllimited", "ctrlrange")):
            if getattr(robot, "actuator_" + limited)[a]:
                close(
                    getattr(actual, "actuator_" + field)[aa], getattr(robot, "actuator_" + field)[a], name + "." + field
                )
        close(actual.jnt_actfrclimited[aj], robot.jnt_actfrclimited[j], name + ".actfrclimited")
        if robot.jnt_actfrclimited[j]:
            close(actual.jnt_actfrcrange[aj], robot.jnt_actfrcrange[j], name + ".actfrcrange")
    contact = read_json(root / "human" / "scene.json")["settings"]["contact"]
    for g in range(actual.ngeom):
        if actual.geom_contype[g] == 0 and actual.geom_conaffinity[g] == 0:
            continue
        for field, key in (
            ("friction", "friction"),
            ("solref", "solref"),
            ("solimp", "solimp"),
            ("condim", "condim"),
            ("margin", "margin_m"),
            ("gap", "gap_m"),
        ):
            close(getattr(actual, "geom_" + field)[g], contact[key], f"geom {g}.{field}")
    close(actual.opt.gravity, source.opt.gravity, "gravity")
    validate_geometry(actual, robot, source)
