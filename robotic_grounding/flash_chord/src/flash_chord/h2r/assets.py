# SPDX-License-Identifier: Apache-2.0
"""Prepare narrowly scoped MJCFs, retaining compiled H2R physical quantities."""

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .inputs import DIGITS

SUPPORT_BODY = "support"  # the one world-fixed body that carries the static environment


def numbers(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).reshape(-1))


def vector(node, name, default):
    return np.array(node.get(name, default).split(), dtype=float)


def portable_xml(source, destination):
    """Resolve mesh paths before moving an XML; leave the source package untouched."""
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if compiler is not None:
        compiler.set("meshdir", "")
    for mesh in root.findall("asset/mesh"):
        resolved = (Path(source).parent / meshdir / mesh.attrib["file"]).resolve()
        if not resolved.is_relative_to(Path(source).parent.resolve()):
            raise ValueError(f"Mesh escapes its common package: {resolved}")
        # Assets and the copied task live below the same preparation root.
        mesh.set("file", os.path.relpath(resolved, Path(destination).parent))
    return root


def robot_xml(robot, destination):
    """The task's robot with the compiled position-servo gains written out."""
    root = portable_xml(robot.mjcf, destination)
    # Materialize compiled gains instead of relying on importer dampratio handling.
    # dampratio itself is not a damping coefficient.
    for actuator in root.findall("actuator/*"):
        joint = robot.compiled["joints"][actuator.attrib["joint"]]
        actuator.attrib.pop("dampratio", None)
        actuator.attrib.pop("inheritrange", None)
        actuator.set("kv", numbers([joint["kv"]]))
        actuator.set("kp", numbers([joint["kp"]]))
        actuator.set("ctrllimited", str(joint["ctrl_limited"]).lower())
        actuator.set("ctrlrange", numbers(joint["ctrlrange"]))
        actuator.set("forcelimited", str(joint["force_limited"]).lower())
        actuator.set("forcerange", numbers(joint["forcerange"]))
    ET.ElementTree(root).write(destination)
    return root


def ik_xml(robot, side, wrist_body, state, destination):
    """One hand below its wrist body, for the native free-flyer IK."""
    from robotic_grounding.retarget import SHARPA_WAVE_XMLS_DIR

    ik = copy.deepcopy(robot)
    palm = copy.deepcopy(ik.find(f".//body[@name='{wrist_body}']"))
    palm.attrib.pop("pos", None)
    palm.attrib.pop("quat", None)
    for joint in list(palm.findall("joint")):
        palm.remove(joint)
    world = ik.find("worldbody")
    world.clear()
    world.append(palm)
    for tag in ("contact", "actuator", "keyframe"):
        for node in ik.findall(tag):
            ik.remove(node)
    # Pinocchio reads defaults in document order. MuJoCo accepts either order.
    defaults = ik.find("default")
    ik.remove(defaults)
    ik.insert(1, defaults)
    # These sites define the native IK objectives, not physical bodies. Keep
    # their native orientations; place tips on the COMMON fingertip markers.
    template = SHARPA_WAVE_XMLS_DIR / f"{side}_sharpawave.xml"
    for body in ET.parse(template).findall(".//body"):
        target = ik.find(f".//body[@name='{body.attrib['name']}']")
        for site in body.findall("site"):
            target.append(copy.deepcopy(site))
    for digit in DIGITS:
        distal = state.body(f"{side}_{digit}_DP")
        tip = state.body(f"{side}_{digit}_fingertip")
        local = distal.xmat.reshape(3, 3).T @ (tip.xpos - distal.xpos)
        ik.find(f".//site[@name='{side}_{digit}_tip_site']").set("pos", numbers(local))
    ET.ElementTree(ik).write(destination)


def scene_xml(demo, destination):
    """The demo scene with its whole static environment on one world-fixed body, placed last.

    Interaction objects keep their bodies, joints and inertias. The table, extra box supports and
    fixed meshes keep their world placement; one body holds them because the native scene binds
    exactly one support component.
    """
    scene = portable_xml(demo.scene_xml, destination)
    world = scene.find("worldbody")
    roots = {obj.root_body for obj in demo.objects.values()}
    support = ET.Element("body", name=SUPPORT_BODY)
    for node in list(world):
        if node.tag == "body" and node.get("name") in roots:
            continue
        world.remove(node)
        if node.tag == "geom":
            support.append(node)
        elif node.tag == "body" and all(child.tag == "geom" for child in node):
            # A fixed body: express its geoms in the world frame.
            position = vector(node, "pos", "0 0 0")
            rotation = Rotation.from_quat(vector(node, "quat", "1 0 0 0"), scalar_first=True)
            for geom in node:
                local = Rotation.from_quat(vector(geom, "quat", "1 0 0 0"), scalar_first=True)
                geom.set("pos", numbers(position + rotation.apply(vector(geom, "pos", "0 0 0"))))
                geom.set("quat", numbers((rotation * local).as_quat(scalar_first=True)))
                support.append(geom)
        else:
            raise ValueError(f"Unsupported static scene element: {node.tag} {node.get('name')}")
    world.append(support)
    static = {geom.get("name") for geom in support}
    declared = {name for entry in demo.environment.values() for name in entry["collision_geoms"]}
    if not declared <= static:
        raise ValueError(f"Environment geometry is missing from the scene: {sorted(declared - static)}")
    bodies = {body.get("name") for body in world.iter("body")}
    for exclude in scene.findall("contact/exclude"):
        if not {exclude.get("body1"), exclude.get("body2")} <= bodies:
            raise ValueError("A collision exclusion names a fixed body that was merged into the support")
    for node in scene.findall("keyframe"):
        scene.remove(node)
    ET.ElementTree(scene).write(destination)


def prepare_assets(task, output):
    """Derive only the XML boundaries needed by the native loader and IK."""
    output = Path(output)
    output.mkdir()
    robot = robot_xml(task.robot, output / "robot.xml")
    state = mujoco.MjData(model := mujoco.MjModel.from_xml_path(str(output / "robot.xml")))
    mujoco.mj_forward(model, state)
    # IK keeps its free-flyer parameterization, using the exact common hand subtree.
    for side in task.hand_sides:
        wrist_body = task.robot.semantics["hands"][side]["wrist_body"]
        ik_xml(robot, side, wrist_body, state, output / f"ik_{side}.xml")
    scene_xml(task.demo, output / "scene.xml")
