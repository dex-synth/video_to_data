# SPDX-License-Identifier: Apache-2.0
"""Prepare narrowly scoped MJCFs, retaining compiled H2R physical quantities."""

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .inputs import DIGITS, SIDES, robot_bundle


def numbers(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).reshape(-1))


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
        # Assets and copied source packages live below the same preparation root.
        mesh.set("file", os.path.relpath(resolved, Path(destination).parent))
    return root


def prepare_assets(human_path, robot_path, output):
    """Derive only the XML boundaries needed by the native loader and IK."""
    from robotic_grounding.retarget import SHARPA_WAVE_XMLS_DIR

    output = Path(output)
    output.mkdir()
    _, _, compiled = robot_bundle(robot_path)
    robot = portable_xml(Path(robot_path) / "robot.xml", output / "robot.xml")
    # Materialize compiled gains instead of relying on importer dampratio handling.
    # dampratio itself is not a damping coefficient.
    for actuator in robot.findall("actuator/*"):
        joint = compiled["joints"][actuator.attrib["joint"]]
        actuator.attrib.pop("dampratio", None)
        actuator.attrib.pop("inheritrange", None)
        actuator.set("kv", numbers([joint["kv"]]))
        actuator.set("kp", numbers([joint["kp"]]))
        actuator.set("ctrllimited", str(joint["ctrl_limited"]).lower())
        actuator.set("ctrlrange", numbers(joint["ctrlrange"]))
        actuator.set("forcelimited", str(joint["force_limited"]).lower())
        actuator.set("forcerange", numbers(joint["forcerange"]))
    ET.ElementTree(robot).write(output / "robot.xml")
    model = mujoco.MjModel.from_xml_path(str(output / "robot.xml"))
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    # IK keeps its free-flyer parameterization, using the exact common hand subtree.
    for side in SIDES:
        ik = copy.deepcopy(robot)
        palm = copy.deepcopy(ik.find(f".//body[@name='{side}_hand_C_MC']"))
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
        ET.ElementTree(ik).write(output / f"ik_{side}.xml")
    scene = portable_xml(Path(human_path) / "scene.xml", output / "object.xml")
    table = scene.find("worldbody/geom[@name='table']")
    scene.find("worldbody").remove(table)
    for node in scene.findall("keyframe"):
        scene.remove(node)
    ET.ElementTree(scene).write(output / "object.xml")
    # Existing support loader accepts a Cube with full dimensions size*scale.
    pos = np.fromstring(table.attrib["pos"], sep=" ")
    size = 2 * np.fromstring(table.attrib["size"], sep=" ")
    (output / "support.usda").write_text(
        '#usda 1.0\n(\n upAxis = "Z"\n metersPerUnit = 1\n)\n'
        'def Cube "table" {\n double size = 1\n'
        f" double3 xformOp:translate = ({', '.join(map(str, pos))})\n"
        f" double3 xformOp:scale = ({', '.join(map(str, size))})\n"
        ' uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n}\n'
    )
