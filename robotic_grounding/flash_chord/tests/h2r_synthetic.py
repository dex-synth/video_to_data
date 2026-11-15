# SPDX-License-Identifier: Apache-2.0
"""Small sealed h2r-bench tasks for tests: compilable box scenes, no dataset, robot assets or GPU."""

import json

import numpy as np

from flash_chord.h2r.h2r_io import demo as demo_io
from flash_chord.h2r.h2r_io import robot as robot_io
from flash_chord.h2r.h2r_io import task as task_io

CONTACT = {"condim": 4, "solref": [0.02, 1.0], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
           "margin_m": 0.0, "gap_m": 0.0, "friction": [2.0, 0.005, 0.0001]}
HINGE = {"type": "hinge", "axis": [1.0, 0.0, 0.0], "origin_m": [0.0, 0.0, 0.0], "limits_rad": [-0.45, 0.32]}
# name -> (interaction objects {object: (parts, joints {joint: (parent, child)})}, extra environment, hands)
STRUCTURES = {
    "single_rigid": ({"target": (["target"], {})}, False, ("left", "right")),
    "two_rigid": ({"tool": (["tool"], {}), "target": (["target"], {})}, False, ("left", "right")),
    "hinged_box": ({"bottom": (["bottom", "top"], {"hinge": ("bottom", "top")})}, False, ("left", "right")),
    "hinged_tool_right_hand": ({"grip": (["grip", "jaw"], {"tongs_hinge": ("grip", "jaw")}), "bowl": (["bowl"], {})},
                               True, ("right",)),
}
CUBE = "".join(f"v {x} {y} {z}\n" for x in (-.02, .02) for y in (-.02, .02) for z in (-.02, .02))


def scene_text(objects, extra, prefix=""):
    """World contents shared by the demo scene and the assembled task scene."""
    bodies, excludes, x = [], [], 0.0
    for parts, joints in objects.values():
        child = ""
        for name, (parent, part) in joints.items():
            child = (f'<body name="object_{part}" pos="0 0 0.05"><joint name="object_{name}" type="hinge" axis="1 0 0" '
                     f'range="-0.45 0.32" damping="0.04" frictionloss="0.0005" stiffness="3" springref="-0.067"/>'
                     f'<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>'
                     f'<geom name="{part}_collision_0" type="box" size=".02 .02 .02"/></body>')
            excludes.append(f'<exclude body1="object_{parent}" body2="object_{part}"/>')
        root = parts[0]
        bodies.append(f'<body name="object_{root}" pos="{x} 0 0.02"><freejoint name="object_{root}_free"/>'
                      f'<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>'
                      f'<geom name="{root}_collision_0" type="box" size=".02 .02 .02"/>{child}</body>')
        x += 0.2
    world = '<geom name="table" type="box" size=".5 .5 .01" pos="0 0 -.01"/>'
    asset = ""
    if extra:  # one more box support in the world, and one fixed mesh on its own rotated body
        world += '<geom name="support_step" type="box" size=".1 .1 .01" pos=".3 .2 .05"/>'
        bodies.append('<body name="object_shelf" pos="-.3 .1 .02" quat="0.9238795 0 0 0.3826834">'
                      '<geom name="shelf_collision_0" type="mesh" mesh="shelf_collision_0" pos="0 .01 0"/></body>')
        asset = f'<asset><mesh name="shelf_collision_0" file="{prefix}meshes/shelf_collision_000.obj"/></asset>'
    return (f'<compiler angle="radian" autolimits="true"/><default><geom friction="2 0.005 0.0001" condim="4" '
            f'density="0"/></default>{asset}', world + "".join(bodies), f'<contact>{"".join(excludes)}</contact>')


def write_demo(path, objects, extra, sides, dataset, frames=4):
    (path / "meshes").mkdir(parents=True)
    (path / "meshes/shelf_collision_000.obj").write_text(CUBE)
    head, world, contact = scene_text(objects, extra)
    (path / "scene.xml").write_text(f"<mujoco>{head}<worldbody>{world}</worldbody>{contact}</mujoco>")
    part_order = [part for parts, _ in objects.values() for part in parts]
    unit = np.array([1., 0., 0., 0.])
    arrays = {"time_s": np.arange(frames) / 30., "source_time_s": np.arange(frames) / 30.,
              "part_pos_m": np.zeros((frames, len(part_order), 3)),
              "part_quat_wxyz": np.tile(unit, (frames, len(part_order), 1))}
    for side in sides:
        arrays.update({f"{side}_joints_m": np.zeros((frames, 21, 3)), f"{side}_vertices_m": np.zeros((frames, 778, 3)),
                       f"{side}_faces": np.zeros((2, 3), dtype=np.int32),
                       f"{side}_mano_quat": np.tile(unit, (frames, 16, 1)),
                       f"{side}_mano_trans": np.zeros((frames, 3)), f"{side}_mano_shape": np.zeros(10)})
    interaction = {}
    for name, (parts, joints) in objects.items():
        described = {}
        for joint, (parent, child) in joints.items():
            arrays[f"joint_{joint}_rad"] = np.linspace(0., 0.2, frames)
            described[joint] = {**HINGE, "parent": parent, "child": child, "mjcf_name": f"object_{joint}",
                                "reference_key": f"joint_{joint}_rad"}
        interaction[name] = {
            "root": {"part": parts[0], "body": f"object_{parts[0]}", "joint": f"object_{parts[0]}_free",
                     "motion": "free"},
            "parts": {part: f"object_{part}" for part in parts}, "joints": described}
    environment = {"table": {"body": "world", "shape": "box", "collision_geoms": ["table"], "contact_role": "table"}}
    if extra:
        environment["support_step"] = {"body": "world", "shape": "box", "collision_geoms": ["support_step"],
                                       "contact_role": "table"}
        environment["shelf"] = {"body": "object_shelf", "shape": "mesh", "collision_geoms": ["shelf_collision_0"],
                                "contact_role": "environment"}
    np.savez_compressed(path / "reference.npz", **arrays)
    (path / "reference.json").write_text(json.dumps({
        "dataset": dataset, "profile": "clip", "task": {"hand_sides": list(sides)},
        "reference_hz": 30., "duration_s": (frames - 1) / 30.,
        "mano": {"convention": "h2r-mano-v1", "left_shapedirs_x_mirrored": False, "assets": []},
        "quaternion_convention": "wxyz-unit-continuous-v1", "object_geometry": {"part_order": part_order}}))
    (path / "scene.json").write_text(json.dumps({
        "interaction_objects": interaction, "scoring_parts": part_order, "environment": environment,
        "settings": {"timestep_s": 1 / 300, "gravity_m_s2": [0., 0., -9.81], "contact": CONTACT}}))
    demo_io.seal_demo(path, input_key="recipe")


def write_robot(path, sides, name, wrist_limit):
    (path / "assets").mkdir(parents=True)
    for file in ("robot.xml", "assets/robot.urdf", *(f"assets/{side}.urdf" for side in sides)):
        (path / file).write_text("<robot/>")
    hands = {side: {"wrist_body": f"{side}_hand_C_MC",
                    "wrist_translation_joints": [f"{side}_{axis}_joint" for axis in "xyz"],
                    "wrist_rotation_joints": [f"{side}_{axis}_joint" for axis in ("roll", "pitch", "yaw")],
                    "finger_joints": []} for side in sides}
    joints = [joint for hand in hands.values()
              for joint in hand["wrist_translation_joints"] + hand["wrist_rotation_joints"]]
    (path / "semantics.json").write_text(json.dumps({"hands": hands, "joint_names": joints, "body_names": []}))
    (path / "compiled.json").write_text(json.dumps({"joints": {
        joint: {"limited": True, "range": [-wrist_limit, wrist_limit]} for joint in joints}}))
    robot_io.seal_robot(path, name, sides, mjcf="robot.xml", urdf="assets/robot.urdf",
                        hand_urdfs={side: f"assets/{side}.urdf" for side in sides})


def write_task(path, structure="hinged_box", *, robot="sharpa_wave", dataset="arctic", fixed_root=False,
               wrist_limit=6.2):
    """A standalone sealed task of one structure; returns its directory."""
    objects, extra, sides = STRUCTURES[structure]
    write_demo(path / "demo", objects, extra, sides, dataset)
    if fixed_root:  # reseal the demo with its first interaction object welded to the world
        scene = json.loads((path / "demo/scene.json").read_text())
        root = next(iter(scene["interaction_objects"].values()))["root"]
        root.update(joint=None, motion="fixed")
        (path / "demo/scene.json").write_text(json.dumps(scene))
        demo_io.seal_demo(path / "demo", input_key="recipe")
    write_robot(path / "robot", sides, robot, wrist_limit)
    head, world, contact = scene_text(objects, extra, prefix="demo/")
    (path / "scene.xml").write_text(f"<mujoco>{head}<worldbody>{world}</worldbody>{contact}</mujoco>")
    joints = {name: {"armature": 0.0, "friction": 0.0005} for _, described in objects.values() for name in described}
    (path / "compiled.json").write_text(json.dumps({"base_scene": {"joints": joints}}))
    task_io.seal_task(path, "demo", "robot")
    return path
