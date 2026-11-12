# SPDX-License-Identifier: Apache-2.0
"""Read and verify immutable H2R files in any of the method's Python environments."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROBOT = "virtual_wrist_sharpa_wave_bimanual"
FORMAT = "flashchord-h2r-v1"
SIDES = ("left", "right")
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
WRIST = ("x", "y", "z", "roll", "pitch", "yaw")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def content_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def member(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Asset must be inside its package: {name}")
    return path


def verify_files(root, files):
    for name, expected in files.items():
        if digest(member(root, name)) != expected:
            raise ValueError(f"Input changed: {root}/{name}")


def validate_oakink_v3(scene, reference):
    """Accept the canonical partition only for the existing rigid/table recipe."""
    parts = reference["object_geometry"]["part_order"]
    objects = scene["interaction_objects"]
    if (set(objects) != set(parts) or set(scene["parts"]) != set(parts)
            or set(scene["resources"]) != set(parts)):
        raise ValueError("OakInk2 interaction objects differ from the reference parts")
    for part in parts:
        obj = objects[part]
        if (obj["parts"] != {part: "object_" + part} or obj["joints"]
                or obj["root"] != {"part": part, "body": "object_" + part,
                                   "joint": "object_" + part + "_free", "motion": "free"}):
            raise ValueError("OakInk2 currently supports independent rigid dynamic parts only")
    environment = scene["environment"]
    if set(environment) != {"table"}:
        raise ValueError("OakInk2 currently supports the common finite table without extra environment bodies")
    table = environment["table"]
    if (table["body"] != "world" or table["shape"] != "box"
            or table["collision_geoms"] != ["table"] or table["contact_role"] != "table"
            or table["quat_wxyz"] != [1., 0., 0., 0.]
            or table["contact"] != scene["settings"]["contact"]):
        raise ValueError("Unsupported canonical OakInk2 table binding")


def validate_arctic_v3(scene, reference):
    """Accept the canonical two-part hinge and finite table ARCTIC contract."""
    parts = ["bottom", "top"]
    if (reference["object_geometry"]["part_order"] != parts
            or set(scene["parts"]) != set(parts) or set(scene["resources"]) != set(parts)
            or scene["scoring_parts"] != parts
            or set(scene["interaction_objects"]) != {scene["object"]}):
        raise ValueError("ARCTIC interaction objects differ from the two-part reference")
    obj = scene["interaction_objects"][scene["object"]]
    expected_joint = {**scene["joint"], "type": "hinge", "parent": "bottom", "child": "top",
                      "rest_pos_m": [0., 0., 0.], "rest_quat_wxyz": [1., 0., 0., 0.],
                      "mjcf_name": "object_hinge", "reference_key": "object_hinge_rad"}
    if (obj["parts"] != {"bottom": "object_bottom", "top": "object_top"}
            or obj["root"] != {"part": "bottom", "body": "object_bottom",
                               "joint": "object_free", "motion": "free"}
            or obj["joints"] != {"top_hinge": expected_joint}):
        raise ValueError("Unsupported canonical ARCTIC articulated object binding")
    if set(scene["environment"]) != {"table"}:
        raise ValueError("ARCTIC requires the common finite table without extra environment bodies")
    table = scene["environment"]["table"]
    authored = scene["recipe"]["scene"]["table"]
    if (table["body"] != "world" or table["shape"] != "box"
            or table["collision_geoms"] != ["table"] or table["contact_role"] != "table"
            or table["quat_wxyz"] != [1., 0., 0., 0.]
            or table["contact"] != scene["settings"]["contact"]
            or table["pos_m"] != [*authored["center_xy_m"], authored["height_m"] - authored["thickness_m"] / 2]
            or table["size_m"] != [*authored["size_xy_m"], authored["thickness_m"]]):
        raise ValueError("Unsupported canonical ARCTIC table binding")


def human_bundle(path):
    path = Path(path).resolve()
    manifest = read_json(path / "bundle.json")
    files = manifest["files"]
    if manifest["format"] not in ("h2r-preprocess-v1", "h2r-preprocess-v2", "h2r-preprocess-v3") or content_hash(files) != manifest["bundle_id"]:
        raise ValueError("Invalid H2R human bundle")
    verify_files(path, files)
    reference, scene = (read_json(path / name) for name in ("reference.json", "scene.json"))
    if reference["reference_sha256"] != files["reference.npz"]:
        raise ValueError("Reference identity mismatch")
    if any(v["scene_id"] != manifest["scene_id"] for v in (reference, scene)):
        raise ValueError("Scene identity mismatch")
    if content_hash(scene["recipe"]) != manifest["input_key"]:
        raise ValueError("Scene recipe changed")
    if reference["reference_hz"] != 30 or reference["speed"] != 1.0:
        raise ValueError("H2R input must retain 30 Hz and 1x speed")
    oakink = reference["dataset"] == "oakink2"
    if manifest["format"] == "h2r-preprocess-v3":
        if manifest.get("kind") != "physical" or scene["format"] != manifest["format"]:
            raise ValueError("H2R v3 requires a physical scene")
        if oakink:
            validate_oakink_v3(scene, reference)
        elif reference["dataset"] == "arctic":
            validate_arctic_v3(scene, reference)
        else:
            raise ValueError("H2R v3 currently supports ARCTIC and OakInk2 only")
    if not oakink and reference["object_geometry"]["part_order"] != ["bottom", "top"]:
        raise ValueError("Only the two-part ARCTIC contract is supported")
    with np.load(path / "reference.npz", allow_pickle=False) as data:
        arrays = dict(data)
    start, stop = reference["source_frame_range"]
    if oakink:
        if manifest["format"] not in ("h2r-preprocess-v2", "h2r-preprocess-v3") or reference["source_hz"] != 120:
            raise ValueError("OakInk2 requires the v2/v3 120 Hz source contract")
        processing = reference.get("processing", {})
        if processing.get("version") != "oakink2-quality-v2" or not read_json(path / "readiness.json")["ready"]:
            raise ValueError("OakInk2 requires an admitted quality-v2 reference")
        physical = scene["recipe"]["physical"]
        if physical["joints"] or set(physical["roots"]) != set(reference["object_geometry"]["part_order"]):
            raise ValueError("OakInk2 currently supports independent rigid dynamic parts only")
        if physical.get("fixed_bodies") or scene["settings"].get("supports"):
            raise ValueError("OakInk2 currently supports the common finite table without extra fixed bodies")
        if not (reference["mano"]["flat_hand_mean"] and reference["mano"].get("wrist_recentered")
                and reference["mano"].get("parameter_coordinates") == "source_world"):
            raise ValueError("Unsupported OakInk2 MANO parameter convention")
        count = len(arrays["time_s"])
        times = np.arange(count) / 30
        np.testing.assert_allclose(arrays["time_s"], times, atol=1e-12, rtol=0)
        np.testing.assert_allclose(arrays["source_time_s"], start / 120 + times, atol=1e-12, rtol=0)
        mapped = arrays["source_frame_left"] + arrays["source_weight"] * (
            arrays["source_frame_right"] - arrays["source_frame_left"])
        np.testing.assert_allclose(mapped, arrays["source_time_s"] * 120, atol=1e-9, rtol=0)
        if count != int(np.floor((stop - start - 1) / 4)) + 1:
            raise ValueError("OakInk2 output does not cover its complete 30 Hz window")
        if not np.isclose(reference["duration_s"], times[-1], atol=1e-12, rtol=0):
            raise ValueError("OakInk2 output duration changed")
        return manifest, reference, scene, arrays
    if not np.array_equal(arrays["source_frame"], np.arange(start, stop)):
        raise ValueError("Source frame mapping changed")
    for key, expected in (("time_s", np.arange(stop - start) / 30), ("source_time_s", np.arange(start, stop) / 30)):
        np.testing.assert_allclose(arrays[key], expected, atol=1e-12, rtol=0)
    if not np.isclose(reference["duration_s"], (stop - start - 1) / 30, atol=1e-12, rtol=0):
        raise ValueError("Source duration changed")
    return manifest, reference, scene, arrays


def robot_bundle(path):
    path = Path(path).resolve()
    manifest = read_json(path / "robot.json")
    if manifest["name"] != ROBOT:
        raise ValueError(f"H2R FlashCHORD currently requires {ROBOT}")
    verify_files(path, manifest["files_sha256"])
    semantics = read_json(path / "semantics.json")
    if len(semantics["joint_names"]) != 56:
        raise ValueError("Expected 56 scalar Sharpa joints")
    compiled = read_json(path / "compiled.json")
    if compiled["joint_order"] != semantics["joint_names"]:
        raise ValueError("Robot joint order differs from its compiled model")
    for side in SIDES:
        for axis in WRIST[3:]:
            joint = compiled["joints"][f"{side}_{axis}_joint"]
            if not joint["limited"] or joint["range"] != [-6.2, 6.2]:
                raise ValueError("The common Sharpa wrist decoder requires the packaged +/-6.2 rad limits")
    return manifest, semantics, compiled


def load_binding(path):
    path = Path(path).resolve()
    binding = read_json(path)
    if binding["format"] != FORMAT:
        raise ValueError("Unsupported FlashCHORD H2R binding")
    verify_files(path.parent, binding["files"])
    return binding


def frame_names(side):
    return [f"{side}_hand_C_MC", *[f"{side}_{d}_DP" for d in DIGITS], *[f"{side}_{d}_fingertip" for d in DIGITS]]
