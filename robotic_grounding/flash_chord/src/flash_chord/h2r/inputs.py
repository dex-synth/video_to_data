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


def human_bundle(path):
    path = Path(path).resolve()
    manifest = read_json(path / "bundle.json")
    files = manifest["files"]
    if manifest["format"] != "h2r-preprocess-v1" or content_hash(files) != manifest["bundle_id"]:
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
    if reference["object_geometry"]["part_order"] != ["bottom", "top"]:
        raise ValueError("Only the two-part ARCTIC contract is supported")
    with np.load(path / "reference.npz", allow_pickle=False) as data:
        arrays = dict(data)
    start, stop = reference["source_frame_range"]
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
