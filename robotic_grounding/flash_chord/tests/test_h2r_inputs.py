# SPDX-License-Identifier: Apache-2.0
"""Input admission regressions against an actual sealed ARCTIC or OakInk2 v3 bundle."""

import os
import shutil
import pytest

from flash_chord.h2r.inputs import content_hash, digest, human_bundle, read_json, write_json


@pytest.fixture
def human(tmp_path):
    source = os.environ.get("FLASHCHORD_H2R_HUMAN")
    if not source:
        pytest.skip("set FLASHCHORD_H2R_HUMAN to a sealed ARCTIC or OakInk2 v3 bundle")
    path = tmp_path / "human"
    shutil.copytree(source, path)
    assert read_json(path / "bundle.json")["format"] == "h2r-preprocess-v3"
    return path


def reseal_scene(path, scene):
    write_json(path / "scene.json", scene)
    manifest = read_json(path / "bundle.json")
    manifest["files"]["scene.json"] = digest(path / "scene.json")
    manifest["bundle_id"] = content_hash(manifest["files"])
    write_json(path / "bundle.json", manifest)


def test_v3_preserves_source_identity_clock_and_parts(human):
    manifest, reference, scene, arrays = human_bundle(human)
    assert manifest["bundle_id"] == read_json(human / "bundle.json")["bundle_id"]
    assert {part for obj in scene["interaction_objects"].values() for part in obj["parts"]} == set(reference["object_geometry"]["part_order"])
    assert arrays["time_s"][-1] == reference["duration_s"]
    assert reference["reference_hz"] == 30


@pytest.mark.parametrize("change", ["extra_environment", "fixed_interaction", "wrong_part_body", "table_contact"])
def test_v3_rejects_unhandled_canonical_scene(human, change):
    scene = read_json(human / "scene.json")
    obj = next(iter(scene["interaction_objects"].values()))
    part = next(iter(obj["parts"]))
    if change == "extra_environment":
        scene["environment"]["extra_support"] = scene["environment"]["table"].copy()
    elif change == "fixed_interaction":
        obj["root"]["motion"] = "fixed"
    elif change == "wrong_part_body":
        obj["parts"][part] = "other_body"
    else:
        scene["environment"]["table"]["contact"]["friction"][0] += 1
    reseal_scene(human, scene)
    dataset = read_json(human / "reference.json")["dataset"]
    with pytest.raises(ValueError, match="ARCTIC" if dataset == "arctic" else "OakInk2"):
        human_bundle(human)


def test_v3_still_rejects_changed_reference(human):
    with (human / "reference.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Input changed"):
        human_bundle(human)


@pytest.mark.parametrize("change", ["hinge_axis", "hinge_reference", "missing_top", "table_size", "table_position"])
def test_arctic_v3_rejects_changed_articulation_and_table(human, change):
    reference = read_json(human / "reference.json")
    if reference["dataset"] != "arctic":
        pytest.skip("ARCTIC articulated object contract")
    scene = read_json(human / "scene.json")
    obj = scene["interaction_objects"][scene["object"]]
    if change == "hinge_axis":
        obj["joints"]["top_hinge"]["axis"] = [0., 0., 1.]
    elif change == "hinge_reference":
        obj["joints"]["top_hinge"]["reference_key"] = "wrong_hinge"
    elif change == "missing_top":
        obj["parts"].pop("top")
    elif change == "table_size":
        scene["environment"]["table"]["size_m"][0] *= 2
    else:
        scene["environment"]["table"]["pos_m"][2] += 0.1
    reseal_scene(human, scene)
    with pytest.raises(ValueError, match="ARCTIC"):
        human_bundle(human)
