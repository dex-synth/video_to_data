# SPDX-License-Identifier: Apache-2.0
"""One task directory is the only input: structure, admission and binding checks on small sealed tasks."""

import mujoco
import numpy as np
import pytest
from h2r_synthetic import STRUCTURES, write_task

from flash_chord.configuration import compose_config
from flash_chord.h2r.assets import SUPPORT_BODY, scene_xml
from flash_chord.h2r.configuration import require_h2r_input
from flash_chord.h2r.h2r_io import export_task
from flash_chord.h2r.inputs import BINDING, FORMAT, load_input, load_prepared, seal_prepared, write_json
from flash_chord.h2r.prepare import ik_settings
from flash_chord.h2r.reference import JOINT_DRIVE, object_assets


def prepared(tmp_path, structure):
    """A sealed preparation holding only what the loaders read before any method data exists."""
    task = load_input(write_task(tmp_path / "source", structure))
    output = tmp_path / "prepared"
    (output / "assets").mkdir(parents=True)
    export_task(task, output / "task")
    scene_xml(load_input(output / "task").demo, output / "assets" / "scene.xml")
    return seal_prepared(output, {})


def world_geoms(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return model, {model.geom(g).name: (data.geom_xpos[g].copy(), data.geom_xmat[g].copy())
                   for g in range(model.ngeom)}


@pytest.mark.parametrize("structure", STRUCTURES)
def test_scene_keeps_every_object_and_static_placement(tmp_path, structure):
    task = load_input(write_task(tmp_path / "task", structure))
    (tmp_path / "assets").mkdir()
    scene_xml(task.demo, tmp_path / "assets" / "scene.xml")
    source, expected = world_geoms(task.demo.scene_xml)
    derived, actual = world_geoms(tmp_path / "assets" / "scene.xml")
    assert (derived.nq, derived.nv, derived.njnt) == (source.nq, source.nv, source.njnt)
    assert set(actual) == set(expected)
    for name, (position, orientation) in expected.items():
        np.testing.assert_allclose(actual[name][0], position, atol=1e-12)
        np.testing.assert_allclose(actual[name][1], orientation, atol=1e-12)
    # The table, extra supports and fixed meshes share one jointless body that comes last.
    support = derived.body(SUPPORT_BODY)
    assert support.id == derived.nbody - 1 and derived.body_jntnum[support.id] == 0
    static = {name for entry in task.demo.environment.values() for name in entry["collision_geoms"]}
    assert {derived.geom(g).name for g in range(derived.ngeom) if derived.geom_bodyid[g] == support.id} == static
    for obj in task.demo.objects.values():
        for body in obj.parts.values():
            np.testing.assert_array_equal(derived.body(body).mass, source.body(body).mass)
        for joint in obj.joints:
            for field in ("stiffness", "range", "axis"):
                np.testing.assert_array_equal(getattr(derived, "jnt_" + field)[derived.joint(joint.mjcf_name).id],
                                              getattr(source, "jnt_" + field)[source.joint(joint.mjcf_name).id])
            address = derived.joint(joint.mjcf_name).qposadr[0]
            assert derived.qpos_spring[address] == source.qpos_spring[source.joint(joint.mjcf_name).qposadr[0]]


@pytest.mark.parametrize("structure", STRUCTURES)
def test_native_assets_follow_the_task_objects(tmp_path, structure):
    task = load_input(write_task(tmp_path / "task", structure))
    assets = object_assets(task, tmp_path / "scene.xml")
    assert [asset.name for asset in assets] == list(task.demo.objects)
    assert [body.reference_name for asset in assets for body in asset.bodies] == list(task.demo.part_order)
    joints = [joint for asset in assets for joint in asset.articulations]
    assert [joint.reference_index for joint in joints] == list(range(len(joints)))
    assert [joint.simulation_joint_name for joint in joints] == [
        joint.mjcf_name for obj in task.demo.objects.values() for joint in obj.joints]
    assert all(joint.drive == JOINT_DRIVE for joint in joints)
    for asset, obj in zip(assets, task.demo.objects.values(), strict=True):
        assert asset.root_reference_name == obj.root_part


def test_native_limits_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="do not cover robot 'other_hand'"):
        load_input(write_task(tmp_path / "robot", robot="other_hand"))
    with pytest.raises(ValueError, match="need a free root"):
        load_input(write_task(tmp_path / "fixed", fixed_root=True))
    with pytest.raises(ValueError, match="wrist decoder requires"):
        load_input(write_task(tmp_path / "wrist", wrist_limit=3.14))


def test_dataset_recipes_are_listed_without_a_default():
    assert {name: ik_settings(name)["mano_to_robot_scale"] for name in ("arctic", "oakink2", "hot3d", "taco")} == {
        "arctic": 1.0, "oakink2": 1.2, "hot3d": 1.2, "taco": 1.2}
    assert ik_settings("arctic")["wrist_offset_xyzw"] and ik_settings("taco")["wrist_offset_xyzw"] is None
    with pytest.raises(ValueError, match="No Sharpa IK recipe for dataset 'grab'"):
        ik_settings("grab")


def test_binding_identifies_the_task_and_rejects_other_inputs(tmp_path):
    binding_path = prepared(tmp_path, "hinged_tool_right_hand")
    binding, task = load_prepared(binding_path)
    assert (binding["format"], binding["task_id"], binding["hand_sides"]) == (FORMAT, task.task_id, ["right"])
    assert set(binding["files"]) == {"assets/scene.xml"}  # the task copy is sealed by its own identity
    # A preparation of the former human/robot bundles.
    write_json(binding_path, {**binding, "format": "flashchord-h2r-v1", "human_bundle_id": "0" * 64})
    with pytest.raises(ValueError, match="Unsupported FlashCHORD preparation 'flashchord-h2r-v1'.*prepare again"):
        load_prepared(binding_path)
    write_json(binding_path, {**binding, "task_id": "0" * 64})
    with pytest.raises(ValueError, match="different task"):
        load_prepared(binding_path)
    write_json(binding_path, binding)
    load_prepared(binding_path)
    with (binding_path.parent / "task/demo/reference.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed file"):
        load_prepared(binding_path)


def test_changed_method_file_is_rejected(tmp_path):
    binding_path = prepared(tmp_path, "single_rigid")
    with (binding_path.parent / "assets/scene.xml").open("a") as stream:
        stream.write("<!-- changed -->")
    with pytest.raises(ValueError, match="changed file"):
        load_prepared(binding_path)


@pytest.mark.parametrize("structure,hands", [("hinged_box", ["right", "left"]), ("hinged_tool_right_hand", ["right"])])
def test_task_hands_select_the_configured_hand_lists(tmp_path, structure, hands):
    binding_path = prepared(tmp_path, structure)
    assert binding_path.name == BINDING
    cfg = compose_config("train_flash_sac", [f"task.parquet='{binding_path}'"])
    require_h2r_input(cfg)
    assert list(cfg.env.observation.sides) == hands == list(cfg.observation.sides)
    assert list(cfg.env.observation.contact_sides) == sorted(hands)  # the native contact order is left, right
    assert list(cfg.env.objective.sides) == list(cfg.env.contact.sides) == hands
