# SPDX-License-Identifier: Apache-2.0
"""Physical geometry of the task's scene must survive the Newton/MuJoCo boundary."""

import mujoco
import pytest

from flash_chord.h2r.validation import validate_geometry

ROBOT = '<body name="robot"><freejoint/><geom name="hand" type="box" size=".02 .02 .02"/></body>'
ROD = '<body name="rod"><freejoint/><geom name="capsule" type="capsule" size=".0033 .0926"/></body>'
TABLE = '<geom name="table" type="box" size=".5 .5 .01" pos="0 0 -.01"/>'
SHELF = '<geom name="shelf" type="box" size=".1 .1 .01" pos=".1 0 0" quat="0.9238795 0 0 0.3826834"/>'


def model(contents):
    return mujoco.MjModel.from_xml_string("<mujoco><worldbody>" + contents + "</worldbody></mujoco>")


def scenes():
    """The task scene (world table, fixed shelf body) and its import with one support body."""
    task = model(TABLE + ROD + ROBOT + '<body name="fixed" pos="0 .2 .1">' + SHELF + "</body>")
    shelf = SHELF.replace('pos=".1 0 0"', 'pos=".1 .2 .1"')
    return model(ROBOT + ROD + '<body name="support">' + TABLE + shelf + "</body>"), task


def test_capsule_padding_is_ignored_but_radius_and_length_are_checked():
    actual, task = scenes()
    capsule = actual.geom("capsule").id
    actual.geom_size[capsule, 2] = .0033
    validate_geometry(actual, task, {"robot"})
    for dimension in (0, 1):
        original = actual.geom_size[capsule, dimension]
        actual.geom_size[capsule, dimension] += .001
        with pytest.raises(ValueError, match="capsule.size"):
            validate_geometry(actual, task, {"robot"})
        actual.geom_size[capsule, dimension] = original


@pytest.mark.parametrize("field,label", [("geom_pos", "shelf.world_pos"), ("geom_friction", "shelf.friction")])
def test_static_environment_placement_and_contact_are_checked(field, label):
    actual, task = scenes()
    getattr(actual, field)[actual.geom("shelf").id, 0] += .01
    with pytest.raises(ValueError, match=label):
        validate_geometry(actual, task, {"robot"})


def test_missing_environment_geometry_is_rejected():
    _, task = scenes()
    with pytest.raises(ValueError, match="added or removed"):
        validate_geometry(model(ROBOT + ROD + '<body name="support">' + TABLE + "</body>"), task, {"robot"})
