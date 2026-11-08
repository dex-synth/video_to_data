# SPDX-License-Identifier: Apache-2.0
"""Physical capsule dimensions must survive the Newton/MuJoCo boundary."""

import mujoco
import pytest

from flash_chord.h2r.validation import validate_geometry


def test_capsule_padding_is_ignored_but_radius_and_length_are_checked():
    robot_body = '<body name="robot"><freejoint/><geom name="hand" type="box" size=".02 .02 .02"/></body>'
    object_body = '<body name="rod"><freejoint/><geom name="capsule" type="capsule" size=".0033 .0926"/></body>'
    table = '<geom name="table" type="box" size=".5 .5 .01" pos="0 0 -.01"/>'

    def model(contents):
        return mujoco.MjModel.from_xml_string('<mujoco><worldbody>' + contents + '</worldbody></mujoco>')

    robot = model(robot_body)
    source = model(object_body + table)
    actual = model(robot_body + object_body + '<body name="support">' + table + '</body>')
    capsule = actual.geom("capsule").id
    actual.geom_size[capsule, 2] = .0033
    validate_geometry(actual, robot, source)
    for dimension in (0, 1):
        original = actual.geom_size[capsule, dimension]
        actual.geom_size[capsule, dimension] += .001
        with pytest.raises(ValueError, match="capsule.size"):
            validate_geometry(actual, robot, source)
        actual.geom_size[capsule, dimension] = original
