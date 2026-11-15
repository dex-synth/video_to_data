# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for setup-time robot reference binding and semantic FK."""

from dataclasses import replace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _common_robot(binding):
    from flash_chord.h2r.builder import CommonModelBuilder
    from flash_chord.h2r.robot import H2rSharpaHands

    builder = CommonModelBuilder(gravity=-9.81)
    layout = H2rSharpaHands(binding).build(builder)
    return builder, layout


def _trajectory(builder):
    first = np.asarray(builder.joint_q, dtype=np.float32)
    second = np.linspace(-0.1, 0.1, len(first), dtype=np.float32)
    return np.stack((first, second))


def test_binding_exposes_readonly_dp_and_fingertip_trajectories(h2r_binding, monkeypatch):
    import newton

    from flash_chord.embodiments.binding import build_robot_reference_binding

    builder, layout = _common_robot(h2r_binding)
    monkeypatch.setattr(
        newton.ModelBuilder,
        "replicate",
        lambda *_args, **_kwargs: pytest.fail("reference FK must not replicate collision geometry"),
    )
    joint_q = _trajectory(builder)
    binding = build_robot_reference_binding(builder, layout, joint_q, joint_q * 2.0, fps=20.0)

    assert binding.num_frames == 2
    assert binding.num_joint_q == binding.num_joint_dof == layout.num_joint_q
    assert binding.fps == 20.0
    np.testing.assert_array_equal(binding.joint_q, joint_q)
    np.testing.assert_array_equal(binding.joint_target, joint_q * 2.0)
    assert not binding.joint_q.flags.writeable
    assert not binding.joint_target.flags.writeable

    for side in layout.sides:
        hand = binding.hand(side)
        assert hand.wrist_pos_w.shape == (2, 3)
        assert hand.wrist_quat_w.shape == (2, 4)
        assert hand.arm_joint_pos.shape == (2, 6)
        np.testing.assert_array_equal(hand.arm_joint_pos, joint_q[:, layout.hand(side).arm_q_ids])
        assert hand.finger_joint_pos.shape == (2, len(layout.hand(side).finger_joint_names))
        assert hand.dp_pos_w.shape == hand.fingertip_pos_w.shape == (2, 5, 3)
        assert hand.dp_quat_w.shape == hand.fingertip_quat_w.shape == (2, 5, 4)
        assert hand.keypoint_pos_w("dp").shape == (2, 6, 3)
        assert hand.keypoint_pos_w("fingertip").shape == (2, 6, 3)
        distance = np.linalg.norm(hand.dp_pos_w - hand.fingertip_pos_w, axis=-1)
        assert np.all(distance > 0.025)
        assert all(
            not array.flags.writeable
            for array in (
                hand.wrist_pos_w,
                hand.wrist_quat_w,
                hand.arm_joint_pos,
                hand.finger_joint_pos,
                hand.dp_pos_w,
                hand.dp_quat_w,
                hand.fingertip_pos_w,
                hand.fingertip_quat_w,
            )
        )

    with pytest.raises(ValueError, match="must be 'dp' or 'fingertip'"):
        binding.hand(layout.sides[0]).keypoint_pos_w("distal")
    with pytest.raises(KeyError, match="no bound hand"):
        binding.hand("center")


def test_device_reference_preserves_named_binding_and_requested_side_order(h2r_binding):
    import warp as wp

    from flash_chord.embodiments.binding import DeviceRobotReference, build_robot_reference_binding

    builder, layout = _common_robot(h2r_binding)
    joint_q = _trajectory(builder)
    binding = build_robot_reference_binding(builder, layout, joint_q, joint_q * 2.0, fps=20.0)
    sides = tuple(reversed(layout.sides))  # the requested order, not the embodiment's
    hands = [binding.hand(side) for side in sides]
    with wp.ScopedDevice("cuda:0"):
        device_reference = DeviceRobotReference.build(binding, layout, sides=sides)
        joint_target = device_reference.joint_target.numpy().reshape(2, layout.num_joint_dof)
        arm_joint_pos = device_reference.arm_joint_pos.numpy().reshape(2, -1)
        finger_joint_pos = device_reference.finger_joint_pos.numpy().reshape(2, -1)
        wrist_pos = device_reference.wrist_pos_w.numpy().reshape(2, len(sides), 3)
        wrist_quat = device_reference.wrist_quat_w.numpy().reshape(2, len(sides), 4)

    assert device_reference.sides == sides
    assert device_reference.arm_counts == tuple(hand.arm_joint_pos.shape[1] for hand in hands)
    assert device_reference.finger_counts == tuple(hand.finger_joint_pos.shape[1] for hand in hands)
    np.testing.assert_array_equal(joint_target, binding.joint_target)
    np.testing.assert_array_equal(arm_joint_pos, np.concatenate([hand.arm_joint_pos for hand in hands], axis=1))
    np.testing.assert_array_equal(finger_joint_pos, np.concatenate([hand.finger_joint_pos for hand in hands], axis=1))
    np.testing.assert_array_equal(wrist_pos[:, 0], hands[0].wrist_pos_w)
    np.testing.assert_array_equal(wrist_quat[:, 0], hands[0].wrist_quat_w[:, (1, 2, 3, 0)])

    with pytest.raises(ValueError, match="device reference sides"):
        DeviceRobotReference.build(binding, layout, sides=("center",))
    bad_hand = replace(hands[0], arm_joint_pos=np.empty((2, 7), dtype=np.float32))
    bad_binding = replace(binding, hands=(bad_hand, *hands[1:]))
    with pytest.raises(ValueError, match=f"{sides[0]} arm_joint_pos has width"):
        DeviceRobotReference.build(bad_binding, layout)


def test_bound_reference_dataclasses_normalize_and_reject_invalid_ownership(h2r_binding):
    from flash_chord.embodiments.binding import build_robot_reference_binding

    builder, layout = _common_robot(h2r_binding)
    joint_q = _trajectory(builder)
    binding = build_robot_reference_binding(builder, layout, joint_q, joint_q, fps=20.0)
    left = binding.hand(layout.sides[0])

    scaled_quaternion = replace(left, wrist_quat_w=left.wrist_quat_w * 2.0)
    np.testing.assert_allclose(np.linalg.norm(scaled_quaternion.wrist_quat_w, axis=1), 1.0)
    assert not scaled_quaternion.wrist_quat_w.flags.writeable

    invalid_fingers = left.finger_joint_pos.copy()
    invalid_fingers[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        replace(left, finger_joint_pos=invalid_fingers)
    with pytest.raises(ValueError, match="sides must be unique"):
        replace(binding, hands=(left, left))
    with pytest.raises(ValueError, match="declares 3 frames"):
        replace(binding, num_frames=3)


def test_named_joint_trajectory_is_permutation_invariant(h2r_binding):
    from flash_chord.embodiments.binding import build_named_joint_trajectory

    builder, layout = _common_robot(h2r_binding)
    scalar = layout.scalar_joints
    lower = np.asarray(builder.joint_limit_lower, dtype=np.float32)[list(scalar.dof_ids)]
    upper = np.asarray(builder.joint_limit_upper, dtype=np.float32)[list(scalar.dof_ids)]
    reference_values = np.stack((0.5 * lower + 0.5 * upper, 0.25 * lower + 0.75 * upper))
    permutation = np.arange(len(scalar.names))[::-1]
    joint_q, joint_target = build_named_joint_trajectory(
        builder,
        layout,
        [scalar.names[index] for index in permutation],
        reference_values[:, permutation],
        joint_limit_tolerance=0.0,
    )

    np.testing.assert_array_equal(joint_q[:, scalar.q_ids], reference_values)
    np.testing.assert_array_equal(joint_target[:, scalar.dof_ids], reference_values)


def test_named_joint_trajectory_rejects_nonbijective_or_incomplete_layouts(h2r_binding):
    from flash_chord.embodiments.binding import build_named_joint_trajectory

    builder, layout = _common_robot(h2r_binding)
    names = list(layout.scalar_joints.names)
    joint_pos = np.zeros((2, len(names)), dtype=np.float32)

    with pytest.raises(ValueError, match="duplicate joints"):
        build_named_joint_trajectory(
            builder,
            layout,
            [names[0], *names[1:-1], names[0]],
            joint_pos,
            joint_limit_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="missing=.*unexpected"):
        build_named_joint_trajectory(
            builder,
            layout,
            [*names[:-1], "unexpected_joint"],
            joint_pos,
            joint_limit_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="does not define named scalar joints"):
        build_named_joint_trajectory(
            builder,
            replace(layout, scalar_joints=None),
            names,
            joint_pos,
            joint_limit_tolerance=0.0,
        )


def test_named_joint_trajectory_preserves_q_and_clamps_only_controller_limit_noise(h2r_binding):
    from flash_chord.embodiments.binding import build_named_joint_trajectory

    builder, layout = _common_robot(h2r_binding)
    scalar = layout.scalar_joints
    lower = np.asarray(builder.joint_limit_lower, dtype=np.float32)[list(scalar.dof_ids)]
    upper = np.asarray(builder.joint_limit_upper, dtype=np.float32)[list(scalar.dof_ids)]
    source = (0.5 * lower + 0.5 * upper)[None, :]
    source[0, 0] = upper[0] + 5.0e-6
    source[0, 1] = lower[1] - 5.0e-6

    joint_q, joint_target = build_named_joint_trajectory(
        builder,
        layout,
        scalar.names,
        source,
        joint_limit_tolerance=1.0e-5,
    )

    np.testing.assert_array_equal(joint_q[:, scalar.q_ids], source)
    assert joint_target[0, scalar.dof_ids[0]] == upper[0]
    assert joint_target[0, scalar.dof_ids[1]] == lower[1]
    with pytest.raises(ValueError, match="named joint .* frame 0 .* tolerance"):
        build_named_joint_trajectory(
            builder,
            layout,
            scalar.names,
            source,
            joint_limit_tolerance=1.0e-7,
        )
    for invalid_tolerance in (-1.0, np.nan):
        with pytest.raises(ValueError, match="tolerance must be nonnegative and finite"):
            build_named_joint_trajectory(
                builder,
                layout,
                scalar.names,
                source,
                joint_limit_tolerance=invalid_tolerance,
            )


@pytest.mark.parametrize(
    ("missing_coordinates", "target_frames", "fps", "message"),
    [
        (1, 2, 20.0, "joint_q must have shape"),
        (0, 3, 20.0, "joint_q has 2 frames"),
        (0, 2, 0.0, "fps must be positive"),
    ],
)
def test_binding_rejects_invalid_trajectory_contract(h2r_binding, missing_coordinates, target_frames, fps, message):
    from flash_chord.embodiments.binding import build_robot_reference_binding

    builder, layout = _common_robot(h2r_binding)
    joint_q = np.zeros((2, layout.num_joint_q - missing_coordinates), dtype=np.float32)
    joint_target = np.zeros((target_frames, layout.num_joint_dof), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        build_robot_reference_binding(builder, layout, joint_q, joint_target, fps)
