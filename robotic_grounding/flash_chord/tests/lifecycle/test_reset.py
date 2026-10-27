# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for lifecycle reset terms."""

import numpy as np
import pytest


pytestmark = pytest.mark.gpu


def test_immediate_reference_reset_matches_explicit_start_and_preserves_other_worlds():
    import warp as wp
    from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

    with wp.ScopedDevice("cuda:0"):
        policy = ReferenceReset.build(2, 195, ReferenceResetConfig(voc_decay_steps=20))
        mask = wp.array([1, 0], dtype=wp.int32)
        scale = wp.array([0.0], dtype=wp.float32)
        policy.reset_frame.assign([17, 23])
        policy.finger_scale.assign([0.2, 0.3])
        policy.steps_since_reset.assign([5, 6])
        policy.applied_voc_scale.assign([1.0, 0.5])
        policy.set_immediate_first_frame_probability(1.0)
        policy.sample(mask, scale)
        np.testing.assert_array_equal(policy.reset_frame.numpy(), [0, 23])
        np.testing.assert_array_equal(policy.finger_scale.numpy(), np.array([1.0, 0.3], np.float32))
        np.testing.assert_array_equal(policy.steps_since_reset.numpy(), [20, 6])
        np.testing.assert_array_equal(policy.applied_voc_scale.numpy(), [0.0, 0.5])
        np.testing.assert_array_equal(policy.reset_count.numpy(), [1, 0])
        phase = wp.array([0, 23], dtype=wp.int32)
        age = wp.zeros(2, dtype=wp.int32)
        policy.advance(wp.array([0, 1], dtype=wp.int32), wp.zeros(2, dtype=wp.int32), phase, age, scale)
        np.testing.assert_array_equal(phase.numpy(), [1, 23])
        np.testing.assert_array_equal(policy.applied_voc_scale.numpy(), [0.0, 0.5])
        assert policy.set_immediate_first_frame_probability(None) == 0.0
        with pytest.raises(ValueError, match="immediate_first_frame_probability"):
            policy.set_immediate_first_frame_probability(float("nan"))


def test_zero_curriculum_still_has_legacy_reset_assistance_unless_immediate_is_selected():
    import warp as wp
    from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

    with wp.ScopedDevice("cuda:0"):
        policy = ReferenceReset.build(32, 195, ReferenceResetConfig(always_reset_to_first_frame=True))
        policy.sample(wp.ones(32, dtype=wp.int32), wp.zeros(1, dtype=wp.float32))
        np.testing.assert_array_equal(policy.reset_frame.numpy(), 0)
        np.testing.assert_array_equal(policy.steps_since_reset.numpy(), 0)
        np.testing.assert_array_equal(policy.applied_voc_scale.numpy(), 1)
        assert (policy.finger_scale.numpy() < 0.7).all()


def test_reset_places_wrists_and_objects_at_reference(h2r_setup):
    import warp as wp

    from flash_chord.lifecycle.reset import reset_scene_to_frame

    with wp.ScopedDevice("cuda:0"):
        setup = h2r_setup(1)
        scene, ref = setup.scene, setup.reference
        state = scene.model.state()
        reset_scene_to_frame(scene, ref, 0, state)
        body_q = state.body_q.numpy()

    # each wrist body (hand_C_MC) snaps to the reference wrist position
    for hand in scene.layout.hands:
        ref_pos = ref.wrist_pos_w(hand.side)[0]
        np.testing.assert_allclose(body_q[hand.wrist_body_id][:3], ref_pos, atol=2e-3)

    # objects snap to their reference frame-0 positions
    obj_pos = ref.object_body_pos_w()[0]
    for binding in scene.objects:
        for body in binding.bodies:
            np.testing.assert_allclose(
                body_q[body.body_id][:3],
                obj_pos[body.reference_body_id],
                atol=2e-3,
            )


def test_masked_reset_changes_only_selected_world_and_zeros_velocity(h2r_setup):
    import warp as wp

    from flash_chord.lifecycle.reset import ReferenceResetTable, reset_worlds

    with wp.ScopedDevice("cuda:0"):
        setup = h2r_setup(2)
        scene, ref = setup.scene, setup.reference
        state = scene.model.state()
        table = ReferenceResetTable.build(scene, ref)
        state.joint_qd.assign(np.ones(scene.model.joint_dof_count, dtype=np.float32))
        joint_q_before = state.joint_q.numpy().reshape(2, table.per_world_q).copy()
        reset_worlds(
            scene.model,
            table,
            state,
            reset_frame=wp.array([0, 1], dtype=wp.int32),
            reset_mask=wp.array([1, 0], dtype=wp.int32),
        )
        joint_q = state.joint_q.numpy().reshape(2, table.per_world_q)
        joint_qd = state.joint_qd.numpy().reshape(2, table.per_world_dof)

    np.testing.assert_allclose(joint_q[0], table.ref_joint_q.numpy()[0])
    np.testing.assert_allclose(joint_qd[0], 0.0)
    np.testing.assert_allclose(joint_q[1], joint_q_before[1])
    np.testing.assert_allclose(joint_qd[1], 1.0)


def test_h2r_reset_uses_bound_named_joint_and_semantic_frame_trajectories(h2r_setup):
    import warp as wp

    from flash_chord.lifecycle.reset import ReferenceResetTable, reset_worlds

    frame = 100
    with wp.ScopedDevice("cuda:0"):
        setup = h2r_setup(1)
        scene, reference = setup.scene, setup.reference
        table = ReferenceResetTable.build(scene, reference)
        state = scene.model.state()
        state.joint_qd.assign(np.ones(scene.model.joint_dof_count, dtype=np.float32))
        reset_worlds(scene.model, table, state, frame_id=frame)
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        body_q = state.body_q.numpy()

    np.testing.assert_array_equal(
        joint_q[: scene.layout.num_joint_q],
        scene.robot_reference.joint_q[frame],
    )
    np.testing.assert_array_equal(joint_qd, 0.0)
    for side in scene.layout.sides:
        hand_layout = scene.layout.hand(side)
        hand_reference = scene.robot_reference.hand(side)
        np.testing.assert_allclose(
            body_q[hand_layout.palm_frame.body_id, :3],
            hand_reference.wrist_pos_w[frame],
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            body_q[list(hand_layout.fingertip_body_ids), :3],
            hand_reference.dp_pos_w[frame],
            atol=2.0e-6,
        )
    assert len(scene.objects) == 1 and len(scene.objects[0].articulations) == 1
    articulation = scene.objects[0].articulations[0]
    np.testing.assert_allclose(
        joint_q[articulation.q_id],
        reference.object_articulation()[frame, articulation.reference_id],
    )


def test_reference_reset_samples_random_frames_open_fingers_and_first_frame_override():
    import warp as wp

    from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

    world_count = 1024
    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count,
            num_frames=11,
            config=ReferenceResetConfig(seed=7),
        )
        mask = wp.ones(world_count, dtype=wp.int32)
        curriculum_voc = wp.array([1.0], dtype=wp.float32)
        reset.sample(mask, curriculum_voc)

        frames = reset.reset_frame.numpy()
        finger_scale = reset.finger_scale.numpy()
        assert frames.min() >= 0
        assert frames.max() <= 9
        assert np.unique(frames).size > 1
        assert np.all((finger_scale >= 0.0) & (finger_scale <= 0.7))
        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 1.0)
        np.testing.assert_array_equal(reset.steps_since_reset.numpy(), 0)
        np.testing.assert_array_equal(reset.reset_count.numpy(), 1)

        first_frame = ReferenceReset.build(
            world_count,
            num_frames=11,
            config=ReferenceResetConfig(
                reset_to_first_frame_probability=1.0,
                first_frame_voc_threshold=0.1,
                seed=7,
            ),
        )
        first_frame.sample(mask, wp.array([0.0], dtype=wp.float32))
        np.testing.assert_array_equal(first_frame.reset_frame.numpy(), 0)


def test_reference_reset_curriculum_probability_updates_inside_existing_graph():
    import warp as wp

    from flash_chord.lifecycle.reset import FirstFrameResetCurriculum, ReferenceReset, ReferenceResetConfig

    world_count = 1024
    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count,
            num_frames=11,
            config=ReferenceResetConfig(reset_to_first_frame_probability=0.0, seed=7),
        )
        mask = wp.ones(world_count, dtype=wp.int32)
        curriculum_voc = wp.array([0.0], dtype=wp.float32)
        with wp.ScopedCapture("cuda:0") as capture:
            reset.sample(mask, curriculum_voc)

        assert isinstance(reset, FirstFrameResetCurriculum)
        assert reset.set_reset_to_first_frame_probability(1.0) == 1.0
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(reset.reset_frame.numpy(), 0)

        assert reset.set_reset_to_first_frame_probability(None) == 0.0
        wp.capture_launch(capture.graph)
        frames = reset.reset_frame.numpy()
        assert np.unique(frames).size > 1
        assert np.any(frames != 0)
        np.testing.assert_array_equal(reset.first_frame_probability.numpy(), [0.0])

        with pytest.raises(ValueError, match="reset_to_first_frame_probability"):
            reset.set_reset_to_first_frame_probability(float("nan"))


def test_reference_reset_holds_reference_and_voc_before_advancing():
    import warp as wp

    from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count=2,
            num_frames=10,
            config=ReferenceResetConfig(voc_decay_steps=3, seed=3),
        )
        mask = wp.ones(2, dtype=wp.int32)
        curriculum_voc = wp.array([0.25], dtype=wp.float32)
        reset.sample(mask, curriculum_voc)
        timestep = wp.array([4, 6], dtype=wp.int32)
        episode_step = wp.zeros(2, dtype=wp.int32)
        terminated = wp.zeros(2, dtype=wp.int32)
        truncated = wp.zeros(2, dtype=wp.int32)

        reset.advance(terminated, truncated, timestep, episode_step, curriculum_voc)
        reset.advance(terminated, truncated, timestep, episode_step, curriculum_voc)
        np.testing.assert_array_equal(timestep.numpy(), [4, 6])
        np.testing.assert_array_equal(episode_step.numpy(), [2, 2])
        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 1.0)

        reset.advance(terminated, truncated, timestep, episode_step, curriculum_voc)
        np.testing.assert_array_equal(timestep.numpy(), [5, 7])
        np.testing.assert_array_equal(episode_step.numpy(), [3, 3])
        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 0.25)

        reset.prepare_explicit(mask, curriculum_voc)
        np.testing.assert_allclose(reset.finger_scale.numpy(), 1.0)
        np.testing.assert_array_equal(reset.steps_since_reset.numpy(), 3)
        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 0.25)


def test_reference_reset_zero_decay_applies_target_voc_during_sample():
    import warp as wp

    from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count=2,
            num_frames=10,
            config=ReferenceResetConfig(voc_decay_steps=0),
        )
        mask = wp.ones(2, dtype=wp.int32)
        target = wp.array([0.25], dtype=wp.float32)

        reset.sample(mask, target)

        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 0.25)
        np.testing.assert_array_equal(reset.steps_since_reset.numpy(), 0)


def test_reference_reset_linear_decay_is_configurable_and_validated():
    import warp as wp

    from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

    with pytest.raises(ValueError, match="reset_finger_openness"):
        ReferenceResetConfig(reset_finger_openness=1.1)
    with pytest.raises(ValueError, match="voc_decay_mode"):
        ReferenceResetConfig(voc_decay_mode="invalid")  # type: ignore[arg-type]

    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count=1,
            num_frames=10,
            config=ReferenceResetConfig(voc_decay_steps=4, voc_decay_mode="linear"),
        )
        mask = wp.ones(1, dtype=wp.int32)
        curriculum_voc = wp.array([0.0], dtype=wp.float32)
        timestep = wp.array([3], dtype=wp.int32)
        episode_step = wp.zeros(1, dtype=wp.int32)
        done = wp.zeros(1, dtype=wp.int32)
        reset.sample(mask, curriculum_voc)
        reset.advance(done, done, timestep, episode_step, curriculum_voc)
        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 0.75)
        np.testing.assert_array_equal(timestep.numpy(), 3)


def test_reference_reset_publishes_applied_target_and_normalized_progress():
    import warp as wp

    from flash_chord.lifecycle.reset import RESET_CONTEXT_NAMES, ReferenceReset, ReferenceResetConfig, ResetContext

    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count=4,
            num_frames=10,
            config=ReferenceResetConfig(voc_decay_steps=4),
        )
        reset.applied_voc_scale.assign(np.asarray([1.0, 0.75, 0.25, 0.25], dtype=np.float32))
        reset.steps_since_reset.assign(np.asarray([0, 1, 4, 7], dtype=np.int32))
        target = wp.array([0.25], dtype=wp.float32)
        context = wp.zeros(4 * len(RESET_CONTEXT_NAMES), dtype=wp.float32)
        result = reset.compute_context(target, context)

        assert isinstance(reset, ResetContext)
        assert reset.context_names == RESET_CONTEXT_NAMES
        assert result is context
        np.testing.assert_allclose(
            context.numpy().reshape(4, 3),
            [
                [1.0, 0.25, 0.0],
                [0.75, 0.25, 0.25],
                [0.25, 0.25, 1.0],
                [0.25, 0.25, 1.0],
            ],
        )
        with pytest.raises(ValueError, match="reset context output has shape"):
            reset.compute_context(target, wp.zeros(11, dtype=wp.float32))

        zero_decay = ReferenceReset.build(
            world_count=1,
            num_frames=10,
            config=ReferenceResetConfig(voc_decay_steps=0),
        )
        mask = wp.ones(1, dtype=wp.int32)
        zero_decay.sample(mask, target)
        zero_context = wp.zeros(3, dtype=wp.float32)
        zero_decay.compute_context(target, zero_context)
        np.testing.assert_allclose(zero_context.numpy(), [0.25, 0.25, 1.0])

        zero_decay.prepare_explicit(mask, target)
        zero_decay.compute_context(target, zero_context)
        np.testing.assert_allclose(zero_context.numpy(), [0.25, 0.25, 1.0])
