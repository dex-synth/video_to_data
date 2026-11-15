# SPDX-License-Identifier: Apache-2.0
"""The task's Sharpa hands with scalar virtual wrists, without the native floating ball wrist."""

from pathlib import Path

import newton
import numpy as np

from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout, HandLayout, ScalarJointLayout
from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.embodiments.frames import capture_body_ids
from flash_chord.embodiments.sharpa_collision import capture_sharpa_link_geometry, sharpa_object_contact_links

from .inputs import frame_names, load_prepared
from .reference import H2rReference
from .wrist import wrist_coordinates


def joint_ids(builder):
    return {
        label.rsplit("/", 1)[-1]: j
        for j, label in enumerate(builder.joint_label)
        if builder.joint_type[j] != newton.JointType.FIXED
    }


class H2rSharpaHands:
    name = "h2r_sharpa_hands"

    def __init__(self, binding_path):
        self.binding_path = Path(binding_path).resolve()
        robot = load_prepared(self.binding_path)[1].robot
        self.semantics, self.sides = robot.semantics, tuple(robot.hands)

    def build(self, builder):
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.balance_inertia = False
        builder.add_mjcf(
            str(self.binding_path.parent / "assets" / "robot.xml"),
            collapse_fixed_joints=False,
            parse_mujoco_options=False,
            mesh_maxhullvert=64,
        )
        names = [name for side in self.sides for name in frame_names(side)]
        bodies = capture_body_ids(builder, names)
        geometry = capture_sharpa_link_geometry(
            builder,
            palm_body_ids={s: bodies[frame_names(s)[0]] for s in self.sides},
            sides=self.sides,
            object_contact_links=sharpa_object_contact_links(),
        )
        # Preserve imported collision exclusions. Do not apply the native hand's extra filters.
        frames = {name: BodyFrame(name=name, body_id=body) for name, body in bodies.items()}
        joints = joint_ids(builder)
        if set(joints) != set(self.semantics["joint_names"]):
            raise ValueError("Newton changed the common joint set")
        hands = []
        for side in self.sides:
            hand = self.semantics["hands"][side]
            if hand["wrist_body"] != frame_names(side)[0]:
                raise ValueError("The robot's wrist body is not the palm frame of the native IK")
            arm = hand["wrist_translation_joints"] + hand["wrist_rotation_joints"]
            fingers = hand["finger_joints"]

            def qids(names):
                return tuple(builder.joint_q_start[joints[n]] for n in names)

            def dofs(names):
                return tuple(builder.joint_qd_start[joints[n]] for n in names)

            hands.append(
                HandLayout(
                    side=side,
                    palm_frame=frames[frame_names(side)[0]],
                    dp_frames=tuple(frames[n] for n in frame_names(side)[1:6]),
                    fingertip_frames=tuple(frames[n] for n in frame_names(side)[6:]),
                    arm_q_ids=qids(arm),
                    arm_dof_ids=dofs(arm),
                    arm_joint_names=tuple(arm),
                    finger_q_ids=qids(fingers),
                    finger_dof_ids=dofs(fingers),
                    finger_joint_names=tuple(fingers),
                    link_geometry=geometry[side],
                )
            )
        ordered = sorted(joints, key=lambda n: builder.joint_q_start[joints[n]])
        return EmbodimentLayout(
            len(builder.joint_q),
            builder.joint_dof_count,
            tuple(hands),
            ScalarJointLayout(
                tuple(builder.joint_q_start[joints[n]] for n in ordered),
                tuple(builder.joint_qd_start[joints[n]] for n in ordered),
                tuple(ordered),
            ),
        )

    def bind_reference(self, builder, layout, reference):
        if not isinstance(reference, H2rReference) or reference.binding_path != self.binding_path:
            raise ValueError("H2R robot and reference must use the same binding")
        with np.load(self.binding_path.parent / "robot_reference.npz", allow_pickle=False) as data:
            names = list(data["joint_names"])
            q = data["joint_q"][:, [names.index(n) for n in layout.scalar_joints.names]].astype(np.float32)
        if reference.metadata.is_resampled:
            initial = q[0].copy()
            q = np.zeros((reference.num_frames, layout.num_joint_q), dtype=np.float32)
            for hand in layout.hands:
                base = reference.wrist_pos_w(hand.side)[0] - initial[list(hand.arm_q_ids[:3])]
                previous = initial[list(hand.arm_q_ids[3:])]
                for t, (position, orientation) in enumerate(zip(
                        reference.wrist_pos_w(hand.side), reference.wrist_quat_w(hand.side), strict=True)):
                    wrist = wrist_coordinates(position, orientation, base, previous)
                    q[t, hand.arm_q_ids] = wrist
                    previous = wrist[3:]
                columns = [reference.finger_joint_names(hand.side).index(n) for n in hand.finger_joint_names]
                q[:, hand.finger_q_ids] = reference.finger_joint_pos(hand.side)[:, columns]
        if q.shape != (reference.num_frames, layout.num_joint_q):
            raise ValueError("Common robot reference must have every robot coordinate at every frame")
        binding = build_robot_reference_binding(builder, layout, q, q.copy(), reference.fps)
        for hand in layout.hands:
            bound = binding.hand(hand.side)
            np.testing.assert_allclose(bound.wrist_pos_w, reference.wrist_pos_w(hand.side), atol=2e-6, rtol=0)
            alignment = np.abs(np.sum(bound.wrist_quat_w * reference.wrist_quat_w(hand.side), axis=-1))
            np.testing.assert_allclose(alignment, 1.0, atol=2e-6, rtol=0)
            if reference.frame_names(hand.side) != frame_names(hand.side):
                raise ValueError("Common robot frame order changed")
            frames = reference.frame_pos_w(hand.side)
            # At source instants FK must still match the sealed IK. Between
            # samples, use FK of interpolated wrist/fingers, as native Sharpa
            # binding does; linearly interpolated link positions are not FK.
            source_frames = reference.sample_times_s * reference.metadata.source_fps
            exact = np.isclose(source_frames, np.round(source_frames), atol=1e-9, rtol=0)
            np.testing.assert_allclose(bound.dp_pos_w[exact], frames[exact, 1:6], atol=2e-6, rtol=0)
            np.testing.assert_allclose(bound.fingertip_pos_w[exact], frames[exact, 6:], atol=2e-6, rtol=0)
        return binding
