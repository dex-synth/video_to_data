# SPDX-License-Identifier: Apache-2.0
"""Keep native hand-pose residuals, then decode the common serial wrist targets."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from flash_chord.runtime.actions.hand_pose import ActionBuffers, ActionConfig, ResidualHandPoseAction
from flash_chord.runtime.delay import TargetDelayBuffer


@wp.func
def near_angle(angle: float, previous: float):
    result = angle + 2.0 * wp.pi * wp.round((previous - angle) / (2.0 * wp.pi))
    if result > 6.2:
        result -= 2.0 * wp.pi
    if result < -6.2:
        result += 2.0 * wp.pi
    return result


@wp.kernel
def decode_wrist(
    processed: wp.array(dtype=wp.float32),
    q: wp.array(dtype=wp.float32),
    processed_starts: wp.array(dtype=wp.int32),
    q_ids: wp.array(dtype=wp.int32),
    pos_ids: wp.array(dtype=wp.int32),
    ori_ids: wp.array(dtype=wp.int32),
    base_pos: wp.array(dtype=wp.vec3),
    num_hands: int,
    processed_dim: int,
    q_per_world: int,
    dofs_per_world: int,
    target: wp.array(dtype=wp.float32),
):
    world, hand = wp.tid()
    p = world * processed_dim + processed_starts[hand]
    quat = wp.normalize(wp.quat(processed[p + 4], processed[p + 5], processed[p + 6], processed[p + 3]))
    r = wp.quat_to_matrix(quat)
    previous = wp.vec3(
        q[world * q_per_world + q_ids[hand * 6 + 3]],
        q[world * q_per_world + q_ids[hand * 6 + 4]],
        q[world * q_per_world + q_ids[hand * 6 + 5]],
    )
    b = wp.atan2(r[0, 2], wp.sqrt(r[0, 0] * r[0, 0] + r[0, 1] * r[0, 1]))
    a = wp.atan2(-r[1, 2], r[2, 2])
    c = wp.atan2(-r[0, 1], r[0, 0])
    if wp.sqrt(r[0, 0] * r[0, 0] + r[0, 1] * r[0, 1]) < 1.0e-6:
        a = previous[0]
        c = wp.atan2(r[1, 0], r[1, 1]) - wp.sign(b) * a
    first = wp.vec3(near_angle(a, previous[0]), near_angle(b, previous[1]), near_angle(c, previous[2]))
    second = wp.vec3(
        near_angle(a + wp.pi, previous[0]), near_angle(wp.pi - b, previous[1]), near_angle(c + wp.pi, previous[2])
    )
    cost1 = wp.length_sq(first - previous)
    cost2 = wp.length_sq(second - previous)
    for axis in range(3):
        if wp.abs(first[axis]) > 6.2:
            cost1 += 1.0e6
        if wp.abs(second[axis]) > 6.2:
            cost2 += 1.0e6
    angles = first
    if cost2 < cost1:
        angles = second
    for axis in range(3):
        target[world * dofs_per_world + pos_ids[hand * 3 + axis]] = processed[p + axis] - base_pos[hand][axis]
        target[world * dofs_per_world + ori_ids[hand * 3 + axis]] = angles[axis]


class PositionController:
    """Position targets only: the solver provides the common servo at each substep."""

    def write_position_targets(self, control, targets):
        wp.copy(control.joint_target_pos, targets)

    def apply_wrist_effort(self, state, control, targets):
        pass


@dataclass(frozen=True)
class H2rActionConfig(ActionConfig):
    def build(self, scene, device=None):
        return H2rHandPoseAction.build(scene, self, device)


@dataclass
class H2rHandPoseAction(ResidualHandPoseAction):
    wrist_q_ids: wp.array | None = None
    base_pos: wp.array | None = None
    q_per_world: int = 0

    @classmethod
    def build(cls, scene, config=None, device=None):
        config = config or H2rActionConfig()
        sides = ("right", "left")
        dofs = scene.model.joint_dof_count // scene.world_count
        mappings = {s: scene.layout.hand(s).arm_dof_ids for s in sides}
        buffers = ActionBuffers.build(
            scene.layout,
            scene.robot_reference,
            scene.world_count,
            config,
            num_joint_dof=dofs,
            device=device,
            wrist_dof_ids=mappings,
        )
        delay = None
        if config.delay.max_steps > 0:
            delay = TargetDelayBuffer.build(
                world_count=scene.world_count,
                num_joint_dof=dofs,
                delayed_dof_ids=tuple(range(56)),
                config=config.delay,
                device=device,
            )
        # Validate this Sharpa-specific chain against imported FK before using analytic decoding.
        import newton

        state = scene.model.state()
        zero_values = scene.model.joint_q.numpy().reshape(scene.world_count, -1)
        zero_values[:, : scene.layout.num_joint_q] = 0
        zero = wp.array(zero_values.reshape(-1), dtype=wp.float32, device=device)
        newton.eval_fk(scene.model, zero, scene.model.joint_qd, state)
        body_q = state.body_q.numpy()
        bases = []
        for side in sides:
            frame = scene.layout.hand(side).palm_frame
            transform = wp.transform_multiply(
                wp.transform(*body_q[frame.body_id]),
                wp.transform(wp.vec3(*frame.body_to_frame_pos), wp.quat(*frame.body_to_frame_quat_xyzw)),
            )
            bases.append(list(transform.p))
            np.testing.assert_allclose(transform.q, [0, 0, 0, 1], atol=1e-6)
            names = scene.layout.hand(side).arm_joint_names
            if names != tuple(f"{side}_{a}_joint" for a in ("x", "y", "z", "roll", "pitch", "yaw")):
                raise ValueError("Unsupported common wrist chain")
        return cls(
            buffers=buffers,
            wrist=PositionController(),
            delay=delay,
            wrist_q_ids=wp.array(
                [i for s in sides for i in scene.layout.hand(s).arm_q_ids], dtype=wp.int32, device=device
            ),
            base_pos=wp.array(bases, dtype=wp.vec3, device=device),
            q_per_world=scene.model.joint_coord_count // scene.world_count,
        )

    def process(self, action, timestep, state=None):
        if state is None:
            raise ValueError("Common wrist decoding needs current coordinates to select a continuous branch")
        self.buffers.process(action, timestep, state)
        layout = self.buffers.layout
        wp.launch(
            decode_wrist,
            dim=(self.buffers.world_count, layout.num_hands),
            inputs=[
                self.processed_target,
                state.joint_q,
                self.buffers.hand_processed_starts,
                self.wrist_q_ids,
                self.buffers.wrist_pos_dof_ids,
                self.buffers.wrist_ori_dof_ids,
                self.base_pos,
                layout.num_hands,
                layout.processed_dim,
                self.q_per_world,
                layout.num_joint_dof,
            ],
            outputs=[self.joint_target],
        )
