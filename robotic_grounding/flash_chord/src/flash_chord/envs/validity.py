# SPDX-License-Identifier: Apache-2.0
"""Reject divergent training states without clamping or changing their physics."""

import math
from dataclasses import dataclass

import newton
import warp as wp


@dataclass(frozen=True)
class StateValidityConfig:
    """Loose failure bounds, not actuator limits or task success thresholds."""

    max_body_displacement_m: float = 10.0
    max_linear_speed_m_s: float = 20.0
    max_angular_speed_rad_s: float = 200.0
    joint_range_margin_fraction: float = 0.5

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@wp.kernel
def check_state(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    initial_body_q: wp.array(dtype=wp.transform),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    q_ids: wp.array(dtype=wp.int32),
    dof_ids: wp.array(dtype=wp.int32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    speed: wp.array(dtype=wp.float32),
    bodies: int,
    coordinates: int,
    dofs: int,
    max_displacement: float,
    max_linear_speed: float,
    max_angular_speed: float,
    reason: wp.array(dtype=wp.int32),
):
    world = wp.tid()
    # Bits: nonfinite=1, position=2, linear speed=4, angular speed=8,
    # scalar joint position=16, scalar joint speed=32.
    failed = int(0)
    for i in range(coordinates):
        if not wp.isfinite(joint_q[world * coordinates + i]):
            failed = failed | 1
    for i in range(dofs):
        if not wp.isfinite(joint_qd[world * dofs + i]):
            failed = failed | 1
    for i in range(bodies):
        index = world * bodies + i
        position = wp.transform_get_translation(body_q[index])
        origin = wp.transform_get_translation(initial_body_q[index])
        rotation = wp.transform_get_rotation(body_q[index])
        linear = wp.spatial_top(body_qd[index])
        angular = wp.spatial_bottom(body_qd[index])
        for axis in range(3):
            if not wp.isfinite(position[axis]) or not wp.isfinite(linear[axis]) or not wp.isfinite(angular[axis]):
                failed = failed | 1
            if wp.abs(position[axis] - origin[axis]) > max_displacement:
                failed = failed | 2
            if wp.abs(linear[axis]) > max_linear_speed:
                failed = failed | 4
            if wp.abs(angular[axis]) > max_angular_speed:
                failed = failed | 8
        for axis in range(4):
            if not wp.isfinite(rotation[axis]):
                failed = failed | 1
    for i in range(q_ids.shape[0]):
        q = joint_q[world * coordinates + q_ids[i]]
        velocity = joint_qd[world * dofs + dof_ids[i]]
        if q < lower[i] or q > upper[i]:
            failed = failed | 16
        if wp.abs(velocity) > speed[i]:
            failed = failed | 32
    reason[world] = failed


class StateValidity:
    """Build scalar-joint bounds once and check each replicated world on device."""

    def __init__(self, model, world_count, config: StateValidityConfig, device):
        self.model, self.world_count, self.config = model, world_count, config
        self.bodies = model.body_count // world_count
        self.coordinates = model.joint_coord_count // world_count
        self.dofs = model.joint_dof_count // world_count
        types = model.joint_type.numpy()
        q_start, d_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        low, high = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        q_ids, d_ids, lower, upper, speed = [], [], [], [], []
        for joint in range(model.joint_count // world_count):
            if types[joint] not in (newton.JointType.PRISMATIC, newton.JointType.REVOLUTE):
                continue
            q, d = int(q_start[joint]), int(d_start[joint])
            margin = config.joint_range_margin_fraction * (high[d] - low[d])
            q_ids.append(q)
            d_ids.append(d)
            lower.append(low[d] - margin)
            upper.append(high[d] + margin)
            speed.append(
                config.max_linear_speed_m_s if types[joint] == newton.JointType.PRISMATIC
                else config.max_angular_speed_rad_s
            )
        self.arrays = [
            wp.array(values, dtype=dtype, device=device)
            for values, dtype in (
                (q_ids, wp.int32), (d_ids, wp.int32),
                (lower, wp.float32), (upper, wp.float32), (speed, wp.float32),
            )
        ]

    def check(self, state, reason):
        wp.launch(check_state, dim=self.world_count, inputs=[
            state.body_q, state.body_qd, self.model.body_q, state.joint_q, state.joint_qd, *self.arrays,
            self.bodies, self.coordinates, self.dofs, self.config.max_body_displacement_m,
            self.config.max_linear_speed_m_s, self.config.max_angular_speed_rad_s,
        ], outputs=[reason])
