# SPDX-License-Identifier: Apache-2.0
"""Check wrist conventions independently using SciPy's rotation composition."""

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from flash_chord.h2r.action import decode_wrist
from flash_chord.h2r.wrist import wrist_coordinates


def poses():
    rng = np.random.default_rng(42)
    angles = np.concatenate(
        (
            rng.uniform(-6.2, 6.2, (120, 3)),
            [
                [0.7, np.pi / 2, -1.2],
                [-0.8, -np.pi / 2, 1.3],
                [0.7, np.pi / 2 - 1e-7, -1.2],
                [-0.8, -np.pi / 2 + 1e-7, 1.3],
                [np.pi - 1e-5, 0.3, 0.4],
                [-np.pi + 1e-5, 0.3, 0.4],
            ],
        )
    )
    rotation = Rotation.from_euler("XYZ", angles)
    return rng, angles, rotation


def test_cpu_wrist_recovers_intrinsic_xyz_pose_and_translation():
    rng, angles, rotation = poses()
    for euler, quat, matrix in zip(angles, rotation.as_quat(), rotation.as_matrix()):
        position = rng.uniform(-1, 1, 3)
        base = np.array([0, -0.15, 0])
        q = wrist_coordinates(position, quat[[3, 0, 1, 2]], base, previous=euler)
        np.testing.assert_allclose(q[:3] + base, position, atol=1e-12)
        np.testing.assert_allclose(Rotation.from_euler("XYZ", q[3:]).as_matrix(), matrix, atol=2e-7)
        assert np.max(np.abs(q[3:])) <= 6.2


def test_wrist_is_continuous_across_pi():
    previous = np.array([3.0, 0.4, -0.2])
    for roll in np.linspace(3.0, 3.4, 50):
        quat = Rotation.from_euler("XYZ", [roll, 0.4, -0.2]).as_quat()
        q = wrist_coordinates([0, 0, 0], quat[[3, 0, 1, 2]], [0, 0, 0], previous)
        assert np.max(np.abs(q[3:] - previous)) < 0.01
        previous = q[3:]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_wrist_target_matches_processed_world_pose(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    rng, angles, rotation = poses()
    count = len(angles)
    position = rng.uniform(-1, 1, (count, 3))
    base = np.array([0, 0.15, 0])
    processed = np.concatenate((position, rotation.as_quat()[:, [3, 0, 1, 2]]), axis=1)
    q = np.concatenate((position - base, angles), axis=1)
    with wp.ScopedDevice(device):
        target = wp.zeros(count * 6, dtype=wp.float32)
        wp.launch(
            decode_wrist,
            dim=(count, 1),
            inputs=[
                wp.array(processed.reshape(-1), dtype=wp.float32),
                wp.array(q.reshape(-1), dtype=wp.float32),
                wp.array([0], dtype=wp.int32),
                wp.array(range(6), dtype=wp.int32),
                wp.array([0, 1, 2], dtype=wp.int32),
                wp.array([3, 4, 5], dtype=wp.int32),
                wp.array([base], dtype=wp.vec3),
                1,
                7,
                6,
                6,
            ],
            outputs=[target],
        )
        decoded = target.numpy().reshape(count, 6)
    np.testing.assert_allclose(decoded[:, :3] + base, position, atol=1e-7)
    np.testing.assert_allclose(Rotation.from_euler("XYZ", decoded[:, 3:]).as_matrix(), rotation.as_matrix(), atol=2e-5)
    assert np.max(np.abs(decoded[:, 3:])) <= 6.2
