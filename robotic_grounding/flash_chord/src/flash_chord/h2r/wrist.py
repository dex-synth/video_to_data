# SPDX-License-Identifier: Apache-2.0
"""World palm poses to the common XYZ slide / intrinsic XYZ hinge chain."""

import numpy as np
from scipy.spatial.transform import Rotation


def xyz_angles(quat_wxyz, previous=None):
    """Choose a continuous solution of R=Rx(a)Ry(b)Rz(c), including gimbal lock."""
    r = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
    b = np.arctan2(r[0, 2], np.hypot(r[0, 0], r[0, 1]))
    previous = np.zeros(3) if previous is None else np.asarray(previous)
    if np.hypot(r[0, 0], r[0, 1]) < 1e-6:
        a = previous[0]
        c = np.arctan2(r[1, 0], r[1, 1]) - np.sign(b) * a
    else:
        a, c = np.arctan2(-r[1, 2], r[2, 2]), np.arctan2(-r[0, 1], r[0, 0])
    candidates = np.array([[a, b, c], [a + np.pi, np.pi - b, c + np.pi]])
    candidates += 2 * np.pi * np.round((previous - candidates) / (2 * np.pi))
    candidates = np.where(candidates > 6.2, candidates - 2 * np.pi, candidates)
    candidates = np.where(candidates < -6.2, candidates + 2 * np.pi, candidates)
    # These are the packaged Sharpa limits, checked against the imported model.
    valid = np.all(np.abs(candidates) <= 6.2, axis=1)
    if not valid.any():
        raise ValueError("Palm orientation has no continuous solution inside wrist limits")
    cost = np.sum((candidates - previous) ** 2, axis=1)
    cost[~valid] = np.inf
    return candidates[np.argmin(cost)]


def wrist_coordinates(position, quat_wxyz, base_pos, previous=None):
    return np.concatenate((np.asarray(position) - base_pos, xyz_angles(quat_wxyz, previous)))
