# SPDX-License-Identifier: Apache-2.0
"""Preserve valid small inertias in the common MJCF model."""

import warnings

import newton
import numpy as np


class CommonModelBuilder(newton.ModelBuilder):
    def __init__(self, gravity):
        super().__init__(gravity=(0.0, 0.0, gravity))
        self.balance_inertia = False

    def finalize(self, *args, **kwargs):
        # Newton adds ~1e-6 to eigenvalues below 1e-10, including the
        # common virtual-wrist links (2e-12). They are valid MuJoCo inertias.
        # Preserve them locally; do not change Newton or the common asset.
        tensor = np.asarray(self.body_inertia, dtype=np.float64).reshape(-1, 3, 3)
        dynamic = np.asarray(self.body_mass) > 0
        if np.any(np.linalg.eigvalsh(tensor[dynamic]) <= 0):
            raise ValueError("Common model has a non-positive inertia")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Inertia validation corrected .* bodies")
            model = super().finalize(*args, **kwargs)
        model.body_inertia.assign(np.asarray(self.body_inertia, dtype=np.float32))
        model.body_inv_inertia.assign(np.asarray(self.body_inv_inertia, dtype=np.float32))
        return model
