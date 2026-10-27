# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures and optional-dependency gating.

Tests are tiered: pure-function and CPU-device Warp-kernel tests run anywhere;
tests marked ``@pytest.mark.gpu`` need a CUDA device and are skipped when absent.
Tests marked ``@pytest.mark.sequence_data`` need the optional reference corpora.
Warp is imported lazily so the suite collects even where Warp is not installed.
"""

import pytest

from flash_chord.assets import ASSETS_DIR


def _cuda_available() -> bool:
    try:
        import warp as wp

        wp.init()
        return bool(wp.is_cuda_available())
    except Exception:  # noqa: BLE001  # an unavailable/misconfigured Warp install means no CUDA tests
        return False


CUDA_AVAILABLE = _cuda_available()
SEQUENCE_DATA_AVAILABLE = (ASSETS_DIR / "human_motion_data").is_dir()


def pytest_collection_modifyitems(config, items):
    """Skip tests whose explicitly marked runtime prerequisites are absent."""
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    skip_sequences = pytest.mark.skip(reason="optional local reference sequences are not installed")
    for item in items:
        if not CUDA_AVAILABLE and "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if not SEQUENCE_DATA_AVAILABLE and "sequence_data" in item.keywords:
            item.add_marker(skip_sequences)


@pytest.fixture
def h2r_binding():
    import os
    from pathlib import Path

    path = os.environ.get("FLASHCHORD_H2R_BINDING")
    if not path:
        pytest.skip("set FLASHCHORD_H2R_BINDING to a prepared h2r_binding.json")
    return Path(path)


@pytest.fixture
def h2r_config(h2r_binding):
    from flash_chord.configuration import compose_config

    def compose(*overrides):
        return compose_config("train_flash_sac", [
            f"task.parquet='{h2r_binding}'", "scene.world_count=2", "logging.mode=disabled", *overrides,
        ])

    return compose


@pytest.fixture
def h2r_setup(h2r_binding):
    from flash_chord.h2r.robot import H2rSharpaHands
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    def build(world_count=1):
        return setup_scene(
            parquet=h2r_binding, control_fps=30, motion_speed=1,
            embodiment=H2rSharpaHands(h2r_binding), collision=CollisionPolicy(ground=False),
            world_count=world_count, include_support=True, decompose_objects=False,
        )

    return build
