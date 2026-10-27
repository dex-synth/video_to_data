# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public H2R recipe composition and rejected legacy training selections."""

from pathlib import Path

import pytest
from hydra.errors import MissingConfigException
from omegaconf import MissingMandatoryValue

from flash_chord.configuration import compose_config, instantiate_typed, resolve_path, resolved_dict
from flash_chord.evaluation.view import PolicyViewConfig
from flash_chord.h2r.configuration import require_h2r_input
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.visualization.viewer import ViewerConfig

_BINDING = "task.parquet=/tmp/h2r_binding.json"


def test_default_is_the_complete_h2r_recipe():
    config = compose_config("train_flash_sac", [_BINDING])
    explicit = compose_config("train_flash_sac", ["experiment=h2r_sharpa_flash_sac", _BINDING])
    assert resolved_dict(config) == resolved_dict(explicit)
    training = instantiate_typed(config.training, TrainingConfig)
    assert training.world_count == 4096
    assert training.total_environment_steps == 250_003_456
    assert training.updates_per_collection == 8
    assert training.network.actor_hidden_dim == 256
    assert training.network.critic_hidden_dim == 512
    assert training.replay.capacity == 4_194_304
    assert config.embodiment._target_ == "flash_chord.h2r.robot.H2rSharpaHands"
    assert config.env.action._target_ == "flash_chord.h2r.action.H2rActionConfig"
    assert config.task.motion_speed == 1
    assert (config.env.sim.fps, config.env.sim.substeps) == (30, 4)
    assert config.scene.support and not config.scene.decompose_objects
    assert config.action.input_mode == "normalized"
    assert config.action.normalized_mapping == "rational"
    assert training.curriculum.thresholds[-2:] == (200_007_680, 250_003_456)
    final = training.curriculum.stage_at(200_007_680)
    assert final.voc_scale == 0
    assert final.immediate_first_frame_probability == 1


@pytest.mark.parametrize("experiment", [
    "sharpa_flash_sac", "sharpa_ppo", "dexmate_sharpa_flash_sac",
    "g1_recon_body_ppo", "g1_recon_body_flash_sac",
])
def test_native_training_recipes_are_not_available(experiment):
    with pytest.raises(MissingConfigException):
        compose_config("train_flash_sac", [f"experiment={experiment}", _BINDING])


@pytest.mark.parametrize("override", [
    "task.parquet=/tmp/reference.parquet",
    "action._target_=flash_chord.runtime.actions.ActionConfig",
    "task.motion_speed=0.5", "task.motion_start_frame=1",
    "task.motion_end_frame=10", "task.source_frame_playback=true",
    "sim.fps=20", "sim.substeps=5",
    "scene.support=false", "scene.decompose_objects=true", "scene.object_scale_min=0.8",
    "embodiment.binding_path=/other/h2r_binding.json",
    "~experiment",
])
def test_native_or_task_changing_overrides_fail_before_reading_assets(override):
    config = compose_config("train_flash_sac", [_BINDING, override])
    with pytest.raises(ValueError, match="H2R"):
        require_h2r_input(config)


def test_input_is_required_and_not_replaced_with_a_default_sequence():
    config = compose_config("train_flash_sac")
    with pytest.raises(MissingMandatoryValue):
        require_h2r_input(config)


def test_partitioned_binding_path_composes_when_quoted():
    path = "/data/sequence_id=clip/h2r_binding.json"
    config = compose_config("train_flash_sac", [f"task.parquet='{path}'"])
    assert config.task.parquet == path


def test_policy_viewer_defaults_to_h2r_and_explicit_reset():
    config = compose_config("view_policy", ["evaluation.checkpoint=/tmp/policy.safetensors"])
    evaluation = instantiate_typed(config.evaluation, PolicyViewConfig)
    assert evaluation.checkpoint == "/tmp/policy.safetensors"
    assert evaluation.deterministic is True
    assert evaluation.reset_mode == "explicit"
    assert config.embodiment._target_ == "flash_chord.h2r.robot.H2rSharpaHands"
    assert config.scene.world_count == 1


def test_viewer_backends():
    for backend in ("gl", "mp4", "viser", "null", "none"):
        assert ViewerConfig(backend=backend, output_path="policy.mp4").backend == backend


def test_resolve_path_uses_invocation_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert resolve_path("checkpoint.ckpt") == Path(tmp_path, "checkpoint.ckpt")


@pytest.mark.parametrize("embodiment", ["sharpa_hands", "sharpa_hands_stiff", "sharpa_hands_primitive", "dexmate_sharpa", "g1_dex3"])
def test_native_robot_configs_are_not_available(embodiment):
    with pytest.raises(MissingConfigException):
        compose_config("train_flash_sac", [f"embodiment={embodiment}", _BINDING])


@pytest.mark.parametrize("runtime", ["env", "replay"])
def test_replay_uses_the_common_robot_clock_and_action(runtime):
    cfg = compose_config("replay", [_BINDING])
    assert cfg.replay.action._target_ == "flash_chord.h2r.action.H2rActionConfig"
    assert (cfg.replay.sim.fps, cfg.replay.sim.substeps) == (30, 4)
    assert cfg.scene.world_count == 1
    assert cfg.replay.voc_scale == 0
    cfg[runtime].action = {"_target_": "flash_chord.runtime.actions.ActionConfig"}
    with pytest.raises(ValueError, match="H2R"):
        require_h2r_input(cfg)
