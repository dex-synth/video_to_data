# SPDX-License-Identifier: Apache-2.0
"""Separate sealed H2R admission from the explicit native Sharpa recipe."""

from pathlib import Path

from omegaconf import OmegaConf

from .inputs import TRAINING_CLOCKS, load_binding

_UNSPECIFIED = object()


def validate_input(config, *, expected_binding=_UNSPECIFIED):
    """Keep sealed H2R admission separate from the restored native Sharpa entry."""
    cfg = config if OmegaConf.is_config(config) else OmegaConf.create(config)
    target = cfg.embodiment._target_
    is_h2r = (Path(cfg.task.parquet).name == "h2r_binding.json"
              or target.startswith("flash_chord.h2r.")
              or str(cfg.env.action._target_).startswith("flash_chord.h2r."))
    if is_h2r:
        binding = require_h2r_input(cfg)
    else:
        if target != "flash_chord.embodiments.sharpa_hands.SharpaHands":
            raise ValueError("Select the native Sharpa or H2R Sharpa recipe")
        binding = None
    if expected_binding is not _UNSPECIFIED and binding != expected_binding:
        raise ValueError("Checkpoint and runtime must use the same input recipe and sealed H2R binding")
    return binding


def require_h2r_input(config, *, expected_binding=None):
    """Validate the H2R branch without allowing fallback to native assets."""
    cfg = config if OmegaConf.is_config(config) else OmegaConf.create(config)
    runtimes = ("env", "replay") if "replay" in cfg else ("env",)
    targets = {
        "embodiment": "flash_chord.h2r.robot.H2rSharpaHands",
        **{f"{name}.action": "flash_chord.h2r.action.H2rActionConfig" for name in runtimes},
    }
    for key, expected in targets.items():
        if OmegaConf.select(cfg, key + "._target_") != expected:
            raise ValueError("H2R inputs require experiment=h2r_sharpa_flash_sac and the common robot/action")
    path = Path(cfg.task.parquet).expanduser().resolve()
    if path.name != "h2r_binding.json":
        raise ValueError("H2R requires a sealed h2r_binding.json, not a native Parquet reference")
    if Path(cfg.embodiment.binding_path).expanduser().resolve() != path:
        raise ValueError("H2R robot and task must use the same sealed input")
    if (cfg.task.motion_speed != 1.0 or cfg.task.source_frame_playback
            or cfg.task.motion_start_frame != 0 or cfg.task.motion_end_frame != -1
            or any((cfg[name].sim.fps, cfg[name].sim.substeps) not in TRAINING_CLOCKS for name in runtimes)):
        raise ValueError("H2R requires its complete 1x window and 20/5 training clock (historical: 20/6 or 30/4)")
    if (not cfg.scene.support or cfg.scene.decompose_objects
            or (cfg.scene.object_scale_min, cfg.scene.object_scale_max) != (1.0, 1.0)):
        raise ValueError("H2R requires its complete support and unchanged object geometry")
    binding = load_binding(path)
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("Checkpoint and runtime must use the same sealed H2R input")
    return binding
