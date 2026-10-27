# SPDX-License-Identifier: Apache-2.0
"""Require sealed H2R inputs at training and checkpoint application boundaries."""

from pathlib import Path

from omegaconf import OmegaConf

from .inputs import load_binding


def require_h2r_input(config, *, expected_binding=None):
    """Reject native recipes and input-changing overrides before building a runtime."""
    cfg = config if OmegaConf.is_config(config) else OmegaConf.create(config)
    runtimes = ("env", "replay") if "replay" in cfg else ("env",)
    targets = {
        "embodiment": "flash_chord.h2r.robot.H2rSharpaHands",
        **{f"{name}.action": "flash_chord.h2r.action.H2rActionConfig" for name in runtimes},
    }
    for key, expected in targets.items():
        if OmegaConf.select(cfg, key + "._target_") != expected:
            raise ValueError("This fork only supports the H2R recipe; use experiment=h2r_sharpa_flash_sac")
    path = Path(cfg.task.parquet).expanduser().resolve()
    if path.name != "h2r_binding.json":
        raise ValueError("H2R requires a sealed h2r_binding.json, not a native Parquet reference")
    if Path(cfg.embodiment.binding_path).expanduser().resolve() != path:
        raise ValueError("H2R robot and task must use the same sealed input")
    if (cfg.task.motion_speed != 1.0 or cfg.task.source_frame_playback
            or cfg.task.motion_start_frame != 0 or cfg.task.motion_end_frame != -1
            or any(cfg[name].sim.fps != 30 or cfg[name].sim.substeps != 4 for name in runtimes)):
        raise ValueError("H2R requires its complete sealed window at 1x speed, 30 Hz and 4 substeps")
    if (not cfg.scene.support or cfg.scene.decompose_objects
            or (cfg.scene.object_scale_min, cfg.scene.object_scale_max) != (1.0, 1.0)):
        raise ValueError("H2R requires its complete support and unchanged object geometry")
    binding = load_binding(path)
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("Checkpoint and runtime must use the same sealed H2R input")
    return binding
