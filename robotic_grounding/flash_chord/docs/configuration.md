# Configuration guide

The only public training recipe is `h2r_sharpa_flash_sac`, selected by default by `train_flash_sac`.
It is self-contained and does not inherit a native robot recipe.

```bash
python scripts/train_flash_sac.py \
  "task.parquet=${PREPARED}/h2r_binding.json" logging.mode=disabled \
  output_dir=/path/to/new-training-output
```

`task.parquet` retains its historical field name but must point to a sealed `h2r_binding.json`.
Prepare it using the [H2R workflow](h2r_inputs.md). There is no default sequence.
The application checks the input before environment construction; ordinary Parquet, native robot/action
configurations, changed source windows, speed, scale or missing support are rejected. The runtime additionally
validates the effective solver model against the common assets.

## Training settings

The default is 4,096 worlds and 250,003,456 environment steps. `h2r_tracking_250m` preserves the VOC
schedule and rewards, with exact first-frame, no-wait resets and VOC=0 in the last approximately 50M steps.
Training curricula are method logic; they do not redefine benchmark initialization or evaluation physics.

Network, optimizer, replay and reward settings remain configurable. The H2R-only change does not change
these parameters. A new configuration does not retroactively change a saved checkpoint.

## Checkpoints

Checkpoints embed the resolved configuration and policy/critic schemas. Training resume, evaluation and
policy viewing reject native checkpoints and require the same sealed H2R reference. The referenced prepared
files must remain available; moving them requires keeping checkpoint paths resolvable.

`policy_<steps>.safetensors` contains the actor; `state_<steps>.safetensors` contains resumable training state.
Historical native files remain on disk but are not supported by this fork's application entry points.

## Viewer and diagnostic evaluation

Use `viewer.backend=viser` or `gl` interactively, `mp4` for bounded recording, and `null`/`none` for headless
stepping. MP4 requires an `.mp4` output path and even image dimensions.

The default `evaluation.reset_mode=explicit` uses exact reference initialization.
The legacy `sampled_settled` mode requires the removed G1 reset strategy and is unsupported for H2R.
`evaluate_policy.py` and `view_policy.py` require H2R checkpoint inputs even when CLI overrides are supplied.
Formal evaluation and metrics remain owned by h2r_bench.

`scripts/debug/view_robot.py` and `view_scene.py` require `--binding /path/to/h2r_binding.json`.
`replay` inherits the H2R recipe and validates its effective action and timing before construction;
its default uses one world, zero residual and VOC=0. Native robot and SONIC selectors are removed.

Reusable configuration groups and semantic/algorithm modules remain for implementation and unit tests;
they do not expose additional supported training recipes.
