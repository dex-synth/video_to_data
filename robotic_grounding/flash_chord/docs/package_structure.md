# Package structure

This fork exposes one H2R training recipe and retains the shared runtime and algorithm components it uses.

```text
src/flash_chord/
├── assets/          # Generic path and mesh utilities; no bundled robot models
├── configs/         # Hydra roots, reusable groups, and one H2R experiment recipe
├── data/            # Reference loading, resampling, and runtime buffers
├── embodiments/     # Semantic layouts, Sharpa contact mapping, and reference binding
├── envs/            # Warp environments and policy observations
├── evaluation/      # Rollout harness and CHORD/ADD-AUC/SPIDER/ManipTrans metrics
├── h2r/             # Common Sharpa inputs, physics checks, and deployment controller
├── lifecycle/       # Curriculum, reset, and termination logic
├── objectives/      # Tracking, wrench, and force-closure rewards
├── runtime/         # Actions, delay, control, contacts, and replay
├── scene/           # Robot/object/support assembly and collision policy
├── training/        # PPO, FlashSAC, checkpoints, and metadata schemas
└── visualization/   # Markers and GL/Viser/MP4/null viewer integration
```

Top-level entry points are in `scripts/`:

- `train_flash_sac.py` trains FlashSAC on sealed H2R inputs; native training recipes and `train_rl.py` are removed;
- `evaluate_policy.py` evaluates checkpoints across many parallel simulation worlds;
- `view_policy.py` visualizes H2R checkpoints;
- `scripts/debug/view_robot.py` and `view_scene.py` inspect the sealed common robot and scene via `--binding`;
- `scripts/debug/replay.py` replays a sealed H2R task with the training action; Viser can also show its kinematic reference.

`workflow/` contains remote training and evaluation templates. `tests/` separates pure/unit coverage from optional
CUDA and local-sequence integration coverage.

The H2R adapter has three module entry points: `h2r.prepare` builds the immutable method reference,
`h2r.evaluation` serves the native controller over the benchmark socket, and `h2r.cpu_evaluation` independently
checks that controller on the common MuJoCo CPU task. Training and visualization use the scripts above. `h2r/configuration.py` checks sealed input identity before training, resume, evaluation, policy viewing and replay; native checkpoints and task-changing overrides are rejected.
The CPU check does not replace benchmark-owned evaluation. Commands and regression inputs are documented in
[the H2R workflow](h2r_inputs.md); generated inputs, checkpoints, and diagnostic runs belong outside the source tree.

`scene/support.py` attaches support surfaces to the world with a fixed joint; support bindings own bodies and shapes,
not dynamic coordinates. `runtime/sim.py` enables MuJoCo MultiCCD. The H2R adapter additionally validates the active
GPU model against the common scene; see [H2R inputs and physics checks](h2r_inputs.md).

Reference sequence corpora, system-identification data/pipelines, retired experiment matrices, policy exporters,
and checkpoint migration layers are not part of the release branch.
