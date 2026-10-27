# FlashCHORD

FlashCHORD fork for h2r_bench: train FlashSAC on sealed benchmark inputs with Newton, Warp, and JAX.

**[Setup](#setup) → [H2R inputs](#h2r-inputs) → [Train](#train) → [Evaluate and view](#evaluate-and-view)**

## Setup

### Host prerequisites

- Linux and NVIDIA driver **580 or newer**.
- [Docker](https://docs.docker.com/engine/install/ubuntu/) with [non-root access](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- Git LFS; access to the repository, `nvidia/cuda`, and `ghcr.io/astral-sh/uv`.
- Training memory depends on the common scene and world count. Replay and one-world viewing use less memory.

### Run in the container

```bash
# Host, from the repository root.
git lfs install
git lfs pull
cd robotic_grounding/flash_chord
docker/run.sh build
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
docker/run.sh start latest 0   # Replace 0 with your host GPU index.
```

`start` opens an interactive shell at `/workspace`. Run the workflow below there.

| Host command | Action |
| --- | --- |
| `docker/run.sh shell latest 0` | Open another shell |
| `docker/run.sh exec latest 0 -- nvidia-smi` | Run a command |
| `docker/run.sh stop latest 0` | Stop and remove the container |

The container uses the pinned [Dockerfile](docker/Dockerfile) and `uv.lock`.

### Host installation (without Docker)

**Skip this section if using Docker.** The image includes Python and the FlashCHORD dependencies; the host prerequisites above still apply.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[train,dev]"
```

### Open Viser

- Open **http://localhost:8080** on the host; for remote machines, forward the viewer port over SSH.
- Use the port in Viser's **`listening *:PORT`** banner. A busy port moves the server elsewhere; Newton's later URL can still show 8080.
- Stop each viewer with **Ctrl+C** before starting the next. Closing the browser tab leaves the process running.

### Debug visualization

The debug tools read the same sealed package as training:

```bash
python scripts/debug/view_robot.py --binding "$PREPARED/h2r_binding.json" --viewer viser
python scripts/debug/view_scene.py --binding "$PREPARED/h2r_binding.json" --viewer viser
python scripts/debug/replay.py "task.parquet=$PREPARED/h2r_binding.json" viewer.backend=viser start_paused=true
```

Replay defaults to zero residual and VOC=0. The Viser controls can also display the kinematic reference.
These are visualization/diagnostic tools, not benchmark scores.

Change `markers.enabled=false` to `markers.enabled=true` in replay/policy commands to show axes, keypoints, and contacts.

| Markers | Reference target | Simulation |
| --- | --- | --- |
| Hand keypoints | Green | Blue |
| Contacts / normals | Yellow | Magenta |

- Training termination thresholds are in [tracking.yaml](src/flash_chord/configs/termination/tracking.yaml). Benchmark success is scored independently by h2r_bench.

## H2R inputs

This fork exposes one training recipe: `h2r_sharpa_flash_sac`. It requires a sealed `h2r_binding.json`
prepared from h2r_bench human/scene and `virtual_wrist_sharpa_wave_bimanual` robot bundles.
See [H2R preparation and deployment](docs/h2r_inputs.md) for the preparation command and independent environments.

The common input fixes the complete 56-coordinate robot, articulated object, finite support, source window,
1× speed and 30 Hz / four physics substeps. Ordinary Parquet files and native robot recipes are rejected.
Method-specific IK, contacts, FlashSAC, rewards and VOC training curriculum remain in this codebase.

## Train

```bash
# Set PREPARED to the directory produced by flash_chord.h2r.prepare.
python scripts/train_flash_sac.py \
  "task.parquet=${PREPARED}/h2r_binding.json" \
  reset.seed=42 logging.mode=disabled output_dir=/path/to/new-training-output
```

The default is `experiment=h2r_sharpa_flash_sac`; specifying it explicitly is equivalent.
The recipe uses 4,096 worlds and 250,003,456 environment steps. The network, optimizer and curriculum
are defined in the [standalone H2R recipe](src/flash_chord/configs/experiment/h2r_sharpa_flash_sac.yaml).
Use `--cfg job --resolve` to inspect the configuration without starting training.

Native Sharpa, Dexmate and G1 recipes, bundled robot models/meshes, their adapters, SONIC policies,
and the PPO training script have been removed.
Training checks the sealed input before constructing the environment. Resume also requires an H2R checkpoint
bound to the same input. The shared algorithm and geometry modules remain implementation components.

## Evaluate and view

```bash
python scripts/evaluate_policy.py "evaluation.checkpoint=${CHECKPOINT}" \
  evaluation.world_count=2 evaluation.metrics_output=/path/to/diagnostic.json logging.mode=disabled
python scripts/view_policy.py "evaluation.checkpoint=${CHECKPOINT}" \
  viewer.backend=viser markers.enabled=false logging.mode=disabled
```

Both applications restore the checkpoint configuration and require H2R inputs. Native checkpoints are rejected,
and input-changing overrides cannot replace the trained reference. Historical checkpoint files are not modified.
These commands are method-environment diagnostics. Formal physical rollouts and scores belong to h2r_bench;
use the external controller described in the [H2R workflow](docs/h2r_inputs.md).

## Configuration

See the [configuration guide](docs/configuration.md) and [package structure](docs/package_structure.md).
Hyperparameters can be overridden for an explicitly identified experiment. Common task speed, source window,
robot, scale and support remain fixed by the sealed package. Choosing H2R inputs does not by itself equalize
training budgets or make different task windows comparable.
