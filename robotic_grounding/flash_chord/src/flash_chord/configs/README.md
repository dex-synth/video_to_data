# Configuration

`experiment/h2r_sharpa_flash_sac.yaml` is the only public training recipe. `train_flash_sac.yaml`
selects it by default. Supply `task.parquet=/path/to/prepared/h2r_binding.json`; an ordinary Parquet
reference is rejected by the application before environment construction.

The H2R recipe owns the robot, timing, action, curriculum and FlashSAC defaults without inheriting a native
experiment. Reusable groups still describe runtime components; they are not standalone training recipes.
Native experiment selections, robot selectors, SONIC/G1 groups and the previous PPO training root are removed.
`replay.yaml` inherits H2R with one world and VOC=0.

Training, resume, evaluation, policy viewing and replay validate H2R input identity. The full input preparation,
physical checks and deployment contract are described in [H2R inputs](../../../docs/h2r_inputs.md);
configuration options are in the [configuration guide](../../../docs/configuration.md).
