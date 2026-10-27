# h2r_bench 工作流：Sharpa Wave 双手

`flash_chord.h2r` 消费 h2r 的 human／scene bundle 与 `virtual_wrist_sharpa_wave_bimanual` robot bundle，在方法侧生成原生 MANO 特征、接触、Sharpa IK 和 FlashCHORD 参考。Bench 无需安装本方法的训练依赖。

本 fork 只保留这条共同机器人上的训练配方，`scripts/train_flash_sac.py` 默认选择 `h2r_sharpa_flash_sac`；原生 Sharpa／Dexmate／G1 训练配方和 PPO 训练入口已移除。训练、续训、原生诊断评估与 policy viewer 均先校验 H2R 封存输入，拒绝普通 Parquet、原生 checkpoint 和改变共同输入的覆盖。配方参数保持原 H2R 设置。

本目录不再打包原生 Sharpa、Vega/Dexmate、G1 的 URDF／网格或 SONIC policy，也不保留对应适配器和 G1 专用配置。共享的参考绑定、Sharpa 碰撞映射、动作处理和训练算法继续保留。准备阶段仍从相邻 V2D 的 Sharpa MJCF 读取 IK site 朝向，再把 site 放到共同机器人子树上，并使用共同 fingertip 位置；这项有效依赖不能作为旧资产删除。

这条路径是共同机器人上的任务适配配方：30 Hz 控制、4 个物理子步、1× 示范、每手 6 个标量腕关节与 22 个手指关节。它保留原生手位姿残差、滤波、观测、奖励和训练主体；原生 ball 腕 checkpoint 不能直接复用。

| 入口 | 职责 |
| --- | --- |
| `python -m flash_chord.h2r.prepare` | 共同输入 → 方法特征、IK、封存参考 |
| `scripts/train_flash_sac.py` | H2R 配方训练，使用 Newton／MuJoCo-Warp |
| `scripts/evaluate_policy.py`、`scripts/view_policy.py` | 原生环境诊断与可视化 |
| `scripts/debug/view_robot.py`、`view_scene.py` | `--binding` 指定封存包，显示共同机器人／完整场景 |
| `scripts/debug/replay.py` | H2R 参考回放，默认零残差、VOC=0；Viser 支持运动学参考预览 |
| `python -m flash_chord.h2r.evaluation` | 通过 socket 接收 bench 状态、提交机器人位置目标 |
| `python -m flash_chord.h2r.cpu_evaluation` | 同一共同模型上的独立 CPU 对齐检查 |

正式共同 rollout 与指标由 bench 执行。准备产物、训练输出和检查记录放在代码目录外；不要删除 checkpoint 仍引用的 prepared 目录或修改其中的封存文件。

## 准备与运行

在 `robotic_grounding/flash_chord` 目录执行。以下解释器路径对应 h2r 开发容器；也可以指定具备相应依赖的独立环境。MANO 模型只读取，不复制到方法产物中。

```bash
export PYTHONPATH="$PWD/src:$PWD/../source/robotic_grounding"
FLASH_PY=/opt/flashchord/venv/bin/python
LOADER_PY=/opt/h2r/v2d-loader/bin/python
IK_PY=/opt/h2r/v2d-preprocess/bin/python

# 先设置 HUMAN_BUNDLE、ROBOT_BUNDLE 和新的 PREPARED 输出目录。
"$FLASH_PY" -m flash_chord.h2r.prepare prepare \
  --human-bundle "$HUMAN_BUNDLE" \
  --robot-bundle "$ROBOT_BUNDLE" \
  --output "$PREPARED" \
  --mano-root /datasets/mano \
  --loader-python "$LOADER_PY" \
  --ik-python "$IK_PY" \
  --device cpu
```

准备入口拒绝覆盖已有输出目录；失败时保留中间产物供诊断。`features`、`ik`、`seal` 子阶段可以通过 `--output` 显式重跑，`features` 还需要 `--mano-root`。只有全部阶段成功后生成的 `h2r_binding.json` 才是训练输入。不要把 `processed.parquet` 单独传给原生配方。

```bash
# 使用完整 H2R 配方；TRAIN_OUTPUT 指向新的外部输出目录。
"$FLASH_PY" scripts/train_flash_sac.py \
  experiment=h2r_sharpa_flash_sac \
  task.parquet="$PREPARED/h2r_binding.json" \
  logging.mode=disabled \
  output_dir="$TRAIN_OUTPUT"

# CHECKPOINT 指向已保存的 policy_*.safetensors。
"$FLASH_PY" scripts/evaluate_policy.py \
  evaluation.checkpoint="$CHECKPOINT" \
  evaluation.world_count=2 \
  evaluation.metrics_output="$TRAIN_OUTPUT/readback.json" \
  logging.mode=disabled

"$FLASH_PY" scripts/view_policy.py \
  evaluation.checkpoint="$CHECKPOINT" \
  viewer.backend=viser markers.enabled=false
```

原生评估与 viewer 使用 checkpoint 中的配置重建方法环境，不代表 h2r 共同成绩。H2R 末阶段的部署初始化已通过续训验证；新增课程的完整从头训练尚未重新验证。更改当前 YAML 不会更改旧 checkpoint 中的配置。

## 训练初始化与部署初始化

`ReferenceReset` 默认在随机参考帧打开部分手指，并停留至 `voc_decay_steps` 结束。这段准备期仍可使用 VOC=1，即使课程的全局 VOC 已降为零。`reset_to_first_frame_probability` 只改变参考帧采样，不取消准备期，也不恢复完整参考手指姿态。共同评估的首帧显式 reset 则使用精确参考手指、零速度，立即开始运动，不执行这段准备。

因此，训练最后阶段需要覆盖实际部署初始化。`reset.immediate_first_frame_probability` 控制采用精确首帧、完整手指姿态、无等待 reset 的比例；这些 reset 从第一步使用当前课程 VOC。默认值为 0，保留原生采样配方。也可在某一课程阶段设置同名 `immediate_first_frame_probability`，阶段未设置时使用 reset 配置值。H2R 专用课程 `h2r_tracking_250m` 保留原生 VOC 日程和奖励，在最后约 5000 万步将此比例设为 1，此时 VOC=0。旧 checkpoint 仍按自身保存的配置加载，不会自动采用新课程。

从已完成 VOC 课程的训练状态继续训练时，可设置 `reset.immediate_first_frame_probability=1.0`，并保留 VOC=0 的末阶段；恢复 `state_*.safetensors`，另存输出和完整配置。该设置不保证旧策略无需适应即可成功，也不会改写 checkpoint 中保存的旧配置。不能通过在共同评估端增加准备时间来替代这项训练覆盖。

训练误差比较的是一次物理推进后的状态与这次推进所保持的控制目标，且统计包含准备期、随机起点和失败后的新 episode。`evaluate_policy.py` 则包含初态，N 个参考采样只执行 N−1 次控制推进，按照各状态的真实参考时刻记录误差。两者均值不能直接当作同一个指标。比较训练与部署效果时，应另跑无辅助、无等待、无提前 reset 的完整轨迹，保留所有失败，并使用同一参考时间轴和计分规则。

## 文件与职责

| 输出 | 用途 |
| --- | --- |
| `human/`、`robot/` | 共同包的独立副本，保持原文件 hash；不回写源包 |
| `assets/robot.xml` | 有效共同机器人，显式化 position actuator 的实际阻尼 |
| `assets/ik_{left,right}.xml` | 共同掌部以下的手指模型，供原生 free-flyer IK 使用 |
| `assets/object.xml`、`assets/support.usda` | 共同两部件单铰链物体与有限桌面 |
| `features.npz` | 原生 MANO 方向和实际计算的接触标签 |
| `processed.parquet` | 原生手位姿、手指、robot frames 和对象参考 |
| `robot_reference.npz` | 按共同关节名称排列的 56 维坐标及源时钟 |
| `checks/mano.json`、`checks/ik.json` | 人手几何核对、实际采样数、FK 和逐帧 IK 误差／迭代数 |
| `h2r_binding.json` | 来源身份、源帧窗口、时长、方法设置及完整产物 hash |

所有运行资源均在准备目录内，XML 和 Parquet 使用相对资源路径；移动整个目录后可以重新加载。修改封存输入或参考会被 hash 校验拒绝。需要短窗口时先由 bench 裁剪共同包；方法入口不再裁剪、重采样或改变速度。

调用链为 `h2r.prepare` → 原生 `read_mano.MANO`／`contact_utils`／`SharpaHandKinematics` → `h2r.reference` → `h2r.scene`／`H2rSharpaHands` → 原生环境与训练。方法专用代码集中在 [h2r/](../src/flash_chord/h2r/)，原生 loader、scene、sim 和 action 只增加局部入口。

## 坐标、资产与控制核验

- 保留共同 world 坐标、米、弧度、`wxyz` 和所有源帧。V2D 与 bench 的 16 个共有 MANO 关节及完整表面须一致。两库选择的 fingertip 顶点不同，方法保留原生顶点，并核对其来自共同表面；这个差异单独记录。
- IK 使用共同手指几何与限位，保留原生任务权重、warm start、求解器与 task-site 方向；tip-site 位置来自共同 fingertip。准备时以非零多轴姿态比较 Pinocchio／MuJoCo FK，运行时逐帧比较 Newton 与共同参考的掌、DP、tip。
- 原生动作依旧计算 `q_target = q_ref * quat_rpy(residual)`，`processed_target` 仍为世界掌位姿。最后才转换为 `Rx(roll) Ry(pitch) Rz(yaw)` 与相对零位的 XYZ 平移；按当前真实关节状态选择连续分支，处理 ±π 和 gimbal lock。动作维度是 56，processed pose 维度是 58。此路径只提交位置目标，不额外施加原生 ball 腕 PD 力矩。
- 物体直接派生 MJCF，避免不必要的 URDF 转换；`ObjectAssetSpec.urdf_path` 沿用原生字段名，在 H2R 专用 scene 中指向此 MJCF。不回退同名 ARCTIC 资产、不再分解碰撞网格、不添加无限地面。保留固定连杆以便逐 body 核对惯量和 frame。
- 名义物理以共同包为准，覆盖原生场景的摩擦和 solver 默认值。只有控制时钟写在 H2R experiment 中；通用 `sim`／`scene.contact_friction` 默认值不覆盖共同包。VOC、随机 reset 和训练课程仍为方法训练机制，不能用于将来的共同评估。

依赖以 `pyproject.toml`／`uv.lock` 为准：MuJoCo / MuJoCo-Warp 3.11.0、Warp 1.16.0、Newton 1.5.0。准备资产从 `compiled.json` 写入实际 `kp/kv`；局部 builder 保留共同虚拟腕／marker 的小惯量；CPU 检查模型和 GPU 模型均恢复共同 scalar joint 总限力开关。

桌面几何来自共同场景导出的 `support.usda`，运行时通过 world-parented fixed joint 建立固定支撑，没有额外 q/DOF。不能用运动学 free joint 代替：Newton 为其增加的巨大 armature 会放大 GPU `meaninertia`，使整个场景提前停止求解。FlashCHORD 的 MuJoCo solver 显式开启 MultiCCD。H2R 还显式写入共同 `margin/gap` 和 raw `solref`，防止 Newton 默认接触包络或原始手指垫材料覆盖共同条件。

H2R 配方设置 `sim.njmax=1024`，为每个 world 分配约束容量。solver 的物理参数来自共同包，容量来自方法配置；两者在同一创建入口组装，之后校验实际 CPU／GPU 模型。

每次创建 solver 都检查最终 MuJoCo 模型的关节、质量、COM、惯量、重力补偿、位置驱动 gain／bias／限幅、接触参数、碰撞凸包、有效碰撞配对和桌面尺寸；额外核验动态自由度、固定支撑、物理开关，以及实际 GPU 模型的 armature、meaninertia 和接触参数。错误在训练前抛出；这些检查不宣称 MuJoCo CPU 与 MuJoCo-Warp 长时闭环轨迹完全相同。

环境 reset 在恢复参考状态后，按 world 调用 Newton solver reset，保留参考 q/qvel 并清理 warmstart、外力和 actuator 缓存；接触观测也按 world 清空。ReplayRunner 同样清理 solver 历史。以上操作支持捕获进 CUDA graph，不影响未选中的 world。`evaluation.deterministic=true` 只选择确定性 actor，默认 MuJoCo-Warp 的并行浮点归约与接触排列仍不保证多圈或独立进程逐位相同；不能据此将 viewer 第一圈当作完整重复性验证。

物理条件不同的旧 checkpoint 不应标为已在当前条件训练；重新运行和训练须另存输出。

## 本地检查

```bash
FLASHCHORD_H2R_BINDING="$PREPARED/h2r_binding.json" \
  "$FLASH_PY" -m pytest -q \
  tests/test_h2r_wrist.py tests/test_h2r_integration.py \
  tests/runtime/test_action.py tests/data/test_mano_sharpa.py tests/data/test_loader.py
```

无准备产物时集成检查显式 skip。GPU 检查覆盖原生 processed action／正则化一致性、共同 FK、零额外腕力矩和短物理 rollout；独立腕转换检查同时覆盖 CPU 与 GPU。输入集合验收应对每个实际共同包执行准备和模型检查，不能从 box 的训练 smoke 推导其他序列的训练性能。

实际动力学回归额外读取 bench 独立编译的完整场景，不从方法模型生成期望值：

```bash
FLASHCHORD_H2R_BINDING="$PREPARED/h2r_binding.json" \
FLASHCHORD_H2R_SCENE=/path/to/bench-task/scene/scene.xml \
  "$FLASH_PY" -m pytest -q tests/test_h2r_physics.py
```

该检查覆盖实际 GPU 参数的故障注入，以及相同首帧和相同关节目标下含接触的短 CPU/GPU rollout。

## 共同物理评估部署

Bench 用此次训练绑定的共同 human／scene 与 robot 包执行 `prepare-evaluation --contacts`，准备 metric spec 后启动 UniSim MuJoCo evaluator。方法在自己的环境启动常驻客户端：

```bash
"$FLASH_PY" -m flash_chord.h2r.evaluation \
  --socket /path/to/eval.sock --checkpoint /path/to/policy.safetensors \
  --prepared "$PREPARED" --output /path/to/new-controller-output
```

`h2r/controller.py` 复用原生 `PolicyObservation`、`ContactTracker`、FlashSAC actor loader 与 `H2rHandPoseAction`。它只建立方法几何与数值 buffer，不创建 solver、VOC、reward 或自动 reset 环境；反馈全部来自 evaluator。任务绑定、完整关节名字、30 Hz／120 Hz 时钟、checkpoint schema 和准备产物逐项核对。当前 checkpoint 的 metadata 引用了训练 prepared 的路径，该路径也必须可读且与部署 prepared 的完整 binding 一致；不能给旧 checkpoint 任意换参考包。

接触使用 `h2r-body-contacts-v1`：最后求解时刻的 body-local 两侧表面点、具名两侧 body，以及作用在第二侧的世界系力。方法恢复原生 shape→link 映射，使用当前真实 body pose 和原生内核进行聚合；零力不等于没有接触。首帧接触特征置零沿用原生规则，后续缺失接触会报错。Bench 的 body 原点速度须先加上 `ω × R(body_com)`，恢复 Newton FK 的质心速度，再送入训练时的原生观测内核；不能直接复制这两个不同参考点的线速度，也不在部署时修改原生观测算法。

策略与原生观测／动作 buffer 保留 CUDA 运算和 checkpoint 的计算 dtype。入口在 JAX 初始化前设置 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`，每次 actor 推理固定为一个 episode，避免独立进程 autotuning 或评估批量改变浮点计算路径；冲突的 XLA 设置明确拒绝。`--device` 默认 `cuda:0`，物理 evaluator 的 MuJoCo CPU 与方法的 CUDA 推理是独立选择。EMA、原生残差旋转组合、历史及连续腕分支均保留；不附加目标机械限位裁剪。当前入口要求零 delay、无 arm integral、无额外方法速度限制的 H2R 配方。

每个 episode 仅提交一次首帧具名机器人初态，后续 tick 每四个物理步更新；终态不再产生动作。控制器输出目录保存 checkpoint hash、配置、输入绑定、XLA flags、观测／动作／目标及耗时，成绩与 physical rollout 由 bench 记录。不同训练输入或 backend 必须另做验证；此入口不保证任务成功。

部署一致性检查需要 checkpoint、binding 和 evaluator `hello` 的封存 JSON：

```bash
FLASHCHORD_H2R_CHECKPOINT=/path/to/policy.safetensors \
FLASHCHORD_H2R_BINDING="$PREPARED/h2r_binding.json" \
FLASHCHORD_H2R_HELLO=/path/to/hello.json \
  "$FLASH_PY" -m pytest -q -s tests/test_h2r_evaluation.py
```

该检查使用原生 GPU 环境的连续真实状态与接触，显式转换质心／原点速度，并打乱关节顺序、交换接触两侧，比较观测、actor、历史与最终控制。运行报告放在各自输出目录，不将训练日志或原生 viewer 的重播当作共同成绩。

共同场景的独立 CPU eval 使用以下入口：

```bash
"$FLASH_PY" -m flash_chord.h2r.cpu_evaluation \
  --task /path/to/fixed-task \
  --metric-spec /path/to/fixed-metric-spec \
  --checkpoint /path/to/policy.safetensors \
  --prepared "$PREPARED" \
  --output /path/to/new-native-cpu-run
```

`h2r.cpu_evaluation` 在方法进程内直接加载固定 task 的共同 MJCF，独立调用 MuJoCo CPU；不导入 bench runtime，也不读取 bench rollout。复用 `h2r.controller` 中原生观测、actor、动作与历史。MuJoCo 保留完整双精度积分状态，只在共同观测／控制边界转换 float32；body 状态使用独立 FK，接触保留最后求解时刻。CPU sensor bridge 的 float32 转换发生在接触坐标与力变换之前，这个顺序也必须一致。物理使用 CPU，策略特征与推理仍使用 CUDA。

不能用 `SolverMuJoCo.use_mujoco_cpu=True` 临时替换 GPU 后就声称与共同 CPU eval 等价：该路径仍使用 Newton 重建模型，并默认每物理步通过 float32 Newton state 回写积分状态。这里的 CPU 入口直接复用共同物理模型，不启用 VOC，不通过方法输出修改共同任务。输出 `physical.npz`、`controls.npz` 与绑定／版本信息，供独立比较与共同指标诊断；正式共同结果仍由 bench 自己执行和记录。

`--metric-spec` 只提供与 task 绑定的手腕测量坐标系，不参与控制或物理推进。记录按该坐标系执行完整位姿变换，包含单位局部旋转时的四元数归一化。

可选回归 `tests/test_h2r_cpu_evaluation.py` 通过 `FLASHCHORD_H2R_CPU_TASK`、`FLASHCHORD_H2R_CHECKPOINT`、`FLASHCHORD_H2R_BINDING`、`FLASHCHORD_H2R_CPU_METRICS` 和 `FLASHCHORD_H2R_CPU_BENCH` 指定输入。最后一个路径包含独立执行的 `server/seed-42/physical.npz` 与 `client/controls.jsonl`。检查完整物理窗口、逐次观测／动作／目标精确相等，并验证读取观测不回写或截断双精度积分状态。已验证范围为 MuJoCo 3.11.0、Sharpa 双手和 box `[45,240)`。
