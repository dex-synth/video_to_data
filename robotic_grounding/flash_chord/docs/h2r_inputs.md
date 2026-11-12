# h2r_bench 工作流：Sharpa Wave 双手

`flash_chord.h2r` 消费 h2r 的 human／scene bundle 与 `virtual_wrist_sharpa_wave_bimanual` robot bundle，在方法侧生成原生 MANO 特征、接触、Sharpa IK 和 FlashCHORD 参考。Bench 无需安装本方法的训练依赖。

`scripts/train_flash_sac.py` 默认选择 `h2r_sharpa_flash_sac`，同时保留显式 `experiment=sharpa_flash_sac` 原生 Sharpa 对照入口。H2R 的训练、续训、诊断与 viewer 严格校验封存输入；原生入口消费普通 ManoSharpaData Parquet 及其物体／支撑资产，不接受混用的 H2R 机器人或动作配置。Dexmate、G1 和 PPO 入口未恢复。

原生 Sharpa 配方、双手 URDF、引用网格和 embodiment 来自上游 v0.4.0（`8c778f90`），仅为 Newton 1.5 调整 builder target 的坐标布局。它保留 ball 腕、20 Hz／5 子步、0.5× 播放和原生 settling，供场景与控制诊断；本 fork 的共享 replay／optimizer 数值保护仍存在，不能据此宣称训练算法与上游完全等价。数据资产需另行提供并展开 Git LFS。原生调用示例为 `python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='/path/to/native.parquet'"`；路径含 `=` 时保留内层引号。上游 `sample_object_vertices` 仅支持可见 mesh，primitive-only `synthbox` 可训练，但诊断评估导出会明确失败；此既有限制保留。

H2R 配方使用 **20 Hz 控制、6 个物理子步、120 Hz 物理、1× 示范**，每手 6 个标量腕关节与 22 个手指关节。H2R 的特征裁剪、状态／样本检查、优化器回滚与 `skip_first_frame_settling=true` 保持启用；这些是保留的行为差异。原生 ball 腕 checkpoint 不能直接用于共同机器人。

共同示范与准备产物保持 30 Hz。加载时复用原生位置／关节线性插值、姿态 SLERP 和接触最近邻采样，按 20 Hz 生成内部参考，并保留精确终点。若终点不落在控制网格上，最后一项是终点样本：例如 6.466667 s 的任务在 6.45 s 最后更新，评估仅再积分 2 个 120 Hz tick。训练环境仍执行均匀控制步，终点样本位于其最后一项；方法内部诊断可能覆盖到 6.5 s，不得视为共同评估时间。`sample_times_s` 明确记录内部参考对应的源时间，新 checkpoint 同时记录这些采样时刻。

历史 30 Hz／4 子步使用 `experiment=h2r_sharpa_flash_sac_30hz`。旧 checkpoint 按保存的频率加载，续训必须显式选择相同频率；不会重写已有 prepared 或 task。准备阶段继续从 V2D 的 Sharpa MJCF 读取 IK site 朝向，并使用共同 fingertip 位置。

| 入口 | 职责 |
| --- | --- |
| `python -m flash_chord.h2r.prepare` | 共同输入 → 方法特征、IK、封存参考 |
| `scripts/train_flash_sac.py` | H2R 或显式原生 Sharpa 配方训练，使用 Newton／MuJoCo-Warp |
| `scripts/evaluate_policy.py`、`scripts/view_policy.py` | 原生环境诊断与可视化 |
| `scripts/debug/view_robot.py`、`view_scene.py` | `--binding` 指定封存包，显示共同机器人／完整场景 |
| `scripts/debug/replay.py` | H2R 参考回放，默认零残差、VOC=0；Viser 支持运动学参考预览 |
| `python -m flash_chord.h2r.evaluation` | 通过 socket 接收 bench 状态、提交机器人位置目标 |
| `python -m flash_chord.h2r.cpu_evaluation` | 同一共同模型上的独立 CPU 对齐检查 |

正式共同 rollout 与指标由 bench 执行。准备产物、训练输出和检查记录放在代码目录外；不要删除 checkpoint 仍引用的 prepared 目录或修改其中的封存文件。

ARCTIC 输入检查接受 `h2r-preprocess-v3` 物理包，核对 `bottom/top`、自由根、单铰链与共同有限桌面的显式绑定，并校验桌面位置和尺寸与封存 recipe 一致。额外环境物体仍会拒绝；这一格式兼容不替代各输入的模型与实际 rollout 检查。

## OakInk2 输入

同一准备入口接受已通过准入的 `h2r-preprocess-v2` 或 `h2r-preprocess-v3`／`oakink2-quality-v2` 物理包。当前范围为独立刚体：每个参考部件对应一个自由根；尚不支持 OakInk2 铰接物体、额外固定物体或桌面以外的支撑代理，遇到这些输入会明确拒绝。v3 额外核对 `interaction_objects` 中每个部件的自由根绑定，并要求 `environment` 仅包含共同有限桌面；资产仍从封存的 `scene.xml` 导入。保留原有碰撞分解、原生碰撞体和有限桌面，不在方法侧重新裁剪、去噪或采样。

输入检查区分 120 Hz 源记录与 30 Hz 输出时钟，核对插值源帧映射和完整输出时长。MANO 使用 flat hand mean、腕中心平移，并将全局姿态和位移转换到共同世界坐标；仍用方法原生 MANO loader、接触和 Sharpa IK，逐序列核对共有关节及表面。接触 part ID、对象参考和训练自由根按共同 `part_order` 一一对应，无铰接时 articulation 维度为零。

OakInk2 IK 使用上游的 `mano_to_robot_scale=1.2`、`wrist_offset_xyzw=None`；ARCTIC 保留 1.0 和 `[-0.5, 0.5, 0.5, 0.5]`。实际设置写入 `checks/ik.json` 与 binding；seal 拒绝将旧 IK 冒充新配方。几何检查会恢复 solver 初态及 FK 缓存，避免影响第一帧 IK。已有封存产物保持原参数，要应用新参数须在新目录重新准备。

准备后可按下文同一训练命令使用默认配方。每个实际输入仍须通过模型参数、动作／FK 和短物理 rollout 检查；训练启动或短检查通过不表示任务成功。共同评估支持范围须单独验证。

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

原生评估与 viewer 使用 checkpoint 中的配置重建方法环境，不代表 h2r 共同成绩。更改当前 YAML 不会更改旧 checkpoint 中的配置；初始化适配的训练效果须在具体任务上验证。

## 训练初始化与部署初始化

`ReferenceReset` 原生行为是在随机参考帧打开部分手指，并保持参考至 `voc_decay_steps` 结束；策略、物理仿真和奖励仍正常运行。默认 `voc_decay_steps=20`、`voc_decay_mode=step`，准备期使用 VOC=1，即使课程的全局 VOC 已降为零。`reset_to_first_frame_probability` 只改变参考帧采样，不恢复完整参考手指姿态。

H2R 配方启用 `reset.skip_first_frame_settling=true`：先按原生规则采样参考帧和手指，若最终选中第 0 帧，则跳过整个准备期，从第一步使用当前课程 VOC 并推进参考；其他帧仍保留原生准备过程。自然抽到第 0 帧、首帧概率覆盖和 `always_reset_to_first_frame` 都遵循同一规则。此开关不改变随机数采样、手指姿态、首帧概率、奖励或训练预算。准备期仍配置为 20 个控制步，在默认 20 Hz 下为 1 秒（历史 30 Hz 为约 0.67 秒）。课程 VOC 尚未归零时，首帧仍有课程规定的辅助，不能将跳过准备期解释为所有阶段均无辅助。

默认课程为上游 `tracking_250m_mixed_reset`：VOC 日程不变，最后约 5000 万步将 `reset_to_first_frame_probability` 提高到 0.5。`skip_first_frame_settling` 的通用默认值为 false，旧配置缺少此字段时保持原行为；当前 H2R 训练可显式设为 false 恢复上游 reset。共同评估仍使用第 0 帧、完整参考手指、零速度和无等待初始化，因此训练保留的随机手指与 eval 仍有差异；此改动不保证任务成功。原生诊断的 `sampled_settled` 模式显式关闭该开关，以保留它所请求的准备期。

`reset.immediate_first_frame_probability` 是另一项可选消融，会直接选择第 0 帧、恢复完整参考手指并跳过准备期；默认值为 0，也可通过同名课程字段设置。`curriculum=h2r_tracking_250m` 保留旧的“末阶段全部使用精确首帧”消融，不再是默认课程。该消融与“采样到首帧才跳过等待”不能混为同一配方。旧 checkpoint 仍按自身保存的配置加载，不自动采用新设置；续训须另存输出和完整配置。

训练误差比较的是一次物理推进后的状态与这次推进所保持的控制目标，且统计包含准备期、随机起点和失败后的新 episode。`evaluate_policy.py` 则包含初态，N 个参考采样只执行 N−1 次控制推进，按照各状态的真实参考时刻记录误差。两者均值不能直接当作同一个指标。比较训练与部署效果时，应另跑无辅助、无等待、无提前 reset 的完整轨迹，保留所有失败，并使用同一参考时间轴和计分规则。

## 文件与职责

| 输出 | 用途 |
| --- | --- |
| `human/`、`robot/` | 共同包的独立副本，保持原文件 hash；不回写源包 |
| `assets/robot.xml` | 有效共同机器人，显式化 position actuator 的实际阻尼 |
| `assets/ik_{left,right}.xml` | 共同掌部以下的手指模型，供原生 free-flyer IK 使用 |
| `assets/object.xml`、`assets/support.usda` | 共同物体场景与有限桌面；ARCTIC 双部件单铰链，OakInk2 独立刚体 |
| `features.npz` | 原生 MANO 方向和实际计算的接触标签 |
| `processed.parquet` | 原生手位姿、手指、robot frames 和对象参考 |
| `robot_reference.npz` | 按共同关节名称排列的 56 维坐标及源时钟 |
| `checks/mano.json`、`checks/ik.json` | 人手几何核对、实际采样数、FK 和逐帧 IK 误差／迭代数 |
| `h2r_binding.json` | 来源身份、源帧窗口、时长、方法设置及完整产物 hash |

所有运行资源均在准备目录内，XML 和 Parquet 使用相对资源路径；移动整个目录后可以重新加载。修改封存输入或参考会被 hash 校验拒绝。需要短窗口时先由 bench 裁剪共同包；方法入口不再裁剪、重采样或改变速度。

调用链为 `h2r.prepare` → 原生 `read_mano.MANO`／`contact_utils`／`SharpaHandKinematics` → `h2r.reference` → `h2r.scene`／`H2rSharpaHands` → 原生环境与训练。方法专用代码集中在 [h2r/](../src/flash_chord/h2r/)，原生 loader、scene、sim 和 action 只增加局部入口。

## 坐标、资产与控制核验

- 保留共同 world 坐标、米、弧度、`wxyz` 和所有源帧。V2D 与 bench 的 16 个共有 MANO 关节及完整表面须一致。两库选择的 fingertip 顶点不同，方法保留原生顶点，并核对其来自共同表面；这个差异单独记录。
- IK 使用共同手指几何与限位，保留原生任务权重、warm start、求解器与 task-site 方向；tip-site 位置来自共同 fingertip。准备时以非零多轴姿态比较 Pinocchio／MuJoCo FK，运行时比较 Newton 掌位姿与插值参考，并在共同采样时刻核对 DP、tip；中间时刻由插值腕／手指的 FK 生成，保持原生 Sharpa 绑定语义。
- 原生动作依旧计算 `q_target = q_ref * quat_rpy(residual)`，`processed_target` 仍为世界掌位姿。最后才转换为 `Rx(roll) Ry(pitch) Rz(yaw)` 与相对零位的 XYZ 平移；按当前真实关节状态选择连续分支，处理 ±π 和 gimbal lock。动作维度是 56，processed pose 维度是 58。此路径只提交位置目标，不额外施加原生 ball 腕 PD 力矩。
- 物体直接派生 MJCF，避免不必要的 URDF 转换；`ObjectAssetSpec.urdf_path` 沿用原生字段名，在 H2R 专用 scene 中指向此 MJCF。不回退同名 ARCTIC 资产、不再分解碰撞网格、不添加无限地面。保留固定连杆以便逐 body 核对惯量和 frame。
- 名义物理以共同包为准，覆盖原生场景的摩擦和 solver 默认值。只有控制时钟写在 H2R experiment 中；通用 `sim`／`scene.contact_friction` 默认值不覆盖共同包。VOC、随机 reset 和训练课程仍为方法训练机制，不能用于将来的共同评估。

依赖以 `pyproject.toml`／`uv.lock` 为准：MuJoCo / MuJoCo-Warp 3.11.0、Warp 1.16.0、Newton 1.5.0。准备资产从 `compiled.json` 写入实际 `kp/kv`；局部 builder 保留共同虚拟腕／marker 的小惯量；CPU 检查模型和 GPU 模型均恢复共同 scalar joint 总限力开关。

桌面几何来自共同场景导出的 `support.usda`，运行时通过 world-parented fixed joint 建立固定支撑，没有额外 q/DOF。不能用运动学 free joint 代替：Newton 为其增加的巨大 armature 会放大 GPU `meaninertia`，使整个场景提前停止求解。FlashCHORD 的 MuJoCo solver 显式开启 MultiCCD。H2R 还显式写入共同 `margin/gap` 和 raw `solref`，防止 Newton 默认接触包络或原始手指垫材料覆盖共同条件。

H2R 配方设置 `sim.njmax=1024`，为每个 world 分配约束容量。solver 的物理参数来自共同包，容量来自方法配置；两者在同一创建入口组装，之后校验实际 CPU／GPU 模型。

每次创建 solver 都检查最终 MuJoCo 模型的关节、质量、COM、惯量、重力补偿、位置驱动 gain／bias／限幅、接触参数、碰撞凸包、有效碰撞配对和桌面尺寸；额外核验动态自由度、固定支撑、物理开关，以及实际 GPU 模型的 armature、meaninertia 和接触参数。错误在训练前抛出；这些检查不宣称 MuJoCo CPU 与 MuJoCo-Warp 长时闭环轨迹完全相同。

原生 capsule 的尺寸按 MuJoCo 语义核对半径与中心线半长；Newton 在 `geom_size[2]` 写入的冗余半径不参与 capsule 几何，校验不将该填充值当作物理差异。碰撞体类型、位姿、有效尺寸和物体惯量仍逐项核对。

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

该检查覆盖实际 GPU 参数的故障注入，以及相同首帧和关节目标下的短 MJWarp 物理对照；CPU MuJoCo 仅核对 FK。捕获 `env.reset()` 时仅执行设备检查；重放 graph 后调用 `env.check_reset_validity(reset_mask)` 报告选中 world 的错误。普通 reset 继续即时报告错误。

## 共同物理评估部署

Bench 用此次训练绑定的共同 human／scene 与 robot 包执行 `prepare-evaluation --contacts --physics-hz 120 --control-hz 20`，准备 metric spec 后启动 UniSim MuJoCo evaluator。方法在自己的环境启动常驻客户端：

```bash
"$FLASH_PY" -m flash_chord.h2r.evaluation \
  --socket /path/to/eval.sock --checkpoint /path/to/policy.safetensors \
  --prepared "$PREPARED" --output /path/to/new-controller-output
```

`h2r/controller.py` 复用原生 `PolicyObservation`、`ContactTracker`、FlashSAC actor loader 与 `H2rHandPoseAction`。它只建立方法几何与数值 buffer，不创建 solver、VOC、reward 或自动 reset 环境；反馈全部来自 evaluator。任务绑定、完整关节名字、checkpoint 控制周期与 task 的整数物理 tick 时钟、checkpoint schema 和准备产物逐项核对。当前 checkpoint 的 metadata 引用了训练 prepared 的路径，该路径也必须可读且与部署 prepared 的完整 binding 一致；不能给旧 checkpoint 任意换参考包。

接触使用 `h2r-body-contacts-v1`：最后求解时刻的 body-local 两侧表面点、具名两侧 body，以及作用在第二侧的世界系力。方法恢复原生 shape→link 映射，使用当前真实 body pose 和原生内核进行聚合；零力不等于没有接触。首帧接触特征置零沿用原生规则，后续缺失接触会报错。Bench 的 body 原点速度须先加上 `ω × R(body_com)`，恢复 Newton FK 的质心速度，再送入训练时的原生观测内核；不能直接复制这两个不同参考点的线速度，也不在部署时修改原生观测算法。

策略与原生观测／动作 buffer 保留 CUDA 运算和 checkpoint 的计算 dtype。入口在 JAX 初始化前设置 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`，每次 actor 推理固定为一个 episode，避免独立进程 autotuning 或评估批量改变浮点计算路径；冲突的 XLA 设置明确拒绝。`--device` 默认 `cuda:0`，物理 evaluator 的 MuJoCo CPU 与方法的 CUDA 推理是独立选择。EMA、原生残差旋转组合、历史及连续腕分支均保留；不附加目标机械限位裁剪。当前入口要求零 delay、无 arm integral、无额外方法速度限制的 H2R 配方。

每个 episode 仅提交一次首帧具名机器人初态，后续按 task 的 `control_ticks` 更新（默认 20／120 Hz 为 6 tick，历史 30／120 Hz 为 4 tick）；终态不再产生动作。控制器输出目录保存 checkpoint hash、配置、输入绑定、XLA flags、观测／动作／目标及耗时，成绩与 physical rollout 由 bench 记录。不同训练输入或 backend 必须另做验证；此入口不保证任务成功。

部署一致性检查需要 checkpoint、binding 和 evaluator `hello` 的封存 JSON：

```bash
FLASHCHORD_H2R_CHECKPOINT=/path/to/policy.safetensors \
FLASHCHORD_H2R_BINDING="$PREPARED/h2r_binding.json" \
FLASHCHORD_H2R_HELLO=/path/to/hello.json \
  "$FLASH_PY" -m pytest -q -s tests/test_h2r_evaluation.py
```

该检查使用 H2R 适配 GPU 环境的连续真实状态与接触，显式转换质心／原点速度，并打乱关节顺序、交换接触两侧，比较观测、actor、历史与最终控制；它验证适配内部部署一致性，不等于完整上游复现。运行报告放在各自输出目录，不将训练日志或原生 viewer 的重播当作共同成绩。

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
