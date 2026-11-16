# h2r_bench 工作流：Sharpa Wave

`flash_chord.h2r` 的唯一输入是一个 h2r_bench **task 目录**（`h2r-task-v1`，独立打包或位于 suite 内）。它包含一个 demo、一个 robot 及两者装配后的完整物理模型。方法侧据此生成原生 MANO 关节坐标系、接触、Sharpa IK 和 FlashCHORD 参考。Bench 无需安装本方法的训练依赖。

task 由 [h2r_io/](../src/flash_chord/h2r/h2r_io/) 读取。该目录是 h2r_bench 加载器的副本（`h2r-bench export-loader` 导出，`SOURCE.json` 记录来源），不在本仓库修改，并排除在 linter 之外；`h2r-bench check-loader src/flash_chord/h2r/h2r_io` 核对它与 bench 一致。四个数据集（ARCTIC、OakInk2、HOT3D、TACO）的 task 格式相同，本方法没有按数据集区分结构的代码。

`scripts/train_flash_sac.py` 默认选择 `h2r_sharpa_flash_sac`，同时保留显式 `experiment=sharpa_flash_sac` 原生 Sharpa 对照入口。H2R 的训练、续训、诊断与 viewer 严格校验封存输入；原生入口消费普通 ManoSharpaData Parquet 及其物体／支撑资产，不接受混用的 H2R 机器人或动作配置。Dexmate、G1 和 PPO 入口未恢复。

原生 Sharpa 配方、双手 URDF、引用网格和 embodiment 来自上游 v0.4.0（`8c778f90`），仅为 Newton 1.5 调整 builder target 的坐标布局。它保留 ball 腕、20 Hz／5 子步、0.5× 播放和原生 settling，供场景与控制诊断；本 fork 的共享 replay／optimizer 数值保护仍存在，不能据此宣称训练算法与上游完全等价。数据资产需另行提供并展开 Git LFS。原生调用示例为 `python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='/path/to/native.parquet'"`；路径含 `=` 时保留内层引号。上游 `sample_object_vertices` 仅支持可见 mesh，primitive-only `synthbox` 可训练，但诊断评估导出会明确失败；此既有限制保留。

| 入口 | 职责 |
| --- | --- |
| `python -m flash_chord.h2r.prepare` | task 目录 → 方法特征、IK、封存参考 |
| `scripts/train_flash_sac.py` | H2R 或显式原生 Sharpa 配方训练，使用 Newton／MuJoCo-Warp |
| `scripts/evaluate_policy.py`、`scripts/view_policy.py` | 原生环境诊断与可视化 |
| `scripts/debug/view_robot.py`、`view_scene.py` | `--binding` 指定封存包，显示任务机器人／完整场景 |
| `scripts/debug/replay.py` | H2R 参考回放，默认零残差、VOC=0；Viser 支持运动学参考预览 |
| `python -m flash_chord.h2r.evaluation` | 通过 socket（`h2r-evaluation-v2`）接收 bench 状态、提交机器人位置目标 |

正式共同 rollout 与指标由 bench 执行。准备产物、训练输出和检查记录放在代码目录外；不要删除 checkpoint 仍引用的 prepared 目录或修改其中的封存文件。

## 支持的任务结构

结构全部来自 task，不依赖数据集名称：

| 内容 | 来源 | 方法侧处理 |
| --- | --- | --- |
| 参与手 | `task.hand_sides`，机器人包只含这些手 | 准备、参考、动作、观测、接触和奖励只覆盖参与手；单手任务的策略维度相应减半。配置中的手列表保持原生顺序，加载 binding 时按参与手筛选 |
| 机器人 | `robot.name`、`robot.hands` 与 `semantics.json` | 腕关节、手指关节和腕 body 取自 semantics；不按关节数量判断。原生 IK、手部碰撞 link 和腕解码只有 Sharpa Wave 实现，其他 `robot.name` 明确拒绝 |
| 交互物体 | `task.demo.objects` | 任意数量的物体；每个物体有一个自由根、任意数量的部件和铰链。每个物体对应一个原生 `ObjectAssetSpec`／`ObjectBinding`，铰链按物体及其关节的顺序占参考的 articulation 列 |
| 物体关节 | 封存场景的轴、限位、刚度、`springref`、阻尼、摩擦 | 随 MJCF 导入并逐项核对；原生虚拟物体控制对每个铰链附加同一辅助驱动（`kp=50, kd=2, effort_limit=50`，上游关节物体默认值），其强度随 VOC 缩放，VOC=0 时为零 |
| 静态环境 | `task.demo.environment`：桌面、额外盒状支撑、固定 mesh | 全部保留原世界位姿，合并到一个固定于 world 的 `support` body（原生场景只绑定一个支撑组件）；不进入观测、奖励或接触特征 |

已在真实数据上走通的结构（2026-11-15，各 2,048,000 步的短训练）：单个刚体（HOT3D）、两个刚体（OakInk2、TACO 的工具与目标）、带铰链的双部件物体（ARCTIC）、带铰链的工具加刚体及两个固定 mesh、仅右手（OakInk2 夹子）。走通指准备、短训练、导出和 MuJoCo `sim` 共同评估链路可运行，不代表任务成功；其中 TACO 一例在默认穿透阈值下以 `initial_collision` 结束，见[训练初始化与部署初始化](#训练初始化与部署初始化)。带额外盒状支撑的 HOT3D task 只走到准备与训练场景的建立和核对。

**原生限制（明确拒绝，不做回退）**：

- 交互物体的根固定在 world 上：原生 reset 与虚拟物体控制都作用在自由根的 7 个坐标上。
- `robot.name` 不是 `sharpa_wave`。
- 数据集没有对应的上游 IK 配方（见下表）。

## 数据集配方

方法只在 IK 上按数据集选择自己的超参数，取自上游各数据集的 Sharpa 重定向入口（`robotic_grounding/scripts/retarget/<dataset>_to_sharpa.py`）。四个数据集逐一列出；未列出的数据集直接报错，没有默认配方。

| 数据集 | `mano_to_robot_scale` | 腕初值偏置 `wrist_offset_xyzw` | 上游入口 |
| --- | --- | --- | --- |
| ARCTIC | 1.0 | `[-0.5, 0.5, 0.5, 0.5]` | `arctic_to_sharpa.py` |
| OakInk2 | 1.2 | 无 | `oakink2_to_sharpa.py` |
| HOT3D | 1.2 | 无 | `hot3d_to_sharpa.py` |
| TACO | 1.2 | 无 | `taco_to_sharpa.py` |

`mano_to_robot_scale` 缩放相对腕部的人手目标，不缩放物体或计分轨迹；腕偏置只用于第一帧 IK 的初值。实际设置写入 `checks/ik.json` 与 binding；seal 拒绝与所选配方不一致的 IK 结果。几何检查会恢复 solver 初态及 FK 缓存，避免影响第一帧 IK。

## MANO

task 的 MANO 只有一种约定（`h2r-mano-v1`）：`mano_quat` 的第 0 个是腕在世界系中的旋转，其余 15 个是相对父关节的绝对旋转（平直平均手、无 PCA），`mano_trans` 是腕关节位置。demo 已给出解码后的关节与表面，方法直接使用，不重新解码几何：

- 关节位置取 demo 的 16 个 MANO 关节；指尖保留原生定义（manotorch 的顶点 745、317、444／445、556、673），从 demo 的同一表面取点。它与 bench 的指尖顶点不同，差值单独记录在 `checks/mano.json`。
- 接触标签用 demo 的手部顶点与原生封闭网格面片计算，物体表面按 `part_order` 逐部件采样（第 `i` 个部件 seed 为 `1+i`）。
- 原生 IK 需要 manotorch `AxisLayerFK` 的解剖学关节坐标系，bench 的解码不提供。这一项仍用方法原生的 MANO 层从 `mano_quat` 计算；它只依赖旋转，与形状无关。
- 为确认原生层与 demo 对同一组参数的解读一致，准备阶段比较两者的关节与表面（阈值 0.1 mm）。`left_shapedirs_x_mirrored` 为真时，比较前对原生左手形状基的 x 分量做同样的镜像。

MANO 模型只读取，不复制到方法产物中；其 hash 须与 demo 记录一致。

## 准备与运行

在 `robotic_grounding/flash_chord` 目录执行。以下解释器路径对应 h2r 开发容器；也可以指定具备相应依赖的独立环境。

```bash
export PYTHONPATH="$PWD/src:$PWD/../source/robotic_grounding"
FLASH_PY=/opt/flashchord/venv/bin/python
LOADER_PY=/opt/h2r/v2d-loader/bin/python
IK_PY=/opt/h2r/v2d-preprocess/bin/python

# TASK 是一个 task 目录；PREPARED 是新的输出目录。
"$FLASH_PY" -m flash_chord.h2r.prepare prepare \
  --task "$TASK" \
  --output "$PREPARED" \
  --mano-root /datasets/mano \
  --loader-python "$LOADER_PY" \
  --ik-python "$IK_PY" \
  --device cpu
```

处理整个 suite 时按分发顺序逐个 task 准备，例如 `h2r-bench list-suite "$SUITE" | while IFS=$'\t' read -r CASE TASK; do ...; done`。

准备入口拒绝覆盖已有输出目录；失败时保留中间产物供诊断。`features`、`ik`、`seal` 子阶段可以通过 `--output` 显式重跑，`features` 还需要 `--mano-root`。每个阶段结束时（含失败）向 `preparation_timing.jsonl` 追加一行墙钟记录（`assets`、`features`、`ik`、`seal` 及整个 `prepare`）；重跑的子阶段同样追加。只有全部阶段成功后生成的 `h2r_binding.json` 才是训练输入。不要把 `processed.parquet` 单独传给原生配方。

```bash
# 使用完整 H2R 配方；TRAIN_OUTPUT 指向新的外部输出目录。
"$FLASH_PY" scripts/train_flash_sac.py \
  experiment=h2r_sharpa_flash_sac \
  "task.parquet='$PREPARED/h2r_binding.json'" \
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

`task.parquet` 沿用原生字段名，值必须是 `h2r_binding.json`。默认（正式）配方为 4096 个 world、250,003,456 个 environment step。只为检查链路的短训练须显式覆盖预算，例如 `scene.world_count=512 training.total_environment_steps=2048000 training.replay.capacity=1048576 training.replay.minimum_size=102400 training.checkpoint.save_interval_environment_steps=2048000`；这些不是默认值，其 checkpoint 不能当作正式结果。

每次训练结束时（含失败）在 `output_dir` 的 `training_timing.jsonl` 追加一行：`setup_s`（配置、场景与环境构建，含 graph capture）、`training_s`（learner 初始化或恢复、编译、交互／更新循环与 checkpoint 写入），其中 `compile_s` 为编译、`loop_s` 为编译之后到结束；另记起止 environment step、world 数、设备与是否续训。续训追加新行。Python 导入与 Hydra 启动不在其中；共同的外部计时由 bench 的 `time-stage` 记录。

原生评估与 viewer 使用 checkpoint 中的配置重建方法环境，不代表 h2r 共同成绩。更改当前 YAML 不会更改旧 checkpoint 中的配置。

## 时钟

| 时钟 | 取值 | 归属 |
| --- | --- | --- |
| 参考 | task 的 `reference_hz`（当前 30 Hz）、1× 速度、完整时长 | task |
| 训练控制 | 20 Hz | 方法（`sim.fps`） |
| 训练物理 | 100 Hz（每控制步 5 个子步） | 方法（`sim.substeps`），训练模型显式使用自己的步长 |
| task 名义步长 | 1/300 s | task；训练不使用它 |
| 评估物理 | `hello["dt_s"]`（默认 300 Hz，评估可覆盖） | 评估 |
| 评估控制 | 方法在 `start` 中声明 `control_dt_s = 1/20 s` | 方法；每周期的 tick 数只由 `hello["dt_s"]` 换算（300 Hz 时为 15） |

参考与准备产物保持 task 的采样率。加载时复用原生位置／关节线性插值、姿态 SLERP 和接触最近邻采样，按 20 Hz 生成内部参考，并保留精确终点。若终点不落在控制网格上，最后一项是终点样本：例如 6.466667 s 的任务在 6.45 s 最后更新，300 Hz 评估时只再积分 5 个物理 tick。训练环境仍执行均匀控制步，终点样本位于其最后一项；方法内部诊断可能覆盖到 6.5 s，不得视为共同评估时间。`sample_times_s` 明确记录内部参考对应的源时间，checkpoint 同时记录这些采样时刻。

`sim.substeps=6`（20 Hz／120 Hz）与 `experiment=h2r_sharpa_flash_sac_30hz`（30 Hz／4 子步）是可显式选择的其他训练时钟；续训必须显式匹配 checkpoint 的策略频率及物理子步数。部署时控制周期始终取 checkpoint 的训练控制周期。

## 训练初始化与部署初始化

共同评估从参考第 0 帧开始：物体取首帧状态，机器人取方法提出的初态，全部速度为零，没有 warm-up。本方法提出的初态是 **第 0 帧 IK 的完整机器人姿态（含完整参考手指）**。部分场景中物体在首帧悬空，只靠手支撑。

方法不对该初态做穿透修正，也不读取评估的穿透阈值。若第 0 帧 IK 姿态与场景的穿透超过评估阈值，bench 以 `initial_collision` 在物理推进前结束该 episode。例如 TACO `seq_20230930_121` 的首帧 IK 中左手拇指远节与桌面穿透 20.6 mm，在默认 20 mm 阈值下两个 seed 都以 `initial_collision` 结束。

训练保留原生 reset，因此与评估初态不同：`ReferenceReset` 在随机参考帧打开部分手指，并保持参考至 `voc_decay_steps` 结束；策略、物理仿真和奖励仍正常运行。默认 `voc_decay_steps=20`、`voc_decay_mode=step`，准备期使用 VOC=1，即使课程的全局 VOC 已降为零。`reset_to_first_frame_probability` 只改变参考帧采样，不恢复完整参考手指姿态。评估使用的“第 0 帧、完整参考手指、零速度、无等待、无辅助”位于训练初态分布的边缘。这一差异按 benchmark 的决定保留，不为评估修改原生训练配方。

H2R 配方启用 `reset.skip_first_frame_settling=true`：先按原生规则采样参考帧和手指，若最终选中第 0 帧，则跳过整个准备期，从第一步使用当前课程 VOC 并推进参考；其他帧仍保留原生准备过程。自然抽到第 0 帧、首帧概率覆盖和 `always_reset_to_first_frame` 都遵循同一规则。此开关不改变随机数采样、手指姿态、首帧概率、奖励或训练预算。准备期仍配置为 20 个控制步，在默认 20 Hz 下为 1 秒。课程 VOC 尚未归零时，首帧仍有课程规定的辅助，不能将跳过准备期解释为所有阶段均无辅助。

默认课程为上游 `tracking_250m_mixed_reset`：VOC 日程不变，最后约 5000 万步将 `reset_to_first_frame_probability` 提高到 0.5。`skip_first_frame_settling` 的通用默认值为 false；当前 H2R 训练可显式设为 false 恢复上游 reset。原生诊断的 `sampled_settled` 模式显式关闭该开关，以保留它所请求的准备期。

`reset.immediate_first_frame_probability` 是另一项可选消融，会直接选择第 0 帧、恢复完整参考手指并跳过准备期；默认值为 0，也可通过同名课程字段设置。`curriculum=h2r_tracking_250m` 保留“末阶段全部使用精确首帧”消融，不是默认课程。该消融与“采样到首帧才跳过等待”不能混为同一配方。

训练误差比较的是一次物理推进后的状态与这次推进所保持的控制目标，且统计包含准备期、随机起点和失败后的新 episode。`evaluate_policy.py` 则包含初态，N 个参考采样只执行 N−1 次控制推进，按照各状态的真实参考时刻记录误差。两者均值不能直接当作同一个指标。比较训练与部署效果时，应另跑无辅助、无等待、无提前 reset 的完整轨迹，保留所有失败，并使用同一参考时间轴和计分规则。

## 文件与职责

| 输出 | 用途 |
| --- | --- |
| `task/` | task 的独立副本（`h2r_io.export_task`，`task_id` 不变），含 `demo/`、`robot/`、装配后的 `scene.xml`；不回写源目录 |
| `assets/robot.xml` | task 的机器人，显式写出 position actuator 的实际增益 |
| `assets/ik_<side>.xml` | 参与手腕部以下的手指模型，供原生 free-flyer IK 使用 |
| `assets/scene.xml` | demo 场景：交互物体原样保留，静态环境合并到最后一个 `support` body |
| `features.npz` | 原生 MANO 关节坐标系和实际计算的接触标签 |
| `processed.parquet` | 原生手位姿、手指、robot frames 和对象参考（只含参与手的列） |
| `robot_reference.npz` | 按机器人关节名称排列的参考坐标 |
| `checks/mano.json`、`checks/ik.json` | 人手几何核对、实际采样数、FK 和逐帧 IK 误差／迭代数 |
| `preparation_timing.jsonl` | 各准备阶段的墙钟与状态；诊断记录，不进入 binding 的 `files`，不影响 prepared 身份 |
| `h2r_binding.json` | 格式 `flashchord-h2r-v3`：`task_id`、`demo_id`、`robot_id`、数据集、参与手、时长、方法设置及方法产物 hash |

`task/` 由自身的 `task_id` 封存，每次加载时经 `h2r_io` 校验 task、demo、robot 的全部文件；binding 的 `files` 覆盖其余方法产物。所有运行资源均在准备目录内，XML 和 Parquet 使用相对资源路径；移动整个目录后可以重新加载。修改任何文件都会被拒绝。

此前格式（`flashchord-h2r-v1`，绑定 human／robot bundle）的准备目录以及引用它的 checkpoint 不被读取，加载时报 `Unsupported FlashCHORD preparation ...: prepare again from a task directory`。旧 checkpoint 与结果全部作废，须从 task 目录重新准备和训练。

调用链为 `h2r.prepare` → 原生 `read_mano.MANO`／`contact_utils`／`SharpaHandKinematics` → `h2r.reference` → `h2r.scene`／`H2rSharpaHands` → 原生环境与训练。方法专用代码集中在 [h2r/](../src/flash_chord/h2r/)，原生 loader、scene、sim 和 action 只增加局部入口。

## 坐标、资产与控制核验

- 保留 task 的 world 坐标、米、弧度、`wxyz` 和所有参考帧。
- IK 使用任务机器人的手指几何与限位，保留原生任务权重、warm start、求解器与 task-site 方向；tip-site 位置来自任务机器人的 fingertip。准备时以非零多轴姿态比较 Pinocchio／MuJoCo FK，运行时比较 Newton 掌位姿与插值参考，并在参考采样时刻核对 DP、tip；中间时刻由插值腕／手指的 FK 生成，保持原生 Sharpa 绑定语义。
- 原生动作依旧计算 `q_target = q_ref * quat_rpy(residual)`，`processed_target` 仍为世界掌位姿。最后才转换为 `Rx(roll) Ry(pitch) Rz(yaw)` 与相对零位的 XYZ 平移；按当前真实关节状态选择连续分支，处理 ±π 和 gimbal lock。每只参与手的动作为 6 个腕残差加手指残差（Sharpa：28 维，双手 56 维），processed pose 每手多 1 维。此路径只提交位置目标，不额外施加原生 ball 腕 PD 力矩。腕旋转关节须为 ±6.2 rad 限位，否则在加载时拒绝。
- 物体直接导入 MJCF；`ObjectAssetSpec.urdf_path` 沿用原生字段名，在 H2R 专用 scene 中指向 `assets/scene.xml`。不回退同名数据集资产、不再分解碰撞网格、不添加无限地面。保留固定连杆以便逐 body 核对惯量和 frame。
- 机器人、物体、材料和 solver 参数以 task 为准，覆盖原生场景默认值。训练／方法内诊断的控制与积分时钟由 H2R experiment 决定（默认 20／100 Hz），与 task 的名义步长和评估步长无关；通用 `sim`／`scene.contact_friction` 默认值不覆盖 task。VOC、随机 reset 和训练课程仍为方法训练机制，不能用于共同评估。

依赖以 `pyproject.toml`／`uv.lock` 为准：MuJoCo / MuJoCo-Warp 3.11.0、Warp 1.16.0、Newton 1.5.0。准备资产从机器人包的 `compiled.json` 写入实际 `kp/kv`；局部 builder 保留虚拟腕／marker 的小惯量；CPU 检查模型和 GPU 模型均恢复机器人 scalar joint 总限力开关。

静态环境随 `assets/scene.xml` 导入为一个没有关节的 `support` body，Newton 以 world-parented fixed joint 表示，没有额外 q/DOF。不能用运动学 free joint 代替：Newton 为其增加的巨大 armature 会放大 GPU `meaninertia`，使整个场景提前停止求解。FlashCHORD 的 MuJoCo solver 显式开启 MultiCCD。H2R 还显式写入场景的 `margin/gap` 和 raw `solref`，防止 Newton 默认接触包络或原始手指垫材料覆盖 task 的接触条件。

H2R 配方设置 `sim.njmax=1024`，为每个 world 分配约束容量。solver 的资产／接触／求解参数来自 task，训练积分步长和容量来自方法配置；两者在同一创建入口组装，之后校验实际 CPU／GPU 模型。

**训练场景与 task 的物理一致性**：每次创建 solver，都把实际的 MuJoCo 模型与 task 装配后的 `scene.xml`（评估运行的那个模型）编译结果逐项比较，而不是与方法自己派生的文件比较：自由度数、每个关节的类型／轴／限位／刚度／`springref`／阻尼／armature／摩擦、每个 body 的质量／质心／惯量／重力补偿、机器人位置驱动的 gain／bias／限幅、每个碰撞体的类型／尺寸／位姿／接触参数／凸包、全部有效碰撞配对、物理开关和重力；另核验固定支撑没有自由度，以及实际 GPU 模型的 armature、meaninertia 和接触参数。静态环境的碰撞体按世界位姿比较。错误在训练前抛出；这些检查不宣称 MuJoCo CPU 与 MuJoCo-Warp 长时闭环轨迹完全相同。

**导入设置与接触一致性**：Newton 导入 MJCF 时，未写 `maxhullvert` 的网格默认取 64，而 task 模型沿用 MuJoCo 的默认值（不设上限）。两者得到的凸包顶点相同，但凸包多边形的划分不同；MuJoCo-Warp 依据这些多边形生成面接触，物体平放在支撑上时接触点数随之不同。`flashchord-h2r-v3` 起，场景以 `mesh_maxhullvert=-1` 导入，`validate_solver` 另外逐个核对每个碰撞网格的凸包多边形（数量、顶点数和顶点坐标，按顺序）；更早的 prepared 与 checkpoint 按旧设置训练，不再接受。

`tests/test_h2r_physics.py` 让方法模型与 task 模型（同为 MJWarp、同一训练步长）从同一状态、以同一目标推进四个控制步，同时推进一份起点移动几个 float32 刻度（机器人关节 2e-7、物体位置 2e-8–6e-8 m）的 task 模型作为舍入参照。方法模型相对 task 模型的偏差（机器人关节与速度、物体位置与姿态）不得超过舍入参照偏差的 3 倍，也不低于固定下限（关节 2e-5、速度 2e-3、物体 1e-5）：手指合拢或轻物体夹在夹子之间的起点会自行放大舍入，固定的小界限对它们不成立。在 ARCTIC 铰接盒上，修正前盒子姿态偏差 5.8e-3 rad（与迁移前的代码在同一输入上相同），舍入参照不到 1e-6；修正后位置 2.4e-7 m、姿态 2.9e-7 rad，接触点数逐子步一致。ARCTIC、OakInk2 双物体、TACO、OakInk2 夹子四个检查任务的两种情形都通过。

已知例外：HOT3D `P0001_a9d6c83d_seg017` 在物体位于参考位姿时，有两个瓶子与桌面的接触（穿透约 0.5–0.6 mm）只在 task 模型的 MuJoCo-Warp 推进中出现，瓶子姿态因此相差 1.1e-3 rad，而舍入参照只有 2e-6 rad。这不是 CPU 与 MuJoCo-Warp 之间的数值差异：同一个 task 模型在 CPU MuJoCo 与 MuJoCo-Warp 中，瓶子姿态在该子步一致到 7e-7 rad，接触数相同；用普通 MuJoCo-Warp 推进 Newton 生成的模型则同样缺少这两个接触，所以差异来自 Newton 生成的模型。几何、凸包与多边形、包围体、候选几何体对、容量、几何体顺序、质量与惯量、`invweight0` 和求解器设置都已核对一致，具体字段尚未查明（Newton 不保留编译用的 `MjSpec`）。这一差异不累积：瓶子姿态差在该子步约 1.4e-3 rad，0.2 s 内衰减到 3e-5 rad。修正前的代码在该任务上同样失败。物体抬起的情形下，手指合拢的起点使舍入参照的机器人偏差达到 2.5e-3 rad，方法模型（8e-5 rad）在其内。

原生 capsule 的尺寸按 MuJoCo 语义核对半径与中心线半长；Newton 在 `geom_size[2]` 写入的冗余半径不参与 capsule 几何，校验不将该填充值当作物理差异。

环境 reset 在恢复参考状态后，按 world 调用 Newton solver reset，保留参考 q/qvel 并清理 warmstart、外力和 actuator 缓存；接触观测也按 world 清空。ReplayRunner 同样清理 solver 历史。以上操作支持捕获进 CUDA graph，不影响未选中的 world。`evaluation.deterministic=true` 只选择确定性 actor，默认 MuJoCo-Warp 的并行浮点归约与接触排列仍不保证多圈或独立进程逐位相同；不能据此将 viewer 第一圈当作完整重复性验证。

## 本地检查

```bash
"$FLASH_PY" -m pytest -q tests/test_h2r_inputs.py tests/test_h2r_geometry.py tests/test_h2r_wrist.py
```

以上检查不需要数据或 GPU：`tests/h2r_synthetic.py` 用加载器的封存函数生成小型 task（单刚体、双刚体、铰接盒、仅右手的铰接工具加额外支撑与固定 mesh），覆盖结构绑定、静态环境位姿、原生限制的拒绝、数据集配方、binding 身份以及旧格式和被改动文件的拒绝。

其余检查需要真实的准备产物，未设置环境变量时显式 skip：

```bash
FLASHCHORD_H2R_BINDING="$PREPARED/h2r_binding.json" \
  "$FLASH_PY" -m pytest -q tests

FLASHCHORD_H2R_CHECKPOINT=/path/to/policy.safetensors \
FLASHCHORD_H2R_BINDING="$PREPARED/h2r_binding.json" \
FLASHCHORD_H2R_HELLO=/path/to/controller-output/controller.json \
  "$FLASH_PY" -m pytest -q -s tests/test_h2r_evaluation.py
```

`tests/test_h2r_integration.py` 覆盖原生 processed action／正则化一致性、任务机器人 FK、零额外腕力矩和短物理 rollout。`tests/test_h2r_physics.py` 以 task 装配后的 `scene.xml` 为对照（不从方法模型生成期望值），覆盖实际 GPU 参数的故障注入，以及相同首帧、关节目标及训练步长下的短 MJWarp 物理对照（自由下落与接触两种情形；仅内存中的对照模型使用训练步长，不修改封存任务）。`tests/test_h2r_evaluation.py` 使用 H2R 适配 GPU 环境的连续真实状态与接触，显式转换质心／原点速度，并打乱关节顺序、交换接触两侧，比较观测、actor、历史与最终控制；`FLASHCHORD_H2R_HELLO` 可以是 hello 消息，也可以是评估客户端写出的 `controller.json`。这些检查对任意结构和参与手的 task 适用。

## 共同物理评估部署

Bench 用同一个 task 启动 evaluator（`h2r-bench evaluate-server task=...`）。方法在自己的环境启动客户端：

```bash
"$FLASH_PY" -m flash_chord.h2r.evaluation \
  --socket /path/to/eval.sock --checkpoint /path/to/policy.safetensors \
  --prepared "$PREPARED" --output /path/to/new-controller-output
```

客户端使用 `h2r_io.client` 收发 `h2r-evaluation-v2` 消息：

- 核对 `hello.protocol`，并以 `hello.task_id` 等于 binding 的 `task_id` 确认 evaluator 运行的就是准备时的 task；不再逐项比较物体、观测等内容。
- `start` 由 `start_request` 组装：`binding = {demo_id, robot_id, method="flashchord", controller}`，`controller` 以 bench 读取的字段名 `checkpoint`、`sha256` 记录 checkpoint 路径与 hash；`requirements` 声明方法读取的全部 body、`contacts=true` 和 `control_dt_s`（checkpoint 的训练控制周期，默认 1/20 s）。
- 每周期的物理 tick 数由 `control_ticks(hello, control_dt_s)` 得出；周期不是 `hello["dt_s"]` 的整数倍时在本地报错。产物中只保存以秒为单位的周期。
- 每个 episode 提交一次首帧具名机器人初态；终态不再产生动作。

`h2r/controller.py` 复用原生 `PolicyObservation`、`ContactTracker`、FlashSAC actor loader 与 `H2rHandPoseAction`。它只建立方法几何与数值 buffer，不创建 solver、VOC、reward 或自动 reset 环境；反馈全部来自 evaluator。checkpoint 的 metadata 引用训练所用的 prepared 路径，该路径必须可读且与部署 prepared 的 binding 完全一致；不能给 checkpoint 换参考。

接触使用 `h2r-body-contacts-v1`：最后求解时刻的 body-local 两侧表面点、具名两侧 body，以及作用在第二侧的世界系力。方法恢复原生 shape→link 映射，使用当前真实 body pose 和原生内核进行聚合；零力不等于没有接触。方法只聚合手—交互物体的接触：对方不是方法模型中 body 的记录（world 上的桌面与支撑、固定 mesh）被忽略。首帧接触特征置零沿用原生规则，后续缺失接触会报错。Bench 的 body 原点速度须先加上 `ω × R(body_com)`，恢复 Newton FK 的质心速度，再送入训练时的原生观测内核。

策略与原生观测／动作 buffer 保留 CUDA 运算和 checkpoint 的计算 dtype。入口在 JAX 初始化前设置 `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`，每次 actor 推理固定为一个 episode，避免独立进程 autotuning 或评估批量改变浮点计算路径；冲突的 XLA 设置明确拒绝。`--device` 默认 `cuda:0`，物理 evaluator 的 MuJoCo CPU 与方法的 CUDA 推理是独立选择。EMA、原生残差旋转组合、历史及连续腕分支均保留；不附加目标机械限位裁剪。当前入口要求零 delay、无 arm integral、无额外方法速度限制的 H2R 配方。

控制器输出目录保存 start 中上报的 binding、控制周期、配置、XLA flags、hello、观测／动作／目标及耗时；成绩与 physical rollout 由 bench 记录。此入口不保证任务成功。
