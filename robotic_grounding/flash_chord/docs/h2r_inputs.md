# h2r_bench 共同输入：Sharpa Wave 双手

`flash_chord.h2r` 消费 h2r 的 human／scene bundle 与 `virtual_wrist_sharpa_wave_bimanual` robot bundle，在方法侧生成原生 MANO 特征、接触、Sharpa IK 和 FlashCHORD 参考。Bench 无需安装本方法的训练依赖。

这条路径是共同机器人上的任务适配配方：30 Hz 控制、4 个物理子步、1× 示范、每手 6 个标量腕关节与 22 个手指关节。它保留原生手位姿残差、滤波、观测、奖励和训练主体；原生 ball 腕 checkpoint 不能直接复用。完整训练、共同 evaluator 客户端和共同成绩不属于这条输入链。

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
# 仅验证启动、参数更新和保存的 smoke；预算和 replay 容量不是正式训练配方。
"$FLASH_PY" scripts/train_flash_sac.py \
  experiment=h2r_sharpa_flash_sac \
  task.parquet="$PREPARED/h2r_binding.json" \
  scene.world_count=4 \
  training.total_environment_steps=128 \
  training.updates_per_collection=1 \
  training.replay.capacity=256 \
  training.replay.minimum_size=16 \
  training.replay.batch_size=16 \
  training.checkpoint.save_interval_environment_steps=128 \
  logging.mode=disabled \
  output_dir="$TRAIN_OUTPUT"

"$FLASH_PY" scripts/evaluate_policy.py \
  evaluation.checkpoint="$TRAIN_OUTPUT/policy_128.safetensors" \
  evaluation.world_count=2 \
  evaluation.metrics_output="$TRAIN_OUTPUT/readback.json" \
  logging.mode=disabled
```

第二个命令使用 checkpoint 中的配置重建方法环境，检查读回与推理；输出是方法内部诊断，不是 h2r 共同测评成绩。正式训练沿用 `h2r_sharpa_flash_sac` 的完整预算，须另行确定实验设置与验收。

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

Newton 1.2 有三处需显式补偿的导入行为：position `dampratio` 在其 joint-target 路径中未转换为实际阻尼，因此从 `compiled.json` 写入 `kv`；小惯量自动修正会抬高虚拟腕／marker 惯量，因此在局部 builder 中保留经过 MuJoCo 验证的原惯量；scalar joint 总限力被默认启用，因此同时恢复 CPU 检查模型和 GPU 模型中的共同限力开关。无需修改 Newton 本身。

每次创建 solver 都检查最终 MuJoCo 模型的关节、质量、COM、惯量、重力补偿、位置驱动 gain／bias／限幅、接触参数、碰撞凸包、有效碰撞配对和桌面尺寸。错误在训练前抛出；这些检查说明当前映射保留了相应参数与几何，不宣称 MuJoCo CPU 与 MuJoCo-Warp 数值轨迹完全相同。

## 本地检查

```bash
FLASHCHORD_H2R_BINDING="$PREPARED/h2r_binding.json" \
  "$FLASH_PY" -m pytest -q \
  tests/test_h2r_wrist.py tests/test_h2r_integration.py \
  tests/runtime/test_action.py tests/data/test_mano_sharpa.py tests/data/test_loader.py
```

无准备产物时集成检查显式 skip。GPU 检查覆盖原生 processed action／正则化一致性、共同 FK、零额外腕力矩和短物理 rollout；独立腕转换检查同时覆盖 CPU 与 GPU。输入集合验收应对每个实际共同包执行准备和模型检查，不能从 box 的训练 smoke 推导其他序列的训练性能。
