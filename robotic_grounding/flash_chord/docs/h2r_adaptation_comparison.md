# FlashCHORD：H2R 适配与 original 对比

本文对比 **Sharpa FlashSAC** 的默认配方，不包含其他机器人、PPO 或单次实验的 CLI 覆盖。核对日期：2026-11-15（输入改为 h2r_bench 的 task 目录之后）。

| 对象 | 版本与入口 |
| --- | --- |
| original | 上游 v0.4.0，commit `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`，`experiment=sharpa_flash_sac` |
| 当前 H2R | fork 分支 `unified-task-distribution`：输入为一个 h2r_bench task 目录，评估协议为 `h2r-evaluation-v2`；默认 `experiment=h2r_sharpa_flash_sac` |

当前版本保留原生 IK、FlashSAC 主体、观测、奖励与 VOC 课程，同时适配共同输入、机器人和物理评估，并修改部分训练行为。因此应称为 **FlashCHORD 的 H2R 适配配方**，不能直接等同于未经修改的 original。表中的“一致”仅指该行列出的设置或逻辑。

## 输入与重定向

| 项目 | original | 当前 H2R |
| --- | --- | --- |
| 输入与准备入口 | V2D loader → 数据集 retarget 脚本 → `ManoSharpaData` Parquet | `h2r.prepare` 读取一个 task 目录（demo、robot 及装配后的场景），生成特征、IK、Parquet、按机器人关节命名的参考及 `h2r_binding.json`；以 `task_id` 绑定并记录方法产物 hash |
| 任务窗口、坐标与资产 | 由原生数据管线和配置选择 | 服从 task；禁止方法侧再次裁剪、缩放物体或替换场景；人手关节与表面直接取 demo，并核对原生 MANO 层的解读与之一致 |
| 参与手 | 双手 | 取 `task.hand_sides`；单手任务只准备、训练和控制该手 |
| 物体结构 | 按数据集选择原生资产：ARCTIC 关节物体或独立刚体 URDF | 取 `task.demo.objects`：任意数量的自由根物体，各含任意数量的部件与铰链；与数据集名称无关 |
| 示范播放速度 `task.motion_speed` | `0.5` | `1.0` |
| 参考时间 | 按目标控制频率和播放速度重采样 | 共同参考保持 30 Hz；内部按 20 Hz 采样，并额外保留精确终点；保存 `sample_times_s` |
| 参考插值 | 位置／关节线性插值，姿态 SLERP，接触最近邻；机器人参考由插值姿态 FK 生成 | 一致 |
| MANO 与接触特征主体 | 原生 MANO、原生 fingertip 顶点、接触检测及逐 link 聚合；接触距离阈值 `0.01 m` | 关节坐标系来自原生 MANO 层，原生 fingertip 顶点、接触检测与逐 link 聚合一致；关节位置与表面取 demo 的解码结果（与原生解码相差 < 0.1 mm） |
| 接触表面采样 | 每部件请求 4096 点；采样调用不指定 seed | 点数一致；固定第 `i` 个部件的 seed 为 `1+i`（从 0 编号），记录实际采样数 |
| 无效接触投影 | 直接保存原生聚合／投影结果 | 若法向长度 ≤ `1e-6` 或部件 ID ≤ 0，清空该接触槽；有效法向归一化 |
| IK 模型 | V2D Sharpa MJCF | 从共同机器人派生手指模型，使用共同几何和限位；保留原生 task-site 朝向，tip-site 位置采用共同 fingertip |
| IK 求解与任务参数 | `SharpaHandKinematics`；`daqp`；最多 100 次迭代；200 Hz 求解步长；收敛阈值 `1e-6`；原生任务权重；逐帧 warm start；`use_relative_frames=false` | 一致 |
| ARCTIC IK 参数 | `mano_to_robot_scale=1.0`；腕初始偏置 `xyzw=[-0.5,0.5,0.5,0.5]` | 一致 |
| OakInk2、HOT3D、TACO IK 参数 | `mano_to_robot_scale=1.2`；腕初始偏置 `None` | 一致 |

IK 的 `mano_to_robot_scale` 缩放相对腕部的人手目标，不缩放共同物体或计分轨迹。四个数据集各自对应上游的 Sharpa 重定向入口，未列出的数据集没有默认配方；支持的结构与原生限制见 [H2R 工作流](h2r_inputs.md)。

## 机器人、动作与物理

| 项目 | original | 当前 H2R |
| --- | --- | --- |
| 机器人与腕结构 | 原生 Sharpa URDF；每手 3 个平移关节、1 个三自由度 ball 腕和 22 个手指关节 | task 的 `sharpa_wave` MJCF（只含参与手）；每手 XYZ＋roll/pitch/yaw 六个标量腕关节和 22 个手指关节，双手共 56 维坐标 |
| 策略动作／processed pose 维数 | `56 / 58` | 双手一致；单手为 `28 / 29` |
| 动作表示 | `residual_hand_pose`；`input_mode=normalized`，`normalized_mapping=rational`；姿态残差右乘参考四元数 | 一致 |
| 残差 scale：腕位置／腕姿态／手指 | `0.05 m / 0.15 rad / 0.15 rad` | 一致 |
| 残差 clip：腕位置／腕姿态／手指 | `0.2 m / 1.0 rad / 1.0 rad` | 一致 |
| EMA／目标延迟 | `ema=0.3`；延迟 `[0,0]` 控制步 | 一致 |
| 腕目标执行 | 平移位置控制；ball 腕每物理子步计算姿态反馈力矩，`wrist_kp=25`、`wrist_tau_max=45`，阻尼 `6` | 原生位姿残差与 EMA 之后，转为共同串联腕位置目标；按真实关节状态选择连续角度分支；不施加额外 ball 腕力矩 |
| 关节驱动、质量与惯量 | 原生资产和 `SharpaHandsConfig`；默认手指 `kp=1.7, kd=0.1, effort_limit=10`，腕平移 `25 / 6 / 45` | 使用 **h2r_bench 统一分发的机器人物理属性**；从共同机器人资产及其 `compiled.json` 导入质量、惯量、实际驱动增益 `kp/kv` 和限力，替代 FlashCHORD 上游默认参数 |
| 策略频率 `sim.fps` | `20 Hz` | 一致 |
| 训练／方法内诊断的物理子步／频率 | `5 / 100 Hz` | 一致；bench 评估仍使用共同任务的物理步长 |
| 训练 backend | Newton `SolverMuJoCo`／MuJoCo-Warp | 一致 |
| 物体／支撑导入 | 原生物体资产；`decompose_objects=true`；支撑使用 kinematic free joint | task 的 MJCF，`decompose_objects=false`，保留已有碰撞体；桌面、额外盒状支撑与固定 mesh 合并为一个固定到 world 的 body，无额外动态自由度 |
| 物体关节辅助驱动 | 关节物体的铰链 `kp=50, kd=2, effort_limit=50`，随 VOC 缩放 | 对 task 的每个铰链使用同一组数值 |
| 物体尺度／支撑开关 | 尺度范围 `[1.0,1.0]`；`support=true` | 一致 |
| 碰撞配置 | `ground=true`；`robot_support_collision=false` | `ground=false`；`robot_support_collision=true`；保留共同有效碰撞配对 |
| solver 与接触参数 | `newton / implicitfast / pyramidal`；`impratio=20`；`iterations=100`；`ls_iterations=50`；材料取原生资产／场景设置 | solver、积分器、锥模型、迭代数、容差、重力，以及摩擦、`solref/solimp`、`margin/gap` 均读取共同包；运行时覆盖通用 YAML 默认值 |
| solver 容量 | `njmax=512`，`nconmax=256` | `njmax=1024`；`nconmax` 一致 |
| backend 兼容与校验 | 使用原生导入和 solver 默认行为 | 显式启用 MultiCCD；保留共同小惯量和 scalar joint 总限力；实际 CPU／GPU 模型与 task 装配后的 `scene.xml` 逐项比较；target API 改用 Newton 1.5 的 `joint_target_q/qd` |
| 依赖版本 | MuJoCo／MuJoCo-Warp `3.8.1`；Newton `1.2.0`；Warp `1.14.0` | 分别为 `3.11.0`、`1.5.0`、`1.16.0`；移除 SONIC 所用 `onnxruntime-gpu` |

`compiled.json` 是 bench 共同机器人包中的编译参数记录。例如，资产用阻尼比 `dampratio` 配置执行器时，FlashCHORD 读取模型编译后解析出的实际阻尼系数 `kv`，显式写入训练侧模型，避免不同导入器对阻尼比的处理产生差异。质量、惯量、驱动参数和限力均沿用共同机器人，并检查导入后的模型是否与共同资产一致。

上述导入发生在 FlashCHORD 方法侧的训练／诊断环境；其积分时钟恢复为上游的 20 Hz 控制、5 子步、100 Hz 物理，独立于共同评估步长。**在 bench 内正式评估时，由 bench 加载共同模型并执行物理仿真；FlashCHORD 只提交机器人关节位置目标，不重新指定这些物理属性。** 共同物理参数随封存任务确定；相对上游，腕结构和驱动已改变，动作维数、残差参数一致不代表动力学等价。

## 训练超参数与课程

以下为合成后的默认配置，已计入 experiment 对基础配置的覆盖。数值保护及 reset 行为差异单列在下一节。

| 项目 | original | 当前 H2R |
| --- | --- | --- |
| 并行环境／训练预算 | `4096 / 250003456` 环境步 | 一致 |
| 更新预算／actor 更新周期 | `updates_per_collection=8`；`actor_update_period=2` | 一致 |
| 网络结构 | actor：2 blocks、宽度 256；critic：2 blocks、宽度 512；`expansion=4`；2 critics、101 atoms；residual actor head、`unit_gain` mean head | 一致 |
| 策略标准差 | `initial_normalized_std=0.1`；`log_std_min=-10`；`log_std_max=-1.8971199848858813` | 一致 |
| 学习率与 Adam | 初始／峰值 `3e-4`，末值 `1.5e-4`；actor 倍率 `1.0`、residual mean 倍率 `0.1`；Adam `β=(0.9,0.999), ε=1e-8` | 一致 |
| discount／target／BC | `discount=0.99`；`target_tau=0.01`；BC 系数 `0` | 一致 |
| Replay | 容量 `4194304`；最小量 `409600`；batch `2048`；`n_step=3`；均匀采样；换阶段不清空；观测存储 `float32` | 一致 |
| 探索 | `target_sigma=0.1`；初始温度 `0.01`；`zeta_exponent=2`；最长噪声重复 `16` 步；warmup 使用 policy | 一致 |
| 归一化与计算精度 | BatchNorm update rate `0.01`、ε `1e-5`；RMSNorm ε `1e-6`；启用 reward normalization，return bound `5`；混合精度 `float16` | 一致 |
| 观测项 | 腕位姿／速度、手指位置／速度、物体位姿、参考命令、动作历史、接触位置／方向；`observation=policy` | 一致 |
| 有效奖励权重 | 物体关键点 `20`；手关键点／关节各 `0.25`；接触 wrench support `10`；漏接触 `-1`；非预期接触 `-10`；终止 `-100`；动作变化／幅值 L2 `-0.005/-0.002`；contact-force L2 和 force-closure 关闭 | 一致 |
| 奖励形状与接触 wrench | 物体／手关键点／关节为 LAPLACIAN，`var=0.1/0.1/1.0`；wrench tolerance／var 均 `0.1`；512 basis、8 cone edges、`friction_coefficient=0.1` | 一致 |
| VOC 参数 | `k_lin=k_ang=400`，`d_lin=d_ang=40`；`max_force=200`，`max_torque=50`，`gravity=9.81` | 一致 |
| VOC 课程 | `tracking_250m_mixed_reset`；阶段阈值不变；scale 依次为 `1, .75, .5, .25, .1, .05, .025, .01, 0, 0, 0`；最后约 50M 步首帧 reset 概率提高到 `0.5` | 一致 |
| 常规训练终止 | 腕位置误差 `0.15 m`、物体位置 `0.10 m`、物体姿态 `0.50 rad` 或参考结束 | 一致 |
| seed／checkpoint 间隔 | `reset.seed=42`，训练 seed 跟随；每 `10000384` 环境步保存 | 一致 |

奖励中的 `friction_coefficient=0.1` 是 wrench 计算参数，不是共同场景的物理摩擦。训练总步数一致，也不表示相同任务难度或相同训练结果。

## 训练行为与部署差异

| 项目 | original | 当前 H2R |
| --- | --- | --- |
| reset 帧与手指采样 | `reset_finger_openness=0.7`；`reset_to_first_frame_probability=0.1`，末阶段 `0.5`；`first_frame_voc_threshold=0.1` | 一致；默认新增 `immediate_first_frame_probability=0`，不强制所有 episode 使用完整参考手指初态 |
| 首帧 settling | `voc_decay_steps=20`、`voc_decay_mode=step`；准备期保持参考，VOC 从 `1` 切到当前课程值 | `skip_first_frame_settling=true`：采样到第 0 帧即跳过准备期、使用当前课程 VOC；非首帧 reset 的 20 步准备期一致 |
| reset 缓存 | 恢复状态并重置方法组件，未显式调用 solver reset 清理全部历史 | 按 world 清理 solver warmstart、外力、actuator 缓存及接触历史；CUDA graph 重放后检查选中 world 的 reset 有效性 |
| 物理状态数值边界 | 无这组额外边界 | 全部状态须有限；body 相对初始位置每轴 ≤ `10 m`；线速度每分量 ≤ `20 m/s`；角速度 ≤ `200 rad/s`；相应标量关节速度同样受限 |
| 关节位置越界处理 | 无这项额外检查及关联终止 | Hydra `env.state_validity.joint_position_limit_mode`：`ignore` 不处理；`discard` 按无效状态终止并过滤样本；默认 `terminate` 按普通失败终止，保留 terminal transition 和关联 n-step 样本，学习已有 termination 惩罚 |
| 无效 transition／Replay | 无新增的有效性标记和关联样本过滤 | 无效 world 终止、该步奖励置零；排除该 transition 及依赖它的 n-step 窗口；reward normalization 排除无效步并清空该 world 的 return 累积；reset 后无效则停止训练 |
| 网络特征裁剪 | 无 | `normalized_feature_limit=100`：每次 BatchNorm 仿射变换后裁剪至 `[-100,100]`，训练与 checkpoint 推理均生效 |
| 优化器失败处理 | 按原生有限梯度检查跳过更新；无整次更新的原子回滚 | 任一梯度跳过或候选状态／指标非有限，回滚 actor、critic、target、温度、BatchNorm、Adam 和调度器；RNG 和尝试计数继续前进 |
| loss scale／无进展停止 | 初始 `65536`，增长因子 `2`、回退 `0.5`、增长间隔 `2000`；无新增下界／停止阈值 | 初始及增长设置一致；新增 `loss_scale_min=1` 和有限上界；`max_failed_updates=1024`，按 critic 尝试／actor 周期检测无进展，保存失败诊断并报错 |
| 部署环境与动作 | 方法内仿真执行原生控制器 | 新增常驻 socket 控制器；从 bench 读取真实关节、body 和接触反馈，复用原生观测／接触聚合／actor／动作历史，仅提交具名机器人位置目标 |
| 反馈格式 | Newton 原生状态与接触 | 按名称对齐关节；转换四元数、body-origin→COM 速度及接触力方向；缺失、错序或错时钟反馈直接拒绝 |
| 评估初始化、VOC 与结束 | 方法自己的 reset、VOC 和 rollout 配置 | bench 掌握评估物理、初始化规则与计分时钟；方法声明自己的 20 Hz 控制周期并提交首帧机器人参考（完整参考手指、零速度、无 warm-up，位于训练 reset 分布的边缘），不向评估世界施加 VOC、物体控制或自动 reset；末段由 bench 精确截断 |
| checkpoint 绑定与指标 | 原生参考及方法内评估工具 | 记录 `task_id` 等输入身份、采样时刻与 schema，拒绝混用输入／时钟；方法内评估作为诊断，正式 rollout 和指标由 h2r_bench 独立执行 |

上述状态边界、样本过滤、特征裁剪、优化器回滚和首帧 settling 会改变训练行为；它们不是只改日志或字段名称。详细处理见 [数值稳定性](numerical_stability.md)。

关节位置阈值仍为标量关节范围两端各放宽原宽度的 `50%`，由 `env.state_validity.joint_range_margin_fraction=0.5` 管理。默认 `terminate` 在奖励计算前设置终止标记，复用 object error 的 termination 奖励项，不做 terminal bootstrap；默认权重 `-100` 在 20 Hz 下对应该步未归一化奖励贡献 `-5`，多个终止原因同时发生只计一次。新增 `termination/joint_position_limit_fraction_per_completed_episode` 统计（沿用原因互斥统计，优先级在 wrist/object 之后）。若同时出现 NaN/Inf、速度或其他数值边界失败，仍按无效样本丢弃。

`discard` 对应此前 H2R 的无效样本处理，不是 original 上游行为；它只过滤失败 transition 和依赖该步的 n-step 样本，保留更早的有效样本。`ignore` 不因关节位置越界结束或过滤。`discard` 与 `terminate` 都拒绝越界 reset 初态，因为 reset 不是可归因于策略动作的 transition。续训须与 checkpoint 保存的模式一致：不含该字段的旧 checkpoint 视为 `discard`，不一致时拒绝，而不是静默改用默认 `terminate`。选项仅控制方法侧训练，不修改物理 joint limit、执行器 control limit 或 bench 独立评估结束规则。

## 其他保留入口

| 入口 | 与当前默认配方的关系 |
| --- | --- |
| `sim.substeps=6` | 可显式选择的训练时钟：20 Hz 策略、6 子步、120 Hz 训练物理；续训须显式匹配子步数 |
| `experiment=h2r_sharpa_flash_sac_30hz` | 可显式选择的训练时钟：30 Hz 策略、4 子步、120 Hz 物理；部署时声明 30 Hz 控制周期，续训不能隐式改频率 |
| `curriculum=h2r_tracking_250m` | 可选 reset 消融：最后约 50M 步 `immediate_first_frame_probability=1`，首帧、完整参考手指、无 settling、VOC=0；不是默认课程 |
| `experiment=sharpa_flash_sac` | 已恢复原生配方、Sharpa URDF／必要网格、ball 腕、20 Hz／5 子步和 0.5× 速度；仍共享本 fork 的 backend 与数值保护改动，不等于独立 original checkout |
| G1／Dexmate／PPO／SONIC | 相关公开训练入口及专用资产已移除；不属于当前 H2R 支持范围 |

核对依据：[H2R 配方](../src/flash_chord/configs/experiment/h2r_sharpa_flash_sac.yaml)、[原生配方](../src/flash_chord/configs/experiment/sharpa_flash_sac.yaml)、[FlashSAC 参数](../src/flash_chord/configs/training/reference_flash_sac.yaml)、[默认课程](../src/flash_chord/configs/curriculum/tracking_250m_mixed_reset.yaml)、[准备代码](../src/flash_chord/h2r/prepare.py)、[动作适配](../src/flash_chord/h2r/action.py)、[场景适配](../src/flash_chord/h2r/scene.py)、[外部控制器](../src/flash_chord/h2r/controller.py)，以及 original commit 的对应文件。表中比较以源码和 Hydra 合成配置为依据；“一致”不表示已证明完整训练或物理轨迹等价。
