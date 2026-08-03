# AV-piper 项目交接文档

> 生成时间：2026-07-15（Asia/Shanghai），2026-07-16 更新 LeRobot 迁移状态  
> 项目路径：`/home/dc/dc_project/AV-piper`  
> Python 环境：`/home/dc/miniforge3/envs/AV-piper/bin/python`  
> 本文档用于在新对话中恢复研究与代码上下文，不代替对当前工作树的实际检查。

## 1. 新对话应先做什么

新对话中的助手应按以下顺序恢复现场：

1. 完整阅读本文档。
2. 运行 `git status --short` 和针对目标文件的 `git diff`。
3. 不要删除、回滚或覆盖现有未提交修改；这些修改都应当作用户工作。
4. 注意耦合双头的核心模型代码位于 AV-piper 仓库内的 `lerobot/` 目录。
5. 在继续修改前先判断用户问的是“当前已实现功能”还是“论文研究设想”。

## 2. 研究主线

项目研究三机械臂主动视觉操作：

- 左右两条机械臂负责操作，对应 Arm 动作 `14` 维（每臂6个关节加1个夹爪）。
- 中间机械臂负责主动视角，对应 View 动作 `6` 维。
- 环境动作总维数为 `20`。
- 策略默认预测 `horizon=16` 步，控制频率为 `25 Hz`。
- MuJoCo 物理步长为 `0.002 s`（500 Hz），每个策略环境步内部执行20次物理步，合计 `0.04 s`。

当前研究的两个层次：

1. **耦合双头扩散策略**：同时生成 Arm 和 View 动作，在去噪过程中交换特征。
2. **事件触发式动作块重规划**：不再预先固定每次执行几步，而是根据当前视觉、机器人状态和未执行动作决定是否重新推理。

DPPO 已跑通并有一定效果，但用户当前不打算把 DPPO 本身作为论文主创新，原因是方法不新且实机应用难度较高。

## 3. 关键本地依赖

### 3.1 AV-piper

```text
/home/dc/dc_project/AV-piper
```

当前工作树有大量未提交修改和未跟踪文件。不要使用 `git reset --hard`、`git checkout --`或其他破坏性命令。

### 3.2 项目内置 LeRobot

```text
/home/dc/dc_project/AV-piper/lerobot
```

主要本地修改文件：

- `lerobot/common/policies/diffusion/modeling_coupled_dual_head_diffusion.py`
- `lerobot/common/policies/diffusion/configuration_diffusion.py`
- `lerobot/common/policies/factory.py`
- `lerobot/common/policies/diffusion/modeling_diffusion.py`
- `lerobot/common/logger.py`

该代码已于2026-07-16从原外部本地修改版整体迁入 AV-piper。原源目录的子模块 Git 元数据损坏，无法恢复可靠的基准 commit；来源、许可证和快照摘要见 `lerobot/ORIGIN.md`。迁移后的源码应直接由 AV-piper 的 Git 工作树保护，不再依赖外部兄弟仓库。

### 3.3 原 DPPO 参考代码

```text
/home/dc/DP+RL/dppo/agent/finetune/train_ppo_diffusion_img_agent.py
```

用户更看重对原 DPPO 的复现，因此 DDIM 转移均值、采样标准差、surrogate likelihood 和 PPO ratio 的实现应优先对照该文件，不要仅凭通用 PPO 直觉重写。

## 4. 耦合双头扩散策略

### 4.1 已实现结构

核心文件：

```text
/home/dc/dc_project/AV-piper/lerobot/common/policies/diffusion/modeling_coupled_dual_head_diffusion.py
```

结构要点：

- 继承原有 `DualHeadDiffusionPolicy` / `DualHeadDiffusionModel` 的双头动作划分。
- Arm UNet 和 View UNet 保持独立参数。
- 两个 UNet 在每一个扩散去噪时间步同步运行。
- 在 UNet bottleneck 处将 `[B,C,T]` 特征转换为 `[B,T,C]` token，进行双向交叉注意力：
  - View 特征为 Arm 提供 context。
  - Arm 特征为 View 提供 context。
- 两个方向使用独立 `nn.MultiheadAttention`，允许学习非对称协作。
- 注意力输出以门控残差方式注入原 bottleneck。
- `conditional_sample_coupled()` 在同一去噪轨迹中同时返回 Arm/View 完整动作块。
- 已兼容 Hugging Face `from_pretrained()` 传入普通 `dict` 配置的情况，会先构造 `DiffusionConfig`。

AV-piper 配置：

- `configs/pretrain/policy/pre_zed_coupled_dual_head_diffusion.yaml`
- `configs/finetune/policy/ft_zed_coupled_dual_head_diffusion.yaml`

### 4.2 门控值

当前计算为：

```python
gates = torch.tanh(self.coupling_timestep_encoder(timesteps))
```

重要含义：

- 输出两个门值，分别用于 View-to-Arm 和 Arm-to-View。
- `tanh` 将门值限制到 `[-1,1]`。
- 门值显式依赖的是“扩散时间步”，不是训练 iteration/epoch。
- 网络参数会随训练更新，因此相同扩散时间步的门值也可随训练变化，但没有手工的 epoch schedule。
- 最后一层权重和偏置零初始化，因此初始门值为0，初始结构退化为独立双头，再逐渐学习耦合。

### 4.3 `coupling_num_heads=8` 的正确理解

`8` 不是因为 Arm 和 View 各有4个 token，也不是“每个 token 一套 QKV”。

Multi-head attention 的 head 数表示将 bottleneck channel 维拆成8个子空间并行计算注意力。Q/K/V 是对所有 token 共享的学习投影。当前仅要求 `bottleneck_dim % coupling_num_heads == 0`。

### 4.4 论文创新性边界

用户已发现下列论文与当前双流协同扩散结构很相似：

- *Separate to Collaborate: Dual-Stream Diffusion Model for Coordinated Piano Hand Motion Synthesis*  
  <https://arxiv.org/abs/2504.09885>

因此不应宣称“双流+交叉注意力耦合”整体架构从未被提出。论文应强调与其区别：

- 机器人操作流与主动视觉流的异质角色，而不是两个对称的手部运动流。
- 去噪时间步感知、零初始化、双向非对称门控的实际消融。
- 与事件触发式主动重规划结合，而不仅是离线轨迹生成架构。
- 遮挡、接触偏离、视角稳定和推理成本之间的机器人任务实验。

尚未完成系统的全面文献检索，投 ICRA 前不应把“架构绝对新颖”作为前提。

### 4.5 RBAC 重规划边界感知前瞻耦合

2026-07-16 新增第一阶段 RBAC（Replanning-Boundary-Aware Anticipatory Coupling）实现：

- `DiffusionConfig.coupling_mode` 支持：
  - `full`：原始全时域瓶颈双向交叉注意力，作为兼容基线；
  - `rbac`：只耦合本轮将执行的 View 前缀与本轮不执行的 Arm 预测后缀；
  - `balanced_lookahead`：见4.6，以两个头的future半段分别修正对方current半段。
- 当前 `horizon=16`、`n_action_steps=8`、瓶颈长度为4时，RBAC路由为：
  - View prefix token `[0:3]` 读取 Arm future token `[3:4]`；
  - Arm future token `[3:4]` 读取 View prefix token `[0:3]`；
  - Arm执行前缀和View未执行后缀不接收耦合残差。
- RBAC不新增可学习参数，`full` checkpoint 可严格加载到 `rbac` 模式。
- `DiffusionConfig`字段默认仍为`coupling_mode: full`，避免旧配置无意改变行为；
  当前预训练策略YAML已在4.6实验中显式切换为`balanced_lookahead`。RBAC实验可覆盖：

```text
policy.coupling_mode=rbac
```

- `s1_pretrain`主入口已移除`init_policy_path`：`resume=false`始终按当前
  env/policy全新训练，`resume=true`只允许完整断点续训。冻结基线后的权重
  初始化仍由`s2_incremental`各独立入口管理。
- 已用130000步、评估成功率49%的真实 full checkpoint 严格加载 RBAC，参数总数保持 `50,230,330`。
- 当前只完成结构、梯度、采样、checkpoint兼容和路由回归测试；尚未完成 RBAC 训练后的成功率对照，因此不能声称性能有效。

### 4.6 Balanced Lookahead 对称前瞻耦合

2026-07-20 新增 `coupling_mode: balanced_lookahead`。它针对当前RBAC在4-token
瓶颈上形成3:1不平衡、Arm future只有一个token且Arm-to-View注意力容易退化为
单key复制的问题，不新增模块，而是改变现有双向Cross-Attention的时间路由：

- 将两个头的4个瓶颈token都等分为current `[0:2]` 与future `[2:4]`；
- `View[2:4]`作为key/value修正`Arm[0:2]`，形成2×2注意力；
- `Arm[2:4]`作为key/value修正`View[0:2]`，同样形成2×2注意力；
- 两路future `[2:4]`只提供前瞻条件，耦合残差不直接写回future瓶颈；
- 两个方向读取耦合前的同一份特征快照，避免一次去噪步内的更新顺序偏差。

该模式只改变路由，不增加或改变参数，因此`full`、`rbac`与
`balanced_lookahead` checkpoint可互相严格加载。当前预训练策略配置已显式设置：

```yaml
coupling_mode: balanced_lookahead
view_to_arm_coupling_scale: 0.5
arm_to_view_coupling_scale: 0.5
```

实现对奇数瓶颈token会直接报错，避免静默产生不对称切分。已覆盖等分切片、
前缀更新/后缀瓶颈不变、双向梯度、端到端前向及严格checkpoint兼容测试；尚未有训练
成功率结果，不能预先声称优于无耦合双扩散头。

## 5. DPPO 现状

### 5.1 共享数学实现

文件：

- `train/s3_finetune/dppo_math.py`
- `train/s3_finetune/finetune_dppo.py`
- `train/s3_finetune/finetune_dppo_dual_head.py`

已实现：

- 原 DPPO 风格的 DDIM 转移均值和标准差 `dppo_ddim_mean_std()`。
- 单头 PPO 使用同一个 old/new transition log-probability 计算 ratio。
- 双头不再分别计算两个 PPO loss 再人工加权；改为 Arm/View log-probability 按动作维数 `14:6` 组合成完整20维的 mean log-probability。
- 联合 log-probability 只计算一次 old/new ratio、PPO clip 和 KL。
- 双头语义常量为 `DPPO_RATIO_MODE="joint_dim_mean"` 和 `DPPO_LIKELIHOOD_VERSION="aligned_ddim_joint_v1"`。
- 保留 Arm/View 分头指标作为诊断，但优化目标是 joint ratio。

需要保留的方法边界：当前对原 DPPO 的 behavior likelihood 依然是 DDIM transition surrogate，不是经过裁剪、反归一化和环境动力学后的真实动作分布精确密度。这是对原方法的复现取舍，不应在文稿中表述为精确 likelihood。

### 5.2 动作切片

当前统一为：

```text
n_obs_steps = 2
n_action_steps = 8
action_slice = [n_obs_steps - 1:n_obs_steps - 1 + n_action_steps] = [1:9]
```

动作起点恢复为原版Diffusion Policy规则，由历史观测长度唯一确定，不再提供`action_start`配置参数。`resolve_action_slice()`会检查越界。

### 5.3 采样噪声与 eta

当前主要微调配置使用：

```yaml
min_sampling_denoising_std: 0.01
min_logprob_denoising_std: 0.01
ddim_eta: 0.0
```

含义：

- `min_sampling_denoising_std` 是采集时扩散转移标准差的下限。
- `min_logprob_denoising_std` 必须与行为采样尺度对齐，否则 old/new likelihood 不可比。
- `ddim_eta` 控制 DDIM 时间步依赖的 sigma；当前配置为0，最终随机性主要由 min std 下限保留。
- 当前 std 不会随训练 iteration 自动变小。如果要实现“接近收敛时逐渐减小探索”，需新增 iteration/success-rate 调度，不是仅设置 `eta=0.02, std=0.002` 就会自动发生。

用户曾讨论过 `0.003–0.006` 的 min std，但认为 eta/总体探索可能太小。最终应用 ratio、KL、clip fraction、成功率和动作平滑度消融，不要仅凭单个 std 数值决定。

### 5.4 PPO clip 监测

`train/s3_finetune/dppo_logging.py` 已从 `finetune_dppo.py` 拆分出 W&B 标签和指标构造。主要监测项包括：

- `ppo_ratio_mean/std/min/max`
- `ppo_ratio_outside_clip_fraction`
- `ppo_objective_clip_fraction`
- `ppo_ratio_upper_clip_fraction`
- `ppo_ratio_lower_clip_fraction`
- 逐去噪步 outside/objective clip fraction
- 更新后 fixed probe 的 ratio 分位数与 clip fraction
- log-probability 变化与 advantage 的相关性、符号一致率
- Critic explained variance 和 value-return correlation

`ratio` 超过区间不等于该样本的优化目标一定被裁剪。PPO `min(surr1,surr2)` 只在“正优势超上界”或“负优势低于下界”时真正选择 clipped objective，因此应优先同时查看 outside fraction 和 objective clip fraction。

### 5.5 评估最佳指标

`train/s3_finetune/eval_selection.py` 为单头 `finetune_dppo.py` 提供共享选择规则：

- `best_eval_success_rate` 和 `best_eval_reward` 使用历史 `max()` 更新，应只上升或不变。
- 当前评估曲线仍可以下降；下降的是 current eval，不是 historical best。
- 如果评估已被判定为 collapse 且启用 rollback，该次结果不参与最佳 Actor/Top-K 竞争。

注意：`finetune_dppo_dual_head.py` 仍有自己的最佳模型和回滚逻辑，不要默认它已与单头 `eval_selection.py` 完全合并。

## 6. 预训练和评估相关修改

### 6.1 Gymnasium VectorEnv info 兼容

新增：

```text
train/s1_pretrain/eval/vector_info.py
```

已在以下入口复用：

- `train/s1_pretrain/eval/eval_policy.py`
- `train/s1_pretrain/eval/eval_train.py`
- `train/s4_adaptive_replanning/eval_dynamic_steps.py`
- `train/s4_adaptive_replanning/eval_fixed_steps.py`

兼容：

- 旧式 `final_info` list/object array。
- 新式 Gymnasium dict-of-arrays。
- `_final_info` 和 `_<key>` 有效位掩码。
- 顶层当前步字段和终止步 `final_info` 覆盖。

### 6.2 预训练 W&B 评估指标

`train/s1_pretrain/eval/eval_train.py` 新增 `build_eval_log_metrics()`，评估时上传：

- success rate / percent
- average/std/min/max reward
- average episode steps
- successful episodes / num episodes
- average/max inference latency

`configs/pretrain/pre_default.yaml` 的 W&B project 已从错误的 `*_finetune` 改为 `*_pretrain`，预训练和微调项目名已分开。

### 6.3 视角动作 target 插值平滑

目前这三个参数只在 `train/s1_pretrain/train/train_pretrain_collect_data.py` 中使用：

```python
+training.view_action_smooth_stride=2
+training.view_action_smooth_start=14
+training.view_action_smooth_dim=6
```

含义：

- `stride=2`：对目标动作每2步选一个关键帧，关键帧之间线性插值回25 Hz；设为1关闭。
- `start=14`：从动作向量第14维开始平滑。
- `dim=6`：共平滑 `action[...,14:20]`，即 View/中间臂动作；前14维 Arm 动作保持原始高频目标。
- 会避开 `action_is_pad` 之后的 padding 区域。
- 这是“训练 target 预处理”，不会改环境执行频率，也不是对模型输出做后处理。

## 7. Quest 策略采集随机种子与运行安全

新增：

- `data_collect/episode_seeding.py`
- `data_collect/collection_run_state.py`

修改：

- `data_collect/quest_policy_collect.py`
- `configs/data_collect/quest_policy_collect.yaml`

当前 seed 策略：

- `cfg.random_seed` 只作为进程基础 seed。
- 全局 Python/NumPy/Torch RNG 在进程开始时播种一次。
- 每个采集 attempt 通过 `numpy.random.SeedSequence([base_seed, attempt_index])` 派生独立 uint32 环境 seed。
- attempt 失败也会消耗一个 seed，以避免重复布局。
- 追加采集会从元数据和已保存 episode 恢复 `next_attempt_index`。
- 变更 base seed 或 seed strategy 时会拒绝追加，避免混合不可追溯的数据。
- 增加原子 JSON 写入和非阻塞单写者文件锁，减少崩溃或多进程同时采集破坏元数据的风险。

## 8. 已删除 mjlab 入口

用户确认 `env/mjlab` 已删除且后续不再使用。因此：

- `setup.py` 已删除 `mjlab.tasks` 下的 `av_piper=env.mjlab` entry point。
- `AV_piper.egg-info/entry_points.txt` 已删除。
- `AV_piper.egg-info/SOURCES.txt` 已同步更新。

不要恢复 `env.mjlab` 导入或安装入口。

## 9. 事件触发式重规划 DQN

### 9.1 研究决策

高层离散动作定义为：

```text
0: Arm 和 View 都继续执行当前缓存
1: 只重新规划 View，Arm 继续执行旧缓存
2: Arm 和 View 联合重新规划
```

目标是让控制器根据当前画面、机器人状态和未执行动作决定是否提前重推理，而不是像固定 action chunk 方法一样预先选定执行长度。

### 9.2 代码结构

```text
train/s4_adaptive_replanning/
├── __init__.py
├── dqn.py
├── data_collection.py
├── train_replanning_dqn.py
└── eval_fixed_execution_steps.py

configs/adaptive_replanning/default.yaml
```

`dqn.py` 包含：

- `ReplanningDecision`
- Dueling Q network
- Double DQN target
- Polyak target update
- 动作合法性 mask
- 固定 horizon 剩余动作块编码
- CPU replay buffer

DQN 状态包含：

- 冻结 RGB encoder 的当前视觉特征。
- 当前归一化机器人状态。
- 尚未执行的 Arm 动作块。
- 尚未执行的 View 动作块。
- 剩余动作有效位 mask。
- 当前 action chunk 执行进度。

`data_collection.py` 包含：

- 显式完整16步归一化/环境动作缓存。
- transition 构造和回放池写入。
- 奖励组合：缩放后环境奖励 - 重规划成本 - Arm 动作跳变惩罚。

`train_replanning_dqn.py` 包含：

- 加载并完全冻结预训练耦合双头策略。
- 自动恢复模型配置、归一化统计、环境和相机。
- 绕过原策略内部的8步动作队列，直接使用 `conditional_sample_coupled()` 获得完整 `horizon=16` 动作块。
- 在单个 MuJoCo 环境中在线采集 transition。
- epsilon-greedy、Double DQN 更新、无探索评估、JSONL/W&B 日志、checkpoint 保存和恢复。
- 文件底部 `default_args` 集中保留常用训练参数。

### 9.3 当前实现限制

这一点非常重要：

- **当前只有动0和动2可用。**
- View-only 条件扩散采样尚未实现，动1被 action mask 屏蔽。
- `configs/adaptive_replanning/default.yaml` 中 `training.view_only_available` 必须保持 `false`。
- 训练入口目前只支持单环境，没有 VectorEnv 采集。
- DQN checkpoint 会恢复网络、优化器和全局步数，但不保存 replay buffer；恢复后需要重新预热回放池。
- 已完成代码连通和1步真实 checkpoint 冒烟测试，但本对话中没有完成20万步 DQN 正式训练，也没有 DQN 成功率结论。
- 当前奖励系数是研究起点，尚需消融。

### 9.4 `training.total_env_steps`

`training.total_env_steps=200000` 表示正式 DQN 训练最多执行20万次 `env.step()`，不是20万个 episode，也不是20万次扩散推理。

当前配置中：

```yaml
learning_starts: 5000
train_frequency: 1
updates_per_step: 1
```

大致意味着前5000个环境步只采集，之后每个环境步一次 DQN 更新，理论最多约195000次更新。评估环境步不计入该数值。

## 10. 不同固定推理执行步长评估

文件：

```text
train/s4_adaptive_replanning/eval_fixed_steps.py
```

这里的 `execution_lengths` 表示“每次扩散推理后，固定执行多少个物理动作再重新推理”，不是 diffusion denoising steps。

当 `horizon=16` 时：

```python
execution_lengths=None
```

会自动评估：

```python
[1, 2, 4, 8, 16]
```

也可设置自定义列表，例如：

```python
execution_lengths=[3, 4, 5, 6, 7, 8]
```

注意：当前实现期望 `None` 或整数列表；只评估一个步长时使用 `[8]`，不要使用标量 `8`。所有值必须在 `[1,horizon]` 内，程序会自动去重并排序。

文件底部使用与 `train/s1_pretrain/eval/eval_policy.py` 相同的 `SimpleNamespace` 集中配置方式。正式使用时至少修改：

```python
model_path="你的checkpoint路径"
```

其他值为 `None` 时从模型 `config.yaml` 自动读取。然后直接运行：

```bash
/home/dc/miniforge3/envs/AV-piper/bin/python \
  train/s4_adaptive_replanning/eval_fixed_steps.py
```

报告默认输出到：

```text
outputs/4_replanning_dqn/fixed_step_eval/日期/时间_任务_checkpoint/
```

生成：

- `report.md`：中文汇总和自动推荐。
- `report.json`：完整结构化结果。
- `summary.csv`：每个执行步长的聚合指标。
- `episodes.csv`：每个 episode 的原始记录。

指标包括成功率及 Wilson 95% 区间、平均回报及标准差、episode 步数、推理次数、实际每次推理执行步数和推理耗时。不同步长共用同一组 episode seeds。

当前限制：

- 只支持提供 `_prepare_global_conditioning()`、`conditional_sample_coupled()` 和 `rgb_encoder` 的耦合双头扩散策略。
- 不支持普通单头 Diffusion、非耦合双头、双模型或 ACT。
- 如果要泛化，应新增统一的 `sample_full_action_chunk` adapter，不要复制评估循环和报告代码。
- 当前 checkpoint 中 `eval.n_episodes=100`，默认5种步长将运行500个 episode；调试时应先设 `episodes=1` 或 `10`。

## 11. W&B 和项目内置 LeRobot 日志修改

### 11.1 续训 run 恢复

`/home/dc/dc_project/AV-piper/lerobot/common/logger.py` 已改为：

- 从本地 W&B 文件中选择已成功初始化的 run ID，避免选中失败日志产生的无效 ID。
- 恢复时读取原 run 的 project/entity；当当前配置不同时，使用 previous project/entity 以恢复原 run。
- 视频以文件路径上传时不再向 `wandb.Video` 传 `fps`，因为 MP4 已编码帧率，消除“`fps` argument does not affect...”警告。
- 使用 `wandb.run.url` 替代已弃用 `get_url()`。

已解决过 `from_pretrained()` 将 config 当作 dict 传给耦合策略时的 `AttributeError: 'dict' object has no attribute 'input_shapes'`。

### 11.2 W&B step 单调性已知风险

如果从较旧 checkpoint（例如30000）恢复一个 W&B 已经记录到更高 step（例如32001）的原 run，W&B 会忽略低于当前 step 的日志：

```text
Tried to log to step 30100 that is less than the current step 32001
```

当前代码仍以 checkpoint 训练 step 记录，没有对该情况做独立 W&B step offset，因此这是一个已知限制。实验中应优先：

1. 从该 W&B run 已记录的最新 checkpoint 恢复；或
2. 从较旧 checkpoint 训练时创建新 W&B run，不恢复原 run。

不要为了消除警告盲目改写真实训练 step。

### 11.3 调试图片路径

`modeling_diffusion.py` 中原本硬编码的：

```text
/home/dc/dc_project/input_images
```

已改为根据文件位置解析的 LeRobot 内部 `input_images` 目录，便于迁移。

## 12. 测试和验证

`tests/` 下的文件是开发回归测试，训练和采集入口不会自动调用它们。

当前测试文件：

```text
tests/test_collection_run_state.py
tests/test_coupled_dual_head_diffusion.py
tests/test_dppo_logging.py
tests/test_dppo_math.py
tests/test_episode_seeding.py
tests/test_finetune_eval_selection.py
tests/test_pretrain_eval_metrics.py
tests/test_replanning_dqn.py
tests/test_replanning_dqn_train.py
tests/test_replanning_fixed_step_eval.py
tests/test_vector_info.py
```

2026-07-15 验证命令：

```bash
/home/dc/miniforge3/envs/AV-piper/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -q
```

2026-07-16 加入 RBAC 测试后的结果：

```text
Ran 80 tests
OK
```

额外已完成：

- 使用真实耦合双头 checkpoint 加载冻结策略。
- 构建 `guided_vision/InsertCylinder-3Arms-v0` MuJoCo 环境。
- 生成完整 `16x20` 动作块并执行1个环境步。
- DQN replay/checkpoint 冒烟测试。
- 固定执行步长评估使用新 `SimpleNamespace` 配置完成1 episode/1 step 冒烟测试，成功生成 JSON/CSV/Markdown 报告。

冒烟测试不等于正式性能实验。

## 13. 当前未提交工作树提醒

当前 `git status --short` 显示修改覆盖以下主要区域：

- `setup.py` 和 `AV_piper.egg-info/*`
- `configs/data_collect/*`
- `configs/pretrain/*`
- `configs/finetune/*`
- `data_collect/quest_policy_collect.py`
- `env/assets/piperx_sim.xml`（仅增加执行器 PD 公式注释）
- `train/s1_pretrain/*`
- `train/s3_finetune/*`
- 新增 `train/s4_adaptive_replanning/*`
- 新增 `configs/adaptive_replanning/*`
- 新增 `tests/*`

其中可能包含用户自己的更早修改。新对话不应根据本文档认定全部 diff 都由同一次任务产生，修改任何重叠文件前都应先查看实际 diff。

## 14. 下一步研究建议

建议按以下顺序推进：

1. **固定步长基线**：用同一 checkpoint 和同一组 seeds 完整比较 `1/2/4/8/16`，获得成功率—推理成本曲线。
2. **动态 DQN 的0/2动作版本**：先证明“继续 vs 联合重规划”能否超过最佳固定步长。
3. **View-only 条件采样**：在固定剩余 Arm 轨迹的条件下只重生成 View，再开放 DQN 动1。
4. **奖励和安全性**：分开消融环境奖励、推理成本、Arm 跳变、遮挡恢复和接触偏离。
5. **耦合架构消融**：独立双头 vs 耦合双头；无门控 vs 固定门控 vs 时间步门控；bottleneck 耦合 vs 其他层耦合；单向 vs 双向。
6. **论文定位**：将主要贡献定位为“主动视觉与操作的去噪期协同 + 观测驱动的异步重规划”，而不是仅宣传双流扩散。

## 15. 可直接复制到新对话的开场语

```text
请先完整阅读：
/home/dc/dc_project/AV-piper/PROJECT_HANDOFF.md

然后运行 git status --short，并检查本次任务涉及文件的实际 diff。
当前工作树存在大量未提交修改，请保留所有现有修改，不要重置或覆盖。
耦合双头的核心代码位于 /home/dc/dc_project/AV-piper/lerobot，也需要同时检查。
请基于已有实现继续工作，不要从头重新设计。
```
