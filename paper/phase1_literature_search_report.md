# AVERT 文献检索报告（Phase 1）

更新日期：2026-08-10  
目标会议：ICRA  
研究问题：角色隔离干预与轨迹对齐恢复监督，能否使统一的主动视觉操作策略在视角或操作偏离后恢复，同时维持任务进程？

> 本报告是 ARS Phase 1 的唯一阶段产物，只完成检索、筛查、证据分级与研究缺口界定，不提前撰写论文正文或章节提纲。

## 1. 结论先行

本轮检索支持 AVERT 作为一个可辩护、但必须精确限定的研究方向。已有工作分别覆盖了主动视角—操作联合学习、模仿学习中的协变量偏移、扰动/纠正数据增强、世界模型或生成模型合成恢复数据、独立安全屏障和在线恢复数据采集。然而，在本轮已检索语料中，尚未发现一项工作同时满足以下四个条件：

1. 主动视角控制本身会改变后续策略接收的视觉观测；
2. 每次只干预一个异构控制角色，其他角色继续执行时间同步的名义指令；
3. 使用环境动力学闭环产生的非生成式观测变化和恢复序列，而非只依赖图像/世界模型合成；当前实现为 MuJoCo，不能据此声称已经完成真实硬件恢复采集；
4. 将受扰角色的恢复目标对齐到持续推进的专家轨迹，并在同一策略中联合学习名义执行与恢复，不依赖在线失败门控、人工纠正或独立恢复模块。

因此，论文不宜声称“双分支扩散架构”本身具有首创性，也不宜笼统声称“首次研究机器人恢复”。更稳健的 novelty claim 是：**AVERT 将环境动力学闭环中的角色隔离干预与面向推进中任务轨迹的恢复对齐结合起来，用于多角色主动视觉操作策略的统一训练。** 该结论仅适用于截至 2026-08-10 的已检索语料，不构成无边界的“全球首次”声明。

## 2. 语料来源与检索策略

### 2.1 用户语料来源

- 来源：用户本地 Zotero 文献库的只读数据库快照及已索引 PDF；未修改 Zotero 数据库。
- 快照日期：2026-08-10。
- 重点集合：`主动视觉`、`机器人/扩散策略`、`OOD问题`、`双扩散头`、`human in loop`、`DP+RL` 等。
- 初始种子：上述相关集合合并去重后共 **101 条**学术记录，均有本地 PDF，其中 **91 条**带摘要元数据。
- 本地全文用途：核对方法、数据生成流程、恢复目标、训练/部署依赖和 AVERT 的差异，而非仅根据题名作判断。

### 2.2 外部补缺来源

为降低仅依赖个人文献库产生的覆盖偏差，额外检索并以一手来源复核：arXiv、PMLR、Robotics: Science and Systems Proceedings、CVF Open Access、OpenReview、NeurIPS Proceedings、IEEE DOI/会议记录及作者项目页。外部补缺登记 **12 条**候选，重点覆盖 2024–2026 年恢复学习和主动视觉新工作。

### 2.3 四层检索策略

1. **概念层**：主动视觉、主动感知、视角选择、视觉运动策略、闭环恢复。
2. **方法层**：扩散策略、动作分块、行为克隆、DAgger、噪声注入、纠正增强、OOD 检测、策略屏障、残差强化学习。
3. **机制层**：视角误差引起的观测分布偏移、扰动后的恢复轨迹、异构角色解耦、时间同步监督、移动参考轨迹。
4. **近邻层**：联合视角—动作扩散、NeRF/扩散模型/世界模型生成恢复数据、人类恢复示范、在线主动恢复数据采集。

代表性检索式如下：

- `("active vision" OR "active perception" OR "viewpoint selection") AND ("robot manipulation" OR "imitation learning") AND (policy OR diffusion)`
- `("covariate shift" OR "compounding error" OR recovery OR corrective) AND ("robot imitation learning" OR "visuomotor policy")`
- `("data augmentation" OR perturbation OR "noise injection") AND (recovery OR corrective) AND (trajectory OR demonstration) AND robot`
- `"active vision" AND (error OR perturbation OR recovery) AND "imitation learning"`
- `("world model" OR NeRF OR diffusion) AND ("recovery data" OR "corrective augmentation") AND robot`
- `(camera OR viewpoint) AND perturbation AND recovery AND (bimanual OR manipulation)`

### 2.4 纳入与排除标准

纳入标准：

- 与机器人主动视觉操作、模仿学习分布偏移、恢复数据生成或闭环纠正直接相关；
- 方法和实验足以判断其训练监督、恢复目标与部署依赖；
- 正式会议/期刊论文优先；对高度相关的 2025–2026 年预印本予以保留并明确降级证据强度；
- 能直接支持论文的动机、近邻比较、方法设计或实验基线。

排除标准：

- 仅做静态视觉增强或识别，不涉及策略闭环；
- 仅讨论一般相机标定/视觉伺服，无法支撑学习型恢复问题；
- 与机器人控制无关的生成模型或扩散架构；
- 缺乏可核查原文、方法描述不足或与研究问题仅有关键词重叠；
- 同一工作的重复版本只保留信息更完整或发表状态更高的一版。

### 2.5 筛查流程与停止条件

| 阶段 | Zotero 种子 | 外部补缺 | 合并结果 |
|---|---:|---:|---:|
| 初始候选 | 101 | 12 | 分通道记录 |
| 题名/摘要后候选 | 28 | 8 | 去重后 34 |
| 全文或权威项目页评估 | — | — | 27 |
| 核心纳入 | 18 | 5 | 23 |

停止条件：连续补缺检索未再发现同时具有“主动视角误差—角色隔离干预—环境动力学闭环恢复—移动轨迹对齐”组合的新方法；新增结果主要落入已有类别（联合主动感知、合成纠正数据、在线门控恢复或通用鲁棒模仿学习），因而达到概念饱和。正式投稿前仍应进行一次临近截止日期的增量检索。

## 3. 主题综合

### T1：主动视觉已经从顺序式规划走向视角—操作联合学习

经典 next-best-view 方法通过闭环视角规划改善抓取观测；后续工作将主动视觉纳入模仿学习或强化学习。AV-ALOHA 证明了三臂平台上从人类示范学习主动视角与双臂动作的可行性；Optimizing Active Perception、Observe Then Act、Observer-Actor、See2Act 与 ActiveGlasses 分别采用联合扩散、异步观察—操作、显式三维重建、耦合去噪或对象中心表示。它们为 AVERT 提供强基线，但主要优化名义条件下的观察与操作协同，并未针对视角控制误差引起的内生观测分布偏移构造恢复监督。

### T2：AVERT 的问题根源属于闭环模仿学习的分布偏移

DAgger 从理论和算法上说明，行为克隆在学习策略诱导的状态分布上会累积错误；DART 通过向示范执行注入优化噪声，让专家数据包含纠正动作。HG-DAgger 与 Diff-DAgger 则用人工门控或策略不确定性触发在线专家介入。这组工作支撑 AVERT 的动机，但其主体通常被视为单一控制策略，未显式处理“视角角色改变所有分支后续观测”这一多角色反馈结构。

### T3：纠正数据正在从在线专家采集转向离线生成

SPARTN 使用 NeRF 合成视角扰动与纠正动作；Diffusion Meets DAgger 用扩散模型生成 OOD 图像/样本；WM-DAgger 用世界模型生成连续眼在手相机偏离与恢复。该路线降低人工采集成本，却引入视觉合成误差、物理动力学缺失或世界模型训练成本。AVERT 的差异不应表述为“首次生成恢复数据”，而应强调由环境动力学实际推进得到的非生成式观测、异构角色隔离和时间推进对齐。

### T4：近期恢复方法多依赖门控、独立模块或额外优化

Latent Policy Barrier 在基础扩散策略之外训练动力学/屏障模块并在推理时优化潜变量；Residual RL 在行为克隆策略上叠加在线强化学习残差；EgoRecovery 使用人类恢复示范和部署时恢复门；RECALL 以不确定性引导在线收集失败恢复数据并处理持续微调的遗忘。它们表明恢复能力的重要性，也明确了 AVERT “单一统一策略、训练期构造恢复监督、部署期无额外失败检测”的位置。

### T5：多分支建模是架构基础，不是核心 novelty

Diffusion Policy 和 ACT 分别提供动作扩散与动作分块基础；Separate to Collaborate 等工作表明角色/肢体特异的双流扩散并非空白。因此，AVERT 的架构贡献应写成适配主动视角与操作角色的建模选择，核心创新放在数据干预协议与恢复监督对齐上。

## 4. 最接近工作的边界比较

| 工作 | 与 AVERT 的重合 | 关键差异 | 论文中的处理方式 |
|---|---|---|---|
| WM-DAgger | 从专家轨迹出发构造连续偏离—恢复数据 | 世界模型合成眼在手观测；恢复到同一专家位姿/锚点；不处理异构主动视角角色及其他角色持续推进 | 必须作为最接近的数据生成工作正面对比，并设计环境闭环 rollout 与移动轨迹对齐消融 |
| SPARTN / Diffusion Meets DAgger | 通过视觉扰动扩展训练分布并提供纠正监督 | 主要生成单帧或合成视觉样本，物理闭环与多角色时间同步不足 | 用作“合成纠正增强”基线/讨论对象 |
| See2Act | 在统一扩散过程中耦合动作与 6-DoF 视角更新 | 面向名义联合规划；相机位姿由渲染/数字孪生支撑；未专门构造视角执行误差的真实恢复数据 | 主动视角—操作联合扩散的最强近邻之一 |
| AV-ALOHA | 同类三臂平台、主动视角与双臂动作联合模仿 | ACT 式名义轨迹学习，缺少角色隔离的偏离恢复监督 | 平台和任务设置的直接基线 |
| EgoRecovery | 专门学习机器人恢复，并强调纠正意图 | 依赖人类第一视角恢复示范、少量机器人恢复数据和部署门控 | 区分“收集人类失败恢复”与“从既有专家轨迹构造角色隔离恢复” |
| RECALL | 主动收集恢复数据，并联合保留原任务能力 | 依赖不确定性/失败触发和持续微调；需要处理灾难性遗忘 | 对比统一离线训练与部署期数据闭环 |
| Latent Policy Barrier | 提升扩散策略 OOD 鲁棒性 | 独立动力学/屏障模块与推理时优化 | 用于说明 AVERT 不需要单独 guardian 或在线优化 |
| DART / DAgger | 通过访问学习器诱导状态或注入噪声缓解协变量偏移 | 需要专家在线标注或未区分多角色监督；没有主动视觉特有观测反馈 | 理论动机与通用恢复数据基线 |

## 5. 注释书目

证据质量分级：**高**＝正式同行评审论文且原文可核查；**中高**＝正式论文但与 AVERT 的直接性有限，或近期工作仅能核对正式记录；**中**＝高度相关但仍为预印本；**背景**＝主要用于架构/历史定位，不能单独支撑核心结论。

### A. 主动视觉与联合视角—操作学习

1. **Zeng et al., “A Survey of View Planning for Object Reconstruction and Inspection,” 2020.** 综述视角规划的目标、表示和优化方法，为主动视觉概念与传统 next-best-view 脉络提供背景；不直接研究模仿学习恢复。质量：高；用途：Related Work 的历史背景。

2. **Breyer et al., “Closed-Loop Next-Best-View Planning for Target-Driven Grasping,” 2022.** 使用闭环 next-best-view 规划改善目标驱动抓取，代表显式视角规划路线。它具备反馈修正，但不是统一学习的主动视角—操作恢复策略。质量：高；用途：传统闭环主动感知基线。

3. **Chuang et al., “Active Vision Might Be All You Need: Exploring Active Vision in Bimanual Robotic Manipulation,” ICRA 2025.** 在三臂 AV-ALOHA 平台上从示范联合预测主动视角臂与双臂操作动作，直接证明平台范式的可行性。训练仍以名义示范为主，是 AVERT 最直接的平台/策略基线。质量：高；用途：Introduction、Related Work、实验基线。

4. **Sun et al., “Optimizing Active Perception for Learning Simultaneous Viewpoint Selection and Manipulation with Diffusion Policy,” 2024/2025 preprint.** 以扩散策略同时学习视角选择和操作，并通过 look-at/IK 约束降低视角动作学习维度。重点是名义协同而非误差恢复。质量：中；用途：联合扩散基线与设计对比。

5. **Wang et al., “Observe Then Act: Asynchronous Active Vision-Action Model for Robotic Manipulation,” RA-L 2025.** 将观察与操作异步/顺序组织，用学习的观察策略为操作获取信息；不统一建模视角偏离后的恢复。质量：高；用途：主动观察的时序设计对比。

6. **Wang et al., “Observer-Actor: Active Vision Imitation Learning with Sparse-View Gaussian Splatting,” 2025/2026 preprint.** 借助稀疏视角 Gaussian Splatting 计算观察者视角，再由 actor 操作。强依赖测试时三维表示与显式观察者—执行者流程，与 AVERT 的端到端环境闭环恢复不同。质量：中；用途：显式三维主动视角路线。

7. **Wang et al., “Learning to See While Learning to Act: Diffusion Models for Active Perception in Robot Imitation” (See2Act), 2026 preprint.** 在单个去噪过程中耦合动作生成与 6-DoF 视角细化，是联合视角—动作扩散的最近邻。其训练依赖与关键动作对齐的相机姿态和在线渲染/数字孪生，未聚焦视角执行误差的恢复监督。质量：中；用途：最强方法近邻与基线候选。

8. **Zou et al., “ActiveGlasses: Learning Manipulation with Active Vision from Ego-centric Human Demonstration,” 2026 preprint.** 从人类第一视角示范学习对象中心表示并联合输出操作与头部/视角运动，扩展了主动视觉示范来源。没有以角色隔离干预构造恢复数据。质量：中；用途：人类示范与主动视觉泛化讨论。

### B. 分布偏移、纠正数据与恢复学习

9. **Ross et al., “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning” (DAgger), AISTATS 2011.** 建立学习策略诱导状态分布下的序列误差分析，并通过迭代专家标注缓解行为克隆的协变量偏移。质量：高；用途：AVERT 动机和分布偏移的理论基础。

10. **Laskey et al., “DART: Noise Injection for Robust Imitation Learning,” CoRL 2017.** 在专家示范执行中注入优化噪声，使示范覆盖学习策略可能访问的状态并包含纠正行为。它是 AVERT 干预式恢复采集的重要前驱，但没有异构角色隔离与主动视觉反馈。质量：高；用途：数据构造前驱与基线。

11. **Kelly et al., “HG-DAgger: Interactive Imitation Learning with Human Experts,” 2019.** 由人类门控学习器执行并在危险/失败前接管，从而采集纠正数据。依赖在线人类专家，与 AVERT 无人工纠正的设定相反。质量：中高；用途：交互式恢复采集对比。

12. **Zhou et al., “NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via Novel-View Synthesis” (SPARTN), CVPR 2023.** 利用 NeRF 合成眼在手相机的视角扰动并生成相应纠正动作，实现离线纠正增强。其视觉变化由渲染产生，缺少真实执行动力学和多角色同步。质量：高；用途：视觉纠正增强的关键基线。

13. **Yu et al., “Diffusion Meets DAgger: Supercharging Eye-in-Hand Imitation Learning,” RSS 2024.** 用生成模型合成 OOD 眼在手观测以扩大策略覆盖范围，避免反复在线示范采集。主要解决合成视觉分布扩展，不等价于物理闭环恢复轨迹。质量：高；用途：生成式 DAgger 路线对比。

14. **Lee et al., “Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation,” ICRA 2025.** 利用扩散策略内部信号估计不确定性并触发专家查询/纠正，减少完全由人门控的负担。仍需要部署或数据收集阶段的失败检测与专家帮助。质量：高；用途：无门控部署优势的对比。

15. **“WM-DAgger: World-Model-Augmented DAgger for Visuomotor Policy Learning,” 2026 preprint.** 用世界模型合成连续的眼在手 OOD 偏离与恢复；纠正动作先偏离再对称恢复到同一专家位姿/时间锚点，并保留恢复段训练。它与 AVERT 的恢复数据构造最接近，但不包含由目标环境动力学直接执行的 rollout、异构角色隔离或对推进中名义轨迹的对齐。质量：中；用途：必须重点比较的最近邻。

16. **Sun and Song, “Latent Policy Barrier: Learning Robust Visuomotor Policies by Staying In-Distribution,” NeurIPS 2025.** 在基础扩散策略之外学习动力学和潜变量策略屏障，在推理时优化未来以避免 OOD。恢复能力来自附加模块与在线优化，而非统一策略的恢复监督。质量：高；用途：模块化鲁棒策略对比。

17. **“Contractive Dynamical Imitation Policies for Efficient Out-of-Sample Recovery,” ICLR 2025.** 通过收缩动力系统提高分布外状态的回归能力，并提供动力系统层面的恢复性质。重点是低维/状态空间动力系统，不直接解决视觉观测被主动相机动作改变的问题。质量：高；用途：恢复稳定性相关工作与理论启发。

18. **“EgoRecovery: Learning Robotic Recovery from Egocentric Human Demonstrations,” 2026 preprint.** 从人类第一视角恢复示范提取纠正意图，再结合少量机器人恢复数据，并在部署时使用恢复门。其恢复数据来源和门控部署均与 AVERT 不同。质量：中；用途：最新恢复学习对比。

19. **“RECALL: Uncertainty-Guided Active Continual Learning for Robot Failure Recovery,” 2026 preprint.** 用不确定性选择失败/恢复片段并持续更新策略，同时通过回放或正则化缓解遗忘。需要在线收集与持续微调；AVERT 则计划在离线训练中联合名义与恢复数据。质量：中；用途：主动恢复数据闭环对比。

### C. 策略建模与训练背景

20. **Chi et al., “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,” RSS 2023 / IJRR 2024.** 奠定以条件去噪扩散建模多模态动作序列的视觉运动策略框架，是 AVERT 两个角色特异扩散分支的直接方法基础。质量：高；用途：Methods 和基线。

21. **Zhao et al., “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware” (ACT), 2023.** 使用动作分块和时序集成完成精细双臂操作，也是 AV-ALOHA 的重要策略基础。质量：高；用途：三臂名义策略基线与动作分块讨论。

22. **Ankile et al., “From Imitation to Refinement—Residual RL for Precise Assembly,” ICRA 2025.** 在冻结的分块行为克隆策略上学习闭环残差，以提升精密装配表现。它需要在线强化学习和独立残差策略，与 AVERT 的纯示范式统一训练不同。质量：高；用途：精密任务恢复/细化基线。

23. **Liu et al., “Separate to Collaborate: Dual-Stream Diffusion Model for Coordinated Piano Hand Motion Synthesis,” 2025 preprint.** 用角色/手特异的双流扩散建模协调运动，说明“分支分离后协同”是一种已有架构思想。它并非机器人主动视觉工作。质量：背景；用途：限制双分支架构的 novelty 表述。

## 6. 文献矩阵

主题编码：T1 主动视觉；T2 联合视角—操作；T3 分布偏移；T4 恢复数据；T5 统一/模块化恢复。

| 编号 | 工作简称 | T1 | T2 | T3 | T4 | T5 | 主要方法 | 质量 |
|---:|---|:---:|:---:|:---:|:---:|:---:|---|---|
| 1 | View Planning Survey | ● |  |  |  |  | 综述 | 高 |
| 2 | Closed-Loop NBV | ● | △ |  |  | △ | 显式 NBV 规划 | 高 |
| 3 | AV-ALOHA | ● | ● | △ |  | ● | ACT/示范学习 | 高 |
| 4 | Optimizing AP | ● | ● | △ |  | ● | 联合扩散策略 | 中 |
| 5 | Observe Then Act | ● | ● |  |  | △ | 异步主动视觉—动作 | 高 |
| 6 | Observer-Actor | ● | ● |  |  | △ | 3DGS + 视角优化 | 中 |
| 7 | See2Act | ● | ● | △ |  | ● | 耦合视角—动作扩散 | 中 |
| 8 | ActiveGlasses | ● | ● | △ |  | ● | 人类第一视角示范 | 中 |
| 9 | DAgger |  |  | ● | ● | △ | 在线数据聚合 | 高 |
| 10 | DART |  |  | ● | ● | ● | 示范噪声注入 | 高 |
| 11 | HG-DAgger |  |  | ● | ● | △ | 人类门控纠正 | 中高 |
| 12 | SPARTN | △ |  | ● | ● | ● | NeRF 纠正增强 | 高 |
| 13 | Diffusion Meets DAgger | △ |  | ● | ● | ● | 生成式 OOD 增强 | 高 |
| 14 | Diff-DAgger |  |  | ● | ● | △ | 扩散不确定性门控 | 高 |
| 15 | WM-DAgger | △ |  | ● | ● | ● | 世界模型连续恢复 | 中 |
| 16 | Latent Policy Barrier |  |  | ● | △ | △ | 屏障/推理时优化 | 高 |
| 17 | Contractive DIL |  |  | ● | △ | ● | 收缩动力系统 | 高 |
| 18 | EgoRecovery |  |  | ● | ● | △ | 人类恢复示范 + 门控 | 中 |
| 19 | RECALL |  |  | ● | ● | △ | 不确定性主动持续学习 | 中 |
| 20 | Diffusion Policy |  |  | △ |  | ● | 动作扩散 | 高 |
| 21 | ACT |  | △ | △ |  | ● | 动作分块 Transformer | 高 |
| 22 | Residual RL |  |  | ● | △ | △ | BC + 在线残差 RL | 高 |
| 23 | Separate to Collaborate |  | △ |  |  | ● | 双流扩散 | 背景 |

注：● 表示直接相关，△ 表示间接相关或只覆盖部分问题。

## 7. 研究缺口与可检验主张

### Gap 1：主动视觉误差的“内生观测分布偏移”缺乏专门建模

通用模仿学习研究的是动作误差导致状态分布偏移；主动视觉中，相机控制误差还会直接改变下一时刻所有视觉分支的输入。现有主动视觉操作工作大多展示视角调节收益，但未把视角执行偏差作为训练分布的显式维度。

可检验主张：相同幅度扰动下，视角扰动相较于只作用于局部执行器的扰动，会造成更广泛或更持久的视觉表征偏移；加入视角恢复数据可显著缩短恢复时间并提高成功率。

### Gap 2：通用噪声注入容易混合异构控制角色的监督

DART 式全动作扰动和常见联合动作增强没有区分主动视角角色与操作角色。若多个角色同时偏离，就难以判断后续纠正动作应归因于哪一误差源，且可能破坏任务进程。

可检验主张：相较于联合扰动，角色隔离干预在相同数据预算下产生更低的监督冲突，并提高单角色及组合扰动下的恢复性能。

### Gap 3：恢复到静态锚点可能与持续推进的任务不一致

WM-DAgger 的恢复目标与同一专家时间点/位姿对齐，适合可回到锚点的设置；精密双臂任务中的其他角色仍在运动时，静态回归可能引入时间错位或跨角色冲突。

可检验主张：移动轨迹对齐相较于固定锚点恢复，可减少恢复后的动作不连续、任务进度回退和双臂/视角不同步。

### Gap 4：恢复通常需要额外部署机制

现有近期方法常依赖不确定性检测、人类接管、恢复门、独立屏障或在线强化学习。它们有效但增加部署复杂度，并可能引入误触发或漏检。

可检验主张：AVERT 在不增加在线门控和独立恢复模块的条件下，能够同时维持名义性能并改善受扰性能。

### Gap 5：环境动力学闭环恢复与合成增强之间缺少直接比较

NeRF、扩散生成和世界模型可以低成本扩大视觉分布，但难以完整复现目标环境中的相机/机械臂动力学、遮挡变化和接触进程。AVERT 的环境闭环采集更昂贵，但可能具有更高的恢复可信度；当前证据只覆盖 MuJoCo，真实硬件优势仍需单独验证。

可检验主张：在相同恢复样本数下，环境动力学闭环数据相较于纯视觉合成增强能更好地应对同一目标环境中的物理扰动；若混合训练更优，则应如实将贡献调整为数据协议而非数据来源本身。跨越仿真到真实硬件的迁移不在未经验证时预设。

## 8. 建议的论文分节用文献

### Introduction

- 主动视觉与三臂平台动机：AV-ALOHA、Closed-Loop NBV、See2Act。
- 协变量偏移与错误累积：DAgger、DART。
- 现有恢复方法的部署代价：Diff-DAgger、Latent Policy Barrier、EgoRecovery、RECALL。
- 最近邻与缺口：WM-DAgger、SPARTN、Diffusion Meets DAgger。

### Related Work

- Active vision for manipulation：View Planning Survey、Closed-Loop NBV、AV-ALOHA、Optimizing AP、Observe Then Act、Observer-Actor、See2Act、ActiveGlasses。
- Robust imitation and corrective data：DAgger、DART、HG-DAgger、Diff-DAgger、SPARTN、Diffusion Meets DAgger、WM-DAgger。
- Unified versus modular recovery：Latent Policy Barrier、Contractive DIL、EgoRecovery、RECALL、Residual RL。
- Policy backbone：Diffusion Policy、ACT；Separate to Collaborate 仅用于架构边界。

### Methods

- Diffusion Policy：条件扩散与动作序列建模依据。
- DART / DAgger：干预式数据覆盖的概念来源。
- WM-DAgger：明确固定锚点恢复与 AVERT 移动轨迹对齐的差别。
- Separate to Collaborate：说明分支隔离是设计基础，避免夸大架构原创性。

### Experiments

- 名义主动视觉基线：AV-ALOHA、Optimizing AP、See2Act。
- 恢复数据基线：DART 式噪声、SPARTN/生成式增强、WM-DAgger 式固定锚点对齐。
- 模块化恢复基线：Latent Policy Barrier 或 Residual RL（若复现成本可接受）。
- 门控/交互式方法只在资源允许时纳入；否则在限制中说明无法公平复现在线专家或世界模型依赖。

## 9. 证据分布与偏差提示

- **方法学偏斜**：23 篇核心文献中，22 篇属于算法、仿真或机器人实验研究，仅 1 篇为综述，计算/实验机器人方法占 **95.7%**，超过 70% 提示阈值。对 ICRA 方法论文而言这一偏斜具有领域合理性，但不能据此推断真实部署的长期安全性或跨平台普适性。
- **时间偏斜**：主动视觉联合扩散和恢复学习的最接近工作集中在 2024–2026 年。近期性有助于定位前沿，但 2026 年多篇工作仍为预印本，其结论只能作为中等强度证据。
- **平台偏斜**：不少纠正增强工作聚焦眼在手相机或单臂操作；直接覆盖独立主动视角臂、双操作臂和多角色同步的工作较少。这既构成研究缺口，也意味着基线复现可能需要适配。
- **地域/群体属性**：多数论文未提供适合本研究问题的统一地域或人口属性，故不对该维度做定量推断。

响应：保留高相关预印本以覆盖快速演进前沿，但所有核心机制主张尽可能由正式同行评审工作提供基础；论文定稿前复查这些预印本的发表状态，并对近邻工作做一次增量检索。

## 10. 已核查的一手入口

- Active Vision Might Be All You Need: https://arxiv.org/abs/2409.17435
- Optimizing Active Perception: https://arxiv.org/abs/2409.14615
- Observe Then Act: https://arxiv.org/abs/2409.14891
- Observer-Actor: https://arxiv.org/abs/2511.18140
- See2Act: https://arxiv.org/abs/2606.23625
- ActiveGlasses: https://arxiv.org/abs/2604.08534
- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- DART: https://proceedings.mlr.press/v78/laskey17a.html
- SPARTN: https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_NeRF_in_the_Palm_of_Your_Hand_Corrective_Augmentation_for_CVPR_2023_paper.html
- Diffusion Meets DAgger: https://www.roboticsproceedings.org/rss20/p048.html
- Diff-DAgger: https://diffdagger.github.io/
- WM-DAgger: https://arxiv.org/html/2604.11351
- Latent Policy Barrier: https://papers.nips.cc/paper_files/paper/2025/hash/feaf23597f3c93803d1b4230aadc8f44-Abstract-Conference.html
- Contractive Dynamical Imitation Policies: https://openreview.net/forum?id=lILEtkWOXD
- EgoRecovery: https://arxiv.org/abs/2607.19745
- RECALL: https://arxiv.org/abs/2606.23617
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- ACT: https://arxiv.org/abs/2304.13705
- Residual RL for Precise Assembly: https://arxiv.org/abs/2407.16677
- Separate to Collaborate: https://arxiv.org/abs/2504.09885

## 11. Phase 2 的输入约束

下一阶段的论文结构必须遵守以下约束：

1. 将贡献核心放在角色隔离干预、由环境动力学闭环产生的非生成式观测和移动轨迹对齐，而不是双扩散头本身。
2. 在 Related Work 和实验中显式讨论 WM-DAgger、See2Act、AV-ALOHA、SPARTN/Diffusion Meets DAgger 与近期恢复工作。
3. 把“内生观测分布偏移”形式化为主动视角动作影响后续观测和两个控制角色预测的反馈链。
4. 所有性能结论保持为待验证假设，直至用户提供或运行实验。
5. 至少设计名义数据、全角色联合扰动、角色隔离但固定锚点、角色隔离且移动轨迹对齐、独立/门控恢复等消融或对比。
