# AVERT ICRA 论文结构（Phase 2）

更新日期：2026-08-10  
状态：**等待用户明确批准；批准前不得进入 Phase 3 论证链或 Phase 4 正文写作。**

## 1. 结构模式与范围

### Structure Pattern

采用 **Conference + IMRaD 混合结构**：ICRA 风格的紧凑会议论文组织，同时保留清晰的 Problem–Method–Experiments 证据链。

当前正文预算为约 **4,900 英文词**，摘要约 **150 词**，参考文献不计。该预算只是结构设计尺度，不代表任何一届 ICRA 的官方页数或格式要求；投稿年份和官方模板尚为 `NOT-CHECKED`，拿到明确年份后需要按模板重新压缩。

### Central Research Question

角色隔离干预与轨迹对齐恢复监督，能否使统一的主动视觉操作策略在视角或操作偏离后恢复，同时维持任务进程？

### Derived Sub-questions

- **SQ1 — 误差机制**：主动视角与操作误差如何通过后续观测和动作预测形成内生闭环分布偏移？
- **SQ2 — 数据构造**：如何从既有成功专家轨迹中构造归因清晰、动力学可行且与任务时间同步的恢复监督？
- **SQ3 — 统一学习**：如何在共享视觉表征上以角色特异扩散分支联合学习名义执行和恢复，而不增加部署期门控或独立恢复器？
- **SQ4 — 实证效果**：AVERT 是否提升视角、左/右操作臂及组合扰动下的恢复和成功率，同时不显著损害名义性能？

上述子问题均由已确认的中心研究问题拆分，不扩大研究范围。

### Paper Flow

论文先说明主动视觉误差与普通操作误差不同：相机位姿偏差会直接改变所有控制角色下一步接收的视觉输入。随后将现有方法分为名义主动视觉、通用纠正模仿学习和模块化/生成式恢复三类，定位其缺少的组合。方法部分先形式化多角色闭环，再给出 AVERT 的策略结构、角色隔离干预和移动专家轨迹对齐。实验围绕“是否能恢复、哪个组件起作用、是否牺牲名义性能、恢复数据是否物理可信”逐级展开，最后讨论成本、成功筛选偏差、仿真/硬件边界与失效条件。

## 2. 详细大纲

### Abstract（约 150 词，不计入正文预算）

**Purpose**：用五个信息单元概括问题、缺口、方法、实验和结论。  
**Serves sub-question**：SQ1–SQ4 总览。  
**Content summary**：

1. 一句说明主动视觉提升可见性，但视角误差会内生地改变后续观测。
2. 一句指出名义示范缺少偏离状态与恢复行为监督。
3. 两句定义 AVERT：角色特异扩散分支；角色隔离、环境闭环的扰动恢复采集；面向推进中专家轨迹的对齐。
4. 一句说明部署期无需失败检测、人工纠正或独立恢复器。
5. 一至两句填入经统计验证的实验结果；在结果未提供前只保留占位符，不使用“显著提升”等结论性措辞。

**Key sources**：AV-ALOHA、DAgger、DART、WM-DAgger、Diffusion Policy。  
**Key argument**：AVERT 针对的是主动视觉引起的多角色闭环恢复监督缺口。  
**Transition**：摘要中的因果链在 Introduction 展开。

### I. Introduction（约 750 词）

**Purpose**：建立问题重要性，解释主动视角误差为何造成特殊的闭环分布偏移，给出受约束的研究缺口与贡献。  
**Serves sub-question**：SQ1 为主，预告 SQ2–SQ4。  
**Content summary**：从精密双臂操作中的遮挡和可见性需求切入，随后给出“视角动作—观测—双角色动作”的反馈链。通过名义主动视觉和通用恢复学习的交叉缺口引出 AVERT，最后用三条贡献收束，不把双分支架构本身表述为首创。

#### I-A. Active vision improves manipulation but creates a coupled failure channel

- 精密双臂任务中的遮挡、局部可见性和操作区域观察需求。
- 主动视觉通过调整相机位姿改善任务相关信息，但相机动作也成为观测生成过程的一部分。
- 以简式因果链表示：
  \[
  \delta a_t^V\!\to\!\delta q_{t+1}^V\!\to\!\delta o_{t+1}
  \!\to\!(\delta a_{t+1}^V,\delta a_{t+1}^A).
  \]
- 与只局部影响执行器状态的普通动作误差作谨慎区分，不声称两者完全独立。

#### I-B. Why nominal demonstrations are insufficient

- 行为克隆只在专家诱导分布上训练，闭环执行误差可能累积。
- DAgger/DART 能覆盖偏离状态，但在线专家标注、统一动作噪声或单角色假设不直接解决主动视角的异构反馈。
- 主动视觉近邻主要学习名义的视角—动作协调；恢复近邻多使用合成观测、门控或独立模块。

#### I-C. AVERT and contributions

- 一句话定义 AVERT 全称及作用对象。
- 贡献 1：环境动力学闭环中的角色隔离恢复数据构造。
- 贡献 2：受扰角色对推进中专家参考的轨迹对齐恢复监督。
- 贡献 3：统一的角色特异扩散策略与三臂平台系统评估。
- 不写“first”；写成“within the literature reviewed”或直接陈述提出了什么。

**Sources**：Closed-Loop NBV、AV-ALOHA、See2Act、DAgger、DART、SPARTN、WM-DAgger、EgoRecovery、RECALL。  
**Key arguments**：主动视角误差具有观测生成层的反馈效应；恢复监督必须保持角色归因和任务时间一致性。  
**Planned visual**：Fig. 1 左半部画误差传播链，右半部画 AVERT 角色隔离恢复概览。  
**Transition to Section II**：Introduction 提出交叉缺口；Related Work 分别验证两条研究脉络为何尚未覆盖该组合。

### II. Related Work（约 700 词）

**Purpose**：在有限篇幅内建立三个必要的比较组，并精确收缩 novelty 边界。  
**Serves sub-question**：SQ1–SQ3。  
**Content summary**：不按论文逐篇罗列，而按“名义主动视觉”“纠正数据”“统一与模块化恢复”综合。每个小节结尾用一到两句落到 AVERT 的差异。

#### II-A. Active perception for robot manipulation

- 从显式 view planning/NBV 过渡到示范学习和联合视角—操作策略。
- AV-ALOHA：同类三臂平台与 ACT 式名义策略。
- Optimizing AP、Observe Then Act、Observer-Actor、See2Act、ActiveGlasses：联合、异步、三维重建、耦合去噪和人类第一视角路线。
- 边界句：这些工作主要优化名义信息获取或协同，没有显式训练视角执行偏离后的角色隔离恢复。

#### II-B. Covariate shift and corrective data augmentation

- DAgger：学习器诱导状态分布；DART：噪声注入获取纠正动作；HG-DAgger/Diff-DAgger：人类或不确定性门控。
- SPARTN、Diffusion Meets DAgger、WM-DAgger：从 NeRF/扩散/世界模型构造离线 OOD 或恢复数据。
- 重点对比 WM-DAgger 的合成连续恢复、固定专家锚点与 AVERT 的目标环境闭环 rollout、异构角色隔离和移动参考。

#### II-C. Unified versus modular recovery

- Latent Policy Barrier、Contractive DIL、Residual RL、EgoRecovery、RECALL 的恢复机制与部署依赖。
- Diffusion Policy 和 ACT 作为统一时序策略主干；Separate to Collaborate 只用于说明双流架构已有先例。
- 边界句：AVERT 的创新中心是恢复监督协议，而不是一般性的多头/双流扩散。

**Sources**：Phase 1 的全部 23 篇文献在本节或其他章节均有明确分配；本节直接覆盖其中 22 篇，Diffusion Policy 还在 Methods 使用。  
**Key arguments**：现有文献提供了问题的三个组成部分，但未覆盖 AVERT 所主张的组合。  
**Planned visual**：若篇幅允许，用一个紧凑比较表替代长段落：方法、是否主动视角、是否环境闭环、是否角色隔离、恢复参考、部署附加模块。  
**Transition to Section III**：文献边界说明“缺什么”；Problem Formulation 把这个缺口转化为可定义的学习问题。

### III. Problem Formulation（约 500 词）

**Purpose**：给出足够精确的符号、角色分解和学习目标，使后续方法不是经验技巧堆叠。  
**Serves sub-question**：SQ1、SQ2。  
**Content summary**：定义专家轨迹、20 维状态/动作、视角与左右操作角色、共享观测以及名义/恢复数据分布。形式化两类误差传播，并定义“恢复到推进中的专家轨迹”而非回到静态姿态。

#### III-A. Multi-role active visual manipulation

- 定义 \(\mathcal D_E=\{\tau_i^E\}\)，\(\tau_i^E=\{(o_t^E,s_t^E,a_t^E)\}\)。
- 分解 \(a_t^E=[u_t^{L,E},g_t^{L,E},u_t^{R,E},g_t^{R,E},u_t^{V,E}]\)。
- 定义角色投影 \(P_\rho\)，\(\rho\in\{V,L,R\}\)，严格区分专家状态 \(q_t^{\rho,E}\) 与动作目标 \(u_t^{\rho,E}\)。
- 策略目标：从共享视觉/本体感知表示输出操作与视角动作序列。

#### III-B. Endogenous observation shift and recovery objective

- 给出 View 与 Arm 两条闭环误差传播路径。
- 定义扰动后的实际环境观测 \(o_t^{\mathrm{rec}}\)，强调它由目标环境动力学和渲染产生，而非图像生成模型直接合成。
- 定义目标：在保持未受扰角色时间同步名义动作的同时，使受扰角色相对移动专家参考的命令偏移收缩，并保留任务成功。
- 明确“解析动作偏移收缩”不等于“实际状态误差必然逐帧单调收缩”。

**Sources**：DAgger、DART、AV-ALOHA、Diffusion Policy、WM-DAgger。  
**Key arguments**：问题的核心不是单独的相机纠偏，而是相机—观测—多角色预测的闭环耦合；恢复参考必须随专家时间推进。  
**Planned visual/equation**：保留 3–4 个核心公式，详细参数和证明式推导移到附录/补充材料（若当届政策允许）。  
**Transition to Section IV**：问题定义给出三个要求——角色隔离、动力学闭环观测、移动参考；AVERT 分别实现它们。

### IV. AVERT（约 1,300 词）

**Purpose**：完整、可复现地描述统一策略与恢复数据生成协议，并明确训练和部署边界。  
**Serves sub-question**：SQ2、SQ3。  
**Content summary**：先介绍共享编码器和角色特异扩散分支，再按时间顺序描述锚点选择、扰动采样、未记录的平滑扰动建立、记录阶段的移动参考恢复、质量筛选和数据合并。正文只保留影响贡献和复现的参数，其余进入表格或补充材料。

#### IV-A. Role-specific diffusion policy with shared visual representation

- 共享双目视觉编码器与本体感知条件。
- 操作分支预测左右臂和夹爪动作序列；视角分支预测 View 臂动作序列。
- 说明共享表示支持跨角色视觉协调，角色特异输出避免把异构动作直接混成一个恢复标签源。
- 名义数据与两类恢复分支共同使用标准扩散监督；无额外恢复损失、恢复门或测试时优化。
- 必须给出准确网络结构、时域、噪声调度、条件注入方式和总损失权重；当前材料不足处标记待补。

#### IV-B. Recovery anchor and bounded perturbation construction

- 每条成功专家轨迹建立合法候选锚点池，满足初始排除、最小间隔和尾窗长度。
- 每条源轨迹固定成功分支配额；失败后先重采样，再在同一时间域搜索临近锚点。
- 六关节有界高斯扰动、限位可行域和最小归一化强度。
- 五次 minimum-jerk/smoothstep 建立与恢复，恢复步数由扰动幅度和附加速度上限自适应决定。
- 注入阶段推进环境时间但不写入训练集；正文说明原因：首个保存观测必须对应已偏离物理状态。

#### IV-C. Role-isolated intervention

- View 分支：只扰动 View，左右臂和夹爪执行同一源时间的专家动作。
- Arm 分支：每次只扰动左或右臂，另一臂、夹爪和 View 执行时间同步专家动作；左右计划分支均衡。
- 用统一选择矩阵写为：
  \[
  a_{t_r^\rho+k}^{\mathrm{rec}}=a_{t_r^\rho+k}^{E}
  +P_\rho^\top\gamma_k^\rho\Delta q_{\mathrm{real}}^\rho.
  \]
- 当前实现以扰动建立后**实际到达的状态偏移**作为 View 和 Arm 恢复曲线初值；采样偏移仅用于建立扰动和记录跟踪误差。
- 对未受扰角色有 \(P_{\rho'}a_t^{\mathrm{rec}}=P_{\rho'}a_t^E\)。

#### IV-D. Trajectory-aligned recovery and quality control

- \(a_{t}^{E}\) 随专家时间推进，AVERT 收缩的是相对移动参考的偏移，而非回到历史静态锚点。
- pre-action 对齐：\((o_k,s_k)\to a_k^{\mathrm{rec}}\)。
- 计划曲线归零后，连续 \(K\) 帧满足状态误差阈值，再记录恢复后稳定段。
- 未受扰角色约束、后缀完整任务成功验证、失败样本丢弃。
- 只保存一份名义轨迹；按 `source_episode` 划分训练/验证，避免原始轨迹与恢复分支跨集合泄漏。
- 明确筛选导致最终经验分布不等于原始截断高斯；需要报告尝试、失败、回退和保存率。

**Sources**：Diffusion Policy、ACT、DART、SPARTN、Diffusion Meets DAgger、WM-DAgger、Separate to Collaborate。  
**Key arguments**：角色隔离提高监督归因清晰度；移动参考保持任务进程；环境闭环 rollout 捕获真实目标环境中的动力学、遮挡和接触演化；统一训练消除部署期附加恢复机制。  
**Planned visuals**：

- Fig. 2：策略架构，明确共享与角色特异部分。
- Fig. 3：时间轴图，比较名义轨迹、View 恢复、Arm 恢复及固定锚点恢复。
- Algorithm 1：压缩版数据生成伪代码。

**Transition to Section V**：方法提出四个可验证设计；实验依次测试总体效果、分角色效果、对齐机制和数据筛选影响。

### V. Experiments（约 1,250 词）

**Purpose**：以可重复、分层的实验回答 SQ4，并分别证明恢复能力和名义性能保持。  
**Serves sub-question**：SQ4，同时回验 SQ1–SQ3。  
**Content summary**：先列研究问题和公平比较协议，再报告主结果、消融、恢复动态与失败案例。当前尚无可核查结果，所有数值和结论均保留为待填，不预写胜出结论。

#### V-A. Experimental setup and evaluation protocol

- 平台：两条操作臂 + 一条主动视角臂；明确当前结果究竟来自 MuJoCo、真实硬件或二者，不能混写。
- 任务：填入多项精密操作的名称、成功判据、时长和遮挡/视角需求。
- 数据：每任务专家 episode 数、每源轨迹 View/Arm 成功分支数、训练/验证/测试按源轨迹划分。
- 训练：统一初始化、训练步数、图像增强、种子和模型选择规则。
- 测试条件：Nominal、View perturbation、Left-arm、Right-arm、Combined/View+Arm；扰动幅度和注入阶段分层。
- 主要指标：任务成功率；辅助指标：恢复成功率、time-to-recover、峰值/终值关节误差、恢复后任务进度、未受扰角色偏差。
- 至少 3 个随机种子；成功率给出置信区间，连续指标报告分布/均值与离散度；统计方案待按试验次数确定。

#### V-B. Baselines and ablations

最低可行训练消融：

1. Expert only；
2. Expert + View recovery；
3. Expert + Arm recovery；
4. Expert + View + Arm recovery（AVERT）；
5. Joint-role perturbation（检验角色隔离）；
6. Fixed-anchor recovery（检验移动轨迹对齐）；
7. Sampled-offset supervision vs. actual-achieved-offset supervision（检验执行偏差处理，若数据允许）。

外部方法/概念基线：AV-ALOHA/ACT 名义策略；DART 式统一动作噪声；SPARTN/WM-DAgger 风格基线仅在可公平实现时纳入，否则提供同预算的近似版本并明确差别。See2Act、Latent Policy Barrier 或 Residual RL 的完整复现取决于代码、渲染器/世界模型和计算预算，不能在不可比条件下制造“更优”结论。

#### V-C. Main results

- Table 1：各任务 × 各测试条件的成功率，报告均值、置信区间和试验次数。
- 分开回答：视角恢复是否提升；Arm 恢复是否提升；联合训练是否互相促进或干扰；名义性能是否保持。
- 若某任务或扰动类型无提升，按真实结果讨论，不只报告平均成功率。
- 对组合扰动结果使用“泛化测试”表述；训练中若从未出现组合扰动，明确说明。

#### V-D. Ablation and recovery dynamics

- 角色隔离 vs. 全角色联合扰动：监督冲突、恢复成功率和未受扰角色偏差。
- 移动参考 vs. 固定锚点：任务进度回退、动作连续性、time-to-recover。
- 名义/恢复数据比例与扰动幅度敏感性。
- 恢复轨迹图：误差随相对恢复时间变化；区分解析命令偏移和实际状态误差。
- 数据生成质量：锚点尝试数、扰动实现误差、失败原因、回退率、最终保存率。

#### V-E. Qualitative analysis and failures

- 成功案例序列：视角偏移后重新建立可见性；Arm 偏移后保持另一角色推进。
- 失败类型：遮挡未解除、接触状态不可恢复、扰动跨角色传播、尾窗不足、策略恢复但任务已不可逆。
- 不把由专家后缀验证通过的数据生成成功率当作学习策略恢复成功率。

**Sources**：AV-ALOHA、See2Act、DART、SPARTN、Diffusion Meets DAgger、WM-DAgger、Diff-DAgger、Latent Policy Barrier、Residual RL。  
**Key arguments**：AVERT 的有效性必须同时表现为受扰条件改善、名义条件保持和恢复动态更合理；单一总成功率不足以证明机制。  
**Transition to Section VI**：实验回答“是否有效”，Discussion 解释有效范围、成本和不能由当前证据推出的结论。

### VI. Discussion and Limitations（约 200 词）

**Purpose**：解释设计权衡、适用边界和威胁，不重复结果。  
**Serves sub-question**：SQ2–SQ4。  
**Content summary**：讨论环境闭环采集相较生成式增强的真实性—成本权衡；成功筛选造成的分布偏差；开环专家相对偏移调度不保证实际误差单调；当前仿真/硬件覆盖边界；对不可逆接触失败和未见扰动组合的限制。

**Sources**：WM-DAgger、SPARTN、Diffusion Meets DAgger、EgoRecovery、RECALL、Contractive DIL。  
**Key arguments**：AVERT 降低了部署复杂度，但训练数据生成仍需要可恢复专家轨迹、环境回放/快照和严格质量筛选。  
**Transition to Section VII**：在承认边界后，用结论回到最小、已被证据支持的贡献。

### VII. Conclusion（约 200 词）

**Purpose**：回答中心研究问题，重述被实验支持的贡献并给出一条克制的未来方向。  
**Serves sub-question**：SQ1–SQ4 汇总。  
**Content summary**：先概括角色隔离干预与移动轨迹对齐，再用真实实验数值回答视角/操作恢复与名义保持；最后指出硬件验证、自动选择干预难度或更复杂不可逆错误是未来工作。若实验只覆盖仿真，结论必须明确限定为 simulation。

**Sources**：不引入新文献；只综合本文证据。  
**Key argument**：AVERT 的价值来自恢复数据协议与统一学习，而非部署时附加恢复系统。  
**Transition**：全文结束。

## 3. Evidence Map

立场标签遵循 ARS handoff：`supports` 表示支持本节动机/设计，`opposes` 表示对过强 novelty 或机制主张构成实质挑战，`neutral` 表示背景或架构定位。

| # | Source | Assigned sections | RQ binding | Stance | Evidence role |
|---:|---|---|---|---|---|
| 1 | View Planning Survey | II-A | SQ1 | neutral | 主动视角规划背景 |
| 2 | Closed-Loop NBV | I-A, II-A | SQ1 | supports | 闭环主动感知前驱 |
| 3 | AV-ALOHA | Abstract, I, II-A, V | SQ1, SQ4 | opposes | 同类三臂名义主动视觉强基线，限制平台 novelty |
| 4 | Optimizing Active Perception | II-A, V | SQ3, SQ4 | opposes | 联合视角—操作扩散近邻 |
| 5 | Observe Then Act | II-A | SQ1, SQ3 | neutral | 异步主动观察路线 |
| 6 | Observer-Actor | II-A | SQ1, SQ3 | neutral | 显式三维观察者—执行者路线 |
| 7 | See2Act | I, II-A, V | SQ1, SQ3, SQ4 | opposes | 耦合视角—动作扩散最强近邻之一 |
| 8 | ActiveGlasses | II-A | SQ1, SQ3 | neutral | 人类第一视角主动视觉示范 |
| 9 | DAgger | Abstract, I-B, II-B, III | SQ1, SQ2 | supports | 协变量偏移理论基础 |
| 10 | DART | Abstract, I-B, II-B, III, IV, V | SQ1, SQ2, SQ4 | supports | 噪声干预与纠正数据前驱 |
| 11 | HG-DAgger | II-B | SQ2, SQ3 | neutral | 人类门控纠正对照 |
| 12 | SPARTN | I, II-B, IV, V, VI | SQ2, SQ4 | opposes | 合成视角纠正增强近邻 |
| 13 | Diffusion Meets DAgger | I, II-B, IV, V, VI | SQ2, SQ4 | opposes | 生成式 OOD 增强近邻 |
| 14 | Diff-DAgger | II-B, V | SQ2, SQ4 | neutral | 不确定性触发专家纠正 |
| 15 | WM-DAgger | Abstract, I, II-B, III, IV, V, VI | SQ2–SQ4 | opposes | 最接近的连续恢复数据生成工作 |
| 16 | Latent Policy Barrier | II-C, V | SQ3, SQ4 | neutral | 独立屏障与推理时优化 |
| 17 | Contractive DIL | II-C, VI | SQ2, SQ3 | neutral | 分布外恢复与收缩性质 |
| 18 | EgoRecovery | I, II-C, VI | SQ2–SQ4 | opposes | 恢复示范与部署门控近邻 |
| 19 | RECALL | I, II-C, VI | SQ2–SQ4 | opposes | 在线主动持续恢复近邻 |
| 20 | Diffusion Policy | Abstract, II-C, III, IV | SQ3 | supports | 策略主干与扩散监督依据 |
| 21 | ACT | II-C, IV, V | SQ3, SQ4 | neutral | 动作分块与平台名义基线 |
| 22 | Residual RL | II-C, V | SQ3, SQ4 | neutral | 在线残差细化对照 |
| 23 | Separate to Collaborate | II-C, IV | SQ3 | opposes | 限制双流/双分支架构 novelty |

所有 Phase 1 核心来源均已至少分配到一个章节。

## 4. 章节转场逻辑

| Boundary | Reader state before transition | Transition logic |
|---|---|---|
| Abstract → Introduction | 知道方法大意，但不知道问题为何特殊 | 展开主动视角动作如何进入观测生成环 |
| Introduction → Related Work | 接受问题与初步缺口 | 用三类文献验证缺口并限制 novelty |
| Related Work → Problem Formulation | 知道现有方法分别缺少哪些条件 | 将交叉缺口写成多角色闭环与移动参考问题 |
| Problem Formulation → AVERT | 明确方法必须满足的三个要求 | 逐一给出策略、干预和对齐机制 |
| AVERT → Experiments | 知道每个设计组件和预期作用 | 将组件转化为 RQ、基线、消融和指标 |
| Experiments → Discussion | 知道性能与恢复动态 | 解释成本、筛选偏差和泛化边界 |
| Discussion → Conclusion | 知道哪些结论不能外推 | 回收为最小且有实验证据的回答 |

## 5. Word Count Summary

| Section | Target English words | Share of body |
|---|---:|---:|
| Abstract | 150 | 不计入正文 |
| I. Introduction | 750 | 15.3% |
| II. Related Work | 700 | 14.3% |
| III. Problem Formulation | 500 | 10.2% |
| IV. AVERT | 1,300 | 26.5% |
| V. Experiments | 1,250 | 25.5% |
| VI. Discussion and Limitations | 200 | 4.1% |
| VII. Conclusion | 200 | 4.1% |
| **Body total** | **4,900** | **100%** |

## 6. 结构质量门与起草前锁定项

### 已通过

- 结构属于认可的 Conference + IMRaD 混合模式。
- 每个顶层章节均有 Purpose、内容摘要、来源、关键论点和转场。
- 正文词数合计准确为 4,900，所有顶层章节不少于 200 词。
- Phase 1 的 23 篇核心文献全部分配到章节，并标注 stance。
- 中心 RQ 的误差机制、数据构造、统一学习和实证验证均有对应章节。

### 起草前必须锁定

1. **用户批准本大纲。** 这是进入 ARS Phase 3 的硬门槛。
2. 提供 ICRA 投稿年份；随后只查当届官方模板并重算版面预算。
3. 确认实验载体：当前恢复生成实现明确使用 MuJoCo；若另有真实硬件结果，需要提供实验路径和数据，不能把仿真写成真实机器人实验。
4. 提供任务名称、数据规模、基线、随机种子、试验次数和当前结果表。
5. 同步方法文档：第 8.3 节仍写 Arm 标签使用采样偏移，而当前代码和第 14.1 节使用实际到达偏移。论文应以当前代码为准，写 \(\Delta q_{\mathrm{real}}\)，并在后续文档修订阶段消除冲突。
6. 明确策略实现细节：共享编码器、两个扩散分支的具体结构、动作维度、预测时域、损失权重和推理方式。

