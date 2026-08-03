# View–Arm 双角色可收缩扰动恢复数据生成方法

建议英文名称：**Expert-Relative Minimum-Jerk Perturbation-and-Recovery Augmentation**（基于移动专家参考与最小加加速度偏移收缩的扰动恢复增强）。

本文档按照机器人学习论文中 Method 章节的写作规范，形式化描述项目当前实现的 View 与 Arm 扰动恢复数据生成方法。除“实现说明与限制”部分外，正文可以直接作为论文方法章节的中文初稿；公式和参数均与当前代码保持一致。

## 1. 问题定义

给定专家示范数据集


\mathcal D_E=\{\tau_i^E\}_{i=1}^{N_E},


其中第 \(i\) 条专家轨迹为


\tau_i^E=\{(o_t^E,s_t^E,a_t^E)\}_{t=0}^{T_i-1}.


o_t^E 表示由主动视觉双目相机获得的图像观测，s_t^E 表示机器人关节状态，a_t^E 表示专家动作。当前三臂系统的状态和动作均为 20 维：


a_t^E=
\left[
u_t^{L,E},g_t^{L,E},
u_t^{R,E},g_t^{R,E},
u_t^{V,E}
\right],


其中 u_t^{L,E},u_t^{R,E},u_t^{V,E}\in\mathbb R^6 分别为左操作臂、右操作臂和 View 臂的六维关节位置动作目标，g_t^{L,E},g_t^{R,E}\in\mathbb R 为左右夹爪动作。本文严格区分专家实际关节状态与专家动作目标。对角色 \rho\in\{V,L,R\}，定义


q_t^{\rho,E}=P_\rho s_t^E,
\qquad
u_t^{\rho,E}=P_\rho a_t^E,


其中 \(P_\rho\in\{0,1\}^{6\times20}\) 是角色投影矩阵。扰动初态和恢复误差相对于 \(q_t^{\rho,E}\) 定义，监督动作则相对于 \(u_t^{\rho,E}\) 构造，二者不能混用。

本文的目标不是学习一个独立恢复控制器，而是从成功专家轨迹构造带有受控分布偏移的恢复示范：在轨迹中注入有界关节扰动，使被扰动角色沿平滑轨迹收缩回随时间变化的专家参考，同时保持未扰动角色的专家动作不变。生成的数据仍使用原双头扩散策略进行普通监督学习，不额外修改模型损失。

在主动视觉闭环中，View 与 Arm 的输出误差分别通过两条物理路径诱发下一时刻的观测分布偏移：


\delta a_t^V
\rightarrow
\delta q_{t+1}^V
\rightarrow
\delta o_{t+1}
\rightarrow
(\delta a_{t+1}^V,\delta a_{t+1}^A),



\delta a_t^A
\rightarrow
\delta s_{t+1}^{\mathrm{robot/object}}
\rightarrow
\delta o_{t+1}
\rightarrow
(\delta a_{t+1}^V,\delta a_{t+1}^A).


因此，分别构造 View 与 Arm 的受控恢复分支，可以在保持另一角色专家监督不变的条件下，隔离并学习两类闭环误差传播。

## 2. 双角色非对称恢复监督

我们考虑两类互斥的恢复分支：

### 2.1 View 恢复分支

只扰动 View 臂，并保持两个操作臂和夹爪动作不变：


\delta q_t^V\neq 0,
\qquad
a_t^{A,\mathrm{rec}}=a_t^{A,E},


其中 \(a^A=[u^L,g^L,u^R,g^R]\) 表示完整 Arm 动作。该分支用于构造“相同任务进程、不同相机位姿”下的视觉观测，并监督 View 动作收缩回专家轨迹。

### 2.2 Arm 恢复分支

每条分支只扰动左臂或右臂中的一只。设被选中的操作臂为 \(B\in\{L,R\}\)，则

\[
\delta q_t^B\neq 0,
\qquad
u_t^{\bar B,\mathrm{rec}}=u_t^{\bar B,E},
\qquad
g_t^{L,\mathrm{rec}}=g_t^{L,E},
\qquad
g_t^{R,\mathrm{rec}}=g_t^{R,E},
\qquad
u_t^{V,\mathrm{rec}}=u_t^{V,E},
\]

其中 \(\bar B\) 表示未受扰动的另一只操作臂。左右臂恢复分支在整个计划数据集中交替分配，使两侧**计划分支**数量之差不超过 1；由于失败分支会被丢弃，最终成功保存的左右臂样本数不保证严格均衡。

这种非对称监督保留了清晰的条件关系：View 恢复时 Arm 标签不变，Arm 恢复时 View 标签不变。因此，双头策略可以从同一监督框架中分别学习视点误差恢复和操作状态误差恢复，而不会混淆两个扰动来源。

## 3. 恢复事件采样

对每条长度为 \(T\) 的成功专家轨迹，在合法时间范围内随机采样恢复事件起点 \(t_0\)。每个候选帧以概率 \(p\) 独立成为候选事件：


z_t\sim\operatorname{Bernoulli}(p).


若伯努利候选在最小间隔约束下不足 3 个，代码会使用同一随机种子从合法区间重构事件集合，保证在数学上可行时达到最小事件数。因此，最终事件分布是受到事件数量、间隔和尾部长度共同约束的条件分布，而不是纯粹的独立伯努利分布。

事件集合还必须满足：


t_0\ge T_{\mathrm{initial}},



|t_i-t_j|\ge T_{\mathrm{interval}},\qquad i\neq j,



t_0\le T-H,


其中 \(H\) 为注入、恢复、稳定确认和恢复后记录所需的最坏情况剩余长度。当前配置为：

- 候选概率 \(p=0.015\)；
- 初始排除长度 \(T_{\mathrm{initial}}=16\) 帧；
- 最小事件间隔 \(T_{\mathrm{interval}}=50\) 帧；
- 每条源轨迹生成 3～5 个事件。

View 的未记录扰动建立阶段不推进物理时间，其最坏情况尾部长度为


H_V=N_{\max}+N_{\mathrm{extra}}+N_{\mathrm{post}}+1=87.


Arm 分支的未记录扰动阶段会推进专家时间轴，因此


H_A=S+N_{\max}+N_{\mathrm{extra}}+N_{\mathrm{post}}+1=97.


## 4. 有界关节扰动模型

对被选角色的六个关节独立采样零均值高斯扰动：


\widetilde{\Delta q}_j\sim\mathcal N(0,\sigma_j^2),
\qquad j=1,\ldots,6.


通过拒绝采样将其限制在有效扰动域内：


|\Delta q_j|\le b_j^{\mathrm{eff}},


其中当前 View 和 Arm 共用


\boldsymbol\sigma=
[0.025,0.025,0.025,0.040,0.040,0.050]\ \mathrm{rad},



\mathbf b^{\mathrm{cfg}}=
[0.060,0.060,0.060,0.100,0.100,0.120]\ \mathrm{rad}.


为避免产生近似专家轨迹的无效样本，还要求归一化扰动强度满足


\left\|\Delta q\oslash\mathbf b^{\mathrm{eff}}\right\|_2\ge\eta,
\qquad \eta=0.5,


其中 \(\oslash\) 表示逐元素除法。

### 4.1 基于关节限位的可行域

令 \(\underline q_j,\overline q_j\) 分别为关节和执行器控制范围交集的下、上界，\(m=0.005\ \mathrm{rad}\) 为安全余量。对恢复窗口 \(\mathcal W\) 内的专家状态与动作目标集合 \(\mathcal Z_j\)，扰动可行区间定义为


\ell_j=
\max\left(
-b_j^{\mathrm{eff}},
\max_{z\in\mathcal Z_j}(\underline q_j+m-z)
\right),



u_j=
\min\left(
b_j^{\mathrm{eff}},
\min_{z\in\mathcal Z_j}(\overline q_j-m-z)
\right).


最终采样必须满足


\ell_j\le\Delta q_j\le u_j.


该约束保证专家状态和专家动作叠加完整扰动后仍不会越过 MuJoCo 关节限位或执行器控制限位。

因此，在给定事件时刻 \(t_0\) 和角色 \(\rho\) 后，被接受扰动的条件密度可写为


p(\Delta q^\rho\mid t_0,\rho)
\propto
\prod_{j=1}^{6}
\exp\left(-\frac{(\Delta q_j^\rho)^2}{2\sigma_j^2}\right)
\mathbb I[\boldsymbol\ell\le\Delta q^\rho\le\mathbf u]
\mathbb I\!\left[
\|\Delta q^\rho\oslash\mathbf b^{\mathrm{eff}}\|_2\ge\eta
\right].


这里描述的是扰动生成器的提议分布；最终写入训练集的样本还会经过执行跟踪、恢复稳定性和完整任务成功筛选，因而最终经验分布并非未经筛选的截断高斯分布。

## 5. 五次最小加加速度收缩曲线

定义归一化进度 \(u\in[0,1]\)，并采用五次平滑函数


s(u)=10u^3-15u^4+6u^5.


它满足


s(0)=0,\quad s(1)=1,


s'(0)=s'(1)=0,



s''(0)=s''(1)=0.


因此，解析附加偏移的连续时间参数化在两个端点均具有零速度和零加速度。该性质不等同于完整动作命令或真实机器人状态在动力学执行中必然连续。恢复阶段使用剩余扰动比例


r(u)=1-s(u)
=1-10u^3+15u^4-6u^5,


满足 \(r(0)=1\) 和 \(r(1)=0\)。

## 6. 自适应恢复时长

五次曲线关于归一化时间的最大一阶导数为 1.875。给定第 \(j\) 个关节允许的最大附加恢复速度 \(v_j^{\max}\)，完成扰动 \(|\Delta q_j|\) 的最小时间为


T_j=\frac{1.875|\Delta q_j|}{v_j^{\max}}.


整个六维恢复过程由最慢关节决定：


T_{\mathrm{rec}}=\max_j T_j.


在控制频率 \(f\) 下，计划恢复步数为


N=
\operatorname{clip}
\left(
\left\lceil fT_{\mathrm{rec}}\right\rceil,
N_{\min},N_{\max}
\right).


当前使用 \(f=25\ \mathrm{Hz}\)、\(N_{\min}=20\)、\(N_{\max}=40\)。

由于恢复步数存在上限，代码还会依据最大恢复速度反向收紧扰动上限：


b_j^{\mathrm{eff}}
=
\min\left(
b_j^{\mathrm{cfg}},
\frac{v_j^{\max}N_{\max}}{1.875f}
\right).


实际拒绝采样使用 \(b_j^{\mathrm{eff}}\)，第 4 节中的扰动上限和归一化强度在运行时也相应使用 \(\mathbf b^{\mathrm{eff}}\)。这避免了恢复时长被截断到 \(N_{\max}\) 后仍突破设定的附加速度上限。

View 分支使用配置中给定的每关节速度上限：

\[
\mathbf v_V^{\max}=
[0.292,0.266,0.160,0.090,0.366,0.149]\ \mathrm{rad/s}.
\]

代入当前 \(N_{\max}=40\) 和 \(f=25\ \mathrm{Hz}\) 后，View 实际使用的有效扰动上限为

\[
\mathbf b_V^{\mathrm{eff}}
=
[0.060,0.060,0.060,0.0768,0.100,0.120]\ \mathrm{rad}.
\]

其中第 4 维由配置值 \(0.100\ \mathrm{rad}\) 自动收紧为 \(0.0768\ \mathrm{rad}\)。

Arm 分支将所有合格且能够容纳完整注入—恢复尾窗的源专家轨迹中，左右臂的关节动作差分合并，并计算每个关节速度绝对值的第 90 百分位：


\mathcal V_{A,j}
=
\left\{
f\left|u_{e,t+1,j}^{B,E}-u_{e,t,j}^{B,E}\right|
\;\middle|\;
e\in\mathcal D_E,\ B\in\{L,R\},\ 0\le t<T_e-1
\right\},



\widehat v_{A,j}=Q_{0.90}(\mathcal V_{A,j}).


应用最小速度下界 \(v_j^{\mathrm{floor}}\) 和统一比例 \(\alpha_v\) 后，得到


v_{A,j}^{\max}
=
\alpha_v\max(\widehat v_{A,j},v_j^{\mathrm{floor}}).


当前 \(v_j^{\mathrm{floor}}=0.05\ \mathrm{rad/s}\)，\(\alpha_v=1\)。

## 7. View 扰动恢复轨迹

### 7.1 未记录的 View 扰动建立

在事件帧 \(t_0\) 恢复完整环境快照。由于 View 臂不参与物体接触，当前实现在不调用 `physics.step()` 的情况下，通过 MuJoCo `qpos` 插值建立反事实相机位姿：


q_{V}^{\mathrm{setup}}(u)
=
q_V^{\mathrm{start}}
+s(u)
\left(
q_{t_0}^{V,E}+\Delta q^V-q_V^{\mathrm{start}}
\right).


设置阶段共使用 \(S_V=10\) 个插值点，但不推进任务物理时间、不渲染图像，也不写入训练数据。最终强制满足


q_{t_0}^{V,\mathrm{actual}}
=q_{t_0}^{V,E}+\Delta q^V.


随后将 View 关节速度置零，并把对应 actuator control 同步到该目标。当前实现使用绝对误差 \(10^{-10}\) 检查实际偏移与采样偏移的一致性。这里的 10 个插值点仅用于建立平滑的运动学状态序列，没有真实控制步或模拟时间含义。

### 7.2 View 恢复标签

对第 \(k\) 个恢复动作，令


u_k=\frac{k+1}{N},
\qquad k=0,\ldots,N-1.


View 动作标签为


u_{t_0+k}^{V,\mathrm{rec}}
=
u_{t_0+k}^{V,E}
+r(u_k)\Delta q^V.


完成 \(N\) 个计划恢复步后，View 动作标签强制恢复为对应专家动作目标，即对 \(k\ge N\) 有 \(u_{t_0+k}^{V,\mathrm{rec}}=u_{t_0+k}^{V,E}\)。

所有 Arm 与夹爪动作严格保持专家值：


a_{t_0+k}^{A,\mathrm{rec}}
=a_{t_0+k}^{A,E}.


由于专家动作参考 \(u_{t_0+k}^{V,E}\) 随时间变化，该方法恢复到的是移动中的专家轨迹，而不是固定关节姿态。

## 8. Arm 扰动恢复轨迹

### 8.1 受扰 Arm 的均衡选择

对所有计划分支身份进行确定性排序，并以随机种子决定起始侧，然后在左、右臂之间交替分配：


B_n\in\{L,R\},
\qquad
B_n\neq B_{n+1}.


该设计使计划分支中的左右臂数量之差不超过 1，同时保证中断续生成时的角色选择可复现。

### 8.2 未记录的物理扰动注入

Arm 可能处于抓取、交接或放置接触状态，因此不能像 View 一样直接修改 `qpos`。当前实现通过真实 MuJoCo 控制步平滑注入扰动，并同步推进专家时间轴。对注入阶段第 \(k\) 步：


u_k^{\mathrm{in}}=\frac{k+1}{S_A},
\qquad S_A=10,



u_{t_0+k}^{B,\mathrm{setup}}
=
u_{t_0+k}^{B,E}
+s(u_k^{\mathrm{in}})\Delta q^B.


同时保持


u_{t_0+k}^{\bar B,\mathrm{setup}}
=u_{t_0+k}^{\bar B,E},



g_{t_0+k}^{L,\mathrm{setup}}=g_{t_0+k}^{L,E},
\qquad
g_{t_0+k}^{R,\mathrm{setup}}=g_{t_0+k}^{R,E},



u_{t_0+k}^{V,\mathrm{setup}}
=u_{t_0+k}^{V,E}.


注入过程执行真实动力学和任务奖励更新，但不保存为训练数据。这 \(S_A=10\) 步替换专家时间轴上原有的 10 个控制步，而不是向轨迹额外插入 10 步。注入结束后的首个记录帧对应专家索引


t_s=t_0+S_A.


此时实现偏移为


\Delta q_{\mathrm{real}}^B
=q_{t_s}^{B,\mathrm{actual}}-q_{t_s}^{B,E}.


仅当


\|\Delta q_{\mathrm{real}}^B-\Delta q^B\|_\infty
\le 0.020\ \mathrm{rad}


时才进入记录阶段；否则恢复快照并重新采样。

### 8.3 Arm 恢复标签

当前代码以采样扰动 \(\Delta q^B\) 为恢复曲线锚点。对恢复阶段第 \(k\) 步：


u_{t_s+k}^{B,\mathrm{rec}}
=
u_{t_s+k}^{B,E}
+r\left(\frac{k+1}{N}\right)\Delta q^B.


其中 \(k=0,\ldots,N-1\)；完成 \(N\) 个计划恢复步后，受扰 Arm 也恢复为对应专家动作目标。

未受扰动角色继续使用对应专家动作：


u_{t_s+k}^{\bar B,\mathrm{rec}}
=u_{t_s+k}^{\bar B,E},



g_{t_s+k}^{L,\mathrm{rec}}=g_{t_s+k}^{L,E},
\qquad
g_{t_s+k}^{R,\mathrm{rec}}=g_{t_s+k}^{R,E},



u_{t_s+k}^{V,\mathrm{rec}}
=u_{t_s+k}^{V,E}.


## 9. 统一形式

令恢复角色 \(\rho\in\{V,L,R\}\)，并定义选择矩阵 \(P_\rho\in\{0,1\}^{6\times20}\)，用于从 20 维动作中选择角色 \(\rho\) 的六个关节。恢复参考起点为


t_r^\rho=
\begin{cases}
t_0, & \rho=V,\\
t_0+S_A, & \rho\in\{L,R\}.
\end{cases}


定义离散剩余扰动系数


\gamma_k^\rho=
\begin{cases}
r\!\left(\dfrac{k+1}{N_\rho}\right),
& 0\le k<N_\rho,\\
0, & k\ge N_\rho.
\end{cases}


其中 \(r(\cdot)\) 只在定义域 \([0,1]\) 内求值。View 和 Arm 恢复动作可以统一写为


a_{t_r^\rho+k}^{\mathrm{rec}}
=
a_{t_r^\rho+k}^{E}
+P_\rho^\top
\gamma_k^\rho
\Delta q^\rho.


由于监督对采用 pre-action 对齐，第一个保存的观测仍对应扰动后的物理状态，而与其配对的第一个动作标签已经使用 \(r(1/N_\rho)\Delta q^\rho\) 开始收缩；完整扰动 \(r(0)\Delta q^\rho\) 不作为恢复阶段的首个动作标签。

对任意未扰动角色 \(\rho'\neq\rho\)，有


P_{\rho'}a_{t_r^\rho+k}^{\mathrm{rec}}
=P_{\rho'}a_{t_r^\rho+k}^{E}.


该表达式体现了方法与双头策略之间的直接对应关系：恢复偏移只写入被扰动角色的输出子空间，未扰动头仍接受原专家监督。

由于 \(\gamma_k^\rho\) 非负且单调不增，解析动作偏移对任意 \(k\ge0\) 满足


\left\|
a_{t_r^\rho+k+1}^{\mathrm{rec}}-a_{t_r^\rho+k+1}^{E}
\right\|_2
\;=\;
\gamma_{k+1}^\rho\|\Delta q^\rho\|_2
\le
\gamma_k^\rho\|\Delta q^\rho\|_2
\;=\;
\left\|
a_{t_r^\rho+k}^{\mathrm{rec}}-a_{t_r^\rho+k}^{E}
\right\|_2.


本文所称“可收缩”严格指动作指令相对移动专家动作的解析偏移单调收缩。该动作由专家参考、预采样扰动和局部恢复进度预先确定，不使用在线状态误差 \(e_k^\rho\) 反馈修正，因而属于**开环的专家相对偏移调度**。在接触动力学和执行器滞后存在时，它不自动保证实际状态误差逐帧单调下降，后者由第 10 节的物理恢复判据单独验证。

## 10. 恢复达标与数据质量判据

### 10.1 移动专家误差

在恢复第 \(k\) 帧，定义被扰动角色的实际关节状态误差为


e_k^\rho
=
q_{t_r^\rho+k}^{\rho,\mathrm{actual}}
-q_{t_r^\rho+k}^{\rho,E}.


当


\|e_k^\rho\|_\infty\le\varepsilon,
\qquad \varepsilon=0.002\ \mathrm{rad},


并连续满足 \(K=3\) 帧时，才判定角色已稳定恢复。代码只在计划曲线归零后，即 \(k\ge N_\rho\) 时开始累计稳定帧。记首次完成连续稳定确认的局部索引为 \(k^\star\)，则要求


k^\star\le N_\rho+N_{\mathrm{extra}}.


计划恢复曲线归零后，最多允许额外执行


N_{\mathrm{extra}}=30


步零偏移专家动作，等待真实执行器进入误差管道。若仍未连续满足条件，则丢弃该次尝试并重新采样扰动。

### 10.2 恢复后稳定段

恢复达标后继续记录


N_{\mathrm{post}}=16


帧专家跟随数据。若恢复后的实际误差再次离开 \(\varepsilon\) 管道，则该分支作废。

按当前零起始局部索引，最终保存的恢复分支长度为


T_{\mathrm{saved}}=k^\star+N_{\mathrm{post}}+1.


### 10.3 未扰动角色约束

View 恢复分支要求左右 Arm 和夹爪实际状态保持在专家误差阈值内。Arm 恢复分支要求另一只 Arm、两个夹爪和 View 实际状态保持在各自阈值内。当前阈值为：

| 恢复分支 | 未扰动 Arm | Gripper | View |
|---|---:|---:|---:|
| View 恢复 | 0.002 rad | 0.005（归一化值） | 不适用 |
| Arm 恢复 | 0.010 rad | 0.010（归一化值） | 0.002 rad |

该检查用于排除扰动经接触动力学传播到未扰动角色的异常分支。恢复 rollout 会在每个记录帧执行上述检查；Arm 的未记录注入阶段则只在 10 个注入步全部完成后检查一次设置结果。

### 10.4 完整任务成功验证

训练文件只保存“恢复过程＋恢复后稳定段”。随后不再渲染或写盘，而是在后台继续执行剩余专家动作：


a_t=a_t^E,
\qquad t>t_{\mathrm{recorded\ end}}.


仅当完整专家后缀最终仍满足环境成功条件时，恢复分支才被保留。因此，数据中的 `success=true` 表示“物理恢复达标且专家后缀最终成功”，并不表示已经训练出的策略能够自主恢复。

### 10.5 观测—动作时间对齐

每个恢复帧均先读取实际机器人状态并渲染 `zed_cam_left`、`zed_cam_right` 双目图像，再执行该帧恢复动作。因此，保存的数据满足 pre-action 对齐：


(o_k,s_k)\longrightarrow a_k^{\mathrm{rec}}.


此外，恢复分支保存 `source_frame_index`、移动专家参考、解析恢复偏移和实际状态误差，用于检查增强帧与源专家轨迹的对应关系。

## 11. 数据生成算法

### Algorithm 1：双角色可收缩扰动恢复数据生成

**输入：** 成功专家轨迹集合 \(\mathcal D_E\)，角色 \(\rho\)，扰动分布参数 \((\boldsymbol\sigma,\mathbf b^{\mathrm{cfg}})\)，恢复参数 \((N_{\min},N_{\max},\varepsilon,K)\)。

**输出：** 原始专家轨迹与恢复分支组成的数据集 \(\mathcal D_R\)。

1. 对每条专家轨迹执行一次无扰动 MuJoCo 重放，验证状态误差和最终任务成功。
2. 在合法时间范围内随机采样 3～5 个恢复事件帧，并保存事件帧的完整环境快照。
3. 对每个事件：
   1. 从事件快照恢复环境；
   2. 根据关节/执行器限位计算扰动可行域；
   3. 从满足局部可行域和最小扰动强度的有界条件高斯分布采样 \(\Delta q^\rho\)；
   4. 若 \(\rho=V\)，通过不推进物理时间的 `qpos` 插值建立 View OOD 状态；
   5. 若 \(\rho\in\{L,R\}\)，通过真实物理步沿移动专家轨迹平滑注入 Arm 扰动；
   6. 根据扰动幅值和速度上限计算恢复步数 \(N\)；
   7. 按五次曲线生成并执行恢复动作，同时记录双目图像、状态和动作；
   8. 完成全部 \(N\) 个计划恢复步后，累计连续 \(K\) 帧满足恢复误差阈值，再记录 \(N_{\mathrm{post}}\) 帧；
   9. 在后台执行完整专家后缀并验证最终任务成功；
   10. 若任一条件失败，恢复快照并重新采样；重试耗尽后跳过该事件。
4. 在单个角色恢复数据集中，原始成功轨迹只保留一次，并与所有成功恢复分支共同组成输出数据集。

## 12. 当前关键参数

| 参数 | View | Arm |
|---|---:|---:|
| 每帧事件概率 | 0.015 | 0.015 |
| 每源轨迹事件数 | 3～5 | 3～5 |
| 最小事件间隔 | 50 帧 | 50 帧 |
| 扰动进入步数 | 10 个 `qpos` 插值循环点，末端再同步一次 | 10 个真实控制步 |
| 扰动进入时是否推进时间轴 | 否 | 是 |
| 恢复步数范围 | 20～40 | 20～40 |
| 恢复速度上限 | 配置固定值 | 专家动作速度 P90 |
| 恢复误差阈值 | 0.002 rad | 0.002 rad |
| 稳定确认 | 连续 3 帧 | 连续 3 帧 |
| 最大额外等待 | 30 帧 | 30 帧 |
| 恢复后记录 | 16 帧 | 16 帧 |
| 单事件最大重试 | 3 | 10 |
| 未扰动动作标签 | Arm/夹爪专家值 | 另一 Arm/夹爪/View 专家值 |
| 最终保留条件 | 完整后缀成功 | 完整后缀成功 |

## 13. 与代码的对应关系

| 方法组成 | 文件 | 函数或位置 |
|---|---|---|
| 五次平滑函数 \(s(u)\) | `data_collect/generate_contractive_view_recovery_trajectories.py` | `_quintic_smoothstep`，158 行附近 |
| 剩余比例 \(r(u)\) | 同上 | `_quintic_remaining_fraction`，172 行附近 |
| 自适应恢复步数 | 同上 | `_adaptive_recovery_steps`，182 行附近 |
| 随机事件帧采样 | 同上 | `_sample_injection_frames`，206 行附近 |
| View 可行扰动域 | 同上 | `_local_feasible_offset_bounds`，290 行附近 |
| View 有界高斯拒绝采样 | 同上 | `_sample_recovery_offset`，330 行附近 |
| View 动作写回 | 同上 | `_recovery_action`，382 行附近 |
| View 未记录扰动建立 | 同上 | `_apply_unrecorded_view_disturbance`，605 行附近 |
| View 恢复与成功检查 | 同上 | `_generate_recovery_branch`，891 行附近 |
| Arm 动作写回 | `data_collect/generate_contractive_arm_recovery_trajectories.py` | `_arm_recovery_action`，80 行附近 |
| 左右 Arm 均衡分配 | 同上 | `_assign_balanced_arms`，98 行附近 |
| Arm 速度 P90 | 同上 | `_arm_velocity_percentile`，116 行附近 |
| Arm 可行扰动域 | 同上 | `_local_arm_feasible_offset_bounds`，179 行附近 |
| Arm 有界高斯拒绝采样 | 同上 | `_sample_arm_recovery_offset`，221 行附近 |
| Arm 未记录物理注入 | 同上 | `_apply_unrecorded_arm_disturbance`，311 行附近 |
| Arm 恢复与成功检查 | 同上 | `_generate_arm_recovery_branch`，475 行附近 |
| View 配置 | `configs/data_collect/contractive_view_trajectory_recovery.yaml` | `event_sampling`、`view_joint_noise`、`recovery` |
| Arm 配置 | `configs/data_collect/contractive_arm_trajectory_recovery.yaml` | `event_sampling`、`arm_joint_noise`、`auto_velocity`、`recovery` |

## 14. 实现说明与论文表述边界

### 14.1 Arm 当前使用采样偏移作为恢复锚点

Arm 扰动通过真实动力学注入，因此注入结束后的实际偏移


\Delta q_{\mathrm{real}}^B
=q_{t_s}^{B,\mathrm{actual}}-q_{t_s}^{B,E}


可能与采样偏移 \(\Delta q^B\) 不完全一致。当前代码只检查


\|\Delta q_{\mathrm{real}}^B-\Delta q^B\|_\infty\le0.02\ \mathrm{rad},


但恢复动作仍使用采样偏移 \(\Delta q^B\)。因此部分分支在记录后的最初几帧仍可能继续增大实际误差。论文若描述当前实现，应明确写为“从采样扰动目标收缩”；若要严格声称“从实际 OOD 状态单调开始恢复”，应先改为使用 \(\Delta q_{\mathrm{real}}^B\) 作为恢复锚点，并增加未记录的自适应稳定阶段。

### 14.2 当前事件采样并非覆盖完整时间轴

为保证恢复、稳定确认和后缀验证有足够长度，事件起点必须位于 \([T_{\mathrm{initial}},T-H]\)。因此，当前 Arm 数据不会在最后 97 帧注入扰动。随机时间采样也会低估持续时间较短的任务阶段。论文当前可以表述为“在所有合法恢复时刻随机采样”，不应直接表述为“覆盖任务的全部阶段”，除非后续增加阶段均衡采样或末端专家动作延长机制。

### 14.3 恢复数据成功不等于策略恢复成功

恢复轨迹由专家参考和解析曲线构造，最终后缀也由专家动作执行。因此，数据生成成功率只衡量轨迹的物理可行性与标注质量。方法有效性必须通过训练后的策略在固定随机种子、受控 View/Arm 扰动和无扰动条件下分别评估。

### 14.4 最终样本分布包含成功筛选

被采样的扰动只是候选。View 事件最多重试 3 次，Arm 事件最多重试 10 次；只有满足注入跟踪、未扰动角色稳定、受扰角色恢复和最终任务成功的分支才会被保存。因此，论文应将最终数据描述为“由物理可行性和任务成功共同筛选的有界扰动恢复分布”，不应将最终样本直接等同于原始截断高斯样本。

### 14.5 View 与 Arm 数据合并需要去重

当前 View 与 Arm 生成器分别输出独立目录，两个目录都会各自保留一份完整原始专家轨迹。因此，构造联合训练集时不能直接拼接两个目录；需要只保留一份原始 episode，并为 View/Arm 恢复分支重新分配不冲突的增强编号。训练集和验证集还应按 `source_episode` 划分，避免同一源轨迹的原始分支与恢复分支跨集合泄漏。

### 14.6 推荐的论文消融

为了分别证明两个角色的贡献，至少应包含：

1. Expert only；
2. Expert + View recovery；
3. Expert + Arm recovery；
4. Expert + View recovery + Arm recovery；
5. 可选的 View–Arm 联合扰动鲁棒性测试。

除总体成功率外，应分别报告无扰动成功率、View 扰动恢复成功率、左/右 Arm 扰动恢复成功率，以及恢复时间、峰值关节误差和恢复后残差。这样才能区分“保持基线能力”和“提升 OOD 恢复能力”两类作用。
