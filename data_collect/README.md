# Data collection code layout

`data_collect` 的可执行入口按数据来源分为两类，底层公共模块保留在本目录。

## Expert data collection

目录：`data_collect/expert_data_collection/`

- `quest_teleop_collect.py`：Quest 专家遥操作数据采集。
- `quest_policy_collect.py`：策略运行与人在环接管数据采集。
- `collect_data_from_model.py`：训练策略自主 rollout 数据采集。
- `quest_receive.py`、`quest_send.py`：Quest/Unity UDP 通信。
- `quest_control.py`、`robot_ik_solver.py`：Quest 动作映射与关节 IK。
- `headset_utils.py`、`quest_pose_filter.py`：头显数据结构与姿态滤波。
- `episode_seeding.py`、`collection_run_state.py`：可复现随机种子与运行目录锁。
- `quest_mujoco_test.py`、`quest_pose_mapping_test.py`：Quest 映射和 IK 诊断入口。

## Recovery data generation

目录：`data_collect/recovery_data_generation/`

- `arm_recovery_trajectories.py`：Arm 扰动恢复数据。
- `view_recovery_trajectories.py`：View 扰动恢复数据。
- `mixed_recovery_trajectories.py`：Arm/View 同时扰动恢复数据。
- `score_expert_policy_risk.py`：专家模型风险扫描与锚点生成。
- `visualize_recovery_dataset.py`：恢复数据分布可视化与统计。
- `backfill_insert_container_state.py`：旧 InsertCylinder 数据兼容回填。
- `trajectory_replay_common.py`：恢复生成器共用的轨迹校验与MuJoCo重放工具。

## Shared modules

`transform_utils.py` 继续位于 `data_collect/` 根目录，因为环境侧的 DiffIK、GradIK
和运动学模块也依赖该坐标变换工具。其余仅服务于 Quest 专家采集的共享模块均已归入
`data_collect/expert_data_collection/`。
