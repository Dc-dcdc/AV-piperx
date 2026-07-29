"""重规划 DQN 的动作缓存、状态构造和 transition 收集工具。

该模块不负责加载 MuJoCo 环境或扩散策略。训练入口在每个物理环境步完成
观测预处理和策略推理后，使用这里的类显式管理16步动作缓存，并把得到的
``(state, decision, reward, next_state, done)`` 写入回放缓冲区。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .dqn import (
    CHUNK_PROGRESS,
    REMAINING_ACTION_MASK,
    REMAINING_ARM_ACTIONS,
    REMAINING_VIEW_ACTIONS,
    ROBOT_STATE,
    VISUAL_FEATURES,
    ReplanningDecision,
    ReplanningDQNConfig,
    ReplanningReplayBuffer,
    ReplanningTransitionBatch,
    build_replanning_action_mask,
    pad_remaining_action_chunk,
)


@dataclass(frozen=True)
class ReplanningRewardConfig:
    """配置环境奖励缩放、推理代价和 Arm 动作突变惩罚。"""

    env_reward_scale: float = 0.01               # 把当前环境约 [-100, 300] 的奖励缩放到稳定范围。
    view_only_replan_cost: float = 0.01          # 只重新规划 View 时扣除的推理代价。
    joint_replan_cost: float = 0.02              # 联合重新规划 Arm/View 时扣除的推理代价。
    arm_discontinuity_coef: float = 0.01         # 新旧 Arm 下一动作均方差的惩罚系数。

    def __post_init__(self) -> None:
        """拒绝会反向奖励重规划或动作突变的负代价配置。"""
        nonnegative_values = {
            "view_only_replan_cost": self.view_only_replan_cost,
            "joint_replan_cost": self.joint_replan_cost,
            "arm_discontinuity_coef": self.arm_discontinuity_coef,
        }
        for name, value in nonnegative_values.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} 必须非负，当前为 {value}")


class ReplanningActionCache:
    """显式保存归一化动作、环境动作和当前动作块执行位置。"""

    def __init__(self, config: ReplanningDQNConfig):
        self.config = config                              # 动作维数和最大 horizon 配置。
        self.normalized_actions: torch.Tensor | None = None  # DQN读取的归一化动作 [H, Da+Dv]。
        self.env_actions: torch.Tensor | None = None         # env.step执行的物理动作 [H, Da+Dv]。
        self.index = 0                                   # 下一条待执行动作在缓存中的下标。
        self.steps_since_replan = 0                      # 最近一次重新推理后已经执行的步数。

    @property
    def total_action_dim(self) -> int:
        """返回 Arm 和 View 拼接后的每步动作维数。"""
        return self.config.arm_action_dim + self.config.view_action_dim

    @property
    def has_remaining(self) -> bool:
        """返回缓存中是否仍有一条可执行的完整动作。"""
        return (
            self.normalized_actions is not None
            and self.env_actions is not None
            and self.index < self.normalized_actions.shape[0]
            and self.index < self.env_actions.shape[0]
        )

    @property
    def remaining_steps(self) -> int:
        """返回当前动作块中尚未执行的动作数量。"""
        if not self.has_remaining:
            return 0
        return int(self.normalized_actions.shape[0] - self.index)

    @property
    def progress(self) -> float:
        """返回当前动作块相对于最大 horizon 的已执行比例。"""
        if self.normalized_actions is None:
            return 0.0
        return min(1.0, float(self.index) / float(self.config.horizon))

    def clear(self) -> None:
        """清空动作缓存，下一次高层决策只能选择联合重规划。"""
        self.normalized_actions = None
        self.env_actions = None
        self.index = 0
        self.steps_since_replan = 0

    def _validate_joint_chunk(
        self,
        normalized_actions: torch.Tensor,
        env_actions: torch.Tensor,
    ) -> None:
        """验证归一化动作和环境动作具有一致的完整双头形状。"""
        expected_shape = (self.config.horizon, self.total_action_dim)
        if tuple(normalized_actions.shape) != expected_shape:
            raise ValueError(
                f"normalized_actions 形状应为 {expected_shape}，"
                f"当前为 {tuple(normalized_actions.shape)}"
            )
        if tuple(env_actions.shape) != expected_shape:
            raise ValueError(
                f"env_actions 形状应为 {expected_shape}，"
                f"当前为 {tuple(env_actions.shape)}"
            )
        if normalized_actions.device != env_actions.device:
            raise ValueError("归一化动作和环境动作必须位于同一个 device")
        if not torch.isfinite(normalized_actions).all():
            raise ValueError("normalized_actions 包含 NaN 或 Inf")
        if not torch.isfinite(env_actions).all():
            raise ValueError("env_actions 包含 NaN 或 Inf")

    def replace_joint(
        self,
        normalized_actions: torch.Tensor,
        env_actions: torch.Tensor,
    ) -> None:
        """使用新生成的完整16步双头动作替换旧缓存。"""
        self._validate_joint_chunk(normalized_actions, env_actions)
        self.normalized_actions = normalized_actions.detach().clone()
        self.env_actions = env_actions.detach().clone()
        self.index = 0
        self.steps_since_replan = 0

    def replace_remaining_view(
        self,
        normalized_view_actions: torch.Tensor,
        env_view_actions: torch.Tensor,
    ) -> None:
        """固定剩余 Arm 动作，只替换与其等长的 View 动作。"""
        if not self.has_remaining:
            raise RuntimeError("没有可供 View-only 重规划复用的 Arm 剩余动作")
        expected_shape = (self.remaining_steps, self.config.view_action_dim)
        if tuple(normalized_view_actions.shape) != expected_shape:
            raise ValueError(
                f"normalized_view_actions 形状应为 {expected_shape}，"
                f"当前为 {tuple(normalized_view_actions.shape)}"
            )
        if tuple(env_view_actions.shape) != expected_shape:
            raise ValueError(
                f"env_view_actions 形状应为 {expected_shape}，"
                f"当前为 {tuple(env_view_actions.shape)}"
            )
        if not torch.isfinite(normalized_view_actions).all():
            raise ValueError("normalized_view_actions 包含 NaN 或 Inf")
        if not torch.isfinite(env_view_actions).all():
            raise ValueError("env_view_actions 包含 NaN 或 Inf")

        view_start = self.config.arm_action_dim             # 拼接动作中 View 维度的起始下标。
        self.normalized_actions[
            self.index :,
            view_start:,
        ] = normalized_view_actions.to(self.normalized_actions)
        self.env_actions[
            self.index :,
            view_start:,
        ] = env_view_actions.to(self.env_actions)
        self.steps_since_replan = 0

    def peek_env_action(self) -> torch.Tensor:
        """读取下一条环境动作，但不推进缓存位置。"""
        if not self.has_remaining:
            raise RuntimeError("动作缓存为空，必须先进行联合重规划")
        return self.env_actions[self.index]

    def peek_normalized_arm_action(self) -> torch.Tensor:
        """读取下一条归一化 Arm 动作，用于计算重规划前后的动作跳变。"""
        if not self.has_remaining:
            raise RuntimeError("动作缓存为空，无法读取下一条 Arm 动作")
        return self.normalized_actions[
            self.index,
            : self.config.arm_action_dim,
        ]

    def advance(self) -> None:
        """在 env.step 成功后把缓存推进一个物理动作。"""
        if not self.has_remaining:
            raise RuntimeError("动作缓存为空，无法推进")
        self.index += 1
        self.steps_since_replan += 1

    def remaining_normalized_heads(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回当前尚未执行的归一化 Arm 和 View 动作。"""
        if not self.has_remaining:
            empty_arm = torch.empty(0, self.config.arm_action_dim)
            empty_view = torch.empty(0, self.config.view_action_dim)
            return empty_arm, empty_view
        remaining = self.normalized_actions[self.index :]
        return (
            remaining[:, : self.config.arm_action_dim],
            remaining[:, self.config.arm_action_dim :],
        )


def _ensure_single_feature(
    tensor: torch.Tensor,
    expected_dim: int,
    name: str,
) -> torch.Tensor:
    """把单条特征统一为 [1, D]，防止误把多环境批量写成一条转移。"""
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    expected_shape = (1, int(expected_dim))
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} 形状应为 {expected_shape}，当前为 {tuple(tensor.shape)}"
        )
    return tensor


def build_replanning_state(
    visual_features: torch.Tensor,
    robot_state: torch.Tensor,
    action_cache: ReplanningActionCache,
) -> dict[str, torch.Tensor]:
    """由当前特征和显式动作缓存构造一条批量大小为1的DQN状态。"""
    config = action_cache.config
    visual_features = _ensure_single_feature(
        visual_features,
        config.visual_feature_dim,
        VISUAL_FEATURES,
    )
    robot_state = _ensure_single_feature(
        robot_state,
        config.robot_state_dim,
        ROBOT_STATE,
    )
    if visual_features.device != robot_state.device:
        raise ValueError("视觉特征和机器人状态必须位于同一个 device")

    arm_remaining, view_remaining = action_cache.remaining_normalized_heads()
    arm_remaining = arm_remaining.to(visual_features.device)
    view_remaining = view_remaining.to(visual_features.device)
    padded_arm, padded_view, valid_mask = pad_remaining_action_chunk(
        arm_remaining,
        view_remaining,
        horizon=config.horizon,
    )
    progress = torch.tensor(
        [[action_cache.progress]],
        dtype=visual_features.dtype,
        device=visual_features.device,
    )
    return {
        VISUAL_FEATURES: visual_features.float(),
        ROBOT_STATE: robot_state.float(),
        REMAINING_ARM_ACTIONS: padded_arm.unsqueeze(0).float(),
        REMAINING_VIEW_ACTIONS: padded_view.unsqueeze(0).float(),
        REMAINING_ACTION_MASK: valid_mask.unsqueeze(0),
        CHUNK_PROGRESS: progress.float(),
    }


def compute_replanning_reward(
    env_reward: float,
    decision: int | ReplanningDecision,
    config: ReplanningRewardConfig,
    *,
    previous_arm_action: torch.Tensor | None = None,
    new_arm_action: torch.Tensor | None = None,
) -> tuple[float, dict[str, float]]:
    """合成环境奖励、重规划成本和联合重规划后的 Arm 动作突变惩罚。"""
    decision = ReplanningDecision(int(decision))
    scaled_env_reward = float(env_reward) * float(config.env_reward_scale)
    view_replan_cost = (
        float(config.view_only_replan_cost)
        if decision == ReplanningDecision.VIEW_ONLY_REPLAN
        else 0.0
    )
    joint_replan_cost = (
        float(config.joint_replan_cost)
        if decision == ReplanningDecision.JOINT_REPLAN
        else 0.0
    )
    arm_discontinuity = 0.0
    if previous_arm_action is not None or new_arm_action is not None:
        if previous_arm_action is None or new_arm_action is None:
            raise ValueError("计算 Arm 动作突变时必须同时提供新旧动作")
        if previous_arm_action.shape != new_arm_action.shape:
            raise ValueError(
                "新旧 Arm 动作形状必须一致，"
                f"当前为 {previous_arm_action.shape} 和 {new_arm_action.shape}"
            )
        arm_discontinuity = float(
            torch.mean(
                (new_arm_action.float() - previous_arm_action.float()).square()
            ).item()
        )
    discontinuity_penalty = (
        float(config.arm_discontinuity_coef) * arm_discontinuity
        if decision == ReplanningDecision.JOINT_REPLAN
        else 0.0
    )
    total_reward = (
        scaled_env_reward
        - view_replan_cost
        - joint_replan_cost
        - discontinuity_penalty
    )
    components = {
        "scaled_env_reward": scaled_env_reward,
        "view_replan_cost": view_replan_cost,
        "joint_replan_cost": joint_replan_cost,
        "arm_discontinuity": arm_discontinuity,
        "arm_discontinuity_penalty": discontinuity_penalty,
        "total_reward": total_reward,
    }
    return float(total_reward), components


class ReplanningDataCollector:
    """把单环境物理步整理成transition并写入DQN回放缓冲区。"""

    def __init__(
        self,
        replay_buffer: ReplanningReplayBuffer,
        reward_config: ReplanningRewardConfig,
    ):
        self.replay_buffer = replay_buffer                # DQN训练使用的循环回放缓冲区。
        self.reward_config = reward_config                # 元控制器奖励的缩放和代价参数。
        self.total_transitions = 0                        # 已写入的物理环境步总数。
        self.decision_counts = {decision: 0 for decision in ReplanningDecision}

    def add_step(
        self,
        *,
        state: Mapping[str, torch.Tensor],
        decision: int | ReplanningDecision,
        env_reward: float,
        next_state: Mapping[str, torch.Tensor],
        done: bool,
        next_action_mask: torch.Tensor,
        previous_arm_action: torch.Tensor | None = None,
        new_arm_action: torch.Tensor | None = None,
    ) -> tuple[float, dict[str, float]]:
        """计算元控制奖励，组装单步transition并立即写入回放缓冲区。"""
        decision = ReplanningDecision(int(decision))
        reward, reward_components = compute_replanning_reward(
            env_reward,
            decision,
            self.reward_config,
            previous_arm_action=previous_arm_action,
            new_arm_action=new_arm_action,
        )
        transition = ReplanningTransitionBatch(
            state={key: value.detach() for key, value in state.items()},
            action=torch.tensor([int(decision)], dtype=torch.long),
            reward=torch.tensor([reward], dtype=torch.float32),
            next_state={key: value.detach() for key, value in next_state.items()},
            done=torch.tensor([bool(done)], dtype=torch.bool),
            next_action_mask=next_action_mask.detach().bool(),
        )
        self.replay_buffer.add_batch(transition)
        self.total_transitions += 1
        self.decision_counts[decision] += 1
        return reward, reward_components

    def build_action_mask(
        self,
        action_cache: ReplanningActionCache,
        *,
        view_only_available: bool,
        force_joint_replan: bool = False,
        min_steps_after_replan: int = 0,
    ) -> torch.Tensor:
        """结合缓存、安全规则和重规划冷却时间生成当前合法决策。"""
        if int(min_steps_after_replan) < 0:
            raise ValueError("min_steps_after_replan 必须非负")
        cooldown_active = (
            action_cache.has_remaining
            and action_cache.steps_since_replan < int(min_steps_after_replan)
        )
        action_mask = build_replanning_action_mask(
            has_cached_plan=action_cache.has_remaining,
            view_only_available=view_only_available,
            force_joint_replan=force_joint_replan,
        )
        if cooldown_active and not force_joint_replan:
            action_mask[:, ReplanningDecision.VIEW_ONLY_REPLAN] = False
            action_mask[:, ReplanningDecision.JOINT_REPLAN] = False
        return action_mask

    def decision_fractions(self) -> dict[str, float]:
        """返回目前三种高层决策在已收集数据中的比例。"""
        denominator = max(1, self.total_transitions)
        return {
            "continue_fraction": (
                self.decision_counts[ReplanningDecision.CONTINUE] / denominator
            ),
            "view_only_replan_fraction": (
                self.decision_counts[ReplanningDecision.VIEW_ONLY_REPLAN]
                / denominator
            ),
            "joint_replan_fraction": (
                self.decision_counts[ReplanningDecision.JOINT_REPLAN]
                / denominator
            ),
        }
