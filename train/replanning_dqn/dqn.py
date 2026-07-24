"""主动视觉双头策略的三决策 Double DQN 核心模块。

该文件只实现独立的重规划元控制器，不会修改或调用现有 DPPO 训练流程。
元控制器在每个物理环境步接收当前观测特征、机器人状态和缓存中的剩余
Arm/View 动作，并在以下三个离散决策之间进行选择：

0. Arm 和 View 都继续执行；
1. 固定 Arm 剩余动作，只重新规划 View；
2. Arm 和 View 联合重新规划。

视觉输入有意采用上游策略提取好的特征，而不是在这里重复创建图像骨干网络。
第一版验证时可以输入随机特征；接入现有策略后，应传入冻结的视觉条件特征。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


VISUAL_FEATURES = "visual_features"                    # 当前真实画面的冻结视觉特征。
ROBOT_STATE = "robot_state"                            # 当前机器人低维状态。
REMAINING_ARM_ACTIONS = "remaining_arm_actions"        # 缓存中尚未执行的 Arm 动作。
REMAINING_VIEW_ACTIONS = "remaining_view_actions"      # 缓存中尚未执行的 View 动作。
REMAINING_ACTION_MASK = "remaining_action_mask"        # 标记剩余动作中的真实位置与补零位置。
CHUNK_PROGRESS = "chunk_progress"                      # 当前动作块已执行比例，范围为 [0, 1]。

REPLANNING_STATE_KEYS = (                               # 一个完整 DQN 状态必须包含的字段。
    VISUAL_FEATURES,                                    # 当前观测的视觉语义特征。
    ROBOT_STATE,                                        # 关节、末端和夹爪等状态特征。
    REMAINING_ARM_ACTIONS,                              # 固定 horizon 的 Arm 剩余动作张量。
    REMAINING_VIEW_ACTIONS,                             # 固定 horizon 的 View 剩余动作张量。
    REMAINING_ACTION_MASK,                              # 两个动作头共用的时间有效位。
    CHUNK_PROGRESS,                                     # 动作块内部的归一化时间位置。
)


class ReplanningDecision(IntEnum):
    """定义 DQN 可以选择的三种高层重规划决策。"""

    CONTINUE = 0                                        # 不推理，继续执行缓存中的 Arm/View 动作。
    VIEW_ONLY_REPLAN = 1                                # 固定 Arm，仅重新生成 View 剩余动作。
    JOINT_REPLAN = 2                                    # 丢弃旧缓存，联合生成新的 Arm/View 动作。


@dataclass(frozen=True)
class ReplanningDQNConfig:
    """保存重规划 DQN 的网络结构和优化参数。"""

    visual_feature_dim: int                             # 上游视觉编码器输出的特征维数。
    robot_state_dim: int                                # 输入 DQN 的机器人状态维数。
    arm_action_dim: int = 14                            # 每一步 Arm 动作的维数。
    view_action_dim: int = 6                            # 每一步 View 动作的维数。
    horizon: int = 16                                   # 双头策略一次预测的最大动作步数。
    visual_embed_dim: int = 256                         # DQN 内部视觉嵌入维数。
    state_embed_dim: int = 128                          # DQN 内部机器人状态嵌入维数。
    chunk_embed_dim: int = 128                          # 每个剩余动作块的编码维数。
    hidden_dim: int = 256                               # 融合主干网络的隐藏层维数。
    gamma: float = 0.99                                 # DQN 计算长期回报的折扣因子。
    learning_rate: float = 1.0e-4                       # 在线 Q 网络的学习率。
    weight_decay: float = 0.0                           # AdamW 的权重衰减系数。
    target_update_tau: float = 0.005                    # 目标网络 Polyak 软更新比例。
    grad_clip_norm: float = 10.0                        # 在线网络梯度范数裁剪上限。

    def __post_init__(self) -> None:
        """尽早拒绝会导致张量形状或优化行为错误的配置。"""
        positive_ints = {                               # 所有必须严格为正的整数维度配置。
            "visual_feature_dim": self.visual_feature_dim,
            "robot_state_dim": self.robot_state_dim,
            "arm_action_dim": self.arm_action_dim,
            "view_action_dim": self.view_action_dim,
            "horizon": self.horizon,
            "visual_embed_dim": self.visual_embed_dim,
            "state_embed_dim": self.state_embed_dim,
            "chunk_embed_dim": self.chunk_embed_dim,
            "hidden_dim": self.hidden_dim,
        }
        for name, value in positive_ints.items():        # 逐项报告具体的非法配置名称。
            if int(value) <= 0:
                raise ValueError(f"{name} 必须大于 0，当前为 {value}")
        if not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError(f"gamma 必须位于 [0, 1]，当前为 {self.gamma}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate 必须大于 0，当前为 {self.learning_rate}")
        if not 0.0 < float(self.target_update_tau) <= 1.0:
            raise ValueError(
                "target_update_tau 必须位于 (0, 1]，"
                f"当前为 {self.target_update_tau}"
            )
        if self.grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm 必须大于 0，当前为 {self.grad_clip_norm}"
            )


def _as_batch_bool(value, *, device: torch.device | str | None = None) -> torch.Tensor:
    """把标量或一维条件转换为批量布尔张量。"""
    tensor = torch.as_tensor(                            # 统一动作约束的类型和所在设备。
        value,
        dtype=torch.bool,
        device=device,
    )
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 1:
        raise ValueError(f"动作掩码条件必须是标量或一维张量，当前形状为 {tensor.shape}")
    return tensor


def _broadcast_batch_bool(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """把单个布尔条件广播到目标批量大小。"""
    if tensor.numel() == batch_size:
        return tensor
    if tensor.numel() == 1:
        return tensor.expand(batch_size)
    raise ValueError(
        f"动作掩码条件批量大小必须为 1 或 {batch_size}，当前为 {tensor.numel()}"
    )


def build_replanning_action_mask(
    has_cached_plan,
    view_only_available=True,
    force_joint_replan=False,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """根据运行状态生成三种决策的合法动作掩码。

    没有缓存动作或触发安全接管时只允许联合重规划；尚未实现 View 条件
    重规划时，可以通过 ``view_only_available=False`` 屏蔽动作 1。
    """
    cached = _as_batch_bool(has_cached_plan, device=device)              # 是否存在可继续执行的缓存计划。
    view_available = _as_batch_bool(view_only_available, device=device)  # View-only 推理是否已经实现并启用。
    force_joint = _as_batch_bool(force_joint_replan, device=device)      # 安全规则是否强制联合重规划。
    batch_size = max(                                                     # 从非标量条件推断统一批量大小。
        cached.numel(),
        view_available.numel(),
        force_joint.numel(),
    )
    cached = _broadcast_batch_bool(cached, batch_size)
    view_available = _broadcast_batch_bool(view_available, batch_size)
    force_joint = _broadcast_batch_bool(force_joint, batch_size)

    mask = torch.zeros(                                  # [B, 3]，True 表示该高层决策当前合法。
        batch_size,
        len(ReplanningDecision),
        dtype=torch.bool,
        device=cached.device,
    )
    normal_execution = cached & ~force_joint             # 允许复用旧计划的正常执行样本。
    mask[:, ReplanningDecision.CONTINUE] = normal_execution
    mask[:, ReplanningDecision.VIEW_ONLY_REPLAN] = (
        normal_execution & view_available
    )
    mask[:, ReplanningDecision.JOINT_REPLAN] = True
    return mask


def pad_remaining_action_chunk(
    arm_actions: torch.Tensor,
    view_actions: torch.Tensor,
    *,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把可变长度的剩余双头动作补零到固定 horizon，并生成有效位掩码。"""
    if arm_actions.ndim not in {2, 3} or view_actions.ndim != arm_actions.ndim:
        raise ValueError(
            "Arm/View 动作必须同时为 [L, D] 或 [B, L, D]，"
            f"当前形状为 {arm_actions.shape} 和 {view_actions.shape}"
        )
    added_batch_dim = arm_actions.ndim == 2               # 记录是否为单条无 batch 输入。
    if added_batch_dim:
        arm_actions = arm_actions.unsqueeze(0)
        view_actions = view_actions.unsqueeze(0)
    if arm_actions.shape[:2] != view_actions.shape[:2]:
        raise ValueError(
            "Arm/View 动作的批量和时间维必须一致，"
            f"当前为 {arm_actions.shape[:2]} 和 {view_actions.shape[:2]}"
        )
    if arm_actions.device != view_actions.device:
        raise ValueError("Arm/View 动作必须位于同一个 device")
    batch_size, remaining_steps = arm_actions.shape[:2]   # 剩余动作的批量数和真实时间长度。
    if remaining_steps > int(horizon):
        raise ValueError(
            f"剩余动作长度 {remaining_steps} 不能超过 horizon={horizon}"
        )

    padded_arm = arm_actions.new_zeros(                    # 补齐后的 Arm 动作 [B, H, Da]。
        batch_size,
        horizon,
        arm_actions.shape[-1],
    )
    padded_view = view_actions.new_zeros(                  # 补齐后的 View 动作 [B, H, Dv]。
        batch_size,
        horizon,
        view_actions.shape[-1],
    )
    padded_arm[:, :remaining_steps] = arm_actions
    padded_view[:, :remaining_steps] = view_actions
    valid_mask = torch.zeros(                              # True 表示该时间位置存在真实缓存动作。
        batch_size,
        horizon,
        dtype=torch.bool,
        device=arm_actions.device,
    )
    valid_mask[:, :remaining_steps] = True

    if added_batch_dim:
        return padded_arm[0], padded_view[0], valid_mask[0]
    return padded_arm, padded_view, valid_mask


def _validate_replanning_state(
    state: Mapping[str, torch.Tensor],
    config: ReplanningDQNConfig,
) -> int:
    """验证一个批量重规划状态并返回批量大小。"""
    missing = set(REPLANNING_STATE_KEYS) - set(state)      # 调用方尚未提供的必需状态字段。
    if missing:
        raise KeyError(f"重规划状态缺少字段: {sorted(missing)}")

    expected_shapes = {                                   # 各字段除 batch 维外的标准形状。
        VISUAL_FEATURES: (config.visual_feature_dim,),
        ROBOT_STATE: (config.robot_state_dim,),
        REMAINING_ARM_ACTIONS: (config.horizon, config.arm_action_dim),
        REMAINING_VIEW_ACTIONS: (config.horizon, config.view_action_dim),
        REMAINING_ACTION_MASK: (config.horizon,),
        CHUNK_PROGRESS: (1,),
    }
    batch_size = None                                     # 从第一个字段推断的统一批量大小。
    for key, trailing_shape in expected_shapes.items():
        tensor = state[key]                               # 当前待检查的状态张量。
        if not torch.is_tensor(tensor):
            raise TypeError(f"状态字段 {key} 必须是 torch.Tensor")
        if tensor.ndim != len(trailing_shape) + 1:
            raise ValueError(
                f"状态字段 {key} 期望形状 [B, {', '.join(map(str, trailing_shape))}]，"
                f"当前为 {tuple(tensor.shape)}"
            )
        if tuple(tensor.shape[1:]) != trailing_shape:
            raise ValueError(
                f"状态字段 {key} 尾部形状应为 {trailing_shape}，"
                f"当前为 {tuple(tensor.shape[1:])}"
            )
        if batch_size is None:
            batch_size = tensor.shape[0]
        elif tensor.shape[0] != batch_size:
            raise ValueError("所有重规划状态字段的批量大小必须一致")
    if batch_size is None or batch_size <= 0:
        raise ValueError("重规划状态批量不能为空")
    return int(batch_size)


class ActionChunkEncoder(nn.Module):
    """编码固定长度补零后的剩余动作，并保留动作的时间顺序。"""

    def __init__(self, horizon: int, action_dim: int, embed_dim: int):
        super().__init__()
        input_dim = int(horizon) * (int(action_dim) + 1)  # 动作值与逐步有效位展平后的总维数。
        self.encoder = nn.Sequential(                     # 保留固定时间顺序的动作块 MLP。
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        actions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """屏蔽无效补零位置后，把完整剩余序列编码为单个向量。"""
        mask = valid_mask.to(                              # 与动作同设备、同浮点类型的有效位。
            device=actions.device,
            dtype=actions.dtype,
        )
        masked_actions = actions * mask.unsqueeze(-1)     # 清除补零位置可能残留的任意数值。
        flattened = torch.cat(                            # 拼入有效位，使网络感知真实剩余长度。
            [masked_actions, mask.unsqueeze(-1)],
            dim=-1,
        ).flatten(start_dim=1)
        return self.encoder(flattened)


class ReplanningDuelingQNetwork(nn.Module):
    """融合观测、状态和剩余双头动作并输出三个决策的 Q 值。"""

    def __init__(self, config: ReplanningDQNConfig):
        super().__init__()
        self.config = config                              # 保存形状检查和训练共享的配置。
        self.visual_encoder = nn.Sequential(              # 把冻结视觉特征映射到 DQN 视觉空间。
            nn.Linear(config.visual_feature_dim, config.visual_embed_dim),
            nn.LayerNorm(config.visual_embed_dim),
            nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(               # 编码关节、末端和夹爪等低维状态。
            nn.Linear(config.robot_state_dim, config.state_embed_dim),
            nn.LayerNorm(config.state_embed_dim),
            nn.SiLU(),
        )
        self.arm_chunk_encoder = ActionChunkEncoder(      # 独立编码 Arm 剩余动作序列。
            config.horizon,
            config.arm_action_dim,
            config.chunk_embed_dim,
        )
        self.view_chunk_encoder = ActionChunkEncoder(     # 独立编码 View 剩余动作序列。
            config.horizon,
            config.view_action_dim,
            config.chunk_embed_dim,
        )

        fusion_dim = (                                    # 四类嵌入加进度和剩余比例的总维数。
            config.visual_embed_dim
            + config.state_embed_dim
            + 2 * config.chunk_embed_dim
            + 2
        )
        self.trunk = nn.Sequential(                       # 融合所有状态信息的共享特征主干。
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(config.hidden_dim, 1) # Dueling DQN 的状态价值 V(s)。
        self.advantage_head = nn.Linear(                  # 三种重规划决策的相对优势 A(s,a)。
            config.hidden_dim,
            len(ReplanningDecision),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """使用正交初始化，并让初始 Q 值保持在零附近。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.orthogonal_(self.advantage_head.weight, gain=0.01)

    def forward(self, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """返回形状为 [B, 3] 的继续、View重规划和联合重规划 Q 值。"""
        _validate_replanning_state(state, self.config)
        valid_mask = state[REMAINING_ACTION_MASK].bool()  # [B, H] 剩余动作有效位置。
        visual_feature = self.visual_encoder(             # [B, visual_embed_dim] 当前视觉嵌入。
            state[VISUAL_FEATURES].float()
        )
        robot_feature = self.state_encoder(               # [B, state_embed_dim] 机器人状态嵌入。
            state[ROBOT_STATE].float()
        )
        arm_feature = self.arm_chunk_encoder(             # [B, chunk_embed_dim] Arm 计划嵌入。
            state[REMAINING_ARM_ACTIONS].float(),
            valid_mask,
        )
        view_feature = self.view_chunk_encoder(           # [B, chunk_embed_dim] View 计划嵌入。
            state[REMAINING_VIEW_ACTIONS].float(),
            valid_mask,
        )
        progress = state[CHUNK_PROGRESS].float().clamp(0.0, 1.0)  # 已执行比例。
        remaining_fraction = valid_mask.float().mean(             # horizon 中仍有效的动作比例。
            dim=1,
            keepdim=True,
        )
        fused = torch.cat(                                # 拼接为 DQN 的完整马尔可夫状态表示。
            [
                visual_feature,
                robot_feature,
                arm_feature,
                view_feature,
                progress,
                remaining_fraction,
            ],
            dim=-1,
        )
        hidden = self.trunk(fused)                        # 三个 Q 头共享的融合隐藏特征。
        state_value = self.value_head(hidden)             # 当前状态与具体决策无关的公共价值。
        advantages = self.advantage_head(hidden)          # 三种决策相对于公共价值的优势。
        return state_value + advantages - advantages.mean(dim=-1, keepdim=True)

    @torch.no_grad()
    def select_action(
        self,
        state: Mapping[str, torch.Tensor],
        action_mask: torch.Tensor | None = None,
        *,
        epsilon: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """在合法动作内执行批量 epsilon-greedy 决策并同时返回原始 Q 值。"""
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon 必须位于 [0, 1]，当前为 {epsilon}")
        q_values = self(state)                            # 未施加动作约束的原始 [B, 3] Q 值。
        if action_mask is None:
            action_mask = torch.ones_like(q_values, dtype=torch.bool)
        else:
            action_mask = action_mask.to(device=q_values.device, dtype=torch.bool)
        if action_mask.shape != q_values.shape:
            raise ValueError(
                f"action_mask 形状应为 {q_values.shape}，当前为 {action_mask.shape}"
            )
        if (~action_mask).all(dim=1).any():
            raise ValueError("每个样本必须至少允许一种重规划决策")

        masked_q_values = q_values.masked_fill(           # 非法动作设为负无穷，禁止被选中。
            ~action_mask,
            -torch.inf,
        )
        greedy_actions = masked_q_values.argmax(dim=-1)   # 当前 Q 值最大的合法决策。
        if epsilon == 0.0:
            return greedy_actions, q_values

        random_scores = torch.rand_like(q_values).masked_fill(  # 为合法动作生成均匀随机排序。
            ~action_mask,
            -1.0,
        )
        random_actions = random_scores.argmax(dim=-1)     # epsilon 探索时采用的随机合法决策。
        explore = (                                       # 每个环境独立判断本次是否随机探索。
            torch.rand(q_values.shape[0], device=q_values.device) < epsilon
        )
        actions = torch.where(                            # 合并探索决策和贪心决策。
            explore,
            random_actions,
            greedy_actions,
        )
        return actions, q_values


@dataclass
class ReplanningTransitionBatch:
    """保存一次 Double DQN 更新所需的批量转移。"""

    state: dict[str, torch.Tensor]                         # 执行高层决策前的 DQN 状态。
    action: torch.Tensor                                  # 已执行决策编号，形状为 [B]。
    reward: torch.Tensor                                  # 执行决策和一个环境步后的即时奖励。
    next_state: dict[str, torch.Tensor]                    # 环境步结束后的下一 DQN 状态。
    done: torch.Tensor                                    # 当前转移是否结束 episode。
    next_action_mask: torch.Tensor                        # 下一状态允许选择的决策，形状为 [B, 3]。

    def to(self, device: torch.device | str) -> "ReplanningTransitionBatch":
        """返回移动到指定设备后的新批量，不修改原始回放数据。"""
        return ReplanningTransitionBatch(
            state={key: value.to(device) for key, value in self.state.items()},
            action=self.action.to(device),
            reward=self.reward.to(device),
            next_state={
                key: value.to(device) for key, value in self.next_state.items()
            },
            done=self.done.to(device),
            next_action_mask=self.next_action_mask.to(device),
        )


class ReplanningReplayBuffer:
    """使用预分配 CPU 张量保存固定形状的重规划转移。"""

    def __init__(self, capacity: int, config: ReplanningDQNConfig):
        if int(capacity) <= 0:
            raise ValueError(f"capacity 必须大于 0，当前为 {capacity}")
        self.capacity = int(capacity)                      # 回放缓冲区最多保存的转移数量。
        self.config = config                              # 各状态字段的固定张量尺寸配置。
        self.position = 0                                 # 下一批数据开始写入的循环下标。
        self.size = 0                                     # 当前已经写入的有效转移数。
        self.state_storage = self._allocate_state_storage()       # 当前状态张量存储。
        self.next_state_storage = self._allocate_state_storage()  # 下一状态张量存储。
        self.action_storage = torch.empty(                # 已执行的三类高层动作编号。
            self.capacity,
            dtype=torch.long,
        )
        self.reward_storage = torch.empty(                # 每条转移对应的即时标量奖励。
            self.capacity,
            dtype=torch.float32,
        )
        self.done_storage = torch.empty(                  # episode 是否在该转移后终止。
            self.capacity,
            dtype=torch.bool,
        )
        self.next_action_mask_storage = torch.empty(      # 下一状态的合法决策掩码。
            self.capacity,
            len(ReplanningDecision),
            dtype=torch.bool,
        )

    def _allocate_state_storage(self) -> dict[str, torch.Tensor]:
        """为所有状态字段分配固定容量的 CPU 存储。"""
        config = self.config                              # 缩短下面各字段尺寸定义的写法。
        return {
            VISUAL_FEATURES: torch.empty(
                self.capacity,
                config.visual_feature_dim,
            ),
            ROBOT_STATE: torch.empty(self.capacity, config.robot_state_dim),
            REMAINING_ARM_ACTIONS: torch.empty(
                self.capacity,
                config.horizon,
                config.arm_action_dim,
            ),
            REMAINING_VIEW_ACTIONS: torch.empty(
                self.capacity,
                config.horizon,
                config.view_action_dim,
            ),
            REMAINING_ACTION_MASK: torch.empty(
                self.capacity,
                config.horizon,
                dtype=torch.bool,
            ),
            CHUNK_PROGRESS: torch.empty(self.capacity, 1),
        }

    def __len__(self) -> int:
        """返回当前已经写入的有效转移数量。"""
        return self.size

    def add_batch(self, batch: ReplanningTransitionBatch) -> None:
        """把一个批量转移复制到循环回放缓冲区。"""
        batch_size = _validate_replanning_state(          # 当前状态的转移批量大小。
            batch.state,
            self.config,
        )
        next_batch_size = _validate_replanning_state(     # 下一状态的转移批量大小。
            batch.next_state,
            self.config,
        )
        if next_batch_size != batch_size:
            raise ValueError("state 和 next_state 的批量大小必须一致")
        expected_vector_shape = (batch_size,)             # 动作、奖励和终止位的标准形状。
        for name, tensor in {
            "action": batch.action,
            "reward": batch.reward,
            "done": batch.done,
        }.items():
            if tuple(tensor.shape) != expected_vector_shape:
                raise ValueError(
                    f"{name} 形状应为 {expected_vector_shape}，当前为 {tensor.shape}"
                )
        expected_mask_shape = (                           # 下一状态动作掩码的标准 [B, 3] 形状。
            batch_size,
            len(ReplanningDecision),
        )
        if tuple(batch.next_action_mask.shape) != expected_mask_shape:
            raise ValueError(
                "next_action_mask 形状应为 "
                f"{expected_mask_shape}，当前为 {batch.next_action_mask.shape}"
            )
        if batch_size > self.capacity:
            start = batch_size - self.capacity            # 丢弃超容量批量中最早的转移。
            batch = ReplanningTransitionBatch(
                state={key: value[start:] for key, value in batch.state.items()},
                action=batch.action[start:],
                reward=batch.reward[start:],
                next_state={
                    key: value[start:] for key, value in batch.next_state.items()
                },
                done=batch.done[start:],
                next_action_mask=batch.next_action_mask[start:],
            )
            batch_size = self.capacity

        indices = (                                       # 当前批量在循环存储中的实际写入下标。
            torch.arange(batch_size, dtype=torch.long) + self.position
        ) % self.capacity
        for key in REPLANNING_STATE_KEYS:
            self.state_storage[key][indices] = batch.state[key].detach().cpu()
            self.next_state_storage[key][indices] = (
                batch.next_state[key].detach().cpu()
            )
        self.action_storage[indices] = batch.action.detach().cpu().long()
        self.reward_storage[indices] = batch.reward.detach().cpu().float()
        self.done_storage[indices] = batch.done.detach().cpu().bool()
        self.next_action_mask_storage[indices] = (
            batch.next_action_mask.detach().cpu().bool()
        )
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
    ) -> ReplanningTransitionBatch:
        """从有效回放范围内均匀采样一个训练批量。"""
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size 必须大于 0，当前为 {batch_size}")
        if self.size < int(batch_size):
            raise ValueError(
                f"回放缓冲区只有 {self.size} 条数据，无法采样 {batch_size} 条"
            )
        indices = torch.randint(                          # 在当前有效范围内均匀随机抽样下标。
            0,
            self.size,
            (int(batch_size),),
        )
        batch = ReplanningTransitionBatch(                # 依据抽样下标组装批量转移。
            state={key: value[indices] for key, value in self.state_storage.items()},
            action=self.action_storage[indices],
            reward=self.reward_storage[indices],
            next_state={
                key: value[indices]
                for key, value in self.next_state_storage.items()
            },
            done=self.done_storage[indices],
            next_action_mask=self.next_action_mask_storage[indices],
        )
        return batch.to(device)


class DoubleDQNTrainer:
    """维护在线网络、目标网络并执行标准 Double DQN 更新。"""

    def __init__(
        self,
        online_network: ReplanningDuelingQNetwork,
        config: ReplanningDQNConfig,
    ):
        self.config = config                              # TD目标和优化器共享的超参数。
        self.online_network = online_network              # 每一步参与梯度更新并负责选动作的 Q 网络。
        self.target_network = copy.deepcopy(              # 计算稳定 TD 目标的慢速目标网络。
            online_network
        ).eval()
        self.target_network.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(               # 只更新在线 Q 网络参数。
            self.online_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def hard_update_target(self) -> None:
        """把在线网络完整复制到目标网络。"""
        self.target_network.load_state_dict(self.online_network.state_dict())

    @torch.no_grad()
    def soft_update_target(self) -> None:
        """使用 Polyak 平均缓慢更新目标网络。"""
        tau = float(self.config.target_update_tau)         # 每次吸收在线网络参数的比例。
        for target_parameter, online_parameter in zip(
            self.target_network.parameters(),
            self.online_network.parameters(),
        ):
            target_parameter.lerp_(online_parameter, tau)

    def train_step(
        self,
        batch: ReplanningTransitionBatch,
    ) -> dict[str, float]:
        """执行一次 Double DQN TD 更新并返回训练诊断数据。"""
        batch_size = _validate_replanning_state(          # 本次参与 TD 更新的转移数量。
            batch.state,
            self.config,
        )
        _validate_replanning_state(batch.next_state, self.config)
        action = batch.action.long().reshape(-1)          # [B]，实际执行的高层决策编号。
        reward = batch.reward.float().reshape(-1)         # [B]，环境和重规划代价合成的奖励。
        done = batch.done.float().reshape(-1)             # [B]，终止样本为 1，禁止价值自举。
        if action.numel() != batch_size:
            raise ValueError("action 数量必须与状态批量大小一致")
        if reward.numel() != batch_size or done.numel() != batch_size:
            raise ValueError("reward 和 done 数量必须与状态批量大小一致")
        if ((action < 0) | (action >= len(ReplanningDecision))).any():
            raise ValueError("action 中包含非法重规划决策编号")

        next_action_mask = batch.next_action_mask.to(     # 下一状态三种决策的合法性约束。
            dtype=torch.bool
        )
        expected_mask_shape = (                           # Double DQN 下一动作掩码的期望形状。
            batch_size,
            len(ReplanningDecision),
        )
        if tuple(next_action_mask.shape) != expected_mask_shape:
            raise ValueError(
                "next_action_mask 形状应为 "
                f"{expected_mask_shape}，当前为 {next_action_mask.shape}"
            )
        if (~next_action_mask).all(dim=1).any():
            raise ValueError("next_action_mask 每行必须至少允许一种决策")

        self.online_network.train()
        q_values = self.online_network(batch.state)       # 在线网络输出的所有当前 Q(s,a)。
        selected_q = q_values.gather(                      # 只取采集时实际执行动作的 Q 值。
            1,
            action.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_online_q = self.online_network(          # 在线网络负责选择下一状态中的最佳动作。
                batch.next_state
            )
            next_online_q = next_online_q.masked_fill(
                ~next_action_mask,
                -torch.inf,
            )
            next_action = next_online_q.argmax(           # Double DQN 选择的下一合法决策。
                dim=1,
                keepdim=True,
            )
            next_target_q = self.target_network(          # 目标网络评估上面选中的下一决策。
                batch.next_state
            )
            next_q = next_target_q.gather(                 # 下一状态最佳合法决策的目标 Q 值。
                1,
                next_action,
            ).squeeze(1)
            td_target = (                                  # r + gamma * (1-done) * Q_target。
                reward
                + self.config.gamma * (1.0 - done) * next_q
            )

        loss = F.smooth_l1_loss(                           # Huber 损失降低异常 TD 误差的影响。
            selected_q,
            td_target,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(        # 裁剪前的在线网络总梯度范数。
            self.online_network.parameters(),
            self.config.grad_clip_norm,
        )
        self.optimizer.step()
        self.soft_update_target()

        with torch.no_grad():
            td_error = td_target - selected_q              # 用于监测和优先回放的 TD 误差。
        return {
            "loss": float(loss.detach()),
            "q_mean": float(selected_q.detach().mean()),
            "target_mean": float(td_target.detach().mean()),
            "td_abs_mean": float(td_error.detach().abs().mean()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
        }


def _make_random_state(
    config: ReplanningDQNConfig,
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """仅为直接运行本文件时的冒烟验证生成随机批量状态。"""
    return {
        VISUAL_FEATURES: torch.randn(
            batch_size,
            config.visual_feature_dim,
            device=device,
        ),
        ROBOT_STATE: torch.randn(
            batch_size,
            config.robot_state_dim,
            device=device,
        ),
        REMAINING_ARM_ACTIONS: torch.randn(
            batch_size,
            config.horizon,
            config.arm_action_dim,
            device=device,
        ),
        REMAINING_VIEW_ACTIONS: torch.randn(
            batch_size,
            config.horizon,
            config.view_action_dim,
            device=device,
        ),
        REMAINING_ACTION_MASK: torch.ones(
            batch_size,
            config.horizon,
            dtype=torch.bool,
            device=device,
        ),
        CHUNK_PROGRESS: torch.rand(batch_size, 1, device=device),
    }


def run_smoke_test() -> dict[str, float]:
    """构造随机转移并验证三动作选择、回放采样和一次 DQN 更新。"""
    torch.manual_seed(0)
    config = ReplanningDQNConfig(                          # 使用较小视觉维数降低冒烟测试开销。
        visual_feature_dim=64,
        robot_state_dim=20,
    )
    network = ReplanningDuelingQNetwork(config)            # 待训练的在线 Dueling Q 网络。
    trainer = DoubleDQNTrainer(network, config)            # 在线/目标网络和优化器管理器。
    state = _make_random_state(config, batch_size=8)       # 随机生成的当前批量状态。
    next_state = _make_random_state(config, batch_size=8)  # 随机生成的下一批量状态。
    action_mask = build_replanning_action_mask(            # 冒烟测试中三种决策全部开放。
        has_cached_plan=torch.ones(8, dtype=torch.bool),
        view_only_available=True,
    )
    action, q_values = network.select_action(              # epsilon-greedy 生成模拟采集动作。
        state,
        action_mask,
        epsilon=0.1,
    )
    batch = ReplanningTransitionBatch(                     # 组装一批随机环境转移。
        state=state,
        action=action,
        reward=torch.randn(8),
        next_state=next_state,
        done=torch.zeros(8, dtype=torch.bool),
        next_action_mask=action_mask,
    )
    replay_buffer = ReplanningReplayBuffer(                # 保存并重新采样随机转移。
        capacity=32,
        config=config,
    )
    replay_buffer.add_batch(batch)
    metrics = trainer.train_step(                          # 执行一次完整 Double DQN 参数更新。
        replay_buffer.sample(8)
    )
    if q_values.shape != (8, len(ReplanningDecision)):
        raise AssertionError(f"DQN 输出形状错误: {q_values.shape}")
    return metrics


if __name__ == "__main__":
    smoke_metrics = run_smoke_test()                       # 直接运行文件时显示最小验证结果。
    print("三决策 Double DQN 冒烟验证通过:", smoke_metrics)
