"""View 动作表示的纯张量变换与归一化统计适配。

数据集始终保存环境使用的绝对关节动作。单头和双头 Diffusion 策略均可在
模型边界把 View 动作转换为相对本次重规划当前状态的位移，并在输出动作
入队前恢复为绝对值。
该模块不注册任何参数或 buffer，因此不会改变旧 ``absolute`` checkpoint
的 state_dict 结构。
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig


VIEW_ACTION_ABSOLUTE = "absolute"
VIEW_ACTION_DELTA_FROM_CURRENT = "delta_from_current"
VIEW_ACTION_DELTA_STATS_KEY = "view_action_delta"


def resolve_dual_head_action_dims(config: DiffusionConfig) -> tuple[int, int]:
    """解析完整动作中的Arm/View切片；单头和双头策略共用相同规则。"""
    action_dim = int(config.output_shapes["action"][0])
    arm_action_dim = getattr(config, "arm_action_dim", None)
    view_action_dim = getattr(config, "view_action_dim", None)
    if arm_action_dim is None and view_action_dim is None:
        arm_action_dim = 14 if action_dim == 20 else action_dim // 2
        view_action_dim = action_dim - arm_action_dim
    elif arm_action_dim is None:
        arm_action_dim = action_dim - int(view_action_dim)
    elif view_action_dim is None:
        view_action_dim = action_dim - int(arm_action_dim)

    arm_action_dim = int(arm_action_dim)
    view_action_dim = int(view_action_dim)
    if arm_action_dim <= 0 or view_action_dim <= 0:
        raise ValueError(
            "Arm/View动作维度必须为正数，"
            f"当前arm={arm_action_dim}, view={view_action_dim}。"
        )
    if arm_action_dim + view_action_dim != action_dim:
        raise ValueError(
            "Arm/View动作维度之和必须等于完整动作维度，"
            f"当前arm={arm_action_dim}, view={view_action_dim}, action={action_dim}。"
        )
    return arm_action_dim, view_action_dim


def prepare_output_dataset_stats(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, Tensor]] | None,
) -> dict[str, dict[str, Tensor]] | None:
    """为混合 ``[Arm绝对动作, View相对位移]`` 构造输出统计量。

    ``absolute`` 模式原样返回输入对象，确保旧路径不发生复制、数值或
    state_dict 变化。新模式仅替换 action 统计量的 View 切片；输入状态
    的统计量仍保持绝对关节语义。
    """
    representation = str(
        getattr(config, "view_action_representation", VIEW_ACTION_ABSOLUTE)
    )
    if representation == VIEW_ACTION_ABSOLUTE or dataset_stats is None:
        return dataset_stats
    if representation != VIEW_ACTION_DELTA_FROM_CURRENT:
        raise ValueError(f"未知View动作表示: {representation!r}。")

    arm_action_dim, view_action_dim = resolve_dual_head_action_dims(config)
    if "action" not in dataset_stats:
        raise KeyError("delta_from_current模式需要dataset_stats['action']。")
    if VIEW_ACTION_DELTA_STATS_KEY not in dataset_stats:
        raise KeyError(
            "delta_from_current模式缺少View增量统计量"
            f"dataset_stats[{VIEW_ACTION_DELTA_STATS_KEY!r}]。"
            "请通过s1预训练入口自动计算，或在构造策略前显式提供。"
        )

    mixed_stats = copy.deepcopy(dataset_stats)
    action_stats = mixed_stats["action"]
    delta_stats = mixed_stats[VIEW_ACTION_DELTA_STATS_KEY]
    action_dim = arm_action_dim + view_action_dim
    for stat_name, action_value in tuple(action_stats.items()):
        if stat_name not in delta_stats:
            raise KeyError(
                f"View增量统计量缺少{stat_name!r}，"
                f"当前字段为{sorted(delta_stats)}。"
            )
        action_tensor = torch.as_tensor(action_value)
        delta_tensor = torch.as_tensor(
            delta_stats[stat_name],
            device=action_tensor.device,
            dtype=action_tensor.dtype,
        )
        if tuple(action_tensor.shape) != (action_dim,):
            raise ValueError(
                f"action/{stat_name}形状应为({action_dim},)，"
                f"当前为{tuple(action_tensor.shape)}。"
            )
        if tuple(delta_tensor.shape) != (view_action_dim,):
            raise ValueError(
                f"{VIEW_ACTION_DELTA_STATS_KEY}/{stat_name}形状应为"
                f"({view_action_dim},)，当前为{tuple(delta_tensor.shape)}。"
            )
        if not torch.isfinite(delta_tensor).all():
            raise ValueError(
                f"{VIEW_ACTION_DELTA_STATS_KEY}/{stat_name}包含非有限值。"
            )
        patched = action_tensor.clone()
        patched[arm_action_dim:] = delta_tensor
        action_stats[stat_name] = patched

    return mixed_stats


def extract_current_view_anchor(
    observation_state: Tensor,
    *,
    arm_action_dim: int,
    view_action_dim: int,
) -> Tensor:
    """从未归一化状态中读取当前 View 关节，返回 ``[B, view_dim]``。"""
    view_end = arm_action_dim + view_action_dim
    if observation_state.ndim == 2:
        current_state = observation_state
    elif observation_state.ndim == 3:
        current_state = observation_state[:, -1]
    else:
        raise ValueError(
            "observation.state应为[B, state_dim]或[B, n_obs_steps, state_dim]，"
            f"当前形状为{tuple(observation_state.shape)}。"
        )
    if current_state.shape[-1] < view_end:
        raise ValueError(
            "delta_from_current模式要求observation.state包含与动作对应的View关节，"
            f"至少需要{view_end}维，当前为{current_state.shape[-1]}维。"
        )
    return current_state[..., arm_action_dim:view_end]


def _broadcast_anchor(anchor: Tensor, actions: Tensor, view_action_dim: int) -> Tensor:
    if anchor.shape[-1] != view_action_dim:
        raise ValueError(
            f"View锚点最后一维应为{view_action_dim}，当前为{anchor.shape[-1]}。"
        )
    while anchor.ndim < actions.ndim:
        anchor = anchor.unsqueeze(-2)
    try:
        torch.broadcast_shapes(anchor.shape, actions[..., -view_action_dim:].shape)
    except RuntimeError as exc:
        raise ValueError(
            "View锚点无法广播到动作时间轴，"
            f"anchor={tuple(anchor.shape)}, actions={tuple(actions.shape)}。"
        ) from exc
    return anchor


def encode_actions_delta_from_current(
    absolute_actions: Tensor,
    view_anchor: Tensor,
    *,
    arm_action_dim: int,
    view_action_dim: int,
) -> Tensor:
    """将绝对动作转换为 ``[Arm绝对值, View相对当前锚点位移]``。"""
    action_dim = arm_action_dim + view_action_dim
    if absolute_actions.shape[-1] != action_dim:
        raise ValueError(
            f"完整动作最后一维应为{action_dim}，当前为{absolute_actions.shape[-1]}。"
        )
    anchor = _broadcast_anchor(view_anchor, absolute_actions, view_action_dim)
    view_delta = absolute_actions[..., arm_action_dim:] - anchor
    return torch.cat(
        (absolute_actions[..., :arm_action_dim], view_delta),
        dim=-1,
    )


def decode_actions_delta_from_current(
    model_actions: Tensor,
    view_anchor: Tensor,
    *,
    arm_action_dim: int,
    view_action_dim: int,
) -> Tensor:
    """把模型的 View 位移恢复为环境执行所需的绝对关节动作。"""
    action_dim = arm_action_dim + view_action_dim
    if model_actions.shape[-1] != action_dim:
        raise ValueError(
            f"完整动作最后一维应为{action_dim}，当前为{model_actions.shape[-1]}。"
        )
    anchor = _broadcast_anchor(view_anchor, model_actions, view_action_dim)
    absolute_view = model_actions[..., arm_action_dim:] + anchor
    return torch.cat(
        (model_actions[..., :arm_action_dim], absolute_view),
        dim=-1,
    )
