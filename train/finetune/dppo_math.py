"""DPPO 扩散转移与联合概率的共享数学函数。"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def optional_float(value):
    """把可选配置值转换为 float，并将常见空值字符串视为 None。"""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def resolve_action_slice(config, *, horizon: int | None = None) -> tuple[int, int]:
    """按原版Diffusion Policy语义，从最后一帧历史观测对应的动作开始执行。"""
    n_obs_steps = int(getattr(config, "n_obs_steps", 1))
    action_steps = int(getattr(config, "n_action_steps", 8))
    if n_obs_steps <= 0:
        raise ValueError(f"n_obs_steps 必须为正，当前为 {n_obs_steps}")
    if action_steps <= 0:
        raise ValueError(f"n_action_steps 必须为正，当前为 {action_steps}")

    action_start = n_obs_steps - 1
    action_end = action_start + action_steps
    if horizon is not None and action_end > int(horizon):
        raise ValueError(
            f"动作切片越界: action_slice=[{action_start}:{action_end}], horizon={horizon}"
        )
    return action_start, action_end


def dppo_ddim_mean_std(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timesteps,
    scheduler,
    *,
    eta: float,
    min_std: float,
    denoised_clip_value: float | None = None,
    eps_clip_value: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算原版 DPPO 风格 DDIM 转移的均值和标准差。"""
    if scheduler.num_inference_steps is None:
        raise RuntimeError("调用 dppo_ddim_mean_std 前必须先设置 scheduler timesteps")
    if min_std < 0:
        raise ValueError(f"min_std 必须非负，当前为 {min_std}")
    if eta < 0:
        raise ValueError(f"eta 必须非负，当前为 {eta}")

    if torch.is_tensor(timesteps):
        timesteps = timesteps.to(device=sample.device, dtype=torch.long)
    else:
        timesteps = torch.as_tensor(timesteps, device=sample.device, dtype=torch.long)

    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=sample.device,
        dtype=sample.dtype,
    )
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1)

    step_ratio = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    if step_ratio <= 0:
        raise ValueError(
            "scheduler 的训练步数必须不少于推理步数，"
            f"当前为 {scheduler.config.num_train_timesteps}/"
            f"{scheduler.num_inference_steps}"
        )
    prev_timesteps = timesteps - step_ratio
    alpha_prod_t_prev = torch.where(
        prev_timesteps >= 0,
        alphas_cumprod[torch.clamp(prev_timesteps, min=0)],
        torch.ones_like(prev_timesteps, device=sample.device, dtype=sample.dtype),
    ).view(-1, 1, 1)

    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t.clamp(min=1e-12))
    sqrt_one_minus_alpha_prod_t = torch.sqrt(
        (1 - alpha_prod_t).clamp(min=1e-12)
    )
    pred_original_sample = (
        sample - sqrt_one_minus_alpha_prod_t * model_output
    ) / sqrt_alpha_prod_t

    if denoised_clip_value is not None:
        pred_original_sample = pred_original_sample.clamp(
            -denoised_clip_value,
            denoised_clip_value,
        )
        model_output = (
            sample - sqrt_alpha_prod_t * pred_original_sample
        ) / sqrt_one_minus_alpha_prod_t

    if eps_clip_value is not None:
        model_output = model_output.clamp(-eps_clip_value, eps_clip_value)

    sigma = float(eta) * torch.sqrt(
        (
            ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t).clamp(min=1e-12))
            * (1 - alpha_prod_t / alpha_prod_t_prev.clamp(min=1e-12))
        ).clamp(min=0)
    )
    sigma = sigma.clamp(min=1e-10)

    dir_xt_coef = torch.sqrt(
        (1 - alpha_prod_t_prev - sigma**2).clamp(min=0)
    )
    mean = (
        torch.sqrt(alpha_prod_t_prev.clamp(min=0)) * pred_original_sample
        + dir_xt_coef * model_output
    )
    std = torch.clamp(sigma, min=float(min_std))
    return mean, std


def combine_head_logprobs(
    head_logprobs: Mapping[str, torch.Tensor],
    head_action_dims: Mapping[str, int],
) -> torch.Tensor:
    """按动作维数合并各头 mean logprob，等价于拼接后统一求 mean。"""
    missing = set(head_action_dims) - set(head_logprobs)
    if missing:
        raise KeyError(f"联合 logprob 缺少动作头: {sorted(missing)}")

    total_dim = sum(int(dim) for dim in head_action_dims.values())
    if total_dim <= 0:
        raise ValueError(f"动作总维数必须大于 0，当前为 {total_dim}")

    joint_logprob = None
    for head, dim in head_action_dims.items():
        dim = int(dim)
        if dim <= 0:
            raise ValueError(f"{head} 动作维数必须大于 0，当前为 {dim}")
        weighted = head_logprobs[head] * dim
        joint_logprob = weighted if joint_logprob is None else joint_logprob + weighted
    return joint_logprob / total_dim


def clipped_ppo_loss(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """对一个联合 logprob 计算一次 PPO ratio、裁剪损失和 k3 KL。"""
    log_ratio = new_logprob - old_logprob
    ratio = torch.exp(log_ratio)
    surrogate = ratio * advantages
    clipped_surrogate = torch.clamp(
        ratio,
        1.0 - clip_coef,
        1.0 + clip_coef,
    ) * advantages
    policy_loss = -torch.min(surrogate, clipped_surrogate).mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
    return policy_loss, log_ratio, ratio, approx_kl


def init_ppo_ratio_stats(denoising_steps: int) -> dict:
    """初始化 PPO ratio 与动态裁剪比例的在线累计量。"""
    denoising_steps = int(denoising_steps)
    if denoising_steps <= 0:
        raise ValueError(f"denoising_steps 必须大于 0，当前为 {denoising_steps}")
    return {
        "denoising_steps": denoising_steps,
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "outside_clip": 0,
        "objective_clipped": 0,
        "upper_clip": 0,
        "lower_clip": 0,
        "per_step_count": [0] * denoising_steps,
        "per_step_outside_clip": [0] * denoising_steps,
        "per_step_objective_clipped": [0] * denoising_steps,
    }


def _flatten_valid_ppo_ratio_inputs(
    ratio: torch.Tensor,
    clip_coef: torch.Tensor,
    advantages: torch.Tensor,
    denoising_indices: torch.Tensor,
    denoising_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """展平并过滤 ratio 诊断所需张量中的非有限或非法样本。"""
    ratio = ratio.detach().float().reshape(-1)
    clip_coef = clip_coef.detach().float().reshape(-1)
    advantages = advantages.detach().float().reshape(-1)
    denoising_indices = denoising_indices.detach().long().reshape(-1)
    sizes = {
        ratio.numel(),
        clip_coef.numel(),
        advantages.numel(),
        denoising_indices.numel(),
    }
    if len(sizes) != 1:
        raise ValueError(
            "ratio、clip_coef、advantages 和 denoising_indices 的元素数必须一致"
        )

    finite_mask = (
        torch.isfinite(ratio)
        & torch.isfinite(clip_coef)
        & torch.isfinite(advantages)
    )
    valid_step_mask = (
        (denoising_indices >= 0) & (denoising_indices < int(denoising_steps))
    )
    valid_mask = finite_mask & valid_step_mask
    return (
        ratio[valid_mask],
        clip_coef[valid_mask],
        advantages[valid_mask],
        denoising_indices[valid_mask],
    )


@torch.no_grad()
def update_ppo_ratio_stats(
    stats: dict,
    ratio: torch.Tensor,
    clip_coef: torch.Tensor,
    advantages: torch.Tensor,
    denoising_indices: torch.Tensor,
) -> None:
    """在线累计 ratio 分布、越界 clipfrac 和实际代理目标裁剪率。"""
    denoising_steps = int(stats["denoising_steps"])
    ratio, clip_coef, advantages, denoising_indices = (
        _flatten_valid_ppo_ratio_inputs(
            ratio,
            clip_coef,
            advantages,
            denoising_indices,
            denoising_steps,
        )
    )
    if ratio.numel() == 0:
        return

    upper_mask = ratio > 1.0 + clip_coef
    lower_mask = ratio < 1.0 - clip_coef
    outside_mask = upper_mask | lower_mask
    # PPO 的 min(surr1, surr2) 只在正优势越过上界或负优势越过下界时
    # 真正选择裁剪后的代理目标；反方向越界不会截断该样本的梯度。
    objective_clipped_mask = (
        ((advantages > 0) & upper_mask)
        | ((advantages < 0) & lower_mask)
    )

    count = int(ratio.numel())
    # 每个 minibatch 只进行一次 GPU->CPU 同步；逐指标调用 .item() 会给长时间
    # DPPO 更新引入大量同步开销。ratio 接近 1 时方差很小，因此用 float64
    # 计算一、二阶矩，避免 E[x²]-E[x]² 的消减误差吞掉真实标准差。
    ratio_for_stats = ratio.double()
    base_values = torch.stack(
        (
            ratio_for_stats.sum(),
            (ratio_for_stats * ratio_for_stats).sum(),
            ratio_for_stats.min(),
            ratio_for_stats.max(),
            outside_mask.sum().to(ratio_for_stats.dtype),
            objective_clipped_mask.sum().to(ratio_for_stats.dtype),
            upper_mask.sum().to(ratio_for_stats.dtype),
            lower_mask.sum().to(ratio_for_stats.dtype),
        )
    )
    per_step_count = torch.bincount(
        denoising_indices,
        minlength=denoising_steps,
    ).to(ratio_for_stats.dtype)
    per_step_outside = torch.bincount(
        denoising_indices,
        weights=outside_mask.to(ratio_for_stats.dtype),
        minlength=denoising_steps,
    )
    per_step_objective = torch.bincount(
        denoising_indices,
        weights=objective_clipped_mask.to(ratio_for_stats.dtype),
        minlength=denoising_steps,
    )
    packed_values = torch.cat(
        (base_values, per_step_count, per_step_outside, per_step_objective)
    ).cpu().tolist()
    (
        ratio_sum,
        ratio_sum_sq,
        ratio_min,
        ratio_max,
        outside_count,
        objective_count,
        upper_count,
        lower_count,
    ) = packed_values[:8]
    offset = 8
    step_counts = packed_values[offset : offset + denoising_steps]
    offset += denoising_steps
    step_outside_counts = packed_values[offset : offset + denoising_steps]
    offset += denoising_steps
    step_objective_counts = packed_values[offset : offset + denoising_steps]

    stats["count"] += count
    stats["sum"] += float(ratio_sum)
    stats["sum_sq"] += float(ratio_sum_sq)
    stats["min"] = min(stats["min"], float(ratio_min))
    stats["max"] = max(stats["max"], float(ratio_max))
    stats["outside_clip"] += int(outside_count)
    stats["objective_clipped"] += int(objective_count)
    stats["upper_clip"] += int(upper_count)
    stats["lower_clip"] += int(lower_count)

    for step in range(denoising_steps):
        stats["per_step_count"][step] += int(step_counts[step])
        stats["per_step_outside_clip"][step] += int(step_outside_counts[step])
        stats["per_step_objective_clipped"][step] += int(
            step_objective_counts[step]
        )


def finalize_ppo_ratio_stats(stats: dict) -> dict:
    """将 PPO ratio 在线累计量转换为可上传 W&B 的标量。"""
    count = int(stats["count"])
    denoising_steps = int(stats["denoising_steps"])
    if count == 0:
        mean = std = minimum = maximum = float("nan")
        outside_fraction = objective_fraction = float("nan")
        upper_fraction = lower_fraction = float("nan")
    else:
        mean = stats["sum"] / count
        variance = max(stats["sum_sq"] / count - mean * mean, 0.0)
        std = math.sqrt(variance)
        minimum = float(stats["min"])
        maximum = float(stats["max"])
        outside_fraction = stats["outside_clip"] / count
        objective_fraction = stats["objective_clipped"] / count
        upper_fraction = stats["upper_clip"] / count
        lower_fraction = stats["lower_clip"] / count

    per_step_outside_fraction = []
    per_step_objective_fraction = []
    for step in range(denoising_steps):
        step_count = int(stats["per_step_count"][step])
        if step_count == 0:
            per_step_outside_fraction.append(float("nan"))
            per_step_objective_fraction.append(float("nan"))
        else:
            per_step_outside_fraction.append(
                stats["per_step_outside_clip"][step] / step_count
            )
            per_step_objective_fraction.append(
                stats["per_step_objective_clipped"][step] / step_count
            )

    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "outside_clip_fraction": outside_fraction,
        "objective_clip_fraction": objective_fraction,
        "upper_clip_fraction": upper_fraction,
        "lower_clip_fraction": lower_fraction,
        "per_step_outside_clip_fraction": per_step_outside_fraction,
        "per_step_objective_clip_fraction": per_step_objective_fraction,
    }


@torch.no_grad()
def summarize_ppo_ratio(
    ratio: torch.Tensor,
    clip_coef: torch.Tensor,
    advantages: torch.Tensor,
    denoising_indices: torch.Tensor,
    denoising_steps: int,
) -> dict:
    """汇总固定 probe 的 ratio、分位数和动态裁剪诊断。"""
    stats = init_ppo_ratio_stats(denoising_steps)
    update_ppo_ratio_stats(
        stats,
        ratio,
        clip_coef,
        advantages,
        denoising_indices,
    )
    summary = finalize_ppo_ratio_stats(stats)
    valid_ratio, _, _, _ = _flatten_valid_ppo_ratio_inputs(
        ratio,
        clip_coef,
        advantages,
        denoising_indices,
        denoising_steps,
    )
    if valid_ratio.numel() == 0:
        summary.update(
            {
                "p05": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
            }
        )
    else:
        quantiles = torch.quantile(
            valid_ratio,
            torch.tensor([0.05, 0.50, 0.95], device=valid_ratio.device),
        )
        summary.update(
            {
                "p05": float(quantiles[0].item()),
                "p50": float(quantiles[1].item()),
                "p95": float(quantiles[2].item()),
            }
        )
    return summary
