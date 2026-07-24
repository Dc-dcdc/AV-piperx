"""单头 DPPO 的 W&B 标签与指标字典构造工具。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf


DPPO_WANDB_PARAMETER_TAGS = (
    ("device", "device"),                                     # 训练设备
    ("n_envs", "env.n_envs"),                                 # 并行环境数量
    ("rollout", "training.rollout_steps"),                    # 每轮采集步数
    ("batch", "training.batch_size"),                         # PPO 批大小
    ("epochs", "training.update_epochs"),                     # 每轮更新次数
    ("grad_acc", "training.grad_accumulate"),                 # 梯度累计步数
    ("actor_lr", "training.actor_lr"),                        # Actor 学习率
    ("critic_lr", "training.critic_lr"),                      # Critic 学习率
    ("gamma", "training.gamma"),                              # 回报折扣因子
    ("gae", "training.gae_lambda"),                           # GAE 衰减系数
    ("clip", "training.clip_ploss_coef"),                     # PPO 裁剪上限
    ("target_kl", "training.target_kl"),                      # KL 早停阈值
    ("reward", "training.reward_source"),                     # 奖励来源
    ("rollout_policy", "training.rollout_policy"),            # 采集策略类型
    ("sample_std", "training.min_sampling_denoising_std"),    # 采样最小噪声
    ("logprob_std", "training.min_logprob_denoising_std"),    # 概率最小噪声
    ("ddim_eta", "training.ddim_eta"),                        # DDIM 随机强度
    ("act_steps", "policy.n_action_steps"),                   # 动作块执行长度
    ("denoise_steps", "policy.ft_denoising_steps"),           # 微调去噪步数
    ("train_vision", "training.train_vision_encoder"),        # 是否训练视觉编码器
)


def _format_wandb_tag_value(
    value: Any,  # 待格式化的配置值
) -> str:
    """把配置值转换成简短且稳定的 W&B 标签文本。"""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        return "-".join(_format_wandb_tag_value(item) for item in value)
    return str(value).replace(" ", "_")


def add_wandb_parameter_tags(
    logger: Any,        # LeRobot 日志记录器
    cfg: DictConfig,    # 当前完整训练配置
) -> None:
    """把实际生效的 DPPO 关键配置追加到当前 W&B run 标签。"""
    wandb_module = getattr(logger, "_wandb", None)
    wandb_run = getattr(wandb_module, "run", None)
    if wandb_run is None:
        return

    configured_tags = OmegaConf.select(cfg, "wandb.tags", default=[])
    if configured_tags is None:
        configured_tags = []
    elif isinstance(configured_tags, str):
        configured_tags = [configured_tags]
    else:
        configured_tags = list(configured_tags)

    parameter_tags = []
    for tag_name, config_path in DPPO_WANDB_PARAMETER_TAGS:
        value = OmegaConf.select(cfg, config_path, default=None)
        if value is not None:
            parameter_tags.append(
                f"{tag_name}:{_format_wandb_tag_value(value)}"
            )

    existing_tags = list(getattr(wandb_run, "tags", ()) or ())
    wandb_run.tags = tuple(
        dict.fromkeys(
            [*existing_tags, *map(str, configured_tags), *parameter_tags]
        )
    )


def _per_step_values(
    summary: Mapping[str, Any],    # ratio 统计汇总字典
    key: str,                      # 需要读取的逐步指标名称
    denoising_steps: int,          # 期望的去噪步数量
) -> list[Any]:
    """读取逐去噪步统计，并在日志上传前检查长度。"""
    values = list(summary[key])
    if len(values) != denoising_steps:
        raise ValueError(
            f"{key} 长度必须等于 denoising_steps，"
            f"当前为 {len(values)}/{denoising_steps}"
        )
    return values


def build_dppo_train_metrics(
    *,                                                          # 以下参数必须使用关键字传入
    completed_episode_count: int,                               # 本轮完成的 episode 数量
    rollout_success_rate: float,                                # 本轮 rollout 成功率
    rollout_average_return: float,                              # 本轮平均回报
    rollout_action_chunks: int,                                 # 本轮采集的动作块数量
    rollout_env_steps: int,                                     # 本轮执行的环境步数量
    critic_loss: float,                                         # Critic 平均损失
    actor_loss: float,                                          # Actor 平均策略损失
    bc_loss: float,                                             # 行为克隆正则损失
    avg_kl: float,                                              # PPO 平均 KL 散度
    max_kl: float,                                              # PPO 最大 KL 散度
    ratio_summary: Mapping[str, Any],                           # 训练期 ratio 汇总结果
    critic_explained_variance: float,                           # Critic 解释方差
    critic_value_return_correlation: float,                     # Value 与 Return 相关性
    logprob_advantage_correlation: float,                       # logprob 变化与优势相关性
    positive_advantage_mean_logprob_delta: float,               # 正优势平均 logprob 变化
    negative_advantage_mean_logprob_delta: float,               # 负优势平均 logprob 变化
    logprob_advantage_sign_agreement: float,                    # logprob 变化与优势符号一致率
    post_probe_size: int,                                       # 更新后固定 probe 样本数
    post_probe_logprob_delta_mean: float,                       # probe 平均 logprob 变化
    post_probe_logprob_advantage_correlation: float,            # probe 变化与优势相关性
    post_probe_positive_advantage_mean_logprob_delta: float,    # probe 正优势平均变化
    post_probe_negative_advantage_mean_logprob_delta: float,    # probe 负优势平均变化
    post_probe_logprob_advantage_sign_agreement: float,         # probe 符号一致率
    post_probe_ratio_summary: Mapping[str, Any],                # 更新后 probe ratio 汇总结果
    actor_update_enabled: bool,                                 # 本轮是否更新 Actor
    early_stop: bool,                                           # 本轮是否触发 KL 早停
    denoising_steps: int,                                       # 微调使用的去噪步数量
) -> dict[str, int | float]:
    """构造单头 DPPO 每轮训练上传的完整标量指标字典。"""
    denoising_steps = int(denoising_steps)
    if denoising_steps <= 0:
        raise ValueError(
            f"denoising_steps 必须大于 0，当前为 {denoising_steps}"
        )

    ratio_outside_by_step = _per_step_values(
        ratio_summary,
        "per_step_outside_clip_fraction",
        denoising_steps,
    )
    ratio_objective_by_step = _per_step_values(
        ratio_summary,
        "per_step_objective_clip_fraction",
        denoising_steps,
    )
    probe_outside_by_step = _per_step_values(
        post_probe_ratio_summary,
        "per_step_outside_clip_fraction",
        denoising_steps,
    )
    probe_objective_by_step = _per_step_values(
        post_probe_ratio_summary,
        "per_step_objective_clip_fraction",
        denoising_steps,
    )

    metrics: dict[str, int | float] = {
        "rollout_completed_episodes": int(completed_episode_count),                                                            # 完成回合数
        "rollout_success_rate": float(rollout_success_rate),                                                                   # 采集成功率
        "rollout_average_return": float(rollout_average_return),                                                               # 平均回报
        "rollout_action_chunks": int(rollout_action_chunks),                                                                   # 动作块数量
        "rollout_env_steps": int(rollout_env_steps),                                                                           # 环境交互步数
        "loss_critic": float(critic_loss),                                                                                     # Critic 损失
        "loss_actor": float(actor_loss),                                                                                       # Actor 策略损失
        "loss_bc": float(bc_loss),                                                                                             # 行为克隆损失
        "ppo_avg_kl": float(avg_kl),                                                                                           # 平均 KL 散度
        "ppo_max_kl": float(max_kl),                                                                                           # 最大 KL 散度
        "ppo_ratio_sample_count": int(ratio_summary["count"]),                                                                 # ratio 样本数
        "ppo_ratio_mean": float(ratio_summary["mean"]),                                                                        # ratio 均值
        "ppo_ratio_std": float(ratio_summary["std"]),                                                                          # ratio 标准差
        "ppo_ratio_min": float(ratio_summary["min"]),                                                                          # ratio 最小值
        "ppo_ratio_max": float(ratio_summary["max"]),                                                                          # ratio 最大值
        "ppo_ratio_outside_clip_fraction": float(ratio_summary["outside_clip_fraction"]),                                      # ratio 超出裁剪区间的比例
        "ppo_objective_clip_fraction": float(ratio_summary["objective_clip_fraction"]),                                        # PPO 代理目标实际被裁剪的比例
        "ppo_ratio_upper_clip_fraction": float(ratio_summary["upper_clip_fraction"]),                                          # ratio 超过裁剪上界的比例
        "ppo_ratio_lower_clip_fraction": float(ratio_summary["lower_clip_fraction"]),                                          # ratio 低于裁剪下界的比例
        "critic_explained_variance": float(critic_explained_variance),                                                         # 解释方差
        "critic_value_return_correlation": float(critic_value_return_correlation),                                             # Value 与 Return 相关性
        "logprob_advantage_correlation": float(logprob_advantage_correlation),                                                 # logprob 变化与优势相关性
        "positive_advantage_mean_logprob_delta": float(positive_advantage_mean_logprob_delta),                                 # 正优势样本平均 logprob 变化
        "negative_advantage_mean_logprob_delta": float(negative_advantage_mean_logprob_delta),                                 # 负优势样本平均 logprob 变化
        "logprob_advantage_sign_agreement": float(logprob_advantage_sign_agreement),                                           # 概率变化与优势符号一致率
        "post_update_probe_size": int(post_probe_size),                                                                        # 固定 probe 样本数
        "post_update_probe_logprob_delta_mean": float(post_probe_logprob_delta_mean),                                          # probe 平均 logprob 变化
        "post_update_probe_logprob_advantage_correlation": float(post_probe_logprob_advantage_correlation),                    # probe 概率变化与优势相关性
        "post_update_probe_positive_advantage_mean_logprob_delta": float(post_probe_positive_advantage_mean_logprob_delta),    # probe 正优势平均 logprob 变化
        "post_update_probe_negative_advantage_mean_logprob_delta": float(post_probe_negative_advantage_mean_logprob_delta),    # probe 负优势平均 logprob 变化
        "post_update_probe_logprob_advantage_sign_agreement": float(post_probe_logprob_advantage_sign_agreement),              # probe 概率变化与优势符号一致率
        "post_update_probe_ratio_mean": float(post_probe_ratio_summary["mean"]),                                               # probe ratio 均值
        "post_update_probe_ratio_std": float(post_probe_ratio_summary["std"]),                                                 # probe ratio 标准差
        "post_update_probe_ratio_min": float(post_probe_ratio_summary["min"]),                                                 # probe ratio 最小值
        "post_update_probe_ratio_max": float(post_probe_ratio_summary["max"]),                                                 # probe ratio 最大值
        "post_update_probe_ratio_p05": float(post_probe_ratio_summary["p05"]),                                                 # probe ratio 的 5% 分位数
        "post_update_probe_ratio_p50": float(post_probe_ratio_summary["p50"]),                                                 # probe ratio 的中位数
        "post_update_probe_ratio_p95": float(post_probe_ratio_summary["p95"]),                                                 # probe ratio 的 95% 分位数
        "post_update_probe_ratio_outside_clip_fraction": float(post_probe_ratio_summary["outside_clip_fraction"]),             # probe ratio 超出裁剪区间的比例
        "post_update_probe_objective_clip_fraction": float(post_probe_ratio_summary["objective_clip_fraction"]),               # probe 代理目标实际被裁剪的比例
        "actor_update_enabled": int(actor_update_enabled),                                                                     # 是否更新 Actor
        "early_stop": int(early_stop),                                                                                         # 是否触发 KL 早停
    }

    for step in range(denoising_steps):
        metrics[f"ppo_ratio_outside_clip_fraction_denoising_step_{step}"] = float(ratio_outside_by_step[step])                  # 当前步 ratio 越界比例
        metrics[f"ppo_objective_clip_fraction_denoising_step_{step}"] = float(ratio_objective_by_step[step])                    # 当前步实际裁剪比例
        metrics[f"post_update_probe_ratio_outside_clip_fraction_denoising_step_{step}"] = float(probe_outside_by_step[step])    # 当前步 probe ratio 越界比例
        metrics[f"post_update_probe_objective_clip_fraction_denoising_step_{step}"] = float(probe_objective_by_step[step])      # 当前步 probe 实际裁剪比例
    return metrics


def build_dppo_eval_metrics(
    *,                                                    # 以下参数必须使用关键字传入
    success_rate: float,                                  # 当前评估成功率
    average_reward: float,                                # 当前评估平均奖励
    best_success_rate: float,                             # 历史最佳成功率
    best_average_reward: float,                           # 历史最佳平均奖励
    new_best_actor: bool,                                 # 当前 Actor 是否刷新最佳记录
    eval_collapsed: bool,                                 # 当前评估是否判定策略塌陷
    candidate_eligible: bool,                             # 当前结果是否可参与模型筛选
    rollback_triggered: bool,                             # 本轮是否触发 Actor 回滚
    rollback_best_success_rate: float | None = None,      # 回滚基线成功率
    rollback_best_average_reward: float | None = None,    # 回滚基线平均奖励
    rollback_best_policy_loss: float | None = None,       # 回滚基线策略损失
) -> dict[str, int | float]:
    """构造评估、历史最佳和可选回滚基线的上传指标。"""
    metrics: dict[str, int | float] = {
        "success_rate": float(success_rate),                  # 当前评估成功率
        "average_reward": float(average_reward),              # 当前评估平均奖励
        "best_success_rate": float(best_success_rate),        # 历史最佳成功率
        "best_average_reward": float(best_average_reward),    # 历史最佳平均奖励
        "new_best_actor": int(new_best_actor),                # 是否刷新最佳 Actor
        "eval_collapsed": int(eval_collapsed),                # 是否判定策略塌陷
        "candidate_eligible": int(candidate_eligible),        # 是否可参与模型筛选
        "rollback_triggered": int(rollback_triggered),        # 是否触发 Actor 回滚
    }

    rollback_values = (
        rollback_best_success_rate,
        rollback_best_average_reward,
        rollback_best_policy_loss,
    )
    if any(value is not None for value in rollback_values):
        if not all(value is not None for value in rollback_values):
            raise ValueError("回滚基线的 success、reward 和 policy loss 必须同时提供")
        metrics.update(
            {
                "rollback_best_success_rate": float(rollback_best_success_rate),        # 回滚基线成功率
                "rollback_best_average_reward": float(rollback_best_average_reward),    # 回滚基线平均奖励
                "rollback_best_policy_loss": float(rollback_best_policy_loss),          # 回滚基线策略损失
            }
        )
    return metrics
