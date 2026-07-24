"""DPPO 评估历史指标与最佳 Actor 选择规则。"""

from __future__ import annotations


def canonical_checkpoint_metric(metric: str) -> str:
    """把配置中的指标别名规范为 success、reward 或 loss。"""
    normalized = str(metric).lower()
    if normalized in {"success", "success_rate", "sr"}:
        return "success"
    if normalized == "reward":
        return "reward"
    return "loss"


def update_historical_eval_best(
    *,
    best_success_rate: float,
    best_average_reward: float,
    success_rate: float,
    average_reward: float,
) -> tuple[float, float]:
    """分别更新历史最高成功率和历史最高平均奖励，保证二者单调不降。"""
    return (
        max(float(best_success_rate), float(success_rate)),
        max(float(best_average_reward), float(average_reward)),
    )


def is_eval_candidate_eligible(*, rollback_enabled: bool, eval_collapsed: bool) -> bool:
    """判断当前评估结果能否参与内存最佳与磁盘 Top-K 选择。"""
    return not (bool(rollback_enabled) and bool(eval_collapsed))


def is_better_eval_candidate(
    *,
    metric: str,
    candidate_loss: float,
    candidate_reward: float,
    candidate_success_rate: float,
    best_loss: float,
    best_reward: float,
    best_success_rate: float,
) -> bool:
    """按 TopKCheckpointManager 的同一规则判断当前 Actor 是否更优。"""
    normalized_metric = canonical_checkpoint_metric(metric)
    if normalized_metric == "success":
        # 成功率优先；成功率相同时，平均奖励更高者更优。
        return (float(candidate_success_rate), float(candidate_reward)) > (
            float(best_success_rate),
            float(best_reward),
        )
    if normalized_metric == "reward":
        return float(candidate_reward) > float(best_reward)
    return float(candidate_loss) < float(best_loss)
