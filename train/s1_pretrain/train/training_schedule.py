"""预训练的step/epoch周期事件判定。"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def uses_epoch_budget(cfg: DictConfig) -> bool:
    """仅在用户配置offline_epochs时启用epoch周期语义。"""

    return OmegaConf.select(
        cfg,
        "training.offline_epochs",
        default=None,
    ) is not None


def _positive_frequency(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name}必须为正整数、0或null，不能是布尔值。")
    numeric = float(value)
    if not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{name}必须为非负整数或null，当前为{value!r}。")
    integer = int(numeric)
    return integer if integer > 0 else None


def _nonnegative_integer(value, name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{name}必须为非负整数或null，不能是布尔值。")
    numeric = float(value)
    if not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{name}必须为非负整数或null，当前为{value!r}。")
    return int(numeric)


def evaluation_has_started(step: int, cfg: DictConfig) -> bool:
    """当前step是否已进入允许周期评估和Top-K筛选的epoch区间。

    ``eval_start_epoch=N`` 表示完成的epoch数小于N时只维护latest。
    固定step模式不应用该阈值。阈值大于总epoch时，最终step也不评估，
    但独立的保存调度仍会保留最终latest checkpoint。
    """

    step = int(step)
    if not uses_epoch_budget(cfg):
        return True

    start_epoch = _nonnegative_integer(
        OmegaConf.select(
            cfg,
            "training.eval_start_epoch",
            default=0,
        ),
        "training.eval_start_epoch",
    )
    if start_epoch == 0:
        return True

    steps_per_epoch = int(cfg.training.steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError(
            "epoch模式要求training.steps_per_epoch为正整数，当前为"
            f"{steps_per_epoch}。"
        )
    return step + 1 >= start_epoch * steps_per_epoch


def should_run_periodic_event(
    step: int,
    cfg: DictConfig,
    *,
    step_frequency_path: str,
    epoch_frequency_path: str,
) -> bool:
    """判断当前更新后是否触发评估或保存，并始终包含最后一步。

    epoch模式使用 ``step + 1``（已完成更新数）判断DataLoader边界，确保
    第N个epoch是在其最后一个batch更新完成后触发，而不是下个epoch首步。
    固定step模式保持项目原有的 ``step % frequency == 0`` 语义。
    """

    step = int(step)
    total_steps = int(cfg.training.offline_steps)
    if step == total_steps - 1:
        return True

    if uses_epoch_budget(cfg):
        steps_per_epoch = int(cfg.training.steps_per_epoch)
        if steps_per_epoch <= 0:
            raise ValueError(
                "epoch模式要求training.steps_per_epoch为正整数，当前为"
                f"{steps_per_epoch}。"
            )
        frequency_epochs = _positive_frequency(
            OmegaConf.select(cfg, epoch_frequency_path, default=None),
            epoch_frequency_path,
        )
        if frequency_epochs is None:
            return False
        completed_steps = step + 1
        if completed_steps % steps_per_epoch != 0:
            return False
        completed_epochs = completed_steps // steps_per_epoch
        return completed_epochs % frequency_epochs == 0

    frequency_steps = _positive_frequency(
        OmegaConf.select(cfg, step_frequency_path, default=None),
        step_frequency_path,
    )
    return bool(
        frequency_steps is not None
        and step > 0
        and step % frequency_steps == 0
    )


def should_evaluate(step: int, cfg: DictConfig) -> bool:
    return bool(
        evaluation_has_started(step, cfg)
        and should_run_periodic_event(
            step,
            cfg,
            step_frequency_path="training.eval_freq",
            epoch_frequency_path="training.eval_freq_epochs",
        )
    )


def should_save_checkpoint(step: int, cfg: DictConfig) -> bool:
    if not bool(OmegaConf.select(cfg, "training.save_checkpoint", default=False)):
        return False
    return should_run_periodic_event(
        step,
        cfg,
        step_frequency_path="training.save_freq",
        epoch_frequency_path="training.save_freq_epochs",
    )
