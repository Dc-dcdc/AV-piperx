"""Learning-rate schedulers used by offline pretraining."""

from __future__ import annotations

import math

from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def resolve_lr_decay_steps(cfg) -> int:
    """Resolve the global step at which cosine decay reaches its floor.

    ``training.lr_decay_steps`` and ``training.lr_decay_epochs`` are mutually
    exclusive.  Epochs are converted with the already-resolved
    ``training.steps_per_epoch``.  When neither is configured, the scheduler
    reaches its floor at the end of offline training for backwards-compatible
    behavior.
    """

    decay_steps_value = OmegaConf.select(cfg, "training.lr_decay_steps", default=None)
    decay_epochs_value = OmegaConf.select(cfg, "training.lr_decay_epochs", default=None)
    if decay_steps_value is not None and decay_epochs_value is not None:
        raise ValueError(
            "training.lr_decay_steps与training.lr_decay_epochs不能同时设置。"
        )

    if decay_steps_value is not None:
        decay_steps_float = float(decay_steps_value)
        if not math.isfinite(decay_steps_float) or not decay_steps_float.is_integer():
            raise ValueError(
                "training.lr_decay_steps必须为正整数，当前为"
                f"{decay_steps_value!r}。"
            )
        decay_steps = int(decay_steps_float)
    elif decay_epochs_value is not None:
        decay_epochs_float = float(decay_epochs_value)
        if not math.isfinite(decay_epochs_float) or decay_epochs_float <= 0:
            raise ValueError(
                "training.lr_decay_epochs必须为正数，当前为"
                f"{decay_epochs_value!r}。"
            )
        steps_per_epoch_value = OmegaConf.select(
            cfg, "training.steps_per_epoch", default=None
        )
        if steps_per_epoch_value is None:
            raise ValueError(
                "使用training.lr_decay_epochs前必须先解析training.steps_per_epoch。"
            )
        steps_per_epoch_float = float(steps_per_epoch_value)
        if (
            not math.isfinite(steps_per_epoch_float)
            or not steps_per_epoch_float.is_integer()
            or steps_per_epoch_float <= 0
        ):
            raise ValueError(
                "training.steps_per_epoch必须为正整数，当前为"
                f"{steps_per_epoch_value!r}。"
            )
        # 支持例如0.5 epoch；向上取整，避免比配置的epoch更早到达下限。
        decay_steps = math.ceil(decay_epochs_float * int(steps_per_epoch_float))
    else:
        offline_steps_value = OmegaConf.select(
            cfg, "training.offline_steps", default=None
        )
        if offline_steps_value is None:
            raise ValueError(
                "未设置学习率衰减终点，且training.offline_steps尚未解析。"
            )
        offline_steps_float = float(offline_steps_value)
        if not math.isfinite(offline_steps_float) or not offline_steps_float.is_integer():
            raise ValueError(
                "training.offline_steps必须为正整数，当前为"
                f"{offline_steps_value!r}。"
            )
        decay_steps = int(offline_steps_float)

    warmup_steps = int(
        OmegaConf.select(cfg, "training.lr_warmup_steps", default=0) or 0
    )
    if decay_steps <= warmup_steps:
        raise ValueError(
            "学习率衰减终点必须晚于warmup终点："
            f"decay_steps={decay_steps}, warmup_steps={warmup_steps}。"
        )
    return decay_steps


def make_cosine_with_floor_scheduler(
    optimizer: Optimizer,
    *,
    num_warmup_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float,
    last_epoch: int = -1,
) -> LambdaLR:
    """Warm up, cosine-decay to a relative LR floor, then hold the floor.

    The same multiplier is applied to every optimizer parameter group.  Thus a
    main-network LR of ``1e-4`` and a backbone LR of ``1e-5`` with a floor ratio
    of ``0.01`` are held at ``1e-6`` and ``1e-7`` respectively.
    """

    num_warmup_steps = int(num_warmup_steps)
    num_decay_steps = int(num_decay_steps)
    min_lr_ratio = float(min_lr_ratio)
    if num_warmup_steps < 0:
        raise ValueError("num_warmup_steps必须大于等于0。")
    if num_decay_steps <= num_warmup_steps:
        raise ValueError("num_decay_steps必须大于num_warmup_steps。")
    if not math.isfinite(min_lr_ratio) or not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio必须位于[0, 1]区间。")

    def lr_multiplier(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        if current_step >= num_decay_steps:
            return min_lr_ratio

        progress = (current_step - num_warmup_steps) / (
            num_decay_steps - num_warmup_steps
        )
        cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_ratio

    return LambdaLR(optimizer, lr_multiplier, last_epoch=last_epoch)
