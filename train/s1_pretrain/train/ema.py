"""预训练策略的指数滑动平均（EMA）控制器。"""

from __future__ import annotations

import copy
import logging

import torch


class PolicyEMA:
    """维护不参与反向传播的策略影子权重。

    在线策略始终由优化器更新；达到 ``update_after_step`` 时，EMA先精确
    复制在线策略，之后才进行指数滑动平均。BatchNorm等buffer每步直接复制，
    避免使用滞后的运行统计量。
    """

    def __init__(
        self,
        online_policy,
        *,
        decay: float,
        update_after_step: int,
    ):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError(f"EMA decay必须位于[0, 1)，当前为{decay}。")
        if isinstance(update_after_step, bool) or int(update_after_step) < 0:
            raise ValueError(
                f"EMA update_after_step必须是非负整数，当前为{update_after_step}。"
            )

        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.ema_policy = copy.deepcopy(online_policy)
        self.ema_policy.requires_grad_(False)
        self.ema_policy.eval()
        self.ready = False
        self.num_updates = 0
        self.last_step = -1

    @classmethod
    def from_policy_config(cls, online_policy) -> "PolicyEMA | None":
        config = online_policy.config
        if not bool(getattr(config, "use_ema", False)):
            return None
        return cls(
            online_policy,
            decay=float(getattr(config, "ema_decay", 0.999)),
            update_after_step=int(
                getattr(config, "ema_update_after_step", 1000)
            ),
        )

    @torch.no_grad()
    def _copy_online(self, online_policy) -> None:
        self.ema_policy.load_state_dict(online_policy.state_dict(), strict=True)
        self.ema_policy.requires_grad_(False)
        self.ema_policy.eval()

    @torch.no_grad()
    def update(self, online_policy, step: int) -> bool:
        """在一次成功的optimizer step后更新EMA；返回本步是否更新。"""
        step = int(step)
        if step < self.update_after_step:
            return False

        if not self.ready:
            self._copy_online(online_policy)
            self.ready = True
            self.num_updates = 1
            self.last_step = step
            logging.info(
                "EMA已在step=%d由在线策略初始化: decay=%g",
                step,
                self.decay,
            )
            return True

        online_parameters = dict(online_policy.named_parameters())
        ema_parameters = dict(self.ema_policy.named_parameters())
        if online_parameters.keys() != ema_parameters.keys():
            raise RuntimeError("EMA与在线策略的参数名称不一致。")

        one_minus_decay = 1.0 - self.decay
        for name, online_parameter in online_parameters.items():
            ema_parameter = ema_parameters[name]
            if online_parameter.requires_grad and torch.is_floating_point(
                online_parameter
            ):
                ema_parameter.mul_(self.decay).add_(
                    online_parameter.detach(),
                    alpha=one_minus_decay,
                )
            else:
                ema_parameter.copy_(online_parameter.detach())

        online_buffers = dict(online_policy.named_buffers())
        ema_buffers = dict(self.ema_policy.named_buffers())
        if online_buffers.keys() != ema_buffers.keys():
            raise RuntimeError("EMA与在线策略的buffer名称不一致。")
        for name, online_buffer in online_buffers.items():
            ema_buffers[name].copy_(online_buffer.detach())

        self.num_updates += 1
        self.last_step = step
        self.ema_policy.eval()
        return True

    def evaluation_policy(self, online_policy):
        """EMA尚未初始化时回退为在线策略。"""
        if not self.ready:
            return online_policy
        self.ema_policy.eval()
        return self.ema_policy

    def checkpoint_policy(self, online_policy):
        """checkpoint的默认部署权重；EMA未就绪时保存在线权重副本。"""
        return self.evaluation_policy(online_policy)

    def metadata(self) -> dict[str, int | float | bool]:
        return {
            "ready": bool(self.ready),
            "decay": float(self.decay),
            "update_after_step": int(self.update_after_step),
            "num_updates": int(self.num_updates),
            "last_step": int(self.last_step),
        }

    def restore(self, ema_policy, metadata: dict) -> None:
        """严格恢复EMA权重及更新进度。"""
        if not metadata:
            raise KeyError("training_state.pth中缺少EMA元数据。")
        saved_decay = float(metadata["decay"])
        saved_start = int(metadata["update_after_step"])
        if saved_decay != self.decay or saved_start != self.update_after_step:
            raise ValueError(
                "EMA配置与checkpoint不一致："
                f"checkpoint=({saved_decay}, {saved_start})，"
                f"current=({self.decay}, {self.update_after_step})。"
            )
        self.ema_policy.load_state_dict(ema_policy.state_dict(), strict=True)
        self.ema_policy.requires_grad_(False)
        self.ema_policy.eval()
        self.ready = bool(metadata["ready"])
        self.num_updates = int(metadata["num_updates"])
        self.last_step = int(metadata["last_step"])

