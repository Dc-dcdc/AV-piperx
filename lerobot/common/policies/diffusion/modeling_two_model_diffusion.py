#!/usr/bin/env python
"""双模型 Diffusion Policy。

该文件用于和 `dual_head_diffusion` 做对比：
- `dual_head_diffusion` 共享视觉/状态编码器，只分开 action denoising U-Net。
- `two_model_diffusion` 使用两个完整 DiffusionModel，视觉/状态编码器和 U-Net 全部分开。

外部数据格式保持不变，仍然输入/输出完整的 `action`。
"""

import copy
from collections import deque

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import populate_queues


class TwoModelDiffusionPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "diffusion-policy"],
):
    """两个完整 DiffusionModel 组成的策略。

    默认将完整 action 拆为：
      - 前 14 维：左右双臂动作，由 arm_model 建模。
      - 后 6 维：头部/视角动作，由 view_model 建模。

    两个子模型都拥有独立的观测编码器、U-Net 和 noise scheduler。
    """

    name = "two_model_diffusion"

    def __init__(
        self,
        config: DiffusionConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        if config is None:
            config = DiffusionConfig()
        self.config = config
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.action_dim = config.output_shapes["action"][0]
        arm_action_dim = getattr(config, "arm_action_dim", None)
        view_action_dim = getattr(config, "view_action_dim", None)
        if arm_action_dim is None and view_action_dim is None:
            arm_action_dim = 14 if self.action_dim == 20 else self.action_dim // 2
            view_action_dim = self.action_dim - arm_action_dim
        elif arm_action_dim is None:
            arm_action_dim = self.action_dim - int(view_action_dim)
        elif view_action_dim is None:
            view_action_dim = self.action_dim - int(arm_action_dim)
        self.arm_action_dim = int(arm_action_dim)
        self.view_action_dim = int(view_action_dim)
        self.view_loss_weight = float(getattr(config, "view_loss_weight", 0.2))
        if self.arm_action_dim <= 0 or self.view_action_dim <= 0:
            raise ValueError(
                f"arm_action_dim and view_action_dim must be positive, got "
                f"{self.arm_action_dim=} {self.view_action_dim=}."
            )
        if self.arm_action_dim + self.view_action_dim != self.action_dim:
            raise ValueError(
                f"Two-model action dims do not match output action dim: "
                f"arm={self.arm_action_dim}, view={self.view_action_dim}, action={self.action_dim}."
            )

        self.arm_model = DiffusionModel(self._make_head_config(self.arm_action_dim))
        self.view_model = DiffusionModel(self._make_head_config(self.view_action_dim))

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        self.use_env_state = "observation.environment_state" in config.input_shapes
        self._queues = None
        self.reset()

    def _make_head_config(self, action_dim: int) -> DiffusionConfig:
        """复制配置，并将输出 action 维度替换成子模型需要的维度。"""
        head_config = copy.deepcopy(self.config)
        head_config.output_shapes = dict(self.config.output_shapes)
        head_config.output_shapes["action"] = [action_dim]
        return head_config

    def reset(self):
        """清空观测和动作缓存，环境 reset 时调用。"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if len(self.expected_image_keys) > 0:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.use_env_state:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """根据当前观测生成单步完整 action。"""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.generate_actions(batch)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._queues["action"].extend(actions.transpose(0, 1))

        return self._queues["action"].popleft()

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """分别调用两个完整 DiffusionModel，并拼接成完整 action chunk。"""
        arm_actions = self.arm_model.generate_actions(batch)
        view_actions = self.view_model.generate_actions(batch)
        return torch.cat([arm_actions, view_actions], dim=-1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """计算两个完整模型的训练损失。"""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)

        if batch["action"].shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, got {batch['action'].shape[-1]}."
            )

        arm_batch = dict(batch)
        arm_batch["action"] = batch["action"][..., : self.arm_action_dim]
        view_batch = dict(batch)
        view_batch["action"] = batch["action"][..., self.arm_action_dim :]

        arm_loss = self.arm_model.compute_loss(arm_batch)
        view_loss = self.view_model.compute_loss(view_batch)
        loss = arm_loss + self.view_loss_weight * view_loss
        return {
            "loss": loss,
            "arm_loss": arm_loss.detach(),
            "view_loss": view_loss.detach(),
        }
