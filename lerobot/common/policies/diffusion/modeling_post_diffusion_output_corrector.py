#!/usr/bin/env python

"""Frozen dual-head diffusion with a post-denoising action residual corrector.

Both original U-Nets finish their complete, independent denoising processes
before this module reads the normalized Arm/View trajectories.  Setting both
external scales to zero takes an explicit bypass and therefore returns the raw
dual-head result without executing any corrector operation.
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.common.policies.diffusion.configuration_post_diffusion_output_corrector import (
    PostDiffusionOutputCorrectorConfig,
)
from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DualHeadDiffusionModel,
    DualHeadDiffusionPolicy,
)


class SharedLinearBipartiteCorrector(nn.Module):
    """A shared 14x6 linear graph used in both correction directions."""

    def __init__(
        self,
        *,
        arm_dim: int,
        view_dim: int,
        residual_limit: float,
    ) -> None:
        super().__init__()
        self.arm_dim = int(arm_dim)
        self.view_dim = int(view_dim)
        self.residual_limit = float(residual_limit)
        self.affinity = nn.Parameter(torch.zeros(self.arm_dim, self.view_dim))
        self.arm_bias = nn.Parameter(torch.zeros(self.arm_dim))
        self.view_bias = nn.Parameter(torch.zeros(self.view_dim))

    def forward(
        self,
        arm_trajectory: Tensor,
        view_trajectory: Tensor,
        *,
        need_view_to_arm: bool,
        need_arm_to_view: bool,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        arm_delta = torch.zeros_like(arm_trajectory)
        view_delta = torch.zeros_like(view_trajectory)
        if need_view_to_arm:
            arm_raw = F.linear(view_trajectory, self.affinity, self.arm_bias)
            arm_delta = self.residual_limit * torch.tanh(arm_raw)
        if need_arm_to_view:
            view_raw = F.linear(
                arm_trajectory,
                self.affinity.transpose(0, 1),
                self.view_bias,
            )
            view_delta = self.residual_limit * torch.tanh(view_raw)

        graph = self.affinity.detach().abs()
        graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return arm_delta, view_delta, {"shared_graph": graph}


class SharedBipartiteAttentionCorrector(nn.Module):
    """One trajectory-level Arm/View affinity graph shared by both directions.

    Each action dimension is one token whose feature vector contains its full
    horizon.  Attention logits have shape [B, arm_dim, view_dim], so each
    trajectory has exactly one interpretable 14x6 graph.  All value heads use
    this same graph.  The reverse direction uses the exact transposed logits
    with a separate source-axis softmax rather than learning another graph.
    """

    def __init__(
        self,
        *,
        horizon: int,
        arm_dim: int,
        view_dim: int,
        d_model: int,
        num_heads: int,
        dropout: float,
        residual_limit: float,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.arm_dim = int(arm_dim)
        self.view_dim = int(view_dim)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.residual_limit = float(residual_limit)

        self.arm_trajectory_encoder = nn.Sequential(
            nn.Linear(self.horizon, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
        )
        self.view_trajectory_encoder = nn.Sequential(
            nn.Linear(self.horizon, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
        )
        self.arm_dimension_embedding = nn.Parameter(
            torch.zeros(1, self.arm_dim, self.d_model)
        )
        self.view_dimension_embedding = nn.Parameter(
            torch.zeros(1, self.view_dim, self.d_model)
        )
        nn.init.normal_(self.arm_dimension_embedding, std=0.02)
        nn.init.normal_(self.view_dimension_embedding, std=0.02)

        self.arm_query = nn.Linear(self.d_model, self.d_model)
        self.view_key = nn.Linear(self.d_model, self.d_model)
        self.arm_value = nn.Linear(self.d_model, self.d_model)
        self.view_value = nn.Linear(self.d_model, self.d_model)
        self.attention_dropout = nn.Dropout(dropout)

        self.arm_output = nn.Linear(self.d_model, self.horizon)
        self.view_output = nn.Linear(self.d_model, self.horizon)
        # Only the action-delta projections are zero initialized.  The graph
        # remains trainable as soon as these projections leave zero.
        nn.init.zeros_(self.arm_output.weight)
        nn.init.zeros_(self.arm_output.bias)
        nn.init.zeros_(self.view_output.weight)
        nn.init.zeros_(self.view_output.bias)

    def _split_heads(self, tokens: Tensor) -> Tensor:
        batch_size, token_count, _ = tokens.shape
        return (
            tokens.reshape(
                batch_size,
                token_count,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def _merge_heads(self, tokens: Tensor) -> Tensor:
        batch_size, _, token_count, _ = tokens.shape
        return (
            tokens.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, token_count, self.d_model)
        )

    def forward(
        self,
        arm_trajectory: Tensor,
        view_trajectory: Tensor,
        *,
        need_view_to_arm: bool,
        need_arm_to_view: bool,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        arm_tokens = self.arm_trajectory_encoder(
            arm_trajectory.transpose(1, 2)
        )
        view_tokens = self.view_trajectory_encoder(
            view_trajectory.transpose(1, 2)
        )
        arm_tokens = arm_tokens + self.arm_dimension_embedding
        view_tokens = view_tokens + self.view_dimension_embedding

        arm_query = self.arm_query(arm_tokens)
        view_key = self.view_key(view_tokens)
        logits = torch.matmul(
            arm_query,
            view_key.transpose(-2, -1),
        ) / math.sqrt(self.d_model)

        # Same graph, different source normalization for each direction.
        view_to_arm_weights = torch.softmax(logits, dim=-1)
        arm_to_view_weights = torch.softmax(
            logits.transpose(-2, -1),
            dim=-1,
        )

        arm_delta = torch.zeros_like(arm_trajectory)
        view_delta = torch.zeros_like(view_trajectory)
        if need_view_to_arm:
            view_values = self._split_heads(self.view_value(view_tokens))
            arm_context = torch.matmul(
                self.attention_dropout(view_to_arm_weights).unsqueeze(1),
                view_values,
            )
            arm_raw = self.arm_output(self._merge_heads(arm_context))
            arm_delta = (
                self.residual_limit * torch.tanh(arm_raw)
            ).transpose(1, 2)

        if need_arm_to_view:
            arm_values = self._split_heads(self.arm_value(arm_tokens))
            view_context = torch.matmul(
                self.attention_dropout(arm_to_view_weights).unsqueeze(1),
                arm_values,
            )
            view_raw = self.view_output(self._merge_heads(view_context))
            view_delta = (
                self.residual_limit * torch.tanh(view_raw)
            ).transpose(1, 2)

        shared_graph = view_to_arm_weights
        view_to_arm_entropy = -(
            shared_graph.clamp_min(1e-12)
            * shared_graph.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        arm_to_view_entropy = -(
            arm_to_view_weights.clamp_min(1e-12)
            * arm_to_view_weights.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        active_entropies = []
        active_max_weights = []
        if need_view_to_arm:
            active_entropies.append(view_to_arm_entropy)
            active_max_weights.append(view_to_arm_weights.max())
        if need_arm_to_view:
            active_entropies.append(arm_to_view_entropy)
            active_max_weights.append(arm_to_view_weights.max())
        return arm_delta, view_delta, {
            "shared_graph": shared_graph,
            "shared_affinity": logits,
            "view_to_arm_graph": view_to_arm_weights,
            "arm_to_view_graph": arm_to_view_weights,
            "graph_entropy": torch.stack(active_entropies).mean(),
            "graph_max_weight": torch.stack(active_max_weights).max(),
            "view_to_arm_graph_entropy": view_to_arm_entropy,
            "arm_to_view_graph_entropy": arm_to_view_entropy,
            "view_to_arm_graph_max_weight": view_to_arm_weights.max(),
            "arm_to_view_graph_max_weight": arm_to_view_weights.max(),
            "shared_affinity_rms": logits.float().square().mean().sqrt(),
        }


class PostDiffusionOutputCorrectorModel(DualHeadDiffusionModel):
    """Independent dual diffusion followed by a bounded output adapter."""

    def __init__(self, config: PostDiffusionOutputCorrectorConfig):
        super().__init__(config)
        self.config = config
        corrector_kwargs = {
            "arm_dim": self.arm_action_dim,
            "view_dim": self.view_action_dim,
            "residual_limit": config.output_corrector_residual_limit,
        }
        if config.output_corrector_type == "linear":
            self.output_corrector = SharedLinearBipartiteCorrector(
                **corrector_kwargs
            )
        else:
            self.output_corrector = SharedBipartiteAttentionCorrector(
                horizon=config.horizon,
                d_model=config.output_corrector_d_model,
                num_heads=config.output_corrector_num_heads,
                dropout=config.output_corrector_dropout,
                **corrector_kwargs,
            )

        self.view_to_arm_output_scale = float(
            config.view_to_arm_output_scale
        )
        self.arm_to_view_output_scale = float(
            config.arm_to_view_output_scale
        )

    @staticmethod
    def _validate_scale(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")
        return value

    def set_output_correction_scales(
        self,
        *,
        view_to_arm: float | None = None,
        arm_to_view: float | None = None,
    ) -> dict[str, float]:
        if view_to_arm is not None:
            self.view_to_arm_output_scale = self._validate_scale(
                "view_to_arm_output_scale",
                view_to_arm,
            )
        if arm_to_view is not None:
            self.arm_to_view_output_scale = self._validate_scale(
                "arm_to_view_output_scale",
                arm_to_view,
            )
        self.config.view_to_arm_output_scale = self.view_to_arm_output_scale
        self.config.arm_to_view_output_scale = self.arm_to_view_output_scale
        return {
            "view_to_arm_output_scale": self.view_to_arm_output_scale,
            "arm_to_view_output_scale": self.arm_to_view_output_scale,
        }

    def generate_baseline_full_trajectories(
        self,
        batch_size: int,
        *,
        global_cond: Tensor | None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the original Arm sampler fully, then the original View sampler."""
        arm_actions = self.conditional_sample(
            self.arm_unet,
            self.arm_noise_scheduler,
            self.arm_action_dim,
            batch_size,
            global_cond=global_cond,
            generator=generator,
        )
        view_actions = self.conditional_sample(
            self.view_unet,
            self.view_noise_scheduler,
            self.view_action_dim,
            batch_size,
            global_cond=global_cond,
            generator=generator,
        )
        return arm_actions, view_actions

    def apply_output_correction(
        self,
        arm_trajectory: Tensor,
        view_trajectory: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        view_to_arm_scale = float(self.view_to_arm_output_scale)
        arm_to_view_scale = float(self.arm_to_view_output_scale)
        if view_to_arm_scale == 0.0 and arm_to_view_scale == 0.0:
            # Explicit bypass: no attention, multiplication, clamp, or rounding.
            return arm_trajectory, view_trajectory, {
                "arm_delta": torch.zeros_like(arm_trajectory),
                "view_delta": torch.zeros_like(view_trajectory),
            }

        arm_delta, view_delta, diagnostics = self.output_corrector(
            arm_trajectory,
            view_trajectory,
            need_view_to_arm=view_to_arm_scale > 0.0,
            need_arm_to_view=arm_to_view_scale > 0.0,
        )
        effective_arm_delta = view_to_arm_scale * arm_delta
        effective_view_delta = arm_to_view_scale * view_delta
        corrected_arm = arm_trajectory + effective_arm_delta
        corrected_view = view_trajectory + effective_view_delta

        if self.config.output_corrector_clamp_actions:
            limit = float(self.config.clip_sample_range)
            corrected_arm = corrected_arm.clamp(-limit, limit)
            corrected_view = corrected_view.clamp(-limit, limit)

        return corrected_arm, corrected_view, {
            **diagnostics,
            "arm_delta": effective_arm_delta,
            "view_delta": effective_view_delta,
        }

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(
                f"Expected {self.config.n_obs_steps} observation steps, got "
                f"{n_obs_steps}."
            )
        global_cond = self._prepare_global_conditioning(batch)
        arm_actions, view_actions = self.generate_baseline_full_trajectories(
            batch_size,
            global_cond=global_cond,
        )
        arm_actions, view_actions, _ = self.apply_output_correction(
            arm_actions,
            view_actions,
        )
        actions = self.combine_action_heads(arm_actions, view_actions)
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def compute_cached_correction_loss(
        self,
        baseline_trajectory: Tensor,
        target_trajectory: Tensor,
        action_is_pad: Tensor | None,
    ) -> dict[str, Tensor]:
        if baseline_trajectory.shape != target_trajectory.shape:
            raise ValueError(
                "Cached baseline and target trajectories must have identical "
                f"shape, got {tuple(baseline_trajectory.shape)} and "
                f"{tuple(target_trajectory.shape)}."
            )
        expected_shape = (
            baseline_trajectory.shape[0],
            self.config.horizon,
            self.action_dim,
        )
        if tuple(baseline_trajectory.shape) != expected_shape:
            raise ValueError(
                f"Expected cached trajectory shape {expected_shape}, got "
                f"{tuple(baseline_trajectory.shape)}."
            )

        baseline_arm, baseline_view = self._prepare_head_trajectories(
            baseline_trajectory
        )
        target_arm, target_view = self._prepare_head_trajectories(
            target_trajectory
        )
        corrected_arm, corrected_view, diagnostics = self.apply_output_correction(
            baseline_arm,
            baseline_view,
        )

        arm_squared_error = (corrected_arm - target_arm).square()
        view_squared_error = (corrected_view - target_view).square()
        baseline_arm_squared_error = (baseline_arm - target_arm).square()
        baseline_view_squared_error = (baseline_view - target_view).square()
        if self.config.do_mask_loss_for_padding:
            if action_is_pad is None:
                raise ValueError(
                    "action_is_pad is required when padding loss masking is enabled."
                )
            valid = (~action_is_pad).unsqueeze(-1).to(arm_squared_error.dtype)
            denominator = valid.sum().clamp_min(1.0)
            arm_loss = (arm_squared_error * valid).sum() / (
                denominator * self.arm_action_dim
            )
            view_loss = (view_squared_error * valid).sum() / (
                denominator * self.view_action_dim
            )
            baseline_arm_loss = (baseline_arm_squared_error * valid).sum() / (
                denominator * self.arm_action_dim
            )
            baseline_view_loss = (
                baseline_view_squared_error * valid
            ).sum() / (denominator * self.view_action_dim)
        else:
            arm_loss = arm_squared_error.mean()
            view_loss = view_squared_error.mean()
            baseline_arm_loss = baseline_arm_squared_error.mean()
            baseline_view_loss = baseline_view_squared_error.mean()

        arm_delta = diagnostics["arm_delta"]
        view_delta = diagnostics["view_delta"]
        residual_loss = arm_delta.square().mean() + (
            self.view_loss_weight * view_delta.square().mean()
        )
        arm_smoothness = (
            arm_delta[:, 1:] - arm_delta[:, :-1]
        ).square().mean()
        view_smoothness = (
            view_delta[:, 1:] - view_delta[:, :-1]
        ).square().mean()
        smoothness_loss = arm_smoothness + (
            self.view_loss_weight * view_smoothness
        )

        task_loss = arm_loss + self.view_loss_weight * view_loss
        total_loss = (
            task_loss
            + self.config.output_residual_loss_weight * residual_loss
            + self.config.output_smoothness_loss_weight * smoothness_loss
        )
        result = {
            "loss": total_loss,
            "task_loss": task_loss.detach(),
            "arm_loss": arm_loss.detach(),
            "view_loss": view_loss.detach(),
            "baseline_arm_loss": baseline_arm_loss.detach(),
            "baseline_view_loss": baseline_view_loss.detach(),
            "residual_loss": residual_loss.detach(),
            "smoothness_loss": smoothness_loss.detach(),
            "arm_delta_rms": arm_delta.detach().float().square().mean().sqrt(),
            "view_delta_rms": view_delta.detach().float().square().mean().sqrt(),
            "arm_delta_max_abs": arm_delta.detach().float().abs().max(),
            "view_delta_max_abs": view_delta.detach().float().abs().max(),
        }
        if "graph_entropy" in diagnostics:
            result["attention_graph_entropy"] = diagnostics[
                "graph_entropy"
            ].detach()
            result["attention_graph_max_weight"] = diagnostics[
                "graph_max_weight"
            ].detach()
            for metric_name in (
                "view_to_arm_graph_entropy",
                "arm_to_view_graph_entropy",
                "view_to_arm_graph_max_weight",
                "arm_to_view_graph_max_weight",
                "shared_affinity_rms",
            ):
                result[f"attention_{metric_name}"] = diagnostics[
                    metric_name
                ].detach()
        return result


class PostDiffusionOutputCorrectorPolicy(DualHeadDiffusionPolicy):
    """Self-contained frozen dual policy plus a trainable output corrector."""

    name = "post_diffusion_output_corrector"

    def __init__(
        self,
        config: PostDiffusionOutputCorrectorConfig | dict | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ) -> None:
        if config is None:
            config = PostDiffusionOutputCorrectorConfig()
        elif isinstance(config, dict):
            config = PostDiffusionOutputCorrectorConfig.from_dict(config)
        super().__init__(config, dataset_stats)

    def _make_diffusion_model(
        self,
        config: PostDiffusionOutputCorrectorConfig,
    ) -> PostDiffusionOutputCorrectorModel:
        return PostDiffusionOutputCorrectorModel(config)

    def reset(self) -> None:
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.expected_image_keys:
            self._queues["observation.images"] = deque(
                maxlen=self.config.n_obs_steps
            )
        if self.use_env_state:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps
            )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Train only from persistent normalized trajectory-cache tensors."""
        required = {
            "baseline_action_trajectory",
            "target_action_trajectory",
            "action_is_pad",
        }
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"Correction cache batch is missing: {sorted(missing)}")
        return self.diffusion.compute_cached_correction_loss(
            batch["baseline_action_trajectory"],
            batch["target_action_trajectory"],
            batch["action_is_pad"],
        )
