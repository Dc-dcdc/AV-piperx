#!/usr/bin/env python

"""A supervised candidate router for post-diffusion Arm-to-View correction."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.common.policies.diffusion.configuration_routed_post_diffusion_output_corrector import (
    RoutedPostDiffusionOutputCorrectorConfig,
)
from lerobot.common.policies.diffusion.modeling_post_diffusion_output_corrector import (
    PostDiffusionOutputCorrectorModel,
    PostDiffusionOutputCorrectorPolicy,
)


class SharedCandidateScorer(nn.Module):
    """Score one candidate using observation and temporal action evidence.

    The same scorer is called for ``none`` and ``Arm→View`` candidates.  Its
    observation-to-time attention is returned as an interpretable 16-step
    importance distribution.
    """

    def __init__(
        self,
        *,
        global_cond_dim: int,
        horizon: int,
        arm_dim: int,
        view_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.arm_dim = int(arm_dim)
        self.view_dim = int(view_dim)
        self.d_model = int(d_model)

        arm_width = self.d_model // 2
        view_width = self.d_model - arm_width
        self.observation_encoder = nn.Sequential(
            nn.Linear(int(global_cond_dim), self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
        )
        self.arm_encoder = nn.Linear(self.arm_dim, arm_width)
        self.view_encoder = nn.Linear(self.view_dim, view_width)
        self.action_norm = nn.LayerNorm(self.d_model)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.horizon, self.d_model)
        )
        nn.init.normal_(self.temporal_position, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.d_model),
        )
        self.observation_to_time = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(self.d_model)
        self.score_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )

    def forward(
        self,
        global_condition: Tensor,
        trajectory: Tensor,
    ) -> tuple[Tensor, Tensor]:
        expected = (
            trajectory.shape[0],
            self.horizon,
            self.arm_dim + self.view_dim,
        )
        if tuple(trajectory.shape) != expected:
            raise ValueError(
                f"Router候选轨迹应为{expected}，实际为"
                f"{tuple(trajectory.shape)}。"
            )
        arm = trajectory[..., : self.arm_dim]
        view = trajectory[..., self.arm_dim :]
        action_tokens = torch.cat(
            [self.arm_encoder(arm), self.view_encoder(view)],
            dim=-1,
        )
        action_tokens = self.action_norm(action_tokens)
        action_tokens = action_tokens + self.temporal_position
        action_tokens = self.temporal_encoder(action_tokens)

        observation = self.observation_encoder(global_condition).unsqueeze(1)
        context, attention = self.observation_to_time(
            query=observation,
            key=action_tokens,
            value=action_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        fused = self.fusion_norm(observation + context).squeeze(1)
        score = self.score_head(fused).squeeze(-1)
        return score, attention.squeeze(1)


class RoutedPostDiffusionOutputCorrectorModel(
    PostDiffusionOutputCorrectorModel
):
    """Generate one dual trajectory, then route between none and Arm→View."""

    def __init__(self, config: RoutedPostDiffusionOutputCorrectorConfig):
        super().__init__(config)
        self.config = config
        global_cond_dim = int(config.input_shapes["observation.state"][0])
        image_keys = [
            key
            for key in config.input_shapes
            if key.startswith("observation.image")
        ]
        if image_keys:
            global_cond_dim += self.rgb_encoder.feature_dim * len(image_keys)
        if "observation.environment_state" in config.input_shapes:
            global_cond_dim += int(
                config.input_shapes["observation.environment_state"][0]
            )
        global_cond_dim *= int(config.n_obs_steps)
        self.global_cond_dim = int(global_cond_dim)
        self.output_router = SharedCandidateScorer(
            global_cond_dim=self.global_cond_dim,
            horizon=config.horizon,
            arm_dim=self.arm_action_dim,
            view_dim=self.view_action_dim,
            d_model=config.router_d_model,
            num_heads=config.router_num_heads,
            num_layers=config.router_num_layers,
            ffn_dim=config.router_ffn_dim,
            dropout=config.router_dropout,
        )
        self.router_threshold = float(config.router_threshold)
        self.router_mode = str(config.router_mode)
        self.reset_router_statistics()

    def set_router_mode(self, mode: str) -> str:
        mode = str(mode)
        if mode not in {"router", "none", "arm_to_view"}:
            raise ValueError(
                "Router推理模式必须是'router'、'none'或'arm_to_view'。"
            )
        self.router_mode = mode
        self.config.router_mode = mode
        return mode

    def set_router_threshold(self, threshold: float) -> float:
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Router阈值必须位于[0, 1]。")
        self.router_threshold = threshold
        self.config.router_threshold = threshold
        return threshold

    def reset_router_statistics(self) -> None:
        self._router_decisions = 0
        self._router_activations = 0
        self._router_probability_sum = 0.0
        self.last_router_gate: Tensor | None = None
        self.last_router_probability: Tensor | None = None

    def router_statistics(self) -> dict[str, float]:
        decisions = int(self._router_decisions)
        return {
            "decisions": decisions,
            "activations": int(self._router_activations),
            "activation_rate": (
                float(self._router_activations) / decisions
                if decisions
                else 0.0
            ),
            "mean_probability": (
                float(self._router_probability_sum) / decisions
                if decisions
                else 0.0
            ),
        }

    def score_candidates(
        self,
        global_condition: Tensor,
        none_trajectory: Tensor,
        arm_to_view_trajectory: Tensor,
    ) -> dict[str, Tensor]:
        q_none, none_attention = self.output_router(
            global_condition,
            none_trajectory,
        )
        q_arm_to_view, arm_to_view_attention = self.output_router(
            global_condition,
            arm_to_view_trajectory,
        )
        logit = q_arm_to_view - q_none
        return {
            "q_none": q_none,
            "q_arm_to_view": q_arm_to_view,
            "router_logit": logit,
            "router_probability": torch.sigmoid(logit),
            "none_temporal_attention": none_attention,
            "arm_to_view_temporal_attention": arm_to_view_attention,
        }

    def compute_router_loss(
        self,
        global_condition: Tensor,
        none_trajectory: Tensor,
        arm_to_view_trajectory: Tensor,
        label: Tensor,
        sample_weight: Tensor | None = None,
        *,
        positive_weight: Tensor | None = None,
    ) -> dict[str, Tensor]:
        diagnostics = self.score_candidates(
            global_condition,
            none_trajectory,
            arm_to_view_trajectory,
        )
        label = label.to(
            device=diagnostics["router_logit"].device,
            dtype=diagnostics["router_logit"].dtype,
        )
        valid = (label >= 0.0) & (label <= 1.0)
        if sample_weight is None:
            weight = valid.to(label.dtype)
        else:
            weight = sample_weight.to(
                device=label.device,
                dtype=label.dtype,
            ) * valid.to(label.dtype)
        if not bool(valid.any()):
            raise ValueError("当前Router batch没有有效的0/1监督标签。")

        per_sample_loss = F.binary_cross_entropy_with_logits(
            diagnostics["router_logit"],
            label.clamp(0.0, 1.0),
            reduction="none",
            pos_weight=positive_weight,
        )
        loss = (per_sample_loss * weight).sum() / weight.sum().clamp_min(1.0)
        prediction = (
            diagnostics["router_probability"] >= self.router_threshold
        )
        target = label >= 0.5
        accuracy = (
            ((prediction == target).to(weight.dtype) * weight).sum()
            / weight.sum().clamp_min(1.0)
        )
        return {
            "loss": loss,
            "router_accuracy": accuracy.detach(),
            "router_probability_mean": (
                diagnostics["router_probability"].detach() * weight
            ).sum()
            / weight.sum().clamp_min(1.0),
            "q_none_mean": diagnostics["q_none"].detach().mean(),
            "q_arm_to_view_mean": diagnostics[
                "q_arm_to_view"
            ].detach().mean(),
            "router_logit_rms": diagnostics["router_logit"]
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
        }

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(
                f"Expected {self.config.n_obs_steps} observation steps, got "
                f"{n_obs_steps}."
            )
        global_condition = self._prepare_global_conditioning(batch)
        raw_arm, raw_view = self.generate_baseline_full_trajectories(
            batch_size,
            global_cond=global_condition,
        )

        if self.router_mode == "none":
            selected_arm, selected_view = raw_arm, raw_view
            self.last_router_gate = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=raw_arm.device,
            )
            self.last_router_probability = torch.zeros(
                batch_size,
                dtype=raw_arm.dtype,
                device=raw_arm.device,
            )
            self._router_decisions += int(batch_size)
        else:
            corrected_arm, corrected_view, _ = self.apply_output_correction(
                raw_arm,
                raw_view,
            )
            if self.router_mode == "arm_to_view":
                selected_arm, selected_view = corrected_arm, corrected_view
                self.last_router_gate = torch.ones(
                    batch_size,
                    dtype=torch.bool,
                    device=raw_arm.device,
                )
                self.last_router_probability = torch.ones(
                    batch_size,
                    dtype=raw_arm.dtype,
                    device=raw_arm.device,
                )
                self._router_decisions += int(batch_size)
                self._router_activations += int(batch_size)
                self._router_probability_sum += float(batch_size)
            else:
                none_trajectory = self.combine_action_heads(raw_arm, raw_view)
                corrected_trajectory = self.combine_action_heads(
                    corrected_arm,
                    corrected_view,
                )
                diagnostics = self.score_candidates(
                    global_condition,
                    none_trajectory,
                    corrected_trajectory,
                )
                probability = diagnostics["router_probability"]
                gate = probability >= self.router_threshold
                self.last_router_gate = gate.detach()
                self.last_router_probability = probability.detach()
                gate_3d = gate[:, None, None]
                # Hard selection keeps the none branch bitwise equal to raw dual
                # diffusion; no multiply-by-zero or post-selection clamp occurs.
                selected_arm = torch.where(gate_3d, corrected_arm, raw_arm)
                selected_view = torch.where(gate_3d, corrected_view, raw_view)
                self._router_decisions += int(gate.numel())
                self._router_activations += int(gate.sum().item())
                self._router_probability_sum += float(
                    probability.detach().sum().item()
                )

        actions = self.combine_action_heads(selected_arm, selected_view)
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]


class RoutedPostDiffusionOutputCorrectorPolicy(
    PostDiffusionOutputCorrectorPolicy
):
    """Self-contained composite policy whose only trainable part is Router."""

    name = "routed_post_diffusion_output_corrector"

    def __init__(
        self,
        config: RoutedPostDiffusionOutputCorrectorConfig | dict | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ) -> None:
        if config is None:
            config = RoutedPostDiffusionOutputCorrectorConfig()
        elif isinstance(config, dict):
            config = RoutedPostDiffusionOutputCorrectorConfig.from_dict(config)
        super().__init__(config, dataset_stats)

    def _make_diffusion_model(
        self,
        config: RoutedPostDiffusionOutputCorrectorConfig,
    ) -> RoutedPostDiffusionOutputCorrectorModel:
        return RoutedPostDiffusionOutputCorrectorModel(config)

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
        if hasattr(self, "diffusion"):
            self.diffusion.reset_router_statistics()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        required = {
            "global_condition",
            "none_trajectory",
            "arm_to_view_trajectory",
            "router_label",
        }
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"Router缓存batch缺少字段: {sorted(missing)}")
        return self.diffusion.compute_router_loss(
            batch["global_condition"],
            batch["none_trajectory"],
            batch["arm_to_view_trajectory"],
            batch["router_label"],
            batch.get("sample_weight"),
        )
