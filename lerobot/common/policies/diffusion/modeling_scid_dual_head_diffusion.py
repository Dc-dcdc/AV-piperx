#!/usr/bin/env python

"""Schur-Complement Innovation Diffusion (SCID) policy.

SCID keeps the shared observation encoder and the two independent diffusion
U-Nets from ``dual_head_diffusion``.  It changes only the coordinates modeled
by the second head: instead of the raw view action ``V``, the head models the
normalized innovation ``R = V - (G A + b)``.  The transform is fixed, stored in
the checkpoint, and inverted after sampling so the external action interface
remains the original concatenated Arm/View command.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DualHeadDiffusionModel,
    DualHeadDiffusionPolicy,
)


class SCIDDualHeadDiffusionPolicy(DualHeadDiffusionPolicy):
    """Dual-head policy whose View head generates an Arm-conditioned innovation."""

    name = "scid_dual_head_diffusion"

    def _make_diffusion_model(self, config: DiffusionConfig) -> "SCIDDualHeadDiffusionModel":
        return SCIDDualHeadDiffusionModel(config)


class SCIDDualHeadDiffusionModel(DualHeadDiffusionModel):
    """Factorize normalized actions into Arm motion and scaled View innovation."""

    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        action_normalization = config.output_normalization_modes.get("action")
        if action_normalization != "min_max":
            raise ValueError(
                "SCID currently requires min_max action normalization so reconstructed "
                f"environment actions have a defined [-1, 1] range; got {action_normalization!r}."
            )
        self.scid_residual_eps = float(getattr(config, "scid_residual_eps", 1e-6))
        self.scid_clamp_reconstructed_view = bool(
            getattr(config, "scid_clamp_reconstructed_view", True)
        )

        # Persistent buffers make the fitted coordinate transform part of the
        # policy checkpoint without allowing gradient updates.
        self.register_buffer(
            "scid_matrix",
            torch.zeros(self.view_action_dim, self.arm_action_dim),
        )
        self.register_buffer("scid_bias", torch.zeros(self.view_action_dim))
        self.register_buffer("scid_residual_scale", torch.ones(self.view_action_dim))
        self.register_buffer("scid_transform_fitted", torch.tensor(False, dtype=torch.bool))

    @property
    def is_scid_transform_fitted(self) -> bool:
        return bool(self.scid_transform_fitted.item())

    def _require_fitted_transform(self) -> None:
        if not self.is_scid_transform_fitted:
            raise RuntimeError(
                "SCID transform is not fitted. Fit it from the training dataset before "
                "training/inference, or load a SCID checkpoint containing the transform buffers."
            )
        if not torch.isfinite(self.scid_matrix).all() or not torch.isfinite(self.scid_bias).all():
            raise RuntimeError("Loaded SCID matrix/bias contains NaN or Inf values.")
        if not torch.isfinite(self.scid_residual_scale).all() or torch.any(
            self.scid_residual_scale < self.scid_residual_eps
        ):
            raise RuntimeError(
                "Loaded SCID residual scale is non-finite or smaller than scid_residual_eps."
            )

    @torch.no_grad()
    def set_scid_transform(
        self,
        matrix: Tensor,
        bias: Tensor,
        residual_scale: Tensor,
    ) -> None:
        """Install a fixed transform fitted in normalized action coordinates."""
        matrix = torch.as_tensor(
            matrix,
            device=self.scid_matrix.device,
            dtype=self.scid_matrix.dtype,
        )
        bias = torch.as_tensor(
            bias,
            device=self.scid_bias.device,
            dtype=self.scid_bias.dtype,
        )
        residual_scale = torch.as_tensor(
            residual_scale,
            device=self.scid_residual_scale.device,
            dtype=self.scid_residual_scale.dtype,
        )

        expected_matrix_shape = (self.view_action_dim, self.arm_action_dim)
        if tuple(matrix.shape) != expected_matrix_shape:
            raise ValueError(
                f"SCID matrix must have shape {expected_matrix_shape}, got {tuple(matrix.shape)}."
            )
        if tuple(bias.shape) != (self.view_action_dim,):
            raise ValueError(
                f"SCID bias must have shape {(self.view_action_dim,)}, got {tuple(bias.shape)}."
            )
        if tuple(residual_scale.shape) != (self.view_action_dim,):
            raise ValueError(
                "SCID residual_scale must have shape "
                f"{(self.view_action_dim,)}, got {tuple(residual_scale.shape)}."
            )
        if not torch.isfinite(matrix).all() or not torch.isfinite(bias).all():
            raise ValueError("SCID matrix and bias must contain only finite values.")
        if not torch.isfinite(residual_scale).all():
            raise ValueError("SCID residual_scale must contain only finite values.")
        if torch.any(residual_scale < self.scid_residual_eps):
            raise ValueError(
                "Every SCID residual scale must be at least "
                f"scid_residual_eps={self.scid_residual_eps}."
            )

        self.scid_matrix.copy_(matrix)
        self.scid_bias.copy_(bias)
        self.scid_residual_scale.copy_(residual_scale)
        self.scid_transform_fitted.fill_(True)

    def nominal_view_action(self, arm_actions: Tensor) -> Tensor:
        """Compute the normalized View motion linearly explained by Arm motion."""
        self._require_fitted_transform()
        if arm_actions.shape[-1] != self.arm_action_dim:
            raise ValueError(
                f"Expected Arm dim {self.arm_action_dim}, got {arm_actions.shape[-1]}."
            )
        return F.linear(arm_actions, self.scid_matrix, self.scid_bias)

    def encode_view_innovation(self, arm_actions: Tensor, view_actions: Tensor) -> Tensor:
        """Map normalized raw View actions to the scaled innovation in [-1, 1]."""
        if view_actions.shape[-1] != self.view_action_dim:
            raise ValueError(
                f"Expected View dim {self.view_action_dim}, got {view_actions.shape[-1]}."
            )
        residual = view_actions - self.nominal_view_action(arm_actions)
        return residual / self.scid_residual_scale

    def decode_view_innovation(
        self,
        arm_actions: Tensor,
        scaled_innovation: Tensor,
        *,
        clamp: bool | None = None,
    ) -> Tensor:
        """Reconstruct normalized raw View actions from Arm and innovation samples."""
        if scaled_innovation.shape[-1] != self.view_action_dim:
            raise ValueError(
                "Expected scaled innovation dim "
                f"{self.view_action_dim}, got {scaled_innovation.shape[-1]}."
            )
        view_actions = self.nominal_view_action(arm_actions) + (
            scaled_innovation * self.scid_residual_scale
        )
        should_clamp = self.scid_clamp_reconstructed_view if clamp is None else bool(clamp)
        if should_clamp:
            view_actions = view_actions.clamp(-1.0, 1.0)
        return view_actions

    def _prepare_head_trajectories(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        arm_actions, view_actions = super()._prepare_head_trajectories(actions)
        scaled_innovation = self.encode_view_innovation(arm_actions, view_actions)
        return arm_actions, scaled_innovation

    def combine_action_heads(self, arm_actions: Tensor, view_head_actions: Tensor) -> Tensor:
        view_actions = self.decode_view_innovation(arm_actions, view_head_actions)
        return torch.cat([arm_actions, view_actions], dim=-1)

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        losses = super().compute_loss(batch)
        # Keep view_loss for existing logging/checkpoint code while exposing the
        # semantically accurate name for SCID-specific diagnostics.
        losses["innovation_loss"] = losses["view_loss"]
        return losses
