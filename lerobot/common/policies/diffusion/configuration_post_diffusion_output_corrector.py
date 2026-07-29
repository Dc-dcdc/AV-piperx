#!/usr/bin/env python

"""Configuration for the frozen dual-head post-diffusion output corrector."""

from __future__ import annotations

import math
from dataclasses import dataclass

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig


@dataclass
class PostDiffusionOutputCorrectorConfig(DiffusionConfig):
    """Dual-head diffusion plus a bounded action-space residual corrector.

    The inherited fields describe the immutable dual-head diffusion baseline.
    The fields below describe only the small corrector that runs after both
    diffusion heads have completed all denoising steps.
    """

    output_corrector_type: str = "bipartite_attention"
    output_corrector_direction: str = "arm_to_view"
    output_corrector_d_model: int = 32
    output_corrector_num_heads: int = 4
    output_corrector_dropout: float = 0.0
    output_corrector_residual_limit: float = 0.1
    output_corrector_clamp_actions: bool = True
    view_to_arm_output_scale: float = 0.0
    arm_to_view_output_scale: float = 1.0
    output_residual_loss_weight: float = 1e-3
    output_smoothness_loss_weight: float = 1e-3

    def __post_init__(self) -> None:
        super().__post_init__()

        supported_types = {"linear", "bipartite_attention"}
        if self.output_corrector_type not in supported_types:
            raise ValueError(
                "`output_corrector_type` must be 'linear' or "
                f"'bipartite_attention'. Got {self.output_corrector_type!r}."
            )

        supported_directions = {
            "view_to_arm",
            "arm_to_view",
            "bidirectional",
        }
        if self.output_corrector_direction not in supported_directions:
            raise ValueError(
                "`output_corrector_direction` must be 'view_to_arm', "
                f"'arm_to_view', or 'bidirectional'. Got "
                f"{self.output_corrector_direction!r}."
            )

        if self.output_corrector_d_model <= 0:
            raise ValueError("`output_corrector_d_model` must be positive.")
        if self.output_corrector_num_heads <= 0:
            raise ValueError("`output_corrector_num_heads` must be positive.")
        if self.output_corrector_d_model % self.output_corrector_num_heads != 0:
            raise ValueError(
                "`output_corrector_num_heads` must divide "
                "`output_corrector_d_model` exactly."
            )
        if not 0.0 <= self.output_corrector_dropout < 1.0:
            raise ValueError("`output_corrector_dropout` must be in [0, 1).")
        if (
            not math.isfinite(self.output_corrector_residual_limit)
            or self.output_corrector_residual_limit <= 0
        ):
            raise ValueError(
                "`output_corrector_residual_limit` must be finite and positive."
            )
        if not isinstance(self.output_corrector_clamp_actions, bool):
            raise ValueError("`output_corrector_clamp_actions` must be a bool.")

        for field_name in (
            "view_to_arm_output_scale",
            "arm_to_view_output_scale",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"`{field_name}` must be finite and in [0, 1].")

        view_to_arm_enabled = self.view_to_arm_output_scale > 0
        arm_to_view_enabled = self.arm_to_view_output_scale > 0
        expected_enabled = {
            "view_to_arm": (True, False),
            "arm_to_view": (False, True),
            "bidirectional": (True, True),
        }[self.output_corrector_direction]
        if (view_to_arm_enabled, arm_to_view_enabled) != expected_enabled:
            raise ValueError(
                "Output direction and scales disagree: "
                f"direction={self.output_corrector_direction!r}, "
                f"view_to_arm_scale={self.view_to_arm_output_scale}, "
                f"arm_to_view_scale={self.arm_to_view_output_scale}."
            )

        for field_name in (
            "output_residual_loss_weight",
            "output_smoothness_loss_weight",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"`{field_name}` must be finite and non-negative.")
