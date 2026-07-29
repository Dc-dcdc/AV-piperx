#!/usr/bin/env python

"""Configuration for a supervised none/Arm-to-View output router."""

from __future__ import annotations

import math
from dataclasses import dataclass

from lerobot.common.policies.diffusion.configuration_post_diffusion_output_corrector import (
    PostDiffusionOutputCorrectorConfig,
)


@dataclass
class RoutedPostDiffusionOutputCorrectorConfig(
    PostDiffusionOutputCorrectorConfig
):
    """Frozen post-diffusion corrector plus a trainable candidate router."""

    router_d_model: int = 64
    router_num_heads: int = 4
    router_num_layers: int = 2
    router_ffn_dim: int = 128
    router_dropout: float = 0.1
    router_threshold: float = 0.7
    router_mode: str = "router"

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.output_corrector_direction != "arm_to_view":
            raise ValueError(
                "none/A→V Router只支持"
                "`output_corrector_direction='arm_to_view'`。"
            )
        if (
            float(self.view_to_arm_output_scale) != 0.0
            or float(self.arm_to_view_output_scale) <= 0.0
        ):
            raise ValueError(
                "none/A→V Router要求View→Arm scale=0且Arm→View scale>0。"
            )
        if self.router_d_model <= 0:
            raise ValueError("`router_d_model`必须为正整数。")
        if self.router_num_heads <= 0:
            raise ValueError("`router_num_heads`必须为正整数。")
        if self.router_d_model % self.router_num_heads != 0:
            raise ValueError(
                "`router_num_heads`必须整除`router_d_model`。"
            )
        if self.router_num_layers <= 0:
            raise ValueError("`router_num_layers`必须为正整数。")
        if self.router_ffn_dim <= 0:
            raise ValueError("`router_ffn_dim`必须为正整数。")
        if not 0.0 <= float(self.router_dropout) < 1.0:
            raise ValueError("`router_dropout`必须位于[0, 1)。")
        if (
            not math.isfinite(float(self.router_threshold))
            or not 0.0 <= float(self.router_threshold) <= 1.0
        ):
            raise ValueError("`router_threshold`必须是[0, 1]内的有限数。")
        if self.router_mode not in {"router", "none", "arm_to_view"}:
            raise ValueError(
                "`router_mode`必须是'router'、'none'或'arm_to_view'。"
            )
