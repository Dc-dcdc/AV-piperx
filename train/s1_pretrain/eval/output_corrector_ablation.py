"""Utilities for inference-only post-diffusion output-scale ablations."""

from __future__ import annotations

import math


OUTPUT_CORRECTOR_SCALE_FIELDS = (
    "view_to_arm_output_scale",
    "arm_to_view_output_scale",
)


def _optional_scale(eval_cfg, field_name: str) -> float | None:
    value = getattr(eval_cfg, field_name, None)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name}必须是[0, 1]内的有限数，当前为{value}")
    return value


def output_corrector_ablation_tag(eval_cfg) -> str:
    """Return a filesystem-safe tag for explicitly requested output scales."""
    labels = []
    for short_name, field_name in (
        ("out_v2a", "view_to_arm_output_scale"),
        ("out_a2v", "arm_to_view_output_scale"),
    ):
        scale = _optional_scale(eval_cfg, field_name)
        if scale is not None:
            labels.append(f"{short_name}={scale:g}")
    return "_" + "_".join(labels) if labels else ""


def apply_output_corrector_ablation_overrides(
    policy,
    eval_cfg,
) -> dict[str, float]:
    """Apply optional output scales after loading a post-diffusion policy.

    When neither override is supplied, the checkpoint configuration remains the
    source of truth. Supplying either field requires the loaded diffusion model
    to expose ``set_output_correction_scales``.
    """
    view_to_arm = _optional_scale(eval_cfg, "view_to_arm_output_scale")
    arm_to_view = _optional_scale(eval_cfg, "arm_to_view_output_scale")
    if view_to_arm is None and arm_to_view is None:
        return {}

    diffusion = getattr(policy, "diffusion", None)
    setter = getattr(diffusion, "set_output_correction_scales", None)
    if not callable(setter):
        raise TypeError(
            "输出修正缩放消融只适用于post_diffusion_output_corrector checkpoint"
        )

    active_scales = setter(
        view_to_arm=view_to_arm,
        arm_to_view=arm_to_view,
    )
    policy_config = getattr(policy, "config", None)
    if policy_config is not None:
        for field_name, value in active_scales.items():
            setattr(policy_config, field_name, value)
    return active_scales
