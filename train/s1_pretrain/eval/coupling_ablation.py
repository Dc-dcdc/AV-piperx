"""Utilities for inference-only coupling ablations on a loaded policy."""

from __future__ import annotations

import math


COUPLING_SCALE_FIELDS = (
    "view_to_arm_coupling_scale",
    "arm_to_view_coupling_scale",
)


def _optional_scale(eval_cfg, field_name: str) -> float | None:
    value = getattr(eval_cfg, field_name, None)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name}必须是[0, 1]内的有限数，当前为{value}")
    return value


def coupling_ablation_tag(eval_cfg) -> str:
    """Return a filesystem-safe tag for explicitly requested scale overrides."""
    labels = []
    for short_name, field_name in (
        ("v2a", "view_to_arm_coupling_scale"),
        ("a2v", "arm_to_view_coupling_scale"),
    ):
        scale = _optional_scale(eval_cfg, field_name)
        if scale is not None:
            labels.append(f"{short_name}={scale:g}")
    return "_" + "_".join(labels) if labels else ""


def apply_coupling_ablation_overrides(policy, eval_cfg) -> dict[str, float]:
    """Apply optional inference-only scales after loading checkpoint weights.

    Checkpoint configs remain the source of truth when neither override is supplied.
    Supplying either field requires a coupled policy with ``set_coupling_scales``.
    """
    view_to_arm = _optional_scale(eval_cfg, "view_to_arm_coupling_scale")
    arm_to_view = _optional_scale(eval_cfg, "arm_to_view_coupling_scale")
    if view_to_arm is None and arm_to_view is None:
        return {}

    diffusion = getattr(policy, "diffusion", None)
    setter = getattr(diffusion, "set_coupling_scales", None)
    if not callable(setter):
        raise TypeError(
            "耦合缩放消融只适用于coupled_dual_head_diffusion checkpoint"
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
