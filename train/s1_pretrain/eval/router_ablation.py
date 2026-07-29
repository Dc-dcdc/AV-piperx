"""Inference-only mode overrides for supervised output Router checkpoints."""

from __future__ import annotations


ROUTER_MODES = {"router", "none", "arm_to_view"}


def _optional_router_mode(eval_cfg) -> str | None:
    value = getattr(eval_cfg, "router_mode", None)
    if value is None:
        return None
    value = str(value)
    if value not in ROUTER_MODES:
        raise ValueError(
            "router_mode必须是'router'、'none'或'arm_to_view'，"
            f"当前为{value!r}。"
        )
    return value


def router_ablation_tag(eval_cfg) -> str:
    mode = _optional_router_mode(eval_cfg)
    return f"_router={mode}" if mode is not None else ""


def apply_router_ablation_override(policy, eval_cfg) -> dict[str, object]:
    mode = _optional_router_mode(eval_cfg)
    threshold = getattr(eval_cfg, "router_threshold", None)
    if mode is None and threshold is None:
        return {}
    diffusion = getattr(policy, "diffusion", None)
    mode_setter = getattr(diffusion, "set_router_mode", None)
    threshold_setter = getattr(diffusion, "set_router_threshold", None)
    if not callable(mode_setter) or not callable(threshold_setter):
        raise TypeError(
            "Router消融只适用于routed_post_diffusion_output_corrector checkpoint。"
        )
    if mode is not None:
        mode_setter(mode)
    if threshold is not None:
        threshold_setter(float(threshold))
    return {
        "router_mode": diffusion.router_mode,
        "router_threshold": float(diffusion.router_threshold),
    }
