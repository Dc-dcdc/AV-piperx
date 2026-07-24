"""Fit and initialize the fixed SCID action-coordinate transform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch import Tensor

from lerobot.common.policies.utils import get_device_from_parameters


@dataclass(frozen=True)
class SCIDTransformFit:
    """Closed-form SCID transform and held-on-training-data diagnostics."""

    matrix: Tensor
    bias: Tensor
    residual_scale: Tensor
    diagnostics: dict[str, float | int | list[float]]


def _leaf_datasets(dataset) -> Iterator:
    """Yield LeRobotDataset leaves from single or concatenated dataset wrappers."""
    children = getattr(dataset, "_datasets", None)
    if children is None:
        if not hasattr(dataset, "hf_dataset"):
            raise TypeError(
                "SCID fitting requires a LeRobotDataset-like object exposing hf_dataset "
                "or a concatenated wrapper exposing _datasets."
            )
        yield dataset
        return

    for child in children:
        yield from _leaf_datasets(child)


def iter_raw_action_batches(dataset, batch_size: int = 8192) -> Iterator[Tensor]:
    """Read raw single-frame actions without horizon padding or image decoding."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    for leaf in _leaf_datasets(dataset):
        if "action" not in leaf.hf_dataset.column_names:
            raise KeyError("Dataset used for SCID fitting does not contain an 'action' column.")
        action_dataset = leaf.hf_dataset.with_format(
            "torch",
            columns=["action"],
            output_all_columns=False,
        )
        for start in range(0, len(action_dataset), batch_size):
            actions = action_dataset[start : start + batch_size]["action"]
            actions = torch.as_tensor(actions, dtype=torch.float32)
            if actions.ndim != 2:
                raise ValueError(
                    "Raw SCID fitting actions must have shape [frames, action_dim], "
                    f"got {tuple(actions.shape)}."
                )
            yield actions


@torch.no_grad()
def fit_scid_transform(
    dataset,
    normalize_actions: Callable[[Tensor], Tensor],
    *,
    arm_action_dim: int,
    view_action_dim: int,
    ridge: float = 1e-3,
    residual_scale_eps: float = 1e-6,
    batch_size: int = 8192,
) -> SCIDTransformFit:
    """Fit ``V = G A + b + R`` in the policy's normalized action coordinates.

    Statistics are accumulated in float64 on CPU. Reading the raw action column
    directly avoids counting horizon padding repeatedly and avoids decoding any
    image/video observations.
    """
    if arm_action_dim <= 0 or view_action_dim <= 0:
        raise ValueError("arm_action_dim and view_action_dim must both be positive.")
    if not torch.isfinite(torch.tensor(ridge)) or ridge < 0:
        raise ValueError(f"ridge must be finite and non-negative, got {ridge}.")
    if not torch.isfinite(torch.tensor(residual_scale_eps)) or residual_scale_eps <= 0:
        raise ValueError(
            "residual_scale_eps must be finite and positive, "
            f"got {residual_scale_eps}."
        )

    action_dim = arm_action_dim + view_action_dim
    count = 0
    sum_arm = torch.zeros(arm_action_dim, dtype=torch.float64)
    sum_view = torch.zeros(view_action_dim, dtype=torch.float64)
    sum_arm_arm = torch.zeros(arm_action_dim, arm_action_dim, dtype=torch.float64)
    sum_view_arm = torch.zeros(view_action_dim, arm_action_dim, dtype=torch.float64)
    sum_view_view = torch.zeros(view_action_dim, view_action_dim, dtype=torch.float64)

    def normalized_batches() -> Iterator[Tensor]:
        for raw_actions in iter_raw_action_batches(dataset, batch_size=batch_size):
            normalized = normalize_actions(raw_actions)
            normalized = torch.as_tensor(normalized).detach().to(device="cpu", dtype=torch.float64)
            if normalized.shape != raw_actions.shape:
                raise ValueError(
                    "Action normalizer changed the action shape: "
                    f"raw={tuple(raw_actions.shape)}, normalized={tuple(normalized.shape)}."
                )
            if normalized.shape[-1] != action_dim:
                raise ValueError(
                    f"Expected action dim {action_dim}, got {normalized.shape[-1]}."
                )
            if not torch.isfinite(normalized).all():
                raise ValueError("Normalized actions contain NaN or Inf values.")
            yield normalized

    for actions in normalized_batches():
        arm = actions[:, :arm_action_dim]
        view = actions[:, arm_action_dim:]
        count += int(actions.shape[0])
        sum_arm += arm.sum(dim=0)
        sum_view += view.sum(dim=0)
        sum_arm_arm += arm.T @ arm
        sum_view_arm += view.T @ arm
        sum_view_view += view.T @ view

    if count < 2:
        raise ValueError(f"SCID fitting requires at least two action frames, got {count}.")

    mean_arm = sum_arm / count
    mean_view = sum_view / count
    cov_arm = sum_arm_arm / count - torch.outer(mean_arm, mean_arm)
    cov_view_arm = sum_view_arm / count - torch.outer(mean_view, mean_arm)
    cov_view = sum_view_view / count - torch.outer(mean_view, mean_view)
    regularized_cov_arm = cov_arm + float(ridge) * torch.eye(
        arm_action_dim,
        dtype=torch.float64,
    )

    try:
        matrix = torch.linalg.solve(regularized_cov_arm, cov_view_arm.T).T
    except RuntimeError as error:
        raise RuntimeError(
            "Failed to solve the SCID ridge system. Increase scid_ridge or inspect "
            "constant/invalid Arm action dimensions."
        ) from error
    bias = mean_view - matrix @ mean_arm

    residual_abs_max = torch.zeros(view_action_dim, dtype=torch.float64)
    residual_squared_sum = torch.zeros(view_action_dim, dtype=torch.float64)
    sum_residual = torch.zeros(view_action_dim, dtype=torch.float64)
    sum_arm_residual = torch.zeros(arm_action_dim, view_action_dim, dtype=torch.float64)
    for actions in normalized_batches():
        arm = actions[:, :arm_action_dim]
        view = actions[:, arm_action_dim:]
        residual = view - torch.nn.functional.linear(arm, matrix, bias)
        residual_abs_max = torch.maximum(residual_abs_max, residual.abs().amax(dim=0))
        residual_squared_sum += residual.square().sum(dim=0)
        sum_residual += residual.sum(dim=0)
        sum_arm_residual += arm.T @ residual

    residual_scale = residual_abs_max.clamp_min(float(residual_scale_eps))
    view_sst = torch.diag(cov_view) * count
    r2 = torch.where(
        view_sst > torch.finfo(torch.float64).eps,
        1.0 - residual_squared_sum / view_sst,
        torch.zeros_like(view_sst),
    )

    mean_residual = sum_residual / count
    cov_arm_residual = sum_arm_residual / count - torch.outer(mean_arm, mean_residual)
    arm_std = torch.diag(cov_arm).clamp_min(0).sqrt()
    view_std = torch.diag(cov_view).clamp_min(0).sqrt()
    residual_var = residual_squared_sum / count - mean_residual.square()
    residual_std = residual_var.clamp_min(0).sqrt()
    raw_denominator = torch.outer(view_std, arm_std).clamp_min(1e-12)
    residual_denominator = torch.outer(arm_std, residual_std).clamp_min(1e-12)
    raw_cross_corr_norm = torch.linalg.vector_norm(cov_view_arm / raw_denominator)
    residual_cross_corr_norm = torch.linalg.vector_norm(
        cov_arm_residual / residual_denominator
    )

    diagnostics: dict[str, float | int | list[float]] = {
        "num_frames": count,
        "ridge": float(ridge),
        "condition_number": float(torch.linalg.cond(regularized_cov_arm).item()),
        "view_r2_mean": float(r2.mean().item()),
        "view_r2_per_dim": [float(value) for value in r2.tolist()],
        "residual_scale": [float(value) for value in residual_scale.tolist()],
        "raw_cross_corr_norm": float(raw_cross_corr_norm.item()),
        "residual_cross_corr_norm": float(residual_cross_corr_norm.item()),
    }
    return SCIDTransformFit(
        matrix=matrix.float(),
        bias=bias.float(),
        residual_scale=residual_scale.float(),
        diagnostics=diagnostics,
    )


@torch.no_grad()
def initialize_scid_transform_from_dataset(
    policy,
    dataset,
    *,
    resume: bool,
    batch_size: int = 8192,
) -> SCIDTransformFit | None:
    """Fit an unfitted SCID policy, or strictly validate a resumed one."""
    diffusion = getattr(policy, "diffusion", None)
    if diffusion is None or not hasattr(diffusion, "is_scid_transform_fitted"):
        raise TypeError("initialize_scid_transform_from_dataset requires a SCID policy.")

    if diffusion.is_scid_transform_fitted:
        return None
    if resume:
        raise RuntimeError(
            "Cannot resume SCID from a checkpoint without a fitted transform. "
            "Use init_policy_path for a raw dual-head checkpoint instead."
        )

    policy_device = get_device_from_parameters(policy)

    def normalize_actions(raw_actions: Tensor) -> Tensor:
        normalized = policy.normalize_targets(
            {"action": raw_actions.to(policy_device)}
        )["action"]
        return normalized.cpu()

    fit = fit_scid_transform(
        dataset,
        normalize_actions,
        arm_action_dim=diffusion.arm_action_dim,
        view_action_dim=diffusion.view_action_dim,
        ridge=float(getattr(policy.config, "scid_ridge", 1e-3)),
        residual_scale_eps=float(getattr(policy.config, "scid_residual_eps", 1e-6)),
        batch_size=batch_size,
    )
    diffusion.set_scid_transform(
        fit.matrix,
        fit.bias,
        fit.residual_scale,
    )
    return fit

