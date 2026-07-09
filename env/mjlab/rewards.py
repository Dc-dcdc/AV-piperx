from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from env.mjlab.observations import compute_task_metrics


def _metrics(env, robot_cfg: SceneEntityCfg, cylinder_cfg: SceneEntityCfg, container_cfg: SceneEntityCfg):
    return compute_task_metrics(
        env,
        robot_cfg=robot_cfg,
        cylinder_cfg=cylinder_cfg,
        container_cfg=container_cfg,
    )


def right_gripper_reach(
    env,
    robot_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    std: float = 0.12,
) -> torch.Tensor:
    metrics = _metrics(env, robot_cfg, cylinder_cfg, container_cfg)
    return torch.exp(-(metrics["right_dist"] ** 2) / (float(std) ** 2))


def left_gripper_reach(
    env,
    robot_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    std: float = 0.12,
) -> torch.Tensor:
    metrics = _metrics(env, robot_cfg, cylinder_cfg, container_cfg)
    return torch.exp(-(metrics["left_dist"] ** 2) / (float(std) ** 2))


def cylinder_to_target(
    env,
    robot_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    xy_std: float = 0.06,
    z_std: float = 0.05,
) -> torch.Tensor:
    metrics = _metrics(env, robot_cfg, cylinder_cfg, container_cfg)
    xy_reward = torch.exp(-(metrics["bottom_xy_error"] ** 2) / (float(xy_std) ** 2))
    z_reward = torch.exp(-(metrics["z_error"] ** 2) / (float(z_std) ** 2))
    return xy_reward * z_reward * metrics["upright_cos"]


def upright_bonus(
    env,
    robot_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
) -> torch.Tensor:
    metrics = _metrics(env, robot_cfg, cylinder_cfg, container_cfg)
    return metrics["upright_cos"]


def success_bonus(
    env,
    robot_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    xy_threshold: float = 0.012,
    upright_threshold: float = 0.9,
) -> torch.Tensor:
    metrics = _metrics(env, robot_cfg, cylinder_cfg, container_cfg)
    success = (
        (metrics["bottom_xy_error"] < float(xy_threshold))
        & (metrics["upright_cos"] > float(upright_threshold))
    )
    return success.float()
