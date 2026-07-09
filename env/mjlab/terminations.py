from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from env.mjlab.observations import compute_task_metrics


def task_success(
    env,
    robot_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    xy_threshold: float = 0.012,
    upright_threshold: float = 0.9,
) -> torch.Tensor:
    metrics = compute_task_metrics(
        env,
        robot_cfg=robot_cfg,
        cylinder_cfg=cylinder_cfg,
        container_cfg=container_cfg,
    )
    return (
        (metrics["bottom_xy_error"] < float(xy_threshold))
        & (metrics["upright_cos"] > float(upright_threshold))
    )


def cylinder_dropped(
    env,
    cylinder_cfg: SceneEntityCfg,
    min_z: float = -0.03,
) -> torch.Tensor:
    cylinder = env.scene[cylinder_cfg.name]
    cylinder_pos = cylinder.data.geom_pos_w[:, cylinder_cfg.geom_ids].reshape(env.num_envs, -1, 3)[:, 0]
    return cylinder_pos[:, 2] < float(min_z)
