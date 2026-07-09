from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat


def _select_one(value: torch.Tensor, ids) -> torch.Tensor:
    selected = value[:, ids]
    if selected.ndim == value.ndim - 1:
        return selected
    return selected.reshape(value.shape[0], -1, value.shape[-1])[:, 0]


def compute_task_metrics(
    env,
    robot_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    cylinder_half_height: float = 0.06,
    target_z: float = 0.04,
) -> dict[str, torch.Tensor]:
    robot: Entity = env.scene[robot_cfg.name]
    cylinder: Entity = env.scene[cylinder_cfg.name]
    container: Entity = env.scene[container_cfg.name]

    left_gripper_pos = _select_one(robot.data.site_pos_w, robot_cfg.site_ids[:1])
    right_gripper_pos = _select_one(robot.data.site_pos_w, robot_cfg.site_ids[1:2])
    cylinder_pos = _select_one(cylinder.data.geom_pos_w, cylinder_cfg.geom_ids)
    cylinder_quat = _select_one(cylinder.data.geom_quat_w, cylinder_cfg.geom_ids)
    target_pos = _select_one(container.data.geom_pos_w, container_cfg.geom_ids).clone()
    target_pos[:, 2] = float(target_z)

    cylinder_axis = matrix_from_quat(cylinder_quat)[..., :, 2]
    cylinder_bottom_pos = cylinder_pos - cylinder_axis * float(cylinder_half_height)

    left_dist = torch.linalg.norm(left_gripper_pos - cylinder_pos, dim=-1)
    right_dist = torch.linalg.norm(right_gripper_pos - cylinder_pos, dim=-1)
    bottom_xy_error = torch.linalg.norm(cylinder_bottom_pos[:, :2] - target_pos[:, :2], dim=-1)
    center_xy_error = torch.linalg.norm(cylinder_pos[:, :2] - target_pos[:, :2], dim=-1)
    z_error = torch.abs(cylinder_pos[:, 2] - target_pos[:, 2])
    upright_cos = torch.abs(cylinder_axis[:, 2]).clamp(0.0, 1.0)

    return {
        "left_gripper_pos": left_gripper_pos,
        "right_gripper_pos": right_gripper_pos,
        "cylinder_pos": cylinder_pos,
        "cylinder_bottom_pos": cylinder_bottom_pos,
        "target_pos": target_pos,
        "left_to_cylinder": cylinder_pos - left_gripper_pos,
        "right_to_cylinder": cylinder_pos - right_gripper_pos,
        "cylinder_to_target": target_pos - cylinder_pos,
        "bottom_to_target": target_pos - cylinder_bottom_pos,
        "left_dist": left_dist,
        "right_dist": right_dist,
        "bottom_xy_error": bottom_xy_error,
        "center_xy_error": center_xy_error,
        "z_error": z_error,
        "upright_cos": upright_cos,
    }


def task_state(
    env,
    robot_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    cylinder_half_height: float = 0.06,
    target_z: float = 0.04,
) -> torch.Tensor:
    metrics = compute_task_metrics(
        env,
        robot_cfg=robot_cfg,
        cylinder_cfg=cylinder_cfg,
        container_cfg=container_cfg,
        cylinder_half_height=cylinder_half_height,
        target_z=target_z,
    )
    return torch.cat(
        [
            metrics["left_to_cylinder"],
            metrics["right_to_cylinder"],
            metrics["cylinder_to_target"],
            metrics["bottom_to_target"],
            metrics["cylinder_pos"],
            metrics["target_pos"],
            metrics["upright_cos"].unsqueeze(-1),
        ],
        dim=-1,
    )
