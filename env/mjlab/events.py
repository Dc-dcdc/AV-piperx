from __future__ import annotations

import torch

from mjlab.envs.mdp.events import resolve_env_ids


def _sample_range(
    low: float,
    high: float,
    shape: tuple[int, ...],
    device: str,
) -> torch.Tensor:
    if low == high:
        return torch.full(shape, float(low), device=device)
    return torch.empty(shape, device=device).uniform_(float(low), float(high))


def reset_insert_cylinder_task(
    env,
    env_ids: torch.Tensor | None,
    cylinder_name: str = "insert_cylinder",
    container_name: str = "cylinder_container",
    cylinder_x_range: tuple[float, float] = (0.045, 0.045),
    cylinder_y_range: tuple[float, float] = (0.05, 0.25),
    cylinder_z_range: tuple[float, float] = (0.01, 0.01),
    container_x_range: tuple[float, float] = (-0.045, -0.045),
    container_y_range: tuple[float, float] = (0.05, 0.25),
    container_z_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    n_envs = len(env_ids)
    origins = env.scene.env_origins[env_ids]

    cylinder = env.scene[cylinder_name]
    root_state = cylinder.data.default_root_state[env_ids].clone()
    root_state[:, 0] = _sample_range(*cylinder_x_range, (n_envs,), env.device) + origins[:, 0]
    root_state[:, 1] = _sample_range(*cylinder_y_range, (n_envs,), env.device) + origins[:, 1]
    root_state[:, 2] = _sample_range(*cylinder_z_range, (n_envs,), env.device) + origins[:, 2]
    root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device)
    root_state[:, 7:13] = 0.0
    cylinder.write_root_state_to_sim(root_state, env_ids=env_ids)

    container = env.scene[container_name]
    container_pose = container.data.default_root_state[env_ids, :7].clone()
    container_pose[:, 0] = _sample_range(*container_x_range, (n_envs,), env.device) + origins[:, 0]
    container_pose[:, 1] = _sample_range(*container_y_range, (n_envs,), env.device) + origins[:, 1]
    container_pose[:, 2] = _sample_range(*container_z_range, (n_envs,), env.device) + origins[:, 2]
    container_pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device)
    container.write_mocap_pose_to_sim(container_pose, env_ids=env_ids)
