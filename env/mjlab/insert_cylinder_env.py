from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv

from env.mjlab.insert_cylinder_cfg import make_insert_cylinder_env_cfg


def make_env(
    *,
    num_envs: int = 128,
    device: str = "cuda:0",
    render_mode: str | None = None,
    play: bool = False,
) -> ManagerBasedRlEnv:
    return ManagerBasedRlEnv(
        cfg=make_insert_cylinder_env_cfg(num_envs=num_envs, play=play),
        device=device,
        render_mode=render_mode,
    )
