from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

from mjlab.tasks.registry import register_mjlab_task

from env.mjlab.insert_cylinder_cfg import (
    TASK_ID,
    make_insert_cylinder_env_cfg,
    make_insert_cylinder_ppo_runner_cfg,
)


def register_tasks() -> None:
    try:
        register_mjlab_task(
            task_id=TASK_ID,
            env_cfg=make_insert_cylinder_env_cfg(),
            play_env_cfg=make_insert_cylinder_env_cfg(play=True),
            rl_cfg=make_insert_cylinder_ppo_runner_cfg(),
        )
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise


register_tasks()

__all__ = [
    "TASK_ID",
    "make_insert_cylinder_env_cfg",
    "make_insert_cylinder_ppo_runner_cfg",
    "register_tasks",
]
