from __future__ import annotations

import argparse
import os

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

import env.mjlab  # noqa: F401
from env.mjlab.insert_cylinder_cfg import TASK_ID
from mjlab.scripts.play import PlayConfig, run_play


def main() -> None:
    parser = argparse.ArgumentParser(description="Play/evaluate PiperX mjlab policy.")
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--checkpoint-file", default="logs/rsl_rl/piperx_insert_cylinder_mjlab/2026-07-07_14-40-40_ppo_state/model_2000.pt")
    parser.add_argument("--agent", choices=["zero", "random", "trained"], default="zero")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=int, default=200)
    parser.add_argument("--viewer", choices=["auto", "native", "viser"], default="native")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = "trained" if args.checkpoint_file and args.agent == "zero" else args.agent
    cfg = PlayConfig(
        agent=agent,
        checkpoint_file=args.checkpoint_file,
        num_envs=args.num_envs,
        device=device,
        video=args.video,
        video_length=args.video_length,
        viewer=args.viewer,
    )
    run_play(args.task_id, cfg)


if __name__ == "__main__":
    main()
