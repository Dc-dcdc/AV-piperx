#!/usr/bin/env python3
"""查看指定遥操轨迹的累计奖励，并绘制奖励曲线。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# 将命令行字符串转换为 bool。
def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


# 解析输入路径，支持 run 目录、episode 目录、arrays.npz 或 reward_debug.jsonl。
def resolve_episode_dir(path: Path, episode: int | None) -> Path:
    path = path.expanduser()

    if path.is_file():
        if path.name not in ("arrays.npz", "reward_debug.jsonl"):
            raise ValueError(f"Expected arrays.npz or reward_debug.jsonl file, got: {path}")
        return path.parent

    if (path / "reward_debug.jsonl").exists() or (path / "arrays.npz").exists():
        return path

    episodes_dir = path / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(
            f"Cannot resolve trajectory from {path}. "
            "Expected reward_debug.jsonl, arrays.npz, episode_xxxxxx/, or a run directory containing episodes/."
        )

    if episode is not None:
        episode_dir = episodes_dir / f"episode_{episode:06d}"
        if not episode_dir.exists():
            raise FileNotFoundError(f"Episode {episode:06d} not found: {episode_dir}")
        return episode_dir

    candidates = sorted(
        path
        for path in episodes_dir.glob("episode_*")
        if path.is_dir() and ((path / "reward_debug.jsonl").exists() or (path / "arrays.npz").exists())
    )
    if not candidates:
        raise FileNotFoundError(f"No episode reward/debug data found under {episodes_dir}.")
    return candidates[-1]


# 从 reward_debug.jsonl 中读取累计奖励曲线。
def load_cumulative_reward_from_jsonl(reward_debug_path: Path) -> np.ndarray:
    cumulative_reward = []
    step_reward = []
    with reward_debug_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "cumulative_reward" in item:
                cumulative_reward.append(float(item["cumulative_reward"]))
            elif "reward" in item:
                step_reward.append(float(item["reward"]))
            else:
                raise ValueError(f"{reward_debug_path}:{line_no} 缺少 reward 或 cumulative_reward 字段。")

    if cumulative_reward:
        return np.asarray(cumulative_reward, dtype=np.float64)
    if step_reward:
        return np.cumsum(np.asarray(step_reward, dtype=np.float64))
    raise ValueError(f"Reward debug file is empty: {reward_debug_path}")


# 从 arrays.npz 中读取累计奖励曲线，兼容旧轨迹。
def load_cumulative_reward_from_arrays(arrays_path: Path, reward_key: str) -> np.ndarray:
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if reward_key not in arrays.files:
            keys = ", ".join(arrays.files)
            raise ValueError(
                "当前轨迹没有保存奖励数据，无法绘制奖励曲线。\n"
                f"轨迹文件: {arrays_path}\n"
                f"缺少字段: {reward_key}\n"
                f"当前已有字段: {keys}\n"
                "请重新采集该轨迹，并在 configs/data_collect/quest_teleop_collect.yaml 中设置:\n"
                "  save_reward_debug: true"
            )
        reward = np.asarray(arrays[reward_key], dtype=np.float64).reshape(-1)

    if reward.size == 0:
        raise ValueError(f"Reward array {reward_key!r} is empty in {arrays_path}.")
    return reward


# 优先从独立 reward_debug.jsonl 读取，旧轨迹则回退到 arrays.npz。
def load_cumulative_reward(episode_dir: Path, reward_key: str) -> tuple[np.ndarray, Path]:
    reward_debug_path = episode_dir / "reward_debug.jsonl"
    if reward_debug_path.exists():
        return load_cumulative_reward_from_jsonl(reward_debug_path), reward_debug_path

    arrays_path = episode_dir / "arrays.npz"
    if arrays_path.exists():
        return load_cumulative_reward_from_arrays(arrays_path, reward_key), arrays_path

    raise FileNotFoundError(
        "当前轨迹没有 reward_debug.jsonl 或 arrays.npz，无法绘制奖励曲线。\n"
        f"轨迹目录: {episode_dir}\n"
        "请确认采集成功保存，并设置 save_reward_debug: true。"
    )


# 根据累计奖励反推出每一步即时奖励。
def cumulative_to_step_reward(cumulative_reward: np.ndarray) -> np.ndarray:
    return np.diff(cumulative_reward, prepend=0.0)


# 绘制累计奖励和每步奖励曲线。
def plot_reward_curve(
    cumulative_reward: np.ndarray,
    source_path: Path,
    output_path: Path | None,
    show: bool,
) -> Path | None:
    if output_path is None and not show:
        return None

    if not show:
        import matplotlib

        matplotlib.use("Agg")
    elif not os.environ.get("DISPLAY"):
        import matplotlib

        matplotlib.use("Agg")
        show = False

    import matplotlib.pyplot as plt

    step_reward = cumulative_to_step_reward(cumulative_reward)
    steps = np.arange(cumulative_reward.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(str(source_path.parent))

    axes[0].plot(steps, cumulative_reward, color="tab:blue", linewidth=2)
    axes[0].set_ylabel("Cumulative reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, step_reward, color="tab:orange", linewidth=1.5)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Step reward")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path is not None:
        output_path = output_path.expanduser()
        if output_path.is_dir():
            output_path = output_path / "reward_curve.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


# 创建命令行参数。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取指定 episode 的 reward_debug.jsonl 或旧版 cumulative_reward，并绘制奖励曲线。"
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="episode 目录、reward_debug.jsonl、arrays.npz 文件，或包含 episodes/ 的 run 目录。",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="当 path 是 run 目录时，指定 episode 编号，例如 0 对应 episode_000000。",
    )
    parser.add_argument(
        "--reward-key",
        default="cumulative_reward",
        help="旧版 arrays.npz 中累计奖励字段名，默认 cumulative_reward。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="曲线图片保存路径。默认保存到 episode 目录下的 reward_curve.png。",
    )
    parser.add_argument(
        "--save-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存奖励曲线图片；可用 --no-save-plot 关闭。",
    )
    parser.add_argument(
        "--show",
        default="true",
        help="是否弹出 matplotlib 窗口显示曲线；无 DISPLAY 时会自动只保存图片。",
    )
    return parser


# 按参数执行奖励读取和绘图。
def run_from_args(args: argparse.Namespace) -> None:
    if args.path is None:
        raise ValueError("请设置轨迹路径 path，或直接在文件底部修改 EPISODE_PATH。")

    episode_dir = resolve_episode_dir(Path(args.path), args.episode)
    cumulative_reward, source_path = load_cumulative_reward(episode_dir, args.reward_key)
    step_reward = cumulative_to_step_reward(cumulative_reward)

    show_plot = str_to_bool(args.show)
    output_path = args.output
    if not args.save_plot:
        output_path = None
    elif output_path is None:
        output_path = episode_dir / "reward_curve.png"

    saved_path = plot_reward_curve(
        cumulative_reward=cumulative_reward,
        source_path=source_path,
        output_path=output_path,
        show=show_plot,
    )

    best_step = int(np.argmax(cumulative_reward))
    worst_step_reward = int(np.argmin(step_reward))
    best_step_reward = int(np.argmax(step_reward))
    print(f"source: {source_path}")
    print(f"steps: {cumulative_reward.shape[0]}")
    print(f"final cumulative reward: {cumulative_reward[-1]:.4f}")
    print(f"max cumulative reward: {cumulative_reward[best_step]:.4f} at step {best_step}")
    print(f"min step reward: {step_reward[worst_step_reward]:.4f} at step {worst_step_reward}")
    print(f"max step reward: {step_reward[best_step_reward]:.4f} at step {best_step_reward}")
    if saved_path is not None:
        print(f"saved plot: {saved_path}")


# 使用 Python 变量调用奖励曲线绘制。
def plot_episode_reward(
    path: str,
    episode: int | None = None,
    reward_key: str = "cumulative_reward",
    output: str | None = None,
    save_plot: bool = True,
    show: bool = True,
) -> None:
    args = argparse.Namespace(
        path=Path(path),
        episode=episode,
        reward_key=reward_key,
        output=Path(output) if output is not None else None,
        save_plot=save_plot,
        show=show,
    )
    run_from_args(args)


# 命令行入口。
def main() -> None:
    run_from_args(build_arg_parser().parse_args())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            main()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"错误：{exc}", file=sys.stderr)
            sys.exit(1)
        raise SystemExit

    # 默认轨迹路径：直接改这里即可查看对应 episode 的奖励曲线。
    EPISODE_PATH = "outputs/4_data_collect/quest_teleop/quest_teleop_InsertCylinder-3Arms-v0_rgb/episodes/episode_000000"

    # 当 EPISODE_PATH 是 run 目录时，指定 episode 编号；普通 episode 目录填 None。
    EPISODE = None

    # arrays.npz 中累计奖励字段名。
    REWARD_KEY = "cumulative_reward"

    # 是否保存奖励曲线图片；False 时只打印统计信息。
    SAVE_PLOT = True

    # 是否弹出 matplotlib 窗口；无 DISPLAY 时会自动只保存图片。
    SHOW = True

    # 图片保存路径；None 表示保存到 episode 目录下的 reward_curve.png。
    OUTPUT = "outputs/reward"

    try:
        plot_episode_reward(
            path=EPISODE_PATH,
            episode=EPISODE,
            reward_key=REWARD_KEY,
            output=OUTPUT,
            save_plot=SAVE_PLOT,
            show=SHOW,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
