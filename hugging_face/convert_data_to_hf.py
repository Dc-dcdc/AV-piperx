#!/usr/bin/env python
"""将本地采集的 raw rollout 数据转换成本地 LeRobot/HF 数据集。

原始采集数据格式:

  metadata.json
  episodes/
    episode_000000/
      info.json
      arrays.npz
        joint_action          # 关节动作，可映射为 LeRobot 的 action
        pose_action           # 可选，末端位姿动作，也可映射为 LeRobot 的 action
        observation_state     # 机器人状态
      images/
        <camera_name>/
          000000.jpg
          000001.jpg
    episode_000000_aug_00/      # 可选：由源episode 000000生成的第0条增强轨迹

转换后的本地 LeRobot/HF 数据集格式:
  data/train-00000-of-00001.parquet         # 每一帧的 action、observation.state、timestamp、episode/frame/index 以及视频帧引用。
        observation.state: 机器人当前关节状态，维度由 raw 数据推断；当前 PiperX 三臂数据为 20 维
        action: 关节动作，维度由 raw 数据推断；当前 PiperX 三臂数据为 20 维，来自 raw 数据的 joint_action
        episode_index: 第几个 episode
        frame_index: episode 内第几帧
        timestamp: 时间戳，25 FPS，所以通常 0.00, 0.04, 0.08...
        next.done: 这一帧后 episode 是否结束
        index: 全局帧编号
        observation.images.<camera>: 不直接存图片，而是存 {path, timestamp}，指向对应 mp4 里的某一帧
  meta_data/info.json                       # 数据集基础信息，例如 fps、视频编码、相机键名和 episode 数量。
  meta_data/stats.safetensors               # 训练归一化统计量：action、observation.state、timestamp、index 等字段的 mean/std/min/max
  meta_data/episode_data_index.safetensors  # 每条 episode 在 parquet 全局帧序列中的起止索引。例如 episode 0 是 [0, 392)
        from: 每条 episode 起始行
        to:   每条 episode 结束行，不包含该行
  videos/observation.images.<camera>_episode_000000.mp4  # 每个相机对应的 episode 视频，parquet 通过 path + timestamp 引用具体帧。
                                                        # 视频会统一重编码为 LeRobot 短 GOP MP4，方便训练时随机读帧。

其中 raw 的 joint_action 或 pose_action 会根据 --action-key 映射到
LeRobot 数据集中的 action 字段。

"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import imageio.v2 as imageio
import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
# LeRobot is vendored as ``ROOT / "lerobot"``. Put the project root first so
# a stale editable installation of the former sibling checkout cannot win.
root_path = str(ROOT)
if root_path in sys.path:
    sys.path.remove(root_path)
sys.path.insert(0, root_path)

from lerobot.common.datasets.utils import calculate_episode_data_index, flatten_dict  # noqa: E402
from lerobot.common.datasets.video_utils import VideoFrame  # noqa: E402
from lerobot.common.datasets.video_utils import encode_video_frames  # noqa: E402
from lerobot.common.datasets.push_dataset_to_hub.utils import (  # noqa: E402
    get_default_encoding,
    save_images_concurrently,
)

DEFAULT_ENCODING = get_default_encoding()
_ENCODING_CHECKED = False
RAW_EPISODE_PATTERN = re.compile(
    r"^episode_(?P<source>\d{6,})(?:_aug_(?P<variant>\d{2,}))?$"
)


# 检查本机 ffmpeg 是否支持指定的视频编码器。
def ffmpeg_has_encoder(vcodec: str) -> bool:
    """Return True if the local ffmpeg build exposes the requested encoder."""
    if not vcodec or shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and vcodec in result.stdout


# 获取运行时视频编码配置，必要时回退到本机可用编码器。
def get_runtime_encoding() -> dict[str, Any]:
    """Use LeRobot's default encoding, with a local ffmpeg-compatible fallback."""
    global DEFAULT_ENCODING, _ENCODING_CHECKED

    if _ENCODING_CHECKED:
        return DEFAULT_ENCODING

    encoding = dict(DEFAULT_ENCODING)
    vcodec = str(encoding.get("vcodec", ""))
    if vcodec and not ffmpeg_has_encoder(vcodec):
        if ffmpeg_has_encoder("libx264"):
            encoding.update({"vcodec": "libx264", "pix_fmt": "yuv420p", "g": 2, "crf": 23})
            logging.warning(
                "当前 ffmpeg 不支持 LeRobot 默认视频编码器 %s，已自动改用 %s。",
                vcodec,
                encoding["vcodec"],
            )
        else:
            raise RuntimeError(
                f"当前 ffmpeg 不支持 LeRobot 默认视频编码器 {vcodec!r}，也未检测到 libx264。"
                "请安装带 libsvtav1 或 libx264 的 ffmpeg，或手动修改 DEFAULT_ENCODING。"
            )

    DEFAULT_ENCODING = encoding
    _ENCODING_CHECKED = True
    return DEFAULT_ENCODING


@dataclass
class EpisodeRecord:
    source_dir: Path
    source_name: str
    info: dict[str, Any]
    arrays: dict[str, np.ndarray]


# 初始化日志输出格式。
def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# 读取 JSON 文件。
def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 写入 JSON 文件。
def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=4, ensure_ascii=False)


# 解析原始采集数据目录。
def resolve_raw_dir(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    if (raw_dir / "metadata.json").exists() and (raw_dir / "episodes").exists():
        return raw_dir

    candidates = [
        path
        for pattern in ("collect_*", "quest_teleop_*")
        for path in raw_dir.glob(pattern)
        if (path / "metadata.json").exists() and (path / "episodes").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find a collect run under {raw_dir}. Expected metadata.json and episodes/."
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    logging.info("Using latest collect run: %s", candidates[0])
    return candidates[0]


# 列出可转换的 episode 目录。
def parse_raw_episode_name(name: str) -> tuple[int, int]:
    """返回(source_episode, variant)，原始轨迹的variant记为-1。"""
    match = RAW_EPISODE_PATTERN.fullmatch(str(name))
    if match is None:
        raise ValueError(
            "episode目录必须形如episode_000003或"
            f"episode_000003_aug_00，当前为{name!r}。"
        )
    variant = match.group("variant")
    return int(match.group("source")), (-1 if variant is None else int(variant))


def list_episode_dirs(raw_dir: Path) -> list[Path]:
    episodes_dir = raw_dir / "episodes"
    episode_dirs = [
        path
        for path in episodes_dir.glob("episode_*")
        if path.is_dir() and not path.name.endswith(".tmp") and (path / "arrays.npz").exists()
    ]
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* folders with arrays.npz found in {episodes_dir}.")
    identities: dict[tuple[int, int], Path] = {}
    for path in episode_dirs:
        identity = parse_raw_episode_name(path.name)
        if identity in identities:
            raise RuntimeError(
                "发现重复的source/variant episode目录: "
                f"{identities[identity]}, {path}"
            )
        identities[identity] = path
    return [
        path
        for _, path in sorted(
            identities.items(),
            key=lambda item: (item[0][0], item[0][1] + 1),
        )
    ]


# 加载单条 episode 的 info.json 和 arrays.npz。
def load_episode(episode_dir: Path) -> EpisodeRecord:
    info = read_json(episode_dir / "info.json")
    with np.load(episode_dir / "arrays.npz", allow_pickle=False) as npz:
        arrays = {key: np.asarray(npz[key]) for key in npz.files}
    return EpisodeRecord(
        source_dir=episode_dir,
        source_name=episode_dir.name,
        info=info,
        arrays=arrays,
    )


# 选择本次需要转换的 episode。
def select_episodes(
    raw_dir: Path,
    max_episodes: int | None,
) -> list[EpisodeRecord]:
    records = []
    for episode_dir in list_episode_dirs(raw_dir):
        record = load_episode(episode_dir)
        records.append(record)
        if max_episodes is not None and len(records) >= max_episodes:
            break

    if not records:
        raise RuntimeError("No episodes selected. Check --max-episodes.")
    return records


# 从 arrays.npz 中读取机器人状态观测数组。
def infer_state_array(arrays: dict[str, np.ndarray]) -> np.ndarray:
    preferred = ("observation_state", "obs__agent_pos", "obs__observation__state")
    for key in preferred:
        if key in arrays:
            return np.asarray(arrays[key], dtype=np.float32)

    obs_keys = [key for key in arrays if key.startswith("obs__")]
    if obs_keys:
        logging.warning("Using fallback state key %s.", obs_keys[0])
        return np.asarray(arrays[obs_keys[0]], dtype=np.float32)

    raise KeyError("Cannot find observation state in arrays.npz.")


# 从 arrays.npz 中读取指定动作数组。
def infer_action_array(arrays: dict[str, np.ndarray], action_key: str) -> np.ndarray:
    if action_key in arrays:
        return np.asarray(arrays[action_key], dtype=np.float32)
    fallback_keys = ("joint_action", "action", "pose_action")
    for key in fallback_keys:
        if key in arrays:
            logging.warning("Action key %s not found; using fallback key %s.", action_key, key)
            return np.asarray(arrays[key], dtype=np.float32)
    raise KeyError(f"Cannot find action key {action_key!r} in arrays.npz.")


# 推断需要写入数据集的相机名称列表。
def infer_cameras(records: list[EpisodeRecord], explicit_cameras: str | None) -> list[str]:
    if explicit_cameras:
        return [item.strip() for item in explicit_cameras.split(",") if item.strip()]

    cameras = set()
    for record in records:
        image_dirs = record.info.get("image_observation_dirs", {})
        for key, rel_path in image_dirs.items():
            if key.startswith("pixels."):
                cameras.add(key.split(".", 1)[1])
            else:
                cameras.add(Path(rel_path).name)

        videos_dir = record.source_dir / "videos"
        if videos_dir.exists():
            cameras.update(path.stem for path in videos_dir.glob("*.mp4"))

    if not cameras:
        raise RuntimeError("No cameras found in episode info or videos/ folders.")
    return sorted(cameras)


# 计算单条 episode 中状态、动作和附加字段共同可用的帧数。
def frame_count_for(record: EpisodeRecord, action_key: str) -> int:
    arrays = record.arrays
    state = infer_state_array(arrays)
    action = infer_action_array(arrays, action_key)
    frame_count = min(len(state), len(action))
    for optional in ("timestamp", "terminated", "truncated", "frame_index"):
        if optional in arrays:
            frame_count = min(frame_count, len(arrays[optional]))
    if frame_count <= 0:
        raise RuntimeError(f"{record.source_dir} has no frames.")
    return int(frame_count)


# 将已有 raw MP4 直接转码为 LeRobot 训练友好的短 GOP MP4，避免中间 PNG 帧带来的大量 I/O。
def transcode_video_to_lerobot_encoding(
    src_video: Path,
    dst_video: Path,
    fps: int,
    encoding: dict[str, Any],
    overwrite: bool,
) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    vcodec = str(encoding.get("vcodec", "libx264"))
    pix_fmt = str(encoding.get("pix_fmt", "yuv420p"))
    g = encoding.get("g", 2)
    crf = encoding.get("crf", 23)

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(src_video),
        "-an",
        "-vf",
        f"fps={int(fps)}",
        "-vcodec",
        vcodec,
        "-pix_fmt",
        pix_fmt,
    ]
    if vcodec == "libx264":
        ffmpeg_cmd += ["-preset", "veryfast"]
    if g is not None:
        ffmpeg_cmd += ["-g", str(g)]
    if crf is not None:
        ffmpeg_cmd += ["-crf", str(crf)]
    ffmpeg_cmd.append(str(dst_video))

    subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL)
    if not dst_video.exists():
        raise OSError(f"Video transcoding did not create output file: {dst_video}")


# 将相机图片帧或已有 MP4 统一重编码为 LeRobot 训练友好的短 GOP MP4。
def copy_or_encode_video(
    record: EpisodeRecord,
    camera: str,
    episode_index: int,
    videos_dir: Path,
    fps: int,
    overwrite: bool,
) -> str:
    video_name = f"observation.images.{camera}_episode_{episode_index:06d}.mp4"
    dst = videos_dir / video_name
    if dst.exists() and not overwrite:
        return f"videos/{video_name}"

    images_dir = record.source_dir / "images" / camera
    frame_paths = sorted(images_dir.glob("*.jpg"))
    if frame_paths:
        # 对齐 LeRobot 官方 push_dataset_to_hub 流程：
        # 先生成 frame_%06d.png，再用 encode_video_frames 写入带短 GOP 的训练友好视频。
        frames = np.stack([imageio.imread(frame_path) for frame_path in frame_paths], axis=0)
        encoding = get_runtime_encoding()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_imgs_dir = Path(tmp_dir) / "frames"
            save_images_concurrently(frames, tmp_imgs_dir)
            encode_video_frames(
                tmp_imgs_dir,
                dst,
                fps,
                overwrite=True,
                **encoding,
            )
        return f"videos/{video_name}"

    src = record.source_dir / "videos" / f"{camera}.mp4"
    if src.exists():
        # 不直接复制采集阶段的 mp4。普通 mp4 的关键帧间隔可能较长，
        # 训练时随机取帧会拖慢 PyAV 解码；这里统一转码为 g=2 的短 GOP 视频。
        encoding = get_runtime_encoding()
        logging.info(
            "Transcoding %s camera=%s episode=%06d encoding=%s",
            src,
            camera,
            episode_index,
            encoding,
        )
        transcode_video_to_lerobot_encoding(
            src_video=src,
            dst_video=dst,
            fps=fps,
            encoding=encoding,
            overwrite=True,
        )
        return f"videos/{video_name}"

    raise FileNotFoundError(f"Missing both {src} and image frames under {images_dir}.")


# 创建 Hugging Face Dataset 的字段定义。
def make_hf_features(cameras: list[str], state_dim: int, action_dim: int) -> datasets.Features:
    features: dict[str, Any] = {}
    for camera in cameras:
        features[f"observation.images.{camera}"] = VideoFrame()
    features["observation.state"] = datasets.Sequence(
        datasets.Value("float32"), length=state_dim
    )
    features["action"] = datasets.Sequence(datasets.Value("float32"), length=action_dim)
    features["episode_index"] = datasets.Value("int64")
    features["frame_index"] = datasets.Value("int64")
    features["timestamp"] = datasets.Value("float32")
    features["next.done"] = datasets.Value("bool")
    features["index"] = datasets.Value("int64")
    return datasets.Features(features)


# 生成 parquet 行数据，并为每条 episode 准备对应视频文件。
def build_rows_and_videos(
    records: list[EpisodeRecord],
    cameras: list[str],
    videos_dir: Path,
    fps: int,
    overwrite: bool,
    action_key: str,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    global_index = 0

    videos_dir.mkdir(parents=True, exist_ok=True)
    for episode_index, record in enumerate(records):
        frame_count = frame_count_for(record, action_key)
        state = infer_state_array(record.arrays)[:frame_count].astype(np.float32, copy=False)
        action = infer_action_array(record.arrays, action_key)[:frame_count].astype(np.float32, copy=False)
        timestamps = record.arrays.get("timestamp")
        if timestamps is None:
            timestamps = np.arange(frame_count, dtype=np.float32) / float(fps)
        timestamps = np.asarray(timestamps[:frame_count], dtype=np.float32)

        terminated = np.asarray(record.arrays.get("terminated", np.zeros(frame_count)), dtype=bool)
        truncated = np.asarray(record.arrays.get("truncated", np.zeros(frame_count)), dtype=bool)
        done_flags = np.logical_or(terminated[:frame_count], truncated[:frame_count])
        done_flags[-1] = True

        video_paths = {
            camera: copy_or_encode_video(
                record=record,
                camera=camera,
                episode_index=episode_index,
                videos_dir=videos_dir,
                fps=fps,
                overwrite=overwrite,
            )
            for camera in cameras
        }

        all_states.append(state)
        all_actions.append(action)
        for frame_index in range(frame_count):
            timestamp = float(timestamps[frame_index])
            row: dict[str, Any] = {
                "observation.state": state[frame_index].tolist(),
                "action": action[frame_index].tolist(),
                "episode_index": int(episode_index),
                "frame_index": int(frame_index),
                "timestamp": timestamp,
                "next.done": bool(done_flags[frame_index]),
                "index": int(global_index),
            }
            for camera in cameras:
                row[f"observation.images.{camera}"] = {
                    "path": video_paths[camera],
                    "timestamp": timestamp,
                }
            rows.append(row)
            global_index += 1

        logging.info(
            "Converted %s -> episode_index=%06d frames=%d success=%s reward=%.2f",
            record.source_name,
            episode_index,
            frame_count,
            bool(record.info.get("success", False)),
            float(record.info.get("reward", 0.0)),
        )

    return rows, all_states, all_actions


# 计算向量字段的均值、方差、最小值和最大值。
def vector_stats(values: np.ndarray) -> dict[str, torch.Tensor]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "mean": torch.from_numpy(array.mean(axis=0).astype(np.float32)),
        "std": torch.from_numpy(np.maximum(array.std(axis=0), 1e-6).astype(np.float32)),
        "min": torch.from_numpy(array.min(axis=0).astype(np.float32)),
        "max": torch.from_numpy(array.max(axis=0).astype(np.float32)),
    }


# 将标量列表转换为二维列向量。
def scalar_column(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


# 收集指定相机在所有 episode 中的图片帧路径。
def find_image_frames(records: list[EpisodeRecord], camera: str) -> list[Path]:
    frames: list[Path] = []
    for record in records:
        frames.extend(sorted((record.source_dir / "images" / camera).glob("*.jpg")))
    return frames


# 计算指定相机图像的通道统计量。
def image_stats(
    records: list[EpisodeRecord],
    camera: str,
    max_frames: int,
) -> dict[str, torch.Tensor]:
    frame_paths = find_image_frames(records, camera)
    if not frame_paths or max_frames == 0:
        mean = torch.full((3, 1, 1), 0.5, dtype=torch.float32)
        std = torch.full((3, 1, 1), 0.25, dtype=torch.float32)
        return {
            "mean": mean,
            "std": std,
            "min": torch.zeros((3, 1, 1), dtype=torch.float32),
            "max": torch.ones((3, 1, 1), dtype=torch.float32),
        }

    if max_frames > 0 and len(frame_paths) > max_frames:
        indices = np.linspace(0, len(frame_paths) - 1, num=max_frames, dtype=np.int64)
        frame_paths = [frame_paths[int(idx)] for idx in indices]

    count = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sumsq = np.zeros(3, dtype=np.float64)
    channel_min = np.full(3, np.inf, dtype=np.float64)
    channel_max = np.full(3, -np.inf, dtype=np.float64)

    for frame_path in frame_paths:
        image = np.asarray(imageio.imread(frame_path), dtype=np.float32) / 255.0
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        image = image[..., :3]
        flat = image.reshape(-1, 3)
        count += flat.shape[0]
        channel_sum += flat.sum(axis=0)
        channel_sumsq += np.square(flat).sum(axis=0)
        channel_min = np.minimum(channel_min, flat.min(axis=0))
        channel_max = np.maximum(channel_max, flat.max(axis=0))

    mean = channel_sum / max(1, count)
    var = np.maximum(channel_sumsq / max(1, count) - np.square(mean), 1e-12)
    std = np.sqrt(var)
    reshape = lambda value: torch.from_numpy(value.astype(np.float32)).view(3, 1, 1)
    return {
        "mean": reshape(mean),
        "std": reshape(std),
        "min": reshape(channel_min),
        "max": reshape(channel_max),
    }


# 汇总 LeRobot 数据集需要的全部统计信息。
def build_stats(
    rows: list[dict[str, Any]],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    records: list[EpisodeRecord],
    cameras: list[str],
    max_image_stat_frames: int,
) -> dict[str, dict[str, torch.Tensor]]:
    state_values = np.concatenate(states, axis=0)
    action_values = np.concatenate(actions, axis=0)
    stats: dict[str, dict[str, torch.Tensor]] = {
        "observation.state": vector_stats(state_values),
        "action": vector_stats(action_values),
        "episode_index": vector_stats(scalar_column([row["episode_index"] for row in rows])),
        "frame_index": vector_stats(scalar_column([row["frame_index"] for row in rows])),
        "timestamp": vector_stats(scalar_column([row["timestamp"] for row in rows])),
        "next.done": vector_stats(scalar_column([float(row["next.done"]) for row in rows])),
        "index": vector_stats(scalar_column([row["index"] for row in rows])),
    }
    for camera in cameras:
        stats[f"observation.images.{camera}"] = image_stats(
            records=records,
            camera=camera,
            max_frames=max_image_stat_frames,
        )
    return stats


# 推断数据集帧率，优先使用手动覆盖值。
def infer_fps(records: list[EpisodeRecord], metadata: dict[str, Any], override_fps: int | None) -> int:
    if override_fps is not None:
        return int(override_fps)
    if "fps" in metadata:
        return int(metadata["fps"])
    for record in records:
        if "fps" in record.info:
            return int(record.info["fps"])
    return 25


# 构造 LeRobot 数据集的 info.json 内容。
def build_info(fps: int, raw_dir: Path, records: list[EpisodeRecord], cameras: list[str]) -> dict[str, Any]:
    return {
        "codebase_version": "v1.6",
        "fps": int(fps),
        "video": 1,
        "encoding": DEFAULT_ENCODING,
        "source_raw_dir": str(raw_dir),
        "total_episodes": int(len(records)),
        "camera_keys": [f"observation.images.{camera}" for camera in cameras],
    }


# 写入数据集 README 说明文件。
def write_dataset_card(local_dir: Path, raw_dir: Path) -> None:
    dataset_name = local_dir.name
    text = (
        "---\n"
        "task_categories:\n"
        "- robotics\n"
        "tags:\n"
        "- LeRobot\n"
        "---\n"
        f"This dataset was converted from `{raw_dir}` for LeRobot training.\n"
        f"Dataset name: `{dataset_name}`.\n"
    )
    (local_dir / "README.md").write_text(text, encoding="utf-8")


# 写入 Git LFS 追踪规则。
def write_gitattributes(local_dir: Path) -> None:
    content = (
        "*.mp4 filter=lfs diff=lfs merge=lfs -text\n"
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
    )
    (local_dir / ".gitattributes").write_text(content, encoding="utf-8")


# 从 raw 数据构建本地 LeRobot/HF 数据集目录。
def build_local_dataset(args: argparse.Namespace) -> Path:
    raw_dir = resolve_raw_dir(args.raw_dir)
    local_dir = Path(args.output_dir)
    if local_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{local_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(local_dir)

    data_dir = local_dir / "data"
    meta_data_dir = local_dir / "meta_data"
    videos_dir = local_dir / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(raw_dir / "metadata.json") if (raw_dir / "metadata.json").exists() else {}
    records = select_episodes(
        raw_dir=raw_dir,
        max_episodes=args.max_episodes,
    )
    cameras = infer_cameras(records, args.cameras)
    fps = infer_fps(records, metadata, args.fps)
    logging.info("Selected %d episode(s), cameras=%s, fps=%d.", len(records), cameras, fps)

    rows, states, actions = build_rows_and_videos(
        records=records,
        cameras=cameras,
        videos_dir=videos_dir,
        fps=fps,
        overwrite=args.overwrite,
        action_key=args.action_key,
    )

    state_dim = int(states[0].shape[-1])
    action_dim = int(actions[0].shape[-1])
    # 将数据转换为 Hugging Face Dataset 格式并保存为 parquet 文件；视频文件已在 build_rows_and_videos 中处理好了。
    hf_dataset = datasets.Dataset.from_list(
        rows,
        features=make_hf_features(cameras=cameras, state_dim=state_dim, action_dim=action_dim),
    )
    parquet_path = data_dir / "train-00000-of-00001.parquet"
    hf_dataset.to_parquet(str(parquet_path))

    episode_data_index = calculate_episode_data_index(hf_dataset)
    save_file(episode_data_index, meta_data_dir / "episode_data_index.safetensors")
    stats = build_stats(
        rows=rows,
        states=states,
        actions=actions,
        records=records,
        cameras=cameras,
        max_image_stat_frames=args.max_image_stat_frames,
    )
    save_file(flatten_dict(stats), meta_data_dir / "stats.safetensors")

    info = build_info(fps=fps, raw_dir=raw_dir, records=records, cameras=cameras)
    write_json(meta_data_dir / "info.json", info)
    write_dataset_card(local_dir=local_dir, raw_dir=raw_dir)
    write_gitattributes(local_dir)

    logging.info("Wrote parquet: %s", parquet_path)
    logging.info("Wrote metadata: %s", meta_data_dir)
    logging.info("Wrote videos: %s", videos_dir)
    return local_dir


# 创建命令行参数解析器。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将本地采集数据转换为 LeRobot/Hugging Face 本地数据集格式。"
    )

    parser.add_argument(
        "--raw-dir",
        default="outputs/4_data_collect/quest_teleop/",
        help="采集数据目录；也可以填父目录，脚本会自动选择最新的 collect_* 或 quest_teleop_* 目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/5_hf_datasets/",
        help="本地生成的 HF 数据集目录。",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="本地输出目录已存在时是否覆盖重建；可用 --no-overwrite 关闭。",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多转换多少条 episode；None 表示转换全部 episode。",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="指定要写入数据集的相机，用逗号分隔；None 表示使用采集数据中保存的全部相机。",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="覆盖数据集帧率；None 表示读取采集时记录的 fps。",
    )
    parser.add_argument(
        "--max-image-stat-frames",
        type=int,
        default=500,
        help="每个相机最多抽样多少帧计算图像统计；设为 0 则使用占位统计，速度更快。",
    )
    parser.add_argument(
        "--action-key",
        default="joint_action",
        choices=("joint_action", "action", "pose_action"),
        help="raw arrays.npz 中映射到 LeRobot action 字段的动作键。",
    )
    parser.add_argument(
        "--http-proxy",
        default="",
        help="HTTP 代理地址；留空表示不修改当前环境变量。",
    )
    parser.add_argument(
        "--https-proxy",
        default="",
        help="HTTPS 代理地址；留空表示不修改当前环境变量。",
    )
    return parser


# 应用运行时代理等环境变量。
def apply_runtime_env(args: argparse.Namespace) -> None:
    if args.http_proxy:
        import os

        os.environ["http_proxy"] = args.http_proxy
    if args.https_proxy:
        import os

        os.environ["https_proxy"] = args.https_proxy


# 按参数执行完整转换流程。
def run_from_args(args: argparse.Namespace) -> None:
    apply_runtime_env(args)
    local_dir = build_local_dataset(args)
    logging.info("Local dataset is ready: %s", local_dir)


# 使用 Python 变量调用转换流程。
def convert_data_folder_to_hf(
    raw_dir: str,
    output_dir: str = "outputs/5_hf_datasets/quest_teleop_insert_cylinder_3arms",
    overwrite: bool = True,
    max_episodes: int | None = None,
    cameras: str | None = None,
    fps: int | None = None,
    max_image_stat_frames: int = 500,
    action_key: str = "joint_action",
    http_proxy: str = "",
    https_proxy: str = "",
) -> None:
    """Use explicit Python variables instead of command line arguments."""
    init_logging()
    args = argparse.Namespace(
        raw_dir=raw_dir,
        output_dir=output_dir,
        overwrite=overwrite,
        max_episodes=max_episodes,
        cameras=cameras,
        fps=fps,
        max_image_stat_frames=max_image_stat_frames,
        action_key=action_key,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
    )
    run_from_args(args)


# 命令行入口函数。
def main() -> None:
    init_logging()
    parser = build_arg_parser()
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit
    
    # raw arrays.npz 中映射到 LeRobot action 字段的动作键：joint_action 或 pose_action。
    ACTION_KEY = "joint_action"

    # 存放原始采集数据的目录
    RAW_DIR = "outputs/4_data_collect/quest_teleop_recovery/SewNeedle-3Arms/quest_teleop_SewNeedle-3Arms-v0_rgb_DART_early_success"

    # 本地 HF 数据集生成目录。
    OUTPUT_DIR = "outputs/5_hf_datasets/SewNeedle-3Arms/expert_100/quest_teleop_SewNeedle-3Arms-v0_rgb_DART_early_success"

    # 本地 HF数据目录 已存在时是否覆盖重建。
    OVERWRITE = True

    # 最多转换多少条 episode；None 表示全部转换。
    MAX_EPISODES = None

    # None 表示默认转换全部相机，方便同一份 HF 数据集复用于不同训练配置。
    CAMERAS = "zed_cam_left,zed_cam_right"  # "zed_cam_left,zed_cam_right,wrist_cam_left,wrist_cam_right,overhead_cam,worms_eye_cam"

    convert_data_folder_to_hf(
        raw_dir=RAW_DIR,
        output_dir=OUTPUT_DIR,
        overwrite=OVERWRITE,
        max_episodes=MAX_EPISODES,
        cameras=CAMERAS,
        action_key=ACTION_KEY,
    )
