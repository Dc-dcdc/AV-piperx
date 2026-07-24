#!/usr/bin/env python
"""将策略运行中的人在环轨迹转换为本地 LeRobot/HF 数据集。

输入目录应来自 ``data_collect/quest_policy_collect.py``，每条 episode 的
``arrays.npz`` 中需要包含至少：

  joint_action          # 最终执行到 env.step 的完整动作
  observation_state     # 执行动作前的观测状态
  teleop_applied        # 当前 step 是否有人类接管/平滑接管影响
  teleop_mask           # 哪些机械臂处在人类接管目标模式
  blend_weight          # 左/右/中间臂的人类动作混合权重
  policy_action         # 接管前策略原本输出的动作
  teleop_action         # 可选，平滑混合前的原始遥操作候选动作

输出数据集仍保持标准 LeRobot 字段：

  observation.state
  observation.images.<camera>
  action                # 默认来自 raw 的 joint_action，即最终执行动作
  episode_index
  frame_index
  timestamp
  next.done
  index

同时额外保留干预分析字段：

  policy_action
  teleop_action         # 旧数据缺失时回退为 action
  teleop_action_available
  action_delta          # action - policy_action，方便做 residual/correction 学习
  teleop_applied
  teleop_mask
  blend_weight
  control_mode
  is_intervention
  intervention_action_mask    # 展开到完整 action 维度的人类控制掩码
  intervention_action_weight  # mask * blend_weight，供 Diffusion loss 使用
  source_episode_index
  source_episode_name
  source_frame_index
  source_timestamp

``intervention_only`` 模式保持旧行为：每段接管成为一个 dataset episode。
``full_rollout`` 模式则完整保留每条原始 episode，供 IWR 从同一 rollout 中分别
采样人类干预数据 D_I 和机器人自主数据 D_R。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
from safetensors.torch import save_file


CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
for path in (CURRENT_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import convert_data_to_hf as base  # noqa: E402
from lerobot.common.datasets.utils import calculate_episode_data_index, flatten_dict  # noqa: E402
from lerobot.common.datasets.video_utils import VideoFrame  # noqa: E402


@dataclass(frozen=True)
class InterventionSegment:
    """源轨迹中的一个连续切片，对应输出数据集中的一条 episode。"""

    record_index: int
    source_start: int
    source_stop: int


DATASET_MODE_INTERVENTION_ONLY = "intervention_only"
DATASET_MODE_FULL_ROLLOUT = "full_rollout"
DATASET_MODES = (DATASET_MODE_INTERVENTION_ONLY, DATASET_MODE_FULL_ROLLOUT)


def safe_array(
    arrays: dict[str, np.ndarray],
    key: str,
    *,
    frame_count: int,
    default: Any,
    dtype,
) -> np.ndarray:
    """读取数组并截断到共同帧数；缺失时用默认值补齐。"""

    if key in arrays:
        return np.asarray(arrays[key], dtype=dtype)[:frame_count]

    value = np.asarray(default, dtype=dtype)
    if value.ndim == 0:
        return np.full((frame_count,), value, dtype=dtype)
    return np.repeat(value[None], frame_count, axis=0).astype(dtype, copy=False)


def resolve_raw_dir(raw_dir: str | Path) -> Path:
    """解析 raw 采集目录；支持 quest_policy_* run 自动发现。"""

    raw_dir = Path(raw_dir)
    if (raw_dir / "metadata.json").exists() and (raw_dir / "episodes").exists():
        return raw_dir

    candidates = [
        path
        for pattern in ("collect_*", "quest_teleop_*", "quest_policy_*")
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


def control_mode_policy_code(record: base.EpisodeRecord) -> int:
    """从 info.json 里的 control_mode_map 推断 policy 模式编码。"""

    mode_map = record.info.get("control_mode_map", {})
    for code, mode in mode_map.items():
        if str(mode) == "policy":
            return int(code)
    return 0


def intervention_mask_for(
    record: base.EpisodeRecord,
    frame_count: int,
    *,
    filter_mode: str,
    min_blend_weight: float,
) -> np.ndarray:
    """根据配置生成“哪些帧属于人类干预”的布尔掩码。"""

    arrays = record.arrays
    teleop_applied = safe_array(
        arrays,
        "teleop_applied",
        frame_count=frame_count,
        default=False,
        dtype=np.bool_,
    )
    teleop_mask = safe_array(
        arrays,
        "teleop_mask",
        frame_count=frame_count,
        default=np.zeros(3, dtype=np.bool_),
        dtype=np.bool_,
    )
    blend_weight = safe_array(
        arrays,
        "blend_weight",
        frame_count=frame_count,
        default=np.zeros(3, dtype=np.float32),
        dtype=np.float32,
    )
    control_mode = safe_array(
        arrays,
        "control_mode",
        frame_count=frame_count,
        default=control_mode_policy_code(record),
        dtype=np.int64,
    )

    if filter_mode == "saved_segments":
        # 直接采用采集阶段保存的连续片段索引，适合需要严格复现当时切分结果的场景。
        # 该文件缺失时回退到 teleop_applied，保证旧数据仍可转换。
        index_path = record.source_dir / "intervention_segments.json"
        if not index_path.exists():
            logging.warning("%s 不存在，将退回 teleop_applied 筛选。", index_path)
            mask = teleop_applied
        else:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            mask = np.zeros(frame_count, dtype=np.bool_)
            for item in index.get("segments", []):
                start = int(item.get("start_frame", 0))
                stop = int(item.get("stop_frame", start))
                start = max(0, min(frame_count, start))
                stop = max(start, min(frame_count, stop))
                mask[start:stop] = True
    elif filter_mode == "teleop_applied":
        # 使用采集端的综合判定：当前有人类目标动作，或平滑混合权重仍未归零。
        # 因此通常同时覆盖接管渐入、完全接管和交还策略时的渐出帧。
        mask = teleop_applied
    elif filter_mode == "teleop_mask":
        # 只看左/右/中间臂是否处于“人类接管目标模式”。筛选更严格，交还
        # 策略后的渐出帧不会被选中，但刚切入且 blend_weight=0 的帧可能被选中。
        mask = np.any(teleop_mask, axis=1)
    elif filter_mode == "blend_weight":
        # 只要任一机械臂的实际人类混合权重达到阈值就选中。它反映最终执行动作
        # 是否确实受人工分量影响，并同时包含接管渐入与交还渐出阶段。
        mask = np.max(blend_weight, axis=1) >= float(min_blend_weight)
    elif filter_mode == "control_mode":
        # 根据离散控制模式判断；只要不是纯 policy 模式就视为干预。它表达操作员
        # 的接管意图，不保证该帧混合权重已经大于零。
        mask = control_mode != control_mode_policy_code(record)
    elif filter_mode == "teleop_or_blend":
        # teleop_applied 与实际 blend_weight 的并集，适合兼容不同版本采集数据中
        # 某个字段记录不完整的情况；通常是覆盖范围最宽、最保守的判定方式。
        mask = teleop_applied | (np.max(blend_weight, axis=1) >= float(min_blend_weight))
    else:
        raise ValueError(f"Unsupported filter_mode: {filter_mode}")

    return np.asarray(mask, dtype=np.bool_)


def true_runs(mask: np.ndarray, *, merge_gap_frames: int) -> list[tuple[int, int]]:
    """把 True 掩码转换为 [start, stop) 片段，可合并短暂断开的 gap。"""

    indices = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
    if indices.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        gap = value - prev - 1
        if gap <= int(merge_gap_frames):
            prev = value
            continue
        runs.append((start, prev + 1))
        start = value
        prev = value
    runs.append((start, prev + 1))
    return runs


def extend_and_merge_segments(
    runs: list[tuple[int, int]],
    *,
    frame_count: int,
    pre_context_frames: int,
    post_context_frames: int,
) -> list[tuple[int, int]]:
    """给干预片段添加上下文帧，并合并重叠片段。"""

    if not runs:
        return []

    extended = [
        (
            max(0, start - int(pre_context_frames)),
            min(frame_count, stop + int(post_context_frames)),
        )
        for start, stop in runs
    ]
    extended.sort()

    merged: list[tuple[int, int]] = []
    cur_start, cur_stop = extended[0]
    for start, stop in extended[1:]:
        if start <= cur_stop:
            cur_stop = max(cur_stop, stop)
        else:
            merged.append((cur_start, cur_stop))
            cur_start, cur_stop = start, stop
    merged.append((cur_start, cur_stop))
    return merged


def select_intervention_segments(
    records: list[base.EpisodeRecord],
    *,
    action_key: str,
    filter_mode: str,
    min_blend_weight: float,
    merge_gap_frames: int,
    pre_context_frames: int,
    post_context_frames: int,
    min_segment_frames: int,
) -> list[InterventionSegment]:
    """从所有 episode 中找出人类接管片段。"""

    segments: list[InterventionSegment] = []
    for record_index, record in enumerate(records):
        frame_count = base.frame_count_for(record, action_key)
        mask = intervention_mask_for(
            record,
            frame_count,
            filter_mode=filter_mode,
            min_blend_weight=min_blend_weight,
        )
        runs = true_runs(mask, merge_gap_frames=merge_gap_frames)
        runs = extend_and_merge_segments(
            runs,
            frame_count=frame_count,
            pre_context_frames=pre_context_frames,
            post_context_frames=post_context_frames,
        )
        for start, stop in runs:
            if stop - start < int(min_segment_frames):
                continue
            segments.append(
                InterventionSegment(
                    record_index=record_index,
                    source_start=int(start),
                    source_stop=int(stop),
                )
            )
    return segments


def select_full_rollouts(
    records: list[base.EpisodeRecord],
    *,
    action_key: str,
) -> list[InterventionSegment]:
    """为每条源 episode 创建一个覆盖全部帧的切片。"""

    return [
        InterventionSegment(
            record_index=record_index,
            source_start=0,
            source_stop=base.frame_count_for(record, action_key),
        )
        for record_index, record in enumerate(records)
        if base.frame_count_for(record, action_key) > 0
    ]


def expand_arm_values_to_action(values: np.ndarray, action_dim: int) -> np.ndarray:
    """把左/右/中间臂的逐臂值展开为逐 action 维度值。

    当前三臂环境的 20 维动作布局固定为左臂 7、右臂 7、中间视角臂 6。
    对已经是逐 action 维度的数据直接原样返回；其他布局显式报错，避免静默
    给错误的 action 维度施加人类监督。
    """

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Expected [frames, dims] arm values, got shape={values.shape}.")
    if values.shape[1] == int(action_dim):
        return values
    if values.shape[1] == 3 and int(action_dim) == 20:
        return np.repeat(values, repeats=(7, 7, 6), axis=1)
    raise ValueError(
        "无法把逐臂干预字段展开到完整 action："
        f"arm_dim={values.shape[1]}, action_dim={action_dim}。"
        "当前仅支持三臂 20 维布局 [7, 7, 6]。"
    )


def infer_source_episode_index(record: base.EpisodeRecord, fallback: int) -> int:
    """读取原始 episode 编号；缺失时用 records 中的顺序。"""

    for key in ("episode", "episode_index"):
        if key in record.info:
            try:
                return int(record.info[key])
            except (TypeError, ValueError):
                pass
    return int(fallback)


def source_timestamps_for(record: base.EpisodeRecord, frame_count: int, fps: int) -> np.ndarray:
    """原始 episode 视频中的时间戳，用于从原视频解码对应帧。"""

    if "timestamp" in record.arrays:
        return np.asarray(record.arrays["timestamp"][:frame_count], dtype=np.float32)
    return np.arange(frame_count, dtype=np.float32) / float(fps)


def make_hf_features(
    cameras: list[str],
    *,
    state_dim: int,
    action_dim: int,
    teleop_mask_dim: int,
    blend_weight_dim: int,
) -> datasets.Features:
    """创建带干预字段的 Hugging Face Dataset features。"""

    features: dict[str, Any] = {}
    for camera in cameras:
        features[f"observation.images.{camera}"] = VideoFrame()
    features["observation.state"] = datasets.Sequence(datasets.Value("float32"), length=state_dim)
    features["action"] = datasets.Sequence(datasets.Value("float32"), length=action_dim)
    features["policy_action"] = datasets.Sequence(datasets.Value("float32"), length=action_dim)
    features["teleop_action"] = datasets.Sequence(datasets.Value("float32"), length=action_dim)
    features["teleop_action_available"] = datasets.Value("bool")
    features["action_delta"] = datasets.Sequence(datasets.Value("float32"), length=action_dim)
    features["teleop_applied"] = datasets.Value("bool")
    features["teleop_mask"] = datasets.Sequence(datasets.Value("bool"), length=teleop_mask_dim)
    features["blend_weight"] = datasets.Sequence(datasets.Value("float32"), length=blend_weight_dim)
    features["control_mode"] = datasets.Value("int64")
    features["is_intervention"] = datasets.Value("bool")
    features["intervention_action_mask"] = datasets.Sequence(
        datasets.Value("bool"), length=action_dim
    )
    features["intervention_action_weight"] = datasets.Sequence(
        datasets.Value("float32"), length=action_dim
    )
    features["source_episode_index"] = datasets.Value("int64")
    features["source_episode_name"] = datasets.Value("string")
    features["source_frame_index"] = datasets.Value("int64")
    features["source_timestamp"] = datasets.Value("float32")
    features["episode_index"] = datasets.Value("int64")
    features["frame_index"] = datasets.Value("int64")
    features["timestamp"] = datasets.Value("float32")
    features["next.done"] = datasets.Value("bool")
    features["index"] = datasets.Value("int64")
    return datasets.Features(features)


def video_path_for_source_episode(
    *,
    record: base.EpisodeRecord,
    camera: str,
    source_video_index: int,
    videos_dir: Path,
    fps: int,
    overwrite: bool,
    cache: dict[tuple[int, str], str],
) -> str:
    """复用原始 episode 视频；同一源 episode 的多个干预段共享同一视频文件。"""

    cache_key = (int(source_video_index), camera)
    if cache_key not in cache:
        cache[cache_key] = base.copy_or_encode_video(
            record=record,
            camera=camera,
            episode_index=int(source_video_index),
            videos_dir=videos_dir,
            fps=fps,
            overwrite=overwrite,
        )
    return cache[cache_key]


def build_rows_and_videos(
    records: list[base.EpisodeRecord],
    segments: list[InterventionSegment],
    cameras: list[str],
    videos_dir: Path,
    fps: int,
    overwrite: bool,
    action_key: str,
    filter_mode: str,
    min_blend_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]]]:
    """生成 parquet 行，并准备每个源 episode 对应的视频文件。"""

    rows: list[dict[str, Any]] = []
    values: dict[str, list[np.ndarray]] = {
        "state": [],
        "action": [],
        "policy_action": [],
        "teleop_action": [],
        "action_delta": [],
        "blend_weight": [],
        "teleop_mask": [],
        "intervention_action_mask": [],
        "intervention_action_weight": [],
    }
    global_index = 0
    video_cache: dict[tuple[int, str], str] = {}

    videos_dir.mkdir(parents=True, exist_ok=True)
    for segment_index, segment in enumerate(segments):
        record = records[segment.record_index]
        frame_count = base.frame_count_for(record, action_key)
        state = base.infer_state_array(record.arrays)[:frame_count].astype(np.float32, copy=False)
        action = base.infer_action_array(record.arrays, action_key)[:frame_count].astype(np.float32, copy=False)

        if "policy_action" in record.arrays:
            policy_action = np.asarray(record.arrays["policy_action"][:frame_count], dtype=np.float32)
            if policy_action.shape[-1] != action.shape[-1]:
                logging.warning(
                    "%s policy_action dim=%s != action dim=%s；将用 action 代替 policy_action。",
                    record.source_dir,
                    policy_action.shape[-1],
                    action.shape[-1],
                )
                policy_action = action.copy()
        else:
            logging.warning("%s 缺少 policy_action；将用 action 代替，action_delta 会为 0。", record.source_dir)
            policy_action = action.copy()

        teleop_action_available = "teleop_action" in record.arrays
        if teleop_action_available:
            teleop_action = np.asarray(record.arrays["teleop_action"][:frame_count], dtype=np.float32)
            if teleop_action.shape != action.shape:
                logging.warning(
                    "%s teleop_action shape=%s != action shape=%s；将回退到最终执行动作。",
                    record.source_dir,
                    teleop_action.shape,
                    action.shape,
                )
                teleop_action = action.copy()
                teleop_action_available = False
        else:
            # 旧采集数据没有保存平滑混合前的候选动作。保留一个等形状回退值，
            # 同时用 available=False 防止训练端将它误判为纯人工动作。
            teleop_action = action.copy()

        teleop_applied = safe_array(
            record.arrays,
            "teleop_applied",
            frame_count=frame_count,
            default=False,
            dtype=np.bool_,
        )
        teleop_mask = safe_array(
            record.arrays,
            "teleop_mask",
            frame_count=frame_count,
            default=np.zeros(3, dtype=np.bool_),
            dtype=np.bool_,
        )
        blend_weight = safe_array(
            record.arrays,
            "blend_weight",
            frame_count=frame_count,
            default=np.zeros(3, dtype=np.float32),
            dtype=np.float32,
        )
        intervention_action_mask = expand_arm_values_to_action(
            teleop_mask.astype(np.bool_, copy=False),
            action.shape[-1],
        ).astype(np.bool_, copy=False)
        intervention_action_weight = expand_arm_values_to_action(
            blend_weight.astype(np.float32, copy=False),
            action.shape[-1],
        ).astype(np.float32, copy=False)
        intervention_action_weight *= intervention_action_mask.astype(np.float32, copy=False)
        control_mode = safe_array(
            record.arrays,
            "control_mode",
            frame_count=frame_count,
            default=control_mode_policy_code(record),
            dtype=np.int64,
        )
        actual_intervention_mask = intervention_mask_for(
            record,
            frame_count,
            filter_mode=filter_mode,
            min_blend_weight=min_blend_weight,
        )
        source_timestamps = source_timestamps_for(record, frame_count, fps)
        source_episode_index = infer_source_episode_index(record, segment.record_index)

        video_paths = {
            camera: video_path_for_source_episode(
                record=record,
                camera=camera,
                source_video_index=segment.record_index,
                videos_dir=videos_dir,
                fps=fps,
                overwrite=overwrite,
                cache=video_cache,
            )
            for camera in cameras
        }

        segment_length = segment.source_stop - segment.source_start
        for local_frame_index, source_frame_index in enumerate(
            range(segment.source_start, segment.source_stop)
        ):
            local_timestamp = float(local_frame_index) / float(fps)
            source_timestamp = float(source_timestamps[source_frame_index])
            action_item = action[source_frame_index].astype(np.float32, copy=False)
            policy_action_item = policy_action[source_frame_index].astype(np.float32, copy=False)
            teleop_action_item = teleop_action[source_frame_index].astype(np.float32, copy=False)
            action_delta = (action_item - policy_action_item).astype(np.float32, copy=False)

            row: dict[str, Any] = {
                "observation.state": state[source_frame_index].astype(np.float32, copy=False).tolist(),
                "action": action_item.tolist(),
                "policy_action": policy_action_item.tolist(),
                "teleop_action": teleop_action_item.tolist(),
                "teleop_action_available": bool(teleop_action_available),
                "action_delta": action_delta.tolist(),
                "teleop_applied": bool(teleop_applied[source_frame_index]),
                "teleop_mask": teleop_mask[source_frame_index].astype(bool, copy=False).tolist(),
                "blend_weight": blend_weight[source_frame_index].astype(np.float32, copy=False).tolist(),
                "control_mode": int(control_mode[source_frame_index]),
                "is_intervention": bool(actual_intervention_mask[source_frame_index]),
                "intervention_action_mask": intervention_action_mask[source_frame_index].tolist(),
                "intervention_action_weight": intervention_action_weight[source_frame_index].tolist(),
                "source_episode_index": int(source_episode_index),
                "source_episode_name": record.source_name,
                "source_frame_index": int(source_frame_index),
                "source_timestamp": source_timestamp,
                "episode_index": int(segment_index),
                "frame_index": int(local_frame_index),
                "timestamp": local_timestamp,
                "next.done": bool(local_frame_index == segment_length - 1),
                "index": int(global_index),
            }
            for camera in cameras:
                row[f"observation.images.{camera}"] = {
                    "path": video_paths[camera],
                    "timestamp": source_timestamp,
                }
            rows.append(row)
            values["state"].append(state[source_frame_index])
            values["action"].append(action_item)
            values["policy_action"].append(policy_action_item)
            values["teleop_action"].append(teleop_action_item)
            values["action_delta"].append(action_delta)
            values["blend_weight"].append(blend_weight[source_frame_index])
            values["teleop_mask"].append(teleop_mask[source_frame_index].astype(np.float32))
            values["intervention_action_mask"].append(
                intervention_action_mask[source_frame_index].astype(np.float32)
            )
            values["intervention_action_weight"].append(
                intervention_action_weight[source_frame_index]
            )
            global_index += 1

        logging.info(
            "Converted %s[%d:%d] -> dataset_episode=%06d frames=%d.",
            record.source_name,
            segment.source_start,
            segment.source_stop,
            segment_index,
            segment_length,
        )

    return rows, values


def scalar_column(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def build_intervention_stats(
    rows: list[dict[str, Any]],
    values: dict[str, list[np.ndarray]],
    records: list[base.EpisodeRecord],
    cameras: list[str],
    max_image_stat_frames: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """汇总 LeRobot 需要的统计信息；额外字段也给出统计，便于检查。"""

    stats: dict[str, dict[str, torch.Tensor]] = {
        "observation.state": base.vector_stats(np.asarray(values["state"], dtype=np.float32)),
        "action": base.vector_stats(np.asarray(values["action"], dtype=np.float32)),
        "policy_action": base.vector_stats(np.asarray(values["policy_action"], dtype=np.float32)),
        "teleop_action": base.vector_stats(np.asarray(values["teleop_action"], dtype=np.float32)),
        "action_delta": base.vector_stats(np.asarray(values["action_delta"], dtype=np.float32)),
        "blend_weight": base.vector_stats(np.asarray(values["blend_weight"], dtype=np.float32)),
        "teleop_mask": base.vector_stats(np.asarray(values["teleop_mask"], dtype=np.float32)),
        "intervention_action_mask": base.vector_stats(
            np.asarray(values["intervention_action_mask"], dtype=np.float32)
        ),
        "intervention_action_weight": base.vector_stats(
            np.asarray(values["intervention_action_weight"], dtype=np.float32)
        ),
        "episode_index": base.vector_stats(scalar_column([row["episode_index"] for row in rows])),
        "frame_index": base.vector_stats(scalar_column([row["frame_index"] for row in rows])),
        "timestamp": base.vector_stats(scalar_column([row["timestamp"] for row in rows])),
        "next.done": base.vector_stats(scalar_column([float(row["next.done"]) for row in rows])),
        "index": base.vector_stats(scalar_column([row["index"] for row in rows])),
        "teleop_applied": base.vector_stats(scalar_column([float(row["teleop_applied"]) for row in rows])),
        "teleop_action_available": base.vector_stats(
            scalar_column([float(row["teleop_action_available"]) for row in rows])
        ),
        "is_intervention": base.vector_stats(scalar_column([float(row["is_intervention"]) for row in rows])),
        "control_mode": base.vector_stats(scalar_column([row["control_mode"] for row in rows])),
        "source_episode_index": base.vector_stats(
            scalar_column([row["source_episode_index"] for row in rows])
        ),
        "source_frame_index": base.vector_stats(scalar_column([row["source_frame_index"] for row in rows])),
        "source_timestamp": base.vector_stats(scalar_column([row["source_timestamp"] for row in rows])),
    }
    for camera in cameras:
        stats[f"observation.images.{camera}"] = base.image_stats(
            records=records,
            camera=camera,
            max_frames=max_image_stat_frames,
        )
    return stats


def build_info(
    *,
    fps: int,
    raw_dir: Path,
    records: list[base.EpisodeRecord],
    segments: list[InterventionSegment],
    rows: list[dict[str, Any]],
    cameras: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """构造 meta_data/info.json。"""

    source_episode_names = sorted({records[s.record_index].source_name for s in segments})
    intervention_frames = sum(bool(row["is_intervention"]) for row in rows)
    is_full_rollout = args.dataset_mode == DATASET_MODE_FULL_ROLLOUT
    return {
        "codebase_version": "v1.6",
        "fps": int(fps),
        "video": 1,
        "encoding": base.DEFAULT_ENCODING,
        "source_raw_dir": str(raw_dir),
        "dataset_kind": "human_in_loop_rollout" if is_full_rollout else "human_intervention",
        "dataset_mode": str(args.dataset_mode),
        "total_episodes": int(len(segments)),
        "total_frames": int(len(rows)),
        "total_intervention_frames": int(intervention_frames),
        "total_robot_frames": int(len(rows) - intervention_frames),
        "total_source_episodes": int(len(source_episode_names)),
        "source_episode_names": source_episode_names,
        "camera_keys": [f"observation.images.{camera}" for camera in cameras],
        "action_key": str(args.action_key),
        "filter_mode": str(args.filter_mode),
        "min_blend_weight": float(args.min_blend_weight),
        "merge_gap_frames": int(args.merge_gap_frames),
        "pre_context_frames": int(args.pre_context_frames),
        "post_context_frames": int(args.post_context_frames),
        "min_segment_frames": int(args.min_segment_frames),
        "description": (
            (
                "Full policy rollouts from quest_policy_collect.py with per-frame human "
                "intervention labels. "
                if is_full_rollout
                else "Frames/segments extracted from quest_policy_collect.py where human takeover was detected. "
            )
            + "action is the full executed action after policy/teleop blending."
        ),
    }


def write_dataset_card(local_dir: Path, raw_dir: Path, info: dict[str, Any]) -> None:
    """写入数据集 README。"""

    text = (
        "---\n"
        "task_categories:\n"
        "- robotics\n"
        "tags:\n"
        "- LeRobot\n"
        "- human-intervention\n"
        "---\n"
        f"This dataset was converted from `{raw_dir}` for LeRobot training.\n\n"
        f"Dataset kind: `{info['dataset_kind']}`.\n"
        f"Mode: `{info['dataset_mode']}`.\n"
        f"Episodes: `{info['total_episodes']}`.\n"
        f"Frames: `{info['total_frames']}`.\n"
        f"Intervention frames: `{info['total_intervention_frames']}`.\n"
        f"Robot frames: `{info['total_robot_frames']}`.\n"
        f"Filter mode: `{info['filter_mode']}`.\n"
    )
    (local_dir / "README.md").write_text(text, encoding="utf-8")


def build_local_intervention_dataset(args: argparse.Namespace) -> Path:
    """从 raw 采集目录构建干预片段或完整 HIL LeRobot/HF 数据集。"""

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

    metadata = base.read_json(raw_dir / "metadata.json") if (raw_dir / "metadata.json").exists() else {}
    records = base.select_episodes(raw_dir=raw_dir, max_episodes=args.max_episodes)
    cameras = base.infer_cameras(records, args.cameras)
    fps = base.infer_fps(records, metadata, args.fps)
    if args.dataset_mode == DATASET_MODE_FULL_ROLLOUT:
        segments = select_full_rollouts(records, action_key=args.action_key)
    else:
        segments = select_intervention_segments(
            records,
            action_key=args.action_key,
            filter_mode=args.filter_mode,
            min_blend_weight=args.min_blend_weight,
            merge_gap_frames=args.merge_gap_frames,
            pre_context_frames=args.pre_context_frames,
            post_context_frames=args.post_context_frames,
            min_segment_frames=args.min_segment_frames,
        )
    if not segments:
        raise RuntimeError(
            "没有找到可转换的轨迹/干预片段。请检查 raw 数据是否来自 quest_policy_collect.py，"
            "或在 intervention_only 模式尝试其他 --filter-mode。"
        )

    logging.info(
        "Selected %d source episode(s), %d output episode(s), mode=%s, cameras=%s, fps=%d.",
        len(records),
        len(segments),
        args.dataset_mode,
        cameras,
        fps,
    )

    rows, values = build_rows_and_videos(
        records=records,
        segments=segments,
        cameras=cameras,
        videos_dir=videos_dir,
        fps=fps,
        overwrite=args.overwrite,
        action_key=args.action_key,
        filter_mode=args.filter_mode,
        min_blend_weight=args.min_blend_weight,
    )
    if not rows:
        raise RuntimeError("Trajectories were selected, but no rows were generated.")

    state_dim = int(np.asarray(values["state"][0]).shape[-1])
    action_dim = int(np.asarray(values["action"][0]).shape[-1])
    teleop_mask_dim = int(np.asarray(values["teleop_mask"][0]).shape[-1])
    blend_weight_dim = int(np.asarray(values["blend_weight"][0]).shape[-1])

    hf_dataset = datasets.Dataset.from_list(
        rows,
        features=make_hf_features(
            cameras=cameras,
            state_dim=state_dim,
            action_dim=action_dim,
            teleop_mask_dim=teleop_mask_dim,
            blend_weight_dim=blend_weight_dim,
        ),
    )
    parquet_path = data_dir / "train-00000-of-00001.parquet"
    hf_dataset.to_parquet(str(parquet_path))

    episode_data_index = calculate_episode_data_index(hf_dataset)
    save_file(episode_data_index, meta_data_dir / "episode_data_index.safetensors")

    stats = build_intervention_stats(
        rows=rows,
        values=values,
        records=records,
        cameras=cameras,
        max_image_stat_frames=args.max_image_stat_frames,
    )
    save_file(flatten_dict(stats), meta_data_dir / "stats.safetensors")

    info = build_info(
        fps=fps,
        raw_dir=raw_dir,
        records=records,
        segments=segments,
        rows=rows,
        cameras=cameras,
        args=args,
    )
    base.write_json(meta_data_dir / "info.json", info)
    write_dataset_card(local_dir=local_dir, raw_dir=raw_dir, info=info)
    base.write_gitattributes(local_dir)

    logging.info("Wrote parquet: %s", parquet_path)
    logging.info("Wrote metadata: %s", meta_data_dir)
    logging.info("Wrote videos: %s", videos_dir)
    return local_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 quest_policy_collect.py 的干预片段或完整人在环轨迹转换为 LeRobot/HF 数据集。"
    )
    parser.add_argument(
        "--dataset-mode",
        default=DATASET_MODE_INTERVENTION_ONLY,
        choices=DATASET_MODES,
        help=(
            "intervention_only=仅提取接管片段（兼容旧行为）；"
            "full_rollout=完整保留每条 policy+human rollout，供 IWR 采样 D_I/D_R。"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default="outputs/4_data_collect/quest_policy/",
        help="采集数据目录；也可以填父目录，脚本会自动选择最新的 collect_*、quest_teleop_* 或 quest_policy_* 目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/5_hf_datasets/quest_policy_human_intervention_rgb_joint",
        help="本地生成的人类干预 HF 数据集目录。",
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
        help="最多读取多少条原始 episode；None 表示读取全部 episode。",
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
        choices=("joint_action", "action"),
        help="raw arrays.npz 中映射到 LeRobot action 字段的动作键；默认使用最终执行的 joint_action。",
    )
    parser.add_argument(
        "--filter-mode",
        default="teleop_applied",
        choices=(
            "saved_segments",
            "teleop_applied",
            "teleop_mask",
            "blend_weight",
            "control_mode",
            "teleop_or_blend",
        ),
        help=(
            "如何判断一帧属于人类干预：saved_segments=读取采集时生成的 intervention_segments.json；"
            "teleop_applied=最终动作受人类/平滑接管影响；teleop_mask=当前目标模式有人类接管；"
            "blend_weight=混合权重超过阈值；control_mode=非 policy 模式；"
            "teleop_or_blend=teleop_applied 或 blend_weight。"
        ),
    )
    parser.add_argument(
        "--min-blend-weight",
        type=float,
        default=1.0e-6,
        help="filter-mode 为 blend_weight/teleop_or_blend 时使用的最小混合权重。",
    )
    parser.add_argument(
        "--merge-gap-frames",
        type=int,
        default=0,
        help="两个接管片段中间如果断开不超过该帧数，则合并为一个片段。",
    )
    parser.add_argument(
        "--pre-context-frames",
        type=int,
        default=0,
        help="每个接管片段前额外保留多少帧上下文；默认 0，表示只保留接管后的帧。",
    )
    parser.add_argument(
        "--post-context-frames",
        type=int,
        default=0,
        help="每个接管片段后额外保留多少帧上下文；默认 0。",
    )
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=1,
        help="丢弃短于该长度的接管片段。Diffusion 训练可设为 horizon，例如 16。",
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


def run_from_args(args: argparse.Namespace) -> None:
    base.apply_runtime_env(args)
    local_dir = build_local_intervention_dataset(args)
    logging.info("Human intervention dataset is ready: %s", local_dir)


def convert_intervention_folder_to_hf(
    raw_dir: str,
    output_dir: str = "outputs/5_hf_datasets/quest_policy_human_intervention_rgb_joint",
    overwrite: bool = True,
    max_episodes: int | None = None,
    cameras: str | None = None,
    fps: int | None = None,
    max_image_stat_frames: int = 500,
    action_key: str = "joint_action",
    filter_mode: str = "teleop_applied",
    min_blend_weight: float = 1.0e-6,
    merge_gap_frames: int = 0,
    pre_context_frames: int = 0,
    post_context_frames: int = 0,
    min_segment_frames: int = 1,
    dataset_mode: str = DATASET_MODE_INTERVENTION_ONLY,
    http_proxy: str = "",
    https_proxy: str = "",
) -> None:
    """用 Python 变量调用转换流程，便于 Notebook/脚本内复用。"""

    base.init_logging()
    args = argparse.Namespace(
        raw_dir=raw_dir,
        output_dir=output_dir,
        dataset_mode=dataset_mode,
        overwrite=overwrite,
        max_episodes=max_episodes,
        cameras=cameras,
        fps=fps,
        max_image_stat_frames=max_image_stat_frames,
        action_key=action_key,
        filter_mode=filter_mode,
        min_blend_weight=min_blend_weight,
        merge_gap_frames=merge_gap_frames,
        pre_context_frames=pre_context_frames,
        post_context_frames=post_context_frames,
        min_segment_frames=min_segment_frames,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
    )
    run_from_args(args)


def main() -> None:
    base.init_logging()
    parser = build_arg_parser()
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit

    # raw arrays.npz 中映射到 LeRobot action 字段的动作键。
    ACTION_KEY = "joint_action"

    # quest_policy_collect.py 生成的原始人在环采集 run。
    RAW_DIR = (
        "outputs/4_data_collect/quest_policy/"
        "quest_policy_InsertCylinder_fail_replay_rgb"
    )

    # 完整 policy+human rollout HF 数据集输出目录。
    OUTPUT_DIR = (
        "outputs/5_hf_datasets/"
        "quest_policy_insert_cylinder_hil_full_rollout_rgb_joint"
    )

    # DATASET_MODE_FULL_ROLLOUT 保留每条原始 episode 的全部 policy/human 帧，供 IWR 使用。
    # DATASET_MODE_INTERVENTION_ONLY 仅提取接管片段，兼容旧行为。
    DATASET_MODE = DATASET_MODE_FULL_ROLLOUT

    # 输出目录已存在时是否覆盖重建。
    OVERWRITE = True

    # 最多读取多少条原始 episode；None 表示全部转换。
    MAX_EPISODES = None

    # 与当前双 ZED 策略输入保持一致；None 表示转换采集 run 中的全部相机。
    CAMERAS = "zed_cam_left,zed_cam_right,wrist_cam_left,wrist_cam_right,overhead_cam,worms_eye_cam"

    # None 表示读取采集 metadata.json 中记录的 fps。
    FPS = None

    # 每个相机最多抽样多少帧计算统计；0 表示使用占位图像统计。
    MAX_IMAGE_STAT_FRAMES = 600

    # 人类干预帧判定配置。
    #
    # 注意：
    # - DATASET_MODE="intervention_only" 时，FILTER_MODE 决定哪些帧会被提取出来。
    # - DATASET_MODE="full_rollout" 时，所有轨迹帧都会保留；FILTER_MODE 只决定
    #   每帧的 is_intervention 标签以及后续 D_I/D_R 索引划分。
    #
    # FILTER_MODE 可选值：
    # 1. "saved_segments"
    #    读取每条原始 episode 的 intervention_segments.json，严格复用采集阶段保存
    #    的 [start_frame, stop_frame) 区间。文件不存在时自动回退到 teleop_applied。
    #
    # 2. "teleop_applied"（推荐默认值）
    #    使用采集端保存的综合布尔标记。只要当前存在有效遥操作目标，或者接管/
    #    交还平滑混合仍在影响执行动作，就标记为干预；一般覆盖范围最符合实际执行。
    #
    # 3. "teleop_mask"
    #    任一机械臂处于人类接管目标模式即标记为干预。它侧重“当前控制权归属”，
    #    不包含切回 policy 后仍有残余 blend_weight 的渐出帧。
    #
    # 4. "blend_weight"
    #    任一机械臂 blend_weight >= MIN_BLEND_WEIGHT 即标记为干预。它侧重“最终执行
    #    动作是否真正混入人工分量”，会包含接管渐入和交还渐出，但可能忽略刚切换
    #    接管且权重仍为 0 的第一帧。
    #
    # 5. "control_mode"
    #    control_mode 不是 policy 即标记为干预。它反映操作员选择的离散控制模式，
    #    但不保证当前帧已经产生非零人工混合权重。
    #
    # 6. "teleop_or_blend"
    #    teleop_applied 或任一 blend_weight 达到阈值即标记为干预。覆盖最宽，适合兼容
    #    旧版/异常数据中 teleop_applied 与 blend_weight 只有一个字段可靠的情况。
    FILTER_MODE = "teleop_applied"
    MIN_BLEND_WEIGHT = 1.0e-6
    MERGE_GAP_FRAMES = 0
    PRE_CONTEXT_FRAMES = 0
    POST_CONTEXT_FRAMES = 0
    MIN_SEGMENT_FRAMES = 1

    # 可选代理；留空表示不修改当前环境变量。
    HTTP_PROXY = ""
    HTTPS_PROXY = ""

    convert_intervention_folder_to_hf(
        raw_dir=RAW_DIR,
        output_dir=OUTPUT_DIR,
        dataset_mode=DATASET_MODE,
        overwrite=OVERWRITE,
        max_episodes=MAX_EPISODES,
        cameras=CAMERAS,
        fps=FPS,
        max_image_stat_frames=MAX_IMAGE_STAT_FRAMES,
        action_key=ACTION_KEY,
        filter_mode=FILTER_MODE,
        min_blend_weight=MIN_BLEND_WEIGHT,
        merge_gap_frames=MERGE_GAP_FRAMES,
        pre_context_frames=PRE_CONTEXT_FRAMES,
        post_context_frames=POST_CONTEXT_FRAMES,
        min_segment_frames=MIN_SEGMENT_FRAMES,
        http_proxy=HTTP_PROXY,
        https_proxy=HTTPS_PROXY,
    )
