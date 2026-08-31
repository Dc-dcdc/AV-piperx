"""按训练 action horizon 计算并缓存 View 当前锚点增量统计量。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock
from torch import Tensor

from lerobot.common.policies.diffusion.view_action_representation import (
    VIEW_ACTION_DELTA_STATS_KEY,
)


_CACHE_SCHEMA_VERSION = 1
_DELTA_DEFINITION = "future_absolute_action_minus_current_observation_state"
_SYMMETRIC_MIN_MAX_MARGIN = 1.001


def _jsonable_dataset_descriptor(dataset) -> dict[str, Any]:
    """生成不会读取视频的稳定数据集描述，用于缓存失效判断。"""
    if hasattr(dataset, "_datasets"):
        return {
            "type": type(dataset).__name__,
            "datasets": [
                _jsonable_dataset_descriptor(child)
                for child in dataset._datasets
            ],
        }
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise TypeError(
            "View增量统计仅支持具有hf_dataset属性的LeRobotDataset。"
        )
    descriptor = {
        "type": type(dataset).__name__,
        "repo_id": str(getattr(dataset, "repo_id", "")),
        "root": str(getattr(dataset, "root", "")),
        "num_samples": int(len(hf_dataset)),
        "fingerprint": str(getattr(hf_dataset, "_fingerprint", "")),
    }
    root = getattr(dataset, "root", None)
    repo_id = str(getattr(dataset, "repo_id", ""))
    if root is not None:
        local_dataset_dir = Path(root).expanduser() / repo_id
        if local_dataset_dir.is_dir():
            tracked_files = sorted((local_dataset_dir / "data").glob("*.parquet"))
            tracked_files.extend(
                path
                for path in (
                    local_dataset_dir / "meta_data" / "info.json",
                    local_dataset_dir / "meta_data" / "episode_data_index.safetensors",
                )
                if path.is_file()
            )
            descriptor["local_files"] = [
                {
                    "path": str(path.relative_to(local_dataset_dir)),
                    "size": int(path.stat().st_size),
                    "mtime_ns": int(path.stat().st_mtime_ns),
                }
                for path in tracked_files
            ]
    return descriptor


def _build_spec(
    dataset,
    *,
    action_delta_timestamps: list[float],
    arm_action_dim: int,
    view_action_dim: int,
    include_padding: bool,
) -> dict[str, Any]:
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "dataset": _jsonable_dataset_descriptor(dataset),
        "action_delta_timestamps": [float(value) for value in action_delta_timestamps],
        "arm_action_dim": int(arm_action_dim),
        "view_action_dim": int(view_action_dim),
        "delta_definition": _DELTA_DEFINITION,
        "include_padding": bool(include_padding),
        "symmetric_min_max_margin": _SYMMETRIC_MIN_MAX_MARGIN,
    }


def _spec_digest(spec: dict[str, Any]) -> str:
    payload = json.dumps(
        spec,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _empty_accumulator(view_action_dim: int) -> dict[str, Tensor | int]:
    return {
        "count": 0,
        "padding_count": 0,
        "candidate_count": 0,
        "sum": torch.zeros(view_action_dim, dtype=torch.float64),
        "sum_sq": torch.zeros(view_action_dim, dtype=torch.float64),
        "min": torch.full((view_action_dim,), torch.inf, dtype=torch.float64),
        "max": torch.full((view_action_dim,), -torch.inf, dtype=torch.float64),
    }


def _merge_values(
    accumulator: dict[str, Tensor | int],
    values: Tensor,
    *,
    padding_count: int,
    candidate_count: int,
) -> None:
    accumulator["padding_count"] += int(padding_count)
    accumulator["candidate_count"] += int(candidate_count)
    if values.numel() == 0:
        return
    values = values.to(dtype=torch.float64)
    accumulator["count"] += int(values.shape[0])
    accumulator["sum"] += values.sum(dim=0)
    accumulator["sum_sq"] += values.square().sum(dim=0)
    accumulator["min"] = torch.minimum(
        accumulator["min"],
        values.amin(dim=0),
    )
    accumulator["max"] = torch.maximum(
        accumulator["max"],
        values.amax(dim=0),
    )


def _numeric_columns(dataset) -> dict[str, Tensor]:
    """一次性读取四个数值列；不会触发任何图像或视频解码。"""
    required_columns = (
        "timestamp",
        "episode_index",
        "observation.state",
        "action",
    )
    missing = set(required_columns).difference(dataset.hf_dataset.column_names)
    if missing:
        raise KeyError(f"数据集缺少View增量统计所需字段: {sorted(missing)}。")
    numeric_dataset = dataset.hf_dataset.select_columns(list(required_columns))
    values = numeric_dataset.with_format("torch")[:]
    return {
        key: torch.as_tensor(values[key]).cpu()
        for key in required_columns
    }


def _accumulate_single_dataset(
    dataset,
    accumulator: dict[str, Tensor | int],
    *,
    action_delta_timestamps: list[float],
    arm_action_dim: int,
    view_action_dim: int,
    include_padding: bool,
) -> None:
    columns = _numeric_columns(dataset)
    timestamps = columns["timestamp"]
    episode_ids = columns["episode_index"]
    states = columns["observation.state"]
    actions = columns["action"]
    action_dim = arm_action_dim + view_action_dim
    if states.ndim != 2 or states.shape[-1] < action_dim:
        raise ValueError(
            "delta_from_current要求observation.state包含完整Arm/View关节，"
            f"至少需要{action_dim}维，当前形状为{tuple(states.shape)}。"
        )
    if actions.ndim != 2 or actions.shape[-1] != action_dim:
        raise ValueError(
            f"action应为[N,{action_dim}]，当前形状为{tuple(actions.shape)}。"
        )
    if not (
        len(timestamps)
        == len(episode_ids)
        == len(states)
        == len(actions)
    ):
        raise ValueError("timestamp/episode_index/state/action样本数不一致。")

    episode_index = getattr(dataset, "episode_data_index", None)
    if episode_index is None or not {"from", "to"}.issubset(episode_index):
        raise AttributeError("数据集缺少episode_data_index['from'/'to']。")
    starts = torch.as_tensor(episode_index["from"], dtype=torch.long)
    ends = torch.as_tensor(episode_index["to"], dtype=torch.long)
    if starts.numel() != ends.numel():
        raise ValueError("episode_data_index的from/to长度不一致。")

    fps = int(getattr(dataset, "fps"))
    tolerance_s = 1.0 / fps - 1e-4
    action_offsets = torch.tensor(
        action_delta_timestamps,
        dtype=timestamps.dtype,
    )
    view_slice = slice(arm_action_dim, action_dim)

    for start_tensor, end_tensor in zip(starts, ends, strict=True):
        start = int(start_tensor.item())
        end = int(end_tensor.item())
        if start < 0 or end <= start or end > len(timestamps):
            raise ValueError(
                f"非法episode范围[{start}, {end})，数据集长度={len(timestamps)}。"
            )

        episode_timestamps = timestamps[start:end]
        episode_id_values = episode_ids[start:end]
        if torch.any(episode_id_values != episode_id_values[0]):
            raise ValueError(
                f"episode范围[{start}, {end})包含多个episode_index。"
            )
        if torch.any(episode_timestamps[1:] < episode_timestamps[:-1]):
            raise ValueError(f"episode范围[{start}, {end})的timestamp没有递增。")
        query_timestamps = (
            episode_timestamps[:, None] + action_offsets[None, :]
        )
        insertion = torch.searchsorted(
            episode_timestamps,
            query_timestamps,
            right=False,
        )
        right = insertion.clamp(0, len(episode_timestamps) - 1)
        left = (insertion - 1).clamp(0, len(episode_timestamps) - 1)
        left_distance = (
            query_timestamps - episode_timestamps[left]
        ).abs()
        right_distance = (
            episode_timestamps[right] - query_timestamps
        ).abs()
        # torch.cdist(...).min()在距离相同时选择更早的索引，因此只在严格
        # 更近时选择right，保持与LeRobot horizon加载逻辑一致。
        choose_right = right_distance < left_distance
        nearest = torch.where(choose_right, right, left)
        nearest_distance = torch.where(
            choose_right,
            right_distance,
            left_distance,
        )
        is_pad = nearest_distance > tolerance_s
        outside_episode = (
            (query_timestamps < episode_timestamps[0])
            | (query_timestamps > episode_timestamps[-1])
        )
        if torch.any(is_pad & ~outside_episode):
            bad_distance = float(nearest_distance[is_pad & ~outside_episode].max())
            raise ValueError(
                "动作时间戳在episode内部找不到容差内帧："
                f"最大最近距离={bad_distance:.8f}s, tolerance={tolerance_s:.8f}s。"
            )

        episode_actions = actions[start:end, view_slice]
        episode_anchors = states[start:end, view_slice]
        view_delta = (
            episode_actions[nearest]
            - episode_anchors[:, None, :]
        )
        valid = torch.ones_like(is_pad) if include_padding else ~is_pad
        _merge_values(
            accumulator,
            view_delta[valid],
            padding_count=int(is_pad.sum().item()),
            candidate_count=int(is_pad.numel()),
        )


def _compute_stats(
    dataset,
    *,
    action_delta_timestamps: list[float],
    arm_action_dim: int,
    view_action_dim: int,
    include_padding: bool,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    accumulator = _empty_accumulator(view_action_dim)
    datasets = (
        list(dataset._datasets)
        if hasattr(dataset, "_datasets")
        else [dataset]
    )
    for child in datasets:
        _accumulate_single_dataset(
            child,
            accumulator,
            action_delta_timestamps=action_delta_timestamps,
            arm_action_dim=arm_action_dim,
            view_action_dim=view_action_dim,
            include_padding=include_padding,
        )

    count = int(accumulator["count"])
    if count <= 0:
        raise ValueError("没有可用于计算View增量统计量的动作目标。")
    mean = accumulator["sum"] / count
    variance = accumulator["sum_sq"] / count - mean.square()
    std = variance.clamp_min(0.0).sqrt().clamp_min(1e-6)
    raw_min = accumulator["min"]
    raw_max = accumulator["max"]
    scale = torch.maximum(raw_min.abs(), raw_max.abs())
    scale = (scale * _SYMMETRIC_MIN_MAX_MARGIN).clamp_min(1e-6)
    stats = {
        "mean": mean.to(dtype=torch.float32),
        "std": std.to(dtype=torch.float32),
        # 对称区间保证零位移严格归一化为0，同时为数值边界留0.1%余量。
        "min": (-scale).to(dtype=torch.float32),
        "max": scale.to(dtype=torch.float32),
    }
    candidate_count = int(accumulator["candidate_count"])
    padding_count = int(accumulator["padding_count"])
    metadata = {
        "valid_or_included_count": count,
        "candidate_count": candidate_count,
        "padding_count": padding_count,
        "padding_fraction": (
            float(padding_count / candidate_count)
            if candidate_count > 0
            else 0.0
        ),
        "raw_min": raw_min.tolist(),
        "raw_max": raw_max.tolist(),
    }
    return stats, metadata


def _load_cache(
    cache_path: Path,
    *,
    spec: dict[str, Any],
    digest: str,
    view_action_dim: int,
) -> tuple[dict[str, Tensor], dict[str, Any]] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("spec_digest") != digest or payload.get("spec") != spec:
            return None
        raw_stats = payload["stats"]
        stats = {
            name: torch.tensor(raw_stats[name], dtype=torch.float32)
            for name in ("mean", "std", "min", "max")
        }
        for name, tensor in stats.items():
            if tensor.shape != (view_action_dim,) or not torch.isfinite(tensor).all():
                raise ValueError(f"缓存统计{name}形状或数值非法。")
        if not torch.all(stats["std"] > 0) or not torch.all(
            stats["max"] > stats["min"]
        ):
            raise ValueError("缓存中的std或min/max范围非法。")
        return stats, dict(payload.get("metadata", {}))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logging.warning("忽略损坏的View增量统计缓存%s: %s", cache_path, exc)
        return None


def _save_cache(
    cache_path: Path,
    *,
    spec: dict[str, Any],
    digest: str,
    stats: dict[str, Tensor],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "spec_digest": digest,
        "spec": spec,
        "stats_key": VIEW_ACTION_DELTA_STATS_KEY,
        "stats": {
            name: tensor.detach().cpu().tolist()
            for name, tensor in stats.items()
        },
        "metadata": metadata,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=cache_path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_name, cache_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_or_compute_view_delta_stats(
    dataset,
    *,
    action_delta_timestamps: list[float],
    arm_action_dim: int,
    view_action_dim: int,
    include_padding: bool,
    cache_dir: str | Path,
) -> tuple[dict[str, Tensor], Path, dict[str, Any]]:
    """返回派生统计量、缓存路径和统计元数据。

    首次调用扫描 action/state/timestamp 数值列并原子写入缓存；后续相同
    数据指纹与 horizon 规格直接复用。缓存不修改数据集 canonical stats。
    """
    if not action_delta_timestamps:
        raise ValueError("action_delta_timestamps不能为空。")
    if not all(math.isfinite(float(value)) for value in action_delta_timestamps):
        raise ValueError("action_delta_timestamps必须全部为有限值。")
    if arm_action_dim <= 0 or view_action_dim <= 0:
        raise ValueError("arm_action_dim和view_action_dim必须为正数。")

    spec = _build_spec(
        dataset,
        action_delta_timestamps=action_delta_timestamps,
        arm_action_dim=arm_action_dim,
        view_action_dim=view_action_dim,
        include_padding=include_padding,
    )
    digest = _spec_digest(spec)
    cache_path = Path(cache_dir).expanduser() / f"view_delta_{digest}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(cache_path) + ".lock")
    with lock:
        cached = _load_cache(
            cache_path,
            spec=spec,
            digest=digest,
            view_action_dim=view_action_dim,
        )
        if cached is not None:
            stats, metadata = cached
            logging.info("复用View增量统计缓存: %s", cache_path)
            return stats, cache_path, metadata

        logging.info(
            "首次计算View增量统计（只读取数值列，不解码视频）: "
            "horizon=%d, include_padding=%s",
            len(action_delta_timestamps),
            include_padding,
        )
        stats, metadata = _compute_stats(
            dataset,
            action_delta_timestamps=action_delta_timestamps,
            arm_action_dim=arm_action_dim,
            view_action_dim=view_action_dim,
            include_padding=include_padding,
        )
        _save_cache(
            cache_path,
            spec=spec,
            digest=digest,
            stats=stats,
            metadata=metadata,
        )
        logging.info("已保存View增量统计缓存: %s", cache_path)
        return stats, cache_path, metadata
