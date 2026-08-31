#!/usr/bin/env python
"""用纯专家 Diffusion Policy 定位专家轨迹中的高风险片段。

该脚本不会调用 ``select_action``，因而不会复用策略内部的动作队列。每个
专家起点都使用真实的 ``n_obs_steps`` 观测，直接生成 K 个扩散动作块，并在
模型归一化空间中与专家的实际执行切片比较。输出包括逐帧分数、经过时序
去噪与 NMS 后的扰动候选，以及足以复现实验的数据/模型元数据。

典型用法::

    python data_collect/recovery_data_generation/score_expert_policy_risk.py \
      --config configs/data_collect/expert_policy_risk.yaml

``pretrained_model`` 是本项目保存的 EMA 模型；脚本故意拒绝 ``online_model``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SCHEMA_VERSION = 1
METRIC_NAMES = ("expected", "min", "p90", "bias", "spread")
CONFIG_SECTIONS = {
    "paths": {
        "dataset_local_dir",
        "raw_dir",
        "checkpoint",
        "output_dir",
    },
    "model": {"policy_name", "action_key", "device"},
    "dataset": {"video_backend", "mapping_atol"},
    "runtime": {
        "batch_size",
        "num_workers",
        "num_samples",
        "seed",
        "frame_stride",
        "min_frame_index",
        "max_frames",
        "max_episodes",
    },
    "risk_scoring": {"smooth_window", "threshold_quantile"},
    "anchor_selection": {
        "merge_gap",
        "nms_distance",
        "arm_total_anchor_budget",
        "view_total_anchor_budget",
        "max_anchors_per_episode",
        "random_exploration_ratio",
    },
    "recovery_compatibility": {
        "arm_required_future_frames",
        "view_required_future_frames",
    },
}
PATH_CONFIG_FIELDS = {
    "dataset_local_dir",
    "raw_dir",
    "checkpoint",
    "output_dir",
}


@dataclass(frozen=True)
class EpisodeMapping:
    """HF episode 与 raw episode 的可审计一一映射。"""

    hf_episode_index: int
    raw_episode_name: str
    source_episode: int
    variant: int
    episode_length: int
    hf_from: int
    hf_to: int
    action_sha256: str
    raw_episode_dir: str


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"无法JSON序列化{type(value).__name__}。")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def canonical_action_sha256(actions: np.ndarray) -> str:
    """对动作数组生成与平台字节序无关的严格 float32 校验和。"""

    array = np.asarray(actions, dtype="<f4")
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("动作数组必须为有限值组成的二维数组。")
    header = np.asarray(array.shape, dtype="<i8").tobytes()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


def parse_raw_episode_name(name: str) -> tuple[int, int]:
    """解析 converter 使用的 episode_XXXXXX[_aug_XX] 命名。"""

    import re

    match = re.fullmatch(r"episode_(\d{6,})(?:_aug_(\d{2,}))?", str(name))
    if match is None:
        raise ValueError(f"非法raw episode名称: {name!r}。")
    return int(match.group(1)), -1 if match.group(2) is None else int(match.group(2))


def strict_episode_mapping(
    hf_episodes: Sequence[Mapping[str, Any]],
    raw_episodes: Sequence[Mapping[str, Any]],
    *,
    atol: float = 0.0,
) -> list[EpisodeMapping]:
    """按 converter 的排序规则严格验证 HF/raw episode 映射。

    两边必须 episode 数相同、HF 编号连续、长度相同，并且动作逐元素一致。
    ``atol`` 默认 0，以避免意外把不对应的轨迹映射到一起；只有确认 parquet
    写入发生可解释的数值舍入时才应提高它。
    """

    raw_sorted = sorted(
        raw_episodes,
        key=lambda item: (
            parse_raw_episode_name(str(item["name"]))[0],
            parse_raw_episode_name(str(item["name"]))[1] + 1,
        ),
    )
    hf_sorted = sorted(hf_episodes, key=lambda item: int(item["episode_index"]))
    if len(hf_sorted) != len(raw_sorted):
        raise ValueError(
            "HF/raw episode数量不一致: "
            f"hf={len(hf_sorted)}, raw={len(raw_sorted)}。"
        )

    mappings: list[EpisodeMapping] = []
    for expected_index, (hf_item, raw_item) in enumerate(zip(hf_sorted, raw_sorted, strict=True)):
        hf_index = int(hf_item["episode_index"])
        if hf_index != expected_index:
            raise ValueError(
                "HF episode_index必须从0连续编号；"
                f"期望{expected_index}，实际{hf_index}。"
            )
        hf_actions = np.asarray(hf_item["actions"], dtype=np.float32)
        raw_actions = np.asarray(raw_item["actions"], dtype=np.float32)
        if hf_actions.shape != raw_actions.shape:
            raise ValueError(
                f"episode {hf_index}长度/动作维度不一致: "
                f"hf={hf_actions.shape}, raw={raw_actions.shape}。"
            )
        if not np.allclose(hf_actions, raw_actions, rtol=0.0, atol=float(atol)):
            difference = float(np.max(np.abs(hf_actions - raw_actions)))
            raise ValueError(
                f"episode {hf_index}动作校验失败，max_abs_diff={difference:.9g}；"
                "禁止猜测HF与raw映射。"
            )
        source_episode, variant = parse_raw_episode_name(str(raw_item["name"]))
        hf_digest = canonical_action_sha256(hf_actions)
        raw_digest = canonical_action_sha256(raw_actions)
        if atol == 0.0 and hf_digest != raw_digest:
            raise ValueError(f"episode {hf_index}动作SHA256校验失败。")
        mappings.append(
            EpisodeMapping(
                hf_episode_index=hf_index,
                raw_episode_name=str(raw_item["name"]),
                source_episode=source_episode,
                variant=variant,
                episode_length=int(hf_actions.shape[0]),
                hf_from=int(hf_item["from"]),
                hf_to=int(hf_item["to"]),
                action_sha256=raw_digest,
                raw_episode_dir=str(Path(raw_item["path"]).resolve()),
            )
        )
    return mappings


def resolve_action_groups(action_dim: int, arm_action_dim: int) -> dict[str, slice]:
    """返回 PiperX 20维动作的角色切片，并保留 Arm/View 综合切片。"""

    action_dim = int(action_dim)
    arm_action_dim = int(arm_action_dim)
    if arm_action_dim < 14:
        raise ValueError(
            "left/right arm和gripper评分要求arm_action_dim至少为14，"
            f"当前为{arm_action_dim}。"
        )
    if action_dim <= arm_action_dim:
        raise ValueError(
            f"动作中没有View维度: action_dim={action_dim}, arm_action_dim={arm_action_dim}。"
        )
    return {
        "left_arm": slice(0, 6),
        "left_gripper": slice(6, 7),
        "right_arm": slice(7, 13),
        "right_gripper": slice(13, 14),
        "view": slice(arm_action_dim, action_dim),
    }


def execution_slice(
    action: torch.Tensor,
    action_is_pad: torch.Tensor,
    *,
    n_obs_steps: int,
    n_action_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """提取 DP 真正执行的动作区间，并同步提取显式 padding mask。"""

    if action.ndim != 3 or action_is_pad.ndim != 2:
        raise ValueError("action须为[B,H,D]，action_is_pad须为[B,H]。")
    if action.shape[:2] != action_is_pad.shape:
        raise ValueError("action与action_is_pad的batch/horizon形状不一致。")
    start = int(n_obs_steps) - 1
    end = start + int(n_action_steps)
    if start < 0 or end > action.shape[1]:
        raise ValueError(
            f"执行切片[{start}:{end}]超出horizon={action.shape[1]}。"
        )
    return action[:, start:end], action_is_pad[:, start:end].bool()


def _masked_rms(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """对最后两个(time, dimension)轴求逐样本masked RMS。"""

    if values.ndim != 4 or valid.ndim != 2:
        raise ValueError("values须为[K,B,T,D]，valid须为[B,T]。")
    weights = valid[None, :, :, None].to(dtype=values.dtype)
    denominator = weights.sum(dim=(-2, -1)) * values.shape[-1]
    numerator = (values.square() * weights).sum(dim=(-2, -1))
    return torch.sqrt(numerator / denominator.clamp_min(1.0))


def score_normalized_predictions(
    predictions: torch.Tensor,
    expert_actions: torch.Tensor,
    action_is_pad: torch.Tensor,
    groups: Mapping[str, slice],
) -> dict[str, torch.Tensor]:
    """在模型归一化空间计算分角色的多样本风险统计。

    Args:
        predictions: ``[K,B,T,D]`` 的 K 次扩散采样。
        expert_actions: ``[B,T,D]`` 的归一化专家动作。
        action_is_pad: ``[B,T]``，True位置从所有统计中显式排除。
    """

    if predictions.ndim != 4 or expert_actions.ndim != 3:
        raise ValueError("predictions须为[K,B,T,D]，expert_actions须为[B,T,D]。")
    if predictions.shape[1:] != expert_actions.shape:
        raise ValueError("预测与专家动作形状不一致。")
    if expert_actions.shape[:2] != action_is_pad.shape:
        raise ValueError("expert_actions与action_is_pad形状不一致。")
    if predictions.shape[0] < 1:
        raise ValueError("至少需要一次扩散采样。")
    valid = ~action_is_pad.bool()
    scorable = valid.any(dim=1)

    result: dict[str, torch.Tensor] = {
        "scorable": scorable,
        "valid_action_steps": valid.sum(dim=1),
        "padding_fraction": action_is_pad.float().mean(dim=1),
    }
    metric_group_values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for group_name, group_slice in groups.items():
        predicted = predictions[..., group_slice]
        expert = expert_actions[..., group_slice]
        if predicted.shape[-1] == 0:
            raise ValueError(f"动作组{group_name!r}为空。")
        per_draw_error = _masked_rms(predicted - expert.unsqueeze(0), valid)
        prediction_mean = predicted.mean(dim=0, keepdim=True)
        bias = _masked_rms(prediction_mean - expert.unsqueeze(0), valid).squeeze(0)
        centered = predicted - prediction_mean
        spread_per_draw = _masked_rms(centered, valid)
        metric_group_values[group_name] = (predicted, expert)
        nan = torch.full_like(bias, torch.nan)
        result[f"{group_name}_expected"] = torch.where(
            scorable, per_draw_error.mean(dim=0), nan
        )
        result[f"{group_name}_min"] = torch.where(
            scorable, per_draw_error.min(dim=0).values, nan
        )
        result[f"{group_name}_p90"] = torch.where(
            scorable, torch.quantile(per_draw_error, 0.9, dim=0), nan
        )
        result[f"{group_name}_bias"] = torch.where(scorable, bias, nan)
        result[f"{group_name}_spread"] = torch.where(
            scorable,
            torch.sqrt(spread_per_draw.square().mean(dim=0)),
            nan,
        )
    # Arm扰动仅作用于左右关节，不把两个夹爪纳入锚点风险。通过拼接12维
    # 关节后直接算masked RMS，避免对左右臂RMSE做简单平均造成维度偏差。
    left_predicted, left_expert = metric_group_values["left_arm"]
    right_predicted, right_expert = metric_group_values["right_arm"]
    joint_predicted = torch.cat((left_predicted, right_predicted), dim=-1)
    joint_expert = torch.cat((left_expert, right_expert), dim=-1)
    per_draw_joint = _masked_rms(
        joint_predicted - joint_expert.unsqueeze(0), valid
    )
    joint_mean = joint_predicted.mean(dim=0, keepdim=True)
    joint_bias = _masked_rms(
        joint_mean - joint_expert.unsqueeze(0), valid
    ).squeeze(0)
    joint_spread = _masked_rms(joint_predicted - joint_mean, valid)
    joint_nan = torch.full_like(joint_bias, torch.nan)
    result["arm_joint_expected"] = torch.where(
        scorable, per_draw_joint.mean(dim=0), joint_nan
    )
    result["arm_joint_min"] = torch.where(
        scorable, per_draw_joint.min(dim=0).values, joint_nan
    )
    result["arm_joint_p90"] = torch.where(
        scorable, torch.quantile(per_draw_joint, 0.9, dim=0), joint_nan
    )
    result["arm_joint_bias"] = torch.where(scorable, joint_bias, joint_nan)
    result["arm_joint_spread"] = torch.where(
        scorable,
        torch.sqrt(joint_spread.square().mean(dim=0)),
        joint_nan,
    )
    result["arm_joint_score"] = result["arm_joint_expected"]
    # 兼容旧分析字段；生成器必须以selected记录中的score_key/score为权威。
    result["arm_score"] = result["arm_joint_score"]
    result["view_score"] = result["view_expected"]
    return result


def prepare_policy_batch(
    policy: Any,
    batch: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """复现策略forward的归一化/动作表示，但不进入训练loss或动作缓存。"""

    if not {"observation.state", "action", "action_is_pad"}.issubset(batch):
        raise KeyError("batch缺少observation.state、action或action_is_pad。")
    working = dict(batch)
    target = working["action"]
    if bool(getattr(policy, "uses_view_delta_from_current", False)):
        anchor = policy._extract_current_view_anchor(working["observation.state"])
        target = policy._encode_actions_for_model(target, anchor)
    working["action"] = target
    normalized = policy.normalize_inputs(working)
    expected_image_keys = list(getattr(policy, "expected_image_keys", []))
    if expected_image_keys:
        missing = [key for key in expected_image_keys if key not in normalized]
        if missing:
            raise KeyError(f"batch缺少checkpoint要求的图像: {missing}。")
        normalized = dict(normalized)
        normalized["observation.images"] = torch.stack(
            [normalized[key] for key in expected_image_keys], dim=-4
        )
    normalized_target = policy.normalize_targets({"action": target})["action"]
    model_batch = {
        key: value
        for key, value in normalized.items()
        if key
        in {
            "observation.state",
            "observation.images",
            "observation.environment_state",
        }
    }
    expert_execution, pad_execution = execution_slice(
        normalized_target,
        working["action_is_pad"],
        n_obs_steps=int(policy.config.n_obs_steps),
        n_action_steps=int(policy.config.n_action_steps),
    )
    return model_batch, expert_execution, pad_execution


@torch.no_grad()
def direct_diffusion_samples(
    policy: Any,
    model_batch: Mapping[str, torch.Tensor],
    seeds: Sequence[int],
) -> torch.Tensor:
    """绕过 ``select_action`` 队列，按固定seed直接生成K个动作块。"""

    if not seeds:
        raise ValueError("seeds不能为空。")
    parameter = next(policy.parameters())
    device = parameter.device
    diffusion = policy.diffusion
    batch_size = int(model_batch["observation.state"].shape[0])
    n_obs_steps = int(policy.config.n_obs_steps)
    start = n_obs_steps - 1
    end = start + int(policy.config.n_action_steps)
    # 图像编码是扫描的主要开销，且同一个观测的K次采样条件完全相同。
    global_cond = diffusion._prepare_global_conditioning(dict(model_batch))
    samples: list[torch.Tensor] = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        if hasattr(diffusion, "conditional_sample_coupled"):
            arm, view = diffusion.conditional_sample_coupled(
                batch_size,
                global_cond=global_cond,
                generator=generator,
            )
            full_sample = torch.cat((arm, view), dim=-1)
        elif hasattr(diffusion, "arm_unet") and hasattr(diffusion, "view_unet"):
            arm = diffusion.conditional_sample(
                diffusion.arm_unet,
                diffusion.arm_noise_scheduler,
                diffusion.arm_action_dim,
                batch_size,
                global_cond=global_cond,
                generator=generator,
            )
            view = diffusion.conditional_sample(
                diffusion.view_unet,
                diffusion.view_noise_scheduler,
                diffusion.view_action_dim,
                batch_size,
                global_cond=global_cond,
                generator=generator,
            )
            full_sample = diffusion.combine_action_heads(arm, view)
        else:
            full_sample = diffusion.conditional_sample(
                batch_size,
                global_cond=global_cond,
                generator=generator,
            )
        if start < 0 or end > full_sample.shape[1]:
            raise ValueError(
                f"执行切片[{start}:{end}]超出生成horizon={full_sample.shape[1]}。"
            )
        samples.append(full_sample[:, start:end])
    return torch.stack(samples, dim=0)


def temporal_median_smooth(
    rows: list[dict[str, Any]],
    score_keys: Sequence[str],
    window: int,
) -> None:
    """按episode进行居中中值平滑，结果原地写入 ``*_smoothed``。"""

    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("smooth_window必须为正奇数。")
    radius = window // 2
    by_episode: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_episode.setdefault(int(row["hf_episode_index"]), []).append(index)
    for indices in by_episode.values():
        indices.sort(key=lambda idx: int(rows[idx]["frame_index"]))
        for score_key in score_keys:
            values = np.asarray([float(rows[idx][score_key]) for idx in indices])
            for local_index, row_index in enumerate(indices):
                lo = max(0, local_index - radius)
                hi = min(len(indices), local_index + radius + 1)
                finite = values[lo:hi][np.isfinite(values[lo:hi])]
                rows[row_index][f"{score_key}_smoothed"] = (
                    float(np.median(finite)) if finite.size else float("nan")
                )


def _merge_candidate_peaks(
    candidates: Sequence[dict[str, Any]],
    *,
    score_key: str,
    merge_gap: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda row: int(row["frame_index"]))
    regions: list[list[dict[str, Any]]] = [[ordered[0]]]
    for row in ordered[1:]:
        if int(row["frame_index"]) - int(regions[-1][-1]["frame_index"]) <= merge_gap:
            regions[-1].append(row)
        else:
            regions.append([row])
    return [
        max(region, key=lambda row: (float(row[score_key]), -int(row["frame_index"])))
        for region in regions
    ]


def _nms_rows(
    candidates: Sequence[dict[str, Any]],
    *,
    score_key: str,
    min_distance: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sorted(
        candidates,
        key=lambda item: (-float(item[score_key]), int(item["frame_index"])),
    ):
        frame = int(row["frame_index"])
        if all(abs(frame - int(other["frame_index"])) >= min_distance for other in selected):
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def select_anchors(
    rows: list[dict[str, Any]],
    *,
    threshold_quantile: float,
    merge_gap: int,
    nms_distance: int,
    arm_total_anchor_budget: int,
    view_total_anchor_budget: int,
    max_anchors_per_episode: int,
    random_exploration_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """在全数据集范围分配锚点预算，并用少量随机锚点保留探索。

    简单轨迹可以不分配任何锚点；困难轨迹可以获得多个锚点，但受到
    ``max_anchors_per_episode`` 的安全上限及episode内NMS约束。
    """

    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile必须位于[0,1]。")
    if not 0.0 <= random_exploration_ratio <= 1.0:
        raise ValueError("random_exploration_ratio必须位于[0,1]。")
    if arm_total_anchor_budget < 1 or view_total_anchor_budget < 1:
        raise ValueError("Arm/View全局锚点预算必须为正整数。")
    if max_anchors_per_episode < 1 or merge_gap < 0 or nms_distance < 1:
        raise ValueError(
            "max_anchors_per_episode和nms_distance须为正数、merge_gap须为非负数。"
        )

    all_selected: list[dict[str, Any]] = []
    selection_summary: dict[str, Any] = {"roles": {}}
    for role in ("arm", "view"):
        rng = random.Random(int(seed) + (0 if role == "arm" else 1_000_003))
        total_budget = int(
            arm_total_anchor_budget if role == "arm" else view_total_anchor_budget
        )
        score_key = (
            "arm_joint_score_smoothed"
            if role == "arm"
            else "view_score_smoothed"
        )
        eligible_key = f"{role}_eligible"
        eligible = [
            row
            for row in rows
            if bool(row[eligible_key])
            and bool(row.get("scorable", True))
            and math.isfinite(float(row[score_key]))
        ]
        if not eligible:
            raise ValueError(f"{role}没有可用于恢复数据生成的候选帧。")
        threshold = float(
            np.quantile(
                np.asarray([float(row[score_key]) for row in eligible]),
                threshold_quantile,
            )
        )
        episode_ids = sorted({int(row["hf_episode_index"]) for row in eligible})
        random_budget = int(round(total_budget * random_exploration_ratio))
        model_budget = total_budget - random_budget
        global_peaks: list[dict[str, Any]] = []
        for episode_id in episode_ids:
            episode_rows = [
                row for row in eligible if int(row["hf_episode_index"]) == episode_id
            ]
            above_threshold = [
                row for row in episode_rows if float(row[score_key]) >= threshold
            ]
            peaks = _merge_candidate_peaks(
                above_threshold, score_key=score_key, merge_gap=merge_gap
            )
            global_peaks.extend(
                _nms_rows(
                    peaks,
                    score_key=score_key,
                    min_distance=nms_distance,
                    limit=max_anchors_per_episode,
                )
            )

        chosen_model: list[dict[str, Any]] = []
        selected_by_episode: dict[int, list[dict[str, Any]]] = {
            episode_id: [] for episode_id in episode_ids
        }
        if model_budget > 0:
            for row in sorted(
                global_peaks,
                key=lambda item: (
                    -float(item[score_key]),
                    int(item["hf_episode_index"]),
                    int(item["frame_index"]),
                ),
            ):
                episode_id = int(row["hf_episode_index"])
                if len(selected_by_episode[episode_id]) >= max_anchors_per_episode:
                    continue
                chosen_model.append(row)
                selected_by_episode[episode_id].append(row)
                if len(chosen_model) >= model_budget:
                    break

        # 随机探索预算也在全数据集范围分配，不强制给每条episode补齐锚点。
        # 因此没有高风险峰的简单轨迹可以保持零扰动。
        random_pool = list(eligible)
        rng.shuffle(random_pool)
        chosen_random: list[dict[str, Any]] = []
        if random_budget > 0:
            for row in random_pool:
                episode_id = int(row["hf_episode_index"])
                episode_selected = selected_by_episode[episode_id]
                if len(episode_selected) >= max_anchors_per_episode:
                    continue
                frame = int(row["frame_index"])
                if any(
                    abs(frame - int(other["frame_index"])) < nms_distance
                    for other in episode_selected
                ):
                    continue
                chosen_random.append(row)
                episode_selected.append(row)
                if len(chosen_random) >= random_budget:
                    break

        role_selected: list[dict[str, Any]] = []
        for source, selected_rows in (
            ("model_risk", chosen_model),
            ("random_exploration", chosen_random),
        ):
            for row in selected_rows:
                record = dict(row)
                record["selection_source"] = source
                record["selection_role"] = role
                record["score_key"] = score_key
                record["score"] = float(row[score_key])
                if role == "arm":
                    left_score = float(row["left_arm_expected"])
                    right_score = float(row["right_arm_expected"])
                    side = "left" if left_score >= right_score else "right"
                    record["target_arm"] = side
                    record["model_risk_side"] = side
                else:
                    record["target_arm"] = None
                    record["model_risk_side"] = None
                role_selected.append(record)
        all_selected.extend(role_selected)
        selected_counts = {
            episode_id: len(selected_rows)
            for episode_id, selected_rows in selected_by_episode.items()
            if selected_rows
        }
        selection_summary["roles"][role] = {
            "threshold": threshold,
            "threshold_quantile": threshold_quantile,
            "eligible_frames": len(eligible),
            "total_anchor_budget": total_budget,
            "model_risk_budget": model_budget,
            "random_exploration_budget": random_budget,
            "max_anchors_per_episode": max_anchors_per_episode,
            "selected": len(role_selected),
            "model_risk_selected": sum(
                row["selection_source"] == "model_risk" for row in role_selected
            ),
            "random_exploration_selected": sum(
                row["selection_source"] == "random_exploration" for row in role_selected
            ),
            "unused_budget": total_budget - len(role_selected),
            "episodes_with_anchors": len(selected_counts),
            "episodes_without_anchors": len(episode_ids) - len(selected_counts),
            "selected_per_episode": {
                str(episode_id): count
                for episode_id, count in sorted(selected_counts.items())
            },
        }
    all_selected.sort(
        key=lambda row: (
            int(row["hf_episode_index"]),
            int(row["frame_index"]),
            str(row["selection_role"]),
        )
    )
    return all_selected, selection_summary


def resolve_pretrained_model_path(path: str | Path) -> Path:
    """解析并强制选择 EMA ``pretrained_model``，拒绝在线模型。"""

    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"checkpoint不存在: {candidate}")
    if candidate.name == "online_model":
        raise ValueError("风险扫描必须使用EMA pretrained_model，不能使用online_model。")
    if (candidate / "pretrained_model" / "config.json").is_file():
        candidate = candidate / "pretrained_model"
    if candidate.name != "pretrained_model" or not (candidate / "config.json").is_file():
        raise ValueError(
            "--checkpoint必须指向pretrained_model，或其直接父checkpoint目录。"
        )
    model_files = list(candidate.glob("*.safetensors")) + list(candidate.glob("*.bin"))
    if not model_files:
        raise FileNotFoundError(f"pretrained_model中没有模型权重: {candidate}")
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_numpy_actions(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return torch.stack(value).cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def load_and_verify_episode_mapping(
    dataset: Any,
    raw_dir: Path,
    action_key: str,
    *,
    atol: float,
) -> list[EpisodeMapping]:
    """从实际 LeRobot/raw 数据加载动作并执行严格映射验证。"""

    episodes_dir = raw_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"raw数据缺少episodes目录: {episodes_dir}")
    raw_items: list[dict[str, Any]] = []
    for episode_dir in episodes_dir.glob("episode_*"):
        arrays_path = episode_dir / "arrays.npz"
        if not episode_dir.is_dir() or not arrays_path.is_file():
            continue
        parse_raw_episode_name(episode_dir.name)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            if action_key not in arrays:
                raise KeyError(f"{arrays_path}缺少动作键{action_key!r}。")
            actions = np.asarray(arrays[action_key], dtype=np.float32)
            state_keys = (
                "observation_state",
                "obs__agent_pos",
                "obs__observation__state",
            )
            state_key = next((key for key in state_keys if key in arrays), None)
            if state_key is None:
                fallback = sorted(key for key in arrays.files if key.startswith("obs__"))
                state_key = fallback[0] if fallback else None
            if state_key is None:
                raise KeyError(f"{arrays_path}缺少converter可识别的状态数组。")
            frame_count = min(len(actions), len(arrays[state_key]))
            for optional in ("timestamp", "terminated", "truncated", "frame_index"):
                if optional in arrays:
                    frame_count = min(frame_count, len(arrays[optional]))
            actions = actions[:frame_count]
        raw_items.append({"name": episode_dir.name, "path": episode_dir, "actions": actions})

    episode_from = dataset.episode_data_index["from"].cpu().tolist()
    episode_to = dataset.episode_data_index["to"].cpu().tolist()
    hf_items: list[dict[str, Any]] = []
    for episode_index, (start, end) in enumerate(zip(episode_from, episode_to, strict=True)):
        values = dataset.hf_dataset.select_columns("action")[int(start) : int(end)]["action"]
        hf_items.append(
            {
                "episode_index": episode_index,
                "from": int(start),
                "to": int(end),
                "actions": _to_numpy_actions(values),
            }
        )
    return strict_episode_mapping(hf_items, raw_items, atol=atol)


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def score_dataset(
    policy: Any,
    dataset: Any,
    mappings: Sequence[EpisodeMapping],
    *,
    batch_size: int,
    num_workers: int,
    num_samples: int,
    seed: int,
    arm_required_future_frames: int,
    view_required_future_frames: int,
    min_frame_index: int,
    max_frames: int | None = None,
    max_episodes: int | None = None,
    frame_stride: int = 1,
) -> list[dict[str, Any]]:
    """顺序遍历全部专家起点并返回逐帧风险记录。"""

    device = next(policy.parameters()).device
    policy.eval()
    mapping_by_id = {item.hf_episode_index: item for item in mappings}
    groups = resolve_action_groups(
        int(policy.config.output_shapes["action"][0]), int(policy.arm_action_dim)
    )
    if frame_stride < 1:
        raise ValueError("frame_stride必须为正整数。")
    if min_frame_index < 0:
        raise ValueError("min_frame_index必须为非负整数。")
    selected_mappings = list(mappings)
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max_episodes必须为正整数或null。")
        selected_mappings = selected_mappings[:max_episodes]
    scan_indices = [
        mapping.hf_from + local_frame
        for mapping in selected_mappings
        for local_frame in range(
            min_frame_index,
            mapping.episode_length,
            frame_stride,
        )
    ]
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames必须为正整数或null。")
        scan_indices = scan_indices[:max_frames]
    if not scan_indices:
        raise ValueError("筛选后没有需要评分的帧。")
    scan_dataset = Subset(dataset, scan_indices)
    loader = DataLoader(
        scan_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    rows: list[dict[str, Any]] = []
    for batch_index, batch_cpu in enumerate(loader):
        batch = _move_batch_to_device(batch_cpu, device)
        model_batch, expert_execution, pad_execution = prepare_policy_batch(policy, batch)
        seeds = [int(seed) + batch_index * 100_000 + sample_index for sample_index in range(num_samples)]
        predictions = direct_diffusion_samples(policy, model_batch, seeds)
        metrics = score_normalized_predictions(
            predictions, expert_execution, pad_execution, groups
        )
        batch_count = expert_execution.shape[0]
        for batch_row in range(batch_count):
            episode_id = int(batch["episode_index"][batch_row].item())
            frame_index = int(batch["frame_index"][batch_row].item())
            global_index = int(batch["index"][batch_row].item())
            mapping = mapping_by_id[episode_id]
            remaining = mapping.episode_length - frame_index - 1
            available_from_anchor = mapping.episode_length - frame_index
            scorable = bool(metrics["scorable"][batch_row].detach().cpu().item())
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "source_raw_dir": str(Path(mapping.raw_episode_dir).parent.parent),
                "raw_episode_name": mapping.raw_episode_name,
                "source_episode": mapping.source_episode,
                "variant": mapping.variant,
                "hf_episode_index": episode_id,
                "frame_index": frame_index,
                "global_index": global_index,
                "episode_length": mapping.episode_length,
                "action_sha256": mapping.action_sha256,
                "remaining_future_frames": remaining,
                "available_frames_from_anchor": available_from_anchor,
                "scorable": scorable,
                "arm_eligible": scorable
                and frame_index >= min_frame_index
                and available_from_anchor >= arm_required_future_frames,
                "view_eligible": scorable
                and frame_index >= min_frame_index
                and available_from_anchor >= view_required_future_frames,
            }
            for role, required in (
                ("arm", arm_required_future_frames),
                ("view", view_required_future_frames),
            ):
                if not scorable:
                    reason = "execution_slice_all_padding"
                elif frame_index < min_frame_index:
                    reason = f"episode_start_guard(min_frame_index={min_frame_index})"
                elif available_from_anchor < required:
                    reason = (
                        "insufficient_future_frames"
                        f"(required_including_anchor={required},"
                        f"available={available_from_anchor})"
                    )
                else:
                    reason = None
                record[f"{role}_unavailable_reason"] = reason
            for key, tensor in metrics.items():
                if key == "scorable":
                    continue
                value = tensor[batch_row].detach().cpu().item()
                record[key] = int(value) if key == "valid_action_steps" else float(value)
            rows.append(record)
        logging.info("已评分 %d/%d 帧", len(rows), len(scan_dataset))
    return rows


def _build_delta_timestamps(policy: Any, dataset: Any) -> dict[str, list[float]]:
    fps = float(dataset.info["fps"])
    result: dict[str, list[float]] = {
        "action": [index / fps for index in range(int(policy.config.horizon))],
        "observation.state": [
            -index / fps
            for index in reversed(range(int(policy.config.n_obs_steps)))
        ],
    }
    for key in policy.expected_image_keys:
        result[key] = list(result["observation.state"])
    return result


def _load_policy(policy_name: str, checkpoint: Path, device: str) -> Any:
    from lerobot.common.policies.factory import get_policy_and_config_classes
    from lerobot.common.utils.utils import get_safe_torch_device
    from safetensors.torch import load_file

    policy_class, _ = get_policy_and_config_classes(policy_name)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # checkpoint 已完整保存视觉骨干参数。构造阶段禁止 torchvision 再下载
    # ImageNet 初始化权重，随后对本地 checkpoint 做严格加载；这样扫描器在
    # 无网络或全新缓存目录中也能复现训练后的模型。
    pretrained_backbone_weights = config.get("pretrained_backbone_weights")
    config["pretrained_backbone_weights"] = None
    policy = policy_class(config)
    safetensor_files = sorted(checkpoint.glob("*.safetensors"))
    binary_files = sorted(checkpoint.glob("*.bin"))
    if len(safetensor_files) == 1:
        state_dict = load_file(str(safetensor_files[0]), device="cpu")
    elif not safetensor_files and len(binary_files) == 1:
        state_dict = torch.load(
            binary_files[0],
            map_location="cpu",
            weights_only=True,
        )
    else:
        raise ValueError(
            "pretrained_model必须恰好包含一个.safetensors或.bin权重文件；"
            f"当前safetensors={len(safetensor_files)}, bin={len(binary_files)}。"
        )
    policy.load_state_dict(state_dict, strict=True)
    policy.config.pretrained_backbone_weights = pretrained_backbone_weights
    policy.to(get_safe_torch_device(device))
    policy.requires_grad_(False)
    policy.eval()
    return policy


def _load_local_dataset(
    dataset_dir: Path,
    delta_timestamps: dict[str, list[float]],
    video_backend: str | None,
) -> Any:
    # 复用训练入口经过项目验证的本地 LeRobot/HF 加载接口。
    from train.s1_pretrain.train.train_pretrain import load_local_lerobot_dataset

    return load_local_lerobot_dataset(
        dataset_dir,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """读取分组式YAML，并严格展开为argparse参数名。"""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"风险扫描YAML配置不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("风险扫描YAML顶层必须是配置分组对象。")

    unknown_sections = sorted(set(payload) - set(CONFIG_SECTIONS))
    if unknown_sections:
        raise ValueError(f"风险扫描YAML包含未知分组: {unknown_sections}")

    flattened: dict[str, Any] = {}
    for section, allowed_fields in CONFIG_SECTIONS.items():
        values = payload.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"风险扫描YAML分组{section!r}必须是对象。")
        unknown_fields = sorted(set(values) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"风险扫描YAML分组{section!r}包含未知字段: {unknown_fields}"
            )
        for key, value in values.items():
            if key in flattened:
                raise ValueError(f"风险扫描YAML字段{key!r}重复定义。")
            if key in PATH_CONFIG_FIELDS:
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ValueError(
                        f"风险扫描YAML路径字段{key!r}必须是非空字符串。"
                    )
                resolved_path = Path(value).expanduser()
                if not resolved_path.is_absolute():
                    resolved_path = ROOT_DIR / resolved_path
                flattened[key] = resolved_path.resolve()
            else:
                flattened[key] = value

    missing = sorted(PATH_CONFIG_FIELDS - set(flattened))
    if missing:
        raise ValueError(f"风险扫描YAML缺少必要路径字段: {missing}")
    return flattened


def build_arg_parser(
    config_defaults: Mapping[str, Any] | None = None,
) -> argparse.ArgumentParser:
    defaults = dict(config_defaults or {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="完整风险扫描YAML配置；显式命令行参数会覆盖YAML值。",
    )
    for argument_name in (
        "dataset_local_dir",
        "raw_dir",
        "checkpoint",
        "output_dir",
    ):
        parser.add_argument(
            f"--{argument_name.replace('_', '-')}",
            dest=argument_name,
            type=Path,
            default=defaults.get(argument_name),
            required=argument_name not in defaults,
        )
    parser.add_argument(
        "--policy-name",
        choices=("diffusion", "dual_head_diffusion", "coupled_dual_head_diffusion"),
        default=defaults.get("policy_name", "dual_head_diffusion"),
    )
    parser.add_argument("--action-key", default=defaults.get("action_key", "joint_action"))
    parser.add_argument("--device", default=defaults.get("device", "cuda"))
    parser.add_argument(
        "--video-backend", default=defaults.get("video_backend", "pyav")
    )
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 16))
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 0))
    parser.add_argument("--num-samples", type=int, default=defaults.get("num_samples", 16))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 1000))
    parser.add_argument(
        "--mapping-atol", type=float, default=defaults.get("mapping_atol", 0.0)
    )
    parser.add_argument(
        "--smooth-window", type=int, default=defaults.get("smooth_window", 5)
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=defaults.get("threshold_quantile", 0.9),
    )
    parser.add_argument("--merge-gap", type=int, default=defaults.get("merge_gap", 3))
    parser.add_argument(
        "--nms-distance", type=int, default=defaults.get("nms_distance", 20)
    )
    parser.add_argument(
        "--arm-total-anchor-budget",
        type=int,
        default=defaults.get("arm_total_anchor_budget", 300),
        help="Arm角色在全数据集范围内最多选择的锚点总数。",
    )
    parser.add_argument(
        "--view-total-anchor-budget",
        type=int,
        default=defaults.get("view_total_anchor_budget", 200),
        help="View角色在全数据集范围内最多选择的锚点总数。",
    )
    parser.add_argument(
        "--max-anchors-per-episode",
        type=int,
        default=defaults.get("max_anchors_per_episode", 6),
        help="单条episode单个角色的锚点安全上限；不设逐episode最低配额。",
    )
    parser.add_argument(
        "--random-exploration-ratio",
        type=float,
        default=defaults.get("random_exploration_ratio", 1.0 / 3.0),
    )
    parser.add_argument(
        "--min-frame-index",
        type=int,
        default=defaults.get("min_frame_index", 16),
        help="避开episode开头的帧数，默认16。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=defaults.get("max_frames"),
        help="从第一个合法锚点开始仅扫描N帧，便于GPU冒烟；默认扫描全部。",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=defaults.get("max_episodes"),
        help="仅扫描按映射排序后的前N条episode，便于GPU冒烟。",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=defaults.get("frame_stride", 1),
        help="每隔多少帧评分一个起点，默认逐帧。",
    )
    parser.add_argument(
        "--arm-required-future-frames",
        type=int,
        default=defaults.get("arm_required_future_frames", 51),
        help="Arm恢复生成器从锚点到完整恢复后缀所需的未来帧预算。",
    )
    parser.add_argument(
        "--view-required-future-frames",
        type=int,
        default=defaults.get("view_required_future_frames", 66),
        help="View恢复生成器从锚点到完整恢复后缀所需的未来帧预算。",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """先解析YAML路径，再以YAML为默认值执行完整命令行解析。"""

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    preliminary, _ = bootstrap.parse_known_args(argv)
    defaults = (
        {}
        if preliminary.config is None
        else load_yaml_config(preliminary.config)
    )
    return build_arg_parser(defaults).parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.num_samples < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("num_samples/batch_size须为正数，num_workers不能为负。")
    checkpoint = resolve_pretrained_model_path(args.checkpoint)
    dataset_dir = args.dataset_local_dir.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    policy = _load_policy(args.policy_name, checkpoint, args.device)
    # 先读取info获得fps，再用完整时间轴重新加载；初次加载不解码任何帧。
    info = _read_json_if_exists(dataset_dir / "meta_data" / "info.json")
    if info is None or "fps" not in info:
        raise FileNotFoundError(f"本地HF数据缺少有效info.json: {dataset_dir}")
    observation_deltas = [
        -index / float(info["fps"])
        for index in reversed(range(int(policy.config.n_obs_steps)))
    ]
    delta_timestamps = {
        "action": [
            index / float(info["fps"])
            for index in range(int(policy.config.horizon))
        ],
        "observation.state": observation_deltas,
        **{key: list(observation_deltas) for key in policy.expected_image_keys},
    }
    dataset = _load_local_dataset(dataset_dir, delta_timestamps, args.video_backend)
    mappings = load_and_verify_episode_mapping(
        dataset, raw_dir, args.action_key, atol=args.mapping_atol
    )
    augmented = [item.raw_episode_name for item in mappings if item.variant >= 0]
    if augmented:
        preview = ", ".join(augmented[:5])
        raise ValueError(
            "风险扫描只接受纯专家数据，但检测到_aug_增强/恢复分支: "
            f"{preview}。请改用纯专家HF/raw数据。"
        )
    rows = score_dataset(
        policy,
        dataset,
        mappings,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=args.seed,
        arm_required_future_frames=args.arm_required_future_frames,
        view_required_future_frames=args.view_required_future_frames,
        min_frame_index=args.min_frame_index,
        max_frames=args.max_frames,
        max_episodes=args.max_episodes,
        frame_stride=args.frame_stride,
    )
    temporal_median_smooth(
        rows,
        ("arm_joint_score", "view_score"),
        args.smooth_window,
    )
    selected, selection_summary = select_anchors(
        rows,
        threshold_quantile=args.threshold_quantile,
        merge_gap=args.merge_gap,
        nms_distance=args.nms_distance,
        arm_total_anchor_budget=args.arm_total_anchor_budget,
        view_total_anchor_budget=args.view_total_anchor_budget,
        max_anchors_per_episode=args.max_anchors_per_episode,
        random_exploration_ratio=args.random_exploration_ratio,
        seed=args.seed,
    )

    checkpoint_config_path = checkpoint / "config.json"
    checkpoint_weights = sorted(checkpoint.glob("*.safetensors")) + sorted(
        checkpoint.glob("*.bin")
    )
    common_metadata = {
        "checkpoint_path": str(checkpoint),
        "dataset_path": str(dataset_dir),
    }
    for row in rows:
        row.update(common_metadata)
    for row in selected:
        row.update(common_metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("写入frame_scores.parquet需要pandas/pyarrow。") from error
    parquet_tmp = output_dir / "frame_scores.parquet.tmp"
    pd.DataFrame(rows).to_parquet(parquet_tmp, index=False)
    parquet_tmp.replace(output_dir / "frame_scores.parquet")
    jsonl = "".join(
        json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
        for row in selected
    )
    _atomic_write_text(output_dir / "selected_anchors.jsonl", jsonl)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": "expert_policy_normalized_action_risk",
        "yaml_config": (
            None if args.config is None else str(args.config.expanduser().resolve())
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "config_sha256": _file_sha256(checkpoint_config_path),
            "weight_files": [
                {"path": str(path), "sha256": _file_sha256(path)}
                for path in checkpoint_weights
            ],
            "is_ema_pretrained_model": True,
            "policy_name": args.policy_name,
            "config": _read_json_if_exists(checkpoint_config_path),
        },
        "dataset": {
            "hf_path": str(dataset_dir),
            "raw_path": str(raw_dir),
            "hf_info": info,
            "raw_metadata": _read_json_if_exists(raw_dir / "metadata.json"),
            "num_frames": len(rows),
            "num_episodes": len(mappings),
            "num_total_episodes": len(mappings),
            "num_scanned_episodes": len(
                {int(row["hf_episode_index"]) for row in rows}
            ),
            "episode_mapping": [asdict(item) for item in mappings],
            "mapping_atol": args.mapping_atol,
            "pure_expert_only": True,
        },
        "scoring": {
            "space": "model_normalized_action_space",
            "action_slice": {
                "start": int(policy.config.n_obs_steps) - 1,
                "length": int(policy.config.n_action_steps),
            },
            "padding_explicitly_masked": True,
            "num_diffusion_samples": args.num_samples,
            "seed": args.seed,
            "seed_scheme": "base + batch_index * 100000 + sample_index",
            "metric_names": list(METRIC_NAMES),
            "smooth_window": args.smooth_window,
            "device": str(next(policy.parameters()).device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_frames": args.max_frames,
            "max_episodes": args.max_episodes,
            "frame_stride": args.frame_stride,
            "interpretation": {
                "primary_selection_signal": (
                    "Monte Carlo expected action-chunk RMSE against expert actions"
                ),
                "dispersion_signal": "Monte Carlo diffusion sample spread",
                "role_scores_scaled_by_training_loss_weight": False,
                "is_calibrated_failure_probability": False,
                "requires_closed_loop_validation": True,
            },
        },
        "selection": {
            **selection_summary,
            "merge_gap": args.merge_gap,
            "nms_distance": args.nms_distance,
            "arm_total_anchor_budget": args.arm_total_anchor_budget,
            "view_total_anchor_budget": args.view_total_anchor_budget,
            "max_anchors_per_episode": args.max_anchors_per_episode,
            "random_exploration_ratio": args.random_exploration_ratio,
            "min_frame_index": args.min_frame_index,
            "arm_required_future_frames": args.arm_required_future_frames,
            "view_required_future_frames": args.view_required_future_frames,
            "excluded": {
                "arm": sum(not bool(row["arm_eligible"]) for row in rows),
                "view": sum(not bool(row["view_eligible"]) for row in rows),
            },
        },
        "outputs": {
            "frame_scores": "frame_scores.parquet",
            "selected_anchors": "selected_anchors.jsonl",
        },
    }
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n",
    )
    logging.info(
        "风险扫描完成: frames=%d, anchors=%d, output=%s",
        len(rows),
        len(selected),
        output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
