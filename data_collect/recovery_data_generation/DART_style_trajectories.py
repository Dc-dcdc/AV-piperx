#!/usr/bin/env python

"""生成统一动作噪声的 DART-style 全轨迹增强数据。

本脚本只实现 ``DART-style Unified-Action Noise Augmentation``，不宣称是
具有在线反馈专家的严格 DART。每个控制步都在左右操作臂与 View 臂的18个
连续关节目标上采样联合高斯噪声，两个夹爪维度始终保持专家动作不变。环境
执行带噪动作，但训练字段 ``joint_action`` 保存同期、未加噪的专家动作：

    observation_state = 执行动作前、已受历史噪声影响的真实状态
    joint_action = 同期记录的干净专家动作（HF训练监督）
    executed_joint_action = 实际推进MuJoCo的带噪动作（仅审计）

噪声协方差可从外部 ``.npz`` 读取，也可在配置中给出18维对角标准差。生成器
支持先按源episode名义长度规划增强帧预算，再根据成功轨迹的实际保存长度
动态补充候选。带噪rollout提前成功时立即保存有效专家前缀，不执行padding，
也不要求与源episode保持相同长度。
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_collect.recovery_data_generation import (  # noqa: E402
    view_recovery_trajectories as replay_common,
)
from data_collect.recovery_data_generation.trajectory_replay_common import (  # noqa: E402
    ACTION_DIM,
    ARM_JOINT_INDICES,
    GRIPPER_INDICES,
    VIEW_JOINT_INDICES,
    SourceEpisode,
    StereoVideoWriter,
    _configure_mujoco_runtime,
    _episode_is_complete,
    _fingerprint,
    _load_json,
    _make_environment,
    _read_agent_state,
    _restore_initial_state,
    _sha256_file,
    _validate_source_arrays,
    _write_json_atomic,
)


SCHEMA_VERSION = 1
EPISODE_NAMING = "source_episode_with_dart_style_full_rollout_variant_v1"
AUGMENTATION_TYPE = "dart_style_unified_action_noise"
SUPERVISOR_MODE = "recorded_time_aligned_reference"
DART_ACTIVE_INDICES = np.concatenate(
    (ARM_JOINT_INDICES, VIEW_JOINT_INDICES)
).astype(np.int64)
ACTIVE_DIM = int(DART_ACTIVE_INDICES.size)
ACTIVE_BLOCKS = (slice(0, 6), slice(6, 12), slice(12, 18))
SUPPORTED_ENVS = {
    "guided_vision/SewNeedle-3Arms-v0",
    "guided_vision/InsertCylinder-3Arms-v0",
    "guided_vision/InsertPeg-3Arms-v0",
    "guided_vision/HookPackage-3Arms-v0",
}


class DartStyleRolloutError(RuntimeError):
    """单条DART-style带噪rollout没有满足数据质量要求。"""


@dataclass(frozen=True)
class NoiseModel:
    covariance: np.ndarray
    sampling_factor: np.ndarray
    max_abs_rad: np.ndarray
    covariance_sha256: str
    source: str
    source_path: str | None
    structure: str
    minimum_eigenvalue: float
    maximum_eigenvalue: float


@dataclass(frozen=True)
class FrameBudgetPlan:
    mode: str
    target_augmented_frames: int
    planned_augmented_frames: int
    variant_counts: dict[int, int]
    candidate_variants: tuple[tuple[int, int], ...]
    exact_match: bool
    selection_seed: int


def _canonical_array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _nonnegative_int(value: Any, name: str) -> int:
    number = float(value)
    if not np.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"{name}必须为非负整数，当前为{value!r}。")
    return int(number)


def _positive_int(value: Any, name: str) -> int:
    number = _nonnegative_int(value, name)
    if number <= 0:
        raise ValueError(f"{name}必须为正整数，当前为{value!r}。")
    return number


def _resolve_optional_path(value: Any) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def _apply_covariance_structure(
    covariance: np.ndarray,
    structure: str,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (ACTIVE_DIM, ACTIVE_DIM):
        raise ValueError(
            f"协方差必须为({ACTIVE_DIM},{ACTIVE_DIM})，当前为{covariance.shape}。"
        )
    if structure == "full":
        return covariance.copy()
    if structure == "diagonal":
        return np.diag(np.diag(covariance))
    if structure == "block_diagonal":
        result = np.zeros_like(covariance)
        for block in ACTIVE_BLOCKS:
            result[block, block] = covariance[block, block]
        return result
    raise ValueError(
        "covariance.structure必须为full、block_diagonal或diagonal，"
        f"当前为{structure!r}。"
    )


def _load_noise_model(cfg: DictConfig) -> NoiseModel:
    covariance_path = _resolve_optional_path(cfg.covariance.path)
    if covariance_path is None:
        std = np.asarray(cfg.covariance.diagonal_std_rad, dtype=np.float64)
        if std.shape != (ACTIVE_DIM,) or not np.isfinite(std).all():
            raise ValueError(
                f"covariance.diagonal_std_rad必须是{ACTIVE_DIM}维有限数组。"
            )
        if np.any(std <= 0.0):
            raise ValueError("covariance.diagonal_std_rad必须全部大于0。")
        covariance = np.diag(np.square(std))
        covariance_source = "configured_diagonal_std"
        source_path = None
    else:
        if not covariance_path.is_file():
            raise FileNotFoundError(f"DART协方差文件不存在: {covariance_path}")
        with np.load(covariance_path, allow_pickle=False) as archive:
            covariance_key = str(cfg.covariance.key)
            if covariance_key not in archive:
                raise KeyError(
                    f"{covariance_path}缺少协方差字段{covariance_key!r}。"
                )
            covariance = np.asarray(archive[covariance_key], dtype=np.float64)
            if "active_indices" in archive:
                active_indices = np.asarray(
                    archive["active_indices"], dtype=np.int64
                )
                if not np.array_equal(active_indices, DART_ACTIVE_INDICES):
                    raise ValueError(
                        "协方差文件active_indices与本项目18维DART动作布局不一致: "
                        f"expected={DART_ACTIVE_INDICES.tolist()}, "
                        f"actual={active_indices.tolist()}"
                    )
        covariance_source = "precomputed_policy_residual_covariance"
        source_path = str(covariance_path)

    if covariance.shape != (ACTIVE_DIM, ACTIVE_DIM):
        raise ValueError(
            f"原始协方差必须为({ACTIVE_DIM},{ACTIVE_DIM})，"
            f"当前为{covariance.shape}。"
        )
    if not np.isfinite(covariance).all():
        raise ValueError("协方差包含NaN或Inf。")
    covariance = 0.5 * (covariance + covariance.T)
    covariance = _apply_covariance_structure(
        covariance, str(cfg.covariance.structure)
    )

    shrinkage = float(cfg.covariance.shrinkage)
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage < 1.0:
        raise ValueError("covariance.shrinkage必须位于[0,1)。")
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(
        np.diag(covariance)
    )

    global_scale = float(cfg.covariance.global_scale)
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        raise ValueError("covariance.global_scale必须为有限正数。")
    covariance *= global_scale**2

    eigenvalue_floor = float(cfg.covariance.eigenvalue_floor)
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor < 0.0:
        raise ValueError("covariance.eigenvalue_floor必须为有限非负数。")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues.min()) < -1e-8 * scale:
        raise ValueError(
            "输入协方差不是半正定矩阵: "
            f"minimum_eigenvalue={float(eigenvalues.min()):.6g}"
        )
    eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
    covariance = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    sampling_factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    max_abs = np.asarray(cfg.noise.max_abs_rad, dtype=np.float64)
    if max_abs.shape != (ACTIVE_DIM,) or not np.isfinite(max_abs).all():
        raise ValueError(f"noise.max_abs_rad必须是{ACTIVE_DIM}维有限数组。")
    if np.any(max_abs <= 0.0):
        raise ValueError("noise.max_abs_rad必须全部大于0。")

    return NoiseModel(
        covariance=covariance,
        sampling_factor=sampling_factor,
        max_abs_rad=max_abs,
        covariance_sha256=_canonical_array_sha256(covariance),
        source=covariance_source,
        source_path=source_path,
        structure=str(cfg.covariance.structure),
        minimum_eigenvalue=float(eigenvalues.min()),
        maximum_eigenvalue=float(eigenvalues.max()),
    )


def _frame_rng(
    *,
    seed: int,
    source_episode: int,
    variant_index: int,
    rollout_attempt: int,
    frame_index: int,
) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(source_episode),
            int(variant_index),
            int(rollout_attempt),
            int(frame_index),
        ]
    )
    return np.random.default_rng(sequence)


def _sample_truncated_noise(
    *,
    model: NoiseModel,
    feasible_lower: np.ndarray,
    feasible_upper: np.ndarray,
    rng: np.random.Generator,
    max_sampling_attempts: int,
) -> tuple[np.ndarray, int]:
    lower = np.maximum(
        np.asarray(feasible_lower, dtype=np.float64), -model.max_abs_rad
    )
    upper = np.minimum(
        np.asarray(feasible_upper, dtype=np.float64), model.max_abs_rad
    )
    if lower.shape != (ACTIVE_DIM,) or upper.shape != (ACTIVE_DIM,):
        raise ValueError("DART噪声可行上下界必须为18维。")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("DART噪声可行上下界包含NaN或Inf。")
    if np.any(lower > upper):
        bad = np.flatnonzero(lower > upper).tolist()
        raise DartStyleRolloutError(
            f"当前专家动作在DART安全关节范围之外，无法采样噪声，维度={bad}。"
        )
    for attempt in range(1, int(max_sampling_attempts) + 1):
        sample = model.sampling_factor @ rng.standard_normal(ACTIVE_DIM)
        if np.all(sample >= lower) and np.all(sample <= upper):
            return sample.astype(np.float64), attempt
    raise DartStyleRolloutError(
        "截断联合高斯在最大尝试次数内没有采到可行动作: "
        f"attempts={max_sampling_attempts}"
    )


def _continuous_control_ranges(env_obj) -> np.ndarray:
    def intersection(joints, actuators) -> np.ndarray:
        joint = np.asarray(
            env_obj._physics.bind(joints).range, dtype=np.float64
        )
        actuator = np.asarray(
            env_obj._physics.bind(actuators).ctrlrange, dtype=np.float64
        )
        return np.stack(
            (
                np.maximum(joint[:, 0], actuator[:, 0]),
                np.minimum(joint[:, 1], actuator[:, 1]),
            ),
            axis=1,
        )

    ranges = np.concatenate(
        (
            intersection(env_obj._left_joints[:6], env_obj._left_actuators[:6]),
            intersection(env_obj._right_joints[:6], env_obj._right_actuators[:6]),
            intersection(env_obj._middle_joints, env_obj._middle_actuators),
        ),
        axis=0,
    )
    if ranges.shape != (ACTIVE_DIM, 2):
        raise RuntimeError(
            f"连续关节控制范围应为({ACTIVE_DIM},2)，当前为{ranges.shape}。"
        )
    if not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise RuntimeError("连续关节与执行器范围没有形成有效交集。")
    return ranges


def _build_noisy_action(
    *,
    clean_action: np.ndarray,
    model: NoiseModel,
    control_ranges: np.ndarray,
    joint_limit_margin_rad: float,
    rng: np.random.Generator,
    max_sampling_attempts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    clean = np.asarray(clean_action, dtype=np.float64)
    if clean.shape != (ACTION_DIM,) or not np.isfinite(clean).all():
        raise ValueError("干净专家动作必须是20维有限数组。")
    margin = float(joint_limit_margin_rad)
    safe_lower = control_ranges[:, 0] + margin
    safe_upper = control_ranges[:, 1] - margin
    if np.any(safe_lower >= safe_upper):
        raise ValueError("joint_limit_margin_rad使连续关节安全范围为空。")
    clean_active = clean[DART_ACTIVE_INDICES]
    active_noise, sampling_attempts = _sample_truncated_noise(
        model=model,
        feasible_lower=safe_lower - clean_active,
        feasible_upper=safe_upper - clean_active,
        rng=rng,
        max_sampling_attempts=max_sampling_attempts,
    )
    requested_noise = np.zeros(ACTION_DIM, dtype=np.float64)
    requested_noise[DART_ACTIVE_INDICES] = active_noise
    requested_noise[GRIPPER_INDICES] = 0.0
    requested_action = clean + requested_noise
    executed_action = requested_action.copy()
    executed_action[DART_ACTIVE_INDICES] = np.clip(
        requested_action[DART_ACTIVE_INDICES],
        control_ranges[:, 0],
        control_ranges[:, 1],
    )
    # 夹爪必须逐位等于专家动作；不能因统一clip静默改变抓取状态。
    executed_action[GRIPPER_INDICES] = clean[GRIPPER_INDICES]
    applied_noise = executed_action - clean
    clipped = np.abs(executed_action - requested_action) > 1e-12
    if np.any(requested_noise[GRIPPER_INDICES] != 0.0) or np.any(
        applied_noise[GRIPPER_INDICES] != 0.0
    ):
        raise RuntimeError("DART-style实现错误：夹爪维度出现非零噪声。")
    return (
        executed_action,
        requested_noise,
        applied_noise,
        clipped,
        sampling_attempts,
    )


def _plan_frame_budget(
    *,
    source_lengths: dict[int, int],
    mode: str,
    target_augmented_frames: int | None,
    variants_per_source: int,
    max_variants_per_source: int,
    exact_match: bool,
    selection_seed: int,
) -> FrameBudgetPlan:
    if not source_lengths:
        raise ValueError("没有可用于规划DART帧预算的源轨迹。")
    if mode == "variants_per_source":
        count = _positive_int(
            variants_per_source,
            "augmentation_budget.variants_per_source",
        )
        variant_counts = {int(source): count for source in source_lengths}
        planned = sum(
            source_lengths[source] * count for source in source_lengths
        )
        return FrameBudgetPlan(
            mode=mode,
            target_augmented_frames=int(planned),
            planned_augmented_frames=int(planned),
            variant_counts=variant_counts,
            candidate_variants=tuple(
                (int(source), int(variant_index))
                for source in sorted(source_lengths)
                for variant_index in range(count)
            ),
            exact_match=True,
            selection_seed=int(selection_seed),
        )
    if mode != "target_augmented_frames":
        raise ValueError(
            "augmentation_budget.mode必须为variants_per_source或"
            f"target_augmented_frames，当前为{mode!r}。"
        )
    if target_augmented_frames is None:
        raise ValueError(
            "target_augmented_frames模式必须设置"
            "augmentation_budget.target_augmented_frames。"
        )
    target = _positive_int(
        target_augmented_frames,
        "augmentation_budget.target_augmented_frames",
    )
    max_variants = _positive_int(
        max_variants_per_source,
        "augmentation_budget.max_variants_per_source",
    )

    items = [
        (int(source), int(length))
        for source, length in sorted(source_lengths.items())
        for _ in range(max_variants)
    ]
    rng = np.random.default_rng(int(selection_seed))
    order = rng.permutation(len(items)).tolist()
    ordered_items = [items[index] for index in order]

    reachable = np.zeros(target + 1, dtype=np.bool_)
    parent_sum = np.full(target + 1, -1, dtype=np.int64)
    parent_item = np.full(target + 1, -1, dtype=np.int64)
    reachable[0] = True
    for item_index, (_, length) in enumerate(ordered_items):
        if length > target:
            continue
        previous_reachable = np.flatnonzero(reachable[: target - length + 1])
        for previous in previous_reachable[::-1]:
            total = int(previous + length)
            if not reachable[total]:
                reachable[total] = True
                parent_sum[total] = int(previous)
                parent_item[total] = int(item_index)

    if reachable[target]:
        selected_total = target
    else:
        if bool(exact_match):
            closest = int(np.flatnonzero(reachable)[-1])
            raise ValueError(
                "完整DART episode无法精确组成目标增强帧数；"
                f"target={target}, closest_not_exceeding={closest}。"
                "请提高max_variants_per_source或改用训练Sampler清单。"
            )
        selected_total = int(np.flatnonzero(reachable)[-1])
        if selected_total <= 0:
            raise ValueError("没有完整DART episode能够装入当前目标帧预算。")

    variant_counts = {int(source): 0 for source in source_lengths}
    cursor = selected_total
    while cursor > 0:
        item_index = int(parent_item[cursor])
        previous = int(parent_sum[cursor])
        if item_index < 0 or previous < 0:
            raise RuntimeError("DART帧预算回溯链损坏。")
        source, _ = ordered_items[item_index]
        variant_counts[source] += 1
        cursor = previous
    variant_counts = {
        source: count for source, count in variant_counts.items() if count > 0
    }
    planned = sum(
        source_lengths[source] * count
        for source, count in variant_counts.items()
    )
    if planned != selected_total:
        raise RuntimeError("DART帧预算规划结果与回溯总和不一致。")

    # 初始候选仍优先使用名义长度恰好组成目标帧数的子集；若轨迹提前
    # 成功、部分候选失败或实际保存帧数不足，则继续使用剩余候选动态补齐。
    initial_candidates = [
        (int(source), int(variant_index))
        for source, count in sorted(variant_counts.items())
        for variant_index in range(count)
    ]
    fallback_candidates = [
        (int(source), int(variant_index))
        for source in sorted(source_lengths)
        for variant_index in range(
            int(variant_counts.get(source, 0)), max_variants
        )
    ]
    candidate_rng = np.random.default_rng(
        np.random.SeedSequence([int(selection_seed), 1])
    )
    if initial_candidates:
        initial_candidates = [
            initial_candidates[index]
            for index in candidate_rng.permutation(len(initial_candidates))
        ]
    if fallback_candidates:
        fallback_candidates = [
            fallback_candidates[index]
            for index in candidate_rng.permutation(len(fallback_candidates))
        ]
    return FrameBudgetPlan(
        mode=mode,
        target_augmented_frames=target,
        planned_augmented_frames=planned,
        variant_counts=variant_counts,
        candidate_variants=tuple(initial_candidates + fallback_candidates),
        exact_match=planned == target,
        selection_seed=int(selection_seed),
    )


def _episode_dir(
    output_run_dir: Path, source_episode: int, variant_index: int
) -> Path:
    return replay_common._episode_dir(
        output_run_dir, source_episode, variant_index
    )


def _base_episode_info(
    *,
    source: SourceEpisode,
    variant_index: int,
    output_run_dir: Path,
    cameras: tuple[str, ...],
    cfg: DictConfig,
    fingerprint: str,
    steps: int,
) -> dict[str, Any]:
    episode_name = replay_common._output_episode_name(
        source.episode_number, variant_index
    )
    return {
        "episode": int(source.episode_number),
        "episode_name": episode_name,
        "episode_naming": EPISODE_NAMING,
        "success": True,
        "steps": int(steps),
        "fps": int(cfg.fps),
        "path": str(
            _episode_dir(
                output_run_dir, source.episode_number, variant_index
            ).relative_to(output_run_dir)
        ),
        "save_rgb": True,
        "save_videos": True,
        "save_depth": False,
        "save_pose_action": False,
        "save_reward_debug": False,
        "observation_npz_keys": {"agent_pos": "observation_state"},
        "depth_npz_keys": {},
        "video_paths": {
            f"pixels.{camera}": f"videos/{camera}.mp4" for camera in cameras
        },
        "source_episode": int(source.episode_number),
        "source_path": str(source.directory),
        "variant_index": int(variant_index),
        "is_augmented": variant_index >= 0,
        "generator": Path(__file__).name,
        "generation_schema_version": SCHEMA_VERSION,
        "generation_fingerprint": fingerprint,
        "observation_action_alignment": "pre_action",
    }


def _validate_completed_identity(
    directory: Path,
    *,
    source_episode: int,
    variant_index: int,
    fingerprint: str,
) -> dict[str, Any]:
    info = _load_json(directory / "info.json")
    expected = {
        "source_episode": int(source_episode),
        "variant_index": int(variant_index),
        "episode_name": replay_common._output_episode_name(
            source_episode, variant_index
        ),
        "generation_fingerprint": fingerprint,
    }
    mismatches = {
        key: {"expected": value, "actual": info.get(key)}
        for key, value in expected.items()
        if info.get(key) != value
    }
    if variant_index >= 0 and info.get("augmentation_type") != AUGMENTATION_TYPE:
        mismatches["augmentation_type"] = {
            "expected": AUGMENTATION_TYPE,
            "actual": info.get("augmentation_type"),
        }
    if mismatches:
        raise RuntimeError(
            f"已完成episode身份与当前DART配置不一致: {directory}, {mismatches}"
        )
    return info


def _save_original_episode(
    *,
    tmp_dir: Path,
    final_dir: Path,
    source: SourceEpisode,
    source_arrays: dict[str, np.ndarray],
    replay_errors: dict[str, float],
    final_reward: float,
    output_run_dir: Path,
    cameras: tuple[str, ...],
    cfg: DictConfig,
    fingerprint: str,
) -> dict[str, Any]:
    np.savez_compressed(
        tmp_dir / "arrays.npz",
        **replay_common._filtered_original_arrays(source_arrays),
    )
    info = _base_episode_info(
        source=source,
        variant_index=-1,
        output_run_dir=output_run_dir,
        cameras=cameras,
        cfg=cfg,
        fingerprint=fingerprint,
        steps=len(source_arrays["joint_action"]),
    )
    info.update(
        {
            "is_augmented": False,
            "source_replay_success": True,
            "final_task_success": True,
            "final_reward": float(final_reward),
            "max_replay_state_abs_error": float(replay_errors["state"]),
            "max_replay_arm_joint_abs_error": float(
                replay_errors["arm_joint"]
            ),
            "max_replay_gripper_abs_error": float(
                replay_errors["gripper"]
            ),
            "max_replay_view_joint_abs_error": float(
                replay_errors["view_joint"]
            ),
            "video_replay_rendered": True,
            "video_replay_cameras": list(cameras),
            "final_info": {
                "is_success": True,
                "source_episode": int(source.episode_number),
                "variant_index": -1,
                "pre_action_rerendered": True,
            },
        }
    )
    _write_json_atomic(tmp_dir / "info.json", info)
    tmp_dir.rename(final_dir)
    return info


def _generate_dart_style_variant(
    *,
    env_obj,
    source: SourceEpisode,
    source_arrays: dict[str, np.ndarray],
    variant_index: int,
    rollout_attempt: int,
    noise_model: NoiseModel,
    control_ranges: np.ndarray,
    output_run_dir: Path,
    cameras: tuple[str, ...],
    cfg: DictConfig,
    fingerprint: str,
) -> dict[str, Any]:
    final_dir = _episode_dir(
        output_run_dir, source.episode_number, variant_index
    )
    tmp_dir = replay_common._prepare_tmp_directory(final_dir)
    try:
        _restore_initial_state(env_obj, source_arrays)
        replay_common._sync_reward_reference_after_initial_restore(env_obj)
        source_actions = np.asarray(
            source_arrays["joint_action"], dtype=np.float64
        )
        source_states = np.asarray(
            source_arrays["observation_state"], dtype=np.float64
        )
        if source_actions.shape != source_states.shape:
            raise ValueError("源专家动作和状态必须同形。")
        initial_arrays = replay_common._capture_episode_initial_arrays(env_obj)

        recorded_states: list[np.ndarray] = []
        clean_actions: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []
        requested_noises: list[np.ndarray] = []
        applied_noises: list[np.ndarray] = []
        clipped_masks: list[np.ndarray] = []
        state_errors: list[np.ndarray] = []
        source_indices: list[int] = []
        sampling_attempt_counts: list[int] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []
        last_reward = 0.0
        early_success = False

        with StereoVideoWriter(tmp_dir / "videos", cameras, cfg) as writer:
            for frame_index, clean_action in enumerate(source_actions):
                actual_state = _read_agent_state(env_obj)
                source_state = source_states[frame_index]
                rng = _frame_rng(
                    seed=int(cfg.noise.seed),
                    source_episode=source.episode_number,
                    variant_index=variant_index,
                    rollout_attempt=rollout_attempt,
                    frame_index=frame_index,
                )
                (
                    executed_action,
                    requested_noise,
                    applied_noise,
                    clipped,
                    sampling_attempts,
                ) = _build_noisy_action(
                    clean_action=clean_action,
                    model=noise_model,
                    control_ranges=control_ranges,
                    joint_limit_margin_rad=float(
                        cfg.noise.joint_limit_margin_rad
                    ),
                    rng=rng,
                    max_sampling_attempts=int(
                        cfg.noise.max_sampling_attempts
                    ),
                )

                replay_common._render_stereo(env_obj, writer, cameras, cfg)
                recorded_states.append(actual_state.astype(np.float32))
                clean_actions.append(clean_action.astype(np.float32))
                executed_actions.append(executed_action.astype(np.float32))
                requested_noises.append(requested_noise.astype(np.float32))
                applied_noises.append(applied_noise.astype(np.float32))
                clipped_masks.append(clipped.astype(np.bool_))
                state_errors.append(
                    (actual_state - source_state).astype(np.float32)
                )
                source_indices.append(int(frame_index))
                sampling_attempt_counts.append(int(sampling_attempts))

                last_reward, terminated, truncated = (
                    replay_common._step_without_render(env_obj, executed_action)
                )
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                if truncated:
                    raise DartStyleRolloutError(
                        f"DART rollout在frame={frame_index}被环境上限截断。"
                    )
                if terminated and not bool(
                    getattr(env_obj, "is_success", False)
                ):
                    raise DartStyleRolloutError(
                        f"DART rollout在frame={frame_index}提前失败终止。"
                    )
                if terminated:
                    early_success = frame_index != len(source_actions) - 1
                    break

        if bool(cfg.rollout.require_final_success) and not bool(
            getattr(env_obj, "is_success", False)
        ):
            raise DartStyleRolloutError("带噪专家动作前缀结束后任务未成功。")

        clean_array = np.asarray(clean_actions, dtype=np.float32)
        saved_steps = int(len(clean_array))
        if saved_steps <= 0 or saved_steps > len(source_actions):
            raise RuntimeError(
                "DART正式保存轨迹长度必须是源专家轨迹的非空前缀。"
            )
        source_action_array = np.asarray(
            source_actions[:saved_steps], dtype=np.float32
        )
        if not np.array_equal(clean_array, source_action_array):
            raise RuntimeError(
                "DART训练标签joint_action不再逐位等于源专家动作前缀。"
            )
        requested_array = np.asarray(requested_noises, dtype=np.float32)
        applied_array = np.asarray(applied_noises, dtype=np.float32)
        if np.any(requested_array[:, GRIPPER_INDICES] != 0.0) or np.any(
            applied_array[:, GRIPPER_INDICES] != 0.0
        ):
            raise RuntimeError("DART输出审计发现夹爪噪声非零。")

        arrays = {
            "joint_action": clean_array,
            "executed_joint_action": np.asarray(
                executed_actions, dtype=np.float32
            ),
            "injected_action_noise_requested": requested_array,
            "injected_action_noise_applied": applied_array,
            "action_clipped": np.asarray(clipped_masks, dtype=np.bool_),
            "observation_state": np.asarray(
                recorded_states, dtype=np.float32
            ),
            "dart_reference_state": source_states[:saved_steps].astype(
                np.float32
            ),
            "dart_state_error": np.asarray(state_errors, dtype=np.float32),
            "source_frame_index": np.asarray(source_indices, dtype=np.int64),
            "noise_sampling_attempts": np.asarray(
                sampling_attempt_counts, dtype=np.int32
            ),
            "timestamp": np.arange(len(clean_actions), dtype=np.float32)
            / float(cfg.fps),
            "terminated": np.asarray(terminated_flags, dtype=np.bool_),
            "truncated": np.asarray(truncated_flags, dtype=np.bool_),
            **initial_arrays,
        }
        np.savez_compressed(tmp_dir / "arrays.npz", **arrays)

        active_requested = requested_array[:, DART_ACTIVE_INDICES]
        active_applied = applied_array[:, DART_ACTIVE_INDICES]
        state_error_array = np.asarray(state_errors, dtype=np.float32)
        clipped_array = np.asarray(clipped_masks, dtype=np.bool_)
        info = _base_episode_info(
            source=source,
            variant_index=variant_index,
            output_run_dir=output_run_dir,
            cameras=cameras,
            cfg=cfg,
            fingerprint=fingerprint,
            steps=len(clean_actions),
        )
        info.update(
            {
                "augmentation_type": AUGMENTATION_TYPE,
                "recovery_type": AUGMENTATION_TYPE,
                "strict_dart": False,
                "supervisor_mode": SUPERVISOR_MODE,
                "off_state_expert_query": False,
                "training_action_semantics": "clean_recorded_expert_action",
                "executed_action_semantics": "expert_plus_truncated_gaussian_noise",
                "noise_active_indices": DART_ACTIVE_INDICES.tolist(),
                "noise_active_dimensions": ACTIVE_DIM,
                "gripper_noise_enabled": False,
                "noise_resample_interval_steps": 1,
                "covariance_source": noise_model.source,
                "covariance_source_path": noise_model.source_path,
                "covariance_structure": noise_model.structure,
                "covariance_sha256": noise_model.covariance_sha256,
                "covariance_trace_rad2": float(
                    np.trace(noise_model.covariance)
                ),
                "covariance_minimum_eigenvalue": (
                    noise_model.minimum_eigenvalue
                ),
                "covariance_maximum_eigenvalue": (
                    noise_model.maximum_eigenvalue
                ),
                "requested_noise_active_rms_rad": float(
                    np.sqrt(np.mean(active_requested.astype(np.float64) ** 2))
                ),
                "applied_noise_active_rms_rad": float(
                    np.sqrt(np.mean(active_applied.astype(np.float64) ** 2))
                ),
                "requested_noise_active_max_abs_rad": float(
                    np.max(np.abs(active_requested))
                ),
                "applied_noise_active_max_abs_rad": float(
                    np.max(np.abs(active_applied))
                ),
                "action_clip_element_fraction": float(clipped_array.mean()),
                "action_clip_frame_fraction": float(
                    clipped_array.any(axis=1).mean()
                ),
                "noise_sampling_attempts_mean": float(
                    np.mean(sampling_attempt_counts)
                ),
                "noise_sampling_attempts_max": int(
                    np.max(sampling_attempt_counts)
                ),
                "state_deviation_rms": float(
                    np.sqrt(np.mean(state_error_array.astype(np.float64) ** 2))
                ),
                "state_deviation_max_abs": float(
                    np.max(np.abs(state_error_array))
                ),
                "source_recorded_first_frame": 0,
                "source_recorded_last_frame": len(clean_actions) - 1,
                "source_total_steps": int(len(source_actions)),
                "saved_source_prefix_steps": saved_steps,
                "source_completion_fraction": float(
                    saved_steps / len(source_actions)
                ),
                "early_success": bool(early_success),
                "full_source_length_preserved": bool(
                    saved_steps == len(source_actions)
                ),
                "source_replay_success": True,
                "final_task_success": bool(
                    getattr(env_obj, "is_success", False)
                ),
                "final_reward": float(last_reward),
                "noise_seed": int(cfg.noise.seed),
                "rollout_attempt": int(rollout_attempt),
                "final_info": {
                    "is_success": bool(getattr(env_obj, "is_success", False)),
                    "source_episode": int(source.episode_number),
                    "variant_index": int(variant_index),
                    "early_success": bool(early_success),
                    "full_source_length_preserved": bool(
                        saved_steps == len(source_actions)
                    ),
                },
            }
        )
        _write_json_atomic(tmp_dir / "info.json", info)
        tmp_dir.rename(final_dir)
        return info
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _generation_dependency_manifest(env_id: str) -> dict[str, str]:
    task_files = {
        "guided_vision/SewNeedle-3Arms-v0": "env/task/sew_needle_env.py",
        "guided_vision/InsertCylinder-3Arms-v0": "env/task/insert_cylinder_env.py",
        "guided_vision/InsertPeg-3Arms-v0": "env/task/insert_peg_env.py",
        "guided_vision/HookPackage-3Arms-v0": "env/task/hook_package_env.py",
    }
    dependencies = [
        Path(__file__).resolve(),
        Path(replay_common.__file__).resolve(),
        ROOT_DIR / "data_collect/recovery_data_generation/trajectory_replay_common.py",
        ROOT_DIR / "env/__init__.py",
        ROOT_DIR / "env/constants.py",
        ROOT_DIR / "env/task/sim_envs.py",
        ROOT_DIR / task_files[env_id],
    ]
    dependencies.extend(sorted((ROOT_DIR / "env/assets").rglob("*.xml")))
    result: dict[str, str] = {}
    for path in dependencies:
        if not path.is_file():
            raise FileNotFoundError(f"DART生成依赖文件不存在: {path}")
        result[str(path.relative_to(ROOT_DIR))] = _sha256_file(path)
    return result


def _semantic_config(
    *,
    cfg: DictConfig,
    input_run_dir: Path,
    env_id: str,
    noise_model: NoiseModel,
    budget_plan: FrameBudgetPlan,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_naming": EPISODE_NAMING,
        "augmentation_type": AUGMENTATION_TYPE,
        "input_run_dir": str(input_run_dir),
        "env_id": env_id,
        "include_original": bool(cfg.rollout.include_original),
        "source_episode_indices": (
            None
            if cfg.source_episode_indices is None
            else [int(value) for value in cfg.source_episode_indices]
        ),
        "max_source_episodes": (
            None
            if cfg.max_source_episodes is None
            else int(cfg.max_source_episodes)
        ),
        "covariance": {
            "source": noise_model.source,
            "source_path": noise_model.source_path,
            "structure": noise_model.structure,
            "sha256": noise_model.covariance_sha256,
            "trace_rad2": float(np.trace(noise_model.covariance)),
            "shrinkage": float(cfg.covariance.shrinkage),
            "eigenvalue_floor": float(cfg.covariance.eigenvalue_floor),
            "global_scale": float(cfg.covariance.global_scale),
        },
        "noise": OmegaConf.to_container(cfg.noise, resolve=True),
        "supervision": OmegaConf.to_container(cfg.supervision, resolve=True),
        "augmentation_budget": {
            **OmegaConf.to_container(cfg.augmentation_budget, resolve=True),
            "resolved_target_augmented_frames": (
                budget_plan.target_augmented_frames
            ),
            "resolved_planned_augmented_frames": (
                budget_plan.planned_augmented_frames
            ),
            "resolved_variant_counts": {
                str(key): value
                for key, value in sorted(budget_plan.variant_counts.items())
            },
            "resolved_candidate_variants": [
                [int(source), int(variant_index)]
                for source, variant_index in budget_plan.candidate_variants
            ],
        },
        "rollout": OmegaConf.to_container(cfg.rollout, resolve=True),
        "validation": OmegaConf.to_container(cfg.validation, resolve=True),
        "cameras": [str(value) for value in cfg.cameras],
        "render_height": int(cfg.render_height),
        "render_width": int(cfg.render_width),
        "fps": int(cfg.fps),
        "video": OmegaConf.to_container(cfg.video, resolve=True),
        "generation_dependencies": _generation_dependency_manifest(env_id),
    }


def _refresh_metadata(
    metadata: dict[str, Any],
    *,
    output_run_dir: Path,
    cameras: tuple[str, ...],
    fingerprint: str,
) -> None:
    completed: list[dict[str, Any]] = []
    episodes_dir = output_run_dir / "episodes"
    for directory in sorted(episodes_dir.glob("episode_*")):
        if not directory.is_dir() or directory.name.endswith(".tmp"):
            continue
        if not _episode_is_complete(directory, cameras):
            continue
        source_episode, variant_index = replay_common._parse_output_identity(
            directory.name
        )
        completed.append(
            _validate_completed_identity(
                directory,
                source_episode=source_episode,
                variant_index=variant_index,
                fingerprint=fingerprint,
            )
        )
    originals = [value for value in completed if not value["is_augmented"]]
    augmented = [value for value in completed if value["is_augmented"]]
    metadata["episodes"] = completed
    metadata["completed_original_episodes"] = len(originals)
    metadata["completed_augmented_episodes"] = len(augmented)
    metadata["completed_original_frames"] = int(
        sum(int(value["steps"]) for value in originals)
    )
    metadata["completed_augmented_frames"] = int(
        sum(int(value["steps"]) for value in augmented)
    )
    metadata["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_json_atomic(output_run_dir / "metadata.json", metadata)


def _load_or_create_metadata(
    *,
    output_run_dir: Path,
    input_run_dir: Path,
    source_metadata: dict[str, Any],
    semantic_config: dict[str, Any],
    fingerprint: str,
    cameras: tuple[str, ...],
    cfg: DictConfig,
) -> dict[str, Any]:
    metadata_path = output_run_dir / "metadata.json"
    if output_run_dir.exists() and not bool(cfg.rollout.resume):
        raise FileExistsError(
            f"DART输出目录已存在且rollout.resume=false: {output_run_dir}"
        )
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "episodes").mkdir(exist_ok=True)
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        if metadata.get("generation_fingerprint") != fingerprint:
            raise ValueError(
                "续生成配置与已有DART输出不一致，请更换output_run_dir: "
                f"existing={metadata.get('generation_fingerprint')!r}, "
                f"current={fingerprint!r}"
            )
    else:
        unexpected = [
            path
            for path in output_run_dir.iterdir()
            if path.name != "episodes"
        ]
        if unexpected or any((output_run_dir / "episodes").iterdir()):
            raise RuntimeError(
                f"DART输出目录非空但缺少metadata.json: {output_run_dir}"
            )
        metadata = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": str(output_run_dir),
            "source_run_dir": str(input_run_dir),
            "source_metadata_created_at": source_metadata.get("created_at"),
            "env_id": semantic_config["env_id"],
            "fps": int(cfg.fps),
            "record_cameras": list(cameras),
            "save_rgb": True,
            "save_videos": True,
            "save_depth": False,
            "save_pose_action": False,
            "video_save_mode": "pre_action_physical_dart_style_rollout",
            "success_semantics": "full_length_noisy_rollout_final_task_success",
            "supervisor_mode": SUPERVISOR_MODE,
            "strict_dart": False,
            "generator": Path(__file__).name,
            "generation_schema_version": SCHEMA_VERSION,
            "generation_fingerprint": fingerprint,
            "generation_config": semantic_config,
            "episode_naming": EPISODE_NAMING,
            "episodes": [],
            "failures": [],
        }
    metadata["generator"] = Path(__file__).name
    _refresh_metadata(
        metadata,
        output_run_dir=output_run_dir,
        cameras=cameras,
        fingerprint=fingerprint,
    )
    return metadata


def _record_failure(
    metadata: dict[str, Any],
    *,
    source_episode: int,
    variant_index: int | None,
    attempts: Sequence[dict[str, Any]],
) -> None:
    key = (int(source_episode), variant_index)
    retained = [
        value
        for value in metadata.setdefault("failures", [])
        if (value.get("source_episode"), value.get("variant_index")) != key
    ]
    retained.append(
        {
            "source_episode": int(source_episode),
            "variant_index": variant_index,
            "attempts": [dict(value) for value in attempts],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    metadata["failures"] = retained


def _clear_failure(
    metadata: dict[str, Any], source_episode: int, variant_index: int | None
) -> None:
    key = (int(source_episode), variant_index)
    metadata["failures"] = [
        value
        for value in metadata.setdefault("failures", [])
        if (value.get("source_episode"), value.get("variant_index")) != key
    ]


def _validate_config(
    cfg: DictConfig, source_metadata: dict[str, Any]
) -> tuple[str, tuple[str, ...]]:
    metadata_env_id = source_metadata.get("env_id")
    env_id = str(metadata_env_id) if cfg.env_id is None else str(cfg.env_id)
    if not env_id:
        raise ValueError("无法从DART配置或源metadata解析env_id。")
    if (
        cfg.env_id is not None
        and metadata_env_id is not None
        and str(metadata_env_id) != env_id
    ):
        raise ValueError(
            f"DART配置env_id={env_id!r}与源metadata="
            f"{metadata_env_id!r}不一致。"
        )
    if env_id not in SUPPORTED_ENVS:
        raise ValueError(f"DART-style生成器不支持env_id={env_id!r}。")
    cameras = tuple(str(value) for value in cfg.cameras)
    if cameras != ("zed_cam_left", "zed_cam_right"):
        raise ValueError(
            "DART-style生成器固定保存zed_cam_left和zed_cam_right，"
            f"当前为{cameras}。"
        )
    source_fps = source_metadata.get("fps")
    if source_fps is not None and int(source_fps) != int(cfg.fps):
        raise ValueError(
            f"DART配置fps={int(cfg.fps)}与源metadata fps="
            f"{int(source_fps)}不一致。"
        )
    for name, value in {
        "fps": cfg.fps,
        "render_height": cfg.render_height,
        "render_width": cfg.render_width,
        "noise.max_sampling_attempts": cfg.noise.max_sampling_attempts,
        "rollout.max_attempts_per_variant": (
            cfg.rollout.max_attempts_per_variant
        ),
    }.items():
        _positive_int(value, name)
    if _positive_int(
        cfg.noise.resample_interval_steps,
        "noise.resample_interval_steps",
    ) != 1:
        raise ValueError(
            "标准DART-style要求noise.resample_interval_steps=1；"
            "块保持噪声应另设消融实验。"
        )
    if bool(cfg.noise.perturb_grippers):
        raise ValueError("当前DART-style基线禁止扰动两个夹爪。")
    if str(cfg.supervision.mode) != SUPERVISOR_MODE:
        raise ValueError(
            f"当前仅支持supervision.mode={SUPERVISOR_MODE!r}。"
        )
    if str(cfg.supervision.action_key) != "joint_action":
        raise ValueError("DART-style训练标签字段必须为joint_action。")
    if not bool(cfg.rollout.require_final_success):
        raise ValueError(
            "第一版DART-style生成器要求rollout.require_final_success=true，"
            "避免保存不可逆失败后的错误同期标签。"
        )
    if str(cfg.rollout.failed_rollout_policy) != "discard":
        raise ValueError("当前仅支持failed_rollout_policy=discard。")
    margin = float(cfg.noise.joint_limit_margin_rad)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("noise.joint_limit_margin_rad必须为有限非负数。")
    for name, value in {
        "validation.max_arm_joint_abs_error": (
            cfg.validation.max_arm_joint_abs_error
        ),
        "validation.max_gripper_abs_error": (
            cfg.validation.max_gripper_abs_error
        ),
        "validation.max_view_joint_abs_error": (
            cfg.validation.max_view_joint_abs_error
        ),
    }.items():
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name}必须为有限正数。")
    if cfg.max_source_episodes is not None:
        _positive_int(cfg.max_source_episodes, "max_source_episodes")
    if cfg.source_episode_indices is not None:
        for value in cfg.source_episode_indices:
            _nonnegative_int(value, "source_episode_indices[]")
    _nonnegative_int(cfg.noise.seed, "noise.seed")
    _nonnegative_int(
        cfg.augmentation_budget.selection_seed,
        "augmentation_budget.selection_seed",
    )
    return env_id, cameras


def _attempt_dart_variant(
    *,
    env_obj,
    source: SourceEpisode,
    source_arrays: dict[str, np.ndarray],
    variant_index: int,
    noise_model: NoiseModel,
    control_ranges: np.ndarray,
    output_run_dir: Path,
    cameras: tuple[str, ...],
    cfg: DictConfig,
    fingerprint: str,
    metadata: dict[str, Any],
) -> bool:
    """生成一个候选分支；成功和失败都会立即同步元数据。"""

    final_dir = _episode_dir(
        output_run_dir, source.episode_number, variant_index
    )
    if _episode_is_complete(final_dir, cameras):
        _validate_completed_identity(
            final_dir,
            source_episode=source.episode_number,
            variant_index=variant_index,
            fingerprint=fingerprint,
        )
        _clear_failure(metadata, source.episode_number, variant_index)
        _refresh_metadata(
            metadata,
            output_run_dir=output_run_dir,
            cameras=cameras,
            fingerprint=fingerprint,
        )
        return True

    attempt_records: list[dict[str, Any]] = []
    for rollout_attempt in range(int(cfg.rollout.max_attempts_per_variant)):
        try:
            info = _generate_dart_style_variant(
                env_obj=env_obj,
                source=source,
                source_arrays=source_arrays,
                variant_index=variant_index,
                rollout_attempt=rollout_attempt,
                noise_model=noise_model,
                control_ranges=control_ranges,
                output_run_dir=output_run_dir,
                cameras=cameras,
                cfg=cfg,
                fingerprint=fingerprint,
            )
            _clear_failure(metadata, source.episode_number, variant_index)
            logging.info(
                "DART分支完成 source=%06d variant=%02d steps=%d/%d "
                "early_success=%s attempt=%d noise_rms=%.6f",
                source.episode_number,
                variant_index,
                int(info["steps"]),
                int(info["source_total_steps"]),
                bool(info["early_success"]),
                rollout_attempt,
                float(info["applied_noise_active_rms_rad"]),
            )
            _refresh_metadata(
                metadata,
                output_run_dir=output_run_dir,
                cameras=cameras,
                fingerprint=fingerprint,
            )
            return True
        except Exception as exc:
            attempt_records.append(
                {
                    "rollout_attempt": int(rollout_attempt),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            logging.warning(
                "DART分支失败 source=%06d variant=%02d attempt=%d/%d: %s",
                source.episode_number,
                variant_index,
                rollout_attempt + 1,
                int(cfg.rollout.max_attempts_per_variant),
                exc,
            )

    _record_failure(
        metadata,
        source_episode=source.episode_number,
        variant_index=variant_index,
        attempts=attempt_records,
    )
    _refresh_metadata(
        metadata,
        output_run_dir=output_run_dir,
        cameras=cameras,
        fingerprint=fingerprint,
    )
    if not bool(cfg.rollout.continue_on_error):
        raise RuntimeError(
            "DART分支重试耗尽: "
            f"source={source.episode_number}, variant={variant_index}"
        )
    return False


def generate_dart_style_run(cfg: DictConfig) -> None:
    input_run_dir = replay_common._resolve_path(cfg.input_run_dir)
    output_run_dir = replay_common._resolve_path(cfg.output_run_dir)
    if input_run_dir == output_run_dir:
        raise ValueError("input_run_dir与output_run_dir不能相同。")
    source_metadata_path = input_run_dir / "metadata.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(
            f"DART输入run缺少metadata.json: {source_metadata_path}"
        )
    source_metadata = _load_json(source_metadata_path)
    env_id, cameras = _validate_config(cfg, source_metadata)
    source_indices = (
        None
        if cfg.source_episode_indices is None
        else [int(value) for value in cfg.source_episode_indices]
    )
    sources, skipped_sources = replay_common._load_successful_sources(
        input_run_dir,
        source_indices,
        None
        if cfg.max_source_episodes is None
        else int(cfg.max_source_episodes),
    )

    source_arrays_by_episode: dict[int, dict[str, np.ndarray]] = {}
    source_lengths: dict[int, int] = {}
    for source in sources:
        arrays = _validate_source_arrays(source.directory / "arrays.npz")
        source_arrays_by_episode[source.episode_number] = arrays
        source_lengths[source.episode_number] = len(arrays["joint_action"])
    budget_plan = _plan_frame_budget(
        source_lengths=source_lengths,
        mode=str(cfg.augmentation_budget.mode),
        target_augmented_frames=(
            None
            if cfg.augmentation_budget.target_augmented_frames is None
            else int(cfg.augmentation_budget.target_augmented_frames)
        ),
        variants_per_source=int(
            cfg.augmentation_budget.variants_per_source
        ),
        max_variants_per_source=int(
            cfg.augmentation_budget.max_variants_per_source
        ),
        exact_match=bool(cfg.augmentation_budget.exact_match),
        selection_seed=int(cfg.augmentation_budget.selection_seed),
    )
    noise_model = _load_noise_model(cfg)
    semantic_config = _semantic_config(
        cfg=cfg,
        input_run_dir=input_run_dir,
        env_id=env_id,
        noise_model=noise_model,
        budget_plan=budget_plan,
    )
    fingerprint = _fingerprint(semantic_config)
    metadata = _load_or_create_metadata(
        output_run_dir=output_run_dir,
        input_run_dir=input_run_dir,
        source_metadata=source_metadata,
        semantic_config=semantic_config,
        fingerprint=fingerprint,
        cameras=cameras,
        cfg=cfg,
    )
    replay_common._update_source_manifest(metadata, sources)
    metadata["skipped_sources"] = skipped_sources
    metadata["planned_variant_counts"] = {
        str(key): value
        for key, value in sorted(budget_plan.variant_counts.items())
    }
    metadata["candidate_variants"] = [
        {
            "source_episode": int(source_episode),
            "variant_index": int(variant_index),
        }
        for source_episode, variant_index in budget_plan.candidate_variants
    ]
    metadata["frame_budget_completion_rule"] = (
        "actual_successful_frames_gte_target_without_padding"
        if budget_plan.mode == "target_augmented_frames"
        else "all_configured_variants_completed_without_padding"
    )
    metadata["target_augmented_frames"] = (
        budget_plan.target_augmented_frames
    )
    metadata["planned_augmented_frames"] = (
        budget_plan.planned_augmented_frames
    )
    metadata["initial_nominal_frame_subset_exact_match"] = (
        budget_plan.exact_match
    )
    _refresh_metadata(
        metadata,
        output_run_dir=output_run_dir,
        cameras=cameras,
        fingerprint=fingerprint,
    )

    logging.info(
        "开始生成DART-style数据: env=%s source=%s output=%s sources=%d "
        "selected_sources=%d initial_variants=%d candidate_variants=%d "
        "target_frames=%d planned_frames=%d covariance=%s structure=%s "
        "fingerprint=%s",
        env_id,
        input_run_dir,
        output_run_dir,
        len(sources),
        len(budget_plan.variant_counts),
        sum(budget_plan.variant_counts.values()),
        len(budget_plan.candidate_variants),
        budget_plan.target_augmented_frames,
        budget_plan.planned_augmented_frames,
        noise_model.covariance_sha256[:12],
        noise_model.structure,
        fingerprint,
    )

    env_obj = _make_environment(
        env_id,
        cameras,
        int(cfg.render_height),
        int(cfg.render_width),
    )
    try:
        env_obj.reset(seed=0)
        required_model_body_names = tuple(
            str(name)
            for name in getattr(env_obj, "replay_model_body_names", ())
        )
        # 带环境要求重新严格校验一次，避免旧源数据缺少随机body初态。
        for source in sources:
            source_arrays_by_episode[source.episode_number] = (
                _validate_source_arrays(
                    source.directory / "arrays.npz",
                    required_model_body_names=required_model_body_names,
                )
            )
        control_ranges = _continuous_control_ranges(env_obj)
        source_by_episode = {
            int(source.episode_number): source for source in sources
        }
        candidate_source_ids = {
            int(source_episode)
            for source_episode, _ in budget_plan.candidate_variants
        }
        nominally_ready_sources: set[int] = set()

        for source in sources:
            source_arrays = source_arrays_by_episode[source.episode_number]
            original_final_dir = _episode_dir(
                output_run_dir, source.episode_number, -1
            )
            original_tmp_dir: Path | None = None
            need_nominal_replay = bool(cfg.rollout.include_original) or (
                source.episode_number in candidate_source_ids
            )
            if not need_nominal_replay:
                continue
            try:
                if bool(cfg.rollout.include_original):
                    if _episode_is_complete(original_final_dir, cameras):
                        _validate_completed_identity(
                            original_final_dir,
                            source_episode=source.episode_number,
                            variant_index=-1,
                            fingerprint=fingerprint,
                        )
                    else:
                        original_tmp_dir = replay_common._prepare_tmp_directory(
                            original_final_dir
                        )
                _, replay_errors, final_reward = (
                    replay_common._replay_source_and_capture(
                        env_obj=env_obj,
                        source=source,
                        source_arrays=source_arrays,
                        event_frames=[],
                        original_tmp_dir=original_tmp_dir,
                        cameras=cameras,
                        cfg=cfg,
                    )
                )
                if original_tmp_dir is not None:
                    _save_original_episode(
                        tmp_dir=original_tmp_dir,
                        final_dir=original_final_dir,
                        source=source,
                        source_arrays=source_arrays,
                        replay_errors=replay_errors,
                        final_reward=final_reward,
                        output_run_dir=output_run_dir,
                        cameras=cameras,
                        cfg=cfg,
                        fingerprint=fingerprint,
                    )
                    original_tmp_dir = None
                nominally_ready_sources.add(int(source.episode_number))
                _clear_failure(metadata, source.episode_number, None)
                _refresh_metadata(
                    metadata,
                    output_run_dir=output_run_dir,
                    cameras=cameras,
                    fingerprint=fingerprint,
                )
            except Exception as exc:
                if original_tmp_dir is not None:
                    shutil.rmtree(original_tmp_dir, ignore_errors=True)
                _record_failure(
                    metadata,
                    source_episode=source.episode_number,
                    variant_index=None,
                    attempts=[{"error": f"{type(exc).__name__}: {exc}"}],
                )
                _refresh_metadata(
                    metadata,
                    output_run_dir=output_run_dir,
                    cameras=cameras,
                    fingerprint=fingerprint,
                )
                logging.exception(
                    "DART源episode=%06d名义重放失败。",
                    source.episode_number,
                )
                if not bool(cfg.rollout.continue_on_error):
                    raise
                continue

        # 增强分支与名义专家副本分阶段处理。这样达到目标帧数后可以停止
        # 生成增强分支，同时仍保证include_original=true时保存全部专家轨迹。
        attempted_candidates = 0
        for source_episode, variant_index in budget_plan.candidate_variants:
            if budget_plan.mode == "target_augmented_frames" and int(
                metadata["completed_augmented_frames"]
            ) >= int(budget_plan.target_augmented_frames):
                break
            source_episode = int(source_episode)
            if source_episode not in nominally_ready_sources:
                logging.warning(
                    "跳过未通过名义重放的DART候选 source=%06d variant=%02d",
                    source_episode,
                    variant_index,
                )
                continue
            attempted_candidates += 1
            source = source_by_episode[source_episode]
            _attempt_dart_variant(
                env_obj=env_obj,
                source=source,
                source_arrays=source_arrays_by_episode[source_episode],
                variant_index=int(variant_index),
                noise_model=noise_model,
                control_ranges=control_ranges,
                output_run_dir=output_run_dir,
                cameras=cameras,
                cfg=cfg,
                fingerprint=fingerprint,
                metadata=metadata,
            )
        metadata["candidate_variants_visited_this_run"] = int(
            attempted_candidates
        )
    finally:
        close = getattr(env_obj, "close", None)
        if callable(close):
            close()

    _refresh_metadata(
        metadata,
        output_run_dir=output_run_dir,
        cameras=cameras,
        fingerprint=fingerprint,
    )
    actual_frames = int(metadata["completed_augmented_frames"])
    planned_frames = int(budget_plan.planned_augmented_frames)
    if budget_plan.mode == "target_augmented_frames":
        budget_complete = actual_frames >= planned_frames
    else:
        budget_complete = int(metadata["completed_augmented_episodes"]) == len(
            budget_plan.candidate_variants
        )
    metadata["frame_budget_complete"] = bool(budget_complete)
    metadata["frame_budget_exact_match"] = actual_frames == planned_frames
    metadata["frame_budget_difference"] = actual_frames - planned_frames
    _write_json_atomic(output_run_dir / "metadata.json", metadata)
    logging.info(
        "DART-style生成结束: original_episodes=%d augmented_episodes=%d "
        "augmented_frames=%d/%d failures=%d",
        int(metadata["completed_original_episodes"]),
        int(metadata["completed_augmented_episodes"]),
        actual_frames,
        planned_frames,
        len(metadata.get("failures", [])),
    )
    if not budget_complete and not bool(cfg.rollout.allow_budget_shortfall):
        raise RuntimeError(
            "DART增强帧预算未完成；已有成功数据和失败记录均已保存，"
            f"actual={actual_frames}, planned={planned_frames}。"
        )


@hydra.main(
    version_base="1.2",
    config_path="../../configs/data_collect",
    config_name="DART_style_augmentation",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _configure_mujoco_runtime(cfg)
    generate_dart_style_run(cfg)


if __name__ == "__main__":
    main()
