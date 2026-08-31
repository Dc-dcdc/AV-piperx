#!/usr/bin/env python

"""生成操作臂关节扰动后的可收缩恢复轨迹。

每个增强 episode 只扰动事件附近真实关节运动更强的一只操作臂；两臂均
低于运动阈值时跳过该事件。事件帧定义为恢复锚点；扰动从锚点之前开始，
使用真实 MuJoCo 控制步沿专家时间轴平滑建立，并恰好在锚点完成。被选中
的六维 Arm 关节在专家目标上平滑叠加偏移。恢复阶段既可让全部角色沿
移动专家轨迹同步前进，也可暂停专家时间轴，使另一只 Arm、两个夹爪和
View 保持恢复锚点动作。设置阶段不写入训练数据。

从恢复锚点开始记录后，Arm 偏移沿五次最小加加速度曲线收缩到所选模式的
移动专家轨迹或静态锚点。
恢复曲线以物理建立阶段结束时实际到达的偏移为起点；采样偏移只作为控制
目标，实际偏移只要位于事件可行域内并满足最小扰动强度就会被接受。
random、specified_region和hybrid模式按每条源轨迹固定成功配额生成分支；
model_risk模式按风险清单中每条轨迹的实际锚点数生成，允许简单轨迹为零、
困难轨迹包含多个锚点。单个锚点重采样扰动仍全部失败时，将该锚点标记为
不可用，并在同一归一化任务域内按距离搜索邻近锚点。
输出只保存“恢复过程＋恢复后若干帧”，剩余专家后缀在后台执行并验证最终
任务成功。输出保持 Quest 原始格式，可直接交给
``hugging_face/convert_data_to_hf.py``。
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_collect.recovery_data_generation import (  # noqa: E402
    view_recovery_trajectories as recovery_common,
)
from data_collect.recovery_data_generation.trajectory_replay_common import (  # noqa: E402
    ACTION_DIM,
    SourceEpisode,
    StereoVideoWriter,
    VIEW_SLICE,
    _configure_mujoco_runtime,
    _episode_is_complete,
    _fingerprint,
    _load_json,
    _make_environment,
    _output_episode_name,
    _read_agent_state,
    _sha256_file,
    _validate_source_arrays,
    _write_json_atomic,
    build_static_anchor_reference_state,
    resolve_recovery_base_action,
    recovery_suffix_start_frame,
    resolve_recovery_timeline_step,
    resolve_trajectory_alignment_mode,
)


SCHEMA_VERSION = 8
EPISODE_NAMING = "source_episode_with_mode_dependent_quota_arm_recovery_branch_v6"
ARM_DIM = 6
LEFT_ARM_SLICE = slice(0, 6)
RIGHT_ARM_SLICE = slice(7, 13)
GRIPPER_INDICES = np.asarray([6, 13], dtype=np.int64)
ARM_SIDES = ("left", "right")


class ArmRecoveryBranchError(RuntimeError):
    """单个 Arm 恢复分支没有满足恢复或最终成功要求。"""


@dataclass(frozen=True)
class ArmMotionSelection:
    """恢复事件附近两只 Arm 的运动强度及主运动臂。"""

    side: str
    window_start_frame: int
    window_end_frame: int
    left_rms_velocity_rad_s: float
    right_rms_velocity_rad_s: float
    dominance_ratio: float


@dataclass(frozen=True)
class ArmRecoveryCandidate:
    """一个可回退的恢复锚点及其任务区域、主运动臂信息。"""

    event: recovery_common.RecoverySamplingEvent
    motion_selection: ArmMotionSelection
    domain_key: str
    min_interval_steps: int
    sampling_weight: float


def _arm_slice(side: str) -> slice:
    if side == "left":
        return LEFT_ARM_SLICE
    if side == "right":
        return RIGHT_ARM_SLICE
    raise ValueError(f"perturbed_arm 必须是 'left' 或 'right'，当前为{side!r}。")


def _other_arm_slice(side: str) -> slice:
    return RIGHT_ARM_SLICE if side == "left" else LEFT_ARM_SLICE


def _arm_recovery_action(
    expert_action: np.ndarray,
    arm_offset: np.ndarray,
    perturbed_arm: str,
) -> np.ndarray:
    """只给选中 Arm 的六维专家关节目标叠加偏移。"""

    expert_action = np.asarray(expert_action, dtype=np.float64)
    arm_offset = np.asarray(arm_offset, dtype=np.float64)
    if expert_action.shape != (ACTION_DIM,) or arm_offset.shape != (ARM_DIM,):
        raise ValueError("expert_action 必须为20维，arm_offset 必须为6维。")
    if not np.isfinite(expert_action).all() or not np.isfinite(arm_offset).all():
        raise ValueError("Arm 恢复动作不能包含 NaN 或 Inf。")
    action = expert_action.copy()
    action[_arm_slice(perturbed_arm)] += arm_offset
    return action


def _select_active_arm(
    *,
    states: np.ndarray,
    event_frame: int,
    fps: int,
    lookback_steps: int,
    lookahead_steps: int,
    min_rms_velocity_rad_s: float,
    requested_side: str | None = None,
) -> ArmMotionSelection | None:
    """依据事件附近真实关节状态速度选择主运动臂；两臂都静止则跳过。"""

    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != ACTION_DIM:
        raise ValueError(f"observation_state 必须为[T,20]，当前为{states.shape}。")
    start = max(0, int(event_frame) - int(lookback_steps))
    stop = min(len(states), int(event_frame) + int(lookahead_steps) + 1)
    if stop - start < 2:
        return None
    velocities = np.diff(states[start:stop], axis=0) * float(fps)
    left_score = float(np.sqrt(np.mean(velocities[:, LEFT_ARM_SLICE] ** 2)))
    right_score = float(np.sqrt(np.mean(velocities[:, RIGHT_ARM_SLICE] ** 2)))
    if requested_side is not None and requested_side not in ARM_SIDES:
        raise ValueError("requested_side必须是left、right或None。")
    side = (
        requested_side
        if requested_side is not None
        else ("left" if left_score > right_score else "right")
    )
    selected_score = left_score if side == "left" else right_score
    other_score = right_score if side == "left" else left_score
    if selected_score < float(min_rms_velocity_rad_s):
        return None
    dominance_ratio = selected_score / max(
        other_score, np.finfo(np.float64).eps
    )
    return ArmMotionSelection(
        side=side,
        window_start_frame=start,
        window_end_frame=stop - 1,
        left_rms_velocity_rad_s=left_score,
        right_rms_velocity_rad_s=right_score,
        dominance_ratio=float(dominance_ratio),
    )


def _build_arm_recovery_candidates(
    *,
    states: np.ndarray,
    source_episode: int,
    setup_steps: int,
    required_tail_steps: int,
    cfg: DictConfig,
    model_risk_anchors: Sequence[recovery_common.ModelRiskAnchor] | None = None,
) -> tuple[list[ArmRecoveryCandidate], list[int]]:
    """构造全部可回退锚点，并按配置概率生成确定性的优先顺序。"""

    num_frames = len(states)
    valid_first = max(
        int(cfg.event_sampling.exclude_initial_steps), int(setup_steps)
    )
    valid_last = int(num_frames) - int(required_tail_steps)
    if valid_last < valid_first:
        raise ValueError(
            "源episode没有同时容纳锚点前扰动建立和锚点后恢复的帧: "
            f"frames={num_frames}, valid=[{valid_first},{valid_last}]"
        )

    mode = str(cfg.event_sampling.mode)
    all_valid = np.arange(valid_first, valid_last + 1, dtype=np.int64)
    if mode == "model_risk":
        anchors = list(model_risk_anchors or ())
        radius = int(cfg.event_sampling.fallback_radius_steps)
        interval = int(cfg.event_sampling.min_injection_interval_steps)
        candidates: list[ArmRecoveryCandidate] = []
        inactive_frames: list[int] = []
        # 重叠风险区间中的同一帧只归属于分数最高且排序最前的区间，
        # 保证后续candidate_by_frame映射和中断续跑身份唯一。
        seen_frames: set[int] = set()
        valid_anchors = [
            anchor
            for anchor in anchors
            if valid_first <= anchor.frame <= valid_last
        ]
        for left_index, left_anchor in enumerate(valid_anchors):
            left_first = max(valid_first, left_anchor.frame - radius)
            left_last = min(valid_last, left_anchor.frame + radius)
            for right_anchor in valid_anchors[left_index + 1 :]:
                right_first = max(valid_first, right_anchor.frame - radius)
                right_last = min(valid_last, right_anchor.frame + radius)
                overlaps = max(left_first, right_first) <= min(
                    left_last, right_last
                )
                if (
                    overlaps
                    and left_anchor.target_arm is not None
                    and right_anchor.target_arm is not None
                    and left_anchor.target_arm != right_anchor.target_arm
                ):
                    raise ValueError(
                        "Arm模型风险锚点的局部回退区间重叠但target_arm冲突: "
                        f"frames={left_anchor.frame}/{right_anchor.frame}, "
                        f"sides={left_anchor.target_arm}/{right_anchor.target_arm}。"
                    )
        for anchor in sorted(anchors, key=lambda value: (-value.score, value.frame)):
            if anchor.source_episode != int(source_episode):
                raise ValueError("传入了属于其他source_episode的模型风险锚点。")
            if (
                anchor.expected_num_frames is not None
                and anchor.expected_num_frames != int(num_frames)
            ):
                raise ValueError(
                    f"风险清单source_episode={source_episode}帧数"
                    f"{anchor.expected_num_frames}与源episode实际帧数{num_frames}不一致。"
                )
            if anchor.frame >= int(num_frames):
                raise ValueError(
                    f"风险锚点frame={anchor.frame}超出source_episode={source_episode}"
                    f"实际帧范围[0,{num_frames - 1}]。"
                )
            if not valid_first <= anchor.frame <= valid_last:
                continue
            first = max(valid_first, anchor.frame - radius)
            last = min(valid_last, anchor.frame + radius)
            domain_key = f"model_risk_interval:{first}:{last}"
            ordered_frames = sorted(
                range(first, last + 1),
                key=lambda frame: (abs(frame - anchor.frame), frame),
            )
            for frame in ordered_frames:
                if frame in seen_frames:
                    continue
                selection = _select_active_arm(
                    states=states,
                    event_frame=frame,
                    fps=int(cfg.fps),
                    lookback_steps=int(cfg.arm_selection.lookback_steps),
                    lookahead_steps=int(cfg.arm_selection.lookahead_steps),
                    min_rms_velocity_rad_s=float(
                        cfg.arm_selection.min_rms_velocity_rad_s
                    ),
                    requested_side=anchor.target_arm,
                )
                if selection is None:
                    inactive_frames.append(frame)
                    continue
                seen_frames.add(frame)
                candidates.append(
                    ArmRecoveryCandidate(
                        event=recovery_common.RecoverySamplingEvent(
                            frame=frame,
                            sampling_mode=mode,
                            sampling_source=(
                                (
                                    "model_risk_anchor"
                                    if anchor.selection_source == "model_risk"
                                    else "manifest_random_exploration_anchor"
                                )
                                if frame == anchor.frame
                                else (
                                    "model_risk_local_fallback"
                                    if anchor.selection_source == "model_risk"
                                    else "manifest_random_exploration_local_fallback"
                                )
                            ),
                            model_risk_anchor_frame=(
                                anchor.frame
                                if anchor.selection_source == "model_risk"
                                else None
                            ),
                            model_risk_score=(
                                float(anchor.score)
                                if anchor.selection_source == "model_risk"
                                else None
                            ),
                            model_risk_score_key=(
                                str(cfg.event_sampling.score_key)
                                if anchor.selection_source == "model_risk"
                                else None
                            ),
                            model_risk_target_arm=(
                                anchor.target_arm
                                if anchor.selection_source == "model_risk"
                                else None
                            ),
                            selection_source=anchor.selection_source,
                            manifest_anchor_frame=anchor.frame,
                            manifest_score=float(anchor.score),
                            manifest_score_key=str(cfg.event_sampling.score_key),
                        ),
                        motion_selection=selection,
                        domain_key=domain_key,
                        min_interval_steps=interval,
                        sampling_weight=float(anchor.score),
                    )
                )

        if bool(cfg.event_sampling.fallback_to_random):
            covered_frames = {candidate.event.frame for candidate in candidates}
            fallback_frames = [
                int(frame) for frame in all_valid if int(frame) not in covered_frames
            ]
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(cfg.seed), int(source_episode), 0x414D5246]
                )
            )
            if fallback_frames:
                for index in rng.permutation(len(fallback_frames)):
                    frame = fallback_frames[int(index)]
                    selection = _select_active_arm(
                        states=states,
                        event_frame=frame,
                        fps=int(cfg.fps),
                        lookback_steps=int(cfg.arm_selection.lookback_steps),
                        lookahead_steps=int(cfg.arm_selection.lookahead_steps),
                        min_rms_velocity_rad_s=float(
                            cfg.arm_selection.min_rms_velocity_rad_s
                        ),
                    )
                    if selection is None:
                        inactive_frames.append(frame)
                        continue
                    candidates.append(
                        ArmRecoveryCandidate(
                            event=recovery_common.RecoverySamplingEvent(
                                frame=frame,
                                sampling_mode=mode,
                                sampling_source="model_risk_random_fallback",
                                model_risk_score_key=str(
                                    cfg.event_sampling.score_key
                                ),
                                selection_source="runtime_random_fallback",
                            ),
                            motion_selection=selection,
                            domain_key="model_risk_random_fallback",
                            min_interval_steps=interval,
                            sampling_weight=0.0,
                        )
                    )
        if not candidates:
            raise ValueError(
                "该源episode没有与恢复有效范围相交且满足局部运动阈值的"
                "Arm模型风险锚点，且未启用可用的随机回退。"
            )
        return candidates, sorted(set(inactive_frames))

    raw_domains: list[
        tuple[str, np.ndarray, str, int | None, float | None, float | None, int, float]
    ] = []
    regions = list(cfg.event_sampling.normalized_regions)
    covered: set[int] = set()

    if mode == "random":
        raw_domains.append(
            (
                "global",
                all_valid,
                "global_random",
                None,
                None,
                None,
                int(cfg.event_sampling.min_injection_interval_steps),
                float(cfg.event_sampling.injection_probability_per_frame),
            )
        )
    else:
        for region_index, region in enumerate(regions):
            first, last = recovery_common._normalized_region_frame_bounds(
                num_frames=num_frames,
                start=float(region.start),
                end=float(region.end),
            )
            first = max(first, valid_first)
            last = min(last, valid_last)
            frames = (
                np.arange(first, last + 1, dtype=np.int64)
                if last >= first
                else np.empty(0, dtype=np.int64)
            )
            covered.update(int(value) for value in frames)
            raw_domains.append(
                (
                    f"region:{region_index}",
                    frames,
                    "normalized_region",
                    region_index,
                    float(region.start),
                    float(region.end),
                    int(region.min_injection_interval_steps),
                    float(region.injection_probability_per_frame),
                )
            )
        if mode == "hybrid":
            outside = np.asarray(
                [frame for frame in all_valid if int(frame) not in covered],
                dtype=np.int64,
            )
            raw_domains.append(
                (
                    "outside",
                    outside,
                    "outside_random",
                    None,
                    None,
                    None,
                    int(cfg.event_sampling.min_injection_interval_steps),
                    float(cfg.event_sampling.injection_probability_per_frame),
                )
            )

    candidates: list[ArmRecoveryCandidate] = []
    inactive_frames: list[int] = []
    for (
        domain_key,
        frames,
        sampling_source,
        region_index,
        region_start,
        region_end,
        min_interval,
        weight,
    ) in raw_domains:
        if weight <= 0.0:
            continue
        for raw_frame in frames:
            frame = int(raw_frame)
            selection = _select_active_arm(
                states=states,
                event_frame=frame,
                fps=int(cfg.fps),
                lookback_steps=int(cfg.arm_selection.lookback_steps),
                lookahead_steps=int(cfg.arm_selection.lookahead_steps),
                min_rms_velocity_rad_s=float(
                    cfg.arm_selection.min_rms_velocity_rad_s
                ),
            )
            if selection is None:
                inactive_frames.append(frame)
                continue
            candidates.append(
                ArmRecoveryCandidate(
                    event=recovery_common.RecoverySamplingEvent(
                        frame=frame,
                        sampling_mode=mode,
                        sampling_source=sampling_source,
                        region_index=region_index,
                        region_start_normalized=region_start,
                        region_end_normalized=region_end,
                    ),
                    motion_selection=selection,
                    domain_key=domain_key,
                    min_interval_steps=int(min_interval),
                    sampling_weight=float(weight),
                )
            )
    if not candidates:
        raise ValueError("所有候选锚点均无主运动臂或采样权重为0。")

    weights = np.asarray(
        [candidate.sampling_weight for candidate in candidates], dtype=np.float64
    )
    weights /= float(np.sum(weights))
    rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg.seed), int(source_episode), 0x414E4348])
    )
    order = rng.choice(len(candidates), size=len(candidates), replace=False, p=weights)
    return [candidates[int(index)] for index in order], sorted(set(inactive_frames))


def _candidate_is_spaced_from_successes(
    candidate: ArmRecoveryCandidate,
    successful_candidates: Sequence[ArmRecoveryCandidate],
) -> bool:
    """使用两个候选中更严格的间隔约束，避免成功锚点过度聚集。"""

    return all(
        abs(candidate.event.frame - existing.event.frame)
        >= max(candidate.min_interval_steps, existing.min_interval_steps, 1)
        for existing in successful_candidates
    )


def _neighbor_candidates(
    *,
    primary: ArmRecoveryCandidate,
    all_candidates: Sequence[ArmRecoveryCandidate],
    unavailable_frames: set[int],
    successful_candidates: Sequence[ArmRecoveryCandidate],
) -> list[ArmRecoveryCandidate]:
    """在同一采样域内按距主锚点由近到远返回尚可尝试的帧。"""

    rank = {
        candidate.event.frame: index for index, candidate in enumerate(all_candidates)
    }
    values = [
        candidate
        for candidate in all_candidates
        if candidate.domain_key == primary.domain_key
        and candidate.event.frame not in unavailable_frames
        and _candidate_is_spaced_from_successes(candidate, successful_candidates)
    ]
    return sorted(
        values,
        key=lambda candidate: (
            abs(candidate.event.frame - primary.event.frame),
            rank[candidate.event.frame],
        ),
    )


def _arm_velocity_percentile(
    action_sequences: Iterable[np.ndarray],
    fps: int,
    percentile: float,
) -> tuple[np.ndarray, int]:
    """合并左右臂专家动作差分，统计各关节速度绝对值分位数。"""

    if int(fps) <= 0:
        raise ValueError("fps 必须为正整数。")
    if not np.isfinite(percentile) or not 0.0 < float(percentile) <= 100.0:
        raise ValueError("percentile 必须位于(0,100]。")
    velocities: list[np.ndarray] = []
    for raw_actions in action_sequences:
        actions = np.asarray(raw_actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"joint_action 必须为[T,20]，当前为{actions.shape}。")
        if not np.isfinite(actions).all():
            raise ValueError("joint_action 包含 NaN 或 Inf。")
        if len(actions) < 2:
            continue
        for arm_slice in (LEFT_ARM_SLICE, RIGHT_ARM_SLICE):
            velocities.append(
                np.abs(np.diff(actions[:, arm_slice], axis=0)) * float(fps)
            )
    if not velocities:
        raise ValueError("没有足够的连续动作帧用于统计 Arm 关节速度。")
    merged = np.concatenate(velocities, axis=0)
    return (
        np.percentile(merged, float(percentile), axis=0).astype(np.float64),
        int(len(merged)),
    )


def _estimate_arm_velocity_statistics(
    sources: Sequence[SourceEpisode],
    fps: int,
    percentile: float,
    floor_rad_s: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    sequences: list[np.ndarray] = []
    for source in sources:
        with np.load(source.directory / "arrays.npz", allow_pickle=False) as data:
            if "joint_action" not in data:
                raise KeyError(f"{source.directory}/arrays.npz 缺少 joint_action。")
            sequences.append(np.asarray(data["joint_action"], dtype=np.float64))
    raw, sample_count = _arm_velocity_percentile(sequences, fps, percentile)
    floor = np.asarray(floor_rad_s, dtype=np.float64)
    if floor.shape != (ARM_DIM,) or not np.isfinite(floor).all() or np.any(floor <= 0):
        raise ValueError("auto_velocity.floor_rad_s 必须是6维有限正数。")
    if not np.isfinite(scale) or float(scale) <= 0.0:
        raise ValueError("auto_velocity.scale 必须为有限正数。")
    resolved = np.maximum(raw, floor) * float(scale)
    return {
        "percentile": float(percentile),
        "sample_count": int(sample_count),
        "raw_percentile_rad_s": raw.tolist(),
        "floor_rad_s": floor.tolist(),
        "scale": float(scale),
        "resolved_max_extra_velocity_rad_s": resolved.tolist(),
    }


def _local_arm_feasible_offset_bounds(
    source_arrays: dict[str, np.ndarray],
    event_frame: int,
    horizon_steps: int,
    control_ranges: np.ndarray,
    max_abs: np.ndarray,
    margin: float,
    perturbed_arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    """计算注入与恢复窗口内所选 Arm 不越过硬限位的偏移域。"""

    states = np.asarray(source_arrays["observation_state"], dtype=np.float64)
    actions = np.asarray(source_arrays["joint_action"], dtype=np.float64)
    arm_slice = _arm_slice(perturbed_arm)
    stop = min(len(actions), int(event_frame) + int(horizon_steps) + 1)
    values = np.concatenate(
        (
            states[int(event_frame) : stop, arm_slice],
            actions[int(event_frame) : stop, arm_slice],
        ),
        axis=0,
    )
    ranges = np.asarray(control_ranges, dtype=np.float64)
    max_abs = np.asarray(max_abs, dtype=np.float64)
    if ranges.shape != (ARM_DIM, 2) or max_abs.shape != (ARM_DIM,):
        raise ValueError("control_ranges 必须为[6,2]，max_abs 必须为6维。")
    lower = np.maximum(
        -max_abs,
        np.max(ranges[:, 0][None, :] + float(margin) - values, axis=0),
    )
    upper = np.minimum(
        max_abs,
        np.min(ranges[:, 1][None, :] - float(margin) - values, axis=0),
    )
    if np.any(lower > upper):
        raise ValueError(
            "该 Arm 恢复事件在关节限位安全余量内没有可行偏移: "
            f"side={perturbed_arm}, lower={lower.tolist()}, upper={upper.tolist()}"
        )
    return lower, upper


def _sample_arm_recovery_offset(
    *,
    seed: int,
    source_episode: int,
    variant_index: int,
    attempt: int,
    anchor_frame: int,
    perturbed_arm: str,
    std: np.ndarray,
    max_abs: np.ndarray,
    feasible_lower: np.ndarray,
    feasible_upper: np.ndarray,
    min_normalized_l2: float,
    max_sampling_attempts: int,
) -> np.ndarray:
    """确定性地为每次分支重试采样独立的截断高斯 Arm 偏移。"""

    std = np.asarray(std, dtype=np.float64)
    max_abs = np.asarray(max_abs, dtype=np.float64)
    lower = np.asarray(feasible_lower, dtype=np.float64)
    upper = np.asarray(feasible_upper, dtype=np.float64)
    if any(value.shape != (ARM_DIM,) for value in (std, max_abs, lower, upper)):
        raise ValueError("Arm 偏移采样的所有向量都必须为6维。")
    if np.any(std <= 0.0) or np.any(max_abs <= 0.0):
        raise ValueError("std_rad 和 max_abs_rad 必须全部大于0。")
    if np.any(lower > upper):
        raise ValueError("feasible_lower 不能大于 feasible_upper。")
    side_code = 0x4C454654 if perturbed_arm == "left" else 0x52494748
    _arm_slice(perturbed_arm)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(source_episode),
                int(variant_index),
                int(attempt),
                int(anchor_frame),
                int(side_code),
            ]
        )
    )
    for _ in range(int(max_sampling_attempts)):
        offset = rng.normal(loc=0.0, scale=std, size=ARM_DIM)
        if np.any(np.abs(offset) > max_abs):
            continue
        if np.any(offset < lower) or np.any(offset > upper):
            continue
        if float(np.linalg.norm(offset / max_abs)) < float(min_normalized_l2):
            continue
        return offset.astype(np.float64)
    raise RuntimeError(
        "截断高斯在最大采样次数内没有得到可行 Arm 偏移；请减小"
        "min_normalized_l2/安全余量，或增大 max_sampling_attempts。"
    )


def _validate_actual_recovery_offset(
    actual_offset: np.ndarray,
    feasible_lower: np.ndarray,
    feasible_upper: np.ndarray,
    max_abs: np.ndarray,
    min_normalized_l2: float,
) -> float:
    """验证物理建立后实际到达的偏移，并返回归一化扰动强度。"""

    actual = np.asarray(actual_offset, dtype=np.float64)
    lower = np.asarray(feasible_lower, dtype=np.float64)
    upper = np.asarray(feasible_upper, dtype=np.float64)
    scale = np.asarray(max_abs, dtype=np.float64)
    if any(value.shape != (ARM_DIM,) for value in (actual, lower, upper, scale)):
        raise ValueError("实际偏移、可行上下限和max_abs必须都是6维。")
    if not all(np.isfinite(value).all() for value in (actual, lower, upper, scale)):
        raise ArmRecoveryBranchError("物理建立后的实际 Arm 偏移或边界包含NaN/Inf。")
    if np.any(scale <= 0.0) or np.any(lower > upper):
        raise ValueError("max_abs必须为正，且可行下限不能大于上限。")

    tolerance = 1e-9
    below = actual < lower - tolerance
    above = actual > upper + tolerance
    if np.any(below) or np.any(above):
        raise ArmRecoveryBranchError(
            "物理建立后的实际 Arm 偏移超出该事件的可行上下限: "
            f"actual={actual.tolist()}, lower={lower.tolist()}, "
            f"upper={upper.tolist()}"
        )
    normalized_l2 = float(np.linalg.norm(actual / scale))
    if normalized_l2 + tolerance < float(min_normalized_l2):
        raise ArmRecoveryBranchError(
            "物理建立后的实际 Arm 偏移强度不足: "
            f"normalized_l2={normalized_l2:.6g} < "
            f"{float(min_normalized_l2):.6g}"
        )
    return normalized_l2


def _branch_errors(
    actual: np.ndarray,
    expert: np.ndarray,
    perturbed_arm: str,
) -> dict[str, Any]:
    error = np.asarray(actual, dtype=np.float64) - np.asarray(
        expert, dtype=np.float64
    )
    selected_slice = _arm_slice(perturbed_arm)
    other_slice = _other_arm_slice(perturbed_arm)
    return {
        "selected_arm_vector": error[selected_slice].copy(),
        "selected_arm": float(np.max(np.abs(error[selected_slice]))),
        "other_arm": float(np.max(np.abs(error[other_slice]))),
        "gripper": float(np.max(np.abs(error[GRIPPER_INDICES]))),
        "view": float(np.max(np.abs(error[VIEW_SLICE]))),
    }


def _validate_unperturbed_roles(
    errors: dict[str, Any],
    cfg: DictConfig,
    phase: str,
    *,
    static_hold: bool = False,
) -> None:
    if static_hold:
        limits = {
            "other_arm": float(
                cfg.validation.get(
                    "static_hold_max_other_arm_joint_drift_rad",
                    cfg.validation.branch_max_other_arm_joint_abs_error,
                )
            ),
            "gripper": float(
                cfg.validation.get(
                    "static_hold_max_gripper_drift",
                    cfg.validation.branch_max_gripper_abs_error,
                )
            ),
            "view": float(
                cfg.validation.get(
                    "static_hold_max_view_joint_drift_rad",
                    cfg.validation.branch_max_view_joint_abs_error,
                )
            ),
        }
    else:
        limits = {
            "other_arm": float(
                cfg.validation.branch_max_other_arm_joint_abs_error
            ),
            "gripper": float(cfg.validation.branch_max_gripper_abs_error),
            "view": float(cfg.validation.branch_max_view_joint_abs_error),
        }
    for name, limit in limits.items():
        if float(errors[name]) > limit:
            raise ArmRecoveryBranchError(
                f"{phase}阶段{name}误差{float(errors[name]):.6g}超过阈值{limit:.6g}。"
            )


def _validate_recovery_unperturbed_roles(
    errors: dict[str, Any],
    cfg: DictConfig,
    *,
    trajectory_alignment_mode: str,
    final: bool,
) -> None:
    """校验恢复阶段的未扰动角色。

    moving_expert 仍在每个恢复步执行原有的跟随误差校验；
    static_anchor_wait 允许制动过程中的瞬时漂移，只在恢复结束时校验。
    """

    static_hold = trajectory_alignment_mode == "static_anchor_wait"
    if static_hold and not final:
        return
    _validate_unperturbed_roles(
        errors,
        cfg,
        "Arm 恢复结束" if final else "Arm 恢复",
        static_hold=static_hold,
    )


def _apply_unrecorded_arm_disturbance(
    *,
    env_obj,
    source_states: np.ndarray,
    source_actions: np.ndarray,
    setup_start_frame: int,
    recovery_anchor_frame: int,
    offset: np.ndarray,
    perturbed_arm: str,
    setup_steps: int,
    cfg: DictConfig,
) -> tuple[int, dict[str, Any], float]:
    """沿专家时间轴用真实物理步平滑建立 Arm OOD 状态，但不记录。"""

    source_index = int(setup_start_frame)
    last_reward = 0.0
    for setup_index in range(int(setup_steps)):
        if source_index >= len(source_actions):
            raise ArmRecoveryBranchError("源轨迹剩余长度不足以完成扰动注入。")
        progress = (setup_index + 1) / float(setup_steps)
        fraction = float(recovery_common._quintic_smoothstep(progress))
        action = _arm_recovery_action(
            source_actions[source_index], fraction * offset, perturbed_arm
        )
        last_reward, terminated, truncated = recovery_common._step_without_render(
            env_obj, action
        )
        if truncated:
            raise ArmRecoveryBranchError("未记录的 Arm 扰动注入阶段被截断。")
        if terminated:
            outcome = "成功" if bool(getattr(env_obj, "is_success", False)) else "失败"
            raise ArmRecoveryBranchError(
                f"Arm 扰动注入阶段提前{outcome}终止，拒绝继续构造分支。"
            )
        source_index += 1

    if source_index != int(recovery_anchor_frame):
        raise RuntimeError(
            "Arm扰动建立结束帧与恢复锚点不一致: "
            f"setup_start={setup_start_frame}, setup_steps={setup_steps}, "
            f"actual_end={source_index}, anchor={recovery_anchor_frame}"
        )

    if source_index >= len(source_states):
        raise ArmRecoveryBranchError(
            "扰动注入完成后没有剩余专家状态可供恢复。"
        )
    actual = _read_agent_state(env_obj)
    expert = source_states[source_index]
    errors = _branch_errors(actual, expert, perturbed_arm)
    _validate_unperturbed_roles(errors, cfg, "扰动注入完成")
    tracking_error = np.asarray(errors["selected_arm_vector"]) - np.asarray(
        offset, dtype=np.float64
    )
    return (
        source_index,
        {
            "actual_offset_rad": np.asarray(
                errors["selected_arm_vector"], dtype=np.float64
            ),
            "offset_tracking_error_rad": tracking_error.astype(np.float64),
            "other_arm_error": float(errors["other_arm"]),
            "gripper_error": float(errors["gripper"]),
            "view_error": float(errors["view"]),
        },
        float(last_reward),
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
    name = _output_episode_name(source.episode_number, variant_index)
    episode_dir = recovery_common._episode_dir(
        output_run_dir, source.episode_number, variant_index
    )
    return {
        "episode": int(source.episode_number),
        "episode_name": name,
        "episode_naming": EPISODE_NAMING,
        "success": True,
        "steps": int(steps),
        "fps": int(cfg.fps),
        "path": str(episode_dir.relative_to(output_run_dir)),
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
        **recovery_common._filtered_original_arrays(source_arrays),
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
            "max_replay_arm_joint_abs_error": float(replay_errors["arm_joint"]),
            "max_replay_gripper_abs_error": float(replay_errors["gripper"]),
            "max_replay_view_joint_abs_error": float(replay_errors["view_joint"]),
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


def _generate_arm_recovery_branch(
    *,
    env_obj,
    source: SourceEpisode,
    source_arrays: dict[str, np.ndarray],
    snapshot: recovery_common.EnvironmentSnapshot,
    event: recovery_common.RecoverySamplingEvent,
    motion_selection: ArmMotionSelection,
    primary_anchor_frame: int,
    anchor_search_history: Sequence[dict[str, Any]],
    variant_index: int,
    attempt: int,
    perturbed_arm: str,
    offset: np.ndarray,
    feasible_lower: np.ndarray,
    feasible_upper: np.ndarray,
    max_extra_velocity_rad_s: np.ndarray,
    output_run_dir: Path,
    cameras: tuple[str, ...],
    cfg: DictConfig,
    fingerprint: str,
) -> recovery_common.RecoveryResult:
    recovery_anchor_frame = int(event.frame)
    setup_steps = int(cfg.recovery.unrecorded_setup_steps)
    setup_start_frame = recovery_anchor_frame - setup_steps
    if setup_start_frame < 0:
        raise ArmRecoveryBranchError(
            f"恢复锚点{recovery_anchor_frame}之前不足{setup_steps}帧建立扰动。"
        )
    final_dir = recovery_common._episode_dir(
        output_run_dir, source.episode_number, variant_index
    )
    tmp_dir = recovery_common._prepare_tmp_directory(final_dir)
    try:
        recovery_common._restore_environment_snapshot(env_obj, snapshot)
        source_states = np.asarray(
            source_arrays["observation_state"], dtype=np.float64
        )
        source_actions = np.asarray(source_arrays["joint_action"], dtype=np.float64)
        source_index, setup_stats, last_reward = _apply_unrecorded_arm_disturbance(
            env_obj=env_obj,
            source_states=source_states,
            source_actions=source_actions,
            setup_start_frame=setup_start_frame,
            recovery_anchor_frame=recovery_anchor_frame,
            offset=offset,
            perturbed_arm=perturbed_arm,
            setup_steps=setup_steps,
            cfg=cfg,
        )
        if source_index != recovery_anchor_frame:
            raise RuntimeError(
                f"恢复记录应从锚点{recovery_anchor_frame}开始，当前为{source_index}。"
            )
        sampled_offset = np.asarray(offset, dtype=np.float64)
        recovery_offset = np.asarray(
            setup_stats["actual_offset_rad"], dtype=np.float64
        )
        configured_max_abs = np.asarray(
            cfg.arm_joint_noise.max_abs_rad, dtype=np.float64
        )
        effective_max_abs = np.minimum(
            configured_max_abs,
            np.asarray(max_extra_velocity_rad_s, dtype=np.float64)
            * int(cfg.recovery.max_steps)
            / (1.875 * int(cfg.fps)),
        )
        actual_offset_normalized_l2 = _validate_actual_recovery_offset(
            recovery_offset,
            feasible_lower,
            feasible_upper,
            effective_max_abs,
            float(cfg.arm_joint_noise.min_normalized_l2),
        )
        recorded_first_source_frame = int(source_index)
        initial_arrays = recovery_common._capture_episode_initial_arrays(env_obj)

        planned_steps = recovery_common._adaptive_recovery_steps(
            recovery_offset,
            int(cfg.fps),
            np.asarray(max_extra_velocity_rad_s, dtype=np.float64),
            int(cfg.recovery.min_steps),
            int(cfg.recovery.max_steps),
        )
        max_extra_steps = int(cfg.recovery.max_extra_zero_offset_steps)
        stable_required = int(cfg.recovery.success_stable_steps)
        post_required = int(cfg.recovery.post_recovery_steps)
        recovery_threshold = float(cfg.recovery.success_max_abs_error_rad)
        trajectory_alignment_mode = resolve_trajectory_alignment_mode(cfg)
        max_recorded_steps = (
            planned_steps + max_extra_steps + post_required + 1
        )
        static_reference_state = source_states[recovery_anchor_frame].copy()
        if trajectory_alignment_mode == "static_anchor_wait":
            static_reference_state = build_static_anchor_reference_state(
                expert_state=source_states[recovery_anchor_frame],
                actual_state=_read_agent_state(env_obj),
                perturbed_indices=_arm_slice(perturbed_arm),
            )

        recorded_states: list[np.ndarray] = []
        recorded_actions: list[np.ndarray] = []
        source_indices: list[int] = []
        reference_arm_states: list[np.ndarray] = []
        command_offsets: list[np.ndarray] = []
        selected_state_errors: list[np.ndarray] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []
        max_selected_error = 0.0
        max_other_arm_error = 0.0
        max_gripper_error = 0.0
        max_view_error = 0.0
        stable_count = 0
        achieved_local_step: int | None = None
        achieved_source_frame: int | None = None
        final_static_errors: dict[str, Any] | None = None
        post_frames = 0
        local_step = 0

        with StereoVideoWriter(tmp_dir / "videos", cameras, cfg) as writer:
            while (
                source_index < len(source_actions)
                and local_step < max_recorded_steps
            ):
                reference_index, next_source_index = (
                    resolve_recovery_timeline_step(
                        trajectory_alignment_mode,
                        recovery_anchor_frame=recovery_anchor_frame,
                        source_frame=source_index,
                    )
                )
                actual_state = _read_agent_state(env_obj)
                expert_state = (
                    source_states[reference_index]
                    if trajectory_alignment_mode == "moving_expert"
                    else static_reference_state
                )
                errors = _branch_errors(
                    actual_state, expert_state, perturbed_arm
                )
                _validate_recovery_unperturbed_roles(
                    errors,
                    cfg,
                    trajectory_alignment_mode=trajectory_alignment_mode,
                    final=False,
                )
                selected_error_vector = np.asarray(
                    errors["selected_arm_vector"], dtype=np.float64
                )
                selected_error = float(errors["selected_arm"])
                max_selected_error = max(max_selected_error, selected_error)
                max_other_arm_error = max(
                    max_other_arm_error, float(errors["other_arm"])
                )
                max_gripper_error = max(
                    max_gripper_error, float(errors["gripper"])
                )
                max_view_error = max(max_view_error, float(errors["view"]))

                if local_step >= planned_steps:
                    stable_count = (
                        stable_count + 1
                        if selected_error <= recovery_threshold
                        else 0
                    )
                    if achieved_local_step is None and stable_count >= stable_required:
                        achieved_local_step = int(local_step)
                        achieved_source_frame = int(reference_index)
                if achieved_local_step is not None and local_step > achieved_local_step:
                    if selected_error > recovery_threshold:
                        raise ArmRecoveryBranchError(
                            "Arm 恢复达标后的专家跟随阶段再次离开误差管道: "
                            f"source_frame={reference_index}, error={selected_error:.6g} > "
                            f"{recovery_threshold:.6g}"
                        )
                    post_frames += 1
                if (
                    achieved_local_step is None
                    and local_step >= planned_steps + max_extra_steps
                ):
                    raise ArmRecoveryBranchError(
                        "计划恢复归零后仍未连续满足 Arm 恢复阈值: "
                        f"planned_steps={planned_steps}, extra={max_extra_steps}, "
                        f"error={selected_error:.6g}, stable={stable_count}/"
                        f"{stable_required}"
                    )

                if local_step < planned_steps:
                    remaining = float(
                        recovery_common._quintic_remaining_fraction(
                            (local_step + 1) / float(planned_steps)
                        )
                    )
                else:
                    remaining = 0.0
                command_offset = remaining * recovery_offset
                base_action = resolve_recovery_base_action(
                    trajectory_alignment_mode,
                    expert_action=source_actions[reference_index],
                    expert_state=expert_state,
                )
                action = _arm_recovery_action(
                    base_action,
                    command_offset,
                    perturbed_arm,
                )

                recovery_common._render_stereo(env_obj, writer, cameras, cfg)
                recorded_states.append(actual_state.astype(np.float32))
                recorded_actions.append(action.astype(np.float32))
                source_indices.append(int(reference_index))
                reference_arm_states.append(
                    expert_state[_arm_slice(perturbed_arm)].astype(np.float32)
                )
                command_offsets.append(command_offset.astype(np.float32))
                selected_state_errors.append(
                    selected_error_vector.astype(np.float32)
                )

                last_reward, terminated, truncated = (
                    recovery_common._step_without_render(env_obj, action)
                )
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                if truncated:
                    raise ArmRecoveryBranchError("Arm 恢复分支被环境步数上限截断。")
                if terminated and not bool(getattr(env_obj, "is_success", False)):
                    raise ArmRecoveryBranchError(
                        "Arm 恢复分支在任务完成前失败终止。"
                    )
                if terminated and reference_index != len(source_actions) - 1:
                    raise ArmRecoveryBranchError(
                        "Arm 恢复分支在专家后缀结束前提前成功终止；"
                        "拒绝在 terminal 状态后继续推进。"
                    )

                source_index = next_source_index
                local_step += 1
                if achieved_local_step is not None and post_frames >= post_required:
                    break

        if achieved_local_step is None:
            raise ArmRecoveryBranchError(
                "源轨迹剩余长度不足以达到 Arm 恢复条件。"
            )
        if post_frames < post_required:
            raise ArmRecoveryBranchError(
                f"Arm 恢复后仅有{post_frames}帧，少于要求的{post_required}帧。"
            )

        if trajectory_alignment_mode == "static_anchor_wait":
            final_static_errors = _branch_errors(
                _read_agent_state(env_obj),
                static_reference_state,
                perturbed_arm,
            )
            _validate_recovery_unperturbed_roles(
                final_static_errors,
                cfg,
                trajectory_alignment_mode=trajectory_alignment_mode,
                final=True,
            )

        source_index = recovery_suffix_start_frame(
            trajectory_alignment_mode,
            recovery_anchor_frame=recovery_anchor_frame,
            source_frame=source_index,
        )

        # 剩余后缀不再渲染和保存，只验证该恢复分支最终仍能完成任务。
        while source_index < len(source_actions):
            last_reward, terminated, truncated = (
                recovery_common._step_without_render(
                    env_obj, source_actions[source_index]
                )
            )
            if truncated:
                raise ArmRecoveryBranchError("后台专家后缀验证被截断。")
            if terminated and not bool(getattr(env_obj, "is_success", False)):
                raise ArmRecoveryBranchError(
                    f"后台专家后缀在 source_frame={source_index} 失败终止。"
                )
            if terminated and source_index != len(source_actions) - 1:
                raise ArmRecoveryBranchError(
                    f"后台专家后缀在 source_frame={source_index} 提前成功终止；"
                    "拒绝在 terminal 状态后继续推进。"
                )
            source_index += 1
        if not bool(getattr(env_obj, "is_success", False)):
            raise ArmRecoveryBranchError("执行完整专家后缀后任务未成功。")

        arrays = {
            "joint_action": np.asarray(recorded_actions, dtype=np.float32),
            "observation_state": np.asarray(recorded_states, dtype=np.float32),
            "timestamp": np.arange(len(recorded_actions), dtype=np.float32)
            / float(cfg.fps),
            "terminated": np.asarray(terminated_flags, dtype=np.bool_),
            "truncated": np.asarray(truncated_flags, dtype=np.bool_),
            "source_frame_index": np.asarray(source_indices, dtype=np.int64),
            "recovery_reference_arm_state": np.asarray(
                reference_arm_states, dtype=np.float32
            ),
            "recovery_command_offset": np.asarray(
                command_offsets, dtype=np.float32
            ),
            "recovery_arm_state_error": np.asarray(
                selected_state_errors, dtype=np.float32
            ),
            **initial_arrays,
        }
        np.savez_compressed(tmp_dir / "arrays.npz", **arrays)

        final_arm_error = float(
            np.max(
                np.abs(np.asarray(selected_state_errors[-1], dtype=np.float64))
            )
        )
        planned_duration_s = planned_steps / float(cfg.fps)
        peak_extra_velocity = 1.875 * np.abs(recovery_offset) / planned_duration_s
        velocity_limited_max_abs = np.minimum(
            configured_max_abs,
            np.asarray(max_extra_velocity_rad_s, dtype=np.float64)
            * int(cfg.recovery.max_steps)
            / (1.875 * int(cfg.fps)),
        )
        info = _base_episode_info(
            source=source,
            variant_index=variant_index,
            output_run_dir=output_run_dir,
            cameras=cameras,
            cfg=cfg,
            fingerprint=fingerprint,
            steps=len(recorded_actions),
        )
        info.update(
            {
                "source_replay_success": True,
                "final_task_success": True,
                "perturbed_arm": perturbed_arm,
                "arm_selection_mode": "local_motion",
                "arm_motion_window_start_frame": int(
                    motion_selection.window_start_frame
                ),
                "arm_motion_window_end_frame": int(
                    motion_selection.window_end_frame
                ),
                "left_arm_rms_velocity_rad_s": float(
                    motion_selection.left_rms_velocity_rad_s
                ),
                "right_arm_rms_velocity_rad_s": float(
                    motion_selection.right_rms_velocity_rad_s
                ),
                "selected_arm_motion_dominance_ratio": float(
                    motion_selection.dominance_ratio
                ),
                # 兼容旧字段：schema v5起它表示恢复锚点，而非扰动建立起点。
                "source_injection_frame": int(recovery_anchor_frame),
                "source_disturbance_start_frame": int(setup_start_frame),
                "source_recovery_anchor_frame": int(recovery_anchor_frame),
                "source_primary_anchor_frame": int(primary_anchor_frame),
                "anchor_search_history": [
                    dict(value) for value in anchor_search_history
                ],
                "anchor_search_failed_frames": int(
                    len(
                        {
                            int(value["candidate_anchor_frame"])
                            for value in anchor_search_history
                            if value.get("anchor_exhausted") is True
                        }
                    )
                ),
                "sampling_mode": event.sampling_mode,
                "sampling_source": event.sampling_source,
                "sampling_region_index": event.region_index,
                "sampling_region_normalized": (
                    None
                    if event.region_index is None
                    else [
                        float(event.region_start_normalized),
                        float(event.region_end_normalized),
                    ]
                ),
                "model_risk_anchor_frame": event.model_risk_anchor_frame,
                "model_risk_score": event.model_risk_score,
                "model_risk_score_key": event.model_risk_score_key,
                "model_risk_target_arm": event.model_risk_target_arm,
                "selection_source": event.selection_source,
                "manifest_anchor_frame": event.manifest_anchor_frame,
                "manifest_score": event.manifest_score,
                "manifest_score_key": event.manifest_score_key,
                "source_recorded_first_frame": recorded_first_source_frame,
                "source_recorded_last_frame": int(source_indices[-1]),
                # 兼容旧分析字段：从v4起该字段表示真实到达并用于恢复的偏移。
                "arm_joint_offset_rad": recovery_offset.tolist(),
                "sampled_arm_joint_offset_rad": sampled_offset.tolist(),
                "recovery_initial_arm_offset_rad": recovery_offset.tolist(),
                "recovery_uses_actual_achieved_offset": True,
                "actual_arm_offset_normalized_l2": actual_offset_normalized_l2,
                "configured_arm_offset_max_abs_rad": configured_max_abs.tolist(),
                "effective_arm_offset_max_abs_rad": velocity_limited_max_abs.tolist(),
                "arm_offset_feasible_lower_rad": feasible_lower.tolist(),
                "arm_offset_feasible_upper_rad": feasible_upper.tolist(),
                "unrecorded_disturbance_setup_steps": int(
                    cfg.recovery.unrecorded_setup_steps
                ),
                "unrecorded_setup_curve": "quintic_minimum_jerk_from_moving_expert",
                "setup_actual_arm_offset_rad": setup_stats[
                    "actual_offset_rad"
                ].tolist(),
                "setup_offset_tracking_error_rad": setup_stats[
                    "offset_tracking_error_rad"
                ].tolist(),
                "planned_recovery_steps": int(planned_steps),
                "planned_recovery_duration_s": float(planned_duration_s),
                "recovery_command_peak_extra_velocity_rad_s": (
                    peak_extra_velocity.tolist()
                ),
                "resolved_max_extra_velocity_rad_s": np.asarray(
                    max_extra_velocity_rad_s, dtype=np.float64
                ).tolist(),
                "actual_recovery_steps": int(achieved_local_step),
                "recovery_achieved_source_frame": int(achieved_source_frame),
                "recovery_success_max_abs_error_rad": recovery_threshold,
                "recovery_success_stable_steps": stable_required,
                "recovery_post_steps": int(post_frames),
                "trajectory_alignment_mode": trajectory_alignment_mode,
                "expert_progress_advanced_during_recovery": bool(
                    trajectory_alignment_mode == "moving_expert"
                ),
                "unperturbed_roles_hold_anchor": bool(
                    trajectory_alignment_mode == "static_anchor_wait"
                ),
                "final_recorded_arm_max_abs_error_rad": final_arm_error,
                "max_recorded_selected_arm_max_abs_error_rad": float(
                    max_selected_error
                ),
                "max_recorded_other_arm_joint_abs_error": float(
                    max_other_arm_error
                ),
                "max_recorded_gripper_abs_error": float(max_gripper_error),
                "max_recorded_view_joint_abs_error": float(max_view_error),
                "static_hold_validation_timing": (
                    "recovery_end"
                    if trajectory_alignment_mode == "static_anchor_wait"
                    else None
                ),
                "final_static_other_arm_joint_drift_rad": (
                    float(final_static_errors["other_arm"])
                    if final_static_errors is not None
                    else None
                ),
                "final_static_gripper_drift": (
                    float(final_static_errors["gripper"])
                    if final_static_errors is not None
                    else None
                ),
                "final_static_view_joint_drift_rad": (
                    float(final_static_errors["view"])
                    if final_static_errors is not None
                    else None
                ),
                "sampling_seed": int(cfg.seed),
                "branch_attempt": int(attempt),
                "final_reward": float(last_reward),
                "background_suffix_validated": True,
                "background_suffix_final_source_frame": int(source_index - 1),
                "recovery_curve": (
                    "quintic_minimum_jerk_to_moving_expert"
                    if trajectory_alignment_mode == "moving_expert"
                    else "quintic_minimum_jerk_to_static_anchor"
                ),
                "unperturbed_action_reference": (
                    "moving_expert"
                    if trajectory_alignment_mode == "moving_expert"
                    else "static_anchor_state"
                ),
                "only_selected_arm_action_modified": True,
                "other_arm_action_uses_expert": True,
                "gripper_action_uses_expert": True,
                "view_action_uses_expert": True,
                "final_info": {
                    "is_success": True,
                    "source_episode": int(source.episode_number),
                    "variant_index": int(variant_index),
                    "source_injection_frame": int(recovery_anchor_frame),
                    "source_disturbance_start_frame": int(setup_start_frame),
                    "source_recovery_anchor_frame": int(recovery_anchor_frame),
                    "trajectory_alignment_mode": trajectory_alignment_mode,
                    "sampling_mode": event.sampling_mode,
                    "sampling_source": event.sampling_source,
                    "sampling_region_index": event.region_index,
                    "model_risk_anchor_frame": event.model_risk_anchor_frame,
                    "model_risk_score": event.model_risk_score,
                    "model_risk_score_key": event.model_risk_score_key,
                    "model_risk_target_arm": event.model_risk_target_arm,
                    "selection_source": event.selection_source,
                    "manifest_anchor_frame": event.manifest_anchor_frame,
                    "manifest_score": event.manifest_score,
                    "manifest_score_key": event.manifest_score_key,
                    "perturbed_arm": perturbed_arm,
                    "arm_selection_mode": "local_motion",
                    "arm_recovery_achieved": True,
                    "background_suffix_validated": True,
                },
            }
        )
        _write_json_atomic(tmp_dir / "info.json", info)
        tmp_dir.rename(final_dir)
        return recovery_common.RecoveryResult(info=info, arrays=arrays)
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
        Path(recovery_common.__file__).resolve(),
        ROOT_DIR / "data_collect/recovery_data_generation/trajectory_replay_common.py",
        ROOT_DIR / "env/__init__.py",
        ROOT_DIR / "env/constants.py",
        ROOT_DIR / "env/task/sim_envs.py",
        ROOT_DIR / task_files[env_id],
    ]
    dependencies.extend(sorted((ROOT_DIR / "env/assets").rglob("*.xml")))
    manifest: dict[str, str] = {}
    for path in dependencies:
        if not path.is_file():
            raise FileNotFoundError(f"生成依赖文件不存在: {path}")
        manifest[str(path.relative_to(ROOT_DIR))] = _sha256_file(path)
    return manifest


def _semantic_config(
    cfg: DictConfig,
    input_run_dir: Path,
    env_id: str,
    velocity_statistics: dict[str, Any],
) -> dict[str, Any]:
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "episode_naming": EPISODE_NAMING,
        "input_run_dir": str(input_run_dir),
        "env_id": env_id,
        "include_original": bool(cfg.include_original),
        "max_source_episodes": (
            None
            if cfg.max_source_episodes is None
            else int(cfg.max_source_episodes)
        ),
        "source_episode_indices": (
            None
            if cfg.source_episode_indices is None
            else sorted(int(value) for value in cfg.source_episode_indices)
        ),
        "seed": int(cfg.seed),
        "arm_assignment": "dominant_local_state_motion",
        "recovery_offset_source": "actual_achieved_state_offset",
        "recovery_anchor_semantics": (
            "disturbance_is_built_before_anchor_and_recording_starts_at_anchor"
        ),
        "anchor_retry_semantics": (
            "resample_offset_then_search_nearest_unused_anchor_in_same_domain"
        ),
        "arm_selection": OmegaConf.to_container(
            cfg.arm_selection, resolve=True
        ),
        "event_sampling": OmegaConf.to_container(cfg.event_sampling, resolve=True),
        "arm_joint_noise": OmegaConf.to_container(
            cfg.arm_joint_noise, resolve=True
        ),
        "auto_velocity": OmegaConf.to_container(cfg.auto_velocity, resolve=True),
        "resolved_velocity_statistics": velocity_statistics,
        "recovery": OmegaConf.to_container(cfg.recovery, resolve=True),
        "validation": OmegaConf.to_container(cfg.validation, resolve=True),
        "cameras": [str(value) for value in cfg.cameras],
        "render_height": int(cfg.render_height),
        "render_width": int(cfg.render_width),
        "fps": int(cfg.fps),
        "video": OmegaConf.to_container(cfg.video, resolve=True),
        "generation_dependencies": _generation_dependency_manifest(env_id),
    }
    if str(cfg.event_sampling.mode) == "model_risk":
        manifest_path = recovery_common._resolve_path(
            cfg.event_sampling.risk_manifest_path
        )
        semantic["model_risk_manifest_sha256"] = _sha256_file(manifest_path)
    return semantic


def _refresh_metadata(
    metadata: dict[str, Any],
    output_run_dir: Path,
    cameras: tuple[str, ...],
    fingerprint: str,
) -> None:
    infos = recovery_common._scan_completed_infos(
        output_run_dir, cameras, fingerprint
    )
    metadata["episodes"] = infos
    metadata["saved_episodes"] = len(infos)
    metadata["successful_episodes"] = sum(
        recovery_common._strict_true(info.get("success")) for info in infos
    )
    metadata["original_episodes"] = sum(
        not bool(info.get("is_augmented", False)) for info in infos
    )
    metadata["recovery_episodes"] = sum(
        bool(info.get("is_augmented", False)) for info in infos
    )
    arm_counts = {side: 0 for side in ARM_SIDES}
    for info in infos:
        side = info.get("perturbed_arm")
        if side in arm_counts:
            arm_counts[str(side)] += 1
    metadata["recovery_episodes_by_arm"] = arm_counts
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
    skipped_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata_path = output_run_dir / "metadata.json"
    if output_run_dir.exists() and not bool(cfg.resume):
        raise FileExistsError(f"输出目录已存在且 resume=false: {output_run_dir}")
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "episodes").mkdir(exist_ok=True)
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        if metadata.get("generation_fingerprint") != fingerprint:
            raise ValueError(
                "续生成配置与已有输出不一致，请更换 output_run_dir: "
                f"existing={metadata.get('generation_fingerprint')!r}, "
                f"current={fingerprint!r}"
            )
    else:
        unexpected = [
            path for path in output_run_dir.iterdir() if path.name != "episodes"
        ]
        if unexpected or any((output_run_dir / "episodes").iterdir()):
            raise RuntimeError(
                f"输出目录非空但缺少 metadata.json: {output_run_dir}"
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
            "video_save_mode": "pre_action_physical_arm_recovery_rollout",
            "success_semantics": "physical_rollout_and_full_suffix_validation",
            "render_width": int(cfg.render_width),
            "render_height": int(cfg.render_height),
            "generator": Path(__file__).name,
            "generation_schema_version": SCHEMA_VERSION,
            "generation_fingerprint": fingerprint,
            "generation_config": semantic_config,
            "episode_naming": EPISODE_NAMING,
            "episodes": [],
            "failures": [],
        }
    metadata["generator"] = Path(__file__).name
    metadata["skipped_sources"] = skipped_sources
    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
    return metadata


def _validate_config(
    cfg: DictConfig,
    source_metadata: dict[str, Any],
) -> tuple[str, tuple[str, ...]]:
    metadata_env_id = source_metadata.get("env_id")
    env_id = str(metadata_env_id) if cfg.env_id is None else str(cfg.env_id)
    if not env_id:
        raise ValueError("无法从配置或源 metadata 解析 env_id。")
    if (
        cfg.env_id is not None
        and metadata_env_id is not None
        and str(metadata_env_id) != env_id
    ):
        raise ValueError(
            f"配置 env_id={env_id!r} 与源 metadata={metadata_env_id!r} 不一致。"
        )
    supported = {
        "guided_vision/SewNeedle-3Arms-v0",
        "guided_vision/InsertCylinder-3Arms-v0",
        "guided_vision/InsertPeg-3Arms-v0",
        "guided_vision/HookPackage-3Arms-v0",
    }
    if env_id not in supported:
        raise ValueError(f"当前 Arm 恢复生成器不支持 env_id={env_id!r}。")
    cameras = tuple(str(value) for value in cfg.cameras)
    if cameras != ("zed_cam_left", "zed_cam_right"):
        raise ValueError(
            "Arm 恢复生成器固定只保存 zed_cam_left 和 zed_cam_right，"
            f"当前为{cameras}。"
        )
    source_fps = source_metadata.get("fps")
    if source_fps is not None and int(source_fps) != int(cfg.fps):
        raise ValueError(
            f"配置 fps={int(cfg.fps)} 与源 metadata fps={int(source_fps)} 不一致。"
        )
    resolve_trajectory_alignment_mode(cfg)

    sampling_mode = str(cfg.event_sampling.mode)
    supported_sampling_modes = {
        "random",
        "specified_region",
        "hybrid",
        "model_risk",
    }
    if sampling_mode not in supported_sampling_modes:
        raise ValueError(
            "event_sampling.mode 必须是 random、specified_region、hybrid 或 model_risk，"
            f"当前为{sampling_mode!r}。"
        )
    regions = list(cfg.event_sampling.normalized_regions)
    if sampling_mode in {"specified_region", "hybrid"} and not regions:
        raise ValueError(
            f"event_sampling.mode={sampling_mode!r} 时必须配置至少一个 "
            "normalized_regions 区间。"
        )
    if sampling_mode == "model_risk":
        if cfg.event_sampling.risk_manifest_path is None:
            raise ValueError("model_risk 模式必须配置 risk_manifest_path。")
        score_key = cfg.event_sampling.score_key
        if not isinstance(score_key, str) or not score_key.strip():
            raise ValueError("model_risk 模式的 score_key 必须是非空字符串。")
        if not isinstance(cfg.event_sampling.fallback_to_random, bool):
            raise ValueError("fallback_to_random 必须是布尔值。")
        fallback_radius = float(cfg.event_sampling.fallback_radius_steps)
        if (
            not np.isfinite(fallback_radius)
            or not fallback_radius.is_integer()
            or fallback_radius < 0
        ):
            raise ValueError("fallback_radius_steps 必须是非负整数。")
    previous_end = 0.0
    for region_index, region in enumerate(regions):
        start = float(region.start)
        end = float(region.end)
        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or not 0.0 <= start < end <= 1.0
        ):
            raise ValueError(
                f"normalized_regions[{region_index}] 必须满足 "
                f"0<=start<end<=1，当前为[{start},{end}]。"
            )
        if region_index > 0 and start < previous_end:
            raise ValueError(
                "normalized_regions 必须按 start 升序排列且互不重叠: "
                f"region[{region_index - 1}].end={previous_end}, "
                f"region[{region_index}].start={start}。"
            )
        previous_end = end
        probability = float(region.injection_probability_per_frame)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"normalized_regions[{region_index}]."
                "injection_probability_per_frame 必须位于[0,1]。"
            )
        region_integer_fields = {
            "min_injection_interval_steps": region.min_injection_interval_steps,
        }
        for name, raw in region_integer_fields.items():
            value = float(raw)
            if (
                not np.isfinite(value)
                or not value.is_integer()
                or value < 0
            ):
                raise ValueError(
                    f"normalized_regions[{region_index}].{name} "
                    "必须为非负整数。"
                )

    if str(cfg.arm_selection.mode) != "local_motion":
        raise ValueError("arm_selection.mode 当前仅支持 local_motion。")
    arm_selection_integer_fields = {
        "arm_selection.lookback_steps": (
            cfg.arm_selection.lookback_steps,
            False,
        ),
        "arm_selection.lookahead_steps": (
            cfg.arm_selection.lookahead_steps,
            True,
        ),
    }
    for name, (raw, strictly_positive) in arm_selection_integer_fields.items():
        value = float(raw)
        minimum = 1 if strictly_positive else 0
        if (
            not np.isfinite(value)
            or not value.is_integer()
            or value < minimum
        ):
            qualifier = "正整数" if strictly_positive else "非负整数"
            raise ValueError(f"{name} 必须为{qualifier}。")
    min_motion = float(cfg.arm_selection.min_rms_velocity_rad_s)
    if not np.isfinite(min_motion) or min_motion < 0.0:
        raise ValueError(
            "arm_selection.min_rms_velocity_rad_s 必须为有限非负数。"
        )

    vector_fields = {
        "arm_joint_noise.std_rad": cfg.arm_joint_noise.std_rad,
        "arm_joint_noise.max_abs_rad": cfg.arm_joint_noise.max_abs_rad,
        "auto_velocity.floor_rad_s": cfg.auto_velocity.floor_rad_s,
    }
    for name, raw in vector_fields.items():
        value = np.asarray(raw, dtype=np.float64)
        if value.shape != (ARM_DIM,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"{name} 必须是6维有限正数。")
    if str(cfg.arm_joint_noise.distribution) != "truncated_gaussian":
        raise ValueError(
            "当前仅支持 arm_joint_noise.distribution=truncated_gaussian。"
        )

    integer_positive = {
        "fps": cfg.fps,
        "render_height": cfg.render_height,
        "render_width": cfg.render_width,
        "event_sampling.recovery_branches_per_source": (
            cfg.event_sampling.recovery_branches_per_source
        ),
        "recovery.min_steps": cfg.recovery.min_steps,
        "recovery.max_steps": cfg.recovery.max_steps,
        "recovery.success_stable_steps": cfg.recovery.success_stable_steps,
        "recovery.unrecorded_setup_steps": cfg.recovery.unrecorded_setup_steps,
        "validation.max_branch_attempts": cfg.validation.max_branch_attempts,
        "arm_joint_noise.max_sampling_attempts": cfg.arm_joint_noise.max_sampling_attempts,
    }
    for name, raw in integer_positive.items():
        value = float(raw)
        if not np.isfinite(value) or not value.is_integer() or value <= 0:
            raise ValueError(f"{name} 必须为正整数。")
    integer_nonnegative = {
        "event_sampling.min_injection_interval_steps": (
            cfg.event_sampling.min_injection_interval_steps
        ),
        "event_sampling.exclude_initial_steps": cfg.event_sampling.exclude_initial_steps,
        "recovery.max_extra_zero_offset_steps": cfg.recovery.max_extra_zero_offset_steps,
        "recovery.post_recovery_steps": cfg.recovery.post_recovery_steps,
    }
    for name, raw in integer_nonnegative.items():
        value = float(raw)
        if not np.isfinite(value) or not value.is_integer() or value < 0:
            raise ValueError(f"{name} 必须为非负整数。")
    if int(cfg.recovery.max_steps) < int(cfg.recovery.min_steps):
        raise ValueError("recovery.max_steps 不能小于 min_steps。")
    if int(cfg.recovery.success_stable_steps) > int(
        cfg.recovery.max_extra_zero_offset_steps
    ) + 1:
        raise ValueError(
            "success_stable_steps 不能大于 max_extra_zero_offset_steps+1。"
        )
    probability = float(cfg.event_sampling.injection_probability_per_frame)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("injection_probability_per_frame 必须位于[0,1]。")
    percentile = float(cfg.auto_velocity.percentile)
    if not np.isfinite(percentile) or not 0.0 < percentile <= 100.0:
        raise ValueError("auto_velocity.percentile 必须位于(0,100]。")
    if not np.isfinite(float(cfg.auto_velocity.scale)) or float(
        cfg.auto_velocity.scale
    ) <= 0.0:
        raise ValueError("auto_velocity.scale 必须为有限正数。")

    positive_float_fields = {
        "arm_joint_noise.joint_limit_margin_rad": cfg.arm_joint_noise.joint_limit_margin_rad,
        "recovery.success_max_abs_error_rad": cfg.recovery.success_max_abs_error_rad,
        "validation.max_arm_joint_abs_error": cfg.validation.max_arm_joint_abs_error,
        "validation.max_gripper_abs_error": cfg.validation.max_gripper_abs_error,
        "validation.max_view_joint_abs_error": cfg.validation.max_view_joint_abs_error,
        "validation.branch_max_other_arm_joint_abs_error": (
            cfg.validation.branch_max_other_arm_joint_abs_error
        ),
        "validation.branch_max_gripper_abs_error": cfg.validation.branch_max_gripper_abs_error,
        "validation.branch_max_view_joint_abs_error": (
            cfg.validation.branch_max_view_joint_abs_error
        ),
        "validation.static_hold_max_other_arm_joint_drift_rad": (
            cfg.validation.get(
                "static_hold_max_other_arm_joint_drift_rad",
                cfg.validation.branch_max_other_arm_joint_abs_error,
            )
        ),
        "validation.static_hold_max_gripper_drift": (
            cfg.validation.get(
                "static_hold_max_gripper_drift",
                cfg.validation.branch_max_gripper_abs_error,
            )
        ),
        "validation.static_hold_max_view_joint_drift_rad": (
            cfg.validation.get(
                "static_hold_max_view_joint_drift_rad",
                cfg.validation.branch_max_view_joint_abs_error,
            )
        ),
    }
    for name, raw in positive_float_fields.items():
        value = float(raw)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须为有限正数。")

    min_normalized_l2 = float(cfg.arm_joint_noise.min_normalized_l2)
    if (
        not np.isfinite(min_normalized_l2)
        or min_normalized_l2 < 0.0
        or min_normalized_l2 > np.sqrt(ARM_DIM)
    ):
        raise ValueError("arm_joint_noise.min_normalized_l2 必须位于[0,sqrt(6)]。")
    if cfg.max_source_episodes is not None:
        max_sources = float(cfg.max_source_episodes)
        if (
            not np.isfinite(max_sources)
            or not max_sources.is_integer()
            or max_sources <= 0
        ):
            raise ValueError("max_source_episodes 必须为正整数或 null。")
    if cfg.source_episode_indices is not None:
        for raw_index in cfg.source_episode_indices:
            index = float(raw_index)
            if not np.isfinite(index) or not index.is_integer() or index < 0:
                raise ValueError("source_episode_indices 必须全部为非负整数。")
    seed = float(cfg.seed)
    if not np.isfinite(seed) or not seed.is_integer() or seed < 0:
        raise ValueError("seed 必须为非负整数。")
    return env_id, cameras


def _load_planning_states(source: SourceEpisode) -> np.ndarray:
    with np.load(source.directory / "arrays.npz", allow_pickle=False) as data:
        missing = [
            key
            for key in ("joint_action", "observation_state")
            if key not in data
        ]
        if missing:
            raise KeyError(f"{source.directory}/arrays.npz 缺少{missing}。")
        actions = np.asarray(data["joint_action"])
        states = np.asarray(data["observation_state"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) == 0:
        raise ValueError(
            f"source_episode={source.episode_number} joint_action 形状非法: "
            f"{actions.shape}"
        )
    if states.shape != actions.shape or not np.isfinite(states).all():
        raise ValueError(
            f"source_episode={source.episode_number} observation_state "
            f"形状或数值非法: {states.shape}"
        )
    return states


def _arm_control_ranges(env_obj, side: str) -> np.ndarray:
    if side == "left":
        joints = env_obj._left_joints[:ARM_DIM]
        actuators = env_obj._left_actuators[:ARM_DIM]
    elif side == "right":
        joints = env_obj._right_joints[:ARM_DIM]
        actuators = env_obj._right_actuators[:ARM_DIM]
    else:
        raise ValueError(f"未知 Arm 侧别: {side!r}")
    joint_ranges = np.asarray(
        env_obj._physics.bind(joints).range, dtype=np.float64
    ).copy()
    actuator_ranges = np.asarray(
        env_obj._physics.bind(actuators).ctrlrange, dtype=np.float64
    ).copy()
    ranges = np.stack(
        (
            np.maximum(joint_ranges[:, 0], actuator_ranges[:, 0]),
            np.minimum(joint_ranges[:, 1], actuator_ranges[:, 1]),
        ),
        axis=1,
    )
    if ranges.shape != (ARM_DIM, 2) or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise RuntimeError(f"{side} Arm 关节限位和执行器限位没有有效交集。")
    return ranges


def generate_contractive_arm_recovery_run(cfg: DictConfig) -> None:
    input_run_dir = recovery_common._resolve_path(cfg.input_run_dir)
    output_run_dir = recovery_common._resolve_path(cfg.output_run_dir)
    if input_run_dir == output_run_dir:
        raise ValueError("input_run_dir 与 output_run_dir 不能相同。")
    source_metadata_path = input_run_dir / "metadata.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"输入 run 缺少 metadata.json: {source_metadata_path}")
    source_metadata = _load_json(source_metadata_path)
    env_id, cameras = _validate_config(cfg, source_metadata)
    source_indices = (
        None
        if cfg.source_episode_indices is None
        else [int(value) for value in cfg.source_episode_indices]
    )
    sources, skipped_sources = recovery_common._load_successful_sources(
        input_run_dir,
        source_indices,
        None if cfg.max_source_episodes is None else int(cfg.max_source_episodes),
    )
    model_risk_anchors_by_episode: dict[
        int, list[recovery_common.ModelRiskAnchor]
    ] = {}
    if str(cfg.event_sampling.mode) == "model_risk":
        model_risk_anchors_by_episode = recovery_common._load_model_risk_manifest(
            manifest_path=cfg.event_sampling.risk_manifest_path,
            input_run_dir=input_run_dir,
            score_key=str(cfg.event_sampling.score_key),
            selection_role="arm",
        )

    setup_steps = int(cfg.recovery.unrecorded_setup_steps)
    required_tail_steps = (
        int(cfg.recovery.max_steps)
        + int(cfg.recovery.max_extra_zero_offset_steps)
        + int(cfg.recovery.post_recovery_steps)
        + 1
    )
    configured_target_branches = int(
        cfg.event_sampling.recovery_branches_per_source
    )
    target_branches_by_source = recovery_common._resolve_recovery_branch_targets(
        [source.episode_number for source in sources],
        sampling_mode=str(cfg.event_sampling.mode),
        configured_branches_per_source=configured_target_branches,
        model_risk_anchors_by_episode=model_risk_anchors_by_episode,
    )
    source_candidates: dict[int, list[ArmRecoveryCandidate]] = {}
    skipped_inactive_events: list[dict[str, Any]] = []
    eligible_sources: list[SourceEpisode] = []
    for source in sources:
        try:
            target_branches = target_branches_by_source[source.episode_number]
            planning_states = _load_planning_states(source)
            if str(cfg.event_sampling.mode) == "model_risk":
                with np.load(
                    source.directory / "arrays.npz", allow_pickle=False
                ) as planning_data:
                    if "joint_action" not in planning_data:
                        raise KeyError(
                            f"{source.directory}/arrays.npz缺少joint_action。"
                        )
                    recovery_common._validate_model_risk_episode_identity(
                        source_episode=source.episode_number,
                        joint_actions=np.asarray(planning_data["joint_action"]),
                        anchors=model_risk_anchors_by_episode.get(
                            source.episode_number, ()
                        ),
                    )
            if target_branches == 0:
                candidates, inactive_frames = [], []
            else:
                candidates, inactive_frames = _build_arm_recovery_candidates(
                    states=planning_states,
                    source_episode=source.episode_number,
                    cfg=cfg,
                    setup_steps=setup_steps,
                    required_tail_steps=required_tail_steps,
                    model_risk_anchors=model_risk_anchors_by_episode.get(
                        source.episode_number, ()
                    ),
                )
            if len(candidates) < target_branches:
                raise ValueError(
                    f"有效候选锚点仅{len(candidates)}个，少于恢复目标"
                    f"{target_branches}。"
                )
            skipped_inactive_events.append(
                {
                    "source_episode": int(source.episode_number),
                    "count": len(inactive_frames),
                    "frames": inactive_frames,
                    "reason": "both_arms_below_motion_threshold",
                }
            )
        except Exception as exc:
            if not bool(cfg.continue_on_error):
                raise
            skipped_sources.append(
                {
                    "source_episode": int(source.episode_number),
                    "directory": str(source.directory),
                    "reason": (
                        "arm_recovery_event_planning_failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )
            logging.warning(
                "源 episode=%06d 无法规划 Arm 恢复事件，已跳过: %s",
                source.episode_number,
                exc,
            )
            continue
        eligible_sources.append(source)
        source_candidates[source.episode_number] = candidates
    sources = eligible_sources
    if not sources:
        raise RuntimeError("没有源 episode 能够容纳当前 Arm 注入与恢复窗口。")

    velocity_statistics = _estimate_arm_velocity_statistics(
        sources,
        int(cfg.fps),
        float(cfg.auto_velocity.percentile),
        np.asarray(cfg.auto_velocity.floor_rad_s, dtype=np.float64),
        float(cfg.auto_velocity.scale),
    )
    max_extra_velocity = np.asarray(
        velocity_statistics["resolved_max_extra_velocity_rad_s"],
        dtype=np.float64,
    )

    semantic_config = _semantic_config(
        cfg, input_run_dir, env_id, velocity_statistics
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
        skipped_sources=skipped_sources,
    )
    recovery_common._update_source_manifest(metadata, sources)
    metadata["trajectory_alignment_mode"] = resolve_trajectory_alignment_mode(cfg)
    metadata["recovery_branches_per_source"] = (
        None
        if str(cfg.event_sampling.mode) == "model_risk"
        else configured_target_branches
    )
    metadata["configured_recovery_branches_per_source"] = (
        configured_target_branches
    )
    metadata["recovery_branch_target_mode"] = (
        "manifest_anchor_count"
        if str(cfg.event_sampling.mode) == "model_risk"
        else "fixed_per_source"
    )
    metadata["recovery_branch_targets_by_source"] = {
        str(source.episode_number): target_branches_by_source[source.episode_number]
        for source in sources
    }
    metadata["total_recovery_branch_target"] = sum(
        target_branches_by_source[source.episode_number] for source in sources
    )
    metadata["candidate_anchor_frames"] = sum(
        len(values) for values in source_candidates.values()
    )
    metadata["skipped_inactive_arm_events"] = skipped_inactive_events
    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)

    logging.info(
        "开始生成可收缩 Arm 恢复数据: env=%s source=%s output=%s "
        "successful_sources=%d skipped=%d target_mode=%s total_target=%d "
        "alignment=%s candidates=%d velocity_p%.1f=%s "
        "fingerprint=%s",
        env_id,
        input_run_dir,
        output_run_dir,
        len(sources),
        len(skipped_sources),
        metadata["recovery_branch_target_mode"],
        metadata["total_recovery_branch_target"],
        metadata["trajectory_alignment_mode"],
        metadata["candidate_anchor_frames"],
        float(cfg.auto_velocity.percentile),
        np.array2string(max_extra_velocity, precision=4),
        fingerprint,
    )

    configured_max_abs = np.asarray(
        cfg.arm_joint_noise.max_abs_rad, dtype=np.float64
    )
    velocity_limited_max_abs = (
        max_extra_velocity
        * int(cfg.recovery.max_steps)
        / (1.875 * int(cfg.fps))
    )
    max_abs = np.minimum(configured_max_abs, velocity_limited_max_abs)
    if np.any(max_abs < configured_max_abs - 1e-12):
        logging.info(
            "为满足自动统计的 Arm 恢复峰值速度上限，"
            "偏移有效上限由%s收紧为%s",
            np.array2string(configured_max_abs, precision=4),
            np.array2string(max_abs, precision=4),
        )
    if np.any(max_abs <= 0.0):
        raise RuntimeError(
            "自动速度统计得到的 Arm 有效扰动上限必须全部大于0。"
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
        control_ranges = {
            side: _arm_control_ranges(env_obj, side) for side in ARM_SIDES
        }
        std = np.asarray(cfg.arm_joint_noise.std_rad, dtype=np.float64)

        for source in sources:
            target_branches = target_branches_by_source[source.episode_number]
            original_final_dir = recovery_common._episode_dir(
                output_run_dir, source.episode_number, -1
            )
            original_complete = _episode_is_complete(original_final_dir, cameras)
            original_tmp_dir: Path | None = None
            try:
                source_arrays = _validate_source_arrays(
                    source.directory / "arrays.npz",
                    required_model_body_names=required_model_body_names,
                )
                episode_fps = source.info.get("fps")
                if episode_fps is not None and int(episode_fps) != int(cfg.fps):
                    raise ValueError(
                        f"source_episode={source.episode_number} fps={episode_fps} "
                        f"与配置 fps={int(cfg.fps)} 不一致。"
                    )
                candidates = source_candidates[source.episode_number]
                snapshot_frames = [
                    candidate.event.frame - setup_steps
                    for candidate in candidates
                ]
                if bool(cfg.include_original):
                    if original_complete:
                        recovery_common._validate_completed_identity(
                            original_final_dir,
                            source.episode_number,
                            -1,
                            fingerprint,
                        )
                    else:
                        original_tmp_dir = recovery_common._prepare_tmp_directory(
                            original_final_dir
                        )
                snapshots, replay_errors, final_reward = (
                    recovery_common._replay_source_and_capture(
                        env_obj=env_obj,
                        source=source,
                        source_arrays=source_arrays,
                        event_frames=snapshot_frames,
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
                recovery_common._clear_failure(
                    metadata, source.episode_number, None
                )
                _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
                logging.info(
                    "源 episode=%06d 名义重放成功，target=%d，"
                    "candidate_anchors=%d，range=%s，max_error=%.3g",
                    source.episode_number,
                    target_branches,
                    len(candidates),
                    (
                        "none"
                        if not candidates
                        else "[%d,%d]"
                        % (
                            min(candidate.event.frame for candidate in candidates),
                            max(candidate.event.frame for candidate in candidates),
                        )
                    ),
                    replay_errors["state"],
                )
            except Exception as exc:
                if original_tmp_dir is not None:
                    shutil.rmtree(original_tmp_dir, ignore_errors=True)
                recovery_common._record_failure(
                    metadata,
                    source.episode_number,
                    None,
                    None,
                    [{"error": f"{type(exc).__name__}: {exc}"}],
                )
                _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
                logging.exception(
                    "源 episode=%06d 准备或名义重放失败。",
                    source.episode_number,
                )
                if not bool(cfg.continue_on_error):
                    raise
                continue

            candidate_by_frame = {
                candidate.event.frame: candidate for candidate in candidates
            }
            successful_candidates: list[ArmRecoveryCandidate] = []
            unavailable_frames: set[int] = set()
            failed_anchor_frames: set[int] = set()

            for variant_index in range(target_branches):
                final_dir = recovery_common._episode_dir(
                    output_run_dir, source.episode_number, variant_index
                )
                if _episode_is_complete(final_dir, cameras):
                    completed = recovery_common._validate_completed_identity(
                        final_dir,
                        source.episode_number,
                        variant_index,
                        fingerprint,
                    )
                    completed_frame = int(completed["source_recovery_anchor_frame"])
                    if completed_frame not in candidate_by_frame:
                        raise RuntimeError(
                            "已完成分支的恢复锚点不在当前确定性候选池中: "
                            f"source={source.episode_number}, variant={variant_index}, "
                            f"anchor={completed_frame}"
                        )
                    completed_candidate = candidate_by_frame[completed_frame]
                    if completed.get("perturbed_arm") != completed_candidate.motion_selection.side:
                        raise RuntimeError("已完成分支的主运动臂与当前候选计划不一致。")
                    successful_candidates.append(completed_candidate)
                    unavailable_frames.add(completed_frame)
                    recovery_common._clear_failure(
                        metadata, source.episode_number, variant_index
                    )
                    logging.info(
                        "跳过已完成 Arm 恢复分支: source=%06d variant=%02d "
                        "anchor=%d arm=%s",
                        source.episode_number,
                        variant_index,
                        completed_frame,
                        completed_candidate.motion_selection.side,
                    )
                    continue

                search_history: list[dict[str, Any]] = []
                for failure in metadata.get("failures", []):
                    if (
                        failure.get("source_episode") == source.episode_number
                        and failure.get("variant_index") == variant_index
                    ):
                        search_history = [dict(value) for value in failure.get("attempts", [])]
                        break
                resumed_failed_frames = {
                    int(value["candidate_anchor_frame"])
                    for value in search_history
                    if value.get("anchor_exhausted") is True
                }
                unavailable_frames.update(resumed_failed_frames)
                failed_anchor_frames.update(resumed_failed_frames)
                pending_primary_frame = next(
                    (
                        int(value["primary_anchor_frame"])
                        for value in reversed(search_history)
                        if value.get("primary_anchor_frame") is not None
                    ),
                    None,
                )

                succeeded = False
                while not succeeded:
                    primary = (
                        candidate_by_frame.get(pending_primary_frame)
                        if pending_primary_frame is not None
                        else None
                    )
                    if primary is None:
                        primary = next(
                            (
                                candidate
                                for candidate in candidates
                                if candidate.event.frame not in unavailable_frames
                                and _candidate_is_spaced_from_successes(
                                    candidate, successful_candidates
                                )
                            ),
                            None,
                        )
                    if primary is None:
                        search_history.append(
                            {
                                "quota_not_met": True,
                                "successful_branches": len(successful_candidates),
                                "target_branches": target_branches,
                                "reason": "same-domain candidate anchors exhausted",
                            }
                        )
                        recovery_common._record_failure(
                            metadata,
                            source.episode_number,
                            variant_index,
                            None,
                            search_history,
                        )
                        _refresh_metadata(
                            metadata, output_run_dir, cameras, fingerprint
                        )
                        logging.error(
                            "源episode恢复目标未满足: source=%06d saved=%d/%d",
                            source.episode_number,
                            len(successful_candidates),
                            target_branches,
                        )
                        break

                    neighbor_queue = _neighbor_candidates(
                        primary=primary,
                        all_candidates=candidates,
                        unavailable_frames=unavailable_frames,
                        successful_candidates=successful_candidates,
                    )
                    if not neighbor_queue:
                        pending_primary_frame = None
                        continue
                    for candidate_rank, candidate in enumerate(neighbor_queue):
                        event = candidate.event
                        event_frame = int(event.frame)
                        setup_start_frame = event_frame - setup_steps
                        motion_selection = candidate.motion_selection
                        perturbed_arm = motion_selection.side

                        try:
                            feasible_lower, feasible_upper = (
                                _local_arm_feasible_offset_bounds(
                                    source_arrays,
                                    setup_start_frame,
                                    setup_steps + required_tail_steps - 1,
                                    control_ranges[perturbed_arm],
                                    max_abs,
                                    float(
                                        cfg.arm_joint_noise.joint_limit_margin_rad
                                    ),
                                    perturbed_arm,
                                )
                            )
                        except Exception as exc:
                            search_history.append(
                                {
                                    "candidate_anchor_frame": event_frame,
                                    "primary_anchor_frame": primary.event.frame,
                                    "candidate_setup_start_frame": setup_start_frame,
                                    "candidate_domain": candidate.domain_key,
                                    "candidate_rank_from_primary": candidate_rank,
                                    "anchor_exhausted": True,
                                    "attempt_count": 0,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            unavailable_frames.add(event_frame)
                            failed_anchor_frames.add(event_frame)
                            continue

                        frame_attempts: list[dict[str, Any]] = []
                        for attempt in range(
                            int(cfg.validation.max_branch_attempts)
                        ):
                            offset: np.ndarray | None = None
                            try:
                                offset = _sample_arm_recovery_offset(
                                    seed=int(cfg.seed),
                                    source_episode=source.episode_number,
                                    variant_index=variant_index,
                                    attempt=attempt,
                                    anchor_frame=event_frame,
                                    perturbed_arm=perturbed_arm,
                                    std=std,
                                    max_abs=max_abs,
                                    feasible_lower=feasible_lower,
                                    feasible_upper=feasible_upper,
                                    min_normalized_l2=float(
                                        cfg.arm_joint_noise.min_normalized_l2
                                    ),
                                    max_sampling_attempts=int(
                                        cfg.arm_joint_noise.max_sampling_attempts
                                    ),
                                )
                                result = _generate_arm_recovery_branch(
                                    env_obj=env_obj,
                                    source=source,
                                    source_arrays=source_arrays,
                                    snapshot=snapshots[setup_start_frame],
                                    event=event,
                                    motion_selection=motion_selection,
                                    primary_anchor_frame=primary.event.frame,
                                    anchor_search_history=(
                                        search_history + frame_attempts
                                    ),
                                    variant_index=variant_index,
                                    attempt=attempt,
                                    perturbed_arm=perturbed_arm,
                                    offset=offset,
                                    feasible_lower=feasible_lower,
                                    feasible_upper=feasible_upper,
                                    max_extra_velocity_rad_s=max_extra_velocity,
                                    output_run_dir=output_run_dir,
                                    cameras=cameras,
                                    cfg=cfg,
                                    fingerprint=fingerprint,
                                )
                                successful_candidates.append(candidate)
                                unavailable_frames.add(event_frame)
                                recovery_common._clear_failure(
                                    metadata,
                                    source.episode_number,
                                    variant_index,
                                )
                                _refresh_metadata(
                                    metadata,
                                    output_run_dir,
                                    cameras,
                                    fingerprint,
                                )
                                logging.info(
                                    "已保存 Arm 恢复分支: source=%06d "
                                    "variant=%02d primary=%d anchor=%d arm=%s "
                                    "attempt=%d searched_frames=%d planned=%d "
                                    "actual=%d final_error=%.3g",
                                    source.episode_number,
                                    variant_index,
                                    primary.event.frame,
                                    event_frame,
                                    perturbed_arm,
                                    attempt,
                                    len(
                                        {
                                            value.get("candidate_anchor_frame")
                                            for value in search_history
                                            if value.get("anchor_exhausted") is True
                                        }
                                    ),
                                    result.info["planned_recovery_steps"],
                                    result.info["actual_recovery_steps"],
                                    result.info[
                                        "final_recorded_arm_max_abs_error_rad"
                                    ],
                                )
                                succeeded = True
                                break
                            except Exception as exc:
                                record = {
                                    "candidate_anchor_frame": event_frame,
                                    "primary_anchor_frame": primary.event.frame,
                                    "candidate_setup_start_frame": setup_start_frame,
                                    "candidate_domain": candidate.domain_key,
                                    "candidate_rank_from_primary": candidate_rank,
                                    "attempt": int(attempt),
                                    "perturbed_arm": perturbed_arm,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "sampled_offset_rad": (
                                        offset.tolist()
                                        if offset is not None
                                        else None
                                    ),
                                }
                                frame_attempts.append(record)
                                logging.warning(
                                    "Arm恢复尝试失败，将在当前锚点重新采样: "
                                    "source=%06d variant=%02d anchor=%d arm=%s "
                                    "attempt=%d/%d error=%s",
                                    source.episode_number,
                                    variant_index,
                                    event_frame,
                                    perturbed_arm,
                                    attempt + 1,
                                    int(cfg.validation.max_branch_attempts),
                                    exc,
                                )
                        if succeeded:
                            break

                        search_history.extend(frame_attempts)
                        search_history.append(
                            {
                                "candidate_anchor_frame": event_frame,
                                "primary_anchor_frame": primary.event.frame,
                                "candidate_setup_start_frame": setup_start_frame,
                                "candidate_domain": candidate.domain_key,
                                "candidate_rank_from_primary": candidate_rank,
                                "anchor_exhausted": True,
                                "attempt_count": len(frame_attempts),
                            }
                        )
                        unavailable_frames.add(event_frame)
                        failed_anchor_frames.add(event_frame)
                        recovery_common._record_failure(
                            metadata,
                            source.episode_number,
                            variant_index,
                            event_frame,
                            search_history,
                        )
                        _refresh_metadata(
                            metadata, output_run_dir, cameras, fingerprint
                        )
                        logging.warning(
                            "当前锚点重试耗尽，转向同域邻近帧: "
                            "source=%06d variant=%02d anchor=%d domain=%s",
                            source.episode_number,
                            variant_index,
                            event_frame,
                            candidate.domain_key,
                        )

                    if succeeded:
                        break
                    # 当前主锚点所在域已经遍历完；外层选择其他尚可用采样域。
                    pending_primary_frame = None

                if not succeeded and not bool(cfg.continue_on_error):
                    raise ArmRecoveryBranchError(
                        f"source={source.episode_number}仅生成"
                        f"{len(successful_candidates)}/{target_branches}条恢复分支。"
                    )
                if not succeeded:
                    break

            quota_status = metadata.setdefault("source_quota_status", {})
            quota_status[str(source.episode_number)] = {
                "target": target_branches,
                "saved": len(successful_candidates),
                "satisfied": len(successful_candidates) == target_branches,
                "used_anchor_frames": len(successful_candidates),
                "failed_anchor_frames": len(failed_anchor_frames),
            }
            _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
    finally:
        env_obj.close()

    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
    saved_by_source: dict[int, int] = {}
    for info in metadata.get("episodes", []):
        if bool(info.get("is_augmented", False)):
            source_episode = int(info["source_episode"])
            saved_by_source[source_episode] = saved_by_source.get(source_episode, 0) + 1
    quota_status = metadata.setdefault("source_quota_status", {})
    for source in sources:
        entry = quota_status.setdefault(str(source.episode_number), {})
        saved = int(saved_by_source.get(source.episode_number, 0))
        target_branches = target_branches_by_source[source.episode_number]
        entry.update(
            {
                "target": target_branches,
                "saved": saved,
                "satisfied": saved == target_branches,
            }
        )
    metadata["quota_satisfied_sources"] = sum(
        bool(value.get("satisfied")) for value in quota_status.values()
    )
    metadata["quota_incomplete_sources"] = sorted(
        int(key)
        for key, value in quota_status.items()
        if not bool(value.get("satisfied"))
    )
    metadata["all_source_quotas_satisfied"] = not metadata[
        "quota_incomplete_sources"
    ]
    _write_json_atomic(output_run_dir / "metadata.json", metadata)
    logging.info(
        "生成完成: saved=%d original=%d recovery=%d by_arm=%s "
        "quota_satisfied=%d/%d incomplete=%s failures=%d output=%s",
        metadata["saved_episodes"],
        metadata["original_episodes"],
        metadata["recovery_episodes"],
        metadata["recovery_episodes_by_arm"],
        metadata["quota_satisfied_sources"],
        len(sources),
        metadata["quota_incomplete_sources"],
        len(metadata.get("failures", [])),
        output_run_dir,
    )
    if metadata["quota_incomplete_sources"]:
        raise RuntimeError(
            "恢复分支目标未全部满足；结果已保存，但不能作为完整恢复数据集使用。"
            "未完成源episode="
            f"{metadata['quota_incomplete_sources']}"
        )


@hydra.main(
    version_base="1.2",
    config_path="../../configs/data_collect",
    config_name="arm_trajectory_recovery",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _configure_mujoco_runtime(cfg)
    generate_contractive_arm_recovery_run(cfg)


if __name__ == "__main__":
    main()
