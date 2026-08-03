#!/usr/bin/env python

"""生成操作臂关节扰动后的可收缩恢复轨迹。

每个增强 episode 只扰动左臂或右臂中的一只，左右臂在全数据集中均衡
分配。扰动建立阶段使用真实 MuJoCo 控制步，并沿专家时间轴执行；被选中
的六维 Arm 关节在移动专家目标上平滑叠加偏移，另一只 Arm、两个夹爪和
View 始终执行同一时刻的专家动作。设置阶段不写入训练数据。

开始记录后，Arm 偏移沿五次最小加加速度曲线收缩到移动中的专家轨迹。
输出只保存“恢复过程＋恢复后若干帧”，剩余专家后缀在后台执行并验证最终
任务成功。输出保持 Quest 原始格式，可直接交给
``hugging_face/convert_data_to_hf.py``。
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_collect import (  # noqa: E402
    generate_contractive_view_recovery_trajectories as recovery_common,
)
from data_collect.augment_view_joint_trajectories import (  # noqa: E402
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
)


SCHEMA_VERSION = 1
EPISODE_NAMING = "source_episode_with_single_arm_recovery_branch_v1"
ARM_DIM = 6
LEFT_ARM_SLICE = slice(0, 6)
RIGHT_ARM_SLICE = slice(7, 13)
GRIPPER_INDICES = np.asarray([6, 13], dtype=np.int64)
ARM_SIDES = ("left", "right")


class ArmRecoveryBranchError(RuntimeError):
    """单个 Arm 恢复分支没有满足恢复或最终成功要求。"""


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


def _assign_balanced_arms(
    branch_identities: Sequence[tuple[int, int]],
    seed: int,
) -> dict[tuple[int, int], str]:
    """确定性地交替分配左右臂，保证计划分支数之差不超过1。"""

    ordered = sorted((int(source), int(variant)) for source, variant in branch_identities)
    if len(set(ordered)) != len(ordered):
        raise ValueError("branch_identities 不能包含重复项。")
    start_index = int(
        np.random.SeedSequence([int(seed), 0x41524D53]).generate_state(1)[0] % 2
    )
    return {
        identity: ARM_SIDES[(start_index + ordinal) % 2]
        for ordinal, identity in enumerate(ordered)
    }


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
) -> None:
    limits = {
        "other_arm": float(cfg.validation.branch_max_other_arm_joint_abs_error),
        "gripper": float(cfg.validation.branch_max_gripper_abs_error),
        "view": float(cfg.validation.branch_max_view_joint_abs_error),
    }
    for name, limit in limits.items():
        if float(errors[name]) > limit:
            raise ArmRecoveryBranchError(
                f"{phase}阶段{name}误差{float(errors[name]):.6g}超过阈值{limit:.6g}。"
            )


def _apply_unrecorded_arm_disturbance(
    *,
    env_obj,
    source_states: np.ndarray,
    source_actions: np.ndarray,
    event_frame: int,
    offset: np.ndarray,
    perturbed_arm: str,
    setup_steps: int,
    cfg: DictConfig,
) -> tuple[int, dict[str, Any], float]:
    """沿专家时间轴用真实物理步平滑建立 Arm OOD 状态，但不记录。"""

    source_index = int(event_frame)
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
    max_tracking_error = float(np.max(np.abs(tracking_error)))
    if max_tracking_error > float(
        cfg.validation.setup_offset_tracking_max_abs_error_rad
    ):
        raise ArmRecoveryBranchError(
            "物理注入后的实际 Arm 偏移未充分跟踪采样目标: "
            f"tracking_error={max_tracking_error:.6g} > "
            f"{float(cfg.validation.setup_offset_tracking_max_abs_error_rad):.6g}"
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
    event_frame: int,
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
            event_frame=event_frame,
            offset=offset,
            perturbed_arm=perturbed_arm,
            setup_steps=int(cfg.recovery.unrecorded_setup_steps),
            cfg=cfg,
        )
        recorded_first_source_frame = int(source_index)
        initial_arrays = recovery_common._capture_episode_initial_arrays(env_obj)

        planned_steps = recovery_common._adaptive_recovery_steps(
            offset,
            int(cfg.fps),
            np.asarray(max_extra_velocity_rad_s, dtype=np.float64),
            int(cfg.recovery.min_steps),
            int(cfg.recovery.max_steps),
        )
        max_extra_steps = int(cfg.recovery.max_extra_zero_offset_steps)
        stable_required = int(cfg.recovery.success_stable_steps)
        post_required = int(cfg.recovery.post_recovery_steps)
        recovery_threshold = float(cfg.recovery.success_max_abs_error_rad)

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
        post_frames = 0
        local_step = 0

        with StereoVideoWriter(tmp_dir / "videos", cameras, cfg) as writer:
            while source_index < len(source_actions):
                actual_state = _read_agent_state(env_obj)
                expert_state = source_states[source_index]
                errors = _branch_errors(
                    actual_state, expert_state, perturbed_arm
                )
                _validate_unperturbed_roles(errors, cfg, "Arm 恢复")
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
                        achieved_source_frame = int(source_index)
                if achieved_local_step is not None and local_step > achieved_local_step:
                    if selected_error > recovery_threshold:
                        raise ArmRecoveryBranchError(
                            "Arm 恢复达标后的专家跟随阶段再次离开误差管道: "
                            f"source_frame={source_index}, error={selected_error:.6g} > "
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
                command_offset = remaining * offset
                action = _arm_recovery_action(
                    source_actions[source_index],
                    command_offset,
                    perturbed_arm,
                )

                recovery_common._render_stereo(env_obj, writer, cameras, cfg)
                recorded_states.append(actual_state.astype(np.float32))
                recorded_actions.append(action.astype(np.float32))
                source_indices.append(int(source_index))
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
                if terminated and source_index != len(source_actions) - 1:
                    raise ArmRecoveryBranchError(
                        "Arm 恢复分支在专家后缀结束前提前成功终止；"
                        "拒绝在 terminal 状态后继续推进。"
                    )

                source_index += 1
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
        peak_extra_velocity = 1.875 * np.abs(offset) / planned_duration_s
        configured_max_abs = np.asarray(
            cfg.arm_joint_noise.max_abs_rad, dtype=np.float64
        )
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
                "source_injection_frame": int(event_frame),
                "source_recorded_first_frame": recorded_first_source_frame,
                "source_recorded_last_frame": int(source_indices[-1]),
                "arm_joint_offset_rad": offset.tolist(),
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
                "final_recorded_arm_max_abs_error_rad": final_arm_error,
                "max_recorded_selected_arm_max_abs_error_rad": float(
                    max_selected_error
                ),
                "max_recorded_other_arm_joint_abs_error": float(
                    max_other_arm_error
                ),
                "max_recorded_gripper_abs_error": float(max_gripper_error),
                "max_recorded_view_joint_abs_error": float(max_view_error),
                "sampling_seed": int(cfg.seed),
                "branch_attempt": int(attempt),
                "final_reward": float(last_reward),
                "background_suffix_validated": True,
                "background_suffix_final_source_frame": int(source_index - 1),
                "recovery_curve": "quintic_minimum_jerk_to_moving_expert",
                "only_selected_arm_action_modified": True,
                "other_arm_action_uses_expert": True,
                "gripper_action_uses_expert": True,
                "view_action_uses_expert": True,
                "final_info": {
                    "is_success": True,
                    "source_episode": int(source.episode_number),
                    "variant_index": int(variant_index),
                    "source_injection_frame": int(event_frame),
                    "perturbed_arm": perturbed_arm,
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
        ROOT_DIR / "data_collect/augment_view_joint_trajectories.py",
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
    return {
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
        "arm_assignment": "globally_balanced_alternating_with_seeded_start",
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
        "event_sampling.max_events_per_source": cfg.event_sampling.max_events_per_source,
        "recovery.min_steps": cfg.recovery.min_steps,
        "recovery.max_steps": cfg.recovery.max_steps,
        "recovery.success_stable_steps": cfg.recovery.success_stable_steps,
        "recovery.post_recovery_steps": cfg.recovery.post_recovery_steps,
        "recovery.unrecorded_setup_steps": cfg.recovery.unrecorded_setup_steps,
        "validation.max_branch_attempts": cfg.validation.max_branch_attempts,
        "arm_joint_noise.max_sampling_attempts": cfg.arm_joint_noise.max_sampling_attempts,
    }
    for name, raw in integer_positive.items():
        value = float(raw)
        if not np.isfinite(value) or not value.is_integer() or value <= 0:
            raise ValueError(f"{name} 必须为正整数。")
    integer_nonnegative = {
        "event_sampling.min_events_per_source": cfg.event_sampling.min_events_per_source,
        "event_sampling.min_injection_interval_steps": (
            cfg.event_sampling.min_injection_interval_steps
        ),
        "event_sampling.exclude_initial_steps": cfg.event_sampling.exclude_initial_steps,
        "recovery.max_extra_zero_offset_steps": cfg.recovery.max_extra_zero_offset_steps,
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
    if int(cfg.event_sampling.max_events_per_source) < int(
        cfg.event_sampling.min_events_per_source
    ):
        raise ValueError("max_events_per_source 不能小于 min_events_per_source。")

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
        "validation.setup_offset_tracking_max_abs_error_rad": (
            cfg.validation.setup_offset_tracking_max_abs_error_rad
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


def _load_action_length(source: SourceEpisode) -> int:
    with np.load(source.directory / "arrays.npz", allow_pickle=False) as data:
        if "joint_action" not in data:
            raise KeyError(f"{source.directory}/arrays.npz 缺少 joint_action。")
        actions = np.asarray(data["joint_action"])
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) == 0:
        raise ValueError(
            f"source_episode={source.episode_number} joint_action 形状非法: "
            f"{actions.shape}"
        )
    return int(len(actions))


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

    required_tail_steps = (
        int(cfg.recovery.unrecorded_setup_steps)
        + int(cfg.recovery.max_steps)
        + int(cfg.recovery.max_extra_zero_offset_steps)
        + int(cfg.recovery.post_recovery_steps)
        + 1
    )
    source_event_frames: dict[int, list[int]] = {}
    branch_identities: list[tuple[int, int]] = []
    eligible_sources: list[SourceEpisode] = []
    for source in sources:
        try:
            event_frames = recovery_common._sample_injection_frames(
                num_frames=_load_action_length(source),
                seed=int(cfg.seed),
                source_episode=source.episode_number,
                probability=float(
                    cfg.event_sampling.injection_probability_per_frame
                ),
                min_interval_steps=int(
                    cfg.event_sampling.min_injection_interval_steps
                ),
                min_events=int(cfg.event_sampling.min_events_per_source),
                max_events=int(cfg.event_sampling.max_events_per_source),
                exclude_initial_steps=int(
                    cfg.event_sampling.exclude_initial_steps
                ),
                required_tail_steps=required_tail_steps,
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
        source_event_frames[source.episode_number] = event_frames
        branch_identities.extend(
            (source.episode_number, variant_index)
            for variant_index in range(len(event_frames))
        )
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
    arm_assignments = _assign_balanced_arms(branch_identities, int(cfg.seed))

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
    metadata["planned_recovery_episodes_by_arm"] = {
        side: sum(value == side for value in arm_assignments.values())
        for side in ARM_SIDES
    }
    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)

    logging.info(
        "开始生成可收缩 Arm 恢复数据: env=%s source=%s output=%s "
        "successful_sources=%d skipped=%d branches=%d left/right=%s velocity_p%.1f=%s "
        "fingerprint=%s",
        env_id,
        input_run_dir,
        output_run_dir,
        len(sources),
        len(skipped_sources),
        len(branch_identities),
        metadata["planned_recovery_episodes_by_arm"],
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
                event_frames = source_event_frames[source.episode_number]
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
                        event_frames=event_frames,
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
                    "源 episode=%06d 名义重放成功，events=%s，max_error=%.3g",
                    source.episode_number,
                    event_frames,
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

            for variant_index, event_frame in enumerate(event_frames):
                perturbed_arm = arm_assignments[
                    (source.episode_number, variant_index)
                ]
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
                    expected = {
                        "source_injection_frame": int(event_frame),
                        "perturbed_arm": perturbed_arm,
                    }
                    mismatch = {
                        key: {"expected": value, "actual": completed.get(key)}
                        for key, value in expected.items()
                        if completed.get(key) != value
                    }
                    if mismatch:
                        raise RuntimeError(
                            f"已完成 Arm 恢复分支与确定性计划不一致: "
                            f"directory={final_dir}, mismatch={mismatch}"
                        )
                    recovery_common._clear_failure(
                        metadata, source.episode_number, variant_index
                    )
                    logging.info(
                        "跳过已完成 Arm 恢复分支: source=%06d variant=%02d "
                        "frame=%d arm=%s",
                        source.episode_number,
                        variant_index,
                        event_frame,
                        perturbed_arm,
                    )
                    continue

                horizon = required_tail_steps - 1
                try:
                    feasible_lower, feasible_upper = (
                        _local_arm_feasible_offset_bounds(
                            source_arrays,
                            event_frame,
                            horizon,
                            control_ranges[perturbed_arm],
                            max_abs,
                            float(cfg.arm_joint_noise.joint_limit_margin_rad),
                            perturbed_arm,
                        )
                    )
                except Exception as exc:
                    attempts = [{"error": f"{type(exc).__name__}: {exc}"}]
                    recovery_common._record_failure(
                        metadata,
                        source.episode_number,
                        variant_index,
                        event_frame,
                        attempts,
                    )
                    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
                    logging.exception(
                        "Arm 恢复可行域计算失败: source=%06d variant=%02d "
                        "frame=%d arm=%s",
                        source.episode_number,
                        variant_index,
                        event_frame,
                        perturbed_arm,
                    )
                    if not bool(cfg.continue_on_error):
                        raise
                    continue

                attempt_records: list[dict[str, Any]] = []
                succeeded = False
                for attempt in range(int(cfg.validation.max_branch_attempts)):
                    offset: np.ndarray | None = None
                    try:
                        offset = _sample_arm_recovery_offset(
                            seed=int(cfg.seed),
                            source_episode=source.episode_number,
                            variant_index=variant_index,
                            attempt=attempt,
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
                            snapshot=snapshots[event_frame],
                            event_frame=event_frame,
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
                        recovery_common._clear_failure(
                            metadata, source.episode_number, variant_index
                        )
                        _refresh_metadata(
                            metadata, output_run_dir, cameras, fingerprint
                        )
                        logging.info(
                            "已保存 Arm 恢复分支: source=%06d variant=%02d "
                            "frame=%d arm=%s attempt=%d offset=%s planned=%d "
                            "actual=%d steps=%d final_error=%.3g",
                            source.episode_number,
                            variant_index,
                            event_frame,
                            perturbed_arm,
                            attempt,
                            np.array2string(offset, precision=4, suppress_small=True),
                            result.info["planned_recovery_steps"],
                            result.info["actual_recovery_steps"],
                            result.info["steps"],
                            result.info["final_recorded_arm_max_abs_error_rad"],
                        )
                        succeeded = True
                        break
                    except Exception as exc:
                        attempt_records.append(
                            {
                                "attempt": int(attempt),
                                "perturbed_arm": perturbed_arm,
                                "error": f"{type(exc).__name__}: {exc}",
                                "offset_rad": (
                                    offset.tolist() if offset is not None else None
                                ),
                            }
                        )
                        logging.warning(
                            "Arm 恢复尝试失败，将重新采样: source=%06d "
                            "variant=%02d frame=%d arm=%s attempt=%d/%d error=%s",
                            source.episode_number,
                            variant_index,
                            event_frame,
                            perturbed_arm,
                            attempt + 1,
                            int(cfg.validation.max_branch_attempts),
                            exc,
                        )
                if not succeeded:
                    recovery_common._record_failure(
                        metadata,
                        source.episode_number,
                        variant_index,
                        event_frame,
                        attempt_records,
                    )
                    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
                    logging.error(
                        "Arm 恢复重试耗尽并跳过: source=%06d variant=%02d "
                        "frame=%d arm=%s",
                        source.episode_number,
                        variant_index,
                        event_frame,
                        perturbed_arm,
                    )
                    if not bool(cfg.continue_on_error):
                        raise ArmRecoveryBranchError(
                            f"Arm 恢复分支{source.episode_number}/{variant_index}失败。"
                        )
    finally:
        env_obj.close()

    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
    logging.info(
        "生成完成: saved=%d original=%d recovery=%d by_arm=%s failures=%d output=%s",
        metadata["saved_episodes"],
        metadata["original_episodes"],
        metadata["recovery_episodes"],
        metadata["recovery_episodes_by_arm"],
        len(metadata.get("failures", [])),
        output_run_dir,
    )


@hydra.main(
    version_base="1.2",
    config_path="../configs/data_collect",
    config_name="contractive_arm_trajectory_recovery",
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
