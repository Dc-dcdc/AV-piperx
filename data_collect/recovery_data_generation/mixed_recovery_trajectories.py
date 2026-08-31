#!/usr/bin/env python

"""生成 Arm/View 同时扰动并在同一窗口恢复的 Mixed Recovery 数据。

每个增强分支在同一专家时刻选择一只操作臂，并与中间 View 臂同步建立
固定关节偏移。设置阶段沿专家时间轴用真实物理步平滑执行但不保存；随后
Arm/View 使用同一五次最小加加速度剩余曲线和共同恢复时长回到持续推进的
专家轨迹。另一只 Arm 和两个夹爪始终采用同一时刻的专家动作。

Arm 与 View 分别沿用单源恢复的截断高斯分布，独立采样完整幅度；物理建立
完成后，以两类执行器实际达到且通过单源边界验收的偏移作为共同恢复起点。
输出保持 Quest 原始格式，可直接交给 hugging_face/convert_data_to_hf.py。
"""

from __future__ import annotations

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
    arm_recovery_trajectories as arm_common,
)
from data_collect.recovery_data_generation import (  # noqa: E402
    view_recovery_trajectories as recovery_common,
)
from data_collect.recovery_data_generation.trajectory_replay_common import (  # noqa: E402
    ACTION_DIM,
    VIEW_DIM,
    VIEW_SLICE,
    SourceEpisode,
    StereoVideoWriter,
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


SCHEMA_VERSION = 2
EPISODE_NAMING = "source_episode_with_fixed_quota_simultaneous_mixed_recovery_branch_v2"
ARM_DIM = arm_common.ARM_DIM
ARM_SIDES = arm_common.ARM_SIDES
GRIPPER_INDICES = arm_common.GRIPPER_INDICES
SUPPORTED_ENVS = {
    "guided_vision/SewNeedle-3Arms-v0",
    "guided_vision/InsertCylinder-3Arms-v0",
    "guided_vision/InsertPeg-3Arms-v0",
    "guided_vision/HookPackage-3Arms-v0",
}


class MixedRecoveryBranchError(RuntimeError):
    """单个 Mixed Recovery 分支没有满足双角色恢复或最终成功要求。"""


@dataclass(frozen=True)
class PlannedEvent:
    frame: int
    perturbed_arm: str
    candidate: arm_common.ArmRecoveryCandidate


@dataclass(frozen=True)
class MixedOffsets:
    arm: np.ndarray
    view: np.ndarray


def _mixed_recovery_action(
    expert_action: np.ndarray,
    arm_offset: np.ndarray,
    view_offset: np.ndarray,
    perturbed_arm: str,
) -> np.ndarray:
    """只给选中 Arm 和 View 的专家目标叠加偏移。"""

    action = np.asarray(expert_action, dtype=np.float64).copy()
    arm_offset = np.asarray(arm_offset, dtype=np.float64)
    view_offset = np.asarray(view_offset, dtype=np.float64)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"expert_action 必须为{ACTION_DIM}维。")
    if arm_offset.shape != (ARM_DIM,) or view_offset.shape != (VIEW_DIM,):
        raise ValueError("Arm/View offset 都必须为6维。")
    if not all(np.isfinite(value).all() for value in (action, arm_offset, view_offset)):
        raise ValueError("Mixed Recovery 动作不能包含NaN或Inf。")
    action[arm_common._arm_slice(perturbed_arm)] += arm_offset
    action[VIEW_SLICE] += view_offset
    return action


def _mixed_errors(
    actual: np.ndarray,
    expert: np.ndarray,
    perturbed_arm: str,
) -> dict[str, Any]:
    error = np.asarray(actual, dtype=np.float64) - np.asarray(
        expert, dtype=np.float64
    )
    selected_slice = arm_common._arm_slice(perturbed_arm)
    other_slice = arm_common._other_arm_slice(perturbed_arm)
    return {
        "all_vector": error.copy(),
        "arm_vector": error[selected_slice].copy(),
        "view_vector": error[VIEW_SLICE].copy(),
        "arm": float(np.max(np.abs(error[selected_slice]))),
        "view": float(np.max(np.abs(error[VIEW_SLICE]))),
        "other_arm": float(np.max(np.abs(error[other_slice]))),
        "gripper": float(np.max(np.abs(error[GRIPPER_INDICES]))),
    }


def _validate_unperturbed_roles(
    errors: dict[str, Any], cfg: DictConfig, phase: str
) -> None:
    limits = {
        "other_arm": float(cfg.validation.branch_max_other_arm_joint_abs_error),
        "gripper": float(cfg.validation.branch_max_gripper_abs_error),
    }
    for name, limit in limits.items():
        if float(errors[name]) > limit:
            raise MixedRecoveryBranchError(
                f"{phase}阶段{name}误差{float(errors[name]):.6g}超过阈值{limit:.6g}。"
            )


def _apply_unrecorded_mixed_disturbance(
    *,
    env_obj,
    source_states: np.ndarray,
    source_actions: np.ndarray,
    setup_start_frame: int,
    recovery_anchor_frame: int,
    offsets: MixedOffsets,
    perturbed_arm: str,
    setup_steps: int,
    cfg: DictConfig,
) -> tuple[int, dict[str, Any], float]:
    """沿移动专家轨迹同步、平滑建立 Arm/View 偏移，但不保存。"""

    source_index = int(setup_start_frame)
    last_reward = 0.0
    for setup_index in range(int(setup_steps)):
        if source_index >= len(source_actions):
            raise MixedRecoveryBranchError("源后缀不足以完成同步扰动设置。")
        fraction = float(
            recovery_common._quintic_smoothstep(
                (setup_index + 1) / float(setup_steps)
            )
        )
        action = _mixed_recovery_action(
            source_actions[source_index],
            fraction * offsets.arm,
            fraction * offsets.view,
            perturbed_arm,
        )
        last_reward, terminated, truncated = recovery_common._step_without_render(
            env_obj, action
        )
        if truncated:
            raise MixedRecoveryBranchError("同步扰动设置阶段被截断。")
        if terminated:
            outcome = "成功" if bool(getattr(env_obj, "is_success", False)) else "失败"
            raise MixedRecoveryBranchError(f"同步扰动设置阶段提前{outcome}终止。")
        source_index += 1
    if source_index != int(recovery_anchor_frame):
        raise RuntimeError(
            "Mixed扰动建立结束帧与恢复锚点不一致: "
            f"setup_start={setup_start_frame}, setup_steps={setup_steps}, "
            f"actual_end={source_index}, anchor={recovery_anchor_frame}"
        )
    if source_index >= len(source_states):
        raise MixedRecoveryBranchError("同步设置后没有剩余专家状态。")
    errors = _mixed_errors(
        _read_agent_state(env_obj), source_states[source_index], perturbed_arm
    )
    _validate_unperturbed_roles(errors, cfg, "同步扰动设置完成")
    arm_tracking = np.asarray(errors["arm_vector"]) - offsets.arm
    view_tracking = np.asarray(errors["view_vector"]) - offsets.view
    return (
        source_index,
        {
            "actual_arm_offset_rad": np.asarray(errors["arm_vector"]),
            "actual_view_offset_rad": np.asarray(errors["view_vector"]),
            "arm_tracking_error_rad": arm_tracking,
            "view_tracking_error_rad": view_tracking,
            "other_arm_error": float(errors["other_arm"]),
            "gripper_error": float(errors["gripper"]),
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


def _generate_mixed_recovery_branch(
    *,
    env_obj,
    source: SourceEpisode,
    source_arrays: dict[str, np.ndarray],
    snapshot: recovery_common.EnvironmentSnapshot,
    event: PlannedEvent,
    primary_anchor_frame: int,
    anchor_search_history: Sequence[dict[str, Any]],
    variant_index: int,
    attempt: int,
    offsets: MixedOffsets,
    arm_lower: np.ndarray,
    arm_upper: np.ndarray,
    view_lower: np.ndarray,
    view_upper: np.ndarray,
    arm_bound: np.ndarray,
    view_bound: np.ndarray,
    arm_velocity: np.ndarray,
    view_velocity: np.ndarray,
    output_run_dir: Path,
    cameras: tuple[str, ...],
    cfg: DictConfig,
    fingerprint: str,
) -> recovery_common.RecoveryResult:
    recovery_anchor_frame = int(event.frame)
    setup_steps = int(cfg.recovery.unrecorded_setup_steps)
    setup_start_frame = recovery_anchor_frame - setup_steps
    if setup_start_frame < 0:
        raise MixedRecoveryBranchError(
            f"恢复锚点{recovery_anchor_frame}之前不足{setup_steps}帧建立扰动。"
        )
    final_dir = recovery_common._episode_dir(
        output_run_dir, source.episode_number, variant_index
    )
    tmp_dir = recovery_common._prepare_tmp_directory(final_dir)
    try:
        recovery_common._restore_environment_snapshot(env_obj, snapshot)
        source_states = np.asarray(source_arrays["observation_state"], dtype=np.float64)
        source_actions = np.asarray(source_arrays["joint_action"], dtype=np.float64)
        source_index, setup_stats, last_reward = _apply_unrecorded_mixed_disturbance(
            env_obj=env_obj,
            source_states=source_states,
            source_actions=source_actions,
            setup_start_frame=setup_start_frame,
            recovery_anchor_frame=recovery_anchor_frame,
            offsets=offsets,
            perturbed_arm=event.perturbed_arm,
            setup_steps=setup_steps,
            cfg=cfg,
        )
        if source_index != recovery_anchor_frame:
            raise RuntimeError(
                f"Mixed恢复记录应从锚点{recovery_anchor_frame}开始，当前为{source_index}。"
            )
        first_source_frame = int(source_index)
        sampled_offsets = offsets
        recovery_offsets = MixedOffsets(
            arm=np.asarray(setup_stats["actual_arm_offset_rad"], dtype=np.float64),
            view=np.asarray(setup_stats["actual_view_offset_rad"], dtype=np.float64),
        )
        arm_normalized_l2 = arm_common._validate_actual_recovery_offset(
            recovery_offsets.arm,
            arm_lower,
            arm_upper,
            arm_bound,
            float(cfg.arm_joint_noise.min_normalized_l2),
        )
        view_normalized_l2 = recovery_common._validate_actual_view_recovery_offset(
            recovery_offsets.view,
            view_lower,
            view_upper,
            view_bound,
            float(cfg.view_joint_noise.min_normalized_l2),
        )
        initial_arrays = recovery_common._capture_episode_initial_arrays(env_obj)
        arm_required_steps = recovery_common._adaptive_recovery_steps(
            recovery_offsets.arm,
            int(cfg.fps),
            arm_velocity,
            int(cfg.recovery.min_steps),
            int(cfg.recovery.max_steps),
        )
        view_required_steps = recovery_common._adaptive_recovery_steps(
            recovery_offsets.view,
            int(cfg.fps),
            view_velocity,
            int(cfg.recovery.min_steps),
            int(cfg.recovery.max_steps),
        )
        planned_steps = max(arm_required_steps, view_required_steps)
        stable_required = int(cfg.recovery.success_stable_steps)
        max_extra_steps = int(cfg.recovery.max_extra_zero_offset_steps)
        post_required = int(cfg.recovery.post_recovery_steps)
        arm_threshold = float(cfg.recovery.success_max_abs_error_rad)
        view_threshold = float(cfg.recovery.success_max_abs_error_rad)

        recorded_states: list[np.ndarray] = []
        recorded_actions: list[np.ndarray] = []
        source_indices: list[int] = []
        arm_references: list[np.ndarray] = []
        view_references: list[np.ndarray] = []
        arm_commands: list[np.ndarray] = []
        view_commands: list[np.ndarray] = []
        arm_errors: list[np.ndarray] = []
        view_errors: list[np.ndarray] = []
        all_errors: list[np.ndarray] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []
        maxima = {"arm": 0.0, "view": 0.0, "other_arm": 0.0, "gripper": 0.0}
        stable_count = 0
        achieved_local_step: int | None = None
        achieved_source_frame: int | None = None
        post_frames = 0
        local_step = 0

        with StereoVideoWriter(tmp_dir / "videos", cameras, cfg) as writer:
            while source_index < len(source_actions):
                actual_state = _read_agent_state(env_obj)
                expert_state = source_states[source_index]
                errors = _mixed_errors(actual_state, expert_state, event.perturbed_arm)
                _validate_unperturbed_roles(errors, cfg, "Mixed恢复")
                for name in maxima:
                    maxima[name] = max(maxima[name], float(errors[name]))

                if local_step >= planned_steps:
                    both_recovered = (
                        float(errors["arm"]) <= arm_threshold
                        and float(errors["view"]) <= view_threshold
                    )
                    stable_count = stable_count + 1 if both_recovered else 0
                    if achieved_local_step is None and stable_count >= stable_required:
                        achieved_local_step = int(local_step)
                        achieved_source_frame = int(source_index)
                if achieved_local_step is not None and local_step > achieved_local_step:
                    if (
                        float(errors["arm"]) > arm_threshold
                        or float(errors["view"]) > view_threshold
                    ):
                        raise MixedRecoveryBranchError(
                            "双角色恢复达标后再次离开误差管道: "
                            f"frame={source_index}, arm={float(errors['arm']):.6g}, "
                            f"view={float(errors['view']):.6g}"
                        )
                    post_frames += 1
                if (
                    achieved_local_step is None
                    and local_step >= planned_steps + max_extra_steps
                ):
                    raise MixedRecoveryBranchError(
                        "共同曲线归零后Arm/View仍未同时稳定恢复: "
                        f"arm={float(errors['arm']):.6g}, view={float(errors['view']):.6g}, "
                        f"stable={stable_count}/{stable_required}"
                    )

                remaining = (
                    float(
                        recovery_common._quintic_remaining_fraction(
                            (local_step + 1) / float(planned_steps)
                        )
                    )
                    if local_step < planned_steps
                    else 0.0
                )
                arm_command = remaining * recovery_offsets.arm
                view_command = remaining * recovery_offsets.view
                action = _mixed_recovery_action(
                    source_actions[source_index],
                    arm_command,
                    view_command,
                    event.perturbed_arm,
                )

                recovery_common._render_stereo(env_obj, writer, cameras, cfg)
                recorded_states.append(actual_state.astype(np.float32))
                recorded_actions.append(action.astype(np.float32))
                source_indices.append(int(source_index))
                arm_references.append(
                    expert_state[arm_common._arm_slice(event.perturbed_arm)].astype(
                        np.float32
                    )
                )
                view_references.append(expert_state[VIEW_SLICE].astype(np.float32))
                arm_commands.append(arm_command.astype(np.float32))
                view_commands.append(view_command.astype(np.float32))
                arm_errors.append(np.asarray(errors["arm_vector"], dtype=np.float32))
                view_errors.append(np.asarray(errors["view_vector"], dtype=np.float32))
                all_errors.append(np.asarray(errors["all_vector"], dtype=np.float32))

                last_reward, terminated, truncated = recovery_common._step_without_render(
                    env_obj, action
                )
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                if truncated:
                    raise MixedRecoveryBranchError("Mixed恢复分支被环境上限截断。")
                if terminated and not bool(getattr(env_obj, "is_success", False)):
                    raise MixedRecoveryBranchError("Mixed恢复分支提前失败终止。")
                if terminated and source_index != len(source_actions) - 1:
                    raise MixedRecoveryBranchError("Mixed恢复分支提前成功终止。")
                source_index += 1
                local_step += 1
                if achieved_local_step is not None and post_frames >= post_required:
                    break

        if achieved_local_step is None or post_frames < post_required:
            raise MixedRecoveryBranchError(
                "源后缀不足以满足双角色恢复稳定帧与恢复后保存帧要求。"
            )

        while source_index < len(source_actions):
            last_reward, terminated, truncated = recovery_common._step_without_render(
                env_obj, source_actions[source_index]
            )
            if truncated:
                raise MixedRecoveryBranchError("后台专家后缀验证被截断。")
            if terminated and not bool(getattr(env_obj, "is_success", False)):
                raise MixedRecoveryBranchError(
                    f"后台专家后缀在frame={source_index}失败终止。"
                )
            if terminated and source_index != len(source_actions) - 1:
                raise MixedRecoveryBranchError("后台专家后缀提前成功终止。")
            source_index += 1
        if not bool(getattr(env_obj, "is_success", False)):
            raise MixedRecoveryBranchError("执行完整专家后缀后任务未成功。")

        arrays = {
            "joint_action": np.asarray(recorded_actions, dtype=np.float32),
            "observation_state": np.asarray(recorded_states, dtype=np.float32),
            "timestamp": np.arange(len(recorded_actions), dtype=np.float32)
            / float(cfg.fps),
            "terminated": np.asarray(terminated_flags, dtype=np.bool_),
            "truncated": np.asarray(truncated_flags, dtype=np.bool_),
            "source_frame_index": np.asarray(source_indices, dtype=np.int64),
            "recovery_reference_arm_state": np.asarray(arm_references, dtype=np.float32),
            "recovery_reference_view_state": np.asarray(view_references, dtype=np.float32),
            "recovery_arm_command_offset": np.asarray(arm_commands, dtype=np.float32),
            "recovery_view_command_offset": np.asarray(view_commands, dtype=np.float32),
            "recovery_command_residual": np.concatenate(
                (
                    np.asarray(arm_commands, dtype=np.float32),
                    np.asarray(view_commands, dtype=np.float32),
                ),
                axis=1,
            ),
            "recovery_arm_state_error": np.asarray(arm_errors, dtype=np.float32),
            "recovery_view_state_error": np.asarray(view_errors, dtype=np.float32),
            "recovery_all_state_error": np.asarray(all_errors, dtype=np.float32),
            **initial_arrays,
        }
        np.savez_compressed(tmp_dir / "arrays.npz", **arrays)

        duration = planned_steps / float(cfg.fps)
        arm_peak_velocity = 1.875 * np.abs(recovery_offsets.arm) / duration
        view_peak_velocity = 1.875 * np.abs(recovery_offsets.view) / duration
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
                "recovery_type": "simultaneous_mixed_recovery",
                "perturbed_arm": event.perturbed_arm,
                "source_injection_frame": int(event.frame),
                "source_recovery_anchor_frame": int(event.frame),
                "source_primary_anchor_frame": int(primary_anchor_frame),
                "anchor_search_history": [
                    dict(value) for value in anchor_search_history
                ],
                "source_recorded_first_frame": first_source_frame,
                "source_recorded_last_frame": int(source_indices[-1]),
                "sampling_mode": event.candidate.event.sampling_mode,
                "sampling_source": event.candidate.event.sampling_source,
                "sampling_region_index": event.candidate.event.region_index,
                "sampling_region_start_normalized": (
                    event.candidate.event.region_start_normalized
                ),
                "sampling_region_end_normalized": (
                    event.candidate.event.region_end_normalized
                ),
                "sampling_domain": event.candidate.domain_key,
                "arm_selection_mode": "local_motion",
                "arm_motion_window_start_frame": int(
                    event.candidate.motion_selection.window_start_frame
                ),
                "arm_motion_window_end_frame": int(
                    event.candidate.motion_selection.window_end_frame
                ),
                "left_arm_rms_velocity_rad_s": float(
                    event.candidate.motion_selection.left_rms_velocity_rad_s
                ),
                "right_arm_rms_velocity_rad_s": float(
                    event.candidate.motion_selection.right_rms_velocity_rad_s
                ),
                "selected_arm_motion_dominance_ratio": float(
                    event.candidate.motion_selection.dominance_ratio
                ),
                "arm_joint_offset_rad": recovery_offsets.arm.tolist(),
                "view_joint_offset_rad": recovery_offsets.view.tolist(),
                "sampled_arm_joint_offset_rad": sampled_offsets.arm.tolist(),
                "sampled_view_joint_offset_rad": sampled_offsets.view.tolist(),
                "recovery_initial_arm_offset_rad": recovery_offsets.arm.tolist(),
                "recovery_initial_view_offset_rad": recovery_offsets.view.tolist(),
                "recovery_uses_actual_achieved_offset": True,
                "actual_arm_offset_normalized_l2": float(arm_normalized_l2),
                "actual_view_offset_normalized_l2": float(view_normalized_l2),
                "effective_arm_offset_max_abs_rad": arm_bound.tolist(),
                "effective_view_offset_max_abs_rad": view_bound.tolist(),
                "arm_offset_feasible_lower_rad": arm_lower.tolist(),
                "arm_offset_feasible_upper_rad": arm_upper.tolist(),
                "view_offset_feasible_lower_rad": view_lower.tolist(),
                "view_offset_feasible_upper_rad": view_upper.tolist(),
                "unrecorded_disturbance_setup_steps": int(
                    cfg.recovery.unrecorded_setup_steps
                ),
                "unrecorded_setup_curve": "synchronous_quintic_minimum_jerk_from_moving_expert",
                "setup_actual_arm_offset_rad": setup_stats[
                    "actual_arm_offset_rad"
                ].tolist(),
                "setup_actual_view_offset_rad": setup_stats[
                    "actual_view_offset_rad"
                ].tolist(),
                "setup_arm_tracking_error_rad": setup_stats[
                    "arm_tracking_error_rad"
                ].tolist(),
                "setup_view_tracking_error_rad": setup_stats[
                    "view_tracking_error_rad"
                ].tolist(),
                "arm_required_recovery_steps": int(arm_required_steps),
                "view_required_recovery_steps": int(view_required_steps),
                "common_recovery_steps": int(planned_steps),
                "planned_recovery_steps": int(planned_steps),
                "planned_recovery_duration_s": float(duration),
                "arm_recovery_peak_extra_velocity_rad_s": arm_peak_velocity.tolist(),
                "view_recovery_peak_extra_velocity_rad_s": view_peak_velocity.tolist(),
                "resolved_arm_max_extra_velocity_rad_s": arm_velocity.tolist(),
                "resolved_view_max_extra_velocity_rad_s": view_velocity.tolist(),
                "actual_recovery_steps": int(achieved_local_step),
                "recovery_achieved_source_frame": int(achieved_source_frame),
                "arm_recovery_success_max_abs_error_rad": arm_threshold,
                "view_recovery_success_max_abs_error_rad": view_threshold,
                "recovery_success_stable_steps": stable_required,
                "recovery_post_steps": int(post_frames),
                "final_recorded_arm_max_abs_error_rad": float(
                    np.max(np.abs(arm_errors[-1]))
                ),
                "final_recorded_view_max_abs_error_rad": float(
                    np.max(np.abs(view_errors[-1]))
                ),
                "max_recorded_arm_max_abs_error_rad": maxima["arm"],
                "max_recorded_view_max_abs_error_rad": maxima["view"],
                "max_recorded_other_arm_joint_abs_error": maxima["other_arm"],
                "max_recorded_gripper_abs_error": maxima["gripper"],
                "sampling_seed": int(cfg.seed),
                "branch_attempt": int(attempt),
                "final_reward": float(last_reward),
                "background_suffix_validated": True,
                "background_suffix_final_source_frame": int(source_index - 1),
                "recovery_curve": "shared_quintic_minimum_jerk_to_moving_expert",
                "selected_arm_and_view_actions_modified": True,
                "other_arm_action_uses_expert": True,
                "gripper_action_uses_expert": True,
                "final_info": {
                    "is_success": True,
                    "source_episode": int(source.episode_number),
                    "variant_index": int(variant_index),
                    "source_injection_frame": int(event.frame),
                    "perturbed_arm": event.perturbed_arm,
                    "arm_and_view_recovery_achieved": True,
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


def _view_control_ranges(env_obj) -> np.ndarray:
    joint_ranges = np.asarray(
        env_obj._physics.bind(env_obj._middle_joints).range, dtype=np.float64
    )
    actuator_ranges = np.asarray(
        env_obj._physics.bind(env_obj._middle_actuators).ctrlrange,
        dtype=np.float64,
    )
    ranges = np.stack(
        (
            np.maximum(joint_ranges[:, 0], actuator_ranges[:, 0]),
            np.minimum(joint_ranges[:, 1], actuator_ranges[:, 1]),
        ),
        axis=1,
    )
    if ranges.shape != (VIEW_DIM, 2) or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise RuntimeError("View关节限位和执行器限位没有有效交集。")
    return ranges


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
        Path(arm_common.__file__).resolve(),
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
            raise FileNotFoundError(f"生成依赖文件不存在: {path}")
        result[str(path.relative_to(ROOT_DIR))] = _sha256_file(path)
    return result


def _semantic_config(
    cfg: DictConfig,
    input_run_dir: Path,
    env_id: str,
    arm_velocity_statistics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_naming": EPISODE_NAMING,
        "input_run_dir": str(input_run_dir),
        "env_id": env_id,
        "include_original": bool(cfg.include_original),
        "max_source_episodes": (
            None if cfg.max_source_episodes is None else int(cfg.max_source_episodes)
        ),
        "source_episode_indices": (
            None
            if cfg.source_episode_indices is None
            else sorted(int(value) for value in cfg.source_episode_indices)
        ),
        "seed": int(cfg.seed),
        "event_sampling": OmegaConf.to_container(cfg.event_sampling, resolve=True),
        "arm_joint_noise": OmegaConf.to_container(cfg.arm_joint_noise, resolve=True),
        "view_joint_noise": OmegaConf.to_container(cfg.view_joint_noise, resolve=True),
        "arm_selection": OmegaConf.to_container(cfg.arm_selection, resolve=True),
        "auto_velocity": OmegaConf.to_container(
            cfg.auto_velocity, resolve=True
        ),
        "resolved_arm_velocity_statistics": arm_velocity_statistics,
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
    metadata["recovery_episodes_by_arm"] = {
        side: sum(info.get("perturbed_arm") == side for info in infos)
        for side in ARM_SIDES
    }
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
        raise FileExistsError(f"输出目录已存在且resume=false: {output_run_dir}")
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "episodes").mkdir(exist_ok=True)
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        if metadata.get("generation_fingerprint") != fingerprint:
            raise ValueError(
                "续生成配置或代码与已有输出不一致，请更换output_run_dir: "
                f"existing={metadata.get('generation_fingerprint')!r}, current={fingerprint!r}"
            )
    else:
        unexpected = [path for path in output_run_dir.iterdir() if path.name != "episodes"]
        if unexpected or any((output_run_dir / "episodes").iterdir()):
            raise RuntimeError(f"输出目录非空但缺少metadata.json: {output_run_dir}")
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
            "video_save_mode": "pre_action_physical_simultaneous_mixed_recovery",
            "success_semantics": "dual_recovery_and_full_suffix_task_success",
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
    cfg: DictConfig, source_metadata: dict[str, Any]
) -> tuple[str, tuple[str, ...]]:
    metadata_env_id = source_metadata.get("env_id")
    env_id = str(metadata_env_id) if cfg.env_id is None else str(cfg.env_id)
    if env_id not in SUPPORTED_ENVS:
        raise ValueError(f"Mixed Recovery不支持env_id={env_id!r}。")
    if cfg.env_id is not None and metadata_env_id is not None and str(metadata_env_id) != env_id:
        raise ValueError("配置env_id与源metadata不一致。")
    cameras = tuple(str(value) for value in cfg.cameras)
    if cameras != ("zed_cam_left", "zed_cam_right"):
        raise ValueError("Mixed Recovery固定只保存zed_cam_left/right。")
    if source_metadata.get("fps") is not None and int(source_metadata["fps"]) != int(cfg.fps):
        raise ValueError("配置fps与源metadata不一致。")
    mode = str(cfg.event_sampling.mode)
    if mode not in {"random", "specified_region", "hybrid"}:
        raise ValueError(
            "event_sampling.mode必须是random、specified_region或hybrid。"
        )
    regions = list(cfg.event_sampling.normalized_regions)
    if mode == "specified_region" and not regions:
        raise ValueError("specified_region模式至少需要一个normalized_regions区间。")
    previous_end = -np.inf
    for region_index, region in enumerate(regions):
        start = float(region.start)
        end = float(region.end)
        if not (np.isfinite(start) and np.isfinite(end) and 0.0 <= start < end <= 1.0):
            raise ValueError(
                f"normalized_regions[{region_index}]必须满足0<=start<end<=1。"
            )
        if start < previous_end:
            raise ValueError("normalized_regions必须按start升序且互不重叠。")
        previous_end = end
        probability = float(region.injection_probability_per_frame)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"normalized_regions[{region_index}]."
                "injection_probability_per_frame必须位于[0,1]。"
            )
        interval = float(region.min_injection_interval_steps)
        if not np.isfinite(interval) or not interval.is_integer() or interval < 0:
            raise ValueError(
                f"normalized_regions[{region_index}]."
                "min_injection_interval_steps必须为非负整数。"
            )
    if str(cfg.arm_selection.mode) != "local_motion":
        raise ValueError("arm_selection.mode当前仅支持local_motion。")
    for name, raw, strictly_positive in (
        ("lookback_steps", cfg.arm_selection.lookback_steps, False),
        ("lookahead_steps", cfg.arm_selection.lookahead_steps, True),
    ):
        value = float(raw)
        minimum = 1 if strictly_positive else 0
        if not np.isfinite(value) or not value.is_integer() or value < minimum:
            qualifier = "正整数" if strictly_positive else "非负整数"
            raise ValueError(f"arm_selection.{name}必须为{qualifier}。")
    min_motion = float(cfg.arm_selection.min_rms_velocity_rad_s)
    if not np.isfinite(min_motion) or min_motion < 0.0:
        raise ValueError("arm_selection.min_rms_velocity_rad_s必须为有限非负数。")
    vector_fields = {
        "arm_joint_noise.std_rad": cfg.arm_joint_noise.std_rad,
        "arm_joint_noise.max_abs_rad": cfg.arm_joint_noise.max_abs_rad,
        "view_joint_noise.std_rad": cfg.view_joint_noise.std_rad,
        "view_joint_noise.max_abs_rad": cfg.view_joint_noise.max_abs_rad,
        "auto_velocity.floor_rad_s": cfg.auto_velocity.floor_rad_s,
        "recovery.max_extra_velocity_rad_s": cfg.recovery.max_extra_velocity_rad_s,
    }
    for name, raw in vector_fields.items():
        value = np.asarray(raw, dtype=np.float64)
        if value.shape != (6,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"{name}必须是6维有限正数。")
    for role in ("arm_joint_noise", "view_joint_noise"):
        if str(cfg[role].distribution) != "truncated_gaussian":
            raise ValueError(f"{role}.distribution必须为truncated_gaussian。")
        min_norm = float(cfg[role].min_normalized_l2)
        if not np.isfinite(min_norm) or not 0.0 <= min_norm <= np.sqrt(6.0):
            raise ValueError(f"{role}.min_normalized_l2范围非法。")
    positive_ints = {
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
        "view_joint_noise.max_sampling_attempts": cfg.view_joint_noise.max_sampling_attempts,
    }
    for name, raw in positive_ints.items():
        value = float(raw)
        if not np.isfinite(value) or not value.is_integer() or value <= 0:
            raise ValueError(f"{name}必须为正整数。")
    nonnegative_ints = {
        "event_sampling.min_injection_interval_steps": (
            cfg.event_sampling.min_injection_interval_steps
        ),
        "event_sampling.exclude_initial_steps": cfg.event_sampling.exclude_initial_steps,
        "recovery.max_extra_zero_offset_steps": cfg.recovery.max_extra_zero_offset_steps,
        "recovery.post_recovery_steps": cfg.recovery.post_recovery_steps,
    }
    for name, raw in nonnegative_ints.items():
        value = float(raw)
        if not np.isfinite(value) or not value.is_integer() or value < 0:
            raise ValueError(f"{name}必须为非负整数。")
    if int(cfg.recovery.max_steps) < int(cfg.recovery.min_steps):
        raise ValueError("recovery.max_steps不能小于min_steps。")
    if int(cfg.recovery.success_stable_steps) > (
        int(cfg.recovery.max_extra_zero_offset_steps) + 1
    ):
        raise ValueError(
            "success_stable_steps不能大于max_extra_zero_offset_steps+1。"
        )
    probability = float(cfg.event_sampling.injection_probability_per_frame)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("注入概率必须位于[0,1]。")
    positive_floats = {
        "arm_joint_noise.joint_limit_margin_rad": cfg.arm_joint_noise.joint_limit_margin_rad,
        "view_joint_noise.joint_limit_margin_rad": cfg.view_joint_noise.joint_limit_margin_rad,
        "auto_velocity.scale": cfg.auto_velocity.scale,
        "recovery.success_max_abs_error_rad": cfg.recovery.success_max_abs_error_rad,
        "validation.max_arm_joint_abs_error": cfg.validation.max_arm_joint_abs_error,
        "validation.max_gripper_abs_error": cfg.validation.max_gripper_abs_error,
        "validation.max_view_joint_abs_error": cfg.validation.max_view_joint_abs_error,
        "validation.branch_max_other_arm_joint_abs_error": (
            cfg.validation.branch_max_other_arm_joint_abs_error
        ),
        "validation.branch_max_gripper_abs_error": cfg.validation.branch_max_gripper_abs_error,
    }
    for name, raw in positive_floats.items():
        if not np.isfinite(float(raw)) or float(raw) <= 0:
            raise ValueError(f"{name}必须为有限正数。")
    if not 0.0 < float(cfg.auto_velocity.percentile) <= 100.0:
        raise ValueError("auto_velocity.percentile必须位于(0,100]。")
    if cfg.max_source_episodes is not None and int(cfg.max_source_episodes) <= 0:
        raise ValueError("max_source_episodes必须为正整数或null。")
    seed = float(cfg.seed)
    if not np.isfinite(seed) or not seed.is_integer() or seed < 0:
        raise ValueError("seed必须为非负整数。")
    if cfg.source_episode_indices is not None:
        for raw_index in cfg.source_episode_indices:
            index = float(raw_index)
            if not np.isfinite(index) or not index.is_integer() or index < 0:
                raise ValueError("source_episode_indices必须全部为非负整数。")
    return env_id, cameras


def generate_simultaneous_mixed_recovery_run(cfg: DictConfig) -> None:
    """按单源v4语义生成固定配额、独立完整幅度的Mixed恢复数据。"""

    input_run_dir = recovery_common._resolve_path(cfg.input_run_dir)
    output_run_dir = recovery_common._resolve_path(cfg.output_run_dir)
    if input_run_dir == output_run_dir:
        raise ValueError("input_run_dir与output_run_dir不能相同。")
    source_metadata_path = input_run_dir / "metadata.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"输入run缺少metadata.json: {source_metadata_path}")
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
    setup_steps = int(cfg.recovery.unrecorded_setup_steps)
    required_tail_steps = (
        int(cfg.recovery.max_steps)
        + int(cfg.recovery.max_extra_zero_offset_steps)
        + int(cfg.recovery.post_recovery_steps)
        + 1
    )
    target_branches = int(cfg.event_sampling.recovery_branches_per_source)

    source_candidates: dict[int, list[arm_common.ArmRecoveryCandidate]] = {}
    skipped_inactive_events: list[dict[str, Any]] = []
    eligible_sources: list[SourceEpisode] = []
    for source in sources:
        try:
            states = arm_common._load_planning_states(source)
            candidates, inactive_frames = arm_common._build_arm_recovery_candidates(
                states=states,
                source_episode=source.episode_number,
                setup_steps=setup_steps,
                required_tail_steps=required_tail_steps,
                cfg=cfg,
            )
            if len(candidates) < target_branches:
                raise ValueError(
                    f"有效候选锚点仅{len(candidates)}个，少于固定配额"
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
                        "mixed_recovery_event_planning_failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )
            logging.warning(
                "源episode=%06d无法规划Mixed恢复事件，已跳过: %s",
                source.episode_number,
                exc,
            )
            continue
        eligible_sources.append(source)
        source_candidates[source.episode_number] = candidates
    sources = eligible_sources
    if not sources:
        raise RuntimeError("没有源episode能够容纳当前Mixed注入与恢复窗口。")

    arm_velocity_statistics = arm_common._estimate_arm_velocity_statistics(
        sources,
        int(cfg.fps),
        float(cfg.auto_velocity.percentile),
        np.asarray(cfg.auto_velocity.floor_rad_s, dtype=np.float64),
        float(cfg.auto_velocity.scale),
    )
    arm_velocity = np.asarray(
        arm_velocity_statistics["resolved_max_extra_velocity_rad_s"],
        dtype=np.float64,
    )
    view_velocity = np.asarray(
        cfg.recovery.max_extra_velocity_rad_s, dtype=np.float64
    )
    duration_factor = int(cfg.recovery.max_steps) / (1.875 * int(cfg.fps))
    arm_configured_bound = np.asarray(
        cfg.arm_joint_noise.max_abs_rad, dtype=np.float64
    )
    view_configured_bound = np.asarray(
        cfg.view_joint_noise.max_abs_rad, dtype=np.float64
    )
    arm_bound = np.minimum(arm_configured_bound, arm_velocity * duration_factor)
    view_bound = np.minimum(view_configured_bound, view_velocity * duration_factor)
    if np.any(arm_bound <= 0.0) or np.any(view_bound <= 0.0):
        raise RuntimeError("速度约束得到的Arm/View有效偏移上限必须为正。")

    semantic_config = _semantic_config(
        cfg, input_run_dir, env_id, arm_velocity_statistics
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
    metadata["recovery_branches_per_source"] = target_branches
    metadata["candidate_anchor_frames"] = sum(
        len(values) for values in source_candidates.values()
    )
    metadata["skipped_inactive_arm_events"] = skipped_inactive_events
    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)

    logging.info(
        "开始生成Mixed Recovery v2: env=%s source=%s output=%s "
        "sources=%d quota/source=%d arm_bound=%s view_bound=%s fingerprint=%s",
        env_id,
        input_run_dir,
        output_run_dir,
        len(sources),
        target_branches,
        np.array2string(arm_bound, precision=4),
        np.array2string(view_bound, precision=4),
        fingerprint,
    )

    env_obj = _make_environment(
        env_id, cameras, int(cfg.render_height), int(cfg.render_width)
    )
    try:
        env_obj.reset(seed=0)
        required_model_body_names = tuple(
            str(name) for name in getattr(env_obj, "replay_model_body_names", ())
        )
        arm_ranges = {
            side: arm_common._arm_control_ranges(env_obj, side) for side in ARM_SIDES
        }
        view_ranges = _view_control_ranges(env_obj)
        arm_std = np.asarray(cfg.arm_joint_noise.std_rad, dtype=np.float64)
        view_std = np.asarray(cfg.view_joint_noise.std_rad, dtype=np.float64)

        for source in sources:
            candidates = source_candidates[source.episode_number]
            original_final_dir = recovery_common._episode_dir(
                output_run_dir, source.episode_number, -1
            )
            original_tmp_dir: Path | None = None
            try:
                source_arrays = _validate_source_arrays(
                    source.directory / "arrays.npz",
                    required_model_body_names=required_model_body_names,
                )
                episode_fps = source.info.get("fps")
                if episode_fps is not None and int(episode_fps) != int(cfg.fps):
                    raise ValueError("源episode fps与Mixed配置不一致。")
                snapshot_frames = [
                    candidate.event.frame - setup_steps for candidate in candidates
                ]
                if bool(cfg.include_original):
                    if _episode_is_complete(original_final_dir, cameras):
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
                recovery_common._clear_failure(metadata, source.episode_number, None)
                _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
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
                    "源episode=%06d准备或名义重放失败。", source.episode_number
                )
                if not bool(cfg.continue_on_error):
                    raise
                continue

            candidate_by_frame = {
                candidate.event.frame: candidate for candidate in candidates
            }
            successful_candidates: list[arm_common.ArmRecoveryCandidate] = []
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
                            "已完成Mixed分支锚点不在当前确定性候选池中。"
                        )
                    completed_candidate = candidate_by_frame[completed_frame]
                    if completed.get("perturbed_arm") != (
                        completed_candidate.motion_selection.side
                    ):
                        raise RuntimeError("已完成Mixed分支主运动臂与候选计划不一致。")
                    successful_candidates.append(completed_candidate)
                    unavailable_frames.add(completed_frame)
                    recovery_common._clear_failure(
                        metadata, source.episode_number, variant_index
                    )
                    continue

                search_history: list[dict[str, Any]] = []
                for failure in metadata.get("failures", []):
                    if (
                        failure.get("source_episode") == source.episode_number
                        and failure.get("variant_index") == variant_index
                    ):
                        search_history = [
                            dict(value) for value in failure.get("attempts", [])
                        ]
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
                                and arm_common._candidate_is_spaced_from_successes(
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
                        break

                    neighbor_queue = arm_common._neighbor_candidates(
                        primary=primary,
                        all_candidates=candidates,
                        unavailable_frames=unavailable_frames,
                        successful_candidates=successful_candidates,
                    )
                    if not neighbor_queue:
                        pending_primary_frame = None
                        continue
                    for candidate_rank, candidate in enumerate(neighbor_queue):
                        event_frame = int(candidate.event.frame)
                        setup_start_frame = event_frame - setup_steps
                        perturbed_arm = candidate.motion_selection.side
                        try:
                            arm_lower, arm_upper = (
                                arm_common._local_arm_feasible_offset_bounds(
                                    source_arrays,
                                    setup_start_frame,
                                    setup_steps + required_tail_steps - 1,
                                    arm_ranges[perturbed_arm],
                                    arm_bound,
                                    float(cfg.arm_joint_noise.joint_limit_margin_rad),
                                    perturbed_arm,
                                )
                            )
                            view_lower, view_upper = (
                                recovery_common._local_feasible_offset_bounds(
                                    source_arrays,
                                    setup_start_frame,
                                    setup_steps + required_tail_steps - 1,
                                    view_ranges,
                                    view_bound,
                                    float(cfg.view_joint_noise.joint_limit_margin_rad),
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
                        for attempt in range(int(cfg.validation.max_branch_attempts)):
                            offsets: MixedOffsets | None = None
                            try:
                                arm_offset = arm_common._sample_arm_recovery_offset(
                                    seed=int(cfg.seed),
                                    source_episode=source.episode_number,
                                    variant_index=variant_index,
                                    attempt=attempt,
                                    anchor_frame=event_frame,
                                    perturbed_arm=perturbed_arm,
                                    std=arm_std,
                                    max_abs=arm_bound,
                                    feasible_lower=arm_lower,
                                    feasible_upper=arm_upper,
                                    min_normalized_l2=float(
                                        cfg.arm_joint_noise.min_normalized_l2
                                    ),
                                    max_sampling_attempts=int(
                                        cfg.arm_joint_noise.max_sampling_attempts
                                    ),
                                )
                                view_offset = recovery_common._sample_recovery_offset(
                                    seed=int(cfg.seed),
                                    source_episode=source.episode_number,
                                    variant_index=variant_index,
                                    attempt=attempt,
                                    anchor_frame=event_frame,
                                    std=view_std,
                                    max_abs=view_bound,
                                    feasible_lower=view_lower,
                                    feasible_upper=view_upper,
                                    min_normalized_l2=float(
                                        cfg.view_joint_noise.min_normalized_l2
                                    ),
                                    max_sampling_attempts=int(
                                        cfg.view_joint_noise.max_sampling_attempts
                                    ),
                                )
                                offsets = MixedOffsets(
                                    arm=arm_offset, view=view_offset
                                )
                                event = PlannedEvent(
                                    frame=event_frame,
                                    perturbed_arm=perturbed_arm,
                                    candidate=candidate,
                                )
                                result = _generate_mixed_recovery_branch(
                                    env_obj=env_obj,
                                    source=source,
                                    source_arrays=source_arrays,
                                    snapshot=snapshots[setup_start_frame],
                                    event=event,
                                    primary_anchor_frame=primary.event.frame,
                                    anchor_search_history=(
                                        search_history + frame_attempts
                                    ),
                                    variant_index=variant_index,
                                    attempt=attempt,
                                    offsets=offsets,
                                    arm_lower=arm_lower,
                                    arm_upper=arm_upper,
                                    view_lower=view_lower,
                                    view_upper=view_upper,
                                    arm_bound=arm_bound,
                                    view_bound=view_bound,
                                    arm_velocity=arm_velocity,
                                    view_velocity=view_velocity,
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
                                    metadata, output_run_dir, cameras, fingerprint
                                )
                                logging.info(
                                    "已保存Mixed分支: source=%06d variant=%02d "
                                    "primary=%d anchor=%d arm=%s attempt=%d H=%d steps=%d",
                                    source.episode_number,
                                    variant_index,
                                    primary.event.frame,
                                    event_frame,
                                    perturbed_arm,
                                    attempt,
                                    result.info["common_recovery_steps"],
                                    result.info["steps"],
                                )
                                succeeded = True
                                break
                            except Exception as exc:
                                frame_attempts.append(
                                    {
                                        "candidate_anchor_frame": event_frame,
                                        "primary_anchor_frame": primary.event.frame,
                                        "candidate_setup_start_frame": setup_start_frame,
                                        "candidate_domain": candidate.domain_key,
                                        "candidate_rank_from_primary": candidate_rank,
                                        "attempt": int(attempt),
                                        "perturbed_arm": perturbed_arm,
                                        "sampled_arm_offset_rad": (
                                            None
                                            if offsets is None
                                            else offsets.arm.tolist()
                                        ),
                                        "sampled_view_offset_rad": (
                                            None
                                            if offsets is None
                                            else offsets.view.tolist()
                                        ),
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )
                                logging.warning(
                                    "Mixed尝试失败，将在当前锚点同时重采Arm/View: "
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
                            "当前Mixed锚点重试耗尽，转向同域邻近帧: "
                            "source=%06d variant=%02d anchor=%d domain=%s",
                            source.episode_number,
                            variant_index,
                            event_frame,
                            candidate.domain_key,
                        )
                    if succeeded:
                        break
                    pending_primary_frame = None

                if not succeeded and not bool(cfg.continue_on_error):
                    raise MixedRecoveryBranchError(
                        f"source={source.episode_number}仅生成"
                        f"{len(successful_candidates)}/{target_branches}条Mixed分支。"
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
        entry.update(
            {"target": target_branches, "saved": saved, "satisfied": saved == target_branches}
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
        "Mixed v2生成完成: saved=%d original=%d recovery=%d by_arm=%s "
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
            "固定Mixed恢复分支配额未全部满足；结果已保存，但不能作为完整"
            "固定配额数据集使用。未完成源episode="
            f"{metadata['quota_incomplete_sources']}"
        )


@hydra.main(
    version_base="1.2",
    config_path="../../configs/data_collect",
    config_name="mixed_trajectory_recovery",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _configure_mujoco_runtime(cfg)
    generate_simultaneous_mixed_recovery_run(cfg)


if __name__ == "__main__":
    main()
