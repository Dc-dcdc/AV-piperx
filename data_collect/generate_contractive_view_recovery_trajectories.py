#!/usr/bin/env python

"""生成主动视觉View关节扰动后的可收缩恢复轨迹。

每个增强episode只包含一个恢复事件：先在不记录、不推进任务物理时间的
设置阶段把View机械臂放到扰动状态，再从该OOD状态开始记录真实MuJoCo
闭环恢复。左右操作臂和夹爪始终使用同一时刻的原专家动作；View动作沿
五次最小加加速度曲线逐步收缩到移动中的专家轨迹。

输出保持Quest原始数据格式，可直接交给``hugging_face/convert_data_to_hf.py``。
原始成功轨迹命名为``episode_<source>``，恢复分支命名为
``episode_<source>_aug_<variant>``。
"""

from __future__ import annotations

import copy
import logging
import re
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_collect.augment_view_joint_trajectories import (  # noqa: E402
    ACTION_DIM,
    ARM_JOINT_INDICES,
    GRIPPER_INDICES,
    MODEL_BODY_INITIAL_KEYS,
    OPTIONAL_FRAME_KEYS,
    VIEW_DIM,
    VIEW_JOINT_INDICES,
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
    _restore_initial_state,
    _sha256_file,
    _validate_source_arrays,
    _write_json_atomic,
)
from env.constants import SIM_PHYSICS_ENV_STEP_RATIO  # noqa: E402


SCHEMA_VERSION = 1
EPISODE_NAMING = "source_episode_with_single_recovery_branch_v1"
SOURCE_EPISODE_PATTERN = re.compile(r"^episode_(?P<episode>\d{6,})$")
OUTPUT_EPISODE_PATTERN = re.compile(
    r"^episode_(?P<source>\d{6,})(?:_aug_(?P<variant>\d{2,}))?$"
)
TASK_STATE_FIELDS = (
    # GuidedVisionEnv基础状态。
    "_current_step",
    "terminated",
    "is_success",
    "reward_debug",
    # InsertCylinder阶段状态。
    "right_has_grasped",
    "left_has_received",
    "right_has_released",
    "cylinder_was_grasped",
    "placement_checked",
    "_prev_metrics",
    # SewNeedle阶段状态。
    "needle_start_through",
    "needle_reached_exit",
    "left_has_grasped",
    "needle_was_grasped",
    "right_released_after_handover",
    "needle_completely_through",
    "success_stable_count",
    "_prev_dists",
    # HookPackage / InsertPeg稳定计数。
    "_success_stable_count",
    "_release_stable_count",
)


class SourceReplayError(RuntimeError):
    """源专家轨迹不能被当前环境精确且成功地重放。"""


class RecoveryBranchError(RuntimeError):
    """单个恢复分支未满足恢复或最终任务成功条件。"""


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """恢复分支起点的完整MuJoCo积分状态和任务Python状态。"""

    integration_state: np.ndarray
    ctrl: np.ndarray
    model_body_pos: np.ndarray
    model_body_quat: np.ndarray
    task_state: dict[str, Any]
    rng_state: dict[str, Any] | None


@dataclass(frozen=True)
class RecoveryResult:
    """一次成功恢复分支生成后用于写info.json的统计。"""

    info: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def _strict_true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _source_marked_success(info: dict[str, Any]) -> bool:
    """只接受显式布尔成功标记，避免字符串``"false"``被误判为真。"""

    if "success" in info:
        return _strict_true(info["success"])
    final_info = info.get("final_info")
    return (
        isinstance(final_info, dict)
        and "is_success" in final_info
        and _strict_true(final_info["is_success"])
    )


def _source_skip_reason(info: dict[str, Any]) -> str:
    if "success" in info:
        return "source_success_is_not_true"
    final_info = info.get("final_info")
    if isinstance(final_info, dict) and "is_success" in final_info:
        return "source_final_info_is_success_is_not_true"
    return "source_has_no_explicit_success_marker"


def _quintic_smoothstep(progress: float | np.ndarray) -> float | np.ndarray:
    progress_array = np.asarray(progress, dtype=np.float64)
    if np.any(~np.isfinite(progress_array)) or np.any(
        (progress_array < 0.0) | (progress_array > 1.0)
    ):
        raise ValueError("progress必须是[0,1]内的有限数。")
    value = (
        10.0 * progress_array**3
        - 15.0 * progress_array**4
        + 6.0 * progress_array**5
    )
    return float(value) if progress_array.ndim == 0 else value


def _quintic_remaining_fraction(
    progress: float | np.ndarray,
) -> float | np.ndarray:
    """五次最小加加速度曲线的剩余偏移比例，严格满足r(1)=0。"""

    value = 1.0 - np.asarray(_quintic_smoothstep(progress), dtype=np.float64)
    value = np.where(np.asarray(progress) == 1.0, 0.0, value)
    return float(value) if value.ndim == 0 else value


def _adaptive_recovery_steps(
    offset: np.ndarray,
    fps: int,
    max_extra_velocity_rad_s: np.ndarray,
    min_steps: int,
    max_steps: int,
) -> int:
    """按五次曲线峰值速度1.875*|offset|/T确定恢复时长。"""

    offset = np.asarray(offset, dtype=np.float64)
    velocity = np.asarray(max_extra_velocity_rad_s, dtype=np.float64)
    if offset.shape != (VIEW_DIM,) or velocity.shape != (VIEW_DIM,):
        raise ValueError("offset和max_extra_velocity_rad_s必须都是6维。")
    if not np.isfinite(offset).all() or not np.isfinite(velocity).all():
        raise ValueError("恢复偏移和速度上限必须为有限数。")
    if np.any(velocity <= 0.0):
        raise ValueError("max_extra_velocity_rad_s必须全部大于0。")
    if fps <= 0 or min_steps <= 0 or max_steps < min_steps:
        raise ValueError("fps/min_steps/max_steps配置非法。")
    required_seconds = float(np.max(1.875 * np.abs(offset) / velocity))
    required_steps = int(np.ceil(required_seconds * int(fps)))
    return int(np.clip(required_steps, int(min_steps), int(max_steps)))


def _sample_injection_frames(
    *,
    num_frames: int,
    seed: int,
    source_episode: int,
    probability: float,
    min_interval_steps: int,
    min_events: int,
    max_events: int,
    exclude_initial_steps: int,
    required_tail_steps: int,
) -> list[int]:
    """以“逐帧概率+最小间隔”确定一个源episode的恢复起点。"""

    if num_frames <= 0:
        raise ValueError("num_frames必须为正整数。")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability必须位于[0,1]。")
    if min_interval_steps < 0 or exclude_initial_steps < 0:
        raise ValueError("帧间隔和排除帧数不能为负数。")
    if required_tail_steps <= 0:
        raise ValueError("required_tail_steps必须为正整数。")
    if min_events < 0 or max_events < min_events:
        raise ValueError("min_events/max_events配置非法。")
    latest_frame = int(num_frames) - int(required_tail_steps)
    first_frame = int(exclude_initial_steps)
    if latest_frame < first_frame:
        raise ValueError(
            "源episode过短，无法同时容纳恢复、稳定确认和恢复后帧: "
            f"frames={num_frames}, first={first_frame}, latest={latest_frame}"
        )
    candidates = np.arange(first_frame, latest_frame + 1, dtype=np.int64)
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(source_episode), 0x45564E54]
    )
    rng = np.random.default_rng(seed_sequence)
    probabilistic = candidates[rng.random(len(candidates)) < float(probability)]

    def select_with_spacing(pool: np.ndarray, selected: list[int]) -> None:
        for frame in rng.permutation(pool):
            frame_value = int(frame)
            if frame_value not in selected and all(
                abs(frame_value - existing) >= int(min_interval_steps)
                for existing in selected
            ):
                selected.append(frame_value)
                if len(selected) >= int(max_events):
                    return

    selected: list[int] = []
    select_with_spacing(probabilistic, selected)
    if len(selected) < int(min_events):
        # 随机贪心可能先选到区间中点，导致明明存在可行组合却无法补足。
        # 这里在压缩坐标中直接采样一个必然满足间隔的min_events子集：
        # x_i = y_i + i * (interval - 1)，其中y_i严格递增。
        spacing = max(1, int(min_interval_steps))
        compressed_count = len(candidates) - (spacing - 1) * (
            int(min_events) - 1
        )
        if compressed_count < int(min_events):
            raise ValueError(
                "可用时间范围无法满足最少恢复事件数和最小间隔: "
                f"available=[{first_frame},{latest_frame}], "
                f"min_events={min_events}, interval={min_interval_steps}"
            )
        compressed = np.sort(
            rng.choice(
                compressed_count,
                size=int(min_events),
                replace=False,
            )
        )
        selected = (
            first_frame
            + compressed
            + np.arange(int(min_events), dtype=np.int64) * (spacing - 1)
        ).astype(int).tolist()
        select_with_spacing(probabilistic, selected)
    if len(selected) > int(max_events):
        chosen = rng.choice(selected, size=int(max_events), replace=False)
        selected = [int(value) for value in chosen]
    return sorted(selected)


def _local_feasible_offset_bounds(
    source_arrays: dict[str, np.ndarray],
    event_frame: int,
    horizon_steps: int,
    control_ranges: np.ndarray,
    max_abs: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """计算恢复窗口内state/action均不越过关节和执行器限位的偏移域。"""

    states = np.asarray(source_arrays["observation_state"], dtype=np.float64)
    actions = np.asarray(source_arrays["joint_action"], dtype=np.float64)
    stop = min(len(actions), int(event_frame) + int(horizon_steps) + 1)
    values = np.concatenate(
        (
            states[int(event_frame) : stop, VIEW_SLICE],
            actions[int(event_frame) : stop, VIEW_SLICE],
        ),
        axis=0,
    )
    ranges = np.asarray(control_ranges, dtype=np.float64)
    max_abs = np.asarray(max_abs, dtype=np.float64)
    if ranges.shape != (VIEW_DIM, 2) or max_abs.shape != (VIEW_DIM,):
        raise ValueError("control_ranges必须为[6,2]，max_abs必须为6维。")
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
            "该恢复事件在关节限位安全余量内没有可行偏移: "
            f"lower={lower.tolist()}, upper={upper.tolist()}"
        )
    return lower, upper


def _sample_recovery_offset(
    *,
    seed: int,
    source_episode: int,
    variant_index: int,
    attempt: int,
    std: np.ndarray,
    max_abs: np.ndarray,
    feasible_lower: np.ndarray,
    feasible_upper: np.ndarray,
    min_normalized_l2: float,
    max_sampling_attempts: int,
) -> np.ndarray:
    """为分支重试派生独立、可复现的截断高斯View偏移。"""

    std = np.asarray(std, dtype=np.float64)
    max_abs = np.asarray(max_abs, dtype=np.float64)
    lower = np.asarray(feasible_lower, dtype=np.float64)
    upper = np.asarray(feasible_upper, dtype=np.float64)
    if any(value.shape != (VIEW_DIM,) for value in (std, max_abs, lower, upper)):
        raise ValueError("偏移采样的所有向量都必须为6维。")
    if np.any(std <= 0.0) or np.any(max_abs <= 0.0):
        raise ValueError("std和max_abs必须全部大于0。")
    if np.any(lower > upper):
        raise ValueError("feasible_lower不能大于feasible_upper。")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(source_episode),
                int(variant_index),
                int(attempt),
                0x4F464653,
            ]
        )
    )
    for _ in range(int(max_sampling_attempts)):
        offset = rng.normal(loc=0.0, scale=std, size=VIEW_DIM)
        if np.any(np.abs(offset) > max_abs):
            continue
        if np.any(offset < lower) or np.any(offset > upper):
            continue
        normalized_l2 = float(np.linalg.norm(offset / max_abs))
        if normalized_l2 < float(min_normalized_l2):
            continue
        return offset.astype(np.float64)
    raise RuntimeError(
        "截断高斯在最大采样次数内没有得到可行恢复偏移；"
        "请减小min_normalized_l2/安全余量，或增大max_sampling_attempts。"
    )


def _recovery_action(
    expert_action: np.ndarray,
    view_offset: np.ndarray,
) -> np.ndarray:
    """保持Arm和夹爪原值，只给View六维专家目标叠加恢复偏移。"""

    expert_action = np.asarray(expert_action, dtype=np.float64)
    view_offset = np.asarray(view_offset, dtype=np.float64)
    if expert_action.shape != (ACTION_DIM,) or view_offset.shape != (VIEW_DIM,):
        raise ValueError("expert_action必须为20维，view_offset必须为6维。")
    if not np.isfinite(expert_action).all() or not np.isfinite(view_offset).all():
        raise ValueError("恢复动作不能包含NaN或Inf。")
    action = expert_action.copy()
    action[VIEW_SLICE] += view_offset
    return action


def _parse_source_episode_number(path: Path) -> int:
    match = SOURCE_EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"源目录名不是纯原始episode格式: {path.name}")
    return int(match.group("episode"))


def _load_successful_sources(
    input_run_dir: Path,
    source_episode_indices: list[int] | None,
    max_source_episodes: int | None,
) -> tuple[list[SourceEpisode], list[dict[str, Any]]]:
    episodes_dir = input_run_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"输入run缺少episodes目录: {episodes_dir}")
    requested = (
        None
        if source_episode_indices is None
        else {int(value) for value in source_episode_indices}
    )
    discovered: dict[int, Path] = {}
    skipped: list[dict[str, Any]] = []
    for path in sorted(episodes_dir.glob("episode_*")):
        if not path.is_dir() or path.name.endswith(".tmp"):
            continue
        match = SOURCE_EPISODE_PATTERN.fullmatch(path.name)
        if match is None:
            skipped.append(
                {
                    "directory": str(path),
                    "reason": "augmented_or_unrecognized_source_directory",
                }
            )
            continue
        episode_number = int(match.group("episode"))
        if requested is not None and episode_number not in requested:
            continue
        discovered[episode_number] = path
    if requested is not None:
        missing = sorted(requested - set(discovered))
        if missing:
            raise FileNotFoundError(f"找不到指定的源episode: {missing}")

    successful: list[SourceEpisode] = []
    for episode_number, path in sorted(discovered.items()):
        arrays_path = path / "arrays.npz"
        info_path = path / "info.json"
        if not arrays_path.is_file() or not info_path.is_file():
            skipped.append(
                {
                    "source_episode": episode_number,
                    "directory": str(path),
                    "reason": "missing_arrays_or_info",
                }
            )
            continue
        try:
            info = _load_json(info_path)
        except Exception as exc:
            skipped.append(
                {
                    "source_episode": episode_number,
                    "directory": str(path),
                    "reason": f"invalid_info: {type(exc).__name__}: {exc}",
                }
            )
            continue
        if not _source_marked_success(info):
            skipped.append(
                {
                    "source_episode": episode_number,
                    "directory": str(path),
                    "reason": _source_skip_reason(info),
                }
            )
            continue
        successful.append(
            SourceEpisode(
                source_index=len(successful),
                episode_number=episode_number,
                directory=path,
                info=info,
            )
        )
        if (
            max_source_episodes is not None
            and len(successful) >= int(max_source_episodes)
        ):
            break
    if not successful:
        raise RuntimeError("没有找到显式标记为成功且格式完整的源episode。")
    return successful, skipped


def _integration_state_signature():
    import mujoco

    return mujoco.mjtState.mjSTATE_INTEGRATION


def _capture_environment_snapshot(env_obj) -> EnvironmentSnapshot:
    physics = env_obj._physics
    task_state = {
        name: copy.deepcopy(getattr(env_obj, name))
        for name in TASK_STATE_FIELDS
        if hasattr(env_obj, name)
    }
    rng_state = None
    if hasattr(env_obj, "np_random") and hasattr(env_obj.np_random, "bit_generator"):
        rng_state = copy.deepcopy(env_obj.np_random.bit_generator.state)
    return EnvironmentSnapshot(
        integration_state=np.asarray(
            physics.get_state(sig=_integration_state_signature())
        ).copy(),
        ctrl=physics.data.ctrl.copy(),
        model_body_pos=physics.model.body_pos.copy(),
        model_body_quat=physics.model.body_quat.copy(),
        task_state=task_state,
        rng_state=rng_state,
    )


def _restore_environment_snapshot(env_obj, snapshot: EnvironmentSnapshot) -> None:
    physics = env_obj._physics
    physics.model.body_pos[:] = snapshot.model_body_pos
    physics.model.body_quat[:] = snapshot.model_body_quat
    physics.set_state(
        snapshot.integration_state,
        sig=_integration_state_signature(),
    )
    physics.data.ctrl[:] = snapshot.ctrl
    for name, value in snapshot.task_state.items():
        setattr(env_obj, name, copy.deepcopy(value))
    if snapshot.rng_state is not None:
        env_obj.np_random.bit_generator.state = copy.deepcopy(snapshot.rng_state)
    physics.forward()


def _sync_reward_reference_after_initial_restore(env_obj) -> None:
    """让任务进度差分参考与已覆盖的源MuJoCo初态保持一致。"""

    if hasattr(env_obj, "_prev_metrics") and hasattr(env_obj, "_calculate_metrics"):
        env_obj._prev_metrics = env_obj._calculate_metrics()
    if hasattr(env_obj, "_prev_dists") and hasattr(env_obj, "_calculate_distances"):
        env_obj._prev_dists = env_obj._calculate_distances()


def _step_without_render(env_obj, action: np.ndarray) -> tuple[float, bool, bool]:
    """执行与env.step相同的控制和奖励更新，但跳过昂贵的get_obs渲染。"""

    action = np.asarray(action, dtype=np.float64)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"动作必须为{ACTION_DIM}维，当前为{action.shape}。")
    if not np.isfinite(action).all():
        raise ValueError("动作包含NaN或Inf，拒绝env.step中的静默全零降级。")
    physics = env_obj._physics
    physics.bind(env_obj._left_actuators[:6]).ctrl = action[:6]
    physics.bind(env_obj._left_actuators[6]).ctrl = env_obj.left_gripper_unnorm_fn(
        np.clip(action[6], 0.0, 1.0)
    )
    physics.bind(env_obj._right_actuators[:6]).ctrl = action[7:13]
    physics.bind(env_obj._right_actuators[6]).ctrl = env_obj.right_gripper_unnorm_fn(
        np.clip(action[13], 0.0, 1.0)
    )
    physics.bind(env_obj._middle_actuators).ctrl = action[VIEW_SLICE]
    for _ in range(SIM_PHYSICS_ENV_STEP_RATIO):
        physics.step()
    env_obj._current_step = int(getattr(env_obj, "_current_step", 0)) + 1
    reward = float(env_obj.get_reward()) if hasattr(env_obj, "get_reward") else 0.0
    terminated = bool(getattr(env_obj, "terminated", False))
    truncated = bool(env_obj._current_step >= env_obj.episode_length)
    return reward, terminated, truncated


def _group_errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    error = np.abs(
        np.asarray(actual, dtype=np.float64)
        - np.asarray(expected, dtype=np.float64)
    )
    return {
        "state": float(np.max(error)),
        "arm_joint": float(np.max(error[ARM_JOINT_INDICES])),
        "gripper": float(np.max(error[GRIPPER_INDICES])),
        "view_joint": float(np.max(error[VIEW_JOINT_INDICES])),
    }


def _validate_source_replay_state(
    errors: dict[str, float],
    cfg: DictConfig,
    source_episode: int,
    frame_index: int,
) -> None:
    limits = {
        "arm_joint": float(cfg.validation.max_arm_joint_abs_error),
        "gripper": float(cfg.validation.max_gripper_abs_error),
        "view_joint": float(cfg.validation.max_view_joint_abs_error),
    }
    for name, limit in limits.items():
        if errors[name] > limit:
            raise SourceReplayError(
                f"source_episode={source_episode}, frame={frame_index}, "
                f"{name}_error={errors[name]:.6g} > {limit:.6g}"
            )


def _apply_unrecorded_view_disturbance(
    env_obj,
    expert_view_state: np.ndarray,
    offset: np.ndarray,
    setup_steps: int,
) -> None:
    """在不调用physics.step的前提下平滑建立分支初始OOD状态。"""

    physics = env_obj._physics
    binding = physics.bind(env_obj._middle_joints)
    start = np.asarray(binding.qpos, dtype=np.float64).copy()
    target = np.asarray(expert_view_state, dtype=np.float64) + np.asarray(
        offset, dtype=np.float64
    )
    for setup_index in range(1, int(setup_steps) + 1):
        fraction = _quintic_smoothstep(setup_index / float(setup_steps))
        binding.qpos = start + float(fraction) * (target - start)
        binding.qvel = np.zeros(VIEW_DIM, dtype=np.float64)
        physics.bind(env_obj._middle_actuators).ctrl = binding.qpos
        physics.forward()
    binding.qpos = target
    binding.qvel = np.zeros(VIEW_DIM, dtype=np.float64)
    physics.bind(env_obj._middle_actuators).ctrl = target
    physics.forward()


def _capture_episode_initial_arrays(env_obj) -> dict[str, np.ndarray]:
    physics = env_obj._physics
    return {
        "initial_time": np.asarray(physics.data.time, dtype=np.float64),
        "initial_qpos": physics.data.qpos.copy().astype(np.float64),
        "initial_qvel": physics.data.qvel.copy().astype(np.float64),
        "initial_ctrl": physics.data.ctrl.copy().astype(np.float64),
        "initial_act": physics.data.act.copy().astype(np.float64),
        "initial_mocap_pos": physics.data.mocap_pos.copy().astype(np.float64),
        "initial_mocap_quat": physics.data.mocap_quat.copy().astype(np.float64),
        "initial_model_body_pos": physics.model.body_pos.copy().astype(np.float64),
        "initial_model_body_quat": physics.model.body_quat.copy().astype(np.float64),
    }


def _render_stereo(env_obj, writer: StereoVideoWriter, cameras: tuple[str, ...], cfg: DictConfig) -> None:
    physics = env_obj._physics
    for camera in cameras:
        frame = physics.render(
            height=int(cfg.render_height),
            width=int(cfg.render_width),
            camera_id=camera,
        )
        writer.append(camera, frame)


def _filtered_original_arrays(source_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = {
        "joint_action",
        "observation_state",
        "initial_time",
        "initial_qpos",
        "initial_qvel",
        "initial_ctrl",
        "initial_act",
        "initial_mocap_pos",
        "initial_mocap_quat",
        *MODEL_BODY_INITIAL_KEYS,
        *OPTIONAL_FRAME_KEYS,
    }
    return {
        key: np.asarray(value).copy()
        for key, value in source_arrays.items()
        if key in keys
    }


def _episode_dir(output_run_dir: Path, source_episode: int, variant_index: int) -> Path:
    return (
        output_run_dir
        / "episodes"
        / _output_episode_name(source_episode, variant_index)
    )


def _parse_output_identity(name: str) -> tuple[int, int]:
    match = OUTPUT_EPISODE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"无法识别输出episode目录名: {name}")
    variant = match.group("variant")
    return int(match.group("source")), -1 if variant is None else int(variant)


def _prepare_tmp_directory(final_dir: Path) -> Path:
    tmp_dir = final_dir.with_name(final_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if final_dir.exists():
        raise FileExistsError(f"目标episode存在但不完整，拒绝覆盖: {final_dir}")
    tmp_dir.mkdir(parents=True)
    return tmp_dir


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
    return {
        "episode": int(source.episode_number),
        "episode_name": name,
        "episode_naming": EPISODE_NAMING,
        "success": True,
        "steps": int(steps),
        "fps": int(cfg.fps),
        "path": str((_episode_dir(output_run_dir, source.episode_number, variant_index)).relative_to(output_run_dir)),
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
    source_episode: int,
    variant_index: int,
    fingerprint: str,
) -> dict[str, Any]:
    info = _load_json(directory / "info.json")
    expected = {
        "source_episode": int(source_episode),
        "variant_index": int(variant_index),
        "episode_name": _output_episode_name(source_episode, variant_index),
        "generation_fingerprint": fingerprint,
    }
    mismatches = {
        key: {"expected": value, "actual": info.get(key)}
        for key, value in expected.items()
        if info.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"已完成episode身份与本次配置不一致: {directory}, {mismatches}"
        )
    return info


def _replay_source_and_capture(
    *,
    env_obj,
    source: SourceEpisode,
    source_arrays: dict[str, np.ndarray],
    event_frames: list[int],
    original_tmp_dir: Path | None,
    cameras: tuple[str, ...],
    cfg: DictConfig,
) -> tuple[dict[int, EnvironmentSnapshot], dict[str, float], float]:
    """名义重放一次，同时捕获全部事件快照并可选重渲染原始双目视频。"""

    _restore_initial_state(env_obj, source_arrays)
    _sync_reward_reference_after_initial_restore(env_obj)
    actions = np.asarray(source_arrays["joint_action"], dtype=np.float64)
    states = np.asarray(source_arrays["observation_state"], dtype=np.float64)
    event_set = set(int(value) for value in event_frames)
    snapshots: dict[int, EnvironmentSnapshot] = {}
    maxima = {"state": 0.0, "arm_joint": 0.0, "gripper": 0.0, "view_joint": 0.0}
    final_reward = 0.0
    writer_context = (
        StereoVideoWriter(original_tmp_dir / "videos", cameras, cfg)
        if original_tmp_dir is not None
        else nullcontext(None)
    )
    with writer_context as writer:
        for frame_index, expert_action in enumerate(actions):
            actual = _read_agent_state(env_obj)
            errors = _group_errors(actual, states[frame_index])
            _validate_source_replay_state(
                errors, cfg, source.episode_number, frame_index
            )
            for name, value in errors.items():
                maxima[name] = max(maxima[name], value)
            if frame_index in event_set:
                snapshots[frame_index] = _capture_environment_snapshot(env_obj)
            if writer is not None:
                _render_stereo(env_obj, writer, cameras, cfg)
            final_reward, terminated, truncated = _step_without_render(
                env_obj, expert_action
            )
            if truncated:
                raise SourceReplayError(
                    f"source_episode={source.episode_number}名义重放被截断。"
                )
            if terminated and not bool(getattr(env_obj, "is_success", False)):
                raise SourceReplayError(
                    f"source_episode={source.episode_number}在frame={frame_index}"
                    "重放时提前失败终止。"
                )
            if terminated and frame_index != len(actions) - 1:
                raise SourceReplayError(
                    f"source_episode={source.episode_number}在frame={frame_index}"
                    "提前成功终止；拒绝在terminal状态后继续重放。"
                )
    missing = sorted(event_set - set(snapshots))
    if missing:
        raise SourceReplayError(f"没有捕获到恢复事件快照: {missing}")
    if not bool(getattr(env_obj, "is_success", False)):
        raise SourceReplayError(
            f"source_episode={source.episode_number}完整名义重放后任务未成功。"
        )
    return snapshots, maxima, float(final_reward)


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
        tmp_dir / "arrays.npz", **_filtered_original_arrays(source_arrays)
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
            "success_inherited_from_source": False,
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


def _generate_recovery_branch(
    *,
    env_obj,
    source: SourceEpisode,
    source_arrays: dict[str, np.ndarray],
    snapshot: EnvironmentSnapshot,
    event_frame: int,
    variant_index: int,
    attempt: int,
    offset: np.ndarray,
    feasible_lower: np.ndarray,
    feasible_upper: np.ndarray,
    output_run_dir: Path,
    cameras: tuple[str, ...],
    cfg: DictConfig,
    fingerprint: str,
) -> RecoveryResult:
    final_dir = _episode_dir(output_run_dir, source.episode_number, variant_index)
    tmp_dir = _prepare_tmp_directory(final_dir)
    try:
        _restore_environment_snapshot(env_obj, snapshot)
        source_states = np.asarray(
            source_arrays["observation_state"], dtype=np.float64
        )
        source_actions = np.asarray(source_arrays["joint_action"], dtype=np.float64)
        _apply_unrecorded_view_disturbance(
            env_obj,
            source_states[event_frame, VIEW_SLICE],
            offset,
            int(cfg.recovery.unrecorded_setup_steps),
        )
        initial_arrays = _capture_episode_initial_arrays(env_obj)
        initial_actual = _read_agent_state(env_obj)
        initial_error = (
            initial_actual[VIEW_SLICE]
            - source_states[event_frame, VIEW_SLICE]
        )
        if not np.allclose(initial_error, offset, rtol=0.0, atol=1e-10):
            raise RecoveryBranchError(
                "外部扰动后的实际View偏移与采样值不一致: "
                f"actual={initial_error.tolist()}, expected={offset.tolist()}"
            )

        planned_steps = _adaptive_recovery_steps(
            offset,
            int(cfg.fps),
            np.asarray(cfg.recovery.max_extra_velocity_rad_s, dtype=np.float64),
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
        reference_views: list[np.ndarray] = []
        command_offsets: list[np.ndarray] = []
        state_errors: list[np.ndarray] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []
        max_arm_error = 0.0
        max_gripper_error = 0.0
        max_view_error = 0.0
        stable_count = 0
        achieved_local_step: int | None = None
        achieved_source_frame: int | None = None
        post_frames = 0
        local_step = 0
        source_index = int(event_frame)
        last_reward = 0.0

        with StereoVideoWriter(tmp_dir / "videos", cameras, cfg) as writer:
            while source_index < len(source_actions):
                actual_state = _read_agent_state(env_obj)
                expert_state = source_states[source_index]
                errors = _group_errors(actual_state, expert_state)
                max_arm_error = max(max_arm_error, errors["arm_joint"])
                max_gripper_error = max(max_gripper_error, errors["gripper"])
                view_error_vector = (
                    actual_state[VIEW_SLICE] - expert_state[VIEW_SLICE]
                )
                view_error = float(np.max(np.abs(view_error_vector)))
                max_view_error = max(max_view_error, view_error)
                if max_arm_error > float(cfg.validation.branch_max_arm_joint_abs_error):
                    raise RecoveryBranchError(
                        f"恢复分支Arm误差{max_arm_error:.6g}超过阈值"
                        f"{float(cfg.validation.branch_max_arm_joint_abs_error):.6g}。"
                    )
                if max_gripper_error > float(cfg.validation.branch_max_gripper_abs_error):
                    raise RecoveryBranchError(
                        f"恢复分支夹爪误差{max_gripper_error:.6g}超过阈值"
                        f"{float(cfg.validation.branch_max_gripper_abs_error):.6g}。"
                    )

                if local_step >= planned_steps:
                    stable_count = stable_count + 1 if view_error <= recovery_threshold else 0
                    if achieved_local_step is None and stable_count >= stable_required:
                        achieved_local_step = int(local_step)
                        achieved_source_frame = int(source_index)
                if achieved_local_step is not None and local_step > achieved_local_step:
                    if view_error > recovery_threshold:
                        raise RecoveryBranchError(
                            "恢复达标后的专家跟随阶段再次离开误差管道: "
                            f"source_frame={source_index}, error={view_error:.6g} > "
                            f"{recovery_threshold:.6g}"
                        )
                    post_frames += 1
                if (
                    achieved_local_step is None
                    and local_step >= planned_steps + max_extra_steps
                ):
                    raise RecoveryBranchError(
                        "计划恢复归零后仍未连续满足恢复阈值: "
                        f"planned_steps={planned_steps}, extra={max_extra_steps}, "
                        f"current_error={view_error:.6g}, stable={stable_count}/"
                        f"{stable_required}"
                    )

                if local_step < planned_steps:
                    remaining = float(
                        _quintic_remaining_fraction(
                            (local_step + 1) / float(planned_steps)
                        )
                    )
                else:
                    remaining = 0.0
                command_offset = remaining * offset
                action = _recovery_action(
                    source_actions[source_index], command_offset
                )

                _render_stereo(env_obj, writer, cameras, cfg)
                recorded_states.append(actual_state.astype(np.float32))
                recorded_actions.append(action.astype(np.float32))
                source_indices.append(source_index)
                reference_views.append(
                    expert_state[VIEW_SLICE].astype(np.float32)
                )
                command_offsets.append(command_offset.astype(np.float32))
                state_errors.append(view_error_vector.astype(np.float32))

                last_reward, terminated, truncated = _step_without_render(
                    env_obj, action
                )
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                if truncated:
                    raise RecoveryBranchError("恢复分支被环境步数上限截断。")
                if terminated and not bool(getattr(env_obj, "is_success", False)):
                    raise RecoveryBranchError("恢复分支在任务完成前失败终止。")
                if terminated and source_index != len(source_actions) - 1:
                    raise RecoveryBranchError(
                        "恢复分支在专家后缀结束前提前成功终止；"
                        "拒绝在terminal状态后继续推进。"
                    )

                source_index += 1
                local_step += 1
                if achieved_local_step is not None and post_frames >= post_required:
                    break

        if achieved_local_step is None:
            raise RecoveryBranchError("源轨迹剩余长度不足以达到恢复成功条件。")
        if post_frames < post_required:
            raise RecoveryBranchError(
                f"恢复后仅有{post_frames}帧，少于要求的{post_required}帧。"
            )

        # 不再写视频，但继续用原专家后缀推进并更新任务阶段，验证最终任务成功。
        while source_index < len(source_actions):
            last_reward, terminated, truncated = _step_without_render(
                env_obj, source_actions[source_index]
            )
            if truncated:
                raise RecoveryBranchError("后台专家后缀验证被截断。")
            if terminated and not bool(getattr(env_obj, "is_success", False)):
                raise RecoveryBranchError(
                    f"后台专家后缀在source_frame={source_index}失败终止。"
                )
            if terminated and source_index != len(source_actions) - 1:
                raise RecoveryBranchError(
                    f"后台专家后缀在source_frame={source_index}提前成功终止；"
                    "拒绝在terminal状态后继续推进。"
                )
            source_index += 1
        final_success = bool(getattr(env_obj, "is_success", False))
        if not final_success:
            raise RecoveryBranchError("执行完整专家后缀后任务未成功。")

        arrays = {
            "joint_action": np.asarray(recorded_actions, dtype=np.float32),
            "observation_state": np.asarray(recorded_states, dtype=np.float32),
            "timestamp": np.arange(len(recorded_actions), dtype=np.float32)
            / float(cfg.fps),
            "terminated": np.asarray(terminated_flags, dtype=np.bool_),
            "truncated": np.asarray(truncated_flags, dtype=np.bool_),
            "source_frame_index": np.asarray(source_indices, dtype=np.int64),
            "recovery_reference_view_state": np.asarray(
                reference_views, dtype=np.float32
            ),
            "recovery_command_offset": np.asarray(
                command_offsets, dtype=np.float32
            ),
            "recovery_view_state_error": np.asarray(
                state_errors, dtype=np.float32
            ),
            **initial_arrays,
        }
        np.savez_compressed(tmp_dir / "arrays.npz", **arrays)

        final_view_error = float(
            np.max(np.abs(np.asarray(state_errors[-1], dtype=np.float64)))
        )
        planned_duration_s = planned_steps / float(cfg.fps)
        peak_extra_velocity = (
            1.875 * np.abs(offset) / planned_duration_s
        )
        configured_offset_max = np.asarray(
            cfg.view_joint_noise.max_abs_rad, dtype=np.float64
        )
        velocity_limited_offset_max = np.minimum(
            configured_offset_max,
            np.asarray(
                cfg.recovery.max_extra_velocity_rad_s,
                dtype=np.float64,
            )
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
                "success_inherited_from_source": False,
                "source_replay_success": True,
                "final_task_success": True,
                "source_injection_frame": int(event_frame),
                "source_recorded_last_frame": int(source_indices[-1]),
                "view_joint_offset_rad": offset.tolist(),
                "configured_view_offset_max_abs_rad": (
                    configured_offset_max.tolist()
                ),
                "effective_view_offset_max_abs_rad": (
                    velocity_limited_offset_max.tolist()
                ),
                "view_offset_feasible_lower_rad": feasible_lower.tolist(),
                "view_offset_feasible_upper_rad": feasible_upper.tolist(),
                "planned_recovery_steps": int(planned_steps),
                "planned_recovery_duration_s": float(planned_duration_s),
                "recovery_command_peak_extra_velocity_rad_s": (
                    peak_extra_velocity.tolist()
                ),
                "actual_recovery_steps": int(achieved_local_step),
                "recovery_achieved_source_frame": int(achieved_source_frame),
                "recovery_success_max_abs_error_rad": recovery_threshold,
                "recovery_success_stable_steps": stable_required,
                "recovery_post_steps": int(post_frames),
                "final_recorded_view_max_abs_error_rad": final_view_error,
                "max_recorded_view_max_abs_error_rad": float(max_view_error),
                "max_recorded_arm_joint_abs_error": float(max_arm_error),
                "max_recorded_gripper_abs_error": float(max_gripper_error),
                "sampling_seed": int(cfg.seed),
                "branch_attempt": int(attempt),
                "final_reward": float(last_reward),
                "background_suffix_validated": True,
                "background_suffix_final_source_frame": int(source_index - 1),
                "unrecorded_disturbance_setup_steps": int(
                    cfg.recovery.unrecorded_setup_steps
                ),
                "recovery_curve": "quintic_minimum_jerk_to_moving_expert",
                "only_view_action_modified": True,
                "final_info": {
                    "is_success": True,
                    "source_episode": int(source.episode_number),
                    "variant_index": int(variant_index),
                    "source_injection_frame": int(event_frame),
                    "view_recovery_achieved": True,
                    "background_suffix_validated": True,
                },
            }
        )
        _write_json_atomic(tmp_dir / "info.json", info)
        tmp_dir.rename(final_dir)
        return RecoveryResult(info=info, arrays=arrays)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _semantic_config(cfg: DictConfig, input_run_dir: Path, env_id: str) -> dict[str, Any]:
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
        "view_joint_noise": OmegaConf.to_container(
            cfg.view_joint_noise, resolve=True
        ),
        "recovery": OmegaConf.to_container(cfg.recovery, resolve=True),
        "validation": OmegaConf.to_container(cfg.validation, resolve=True),
        "cameras": [str(value) for value in cfg.cameras],
        "render_height": int(cfg.render_height),
        "render_width": int(cfg.render_width),
        "fps": int(cfg.fps),
        "video": OmegaConf.to_container(cfg.video, resolve=True),
        # 防止中断续生成期间修改生成器、任务奖励或MuJoCo XML后静默混合数据。
        "generation_dependencies": _generation_dependency_manifest(env_id),
    }


def _generation_dependency_manifest(env_id: str) -> dict[str, str]:
    task_files = {
        "guided_vision/SewNeedle-3Arms-v0": "env/task/sew_needle_env.py",
        "guided_vision/InsertCylinder-3Arms-v0": "env/task/insert_cylinder_env.py",
        "guided_vision/InsertPeg-3Arms-v0": "env/task/insert_peg_env.py",
        "guided_vision/HookPackage-3Arms-v0": "env/task/hook_package_env.py",
    }
    dependencies = [
        Path(__file__).resolve(),
        ROOT_DIR / "data_collect/augment_view_joint_trajectories.py",
        ROOT_DIR / "env/__init__.py",
        ROOT_DIR / "env/constants.py",
        ROOT_DIR / "env/task/sim_envs.py",
        ROOT_DIR / task_files[env_id],
    ]
    dependencies.extend(sorted((ROOT_DIR / "env/assets").rglob("*.xml")))
    manifest = {}
    for path in dependencies:
        if not path.is_file():
            raise FileNotFoundError(f"生成依赖文件不存在: {path}")
        key = str(path.relative_to(ROOT_DIR))
        manifest[key] = _sha256_file(path)
    return manifest


def _scan_completed_infos(
    output_run_dir: Path,
    cameras: tuple[str, ...],
    fingerprint: str,
) -> list[dict[str, Any]]:
    episodes_dir = output_run_dir / "episodes"
    if not episodes_dir.exists():
        return []
    infos = []
    identities: dict[tuple[int, int], Path] = {}
    for directory in episodes_dir.glob("episode_*"):
        if not directory.is_dir() or directory.name.endswith(".tmp"):
            continue
        identity = _parse_output_identity(directory.name)
        if identity in identities:
            raise RuntimeError(
                f"发现重复输出身份: {identities[identity]}, {directory}"
            )
        identities[identity] = directory
    for (source_episode, variant_index), directory in sorted(
        identities.items(), key=lambda item: (item[0][0], item[0][1] + 1)
    ):
        if not _episode_is_complete(directory, cameras):
            raise RuntimeError(f"发现不完整的最终episode目录: {directory}")
        infos.append(
            _validate_completed_identity(
                directory,
                source_episode,
                variant_index,
                fingerprint,
            )
        )
    return infos


def _refresh_metadata(
    metadata: dict[str, Any],
    output_run_dir: Path,
    cameras: tuple[str, ...],
    fingerprint: str,
) -> None:
    infos = _scan_completed_infos(output_run_dir, cameras, fingerprint)
    metadata["episodes"] = infos
    metadata["saved_episodes"] = len(infos)
    metadata["successful_episodes"] = sum(
        _strict_true(info.get("success")) for info in infos
    )
    metadata["original_episodes"] = sum(
        not bool(info.get("is_augmented", False)) for info in infos
    )
    metadata["recovery_episodes"] = sum(
        bool(info.get("is_augmented", False)) for info in infos
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
                "续生成配置与已有输出不一致，请更换output_run_dir: "
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
                f"输出目录非空但缺少metadata.json: {output_run_dir}"
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
            "video_save_mode": "pre_action_physical_recovery_rollout",
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
    metadata["skipped_sources"] = skipped_sources
    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
    return metadata


def _update_source_manifest(
    metadata: dict[str, Any], sources: list[SourceEpisode]
) -> None:
    manifest = metadata.setdefault("source_manifest", {})
    for source in sources:
        entry = {
            "source_episode": int(source.episode_number),
            "arrays_sha256": _sha256_file(source.directory / "arrays.npz"),
            "info_sha256": _sha256_file(source.directory / "info.json"),
        }
        key = str(source.episode_number)
        if key in manifest and manifest[key] != entry:
            raise RuntimeError(
                "源episode自上次生成后发生变化，拒绝混合结果: "
                f"source_episode={source.episode_number}"
            )
        manifest[key] = entry


def _record_failure(
    metadata: dict[str, Any],
    source_episode: int,
    variant_index: int | None,
    event_frame: int | None,
    attempts: list[dict[str, Any]],
) -> None:
    key = (int(source_episode), variant_index)
    failures = [
        value
        for value in metadata.setdefault("failures", [])
        if (value.get("source_episode"), value.get("variant_index")) != key
    ]
    failures.append(
        {
            "source_episode": int(source_episode),
            "variant_index": variant_index,
            "source_injection_frame": event_frame,
            "attempts": attempts,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    metadata["failures"] = failures


def _clear_failure(
    metadata: dict[str, Any], source_episode: int, variant_index: int | None
) -> None:
    key = (int(source_episode), variant_index)
    metadata["failures"] = [
        value
        for value in metadata.setdefault("failures", [])
        if (value.get("source_episode"), value.get("variant_index")) != key
    ]


def _validate_config(cfg: DictConfig, source_metadata: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    metadata_env_id = source_metadata.get("env_id")
    env_id = str(metadata_env_id) if cfg.env_id is None else str(cfg.env_id)
    if not env_id:
        raise ValueError("无法从配置或源metadata解析env_id。")
    if cfg.env_id is not None and metadata_env_id is not None and str(metadata_env_id) != env_id:
        raise ValueError(
            f"配置env_id={env_id!r}与源metadata={metadata_env_id!r}不一致。"
        )
    supported = {
        "guided_vision/SewNeedle-3Arms-v0",
        "guided_vision/InsertCylinder-3Arms-v0",
        "guided_vision/InsertPeg-3Arms-v0",
        "guided_vision/HookPackage-3Arms-v0",
    }
    if env_id not in supported:
        raise ValueError(f"当前恢复生成器不支持env_id={env_id!r}。")
    cameras = tuple(str(value) for value in cfg.cameras)
    if cameras != ("zed_cam_left", "zed_cam_right"):
        raise ValueError(
            "恢复生成器固定只保存zed_cam_left和zed_cam_right，"
            f"当前为{cameras}。"
        )
    source_fps = source_metadata.get("fps")
    if source_fps is not None and int(source_fps) != int(cfg.fps):
        raise ValueError(
            f"配置fps={int(cfg.fps)}与源metadata fps={int(source_fps)}不一致。"
        )
    vector_fields = {
        "view_joint_noise.std_rad": cfg.view_joint_noise.std_rad,
        "view_joint_noise.max_abs_rad": cfg.view_joint_noise.max_abs_rad,
        "recovery.max_extra_velocity_rad_s": cfg.recovery.max_extra_velocity_rad_s,
    }
    for name, value in vector_fields.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (VIEW_DIM,) or not np.isfinite(array).all() or np.any(array <= 0):
            raise ValueError(f"{name}必须是6维有限正数。")
    if str(cfg.view_joint_noise.distribution) != "truncated_gaussian":
        raise ValueError("当前仅支持view_joint_noise.distribution=truncated_gaussian。")
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
        "view_joint_noise.max_sampling_attempts": cfg.view_joint_noise.max_sampling_attempts,
    }
    for name, raw in integer_positive.items():
        value = float(raw)
        if not np.isfinite(value) or not value.is_integer() or value <= 0:
            raise ValueError(f"{name}必须为正整数。")
    integer_nonnegative = {
        "event_sampling.min_events_per_source": cfg.event_sampling.min_events_per_source,
        "event_sampling.min_injection_interval_steps": cfg.event_sampling.min_injection_interval_steps,
        "event_sampling.exclude_initial_steps": cfg.event_sampling.exclude_initial_steps,
        "recovery.max_extra_zero_offset_steps": cfg.recovery.max_extra_zero_offset_steps,
    }
    for name, raw in integer_nonnegative.items():
        value = float(raw)
        if not np.isfinite(value) or not value.is_integer() or value < 0:
            raise ValueError(f"{name}必须为非负整数。")
    if int(cfg.recovery.max_steps) < int(cfg.recovery.min_steps):
        raise ValueError("recovery.max_steps不能小于min_steps。")
    if int(cfg.recovery.success_stable_steps) > (
        int(cfg.recovery.max_extra_zero_offset_steps) + 1
    ):
        raise ValueError(
            "recovery.success_stable_steps不能大于"
            "max_extra_zero_offset_steps+1，否则任何分支都无法达标。"
        )
    if int(cfg.event_sampling.max_events_per_source) < int(cfg.event_sampling.min_events_per_source):
        raise ValueError("max_events_per_source不能小于min_events_per_source。")
    probability = float(cfg.event_sampling.injection_probability_per_frame)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("injection_probability_per_frame必须位于[0,1]。")
    positive_float_fields = {
        "view_joint_noise.joint_limit_margin_rad": cfg.view_joint_noise.joint_limit_margin_rad,
        "recovery.success_max_abs_error_rad": cfg.recovery.success_max_abs_error_rad,
        "validation.max_arm_joint_abs_error": cfg.validation.max_arm_joint_abs_error,
        "validation.max_gripper_abs_error": cfg.validation.max_gripper_abs_error,
        "validation.max_view_joint_abs_error": cfg.validation.max_view_joint_abs_error,
        "validation.branch_max_arm_joint_abs_error": cfg.validation.branch_max_arm_joint_abs_error,
        "validation.branch_max_gripper_abs_error": cfg.validation.branch_max_gripper_abs_error,
    }
    for name, raw in positive_float_fields.items():
        value = float(raw)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name}必须为有限正数。")
    if cfg.max_source_episodes is not None and int(cfg.max_source_episodes) <= 0:
        raise ValueError("max_source_episodes必须为正整数或null。")
    if cfg.max_source_episodes is not None:
        max_sources = float(cfg.max_source_episodes)
        if not np.isfinite(max_sources) or not max_sources.is_integer():
            raise ValueError("max_source_episodes必须为正整数或null。")
    min_normalized_l2 = float(cfg.view_joint_noise.min_normalized_l2)
    if (
        not np.isfinite(min_normalized_l2)
        or min_normalized_l2 < 0.0
        or min_normalized_l2 > np.sqrt(VIEW_DIM)
    ):
        raise ValueError(
            "view_joint_noise.min_normalized_l2必须位于[0,sqrt(6)]。"
        )
    seed_value = float(cfg.seed)
    if (
        not np.isfinite(seed_value)
        or not seed_value.is_integer()
        or seed_value < 0
    ):
        raise ValueError("seed必须为非负整数。")
    if cfg.source_episode_indices is not None:
        for raw_index in cfg.source_episode_indices:
            index_value = float(raw_index)
            if (
                not np.isfinite(index_value)
                or not index_value.is_integer()
                or index_value < 0
            ):
                raise ValueError("source_episode_indices必须全部为非负整数。")
    return env_id, cameras


def generate_contractive_view_recovery_run(cfg: DictConfig) -> None:
    input_run_dir = _resolve_path(cfg.input_run_dir)
    output_run_dir = _resolve_path(cfg.output_run_dir)
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
    sources, skipped_sources = _load_successful_sources(
        input_run_dir,
        source_indices,
        None if cfg.max_source_episodes is None else int(cfg.max_source_episodes),
    )
    semantic_config = _semantic_config(cfg, input_run_dir, env_id)
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
    _update_source_manifest(metadata, sources)
    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)

    logging.info(
        "开始生成可收缩View恢复数据: env=%s source=%s output=%s "
        "successful_sources=%d skipped=%d fingerprint=%s",
        env_id,
        input_run_dir,
        output_run_dir,
        len(sources),
        len(skipped_sources),
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
        joint_ranges = np.asarray(
            env_obj._physics.bind(env_obj._middle_joints).range,
            dtype=np.float64,
        ).copy()
        actuator_ranges = np.asarray(
            env_obj._physics.bind(env_obj._middle_actuators).ctrlrange,
            dtype=np.float64,
        ).copy()
        control_ranges = np.stack(
            (
                np.maximum(joint_ranges[:, 0], actuator_ranges[:, 0]),
                np.minimum(joint_ranges[:, 1], actuator_ranges[:, 1]),
            ),
            axis=1,
        )
        if np.any(control_ranges[:, 0] >= control_ranges[:, 1]):
            raise RuntimeError("View关节限位和执行器限位没有有效交集。")

        required_tail_steps = (
            int(cfg.recovery.max_steps)
            + int(cfg.recovery.max_extra_zero_offset_steps)
            + int(cfg.recovery.post_recovery_steps)
            + 1
        )
        std = np.asarray(cfg.view_joint_noise.std_rad, dtype=np.float64)
        configured_max_abs = np.asarray(
            cfg.view_joint_noise.max_abs_rad, dtype=np.float64
        )
        max_extra_velocity = np.asarray(
            cfg.recovery.max_extra_velocity_rad_s, dtype=np.float64
        )
        # 五次曲线峰值速度为1.875*|offset|/T；在max_steps固定上限下
        # 反推允许的最大偏移，避免“时长被clip后实际速度突破上限”。
        velocity_limited_max_abs = (
            max_extra_velocity
            * int(cfg.recovery.max_steps)
            / (1.875 * int(cfg.fps))
        )
        max_abs = np.minimum(configured_max_abs, velocity_limited_max_abs)
        if np.any(max_abs < configured_max_abs - 1e-12):
            logging.info(
                "为满足五次恢复峰值速度上限，View偏移有效上限由%s收紧为%s",
                np.array2string(configured_max_abs, precision=4),
                np.array2string(max_abs, precision=4),
            )

        for source in sources:
            source_attempt_errors: list[dict[str, Any]] = []
            original_final_dir = _episode_dir(
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
                        f"source_episode={source.episode_number} fps={episode_fps}"
                        f"与配置fps={int(cfg.fps)}不一致。"
                    )
                event_frames = _sample_injection_frames(
                    num_frames=len(source_arrays["joint_action"]),
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
                if bool(cfg.include_original):
                    if original_complete:
                        _validate_completed_identity(
                            original_final_dir,
                            source.episode_number,
                            -1,
                            fingerprint,
                        )
                    else:
                        original_tmp_dir = _prepare_tmp_directory(
                            original_final_dir
                        )
                snapshots, replay_errors, final_reward = _replay_source_and_capture(
                    env_obj=env_obj,
                    source=source,
                    source_arrays=source_arrays,
                    event_frames=event_frames,
                    original_tmp_dir=original_tmp_dir,
                    cameras=cameras,
                    cfg=cfg,
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
                _clear_failure(metadata, source.episode_number, None)
                _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
                logging.info(
                    "源episode=%06d名义重放成功，events=%s，max_error=%.3g",
                    source.episode_number,
                    event_frames,
                    replay_errors["state"],
                )
            except Exception as exc:
                if original_tmp_dir is not None:
                    shutil.rmtree(original_tmp_dir, ignore_errors=True)
                source_attempt_errors.append(
                    {"error": f"{type(exc).__name__}: {exc}"}
                )
                _record_failure(
                    metadata,
                    source.episode_number,
                    None,
                    None,
                    source_attempt_errors,
                )
                _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
                logging.exception(
                    "源episode=%06d准备或名义重放失败。",
                    source.episode_number,
                )
                if not bool(cfg.continue_on_error):
                    raise
                continue

            for variant_index, event_frame in enumerate(event_frames):
                final_dir = _episode_dir(
                    output_run_dir, source.episode_number, variant_index
                )
                if _episode_is_complete(final_dir, cameras):
                    completed = _validate_completed_identity(
                        final_dir,
                        source.episode_number,
                        variant_index,
                        fingerprint,
                    )
                    if int(completed.get("source_injection_frame", -1)) != int(event_frame):
                        raise RuntimeError(
                            "已完成恢复分支的事件帧与确定性采样结果不一致: "
                            f"directory={final_dir}, recorded="
                            f"{completed.get('source_injection_frame')}, expected={event_frame}"
                        )
                    _clear_failure(
                        metadata, source.episode_number, variant_index
                    )
                    logging.info(
                        "跳过已完成恢复分支: source=%06d variant=%02d frame=%d",
                        source.episode_number,
                        variant_index,
                        event_frame,
                    )
                    continue

                horizon = (
                    int(cfg.recovery.max_steps)
                    + int(cfg.recovery.max_extra_zero_offset_steps)
                    + int(cfg.recovery.post_recovery_steps)
                    + 1
                )
                try:
                    feasible_lower, feasible_upper = _local_feasible_offset_bounds(
                        source_arrays,
                        event_frame,
                        horizon,
                        control_ranges,
                        max_abs,
                        float(cfg.view_joint_noise.joint_limit_margin_rad),
                    )
                except Exception as exc:
                    attempts = [{"error": f"{type(exc).__name__}: {exc}"}]
                    _record_failure(
                        metadata,
                        source.episode_number,
                        variant_index,
                        event_frame,
                        attempts,
                    )
                    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
                    logging.exception(
                        "恢复分支可行域计算失败: source=%06d variant=%02d frame=%d",
                        source.episode_number,
                        variant_index,
                        event_frame,
                    )
                    if not bool(cfg.continue_on_error):
                        raise
                    continue

                attempt_records: list[dict[str, Any]] = []
                succeeded = False
                for attempt in range(int(cfg.validation.max_branch_attempts)):
                    offset: np.ndarray | None = None
                    try:
                        offset = _sample_recovery_offset(
                            seed=int(cfg.seed),
                            source_episode=source.episode_number,
                            variant_index=variant_index,
                            attempt=attempt,
                            std=std,
                            max_abs=max_abs,
                            feasible_lower=feasible_lower,
                            feasible_upper=feasible_upper,
                            min_normalized_l2=float(
                                cfg.view_joint_noise.min_normalized_l2
                            ),
                            max_sampling_attempts=int(
                                cfg.view_joint_noise.max_sampling_attempts
                            ),
                        )
                        result = _generate_recovery_branch(
                            env_obj=env_obj,
                            source=source,
                            source_arrays=source_arrays,
                            snapshot=snapshots[event_frame],
                            event_frame=event_frame,
                            variant_index=variant_index,
                            attempt=attempt,
                            offset=offset,
                            feasible_lower=feasible_lower,
                            feasible_upper=feasible_upper,
                            output_run_dir=output_run_dir,
                            cameras=cameras,
                            cfg=cfg,
                            fingerprint=fingerprint,
                        )
                        _clear_failure(
                            metadata, source.episode_number, variant_index
                        )
                        _refresh_metadata(
                            metadata, output_run_dir, cameras, fingerprint
                        )
                        logging.info(
                            "已保存恢复分支: source=%06d variant=%02d frame=%d "
                            "attempt=%d offset=%s planned=%d actual=%d steps=%d "
                            "final_error=%.3g",
                            source.episode_number,
                            variant_index,
                            event_frame,
                            attempt,
                            np.array2string(offset, precision=4, suppress_small=True),
                            result.info["planned_recovery_steps"],
                            result.info["actual_recovery_steps"],
                            result.info["steps"],
                            result.info["final_recorded_view_max_abs_error_rad"],
                        )
                        succeeded = True
                        break
                    except Exception as exc:
                        attempt_records.append(
                            {
                                "attempt": int(attempt),
                                "error": f"{type(exc).__name__}: {exc}",
                                "offset_rad": (
                                    offset.tolist() if offset is not None else None
                                ),
                            }
                        )
                        logging.warning(
                            "恢复分支尝试失败，将重新采样: source=%06d "
                            "variant=%02d frame=%d attempt=%d/%d error=%s",
                            source.episode_number,
                            variant_index,
                            event_frame,
                            attempt + 1,
                            int(cfg.validation.max_branch_attempts),
                            exc,
                        )
                if not succeeded:
                    _record_failure(
                        metadata,
                        source.episode_number,
                        variant_index,
                        event_frame,
                        attempt_records,
                    )
                    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
                    logging.error(
                        "恢复分支重试耗尽并跳过: source=%06d variant=%02d frame=%d",
                        source.episode_number,
                        variant_index,
                        event_frame,
                    )
                    if not bool(cfg.continue_on_error):
                        raise RecoveryBranchError(
                            f"恢复分支{source.episode_number}/{variant_index}失败。"
                        )
    finally:
        env_obj.close()

    _refresh_metadata(metadata, output_run_dir, cameras, fingerprint)
    logging.info(
        "生成完成: saved=%d original=%d recovery=%d failures=%d output=%s",
        metadata["saved_episodes"],
        metadata["original_episodes"],
        metadata["recovery_episodes"],
        len(metadata.get("failures", [])),
        output_run_dir,
    )


@hydra.main(
    version_base="1.2",
    config_path="../configs/data_collect",
    config_name="contractive_view_trajectory_recovery",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _configure_mujoco_runtime(cfg)
    generate_contractive_view_recovery_run(cfg)


if __name__ == "__main__":
    main()
