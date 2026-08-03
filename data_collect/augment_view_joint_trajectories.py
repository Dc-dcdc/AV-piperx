#!/usr/bin/env python

"""用固定中间臂关节偏移扩展Quest专家轨迹。

生成器采用“原轨迹真实重放 + 增强视角临时分支渲染”：

1. 从原始episode的完整MuJoCo初态恢复世界；
2. 在执行 ``action_t`` 前检查并渲染当前状态；
3. 增强分支只临时修改中间臂qpos并调用 ``physics.forward()``，
   不执行物理步进，因此针、墙、圆柱及左右操作臂保持原轨迹状态；
4. 恢复原世界状态，再用原专家动作推进到下一帧。

输出仍是 ``quest_teleop`` 原始episode格式，可继续交给
``hugging_face/convert_data_to_hf.py`` 转换。原始轨迹使用
``episode_<source>``，增强轨迹使用 ``episode_<source>_aug_<variant>``。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import imageio.v2 as imageio
import numpy as np
from omegaconf import DictConfig, OmegaConf


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 修改渲染时序、标签语义或输出映射时必须递增，防止续跑混合不同版本。
# v2 增加 MuJoCo model.body_pos/body_quat 初态的保存与恢复。
SCHEMA_VERSION = 2
EPISODE_NAMING = "source_episode_with_aug_variant_v1"
NAMING_MIGRATION_FILE = ".episode_naming_migration.json"
OUTPUT_EPISODE_PATTERN = re.compile(
    r"^episode_(?P<source>\d{6,})(?:_aug_(?P<variant>\d{2,}))?$"
)
VIEW_SLICE = slice(14, 20)
ACTION_DIM = 20
VIEW_DIM = 6
ARM_JOINT_INDICES = np.asarray(
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    dtype=np.int64,
)
GRIPPER_INDICES = np.asarray([6, 13], dtype=np.int64)
VIEW_JOINT_INDICES = np.arange(14, 20, dtype=np.int64)
REPLAY_VALIDATION_MODES = {"strict", "fallback"}
REQUIRED_INITIAL_KEYS = (
    "initial_time",
    "initial_qpos",
    "initial_qvel",
    "initial_ctrl",
    "initial_act",
    "initial_mocap_pos",
    "initial_mocap_quat",
)
MODEL_BODY_INITIAL_KEYS = (
    "initial_model_body_pos",
    "initial_model_body_quat",
)
OPTIONAL_FRAME_KEYS = ("timestamp", "terminated", "truncated")


class ReplayStateMismatchError(RuntimeError):
    """专家动作重放没有重建原记录状态。"""


@dataclass(frozen=True)
class SourceEpisode:
    source_index: int
    episode_number: int
    directory: Path
    info: dict[str, Any]


@dataclass(frozen=True)
class Variant:
    output_index: int
    variant_index: int
    is_augmented: bool
    raw_offset: np.ndarray
    actual_offset: np.ndarray
    feasible_lower: np.ndarray
    feasible_upper: np.ndarray


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(value), file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def _episode_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("episode_"))
    except ValueError as exc:
        raise ValueError(f"无法从episode目录名解析编号: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"JSON根节点必须是对象: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sources(
    input_run_dir: Path,
    source_episode_indices: list[int] | None,
    max_source_episodes: int | None,
) -> list[SourceEpisode]:
    episodes_dir = input_run_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"输入run缺少episodes目录: {episodes_dir}")

    requested = (
        None
        if source_episode_indices is None
        else {int(value) for value in source_episode_indices}
    )
    directories = sorted(
        (
            path
            for path in episodes_dir.glob("episode_*")
            if path.is_dir() and (path / "arrays.npz").is_file()
        ),
        key=_episode_number,
    )
    if requested is not None:
        existing_numbers = {_episode_number(path) for path in directories}
        missing = sorted(requested - existing_numbers)
        if missing:
            raise FileNotFoundError(f"找不到指定的源episode: {missing}")
        directories = [
            path for path in directories if _episode_number(path) in requested
        ]
    if max_source_episodes is not None:
        if int(max_source_episodes) <= 0:
            raise ValueError("max_source_episodes必须为正整数或null。")
        directories = directories[: int(max_source_episodes)]
    if not directories:
        raise RuntimeError(f"输入run中没有可处理episode: {input_run_dir}")

    sources = []
    for source_index, directory in enumerate(directories):
        info_path = directory / "info.json"
        info = _load_json(info_path) if info_path.is_file() else {}
        sources.append(
            SourceEpisode(
                source_index=source_index,
                episode_number=_episode_number(directory),
                directory=directory,
                info=info,
            )
        )
    return sources


def _validate_source_arrays(
    path: Path,
    *,
    required_model_body_names: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}

    missing = [
        key
        for key in (*REQUIRED_INITIAL_KEYS, "joint_action", "observation_state")
        if key not in arrays
    ]
    if missing:
        raise KeyError(f"{path}缺少字段: {missing}")

    present_model_body_keys = [
        key for key in MODEL_BODY_INITIAL_KEYS if key in arrays
    ]
    if present_model_body_keys and len(present_model_body_keys) != len(
        MODEL_BODY_INITIAL_KEYS
    ):
        missing_model_keys = [
            key for key in MODEL_BODY_INITIAL_KEYS if key not in arrays
        ]
        raise KeyError(
            f"{path}的MuJoCo model body初态字段不完整，"
            f"缺少: {missing_model_keys}"
        )
    if required_model_body_names and not present_model_body_keys:
        raise KeyError(
            f"{path}缺少{list(MODEL_BODY_INITIAL_KEYS)}；当前环境会在reset()"
            "中随机修改无关节静态body "
            f"{list(required_model_body_names)}，仅靠initial_qpos无法恢复。"
            "请使用包含model body初态的新采集数据，或先为旧数据回填该状态。"
        )

    states = arrays["observation_state"]
    actions = arrays["joint_action"]
    if states.ndim != 2 or states.shape[1] != ACTION_DIM:
        raise ValueError(
            f"observation_state必须为[T,{ACTION_DIM}]，当前{states.shape}: {path}"
        )
    if actions.shape != states.shape:
        raise ValueError(
            f"joint_action必须与observation_state同形，"
            f"state={states.shape}, action={actions.shape}: {path}"
        )
    if len(states) == 0:
        raise ValueError(f"episode不能是空轨迹: {path}")
    for key in (
        *REQUIRED_INITIAL_KEYS,
        *present_model_body_keys,
        "joint_action",
        "observation_state",
    ):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{path}字段{key}包含NaN或Inf。")
    if present_model_body_keys:
        body_pos = arrays["initial_model_body_pos"]
        body_quat = arrays["initial_model_body_quat"]
        if body_pos.ndim != 2 or body_pos.shape[1] != 3:
            raise ValueError(
                "initial_model_body_pos必须为[nbody,3]，"
                f"当前{body_pos.shape}: {path}"
            )
        if body_quat.shape != (body_pos.shape[0], 4):
            raise ValueError(
                "initial_model_body_quat必须为[nbody,4]且与body_pos"
                f"具有相同nbody，当前{body_quat.shape}: {path}"
            )
    if arrays["initial_ctrl"].shape != (ACTION_DIM,):
        raise ValueError(
            f"initial_ctrl必须为({ACTION_DIM},)，当前"
            f"{arrays['initial_ctrl'].shape}: {path}"
        )
    for key in OPTIONAL_FRAME_KEYS:
        if key in arrays and len(arrays[key]) != len(states):
            raise ValueError(
                f"{path}字段{key}长度必须等于轨迹长度{len(states)}，"
                f"当前为{len(arrays[key])}。"
            )
    if "timestamp" in arrays and not np.isfinite(arrays["timestamp"]).all():
        raise ValueError(f"{path}字段timestamp包含NaN或Inf。")
    return arrays


def _configure_mujoco_runtime(cfg: DictConfig) -> None:
    backend = str(cfg.mujoco_gl).strip().lower()
    if backend not in {"auto", "glfw", "egl", "osmesa"}:
        raise ValueError(
            "mujoco_gl必须是auto/glfw/egl/osmesa，"
            f"当前为{cfg.mujoco_gl!r}。"
        )
    if backend == "auto":
        # 有桌面显示时便于本机调试；无DISPLAY的服务器自动使用EGL离屏渲染。
        os.environ.setdefault(
            "MUJOCO_GL",
            "glfw" if os.environ.get("DISPLAY") else "egl",
        )
    else:
        os.environ["MUJOCO_GL"] = backend
    if cfg.render_device is not None:
        render_device_value = float(cfg.render_device)
        if (
            not np.isfinite(render_device_value)
            or not render_device_value.is_integer()
            or render_device_value < 0
        ):
            raise ValueError("render_device必须为非负整数或null。")
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(render_device_value))


def _make_environment(
    env_id: str,
    cameras: tuple[str, ...],
    render_height: int,
    render_width: int,
):
    # 必须在配置MUJOCO_GL之后导入，避免EGL设备提前绑定。
    import gymnasium as gym

    import env as _register_guided_vision_envs  # noqa: F401

    environment = gym.make(
        env_id,
        disable_env_checker=True,
        cameras=list(cameras),
        episode_length=10**9,
        observation_height=int(render_height),
        observation_width=int(render_width),
    )
    return environment.unwrapped

# 恢复不属于 data.qpos 的 MuJoCo model body 初态。
def _restore_initial_model_body_state(
    env_obj,
    arrays: dict[str, np.ndarray],
) -> None:
    physics = env_obj._physics
    required_body_names = tuple(
        str(name)
        for name in getattr(env_obj, "replay_model_body_names", ())
    )
    present_keys = [key for key in MODEL_BODY_INITIAL_KEYS if key in arrays]
    if present_keys and len(present_keys) != len(MODEL_BODY_INITIAL_KEYS):
        missing = [key for key in MODEL_BODY_INITIAL_KEYS if key not in arrays]
        raise KeyError(f"MuJoCo model body初态字段不完整，缺少: {missing}")
    if required_body_names and not present_keys:
        raise KeyError(
            f"当前环境要求恢复静态任务body {list(required_body_names)}，"
            f"但源episode缺少{list(MODEL_BODY_INITIAL_KEYS)}。"
        )
    if not present_keys:
        return

    body_pos = np.asarray(arrays["initial_model_body_pos"], dtype=np.float64)
    body_quat = np.asarray(
        arrays["initial_model_body_quat"],
        dtype=np.float64,
    )
    if physics.model.body_pos.shape != body_pos.shape:
        raise ValueError(
            "initial_model_body_pos形状与当前MuJoCo模型不一致: "
            f"model={physics.model.body_pos.shape}, saved={body_pos.shape}"
        )
    if physics.model.body_quat.shape != body_quat.shape:
        raise ValueError(
            "initial_model_body_quat形状与当前MuJoCo模型不一致: "
            f"model={physics.model.body_quat.shape}, saved={body_quat.shape}"
        )
    physics.model.body_pos[:] = body_pos
    physics.model.body_quat[:] = body_quat


# 用保存的 model/data 状态覆盖 reset() 产生的临时初态。
def _restore_initial_state(env_obj, arrays: dict[str, np.ndarray]) -> None:
    physics = env_obj._physics
    env_obj.reset(seed=0)
    # 必须在forward()之前恢复 model 侧静态 body 初态；这些值不会包含在
    # data.qpos 中，例如 InsertCylinder 的 cylinder_container.pos。
    _restore_initial_model_body_state(env_obj, arrays)
    physics.data.qpos[:] = arrays["initial_qpos"]
    physics.data.qvel[:] = arrays["initial_qvel"]
    physics.data.time = float(arrays["initial_time"])
    physics.data.ctrl[:] = arrays["initial_ctrl"]
    if physics.data.act.shape != arrays["initial_act"].shape:
        raise ValueError(
            "initial_act形状与当前MuJoCo模型不一致: "
            f"data={physics.data.act.shape}, saved={arrays['initial_act'].shape}"
        )
    physics.data.act[:] = arrays["initial_act"]
    if physics.data.mocap_pos.shape != arrays["initial_mocap_pos"].shape:
        raise ValueError(
            "initial_mocap_pos形状与当前MuJoCo模型不一致: "
            f"data={physics.data.mocap_pos.shape}, "
            f"saved={arrays['initial_mocap_pos'].shape}"
        )
    if physics.data.mocap_quat.shape != arrays["initial_mocap_quat"].shape:
        raise ValueError(
            "initial_mocap_quat形状与当前MuJoCo模型不一致: "
            f"data={physics.data.mocap_quat.shape}, "
            f"saved={arrays['initial_mocap_quat'].shape}"
        )
    physics.data.mocap_pos[:] = arrays["initial_mocap_pos"]
    physics.data.mocap_quat[:] = arrays["initial_mocap_quat"]
    physics.forward()


def _read_agent_state(env_obj) -> np.ndarray:
    physics = env_obj._physics
    left = physics.bind(env_obj._left_joints).qpos.copy()
    right = physics.bind(env_obj._right_joints).qpos.copy()
    middle = physics.bind(env_obj._middle_joints).qpos.copy()
    left[6] = env_obj.left_gripper_norm_fn(left[6])
    right[6] = env_obj.right_gripper_norm_fn(right[6])
    return np.concatenate((left, right, middle)).astype(np.float64)


def _replay_error_groups(abs_error: np.ndarray) -> dict[str, float]:
    abs_error = np.asarray(abs_error, dtype=np.float64)
    if abs_error.shape != (ACTION_DIM,):
        raise ValueError(
            f"重放状态误差必须为({ACTION_DIM},)，当前{abs_error.shape}。"
        )
    return {
        "state": float(abs_error.max()),
        "arm_joint": float(abs_error[ARM_JOINT_INDICES].max()),
        "gripper": float(abs_error[GRIPPER_INDICES].max()),
        "view_joint": float(abs_error[VIEW_JOINT_INDICES].max()),
    }


def _replay_validation_limits(
    cfg: DictConfig,
    mode: str,
) -> dict[str, float]:
    if mode == "strict":
        return {
            "state": float(cfg.replay_validation.max_state_abs_error),
        }
    if mode == "fallback":
        return {
            "arm_joint": float(
                cfg.replay_validation.retry_max_arm_joint_abs_error
            ),
            "gripper": float(
                cfg.replay_validation.retry_max_gripper_abs_error
            ),
            "view_joint": float(
                cfg.replay_validation.retry_max_view_joint_abs_error
            ),
        }
    raise ValueError(
        f"未知replay validation mode={mode!r}，"
        f"必须是{sorted(REPLAY_VALIDATION_MODES)}。"
    )


def _replay_validation_violation(
    groups: dict[str, float],
    limits: dict[str, float],
) -> tuple[str, float, float] | None:
    violations = [
        (name, float(groups[name]), float(limit))
        for name, limit in limits.items()
        if float(groups[name]) > float(limit)
    ]
    if not violations:
        return None
    return max(
        violations,
        key=lambda item: (
            float("inf") if item[2] == 0 else item[1] / item[2]
        ),
    )


def _apply_original_action(env_obj, action: np.ndarray) -> None:
    """只推进物理，不额外渲染get_obs或计算任务奖励。"""
    from env.constants import SIM_PHYSICS_ENV_STEP_RATIO

    physics = env_obj._physics
    action = np.asarray(action, dtype=np.float64)
    physics.bind(env_obj._left_actuators[:6]).ctrl = action[:6]
    physics.bind(env_obj._right_actuators[:6]).ctrl = action[7:13]
    physics.bind(env_obj._left_actuators[6]).ctrl = (
        env_obj.left_gripper_unnorm_fn(np.clip(action[6], 0.0, 1.0))
    )
    physics.bind(env_obj._right_actuators[6]).ctrl = (
        env_obj.right_gripper_unnorm_fn(np.clip(action[13], 0.0, 1.0))
    )
    physics.bind(env_obj._middle_actuators).ctrl = action[VIEW_SLICE]
    for _ in range(SIM_PHYSICS_ENV_STEP_RATIO):
        physics.step()


def _feasible_offset_bounds(
    arrays: dict[str, np.ndarray],
    joint_ranges: np.ndarray,
    max_abs: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    if joint_ranges.shape != (VIEW_DIM, 2):
        raise ValueError(f"中间臂关节限位必须为[{VIEW_DIM},2]，当前{joint_ranges.shape}")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("joint_limit_margin_rad必须非负。")
    if not np.isfinite(joint_ranges).all() or not np.isfinite(max_abs).all():
        raise ValueError("关节限位和max_abs_rad必须都是有限数。")
    joint_min = joint_ranges[:, 0] + margin
    joint_max = joint_ranges[:, 1] - margin
    if np.any(joint_min >= joint_max):
        raise ValueError("joint_limit_margin_rad过大，导致关节可行区间为空。")

    values = np.concatenate(
        (
            arrays["observation_state"][:, VIEW_SLICE],
            arrays["joint_action"][:, VIEW_SLICE],
            arrays["initial_ctrl"][None, VIEW_SLICE],
        ),
        axis=0,
    ).astype(np.float64)
    lower = np.maximum(-max_abs, joint_min - values.min(axis=0))
    upper = np.minimum(max_abs, joint_max - values.max(axis=0))
    if np.any(lower > upper):
        raise ValueError(
            "原轨迹本身没有可用的固定View偏移区间: "
            f"lower={lower.tolist()}, upper={upper.tolist()}"
        )
    return lower, upper


def _variant_rng(seed: int, source_episode: int, variant_index: int):
    sequence = np.random.SeedSequence(
        [int(seed), int(source_episode), int(variant_index)]
    )
    return np.random.default_rng(sequence)


def _sample_offset(
    *,
    seed: int,
    source_episode: int,
    variant_index: int,
    std: np.ndarray,
    max_abs: np.ndarray,
    feasible_lower: np.ndarray,
    feasible_upper: np.ndarray,
    min_normalized_l2: float,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = _variant_rng(seed, source_episode, variant_index)
    safe_std = np.maximum(std, 1e-12)
    last_raw = np.zeros(VIEW_DIM, dtype=np.float64)
    last_actual = np.zeros(VIEW_DIM, dtype=np.float64)

    for _ in range(max_attempts):
        raw = rng.normal(loc=0.0, scale=std, size=VIEW_DIM)
        if np.any(np.abs(raw) > max_abs):
            continue
        actual = np.clip(raw, feasible_lower, feasible_upper)
        if np.linalg.norm(actual / safe_std) < min_normalized_l2:
            last_raw, last_actual = raw, actual
            continue
        return raw.astype(np.float64), actual.astype(np.float64)

    raise RuntimeError(
        f"在{max_attempts}次内没有采到有效固定偏移；"
        f"last_raw={last_raw.tolist()}, last_actual={last_actual.tolist()}。"
    )


def _build_variants(
    *,
    source: SourceEpisode,
    source_position: int,
    arrays: dict[str, np.ndarray],
    joint_ranges: np.ndarray,
    cfg: DictConfig,
) -> list[Variant]:
    std = np.asarray(cfg.view_joint_noise.std_rad, dtype=np.float64)
    max_abs = np.asarray(cfg.view_joint_noise.max_abs_rad, dtype=np.float64)
    if (
        std.shape != (VIEW_DIM,)
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError(f"std_rad必须是{VIEW_DIM}个有限正数。")
    if (
        max_abs.shape != (VIEW_DIM,)
        or not np.isfinite(max_abs).all()
        or np.any(max_abs <= 0)
    ):
        raise ValueError(f"max_abs_rad必须是{VIEW_DIM}个有限正数。")
    if str(cfg.view_joint_noise.distribution) != "truncated_gaussian":
        raise ValueError("当前仅支持view_joint_noise.distribution=truncated_gaussian。")
    min_normalized_l2 = float(cfg.view_joint_noise.min_normalized_l2)
    max_sampling_attempts = int(cfg.view_joint_noise.max_sampling_attempts)
    if not np.isfinite(min_normalized_l2) or min_normalized_l2 < 0:
        raise ValueError("min_normalized_l2必须是有限非负数。")
    if max_sampling_attempts <= 0:
        raise ValueError("max_sampling_attempts必须为正整数。")

    lower, upper = _feasible_offset_bounds(
        arrays,
        joint_ranges,
        max_abs,
        float(cfg.view_joint_noise.joint_limit_margin_rad),
    )
    slots_per_source = int(cfg.variants_per_episode) + int(bool(cfg.include_original))
    variants = []
    next_slot = 0
    if bool(cfg.include_original):
        variants.append(
            Variant(
                output_index=source_position * slots_per_source,
                variant_index=-1,
                is_augmented=False,
                raw_offset=np.zeros(VIEW_DIM, dtype=np.float64),
                actual_offset=np.zeros(VIEW_DIM, dtype=np.float64),
                feasible_lower=lower,
                feasible_upper=upper,
            )
        )
        next_slot = 1

    for variant_index in range(int(cfg.variants_per_episode)):
        raw, actual = _sample_offset(
            seed=int(cfg.seed),
            source_episode=source.episode_number,
            variant_index=variant_index,
            std=std,
            max_abs=max_abs,
            feasible_lower=lower,
            feasible_upper=upper,
            min_normalized_l2=min_normalized_l2,
            max_attempts=max_sampling_attempts,
        )
        variants.append(
            Variant(
                output_index=source_position * slots_per_source + next_slot,
                variant_index=variant_index,
                is_augmented=True,
                raw_offset=raw,
                actual_offset=actual,
                feasible_lower=lower,
                feasible_upper=upper,
            )
        )
        next_slot += 1
    return variants


class StereoVideoWriter:
    def __init__(self, videos_dir: Path, cameras: tuple[str, ...], cfg: DictConfig):
        videos_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            camera: videos_dir / f"{camera}.mp4" for camera in cameras
        }
        self.writers: dict[str, Any] = {}
        self.closed = False
        try:
            for camera, path in self.paths.items():
                self.writers[camera] = imageio.get_writer(
                    str(path),
                    format="FFMPEG",
                    mode="I",
                    fps=int(cfg.fps),
                    codec=str(cfg.video.codec),
                    pixelformat=str(cfg.video.pixel_format),
                    macro_block_size=int(cfg.video.macro_block_size),
                    ffmpeg_params=[
                        "-crf",
                        str(int(cfg.video.crf)),
                        "-g",
                        str(int(cfg.video.gop)),
                    ],
                )
        except BaseException:
            # 第二路视频初始化失败时，也要关闭已经启动的第一路ffmpeg。
            for writer in self.writers.values():
                try:
                    writer.close()
                except BaseException:
                    pass
            self.closed = True
            raise

    def append(self, camera: str, frame: np.ndarray) -> None:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            if frame.max(initial=0) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.writers[camera].append_data(np.ascontiguousarray(frame))

    def close(self) -> None:
        if self.closed:
            return
        first_error = None
        for writer in self.writers.values():
            try:
                writer.close()
            except BaseException as exc:  # ffmpeg关闭错误也必须清理其余writer。
                if first_error is None:
                    first_error = exc
        self.closed = True
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "StereoVideoWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.close()
        except BaseException:
            # 保留循环内部更接近根因的异常；正常退出时仍传播ffmpeg关闭错误。
            if exc_type is None:
                raise
            logging.exception("关闭双目视频writer时发生次级异常。")
        return False


def _make_output_arrays(
    env_obj,
    source_arrays: dict[str, np.ndarray],
    offset: np.ndarray,
) -> dict[str, np.ndarray]:
    states = source_arrays["observation_state"].astype(np.float32, copy=True)
    actions = source_arrays["joint_action"].astype(np.float32, copy=True)
    states[:, VIEW_SLICE] += offset.astype(np.float32)
    actions[:, VIEW_SLICE] += offset.astype(np.float32)

    _restore_initial_state(env_obj, source_arrays)
    physics = env_obj._physics
    physics.bind(env_obj._middle_joints).qpos = (
        physics.bind(env_obj._middle_joints).qpos + offset
    )
    physics.bind(env_obj._middle_actuators).ctrl = (
        physics.bind(env_obj._middle_actuators).ctrl + offset
    )
    physics.forward()

    output = {
        "joint_action": actions,
        "observation_state": states,
        "initial_time": np.asarray(source_arrays["initial_time"]).copy(),
        "initial_qpos": physics.data.qpos.copy().astype(np.float64),
        "initial_qvel": np.asarray(source_arrays["initial_qvel"]).copy(),
        "initial_ctrl": physics.data.ctrl.copy().astype(np.float64),
        "initial_act": np.asarray(source_arrays["initial_act"]).copy(),
        "initial_mocap_pos": np.asarray(
            source_arrays["initial_mocap_pos"]
        ).copy(),
        "initial_mocap_quat": np.asarray(
            source_arrays["initial_mocap_quat"]
        ).copy(),
        "initial_model_body_pos": physics.model.body_pos.copy().astype(
            np.float64
        ),
        "initial_model_body_quat": physics.model.body_quat.copy().astype(
            np.float64
        ),
    }
    for key in OPTIONAL_FRAME_KEYS:
        if key in source_arrays:
            output[key] = np.asarray(source_arrays[key]).copy()
    return output


def _output_episode_name(source_episode: int, variant_index: int) -> str:
    source_episode = int(source_episode)
    variant_index = int(variant_index)
    if source_episode < 0:
        raise ValueError("source_episode必须为非负整数。")
    if variant_index == -1:
        return f"episode_{source_episode:06d}"
    if variant_index < 0:
        raise ValueError("增强variant_index必须为非负整数。")
    return f"episode_{source_episode:06d}_aug_{variant_index:02d}"


def _parse_output_episode_name(name: str) -> tuple[int, int]:
    match = OUTPUT_EPISODE_PATTERN.fullmatch(str(name))
    if match is None:
        raise ValueError(
            "输出episode目录必须形如episode_000003或"
            f"episode_000003_aug_00，当前为{name!r}。"
        )
    source_episode = int(match.group("source"))
    variant = match.group("variant")
    return source_episode, (-1 if variant is None else int(variant))


def _episode_dir(
    output_run_dir: Path,
    source_episode: int,
    variant_index: int,
) -> Path:
    return (
        output_run_dir
        / "episodes"
        / _output_episode_name(source_episode, variant_index)
    )


def _episode_is_complete(directory: Path, cameras: tuple[str, ...]) -> bool:
    required = [directory / "arrays.npz", directory / "info.json"]
    required.extend(directory / "videos" / f"{camera}.mp4" for camera in cameras)
    return directory.is_dir() and all(path.is_file() and path.stat().st_size > 0 for path in required)


def _validate_completed_episode_identity(
    directory: Path,
    source: SourceEpisode,
    variant: Variant,
) -> None:
    info = _load_json(directory / "info.json")
    expected_name = _output_episode_name(
        source.episode_number,
        variant.variant_index,
    )
    expected = {
        "episode": int(variant.output_index),
        "output_index": int(variant.output_index),
        "episode_name": expected_name,
        "episode_naming": EPISODE_NAMING,
        "path": f"episodes/{expected_name}",
        "source_episode": int(source.episode_number),
        "variant_index": int(variant.variant_index),
        "is_augmented": bool(variant.is_augmented),
    }
    mismatches = {
        key: {"expected": value, "actual": info.get(key)}
        for key, value in expected.items()
        if info.get(key) != value
    }
    if directory.name != expected_name:
        mismatches["directory_name"] = {
            "expected": expected_name,
            "actual": directory.name,
        }
    actual_offset = np.asarray(info.get("view_joint_offset_rad", []), dtype=np.float64)
    if (
        actual_offset.shape != (VIEW_DIM,)
        or not np.allclose(
            actual_offset,
            variant.actual_offset,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        mismatches["view_joint_offset_rad"] = {
            "expected": variant.actual_offset.tolist(),
            "actual": info.get("view_joint_offset_rad"),
        }
    if mismatches:
        raise RuntimeError(
            f"已完成episode与当前源/变体映射不一致: {directory}, "
            f"mismatches={mismatches}"
        )


def _make_episode_info(
    *,
    output_run_dir: Path,
    source: SourceEpisode,
    variant: Variant,
    steps: int,
    cameras: tuple[str, ...],
    cfg: DictConfig,
    max_state_error: float,
    max_group_errors: dict[str, float],
    replay_validation_mode: str,
    replay_validation_limits: dict[str, float],
) -> dict[str, Any]:
    source_final_info = source.info.get("final_info", {})
    if not isinstance(source_final_info, dict):
        source_final_info = {}
    source_success = bool(
        source.info.get(
            "success",
            source_final_info.get("is_success", False),
        )
    )
    return {
        "episode": int(variant.output_index),
        "output_index": int(variant.output_index),
        "episode_name": _output_episode_name(
            source.episode_number,
            variant.variant_index,
        ),
        "episode_naming": EPISODE_NAMING,
        "success": source_success,
        # 增强分支没有重新执行任务；成功标签继承自同一条专家轨迹。
        "success_inherited_from_source": True,
        "steps": int(steps),
        "fps": int(cfg.fps),
        "path": str(
            _episode_dir(
                output_run_dir,
                source.episode_number,
                variant.variant_index,
            ).relative_to(output_run_dir)
        ),
        "save_rgb": True,
        "save_videos": True,
        "save_depth": False,
        "save_reward_debug": False,
        "reward_debug_path": None,
        "final_cumulative_reward": source.info.get("final_cumulative_reward"),
        "observation_npz_keys": {"agent_pos": "observation_state"},
        "depth_npz_keys": {},
        "video_paths": {
            f"pixels.{camera}": f"videos/{camera}.mp4"
            for camera in cameras
        },
        "source_episode": int(source.episode_number),
        "source_path": str(source.directory),
        "variant_index": int(variant.variant_index),
        "is_augmented": bool(variant.is_augmented),
        "view_joint_offset_rad": variant.actual_offset.tolist(),
        "raw_view_joint_offset_rad": variant.raw_offset.tolist(),
        "offset_limited": bool(
            not np.allclose(
                variant.raw_offset,
                variant.actual_offset,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "view_offset_feasible_lower_rad": variant.feasible_lower.tolist(),
        "view_offset_feasible_upper_rad": variant.feasible_upper.tolist(),
        "max_replay_state_abs_error": float(max_state_error),
        "max_replay_arm_joint_abs_error": float(
            max_group_errors["arm_joint"]
        ),
        "max_replay_gripper_abs_error": float(
            max_group_errors["gripper"]
        ),
        "max_replay_view_joint_abs_error": float(
            max_group_errors["view_joint"]
        ),
        "replay_validation_mode": replay_validation_mode,
        "replay_validation_retried": replay_validation_mode == "fallback",
        "replay_validation_limits": replay_validation_limits,
        "final_info": {
            **source_final_info,
            "source_episode": int(source.episode_number),
            "variant_index": int(variant.variant_index),
            "is_augmented": bool(variant.is_augmented),
            "pre_action_rerendered": True,
            "counterfactual_view_branch": bool(variant.is_augmented),
            "counterfactual_success_inherited": True,
            "video_replay_rendered": True,
            "video_replay_cameras": list(cameras),
        },
    }


def _generate_episode(
    *,
    env_obj,
    source: SourceEpisode,
    source_arrays: dict[str, np.ndarray],
    variant: Variant,
    output_run_dir: Path,
    cameras: tuple[str, ...],
    cfg: DictConfig,
    replay_validation_mode: str = "strict",
) -> dict[str, Any]:
    final_dir = _episode_dir(
        output_run_dir,
        source.episode_number,
        variant.variant_index,
    )
    tmp_dir = final_dir.with_name(final_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if final_dir.exists():
        raise FileExistsError(
            f"目标episode已存在但不完整，拒绝覆盖: {final_dir}"
        )
    tmp_dir.mkdir(parents=True)

    try:
        _restore_initial_state(env_obj, source_arrays)
        physics = env_obj._physics
        max_state_error = 0.0
        max_group_errors = {
            "arm_joint": 0.0,
            "gripper": 0.0,
            "view_joint": 0.0,
        }
        validation_limits = _replay_validation_limits(
            cfg,
            replay_validation_mode,
        )
        actions = source_arrays["joint_action"]
        recorded_states = source_arrays["observation_state"]

        with StereoVideoWriter(tmp_dir / "videos", cameras, cfg) as writer:
            for frame_index, original_action in enumerate(actions):
                replayed_state = _read_agent_state(env_obj)
                abs_error = np.abs(
                    replayed_state
                    - recorded_states[frame_index].astype(np.float64)
                )
                error_groups = _replay_error_groups(abs_error)
                state_error = error_groups["state"]
                max_state_error = max(max_state_error, state_error)
                for group_name in max_group_errors:
                    max_group_errors[group_name] = max(
                        max_group_errors[group_name],
                        error_groups[group_name],
                    )
                violation = _replay_validation_violation(
                    error_groups,
                    validation_limits,
                )
                if violation is not None:
                    group_name, group_error, group_limit = violation
                    violation_text = (
                        f"state_error={group_error:.6g} > "
                        f"{group_limit:.6g}"
                        if group_name == "state"
                        else (
                            f"{group_name}_error={group_error:.6g} > "
                            f"{group_limit:.6g}, "
                            f"state_error={state_error:.6g}"
                        )
                    )
                    raise ReplayStateMismatchError(
                        f"source_episode={source.episode_number}, "
                        f"frame={frame_index}, mode={replay_validation_mode}, "
                        f"{violation_text}"
                    )

                if variant.is_augmented:
                    physics_state = np.asarray(physics.get_state()).copy()
                    actuator_ctrl = physics.data.ctrl.copy()
                    augmented_view_state = (
                        recorded_states[frame_index, VIEW_SLICE].astype(
                            np.float64
                        )
                        + variant.actual_offset
                    )
                    physics.bind(env_obj._middle_joints).qpos = (
                        augmented_view_state
                    )
                    physics.bind(env_obj._middle_actuators).ctrl = (
                        augmented_view_state
                    )
                    physics.forward()

                for camera in cameras:
                    frame = physics.render(
                        height=int(cfg.render_height),
                        width=int(cfg.render_width),
                        camera_id=camera,
                    )
                    writer.append(camera, frame)

                if variant.is_augmented:
                    physics.set_state(physics_state)
                    physics.data.ctrl[:] = actuator_ctrl
                    physics.forward()

                _apply_original_action(env_obj, original_action)

        output_arrays = _make_output_arrays(
            env_obj,
            source_arrays,
            variant.actual_offset,
        )
        np.savez_compressed(tmp_dir / "arrays.npz", **output_arrays)
        info = _make_episode_info(
            output_run_dir=output_run_dir,
            source=source,
            variant=variant,
            steps=len(actions),
            cameras=cameras,
            cfg=cfg,
            max_state_error=max_state_error,
            max_group_errors=max_group_errors,
            replay_validation_mode=replay_validation_mode,
            replay_validation_limits=validation_limits,
        )
        _write_json_atomic(tmp_dir / "info.json", info)
        tmp_dir.rename(final_dir)
        return info
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _semantic_config(cfg: DictConfig, input_run_dir: Path, env_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "input_run_dir": str(input_run_dir),
        "env_id": env_id,
        "episode_naming": EPISODE_NAMING,
        "include_original": bool(cfg.include_original),
        "variants_per_episode": int(cfg.variants_per_episode),
        "source_episode_indices": (
            None
            if cfg.source_episode_indices is None
            else sorted(int(value) for value in cfg.source_episode_indices)
        ),
        "seed": int(cfg.seed),
        "view_joint_noise": OmegaConf.to_container(
            cfg.view_joint_noise,
            resolve=True,
        ),
        "cameras": list(cfg.cameras),
        "render_height": int(cfg.render_height),
        "render_width": int(cfg.render_width),
        "fps": int(cfg.fps),
        "video": OmegaConf.to_container(cfg.video, resolve=True),
    }


def _fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _rewrite_episode_naming_info(
    directory: Path,
    output_run_dir: Path,
) -> None:
    info_path = directory / "info.json"
    info = _load_json(info_path)
    if "source_episode" not in info or "variant_index" not in info:
        raise KeyError(
            f"旧episode缺少source_episode/variant_index，无法迁移: {info_path}"
        )
    expected_name = _output_episode_name(
        int(info["source_episode"]),
        int(info["variant_index"]),
    )
    if directory.name != expected_name:
        raise RuntimeError(
            f"episode目录名与记录身份不一致: directory={directory.name}, "
            f"expected={expected_name}"
        )
    output_index = int(info.get("output_index", info["episode"]))
    updates = {
        "episode": output_index,
        "output_index": output_index,
        "episode_name": expected_name,
        "episode_naming": EPISODE_NAMING,
        "path": str(directory.relative_to(output_run_dir)),
    }
    if any(info.get(key) != value for key, value in updates.items()):
        info.update(updates)
        _write_json_atomic(info_path, info)


def _migrate_legacy_episode_names(
    output_run_dir: Path,
    cameras: tuple[str, ...],
) -> int:
    """把旧的连续输出编号安全迁移为“源episode + 增强序号”命名。"""
    episodes_dir = output_run_dir / "episodes"
    if not episodes_dir.is_dir():
        return 0

    active_tmp_dirs = sorted(episodes_dir.glob("episode_*.tmp"))
    if active_tmp_dirs:
        raise RuntimeError(
            "检测到正在生成或未清理的episode临时目录，拒绝并发重命名: "
            f"{[path.name for path in active_tmp_dirs]}"
        )

    marker_path = output_run_dir / NAMING_MIGRATION_FILE
    if marker_path.is_file():
        plan = _load_json(marker_path)
        if plan.get("episode_naming") != EPISODE_NAMING:
            raise RuntimeError(f"无法识别命名迁移记录: {marker_path}")
    else:
        stale_tmp_dirs = sorted(episodes_dir.glob(".episode_naming_tmp_*"))
        if stale_tmp_dirs:
            raise RuntimeError(
                "发现命名迁移临时目录但缺少迁移记录，请先人工检查: "
                f"{[path.name for path in stale_tmp_dirs]}"
            )

        directories = sorted(
            path
            for path in episodes_dir.glob("episode_*")
            if path.is_dir()
        )
        identities = []
        desired_names = set()
        for directory in directories:
            if not _episode_is_complete(directory, cameras):
                raise RuntimeError(
                    f"发现不完整episode，拒绝迁移命名: {directory}"
                )
            info = _load_json(directory / "info.json")
            if "source_episode" not in info or "variant_index" not in info:
                raise KeyError(
                    "旧episode缺少source_episode/variant_index，"
                    f"无法迁移: {directory}"
                )
            desired_name = _output_episode_name(
                int(info["source_episode"]),
                int(info["variant_index"]),
            )
            if desired_name in desired_names:
                raise RuntimeError(
                    f"多个episode映射到同一目标目录: {desired_name}"
                )
            desired_names.add(desired_name)
            identities.append((directory.name, desired_name))

        current_names = {current for current, _ in identities}
        mappings = [
            (current, desired)
            for current, desired in identities
            if current != desired
        ]
        for _, desired in mappings:
            destination = episodes_dir / desired
            if destination.exists() and desired not in current_names:
                raise FileExistsError(f"目标episode目录已被占用: {destination}")

        if not mappings:
            for directory in directories:
                _rewrite_episode_naming_info(directory, output_run_dir)
            return 0

        plan = {
            "episode_naming": EPISODE_NAMING,
            "stage": "planned",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entries": [
                {
                    "current": current,
                    "temporary": f".episode_naming_tmp_{index:06d}",
                    "desired": desired,
                }
                for index, (current, desired) in enumerate(mappings)
            ],
        }
        _write_json_atomic(marker_path, plan)

    entries = plan.get("entries")
    if not isinstance(entries, list):
        raise TypeError(f"命名迁移记录entries必须是列表: {marker_path}")

    if plan.get("stage") == "planned":
        for entry in entries:
            current = episodes_dir / str(entry["current"])
            temporary = episodes_dir / str(entry["temporary"])
            if temporary.exists():
                if current.exists():
                    raise RuntimeError(
                        f"命名迁移同时存在源和临时目录: {current}, {temporary}"
                    )
                continue
            if not current.is_dir():
                raise FileNotFoundError(f"命名迁移找不到源目录: {current}")
            current.rename(temporary)
        plan["stage"] = "staged"
        _write_json_atomic(marker_path, plan)

    if plan.get("stage") == "staged":
        for entry in entries:
            temporary = episodes_dir / str(entry["temporary"])
            destination = episodes_dir / str(entry["desired"])
            if destination.exists():
                if temporary.exists():
                    raise RuntimeError(
                        f"命名迁移目标和临时目录同时存在: "
                        f"{destination}, {temporary}"
                    )
                continue
            if not temporary.is_dir():
                raise FileNotFoundError(
                    f"命名迁移找不到临时目录: {temporary}"
                )
            temporary.rename(destination)
        plan["stage"] = "renamed"
        _write_json_atomic(marker_path, plan)

    if plan.get("stage") != "renamed":
        raise RuntimeError(
            f"未知命名迁移阶段: {plan.get('stage')!r}, marker={marker_path}"
        )

    for directory in sorted(episodes_dir.glob("episode_*")):
        if directory.is_dir() and not directory.name.endswith(".tmp"):
            _rewrite_episode_naming_info(directory, output_run_dir)
    marker_path.unlink()
    return len(entries)


def _scan_completed_infos(
    output_run_dir: Path,
    cameras: tuple[str, ...],
) -> list[dict[str, Any]]:
    infos = []
    episodes_dir = output_run_dir / "episodes"
    if not episodes_dir.exists():
        return infos
    identities: dict[tuple[int, int], Path] = {}
    for directory in episodes_dir.glob("episode_*"):
        if directory.name.endswith(".tmp") or not directory.is_dir():
            continue
        identity = _parse_output_episode_name(directory.name)
        if identity in identities:
            raise RuntimeError(
                "发现重复的source/variant输出身份: "
                f"{identities[identity]}, {directory}"
            )
        identities[identity] = directory

    for identity, directory in sorted(
        identities.items(),
        key=lambda item: (item[0][0], item[0][1] + 1),
    ):
        if not _episode_is_complete(directory, cameras):
            raise RuntimeError(
                f"发现不完整的最终episode目录，请人工检查后再续传: {directory}"
            )
        info = _load_json(directory / "info.json")
        source_episode, variant_index = identity
        if (
            int(info.get("source_episode", -1)) != source_episode
            or int(info.get("variant_index", -2)) != variant_index
        ):
            raise RuntimeError(
                f"目录名与info身份不一致: directory={directory}, "
                f"directory_identity={identity}, "
                f"info_identity="
                f"{(info.get('source_episode'), info.get('variant_index'))}"
            )
        infos.append(info)
    return infos


def _update_source_manifest(
    metadata: dict[str, Any],
    sources: list[SourceEpisode],
) -> None:
    manifest = metadata.setdefault("source_manifest", {})
    if not isinstance(manifest, dict):
        raise TypeError("metadata.source_manifest必须是对象。")

    occupied_positions = {
        int(entry["source_position"]): int(episode_number)
        for episode_number, entry in manifest.items()
    }
    for source in sources:
        arrays_path = source.directory / "arrays.npz"
        info_path = source.directory / "info.json"
        entry = {
            "source_episode": int(source.episode_number),
            "source_position": int(source.source_index),
            "arrays_sha256": _sha256_file(arrays_path),
            "info_sha256": (
                _sha256_file(info_path) if info_path.is_file() else None
            ),
        }
        key = str(source.episode_number)
        previous = manifest.get(key)
        if previous is not None and previous != entry:
            raise RuntimeError(
                "源episode自上次生成后发生变化，拒绝混合新旧结果: "
                f"source_episode={source.episode_number}, "
                f"previous={previous}, current={entry}"
            )
        position_owner = occupied_positions.get(source.source_index)
        if position_owner is not None and position_owner != source.episode_number:
            raise RuntimeError(
                "源episode排序发生变化，已有输出编号不能安全复用: "
                f"source_position={source.source_index}, "
                f"existing_episode={position_owner}, "
                f"current_episode={source.episode_number}"
            )
        manifest[key] = entry
        occupied_positions[source.source_index] = source.episode_number


def _failure_key(failure: dict[str, Any]) -> tuple[int | None, int | None]:
    return (
        failure.get("source_episode"),
        failure.get("variant_index"),
    )


def _record_failure(metadata: dict[str, Any], failure: dict[str, Any]) -> None:
    key = _failure_key(failure)
    failures = [
        item
        for item in metadata.setdefault("failures", [])
        if _failure_key(item) != key
    ]
    failures.append(failure)
    metadata["failures"] = failures


def _clear_failure(
    metadata: dict[str, Any],
    source_episode: int,
    variant_index: int | None,
) -> None:
    key = (int(source_episode), variant_index)
    metadata["failures"] = [
        item
        for item in metadata.setdefault("failures", [])
        if _failure_key(item) != key
    ]


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
    if output_run_dir.exists() and not bool(cfg.resume):
        raise FileExistsError(
            f"输出目录已存在且resume=false: {output_run_dir}"
        )
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "episodes").mkdir(exist_ok=True)

    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        previous = metadata.get("generation_fingerprint")
        if previous != fingerprint:
            raise ValueError(
                "续生成配置与已有输出不一致，请使用新的output_run_dir。"
                f" existing={previous!r}, current={fingerprint!r}"
            )
    else:
        non_metadata_entries = [
            path
            for path in output_run_dir.iterdir()
            if path.name != "episodes"
        ]
        existing_episode_entries = list(
            (output_run_dir / "episodes").iterdir()
        )
        if non_metadata_entries or existing_episode_entries:
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
            "save_reward_debug": False,
            "video_save_mode": "pre_action_counterfactual_view_rerender",
            "success_semantics": "inherited_from_source_trajectory",
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

    metadata["episode_naming"] = EPISODE_NAMING
    metadata["episodes"] = _scan_completed_infos(output_run_dir, cameras)
    metadata["saved_episodes"] = len(metadata["episodes"])
    metadata["successful_episodes"] = sum(
        bool(info.get("success", False)) for info in metadata["episodes"]
    )
    metadata["original_episodes"] = sum(
        not bool(info.get("is_augmented", False))
        for info in metadata["episodes"]
    )
    metadata["augmented_episodes"] = sum(
        bool(info.get("is_augmented", False))
        for info in metadata["episodes"]
    )
    metadata["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_json_atomic(metadata_path, metadata)
    return metadata


def _upgrade_metadata_episode_naming(
    output_run_dir: Path,
    semantic_config: dict[str, Any],
    fingerprint: str,
) -> bool:
    """只允许旧配置通过增加episode_naming升级到当前fingerprint。"""
    metadata_path = output_run_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = _load_json(metadata_path)
    if metadata.get("generation_fingerprint") == fingerprint:
        return False
    previous_config = metadata.get("generation_config")
    if not isinstance(previous_config, dict):
        return False
    upgraded_config = dict(previous_config)
    previous_naming = upgraded_config.get("episode_naming")
    if previous_naming not in {None, EPISODE_NAMING}:
        return False
    upgraded_config["episode_naming"] = EPISODE_NAMING
    if upgraded_config != semantic_config:
        return False
    if _fingerprint(upgraded_config) != fingerprint:
        return False
    metadata["generation_config"] = upgraded_config
    metadata["generation_fingerprint"] = fingerprint
    metadata["episode_naming"] = EPISODE_NAMING
    metadata["episode_naming_fingerprint_upgraded_at"] = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    _write_json_atomic(metadata_path, metadata)
    return True


def _record_metadata_progress(
    metadata: dict[str, Any],
    output_run_dir: Path,
    cameras: tuple[str, ...],
) -> None:
    metadata["episodes"] = _scan_completed_infos(output_run_dir, cameras)
    metadata["saved_episodes"] = len(metadata["episodes"])
    metadata["successful_episodes"] = sum(
        bool(info.get("success", False)) for info in metadata["episodes"]
    )
    metadata["original_episodes"] = sum(
        not bool(info.get("is_augmented", False))
        for info in metadata["episodes"]
    )
    metadata["augmented_episodes"] = sum(
        bool(info.get("is_augmented", False))
        for info in metadata["episodes"]
    )
    metadata["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_json_atomic(output_run_dir / "metadata.json", metadata)


def migrate_existing_output_episode_names(
    output_run_dir: str | Path,
    cameras: tuple[str, ...] = ("zed_cam_left", "zed_cam_right"),
) -> int:
    """仅迁移已有输出命名，不启动MuJoCo或继续生成新episode。"""
    resolved_dir = _resolve_path(output_run_dir)
    migrated = _migrate_legacy_episode_names(resolved_dir, cameras)
    metadata_path = resolved_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        metadata["episode_naming"] = EPISODE_NAMING
        generation_config = metadata.get("generation_config")
        if isinstance(generation_config, dict):
            generation_config = dict(generation_config)
            generation_config["episode_naming"] = EPISODE_NAMING
            metadata["generation_config"] = generation_config
            metadata["generation_fingerprint"] = _fingerprint(
                generation_config
            )
        if migrated:
            metadata["last_episode_naming_migration"] = {
                "migrated_directories": int(migrated),
                "completed_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        _record_metadata_progress(metadata, resolved_dir, cameras)
    return migrated


def generate_view_augmented_run(cfg: DictConfig) -> None:
    input_run_dir = _resolve_path(cfg.input_run_dir)
    output_run_dir = _resolve_path(cfg.output_run_dir)
    if input_run_dir == output_run_dir:
        raise ValueError("input_run_dir与output_run_dir不能相同。")
    source_metadata_path = input_run_dir / "metadata.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"输入run缺少metadata.json: {source_metadata_path}")
    source_metadata = _load_json(source_metadata_path)

    metadata_env_id = source_metadata.get("env_id")
    env_id = (
        str(metadata_env_id)
        if cfg.env_id is None
        else str(cfg.env_id)
    )
    if not env_id:
        raise ValueError("无法从配置或源metadata解析env_id。")
    if (
        cfg.env_id is not None
        and metadata_env_id is not None
        and str(metadata_env_id) != env_id
    ):
        raise ValueError(
            f"配置env_id={env_id!r}与源metadata={metadata_env_id!r}不一致。"
        )
    variants_value = float(cfg.variants_per_episode)
    if (
        not np.isfinite(variants_value)
        or not variants_value.is_integer()
        or variants_value < 0
    ):
        raise ValueError("variants_per_episode必须为非负整数。")
    if not bool(cfg.include_original) and int(cfg.variants_per_episode) == 0:
        raise ValueError("include_original=false时至少需要一个增强变体。")

    cameras = tuple(str(camera) for camera in cfg.cameras)
    if cameras != ("zed_cam_left", "zed_cam_right"):
        raise ValueError(
            "当前生成器固定只保存zed_cam_left和zed_cam_right，"
            f"当前为{cameras}。"
        )
    integer_fields = {
        "render_height": cfg.render_height,
        "render_width": cfg.render_width,
        "fps": cfg.fps,
        "video.gop": cfg.video.gop,
        "video.macro_block_size": cfg.video.macro_block_size,
    }
    for name, value in integer_fields.items():
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
            raise ValueError(f"{name}必须为正整数。")
    crf = float(cfg.video.crf)
    if not np.isfinite(crf) or not crf.is_integer() or not 0 <= crf <= 51:
        raise ValueError("video.crf必须是0到51之间的整数。")
    state_tolerance = float(cfg.replay_validation.max_state_abs_error)
    if not np.isfinite(state_tolerance) or state_tolerance < 0:
        raise ValueError("replay_validation.max_state_abs_error必须是有限非负数。")
    retry_limit_fields = (
        "retry_max_arm_joint_abs_error",
        "retry_max_gripper_abs_error",
        "retry_max_view_joint_abs_error",
    )
    for field_name in retry_limit_fields:
        value = float(cfg.replay_validation[field_name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"replay_validation.{field_name}必须是有限正数。"
            )
    if cfg.max_source_episodes is not None:
        max_sources = float(cfg.max_source_episodes)
        if (
            not np.isfinite(max_sources)
            or not max_sources.is_integer()
            or max_sources <= 0
        ):
            raise ValueError("max_source_episodes必须为正整数或null。")

    source_fps = source_metadata.get("fps")
    if source_fps is not None and int(source_fps) != int(cfg.fps):
        raise ValueError(
            f"配置fps={int(cfg.fps)}与源metadata fps={int(source_fps)}不一致；"
            "固定轨迹不做时间重采样，不能只修改视频fps。"
        )

    source_indices = (
        None
        if cfg.source_episode_indices is None
        else [int(value) for value in cfg.source_episode_indices]
    )
    sources = _load_sources(
        input_run_dir,
        source_indices,
        (
            None
            if cfg.max_source_episodes is None
            else int(cfg.max_source_episodes)
        ),
    )
    for source in sources:
        episode_fps = source.info.get("fps")
        if episode_fps is not None and int(episode_fps) != int(cfg.fps):
            raise ValueError(
                f"source_episode={source.episode_number}的fps={episode_fps}"
                f"与配置fps={int(cfg.fps)}不一致。"
            )
    semantic_config = _semantic_config(cfg, input_run_dir, env_id)
    generation_fingerprint = _fingerprint(semantic_config)
    migrated_names = 0
    if bool(cfg.resume):
        migrated_names = _migrate_legacy_episode_names(
            output_run_dir,
            cameras,
        )
        _upgrade_metadata_episode_naming(
            output_run_dir,
            semantic_config,
            generation_fingerprint,
        )
    metadata = _load_or_create_metadata(
        output_run_dir=output_run_dir,
        input_run_dir=input_run_dir,
        source_metadata=source_metadata,
        semantic_config=semantic_config,
        fingerprint=generation_fingerprint,
        cameras=cameras,
        cfg=cfg,
    )
    if migrated_names:
        metadata["last_episode_naming_migration"] = {
            "migrated_directories": int(migrated_names),
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    _update_source_manifest(metadata, sources)
    _record_metadata_progress(metadata, output_run_dir, cameras)

    logging.info(
        "开始View轨迹增强: env=%s, source=%s, output=%s, sources=%d, "
        "variants=%d, include_original=%s, fingerprint=%s",
        env_id,
        input_run_dir,
        output_run_dir,
        len(sources),
        int(cfg.variants_per_episode),
        bool(cfg.include_original),
        generation_fingerprint,
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
        for source_position, source in enumerate(sources):
            arrays_path = source.directory / "arrays.npz"
            try:
                source_arrays = _validate_source_arrays(
                    arrays_path,
                    required_model_body_names=required_model_body_names,
                )
                variants = _build_variants(
                    source=source,
                    source_position=source_position,
                    arrays=source_arrays,
                    joint_ranges=joint_ranges,
                    cfg=cfg,
                )
                _clear_failure(
                    metadata,
                    source.episode_number,
                    None,
                )
            except Exception as exc:
                failure = {
                    "source_episode": source.episode_number,
                    "variant_index": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                _record_failure(metadata, failure)
                _record_metadata_progress(metadata, output_run_dir, cameras)
                logging.exception(
                    "源episode=%06d准备失败。", source.episode_number
                )
                if not bool(cfg.continue_on_source_error):
                    raise
                continue

            source_failed = False
            source_validation_mode = "strict"
            for variant in variants:
                final_dir = _episode_dir(
                    output_run_dir,
                    source.episode_number,
                    variant.variant_index,
                )
                if _episode_is_complete(final_dir, cameras):
                    _validate_completed_episode_identity(
                        final_dir,
                        source,
                        variant,
                    )
                    completed_info = _load_json(final_dir / "info.json")
                    if (
                        completed_info.get("replay_validation_mode")
                        == "fallback"
                    ):
                        source_validation_mode = "fallback"
                    _clear_failure(
                        metadata,
                        source.episode_number,
                        variant.variant_index,
                    )
                    logging.info(
                        "跳过已完成: output=%06d source=%06d variant=%d",
                        variant.output_index,
                        source.episode_number,
                        variant.variant_index,
                    )
                    continue
                try:
                    try:
                        info = _generate_episode(
                            env_obj=env_obj,
                            source=source,
                            source_arrays=source_arrays,
                            variant=variant,
                            output_run_dir=output_run_dir,
                            cameras=cameras,
                            cfg=cfg,
                            replay_validation_mode=source_validation_mode,
                        )
                    except ReplayStateMismatchError as strict_error:
                        if (
                            source_validation_mode != "strict"
                            or not bool(
                                cfg.replay_validation.retry_on_mismatch
                            )
                        ):
                            raise
                        source_validation_mode = "fallback"
                        logging.warning(
                            "严格重放校验失败，使用分组回退阈值"
                            "从头重新生成: output=%06d source=%06d "
                            "variant=%d, strict_error=%s",
                            variant.output_index,
                            source.episode_number,
                            variant.variant_index,
                            strict_error,
                        )
                        info = _generate_episode(
                            env_obj=env_obj,
                            source=source,
                            source_arrays=source_arrays,
                            variant=variant,
                            output_run_dir=output_run_dir,
                            cameras=cameras,
                            cfg=cfg,
                            replay_validation_mode=source_validation_mode,
                        )
                    _clear_failure(
                        metadata,
                        source.episode_number,
                        variant.variant_index,
                    )
                    _record_metadata_progress(
                        metadata,
                        output_run_dir,
                        cameras,
                    )
                    logging.info(
                        "已保存output=%06d source=%06d variant=%d "
                        "offset=%s max_state_error=%.3g validation=%s "
                        "arm=%.3g gripper=%.3g view=%.3g",
                        variant.output_index,
                        source.episode_number,
                        variant.variant_index,
                        np.array2string(
                            variant.actual_offset,
                            precision=4,
                            suppress_small=True,
                        ),
                        float(info["max_replay_state_abs_error"]),
                        info["replay_validation_mode"],
                        float(info["max_replay_arm_joint_abs_error"]),
                        float(info["max_replay_gripper_abs_error"]),
                        float(info["max_replay_view_joint_abs_error"]),
                    )
                except Exception as exc:
                    failure = {
                        "source_episode": source.episode_number,
                        "output_episode": variant.output_index,
                        "variant_index": variant.variant_index,
                        "error": f"{type(exc).__name__}: {exc}",
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    _record_failure(metadata, failure)
                    _record_metadata_progress(
                        metadata,
                        output_run_dir,
                        cameras,
                    )
                    logging.exception(
                        "生成失败: output=%06d source=%06d variant=%d",
                        variant.output_index,
                        source.episode_number,
                        variant.variant_index,
                    )
                    source_failed = True
                    if not bool(cfg.continue_on_source_error):
                        raise
                    break

            if source_failed:
                logging.warning(
                    "跳过源episode=%06d的剩余变体。", source.episode_number
                )
    finally:
        env_obj.close()

    _record_metadata_progress(metadata, output_run_dir, cameras)
    logging.info(
        "生成完成: saved=%d, original=%d, augmented=%d, failures=%d, output=%s",
        metadata["saved_episodes"],
        metadata["original_episodes"],
        metadata["augmented_episodes"],
        len(metadata.get("failures", [])),
        output_run_dir,
    )


@hydra.main(
    version_base="1.2",
    config_path="../configs/data_collect",
    config_name="view_joint_trajectory_augmentation",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _configure_mujoco_runtime(cfg)
    generate_view_augmented_run(cfg)


if __name__ == "__main__":
    main()
