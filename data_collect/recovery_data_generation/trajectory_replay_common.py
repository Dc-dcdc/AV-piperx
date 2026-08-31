"""恢复轨迹生成与数据审计共用的 Quest 轨迹重放工具。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from omegaconf import DictConfig


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VIEW_SLICE = slice(14, 20)
ACTION_DIM = 20
VIEW_DIM = 6
ARM_JOINT_INDICES = np.asarray(
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    dtype=np.int64,
)
GRIPPER_INDICES = np.asarray([6, 13], dtype=np.int64)
VIEW_JOINT_INDICES = np.arange(14, 20, dtype=np.int64)
TRAJECTORY_ALIGNMENT_MODES = ("moving_expert", "static_anchor_wait")
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


@dataclass(frozen=True)
class SourceEpisode:
    source_index: int
    episode_number: int
    directory: Path
    info: dict[str, Any]


def resolve_trajectory_alignment_mode(cfg: DictConfig) -> str:
    """读取并校验恢复阶段的专家时间轴模式。"""

    mode = str(cfg.recovery.get("trajectory_alignment_mode", "moving_expert"))
    if mode not in TRAJECTORY_ALIGNMENT_MODES:
        raise ValueError(
            "recovery.trajectory_alignment_mode 必须是 moving_expert 或 "
            f"static_anchor_wait，当前为{mode!r}。"
        )
    return mode


def resolve_recovery_timeline_step(
    mode: str,
    *,
    recovery_anchor_frame: int,
    source_frame: int,
) -> tuple[int, int]:
    """返回本恢复步的专家参考帧和下一恢复步的源帧。"""

    if mode == "moving_expert":
        return int(source_frame), int(source_frame) + 1
    if mode == "static_anchor_wait":
        return int(recovery_anchor_frame), int(source_frame)
    raise ValueError(f"未知轨迹对齐模式: {mode!r}。")


def resolve_recovery_base_action(
    mode: str,
    *,
    expert_action: np.ndarray,
    expert_state: np.ndarray,
) -> np.ndarray:
    """选择恢复动作的基准目标；静态模式用锚点状态实现位置保持。"""

    if mode == "moving_expert":
        base_action = np.asarray(expert_action, dtype=np.float64)
    elif mode == "static_anchor_wait":
        # 位置控制下，锚点处的专家 action 可能仍指向未来状态，重复执行会
        # 继续推动关节；因此静态等待必须以锚点 observation_state 为目标。
        base_action = np.asarray(expert_state, dtype=np.float64)
    else:
        raise ValueError(f"未知轨迹对齐模式: {mode!r}。")
    if base_action.shape != (ACTION_DIM,) or not np.isfinite(base_action).all():
        raise ValueError("恢复基准动作必须是20维有限数组。")
    return base_action


def build_static_anchor_reference_state(
    *,
    expert_state: np.ndarray,
    actual_state: np.ndarray,
    perturbed_indices: slice | np.ndarray,
) -> np.ndarray:
    """构造静态恢复参考：受扰角色回专家锚点，其余角色保持实际锚点。"""

    expert = np.asarray(expert_state, dtype=np.float64)
    actual = np.asarray(actual_state, dtype=np.float64)
    if (
        expert.shape != (ACTION_DIM,)
        or actual.shape != (ACTION_DIM,)
        or not np.isfinite(expert).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("专家锚点状态和实际锚点状态必须是20维有限数组。")
    reference = actual.copy()
    reference[perturbed_indices] = expert[perturbed_indices]
    return reference


def recovery_suffix_start_frame(
    mode: str,
    *,
    recovery_anchor_frame: int,
    source_frame: int,
) -> int:
    """解析恢复完成后继续执行专家后缀的首帧。"""

    if mode == "moving_expert":
        return int(source_frame)
    if mode == "static_anchor_wait":
        # 静态等待发送的是锚点 state 保持目标，并未执行锚点 expert action；
        # 因此后缀必须从锚点 action 本身恢复，不能跳到下一帧。
        return int(recovery_anchor_frame)
    raise ValueError(f"未知轨迹对齐模式: {mode!r}。")


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
            "joint_action必须与observation_state同形，"
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
    body_quat = np.asarray(arrays["initial_model_body_quat"], dtype=np.float64)
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


def _restore_initial_state(env_obj, arrays: dict[str, np.ndarray]) -> None:
    physics = env_obj._physics
    env_obj.reset(seed=0)
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


def _apply_original_action(env_obj, action: np.ndarray) -> None:
    """只推进物理，不额外渲染观测或计算任务奖励。"""
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
            except BaseException as exc:
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
            if exc_type is None:
                raise
            logging.exception("关闭双目视频writer时发生次级异常。")
        return False


def _output_episode_name(source_episode: int, variant_index: int) -> str:
    source_episode = int(source_episode)
    variant_index = int(variant_index)
    if source_episode < 0:
        raise ValueError("source_episode必须为非负整数。")
    if variant_index == -1:
        return f"episode_{source_episode:06d}"
    if variant_index < 0:
        raise ValueError("恢复variant_index必须为非负整数。")
    return f"episode_{source_episode:06d}_aug_{variant_index:02d}"


def _episode_is_complete(directory: Path, cameras: tuple[str, ...]) -> bool:
    required = [directory / "arrays.npz", directory / "info.json"]
    required.extend(directory / "videos" / f"{camera}.mp4" for camera in cameras)
    return directory.is_dir() and all(
        path.is_file() and path.stat().st_size > 0 for path in required
    )


def _fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
