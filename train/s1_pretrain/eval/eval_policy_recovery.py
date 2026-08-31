import os
DETERMINISTIC_EVAL = os.environ.get("DPPO_EVAL_DETERMINISTIC", "0").lower() in {"1", "true", "yes"}
if DETERMINISTIC_EVAL:
    # 确定性模式会牺牲速度；只在需要逐 bit 复现实验时开启。
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
# 独立评估优先使用 EGL 离屏渲染，避免 GLFW/X11 窗口后端带来的额外波动。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
import torch
import csv
import gc
import hashlib
import logging
import numpy as np
import imageio
import json
import re
import shutil
from pathlib import Path
from contextlib import nullcontext
from functools import partial
import gymnasium as gym
import yaml
from tqdm import tqdm
import random
import time
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

if __package__:
    from .coupling_ablation import (
        apply_coupling_ablation_overrides,
        coupling_ablation_tag,
    )
    from .output_corrector_ablation import (
        apply_output_corrector_ablation_overrides,
        output_corrector_ablation_tag,
    )
    from .router_ablation import (
        apply_router_ablation_override,
        router_ablation_tag,
    )
    from .vector_info import as_bool_array as _as_bool_array
    from .vector_info import extract_info_bool as _extract_info_bool
else:
    from coupling_ablation import (
        apply_coupling_ablation_overrides,
        coupling_ablation_tag,
    )
    from output_corrector_ablation import (
        apply_output_corrector_ablation_overrides,
        output_corrector_ablation_tag,
    )
    from router_ablation import (
        apply_router_ablation_override,
        router_ablation_tag,
    )
    from vector_info import as_bool_array as _as_bool_array
    from vector_info import extract_info_bool as _extract_info_bool

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ROOT_PATH = Path(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import env.task.sim_envs


def seed_runtime(seed: int):
    """重置一次运行时 RNG；评估时每个 episode 都会调用，避免上一局步数污染下一局扩散噪声。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def seed_env_spaces(env, seed: int):
    """Gym space 自己也可能持有 RNG，显式对齐到当前 episode seed。"""
    for space_name in ("action_space", "observation_space"):
        space = getattr(env, space_name, None)
        if hasattr(space, "seed"):
            space.seed(seed)

def configure_torch_runtime(deterministic: bool):
    if deterministic:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.backends.mkldnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        return

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(False)


def prepare_policy_observation(raw_obs: dict, expected_keys: set[str], device) -> dict[str, torch.Tensor]:
    """只转换 policy 真正需要的观测键；兼容单环境和 gym.vector 的 batch 观测。"""
    batch = {}

    if "observation.state" in expected_keys:
        state = np.asarray(raw_obs["agent_pos"], dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if not state.flags.c_contiguous:
            state = np.ascontiguousarray(state)
        batch["observation.state"] = torch.from_numpy(state).to(device, non_blocking=True)

    if "observation.environment_state" in expected_keys and "environment_state" in raw_obs:
        env_state = np.asarray(raw_obs["environment_state"], dtype=np.float32)
        if env_state.ndim == 1:
            env_state = env_state[None, :]
        if not env_state.flags.c_contiguous:
            env_state = np.ascontiguousarray(env_state)
        batch["observation.environment_state"] = torch.from_numpy(env_state).to(device, non_blocking=True)

    pixels = raw_obs.get("pixels", {})
    for key in expected_keys:
        if not key.startswith("observation.images."):
            continue
        camera = key.removeprefix("observation.images.")
        if camera not in pixels:
            raise KeyError(f"环境观测中缺少策略需要的相机: {camera}")

        image = np.asarray(pixels[camera])
        if image.ndim == 3:
            image = image[None, ...]
        if image.ndim != 4:
            raise ValueError(f"相机 {camera} 的观测维度异常: {image.shape}")
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        image_tensor = torch.from_numpy(image).to(device, non_blocking=True)
        image_tensor = image_tensor.permute(0, 3, 1, 2).contiguous()
        batch[key] = image_tensor.to(dtype=torch.float32).div_(255.0)

    return batch


def is_vector_env(env) -> bool:
    return int(getattr(env, "num_envs", 1)) > 1


def get_last_router_decision(policy):
    diffusion = getattr(policy, "diffusion", None)
    gate = getattr(diffusion, "last_router_gate", None)
    probability = getattr(diffusion, "last_router_probability", None)
    if gate is None or probability is None:
        return None
    return (
        gate.detach().cpu().numpy().astype(bool, copy=False),
        probability.detach().float().cpu().numpy(),
    )


def make_single_eval_env(env_id: str, cameras: list[str], episode_length: int):
    import env as _registered_env  # noqa: F401 - 子进程里触发 Gym 注册

    env_obj = gym.make(
        id=env_id,
        cameras=cameras,
        episode_length=episode_length,
    )
    return env_obj.unwrapped


def make_eval_env(env_id: str, cameras: list[str], eval_cfg):
    batch_size = int(getattr(eval_cfg, "batch_size", getattr(eval_cfg, "num_envs", 1)))
    n_episodes = int(getattr(eval_cfg, "n_episodes", 1))
    batch_size = max(1, min(batch_size, max(1, n_episodes)))
    episode_length = int(getattr(eval_cfg, "max_steps", 300))

    if batch_size <= 1:
        logging.info("评估使用单环境模式。")
        return make_single_eval_env(env_id, cameras, episode_length)

    env_fns = [partial(make_single_eval_env, env_id, cameras, episode_length) for _ in range(batch_size)]
    use_async_envs = bool(getattr(eval_cfg, "use_async_envs", True))
    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    vector_kwargs = {}
    if use_async_envs:
        vector_kwargs.update(shared_memory=True, context="spawn")

    try:
        eval_env = vector_cls(env_fns, **vector_kwargs)
    except TypeError:
        # 兼容较旧 gymnasium：部分版本不支持 context/shared_memory 参数。
        eval_env = vector_cls(env_fns)

    mode_name = "AsyncVectorEnv" if use_async_envs else "SyncVectorEnv"
    logging.info(f"评估使用多环境模式: {mode_name}, num_envs={batch_size}")
    return eval_env


def describe_eval_env(env_id: str, eval_env) -> str:
    if is_vector_env(eval_env):
        return (
            f"{env_id} -> {eval_env.__class__.__module__}.{eval_env.__class__.__name__}"
            f"[num_envs={int(eval_env.num_envs)}]"
        )
    return f"{env_id} -> {eval_env.unwrapped.__class__.__module__}.{eval_env.unwrapped.__class__.__name__}"


def _as_float_array(value, n_envs: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(n_envs, default, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == ():
        return np.full(n_envs, float(arr), dtype=np.float32)
    arr = arr.reshape(-1)
    if arr.shape[0] < n_envs:
        padded = np.full(n_envs, default, dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded
    return arr[:n_envs]


def _render_vector_frames(env, camera: str, n_envs: int):
    if hasattr(env, "call"):
        frames = env.call("render", [camera])
        return list(frames)
    return [env.envs[idx].unwrapped.render([camera]) for idx in range(n_envs)]


ACTION_DIM = 20
LEFT_ARM_INDICES = np.arange(0, 6, dtype=np.int64)
RIGHT_ARM_INDICES = np.arange(7, 13, dtype=np.int64)
VIEW_INDICES = np.arange(14, 20, dtype=np.int64)
PIPER_CTRL_LOW = np.asarray(
    [-2.6179938, 0.0, -2.9670597, -1.553343, -1.553343, -2.0943951],
    dtype=np.float64,
)
PIPER_CTRL_HIGH = np.asarray(
    [2.6179938, 3.1415926, 0.0, 1.553343, 1.553343, 2.0943951],
    dtype=np.float64,
)
DEFAULT_RECOVERY_STD_RAD = (0.016, 0.016, 0.016, 0.026, 0.026, 0.032)
DEFAULT_RECOVERY_MAX_ABS_RAD = (0.040, 0.040, 0.040, 0.065, 0.065, 0.080)
RECOVERY_MODE_CODES = {"clean": 0, "arm": 1, "view": 2, "mixed": 3, "arm_view_seq": 4}


def _six_finite_values(value, name: str, *, positive: bool) -> tuple[float, ...]:
    values = tuple(float(item) for item in value)
    if len(values) != 6 or not np.isfinite(values).all():
        raise ValueError(f"{name}必须包含6个有限数值，当前为{values!r}。")
    if positive and any(item <= 0.0 for item in values):
        raise ValueError(f"{name}的所有元素必须大于0。")
    return values


@dataclass(frozen=True)
class RecoveryPerturbationSpec:
    """独立鲁棒性评估使用的物理关节目标扰动配置。"""

    mode: str
    seed: int
    severity: float
    setup_steps: int
    hold_steps: int
    trigger_step_min: int
    trigger_step_max: int
    arm_selection: str
    motion_window_steps: int
    fps: float
    mixed_equal_energy: bool
    std_rad: tuple[float, ...]
    max_abs_rad: tuple[float, ...]
    min_normalized_l2: float
    joint_limit_margin_rad: float
    recovery_error_rad: float
    recovery_stable_steps: int
    max_sampling_attempts: int
    trigger_description: str
    align_to_chunk: bool

    @classmethod
    def from_eval_cfg(
        cls,
        cfg_eval,
        *,
        max_steps: int | None = None,
        fps: float | None = None,
    ) -> "RecoveryPerturbationSpec":
        max_steps = int(max_steps if max_steps is not None else getattr(cfg_eval, "max_steps", 300))
        fps = float(fps if fps is not None else getattr(cfg_eval, "fps", 25))
        if max_steps < 2:
            raise ValueError("max_steps必须至少为2，才能建立并释放扰动。")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps必须为有限正数。")

        mode = str(getattr(cfg_eval, "recovery_perturbation", "clean")).strip().lower()
        if mode not in RECOVERY_MODE_CODES:
            raise ValueError("recovery_perturbation只能是clean/arm/view/mixed。")
        severity = float(getattr(cfg_eval, "recovery_severity", 1.0))
        if not np.isfinite(severity) or severity <= 0.0:
            raise ValueError("recovery_severity必须为有限正数。")
        setup_steps = int(getattr(cfg_eval, "recovery_setup_steps", 15))
        if setup_steps < 1:
            raise ValueError("recovery_setup_steps必须大于等于1。")
        hold_steps = int(getattr(cfg_eval, "recovery_hold_steps", 0))
        if hold_steps < 0:
            raise ValueError("recovery_hold_steps必须大于等于0。")

        exclude_initial = int(getattr(cfg_eval, "recovery_exclude_initial_steps", 16))
        if exclude_initial < 0:
            raise ValueError("recovery_exclude_initial_steps不能为负数。")
        latest_trigger = max_steps - setup_steps - hold_steps - 1
        if latest_trigger < exclude_initial:
            raise ValueError(
                "max_steps不足以容纳开头保护、扰动建立/保持和释放后观测: "
                f"max_steps={max_steps}, exclude={exclude_initial}, setup={setup_steps}, hold={hold_steps}。"
            )

        explicit_steps = getattr(cfg_eval, "recovery_trigger_step_range", None)
        if explicit_steps is not None:
            values = tuple(int(item) for item in explicit_steps)
            if len(values) != 2:
                raise ValueError("recovery_trigger_step_range必须为[start, end]闭区间。")
            trigger_min, trigger_max = values
            trigger_description = f"steps[{trigger_min},{trigger_max}]"
        else:
            normalized = tuple(
                float(item)
                for item in getattr(
                    cfg_eval,
                    "recovery_trigger_normalized_range",
                    (0.35, 0.47),
                )
            )
            if (
                len(normalized) != 2
                or not np.isfinite(normalized).all()
                or not 0.0 <= normalized[0] < normalized[1] <= 1.0
            ):
                raise ValueError(
                    "recovery_trigger_normalized_range必须满足0<=start<end<=1。"
                )
            reference_value = getattr(cfg_eval, "recovery_trigger_reference_steps", None)
            reference_steps = max_steps if reference_value is None else int(reference_value)
            if reference_steps < 1:
                raise ValueError("recovery_trigger_reference_steps必须为正整数或None。")
            trigger_min = int(np.ceil(normalized[0] * reference_steps))
            trigger_max = int(np.floor(normalized[1] * reference_steps))
            trigger_description = (
                f"normalized[{normalized[0]:g},{normalized[1]:g}]@{reference_steps}"
            )

        trigger_min = max(trigger_min, exclude_initial)
        trigger_max = min(trigger_max, latest_trigger)
        if trigger_min > trigger_max:
            raise ValueError(
                "扰动触发区间与有效评估窗口没有交集: "
                f"trigger=[{trigger_min},{trigger_max}], latest={latest_trigger}。"
            )

        arm_selection = str(
            getattr(cfg_eval, "recovery_arm_selection", "local_motion")
        ).strip().lower()
        if arm_selection not in {"local_motion", "alternate", "random", "left", "right"}:
            raise ValueError(
                "recovery_arm_selection只能是local_motion/alternate/random/left/right。"
            )
        motion_window_steps = int(getattr(cfg_eval, "recovery_motion_window_steps", 16))
        if motion_window_steps < 2:
            raise ValueError("recovery_motion_window_steps必须至少为2。")

        std_rad = _six_finite_values(
            getattr(cfg_eval, "recovery_std_rad", DEFAULT_RECOVERY_STD_RAD),
            "recovery_std_rad",
            positive=True,
        )
        max_abs_rad = _six_finite_values(
            getattr(cfg_eval, "recovery_max_abs_rad", DEFAULT_RECOVERY_MAX_ABS_RAD),
            "recovery_max_abs_rad",
            positive=True,
        )
        min_normalized_l2 = float(getattr(cfg_eval, "recovery_min_normalized_l2", 0.5))
        if not np.isfinite(min_normalized_l2) or not 0.0 <= min_normalized_l2 <= np.sqrt(6.0):
            raise ValueError("recovery_min_normalized_l2必须位于[0,sqrt(6)]。")
        joint_limit_margin_rad = float(
            getattr(cfg_eval, "recovery_joint_limit_margin_rad", 0.005)
        )
        if (
            not np.isfinite(joint_limit_margin_rad)
            or joint_limit_margin_rad < 0.0
            or np.any(PIPER_CTRL_LOW + joint_limit_margin_rad >= PIPER_CTRL_HIGH - joint_limit_margin_rad)
        ):
            raise ValueError("recovery_joint_limit_margin_rad非法。")
        recovery_error_rad = float(getattr(cfg_eval, "recovery_success_max_abs_error_rad", 0.004))
        if not np.isfinite(recovery_error_rad) or recovery_error_rad <= 0.0:
            raise ValueError("recovery_success_max_abs_error_rad必须为有限正数。")
        recovery_stable_steps = int(getattr(cfg_eval, "recovery_success_stable_steps", 3))
        if recovery_stable_steps < 1:
            raise ValueError("recovery_success_stable_steps必须大于等于1。")
        max_sampling_attempts = int(getattr(cfg_eval, "recovery_max_sampling_attempts", 1000))
        if max_sampling_attempts < 1:
            raise ValueError("recovery_max_sampling_attempts必须大于等于1。")
        align_to_chunk = bool(getattr(cfg_eval, "recovery_align_to_chunk", True))

        return cls(
            mode=mode,
            seed=int(getattr(cfg_eval, "recovery_seed", 20260819)),
            severity=severity,
            setup_steps=setup_steps,
            hold_steps=hold_steps,
            trigger_step_min=trigger_min,
            trigger_step_max=trigger_max,
            arm_selection=arm_selection,
            motion_window_steps=motion_window_steps,
            fps=fps,
            mixed_equal_energy=bool(getattr(cfg_eval, "recovery_mixed_equal_energy", True)),
            std_rad=std_rad,
            max_abs_rad=max_abs_rad,
            min_normalized_l2=min_normalized_l2,
            joint_limit_margin_rad=joint_limit_margin_rad,
            recovery_error_rad=recovery_error_rad,
            recovery_stable_steps=recovery_stable_steps,
            max_sampling_attempts=max_sampling_attempts,
            trigger_description=trigger_description,
            align_to_chunk=align_to_chunk,
        )

    def tag(self) -> str:
        config_id = hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:8]
        if self.mode == "clean":
            return f"recovery-clean-c{config_id}"
        equal_energy = "-eq" if self.mode == "mixed" and self.mixed_equal_energy else ""
        hold_part = f"-hold{self.hold_steps}" if self.hold_steps else ""
        return (
            f"recovery-{self.mode}{equal_energy}-s{self.severity:g}"
            f"-setup{self.setup_steps}{hold_part}-t{self.trigger_step_min}-{self.trigger_step_max}"
            f"-c{config_id}"
        )

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "severity": self.severity,
            "setup_steps": self.setup_steps,
            "hold_steps": self.hold_steps,
            "trigger_step_range": [self.trigger_step_min, self.trigger_step_max],
            "trigger_description": self.trigger_description,
            "arm_selection": self.arm_selection,
            "motion_window_steps": self.motion_window_steps,
            "fps": self.fps,
            "mixed_equal_energy": self.mixed_equal_energy,
            "std_rad": list(self.std_rad),
            "max_abs_rad": list(self.max_abs_rad),
            "min_normalized_l2": self.min_normalized_l2,
            "joint_limit_margin_rad": self.joint_limit_margin_rad,
            "recovery_success_max_abs_error_rad": self.recovery_error_rad,
            "recovery_success_stable_steps": self.recovery_stable_steps,
            "max_sampling_attempts": self.max_sampling_attempts,
            "align_to_chunk": self.align_to_chunk,
        }


def _quintic_smoothstep(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


_ANNOTATION_COLORS = {
    "arm:perturb": (220, 40, 40),     # 红：arm 扰动建立
    "view:perturb": (255, 140, 0),    # 橙：view 扰动建立
    "mixed:perturb": (200, 40, 200),  # 紫：mixed 扰动建立
    "arm:recover": (40, 140, 220),    # 蓝：arm 恢复
    "view:recover": (40, 180, 90),    # 绿：view 恢复
    "mixed:recover": (0, 170, 170),   # 青：mixed 恢复
}


def _annotate_frame(frame: np.ndarray, label: str | None) -> np.ndarray:
    """给帧叠加扰动状态边框（按角色+状态着色）；label 为 None 时原样返回。"""
    if label is None:
        return frame
    color = _ANNOTATION_COLORS.get(label)
    if color is None:
        return frame
    out = np.asarray(frame).copy()
    h, w = out.shape[:2]
    border = max(3, h // 90)
    out[:border] = color
    out[-border:] = color
    out[:, :border] = color
    out[:, -border:] = color
    return out


_MODE_ROLES: dict[str, list[str]] = {
    "clean": [],
    "arm": ["arm"],
    "view": ["view"],
    "mixed": ["mixed"],
    "arm_view_seq": ["arm", "view"],
}


class _RecoveryEvent:
    """单次角色扰动事件：偏移采样、平滑建立、释放后控制恢复统计。"""

    def __init__(
        self,
        role: str,
        trigger_step: int,
        release_step: int,
        spec: RecoveryPerturbationSpec,
        rng,
        state_history: list,
        episode: int,
    ):
        self.role = role  # "arm" | "view" | "mixed"
        self.trigger_step = int(trigger_step)
        self.release_step = int(release_step)
        self.spec = spec
        self.rng = rng
        self.state_history = state_history
        self.episode = int(episode)
        self.triggered = False
        self.arm_side: str | None = None
        self.arm_motion_rms = {"left": None, "right": None}
        self.arm_offset = np.zeros(6, dtype=np.float64)
        self.view_offset = np.zeros(6, dtype=np.float64)
        self.trigger_state: np.ndarray | None = None
        self.release_state: np.ndarray | None = None
        self.realized_arm_delta: np.ndarray | None = None
        self.realized_view_delta: np.ndarray | None = None
        self.max_alpha = 0.0
        self.clipped_values = 0
        self.perturbed_action_values = 0
        self.recovery_stable_count = 0
        self.control_recovered = False
        self.control_recovery_step: int | None = None
        self.max_post_release_control_error_rad: float | None = None

    def _select_arm_side(self) -> str:
        mode = self.spec.arm_selection
        if mode in {"left", "right"}:
            return mode
        if mode == "alternate":
            return "left" if self.episode % 2 == 0 else "right"
        if mode == "random":
            return "left" if int(self.rng.integers(0, 2)) == 0 else "right"

        history = np.asarray(
            self.state_history[-(self.spec.motion_window_steps + 1) :],
            dtype=np.float64,
        )
        if history.shape[0] < 2:
            return "left" if self.episode % 2 == 0 else "right"
        velocity = np.diff(history, axis=0) * self.spec.fps
        left_rms = float(np.sqrt(np.mean(np.square(velocity[:, LEFT_ARM_INDICES]))))
        right_rms = float(np.sqrt(np.mean(np.square(velocity[:, RIGHT_ARM_INDICES]))))
        self.arm_motion_rms = {"left": left_rms, "right": right_rms}
        if np.isclose(left_rms, right_rms, rtol=0.0, atol=1e-12):
            return "left" if self.episode % 2 == 0 else "right"
        return "left" if left_rms > right_rms else "right"

    def _sample_offset(self, current_state: np.ndarray, scale: float) -> np.ndarray:
        std = np.asarray(self.spec.std_rad, dtype=np.float64) * scale
        max_abs = np.asarray(self.spec.max_abs_rad, dtype=np.float64) * scale
        lower = np.maximum(
            -max_abs,
            PIPER_CTRL_LOW + self.spec.joint_limit_margin_rad - current_state,
        )
        upper = np.minimum(
            max_abs,
            PIPER_CTRL_HIGH - self.spec.joint_limit_margin_rad - current_state,
        )
        if np.any(lower > upper):
            raise RuntimeError(
                "当前关节状态位于带安全余量的控制范围之外，无法建立恢复扰动。"
            )
        for _ in range(self.spec.max_sampling_attempts):
            offset = self.rng.normal(0.0, std, size=6)
            if np.any(offset < lower) or np.any(offset > upper):
                continue
            if float(np.linalg.norm(offset / max_abs)) < self.spec.min_normalized_l2:
                continue
            return offset
        raise RuntimeError(
            "在recovery_max_sampling_attempts内未采到满足限位和最小强度的扰动。"
        )

    def _initialize_offsets(self, state: np.ndarray) -> None:
        self.triggered = True
        self.trigger_state = state.copy()
        role_scale = self.spec.severity
        if self.role == "mixed" and self.spec.mixed_equal_energy:
            role_scale /= np.sqrt(2.0)
        if self.role in {"arm", "mixed"}:
            self.arm_side = self._select_arm_side()
            arm_indices = LEFT_ARM_INDICES if self.arm_side == "left" else RIGHT_ARM_INDICES
            self.arm_offset = self._sample_offset(state[arm_indices], role_scale)
        if self.role in {"view", "mixed"}:
            self.view_offset = self._sample_offset(state[VIEW_INDICES], role_scale)

    def _target_indices(self) -> np.ndarray:
        indices: list[int] = []
        if self.role in {"arm", "mixed"}:
            arm_indices = LEFT_ARM_INDICES if self.arm_side == "left" else RIGHT_ARM_INDICES
            indices.extend(int(item) for item in arm_indices)
        if self.role in {"view", "mixed"}:
            indices.extend(int(item) for item in VIEW_INDICES)
        return np.asarray(indices, dtype=np.int64)

    def _capture_release_state(self, state: np.ndarray) -> None:
        if self.release_state is not None or self.trigger_state is None:
            return
        self.release_state = state.copy()
        if self.role in {"arm", "mixed"}:
            arm_indices = LEFT_ARM_INDICES if self.arm_side == "left" else RIGHT_ARM_INDICES
            self.realized_arm_delta = state[arm_indices] - self.trigger_state[arm_indices]
        if self.role in {"view", "mixed"}:
            self.realized_view_delta = state[VIEW_INDICES] - self.trigger_state[VIEW_INDICES]

    def apply(self, step: int, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        if step == self.trigger_step:
            self._initialize_offsets(state)

        disturbed = action.copy()
        if self.triggered and self.trigger_step <= step < self.release_step:
            setup = int(self.spec.setup_steps)
            if step < self.trigger_step + setup:
                # 建立阶段：五次 smoothstep 从 0 平滑爬升到完整偏移
                progress = (step - self.trigger_step + 1) / float(setup)
                alpha = _quintic_smoothstep(progress)
            else:
                # 保持阶段：维持完整偏移，测持续扰动下的纠正能力
                alpha = 1.0
            self.max_alpha = max(self.max_alpha, alpha)
            if self.role in {"arm", "mixed"}:
                arm_indices = LEFT_ARM_INDICES if self.arm_side == "left" else RIGHT_ARM_INDICES
                disturbed[arm_indices] += alpha * self.arm_offset
            if self.role in {"view", "mixed"}:
                disturbed[VIEW_INDICES] += alpha * self.view_offset

            target_indices = self._target_indices()
            before_clip = disturbed[target_indices].copy()
            tiled_low = np.tile(PIPER_CTRL_LOW, len(target_indices) // 6)
            tiled_high = np.tile(PIPER_CTRL_HIGH, len(target_indices) // 6)
            disturbed[target_indices] = np.clip(
                before_clip,
                tiled_low + self.spec.joint_limit_margin_rad,
                tiled_high - self.spec.joint_limit_margin_rad,
            )
            self.clipped_values += int(
                np.count_nonzero(~np.isclose(before_clip, disturbed[target_indices]))
            )
            self.perturbed_action_values += int(len(target_indices))
        elif self.triggered and step >= self.release_step:
            self._capture_release_state(state)
            if not self.control_recovered:
                indices = self._target_indices()
                control_low = np.tile(PIPER_CTRL_LOW, len(indices) // 6)
                control_high = np.tile(PIPER_CTRL_HIGH, len(indices) // 6)
                target = np.clip(action[indices], control_low, control_high)
                error = float(np.max(np.abs(state[indices] - target)))
                if self.max_post_release_control_error_rad is None:
                    self.max_post_release_control_error_rad = error
                else:
                    self.max_post_release_control_error_rad = max(
                        self.max_post_release_control_error_rad,
                        error,
                    )
                if error <= self.spec.recovery_error_rad:
                    self.recovery_stable_count += 1
                else:
                    self.recovery_stable_count = 0
                if self.recovery_stable_count >= self.spec.recovery_stable_steps:
                    self.control_recovered = True
                    self.control_recovery_step = int(step)
        return disturbed

    def result(self, episode_steps: int) -> dict:
        recovery_steps = (
            self.control_recovery_step - self.release_step + 1
            if self.control_recovery_step is not None
            else None
        )
        clipping_rate = (
            self.clipped_values / self.perturbed_action_values
            if self.perturbed_action_values
            else 0.0
        )
        return {
            "role": self.role,
            "trigger_step": self.trigger_step,
            "triggered": bool(self.triggered),
            "release_step": self.release_step if self.triggered else None,
            "arm_side": self.arm_side,
            "arm_motion_rms_rad_s": self.arm_motion_rms,
            "arm_offset_rad": self.arm_offset.tolist(),
            "view_offset_rad": self.view_offset.tolist(),
            "realized_arm_delta_rad": (
                self.realized_arm_delta.tolist() if self.realized_arm_delta is not None else None
            ),
            "realized_view_delta_rad": (
                self.realized_view_delta.tolist() if self.realized_view_delta is not None else None
            ),
            "max_alpha": self.max_alpha,
            "clipped_values": self.clipped_values,
            "perturbed_action_values": self.perturbed_action_values,
            "clipping_rate": clipping_rate,
            "control_recovered": bool(self.control_recovered),
            "control_recovery_step": self.control_recovery_step,
            "control_recovery_steps_after_release": recovery_steps,
            "max_post_release_control_error_rad": self.max_post_release_control_error_rad,
            "ended_before_trigger": bool(episode_steps <= self.trigger_step),
        }


class EpisodeRecoveryPerturbation:
    """管理单个 episode 的一次或多次（顺序）扰动建立、释放与控制恢复统计。"""

    def __init__(
        self,
        spec: RecoveryPerturbationSpec,
        episode: int,
        episode_seed: int,
        n_action_steps: int = 8,
    ):
        self.spec = spec
        self.episode = int(episode)
        self.episode_seed = int(episode_seed)
        seed_sequence = np.random.SeedSequence(
            [spec.seed, self.episode_seed, self.episode, RECOVERY_MODE_CODES[spec.mode]]
        )
        self.rng = np.random.default_rng(seed_sequence)
        self.state_history: list[np.ndarray] = []
        self.events: list[_RecoveryEvent] = self._build_events(int(n_action_steps))

    def _build_events(self, n_action_steps: int) -> list[_RecoveryEvent]:
        roles = _MODE_ROLES[self.spec.mode]
        if not roles:
            return []
        releases = self._schedule_releases(n_action_steps, len(roles))
        setup = int(self.spec.setup_steps)
        hold = int(self.spec.hold_steps)
        events = []
        for role, release in zip(roles, releases):
            trigger = max(0, int(release) - setup - hold)
            events.append(
                _RecoveryEvent(
                    role,
                    trigger,
                    int(release),
                    self.spec,
                    self.rng,
                    self.state_history,
                    self.episode,
                )
            )
        return events

    def _schedule_releases(self, n_action_steps: int, n_events: int) -> list[int]:
        """为 n_events 个顺序事件调度互不重叠的 release step。

        align_to_chunk=True 时每个 release 对齐到 chunk 边界（8 的整数倍），相邻事件
        至少间隔一个完整 chunk，使前一扰动有足够时间恢复后再触发下一个。
        """
        lo = int(self.spec.trigger_step_min)
        hi = int(self.spec.trigger_step_max)
        # 间隔需覆盖整个扰动时长（建立+保持），并给前一扰动留出恢复窗口，
        # 再触发下一个角色，避免两个恢复测量互相污染。
        gap = max(
            int(self.spec.setup_steps) + int(self.spec.hold_steps) + 1,
            2 * n_action_steps,
        )

        if self.spec.align_to_chunk and n_action_steps > 0:
            boundaries = [b for b in range(lo, hi + 1) if b % n_action_steps == 0]
            spaced = []
            for b in boundaries:
                if not spaced or b - spaced[-1] >= gap:
                    spaced.append(b)
            if len(spaced) >= n_events:
                start_idx = int(self.rng.integers(0, len(spaced) - n_events + 1))
                return spaced[start_idx : start_idx + n_events]

        # 回退：随机 release，保证最小间隔
        releases: list[int] = []
        for _ in range(1000):
            r = int(self.rng.integers(lo, hi + 1))
            if all(abs(r - existing) >= gap for existing in releases):
                releases.append(r)
            if len(releases) >= n_events:
                break
        releases.sort()
        if len(releases) < n_events:
            releases = [int(v) for v in np.linspace(lo, hi, n_events)]
        return releases[:n_events]

    def apply(self, step: int, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        state = np.asarray(state, dtype=np.float64)
        if action.shape != (ACTION_DIM,) or state.shape != (ACTION_DIM,):
            raise ValueError(
                f"恢复扰动要求action/state均为({ACTION_DIM},)，"
                f"当前为{action.shape}/{state.shape}。"
            )
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError("恢复扰动收到NaN/Inf action或state。")

        self.state_history.append(state.copy())
        if len(self.state_history) > self.spec.motion_window_steps + 1:
            self.state_history.pop(0)
        for event in self.events:
            action = event.apply(step, action, state)
        return action

    def result(self, episode_steps: int) -> dict:
        event_results = [event.result(episode_steps) for event in self.events]
        if not event_results:
            return {
                "perturbation_mode": self.spec.mode,
                "perturbation_severity": self.spec.severity,
                "perturbation_scheduled_step": None,
                "perturbation_triggered": False,
                "perturbation_release_step": None,
                "perturbation_arm_side": None,
                "perturbation_arm_motion_rms_rad_s": {"left": None, "right": None},
                "perturbation_arm_offset_rad": [0.0] * 6,
                "perturbation_view_offset_rad": [0.0] * 6,
                "perturbation_realized_arm_delta_rad": None,
                "perturbation_realized_view_delta_rad": None,
                "perturbation_max_alpha": 0.0,
                "perturbation_clipped_values": 0,
                "perturbation_clipping_rate": 0.0,
                "control_recovered": False,
                "control_recovery_step": None,
                "control_recovery_steps_after_release": None,
                "max_post_release_control_error_rad": None,
                "episode_ended_before_perturbation": False,
            }

        primary = event_results[0]
        total_clipped = sum(e["clipped_values"] for e in event_results)
        total_perturbed = sum(e["perturbed_action_values"] for e in event_results)
        recovery_steps_values = [
            e["control_recovery_steps_after_release"]
            for e in event_results
            if e["control_recovery_steps_after_release"] is not None
        ]
        max_error_values = [
            e["max_post_release_control_error_rad"]
            for e in event_results
            if e["max_post_release_control_error_rad"] is not None
        ]
        flat = {
            "perturbation_mode": self.spec.mode,
            "perturbation_severity": self.spec.severity,
            "perturbation_scheduled_step": primary["trigger_step"],
            "perturbation_triggered": all(e["triggered"] for e in event_results),
            "perturbation_release_step": primary["release_step"],
            "perturbation_arm_side": primary["arm_side"],
            "perturbation_arm_motion_rms_rad_s": primary["arm_motion_rms_rad_s"],
            "perturbation_arm_offset_rad": primary["arm_offset_rad"],
            "perturbation_view_offset_rad": primary["view_offset_rad"],
            "perturbation_realized_arm_delta_rad": primary["realized_arm_delta_rad"],
            "perturbation_realized_view_delta_rad": primary["realized_view_delta_rad"],
            "perturbation_max_alpha": max(e["max_alpha"] for e in event_results),
            "perturbation_clipped_values": total_clipped,
            "perturbation_clipping_rate": (
                total_clipped / total_perturbed if total_perturbed else 0.0
            ),
            "control_recovered": all(e["control_recovered"] for e in event_results),
            "control_recovery_step": (
                max(
                    e["control_recovery_step"]
                    for e in event_results
                    if e["control_recovery_step"] is not None
                )
                if any(e["control_recovery_step"] is not None for e in event_results)
                else None
            ),
            "control_recovery_steps_after_release": (
                max(recovery_steps_values) if recovery_steps_values else None
            ),
            "max_post_release_control_error_rad": (
                max(max_error_values) if max_error_values else None
            ),
            "episode_ended_before_perturbation": any(
                e["ended_before_trigger"] for e in event_results
            ),
        }
        if len(event_results) > 1:
            flat["perturbation_events"] = event_results
        return flat

    def label_for_step(self, step: int) -> str | None:
        """返回给定 step 的扰动状态标签（"<role>:perturb" / "<role>:recover"），无扰动返回 None。

        优先返回"扰动建立"（红）：顺序模式下若前一角色尚未恢复、后一角色已开始扰动，
        后一角色的红色标注不应被前一角色的蓝色恢复标注覆盖。
        """
        for event in self.events:
            if event.triggered and event.trigger_step <= step < event.release_step:
                return f"{event.role}:perturb"
        for event in reversed(self.events):
            if not event.triggered:
                continue
            recovery_end = event.control_recovery_step
            if recovery_end is None:
                recovery_end = float("inf")
            if event.release_step <= step < recovery_end:
                return f"{event.role}:recover"
        return None
def summarize_recovery_perturbations(episode_records: list[dict]) -> dict:
    if not episode_records:
        return {
            "perturbation_trigger_rate": 0.0,
            "control_recovery_rate": 0.0,
            "mean_control_recovery_steps": None,
        }
    triggered = [record for record in episode_records if record.get("perturbation_triggered")]
    recovered = [record for record in triggered if record.get("control_recovered")]
    recovery_steps = [
        int(record["control_recovery_steps_after_release"])
        for record in recovered
        if record.get("control_recovery_steps_after_release") is not None
    ]
    return {
        "perturbation_trigger_rate": len(triggered) / len(episode_records),
        "perturbation_triggered_episodes": len(triggered),
        "control_recovery_rate": len(recovered) / len(triggered) if triggered else 0.0,
        "control_recovered_episodes": len(recovered),
        "mean_control_recovery_steps": (
            float(np.mean(recovery_steps)) if recovery_steps else None
        ),
    }


def patch_act_position_embedding_for_determinism():
    """ACT 的 CUDA cumsum 非确定；评估时用等价 arange 坐标替代。"""
    try:
        from lerobot.common.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d
    except Exception as exc:
        logging.warning(f"无法应用 ACT 确定性补丁: {exc}")
        return

    if getattr(ACTSinusoidalPositionEmbedding2d, "_dppo_deterministic_patch", False):
        return

    def deterministic_forward(self, x):
        height, width = x.shape[-2:]
        y_range = torch.arange(1, height + 1, dtype=torch.float32, device=x.device)
        y_range = y_range.view(1, height, 1).expand(1, height, width)
        x_range = torch.arange(1, width + 1, dtype=torch.float32, device=x.device)
        x_range = x_range.view(1, 1, width).expand(1, height, width)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        return torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

    ACTSinusoidalPositionEmbedding2d.forward = deterministic_forward
    ACTSinusoidalPositionEmbedding2d._dppo_deterministic_patch = True
    logging.info("已启用 ACT 评估确定性补丁: positional embedding 使用 arange 替代 CUDA cumsum")


def custom_eval_policy_vectorized(env, policy, cfg_eval, videos_dir, device):
    """用 gym.vector 并行评估多个 episode；policy action chunk 队列按 batch 同步推进。"""
    policy.eval()
    n_envs = int(env.num_envs)
    successes = []
    rewards = []
    episode_records = []
    saved_video_paths = []

    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = int(getattr(cfg_eval, "n_episodes", 10))
    max_rendered = int(getattr(cfg_eval, "max_episodes_rendered", 4))
    fps = int(getattr(cfg_eval, "fps", 25))
    max_steps = int(getattr(cfg_eval, "max_steps", 300))
    raw_camera = getattr(cfg_eval, "render_camera", "overhead_cam")
    render_cameras = [raw_camera] if isinstance(raw_camera, str) else list(raw_camera)
    expected_keys = set(policy.config.input_shapes)
    base_seed = int(getattr(cfg_eval, "seed", 1000))
    perturbation_spec = RecoveryPerturbationSpec.from_eval_cfg(
        cfg_eval,
        max_steps=max_steps,
        fps=fps,
    )
    logging.info("恢复鲁棒性评估配置: %s", perturbation_spec.to_dict())
    n_action_steps = int(getattr(policy.config, "n_action_steps", 8))
    global_real_inference_times = []
    router_totals = {
        "decisions": 0,
        "activations": 0,
        "probability_sum": 0.0,
    }

    episode_cursor = 0
    progress = tqdm(total=n_episodes, leave=False)
    while episode_cursor < n_episodes:
        active_count = min(n_envs, n_episodes - episode_cursor)
        episode_ids = np.arange(episode_cursor, episode_cursor + active_count, dtype=np.int64)
        active = np.zeros(n_envs, dtype=bool)
        active[:active_count] = True
        seeds = [
            base_seed + int(episode_ids[idx]) if idx < active_count else base_seed + n_episodes + idx
            for idx in range(n_envs)
        ]

        seed_runtime(int(seeds[0]))
        try:
            obs, _ = env.reset(seed=seeds)
        except TypeError:
            logging.warning("当前 VectorEnv 不支持列表 seed，回退为无 seed reset。")
            obs, _ = env.reset()

        policy.reset()
        seed_runtime(int(seeds[0]))
        episode_perturbations = [
            EpisodeRecoveryPerturbation(
                perturbation_spec,
                episode=(int(episode_ids[idx]) if idx < active_count else episode_cursor + idx),
                episode_seed=int(seeds[idx]),
                n_action_steps=n_action_steps,
            )
            for idx in range(n_envs)
        ]

        done = np.zeros(n_envs, dtype=bool)
        episode_success = np.zeros(n_envs, dtype=bool)
        episode_rewards = np.zeros(n_envs, dtype=np.float32)
        steps_taken = np.zeros(n_envs, dtype=np.int32)
        frames_by_env = {
            idx: {camera: [] for camera in render_cameras}
            for idx in range(active_count)
            if int(episode_ids[idx]) < max_rendered
        }
        render_failed_cameras = set()
        batch_inference_times = []

        for step in range(max_steps):
            step_active = active & ~done
            if not step_active.any():
                break

            if frames_by_env:
                render_indices = [idx for idx in frames_by_env if step_active[idx]]
                if render_indices:
                    for camera in render_cameras:
                        if camera in render_failed_cameras:
                            continue
                        try:
                            frames = _render_vector_frames(env, camera, n_envs)
                        except Exception as exc:
                            render_failed_cameras.add(camera)
                            logging.warning(f"VectorEnv 渲染相机 {camera} 失败，本批次跳过该视角录像: {exc}")
                            continue
                        for env_idx in render_indices:
                            frames_by_env[env_idx][camera].append(frames[env_idx])

            raw_states = np.asarray(obs["agent_pos"], dtype=np.float64)
            if raw_states.ndim == 1:
                raw_states = raw_states[None, :]
            if raw_states.shape != (n_envs, ACTION_DIM):
                raise ValueError(
                    f"向量环境agent_pos应为({n_envs},{ACTION_DIM})，当前为{raw_states.shape}。"
                )
            batch_obs = prepare_policy_observation(obs, expected_keys, device)
            router_decision_due = len(policy._queues["action"]) == 0
            start_time = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch_obs)
            if router_decision_due:
                router_decision = get_last_router_decision(policy)
                if router_decision is not None:
                    gate, probability = router_decision
                    router_totals["decisions"] += int(step_active.sum())
                    router_totals["activations"] += int(
                        gate[step_active].sum()
                    )
                    router_totals["probability_sum"] += float(
                        probability[step_active].sum()
                    )
            action_np = action.detach().cpu().numpy().copy()
            if action_np.ndim == 1:
                action_np = action_np[None, :]
            if action_np.shape[0] != n_envs:
                raise ValueError(f"策略输出 batch={action_np.shape[0]}，但评估环境 num_envs={n_envs}")
            for env_idx in np.flatnonzero(step_active):
                action_np[env_idx] = episode_perturbations[env_idx].apply(
                    step,
                    action_np[env_idx],
                    raw_states[env_idx],
                )
            action_np[~step_active] = 0.0

            inference_time_ms = (time.perf_counter() - start_time) * 1000
            batch_inference_times.append(inference_time_ms)
            steps_taken[step_active] = step + 1

            try:
                obs, reward, terminated, truncated, info = env.step(action_np)
            except Exception as exc:
                logging.exception(f"VectorEnv 物理引擎崩溃 (Step {step})，本批未完成 episode 记为失败: {exc}")
                unfinished = active & ~done
                episode_rewards[unfinished] = -1000.0
                episode_success[unfinished] = False
                done[unfinished] = True
                break

            reward_arr = _as_float_array(reward, n_envs)
            terminated_arr = _as_bool_array(terminated, n_envs)
            truncated_arr = _as_bool_array(truncated, n_envs)
            step_done = terminated_arr | truncated_arr | (step + 1 >= max_steps)
            success_arr = _extract_info_bool(info, "is_success", n_envs, default=False)

            episode_rewards[step_active] += reward_arr[step_active]
            newly_done = step_active & step_done
            episode_success[newly_done] = success_arr[newly_done]
            done[newly_done] = True

        unfinished = active & ~done
        if unfinished.any():
            steps_taken[unfinished] = np.maximum(steps_taken[unfinished], max_steps)
            episode_success[unfinished] = False
            done[unfinished] = True

        for local_idx in range(active_count):
            ep = int(episode_ids[local_idx])
            success = bool(episode_success[local_idx])
            ep_reward = float(episode_rewards[local_idx])
            successes.append(success)
            rewards.append(ep_reward)
            episode_record = {
                "episode": ep,
                "seed": base_seed + ep,
                "success": success,
                "reward": ep_reward,
                "steps": int(steps_taken[local_idx]),
            }
            episode_record.update(
                episode_perturbations[local_idx].result(
                    episode_steps=int(steps_taken[local_idx])
                )
            )
            episode_records.append(episode_record)

            if local_idx in frames_by_env:
                status = "Success" if success else "Fail"
                annotate = bool(getattr(cfg_eval, "recovery_annotate_video", True))
                for camera, frames in frames_by_env[local_idx].items():
                    if len(frames) == 0:
                        continue
                    if annotate:
                        pert = episode_perturbations[local_idx]
                        frames = [
                            _annotate_frame(frame, pert.label_for_step(i))
                            for i, frame in enumerate(frames)
                        ]
                    video_name = f"{camera}_ep_{ep}_reward={ep_reward:.1f}_{status}.mp4"
                    video_path = videos_dir / video_name
                    imageio.mimsave(str(video_path), frames, fps=fps)
                    saved_video_paths.append(str(video_path))

        real_inferences = [t for t in batch_inference_times if t > 5.0]
        if real_inferences:
            global_real_inference_times.extend(real_inferences)

        episode_cursor += active_count
        progress.update(active_count)

    progress.close()

    if len(global_real_inference_times) > 0:
        warmup_steps = 3
        stable_times = (
            global_real_inference_times[warmup_steps:]
            if len(global_real_inference_times) > warmup_steps
            else global_real_inference_times
        )
        max_time = max(stable_times)
        avg_real_time = sum(stable_times) / len(stable_times)
    else:
        avg_real_time = 0.0
        max_time = 0.0

    router_decisions = int(router_totals["decisions"])
    recovery_metrics = summarize_recovery_perturbations(episode_records)
    return {
        "aggregated": {
            "success_rate": float(np.mean(successes)),
            "average_reward": float(np.mean(rewards)),
            "avg_inference_ms": float(avg_real_time),
            "max_inference_ms": float(max_time),
            "router_decisions": router_decisions,
            "router_activations": int(router_totals["activations"]),
            "router_activation_rate": (
                float(router_totals["activations"]) / router_decisions
                if router_decisions
                else 0.0
            ),
            "router_mean_probability": (
                float(router_totals["probability_sum"]) / router_decisions
                if router_decisions
                else 0.0
            ),
            **recovery_metrics,
        },
        "video_paths": saved_video_paths,
        "episodes": episode_records,
    }


def custom_eval_policy(env, policy, cfg_eval, videos_dir, device):
    """
    完全自主实现的评估代码。没有任何黑盒。
    接收标准 Gym 环境，处理图像归一化，跑策略推理，保存视频。
    """
    if is_vector_env(env):
        return custom_eval_policy_vectorized(env, policy, cfg_eval, videos_dir, device)

    policy.eval() # 必须开启评估模式
    successes = []
    rewards = []
    episode_records = []

    # 用来动态记录实际保存的视频路径
    saved_video_paths = []
    
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # 从配置中动态读取参数，提供后备默认值防崩溃
    n_episodes = getattr(cfg_eval, "n_episodes", 10)
    max_rendered = getattr(cfg_eval, "max_episodes_rendered", 4)
    fps = getattr(cfg_eval, "fps", 25)
    max_steps = getattr(cfg_eval, "max_steps", 300)
    raw_camera = getattr(cfg_eval, "render_camera", 'overhead_cam')
    render_cameras = [raw_camera] if isinstance(raw_camera, str) else list(raw_camera)
    expected_keys = set(policy.config.input_shapes)
    # 从配置中提取基础种子，默认为 1000
    base_seed = getattr(cfg_eval, "seed", 1000)
    perturbation_spec = RecoveryPerturbationSpec.from_eval_cfg(
        cfg_eval,
        max_steps=max_steps,
        fps=fps,
    )
    logging.info("恢复鲁棒性评估配置: %s", perturbation_spec.to_dict())
    n_action_steps = int(getattr(policy.config, "n_action_steps", 8))
    # 计算所有评估回合推理的总耗时
    global_real_inference_times = []
    router_totals = {
        "decisions": 0,
        "activations": 0,
        "probability_sum": 0.0,
    }
    # 动作执行循环
    for ep in tqdm(range(n_episodes), leave=False):
        # 计算当前回合的专属种子，并传给环境
        ep_seed = base_seed + ep
        # 先固定全局 RNG，再 reset 环境；这样 reset 内部若用 random/np/torch 也可复现。
        seed_runtime(ep_seed)
        seed_env_spaces(env, ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        done = False
        frames_by_camera = {camera: [] for camera in render_cameras} if ep < max_rendered else {}
        ep_reward = 0
        # LeRobot/DPPO 的 Policy 内置了 action chunking 队列
        policy.reset() # 清空模型的动作缓冲历史
        # policy.reset/env.reset 之后再对齐一次，保证 diffusion select_action 的首个噪声固定。
        seed_runtime(ep_seed)
        episode_perturbation = EpisodeRecoveryPerturbation(
            perturbation_spec,
            episode=ep,
            episode_seed=ep_seed,
            n_action_steps=n_action_steps,
        )

        # 🌟 1. 每回合新建一个列表，静默记录本回合的所有耗时
        ep_inference_times = []
        episode_router = {
            "decisions": 0,
            "activations": 0,
            "probability_sum": 0.0,
        }
        steps_taken = 0
        for step in range(max_steps):
            steps_taken = step + 1
            # 1. 如果还在需要渲染的额度内，才调用渲染 (提升非渲染 episode 的评估速度)
            if ep < max_rendered:
                for camera in render_cameras:
                    frames_by_camera[camera].append(env.unwrapped.render([camera])) # gym创建需要加上 .unwrapped

            # 2. 只转换模型真正需要的输入特征，推入 GPU。
            raw_state = np.asarray(obs["agent_pos"], dtype=np.float64).copy()
            obs = prepare_policy_observation(obs, expected_keys, device)
            # ==========================================
            # ⏱️ 开始计时：使用高精度的 perf_counter
            # ==========================================
            start_time = time.perf_counter()

            # 3. 推理获取动作
            router_decision_due = len(policy._queues["action"]) == 0
            with torch.inference_mode():
                # 使用lerobot自带的推理函数，obs是单帧的，模型会自动处理历史动作的拼接和缓存
                action = policy.select_action(obs) # 这里每次取出一个动作，推理依旧一次生成8个动作，只是一个个往外取
            if router_decision_due:
                router_decision = get_last_router_decision(policy)
                if router_decision is not None:
                    gate, probability = router_decision
                    episode_router["decisions"] += 1
                    episode_router["activations"] += int(gate[0])
                    episode_router["probability_sum"] += float(
                        probability[0]
                    )
            # 4. 把模型输出的 Tensor 动作转回 Numpy (包含在计时内)
            action_np = action.squeeze(0).cpu().numpy()
            action_np = episode_perturbation.apply(step, action_np, raw_state)

            # ⏱️ 结束计时
            inference_time_ms = (time.perf_counter() - start_time) * 1000 # 转换为毫秒 (ms)
            ep_inference_times.append(inference_time_ms)
            # print(f"👉 Step {step} 推理耗时: {inference_time_ms:.2f} ms")
            # ==========================================
            try:
                # 5. 与环境交互
                obs, reward, terminated, truncated, info = env.step(action_np)
                ep_reward += float(reward)

                done = terminated or truncated
            except Exception as e:
                # 🌟 核心拦截器：无论物理引擎报什么错，全部强行吃掉！
                logging.error(f"物理引擎崩溃 (Step {step}): {e}")
                logging.error("判定本回合评估失败，直接结束当前回合，继续训练...")
                
                done = True
                ep_reward = -1000.0          # 发生物理崩溃，不给奖励
                info = {"is_success": False} # 标记为失败
            if done:
                break
        # 记录指标
        successes.append(info.get("is_success", False))
        rewards.append(ep_reward)
        episode_record = {
            "episode": ep,
            "seed": ep_seed,
            "success": bool(info.get("is_success", False)),
            "reward": float(ep_reward),
            "steps": int(steps_taken),
        }
        episode_record.update(
            episode_perturbation.result(episode_steps=int(steps_taken))
        )
        episode_records.append(episode_record)
        decisions = int(episode_router["decisions"])
        if decisions:
            router_totals["decisions"] += decisions
            router_totals["activations"] += int(
                episode_router["activations"]
            )
            router_totals["probability_sum"] += float(
                episode_router["probability_sum"]
            )
            episode_records[-1].update(
                {
                    "router_decisions": decisions,
                    "router_activations": int(
                        episode_router["activations"]
                    ),
                    "router_activation_rate": (
                        float(episode_router["activations"]) / decisions
                    ),
                    "router_mean_probability": (
                        float(episode_router["probability_sum"]) / decisions
                    ),
                }
            )

        # 6. 根据配置的帧率和最大渲染数量保存视频
        if ep < max_rendered and frames_by_camera:
            status = "Success" if successes[ep] else "Fail"
            annotate = bool(getattr(cfg_eval, "recovery_annotate_video", True))
            for camera, frames in frames_by_camera.items():
                if len(frames) == 0:
                    continue
                if annotate:
                    frames = [
                        _annotate_frame(frame, episode_perturbation.label_for_step(i))
                        for i, frame in enumerate(frames)
                    ]
                video_name = f"{camera}_ep_{ep}_reward={ep_reward:.1f}_{status}.mp4"
                video_path = videos_dir / video_name
                imageio.mimsave(str(video_path), frames, fps=fps)
                saved_video_paths.append(str(video_path))

        # 记录每回合的推理时间
        if len(ep_inference_times) > 0:
            
            # 过滤出“真实推理”步骤（比如耗时超过 5ms 的肯定是在跑网络，排除了 0.1ms 的出队操作）
            real_inferences = [t for t in ep_inference_times if t > 5.0]
            
            if real_inferences:
                global_real_inference_times.extend(real_inferences) # 加入全局统计

    # 结算全局指标：剔除前几次全局预热     
    if len(global_real_inference_times) > 0:

        # 全局剔除前 3 次真正的网络推理作为 Warm-up
        warmup_steps = 3
        # 尝试获取稳定状态的数据
        if len(global_real_inference_times) > warmup_steps:
            stable_times = global_real_inference_times[warmup_steps:]
        else:
            # 数据太少，不够剔除，只能全量使用
            stable_times = global_real_inference_times
            
        # 安全地计算最大值和均值
        max_time = max(stable_times)

        avg_real_time = sum(stable_times) / len(stable_times)
        # logging.info(f"[总计{n_episodes}回合] 真实推理触发: {len(global_real_inference_times)} 次 | 峰值耗时: {max_time:.2f} ms | 平均耗时: {avg_real_time:.2f} ms")
    else:
        avg_real_time = 0.0  # 提供默认值防报错
        max_time = 0.0
    router_decisions = int(router_totals["decisions"])
    recovery_metrics = summarize_recovery_perturbations(episode_records)
    return {
        "aggregated": {
            "success_rate": float(np.mean(successes)),
            "average_reward": float(np.mean(rewards)),
            "avg_inference_ms": float(avg_real_time),
            "max_inference_ms": float(max_time),
            "router_decisions": router_decisions,
            "router_activations": int(router_totals["activations"]),
            "router_activation_rate": (
                float(router_totals["activations"]) / router_decisions
                if router_decisions
                else 0.0
            ),
            "router_mean_probability": (
                float(router_totals["probability_sum"]) / router_decisions
                if router_decisions
                else 0.0
            ),
            **recovery_metrics,
        },
        "video_paths": saved_video_paths,
        "episodes": episode_records,
    }


def seed_everything(seed: int, deterministic: bool = False):
    """Set seed for absolute reproducibility."""
    
    # 1. 锁死 Python 字典和集合的哈希种子 
    # (防止字典遍历顺序在不同运行中发生变化，导致 Batch 数据错位)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. 锁死 Python / Numpy / PyTorch (CPU & 所有 GPU)
    seed_runtime(seed)

    configure_torch_runtime(deterministic)

def ensure_python_hash_seed(seed: int):
    """PYTHONHASHSEED 必须在解释器启动前生效；独立运行时自动重启一次补齐。"""
    desired = str(seed)
    if os.environ.get("PYTHONHASHSEED") == desired:
        return

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = desired
    logging.warning(f"PYTHONHASHSEED 需要在 Python 启动前设置，正在用 PYTHONHASHSEED={desired} 自动重启 eval_policy.py。")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

def resolve_eval_path(path_like) -> Path:
    path = Path(path_like).expanduser()
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        root_path = ROOT_PATH / path
        if root_path.exists():
            return root_path.resolve()
        return root_path.resolve()
    return path


def checkpoint_step(checkpoint_path: Path) -> int:
    match = re.match(r"^(\d+)", checkpoint_path.name)
    return int(match.group(1)) if match else -1


def is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "pretrained_model").is_dir()
        or (path / "config.yaml").exists()
        or (path / "training_state.pth").exists()
    )


def resolve_record_checkpoint_path(raw_path, checkpoints_dir: Path) -> Path:
    raw = Path(str(raw_path)).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        Path.cwd() / raw,
        ROOT_PATH / raw,
        checkpoints_dir / raw,
        checkpoints_dir / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT_PATH / raw).resolve()


def unique_paths(paths: list[Path]) -> list[Path]:
    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path.resolve())
    return unique


def discover_checkpoints(path_like, checkpoint_source: str = "all") -> tuple[list[Path], Path, bool]:
    """支持输入单个 checkpoint、checkpoints 目录或完整训练 run 目录。"""
    input_path = resolve_eval_path(path_like)
    if not input_path.exists():
        raise FileNotFoundError(f"❌ 找不到评估路径: {input_path}\n请检查 ckpt_path 是否正确。")

    if is_checkpoint_dir(input_path):
        run_dir = input_path.parent.parent if input_path.parent.name == "checkpoints" else input_path.parent
        return [input_path], run_dir.resolve(), False

    if input_path.name == "checkpoints":
        checkpoints_dir = input_path
        run_dir = input_path.parent
    else:
        checkpoints_dir = input_path / "checkpoints"
        run_dir = input_path

    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"❌ 未找到 checkpoints 目录: {checkpoints_dir}\n"
            "ckpt_path 可以指向单个 checkpoint、checkpoints 目录，或训练输出 run 目录。"
        )

    checkpoint_source = checkpoint_source.lower()
    if checkpoint_source == "all":
        checkpoints = [p.resolve() for p in checkpoints_dir.iterdir() if is_checkpoint_dir(p)]
        checkpoints.sort(key=lambda p: (checkpoint_step(p), p.name))
    elif checkpoint_source in {"top_k", "latest"}:
        records_path = checkpoints_dir / "top_k_records.json"
        if not records_path.exists():
            raise FileNotFoundError(f"❌ {checkpoint_source} 需要 top_k_records.json，但未找到: {records_path}")

        with open(records_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if checkpoint_source == "latest":
            raw_paths = [records["latest"]]
        else:
            raw_paths = [item["path"] for item in records.get("top_k", [])]

        checkpoints = [resolve_record_checkpoint_path(raw_path, checkpoints_dir) for raw_path in raw_paths]
        missing = [str(path) for path in checkpoints if not is_checkpoint_dir(path)]
        if missing:
            raise FileNotFoundError("top_k_records.json 中存在不可用 checkpoint:\n" + "\n".join(missing))
    else:
        raise ValueError("checkpoint_source 只能是 all / top_k / latest")

    checkpoints = unique_paths(checkpoints)
    if not checkpoints:
        raise FileNotFoundError(f"❌ 没有在目录中发现可评估 checkpoint: {checkpoints_dir}")

    return checkpoints, run_dir.resolve(), True


def build_batch_output_dir(eval_cfg, run_dir: Path) -> Path:
    output_dir = getattr(eval_cfg, "eval_output_dir", None)
    if output_dir:
        output_path = Path(output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = ROOT_PATH / output_path
        return output_path.resolve()

    source = getattr(eval_cfg, "checkpoint_source", "all")
    mode = getattr(eval_cfg, "mode", "fast_repro")
    precision = "amp" if getattr(eval_cfg, "use_amp", False) and mode != "strict" else "fp32"
    ablation_tag = (
        coupling_ablation_tag(eval_cfg)
        + output_corrector_ablation_tag(eval_cfg)
    )
    recovery_tag = RecoveryPerturbationSpec.from_eval_cfg(eval_cfg).tag()
    return (
        run_dir
        / "policy_eval_recovery"
        / (
            f"{source}_{mode}_{precision}_seed={eval_cfg.seed}_ep={eval_cfg.n_episodes}"
            f"_steps={eval_cfg.max_steps}_{recovery_tag}{ablation_tag}"
        )
    ).resolve()


def make_checkpoint_eval_cfg(eval_cfg, checkpoint_path: Path):
    checkpoint_cfg = SimpleNamespace(**vars(eval_cfg))
    checkpoint_cfg.ckpt_path = str(checkpoint_path)
    if isinstance(getattr(eval_cfg, "render_camera", None), list):
        checkpoint_cfg.render_camera = list(eval_cfg.render_camera)
    return checkpoint_cfg


def checkpoint_identity_keys(checkpoint_path: Path) -> set[str]:
    checkpoint_path = checkpoint_path.resolve()
    return {str(checkpoint_path), checkpoint_path.name}


def row_identity_keys(row: dict) -> set[str]:
    keys = set()
    for field in ("checkpoint_key", "checkpoint_path", "checkpoint_name"):
        value = row.get(field)
        if not value:
            continue
        keys.add(str(value))
        if field == "checkpoint_path":
            try:
                keys.add(str(resolve_eval_path(value)))
            except Exception:
                pass
    return keys


def load_existing_eval_summary(summary_dir: Path, eval_cfg=None) -> list[dict]:
    summary_path = summary_dir / "policy_eval_summary.json"
    if not summary_path.exists():
        return []

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logging.warning(f"读取已有评估汇总失败，将从空表继续: {summary_path} ({exc})")
        return []

    if eval_cfg is not None and isinstance(payload, dict):
        existing_spec = payload.get("recovery_perturbation")
        current_spec = RecoveryPerturbationSpec.from_eval_cfg(eval_cfg).to_dict()
        if existing_spec is not None and existing_spec != current_spec:
            raise ValueError(
                "eval_output_dir中已有另一套恢复扰动配置，拒绝混写或错误跳过checkpoint。"
                f" existing={existing_spec}, current={current_spec}"
            )

    rows = payload.get("results", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list):
        logging.warning(f"已有评估汇总格式异常，将从空表继续: {summary_path}")
        return []

    rows = clean_eval_rows(rows)
    logging.info(f"已读取已有评估汇总: {summary_path}，包含 {len(rows)} 条历史结果")
    return rows


def clean_eval_rows(rows: list[dict]) -> list[dict]:
    cleaned_rows = []
    for row in rows:
        cleaned = dict(row)
        cleaned.pop("eval_order", None)
        cleaned.pop("completed_at", None)
        cleaned_rows.append(cleaned)
    return cleaned_rows


def completed_checkpoint_keys(rows: list[dict]) -> set[str]:
    keys = set()
    for row in rows:
        if row.get("status") != "ok":
            continue
        keys.update(row_identity_keys(row))
    return keys


def latest_rows_by_checkpoint(rows: list[dict]) -> list[dict]:
    latest = {}
    fallback_index = 0
    for row in rows:
        keys = row_identity_keys(row)
        if not keys:
            fallback_index += 1
            key = f"__row_{fallback_index}"
        else:
            key = sorted(keys)[0]
        latest[key] = row
    return list(latest.values())


def ranked_eval_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    latest_rows = latest_rows_by_checkpoint(rows)
    ok_rows = [row for row in latest_rows if row.get("status") == "ok"]
    failed_rows = [row for row in latest_rows if row.get("status") != "ok"]
    sorted_ok_rows = sorted(
        ok_rows,
        key=lambda row: (row.get("success_rate", -1.0), row.get("average_reward", float("-inf"))),
        reverse=True,
    )
    ranked_rows = []
    for rank, row in enumerate(sorted_ok_rows, start=1):
        ranked_row = dict(row)
        ranked_row["rank"] = rank
        ranked_rows.append(ranked_row)
    for row in failed_rows:
        ranked_row = dict(row)
        ranked_row["rank"] = None
        ranked_rows.append(ranked_row)
    return latest_rows, ok_rows, ranked_rows


def write_eval_summary(summary_dir: Path, rows: list[dict], eval_cfg, source_path) -> tuple[Path, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = clean_eval_rows(rows)
    latest_rows, ok_rows, ranked_rows = ranked_eval_rows(rows)

    perturbation_spec = RecoveryPerturbationSpec.from_eval_cfg(eval_cfg)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_path": str(source_path),
        "checkpoint_source": getattr(eval_cfg, "checkpoint_source", "all"),
        "mode": getattr(eval_cfg, "mode", "fast_repro"),
        "seed": eval_cfg.seed,
        "n_episodes": eval_cfg.n_episodes,
        "eval_batch_size": int(getattr(eval_cfg, "batch_size", getattr(eval_cfg, "num_envs", 1))),
        "max_steps": eval_cfg.max_steps,
        "recovery_perturbation": perturbation_spec.to_dict(),
        "total_records": len(rows),
        "latest_checkpoint_records": len(latest_rows),
        "best": ranked_rows[0] if ok_rows else None,
        "ranking": ranked_rows,
        "results": rows,
    }
    json_path = summary_dir / "policy_eval_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = summary_dir / "policy_eval_summary.csv"
    fieldnames = [
        "rank",
        "status",
        "checkpoint_name",
        "checkpoint_key",
        "step",
        "success_rate",
        "success_rate_percent",
        "average_reward",
        "avg_inference_ms",
        "max_inference_ms",
        "router_mode",
        "router_threshold",
        "router_decisions",
        "router_activations",
        "router_activation_rate",
        "router_mean_probability",
        "perturbation_mode",
        "perturbation_severity",
        "perturbation_trigger_rate",
        "perturbation_triggered_episodes",
        "control_recovery_rate",
        "control_recovered_episodes",
        "mean_control_recovery_steps",
        "n_episodes",
        "eval_batch_size",
        "max_steps",
        "seed",
        "mode",
        "videos_dir",
        "episode_results_path",
        "checkpoint_path",
        "error",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(ranked_rows, start=1):
            csv_row = {field: row.get(field, "") for field in fieldnames}
            csv_row["rank"] = rank if row.get("status") == "ok" else ""
            writer.writerow(csv_row)

    logging.info(f"批量评估汇总 JSON 已保存: {json_path}")
    logging.info(f"批量评估汇总 CSV 已保存: {csv_path}")
    return json_path, csv_path


def prune_ranked_eval_artifacts(
    run_dir: Path,
    summary_dir: Path,
    rows: list[dict],
    keep_top: int,
    prune_checkpoints: bool = True,
    prune_eval_outputs: bool = True,
    prune_unranked_checkpoints: bool = False,
) -> dict:
    if keep_top <= 0:
        raise ValueError("keep_top_after_eval 必须大于 0，或者设为 None 关闭清理。")

    latest_rows, _, ranked_rows = ranked_eval_rows(rows)
    keep_rows = [row for row in ranked_rows if row.get("status") == "ok"][:keep_top]
    keep_names = {row.get("checkpoint_name") for row in keep_rows if row.get("checkpoint_name")}
    evaluated_names = {row.get("checkpoint_name") for row in latest_rows if row.get("checkpoint_name")}

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "keep_top": keep_top,
        "kept_checkpoints": [row.get("checkpoint_name") for row in keep_rows],
        "deleted_checkpoints": [],
        "deleted_eval_outputs": [],
        "skipped_unranked_checkpoints": [],
        "skipped": [],
    }

    if not keep_names:
        logging.warning("没有成功评估的 checkpoint，跳过按排名清理。")
        report["skipped"].append("no_successful_checkpoint")
        return report

    if prune_checkpoints:
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            for checkpoint_dir in checkpoints_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue
                if checkpoint_dir.name in keep_names:
                    continue
                if checkpoint_dir.name not in evaluated_names and not prune_unranked_checkpoints:
                    report["skipped_unranked_checkpoints"].append(str(checkpoint_dir))
                    continue
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
                report["deleted_checkpoints"].append(str(checkpoint_dir))
                logging.info(f"已按评估排名删除低排名模型: {checkpoint_dir}")
        else:
            report["skipped"].append(f"missing_checkpoints_dir:{checkpoints_dir}")

    if prune_eval_outputs and summary_dir.exists():
        for eval_dir in summary_dir.iterdir():
            if not eval_dir.is_dir() or eval_dir.name not in evaluated_names:
                continue
            if eval_dir.name in keep_names:
                continue
            shutil.rmtree(eval_dir, ignore_errors=True)
            report["deleted_eval_outputs"].append(str(eval_dir))
            logging.info(f"已按评估排名删除低排名评估输出: {eval_dir}")

    report_path = summary_dir / "policy_eval_prune_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info(f"评估清理报告已保存: {report_path}")
    return report


def prune_if_configured(
    eval_cfg,
    run_dir: Path,
    batch_output_dir: Path | None,
    rows: list[dict],
    stage: str,
    prune_unranked_checkpoints: bool = False,
):
    if batch_output_dir is None:
        return None

    keep_top_after_eval = getattr(eval_cfg, "keep_top_after_eval", None)
    if keep_top_after_eval is None:
        return None

    if getattr(eval_cfg, "max_checkpoints", None) and not getattr(eval_cfg, "allow_prune_with_max_checkpoints", False):
        logging.warning("当前设置了 max_checkpoints，跳过清理，避免删除尚未完整评估的 checkpoint。")
        return None

    logging.info(f"{stage}: 按 ranking 清理评估产物，仅保留前 {int(keep_top_after_eval)} 个 checkpoint。")
    return prune_ranked_eval_artifacts(
        run_dir=run_dir,
        summary_dir=batch_output_dir,
        rows=rows,
        keep_top=int(keep_top_after_eval),
        prune_checkpoints=bool(getattr(eval_cfg, "prune_checkpoints", True)),
        prune_eval_outputs=bool(getattr(eval_cfg, "prune_eval_outputs", True)),
        prune_unranked_checkpoints=prune_unranked_checkpoints,
    )


def evaluate_one_checkpoint(eval_cfg, checkpoint_path: Path, device, output_root: Path | None = None) -> dict:
    ckpt_path = checkpoint_path.resolve()
    eval_mode = getattr(eval_cfg, "mode", "fast_repro")
    eval_env = None
    policy = None
    active_coupling_scales = {}
    active_output_corrector_scales = {}
    active_router = {}

    try:
        # ==========================================
        # 1.自动探测 LeRobot 的 pretrained_model 子文件夹
        # ==========================================
        hf_model_dir = ckpt_path / "pretrained_model"
        if hf_model_dir.exists():
            logging.info("检测到 LeRobot 标准快照结构，将自动读取子目录: pretrained_model")
            load_dir = hf_model_dir
        else:
            load_dir = ckpt_path

        # ==========================================
        # 2.实例化 Policy 并加载权重
        # ==========================================
        logging.info(f"正在从目录重建网络并加载权重: {load_dir}")
        try:
            from lerobot.common.utils.utils import init_hydra_config

            config_yaml_path = Path(load_dir) / "config.yaml"
            if not config_yaml_path.exists():
                config_yaml_path = Path(load_dir).parent / "config.yaml"

            if not config_yaml_path.exists():
                raise FileNotFoundError("找不到 config.yaml，无法初始化 hydra_cfg！")

            hydra_cfg = init_hydra_config(str(config_yaml_path))
            hydra_cfg.device = str(device)

            policy = make_policy(
                hydra_cfg=hydra_cfg,
                pretrained_policy_name_or_path=str(load_dir)
            )

            logging.info("  成功使用 make_policy 加载策略！底层 Normalizer 与平滑权重已自动生效。")
            policy.to(device)
            active_coupling_scales = apply_coupling_ablation_overrides(
                policy, eval_cfg
            )
            if active_coupling_scales:
                logging.info(
                    "耦合推理消融已生效: "
                    f"View→Arm={active_coupling_scales['view_to_arm_coupling_scale']:g}, "
                    f"Arm→View={active_coupling_scales['arm_to_view_coupling_scale']:g}"
                )
            active_output_corrector_scales = (
                apply_output_corrector_ablation_overrides(policy, eval_cfg)
            )
            if active_output_corrector_scales:
                logging.info(
                    "最终输出修正消融已生效: "
                    f"View→Arm={active_output_corrector_scales['view_to_arm_output_scale']:g}, "
                    f"Arm→View={active_output_corrector_scales['arm_to_view_output_scale']:g}"
                )
            active_router = apply_router_ablation_override(policy, eval_cfg)
            if active_router:
                logging.info(
                    "Router推理消融已生效: "
                    f"mode={active_router['router_mode']}, "
                    f"threshold={active_router['router_threshold']:g}"
                )
        except Exception as e:
            raise RuntimeError(f"❌ 权重加载失败！详细报错: {e}")

        # ==========================================
        # 🌟 3.读取快照中的配置，使环境和训练时的对齐
        # ==========================================
        all_obs_keys = policy.config.input_shapes.keys()
        ref_cams = [k.replace("observation.images.", "") for k in all_obs_keys if "observation.images." in k]
        if not ref_cams:
            raise ValueError("❌ 严重冲突：模型中未找到相机相关参数。请检查模型输入是否正确。")

        render_cameras = [eval_cfg.render_camera] if isinstance(eval_cfg.render_camera, str) else list(eval_cfg.render_camera)
        eval_cfg.render_camera = render_cameras
        obs_cameras = list(dict.fromkeys(ref_cams))

        config_yaml_path = Path(load_dir) / "config.yaml"
        if config_yaml_path.exists():
            with open(config_yaml_path, "r", encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f)

                env_cfg = full_cfg.get("env", {})
                env_name = env_cfg.get("name", getattr(env_cfg, "name", "guided_vision"))
                env_task = env_cfg.get("task", getattr(env_cfg, "task", "InsertCylinder-3Arms-v0"))
                logging.info(f"成功从预训练文件夹读取完整环境配置: {env_name}/{env_task}")
        else:
            env_name = getattr(eval_cfg, "name", "guided_vision")
            env_task = getattr(eval_cfg, "task", "InsertCylinder-3Arms-v0")
            logging.warning(f"  未找到 config.yaml，使用本地设定的后备环境: {env_name}/{env_task}")

        env_id = f"{env_name}/{env_task}"
        logging.info(f"正在通过 Gym 注册表构建环境: {env_id}")
        eval_env = make_eval_env(env_id, obs_cameras, eval_cfg)
        env_desc = describe_eval_env(env_id, eval_env)
        logging.info(f"当前评估环境: {env_desc}")
        logging.info(f"环境加载成功！最终挂载的相机: {obs_cameras}")

        # ==========================================
        # 🌟 4.设置视频和明细输出目录
        # ==========================================
        if output_root is None:
            videos_dir = ckpt_path / "extra_eval_videos"
        else:
            videos_dir = Path(output_root)
        logging.info(f"开始测试! 录像和明细将保存在: {videos_dir}")

        # ==========================================
        # 🌟 5.调用评估函数
        # ==========================================
        with torch.autocast(device_type=device.type) if getattr(eval_cfg, "use_amp", False) else nullcontext():
            eval_info = custom_eval_policy(
                env=eval_env,
                policy=policy,
                cfg_eval=eval_cfg,
                videos_dir=videos_dir,
                device=device
            )

        # ==========================================
        # 🌟 6.整理指标，归档视频和逐 episode 明细
        # ==========================================
        sr = eval_info["aggregated"]["success_rate"]
        ar = eval_info["aggregated"]["average_reward"]
        avg_infer = eval_info["aggregated"]["avg_inference_ms"]
        max_infer = eval_info["aggregated"]["max_inference_ms"]
        ablation_tag = (
            coupling_ablation_tag(eval_cfg)
            + output_corrector_ablation_tag(eval_cfg)
            + router_ablation_tag(eval_cfg)
        )
        perturbation_spec = RecoveryPerturbationSpec.from_eval_cfg(eval_cfg)
        new_folder_name = (
            f"eval_seed={eval_cfg.seed}_ep={eval_cfg.n_episodes}"
            f"_{perturbation_spec.tag()}{ablation_tag}"
            f"_sr={sr*100:.1f}_ar={ar:.2f}"
        )
        new_videos_dir = Path(videos_dir) / new_folder_name
        new_videos_dir.mkdir(parents=True, exist_ok=True)

        for video_path in eval_info["video_paths"]:
            video_path = Path(video_path)
            if video_path.exists():
                shutil.move(str(video_path), str(new_videos_dir / video_path.name))

        episode_results_path = new_videos_dir / "episode_results.json"
        with open(episode_results_path, "w", encoding="utf-8") as f:
            json.dump(eval_info["episodes"], f, indent=2, ensure_ascii=False)
        logging.info(f"逐 episode 评估明细已保存: {episode_results_path}")

        result = {
            "status": "ok",
            "checkpoint_name": ckpt_path.name,
            "checkpoint_key": str(ckpt_path),
            "checkpoint_path": str(ckpt_path),
            "step": checkpoint_step(ckpt_path),
            "mode": eval_mode,
            "seed": eval_cfg.seed,
            "n_episodes": eval_cfg.n_episodes,
            "eval_batch_size": int(getattr(eval_env, "num_envs", 1)),
            "max_steps": eval_cfg.max_steps,
            "success_rate": float(sr),
            "success_rate_percent": float(sr * 100.0),
            "average_reward": float(ar),
            "avg_inference_ms": float(avg_infer),
            "max_inference_ms": float(max_infer),
            "perturbation_mode": perturbation_spec.mode,
            "perturbation_severity": perturbation_spec.severity,
            "perturbation_trigger_rate": float(
                eval_info["aggregated"].get("perturbation_trigger_rate", 0.0)
            ),
            "perturbation_triggered_episodes": int(
                eval_info["aggregated"].get("perturbation_triggered_episodes", 0)
            ),
            "control_recovery_rate": float(
                eval_info["aggregated"].get("control_recovery_rate", 0.0)
            ),
            "control_recovered_episodes": int(
                eval_info["aggregated"].get("control_recovered_episodes", 0)
            ),
            "mean_control_recovery_steps": eval_info["aggregated"].get(
                "mean_control_recovery_steps"
            ),
            "router_decisions": int(
                eval_info["aggregated"].get("router_decisions", 0)
            ),
            "router_activations": int(
                eval_info["aggregated"].get("router_activations", 0)
            ),
            "router_activation_rate": float(
                eval_info["aggregated"].get(
                    "router_activation_rate",
                    0.0,
                )
            ),
            "router_mean_probability": float(
                eval_info["aggregated"].get(
                    "router_mean_probability",
                    0.0,
                )
            ),
            "videos_dir": str(new_videos_dir),
            "episode_results_path": str(episode_results_path),
            "error": "",
            **active_coupling_scales,
            **active_output_corrector_scales,
            **active_router,
        }

        logging.info("="*50)
        logging.info("--独立评估完成！")
        logging.info(f"--Checkpoint: {ckpt_path.name}")
        logging.info(f"--评估模式: {eval_mode}")
        logging.info(f"--成功率 (Success Rate): {sr*100:.1f}%")
        logging.info(f"--平均奖励 (Average Reward): {ar:.2f}")
        logging.info(
            "--恢复扰动: mode=%s, severity=%g, trigger=%.1f%%, "
            "control_recovery=%.1f%%",
            perturbation_spec.mode,
            perturbation_spec.severity,
            result["perturbation_trigger_rate"] * 100.0,
            result["control_recovery_rate"] * 100.0,
        )
        logging.info(f"--平均推理时间 (Average Inference Time): {avg_infer:.2f} ms")
        logging.info(f"--最大推理时间 (Max Inference Time): {max_infer:.2f} ms")
        logging.info("="*50)
        return result
    finally:
        if eval_env is not None:
            eval_env.close()
        del policy
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()


def main(eval_cfg):
    eval_mode = getattr(eval_cfg, "mode", "fast_repro")
    if eval_mode not in {"fast_repro", "strict"}:
        raise ValueError(f"未知评估模式: {eval_mode}. 可选: fast_repro / strict")

    perturbation_spec = RecoveryPerturbationSpec.from_eval_cfg(eval_cfg)
    logging.info("独立恢复鲁棒性评估: %s", perturbation_spec.to_dict())

    deterministic = eval_mode == "strict" or bool(getattr(eval_cfg, "deterministic", DETERMINISTIC_EVAL))
    if deterministic and getattr(eval_cfg, "use_amp", False):
        logging.warning("strict/deterministic 模式下自动关闭 AMP，避免混合精度带来的数值差异。")
        eval_cfg.use_amp = False

    seed_everything(eval_cfg.seed, deterministic=deterministic)
    if deterministic:
        patch_act_position_embedding_for_determinism()

    checkpoint_source = getattr(eval_cfg, "checkpoint_source", "all")
    checkpoints, run_dir, is_batch_input = discover_checkpoints(eval_cfg.ckpt_path, checkpoint_source)

    max_checkpoints = getattr(eval_cfg, "max_checkpoints", None)
    if max_checkpoints:
        checkpoints = checkpoints[: int(max_checkpoints)]

    device = get_safe_torch_device(getattr(eval_cfg, "device", "cuda"))
    logging.info(f"初始化评估程序... 使用设备: {device}")
    logging.info(f"将评估 {len(checkpoints)} 个 checkpoint")

    has_explicit_output_dir = bool(getattr(eval_cfg, "eval_output_dir", None))
    batch_output_dir = (
        build_batch_output_dir(eval_cfg, run_dir)
        if is_batch_input or has_explicit_output_dir
        else None
    )
    if batch_output_dir is not None:
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"批量评估输出目录: {batch_output_dir}")

    rows = (
        load_existing_eval_summary(batch_output_dir, eval_cfg)
        if batch_output_dir is not None
        else []
    )
    completed_keys = completed_checkpoint_keys(rows)
    if completed_keys:
        completed_rows = sum(1 for row in rows if row.get("status") == "ok")
        logging.info(f"已发现 {completed_rows} 条成功历史结果，将跳过对应 checkpoint")

    continue_on_error = bool(getattr(eval_cfg, "continue_on_error", len(checkpoints) > 1))
    for eval_order, checkpoint_path in enumerate(tqdm(checkpoints, desc="Evaluating checkpoints"), start=1):
        checkpoint_keys = checkpoint_identity_keys(checkpoint_path)
        if batch_output_dir is not None and checkpoint_keys & completed_keys:
            logging.info(f"[{eval_order}/{len(checkpoints)}] 跳过已评估 checkpoint: {checkpoint_path.name}")
            continue

        checkpoint_cfg = make_checkpoint_eval_cfg(eval_cfg, checkpoint_path)
        output_root = batch_output_dir / checkpoint_path.name if batch_output_dir is not None else None
        logging.info(f"[{eval_order}/{len(checkpoints)}] 开始评估 checkpoint: {checkpoint_path.name}")

        try:
            result = evaluate_one_checkpoint(
                eval_cfg=checkpoint_cfg,
                checkpoint_path=checkpoint_path,
                device=device,
                output_root=output_root,
            )
            rows.append(result)
            completed_keys.update(checkpoint_keys)
            if batch_output_dir is not None:
                write_eval_summary(batch_output_dir, rows, eval_cfg, source_path=resolve_eval_path(eval_cfg.ckpt_path))
                prune_if_configured(eval_cfg, run_dir, batch_output_dir, rows, stage="单个 checkpoint 评估完成")
        except Exception as exc:
            error_row = {
                "status": "failed",
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_key": str(checkpoint_path.resolve()),
                "checkpoint_path": str(checkpoint_path),
                "step": checkpoint_step(checkpoint_path),
                "mode": eval_mode,
                "seed": eval_cfg.seed,
                "n_episodes": eval_cfg.n_episodes,
                "eval_batch_size": int(getattr(eval_cfg, "batch_size", getattr(eval_cfg, "num_envs", 1))),
                "max_steps": eval_cfg.max_steps,
                "perturbation_mode": perturbation_spec.mode,
                "perturbation_severity": perturbation_spec.severity,
                "error": repr(exc),
            }
            rows.append(error_row)
            if batch_output_dir is not None:
                write_eval_summary(batch_output_dir, rows, eval_cfg, source_path=resolve_eval_path(eval_cfg.ckpt_path))
                prune_if_configured(eval_cfg, run_dir, batch_output_dir, rows, stage="单个 checkpoint 评估失败")
            logging.exception(f"评估 checkpoint 失败: {checkpoint_path}")
            if not continue_on_error:
                raise

    if batch_output_dir is not None:
        write_eval_summary(batch_output_dir, rows, eval_cfg, source_path=resolve_eval_path(eval_cfg.ckpt_path))
        prune_if_configured(
            eval_cfg,
            run_dir,
            batch_output_dir,
            rows,
            stage="批量评估结束",
            prune_unranked_checkpoints=True,
        )
    return rows

# =========================================================================
# 🌟 独立评估测试入口，推荐使用lerobot保存的快照格式，只需要给路径，环境配置会自动对齐
# =========================================================================
if __name__ == "__main__":

    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)

    # ==========================================
    # 🎯 核心配置区：在这里自由修改你的评估参数！
    # ==========================================
    eval_cfg = SimpleNamespace(
        seed=1000,
        # 可以指向单个 checkpoint，也可以指向整次训练 run 目录或 run/checkpoints 目录。
        ckpt_path="outputs/2_pretrain/InsertPeg-3Arms-v0/baseline_expert/13-06-48_InsertPeg-3Arms-v0_dual_head_diffusion/checkpoints/144923_loss=0.0019_sr=65.0_ar=412.19",
        checkpoint_source="all",  # all: 读取目录下 checkpoint全部文件；   top_k/latest: 读取 checkpoints/top_k_records.json中记录的模型
        max_checkpoints=None,     # 调试时可设为 1/2，正式评估保持 None
        eval_output_dir=None,     # None 时自动保存到 run_dir/policy_eval/固定配置名，方便断点续评
        continue_on_error=True,   # 某个 checkpoint 失败时继续评估后面的模型
        # 鲁棒性评估不参与训练Top-K筛选，默认禁止删除任何训练快照或历史评估结果。
        keep_top_after_eval=None,
        prune_checkpoints=False,
        prune_eval_outputs=False,
        allow_prune_with_max_checkpoints=False,  # max_checkpoints 调试时默认不清理，避免误删未完整评估的模型
        
        # ⚙️ 评估参数设置
        mode="fast_repro",          # fast_repro: 快速且固定 seeds；  strict: 最强可复现但更慢
        n_episodes=100,             # 评估多少个任务                 
        max_episodes_rendered=5,    # 全量评估建议 0；需要视频时再改为 1/2
        fps=25,                     # 视频帧率，和环境控制频率对齐
        max_steps=400,              # 每个任务的最大步数
        batch_size=15,               # 并行评估环境数量；设为 1 即回到单环境评估
        use_async_envs=True,        # True 使用多进程 AsyncVectorEnv，False 使用单进程 SyncVectorEnv
        device="cuda",              # 如需完全规避 CUDA 非确定算子，可临时改成 "cpu"
        deterministic=False,        # 通常不用手动改；strict 模式会自动开启
        
        # 相机设置
        # ['zed_cam_left', 'zed_cam_right', 'overhead_cam', 'worms_eye_cam' , 'wrist_cam_left', 'wrist_cam_right'],
        render_camera=['zed_cam_left'],         # 保存video的相机视角    
        # ⚡ 快速评估默认开启混合精度；严格对比指标时可改 False
        use_amp=True,

        # ================================================================
        # 恢复鲁棒性评估： clean / arm / view / mixed / arm_view_seq （顺序：arm→view 各一次），单次运行选择一种条件。
        # 扰动只改变实际执行的关节目标；15步建立结束后不运行人工恢复控制器，
        # 也不清空策略动作队列，后续完全由策略根据真实观测自主恢复。
        # ================================================================
        recovery_perturbation="arm_view_seq",
        recovery_seed=1000,          # 扰动RNG；与环境/扩散模型RNG相互独立。
        recovery_severity=0.5,        # 0.5=mild，1.0=训练同分布，1.5=分布外压力测试。
        recovery_setup_steps=20,      # 五次smoothstep从0平滑建立到完整偏移的步数。
        recovery_hold_steps=0,        # 建立后保持完整偏移的步数(0=立即释放)；>0测持续扰动下的纠正能力
        recovery_align_to_chunk=True,  # True对齐chunk边界测恢复能力；False随机注入测端到端鲁棒性
        recovery_annotate_video=True,  # 视频帧叠加边框标注：红=扰动建立，蓝=释放后恢复
        recovery_exclude_initial_steps=16,

        # 若填写[start,end]闭区间则优先按绝对step触发；None时使用下方归一化区间。
        recovery_trigger_step_range=None,
        recovery_trigger_normalized_range=(0.1, 0.5),
        # None表示以上比例相对max_steps；SewNeedle若要严格对齐约310帧专家轨迹可设310。
        recovery_trigger_reference_steps=310,

        # local_motion根据触发前最近16帧真实state速度选择主运动臂；不会扰动夹爪。
        recovery_arm_selection="local_motion",  # local_motion/alternate/random/left/right
        recovery_motion_window_steps=16,
        # Mixed下Arm/View各缩放1/sqrt(2)，使总扰动能量与单角色近似一致。
        recovery_mixed_equal_energy=True,

        # 与当前Arm/View恢复数据生成配置一致的截断高斯，单位rad。
        recovery_std_rad=(0.016, 0.016, 0.016, 0.026, 0.026, 0.032),
        recovery_max_abs_rad=(0.040, 0.040, 0.040, 0.065, 0.065, 0.080),
        recovery_min_normalized_l2=0.5,
        recovery_joint_limit_margin_rad=0.005,
        recovery_max_sampling_attempts=1000,

        # “控制恢复”仅作为辅助指标：释放后实际关节连续3帧贴近策略当前目标。
        # 最终任务成功率仍是鲁棒性评估的主要指标。
        recovery_success_max_abs_error_rad=0.004,
        recovery_success_stable_steps=3,

        # 耦合推理消融；None沿用checkpoint，0关闭对应方向，1保持完整耦合。
        view_to_arm_coupling_scale=None,
        arm_to_view_coupling_scale=None,
        # 最终输出修正消融；None沿用checkpoint，0关闭对应方向。
        view_to_arm_output_scale=None,
        arm_to_view_output_scale=None,
    )
    if eval_cfg.deterministic or eval_cfg.mode == "strict" or DETERMINISTIC_EVAL:
        ensure_python_hash_seed(eval_cfg.seed)
    # ==========================================
    # 启动
    main(eval_cfg=eval_cfg)
                                         
