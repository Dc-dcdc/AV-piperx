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
from types import SimpleNamespace
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT_PATH = Path(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

if __package__:
    from .action_value_critic import (
        ActionValueCritic,
        ActionValueCriticConfig,
        RelativeReplanningConfig,
        RelativeValueReplanningDecider,
    )
    from ..s1_pretrain.eval.coupling_ablation import (
        apply_coupling_ablation_overrides,
        coupling_ablation_tag,
    )
    from ..s1_pretrain.eval.vector_info import as_bool_array as _as_bool_array
    from ..s1_pretrain.eval.vector_info import extract_info_bool as _extract_info_bool
else:
    from train.s4_adaptive_replanning.action_value_critic import (
        ActionValueCritic,
        ActionValueCriticConfig,
        RelativeReplanningConfig,
        RelativeValueReplanningDecider,
    )
    from train.s1_pretrain.eval.coupling_ablation import (
        apply_coupling_ablation_overrides,
        coupling_ablation_tag,
    )
    from train.s1_pretrain.eval.vector_info import as_bool_array as _as_bool_array
    from train.s1_pretrain.eval.vector_info import extract_info_bool as _extract_info_bool

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


def resolve_critic_checkpoint_path(path_like) -> Path:
    """解析Critic文件、checkpoints目录或完整训练目录。"""
    if path_like is None or str(path_like).strip().lower() in {
        "",
        "none",
        "null",
    }:
        raise ValueError("必须设置critic_ckpt_path")
    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = ROOT_PATH / path
    path = path.resolve()
    if path.is_file():
        return path
    candidates = (
        path / "best.pt",
        path / "latest.pt",
        path / "checkpoints" / "best.pt",
        path / "checkpoints" / "latest.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"找不到Critic快照: {path}；支持具体.pt、checkpoints目录或训练目录"
    )


def make_eval_replanning_config(cfg_eval) -> RelativeReplanningConfig:
    """从动态评估配置构造相对价值触发参数。"""
    return RelativeReplanningConfig(
        gamma=float(getattr(cfg_eval, "critic_gamma", 0.99)),
        anchor_ratio_threshold=float(
            getattr(cfg_eval, "critic_anchor_ratio_threshold", 0.70)
        ),
        local_drop_threshold=float(
            getattr(cfg_eval, "critic_local_drop_threshold", 0.15)
        ),
        consecutive_bad_steps=int(
            getattr(cfg_eval, "critic_consecutive_bad_steps", 2)
        ),
        ema_alpha=float(getattr(cfg_eval, "critic_ema_alpha", 0.20)),
        min_reference_q=float(
            getattr(cfg_eval, "critic_min_reference_q", 0.05)
        ),
    )


def critic_variant_key(cfg_eval) -> str:
    """生成断点续评标识，避免不同Critic配置互相误判为已完成。"""
    raw_path = str(getattr(cfg_eval, "critic_ckpt_path", ""))
    config = make_eval_replanning_config(cfg_eval)
    return (
        f"critic=on::ckpt={raw_path}"
        f"::min_steps={int(getattr(cfg_eval, 'min_execution_steps', 1))}"
        f"::max_steps={int(getattr(cfg_eval, 'max_execution_steps', 8))}"
        f"::gamma={config.gamma:g}"
        f"::anchor={config.anchor_ratio_threshold:g}"
        f"::local={config.local_drop_threshold:g}"
        f"::consecutive={config.consecutive_bad_steps}"
        f"::ema={config.ema_alpha:g}"
        f"::min_ref={config.min_reference_q:g}"
    )


def critic_output_tag(cfg_eval) -> str:
    """生成简短目录标签；完整Critic配置仍会写入JSON结果。"""
    config = make_eval_replanning_config(cfg_eval)
    return (
        f"_critic=on"
        f"_min={int(getattr(cfg_eval, 'min_execution_steps', 1))}"
        f"_max={int(getattr(cfg_eval, 'max_execution_steps', 8))}"
        f"_qar={config.anchor_ratio_threshold:g}"
        f"_qdrop={config.local_drop_threshold:g}"
        f"_qbad={config.consecutive_bad_steps}"
    )


def effective_eval_batch_size(cfg_eval) -> int:
    """Critic需要为每个episode维护独立动作队列，当前固定为单环境。"""
    del cfg_eval
    return 1


def load_action_value_critic(cfg_eval, policy, device):
    """加载与当前策略输入契约一致的sigmoid动作价值Critic。"""
    checkpoint_path = resolve_critic_checkpoint_path(
        getattr(cfg_eval, "critic_ckpt_path", None)
    )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    stored_config = dict(payload.get("critic_config", {}))
    if not stored_config:
        raise KeyError(f"Critic快照缺少critic_config: {checkpoint_path}")
    stored_activation = str(
        stored_config.get("output_activation", "identity")
    )
    # 兼容本项目已经训练出的旧版线性Critic；其结构相同，只是快照里没有
    # output_activation/initial_q。二值终止回报的理论Q仍位于[0,1]，
    # 在线判定前会进行安全裁剪。
    stored_config["output_activation"] = stored_activation
    stored_config.setdefault("initial_q", 0.05)
    stored_config["hidden_dims"] = tuple(stored_config["hidden_dims"])
    critic_config = ActionValueCriticConfig(**stored_config)
    critic = ActionValueCritic(critic_config)
    critic.load_state_dict(payload["online_critic"])
    critic.to(device)
    critic.requires_grad_(False)
    critic.eval()
    critic._adaptive_replanning_clamp_q = (
        stored_activation == "identity"
    )
    if critic._adaptive_replanning_clamp_q:
        logging.warning(
            "正在使用旧版identity Critic；在线评分会裁剪到[0,1]。"
            "后续重新训练时建议使用当前默认sigmoid Critic。"
        )

    if not hasattr(policy, "diffusion") or not hasattr(
        policy.diffusion,
        "rgb_encoder",
    ):
        raise TypeError(
            "Critic动态重规划要求策略提供diffusion.rgb_encoder视觉底座"
        )
    expected_camera_names = [
        key.removeprefix("observation.images.")
        for key in policy.expected_image_keys
    ]
    stored_camera_names = list(payload.get("camera_names", ()))
    if stored_camera_names and stored_camera_names != expected_camera_names:
        raise ValueError(
            "Critic与策略相机顺序不一致: "
            f"critic={stored_camera_names}, policy={expected_camera_names}"
        )
    expected_action_dim = int(policy.config.output_shapes["action"][0])
    if critic_config.action_dim != expected_action_dim:
        raise ValueError(
            "Critic与策略action维度不一致: "
            f"critic={critic_config.action_dim}, policy={expected_action_dim}"
        )
    stored_policy_path = payload.get("pretrained_model_path")
    if stored_policy_path:
        active_policy_path = Path(
            str(getattr(cfg_eval, "_active_policy_model_path", ""))
        )
        if (
            str(active_policy_path)
            and active_policy_path.expanduser().resolve()
            != Path(stored_policy_path).expanduser().resolve()
        ):
            logging.warning(
                "Critic记录的预训练策略与当前待评估策略路径不同；"
                "结构虽一致，但Q分布可能失配: critic_policy=%s, eval_policy=%s",
                stored_policy_path,
                active_policy_path,
            )

    trained_steps = [
        int(value)
        for value in payload.get("training_execution_steps", ())
    ]
    critic.training_execution_steps = tuple(trained_steps)
    replanning_config = make_eval_replanning_config(cfg_eval)
    logging.info(
        "Critic已加载: %s, trained_steps=%s, "
        "anchor_ratio=%.3f, local_drop=%.3f, consecutive=%d",
        checkpoint_path,
        trained_steps,
        replanning_config.anchor_ratio_threshold,
        replanning_config.local_drop_threshold,
        replanning_config.consecutive_bad_steps,
    )
    return critic, replanning_config, checkpoint_path


def copy_critic_observation(
    observation: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """只保留Critic需要的当前原始策略输入，作为下一步历史帧。"""
    return {
        key: value.detach()
        for key, value in observation.items()
        if key == "observation.state"
        or key.startswith("observation.images.")
    }


@torch.no_grad()
def score_action_with_critic(
    policy,
    critic: ActionValueCritic,
    previous_observation: dict[str, torch.Tensor] | None,
    current_observation: dict[str, torch.Tensor],
    action: torch.Tensor,
) -> float:
    """实时编码最新两帧观测和待执行动作，返回单环境Q值。"""
    if action.ndim == 1:
        action = action.unsqueeze(0)
    if int(action.shape[0]) != 1:
        raise ValueError("Critic动态重规划当前只支持batch_size=1")
    previous = (
        current_observation
        if previous_observation is None
        else previous_observation
    )
    state_key = "observation.state"
    if state_key not in previous or state_key not in current_observation:
        raise KeyError("Critic评分缺少observation.state")

    history_inputs = {
        state_key: torch.stack(
            [previous[state_key], current_observation[state_key]],
            dim=1,
        ).flatten(0, 1)
    }
    for image_key in policy.expected_image_keys:
        if image_key not in previous or image_key not in current_observation:
            raise KeyError(f"Critic评分缺少相机输入: {image_key}")
        history_inputs[image_key] = torch.stack(
            [previous[image_key], current_observation[image_key]],
            dim=1,
        ).flatten(0, 1)

    normalized_inputs = policy.normalize_inputs(history_inputs)
    normalized_state = normalized_inputs[state_key].reshape(1, 2, -1)
    normalized_action = policy.normalize_targets(
        {"action": action.unsqueeze(1)}
    )["action"][:, 0]
    camera_images = torch.stack(
        [
            normalized_inputs[image_key]
            for image_key in policy.expected_image_keys
        ],
        dim=1,
    )
    visual_features = policy.diffusion.rgb_encoder(
        camera_images.flatten(0, 1)
    )
    visual_features = visual_features.reshape(
        1,
        2,
        len(policy.expected_image_keys),
        -1,
    ).flatten(start_dim=1)
    joint_history = normalized_state.flatten(start_dim=1)
    q_value = critic(
        visual_features,
        joint_history,
        normalized_action,
    )
    q_value = q_value.float()
    if bool(getattr(critic, "_adaptive_replanning_clamp_q", False)):
        q_value = q_value.clamp(0.0, 1.0)
    return float(q_value.item())


def is_vector_env(env) -> bool:
    return int(getattr(env, "num_envs", 1)) > 1


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


def custom_eval_policy_vectorized(
    env,
    policy,
    cfg_eval,
    videos_dir,
    device,
    critic=None,
    replanning_config=None,
):
    """用 gym.vector 并行评估多个 episode；policy action chunk 队列按 batch 同步推进。"""
    if critic is not None:
        raise ValueError(
            "Critic动态重规划需要为每个环境维护独立动作队列，"
            "当前仅支持batch_size=1"
        )
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
    execution_limit = int(getattr(policy.config, "n_action_steps", 1))
    global_real_inference_times = []

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

            batch_obs = prepare_policy_observation(obs, expected_keys, device)
            start_time = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch_obs)
            action_np = action.detach().cpu().numpy().copy()
            if action_np.ndim == 1:
                action_np = action_np[None, :]
            if action_np.shape[0] != n_envs:
                raise ValueError(f"策略输出 batch={action_np.shape[0]}，但评估环境 num_envs={n_envs}")
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
            episode_records.append(
                {
                    "episode": ep,
                    "seed": base_seed + ep,
                    "success": success,
                    "reward": ep_reward,
                    "steps": int(steps_taken[local_idx]),
                    "action_chunk_capacity": execution_limit,
                }
            )

            if local_idx in frames_by_env:
                status = "Success" if success else "Fail"
                for camera, frames in frames_by_env[local_idx].items():
                    if len(frames) == 0:
                        continue
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

    return {
        "aggregated": {
            "success_rate": float(np.mean(successes)),
            "average_reward": float(np.mean(rewards)),
            "avg_inference_ms": float(avg_real_time),
            "max_inference_ms": float(max_time),
            "action_chunk_capacity": execution_limit,
        },
        "video_paths": saved_video_paths,
        "episodes": episode_records,
    }


def custom_eval_policy(
    env,
    policy,
    cfg_eval,
    videos_dir,
    device,
    critic: ActionValueCritic | None = None,
    replanning_config: RelativeReplanningConfig | None = None,
):
    """
    完全自主实现的评估代码。没有任何黑盒。
    接收标准 Gym 环境，处理图像归一化，跑策略推理，保存视频。
    """
    if is_vector_env(env):
        return custom_eval_policy_vectorized(
            env,
            policy,
            cfg_eval,
            videos_dir,
            device,
            critic=critic,
            replanning_config=replanning_config,
        )
    if critic is None or replanning_config is None:
        raise ValueError(
            "eval_dynamic_steps.py是纯Critic评估入口，"
            "必须同时加载critic和replanning_config"
        )

    policy.eval() # 必须开启评估模式
    if critic is not None:
        critic.eval()
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
    execution_limit = int(getattr(policy.config, "n_action_steps", 1))
    min_execution_steps = int(
        getattr(cfg_eval, "min_execution_steps", 1)
    )
    max_execution_steps = int(
        getattr(cfg_eval, "max_execution_steps", execution_limit)
    )
    if execution_limit != max_execution_steps:
        raise ValueError(
            "策略动作队列容量必须等于max_execution_steps: "
            f"queue={execution_limit}, max={max_execution_steps}"
        )
    if not 1 <= min_execution_steps <= max_execution_steps:
        raise ValueError(
            "执行边界必须满足1 <= min_execution_steps <= "
            f"max_execution_steps，当前为[{min_execution_steps}, "
            f"{max_execution_steps}]"
        )
    # 计算所有评估回合推理的总耗时
    global_real_inference_times = []
    global_critic_times = []
    global_executed_q_values = []
    global_anchor_ratios = []
    global_td_changes = []
    global_q_improvements = []
    global_inference_intervals = []
    total_critic_replans = 0
    total_policy_inferences = 0
    total_natural_inferences = 0
    episodes_with_critic_replan = 0
    total_max_forced_replans = 0
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

        # 🌟 1. 每回合新建一个列表，静默记录本回合的所有耗时
        ep_inference_times = []
        ep_critic_times = []
        ep_executed_q_values = []
        ep_anchor_ratios = []
        ep_td_changes = []
        ep_q_improvements = []
        ep_trigger_steps = []
        ep_max_replan_steps = []
        ep_critic_trace = []
        ep_policy_inferences = 0
        ep_natural_inferences = 0
        ep_policy_inference_steps = []
        previous_critic_observation = None
        actions_executed_since_inference = 0
        replanning_decider = (
            RelativeValueReplanningDecider(replanning_config)
            if critic is not None
            else None
        )
        steps_taken = 0
        info = {"is_success": False}
        for step in range(max_steps):
            steps_taken = step + 1
            # 1. 如果还在需要渲染的额度内，才调用渲染 (提升非渲染 episode 的评估速度)
            if ep < max_rendered:
                for camera in render_cameras:
                    frames_by_camera[camera].append(env.unwrapped.render([camera])) # gym创建需要加上 .unwrapped

            # 2. 只转换模型真正需要的输入特征，推入 GPU。
            policy_observation = prepare_policy_observation(
                obs,
                expected_keys,
                device,
            )
            action_queue = policy._queues.get("action")
            if action_queue is None:
                raise RuntimeError("策略缺少action chunk队列，无法进行动态重规划")
            queue_was_empty = len(action_queue) == 0
            critic_triggered = False
            max_forced_replan = (
                actions_executed_since_inference
                >= max_execution_steps
            )
            if max_forced_replan:
                action_queue.clear()
                queue_was_empty = True
                ep_max_replan_steps.append(int(step))
            cached_q = None
            critic_decision = None

            # Critic只检查尚未执行的缓存动作；满足相对下降条件时先清空动作块，
            # 随后的select_action会用当前最新观测重新生成一整段动作。
            if not queue_was_empty:
                critic_start = time.perf_counter()
                cached_q = score_action_with_critic(
                    policy,
                    critic,
                    previous_critic_observation,
                    policy_observation,
                    action_queue[0],
                )
                ep_critic_times.append(
                    (time.perf_counter() - critic_start) * 1000.0
                )
                critic_decision = replanning_decider.evaluate(cached_q)
                ep_anchor_ratios.append(critic_decision.anchor_ratio)
                ep_td_changes.append(
                    critic_decision.normalized_td_change
                )
                min_steps_satisfied = (
                    actions_executed_since_inference
                    >= min_execution_steps
                )
                if (
                    critic_decision.should_replan
                    and min_steps_satisfied
                ):
                    critic_triggered = True
                    action_queue.clear()
                    ep_trigger_steps.append(step)

            # ==========================================
            # ⏱️ 开始计时：使用高精度的 perf_counter
            # ==========================================
            start_time = time.perf_counter()

            # 3. 推理获取动作
            with torch.inference_mode():
                # 使用lerobot自带的推理函数，obs是单帧的，模型会自动处理历史动作的拼接和缓存
                action = policy.select_action(policy_observation)
            # 4. 把模型输出的 Tensor 动作转回 Numpy (包含在计时内)
            action_np = action.squeeze(0).cpu().numpy()

            # ⏱️ 结束计时
            inference_time_ms = (time.perf_counter() - start_time) * 1000 # 转换为毫秒 (ms)
            ep_inference_times.append(inference_time_ms)
            generated_new_chunk = queue_was_empty or critic_triggered
            if generated_new_chunk:
                ep_policy_inferences += 1
                ep_policy_inference_steps.append(int(step))
                if not critic_triggered:
                    ep_natural_inferences += 1

            if generated_new_chunk:
                critic_start = time.perf_counter()
                executed_q = score_action_with_critic(
                    policy,
                    critic,
                    previous_critic_observation,
                    policy_observation,
                    action,
                )
                ep_critic_times.append(
                    (time.perf_counter() - critic_start) * 1000.0
                )
                replanning_decider.start_chunk(executed_q)
                if cached_q is not None:
                    ep_q_improvements.append(executed_q - cached_q)
            else:
                executed_q = cached_q
            ep_executed_q_values.append(float(executed_q))
            ep_critic_trace.append(
                {
                    "step": int(step),
                    "action_source": (
                        "critic_replanned"
                        if critic_triggered
                        else (
                            "max_steps_replanned"
                            if max_forced_replan
                            else (
                                "natural_inference"
                                if queue_was_empty
                                else "cached_chunk"
                            )
                        )
                    ),
                    "actions_executed_before_action": int(
                        actions_executed_since_inference
                    ),
                    "min_steps_satisfied": bool(
                        actions_executed_since_inference
                        >= min_execution_steps
                    ),
                    "max_steps_reached": bool(max_forced_replan),
                    "cached_q": (
                        float(cached_q)
                        if cached_q is not None
                        else None
                    ),
                    "executed_q": float(executed_q),
                    "replanned": bool(critic_triggered),
                    "anchor_ratio": (
                        float(critic_decision.anchor_ratio)
                        if critic_decision is not None
                        else 1.0
                    ),
                    "normalized_td_change": (
                        float(
                            critic_decision.normalized_td_change
                        )
                        if critic_decision is not None
                        else 0.0
                    ),
                    "anchor_bad": (
                        bool(critic_decision.anchor_bad)
                        if critic_decision is not None
                        else False
                    ),
                    "local_bad": (
                        bool(critic_decision.local_bad)
                        if critic_decision is not None
                        else False
                    ),
                    "consecutive_bad_steps": (
                        int(
                            critic_decision.consecutive_bad_steps
                        )
                        if critic_decision is not None
                        else 0
                    ),
                }
            )
            previous_critic_observation = copy_critic_observation(
                policy_observation
            )
            if generated_new_chunk:
                actions_executed_since_inference = 0
            actions_executed_since_inference += 1
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
        critic_replans = len(ep_trigger_steps)
        if critic_replans:
            episodes_with_critic_replan += 1
        total_critic_replans += critic_replans
        total_max_forced_replans += len(ep_max_replan_steps)
        total_policy_inferences += ep_policy_inferences
        total_natural_inferences += ep_natural_inferences
        global_critic_times.extend(ep_critic_times)
        global_executed_q_values.extend(ep_executed_q_values)
        global_anchor_ratios.extend(ep_anchor_ratios)
        global_td_changes.extend(ep_td_changes)
        global_q_improvements.extend(ep_q_improvements)
        ep_inference_intervals = [
            current_step - previous_step
            for previous_step, current_step in zip(
                ep_policy_inference_steps,
                ep_policy_inference_steps[1:],
            )
        ]
        global_inference_intervals.extend(ep_inference_intervals)
        episode_records.append(
            {
                "episode": ep,
                "seed": ep_seed,
                "success": bool(info.get("is_success", False)),
                "reward": float(ep_reward),
                "steps": int(steps_taken),
                "action_chunk_capacity": execution_limit,
                "critic_enabled": critic is not None,
                "critic_replans": critic_replans,
                "critic_trigger_steps": ep_trigger_steps,
                "max_forced_replan_steps": ep_max_replan_steps,
                "critic_trace": ep_critic_trace,
                "min_execution_steps": min_execution_steps,
                "max_execution_steps": max_execution_steps,
                "critic_mean_q": (
                    float(np.mean(ep_executed_q_values))
                    if ep_executed_q_values
                    else None
                ),
                "critic_min_q": (
                    float(np.min(ep_executed_q_values))
                    if ep_executed_q_values
                    else None
                ),
                "critic_mean_anchor_ratio": (
                    float(np.mean(ep_anchor_ratios))
                    if ep_anchor_ratios
                    else None
                ),
                "critic_mean_normalized_td_change": (
                    float(np.mean(ep_td_changes))
                    if ep_td_changes
                    else None
                ),
                "critic_mean_replan_q_improvement": (
                    float(np.mean(ep_q_improvements))
                    if ep_q_improvements
                    else None
                ),
                "critic_evaluations": len(ep_critic_times),
                "critic_avg_ms": (
                    float(np.mean(ep_critic_times))
                    if ep_critic_times
                    else 0.0
                ),
                "policy_inference_count": ep_policy_inferences,
                "natural_inference_count": ep_natural_inferences,
                "policy_inference_steps": ep_policy_inference_steps,
                "mean_policy_inference_interval": (
                    float(np.mean(ep_inference_intervals))
                    if ep_inference_intervals
                    else None
                ),
                "min_policy_inference_interval": (
                    int(min(ep_inference_intervals))
                    if ep_inference_intervals
                    else None
                ),
                "max_policy_inference_interval": (
                    int(max(ep_inference_intervals))
                    if ep_inference_intervals
                    else None
                ),
            }
        )

        # 6. 根据配置的帧率和最大渲染数量保存视频
        if ep < max_rendered and frames_by_camera:
            status = "Success" if successes[ep] else "Fail"
            for camera, frames in frames_by_camera.items():
                if len(frames) == 0:
                    continue
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
    return {
        "aggregated": {
            "success_rate": float(np.mean(successes)),
            "average_reward": float(np.mean(rewards)),
            "avg_inference_ms": float(avg_real_time),
            "max_inference_ms": float(max_time),
            "action_chunk_capacity": execution_limit,
            "critic_enabled": critic is not None,
            "critic_total_replans": int(total_critic_replans),
            "max_forced_replans": int(total_max_forced_replans),
            "min_execution_steps": min_execution_steps,
            "max_execution_steps": max_execution_steps,
            "critic_replans_per_episode": (
                float(total_critic_replans / n_episodes)
                if n_episodes
                else 0.0
            ),
            "critic_trigger_episode_rate": (
                float(episodes_with_critic_replan / n_episodes)
                if n_episodes
                else 0.0
            ),
            "critic_mean_q": (
                float(np.mean(global_executed_q_values))
                if global_executed_q_values
                else None
            ),
            "critic_min_q": (
                float(np.min(global_executed_q_values))
                if global_executed_q_values
                else None
            ),
            "critic_mean_anchor_ratio": (
                float(np.mean(global_anchor_ratios))
                if global_anchor_ratios
                else None
            ),
            "critic_mean_normalized_td_change": (
                float(np.mean(global_td_changes))
                if global_td_changes
                else None
            ),
            "critic_mean_replan_q_improvement": (
                float(np.mean(global_q_improvements))
                if global_q_improvements
                else None
            ),
            "critic_evaluations": len(global_critic_times),
            "critic_avg_ms": (
                float(np.mean(global_critic_times))
                if global_critic_times
                else 0.0
            ),
            "policy_inference_count": int(total_policy_inferences),
            "natural_inference_count": int(total_natural_inferences),
            "mean_policy_inference_interval": (
                float(np.mean(global_inference_intervals))
                if global_inference_intervals
                else None
            ),
            "min_policy_inference_interval": (
                int(min(global_inference_intervals))
                if global_inference_intervals
                else None
            ),
            "max_policy_inference_interval": (
                int(max(global_inference_intervals))
                if global_inference_intervals
                else None
            ),
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


def configure_policy_for_critic(
    policy,
    critic: ActionValueCritic,
    min_execution_steps: int,
    max_execution_steps: int,
) -> tuple[int, int, int]:
    """配置Critic调度边界，并用最大执行步数设置动作队列容量。"""

    config = getattr(policy, "config", None)
    if config is None or not hasattr(config, "n_action_steps"):
        raise ValueError("当前策略配置不包含n_action_steps，无法建立动作队列。")

    checkpoint_limit = int(config.n_action_steps)
    min_execution_steps = int(min_execution_steps)
    max_execution_steps = int(max_execution_steps)
    if min_execution_steps <= 0:
        raise ValueError(
            "min_execution_steps必须大于0，"
            f"当前为{min_execution_steps}"
        )
    if max_execution_steps < min_execution_steps:
        raise ValueError(
            "max_execution_steps必须大于等于min_execution_steps，"
            f"当前为min={min_execution_steps}, max={max_execution_steps}"
        )
    trained_steps = tuple(
        int(value)
        for value in getattr(critic, "training_execution_steps", ())
    )
    active_limit = max_execution_steps
    if active_limit <= 0:
        raise ValueError(f"Critic动作块容量必须为正整数，当前为{active_limit}。")

    upper_bounds = []
    horizon = getattr(config, "horizon", None)
    if horizon is not None:
        action_start = max(0, int(getattr(config, "n_obs_steps", 1)) - 1)
        horizon_limit = int(horizon) - action_start
        if getattr(config, "coupling_mode", None) in {
            "rbac",
            "bidirectional_prefix_to_suffix",
        }:
            # 这些耦合方式要求执行前缀后至少保留一个未执行的 suffix action。
            horizon_limit -= 1
        upper_bounds.append(("horizon", horizon_limit))

    chunk_size = getattr(config, "chunk_size", None)
    if chunk_size is not None:
        upper_bounds.append(("chunk_size", int(chunk_size)))

    if (
        getattr(config, "temporal_ensemble_coeff", None) is not None
        and active_limit != 1
    ):
        raise ValueError("ACT temporal ensembling 模式只支持 n_action_steps=1。")

    if upper_bounds:
        limit_name, model_limit = min(
            upper_bounds,
            key=lambda item: item[1],
        )
        if active_limit > model_limit:
            raise ValueError(
                f"Critic动作块容量{active_limit}超过模型允许上限"
                f"{model_limit}（由{limit_name}决定）。"
            )

    config.n_action_steps = active_limit
    policy.reset()
    if trained_steps and active_limit not in trained_steps:
        logging.warning(
            "max_execution_steps不在Critic训练步长中，可能存在分布偏移: "
            "trained=%s, active=%d",
            list(trained_steps),
            active_limit,
        )
    logging.info(
        "Critic执行边界已生效: min=%d, max=%d, "
        "checkpoint_chunk=%d, critic_trained_steps=%s",
        min_execution_steps,
        active_limit,
        checkpoint_limit,
        list(trained_steps),
    )
    return checkpoint_limit, min_execution_steps, active_limit


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
    ablation_tag = coupling_ablation_tag(eval_cfg)
    critic_tag = critic_output_tag(eval_cfg)
    return (
        run_dir
        / "critic_eval"
        / (
            f"{source}_{mode}_{precision}_seed={eval_cfg.seed}_ep={eval_cfg.n_episodes}"
            f"_steps={eval_cfg.max_steps}"
            f"{ablation_tag}{critic_tag}"
        )
    ).resolve()


def make_checkpoint_eval_cfg(
    eval_cfg,
    checkpoint_path: Path,
):
    checkpoint_cfg = SimpleNamespace(**vars(eval_cfg))
    checkpoint_cfg.ckpt_path = str(checkpoint_path)
    if isinstance(getattr(eval_cfg, "render_camera", None), list):
        checkpoint_cfg.render_camera = list(eval_cfg.render_camera)
    return checkpoint_cfg


def checkpoint_identity_keys(
    checkpoint_path: Path,
    critic_variant: str = "",
) -> set[str]:
    checkpoint_path = checkpoint_path.resolve()
    suffix = f"::{critic_variant}"
    return {
        f"{checkpoint_path}{suffix}",
        f"{checkpoint_path.name}{suffix}",
    }


def row_identity_keys(row: dict) -> set[str]:
    keys = set()
    suffix = f"::{row.get('critic_variant', '')}"
    for field in ("checkpoint_key", "checkpoint_path", "checkpoint_name"):
        value = row.get(field)
        if not value:
            continue
        keys.add(f"{value}{suffix}")
        if field == "checkpoint_path":
            try:
                keys.add(f"{resolve_eval_path(value)}{suffix}")
            except Exception:
                pass
    return keys


def load_existing_eval_summary(summary_dir: Path) -> list[dict]:
    summary_path = summary_dir / "critic_eval_summary.json"
    if not summary_path.exists():
        return []

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logging.warning(f"读取已有评估汇总失败，将从空表继续: {summary_path} ({exc})")
        return []

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

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_path": str(source_path),
        "checkpoint_source": getattr(eval_cfg, "checkpoint_source", "all"),
        "mode": getattr(eval_cfg, "mode", "fast_repro"),
        "seed": eval_cfg.seed,
        "n_episodes": eval_cfg.n_episodes,
        "eval_batch_size": effective_eval_batch_size(eval_cfg),
        "max_steps": eval_cfg.max_steps,
        "min_execution_steps": int(eval_cfg.min_execution_steps),
        "max_execution_steps": int(eval_cfg.max_execution_steps),
        "critic": {
            "enabled": True,
            "checkpoint_path": getattr(
                eval_cfg,
                "critic_ckpt_path",
                None,
            ),
            "variant": critic_variant_key(eval_cfg),
            "gamma": float(getattr(eval_cfg, "critic_gamma", 0.99)),
            "anchor_ratio_threshold": float(
                getattr(
                    eval_cfg,
                    "critic_anchor_ratio_threshold",
                    0.70,
                )
            ),
            "local_drop_threshold": float(
                getattr(
                    eval_cfg,
                    "critic_local_drop_threshold",
                    0.15,
                )
            ),
            "consecutive_bad_steps": int(
                getattr(
                    eval_cfg,
                    "critic_consecutive_bad_steps",
                    2,
                )
            ),
            "ema_alpha": float(
                getattr(eval_cfg, "critic_ema_alpha", 0.20)
            ),
            "min_reference_q": float(
                getattr(
                    eval_cfg,
                    "critic_min_reference_q",
                    0.05,
                )
            ),
        },
        "total_records": len(rows),
        "latest_checkpoint_records": len(latest_rows),
        "best": ranked_rows[0] if ok_rows else None,
        "ranking": ranked_rows,
        "results": rows,
    }
    json_path = summary_dir / "critic_eval_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = summary_dir / "critic_eval_summary.csv"
    fieldnames = [
        "rank",
        "status",
        "checkpoint_name",
        "checkpoint_key",
        "step",
        "action_chunk_capacity",
        "checkpoint_action_chunk_capacity",
        "success_rate",
        "success_rate_percent",
        "average_reward",
        "avg_inference_ms",
        "max_inference_ms",
        "critic_enabled",
        "critic_total_replans",
        "max_forced_replans",
        "min_execution_steps",
        "max_execution_steps",
        "critic_replans_per_episode",
        "critic_trigger_episode_rate",
        "critic_mean_q",
        "critic_min_q",
        "critic_mean_anchor_ratio",
        "critic_mean_normalized_td_change",
        "critic_mean_replan_q_improvement",
        "critic_evaluations",
        "critic_avg_ms",
        "policy_inference_count",
        "natural_inference_count",
        "mean_policy_inference_interval",
        "min_policy_inference_interval",
        "max_policy_inference_interval",
        "critic_checkpoint_path",
        "critic_variant",
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

    logging.info(f"Critic评估汇总JSON已保存: {json_path}")
    logging.info(f"Critic评估汇总CSV已保存: {csv_path}")
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

    report_path = summary_dir / "critic_eval_prune_report.json"
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
    critic = None
    replanning_config = None
    critic_checkpoint_path = None
    active_coupling_scales = {}
    checkpoint_execution_limit = None
    active_min_execution_steps = None
    active_execution_limit = None

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
            eval_cfg._active_policy_model_path = str(
                Path(load_dir).resolve()
            )
            critic, replanning_config, critic_checkpoint_path = (
                load_action_value_critic(
                    eval_cfg,
                    policy,
                    device,
                )
            )
            (
                checkpoint_execution_limit,
                active_min_execution_steps,
                active_execution_limit,
            ) = configure_policy_for_critic(
                policy,
                critic,
                int(getattr(eval_cfg, "min_execution_steps", 1)),
                int(getattr(eval_cfg, "max_execution_steps", 8)),
            )
            eval_cfg.batch_size = 1
            if hasattr(eval_cfg, "num_envs"):
                eval_cfg.num_envs = 1
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
                device=device,
                critic=critic,
                replanning_config=replanning_config,
            )

        # ==========================================
        # 🌟 6.整理指标，归档视频和逐 episode 明细
        # ==========================================
        sr = eval_info["aggregated"]["success_rate"]
        ar = eval_info["aggregated"]["average_reward"]
        avg_infer = eval_info["aggregated"]["avg_inference_ms"]
        max_infer = eval_info["aggregated"]["max_inference_ms"]
        ablation_tag = coupling_ablation_tag(eval_cfg)
        critic_tag = critic_output_tag(eval_cfg)
        critic_replans = int(
            eval_info["aggregated"].get("critic_total_replans", 0)
        )
        max_forced_replans = int(
            eval_info["aggregated"].get("max_forced_replans", 0)
        )
        new_folder_name = (
            f"eval_seed={eval_cfg.seed}_ep={eval_cfg.n_episodes}"
            f"{ablation_tag}{critic_tag}"
            f"_critic_replans={critic_replans}"
            f"_max_replans={max_forced_replans}"
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
            "action_chunk_capacity": int(active_execution_limit),
            "checkpoint_action_chunk_capacity": int(
                checkpoint_execution_limit
            ),
            "min_execution_steps": int(active_min_execution_steps),
            "max_execution_steps": int(active_execution_limit),
            "success_rate": float(sr),
            "success_rate_percent": float(sr * 100.0),
            "average_reward": float(ar),
            "avg_inference_ms": float(avg_infer),
            "max_inference_ms": float(max_infer),
            "critic_checkpoint_path": (
                str(critic_checkpoint_path)
                if critic_checkpoint_path is not None
                else None
            ),
            "critic_variant": critic_variant_key(eval_cfg),
            "videos_dir": str(new_videos_dir),
            "episode_results_path": str(episode_results_path),
            "error": "",
            **{
                key: value
                for key, value in eval_info["aggregated"].items()
                if key.startswith("critic_")
                or key
                in {
                    "policy_inference_count",
                    "natural_inference_count",
                    "max_forced_replans",
                    "min_execution_steps",
                    "max_execution_steps",
                    "mean_policy_inference_interval",
                    "min_policy_inference_interval",
                    "max_policy_inference_interval",
                }
            },
            **active_coupling_scales,
        }

        logging.info("="*50)
        logging.info("--独立评估完成！")
        logging.info(f"--Checkpoint: {ckpt_path.name}")
        logging.info(f"--评估模式: {eval_mode}")
        logging.info(
            f"--Critic执行边界: min={active_min_execution_steps}, "
            f"max={active_execution_limit} "
            f"(checkpoint动作块={checkpoint_execution_limit})"
        )
        logging.info(f"--成功率 (Success Rate): {sr*100:.1f}%")
        logging.info(f"--平均奖励 (Average Reward): {ar:.2f}")
        logging.info(f"--平均推理时间 (Average Inference Time): {avg_infer:.2f} ms")
        logging.info(f"--最大推理时间 (Max Inference Time): {max_infer:.2f} ms")
        logging.info(
            "--Critic动态重规划: %d次, 最大步数强制重规划: %d次, "
            "%.3f次Critic触发/episode, "
            "触发episode比例=%.1f%%, mean_Q=%.4f, Critic平均耗时=%.2f ms",
            critic_replans,
            eval_info["aggregated"]["max_forced_replans"],
            eval_info["aggregated"][
                "critic_replans_per_episode"
            ],
            100.0
            * eval_info["aggregated"][
                "critic_trigger_episode_rate"
            ],
            eval_info["aggregated"]["critic_mean_q"],
            eval_info["aggregated"]["critic_avg_ms"],
        )
        logging.info(
            "--实际策略推理间隔: mean=%s, min=%s, max=%s",
            eval_info["aggregated"]["mean_policy_inference_interval"],
            eval_info["aggregated"]["min_policy_inference_interval"],
            eval_info["aggregated"]["max_policy_inference_interval"],
        )
        logging.info("="*50)
        return result
    finally:
        if eval_env is not None:
            eval_env.close()
        del critic
        del policy
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()


def main(eval_cfg):
    eval_mode = getattr(eval_cfg, "mode", "fast_repro")
    if eval_mode not in {"fast_repro", "strict"}:
        raise ValueError(f"未知评估模式: {eval_mode}. 可选: fast_repro / strict")

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
    logging.info(
        "Critic自适应重规划评估: checkpoints=%d, "
        "execution_bounds=[%d,%d], critic=%s",
        len(checkpoints),
        int(eval_cfg.min_execution_steps),
        int(eval_cfg.max_execution_steps),
        getattr(eval_cfg, "critic_ckpt_path", None),
    )
    logging.info("Critic评估固定使用单环境；实际重推理时机由Critic决定。")

    has_explicit_output_dir = bool(getattr(eval_cfg, "eval_output_dir", None))
    batch_output_dir = (
        build_batch_output_dir(eval_cfg, run_dir)
        if is_batch_input or has_explicit_output_dir
        else None
    )
    if batch_output_dir is not None:
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"批量评估输出目录: {batch_output_dir}")

    rows = load_existing_eval_summary(batch_output_dir) if batch_output_dir is not None else []
    completed_keys = completed_checkpoint_keys(rows)
    if completed_keys:
        completed_rows = sum(1 for row in rows if row.get("status") == "ok")
        logging.info(
            f"已发现{completed_rows}条成功历史结果，将跳过对应checkpoint"
        )

    continue_on_error = bool(
        getattr(eval_cfg, "continue_on_error", len(checkpoints) > 1)
    )
    active_critic_variant = critic_variant_key(eval_cfg)
    for eval_order, checkpoint_path in enumerate(
        tqdm(checkpoints, desc="Evaluating checkpoints with Critic"),
        start=1,
    ):
        checkpoint_keys = checkpoint_identity_keys(
            checkpoint_path,
            active_critic_variant,
        )
        if batch_output_dir is not None and checkpoint_keys & completed_keys:
            logging.info(
                f"[{eval_order}/{len(checkpoints)}] "
                f"跳过已完成Critic评估: {checkpoint_path.name}"
            )
            continue

        checkpoint_cfg = make_checkpoint_eval_cfg(
            eval_cfg,
            checkpoint_path,
        )
        output_root = (
            batch_output_dir / checkpoint_path.name
            if batch_output_dir is not None
            else None
        )
        logging.info(
            f"[{eval_order}/{len(checkpoints)}] "
            f"开始Critic评估: {checkpoint_path.name}"
        )

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
                "eval_batch_size": effective_eval_batch_size(eval_cfg),
                "max_steps": eval_cfg.max_steps,
                "min_execution_steps": int(
                    eval_cfg.min_execution_steps
                ),
                "max_execution_steps": int(
                    eval_cfg.max_execution_steps
                ),
                "critic_enabled": True,
                "critic_checkpoint_path": getattr(
                    eval_cfg,
                    "critic_ckpt_path",
                    None,
                ),
                "critic_variant": active_critic_variant,
                "error": repr(exc),
            }
            rows.append(error_row)
            if batch_output_dir is not None:
                write_eval_summary(batch_output_dir, rows, eval_cfg, source_path=resolve_eval_path(eval_cfg.ckpt_path))
                prune_if_configured(eval_cfg, run_dir, batch_output_dir, rows, stage="单个 checkpoint 评估失败")
            logging.exception(
                f"Critic评估失败: {checkpoint_path}"
            )
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
        seed=100,
        # 可以指向单个 checkpoint，也可以指向整次训练 run 目录或 run/checkpoints 目录。
        ckpt_path="outputs/2_pretrain/train/2026-07-16/20-59-53_InsertCylinder-3Arms-v0_pre_zed_dual_head_diffusion/checkpoints/162000_loss=0.0051_sr=87.0_ar=751.50",
        checkpoint_source="all",  # all: 读取目录下 checkpoint全部文件；   top_k/latest: 读取 checkpoints/top_k_records.json中记录的模型
        max_checkpoints=None,     # 调试时可设为 1/2，正式评估保持 None
        eval_output_dir=None,     # None时自动保存到run_dir/critic_eval，方便断点续评
        continue_on_error=True,   # 某个 checkpoint 失败时继续评估后面的模型
        keep_top_after_eval=None, # 设置整数N才按成功率保留排名靠前的checkpoint
        prune_checkpoints=False,  # 默认不删除原checkpoint
        prune_eval_outputs=False, # 默认不删除低排名评估输出
        allow_prune_with_max_checkpoints=False,  # max_checkpoints 调试时默认不清理，避免误删未完整评估的模型
        
        # ⚙️ 评估参数设置
        mode="fast_repro",          # fast_repro: 快速且固定 seeds；  strict: 最强可复现但更慢
        n_episodes=200,             # 评估多少个任务                 
        max_episodes_rendered=0,    # 全量评估建议 0；需要视频时再改为 1/2
        fps=25,                     # 视频帧率，和环境控制频率对齐
        max_steps=400,              # 每条episode的最大环境步数
        device="cuda",              # 如需完全规避 CUDA 非确定算子，可临时改成 "cpu"
        deterministic=False,        # 通常不用手动改；strict 模式会自动开启
        
        # 相机设置
        # ['zed_cam_left', 'zed_cam_right', 'overhead_cam', 'worms_eye_cam' , 'wrist_cam_left', 'wrist_cam_right'],
        render_camera=['overhead_cam'],         # 保存video的相机视角    
        # ⚡ 快速评估默认开启混合精度；严格对比指标时可改 False
        use_amp=True,

        # 耦合推理消融；None沿用checkpoint，0关闭对应方向，1保持完整耦合。
        view_to_arm_coupling_scale=None,
        arm_to_view_coupling_scale=None,

        # 本脚本始终启用Critic，并固定使用单环境评估。
        # 支持具体.pt、Critic的checkpoints目录或完整训练输出目录。
        critic_ckpt_path="outputs/7_replanning_dqn/action_value/train/2026-07-24/18-20-46_InsertCylinder-3Arms-v0/checkpoints/latest.pt",
        # 每次生成动作块后，至少执行2步才允许Critic提前重规划。
        min_execution_steps=4,
        # 无论Critic判断如何，最多执行8步就强制重新推理。
        max_execution_steps=15,
        critic_gamma=0.99,
        # 当前缓存动作的平滑Q低于动作块预测Q的70%时记为一次异常。
        critic_anchor_ratio_threshold=0.70,
        # 相邻评分的归一化Bellman下降超过15%时记为一次异常。
        critic_local_drop_threshold=0.05,
        # 连续异常达到2步才清空动作块并重新推理，抑制单次Q噪声。
        critic_consecutive_bad_steps=2,
        # Critic Q的指数滑动平均系数；越小越平滑、触发越慢。
        critic_ema_alpha=0.20,
        # 相对比值的数值下限，避免接近0的Q放大噪声。
        critic_min_reference_q=0.05,
    )
    if eval_cfg.deterministic or eval_cfg.mode == "strict" or DETERMINISTIC_EVAL:
        ensure_python_hash_seed(eval_cfg.seed)
    # ==========================================
    # 启动
    main(eval_cfg=eval_cfg)
                                         
