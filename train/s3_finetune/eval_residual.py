import contextlib
import json
import logging
import os
import random
import sys
import time
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import gymnasium as gym
import imageio
import numpy as np
import torch
import yaml
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import env.task.sim_envs  # noqa: F401
from train.s3_finetune.finetune_residual import FrozenDiffusionResidualPolicy




@contextlib.contextmanager
def maybe_suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull):
            yield


def seed_runtime(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_env_spaces(env, seed: int):
    for space_name in ("action_space", "observation_space"):
        space = getattr(env, space_name, None)
        if hasattr(space, "seed"):
            space.seed(seed)


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    seed_runtime(seed)
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


def ensure_python_hash_seed(seed: int):
    desired = str(seed)
    if os.environ.get("PYTHONHASHSEED") == desired:
        return

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = desired
    logging.warning(
        f"PYTHONHASHSEED must be set before Python starts; restarting eval_residual.py with {desired}."
    )
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def patch_act_position_embedding_for_determinism():
    try:
        from lerobot.common.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d
    except Exception as exc:
        logging.warning(f"Cannot apply ACT deterministic patch: {exc}")
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
    logging.info("Applied ACT deterministic positional embedding patch.")


def resolve_checkpoint_dirs(ckpt_path: str | Path):
    ckpt_root = Path(ckpt_path).expanduser()
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Cannot find checkpoint path: {ckpt_root}")

    if ckpt_root.name == "pretrained_model":
        load_dir = ckpt_root
        root_dir = ckpt_root.parent
    elif (ckpt_root / "pretrained_model").is_dir():
        root_dir = ckpt_root
        load_dir = ckpt_root / "pretrained_model"
        logging.info("Detected checkpoint/pretrained_model layout; loading pretrained_model.")
    else:
        root_dir = ckpt_root
        load_dir = ckpt_root

    config_yaml_path = load_dir / "config.yaml"
    if not config_yaml_path.exists():
        config_yaml_path = root_dir / "config.yaml"
    if not config_yaml_path.exists():
        raise FileNotFoundError(f"Cannot find config.yaml near {load_dir} or {root_dir}")

    residual_policy_path = load_dir / "residual_policy.pt"
    if not residual_policy_path.exists():
        raise FileNotFoundError(
            f"Cannot find residual_policy.pt in {load_dir}. "
            "Please pass a finetune_residual checkpoint or its pretrained_model directory."
        )

    return root_dir, load_dir, config_yaml_path


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def nested_get(data: dict, dotted_key: str, default=None):
    current = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def infer_actor_shape(model_state: dict, action_dim: int):
    actor_weight_keys = sorted(
        [key for key in model_state if key.startswith("actor_mean.") and key.endswith(".weight")],
        key=lambda key: int(key.split(".")[1]),
    )
    critic_weight_keys = sorted(
        [key for key in model_state if key.startswith("critic.") and key.endswith(".weight")],
        key=lambda key: int(key.split(".")[1]),
    )

    actor_hidden_dim = 512
    actor_depth = 2
    critic_hidden_dim = 512
    critic_depth = 2
    if actor_weight_keys:
        actor_hidden_dim = int(model_state[actor_weight_keys[0]].shape[0])
        actor_depth = max(1, len(actor_weight_keys) - 1)
        if int(model_state[actor_weight_keys[-1]].shape[0]) != int(action_dim):
            logging.warning("Actor output shape does not match action_dim; check checkpoint compatibility.")
    if critic_weight_keys:
        critic_hidden_dim = int(model_state[critic_weight_keys[0]].shape[0])
        critic_depth = max(1, len(critic_weight_keys) - 1)
    return actor_hidden_dim, actor_depth, critic_hidden_dim, critic_depth


def load_frozen_diffusion_residual_policy(base_policy, load_dir: str | Path, config_yaml_path: Path, device):
    load_dir = Path(load_dir)
    residual_ckpt_path = load_dir / "residual_policy.pt"
    residual_config_path = load_dir / "residual_policy_config.json"

    try:
        residual_ckpt = torch.load(residual_ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        residual_ckpt = torch.load(residual_ckpt_path, map_location=device)

    if "model" not in residual_ckpt:
        raise ValueError(f"{residual_ckpt_path} is not a finetune_residual checkpoint: missing key 'model'.")

    residual_json = {}
    if residual_config_path.exists():
        with open(residual_config_path, "r", encoding="utf-8") as f:
            residual_json = json.load(f)

    config_yaml = load_yaml(config_yaml_path)
    training_cfg = config_yaml.get("training", {})
    model_state = residual_ckpt["model"]

    def pick(key: str, default=None):
        if key in residual_ckpt:
            return residual_ckpt[key]
        if key in residual_json:
            return residual_json[key]
        return training_cfg.get(key, default)

    action_dim = int(pick("action_dim", base_policy.config.output_shapes["action"][0]))
    action_start = int(pick("action_start", int(getattr(base_policy.config, "n_obs_steps", 2)) - 1))
    action_end = int(pick("action_end", action_start + int(getattr(base_policy.config, "n_action_steps", 8))))
    global_cond_dim = int(pick("global_cond_dim", 0))
    stepwise_obs = bool(pick("stepwise_obs", training_cfg.get("residual_stepwise_obs", True)))
    action_scale = float(pick("action_scale", training_cfg.get("residual_action_scale", 0.1)))
    max_delta = float(pick("max_delta", training_cfg.get("residual_max_delta", 0.0)))
    logprob_reduction = str(pick("logprob_reduction", training_cfg.get("logprob_reduction", "mean")))
    activation = str(training_cfg.get("residual_activation", "SiLU"))

    inferred_actor_hidden, inferred_actor_depth, inferred_critic_hidden, inferred_critic_depth = infer_actor_shape(
        model_state,
        action_dim=action_dim,
    )
    actor_hidden_dim = int(training_cfg.get("residual_hidden_dim", inferred_actor_hidden))
    actor_depth = int(training_cfg.get("residual_depth", inferred_actor_depth))
    critic_hidden_dim = int(training_cfg.get("residual_critic_hidden_dim", inferred_critic_hidden))
    critic_depth = int(training_cfg.get("residual_critic_depth", inferred_critic_depth))
    learn_std = bool(training_cfg.get("residual_learn_std", True))
    action_head_std = float(training_cfg.get("residual_action_head_std", 0.0))

    if "actor_logstd" in model_state:
        init_std = float(model_state["actor_logstd"].float().exp().mean().item())
    else:
        init_std = float(training_cfg.get("residual_std", 0.02))

    policy = FrozenDiffusionResidualPolicy(
        base_policy=base_policy,
        action_dim=action_dim,
        action_start=action_start,
        action_end=action_end,
        global_cond_dim=global_cond_dim,
        actor_hidden_dim=actor_hidden_dim,
        actor_depth=actor_depth,
        critic_hidden_dim=critic_hidden_dim,
        critic_depth=critic_depth,
        activation=activation,
        init_std=init_std,
        learn_std=learn_std,
        action_head_std=action_head_std,
        action_scale=action_scale,
        max_delta=max_delta,
        logprob_reduction=logprob_reduction,
        stepwise_obs=stepwise_obs,
    ).to(device)
    policy.residual_policy.load_state_dict(model_state, strict=True)
    policy.eval()

    logging.info(
        "Loaded DP+Residual policy | "
        f"action_dim={action_dim}, action_slice=[{action_start}:{action_end}], "
        f"global_cond_dim={global_cond_dim}, residual_obs_dim={policy.residual_obs_dim}, "
        f"stepwise_obs={stepwise_obs}, action_scale={action_scale:.4f}, max_delta={max_delta:.5f}, "
        f"std={float(policy.action_std.mean().detach().cpu().item()):.5f}, "
        f"actor={actor_hidden_dim}x{actor_depth}, critic={critic_hidden_dim}x{critic_depth}, "
        f"activation={activation}"
    )
    return policy


def prepare_single_obs(obs):
    if isinstance(obs, dict):
        return {key: prepare_single_obs(value) for key, value in obs.items()}
    if hasattr(obs, "copy"):
        return np.expand_dims(obs.copy(), axis=0).copy()
    return obs


def render_frame(env, camera: str):
    return env.unwrapped.render([camera])


def eval_policy(env, policy, cfg_eval, videos_dir, device):
    policy.eval()
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = int(getattr(cfg_eval, "n_episodes", 20))
    max_rendered = int(getattr(cfg_eval, "max_episodes_rendered", 1))
    fps = int(getattr(cfg_eval, "fps", 25))
    max_steps = int(getattr(cfg_eval, "max_steps", 400))
    base_seed = int(getattr(cfg_eval, "seed", 100))
    quiet = bool(getattr(cfg_eval, "quiet", True))
    show_progress = bool(getattr(cfg_eval, "show_progress", True))
    suppress_model_stdout = bool(getattr(cfg_eval, "suppress_model_stdout", quiet))
    raw_cameras = getattr(cfg_eval, "render_camera", ["overhead_cam"])
    render_cameras = [raw_cameras] if isinstance(raw_cameras, str) else list(raw_cameras)
    render_cameras = render_cameras or ["overhead_cam"]

    successes = []
    rewards = []
    episode_records = []
    saved_video_paths = []
    global_real_inference_times = []

    episode_iter = tqdm(
        range(n_episodes),
        desc="Residual eval",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for ep in episode_iter:
        ep_seed = base_seed + ep
        seed_runtime(ep_seed)
        seed_env_spaces(env, ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        policy.reset()
        seed_runtime(ep_seed)

        info = {"is_success": False}
        ep_reward = 0.0
        steps_taken = 0
        ep_inference_times = []
        frames_by_camera = {camera: [] for camera in render_cameras} if ep < max_rendered else {}

        for step in range(max_steps):
            steps_taken = step + 1
            if ep < max_rendered:
                for camera in render_cameras:
                    frames_by_camera[camera].append(render_frame(env, camera))

            policy_obs = preprocess_observation(prepare_single_obs(obs))
            policy_obs = {
                key: value.to(device)
                for key, value in policy_obs.items()
                if key in policy.config.input_shapes
            }

            start_time = time.perf_counter()
            with torch.no_grad():
                with maybe_suppress_stdout(suppress_model_stdout):
                    action = policy.select_action(policy_obs)
            action_np = action.squeeze(0).detach().cpu().numpy()
            inference_time_ms = (time.perf_counter() - start_time) * 1000.0
            ep_inference_times.append(inference_time_ms)

            try:
                obs, reward, terminated, truncated, info = env.step(action_np)
                ep_reward += float(reward)
                done = bool(terminated or truncated)
            except Exception as exc:
                logging.error(f"Physics/env error at episode={ep}, step={step}: {exc}")
                info = {"is_success": False}
                ep_reward = -1000.0
                done = True

            if done:
                break

        success = bool(info.get("is_success", False))
        successes.append(success)
        rewards.append(float(ep_reward))
        episode_records.append(
            {
                "episode": int(ep),
                "seed": int(ep_seed),
                "success": success,
                "reward": float(ep_reward),
                "steps": int(steps_taken),
            }
        )

        real_inferences = [value for value in ep_inference_times if value > 5.0]
        global_real_inference_times.extend(real_inferences)

        if show_progress:
            episode_iter.set_postfix(
                success=f"{np.mean(successes) * 100:.1f}%",
                avg_reward=f"{np.mean(rewards):.1f}",
                last_steps=f"{steps_taken}/{max_steps}",
            )

        if ep < max_rendered:
            status = "Success" if success else "Fail"
            for camera, frames in frames_by_camera.items():
                if not frames:
                    continue
                video_path = videos_dir / f"{camera}_ep_{ep}_reward={ep_reward:.1f}_{status}.mp4"
                imageio.mimsave(str(video_path), frames, fps=fps)
                saved_video_paths.append(str(video_path))

    warmup_steps = 3
    stable_times = (
        global_real_inference_times[warmup_steps:]
        if len(global_real_inference_times) > warmup_steps
        else global_real_inference_times
    )
    avg_real_time = float(np.mean(stable_times)) if stable_times else 0.0
    max_time = float(np.max(stable_times)) if stable_times else 0.0

    return {
        "aggregated": {
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "average_reward": float(np.mean(rewards)) if rewards else 0.0,
            "avg_inference_ms": avg_real_time,
            "max_inference_ms": max_time,
        },
        "video_paths": saved_video_paths,
        "episodes": episode_records,
    }


def load_policy_and_env(eval_cfg, device):
    root_dir, load_dir, config_yaml_path = resolve_checkpoint_dirs(eval_cfg.ckpt_path)

    logging.info(f"Loading frozen Diffusion base from: {load_dir}")
    hydra_cfg = init_hydra_config(str(config_yaml_path))
    hydra_cfg.device = str(device)
    base_policy = make_policy(
        hydra_cfg=hydra_cfg,
        pretrained_policy_name_or_path=str(load_dir),
    )
    base_policy.to(device)

    policy = load_frozen_diffusion_residual_policy(
        base_policy=base_policy,
        load_dir=load_dir,
        config_yaml_path=config_yaml_path,
        device=device,
    )

    all_obs_keys = policy.config.input_shapes.keys()
    ref_cams = [
        key.replace("observation.images.", "")
        for key in all_obs_keys
        if "observation.images." in key
    ]
    if not ref_cams:
        raise ValueError("The policy config does not contain observation.images.* inputs.")

    render_cameras = getattr(eval_cfg, "render_camera", ["overhead_cam"])
    render_cameras = [render_cameras] if isinstance(render_cameras, str) else list(render_cameras)
    obs_cameras = list(dict.fromkeys(ref_cams + render_cameras))

    config_yaml = load_yaml(config_yaml_path)
    env_cfg = config_yaml.get("env", {})
    env_name = env_cfg.get("name", getattr(eval_cfg, "env_name", "guided_vision"))
    env_task = env_cfg.get("task", getattr(eval_cfg, "env_task", "InsertCylinder-3Arms-v0"))
    env_id = f"{env_name}/{env_task}"
    logging.info(f"Creating eval env: {env_id}, cameras={obs_cameras}")
    eval_env = gym.make(id=env_id, cameras=obs_cameras)
    return policy, eval_env, root_dir


def main(eval_cfg):
    seed_everything(int(eval_cfg.seed))
    patch_act_position_embedding_for_determinism()
    device = get_safe_torch_device(getattr(eval_cfg, "device", "cuda"), log=True)

    policy, eval_env, root_dir = load_policy_and_env(eval_cfg, device)
    videos_dir = Path(getattr(eval_cfg, "videos_dir", "") or (Path(root_dir) / "extra_eval_videos"))
    videos_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Evaluation videos/results will be written under: {videos_dir}")

    with torch.autocast(device_type=device.type) if bool(getattr(eval_cfg, "use_amp", False)) else nullcontext():
        eval_info = eval_policy(
            env=eval_env,
            policy=policy,
            cfg_eval=eval_cfg,
            videos_dir=videos_dir,
            device=device,
        )

    sr = eval_info["aggregated"]["success_rate"]
    ar = eval_info["aggregated"]["average_reward"]
    avg_infer = eval_info["aggregated"]["avg_inference_ms"]
    max_infer = eval_info["aggregated"]["max_inference_ms"]

    result_dir = videos_dir / f"eval_seed={eval_cfg.seed}_ep={eval_cfg.n_episodes}_sr={sr * 100:.1f}_ar={ar:.2f}"
    result_dir.mkdir(parents=True, exist_ok=True)

    for video_path in eval_info["video_paths"]:
        src = Path(video_path)
        if src.exists() and src.parent != result_dir:
            src.replace(result_dir / src.name)

    episode_results_path = result_dir / "episode_results.json"
    metrics_path = result_dir / "metrics.json"
    with open(episode_results_path, "w", encoding="utf-8") as f:
        json.dump(eval_info["episodes"], f, indent=2, ensure_ascii=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_info["aggregated"], f, indent=2, ensure_ascii=False)

    logging.info("=" * 60)
    logging.info("Residual evaluation finished.")
    logging.info(f"Success Rate     : {sr * 100:.1f}%")
    logging.info(f"Average Reward   : {ar:.2f}")
    logging.info(f"Avg Inference    : {avg_infer:.2f} ms")
    logging.info(f"Max Inference    : {max_infer:.2f} ms")
    logging.info(f"Episode JSON     : {episode_results_path}")
    logging.info(f"Metrics JSON     : {metrics_path}")
    logging.info("=" * 60)
    return eval_info


if __name__ == "__main__":
    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)

    eval_cfg = SimpleNamespace(
        # residual checkpoint 路径，可填 checkpoint 目录或 pretrained_model 子目录
        ckpt_path="outputs/3_finetune/train/2026-07-06/23-04-58_InsertCylinder-3Arms-v0_ft_zed_diffusion_residual/checkpoints/000030_sr=0.80_reward=666.52_Rloss=-0.0049_Vloss=0.0016",  
        seed=100,  # 评估随机种子
        n_episodes=100,  # 评估 episode 数量
        max_episodes_rendered=1,  # 保存视频的 episode 数量
        max_steps=400,  # 每个 episode 最大步数
        fps=25,  # 保存视频帧率
        device="cuda",  # 评估设备，可改为 cuda:0 或 cpu
        render_camera=["overhead_cam"],  # 保存视频使用的相机列表
        videos_dir="",  # 视频输出目录，空字符串表示写到 checkpoint/extra_eval_videos
        use_amp=False,  # 是否启用混合精度评估
        quiet=True,  # 是否减少终端冗余输出
        show_progress=True,  # 是否显示 episode 进度条
        suppress_model_stdout=True,  # 是否屏蔽模型内部 print 输出
    )
    ensure_python_hash_seed(eval_cfg.seed)
    main(eval_cfg)
