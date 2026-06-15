#!/usr/bin/env python
"""Collect policy rollout data and save it locally.

This script intentionally saves a raw, lossless-enough dataset first:

outputs/data_collect/<run_name>/
  metadata.json
  episodes/
    episode_000000/
      info.json
      arrays.npz
      images/<camera>/000000.jpg
      videos/<camera>.mp4

The raw layout is easier to inspect and can be converted to the local
LeRobotDataset layout later without mixing generated rollouts into checkpoint
folders.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

# Keep the same deterministic/offscreen defaults used by eval.py.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import env  # noqa: F401  # Ensure Gym environments are registered.
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging

# Reuse eval.py's seeding and ACT deterministic patch logic.
try:
    from eval import (  # noqa: E402
        ensure_python_hash_seed,
        patch_act_position_embedding_for_determinism,
        seed_env_spaces,
        seed_everything,
        seed_runtime,
    )
except ModuleNotFoundError:
    from train.pretrain.eval import (  # noqa: E402
        ensure_python_hash_seed,
        patch_act_position_embedding_for_determinism,
        seed_env_spaces,
        seed_everything,
        seed_runtime,
    )

ZED_CAMERA_NAMES = ["zed_cam_left", "zed_cam_right"]

DEFAULT_2ARM_CAMERA_NAMES = [
    "wrist_cam_left",
    "wrist_cam_right",
    "overhead_cam",
    "worms_eye_cam",
]

DEFAULT_3ARM_CAMERA_NAMES = [
    "zed_cam_left",
    "zed_cam_right",
    "wrist_cam_left",
    "wrist_cam_right",
    "overhead_cam",
    "worms_eye_cam",
]


def get_default_camera_names(env_id: str) -> list[str]:
    env_id_lower = env_id.lower() 
    if "2arms" in env_id_lower or "2arm" in env_id_lower:
        return DEFAULT_2ARM_CAMERA_NAMES
    return DEFAULT_3ARM_CAMERA_NAMES


def prepare_obs_for_policy(obs):
    """Add batch dimension, copy arrays, then run LeRobot observation preprocessing."""

    def prepare(obj):
        if isinstance(obj, dict):
            return {k: prepare(v) for k, v in obj.items()}
        if hasattr(obj, "copy"):
            return np.expand_dims(obj.copy(), axis=0).copy()
        return obj

    return preprocess_observation(prepare(obs))


def get_policy_input_cameras(policy) -> list[str]:
    input_shapes = getattr(policy.config, "input_shapes", {})
    cameras = [
        key.replace("observation.images.", "")
        for key in input_shapes.keys()
        if key.startswith("observation.images.")
    ]
    return list(dict.fromkeys(cameras))


def resolve_checkpoint_dir(ckpt_path: str | Path) -> Path:
    ckpt_path = Path(ckpt_path)
    pretrained_model = ckpt_path / "pretrained_model"
    return pretrained_model if pretrained_model.exists() else ckpt_path


def load_policy_from_checkpoint(ckpt_path: str | Path, device: torch.device):
    load_dir = resolve_checkpoint_dir(ckpt_path)
    config_yaml = load_dir / "config.yaml"
    if not config_yaml.exists():
        config_yaml = load_dir.parent / "config.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(f"Cannot find config.yaml around checkpoint: {ckpt_path}")

    hydra_cfg = init_hydra_config(str(config_yaml))
    hydra_cfg.device = str(device)
    policy = make_policy(
        hydra_cfg=hydra_cfg,
        pretrained_policy_name_or_path=str(load_dir),
    )
    policy.to(device)
    policy.eval()
    return policy, load_dir, config_yaml


def disable_diffusion_debug_image_saving(policy):
    """Disable local debug image dumps added inside the project's diffusion encoder.

    The local modeling_diffusion.py currently saves the first few resized/cropped
    images to a hard-coded directory outside this repo. Data collection already
    saves images explicitly, so those debug dumps only add noise and can fail in
    sandboxed runs.
    """
    patched = 0
    for module in policy.modules():
        if module.__class__.__name__ == "DiffusionRgbEncoder":
            module._debug_img_counter = 3
            patched += 1
    if patched:
        logging.info(f"Disabled diffusion debug image saving for {patched} image encoder(s).")


def read_env_id(config_yaml: Path) -> str:
    if not config_yaml.exists():
        raise FileNotFoundError(f"Cannot find checkpoint config.yaml: {config_yaml}")

    with open(config_yaml, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}

    env_cfg = full_cfg.get("env", {})
    env_name = env_cfg.get("name")
    env_task = env_cfg.get("task")
    if not env_name or not env_task:
        raise ValueError(f"config.yaml 缺少 env.name 或 env.task，无法确定采集环境: {config_yaml}")
    return f"{env_name}/{env_task}"


def collect_env_id_for(env_id: str) -> str:
    if env_id.endswith("SewNeedle-2Arms-v0"):
        return env_id.replace("SewNeedle-2Arms-v0", "SewNeedle-2Arms-Collect-v0")
    if env_id.endswith("SewNeedle-3Arms-v0"):
        return env_id.replace("SewNeedle-3Arms-v0", "SewNeedle-3Arms-Collect-v0")
    return env_id


def build_env(env_id: str, cameras: list[str], image_height: int, image_width: int):
    return gym.make(
        id=env_id,
        cameras=cameras,
        observation_height=image_height,
        observation_width=image_width,
    )


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._=-]+", "_", value).strip("_")
    return sanitized or "unknown"


def infer_arm_tag(task_name: str) -> str:
    match = re.search(r"(\d+)Arms?", task_name, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}Arms"
    return "UnknownArms"


def make_default_run_name(env_id: str) -> str:
    task_name = env_id.split("/", 1)[-1]
    arm_tag = infer_arm_tag(task_name)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"collect_{sanitize_path_component(task_name)}_{arm_tag}_{timestamp}"


def make_run_dir(output_dir: str | Path, run_name: str | None, env_id: str) -> Path:
    output_dir = Path(output_dir)
    if run_name is None:
        run_name = make_default_run_name(env_id)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    return run_dir


def list_existing_episode_dirs(run_dir: Path) -> list[Path]:
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.exists():
        return []
    return sorted(
        path
        for path in episodes_dir.glob("episode_*")
        if path.is_dir() and not path.name.endswith(".tmp") and (path / "arrays.npz").exists()
    )


def episode_index_from_dir(episode_dir: Path) -> int:
    match = re.search(r"episode_(\d+)$", episode_dir.name)
    if not match:
        return -1
    return int(match.group(1))


def next_episode_index(run_dir: Path) -> int:
    indices = [episode_index_from_dir(path) for path in list_existing_episode_dirs(run_dir)]
    indices = [index for index in indices if index >= 0]
    return max(indices, default=-1) + 1


def load_existing_episode_infos(run_dir: Path) -> list[dict]:
    infos = []
    for episode_dir in list_existing_episode_dirs(run_dir):
        info_path = episode_dir / "info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        else:
            info = {
                "episode": episode_index_from_dir(episode_dir),
                "success": True,
            }
        info["path"] = str(episode_dir.relative_to(run_dir))
        infos.append(info)
    return infos


def image_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def extract_state(obs) -> np.ndarray:
    if "agent_pos" in obs:
        return np.asarray(obs["agent_pos"], dtype=np.float32)
    if "observation.state" in obs:
        return np.asarray(obs["observation.state"], dtype=np.float32)
    raise KeyError("Observation does not contain 'agent_pos' or 'observation.state'.")


def extract_pixels(obs) -> dict[str, np.ndarray]:
    if "pixels" in obs:
        return {k: np.asarray(v) for k, v in obs["pixels"].items()}
    prefix = "observation.images."
    return {k.replace(prefix, ""): np.asarray(v) for k, v in obs.items() if k.startswith(prefix)}


def obs_key_to_npz_key(raw_key: str) -> str:
    safe = raw_key.replace(".", "__").replace("/", "__")
    return f"obs__{safe}"


def flatten_numeric_obs(obs, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten every non-image numeric value from the raw environment obs.

    Image observations are saved separately under images/<camera>/ so this
    function skips the standard image branches.
    """
    if isinstance(obs, dict):
        flattened = {}
        for key, value in obs.items():
            raw_key = f"{prefix}.{key}" if prefix else str(key)
            if raw_key == "pixels" or raw_key.startswith("observation.images"):
                continue
            flattened.update(flatten_numeric_obs(value, raw_key))
        return flattened

    try:
        array = np.asarray(obs)
    except Exception:
        return {}

    if array.dtype.kind not in "biufc":
        return {}
    return {prefix: array.astype(np.float32, copy=False)}


def stack_trace(values: list[np.ndarray]) -> np.ndarray:
    try:
        return np.stack(values, axis=0)
    except ValueError:
        return np.asarray(values, dtype=object)


def should_keep_episode(save_filter: str, success: bool) -> bool:
    if save_filter == "all":
        return True
    if save_filter == "success":
        return success
    if save_filter == "failure":
        return not success
    raise ValueError(f"Unknown save_filter: {save_filter}")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def save_episode_arrays(
    episode_dir: Path,
    obs_traces: dict[str, list[np.ndarray]],
    actions: list[np.ndarray],
    rewards: list[float],
    terminated: list[bool],
    truncated: list[bool],
    timestamps: list[float],
):
    obs_key_map = {}
    arrays = {
        "action": np.asarray(actions, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "timestamp": np.asarray(timestamps, dtype=np.float32),
        "frame_index": np.arange(len(actions), dtype=np.int64),
    }

    for raw_key, values in sorted(obs_traces.items()):
        if not values:
            continue
        npz_key = obs_key_to_npz_key(raw_key)
        obs_key_map[raw_key] = npz_key
        arrays[npz_key] = stack_trace(values)

    # Keep the old convenient alias used by the first version of the collector.
    if "agent_pos" in obs_key_map:
        arrays["observation_state"] = arrays[obs_key_map["agent_pos"]]
    elif "observation.state" in obs_key_map:
        arrays["observation_state"] = arrays[obs_key_map["observation.state"]]

    np.savez_compressed(episode_dir / "arrays.npz", **arrays)
    return obs_key_map


def create_episode_videos(episode_dir: Path, fps: int) -> list[str]:
    """Create one preview mp4 per camera from the saved image frames."""
    videos_dir = episode_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    saved_videos = []
    images_root = episode_dir / "images"
    if not images_root.exists():
        return saved_videos

    for cam_dir in sorted(path for path in images_root.iterdir() if path.is_dir()):
        frame_paths = sorted(cam_dir.glob("*.jpg"))
        if not frame_paths:
            continue

        video_path = videos_dir / f"{cam_dir.name}.mp4"
        with imageio.get_writer(str(video_path), fps=fps, macro_block_size=1) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        saved_videos.append(str(video_path))

    return saved_videos


def collect_episode(
    *,
    env,
    policy,
    device,
    episode_dir: Path,
    episode_index: int,
    episode_seed: int,
    max_steps: int,
    fps: int,
    camera_save_quality: int,
    use_amp: bool,
):
    seed_runtime(episode_seed)
    seed_env_spaces(env, episode_seed)
    obs, _ = env.reset(seed=episode_seed)
    policy.reset()
    seed_runtime(episode_seed)

    obs_traces: dict[str, list[np.ndarray]] = {}
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    timestamps: list[float] = []
    inference_times_ms: list[float] = []

    image_root = episode_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    final_info = {"is_success": False}
    ep_reward = 0.0
    steps_taken = 0

    for step in range(max_steps):
        steps_taken = step + 1

        raw_obs_fields = flatten_numeric_obs(obs)
        raw_pixels = extract_pixels(obs)

        for cam, image in raw_pixels.items():
            cam_dir = image_root / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(
                cam_dir / f"{step:06d}.jpg",
                image_to_uint8_hwc(image),
                quality=camera_save_quality,
            )

        policy_obs = prepare_obs_for_policy(obs)
        policy_obs = {
            k: v.to(device)
            for k, v in policy_obs.items()
            if k in policy.config.input_shapes
        }

        start = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(device_type=device.type) if use_amp else nullcontext():
                action = policy.select_action(policy_obs)
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        inference_times_ms.append((time.perf_counter() - start) * 1000.0)

        for raw_key, value in raw_obs_fields.items():
            obs_traces.setdefault(raw_key, []).append(value)
        actions.append(action_np)
        timestamps.append(step / float(fps))

        try:
            obs, reward, terminated, truncated, info = env.step(action_np)
        except Exception as exc:
            logging.exception("Physics/env exception at episode=%s step=%s: %s", episode_index, step, exc)
            reward = -1000.0
            terminated = True
            truncated = False
            info = {"is_success": False, "exception": str(exc)}

        reward = float(reward)
        ep_reward += reward
        rewards.append(reward)
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        final_info = dict(info or {})

        if terminated or truncated:
            break

    success = bool(final_info.get("is_success", False))
    obs_key_map = save_episode_arrays(
        episode_dir=episode_dir,
        obs_traces=obs_traces,
        actions=actions,
        rewards=rewards,
        terminated=terminated_flags,
        truncated=truncated_flags,
        timestamps=timestamps,
    )

    episode_info = {
        "episode": int(episode_index),
        "seed": int(episode_seed),
        "success": success,
        "reward": float(ep_reward),
        "steps": int(steps_taken),
        "fps": int(fps),
        "inference_ms_mean": float(np.mean(inference_times_ms)) if inference_times_ms else 0.0,
        "inference_ms_max": float(np.max(inference_times_ms)) if inference_times_ms else 0.0,
        "observation_npz_keys": obs_key_map,
        "image_observation_dirs": {
            f"pixels.{camera}": f"images/{camera}"
            for camera in sorted(extract_pixels(obs).keys())
        },
        "final_info": final_info,
    }
    with open(episode_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(episode_info, f, indent=2, ensure_ascii=False)

    return episode_info


def write_metadata(run_dir: Path, metadata: dict):
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main(args):
    init_logging()
    logging.getLogger().setLevel(logging.INFO)

    seed_everything(args.seed)
    patch_act_position_embedding_for_determinism()

    device = get_safe_torch_device(args.device)
    logging.info(f"Using device: {device}")
    policy, load_dir, config_yaml = load_policy_from_checkpoint(args.ckpt_path, device)
    disable_diffusion_debug_image_saving(policy)

    policy_cameras = get_policy_input_cameras(policy)
    if not policy_cameras:
        raise ValueError("No observation.images.* camera key found in policy input_shapes.")

    source_env_id = read_env_id(config_yaml)
    env_id = collect_env_id_for(source_env_id)
    cameras = get_default_camera_names(source_env_id)
    if args.image_height <= 0 or args.image_width <= 0:
        raise ValueError("image_height 和 image_width 必须大于 0。")

    missing_policy_cameras = [camera for camera in policy_cameras if camera not in cameras]
    if missing_policy_cameras:
        raise ValueError(
            "当前相机选择缺少策略推理必需相机: "
            f"{missing_policy_cameras}. source_env_id={source_env_id}, env_id={env_id}, selected={cameras}. "
            "如果是 2-arm wrist 数据采集，请换用 wrist-only checkpoint；如果 checkpoint 需要额外相机，"
            "请先在 get_default_camera_names() 的默认相机列表中加入。"
        )

    eval_env = build_env(env_id, cameras, args.image_height, args.image_width)
    env_desc = f"{env_id} -> {eval_env.unwrapped.__class__.__module__}.{eval_env.unwrapped.__class__.__name__}"

    append_run_dir = str(getattr(args, "append_run_dir", "")).strip()
    if append_run_dir:
        run_dir = Path(append_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"append_run_dir 不存在: {run_dir}")
        (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        existing_episode_infos = load_existing_episode_infos(run_dir)
        logging.info(f"Append mode: {run_dir}，已发现 {len(existing_episode_infos)} 条现有轨迹。")
    else:
        run_dir = make_run_dir(args.output_dir, args.run_name, source_env_id)
        metadata = {}
        existing_episode_infos = []

    metadata.update({
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt_path": str(args.ckpt_path),
        "load_dir": str(load_dir),
        "config_yaml": str(config_yaml),
        "source_env_id": source_env_id,
        "env_id": env_id,
        "env_desc": env_desc,
        "policy_input_cameras": policy_cameras,
        "default_cameras": get_default_camera_names(source_env_id),
        "saved_cameras": cameras,
        "seed": int(args.seed),
        "target_successes": int(args.target_successes),
        "max_steps": int(args.max_steps),
        "fps": int(args.fps),
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "save_filter": args.save_filter,
        "episodes": existing_episode_infos,
    })

    logging.info(f"Collecting data into: {run_dir}")
    logging.info(f"Environment: {env_desc}")
    logging.info(f"Saved cameras: {cameras}")
    logging.info(f"Save filter: {args.save_filter}")

    saved_count = len(existing_episode_infos)
    attempted_count = max(int(metadata.get("attempted_episodes", 0)), saved_count)
    success_count = sum(1 for info in existing_episode_infos if bool(info.get("success", False)))
    metadata["attempted_episodes"] = attempted_count
    metadata["saved_episodes"] = saved_count
    metadata["successful_episodes"] = success_count
    metadata["success_rate_attempted"] = success_count / max(1, attempted_count)
    write_metadata(run_dir, metadata)

    if args.target_successes <= 0:
        raise ValueError("target_successes 必须大于 0。")

    episode_index = next_episode_index(run_dir)
    while success_count < args.target_successes:
        attempted_count += 1
        episode_seed = args.seed + episode_index
        tmp_episode_dir = run_dir / "episodes" / f"episode_{episode_index:06d}.tmp"
        final_episode_dir = run_dir / "episodes" / f"episode_{episode_index:06d}"
        if tmp_episode_dir.exists():
            shutil.rmtree(tmp_episode_dir)
        tmp_episode_dir.mkdir(parents=True, exist_ok=False)

        info = collect_episode(
            env=eval_env,
            policy=policy,
            device=device,
            episode_dir=tmp_episode_dir,
            episode_index=episode_index,
            episode_seed=episode_seed,
            max_steps=args.max_steps,
            fps=args.fps,
            camera_save_quality=args.camera_quality,
            use_amp=args.use_amp,
        )

        if info["success"]:
            success_count += 1

        keep = should_keep_episode(args.save_filter, info["success"])
        info["kept"] = bool(keep)
        if keep:
            tmp_episode_dir.rename(final_episode_dir)
            info["path"] = str(final_episode_dir.relative_to(run_dir))
            if args.save_videos:
                info["video_paths"] = create_episode_videos(final_episode_dir, args.fps)
            saved_count += 1
        else:
            shutil.rmtree(tmp_episode_dir, ignore_errors=True)
            info["path"] = None
            info["video_paths"] = []

        metadata["episodes"].append(info)
        metadata["attempted_episodes"] = attempted_count
        metadata["saved_episodes"] = saved_count
        metadata["successful_episodes"] = success_count
        metadata["success_rate_attempted"] = success_count / max(1, attempted_count)
        write_metadata(run_dir, metadata)

        logging.info(
            f"episode={episode_index:04d} seed={episode_seed} success={info['success']} "
            f"reward={info['reward']:.2f} steps={info['steps']} kept={keep} "
            f"saved={saved_count}/{args.target_successes}"
        )
        episode_index += 1

    logging.info("=" * 60)
    logging.info("Data collection finished.")
    logging.info(f"Attempted episodes: {attempted_count}")   # 总共尝试了多少条轨迹（包括成功和失败）。
    logging.info(f"Successful episodes: {success_count}")    # 满足成功条件的轨迹数量。
    logging.info(f"Saved episodes: {saved_count}")           # 最终保存的轨迹数量，取决于 save_filter 和成功数量。
    logging.info(f"Output: {run_dir}")
    logging.info("=" * 60)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect rollout data from a checkpoint policy.")
    parser.add_argument(
        "--ckpt-path",
        default="outputs/1_hugging_model/pre_sim_sew_needle_3arms_zed_wrist_diffusion",
    )
    
    parser.add_argument("--output-dir", default="outputs/4_data_collect")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--append-run-dir",
        default="",
        help="已有采集目录；填写后会在该目录继续补采，留空则新建目录。",
    )
    parser.add_argument(
        "--target-successes",
        type=int,
        default=100,
        help="目标成功轨迹数量；脚本会一直尝试，直到成功轨迹达到这个数量。",
    )
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=480, help="采集图像和生成视频的高度。")
    parser.add_argument("--image-width", type=int, default=640, help="采集图像和生成视频的宽度。")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--camera-quality", type=int, default=95)
    parser.add_argument(
        "--save-videos",
        type=parse_bool,
        default=True,
        help="是否为每条保存轨迹生成 mp4 预览视频，true/false。",
    )
    parser.add_argument(
        "--save-filter",
        choices=["success", "failure", "all"],
        default="success",
        help="保留哪些评估轨迹.",
    )
    return parser


def collect_data_from_checkpoint(
    *,
    ckpt_path: str,
    output_dir: str,
    append_run_dir: str,
    max_steps: int,
    image_height: int,
    image_width: int,
    seed: int,
    target_successes: int,
    save_videos: bool,
):
    """使用代码变量启动数据采集，其他参数沿用 build_arg_parser() 中的默认值。"""
    parser = build_arg_parser()
    args = parser.parse_args([])
    args.ckpt_path = ckpt_path
    args.output_dir = output_dir
    args.append_run_dir = append_run_dir
    args.max_steps = max_steps
    args.image_height = image_height
    args.image_width = image_width
    args.seed = seed
    args.target_successes = target_successes
    args.save_videos = save_videos
    ensure_python_hash_seed(args.seed)
    main(args)


def train_cli():
    parser = build_arg_parser()
    args = parser.parse_args()
    ensure_python_hash_seed(args.seed)
    main(args)


if __name__ == "__main__":
    # 需要采集数据的 checkpoint 路径；可以指向 checkpoint 文件夹或 pretrained_model 子文件夹。
    CKPT_PATH = "outputs/1_hugging_model/pre_sim_sew_needle_3arms_zed_wrist_diffusion"

    # 已有采集目录；为空时新建目录。人工剔除后要补采，就填旧目录路径。
    APPEND_RUN_DIR = ""

    # 原始采集数据保存目录。
    OUTPUT_DIR = "outputs/4_data_collect"

    # 每条轨迹最多执行多少步；通常不要超过环境 episode_length。
    MAX_STEPS = 160

    # 采集图像与生成视频的尺寸；视频尺寸会跟随图片帧尺寸。
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640

    # 随机种子；第 i 条轨迹会使用 SEED + i。
    SEED = 1000

    # 目标成功轨迹数量；脚本会一直尝试，直到成功轨迹达到这个数量。
    TARGET_SUCCESSES = 100

    # 是否额外生成每条轨迹的 mp4 预览视频；图片帧和数组数据始终会保存。
    SAVE_VIDEOS = True

    collect_data_from_checkpoint(
        ckpt_path=CKPT_PATH,
        output_dir=OUTPUT_DIR,
        append_run_dir=APPEND_RUN_DIR,
        max_steps=MAX_STEPS,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        seed=SEED,
        target_successes=TARGET_SUCCESSES,
        save_videos=SAVE_VIDEOS,
    )
