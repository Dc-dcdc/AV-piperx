#!/usr/bin/env python
"""用冻结扩散策略和多种固定执行步长采集动作价值离线轨迹。"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from lerobot.common.envs.utils import preprocess_observation
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

if __package__:
    from .train_replanning_dqn import (
        load_frozen_policy,
        make_training_env,
        policy_camera_names,
        resolve_device,
        resolve_project_path,
        seed_everything,
    )
else:
    from train.s4_adaptive_replanning.train_replanning_dqn import (
        load_frozen_policy,
        make_training_env,
        policy_camera_names,
        resolve_device,
        resolve_project_path,
        seed_everything,
    )


DATASET_SCHEMA_VERSION = 1


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    """先写临时文件再替换，避免中断后留下半份JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(dict(payload), file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def _add_batch_dim(value):
    """递归给单环境Gym观测添加batch维。"""
    if isinstance(value, dict):
        return {key: _add_batch_dim(item) for key, item in value.items()}
    array = np.asarray(value)
    return np.expand_dims(array.copy(), axis=0)


def prepare_policy_input(
    observation: Mapping,
    policy,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """把原始Gym观测转换为策略select_action需要的未归一化张量。"""
    batch = preprocess_observation(_add_batch_dim(observation))
    expected_keys = set(policy.config.input_shapes)
    policy_input = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in expected_keys
    }
    missing = expected_keys - set(policy_input)
    if missing:
        raise KeyError(f"环境观测缺少策略输入: {sorted(missing)}")
    return policy_input


def validate_collection_config(cfg: DictConfig) -> list[int]:
    """在加载模型和创建环境之前校验采集参数。"""
    if cfg.pretrained_ckpt_path is None:
        raise ValueError("必须设置 pretrained_ckpt_path")
    if cfg.dataset_dir is None:
        raise ValueError("必须设置 dataset_dir")
    raw_steps = list(cfg.collection.execution_steps)
    execution_steps = []
    seen = set()
    for raw_value in raw_steps:
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f"execution_steps必须为正整数，当前为{value}")
        if value not in seen:
            execution_steps.append(value)
            seen.add(value)
    if not execution_steps:
        raise ValueError("collection.execution_steps不能为空")
    if int(cfg.collection.episodes_per_step) <= 0:
        raise ValueError("collection.episodes_per_step必须大于0")
    if int(cfg.env.max_episode_steps) <= 0:
        raise ValueError("env.max_episode_steps必须大于0")
    quality = int(cfg.collection.image_quality)
    if not 1 <= quality <= 100:
        raise ValueError("collection.image_quality必须位于[1,100]")
    return execution_steps


def max_supported_execution_steps(policy) -> int:
    """计算策略动作切片和耦合方式允许的最大执行长度。"""
    config = policy.config
    start = max(0, int(config.n_obs_steps) - 1)
    maximum = int(config.horizon) - start
    if getattr(config, "coupling_mode", None) in {
        "rbac",
        "bidirectional_prefix_to_suffix",
    }:
        # 与固定步长评估保持一致：为这些前缀到后缀耦合方式保留一个suffix。
        maximum -= 1
    return maximum


def configure_policy_execution_steps(policy, execution_steps: int) -> None:
    """修改策略内部动作队列长度，使其每K步自动重新扩散推理。"""
    execution_steps = int(execution_steps)
    maximum = max_supported_execution_steps(policy)
    if execution_steps > maximum:
        raise ValueError(
            f"execution_steps={execution_steps}超过当前快照允许上限{maximum}；"
            f"horizon={policy.config.horizon}, n_obs_steps={policy.config.n_obs_steps}, "
            f"coupling_mode={getattr(policy.config, 'coupling_mode', None)}"
        )
    policy.config.n_action_steps = execution_steps
    policy.reset()


def extract_raw_observation(
    observation: Mapping,
    camera_names: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """提取原始20维状态和各策略相机的uint8 RGB图像。"""
    if "agent_pos" not in observation:
        raise KeyError("环境观测缺少agent_pos")
    state = np.asarray(observation["agent_pos"], dtype=np.float32).reshape(-1)
    pixels = observation.get("pixels")
    if not isinstance(pixels, Mapping):
        raise KeyError("环境观测缺少pixels字典")
    images = {}
    for camera in camera_names:
        if camera not in pixels:
            raise KeyError(f"环境观测缺少相机{camera}")
        image = np.asarray(pixels[camera])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"相机{camera}应为HWC RGB图像，当前为{image.shape}"
            )
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255.0, 0, 255)
            image = image.astype(np.uint8)
        images[camera] = np.ascontiguousarray(image)
    return state, images


def save_observation_frame(
    episode_dir: Path,
    frame_index: int,
    observation: Mapping,
    camera_names: list[str],
    image_quality: int,
) -> np.ndarray:
    """保存一帧多相机图像并返回对应关节状态。"""
    state, images = extract_raw_observation(observation, camera_names)
    for camera, image in images.items():
        image_path = (
            episode_dir
            / "images"
            / camera
            / f"{frame_index:06d}.jpg"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(image_path, image, quality=int(image_quality))
    return state


def execution_steps_group_name(execution_steps: int) -> str:
    """返回稳定、可排序的执行步长分组目录名。"""
    execution_steps = int(execution_steps)
    if execution_steps <= 0:
        raise ValueError(f"execution_steps必须为正整数，当前为{execution_steps}")
    return f"execution_steps_{execution_steps:03d}"


def existing_episode_infos(dataset_dir: Path) -> list[dict]:
    """递归读取完整episode信息，兼容新分组目录和旧扁平目录。"""
    infos = []
    episodes_dir = dataset_dir / "episodes"
    if not episodes_dir.is_dir():
        return infos
    for episode_dir in sorted(episodes_dir.rglob("episode_*")):
        if not episode_dir.is_dir() or episode_dir.name.endswith(".tmp"):
            continue
        suffix = episode_dir.name.removeprefix("episode_")
        if len(suffix) != 6 or not suffix.isdigit():
            continue
        info_path = episode_dir / "info.json"
        arrays_path = episode_dir / "arrays.npz"
        if not info_path.is_file() or not arrays_path.is_file():
            continue
        with info_path.open("r", encoding="utf-8") as file:
            info = json.load(file)
        info["_storage_local_episode_index"] = int(suffix)
        try:
            expected_group = execution_steps_group_name(
                int(info["execution_steps"])
            )
        except (KeyError, TypeError, ValueError):
            expected_group = None
        info["_uses_grouped_layout"] = (
            expected_group is not None
            and episode_dir.parent.name == expected_group
        )
        infos.append(info)
    return infos


def next_collection_episode_indices(
    episode_infos: list[dict],
    execution_steps: list[int],
) -> tuple[dict[int, int], int]:
    """恢复每个步长的局部编号以及整个数据集的全局编号。"""
    next_global_index = (
        max(
            (
                int(
                    info.get(
                        "global_episode_index",
                        info.get("episode_index", -1),
                    )
                )
                for info in episode_infos
            ),
            default=-1,
        )
        + 1
    )
    next_local_indices = {}
    for value in execution_steps:
        step_infos = [
            info
            for info in episode_infos
            if int(info.get("execution_steps", -1)) == value
        ]
        grouped_local_indices = [
            int(info["_storage_local_episode_index"])
            for info in step_infos
            if bool(info.get("_uses_grouped_layout", False))
        ]
        # 纯旧版扁平数据没有局部路径编号，此时从该步长已有数量继续。
        next_local_indices[value] = (
            max(grouped_local_indices) + 1
            if grouped_local_indices
            else len(step_infos)
        )
    return next_local_indices, next_global_index


def make_dataset_metadata(
    cfg: DictConfig,
    model_dir: Path,
    policy,
    camera_names: list[str],
    execution_steps: list[int],
    env_id: str,
    episode_infos: list[dict],
) -> dict:
    """构造可独立校验和恢复采集语义的数据集元数据。"""
    all_execution_steps = sorted(
        {
            *(int(value) for value in execution_steps),
            *(
                int(info["execution_steps"])
                for info in episode_infos
                if "execution_steps" in info
            ),
        }
    )
    counts = {
        str(value): sum(
            int(info.get("execution_steps", -1)) == value
            for info in episode_infos
        )
        for value in all_execution_steps
    }
    successes = {
        str(value): sum(
            int(info.get("execution_steps", -1)) == value
            and bool(info.get("success", False))
            for info in episode_infos
        )
        for value in all_execution_steps
    }
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_type": "offline_action_value_sarsa",
        "pretrained_model_path": str(model_dir),
        "policy_name": str(policy.name),
        "env_id": env_id,
        "camera_names": list(camera_names),
        "state_dim": int(policy.config.input_shapes["observation.state"][0]),
        "action_dim": int(policy.config.output_shapes["action"][0]),
        "n_obs_steps": 2,
        "execution_steps": all_execution_steps,
        "episode_layout": (
            "episodes/execution_steps_{execution_steps:03d}/"
            "episode_{episode_index:06d}"
        ),
        "episode_index_scope": "per_execution_steps",
        "global_episode_index_scope": "whole_dataset",
        "episodes_per_step": int(cfg.collection.episodes_per_step),
        "max_episode_steps": int(cfg.env.max_episode_steps),
        "image_quality": int(cfg.collection.image_quality),
        "episode_count": len(episode_infos),
        "episode_counts_by_execution_steps": counts,
        "success_counts_by_execution_steps": successes,
        "reward_definition": "1 only on successful transition, otherwise 0",
    }


def validate_existing_metadata(
    metadata: Mapping,
    policy,
    camera_names: list[str],
) -> None:
    """拒绝把不同输入结构的数据追加到同一目录。"""
    expected = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "policy_name": str(policy.name),
        "camera_names": list(camera_names),
        "state_dim": int(policy.config.input_shapes["observation.state"][0]),
        "action_dim": int(policy.config.output_shapes["action"][0]),
        "n_obs_steps": 2,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"现有数据集与当前快照结构不一致: {mismatches}")


def collect_one_episode(
    *,
    env,
    policy,
    device: torch.device,
    episode_dir: Path,
    episode_index: int,
    global_episode_index: int,
    episode_seed: int,
    execution_steps: int,
    camera_names: list[str],
    max_episode_steps: int,
    image_quality: int,
    use_amp: bool,
) -> dict:
    """采集并原子写入一条满足T+1观测/T动作约束的轨迹。"""
    temporary_dir = episode_dir.with_name(f".{episode_dir.name}.tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    actions: list[np.ndarray] = []
    rewards: list[float] = []
    environment_rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    states: list[np.ndarray] = []
    inference_times_ms: list[float] = []
    success = False
    physics_error = None

    try:
        seed_everything(episode_seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(episode_seed)
        observation, _ = env.reset(seed=episode_seed)
        configure_policy_execution_steps(policy, execution_steps)
        states.append(
            save_observation_frame(
                temporary_dir,
                0,
                observation,
                camera_names,
                image_quality,
            )
        )

        for transition_index in range(max_episode_steps):
            policy_input = prepare_policy_input(observation, policy, device)
            queue_empty = len(policy._queues["action"]) == 0
            if queue_empty and device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast_context:
                action_tensor = policy.select_action(policy_input)
            if queue_empty:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_times_ms.append(
                    (time.perf_counter() - inference_start) * 1000.0
                )
            action = (
                action_tensor.squeeze(0)
                .detach()
                .to(dtype=torch.float32, device="cpu")
                .numpy()
            )

            try:
                (
                    next_observation,
                    environment_reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)
            except Exception as exc:  # MuJoCo异常也必须形成一个合法失败终止转移。
                logging.exception(
                    "episode=%d global_episode=%d step=%d"
                    "物理环境异常，按失败截断保存",
                    episode_index,
                    global_episode_index,
                    transition_index,
                )
                next_observation = observation
                environment_reward = 0.0
                terminated = False
                truncated = True
                info = {"is_success": False}
                physics_error = repr(exc)

            success = bool(info.get("is_success", False))
            if success:
                terminated = True
            is_last_allowed_step = transition_index + 1 >= max_episode_steps
            if is_last_allowed_step and not terminated:
                truncated = True

            actions.append(np.asarray(action, dtype=np.float32).copy())
            rewards.append(1.0 if success else 0.0)
            environment_rewards.append(float(environment_reward))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            states.append(
                save_observation_frame(
                    temporary_dir,
                    transition_index + 1,
                    next_observation,
                    camera_names,
                    image_quality,
                )
            )
            observation = next_observation
            if terminated or truncated:
                break

        if not actions:
            raise RuntimeError("episode未产生任何环境transition")
        np.savez_compressed(
            temporary_dir / "arrays.npz",
            observation_state=np.asarray(states, dtype=np.float32),
            action=np.asarray(actions, dtype=np.float32),
            reward=np.asarray(rewards, dtype=np.float32),
            environment_reward=np.asarray(
                environment_rewards,
                dtype=np.float32,
            ),
            terminated=np.asarray(terminated_flags, dtype=np.bool_),
            truncated=np.asarray(truncated_flags, dtype=np.bool_),
        )
        info_payload = {
            "episode_index": int(episode_index),
            "global_episode_index": int(global_episode_index),
            "seed": int(episode_seed),
            "execution_steps": int(execution_steps),
            "success": bool(success),
            "length": len(actions),
            "sparse_return": float(sum(rewards)),
            "environment_return": float(sum(environment_rewards)),
            "inference_count": len(inference_times_ms),
            "average_inference_ms": (
                float(np.mean(inference_times_ms))
                if inference_times_ms
                else 0.0
            ),
            "physics_error": physics_error,
        }
        _write_json_atomic(temporary_dir / "info.json", info_payload)
        if episode_dir.exists():
            raise FileExistsError(f"目标episode已存在: {episode_dir}")
        temporary_dir.rename(episode_dir)
        return info_payload
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def collect_action_value_dataset(cfg: DictConfig) -> None:
    """主采集流程：每种固定执行步长保存相同数量的自然分布episode。"""
    execution_steps_values = validate_collection_config(cfg)
    seed_everything(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    policy, saved_policy_config, _, model_dir = load_frozen_policy(
        cfg.pretrained_ckpt_path,
        device,
    )
    if int(policy.config.n_obs_steps) != 2:
        raise ValueError(
            "当前动作价值数据定义固定使用两帧历史，但快照"
            f"n_obs_steps={policy.config.n_obs_steps}"
        )
    if bool(getattr(policy, "use_env_state", False)):
        raise NotImplementedError(
            "当前动作价值采集仅支持图像和observation.state，不支持environment_state"
        )
    camera_names = policy_camera_names(policy)
    original_action_steps = int(policy.config.n_action_steps)
    maximum_execution_steps = max_supported_execution_steps(policy)
    for execution_steps in execution_steps_values:
        if execution_steps > maximum_execution_steps:
            raise ValueError(
                f"execution_steps={execution_steps}超过当前快照允许上限"
                f"{maximum_execution_steps}；horizon={policy.config.horizon}, "
                f"n_obs_steps={policy.config.n_obs_steps}, "
                f"coupling_mode={getattr(policy.config, 'coupling_mode', None)}"
            )

    dataset_dir = resolve_project_path(cfg.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "episodes").mkdir(exist_ok=True)
    existing_infos = existing_episode_infos(dataset_dir)
    if existing_infos and not bool(cfg.collection.resume):
        raise FileExistsError(
            f"数据集已有{len(existing_infos)}条episode；"
            "继续采集请设置collection.resume=true，或使用新dataset_dir"
        )

    env, env_id = make_training_env(
        policy,
        saved_policy_config,
        cfg.env,
    )
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            validate_existing_metadata(
                json.load(file),
                policy,
                camera_names,
            )

    completed_counts = {
        value: sum(
            int(info.get("execution_steps", -1)) == value
            for info in existing_infos
        )
        for value in execution_steps_values
    }
    (
        next_local_episode_indices,
        next_global_episode_index,
    ) = next_collection_episode_indices(
        existing_infos,
        execution_steps_values,
    )
    logging.info(
        "动作价值轨迹采集开始: env=%s, policy=%s, steps=%s, "
        "episodes_per_step=%d, dataset=%s",
        env_id,
        policy.name,
        execution_steps_values,
        int(cfg.collection.episodes_per_step),
        dataset_dir,
    )

    try:
        for execution_steps in execution_steps_values:
            remaining = (
                int(cfg.collection.episodes_per_step)
                - completed_counts[execution_steps]
            )
            if remaining <= 0:
                logging.info(
                    "execution_steps=%d已完成%d条，跳过",
                    execution_steps,
                    completed_counts[execution_steps],
                )
                continue
            progress = tqdm(
                range(remaining),
                desc=f"collect K={execution_steps}",
                leave=True,
            )
            for collection_offset in progress:
                episode_index = next_local_episode_indices[execution_steps]
                global_episode_index = next_global_episode_index
                episode_seed = (
                    int(cfg.seed)
                    + execution_steps * 1_000_000
                    + completed_counts[execution_steps]
                    + collection_offset
                )
                episode_dir = (
                    dataset_dir
                    / "episodes"
                    / execution_steps_group_name(execution_steps)
                    / f"episode_{episode_index:06d}"
                )
                info = collect_one_episode(
                    env=env,
                    policy=policy,
                    device=device,
                    episode_dir=episode_dir,
                    episode_index=episode_index,
                    global_episode_index=global_episode_index,
                    episode_seed=episode_seed,
                    execution_steps=execution_steps,
                    camera_names=camera_names,
                    max_episode_steps=int(cfg.env.max_episode_steps),
                    image_quality=int(cfg.collection.image_quality),
                    use_amp=bool(cfg.use_amp),
                )
                existing_infos.append(info)
                next_local_episode_indices[execution_steps] += 1
                next_global_episode_index += 1
                metadata = make_dataset_metadata(
                    cfg,
                    model_dir,
                    policy,
                    camera_names,
                    execution_steps_values,
                    env_id,
                    existing_infos,
                )
                _write_json_atomic(metadata_path, metadata)
                step_infos = [
                    item
                    for item in existing_infos
                    if int(item.get("execution_steps", -1))
                    == execution_steps
                ]
                progress.set_postfix(
                    success_rate=(
                        sum(bool(item["success"]) for item in step_infos)
                        / len(step_infos)
                    ),
                    length=info["length"],
                )
    finally:
        policy.config.n_action_steps = original_action_steps
        policy.reset()
        env.close()

    logging.info(
        "采集完成: episodes=%d, dataset=%s",
        len(existing_infos),
        dataset_dir,
    )


@hydra.main(
    version_base="1.2",
    config_path="../../configs/adaptive_replanning",
    config_name="action_value_collect",
)
def collect_cli(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("动作价值采集配置:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    collect_action_value_dataset(cfg)


def has_cli_override(args: list[str], key: str) -> bool:
    """判断用户是否已经在命令行显式设置某个Hydra参数。"""
    for argument in args:
        argument_key = argument.split("=", maxsplit=1)[0].lstrip("+")
        if argument_key == key:
            return True
    return False


if __name__ == "__main__":
    # ==========================================
    # 常用采集参数：直接修改这里即可
    # ==========================================
    # 这些值相当于Hydra命令行覆盖项；若启动命令中显式传入同名参数，
    # 则以命令行值为准，不会被下面的本地默认值覆盖。
    default_args = [
        # 冻结的预训练扩散策略快照，用来生成并执行采集动作。
        "pretrained_ckpt_path='outputs/2_pretrain/train/2026-07-16/"
        "20-59-53_InsertCylinder-3Arms-v0_pre_zed_dual_head_diffusion/"
        "checkpoints/162000_loss=0.0051_sr=87.0_ar=751.50'",
        # 原始图像、关节、动作和终止信息的数据集保存目录。
        "dataset_dir=outputs/6_action_value_datasets/insert_cylinder",
        # 推理设备与自动混合精度。
        "device=cuda:0",
        "use_amp=true",
        # 环境允许的最大交互步数。
        "env.max_episode_steps=400",
        # 固定动作执行步长；每种步长分别保留自然成功率。
        "collection.execution_steps=[8]",
        # 每种执行步长最终需要达到的episode数量。
        "collection.episodes_per_step=100",
        # JPEG保存质量，数值越高图像越清晰、占用空间越大。
        "collection.image_quality=95",
        # true表示已有数据时继续补采，false要求目标目录为空。
        "collection.resume=true",
        # 环境和episode随机种子。
        "seed=1000",
    ]

    original_cli_args = sys.argv[1:]
    for argument in default_args:
        key = argument.split("=", maxsplit=1)[0].lstrip("+")
        if not has_cli_override(original_cli_args, key):
            sys.argv.append(argument)

    collect_cli()
