#!/usr/bin/env python

"""Collect task-outcome supervision for a none/Arm-to-View Router."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import env  # noqa: F401 - register Gym environments
from lerobot.common.policies.diffusion.modeling_post_diffusion_output_corrector import (
    PostDiffusionOutputCorrectorPolicy,
)
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
)
from train.s1_pretrain.eval.eval_policy import (
    prepare_policy_observation,
    seed_env_spaces,
    seed_runtime,
)
from train.s2_incremental.router.counterfactual_dataset import (
    ROUTER_CACHE_ARRAYS,
    ROUTER_CACHE_SCHEMA_VERSION,
    BranchOutcome,
    capture_environment_state,
    capture_runtime_rng,
    clone_policy_queues,
    extract_stage,
    make_counterfactual_label,
    next_none_action_and_candidates,
    restore_environment_state,
    restore_policy_queues,
    restore_runtime_rng,
    stable_cache_hash,
    update_observation_queues,
)


def _resolve_model_dir(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    if not path.is_dir():
        raise FileNotFoundError(f"输出修正器checkpoint不存在: {path}")
    return path


def _find_config(model_dir: Path) -> Path:
    for path in (
        model_dir / "config.yaml",
        model_dir.parent / "config.yaml",
        model_dir.parent.parent / ".hydra" / "config.yaml",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"checkpoint附近找不到config.yaml: {model_dir}")


def _checkpoint_fingerprint(model_dir: Path) -> str:
    hasher = hashlib.sha256()
    files = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file()
        and (
            path.suffix in {".safetensors", ".bin", ".pt", ".pth"}
            or path.name in {"config.json", "config.yaml"}
        )
    )
    if not files:
        raise FileNotFoundError(f"模型目录没有可哈希的权重: {model_dir}")
    for path in files:
        hasher.update(path.name.encode("utf-8"))
        with open(path, "rb") as file:
            while chunk := file.read(8 * 1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def _load_source_policy(checkpoint: str | Path, device):
    model_dir = _resolve_model_dir(checkpoint)
    source_cfg = init_hydra_config(str(_find_config(model_dir)))
    source_cfg.device = str(device)
    policy = make_policy(
        hydra_cfg=source_cfg,
        pretrained_policy_name_or_path=str(model_dir),
        strict_pretrained_loading=True,
    )
    if not isinstance(policy, PostDiffusionOutputCorrectorPolicy):
        raise TypeError(
            "Router数据必须由post_diffusion_output_corrector checkpoint生成，"
            f"实际为{type(policy).__name__}。"
        )
    policy.eval()
    policy.to(device)
    return policy, source_cfg, model_dir


def _make_env(source_cfg, policy, collection_cfg):
    env_id = f"{source_cfg.env.name}/{source_cfg.env.task}"
    cameras = [
        key.removeprefix("observation.images.")
        for key in policy.config.input_shapes
        if key.startswith("observation.images.")
    ]
    kwargs = {
        "cameras": cameras,
        "episode_length": int(collection_cfg.max_steps),
    }
    if bool(collection_cfg.enable_reward_debug):
        kwargs["enable_reward_debug"] = True
    try:
        environment = gym.make(id=env_id, **kwargs).unwrapped
    except TypeError:
        kwargs.pop("enable_reward_debug", None)
        environment = gym.make(id=env_id, **kwargs).unwrapped
        logging.warning(
            "当前任务不支持enable_reward_debug，Router标签将退化为成功率和回报。"
        )
    return environment, env_id, cameras


def _run_branch(
    *,
    environment,
    policy,
    environment_snapshot,
    runtime_rng_snapshot,
    first_chunk: np.ndarray,
    expected_keys: set[str],
    device,
    max_steps: int,
) -> BranchOutcome:
    observation = restore_environment_state(environment, environment_snapshot)
    restore_runtime_rng(runtime_rng_snapshot)
    policy.reset()
    original_scales = (
        float(policy.diffusion.view_to_arm_output_scale),
        float(policy.diffusion.arm_to_view_output_scale),
    )
    policy.diffusion.set_output_correction_scales(
        view_to_arm=0.0,
        arm_to_view=0.0,
    )

    reward_sum = 0.0
    max_stage = 0
    success = False
    done = False
    steps = 0
    try:
        for action in first_chunk:
            observation_batch = prepare_policy_observation(
                observation,
                expected_keys,
                device,
            )
            update_observation_queues(policy, observation_batch)
            observation, reward, terminated, truncated, info = (
                environment.step(np.asarray(action, dtype=np.float32))
            )
            steps += 1
            reward_sum += float(reward)
            max_stage = max(max_stage, extract_stage(info))
            success = bool(info.get("is_success", False))
            done = bool(terminated or truncated)
            if done:
                break

        # The open-loop chunk advanced the observation-only queue before each
        # action.  The first continuation call appends the final observation,
        # reproducing the same two-frame history used by normal action chunking.
        while (
            not done
            and int(environment._current_step) < int(max_steps)
        ):
            observation_batch = prepare_policy_observation(
                observation,
                expected_keys,
                device,
            )
            with torch.inference_mode():
                action = policy.select_action(observation_batch)
            observation, reward, terminated, truncated, info = environment.step(
                action[0].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            steps += 1
            reward_sum += float(reward)
            max_stage = max(max_stage, extract_stage(info))
            success = bool(info.get("is_success", False))
            done = bool(terminated or truncated)
    finally:
        policy.diffusion.set_output_correction_scales(
            view_to_arm=original_scales[0],
            arm_to_view=original_scales[1],
        )

    return BranchOutcome(
        success=success,
        max_stage=max_stage,
        reward=reward_sum,
        steps=steps,
    )


def _empty_records() -> dict[str, list]:
    return {name: [] for name in ROUTER_CACHE_ARRAYS}


def _append_record(
    records: dict[str, list],
    *,
    candidates,
    label: int,
    sample_weight: float,
    reason: int,
    none_outcome: BranchOutcome,
    arm_to_view_outcome: BranchOutcome,
    episode_seed: int,
    decision_step: int,
) -> None:
    records["global_condition"].append(
        candidates.global_condition[0].numpy()
    )
    records["none_trajectory"].append(
        candidates.none_trajectory[0].numpy()
    )
    records["arm_to_view_trajectory"].append(
        candidates.arm_to_view_trajectory[0].numpy()
    )
    records["router_label"].append(label)
    records["sample_weight"].append(sample_weight)
    records["quality_none"].append(none_outcome.as_array())
    records["quality_arm_to_view"].append(arm_to_view_outcome.as_array())
    records["episode_seed"].append(episode_seed)
    records["decision_step"].append(decision_step)
    records["label_reason"].append(reason)


def _save_records(
    cache_dir: Path,
    records: dict[str, list],
    *,
    identity: dict,
    env_id: str,
    checkpoint: Path,
) -> Path:
    if not records["router_label"]:
        raise RuntimeError("没有收集到任何Router反事实样本。")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dtypes = {
        "global_condition": np.float16,
        "none_trajectory": np.float16,
        "arm_to_view_trajectory": np.float16,
        "router_label": np.int8,
        "sample_weight": np.float32,
        "quality_none": np.float32,
        "quality_arm_to_view": np.float32,
        "episode_seed": np.int64,
        "decision_step": np.int32,
        "label_reason": np.int8,
    }
    arrays = {}
    for name, filename in ROUTER_CACHE_ARRAYS.items():
        array = np.asarray(records[name], dtype=dtypes[name])
        np.save(cache_dir / filename, array)
        arrays[name] = filename

    labels = np.asarray(records["router_label"], dtype=np.int8)
    manifest = {
        "schema_version": ROUTER_CACHE_SCHEMA_VERSION,
        "cache_identity": identity,
        "source_checkpoint": str(checkpoint),
        "environment": env_id,
        "num_samples": int(len(labels)),
        "num_labeled": int((labels >= 0).sum()),
        "num_none": int((labels == 0).sum()),
        "num_arm_to_view": int((labels == 1).sum()),
        "num_ambiguous": int((labels < 0).sum()),
        "arrays": arrays,
    }
    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    return manifest_path


def collect(cfg: DictConfig) -> Path:
    collection_cfg = cfg.router_collection
    device = get_safe_torch_device(str(collection_cfg.device))
    policy, source_cfg, model_dir = _load_source_policy(
        cfg.init_policy_path,
        device,
    )
    environment, env_id, cameras = _make_env(
        source_cfg,
        policy,
        collection_cfg,
    )
    expected_keys = set(policy.config.input_shapes)
    source_fingerprint = _checkpoint_fingerprint(model_dir)
    start_seed = int(collection_cfg.start_seed)
    n_episodes = int(collection_cfg.n_episodes)
    group_identity = {
        "source_checkpoint_fingerprint": source_fingerprint,
        "env_id": env_id,
        "max_steps": int(collection_cfg.max_steps),
        "collect_every_decisions": int(
            collection_cfg.collect_every_decisions
        ),
        "max_samples_per_episode": int(
            collection_cfg.max_samples_per_episode
        ),
        "arm_to_view_scale": float(collection_cfg.arm_to_view_scale),
        "reward_margin": float(collection_cfg.reward_margin),
        "horizon": int(policy.config.horizon),
        "n_action_steps": int(policy.config.n_action_steps),
    }
    identity = {
        **group_identity,
        "start_seed": start_seed,
        "n_episodes": n_episodes,
    }
    group_name = f"none_av_router_{stable_cache_hash(group_identity)}"
    seed_name = f"seed={start_seed}_ep={n_episodes}"
    cache_dir = (
        Path(collection_cfg.root_dir).expanduser().resolve()
        / group_name
        / seed_name
    )
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file() and not bool(collection_cfg.rebuild):
        logging.info("复用已有Router反事实缓存: %s", manifest_path)
        return manifest_path

    records = _empty_records()
    try:
        progress = tqdm(range(n_episodes), desc="Router counterfactual")
        for episode_offset in progress:
            episode_seed = start_seed + episode_offset
            seed_runtime(episode_seed)
            seed_env_spaces(environment, episode_seed)
            observation, _ = environment.reset(seed=episode_seed)
            policy.reset()
            seed_runtime(episode_seed)
            done = False
            decision_index = 0
            collected_this_episode = 0

            while (
                not done
                and int(environment._current_step)
                < int(collection_cfg.max_steps)
            ):
                branch_snapshot = (
                    capture_environment_state(environment)
                    if len(policy._queues["action"]) == 0
                    else None
                )
                observation_batch = prepare_policy_observation(
                    observation,
                    expected_keys,
                    device,
                )
                action, candidates = next_none_action_and_candidates(
                    policy,
                    observation_batch,
                    arm_to_view_scale=float(
                        collection_cfg.arm_to_view_scale
                    ),
                )

                if candidates is not None:
                    should_collect = (
                        decision_index
                        % int(collection_cfg.collect_every_decisions)
                        == 0
                        and collected_this_episode
                        < int(collection_cfg.max_samples_per_episode)
                    )
                    decision_index += 1
                    if should_collect:
                        if branch_snapshot is None:
                            raise RuntimeError("Router分支点缺少环境快照。")
                        main_queues = clone_policy_queues(policy)
                        post_candidate_rng = capture_runtime_rng()
                        none_outcome = _run_branch(
                            environment=environment,
                            policy=policy,
                            environment_snapshot=branch_snapshot,
                            runtime_rng_snapshot=post_candidate_rng,
                            first_chunk=candidates.none_chunk,
                            expected_keys=expected_keys,
                            device=device,
                            max_steps=int(collection_cfg.max_steps),
                        )
                        arm_to_view_outcome = _run_branch(
                            environment=environment,
                            policy=policy,
                            environment_snapshot=branch_snapshot,
                            runtime_rng_snapshot=post_candidate_rng,
                            first_chunk=candidates.arm_to_view_chunk,
                            expected_keys=expected_keys,
                            device=device,
                            max_steps=int(collection_cfg.max_steps),
                        )
                        label, weight, reason = make_counterfactual_label(
                            none_outcome,
                            arm_to_view_outcome,
                            reward_margin=float(
                                collection_cfg.reward_margin
                            ),
                        )
                        _append_record(
                            records,
                            candidates=candidates,
                            label=label,
                            sample_weight=weight,
                            reason=reason,
                            none_outcome=none_outcome,
                            arm_to_view_outcome=arm_to_view_outcome,
                            episode_seed=episode_seed,
                            decision_step=int(branch_snapshot.current_step),
                        )
                        collected_this_episode += 1

                        observation = restore_environment_state(
                            environment,
                            branch_snapshot,
                        )
                        restore_policy_queues(policy, main_queues)
                        restore_runtime_rng(post_candidate_rng)

                observation, _, terminated, truncated, _ = environment.step(
                    action[0].detach().cpu().numpy().astype(
                        np.float32,
                        copy=False,
                    )
                )
                done = bool(terminated or truncated)

            progress.set_postfix(samples=len(records["router_label"]))
    finally:
        environment.close()

    manifest_path = _save_records(
        cache_dir,
        records,
        identity=identity,
        env_id=env_id,
        checkpoint=model_dir,
    )
    logging.info("Router反事实缓存已保存: %s", manifest_path)
    return manifest_path


@hydra.main(
    version_base="1.2",
    config_name="pre_default",
    config_path="../../../configs/pretrain",
)
def collect_cli(cfg: DictConfig) -> None:
    init_logging()
    manifest = collect(cfg)
    print(f"Router manifest: {manifest}")


if __name__ == "__main__":
    default_args = [
        # 已完成训练的后置修正器复合checkpoint。
        "init_policy_path='outputs/2_pretrain/post_diffusion_output_corrector/InsertCylinder-3Arms-v0/2026-07-28/00-20-36_InsertCylinder-3Arms-v0_pre_zed_post_diffusion_output_corrector/checkpoints/099999_loss=0.0016_sr=86.0_ar=738.49'",
        # 环境字段用于Hydra补全；实际任务以checkpoint配置为准。
        "env=sim_insert_cylinder_3arms",
        # Router配置同时提供采集参数和后续目标网络结构。
        "policy=pre_zed_none_av_router",
        # 数据采集不需要训练数据集，但pre_default要求保留合法字段。
        "dataset_local_dir=outputs/5_hf_datasets/quest_teleop_insert_cylinder_3arms_rgb_joint",
        "dataset_repo_id=Dc-dc/quest_teleop_insert_cylinder_3arms_rgb_joint",
        "wandb.enable=false",
    ]
    for argument in default_args:
        key = argument.split("=", 1)[0]
        if not any(
            item.split("=", 1)[0] == key for item in sys.argv[1:]
        ):
            sys.argv.append(argument)
    collect_cli()
