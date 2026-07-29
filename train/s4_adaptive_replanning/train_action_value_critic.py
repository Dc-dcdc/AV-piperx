#!/usr/bin/env python
"""使用原始多步长轨迹离线训练标量动作价值 Critic。"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

if __package__:
    from .action_value_critic import (
        ActionValueCritic,
        ActionValueCriticConfig,
        ActionValueTransitionDataset,
        load_action_value_episodes,
        make_target_critic,
        polyak_update,
        split_episodes_by_execution_steps,
    )
    from .train_replanning_dqn import (
        load_frozen_policy,
        policy_camera_names,
        resolve_device,
        resolve_project_path,
        seed_everything,
    )
else:
    from train.s4_adaptive_replanning.action_value_critic import (
        ActionValueCritic,
        ActionValueCriticConfig,
        ActionValueTransitionDataset,
        load_action_value_episodes,
        make_target_critic,
        polyak_update,
        split_episodes_by_execution_steps,
    )
    from train.s4_adaptive_replanning.train_replanning_dqn import (
        load_frozen_policy,
        policy_camera_names,
        resolve_device,
        resolve_project_path,
        seed_everything,
    )


def validate_training_config(cfg: DictConfig) -> None:
    """校验离线单步SARSA训练所需的关键超参数。"""
    if cfg.pretrained_ckpt_path is None:
        raise ValueError("必须设置 pretrained_ckpt_path")
    if cfg.dataset_dir is None:
        raise ValueError("必须设置 dataset_dir")
    positive_ints = {
        "training.epochs": cfg.training.epochs,
        "training.batch_size": cfg.training.batch_size,
        "training.save_freq_epochs": cfg.training.save_freq_epochs,
        "training.log_freq_steps": cfg.training.log_freq_steps,
        "critic.visual_embed_dim": cfg.critic.visual_embed_dim,
        "critic.state_embed_dim": cfg.critic.state_embed_dim,
        "critic.action_embed_dim": cfg.critic.action_embed_dim,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"{name}必须大于0，当前为{value}")
    if int(cfg.training.num_workers) < 0:
        raise ValueError("training.num_workers不能为负数")
    if not 0.0 <= float(cfg.training.validation_ratio) < 1.0:
        raise ValueError("training.validation_ratio必须位于[0,1)")
    if not 0.0 <= float(cfg.training.gamma) <= 1.0:
        raise ValueError("training.gamma必须位于[0,1]")
    if not 0.0 < float(cfg.training.target_update_tau) <= 1.0:
        raise ValueError("training.target_update_tau必须位于(0,1]")
    if float(cfg.training.learning_rate) <= 0.0:
        raise ValueError("training.learning_rate必须大于0")
    if float(cfg.training.grad_clip_norm) <= 0.0:
        raise ValueError("training.grad_clip_norm必须大于0")
    hidden_dims = [int(value) for value in cfg.critic.hidden_dims]
    if not hidden_dims or any(value <= 0 for value in hidden_dims):
        raise ValueError("critic.hidden_dims必须包含正整数")


def requested_training_execution_steps(
    cfg: DictConfig,
) -> list[int] | None:
    """解析训练数据步长选择；null表示使用数据集中的全部步长。"""
    value = OmegaConf.select(cfg, "training.execution_steps", default=None)
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        raw_values = [int(value)]
    else:
        try:
            raw_values = [int(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "training.execution_steps应为null、单个整数或整数列表，"
                "例如null、4、[4]、[4,8]"
            ) from exc
    selected = list(dict.fromkeys(raw_values))
    if not selected or any(item <= 0 for item in selected):
        raise ValueError(
            "training.execution_steps必须包含至少一个正整数，"
            "使用全部步长请设为null"
        )
    return selected


def validate_dataset_against_policy(
    metadata: Mapping,
    policy,
) -> list[str]:
    """确保轨迹的相机、状态和动作结构与冻结快照完全一致。"""
    camera_names = policy_camera_names(policy)
    expected = {
        "schema_version": 1,
        "policy_name": str(policy.name),
        "camera_names": camera_names,
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
        raise ValueError(f"动作价值数据集与预训练策略不一致: {mismatches}")
    if int(policy.config.n_obs_steps) != 2:
        raise ValueError(
            f"动作价值Critic固定使用两帧，但策略n_obs_steps={policy.config.n_obs_steps}"
        )
    if bool(getattr(policy, "use_env_state", False)):
        raise NotImplementedError(
            "当前Critic仅支持图像和observation.state，不支持environment_state"
        )
    return camera_names


def make_dataloader(
    dataset: ActionValueTransitionDataset,
    cfg: DictConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """构造适合JPEG随机读取的DataLoader。"""
    num_workers = int(cfg.training.num_workers)
    kwargs = {
        "batch_size": int(cfg.training.batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": torch.Generator().manual_seed(int(seed)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            cfg.training.persistent_workers
        )
        prefetch_factor = int(cfg.training.prefetch_factor)
        if prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def _autocast_context(device: torch.device, enabled: bool):
    """动作价值训练默认使用CUDA bfloat16，不需要GradScaler。"""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def encode_transition_batch(
    batch: Mapping[str, torch.Tensor],
    policy,
    camera_names: list[str],
    device: torch.device,
    use_amp: bool,
) -> dict[str, torch.Tensor]:
    """实时编码三帧观测，并共享中间帧构造当前/下一两帧特征。"""
    images = batch["images"].to(device, non_blocking=True)  # [B,3,N,H,W,C]
    joint_states = batch["joint_states"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    actions = torch.stack(
        [batch["action"], batch["next_action"]],
        dim=1,
    ).to(device=device, dtype=torch.float32, non_blocking=True)

    if images.ndim != 6 or images.shape[1] != 3:
        raise ValueError(f"images应为[B,3,N,H,W,C]，当前为{images.shape}")
    if images.shape[2] != len(camera_names) or images.shape[-1] != 3:
        raise ValueError(
            f"images相机/RGB维度异常: shape={images.shape}, cameras={camera_names}"
        )
    if joint_states.ndim != 3 or joint_states.shape[1] != 3:
        raise ValueError(
            f"joint_states应为[B,3,D]，当前为{joint_states.shape}"
        )

    batch_size, frame_count = joint_states.shape[:2]
    flat_inputs = {
        "observation.state": joint_states.reshape(
            batch_size * frame_count,
            joint_states.shape[-1],
        )
    }
    images = (
        images.permute(0, 1, 2, 5, 3, 4)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    for camera_index, image_key in enumerate(policy.expected_image_keys):
        flat_inputs[image_key] = images[:, :, camera_index].reshape(
            batch_size * frame_count,
            *images.shape[3:],
        )
    normalized_inputs = policy.normalize_inputs(flat_inputs)
    normalized_states = normalized_inputs["observation.state"].reshape(
        batch_size,
        frame_count,
        -1,
    )
    normalized_actions = policy.normalize_targets(
        {"action": actions}
    )["action"]

    flat_camera_images = torch.stack(
        [
            normalized_inputs[image_key]
            for image_key in policy.expected_image_keys
        ],
        dim=1,
    )
    with _autocast_context(device, use_amp):
        flat_visual_features = policy.diffusion.rgb_encoder(
            flat_camera_images.flatten(0, 1)
        )
    if flat_visual_features.ndim != 2:
        raise ValueError(
            "rgb_encoder输出必须为二维，当前为"
            f"{flat_visual_features.shape}"
        )
    feature_dim = flat_visual_features.shape[-1]
    visual_features = flat_visual_features.reshape(
        batch_size,
        frame_count,
        len(camera_names),
        feature_dim,
    ).flatten(start_dim=2)

    return {
        "current_visual": visual_features[:, 0:2].flatten(start_dim=1),
        "next_visual": visual_features[:, 1:3].flatten(start_dim=1),
        "current_state": normalized_states[:, 0:2].flatten(start_dim=1),
        "next_state": normalized_states[:, 1:3].flatten(start_dim=1),
        "action": normalized_actions[:, 0],
        "next_action": normalized_actions[:, 1],
        "reward": batch["reward"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "done": batch["done"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "episode_success": batch["episode_success"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
    }


def compute_sarsa_loss(
    encoded: Mapping[str, torch.Tensor],
    online_critic: ActionValueCritic,
    target_critic: ActionValueCritic,
    gamma: float,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算严格单步SARSA TD目标和SmoothL1损失。"""
    with _autocast_context(device, use_amp):
        current_q = online_critic(
            encoded["current_visual"],
            encoded["current_state"],
            encoded["action"],
        )
        with torch.no_grad():
            next_q = target_critic(
                encoded["next_visual"],
                encoded["next_state"],
                encoded["next_action"],
            )
            td_target = (
                encoded["reward"]
                + float(gamma) * (1.0 - encoded["done"]) * next_q
            )
        loss = nn.functional.smooth_l1_loss(current_q, td_target)
    return loss, {
        "current_q": current_q.detach(),
        "next_q": next_q.detach(),
        "td_target": td_target.detach(),
    }


def _new_metric_accumulator() -> dict[str, float]:
    return {
        "loss_sum": 0.0,
        "q_sum": 0.0,
        "target_sum": 0.0,
        "success_q_sum": 0.0,
        "success_count": 0.0,
        "failure_q_sum": 0.0,
        "failure_count": 0.0,
        "sample_count": 0.0,
    }


def _accumulate_metrics(
    accumulator: dict[str, float],
    loss: torch.Tensor,
    diagnostics: Mapping[str, torch.Tensor],
    episode_success: torch.Tensor,
) -> None:
    batch_size = int(diagnostics["current_q"].numel())
    q_values = diagnostics["current_q"].float()
    targets = diagnostics["td_target"].float()
    success_mask = episode_success > 0.5
    failure_mask = ~success_mask
    accumulator["loss_sum"] += float(loss.detach()) * batch_size
    accumulator["q_sum"] += float(q_values.sum())
    accumulator["target_sum"] += float(targets.sum())
    accumulator["sample_count"] += batch_size
    if success_mask.any():
        accumulator["success_q_sum"] += float(q_values[success_mask].sum())
        accumulator["success_count"] += int(success_mask.sum())
    if failure_mask.any():
        accumulator["failure_q_sum"] += float(q_values[failure_mask].sum())
        accumulator["failure_count"] += int(failure_mask.sum())


def _finalize_metrics(accumulator: Mapping[str, float]) -> dict[str, float]:
    count = max(1.0, accumulator["sample_count"])
    return {
        "loss": accumulator["loss_sum"] / count,
        "q_mean": accumulator["q_sum"] / count,
        "td_target_mean": accumulator["target_sum"] / count,
        "successful_episode_q_mean": (
            accumulator["success_q_sum"]
            / max(1.0, accumulator["success_count"])
        ),
        "failed_episode_q_mean": (
            accumulator["failure_q_sum"]
            / max(1.0, accumulator["failure_count"])
        ),
        "samples": int(accumulator["sample_count"]),
    }


def run_training_epoch(
    loader: DataLoader,
    policy,
    camera_names: list[str],
    online_critic: ActionValueCritic,
    target_critic: ActionValueCritic,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    global_step: int,
) -> tuple[dict[str, float], int]:
    """执行一个离线单步SARSA训练epoch。"""
    online_critic.train()
    target_critic.eval()
    policy.eval()
    metrics = _new_metric_accumulator()
    for batch_index, batch in enumerate(loader):
        encoded = encode_transition_batch(
            batch,
            policy,
            camera_names,
            device,
            bool(cfg.use_amp),
        )
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = compute_sarsa_loss(
            encoded,
            online_critic,
            target_critic,
            float(cfg.training.gamma),
            device,
            bool(cfg.use_amp),
        )
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            online_critic.parameters(),
            float(cfg.training.grad_clip_norm),
        )
        optimizer.step()
        polyak_update(
            online_critic,
            target_critic,
            float(cfg.training.target_update_tau),
        )
        global_step += 1
        _accumulate_metrics(
            metrics,
            loss,
            diagnostics,
            encoded["episode_success"],
        )
        if global_step % int(cfg.training.log_freq_steps) == 0:
            logging.info(
                "step=%d batch=%d loss=%.6f q=%.4f target=%.4f grad=%.4f",
                global_step,
                batch_index,
                float(loss.detach()),
                float(diagnostics["current_q"].float().mean()),
                float(diagnostics["td_target"].float().mean()),
                float(grad_norm),
            )
    return _finalize_metrics(metrics), global_step


@torch.no_grad()
def run_validation(
    loader: DataLoader | None,
    policy,
    camera_names: list[str],
    online_critic: ActionValueCritic,
    target_critic: ActionValueCritic,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float] | None:
    """计算episode隔离验证集上的Bellman残差和Q值诊断。"""
    if loader is None:
        return None
    online_critic.eval()
    target_critic.eval()
    policy.eval()
    metrics = _new_metric_accumulator()
    for batch in loader:
        encoded = encode_transition_batch(
            batch,
            policy,
            camera_names,
            device,
            bool(cfg.use_amp),
        )
        loss, diagnostics = compute_sarsa_loss(
            encoded,
            online_critic,
            target_critic,
            float(cfg.training.gamma),
            device,
            bool(cfg.use_amp),
        )
        _accumulate_metrics(
            metrics,
            loss,
            diagnostics,
            encoded["episode_success"],
        )
    return _finalize_metrics(metrics)


def move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_critic_checkpoint(
    raw_path,
    online_critic: ActionValueCritic,
    target_critic: ActionValueCritic,
    optimizer: torch.optim.Optimizer,
    critic_config: ActionValueCriticConfig,
    device: torch.device,
) -> tuple[int, int, float]:
    """可选恢复Critic、Target、优化器、epoch和最优验证损失。"""
    if raw_path is None or str(raw_path).strip().lower() in {
        "",
        "none",
        "null",
    }:
        return 0, 0, float("inf")
    path = resolve_project_path(raw_path)
    if path.is_dir():
        path = path / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"找不到Critic checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stored_config = checkpoint.get("critic_config", {})
    for name in (
        "visual_feature_dim",
        "joint_history_dim",
        "action_dim",
    ):
        if int(stored_config.get(name, -1)) != int(getattr(critic_config, name)):
            raise ValueError(
                f"Critic恢复维度{name}不一致: "
                f"checkpoint={stored_config.get(name)}, "
                f"current={getattr(critic_config, name)}"
            )
    stored_activation = str(
        stored_config.get("output_activation", "identity")
    )
    if stored_activation != critic_config.output_activation:
        raise ValueError(
            "Critic恢复输出语义不一致: "
            f"checkpoint={stored_activation}, "
            f"current={critic_config.output_activation}。"
            "旧版identity快照不能用于相对价值比值，请从头训练sigmoid Critic。"
        )
    online_critic.load_state_dict(checkpoint["online_critic"])
    target_critic.load_state_dict(
        checkpoint.get("target_critic", checkpoint["online_critic"])
    )
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state(optimizer, device)
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_validation_loss", float("inf"))),
    )


def save_critic_checkpoint(
    output_dir: Path,
    online_critic: ActionValueCritic,
    target_critic: ActionValueCritic,
    optimizer: torch.optim.Optimizer,
    critic_config: ActionValueCriticConfig,
    *,
    epoch: int,
    global_step: int,
    checkpoint_loss: float,
    best_validation_loss: float,
    pretrained_model_dir: Path,
    dataset_dir: Path,
    camera_names: list[str],
    training_execution_steps: list[int],
    save_epoch_copy: bool,
    is_best: bool,
) -> None:
    """保存训练恢复状态以及Critic和原策略之间的结构契约。"""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "online_critic": online_critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "critic_config": asdict(critic_config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_loss": float(checkpoint_loss),
        "best_validation_loss": float(best_validation_loss),
        "pretrained_model_path": str(pretrained_model_dir),
        "dataset_dir": str(dataset_dir),
        "camera_names": list(camera_names),
        "training_execution_steps": list(training_execution_steps),
        "history_steps": 2,
        "target_type": "one_step_offline_sarsa",
    }
    torch.save(payload, checkpoint_dir / "latest.pt")
    if save_epoch_copy:
        epoch_filename = (
            f"epoch_{epoch:06d}_loss={float(checkpoint_loss):.6f}.pt"
        )
        torch.save(payload, checkpoint_dir / epoch_filename)
    if is_best:
        torch.save(payload, checkpoint_dir / "best.pt")


def append_jsonl(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def maybe_init_wandb(cfg: DictConfig, output_dir: Path):
    if not bool(cfg.wandb.enable):
        return None
    import wandb

    return wandb.init(
        project=str(cfg.wandb.project),
        name=hydra.core.hydra_config.HydraConfig.get().job.name,
        notes=str(cfg.wandb.notes),
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(
            x_disable_stats=bool(cfg.wandb.disable_system_stats),
            x_disable_machine_info=bool(cfg.wandb.disable_machine_info),
        ),
    )


def describe_episode_split(name: str, episodes) -> None:
    """输出各执行步长的自然episode数量和成功率。"""
    for execution_steps in sorted({item.execution_steps for item in episodes}):
        group = [
            item for item in episodes if item.execution_steps == execution_steps
        ]
        logging.info(
            "%s: execution_steps=%d, episodes=%d, transitions=%d, success_rate=%.3f",
            name,
            execution_steps,
            len(group),
            sum(item.transition_count for item in group),
            np.mean([item.success for item in group]),
        )


def train_action_value_critic(
    cfg: DictConfig,
    output_dir: str | Path,
) -> None:
    """加载冻结快照和原始轨迹，训练独立标量动作价值网络。"""
    validate_training_config(cfg)
    seed_everything(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    policy, _, _, pretrained_model_dir = load_frozen_policy(
        cfg.pretrained_ckpt_path,
        device,
    )
    dataset_dir = resolve_project_path(cfg.dataset_dir)
    requested_execution_steps = requested_training_execution_steps(cfg)
    episodes, metadata = load_action_value_episodes(
        dataset_dir,
        execution_steps=requested_execution_steps,
    )
    training_execution_steps = sorted(
        {episode.execution_steps for episode in episodes}
    )
    logging.info(
        "Critic训练选择execution_steps=%s（null表示加载全部可用步长）",
        training_execution_steps,
    )
    camera_names = validate_dataset_against_policy(metadata, policy)
    stored_model_path = metadata.get("pretrained_model_path")
    if stored_model_path and Path(stored_model_path).expanduser() != pretrained_model_dir:
        logging.warning(
            "数据集记录的快照路径与当前路径不同；已通过策略结构校验，"
            "请确认二者权重确实相同: dataset=%s, current=%s",
            stored_model_path,
            pretrained_model_dir,
        )

    train_episodes, validation_episodes = split_episodes_by_execution_steps(
        episodes,
        float(cfg.training.validation_ratio),
        int(cfg.seed),
    )
    describe_episode_split("train", train_episodes)
    if validation_episodes:
        describe_episode_split("validation", validation_episodes)
    train_dataset = ActionValueTransitionDataset(train_episodes)
    validation_dataset = (
        ActionValueTransitionDataset(validation_episodes)
        if validation_episodes
        else None
    )
    train_loader = make_dataloader(
        train_dataset,
        cfg,
        shuffle=True,
        seed=int(cfg.seed),
    )
    validation_loader = (
        make_dataloader(
            validation_dataset,
            cfg,
            shuffle=False,
            seed=int(cfg.seed) + 1,
        )
        if validation_dataset is not None
        else None
    )

    probe_batch = next(iter(train_loader))
    probe_encoded = encode_transition_batch(
        probe_batch,
        policy,
        camera_names,
        device,
        bool(cfg.use_amp),
    )
    critic_config = ActionValueCriticConfig(
        visual_feature_dim=int(probe_encoded["current_visual"].shape[-1]),
        joint_history_dim=int(probe_encoded["current_state"].shape[-1]),
        action_dim=int(probe_encoded["action"].shape[-1]),
        visual_embed_dim=int(cfg.critic.visual_embed_dim),
        state_embed_dim=int(cfg.critic.state_embed_dim),
        action_embed_dim=int(cfg.critic.action_embed_dim),
        hidden_dims=tuple(int(value) for value in cfg.critic.hidden_dims),
        output_activation=str(cfg.critic.output_activation),
        initial_q=float(cfg.critic.initial_q),
    )
    online_critic = ActionValueCritic(critic_config).to(device)
    target_critic = make_target_critic(online_critic).to(device)
    optimizer = torch.optim.AdamW(
        online_critic.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    start_epoch, global_step, best_validation_loss = restore_critic_checkpoint(
        cfg.resume_critic_path,
        online_critic,
        target_critic,
        optimizer,
        critic_config,
        device,
    )
    wandb_run = maybe_init_wandb(cfg, output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    logging.info(
        "动作价值训练开始: train_transitions=%d, validation_transitions=%d, "
        "visual_dim=%d, joint_history_dim=%d, action_dim=%d, device=%s",
        len(train_dataset),
        len(validation_dataset) if validation_dataset is not None else 0,
        critic_config.visual_feature_dim,
        critic_config.joint_history_dim,
        critic_config.action_dim,
        device,
    )

    try:
        for epoch in range(start_epoch, int(cfg.training.epochs)):
            train_metrics, global_step = run_training_epoch(
                train_loader,
                policy,
                camera_names,
                online_critic,
                target_critic,
                optimizer,
                cfg,
                device,
                global_step,
            )
            validation_metrics = run_validation(
                validation_loader,
                policy,
                camera_names,
                online_critic,
                target_critic,
                cfg,
                device,
            )
            selection_loss = (
                validation_metrics["loss"]
                if validation_metrics is not None
                else train_metrics["loss"]
            )
            is_best = selection_loss < best_validation_loss
            if is_best:
                best_validation_loss = selection_loss
            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "best_validation_loss": best_validation_loss,
                **{
                    f"train/{key}": value
                    for key, value in train_metrics.items()
                },
            }
            if validation_metrics is not None:
                payload.update(
                    {
                        f"validation/{key}": value
                        for key, value in validation_metrics.items()
                    }
                )
            append_jsonl(metrics_path, payload)
            if wandb_run is not None:
                wandb_run.log(payload, step=global_step)
            logging.info(
                "epoch=%d train_loss=%.6f validation_loss=%s "
                "train_q=%.4f best=%.6f",
                epoch,
                train_metrics["loss"],
                (
                    f"{validation_metrics['loss']:.6f}"
                    if validation_metrics is not None
                    else "none"
                ),
                train_metrics["q_mean"],
                best_validation_loss,
            )
            save_critic_checkpoint(
                output_dir,
                online_critic,
                target_critic,
                optimizer,
                critic_config,
                epoch=epoch,
                global_step=global_step,
                checkpoint_loss=selection_loss,
                best_validation_loss=best_validation_loss,
                pretrained_model_dir=pretrained_model_dir,
                dataset_dir=dataset_dir,
                camera_names=camera_names,
                training_execution_steps=training_execution_steps,
                save_epoch_copy=(
                    (epoch + 1) % int(cfg.training.save_freq_epochs) == 0
                    or epoch + 1 == int(cfg.training.epochs)
                ),
                is_best=is_best,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


@hydra.main(
    version_base="1.2",
    config_path="../../configs/adaptive_replanning",
    config_name="action_value_train",
)
def train_cli(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    train_action_value_critic(
        cfg,
        hydra.core.hydra_config.HydraConfig.get().run.dir,
    )


def has_cli_override(args: list[str], key: str) -> bool:
    """判断用户是否已经在命令行显式设置某个Hydra参数。"""
    for argument in args:
        argument_key = argument.split("=", maxsplit=1)[0].lstrip("+")
        if argument_key == key:
            return True
    return False


if __name__ == "__main__":
    # ==========================================
    # 常用训练参数：直接修改这里即可
    # ==========================================
    # 这些值相当于Hydra命令行覆盖项；若启动命令中显式传入同名参数，
    # 则以命令行值为准，不会被下面的本地默认值覆盖。
    default_args = [
        # 冻结的预训练扩散策略快照，用于实时提取两帧视觉特征。
        "pretrained_ckpt_path='outputs/2_pretrain/train/2026-07-16/20-59-53_InsertCylinder-3Arms-v0_pre_zed_dual_head_diffusion/checkpoints/162000_loss=0.0051_sr=87.0_ar=751.50'",
        # collect_action_value_dataset.py生成的原始轨迹数据集。
        "dataset_dir=outputs/6_action_value_datasets/insert_cylinder",
        # 全新训练设为null；断点续训时改为latest.pt或checkpoints目录。
        "resume_critic_path=null",
        # 训练设备与自动混合精度。
        "device=cuda:0",
        "use_amp=true",
        # 严格单步SARSA的核心训练超参数。
        # null使用全部数据；可改为[4]或[4,8]选择指定执行步长。
        "training.execution_steps=null",
        "training.gamma=0.99",
        "training.learning_rate=1.0e-4",
        "training.batch_size=32",
        "training.epochs=100",
        "training.num_workers=4",
        # Critic使用[0,1]有界Q，供在线评估进行相对价值判断。
        "critic.output_activation=sigmoid",
        "critic.initial_q=0.05",
        # 是否记录到Weights & Biases。
        "wandb.enable=false",
    ]

    original_cli_args = sys.argv[1:]
    for argument in default_args:
        key = argument.split("=", maxsplit=1)[0].lstrip("+")
        if not has_cli_override(original_cli_args, key):
            sys.argv.append(argument)

    train_cli()
