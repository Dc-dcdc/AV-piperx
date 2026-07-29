#!/usr/bin/env python

"""Train only the supervised none/Arm-to-View candidate Router."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import hydra
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from lerobot.common.policies.diffusion.configuration_routed_post_diffusion_output_corrector import (
    RoutedPostDiffusionOutputCorrectorConfig,
)
from lerobot.common.policies.diffusion.modeling_post_diffusion_output_corrector import (
    PostDiffusionOutputCorrectorPolicy,
)
from lerobot.common.policies.diffusion.modeling_routed_post_diffusion_output_corrector import (
    RoutedPostDiffusionOutputCorrectorPolicy,
)
from lerobot.common.policies.factory import _policy_cfg_from_hydra_cfg
from lerobot.common.utils.utils import get_safe_torch_device, init_logging
from train.s2_incremental.router.collect_none_av_router_dataset import (
    _load_source_policy,
    _resolve_model_dir,
)
from train.s2_incremental.router.counterfactual_dataset import (
    CounterfactualRouterDataset,
    split_indices_by_episode_seed,
)


def migrate_corrector_into_router(
    source: PostDiffusionOutputCorrectorPolicy,
    target: RoutedPostDiffusionOutputCorrectorPolicy,
) -> dict[str, Any]:
    """Copy every frozen tensor; only Router tensors may be missing."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    unexpected = sorted(set(source_state).difference(target_state))
    if unexpected:
        raise RuntimeError(
            "Router目标策略无法识别源修正器张量:\n"
            + "\n".join(f"  - {name}" for name in unexpected)
        )
    mismatches = [
        (
            name,
            tuple(tensor.shape),
            tuple(target_state[name].shape),
        )
        for name, tensor in source_state.items()
        if tensor.shape != target_state[name].shape
    ]
    if mismatches:
        raise RuntimeError(f"修正器→Router形状不兼容: {mismatches}")
    incompatible = target.load_state_dict(source_state, strict=False)
    expected_missing = {
        name
        for name in target_state
        if name.startswith("diffusion.output_router.")
    }
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "不安全的修正器→Router迁移: "
            f"missing={incompatible.missing_keys}, "
            f"expected={sorted(expected_missing)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    copied = target.state_dict()
    unequal = [
        name
        for name, tensor in source_state.items()
        if not torch.equal(tensor, copied[name])
    ]
    if unequal:
        raise RuntimeError(f"冻结源权重迁移后发生变化: {unequal}")
    return {
        "frozen_tensor_count": len(source_state),
        "router_tensor_count": len(expected_missing),
    }


def validate_source_compatibility(
    source_config,
    target_config: RoutedPostDiffusionOutputCorrectorConfig,
) -> None:
    """Protect all behavior-defining dimensions and corrector weights."""
    fields = (
        "n_obs_steps",
        "horizon",
        "n_action_steps",
        "input_shapes",
        "output_shapes",
        "input_normalization_modes",
        "output_normalization_modes",
        "arm_action_dim",
        "view_action_dim",
        "vision_backbone",
        "pretrained_backbone_weights",
        "resize_shape",
        "crop_shape",
        "use_group_norm",
        "spatial_softmax_num_keypoints",
        "down_dims",
        "kernel_size",
        "n_groups",
        "diffusion_step_embed_dim",
        "use_film_scale_modulation",
        "noise_scheduler_type",
        "num_train_timesteps",
        "num_inference_steps",
        "beta_schedule",
        "beta_start",
        "beta_end",
        "prediction_type",
        "clip_sample",
        "clip_sample_range",
        "output_corrector_type",
        "output_corrector_d_model",
        "output_corrector_num_heads",
        "output_corrector_dropout",
        "output_corrector_residual_limit",
        "output_corrector_clamp_actions",
    )
    mismatches = []
    for field in fields:
        source_value = getattr(source_config, field)
        target_value = getattr(target_config, field)
        if isinstance(source_value, (list, tuple)):
            source_value = tuple(source_value)
        if isinstance(target_value, (list, tuple)):
            target_value = tuple(target_value)
        if source_value != target_value:
            mismatches.append(
                f"{field}: source={source_value!r}, target={target_value!r}"
            )
    if mismatches:
        raise ValueError(
            "Router配置与源输出修正器不兼容:\n"
            + "\n".join(f"  - {item}" for item in mismatches)
        )


def configure_router_only(policy) -> dict[str, Any]:
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.diffusion.output_router.parameters():
        parameter.requires_grad_(True)
    trainable = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    invalid = [
        name
        for name, _ in trainable
        if not name.startswith("diffusion.output_router.")
    ]
    if invalid or not trainable:
        raise RuntimeError(
            f"Router训练参数范围异常: invalid={invalid}, count={len(trainable)}"
        )
    return {
        "names": [name for name, _ in trainable],
        "tensor_count": len(trainable),
        "parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in policy.parameters()
            if not parameter.requires_grad
        ),
    }


def _make_datasets(cfg: DictConfig):
    raw_manifests = OmegaConf.to_container(
        cfg.router_cache.manifests,
        resolve=True,
    )
    manifests = [
        Path(str(value)).expanduser().resolve()
        for value in raw_manifests
        if str(value).strip()
    ]
    if not manifests:
        raise ValueError(
            "router_cache.manifests为空；请先运行"
            "collect_none_av_router_dataset.py。"
        )
    train_sets = []
    validation_sets = []
    for manifest in manifests:
        train_indices, validation_indices = split_indices_by_episode_seed(
            manifest,
            validation_fraction=float(
                cfg.router_training.validation_fraction
            ),
            seed=int(cfg.seed),
        )
        common = {
            "manifest_path": manifest,
            "memory_limit_gb": float(cfg.router_cache.memory_limit_gb),
        }
        train_sets.append(
            CounterfactualRouterDataset(
                **common,
                indices=train_indices,
            )
        )
        validation_sets.append(
            CounterfactualRouterDataset(
                **common,
                indices=validation_indices,
            )
        )
    train_dataset = (
        train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    )
    validation_dataset = (
        validation_sets[0]
        if len(validation_sets) == 1
        else ConcatDataset(validation_sets)
    )
    return train_dataset, validation_dataset, manifests


def _train_label_counts(dataset) -> tuple[int, int]:
    labels = []
    datasets = (
        dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    )
    for child in datasets:
        labels.extend(
            child.arrays["router_label"][child.indices].astype(np.int64).tolist()
        )
    labels_array = np.asarray(labels)
    return int((labels_array == 0).sum()), int((labels_array == 1).sum())


def _move_batch(batch, device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def validate(policy, loader, device) -> dict[str, Any]:
    policy.diffusion.output_router.eval()
    probabilities = []
    labels = []
    weights = []
    losses = []
    for batch in loader:
        batch = _move_batch(batch, device)
        result = policy(batch)
        losses.append(float(result["loss"]))
        diagnostics = policy.diffusion.score_candidates(
            batch["global_condition"],
            batch["none_trajectory"],
            batch["arm_to_view_trajectory"],
        )
        probabilities.append(
            diagnostics["router_probability"].detach().cpu()
        )
        labels.append(batch["router_label"].detach().cpu())
        weights.append(batch["sample_weight"].detach().cpu())
    probability = torch.cat(probabilities)
    label = torch.cat(labels)
    weight = torch.cat(weights)
    return {
        "loss": float(np.mean(losses)),
        "probability": probability,
        "label": label,
        "weight": weight,
    }


def calibrate_threshold(
    validation: dict[str, Any],
    *,
    false_positive_cost: float,
) -> dict[str, float]:
    probability = validation["probability"]
    target = validation["label"] >= 0.5
    weight = validation["weight"]
    best = None
    for threshold in torch.linspace(0.05, 0.95, 91):
        prediction = probability >= threshold
        true_positive = weight[prediction & target].sum()
        true_negative = weight[(~prediction) & (~target)].sum()
        false_positive = weight[prediction & (~target)].sum()
        score = (
            true_positive
            + true_negative
            - (float(false_positive_cost) - 1.0) * false_positive
        )
        candidate = (float(score), float(threshold))
        if best is None or candidate > best:
            best = candidate
    threshold = best[1]
    prediction = probability >= threshold
    true_positive = int((prediction & target).sum())
    false_positive = int((prediction & (~target)).sum())
    true_negative = int(((~prediction) & (~target)).sum())
    false_negative = int(((~prediction) & target).sum())
    accuracy = float((prediction == target).float().mean())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "score": best[0],
    }


def _is_no_decay(name: str, parameter: nn.Parameter) -> bool:
    return (
        parameter.ndim <= 1
        or name.endswith(".bias")
        or "temporal_position" in name
    )


def _make_optimizer(cfg, policy):
    named = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    decay = [
        parameter
        for name, parameter in named
        if not _is_no_decay(name, parameter)
    ]
    no_decay = [
        parameter
        for name, parameter in named
        if _is_no_decay(name, parameter)
    ]
    groups = [
        {
            "params": decay,
            "weight_decay": float(cfg.router_training.weight_decay),
        },
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=float(cfg.router_training.lr),
        betas=tuple(cfg.training.adam_betas),
        eps=float(cfg.training.adam_eps),
    )


def _save_checkpoint(
    *,
    out_dir: Path,
    step: int,
    policy,
    optimizer,
    scheduler,
    cfg,
    validation: dict[str, float],
) -> Path:
    identifier = (
        f"{step:06d}_val_loss={validation['loss']:.4f}"
        f"_acc={validation['accuracy'] * 100.0:.1f}"
    )
    checkpoint = out_dir / "checkpoints" / identifier
    model_dir = checkpoint / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(model_dir)
    OmegaConf.update(
        cfg,
        "policy.router_threshold",
        float(policy.diffusion.router_threshold),
        merge=False,
        force_add=True,
    )
    OmegaConf.save(cfg, model_dir / "config.yaml")
    torch.save(
        {
            "step": int(step),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint / "training_state.pth",
    )
    with open(
        checkpoint / "router_validation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(validation, file, indent=2, ensure_ascii=False)
    return checkpoint


def train(cfg: DictConfig, out_dir: str | Path) -> Path:
    device = get_safe_torch_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    train_dataset, validation_dataset, manifests = _make_datasets(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.router_training.batch_size),
        shuffle=True,
        num_workers=int(cfg.router_training.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=(
            int(cfg.router_training.num_workers) > 0
            and bool(cfg.router_training.persistent_workers)
        ),
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(cfg.router_training.batch_size),
        shuffle=False,
        num_workers=int(cfg.router_training.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=(
            int(cfg.router_training.num_workers) > 0
            and bool(cfg.router_training.persistent_workers)
        ),
        drop_last=False,
    )

    resume_path = str(cfg.resume_path).strip() if cfg.resume_path else ""
    start_step = 0
    if bool(cfg.resume):
        model_dir = _resolve_model_dir(resume_path)
        policy = RoutedPostDiffusionOutputCorrectorPolicy.from_pretrained(
            model_dir,
            strict=True,
        )
        policy.to(device)
        migration = {"mode": "resume"}
    else:
        source, _, _ = _load_source_policy(cfg.init_policy_path, device)
        target_config = _policy_cfg_from_hydra_cfg(
            RoutedPostDiffusionOutputCorrectorConfig,
            cfg,
        )
        validate_source_compatibility(source.config, target_config)
        policy = RoutedPostDiffusionOutputCorrectorPolicy(target_config)
        migration = migrate_corrector_into_router(source, policy)
        policy.to(device)
        del source
    scope = configure_router_only(policy)
    policy.eval()
    policy.diffusion.output_router.train(True)

    optimizer = _make_optimizer(cfg, policy)
    total_steps = int(cfg.router_training.steps)
    scheduler = get_scheduler(
        str(cfg.training.lr_scheduler),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.router_training.warmup_steps),
        num_training_steps=total_steps,
    )
    if bool(cfg.resume):
        state_path = (
            Path(resume_path)
            if Path(resume_path).name != "pretrained_model"
            else Path(resume_path).parent
        ) / "training_state.pth"
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"]) + 1

    none_count, positive_count = _train_label_counts(train_dataset)
    positive_weight = torch.tensor(
        none_count / max(1, positive_count),
        dtype=torch.float32,
        device=device,
    )
    logging.info(
        "Router数据: train=%d, validation=%d, none=%d, A→V=%d, "
        "pos_weight=%.3f",
        len(train_dataset),
        len(validation_dataset),
        none_count,
        positive_count,
        float(positive_weight),
    )
    logging.info(
        "Router参数: trainable=%d tensors/%d params, frozen=%d params",
        scope["tensor_count"],
        scope["parameter_count"],
        scope["frozen_parameter_count"],
    )
    logging.info("源迁移: %s; manifests=%s", migration, manifests)

    out_dir = Path(out_dir)
    (out_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / ".hydra" / "config.yaml")
    wandb_run = None
    if bool(OmegaConf.select(cfg, "wandb.enable", default=False)):
        import wandb

        wandb_run = wandb.init(
            project=str(cfg.wandb.project),
            entity=OmegaConf.select(cfg, "wandb.entity", default=None),
            name=out_dir.name,
            tags=list(OmegaConf.select(cfg, "wandb.tags", default=[])),
            dir=str(out_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            job_type="train",
        )
    iterator = iter(train_loader)
    use_amp = bool(cfg.use_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16
    best_key = (-math.inf, -math.inf)
    best_checkpoint = None

    for step in range(start_step, total_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else nullcontext()
        )
        with amp_context:
            result = policy.diffusion.compute_router_loss(
                batch["global_condition"],
                batch["none_trajectory"],
                batch["arm_to_view_trajectory"],
                batch["router_label"],
                batch["sample_weight"],
                positive_weight=positive_weight,
            )
        result["loss"].backward()
        nn.utils.clip_grad_norm_(
            policy.diffusion.output_router.parameters(),
            float(cfg.training.grad_clip_norm),
        )
        optimizer.step()
        scheduler.step()

        if step % int(cfg.router_training.log_freq) == 0:
            logging.info(
                "step=%d loss=%.5f acc=%.3f p=%.3f lr=%.3e",
                step,
                float(result["loss"]),
                float(result["router_accuracy"]),
                float(result["router_probability_mean"]),
                optimizer.param_groups[0]["lr"],
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": float(result["loss"]),
                        "train/router_accuracy": float(
                            result["router_accuracy"]
                        ),
                        "train/router_probability_mean": float(
                            result["router_probability_mean"]
                        ),
                        "train/q_none_mean": float(result["q_none_mean"]),
                        "train/q_arm_to_view_mean": float(
                            result["q_arm_to_view_mean"]
                        ),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=step,
                )

        due = (
            (step + 1) % int(cfg.router_training.eval_freq) == 0
            or step == total_steps - 1
        )
        if not due:
            continue
        validation_raw = validate(policy, validation_loader, device)
        calibrated = calibrate_threshold(
            validation_raw,
            false_positive_cost=float(
                cfg.router_training.false_positive_cost
            ),
        )
        calibrated["loss"] = float(validation_raw["loss"])
        calibrated["step"] = int(step)
        policy.diffusion.set_router_threshold(calibrated["threshold"])
        logging.info("Router validation: %s", calibrated)
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"validation/{name}": value
                    for name, value in calibrated.items()
                    if name != "step"
                },
                step=step,
            )
        candidate_key = (
            float(calibrated["score"]),
            -float(calibrated["loss"]),
        )
        if candidate_key > best_key:
            best_key = candidate_key
            best_checkpoint = _save_checkpoint(
                out_dir=out_dir,
                step=step,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                validation=calibrated,
            )
            logging.info("保存新的最佳Router checkpoint: %s", best_checkpoint)
        policy.eval()
        policy.diffusion.output_router.train(True)

    if best_checkpoint is None:
        raise RuntimeError("Router训练结束但没有保存checkpoint。")
    if wandb_run is not None:
        wandb_run.finish()
    return best_checkpoint


@hydra.main(
    version_base="1.2",
    config_name="pre_default",
    config_path="../../../configs/pretrain",
)
def train_cli(cfg: DictConfig) -> None:
    init_logging()
    runtime = hydra.core.hydra_config.HydraConfig.get()
    checkpoint = train(cfg, runtime.run.dir)
    print(f"Best Router checkpoint: {checkpoint}")


if __name__ == "__main__":
    default_args = [
        # 冻结的后置A→V修正器checkpoint；只用于首次Router初始化。
        "init_policy_path='outputs/2_pretrain/post_diffusion_output_corrector/InsertCylinder-3Arms-v0/2026-07-28/00-20-36_InsertCylinder-3Arms-v0_pre_zed_post_diffusion_output_corrector/checkpoints/099999_loss=0.0016_sr=86.0_ar=738.49'",
        "env=sim_insert_cylinder_3arms",
        "policy=pre_zed_none_av_router",
        # Router训练只读反事实manifest，不会读取本字段中的示范图像。
        "dataset_local_dir=outputs/5_hf_datasets/quest_teleop_insert_cylinder_3arms_rgb_joint",
        "dataset_repo_id=Dc-dc/quest_teleop_insert_cylinder_3arms_rgb_joint",
        "resume=false",
        "resume_path=null",
        "wandb.enable=false",
    ]
    for argument in default_args:
        key = argument.split("=", 1)[0]
        if not any(
            item.split("=", 1)[0] == key for item in sys.argv[1:]
        ):
            sys.argv.append(argument)
    train_cli()
