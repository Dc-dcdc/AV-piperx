#!/usr/bin/env python
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 强行指向国内镜像站
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import gymnasium as gym
import env
import logging
import time
from contextlib import nullcontext
from pprint import pformat

from train.pretrain.eval_train import evaluate_and_checkpoint_if_needed, TopKCheckpointManager, make_eval_env

import hydra
import datasets
import numpy as np
import torch
import torchvision.transforms as v2
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler

# ==========================================
# 🌟 采用官方最新极简 API，抛弃 factory.py
# ==========================================
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.transforms import get_image_transforms
from lerobot.common.datasets.utils import hf_transform_to_torch, unflatten_dict
from lerobot.common.datasets.video_utils import VideoFrame  # noqa: F401  注册本地 parquet 中的 VideoFrame 字段
# 复用 LeRobot 的其他核心组件
from lerobot.common.logger import Logger
from lerobot.common.policies.factory import make_policy # 用于获取训练策略模型
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from train.pretrain.optimizer_utils import partition_optimizer_parameters
from train.pretrain.scid_transform import initialize_scid_transform_from_dataset
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
    set_global_seed,
)


PRETRAIN_WANDB_PARAMETER_TAGS = (
    ("device", "device"),                                  # 训练设备
    ("batch", "training.batch_size"),                     # 训练批大小
    ("hil_ratio", "training.intervention_ratio"),         # 人类干预样本占比
    ("lr", "training.lr"),                                # 主网络学习率
    ("backbone_lr", "training.lr_backbone"),              # 视觉底座学习率
    ("steps", "training.offline_steps"),                  # 预训练总步数
    ("lr_scheduler", "training.lr_scheduler"),            # 学习率调度器
    ("lr_warmup", "training.lr_warmup_steps"),            # 学习率预热步数
    ("weight_decay", "training.weight_decay"),            # 权重衰减
    ("grad_clip", "training.grad_clip_norm"),             # 梯度裁剪阈值
    ("amp", "use_amp"),                                   # 是否使用混合精度
    ("image_aug", "training.image_transforms.enable"),    # 是否启用图像增强
    ("obs_steps", "policy.n_obs_steps"),                  # 历史观测步数
    ("horizon", "policy.horizon"),                        # 动作预测长度
    ("act_steps", "policy.n_action_steps"),               # 动作块执行长度
    ("down_dims", "policy.down_dims"),                    # U-Net通道结构
    ("n_groups", "policy.n_groups"),                      # GroupNorm组数
    ("noise_scheduler", "policy.noise_scheduler_type"),   # 扩散噪声调度器
    ("train_noise_steps", "policy.num_train_timesteps"),  # 训练加噪步数
    ("infer_steps", "policy.num_inference_steps"),        # 推理去噪步数
    ("prediction", "policy.prediction_type"),             # 扩散预测目标
    ("ema", "policy.use_ema"),                            # 是否使用EMA
    ("ema_decay", "policy.ema_decay"),                    # EMA衰减系数
    ("arm_dim", "policy.arm_action_dim"),                 # 双臂动作维度
    ("view_dim", "policy.view_action_dim"),               # 视角动作维度
    ("view_weight", "policy.view_loss_weight"),           # 视角损失权重
    ("coupling", "policy.coupling_mode"),                 # full、RBAC或balanced lookahead耦合
    ("coupling_block", "policy.coupling_block_type"),    # scalar gate或role adaLN-Zero
    ("coupling_pos", "policy.coupling_use_temporal_pos_emb"),
    ("coupling_ffn", "policy.coupling_use_ffn"),
    ("coupling_ffn_ratio", "policy.coupling_ffn_ratio"),
    ("v2a_scale", "policy.view_to_arm_coupling_scale"),   # View上下文注入Arm的外部缩放
    ("a2v_scale", "policy.arm_to_view_coupling_scale"),   # Arm上下文注入View的外部缩放
    ("scid_ridge", "policy.scid_ridge"),                 # SCID闭式Arm->View映射的岭正则
    ("iwr", "training.iwr.enabled"),                     # 是否使用论文式 IWR
    ("iwr_i_bs", "training.iwr.intervention_batch_size"),
    ("iwr_r_bs", "training.iwr.robot_batch_size"),
    ("iwr_e_bs", "training.iwr.expert_batch_size"),
)

IWR_TEMPORAL_KEYS = {
    "teleop_action",
    "teleop_action_available",
    "is_intervention",
    "intervention_action_mask",
    "intervention_action_weight",
}


# 这些字段决定模型结构、时序语义或训练目标。HIL 微调必须与专家
# checkpoint 保持一致，避免同形状参数被加载后却采用了不同的策略语义。
BASE_POLICY_COMPATIBILITY_FIELDS = (
    "n_obs_steps",
    "horizon",
    "n_action_steps",
    "input_shapes",
    "output_shapes",
    "input_normalization_modes",
    "output_normalization_modes",
    "vision_backbone",
    "resize_shape",
    "crop_shape",
    "crop_is_random",
    "pretrained_backbone_weights",
    "use_group_norm",
    "spatial_softmax_num_keypoints",
    "down_dims",
    "kernel_size",
    "n_groups",
    "diffusion_step_embed_dim",
    "use_film_scale_modulation",
    "noise_scheduler_type",
    "num_inference_steps",
    "num_train_timesteps",
    "beta_schedule",
    "beta_start",
    "beta_end",
    "prediction_type",
    "clip_sample",
    "clip_sample_range",
)

DUAL_HEAD_COMPATIBILITY_FIELDS = (
    "arm_action_dim",
    "view_action_dim",
    "view_loss_weight",
)

COUPLED_COMPATIBILITY_FIELDS = (
    "coupling_num_heads",
    "coupling_dropout",
    "coupling_mode",
    "coupling_block_type",
    "coupling_use_temporal_pos_emb",
    "coupling_use_ffn",
    "coupling_ffn_ratio",
    "view_to_arm_coupling_scale",
    "arm_to_view_coupling_scale",
)


def _format_wandb_tag_value(value) -> str:
    """将配置值转换为简短稳定的 W&B 标签文本。"""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        return "-".join(_format_wandb_tag_value(item) for item in value)
    return str(value).replace(" ", "_")


def add_wandb_parameter_tags(logger: Logger, cfg: DictConfig) -> None:
    """把实际生效的预训练关键配置追加到当前 W&B run 标签。"""
    wandb_module = getattr(logger, "_wandb", None)
    wandb_run = getattr(wandb_module, "run", None)
    if wandb_run is None:
        return

    configured_tags = OmegaConf.select(cfg, "wandb.tags", default=[])
    if configured_tags is None:
        configured_tags = []
    elif isinstance(configured_tags, str):
        configured_tags = [configured_tags]
    else:
        configured_tags = list(configured_tags)

    parameter_tags = []
    for tag_name, config_path in PRETRAIN_WANDB_PARAMETER_TAGS:
        value = OmegaConf.select(cfg, config_path, default=None)
        if value is not None:
            parameter_tags.append(f"{tag_name}:{_format_wandb_tag_value(value)}")

    existing_tags = list(wandb_run.tags or ())
    wandb_run.tags = tuple(
        dict.fromkeys([*existing_tags, *map(str, configured_tags), *parameter_tags])
    )


def count_trainable_parameters_by_component(policy):
    """按视觉编码器、扩散 U-Net 和其他模块统计可学习参数，参数只计一次。"""
    trainable_params = {id(param): param for param in policy.parameters() if param.requires_grad}
    visual_param_ids = set()
    diffusion_param_ids = set()

    visual_module_names = {"rgb_encoder", "image_encoder", "visual_encoders", "backbone"}
    diffusion_module_names = {
        "unet",
        "arm_unet",
        "view_unet",
        "view_to_arm_attention",
        "arm_to_view_attention",
        "arm_coupling_norm",
        "view_coupling_norm",
        "coupling_timestep_encoder",
        "role_adaln_coupling",
    }
    for module_name, module in policy.named_modules():
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name in visual_module_names:
            visual_param_ids.update(
                id(param) for param in module.parameters() if param.requires_grad
            )
        if leaf_name in diffusion_module_names:
            diffusion_param_ids.update(
                id(param) for param in module.parameters() if param.requires_grad
            )

    # 若模块存在嵌套，视觉部分优先归类，避免同一个参数重复计数。
    visual_param_ids &= trainable_params.keys()
    diffusion_param_ids &= trainable_params.keys()
    diffusion_param_ids -= visual_param_ids
    categorized_param_ids = visual_param_ids | diffusion_param_ids
    other_param_ids = trainable_params.keys() - categorized_param_ids

    def count(param_ids):
        return sum(trainable_params[param_id].numel() for param_id in param_ids)

    return {
        "vision": count(visual_param_ids),
        "diffusion": count(diffusion_param_ids),
        "other": count(other_param_ids),
        "total": count(trainable_params.keys()),
    }


def log_trainable_parameter_counts(policy):
    """输出模型各主要部分及总计的可学习参数量。"""
    counts = count_trainable_parameters_by_component(policy)
    logging.info(
        f"模型可学习参数量: "
        f"视觉部分={counts['vision']} ({format_big_number(counts['vision'])}), "
        f"扩散模型部分={counts['diffusion']} ({format_big_number(counts['diffusion'])}), "
        f"其他={counts['other']} ({format_big_number(counts['other'])}), "
        f"总计={counts['total']} ({format_big_number(counts['total'])})"
    )


def is_vision_backbone_parameter(parameter_name: str) -> bool:
    """识别不同 Diffusion Policy 实现中的视觉 backbone 参数。"""
    return parameter_name.startswith("model.backbone") or any(
        marker in parameter_name
        for marker in (
            "rgb_encoder.backbone.",
            "image_encoder.",
            "visual_encoders.",
        )
    )



def make_optimizer_and_scheduler(cfg, policy):
    if cfg.policy.name == "act":
        optimizer_params_dicts = [
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": cfg.training.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_params_dicts, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        lr_scheduler = None
    elif cfg.policy.name in [
        "diffusion",
        "dual_head_diffusion",
        "scid_dual_head_diffusion",
        "coupled_dual_head_diffusion",
        "two_model_diffusion",
    ]:
        # 分离视觉 Backbone 和 U-Net 的学习率，并将所有参数纳入优化器。
        # 当前 Diffusion 模型的真实前缀是 diffusion.rgb_encoder.backbone.*。
        candidate_named_parameters = [
            (name, parameter)
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        ]
        backbone_named_parameters = [
            (name, parameter)
            for name, parameter in candidate_named_parameters
            if is_vision_backbone_parameter(name)
        ]
        if not backbone_named_parameters:
            raise RuntimeError(
                "未识别到视觉 backbone 参数，无法应用 training.lr_backbone；"
                "请检查策略参数命名。"
            )

        backbone_lr = float(cfg.training.lr_backbone)
        freeze_backbone = backbone_lr <= 0.0
        if freeze_backbone:
            for _, parameter in backbone_named_parameters:
                parameter.requires_grad_(False)
            logging.info(
                f"training.lr_backbone={backbone_lr:g}，已显式冻结 "
                f"{len(backbone_named_parameters)} 个视觉 backbone 参数张量。"
            )

        trainable_named_parameters = [
            (name, parameter)
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        ]
        main_parameters, coupling_parameters, backbone_parameters = (
            partition_optimizer_parameters(
                trainable_named_parameters,
                is_backbone_parameter=is_vision_backbone_parameter,
            )
        )
        optimizer_params_dicts = [
            {
                "name": "main",
                "params": main_parameters,
                "weight_decay": cfg.training.weight_decay,
            },
        ]
        if coupling_parameters:
            optimizer_params_dicts.append(
                {
                    "name": "coupling",
                    "params": coupling_parameters,
                    "weight_decay": 0.0,
                }
            )
        if not freeze_backbone:
            optimizer_params_dicts.append(
                {
                    "name": "backbone",
                    "params": backbone_parameters,
                    "lr": backbone_lr,
                }
            )
        logging.info(
            f"优化器参数组: main={len(main_parameters)} tensors "
            f"(lr={float(cfg.training.lr):g}, weight_decay={float(cfg.training.weight_decay):g}), "
            f"coupling={len(coupling_parameters)} tensors "
            f"(lr={float(cfg.training.lr):g}, weight_decay=0), "
            f"backbone={len(backbone_parameters)} tensors "
            f"(lr={'frozen' if freeze_backbone else f'{backbone_lr:g}'}, "
            f"weight_decay={float(cfg.training.weight_decay):g})"
        )

        #对于扩散模型（diffusion model），我们使用了Adam优化器来更新模型的参数
        optimizer = torch.optim.Adam(
            # policy.diffusion.parameters(),
            optimizer_params_dicts,
            lr=cfg.training.lr,
            betas=cfg.training.adam_betas,
            eps=cfg.training.adam_eps,
            weight_decay=cfg.training.weight_decay,
        )
        from diffusers.optimization import get_scheduler

        #使用了diffusers库中的get_scheduler函数来创建一个学习率调度器
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps, #预热步数，前num_warmup_steps步，学习率会从0线性增加到cfg.training.lr指定的初始学习率，这有助于模型在训练初期更稳定地收敛。
            num_training_steps=cfg.training.offline_steps, #总训练步数
        )


    elif cfg.policy.name == "tdmpc": #对于TDMPC模型，我们使用了Adam优化器来更新模型的参数
        optimizer = torch.optim.Adam(policy.parameters(), cfg.training.lr)
        lr_scheduler = None
    elif cfg.policy.name == "vqbet": #对于VQBeT模型，我们使用了自定义的VQBeTOptimizer来更新模型的参数
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTOptimizer, VQBeTScheduler

        optimizer = VQBeTOptimizer(policy, cfg)
        lr_scheduler = VQBeTScheduler(optimizer, cfg)
    else:
        raise NotImplementedError()

    return optimizer, lr_scheduler #返回创建好的优化器和学习率调度器，这些会在训练过程中被用来更新模型的参数和调整学习率。





def update_policy(
    policy,
    batch,
    optimizer,
    grad_clip_norm,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
):
    """进行一次训练更新，计算损失，反向传播，更新模型参数，并返回一个包含训练信息的字典."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train() # 设置模型为训练模式激活模型里的 Dropout和BatchNorm等操作
    # ==========================================
    # 1. 向前传播计算loss
    # ==========================================
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp else nullcontext(): # 如果 use_amp = True，它会开启 torch.autocast，意味着接下来的计算会自动在 Float32 和 Float16 之间切换，省显存且加速
        output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
        loss = output_dict["loss"]
    
    # ==========================================
    # 2. 反向传播与梯度裁剪
    # ==========================================
    grad_scaler.scale(loss).backward() #先放大loss,即梯度信息，再反向传播，避免精度丢失
    grad_scaler.unscale_(optimizer) #把刚才放大的梯度进行还原
    # 梯度裁剪
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # ==========================================
    # 3. 更新权重
    # ==========================================
    with lock if lock is not None else nullcontext():
        scale_before = grad_scaler.get_scale() # 记录更新前的缩放因子
        grad_scaler.step(optimizer) # 判断是否出现NaN，如果出现就跳过，没有则更新模型权重
        grad_scaler.update() # 更新梯度放大系数，正常会增大一些，如果出现nan会减小
        scale_after = grad_scaler.get_scale()  # 记录更新后的缩放因子
    # grad_scaler.update()
    optimizer.zero_grad(set_to_none=True) #清空梯度，以便下一次反向传播

    # ==========================================
    # 4. 更新调度器与内部状态
    # ==========================================
    # 如果 scaler 被缩小了，说明触发了 NaN/Inf 并跳过了 optimizer.step()
    # 此时应当跳过 scheduler 和 EMA 的更新
    step_was_skipped = scale_after < scale_before

    if not step_was_skipped:
        if lr_scheduler is not None:
            lr_scheduler.step() # 更新学习率
        if isinstance(policy, PolicyWithUpdate):
            with torch.no_grad():
                policy.update()# 模型平滑处理（EMA）
    else:
        print("Warning: Gradient overflow, skipping LR and EMA update.")
    
    info = {
        "loss": loss.item(),
        "grad_norm": float(grad_norm),
        "lr": optimizer.param_groups[0]["lr"],
        "update_s": time.perf_counter() - start_time,
    }
    backbone_group = next(
        (
            group
            for group in optimizer.param_groups
            if group.get("name") == "backbone"
        ),
        None,
    )
    if backbone_group is not None:
        info["backbone_lr"] = backbone_group["lr"]
    elif len(optimizer.param_groups) > 1 and not any(
        "name" in group for group in optimizer.param_groups
    ):
        # 兼容未命名参数组的ACT等原有优化器。
        info["backbone_lr"] = optimizer.param_groups[1]["lr"]
    # 遍历 output_dict，安全地提取数据
    for k, v in output_dict.items():
        if k == "loss":
            continue
        if isinstance(v, torch.Tensor):
            # 如果是标量（单值 Tensor），直接取 .item() 转为普通的 Python float
            if v.numel() == 1:
                info[k] = v.item()
            # 如果是多维张量，必须把它从计算图剥离并转移到 CPU 内存
            else:
                info[k] = v.detach().cpu()
        else:
            info[k] = v

    return info


def update_policy_iwr(
    policy,
    source_batches: dict[str, dict],
    source_weights: dict[str, float],
    optimizer,
    grad_clip_norm,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
):
    """对 D_I、D_R 和可选专家 replay 分别归一化 loss 后完成一次更新。

    分来源前向传播很重要：D_I 通常只有少量受控 action 维度有效，如果先把
    batch 拼起来再全局平均，它的贡献会随受控维度数量变化，无法实现 IWR 的
    等权数据源目标。
    """

    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    outputs = {}
    weighted_losses = []
    active_weight_sum = 0.0

    autocast_context = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )
    with autocast_context:
        for source_name, batch in source_batches.items():
            source_weight = float(source_weights.get(source_name, 0.0))
            if source_weight <= 0.0:
                continue
            output = policy.forward(batch)
            outputs[source_name] = output
            weighted_losses.append(source_weight * output["loss"])
            active_weight_sum += source_weight
        if not weighted_losses:
            raise ValueError("IWR update requires at least one positive source loss weight.")
        # 使用加权平均保持整体梯度尺度与普通单 batch BC 接近。
        loss = torch.stack(weighted_losses).sum() / active_weight_sum

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    scale_before = grad_scaler.get_scale()
    grad_scaler.step(optimizer)
    grad_scaler.update()
    scale_after = grad_scaler.get_scale()
    optimizer.zero_grad(set_to_none=True)

    step_was_skipped = scale_after < scale_before
    if not step_was_skipped:
        if lr_scheduler is not None:
            lr_scheduler.step()
        if isinstance(policy, PolicyWithUpdate):
            with torch.no_grad():
                policy.update()
    else:
        logging.warning("Gradient overflow: skipped IWR optimizer/LR/EMA update.")

    info = {
        "loss": float(loss.detach().item()),
        "grad_norm": float(grad_norm),
        "lr": optimizer.param_groups[0]["lr"],
        "update_s": time.perf_counter() - start_time,
    }
    backbone_group = next(
        (
            group
            for group in optimizer.param_groups
            if group.get("name") == "backbone"
        ),
        None,
    )
    if backbone_group is not None:
        info["backbone_lr"] = backbone_group["lr"]
    elif len(optimizer.param_groups) > 1 and not any(
        "name" in group for group in optimizer.param_groups
    ):
        # 兼容未命名参数组的ACT等原有优化器。
        info["backbone_lr"] = optimizer.param_groups[1]["lr"]
    for source_name, output in outputs.items():
        info[f"{source_name}_loss"] = float(output["loss"].detach().item())
        for key in ("arm_loss", "view_loss"):
            value = output.get(key)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                info[f"{source_name}_{key}"] = float(value.detach().item())
    return info





# 更新训练信息
def log_train_info(logger: Logger, info, step, cfg, dataset):
    loss = info["loss"]
    grad_norm = info["grad_norm"] #梯度范数，衡量了模型参数更新的幅度，过大可能导致训练不稳定，过小可能导致训练停滞
    lr = info["lr"]
    update_s = info["update_s"]   #模型参数更新所花费的时间，单位是秒，这个时间包括了前向传播、损失计算、反向传播、梯度裁剪、优化器更新等操作的时间
    dataloading_s = info["dataloading_s"] #从数据迭代器中获取数据所花费的时间，这个时间包括了数据加载、预处理等操作的时间

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.training.batch_size #已经训练的样本数 = step x batch
    avg_samples_per_ep = dataset.num_samples / dataset.num_episodes #每条轨迹平均产出几条数据 = 数据集总样本数 ÷ 轨迹条数
    num_episodes = num_samples / avg_samples_per_ep #已经训练的轨迹条数 = 已经训练的样本数 ÷ 每条轨迹平均样本数   （相当于遍历了多少条数据）
    num_epochs = num_samples / dataset.num_samples  # Epoch = 训练总样本数 ÷ 数据集总样本数     （相当于遍历了整个数据集多少轮了）
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during training
        f"smpl:{format_big_number(num_samples)}", #已经训练的样本数
        # number of episodes seen during training
        f"ep:{format_big_number(num_episodes)}", #计算得到的遍历数据条数
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}", #计算得到的训练轮次
        f"loss:{loss:.3f}",
    ]
    if "arm_loss" in info:
        log_items.append(f"arm_loss:{info['arm_loss']:.3f}")
    if "view_loss" in info:
        log_items.append(f"view_loss:{info['view_loss']:.3f}")
    for source_name in ("intervention", "robot", "expert"):
        key = f"{source_name}_loss"
        if key in info:
            log_items.append(f"{source_name}_loss:{info[key]:.5f}")
    log_items.extend([
        f"grdn:{grad_norm:.3f}", #梯度范数，衡量了模型参数更新的幅度，过大可能导致训练不稳定，过小可能导致训练停滞
        f"lr:{lr:0.1e}",
        # in seconds
        f"updt_s:{update_s:.3f}", #模型参数更新所花费的时间
        f"data_s:{dataloading_s:.3f}",  # 一般趋近于0，如果这个时间过长，说明cpu太弱了
    ])
    if "backbone_lr" in info:
        log_items.insert(-2, f"bb_lr:{info['backbone_lr']:0.1e}")
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs

    logger.log_dict(info, step, mode="train")





# 更新评估信息
def log_eval_info(logger, info, step, cfg, dataset):
    eval_s = info["eval_s"]
    avg_sum_reward = info["avg_sum_reward"]
    pc_success = info["pc_success"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.training.batch_size
    avg_samples_per_ep = dataset.num_samples / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_samples
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during training
        f"smpl:{format_big_number(num_samples)}",
        # number of episodes seen during training
        f"ep:{format_big_number(num_episodes)}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"∑rwrd:{avg_sum_reward:.3f}",
        f"success:{pc_success:.1f}%",
        f"eval_s:{eval_s:.3f}",
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs

    logger.log_dict(info, step, mode="eval")



def get_resolved_delta_timestamps(cfg: DictConfig) -> dict:
    """
    解析配置文件中的字符串形式的时间戳为真实的 Python 列表，
    并提供严格的 Fail-Fast 防御性检查。
    """
    # 1. 获取配置
    delta_timestamps_cfg = cfg.training.get("delta_timestamps")
    
    # 如果整个节点都不存在，立刻终止！
    if not delta_timestamps_cfg:
        raise ValueError("配置文件中缺失 `training.delta_timestamps` 参数！\n")
        
    # 2. 解析
    delta_timestamps_dict = {}
    for key, value in delta_timestamps_cfg.items():
        if isinstance(value, str):
            delta_timestamps_dict[key] = eval(value)
        else:
            delta_timestamps_dict[key] = list(value)
            
    # 如果节点存在，但漏写了最重要的 `action`，立刻终止！
    if "action" not in delta_timestamps_dict:
        raise ValueError("配置文件`delta_timestamps` 中缺失了最核心的 `action` 时间轴！\n")
        
    # ⚠️ 软警告（可选）：Diffusion 通常还需要历史视觉帧，如果没写，可以给个黄字警告
    if cfg.policy.name in [
        "diffusion",
        "dual_head_diffusion",
        "scid_dual_head_diffusion",
        "coupled_dual_head_diffusion",
        "two_model_diffusion",
    ] and not any("images" in k for k in delta_timestamps_dict.keys()):
        import logging
        logging.warning("警告: 你的 `delta_timestamps` 中没有包含任何图片的过去时间帧。\n")

    return delta_timestamps_dict


def clean_optional_path(value) -> str | None:
    """把 Hydra 中的 none/null/空字符串统一转为 None。"""
    if value is None:
        return None
    text = str(value).strip().strip("'\"")
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def resolve_existing_path(value, option_name: str) -> Path:
    """解析配置路径；相对路径统一以项目根目录为基准。"""
    cleaned_path = clean_optional_path(value)
    if cleaned_path is None:
        raise ValueError(f"HIL 训练必须配置 `{option_name}`。")

    path = Path(cleaned_path).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"`{option_name}` 指向的路径不存在: {path}")
    return path.resolve()


def resolve_pretrained_model_dir(value, option_name: str) -> Path:
    """接受 checkpoint 根目录或其 pretrained_model 子目录。"""
    path = resolve_existing_path(value, option_name)
    model_dir = path / "pretrained_model" if (path / "pretrained_model").is_dir() else path
    if not model_dir.is_dir():
        raise NotADirectoryError(f"`{option_name}` 不是模型目录: {model_dir}")

    required_files = ("config.json", "config.yaml", "model.safetensors")
    missing_files = [name for name in required_files if not (model_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"预训练模型目录不完整: {model_dir}; 缺少 {', '.join(missing_files)}"
        )
    return model_dir


def resolve_resume_checkpoint(value) -> tuple[Path, Path, Path]:
    """解析 HIL checkpoint，并要求完整训练状态可用于真正的断点续训。"""
    path = resolve_existing_path(value, "resume_path")
    checkpoint_dir = path.parent if path.name == "pretrained_model" else path
    model_dir = resolve_pretrained_model_dir(path, "resume_path")
    training_state_file = checkpoint_dir / "training_state.pth"
    if not training_state_file.is_file():
        raise FileNotFoundError(
            "resume=true 必须恢复完整的 HIL 训练状态，但未找到: "
            f"{training_state_file}"
        )
    return checkpoint_dir, model_dir, training_state_file


def _plain_config_value(value):
    """把 tuple/OmegaConf 容器转换为可稳定比较的普通 Python 值。"""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): _plain_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config_value(item) for item in value]
    return value


def validate_pretrained_policy_compatibility(model_dir: Path, cfg: DictConfig) -> None:
    """确保 HIL 策略与专家 checkpoint 是同一模型，而不是隐式架构迁移。"""
    saved_run_cfg = OmegaConf.load(model_dir / "config.yaml")
    saved_policy_name = OmegaConf.select(saved_run_cfg, "policy.name", default=None)
    current_policy_name = str(cfg.policy.name)
    if saved_policy_name is None:
        raise ValueError(f"checkpoint 配置未记录 policy.name: {model_dir / 'config.yaml'}")
    if str(saved_policy_name) != current_policy_name:
        raise ValueError(
            "HIL 策略与专家 checkpoint 模型类型不一致，禁止隐式跨架构加载: "
            f"checkpoint={saved_policy_name!r}, HIL={current_policy_name!r}。"
        )

    with (model_dir / "config.json").open("r", encoding="utf-8") as file:
        saved_policy_cfg = json.load(file)
    current_policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)

    fields = list(BASE_POLICY_COMPATIBILITY_FIELDS)
    if current_policy_name in {
        "dual_head_diffusion",
        "scid_dual_head_diffusion",
        "coupled_dual_head_diffusion",
        "two_model_diffusion",
    }:
        fields.extend(DUAL_HEAD_COMPATIBILITY_FIELDS)
    if current_policy_name == "coupled_dual_head_diffusion":
        fields.extend(COUPLED_COMPATIBILITY_FIELDS)

    mismatches = []
    for field in fields:
        if field not in saved_policy_cfg:
            mismatches.append(f"{field}: checkpoint 中缺失")
            continue
        if field not in current_policy_cfg:
            mismatches.append(f"{field}: HIL 配置中缺失")
            continue
        saved_value = _plain_config_value(saved_policy_cfg[field])
        current_value = _plain_config_value(current_policy_cfg[field])
        if saved_value != current_value:
            mismatches.append(
                f"{field}: checkpoint={saved_value!r}, HIL={current_value!r}"
            )

    if mismatches:
        mismatch_text = "\n  - ".join(mismatches)
        raise ValueError(
            "HIL 配置与专家 checkpoint 不兼容。首次 HIL 必须在同一策略结构上微调:"
            f"\n  - {mismatch_text}"
        )


def load_local_lerobot_dataset(
    local_dir: str | Path,
    delta_timestamps: dict,
    video_backend: str | None,
    image_transforms=None,
    cache_dir: str | Path | None = None,
) -> LeRobotDataset:
    """从 convert_data_to_hf.py 生成的本地 LeRobot/HF 目录加载数据集。"""
    local_dir = Path(local_dir).expanduser()
    required_paths = [
        local_dir / "data",
        local_dir / "meta_data" / "info.json",
        local_dir / "meta_data" / "stats.safetensors",
        local_dir / "meta_data" / "episode_data_index.safetensors",
        local_dir / "videos",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"本地数据集目录不完整: {local_dir}\n缺失:\n{missing_text}"
        )

    parquet_files = sorted((local_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"本地数据集目录中没有 parquet 文件: {local_dir / 'data'}")

    hf_dataset = datasets.Dataset.from_parquet(
        [str(path) for path in parquet_files],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    requested_camera_keys = {
        key for key in delta_timestamps if key.startswith("observation.images.")
    }
    unused_camera_keys = [
        key
        for key, feature in hf_dataset.features.items()
        if isinstance(feature, VideoFrame)
        and requested_camera_keys
        and key not in requested_camera_keys
    ]
    if unused_camera_keys:
        # LeRobotDataset 会解码 features 中的全部视频列；HIL 同时开 D_I/D_R
        # 两个 loader，只保留策略实际使用的相机可显著降低 CPU/显存压力。
        hf_dataset = hf_dataset.remove_columns(unused_camera_keys)
        logging.info(f"数据集 {local_dir.name} 忽略未使用相机: {unused_camera_keys}")
    hf_dataset.set_transform(hf_transform_to_torch)

    # 同一份配置会同时加载普通专家数据和带额外 IWR 字段的完整 HIL 数据。
    # 仅向各数据集请求它实际拥有的 temporal key，避免专家数据因缺少
    # teleop_action/intervention_action_weight 而报错。
    available_features = set(hf_dataset.features)
    filtered_delta_timestamps = {
        key: value
        for key, value in delta_timestamps.items()
        if key in available_features
    }
    skipped_temporal_keys = sorted(set(delta_timestamps) - available_features)
    if skipped_temporal_keys:
        logging.info(
            f"数据集 {local_dir.name} 跳过不存在的时间字段: {skipped_temporal_keys}"
        )

    with open(local_dir / "meta_data" / "info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    stats = unflatten_dict(load_file(local_dir / "meta_data" / "stats.safetensors"))
    episode_data_index = load_file(local_dir / "meta_data" / "episode_data_index.safetensors")

    dataset = LeRobotDataset.from_preloaded(
        repo_id=local_dir.name,
        root=local_dir.parent,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        stats=stats,
        info=info,
        videos_dir=local_dir / "videos",
        video_backend=video_backend,
        delta_timestamps=filtered_delta_timestamps,
        transform=image_transforms,
    )

    # from_preloaded 不会执行 LeRobotDataset.__init__，这里补齐当前项目依赖的 resize 行为。
    dataset.resize = v2.Resize((480, 640))
    return dataset



# ✅ 这是一个完美且内存安全的 PyTorch 无限数据生成器
def get_infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def make_episode_aware_sampler_if_needed(dataset, cfg):
    """按配置创建 LeRobot 的 EpisodeAwareSampler。"""
    if cfg.training.get("drop_n_last_frames"):
        return EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.training.get("drop_n_last_frames"),
            shuffle=True,
        )
    return None


def make_training_dataloader(dataset, cfg, device, *, batch_size: int):
    """创建训练 dataloader；兼容专家数据和人类干预数据。"""
    sampler = make_episode_aware_sampler_if_needed(dataset, cfg)
    return DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=int(batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=(device.type != "cpu"),
        drop_last=True,
    )


def _raw_hf_column(dataset: LeRobotDataset, key: str):
    """直接读取 Arrow 列，绕过图像/torch transform。"""

    if key not in dataset.hf_dataset.column_names:
        raise KeyError(
            f"HIL 数据集缺少 IWR 必需字段 {key!r}。请用 "
            "convert_intervention_to_hf.py --dataset-mode full_rollout 重新转换。"
        )
    return dataset.hf_dataset.data.column(key).to_pylist()


def classify_iwr_sample_indices(
    *,
    episode_from: np.ndarray,
    episode_to: np.ndarray,
    is_intervention: np.ndarray,
    intervention_action_weight: np.ndarray,
    horizon: int,
    min_intervention_weight: float,
    drop_n_last_frames: int = 0,
) -> tuple[list[int], list[int]]:
    """构造 IWR 的 D_I 与纯自主 D_R anchor 索引。

    D_I 要求当前帧确实有至少一个受人工控制且权重大于阈值的 action 维度。
    D_R 要求从当前 anchor 开始的整个 Diffusion action horizon 内都没有人工
    监督，防止机器人 loss 意外学习未来的人类动作。
    """

    is_intervention = np.asarray(is_intervention, dtype=np.bool_).reshape(-1)
    action_weight = np.asarray(intervention_action_weight, dtype=np.float32)
    if action_weight.ndim != 2 or action_weight.shape[0] != is_intervention.shape[0]:
        raise ValueError(
            "intervention_action_weight must be [frames, action_dim], got "
            f"{action_weight.shape} for {is_intervention.shape[0]} frames."
        )
    human_frame = is_intervention & (
        np.max(action_weight, axis=1) > float(min_intervention_weight)
    )
    intervention_indices: list[int] = []
    robot_indices: list[int] = []
    horizon = max(1, int(horizon))
    drop_n_last_frames = max(0, int(drop_n_last_frames))

    for start, stop in zip(episode_from.tolist(), episode_to.tolist(), strict=True):
        start = int(start)
        stop = int(stop)
        anchor_stop = max(start, stop - drop_n_last_frames)
        for index in range(start, anchor_stop):
            if human_frame[index]:
                intervention_indices.append(index)
                continue
            future_stop = min(stop, index + horizon)
            # release blending 等 teleop_applied 帧可能没有主动 human mask，但执行
            # 动作仍不是纯 policy action；它们既不进入 D_I，也不能污染 D_R。
            if not np.any(is_intervention[index:future_stop]):
                robot_indices.append(index)

    return intervention_indices, robot_indices


def build_iwr_sample_indices(
    dataset: LeRobotDataset,
    cfg: DictConfig,
    *,
    require_robot: bool = True,
) -> tuple[list[int], list[int]]:
    """从完整 HIL 数据集元数据中创建 D_I/D_R 索引。"""

    dataset_mode = str(dataset.info.get("dataset_mode", ""))
    dataset_kind = str(dataset.info.get("dataset_kind", ""))
    if dataset_mode != "full_rollout" and dataset_kind != "human_in_loop_rollout":
        raise ValueError(
            "IWR 需要完整 policy+human rollout 数据集，当前数据集为 "
            f"dataset_mode={dataset_mode!r}, dataset_kind={dataset_kind!r}。"
            "请用 --dataset-mode full_rollout 重新转换，不能使用仅含干预片段的数据集。"
        )

    iwr_cfg = cfg.training.iwr
    intervention_indices, robot_indices = classify_iwr_sample_indices(
        episode_from=np.asarray(dataset.episode_data_index["from"].cpu()),
        episode_to=np.asarray(dataset.episode_data_index["to"].cpu()),
        is_intervention=np.asarray(_raw_hf_column(dataset, "is_intervention"), dtype=np.bool_),
        intervention_action_weight=np.asarray(
            _raw_hf_column(dataset, "intervention_action_weight"),
            dtype=np.float32,
        ),
        horizon=int(cfg.policy.horizon),
        min_intervention_weight=float(iwr_cfg.get("min_intervention_weight", 1.0e-6)),
        drop_n_last_frames=int(cfg.training.get("drop_n_last_frames", 0) or 0),
    )
    if not intervention_indices:
        raise ValueError("完整 HIL 数据集中没有可用的人类干预 anchor (D_I)。")
    if require_robot and not robot_indices:
        raise ValueError(
            "完整 HIL 数据集中没有 action horizon 内完全自主的机器人 anchor (D_R)。"
        )
    return intervention_indices, robot_indices


def make_indexed_dataloader(dataset, cfg, device, *, indices: list[int], batch_size: int):
    """为 D_I/D_R 指定索引创建随机 dataloader。"""

    if int(batch_size) <= 0:
        return None
    if not indices:
        raise ValueError("Indexed dataloader received an empty index list.")
    return DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=int(batch_size),
        sampler=SubsetRandomSampler(indices),
        pin_memory=(device.type != "cpu"),
        # 小规模新一轮干预可能少于 batch_size；保留最后一个 batch 避免无限
        # dataloader 在没有任何产出时空转。
        drop_last=False,
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """只搬运 Tensor，字符串等溯源字段保持原样。"""

    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def prepare_intervention_batch(batch: dict, cfg: DictConfig) -> dict:
    """为 D_I 构造完整 action target 和逐维监督权重。

    ``action`` 是环境实际执行的完整动作，因此它天然包含人工控制维度以及
    同一时刻仍由策略控制的其他维度。raw ``teleop_action`` 可用时，只替换
    ``intervention_action_weight`` 标记为人工控制的维度，不能覆盖右臂/视角等
    非人工维度。

    所有 action 维度的基础 loss 权重均为 1。``human_action_weight`` 只用于在
    此基础上增强人工维度；它不会再像旧的二值 mask 一样删除非人工维度监督。
    """

    required = {"action", "intervention_action_weight"}
    missing = required.difference(batch)
    if missing:
        raise KeyError(
            f"D_I batch 缺少 {sorted(missing)}；请确认转换和 delta_timestamps 配置。"
        )
    batch = dict(batch)
    executed_action = batch["action"]
    intervention_weight = batch["intervention_action_weight"].to(
        dtype=executed_action.dtype
    )
    if intervention_weight.shape != executed_action.shape:
        raise ValueError(
            "intervention_action_weight 必须与 action 同为 [B,H,D]，"
            f"当前 mask={tuple(intervention_weight.shape)}, "
            f"action={tuple(executed_action.shape)}。"
        )

    target_action = executed_action.clone()

    prefer_teleop_action = bool(cfg.training.iwr.get("prefer_teleop_action", True))
    if prefer_teleop_action and "teleop_action" in batch:
        teleop_action = batch["teleop_action"].to(dtype=executed_action.dtype)
        if teleop_action.shape != executed_action.shape:
            raise ValueError(
                f"teleop_action shape={tuple(teleop_action.shape)} "
                f"!= action {tuple(executed_action.shape)}."
            )
        available = batch.get("teleop_action_available")
        if available is None:
            # 兼容新字段出现前人工补建的数据集：存在 teleop_action 即视为可用。
            available = torch.ones(
                executed_action.shape[:-1],
                dtype=torch.bool,
                device=executed_action.device,
            )
        else:
            available = available.to(device=executed_action.device, dtype=torch.bool)
        while available.ndim < executed_action.ndim:
            available = available.unsqueeze(-1)

        min_intervention_weight = float(
            cfg.training.iwr.get("min_intervention_weight", 1.0e-6)
        )
        human_dimension = intervention_weight > min_intervention_weight
        replace_mask = available & human_dimension
        target_action = torch.where(replace_mask, teleop_action, executed_action)

    human_action_weight = float(cfg.training.iwr.get("human_action_weight", 1.0))
    if human_action_weight <= 0.0:
        raise ValueError(
            "training.iwr.human_action_weight 必须大于 0，"
            f"当前为 {human_action_weight}。"
        )

    # 基础权重为 1，保证左右臂和视角等完整 action 都有监督。人工维度可在
    # [1, human_action_weight] 内随接管 blend 权重平滑增强。
    loss_mask = torch.ones_like(executed_action)
    loss_mask = loss_mask + (human_action_weight - 1.0) * intervention_weight.clamp(
        0.0, 1.0
    )

    batch["action"] = target_action
    batch["loss_mask"] = loss_mask
    return batch


def concat_batches(*batches: dict) -> dict:
    """沿 batch 维拼接多个 dataloader batch，只保留所有 batch 共同拥有的字段。"""
    if not batches:
        raise ValueError("concat_batches requires at least one batch.")

    common_keys = set(batches[0].keys())
    for batch in batches[1:]:
        common_keys &= set(batch.keys())

    merged = {}
    for key in sorted(common_keys):
        values = [batch[key] for batch in batches]
        first = values[0]
        if isinstance(first, torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(first, np.ndarray):
            merged[key] = np.concatenate(values, axis=0)
        elif isinstance(first, list):
            output = []
            for value in values:
                output.extend(value)
            merged[key] = output
        else:
            # 训练用的关键字段通常都是 Tensor；保留这个分支只是为了兼容极少数元数据。
            output = []
            for value in values:
                if isinstance(value, tuple):
                    output.extend(value)
                else:
                    output.append(value)
            merged[key] = output
    return merged



def train_human_in_loop(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    """
    从专家预训练策略初始化，使用专家数据与人类干预数据进行离线 HIL 微调。

    首次 HIL 只继承专家模型权重和归一化统计，optimizer/scheduler/step 全部重置；
    resume=true 仅用于恢复已经启动过的 HIL 训练。
    """

    # 在创建 logger 和加载数据集之前完成 fail-fast 校验，HIL 永远不允许随机初始化。
    start_step = 0
    checkpoint_dir = None
    training_state_file = None
    pretrained_model_path = clean_optional_path(cfg.get("pretrained_model_path", None))
    legacy_init_policy_path = clean_optional_path(cfg.get("init_policy_path", None))

    if cfg.resume:
        checkpoint_dir, policy_load_path, training_state_file = resolve_resume_checkpoint(
            cfg.get("resume_path", None)
        )
        model_source_description = f"恢复 HIL checkpoint: {checkpoint_dir}"
    else:
        if pretrained_model_path is not None and legacy_init_policy_path is not None:
            raise ValueError(
                "请只使用 HIL 专用接口 `pretrained_model_path`；"
                "不要同时设置旧接口 `init_policy_path`。"
            )
        if pretrained_model_path is None:
            if legacy_init_policy_path is None:
                raise ValueError(
                    "首次 HIL 训练必须提供专家模型："
                    "`pretrained_model_path=<checkpoint目录或pretrained_model目录>`。"
                )
            pretrained_model_path = legacy_init_policy_path
        policy_load_path = resolve_pretrained_model_dir(
            pretrained_model_path,
            "pretrained_model_path",
        )
        model_source_description = (
            "从专家 checkpoint 初始化新的 HIL 实验（重置 optimizer/scheduler/step）: "
            f"{policy_load_path}"
        )

    validate_pretrained_policy_compatibility(policy_load_path, cfg)

    init_logging() #初始化日志
    logging.info(pformat(OmegaConf.to_container(cfg))) #打印配置cfg
    logging.info("模型初始化来源: %s", model_source_description)
    logging.info("模型归一化统计将严格继承该 checkpoint，不会由干预小数据集重新估计。")

    # 初始化日志记录器与设备
    wandb_enabled = bool(getattr(cfg.wandb, "enable", False))
    disable_system_stats = bool(getattr(cfg.wandb, "disable_system_stats", False))
    disable_machine_info = bool(getattr(cfg.wandb, "disable_machine_info", False))
    if wandb_enabled and (disable_system_stats or disable_machine_info):
        import wandb

        wandb.setup(
            settings=wandb.Settings(
                x_disable_stats=disable_system_stats,
                x_disable_machine_info=disable_machine_info,
            )
        )
        logging.info(
            f"W&B系统数据设置: disable_system_stats={disable_system_stats}, "
            f"disable_machine_info={disable_machine_info}"
        )
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)
    add_wandb_parameter_tags(logger, cfg)
    set_global_seed(cfg.seed)
    device = get_safe_torch_device(cfg.device, log=True)

    # 开启 CuDNN 加速和 TF32 支持
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    # ==========================================
    # 🌟 1. 动态构建时间戳与挂载数据集
    # ==========================================
    logging.info("📦 正在挂载离线专家数据集...")
    image_transforms = None
    # 色彩/像素级增强，对图像添加光亮色彩等随机扰动
    if cfg.training.image_transforms.enable:
        cfg_tf = cfg.training.image_transforms
        image_transforms = get_image_transforms(
            brightness_weight=cfg_tf.brightness.weight,
            brightness_min_max=cfg_tf.brightness.min_max,
            contrast_weight=cfg_tf.contrast.weight,
            contrast_min_max=cfg_tf.contrast.min_max,
            saturation_weight=cfg_tf.saturation.weight,
            saturation_min_max=cfg_tf.saturation.min_max,
            hue_weight=cfg_tf.hue.weight,
            hue_min_max=cfg_tf.hue.min_max,
            sharpness_weight=cfg_tf.sharpness.weight,
            sharpness_min_max=cfg_tf.sharpness.min_max,
            max_num_transforms=cfg_tf.max_num_transforms,
            random_order=cfg_tf.random_order,
        )
    # # 解析配置文件中的时间戳
    resolved_delta_timestamps = get_resolved_delta_timestamps(cfg)
    logging.info(f"解析到的动作时间轴: {resolved_delta_timestamps.get('action', [])[:5]} ...")
    logging.info(f"解析到的视觉时间轴: {resolved_delta_timestamps.get('observation.state', [])}")


    dataset_local_dir = clean_optional_path(cfg.get("dataset_local_dir", None))
    hil_dataset_local_dir = clean_optional_path(cfg.get("hil_dataset_local_dir", None))
    legacy_intervention_dir = clean_optional_path(cfg.get("intervention_dataset_local_dir", None))
    if hil_dataset_local_dir is None:
        hil_dataset_local_dir = legacy_intervention_dir
    dataset_cache_dir = clean_optional_path(cfg.get("dataset_cache_dir", None))
    iwr_enabled = bool(OmegaConf.select(cfg, "training.iwr.enabled", default=False))
    if iwr_enabled and str(cfg.policy.name) != "dual_head_diffusion":
        raise ValueError(
            "当前逐 action 维度 IWR loss 仅在 dual_head_diffusion 中实现；"
            f"当前 policy={cfg.policy.name!r}。"
        )
    if dataset_local_dir:
        logging.info(f"使用服务器本地数据集: {dataset_local_dir}")
        offline_dataset = load_local_lerobot_dataset(
            local_dir=dataset_local_dir,
            delta_timestamps=resolved_delta_timestamps,
            video_backend=cfg.video_backend,
            image_transforms=image_transforms,
            cache_dir=dataset_cache_dir,
        )
    else:
        logging.info(f"使用 Hugging Face 数据集: {cfg.dataset_repo_id}")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset_repo_id, #根据id下载或者加载本地数据（/home/dc/.cache/huggingface/datasets）
            delta_timestamps=resolved_delta_timestamps,
            video_backend=cfg.video_backend,
            image_transforms=image_transforms,
        )

    hil_dataset = None
    if hil_dataset_local_dir:
        logging.info(f"使用完整人在环本地数据集: {hil_dataset_local_dir}")
        hil_dataset = load_local_lerobot_dataset(
            local_dir=hil_dataset_local_dir,
            delta_timestamps=resolved_delta_timestamps,
            video_backend=cfg.video_backend,
            image_transforms=image_transforms,
            cache_dir=dataset_cache_dir,
        )
        logging.info(
            "人在环数据: "
            f"kind={hil_dataset.info.get('dataset_kind')} "
            f"mode={hil_dataset.info.get('dataset_mode')} "
            f"episodes={hil_dataset.num_episodes} samples={hil_dataset.num_samples}"
        )
    if iwr_enabled and hil_dataset is None:
        raise ValueError(
            "training.iwr.enabled=true 时必须配置 hil_dataset_local_dir，"
            "并指向 full_rollout 转换结果。"
        )
    # # 使用官方函数解析并挂载到 cfg，这样 make_dataset 内部才能正确读取
    # resolve_delta_timestamps(cfg)
    
    # ==========================================
    # 🌟 3. 初始化模型与优化器 (顺序极其重要！)
    # ==========================================
    logging.info("🧠 正在初始化 Diffusion Policy...")
    
    # 3.1 首次 HIL 与 resume 都严格加载已有权重；此脚本不存在随机初始化分支。
    policy = make_policy(
        hydra_cfg=cfg,
        dataset_stats=None,
        pretrained_policy_name_or_path=str(policy_load_path),
        allow_scid_dual_init=(
            cfg.policy.name == "scid_dual_head_diffusion"
            and not cfg.resume
        ),
        strict_pretrained_loading=True,
    )
    policy.to(device)

    if cfg.policy.name == "scid_dual_head_diffusion":
        scid_fit = initialize_scid_transform_from_dataset(
            policy,
            offline_dataset,
            resume=bool(cfg.resume),
        )
        if scid_fit is None:
            logging.info("SCID变换已从checkpoint严格恢复，跳过重新拟合。")
        else:
            diagnostics = scid_fit.diagnostics
            logging.info(
                "SCID变换拟合完成: frames=%d, mean_R2=%.4f, "
                "cross_corr=%.4f->%.4f, condition=%.3e, scale=%s",
                diagnostics["num_frames"],
                diagnostics["view_r2_mean"],
                diagnostics["raw_cross_corr_norm"],
                diagnostics["residual_cross_corr_norm"],
                diagnostics["condition_number"],
                [round(value, 6) for value in diagnostics["residual_scale"]],
            )

    # 3.2 无论是不是 resume，都必须先根据模型初始化出全新的优化器！
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(enabled=cfg.use_amp) # 用于自动计算梯度缩放因子

    # ==========================================
    # 🌟 4. 恢复优化器与步数状态
    # ==========================================
    if cfg.resume:
        logging.info("🔄 正在恢复优化器与训练步数...")
        checkpoint_dict = torch.load(
            training_state_file,
            map_location="cpu",
            weights_only=False,
        )
        required_state_keys = {"optimizer", "step"}
        missing_state_keys = required_state_keys.difference(checkpoint_dict)
        if missing_state_keys:
            raise KeyError(
                f"HIL training_state.pth 缺少字段: {sorted(missing_state_keys)}"
            )

        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        logging.info("✅ Optimizer (优化器) 状态已恢复")

        if lr_scheduler is not None:
            scheduler_state = checkpoint_dict.get(
                "lr_scheduler",
                checkpoint_dict.get("scheduler"),
            )
            if scheduler_state is None:
                raise KeyError("HIL training_state.pth 中未找到 scheduler 状态")
            lr_scheduler.load_state_dict(scheduler_state)
            logging.info("✅ LR Scheduler (调度器) 状态已恢复")

        if cfg.use_amp and "grad_scaler" in checkpoint_dict:
            grad_scaler.load_state_dict(checkpoint_dict["grad_scaler"])
            logging.info("✅ GradScaler 状态已恢复")

        start_step = int(checkpoint_dict["step"]) + 1
        logging.info(f"⏭️ HIL 训练将从 step {start_step} 继续")


    # ==========================================
    # 🌟 5. 构建标准的高并发数据加载器 (彻底解耦)
    # ==========================================
    total_batch_size = int(cfg.training.batch_size)
    dl_iter = None
    intervention_dl_iter = None
    iwr_source_iters: dict[str, object] = {}
    iwr_source_weights: dict[str, float] = {}
    training_log_dataset = offline_dataset

    if iwr_enabled:
        iwr_cfg = cfg.training.iwr
        intervention_batch_size = int(iwr_cfg.intervention_batch_size)
        robot_batch_size = int(iwr_cfg.robot_batch_size)
        expert_batch_size = int(iwr_cfg.get("expert_batch_size", 0))
        if (
            intervention_batch_size <= 0
            or robot_batch_size < 0
            or expert_batch_size < 0
            or robot_batch_size + expert_batch_size <= 0
        ):
            raise ValueError(
                "IWR batch size 要求 intervention>0、robot/expert>=0，且 robot 或 "
                "expert 至少一路启用，当前为 "
                f"{intervention_batch_size}/{robot_batch_size}/{expert_batch_size}。"
            )
        configured_total = intervention_batch_size + robot_batch_size + expert_batch_size
        if configured_total != total_batch_size:
            raise ValueError(
                "training.batch_size 必须等于 IWR 三个来源 batch size 之和："
                f"batch_size={total_batch_size}, sources={configured_total}。"
            )

        intervention_indices, robot_indices = build_iwr_sample_indices(
            hil_dataset,
            cfg,
            require_robot=robot_batch_size > 0,
        )
        intervention_dataloader = make_indexed_dataloader(
            hil_dataset,
            cfg,
            device,
            indices=intervention_indices,
            batch_size=intervention_batch_size,
        )
        iwr_source_iters["intervention"] = iter(get_infinite_dataloader(intervention_dataloader))
        iwr_source_weights = {
            "intervention": float(iwr_cfg.get("intervention_loss_weight", 1.0)),
        }
        if robot_batch_size > 0:
            robot_dataloader = make_indexed_dataloader(
                hil_dataset,
                cfg,
                device,
                indices=robot_indices,
                batch_size=robot_batch_size,
            )
            iwr_source_iters["robot"] = iter(get_infinite_dataloader(robot_dataloader))
            iwr_source_weights["robot"] = float(iwr_cfg.get("robot_loss_weight", 1.0))
        if expert_batch_size > 0:
            expert_dataloader = make_training_dataloader(
                offline_dataset,
                cfg,
                device,
                batch_size=expert_batch_size,
            )
            iwr_source_iters["expert"] = iter(get_infinite_dataloader(expert_dataloader))
            iwr_source_weights["expert"] = float(iwr_cfg.get("expert_loss_weight", 0.25))

        training_log_dataset = hil_dataset
        logging.info(
            f"IWR batch: D_I={intervention_batch_size}/{len(intervention_indices)} anchors, "
            f"D_R={robot_batch_size}/{len(robot_indices)} anchors, expert={expert_batch_size}; "
            f"loss weights={iwr_source_weights}"
        )
    elif hil_dataset is None:
        expert_batch_size = total_batch_size
        dataloader = make_training_dataloader(
            offline_dataset,
            cfg,
            device,
            batch_size=expert_batch_size,
        )
        dl_iter = iter(get_infinite_dataloader(dataloader))
        logging.info(f"训练 batch 构成: expert={expert_batch_size}, intervention=0")
    else:
        if total_batch_size < 2:
            raise ValueError("启用 intervention_dataset_local_dir 时，training.batch_size 至少需要为 2。")
        intervention_ratio = float(cfg.training.get("intervention_ratio", 0.5))
        if not 0.0 < intervention_ratio < 1.0:
            raise ValueError(f"training.intervention_ratio 必须在 (0, 1) 内，当前为 {intervention_ratio}")
        intervention_batch_size = int(round(total_batch_size * intervention_ratio))
        intervention_batch_size = max(1, min(total_batch_size - 1, intervention_batch_size))
        expert_batch_size = total_batch_size - intervention_batch_size

        expert_dataloader = make_training_dataloader(
            offline_dataset,
            cfg,
            device,
            batch_size=expert_batch_size,
        )
        intervention_dataloader = make_training_dataloader(
            hil_dataset,
            cfg,
            device,
            batch_size=intervention_batch_size,
        )
        dl_iter = iter(get_infinite_dataloader(expert_dataloader))
        intervention_dl_iter = iter(get_infinite_dataloader(intervention_dataloader))
        logging.info(
            "训练 batch 构成: expert=%d, intervention=%d, intervention_ratio=%.3f",
            expert_batch_size,
            intervention_batch_size,
            intervention_batch_size / total_batch_size,
        )

    log_trainable_parameter_counts(policy)
    logging.info(f"HIL 微调目标步数: {cfg.training.offline_steps}")

    # ==========================================
    # 🌟 5. 动态拼接环境 ID 并创建环境
    # ==========================================
    # 观测要用的相机列表  =  模型推理要用的相机列表 + 评估时保存的video视角相机
    all_obs_keys = policy.config.input_shapes.keys()
    ref_cams = [k.replace("observation.images.", "") for k in all_obs_keys if "observation.images." in k]
    if not ref_cams:
        raise ValueError(f"❌ 严重冲突：模型中未找到相机相关参数。请检查模型输入是否正确。")
    obs_cameras = list(dict.fromkeys(ref_cams + cfg.eval.render_camera))

    # 读取 YAML 中的 name ("guided_vision") 和 task ("InsertCylinder-3Arms-v0")
    # 拼接出 "guided_vision/InsertCylinder-3Arms-v0"
    env_id = f"{cfg.env.name}/{cfg.env.task}" 
    
    logging.info(f"正在通过 Gym 注册表构建环境: {env_id}")

    eval_env = make_eval_env(env_id, obs_cameras, cfg.eval)
    logging.info(f"✅ 环境加载成功！最终挂载的相机: {obs_cameras}")

    # ==========================================
    # 🌟 6. HIL 离线微调主循环
    # ==========================================
    max_checkpoints = getattr(cfg.eval, "max_checkpoints", 5)
    records_resume = getattr(cfg.eval, "records_resume", True)
    checkpoint_metric = getattr(cfg.eval, "checkpoint_metric", "loss")
    manager = TopKCheckpointManager(out_dir=out_dir, 
                                    max_keep=max_checkpoints, 
                                    records_resume=records_resume, 
                                    metric=checkpoint_metric)
    policy.train()
    logging.info("🔥 开始 HIL 混合数据微调...")
    
    # 从 start_step 开始，避免覆盖之前的进度！
    for step in range(start_step, cfg.training.offline_steps):
        start_time = time.perf_counter()
        
        # 获取数据并推入 GPU
        if iwr_enabled:
            source_batches = {
                source_name: move_batch_to_device(next(source_iter), device)
                for source_name, source_iter in iwr_source_iters.items()
            }
            source_batches["intervention"] = prepare_intervention_batch(
                source_batches["intervention"],
                cfg,
            )
            dataloading_s = time.perf_counter() - start_time
            train_info = update_policy_iwr(
                policy,
                source_batches,
                iwr_source_weights,
                optimizer,
                cfg.training.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.use_amp,
            )
        else:
            batch = next(dl_iter) # 取出专家 batch
            if intervention_dl_iter is not None:
                intervention_batch = next(intervention_dl_iter)
                batch = concat_batches(batch, intervention_batch)
            dataloading_s = time.perf_counter() - start_time
            batch = move_batch_to_device(batch, device)
            train_info = update_policy(
                policy,
                batch,
                optimizer,
                cfg.training.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.use_amp,
            )
        train_info["dataloading_s"] = dataloading_s

        # 日志记录
        if step % cfg.training.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, training_log_dataset)

        # ==========================================
        # 评估和保存函数
        # ==========================================
        evaluate_and_checkpoint_if_needed(
            step=step,
            policy=policy,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=logger,
            cfg=cfg,
            device=device,
            out_dir=out_dir,
            eval_env=eval_env,        # 预训练阶段如果没有验证环境，直接传 None
            train_loss=train_info["loss"],
            manager=manager,
        )
    logging.info("HIL 微调结束！")

# ==========================================
# 🌟 Hydra 启动入口 (保留配置功能与 Args 注入)
# ==========================================
@hydra.main(version_base="1.2", config_name="pre_default", config_path="../../configs/pretrain") #配置文件存放位置
def train_cli(cfg: DictConfig):

    train_human_in_loop(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,  # 获取当前训练运行的输出目录，用于保存训练输出的数据
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name, # 获取当前训练运行的作业名称，用于wandb
    )

if __name__ == "__main__":
    # checkpoint 名通常含有 loss=.../sr=...。Hydra 会把路径中的第二个等号
    # 误判为 override 语法，因此在交给 Hydra 前自动把路径值包成字符串。
    path_override_keys = {
        "pretrained_model_path",
        "resume_path",
        "init_policy_path",
    }
    for argument_index, argument in enumerate(sys.argv[1:], start=1):
        if "=" not in argument:
            continue
        raw_key, raw_value = argument.split("=", 1)
        if raw_key.lstrip("+") not in path_override_keys:
            continue
        if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {"'", '"'}:
            continue
        escaped_value = raw_value.replace("\\", "\\\\").replace('"', '\\"')
        sys.argv[argument_index] = f'{raw_key}="{escaped_value}"'

    # 强行注入命令行参数 (极大提升本地调试和修改效率)
    # 这里面也可以随时添加你想覆盖的 args 参数
    default_args = [
        "env=sim_insert_cylinder_3arms", # 环境，这俩定义在default文件中
        "policy=pre_human_in_loop", # HIL 专用配置，见 configs/pretrain/policy/pre_human_in_loop.yaml
        # 专家 checkpoint 默认写在 HIL 配置中，也可用 pretrained_model_path=<路径> 覆盖。
        "pretrained_model_path='outputs/2_pretrain/train/2026-07-16/20-59-53_InsertCylinder-3Arms-v0_pre_zed_dual_head_diffusion/checkpoints/162000_loss=0.0051_sr=87.0_ar=751.50'",
        # resume=true 只用于继续同一个训练 run，会恢复 optimizer/scheduler/step。
        "wandb.enable=true",
    ]

    # 只比较 Hydra override 等号左边的完整键名，避免 pretrained_model_path
    # 中包含 "policy" 而错误阻止 policy=pre_human_in_loop 的注入。
    provided_arg_keys = {
        argument.split("=", 1)[0].lstrip("+")
        for argument in sys.argv[1:]
        if "=" in argument
    }
    for arg in default_args:
        arg_key = arg.split("=", 1)[0]
        if arg_key not in provided_arg_keys:
            sys.argv.append(arg)

    # ==========================================
    # 🌟 核心修复：在 Hydra 启动前截胡！强行修改底层输出目录
    # ==========================================
    # 使用 replace(" ", "") 过滤掉所有可能的空格干扰
    is_resume = any(arg.lower().replace(" ", "") == "resume=true" for arg in sys.argv)
    resume_path_arg = next((arg for arg in sys.argv if arg.startswith("resume_path=")), None)

    if is_resume and resume_path_arg:
        resume_path = resume_path_arg.split("=", 1)[1].strip("'\"")
        
        # 只要路径有效，就强行重定向
        if resume_path.lower() not in ["none", "null", ""]:
            ckpt_path = Path(resume_path)
            if not ckpt_path.is_absolute():
                ckpt_path = ROOT_DIR / ckpt_path
            checkpoint_dir = ckpt_path.parent if ckpt_path.name == "pretrained_model" else ckpt_path
            # checkpoints/last 的上一级的上一级，就是原本的训练根目录
            original_out_dir = str(checkpoint_dir.parent.parent.absolute())
            
            # 告诉 Hydra：不要建新文件夹了，日志、配置、视频统统给我存进这个老目录！
            sys.argv.append(f'hydra.run.dir="{original_out_dir}"')
            print(f"🔄 [预处理] 检测到断点续训，已强制重定向所有输出至旧目录:\n   👉 {original_out_dir}")
    
    train_cli()
