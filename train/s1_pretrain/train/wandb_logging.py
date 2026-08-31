"""W&B-facing logging helpers for pretrain runs."""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from omegaconf import DictConfig, OmegaConf

from lerobot.common.logger import Logger
from lerobot.common.utils.utils import format_big_number


WANDB_TAG_MAX_LENGTH = 64  # W&B单个run标签允许的最大字符数


PRETRAIN_WANDB_PARAMETER_TAGS = (
    ("device", "device"),                                 # 训练设备
    ("batch", "training.batch_size"),                     # 训练批大小
    ("lr", "training.lr"),                                # 主网络学习率
    ("backbone_lr", "training.lr_backbone"),              # 视觉底座学习率
    ("steps", "training.offline_steps"),                  # 预训练总步数
    ("epochs", "training.offline_epochs"),                # 用户配置的epoch预算
    ("steps_per_epoch", "training.steps_per_epoch"),      # Sampler解析的每epoch步数
    ("eval_epochs", "training.eval_freq_epochs"),         # epoch模式评估周期
    ("eval_start", "training.eval_start_epoch"),          # epoch模式开始评估的轮次
    ("save_epochs", "training.save_freq_epochs"),         # epoch模式保存周期
    ("lr_scheduler", "training.lr_scheduler"),            # 学习率调度器
    ("lr_warmup", "training.lr_warmup_steps"),            # 学习率预热步数
    ("lr_decay_epochs", "training.lr_decay_epochs"),      # 余弦衰减到下限的epoch
    ("lr_decay_steps", "training.lr_decay_steps"),        # 余弦衰减到下限的全局step
    ("min_lr_ratio", "training.min_lr_ratio"),            # 学习率下限/初始学习率
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
    ("ema_start", "policy.ema_update_after_step"),        # EMA开始更新的训练步
    ("arm_dim", "policy.arm_action_dim"),                 # 双臂动作维度
    ("view_dim", "policy.view_action_dim"),               # 视角动作维度
    ("view_repr", "policy.view_action_representation"),   # View输出为绝对关节或当前锚点增量
    ("view_weight", "policy.view_loss_weight"),           # 视角损失权重
    ("eef_pos_weight", "policy.eef_pose_position_loss_weight"),  # 可微FK末端位置损失权重
    ("eef_rot_weight", "policy.eef_pose_rotation_loss_weight"),  # 可微FK末端旋转损失权重
    ("eef_pose_tmax", "policy.eef_pose_loss_max_timestep"),      # 启用末端位姿监督的最大扩散时间步
    ("coupling", "policy.coupling_mode"),                 # 双头瓶颈层token路由模式
    ("coupling_block", "policy.coupling_block_type"),     # scalar gate或role adaLN-Zero
    ("coupling_pos", "policy.coupling_use_temporal_pos_emb"),  # 耦合分支是否加入时间位置编码
    ("coupling_ffn", "policy.coupling_use_ffn"),                # 是否启用耦合后的FFN分支
    ("coupling_ffn_ratio", "policy.coupling_ffn_ratio"),        # 耦合FFN隐藏层倍率
    ("coupling_tmax", "policy.coupling_active_max_timestep"),   # 开放耦合残差的最大扩散时间步
    ("v2a_scale", "policy.view_to_arm_coupling_scale"),   # View上下文注入Arm的外部缩放
    ("a2v_scale", "policy.arm_to_view_coupling_scale"),   # Arm上下文注入View的外部缩放
    ("output_corrector", "policy.output_corrector_type"),  # 最终动作修正器类型
    ("output_direction", "policy.output_corrector_direction"),  # 最终动作修正方向
    ("output_dim", "policy.output_corrector_d_model"),     # 动作维度Token隐藏维度
    ("output_heads", "policy.output_corrector_num_heads"), # 单一14x6图共享的Value子空间头数
    ("output_v2a_scale", "policy.view_to_arm_output_scale"), # View最终轨迹修正Arm的scale
    ("output_a2v_scale", "policy.arm_to_view_output_scale"), # Arm最终轨迹修正View的scale
    ("output_limit", "policy.output_corrector_residual_limit"), # 单维归一化动作修正上限
)


def configure_wandb_runtime(cfg: DictConfig) -> None:
    """Apply process-level W&B settings before ``Logger`` initializes a run."""

    wandb_cfg = getattr(cfg, "wandb", {})  # W&B配置节点
    wandb_enabled = bool(getattr(wandb_cfg, "enable", False))  # 是否启用W&B
    disable_system_stats = bool(getattr(wandb_cfg, "disable_system_stats", False))  # 是否禁用系统监控
    disable_machine_info = bool(getattr(wandb_cfg, "disable_machine_info", False))  # 是否禁用机器信息
    if not (wandb_enabled and (disable_system_stats or disable_machine_info)):
        return

    import wandb

    wandb.setup(
        settings=wandb.Settings(
            x_disable_stats=disable_system_stats,
            x_disable_machine_info=disable_machine_info,
        )
    )
    logging.info(
        "W&B系统数据设置: disable_system_stats=%s, disable_machine_info=%s",
        disable_system_stats,
        disable_machine_info,
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


def _sanitize_wandb_tag(value) -> str | None:
    """生成合法的W&B标签；超长时保留更有区分度的末尾部分。"""

    tag = str(value).strip().replace(" ", "_")
    if not tag:
        return None
    if len(tag) <= WANDB_TAG_MAX_LENGTH:
        return tag

    truncation_marker = "..."
    suffix_length = WANDB_TAG_MAX_LENGTH - len(truncation_marker)
    return f"{truncation_marker}{tag[-suffix_length:]}"


def _dataset_name_tags(cfg: DictConfig) -> list[str]:
    """从本地路径或HF repo id中提取数据集名称，避免把完整路径作为tag上传。"""

    dataset_value = OmegaConf.select(cfg, "dataset_local_dir", default=None)
    if dataset_value in (None, "null", "none", ""):
        dataset_value = OmegaConf.select(cfg, "dataset_repo_id", default=None)
    if dataset_value in (None, "null", "none", ""):
        return []

    values = dataset_value if OmegaConf.is_list(dataset_value) else [dataset_value]
    tags = []
    for value in values:
        text = str(value).replace("\\", "/").rstrip("/")
        if not text or text.lower() in {"null", "none"}:
            continue
        dataset_tag = _sanitize_wandb_tag(PurePosixPath(text).name)
        if dataset_tag is not None:
            tags.append(dataset_tag)
    return tags


def add_wandb_parameter_tags(logger: Logger, cfg: DictConfig) -> None:
    """把实际生效的预训练关键配置追加到当前 W&B run 标签。"""

    wandb_module = getattr(logger, "_wandb", None)  # Logger内部持有的wandb模块
    wandb_run = getattr(wandb_module, "run", None)  # 当前W&B run对象
    if wandb_run is None:
        return

    configured_tags = OmegaConf.select(cfg, "wandb.tags", default=[])  # 配置文件中手动指定的tags
    if configured_tags is None:
        configured_tags = []
    elif isinstance(configured_tags, str):
        configured_tags = [configured_tags]
    else:
        configured_tags = list(configured_tags)

    dataset_tags = _dataset_name_tags(cfg)  # 只保留数据集目录名/repo名，不上传完整路径
    parameter_tags = []  # 从关键配置自动生成的参数tags
    for tag_name, config_path in PRETRAIN_WANDB_PARAMETER_TAGS:
        value = OmegaConf.select(cfg, config_path, default=None)  # 当前配置路径对应的实际值
        if value is not None:
            parameter_tags.append(f"{tag_name}:{_format_wandb_tag_value(value)}")

    existing_tags = list(wandb_run.tags or ())  # W&B初始化时已有的tags
    raw_tags = [
        *existing_tags,
        *map(str, configured_tags),
        *dataset_tags,
        *parameter_tags,
    ]
    valid_tags = []
    for raw_tag in raw_tags:
        tag = _sanitize_wandb_tag(raw_tag)
        if tag is not None:
            valid_tags.append(tag)

    # W&B会在任一标签不合法时拒绝整批更新，因此提交前统一清洗并去重。
    wandb_run.tags = tuple(dict.fromkeys(valid_tags))
    logging.info("W&B标签已更新: count=%d, tags=%s", len(wandb_run.tags), list(wandb_run.tags))


def _add_progress_metrics(info: dict, step: int, cfg: DictConfig, dataset) -> tuple[float, float, float]:
    """Append dataset progress metrics to the payload sent to W&B."""

    batch_size = int(cfg.training.batch_size)
    effective_num_samples = OmegaConf.select(
        cfg,
        "training.effective_num_samples",
        default=dataset.num_samples,
    )
    steps_per_epoch = OmegaConf.select(
        cfg,
        "training.steps_per_epoch",
        default=None,
    )
    if steps_per_epoch is None:
        num_samples = (step + 1) * batch_size
        num_epochs = num_samples / dataset.num_samples
    else:
        steps_per_epoch = int(steps_per_epoch)
        completed_epochs, batches_in_epoch = divmod(step + 1, steps_per_epoch)
        drop_last = bool(
            OmegaConf.select(
                cfg,
                "training.resolved_drop_last",
                default=True,
            )
        )
        samples_per_epoch = (
            steps_per_epoch * batch_size
            if drop_last
            else int(effective_num_samples)
        )
        partial_samples = min(
            batches_in_epoch * batch_size,
            samples_per_epoch,
        )
        num_samples = completed_epochs * samples_per_epoch + partial_samples
        num_epochs = completed_epochs + partial_samples / samples_per_epoch

    avg_samples_per_ep = effective_num_samples / dataset.num_episodes  # 每条episode平均有效样本数
    num_episodes = num_samples / avg_samples_per_ep  # 约等于已遍历episode数

    info["step"] = step  # 上传到train/step或eval/step
    info["num_samples"] = num_samples  # 上传累计样本数
    info["num_episodes"] = num_episodes  # 上传累计episode等效数
    info["num_epochs"] = num_epochs  # 上传累计epoch等效数
    return num_samples, num_episodes, num_epochs


def log_train_info(logger: Logger, info: dict, step: int, cfg: DictConfig, dataset) -> None:
    """Log and upload one pretrain optimization step."""

    loss = info["loss"]  # 总训练损失
    grad_norm = info["grad_norm"]  # 梯度裁剪前后的范数记录
    lr = info["lr"]  # 当前学习率
    update_s = info["update_s"]  # 单步模型更新耗时
    dataloading_s = info["dataloading_s"]  # 单步数据加载耗时

    num_samples, num_episodes, num_epochs = _add_progress_metrics(info, step, cfg, dataset)  # 训练进度统计
    log_items = [
        f"step:{format_big_number(step)}",
        f"smpl:{format_big_number(num_samples)}",
        f"ep:{format_big_number(num_episodes)}",
        f"epch:{num_epochs:.2f}",
        f"loss:{loss:.3f}",
    ]
    if "arm_loss" in info:
        log_items.append(f"arm_loss:{info['arm_loss']:.3f}")
    if "view_loss" in info:
        log_items.append(f"view_loss:{info['view_loss']:.3f}")
    if "view_delta_target_abs_mean_rad" in info:
        log_items.append(
            "view_delta:"
            f"{info['view_delta_target_abs_mean_rad']:.4f}rad"
        )
    if "eef_position_error_m" in info:
        log_items.append(f"eef_pos:{info['eef_position_error_m']:.4f}m")
    if "eef_rotation_error_rad" in info:
        log_items.append(f"eef_rot:{info['eef_rotation_error_rad']:.4f}rad")
    log_items.extend(
        [
            f"grdn:{grad_norm:.3f}",
            f"lr:{lr:0.1e}",
            f"updt_s:{update_s:.3f}",
            f"data_s:{dataloading_s:.3f}",
        ]
    )
    logging.info(" ".join(log_items))
    logger.log_dict(info, step, mode="train")


def log_eval_info(logger: Logger, info: dict, step: int, cfg: DictConfig, dataset) -> None:
    """Log and upload legacy synchronous evaluation metrics."""

    eval_s = info["eval_s"]  # 本轮评估耗时
    avg_sum_reward = info["avg_sum_reward"]  # 平均累计奖励
    pc_success = info["pc_success"]  # 成功率百分比

    num_samples, num_episodes, num_epochs = _add_progress_metrics(info, step, cfg, dataset)  # 评估对应训练进度
    logging.info(
        " ".join(
            [
                f"step:{format_big_number(step)}",
                f"smpl:{format_big_number(num_samples)}",
                f"ep:{format_big_number(num_episodes)}",
                f"epch:{num_epochs:.2f}",
                f"∑rwrd:{avg_sum_reward:.3f}",
                f"success:{pc_success:.1f}%",
                f"eval_s:{eval_s:.3f}",
            ]
        )
    )
    logger.log_dict(info, step, mode="eval")
