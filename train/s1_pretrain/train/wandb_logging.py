"""W&B-facing logging helpers for pretrain runs."""

from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf

from lerobot.common.logger import Logger
from lerobot.common.utils.utils import format_big_number


PRETRAIN_WANDB_PARAMETER_TAGS = (
    ("device", "device"),                                 # 训练设备
    ("batch", "training.batch_size"),                     # 训练批大小
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
    ("ema_start", "policy.ema_update_after_step"),        # EMA开始更新的训练步
    ("arm_dim", "policy.arm_action_dim"),                 # 双臂动作维度
    ("view_dim", "policy.view_action_dim"),               # 视角动作维度
    ("view_weight", "policy.view_loss_weight"),           # 视角损失权重
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
    ("scid_ridge", "policy.scid_ridge"),                  # SCID闭式Arm->View映射的岭正则
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

    parameter_tags = []  # 从关键配置自动生成的参数tags
    for tag_name, config_path in PRETRAIN_WANDB_PARAMETER_TAGS:
        value = OmegaConf.select(cfg, config_path, default=None)  # 当前配置路径对应的实际值
        if value is not None:
            parameter_tags.append(f"{tag_name}:{_format_wandb_tag_value(value)}")

    existing_tags = list(wandb_run.tags or ())  # W&B初始化时已有的tags
    wandb_run.tags = tuple(
        dict.fromkeys([*existing_tags, *map(str, configured_tags), *parameter_tags])
    )


def _add_progress_metrics(info: dict, step: int, cfg: DictConfig, dataset) -> tuple[float, float, float]:
    """Append dataset progress metrics to the payload sent to W&B."""

    num_samples = (step + 1) * cfg.training.batch_size  # 已训练样本数
    avg_samples_per_ep = dataset.num_samples / dataset.num_episodes  # 每条episode平均样本数
    num_episodes = num_samples / avg_samples_per_ep  # 约等于已遍历episode数
    num_epochs = num_samples / dataset.num_samples  # 约等于已遍历数据集轮数

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
