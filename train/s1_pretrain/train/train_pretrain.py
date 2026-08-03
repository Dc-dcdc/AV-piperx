#!/usr/bin/env python
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 强行指向国内镜像站
import copy
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import gymnasium as gym
import env
import logging
import time
from contextlib import nullcontext
from pprint import pformat

from train.s1_pretrain.eval.async_eval import (
    AsyncEvalController,
    finalize_async_eval_result,
)
from train.s1_pretrain.eval.eval_train import (
    TopKCheckpointManager,
    evaluate_and_checkpoint_if_needed,
    make_checkpoint_identifier,
    make_eval_env,
)

import hydra
import datasets
import torch
import torchvision.transforms as v2
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

# ==========================================
# 🌟 采用官方最新极简 API，抛弃 factory.py
# ==========================================
from lerobot.common.datasets.factory import apply_dataset_stats_overrides
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.transforms import get_image_transforms
from lerobot.common.datasets.utils import hf_transform_to_torch, unflatten_dict
from lerobot.common.datasets.view_delta_stats import (
    load_or_compute_view_delta_stats,
)
from lerobot.common.datasets.video_utils import VideoFrame  # noqa: F401  注册本地 parquet 中的 VideoFrame 字段
# 复用 LeRobot 的其他核心组件
from lerobot.common.logger import Logger
from lerobot.common.policies.factory import make_policy # 用于获取训练策略模型
from lerobot.common.policies.diffusion.view_action_representation import (
    VIEW_ACTION_DELTA_FROM_CURRENT,
    VIEW_ACTION_DELTA_STATS_KEY,
    resolve_dual_head_action_dims,
)
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from train.s1_pretrain.train.optimizer_utils import (
    is_visual_backbone_parameter,
    partition_optimizer_parameters,
)
from train.s1_pretrain.train.ema import PolicyEMA
from train.s1_pretrain.train.wandb_logging import (
    add_wandb_parameter_tags,
    configure_wandb_runtime,
    log_train_info,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
    set_global_seed,
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


def resolve_amp_dtype(cfg: DictConfig) -> torch.dtype:
    """解析训练用 autocast dtype。默认沿用当前配置的 bfloat16。"""
    dtype_name = str(getattr(cfg, "amp_dtype", "bfloat16")).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"不支持的 amp_dtype={dtype_name!r}，可选 bfloat16 或 float16。")


def resolve_grad_scaler_enabled(cfg: DictConfig, device: torch.device, amp_dtype: torch.dtype) -> bool:
    """GradScaler 默认只在 CUDA fp16 AMP 下启用，必要时可用 use_grad_scaler 覆盖。"""
    mode = str(getattr(cfg, "use_grad_scaler", "auto")).lower()
    if mode == "auto":
        return bool(cfg.use_amp and device.type == "cuda" and amp_dtype == torch.float16)
    if mode in {"true", "1", "yes", "on"}:
        return bool(cfg.use_amp and device.type == "cuda")
    if mode in {"false", "0", "no", "off"}:
        return False
    raise ValueError("use_grad_scaler 只能是 auto/true/false。")


def should_run_periodic_step(step: int, total_steps: int, freq: int | None) -> bool:
    """训练最后一步也执行周期任务，避免尾部 checkpoint/eval 被漏掉。"""
    is_last_step = step == total_steps - 1
    if is_last_step:
        return True
    return bool(freq and freq > 0 and step > 0 and step % freq == 0)


def should_checkpoint_or_eval(step: int, cfg: DictConfig) -> bool:
    """提前判断本 step 是否真的需要进入较重的评估/保存逻辑。"""
    total_steps = int(cfg.training.offline_steps)
    should_eval = should_run_periodic_step(
        step,
        total_steps,
        int(getattr(cfg.training, "eval_freq", 0)),
    )
    save_freq = int(getattr(cfg.training, "save_freq", 10000))
    should_checkpoint = bool(getattr(cfg.training, "save_checkpoint", False)) and (
        should_run_periodic_step(step, total_steps, save_freq)
    )
    return should_eval or should_checkpoint


def process_async_eval_results(
    evaluator: AsyncEvalController,
    *,
    logger,
    cfg: DictConfig,
    manager: TopKCheckpointManager,
    logging_step: int,
) -> int:
    """非阻塞收集已完成评估；所有日志和Top-K文件操作只在主进程执行。"""
    results = evaluator.poll()
    for result in results:
        try:
            finalize_async_eval_result(
                result,
                logger=logger,
                cfg=cfg,
                manager=manager,
                logging_step=logging_step,
            )
        finally:
            evaluator.cleanup_result_snapshot(result)
    return len(results)


def wait_for_async_eval_capacity(
    evaluator: AsyncEvalController,
    *,
    logger,
    cfg: DictConfig,
    manager: TopKCheckpointManager,
    logging_step: int,
):
    """等待一个队列槽位，同时持续消费结果，主要用于最终step或禁止跳过时。"""
    while not evaluator.has_capacity:
        completed = process_async_eval_results(
            evaluator,
            logger=logger,
            cfg=cfg,
            manager=manager,
            logging_step=logging_step,
        )
        if completed == 0:
            time.sleep(0.1)


def make_train_dataloader_kwargs(
    cfg: DictConfig,
    device: torch.device,
    shuffle: bool,
    sampler,
) -> dict:
    """构建 DataLoader 参数；多 worker 时保持 worker 常驻并允许配置预取深度。"""
    num_workers = int(cfg.training.num_workers)
    dataloader_kwargs = {
        "num_workers": num_workers,
        "batch_size": int(cfg.training.batch_size),
        "shuffle": shuffle,
        "sampler": sampler,
        "pin_memory": (device.type != "cpu"),
        "drop_last": True,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(
            cfg.training.get("persistent_workers", True)
        )
        prefetch_factor = cfg.training.get("prefetch_factor", None)
        if prefetch_factor is not None:
            prefetch_factor = int(prefetch_factor)
            if prefetch_factor > 0:
                dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return dataloader_kwargs


def tensor_to_float(value) -> float:
    """只在日志/保存需要数值时把 Tensor 同步到 Python float。"""
    if isinstance(value, torch.Tensor):
        return float(value.detach())
    return float(value)


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
        "coupled_dual_head_diffusion",
        "two_model_diffusion",
    ]:
        # 分离主网络、零门控耦合分支和视觉Backbone。耦合分支在gate=0时
        # 暂时没有任务梯度，因此关闭其weight decay，避免Attention/AdaLN/
        # FFN/时间编码器在gate打开前被额外压缩。
        trainable_named_parameters = [
            (name, parameter)
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        ]
        main_parameters, coupling_parameters, backbone_parameters = (
            partition_optimizer_parameters(
                trainable_named_parameters,
                is_backbone_parameter=is_visual_backbone_parameter,
            )
        )
        optimizer_params_dicts = [
            {
                "name": "main",
                "params": main_parameters,
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
        optimizer_params_dicts.append(
            {
                "name": "backbone",
                "params": backbone_parameters,
                # 视觉网络给予 10 倍小的学习率，保护预训练特征
                "lr": getattr(cfg.training, "lr_backbone", 1e-5),
            }
        )
        optimizer_class = (
            torch.optim.AdamW
            if cfg.policy.name == "coupled_dual_head_diffusion"
            else torch.optim.Adam
        )
        logging.info(
            "优化器=%s; 参数组: main=%d tensors (weight_decay=%g), "
            "coupling=%d tensors (weight_decay=0), "
            "backbone=%d tensors (lr=%g, weight_decay=%g)",
            optimizer_class.__name__,
            len(main_parameters),
            float(cfg.training.weight_decay),
            len(coupling_parameters),
            len(backbone_parameters),
            float(getattr(cfg.training, "lr_backbone", 1e-5)),
            float(cfg.training.weight_decay),
        )

        # 耦合双头使用AdamW解耦梯度更新与weight decay；其余策略保持原优化器。
        optimizer = optimizer_class(
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
    amp_dtype: torch.dtype = torch.bfloat16,
    collect_metrics: bool = True,
    lock=None,
    ema: PolicyEMA | None = None,
    step: int | None = None,
):
    """进行一次训练更新，计算损失，反向传播，更新模型参数，并返回一个包含训练信息的字典."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    # ==========================================
    # 1. 向前传播计算loss
    # ==========================================
    with torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext(): # 如果 use_amp = True，它会开启 torch.autocast，意味着接下来的计算会自动在 Float32 和 Float16 之间切换，省显存且加速
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
                policy.update()  # 更新策略自带的非梯度训练状态（若实现）。
        if ema is not None:
            if step is None:
                raise ValueError("启用EMA时必须向update_policy传入当前step。")
            ema.update(policy, step)
    else:
        print("Warning: Gradient overflow, skipping LR and EMA update.")

    if not collect_metrics:
        return {}
    
    info = {
        "loss": tensor_to_float(loss),
        "grad_norm": tensor_to_float(grad_norm),
        "lr": optimizer.param_groups[0]["lr"],
        "update_s": time.perf_counter() - start_time,
    }
    if ema is not None:
        info["ema_ready"] = int(ema.ready)
        info["ema_updates"] = int(ema.num_updates)
    # 遍历 output_dict，安全地提取数据
    for k, v in output_dict.items():
        if k == "loss":
            continue
        if isinstance(v, torch.Tensor):
            # 如果是标量（单值 Tensor），直接取 .item() 转为普通的 Python float
            if v.numel() == 1:
                info[k] = tensor_to_float(v)
            # 如果是多维张量，必须把它从计算图剥离并转移到 CPU 内存
            else:
                info[k] = v.detach().cpu()
        else:
            info[k] = v

    return info

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


# 严格断点续训时，快照配置是训练语义的基线。这里只允许当前运行覆盖与
# 运行资源、训练时长/记录频率和评估吞吐相关的字段。
RESUME_CURRENT_CONFIG_PATHS = (
    "device",                         # 当前训练使用的计算设备，如 cuda:0
    "use_amp",                        # 是否启用自动混合精度训练
    "training.offline_steps",         # 本次训练计划达到的总离线训练步数
    "training.eval_freq",             # 每隔多少训练步执行一次评估
    "training.save_freq",             # 每隔多少训练步保存一次快照
    "training.log_freq",              # 每隔多少训练步记录一次训练日志
    "training.num_workers",           # DataLoader 并行加载数据的进程数
    "training.persistent_workers",    # 是否让 DataLoader worker 跨轮次常驻
    "training.prefetch_factor",       # 每个 DataLoader worker 预取的 batch 数
    "eval.device",                    # 模型评估使用的计算设备
    "eval.batch_size",                # 并行评估时同时运行的环境数量
    "eval.n_episodes",                # 每次评估执行的 episode 总数
    "eval.max_episodes_rendered",     # 每次评估最多保存录像的 episode 数
    "eval.use_async_envs",            # 是否使用异步向量环境进行并行评估
    "wandb",                          # 当前实验的完整 W&B 日志配置
)

# 这些字段负责发起本次恢复，不能被快照中保存的旧 resume 状态覆盖。
RESUME_CONTROL_CONFIG_PATHS = (
    "resume",
    "resume_path",
)


def get_resume_checkpoint_dir(resume_path: str | Path) -> Path:
    """将 checkpoint 或 pretrained_model 路径统一转换为 checkpoint 目录。"""
    checkpoint_dir = Path(resume_path).expanduser()
    if checkpoint_dir.name == "pretrained_model":
        checkpoint_dir = checkpoint_dir.parent
    return checkpoint_dir


def get_resume_run_dir(resume_path: str | Path) -> Path | None:
    """从 checkpoint 路径向上查找包含 .hydra 的原实验目录。"""
    checkpoint_dir = get_resume_checkpoint_dir(resume_path)
    for candidate in (checkpoint_dir, *checkpoint_dir.parents):
        if (candidate / ".hydra").is_dir():
            return candidate
    if checkpoint_dir.parent.name == "checkpoints":
        return checkpoint_dir.parent.parent
    return None


def load_resume_snapshot_config(resume_path: str | Path) -> tuple[DictConfig, Path]:
    """读取 checkpoint 中 Logger 保存的完整 Hydra 配置。"""
    checkpoint_dir = get_resume_checkpoint_dir(resume_path)
    snapshot_config_path = checkpoint_dir / "pretrained_model" / "config.yaml"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"断点续训目录不存在: {checkpoint_dir}")
    if not snapshot_config_path.is_file():
        raise FileNotFoundError(
            "断点续训必须读取快照配置，但未找到: "
            f"{snapshot_config_path}"
        )

    snapshot_cfg = OmegaConf.load(snapshot_config_path)
    if not isinstance(snapshot_cfg, DictConfig):
        raise TypeError(f"快照配置不是 DictConfig: {snapshot_config_path}")
    return snapshot_cfg, snapshot_config_path


def _copy_config_value(value):
    """复制 OmegaConf 节点，避免把当前配置节点直接挂到合并配置中。"""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    return copy.deepcopy(value)


def build_resume_config(cfg: DictConfig) -> tuple[DictConfig, Path | None]:
    """
    构造实际训练配置。

    resume=false 时原样返回当前 Hydra 配置；resume=true 时以当前配置补齐
    新版本缺省字段，再让快照覆盖训练语义，最后恢复运行字段白名单。
    """
    if not bool(OmegaConf.select(cfg, "resume", default=False)):
        return cfg, None

    resume_path = clean_optional_path(OmegaConf.select(cfg, "resume_path", default=None))
    if resume_path is None:
        raise ValueError("resume=true 时必须提供有效的 resume_path。")

    snapshot_cfg, snapshot_config_path = load_resume_snapshot_config(resume_path)

    # 当前配置放在前面，只负责给旧快照补充后来新增且快照中不存在的字段；
    # 对于双方都存在的字段，快照配置优先。先转成普通的非struct配置，
    # 否则从普通dual-head恢复coupled策略时，coupling_*新增键会被拒绝。
    current_cfg_copy = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=False)
    )
    snapshot_cfg_copy = OmegaConf.create(
        OmegaConf.to_container(snapshot_cfg, resolve=False)
    )
    # 旧快照缺少该字段时，其真实语义必然是绝对View动作。必须在merge前
    # 显式补为absolute，不能让当前配置中的delta值悄悄改变旧权重语义。
    if (
        OmegaConf.select(
            snapshot_cfg_copy,
            "policy.view_action_representation",
            default=None,
        )
        is None
    ):
        OmegaConf.update(
            snapshot_cfg_copy,
            "policy.view_action_representation",
            "absolute",
            merge=False,
            force_add=True,
        )
    effective_cfg = OmegaConf.merge(current_cfg_copy, snapshot_cfg_copy)
    for config_path in (
        *RESUME_CONTROL_CONFIG_PATHS,
        *RESUME_CURRENT_CONFIG_PATHS,
    ):
        current_value = OmegaConf.select(cfg, config_path, default=None)
        OmegaConf.update(
            effective_cfg,
            config_path,
            _copy_config_value(current_value),
            merge=False,
            force_add=True,
        )

    # s1主预训练已移除init_policy_path；旧s1快照中的遗留字段不能重新混入。
    # s2增量入口仍复用本函数并显式定义该字段，因此只为这些调用者保留它。
    if "init_policy_path" in cfg:
        OmegaConf.update(
            effective_cfg,
            "init_policy_path",
            _copy_config_value(cfg.init_policy_path),
            merge=False,
            force_add=True,
        )
    elif "init_policy_path" in effective_cfg:
        del effective_cfg["init_policy_path"]

    return effective_cfg, snapshot_config_path


RESUME_CONFIG_GROUP_IDENTITY_PATHS = {
    "env": (
        "env.name",
        "env.task",
        "env.state_dim",
        "env.action_dim",
    ),
    "policy": ("policy.name",),
}


def _flatten_config_leaves(value, prefix: str = "") -> dict[str, object]:
    """把配置展开成叶子路径，用于从完整快照反查最匹配的配置组文件。"""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=False)

    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config_leaves(child, child_prefix))
        return flattened
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def _matching_snapshot_group_candidates(
    snapshot_cfg: DictConfig,
    group_name: str,
) -> list[tuple[str, int]]:
    """返回身份字段一致的配置组候选及其与快照叶子字段的匹配分数。"""
    group_dir = ROOT_DIR / "configs" / "pretrain" / group_name
    identity_paths = RESUME_CONFIG_GROUP_IDENTITY_PATHS[group_name]
    snapshot_leaves = _flatten_config_leaves(snapshot_cfg)
    candidates = []

    for config_path in sorted(group_dir.glob("*.yaml")):
        candidate_cfg = OmegaConf.load(config_path)
        identity_matches = all(
            OmegaConf.select(candidate_cfg, path, default=None)
            == OmegaConf.select(snapshot_cfg, path, default=None)
            for path in identity_paths
        )
        if not identity_matches:
            continue

        candidate_leaves = _flatten_config_leaves(candidate_cfg)
        match_score = sum(
            snapshot_leaves.get(path, object()) == value
            for path, value in candidate_leaves.items()
        )
        candidates.append((config_path.stem, match_score))

    return candidates


def infer_resume_hydra_choice(
    resume_path: str | Path,
    snapshot_cfg: DictConfig,
    group_name: str,
    recorded_choice: str | None,
) -> str | None:
    """
    从checkpoint内容校验或反推配置组。

    `.hydra/hydra.yaml` 会在重用原输出目录时被后续启动覆盖，因此它只能
    作为候选；checkpoint内的完整config.yaml才是最终权威来源。
    """
    candidates_with_scores = _matching_snapshot_group_candidates(
        snapshot_cfg,
        group_name,
    )
    candidate_names = [name for name, _ in candidates_with_scores]

    # 原Hydra记录只有在身份字段仍与checkpoint一致时才可信。
    if recorded_choice in candidate_names:
        return recorded_choice
    if len(candidate_names) == 1:
        return candidate_names[0]
    if not candidate_names:
        return None

    # 多个配置文件可能使用同一种policy.name；优先参考原实验目录名称。
    run_dir = get_resume_run_dir(resume_path)
    if run_dir is not None:
        path_matches = [
            name for name in candidate_names if run_dir.name.endswith(f"_{name}")
        ]
        if len(path_matches) == 1:
            return path_matches[0]

    # 最后选择与完整快照静态字段匹配数最高且没有并列的配置文件。
    best_score = max(score for _, score in candidates_with_scores)
    best_matches = [
        name for name, score in candidates_with_scores if score == best_score
    ]
    if len(best_matches) == 1:
        return best_matches[0]
    return None


def load_resume_hydra_choices(resume_path: str | Path) -> dict[str, str]:
    """以checkpoint配置为权威来源，恢复env/policy Hydra配置组名称。"""
    snapshot_cfg, _ = load_resume_snapshot_config(resume_path)
    run_dir = get_resume_run_dir(resume_path)
    recorded_choices = {}
    if run_dir is not None:
        hydra_config_path = run_dir / ".hydra" / "hydra.yaml"
        if hydra_config_path.is_file():
            hydra_cfg = OmegaConf.load(hydra_config_path)
            for group_name in ("env", "policy"):
                value = OmegaConf.select(
                    hydra_cfg,
                    f"hydra.runtime.choices.{group_name}",
                    default=None,
                )
                if value is not None:
                    recorded_choices[group_name] = str(value)

    restored_choices = {}
    for group_name in ("env", "policy"):
        inferred_choice = infer_resume_hydra_choice(
            resume_path,
            snapshot_cfg,
            group_name,
            recorded_choices.get(group_name),
        )
        if inferred_choice is not None:
            restored_choices[group_name] = inferred_choice
    return restored_choices


def get_cli_override_value(args: list[str] | tuple[str, ...], key: str) -> str | None:
    """读取一个精确匹配的 Hydra 命令行覆盖值。"""
    for arg in reversed(args):
        if "=" not in arg:
            continue
        arg_key, value = arg.split("=", 1)
        if arg_key.lstrip("+") == key:
            return value.strip().strip("'\"")
    return None


def replace_cli_override(args: list[str], key: str, value: str) -> None:
    """替换同名 Hydra 覆盖，避免默认参数与恢复参数重复。"""
    args[:] = [
        arg
        for arg in args
        if "=" not in arg or arg.split("=", 1)[0].lstrip("+") != key
    ]
    args.append(f"{key}={value}")


def restore_resume_hydra_choices(
    args: list[str],
    user_cli_args: tuple[str, ...],
    resume_path: str | Path,
) -> dict[str, str]:
    """
    在 Hydra 组合配置前恢复原实验的 env/policy 配置组。

    用户若显式请求不同配置组则拒绝严格续训，避免模型结构或环境语义被
    悄悄改变；需要切换配置时应使用resume=false启动独立实验。
    """
    snapshot_choices = load_resume_hydra_choices(resume_path)
    for group_name, snapshot_choice in snapshot_choices.items():
        explicit_choice = get_cli_override_value(user_cli_args, group_name)
        if explicit_choice is not None and explicit_choice != snapshot_choice:
            raise ValueError(
                f"严格断点续训不允许修改 {group_name} 配置组: "
                f"快照={snapshot_choice!r}, 当前命令行={explicit_choice!r}。"
                "如需切换策略或环境，请使用resume=false启动独立实验。"
            )
        replace_cli_override(args, group_name, snapshot_choice)
    return snapshot_choices


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
    hf_dataset.set_transform(hf_transform_to_torch)

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
        delta_timestamps=delta_timestamps,
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



def train_dppo_pretrain(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    """
    DPPO 第一阶段：基于专家数据的 Diffusion 策略预训练 (Offline Behavior Cloning)
    结合了 Hydra 配置管理与 LeRobot 最新极简数据加载 API。
    """
    
    init_logging() #初始化日志
    logging.info(pformat(OmegaConf.to_container(cfg))) #打印配置cfg

    # 初始化日志记录器与设备
    configure_wandb_runtime(cfg)
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)
    add_wandb_parameter_tags(logger, cfg)
    set_global_seed(cfg.seed)
    device = get_safe_torch_device(cfg.device, log=True)

    # 开启 CuDNN 加速和 TF32 支持
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    amp_dtype = resolve_amp_dtype(cfg)
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
    dataset_cache_dir = clean_optional_path(cfg.get("dataset_cache_dir", None))
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
    apply_dataset_stats_overrides(
        offline_dataset,
        cfg.get("override_dataset_stats"),
    )
    # # 使用官方函数解析并挂载到 cfg，这样 make_dataset 内部才能正确读取
    # resolve_delta_timestamps(cfg)
    
    # # 必须使用 make_dataset 以激活 ACT 依赖的 image_transforms
    # offline_dataset = make_dataset(cfg)
    # ==========================================
    # 🌟 2. 严格断点续训：resume=false时始终按当前policy/env全新训练
    # ==========================================
    start_step = 0
    policy_load_path = None
    ema_policy_load_path = None
    training_state_file = None
    configured_use_ema = bool(getattr(cfg.policy, "use_ema", False))

    if cfg.resume:
        # 将配置获取的内容强制转为字符串并小写，防范 "none", "null", NoneType 导致崩溃
        raw_path = str(getattr(cfg, "resume_path", "")).strip().lower()
        
        if raw_path in ["", "none", "null"]:
            raise ValueError("resume=true时必须提供有效的resume_path。")
        else:
            chkpt_dir = get_resume_checkpoint_dir(getattr(cfg, "resume_path"))
            if not chkpt_dir.exists():
                raise FileNotFoundError(f"严格续训路径不存在: {chkpt_dir}")
            else:
                logging.info(f"🎯 成功检测到有效的恢复路径: {chkpt_dir}")
                training_state_file = chkpt_dir / "training_state.pth"
                if configured_use_ema:
                    online_path = chkpt_dir / "online_pretrained_model"
                    ema_path = chkpt_dir / "pretrained_model"
                    if not online_path.is_dir() or not ema_path.is_dir():
                        raise FileNotFoundError(
                            "严格EMA续训需要checkpoint同时包含"
                            "online_pretrained_model和pretrained_model。"
                            "旧checkpoint没有真实EMA历史，不能伪造为完整续训。"
                        )
                    policy_load_path = online_path
                    ema_policy_load_path = ema_path
                else:
                    policy_load_path = chkpt_dir / "pretrained_model"

    # ==========================================
    # 🌟 3. 初始化模型与优化器 (顺序极其重要！)
    # ==========================================
    logging.info("🧠 正在初始化 Diffusion Policy...")

    # 新训练的View相对动作模式需要与实际action horizon一致的派生统计量。
    # resume时normalizer统计已保存在checkpoint中，不应重新扫描或覆盖。
    view_action_representation = str(
        getattr(cfg.policy, "view_action_representation", "absolute")
    )
    if (
        policy_load_path is None
        and view_action_representation == VIEW_ACTION_DELTA_FROM_CURRENT
    ):
        if str(cfg.policy.name) not in {
            "diffusion",
            "dual_head_diffusion",
            "coupled_dual_head_diffusion",
        }:
            raise ValueError(
                "delta_from_current当前仅支持diffusion、"
                "dual_head_diffusion和coupled_dual_head_diffusion，"
                f"当前policy.name={cfg.policy.name!r}。"
            )
        state_delta_timestamps = resolved_delta_timestamps.get(
            "observation.state",
            [],
        )
        if (
            not state_delta_timestamps
            or abs(float(state_delta_timestamps[-1])) > 1e-8
        ):
            raise ValueError(
                "delta_from_current要求observation.state时间轴的最后一项为0，"
                "以保证batch[:, -1]就是当前真实View关节；当前时间轴为"
                f"{state_delta_timestamps}。"
            )
        arm_action_dim, view_action_dim = resolve_dual_head_action_dims(
            cfg.policy
        )
        configured_cache_dir = clean_optional_path(
            cfg.get("view_delta_stats_cache_dir", None)
        )
        if configured_cache_dir is None:
            configured_cache_dir = "outputs/buffer/view_action_delta_stats"
        cache_dir = Path(configured_cache_dir).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = ROOT_DIR / cache_dir
        view_delta_stats, stats_cache_path, stats_metadata = (
            load_or_compute_view_delta_stats(
                offline_dataset,
                action_delta_timestamps=resolved_delta_timestamps["action"],
                arm_action_dim=arm_action_dim,
                view_action_dim=view_action_dim,
                include_padding=not bool(
                    getattr(cfg.policy, "do_mask_loss_for_padding", False)
                ),
                cache_dir=cache_dir,
            )
        )
        offline_dataset.stats[VIEW_ACTION_DELTA_STATS_KEY] = view_delta_stats
        logging.info(
            "View动作使用当前锚点增量表示: arm_dim=%d, view_dim=%d, "
            "stats_cache=%s, targets=%d, padding_fraction=%.4f, "
            "delta_min=%s, delta_max=%s",
            arm_action_dim,
            view_action_dim,
            stats_cache_path,
            int(stats_metadata.get("valid_or_included_count", 0)),
            float(stats_metadata.get("padding_fraction", 0.0)),
            view_delta_stats["min"].tolist(),
            view_delta_stats["max"].tolist(),
        )

    # 3.1 resume加载online权重；resume=false使用当前policy/env随机初始化。
    policy = make_policy(
        hydra_cfg=cfg,
        dataset_stats=offline_dataset.stats if policy_load_path is None else None,
        pretrained_policy_name_or_path=(
            str(policy_load_path) if policy_load_path is not None else None
        ),
    )
    policy.to(device)
    ema = PolicyEMA.from_policy_config(policy)
    if ema is not None:
        logging.info(
            "EMA已启用: decay=%g, update_after_step=%d；"
            "在线模型用于反向传播，EMA模型用于评估/部署。",
            ema.decay,
            ema.update_after_step,
        )

    # 3.2 无论是不是 resume，都必须先根据模型初始化出全新的优化器！
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # bfloat16 拥有接近 fp32 的指数范围，训练时不需要 GradScaler；fp16 才启用。
    grad_scaler = GradScaler(
        enabled=resolve_grad_scaler_enabled(cfg, device, amp_dtype)
    )
    logging.info(
        "AMP: enabled=%s, dtype=%s, grad_scaler=%s",
        bool(cfg.use_amp),
        str(amp_dtype).replace("torch.", ""),
        grad_scaler.is_enabled(),
    )

    # ==========================================
    # 🌟 4. 恢复优化器与步数状态
    # ==========================================
    if cfg.resume and training_state_file and training_state_file.exists():
        import json
        logging.info("🔄 正在恢复优化器与训练步数...")
        
        try:
            # 1. 一次性读取整个综合大字典，全部丢到内存(CPU)里准备分发
            checkpoint_dict = torch.load(training_state_file, map_location="cpu", weights_only=False)

            # 2. 恢复 Optimizer
            if "optimizer" in checkpoint_dict:
                optimizer.load_state_dict(checkpoint_dict["optimizer"])
                logging.info("✅ Optimizer (优化器) 状态已恢复")
            else:
                logging.warning("⚠️ 存档中未找到 optimizer 状态，动量将重置！")

            # 3. 恢复 LR Scheduler 
            # 兼容不同库的命名习惯（有时叫 lr_scheduler，有时叫 scheduler）
            if lr_scheduler is not None:
                if "lr_scheduler" in checkpoint_dict:
                    lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                    logging.info("✅ LR Scheduler (调度器) 状态已恢复")
                elif "scheduler" in checkpoint_dict:
                    lr_scheduler.load_state_dict(checkpoint_dict["scheduler"])
                    logging.info("✅ LR Scheduler (调度器) 状态已恢复")
                else:
                    logging.warning("⚠️ 存档中未找到 lr_scheduler 状态，学习率将重置！")
            # 4. 恢复 GradScaler (只有 fp16 混合精度时才需要)
            if grad_scaler.is_enabled() and "grad_scaler" in checkpoint_dict:
                grad_scaler.load_state_dict(checkpoint_dict["grad_scaler"])
                logging.info("✅ GradScaler 状态已恢复")
            elif grad_scaler.is_enabled():
                logging.warning("⚠️ 存档中未找到 grad_scaler 状态，梯度缩放因子将重置！")
            else:
                logging.info("当前 AMP dtype 不需要 GradScaler，跳过 scaler 状态恢复。")

            # 5. 严格恢复EMA权重与更新进度。
            if ema is not None:
                if ema_policy_load_path is None:
                    raise FileNotFoundError("EMA续训缺少pretrained_model权重目录。")
                ema_loaded_policy = make_policy(
                    hydra_cfg=cfg,
                    dataset_stats=None,
                    pretrained_policy_name_or_path=str(ema_policy_load_path),
                    strict_pretrained_loading=True,
                )
                ema.restore(
                    ema_loaded_policy,
                    checkpoint_dict.get("ema"),
                )
                del ema_loaded_policy
                logging.info(
                    "✅ EMA状态已恢复: ready=%s, updates=%d, last_step=%d",
                    ema.ready,
                    ema.num_updates,
                    ema.last_step,
                )
            elif "ema" in checkpoint_dict:
                raise ValueError(
                    "checkpoint包含EMA状态，但当前配置use_ema=false；"
                    "严格续训不允许切换EMA语义。"
                )

            # 6. 恢复 Step 步数
            if "step" in checkpoint_dict:
                start_step = checkpoint_dict["step"] + 1
                logging.info(f"⏭️ 从字典成功读取，训练将从 step {start_step} 无缝继续...")
            else:
                # 容错：如果字典里真没存步数，就退化为看文件夹的名字 (比如 000500)
                # 提取下划线前的纯数字部分再转换
                start_step = int(chkpt_dir.name.split('_')[0]) + 1
                logging.info(f"⏭️ 字典中未记录步数，从目录名推断，训练将从 step {start_step} 无缝继续...")

        except Exception as e:
            raise RuntimeError(
                f"严格续训恢复{training_state_file.name}失败，已停止训练，"
                "不会使用旧step搭配新优化器或伪造EMA状态。"
            ) from e
    elif cfg.resume:
        raise FileNotFoundError(
            f"严格续训找不到状态文件 {training_state_file}；"
            "resume=false只会按当前policy/env启动全新训练。"
        )


    # ==========================================
    # 🌟 5. 构建标准的高并发数据加载器 (彻底解耦)
    # ==========================================
    # 如果配置中指定了丢弃最后n帧数据，就使用EpisodeAwareSampler采样器，并且不进行shuffle，这样可以确保在每个训练周期内，模型不会看到每个episode的最后n帧数据，
    # 这对于某些任务可能有帮助，比如那些episode的最后几帧可能包含一些特殊的状态或者奖励信号，丢弃它们可以让模型更好地学习到一般性的行为模式。
    if cfg.training.get("drop_n_last_frames"): 
        shuffle = False
        sampler = EpisodeAwareSampler( 
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.training.get("drop_n_last_frames"),
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader_kwargs = make_train_dataloader_kwargs(cfg, device, shuffle, sampler)
    dataloader = DataLoader(offline_dataset, **dataloader_kwargs)
    logging.info(
        "DataLoader: workers=%s, persistent_workers=%s, prefetch_factor=%s, pin_memory=%s",
        dataloader_kwargs["num_workers"],
        dataloader_kwargs.get("persistent_workers", False),
        dataloader_kwargs.get("prefetch_factor", "default"),
        dataloader_kwargs["pin_memory"],
    )
    
    # 使用 Python 内置的 cycle 将其变为无限迭代器，使用 next(dl_iter) 进行取出一个batch的数据
    # dl_iter = cycle(dataloader) #会存有历史数据，导致显存溢出
    dl_iter = iter(get_infinite_dataloader(dataloader)) 
    log_trainable_parameter_counts(policy)
    logging.info(f"预训练目标步数: {cfg.training.offline_steps}")

    # ==========================================
    # 🌟 5. 动态拼接环境 ID 并创建环境
    # ==========================================
    # 观测要用的相机列表  =  模型推理要用的相机列表 + 评估时保存的video视角相机
    all_obs_keys = policy.config.input_shapes.keys()
    ref_cams = [k.replace("observation.images.", "") for k in all_obs_keys if "observation.images." in k]
    if not ref_cams:
        raise ValueError(f"❌ 严重冲突：模型中未找到相机相关参数。请检查模型输入是否正确。")
    configured_render_cameras = cfg.eval.render_camera
    if isinstance(configured_render_cameras, str):
        configured_render_cameras = [configured_render_cameras]
    obs_cameras = list(dict.fromkeys(ref_cams + list(configured_render_cameras)))

    # 读取 YAML 中的 name ("guided_vision") 和 task ("InsertCylinder-3Arms-v0")
    # 拼接出 "guided_vision/InsertCylinder-3Arms-v0"
    env_id = f"{cfg.env.name}/{cfg.env.task}" 
    
    async_eval_enabled = bool(getattr(cfg.eval, "async_enabled", False))
    async_evaluator = None
    eval_env = None
    if async_eval_enabled:
        eval_device = str(getattr(cfg.eval, "device", cfg.device))
        if eval_device == str(cfg.device):
            logging.warning(
                "异步训练和评估使用同一设备%s，将竞争GPU算力；"
                "双GPU服务器建议设置device=cuda:0、eval.device=cuda:1。",
                eval_device,
            )
        async_evaluator = AsyncEvalController(
            policy_name=cfg.policy.name,
            env_id=env_id,
            obs_cameras=obs_cameras,
            eval_cfg=cfg.eval,
            eval_device=eval_device,
            use_amp=cfg.use_amp,
            amp_dtype=str(getattr(cfg, "amp_dtype", "bfloat16")),
            out_dir=out_dir,
            max_pending=int(getattr(cfg.eval, "max_pending", 1)),
            startup_timeout_s=float(
                getattr(cfg.eval, "startup_timeout_s", 180.0)
            ),
            shutdown_timeout_s=float(
                getattr(cfg.eval, "shutdown_timeout_s", 30.0)
            ),
        )
        async_evaluator.start()
        logging.info(
            "训练/评估并行模式已启用: train_device=%s, eval_device=%s, cameras=%s",
            device,
            eval_device,
            obs_cameras,
        )
    else:
        logging.info(f"正在通过 Gym 注册表构建环境: {env_id}")
        eval_env = make_eval_env(env_id, obs_cameras, cfg.eval)
        logging.info(f"✅ 环境加载成功！最终挂载的相机: {obs_cameras}")

    # ==========================================
    # 🌟 6. DP 预训练主循环
    # ==========================================
    max_checkpoints = getattr(cfg.eval, "max_checkpoints", 5)
    records_resume = getattr(cfg.eval, "records_resume", True)
    checkpoint_metric = getattr(cfg.eval, "checkpoint_metric", "loss")
    manager = TopKCheckpointManager(out_dir=out_dir, 
                                    max_keep=max_checkpoints, 
                                    records_resume=records_resume, 
                                    metric=checkpoint_metric)
    policy.train()
    logging.info("🔥 开始预训练 (模仿学习阶段)...")
    
    log_freq = int(getattr(cfg.training, "log_freq", 0))
    # 从 start_step 开始，避免覆盖之前的进度！
    for step in range(start_step, cfg.training.offline_steps):
        start_time = time.perf_counter()
        
        # 获取数据并推入 GPU
        batch = next(dl_iter) # 取出一个batch的数据
        dataloading_s = time.perf_counter() - start_time # 计算数据加载时间
        for key in batch: # 这里的key对应的是类别，如action/observation
            if isinstance(batch[key], torch.Tensor):
                # 最好加上非阻塞传输non_blocking，并确保原有的引用随着循环覆盖而消失
                batch[key] = batch[key].to(device, non_blocking=True)

        should_log = bool(log_freq and log_freq > 0 and step % log_freq == 0)
        should_run_eval_or_checkpoint = should_checkpoint_or_eval(step, cfg)
        collect_metrics = should_log or should_run_eval_or_checkpoint

        # 前向传播、Loss 计算、反向传播与 EMA 更新
        train_info = update_policy(
            policy,
            batch,
            optimizer,
            cfg.training.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.use_amp, # 是否使用混合精度训练，把部分计算从 float32 改成 float16，速度快 30%~100%
            amp_dtype=amp_dtype,
            collect_metrics=collect_metrics,
            ema=ema,
            step=step,
        )
        if collect_metrics:
            train_info["dataloading_s"] = dataloading_s

        if async_evaluator is not None:
            process_async_eval_results(
                async_evaluator,
                logger=logger,
                cfg=cfg,
                manager=manager,
                logging_step=step,
            )

        # 日志记录
        if should_log:
            log_train_info(logger, train_info, step, cfg, offline_dataset)

        # ==========================================
        # 评估和保存函数
        # ==========================================
        if should_run_eval_or_checkpoint:
            if async_evaluator is None:
                evaluate_and_checkpoint_if_needed(
                    step=step,
                    policy=policy,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    logger=logger,
                    cfg=cfg,
                    device=device,
                    out_dir=out_dir,
                    eval_env=eval_env,
                    train_loss=train_info["loss"],
                    manager=manager,
                    ema=ema,
                    grad_scaler=grad_scaler,
                )
            else:
                total_steps = int(cfg.training.offline_steps)
                is_last_step = step == total_steps - 1
                eval_due = should_run_periodic_step(
                    step,
                    total_steps,
                    int(getattr(cfg.training, "eval_freq", 0)),
                )
                save_due = bool(
                    getattr(cfg.training, "save_checkpoint", False)
                ) and should_run_periodic_step(
                    step,
                    total_steps,
                    int(getattr(cfg.training, "save_freq", 10000)),
                )

                if eval_due and not async_evaluator.has_capacity:
                    skip_if_busy = bool(
                        getattr(cfg.eval, "skip_if_busy", True)
                    )
                    if is_last_step or not skip_if_busy:
                        logging.info(
                            "异步评估队列已满，等待空闲槽位: step=%d",
                            step,
                        )
                        wait_for_async_eval_capacity(
                            async_evaluator,
                            logger=logger,
                            cfg=cfg,
                            manager=manager,
                            logging_step=step,
                        )
                    else:
                        logging.warning(
                            "异步评估仍在运行，跳过step=%d的评估请求；训练继续。",
                            step,
                        )
                        eval_due = False

                base_identifier = make_checkpoint_identifier(
                    step,
                    total_steps,
                    train_info["loss"],
                )
                checkpoint_path = None
                if save_due:
                    logger.save_checkpoint(
                        step,
                        policy,
                        optimizer,
                        lr_scheduler,
                        identifier=base_identifier,
                        ema_policy=(
                            ema.checkpoint_policy(policy)
                            if ema is not None
                            else None
                        ),
                        ema_state=ema.metadata() if ema is not None else None,
                        grad_scaler=grad_scaler,
                    )
                    checkpoint_path = (
                        Path(out_dir) / "checkpoints" / base_identifier
                    )

                if eval_due:
                    cleanup_snapshot_dir = None
                    if checkpoint_path is not None:
                        snapshot_path = checkpoint_path / "pretrained_model"
                        manager.protect(checkpoint_path)
                    else:
                        (
                            snapshot_path,
                            cleanup_snapshot_dir,
                        ) = async_evaluator.save_temporary_snapshot(
                            (
                                ema.evaluation_policy(policy)
                                if ema is not None
                                else policy
                            ),
                            step=step,
                        )

                    videos_dir = (
                        Path(out_dir) / "eval" / f"videos_{base_identifier}"
                    )
                    submitted = async_evaluator.submit(
                        step=step,
                        train_loss=train_info["loss"],
                        base_identifier=base_identifier,
                        snapshot_path=snapshot_path,
                        videos_dir=videos_dir,
                        checkpoint_path=checkpoint_path,
                        cleanup_snapshot_dir=cleanup_snapshot_dir,
                    )
                    if not submitted:
                        if checkpoint_path is not None:
                            manager.release(checkpoint_path)
                            manager.update(
                                step,
                                train_info["loss"],
                                checkpoint_path,
                            )
                        if cleanup_snapshot_dir is not None:
                            shutil.rmtree(
                                cleanup_snapshot_dir,
                                ignore_errors=True,
                            )
                        logging.warning(
                            "异步评估任务提交失败，已跳过step=%d。",
                            step,
                        )
                    else:
                        logging.info(
                            "已提交异步评估: step=%d, snapshot=%s, pending=%d",
                            step,
                            snapshot_path,
                            async_evaluator.pending_count,
                        )
                elif checkpoint_path is not None:
                    # 未安排环境评估时，仍按现有指标登记并清理checkpoint。
                    manager.update(
                        step,
                        train_info["loss"],
                        checkpoint_path,
                    )

    if async_evaluator is not None:
        if bool(getattr(cfg.eval, "wait_at_end", True)):
            while async_evaluator.pending_count > 0:
                completed = process_async_eval_results(
                    async_evaluator,
                    logger=logger,
                    cfg=cfg,
                    manager=manager,
                    logging_step=int(cfg.training.offline_steps) - 1,
                )
                if completed == 0:
                    time.sleep(0.1)
            async_evaluator.close()
        else:
            logging.warning(
                "eval.wait_at_end=false，将终止尚未完成的异步评估任务。"
            )
            async_evaluator.close(force=True)
    elif eval_env is not None and hasattr(eval_env, "close"):
        eval_env.close()
    logging.info("DPPO 预训练结束！")

# ==========================================
# 🌟 Hydra 启动入口 (保留配置功能与 Args 注入)
# ==========================================
@hydra.main(version_base="1.2", config_name="pre_default", config_path="../../../configs/pretrain") #配置文件存放位置
def train_cli(cfg: DictConfig):
    hydra_runtime = hydra.core.hydra_config.HydraConfig.get()
    out_dir = hydra_runtime.run.dir
    cfg, snapshot_config_path = build_resume_config(cfg)
    if snapshot_config_path is not None:
        # Hydra 在进入函数前保存的是“组合后、快照合并前”的配置。用实际
        # 生效配置覆盖它，确保原运行目录和后续排查记录与真实训练一致。
        effective_config_path = Path(out_dir) / ".hydra" / "config.yaml"
        effective_config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, effective_config_path)
        print(
            "🔒 [配置恢复] 已使用快照配置作为训练基线，并仅保留当前运行字段白名单:\n"
            f"   👉 {snapshot_config_path}\n"
            f"   👉 实际配置已记录到 {effective_config_path}\n"
            f"   👉 env={cfg.env.name}/{cfg.env.task}, policy={cfg.policy.name}"
        )

    train_dppo_pretrain(
        cfg,
        out_dir=out_dir,  # 获取当前训练运行的输出目录，用于保存训练输出的数据
        job_name=hydra_runtime.job.name, # 获取当前训练运行的作业名称，用于wandb
    )

if __name__ == "__main__":
    # 区分用户显式命令行覆盖与下面自动注入的本地默认值。严格续训时，
    # 用户显式指定且与快照冲突的 env/policy 会被拒绝。
    user_cli_args = tuple(sys.argv[1:])

    # 强行注入命令行参数 (极大提升本地调试和修改效率)
    # 这里面也可以随时添加你想覆盖的 args 参数
    default_args = [
        "dataset_local_dir=outputs/5_hf_datasets/quest_teleop_InsertCylinder-3Arms-v0_rgb_arm_recovery",
        "dataset_repo_id=Dc-dc/quest_teleop_SewNeedle-3Arms-v0_rgb_joint",
        # 全新训练时使用下面的默认配置组；resume=true 时会在 Hydra 启动前
        # 自动替换为原实验 .hydra/hydra.yaml 中记录的 env/policy。
        "env=sim_insert_cylinder_3arms",
        "policy=pre_zed_diffusion",
        # resume=false按当前env/policy全新训练；resume=true严格恢复原实验。
        "resume=false",
        "resume_path='outputs/2_pretrain/train/2026-07-21/23-22-46_InsertCylinder-3Arms-v0_pre_zed_coupled_dual_head_diffusion/checkpoints/100000_loss=0.0087_sr=53.0_ar=556.51'",
        "training.num_workers=5",
        "wandb.enable=true",
    ]
    
    for arg in default_args:
        arg_key = arg.split("=")[0]
        if get_cli_override_value(sys.argv, arg_key) is None:
            sys.argv.append(arg)

    # ==========================================
    # 🌟 核心修复：在 Hydra 启动前截胡！强行修改底层输出目录
    # ==========================================
    # 使用 replace(" ", "") 过滤掉所有可能的空格干扰
    is_resume = str(get_cli_override_value(sys.argv, "resume")).lower() == "true"
    resume_path = get_cli_override_value(sys.argv, "resume_path")

    if is_resume and resume_path:
        # 只要路径有效，就强行重定向
        if resume_path.lower() not in ["none", "null", ""]:
            snapshot_choices = restore_resume_hydra_choices(
                sys.argv,
                user_cli_args,
                resume_path,
            )
            if snapshot_choices:
                print(
                    "🔄 [预处理] 已从原实验恢复 Hydra 配置组: "
                    + ", ".join(
                        f"{name}={choice}"
                        for name, choice in snapshot_choices.items()
                    )
                )
            else:
                print(
                    "⚠️ [预处理] 原实验未保存 Hydra 配置组名称；"
                    "训练函数仍会从 checkpoint 的完整 config.yaml 恢复实际配置。"
                )

            original_run_dir = get_resume_run_dir(resume_path)
            if original_run_dir is None:
                checkpoint_dir = get_resume_checkpoint_dir(resume_path)
                original_run_dir = checkpoint_dir.parent.parent
            original_out_dir = str(original_run_dir.absolute())
            
            # 告诉 Hydra：不要建新文件夹了，日志、配置、视频统统给我存进这个老目录！
            replace_cli_override(
                sys.argv,
                "hydra.run.dir",
                f'"{original_out_dir}"',
            )
            print(f"🔄 [预处理] 检测到断点续训，已强制重定向所有输出至旧目录:\n   👉 {original_out_dir}")
    
    train_cli()
