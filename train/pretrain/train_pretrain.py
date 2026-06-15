#!/usr/bin/env python
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 强行指向国内镜像站
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import gymnasium as gym
import env
from env.task.sew_needle_env import SewNeedleEnv
import logging
import time
from contextlib import nullcontext
from pprint import pformat

from train.pretrain.eval import  evaluate_and_checkpoint_if_needed,TopKCheckpointManager

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

# ==========================================
# 🌟 采用官方最新极简 API，抛弃 factory.py
# ==========================================
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.transforms import get_image_transforms
# 复用 LeRobot 的其他核心组件
from lerobot.common.logger import Logger
from lerobot.common.policies.factory import make_policy # 用于获取训练策略模型
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
    set_global_seed,
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
    elif cfg.policy.name == "diffusion":
        # 🌟 修复：分离视觉 Backbone 和 U-Net 的学习率，并将所有参数纳入优化器
        optimizer_params_dicts = [
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    # 假设非 backbone 的参数（即 UNet 和相关投影层）
                    if not n.startswith("model.backbone") and not n.startswith("image_encoder") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    # 抓取视觉编码器的参数
                    if (n.startswith("model.backbone") or n.startswith("image_encoder") or n.startswith("visual_encoders")) and p.requires_grad
                ],
                # 视觉网络给予 10 倍小的学习率，保护预训练特征
                "lr": getattr(cfg.training, "lr_backbone", 1e-5), 
            },
        ]

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
        f"grdn:{grad_norm:.3f}", #梯度范数，衡量了模型参数更新的幅度，过大可能导致训练不稳定，过小可能导致训练停滞
        f"lr:{lr:0.1e}",
        # in seconds
        f"updt_s:{update_s:.3f}", #模型参数更新所花费的时间
        f"data_s:{dataloading_s:.3f}",  # 一般趋近于0，如果这个时间过长，说明cpu太弱了
    ]
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
    if cfg.policy.name == "diffusion" and not any("images" in k for k in delta_timestamps_dict.keys()):
        import logging
        logging.warning("警告: 你的 `delta_timestamps` 中没有包含任何图片的过去时间帧。\n")

    return delta_timestamps_dict



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
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)
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
    logging.info(f"⏱️ 解析到的动作时间轴: {resolved_delta_timestamps.get('action', [])[:5]} ...")
    logging.info(f"⏱️ 解析到的视觉时间轴: {resolved_delta_timestamps.get('observation.state', [])}")


    offline_dataset = LeRobotDataset(
        repo_id=cfg.dataset_repo_id, #根据id下载或者加载本地数据（/home/dc/.cache/huggingface/datasets）
        delta_timestamps=resolved_delta_timestamps,
        video_backend=cfg.video_backend, 
        image_transforms=image_transforms,
    )
    # # 使用官方函数解析并挂载到 cfg，这样 make_dataset 内部才能正确读取
    # resolve_delta_timestamps(cfg)
    
    # # 必须使用 make_dataset 以激活 ACT 依赖的 image_transforms
    # offline_dataset = make_dataset(cfg)
    # ==========================================
    # 🌟 2. 断点续训：路径校验与防雷处理
    # ==========================================
    start_step = 0
    policy_load_path = None
    training_state_file = None

    if cfg.resume:
        # 将配置获取的内容强制转为字符串并小写，防范 "none", "null", NoneType 导致崩溃
        raw_path = str(getattr(cfg, "resume_path", "")).strip().lower()
        
        if raw_path in ["", "none", "null"]:
            logging.warning("⚠️ 警告：开启了 resume=True，但 resume_path 为空，将从头开始全新训练。")
            cfg.resume = False # 强制关闭恢复标志
        else:
            chkpt_dir = Path(getattr(cfg, "resume_path"))
            if not chkpt_dir.exists():
                logging.warning(f"⚠️ 警告：指定的恢复路径不存在 [{chkpt_dir}]，将从头开始全新训练。")
                cfg.resume = False # 强制关闭恢复标志
            else:
                logging.info(f"🎯 成功检测到有效的恢复路径: {chkpt_dir}")
                policy_load_path = chkpt_dir / "pretrained_model"
                training_state_file = chkpt_dir / "training_state.pth"

    # ==========================================
    # 🌟 3. 初始化模型与优化器 (顺序极其重要！)
    # ==========================================
    logging.info("🧠 正在初始化 Diffusion Policy...")
    
    # 3.1 先创建模型 (如果 cfg.resume 为 True，底层会自动从 policy_load_path 加载旧权重)
    policy = make_policy(
        hydra_cfg=cfg,
        dataset_stats=offline_dataset.stats if not cfg.resume else None, 
        pretrained_policy_name_or_path=str(policy_load_path) if cfg.resume else None, 
    )
    policy.to(device)

    # 3.2 无论是不是 resume，都必须先根据模型初始化出全新的优化器！
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(enabled=cfg.use_amp) # 用于自动计算梯度缩放因子

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
            # 4. 恢复 GradScaler (只有在使用混合精度时才需要)
            if cfg.use_amp and "grad_scaler" in checkpoint_dict:
                grad_scaler.load_state_dict(checkpoint_dict["grad_scaler"])
                logging.info("✅ GradScaler 状态已恢复")
            else:
                logging.warning("⚠️ 存档中未找到 grad_scaler 状态，梯度缩放因子将重置！")

            # 5. 恢复 Step 步数
            if "step" in checkpoint_dict:
                start_step = checkpoint_dict["step"] + 1
                logging.info(f"⏭️ 从字典成功读取，训练将从 step {start_step} 无缝继续...")
            else:
                # 容错：如果字典里真没存步数，就退化为看文件夹的名字 (比如 000500)
                # 提取下划线前的纯数字部分再转换
                start_step = int(chkpt_dir.name.split('_')[0]) + 1
                logging.info(f"⏭️ 字典中未记录步数，从目录名推断，训练将从 step {start_step} 无缝继续...")

        except Exception as e:
            logging.error(f"❌ 解析 {training_state_file.name} 失败: {e}")
            try:
                # 提取下划线前的纯数字部分再转换
                start_step = int(chkpt_dir.name.split('_')[0]) + 1
                logging.info(f"⏭️ 降级方案：从目录名推断，训练将从 step {start_step} 无缝继续...")
            except ValueError:
                pass
    elif cfg.resume:
        logging.warning(f"⚠️ 找不到状态文件 {training_state_file}，只能恢复模型权重，优化器将重新归零。")


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

    dataloader = DataLoader(
        offline_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,         # 开启全局打乱
        sampler=sampler,                   
        pin_memory=(device.type != "cpu"),
        drop_last=True, # 开启丢弃最后一个不完整的批次
    )
    
    # 使用 Python 内置的 cycle 将其变为无限迭代器，使用 next(dl_iter) 进行取出一个batch的数据
    # dl_iter = cycle(dataloader) #会存有历史数据，导致显存溢出
    dl_iter = iter(get_infinite_dataloader(dataloader)) 
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"📐 模型可学习参数量: {num_learnable_params} ({format_big_number(num_learnable_params)})")
    logging.info(f"🎯 预训练目标步数: {cfg.training.offline_steps}")

    # ==========================================
    # 🌟 5. 动态拼接环境 ID 并创建环境
    # ==========================================
    # 观测要用的相机列表  =  模型推理要用的相机列表 + 评估时保存的video视角相机
    all_obs_keys = policy.config.input_shapes.keys()
    ref_cams = [k.replace("observation.images.", "") for k in all_obs_keys if "observation.images." in k]
    if not ref_cams:
        raise ValueError(f"❌ 严重冲突：模型中未找到相机相关参数。请检查模型输入是否正确。")
    obs_cameras = list(dict.fromkeys(ref_cams + cfg.eval.render_camera))

    # 读取 YAML 中的 name ("guided_vision") 和 task ("SewNeedle-2Arms-v0")
    # 拼接出 "guided_vision/SewNeedle-2Arms-v0"
    env_id = f"{cfg.env.name}/{cfg.env.task}" 
    
    logging.info(f"正在通过 Gym 注册表构建环境: {env_id}")

    # 使用 gym.make 创建环境，并通过 kwargs 强行覆盖你需要的相机
    eval_env = gym.make(
        id=env_id, 
        disable_env_checker=True,  
        cameras=obs_cameras,  # 👈 这里的传参会直接覆盖 __init__.py 里的默认套餐！
    )
    logging.info(f"✅ 环境加载成功！最终挂载的相机: {obs_cameras}")

    # ==========================================
    # 🌟 6. DPPO 预训练主循环
    # ==========================================
    max_checkpoints = getattr(cfg.eval, "max_checkpoints", 5)
    records_resume = getattr(cfg.eval, "records_resume", True)
    checkpoint_metric = getattr(cfg.eval, "checkpoint_metric", "loss")
    manager = TopKCheckpointManager(out_dir=out_dir, 
                                    max_keep=max_checkpoints, 
                                    records_resume=records_resume, 
                                    metric=checkpoint_metric)
    policy.train()
    logging.info("🔥 开始 DPPO 预训练 (模仿学习阶段)...")
    
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

        # 前向传播、Loss 计算、反向传播与 EMA 更新
        train_info = update_policy(
            policy,
            batch,
            optimizer,
            cfg.training.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.use_amp, # 是否使用混合精度训练，把部分计算从 float32 改成 float16，速度快 30%~100%
        )
        train_info["dataloading_s"] = dataloading_s

        # 日志记录
        if step % cfg.training.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset)

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
    logging.info("🎉 DPPO 预训练圆满结束！你现在可以拿着这个 Checkpoint 去跑 PPO 微调了。")

# ==========================================
# 🌟 Hydra 启动入口 (保留配置功能与 Args 注入)
# ==========================================
@hydra.main(version_base="1.2", config_name="pre_default", config_path="../../configs/pretrain") #配置文件存放位置
def train_cli(cfg: DictConfig):

    train_dppo_pretrain(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,  # 获取当前训练运行的输出目录，用于保存训练输出的数据
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name, # 获取当前训练运行的作业名称，用于wandb
    )

if __name__ == "__main__":
    # 强行注入命令行参数 (极大提升本地调试和修改效率)
    # 这里面也可以随时添加你想覆盖的 args 参数
    default_args = [
        # "dataset_repo_id=Dc-dc/quest_teleop_sew_needle_3arms_jonit",
        "env=sim_sew_needle_3arms", # 环境，这俩定义在default文件中
        "policy=pre_zed_wrist_diffusion", # 策略
        "resume=false",
        "resume_path='outputs/2_pretrain/train/2026-05-19/00-57-05_SewNeedle-3Arms-v0_pre_zed_static_wrist_diffusion/checkpoints/108000_loss=0.0111_sr=0.0_ar=-64.33'",
        "training.batch_size=16",
        "training.num_workers=4",
        "wandb.enable=false", # 关闭 wandb，不需要aLse" ,
    ]
    
    for arg in default_args:
        arg_key = arg.split("=")[0]
        if not any(arg_key in sys_arg for sys_arg in sys.argv):
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
            # checkpoints/last 的上一级的上一级，就是原本的训练根目录
            original_out_dir = str(ckpt_path.parent.parent.absolute())
            
            # 告诉 Hydra：不要建新文件夹了，日志、配置、视频统统给我存进这个老目录！
            sys.argv.append(f'hydra.run.dir="{original_out_dir}"')
            print(f"🔄 [预处理] 检测到断点续训，已强制重定向所有输出至旧目录:\n   👉 {original_out_dir}")
    
    train_cli()
