"""冻结预训练耦合双头扩散策略，训练事件触发式重规划 Double DQN。"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

ROOT_DIR = Path(__file__).resolve().parents[2]             # AV-piper项目根目录。
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))                     # 支持直接执行本文件而非仅使用python -m。

os.environ.setdefault("MUJOCO_GL", "egl")                 # MuJoCo离屏渲染使用EGL后端。
os.environ.setdefault(                                    # PyOpenGL与MuJoCo使用同一个图形后端。
    "PYOPENGL_PLATFORM",
    os.environ["MUJOCO_GL"],
)

import einops
import gymnasium as gym
import hydra
import numpy as np
import torch
import yaml
from lerobot.common.envs.utils import preprocess_observation
from omegaconf import DictConfig, OmegaConf

import env as _registered_env  # noqa: F401  注册项目中的Gym环境。

if __package__:
    from .data_collection import (
        ReplanningActionCache,
        ReplanningDataCollector,
        ReplanningRewardConfig,
        build_replanning_state,
    )
    from .dqn import (
        DoubleDQNTrainer,
        ReplanningDecision,
        ReplanningDQNConfig,
        ReplanningDuelingQNetwork,
        ReplanningReplayBuffer,
    )
else:
    from train.s4_adaptive_replanning.data_collection import (
        ReplanningActionCache,
        ReplanningDataCollector,
        ReplanningRewardConfig,
        build_replanning_state,
    )
    from train.s4_adaptive_replanning.dqn import (
        DoubleDQNTrainer,
        ReplanningDecision,
        ReplanningDQNConfig,
        ReplanningDuelingQNetwork,
        ReplanningReplayBuffer,
    )


def seed_everything(seed: int) -> None:
    """固定Python、NumPy、PyTorch和CUDA随机数种子。"""
    seed = int(seed)                                      # 统一转换为各随机库接受的整数种子。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_training_config(cfg: DictConfig) -> None:
    """在加载大模型和环境前检查训练循环依赖的关键配置。"""
    positive_ints = {                                    # 所有必须严格为正的循环与容量参数。
        "env.max_episode_steps": cfg.env.max_episode_steps,          # 单回合最多物理步数。
        "training.total_env_steps": cfg.training.total_env_steps,    # 整次训练目标环境步数。
        "training.replay_capacity": cfg.training.replay_capacity,    # 回放缓冲区最大transition数。
        "training.learning_starts": cfg.training.learning_starts,    # 开始TD更新前的预热数据量。
        "training.batch_size": cfg.training.batch_size,              # 每次DQN更新的采样数量。
        "training.train_frequency": cfg.training.train_frequency,    # 相邻DQN更新间隔的环境步数。
        "training.updates_per_step": cfg.training.updates_per_step,  # 每个更新点连续执行的梯度次数。
        "training.log_freq_episodes": cfg.training.log_freq_episodes,# 终端日志episode间隔。
        "training.eval_freq_steps": cfg.training.eval_freq_steps,    # 自动评估环境步间隔。
        "training.save_freq_steps": cfg.training.save_freq_steps,    # 自动保存环境步间隔。
        "eval.n_episodes": cfg.eval.n_episodes,                      # 每轮评估的回合数量。
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"{name}必须大于0，当前为{value}")
    if int(cfg.training.batch_size) > int(cfg.training.replay_capacity):
        raise ValueError("training.batch_size不能大于replay_capacity")
    if int(cfg.training.learning_starts) > int(cfg.training.replay_capacity):
        raise ValueError("training.learning_starts不能大于replay_capacity")
    if int(cfg.training.min_steps_after_replan) < 0:
        raise ValueError("training.min_steps_after_replan必须非负")
    probabilities = {                                    # 所有应限制在[0,1]内的探索概率。
        "training.warmup_continue_probability": (
            cfg.training.warmup_continue_probability
        ),
        "training.epsilon_start": cfg.training.epsilon_start,
        "training.epsilon_end": cfg.training.epsilon_end,
    }
    for name, value in probabilities.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name}必须位于[0,1]，当前为{value}")


def resolve_device(device_name: str) -> torch.device:
    """解析训练设备，并在CUDA不可用时安全退回CPU。"""
    requested = torch.device(str(device_name))            # 用户配置中期望使用的计算设备。
    if requested.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA不可用，重规划DQN训练将退回CPU。")
        return torch.device("cpu")
    return requested


def resolve_project_path(path: str | Path) -> Path:
    """把相对路径解析到项目根目录，并展开用户目录。"""
    resolved = Path(path).expanduser()                    # 展开路径中的“~”用户目录。
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved.resolve()


def resolve_policy_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[Path, Path, Path]:
    """解析checkpoint根目录、模型目录和训练时保存的config.yaml。"""
    checkpoint_root = resolve_project_path(checkpoint_path)  # 用户传入的checkpoint或模型目录。
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"找不到预训练checkpoint: {checkpoint_root}")

    if (checkpoint_root / "pretrained_model").is_dir():
        model_dir = checkpoint_root / "pretrained_model"    # LeRobot权重与config.json所在目录。
        run_or_checkpoint_dir = checkpoint_root              # 当前具体checkpoint目录。
    elif checkpoint_root.name == "pretrained_model":
        model_dir = checkpoint_root                          # 用户已经直接传入pretrained_model。
        run_or_checkpoint_dir = checkpoint_root.parent       # pretrained_model所属checkpoint目录。
    else:
        model_dir = checkpoint_root                          # 兼容直接可被from_pretrained加载的目录。
        run_or_checkpoint_dir = checkpoint_root              # 同时将其视为配置搜索起点。

    config_candidates = [                                # 从模型层逐级向运行根目录搜索完整Hydra配置。
        model_dir / "config.yaml",
        run_or_checkpoint_dir / "config.yaml",
        run_or_checkpoint_dir.parent / "config.yaml",
        run_or_checkpoint_dir.parent.parent / "config.yaml",
    ]
    config_path = next(                                  # 实际存在的第一份训练config.yaml。
        (path for path in config_candidates if path.is_file()),
        None,
    )
    if config_path is None:
        searched = "\n".join(                            # 错误信息中展示全部已检查位置。
            f"  - {path}" for path in config_candidates
        )
        raise FileNotFoundError(f"找不到预训练策略config.yaml，已检查:\n{searched}")
    return run_or_checkpoint_dir, model_dir, config_path


def load_frozen_policy(
    checkpoint_path: str | Path,
    device: torch.device,
):
    """按项目现有LeRobot格式加载策略，并冻结全部预训练参数。"""
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.utils.utils import init_hydra_config

    checkpoint_root, model_dir, config_path = resolve_policy_checkpoint(  # 解析权重和完整配置路径。
        checkpoint_path
    )
    hydra_cfg = init_hydra_config(str(config_path))        # 恢复预训练时的模型结构和归一化配置。
    hydra_cfg.device = str(device)                         # 避免旧config中的CUDA设备在CPU机器上提前报错。
    policy = make_policy(                                 # 从本地pretrained_model恢复双头策略权重。
        hydra_cfg=hydra_cfg,
        pretrained_policy_name_or_path=str(model_dir),
    )
    policy.to(device)
    policy.eval()
    policy.requires_grad_(False)
    for module in policy.modules():
        if "RgbEncoder" in module.__class__.__name__:
            module._debug_img_counter = max(             # 禁止训练rollout反复保存视觉调试图片。
                int(getattr(module, "_debug_img_counter", 0)),
                3,
            )

    required_methods = (                                 # 所有双头策略共享的底层接口。
        "_prepare_global_conditioning",
        "combine_action_heads",
    )
    missing = [
        name for name in required_methods if not hasattr(policy.diffusion, name)
    ]
    if missing:
        raise TypeError(
            "该训练入口需要双头扩散策略，底层缺少接口: "
            f"{missing}"
        )
    has_coupled_sampler = callable(
        getattr(policy.diffusion, "conditional_sample_coupled", None)
    )
    has_independent_sampler = callable(
        getattr(policy.diffusion, "conditional_sample", None)
    ) and all(
        hasattr(policy.diffusion, name)
        for name in (
            "arm_unet",
            "view_unet",
            "arm_noise_scheduler",
            "view_noise_scheduler",
            "arm_action_dim",
            "view_action_dim",
        )
    )
    if not has_coupled_sampler and not has_independent_sampler:
        raise TypeError(
            "双头策略必须提供conditional_sample_coupled，或提供完整的独立双头采样接口。"
        )
    if not hasattr(policy.diffusion, "rgb_encoder"):
        raise TypeError("重规划DQN当前要求预训练策略包含rgb_encoder")

    with config_path.open("r", encoding="utf-8") as file:
        saved_config = yaml.safe_load(file) or {}         # 用于恢复环境name、task等训练元信息。
    logging.info("冻结的预训练策略已加载: %s", model_dir)
    return policy, saved_config, checkpoint_root, model_dir


def make_diffusion_generator(
    device: torch.device,
    seed: int,
) -> torch.Generator:
    """创建与策略采样设备一致且相互独立的PyTorch随机数生成器。"""
    generator_device = (                                 # torch.Generator必须与扩散采样张量同设备。
        device if device.type == "cuda" else torch.device("cpu")
    )
    generator = torch.Generator(device=generator_device) # 与环境和epsilon探索相互独立的扩散随机流。
    generator.manual_seed(int(seed))
    return generator


def policy_camera_names(policy) -> list[str]:
    """按策略输入配置顺序返回需要挂载的相机名称。"""
    cameras = [                                          # 保持预训练输入键顺序的相机名称列表。
        key.replace("observation.images.", "")
        for key in policy.config.input_shapes
        if key.startswith("observation.images.")
    ]
    if not cameras:
        raise ValueError("预训练策略没有observation.images.*输入")
    return cameras


def make_training_env(
    policy,
    saved_policy_config: Mapping,
    cfg_env: DictConfig,
):
    """使用预训练策略的相机输入创建单个MuJoCo训练环境。"""
    saved_env = saved_policy_config.get("env", {})       # checkpoint保存的原始环境配置。
    env_name = str(                                      # 优先使用当前DQN配置中的环境命名空间。
        getattr(cfg_env, "name", None) or saved_env.get("name")
    )
    env_task = str(                                      # 优先使用当前DQN配置中的任务名称。
        getattr(cfg_env, "task", None) or saved_env.get("task")
    )
    if not env_name or env_name == "None" or not env_task or env_task == "None":
        raise ValueError("无法从DQN配置或预训练config.yaml确定环境name/task")
    env_id = f"{env_name}/{env_task}"                    # Gym注册表中的完整环境ID。
    cameras = policy_camera_names(policy)                 # 只渲染预训练策略真正需要的相机。
    env = gym.make(                                       # 单环境实例；动作缓存暂不支持向量环境。
        id=env_id,
        disable_env_checker=True,
        cameras=cameras,
        episode_length=int(cfg_env.max_episode_steps),
        enable_reward_debug=bool(cfg_env.enable_reward_debug),
    )
    logging.info("MuJoCo环境已创建: %s，相机=%s", env_id, cameras)
    return env, env_id


def _add_batch_dim(value):
    """递归复制NumPy观测，并在最前面增加单环境batch维。"""
    if isinstance(value, dict):
        return {key: _add_batch_dim(item) for key, item in value.items()}
    if hasattr(value, "copy"):
        return np.expand_dims(value.copy(), axis=0).copy()
    return value


@torch.no_grad()
def prepare_normalized_observation(
    observation: Mapping,
    policy,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """把Gym观测转换为策略归一化后的单帧状态和多相机图像。"""
    batch = preprocess_observation(                       # 转换为LeRobot扁平键和CHW浮点图像。
        _add_batch_dim(observation)
    )
    policy_inputs = {                                     # 过滤环境中与预训练策略无关的观测字段。
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in policy.config.input_shapes
    }
    normalized = policy.normalize_inputs(policy_inputs)  # 使用预训练数据集统计量归一化输入。
    result = {                                            # 单帧归一化状态，形状为[B, state_dim]。
        "observation.state": normalized["observation.state"],
    }
    image_tensors = [                                     # 按策略输入键顺序排列的多相机图像。
        normalized[key]
        for key in policy.expected_image_keys
    ]
    if image_tensors:
        result["observation.images"] = torch.stack(      # [B,N,C,H,W]，N为相机数量。
            image_tensors,
            dim=-4,
        )
    if policy.use_env_state:
        result["observation.environment_state"] = normalized[
            "observation.environment_state"
        ]
    return result


class NormalizedObservationHistory:
    """显式维护扩散策略所需的最近n_obs_steps条归一化观测。"""

    def __init__(self, n_obs_steps: int):
        if int(n_obs_steps) <= 0:
            raise ValueError("n_obs_steps必须大于0")
        self.n_obs_steps = int(n_obs_steps)                # 策略条件使用的历史观测长度。
        self.queues: dict[str, deque] = {}                 # 每个观测字段独立的定长队列。

    def reset(self, first_observation: Mapping[str, torch.Tensor]) -> None:
        """用episode第一帧复制填满全部历史位置。"""
        self.queues = {
            key: deque(maxlen=self.n_obs_steps)
            for key in first_observation
        }
        for key, value in first_observation.items():
            for _ in range(self.n_obs_steps):
                self.queues[key].append(value.detach().clone())

    def append(self, observation: Mapping[str, torch.Tensor]) -> None:
        """追加一条新观测，并自动丢弃最旧观测。"""
        if set(observation) != set(self.queues):
            raise KeyError(
                "观测历史字段发生变化: "
                f"expected={sorted(self.queues)}, got={sorted(observation)}"
            )
        for key, value in observation.items():
            self.queues[key].append(value.detach())

    def stacked(self) -> dict[str, torch.Tensor]:
        """把各字段堆叠为扩散模型要求的[B, S, ...]格式。"""
        if not self.queues:
            raise RuntimeError("观测历史尚未reset")
        return {
            key: torch.stack(list(queue), dim=1)
            for key, queue in self.queues.items()
        }


@torch.no_grad()
def extract_visual_features(
    policy,
    history_batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """复用冻结rgb_encoder提取两帧、多相机拼接后的纯视觉特征。"""
    images = history_batch.get("observation.images")      # [B,S,N,C,H,W]归一化图像历史。
    if images is None:
        raise KeyError("观测历史缺少observation.images")
    if images.ndim != 6:
        raise ValueError(
            "observation.images应为[B,S,N,C,H,W]，"
            f"当前为{tuple(images.shape)}"
        )
    batch_size, history_steps, camera_count = images.shape[:3]  # 批量、历史长度和相机数。
    flat_images = einops.rearrange(                      # 合并前三维以批量运行一次冻结RGB编码器。
        images,
        "b s n ... -> (b s n) ...",
    )
    flat_features = policy.diffusion.rgb_encoder(        # [(B*S*N), image_feature_dim]。
        flat_images
    )
    if flat_features.ndim != 2:
        raise ValueError(
            f"rgb_encoder输出应为二维特征，当前为{tuple(flat_features.shape)}"
        )
    return einops.rearrange(
        flat_features,
        "(b s n) d -> b (s n d)",
        b=batch_size,
        s=history_steps,
        n=camera_count,
    )


def current_robot_state(
    history_batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """返回历史中最新一帧的归一化机器人状态。"""
    return history_batch["observation.state"][:, -1]


@torch.no_grad()
def infer_full_joint_chunk(
    policy,
    history_batch: Mapping[str, torch.Tensor],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """绕过内部8步队列，联合生成完整horizon的归一化和环境动作。"""
    device = history_batch["observation.state"].device   # 当前扩散条件和动作采样所在设备。
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()                      # 联合扩散推理的高精度起始时间。
    global_cond = policy.diffusion._prepare_global_conditioning(  # [B, global_cond_dim]观测条件。
        history_batch
    )
    if callable(getattr(policy.diffusion, "conditional_sample_coupled", None)):
        arm_actions, view_head_actions = policy.diffusion.conditional_sample_coupled(
            batch_size=global_cond.shape[0],
            global_cond=global_cond,
            generator=generator,
        )
    else:
        arm_actions = policy.diffusion.conditional_sample(
            policy.diffusion.arm_unet,
            policy.diffusion.arm_noise_scheduler,
            policy.diffusion.arm_action_dim,
            global_cond.shape[0],
            global_cond=global_cond,
            generator=generator,
        )
        view_head_actions = policy.diffusion.conditional_sample(
            policy.diffusion.view_unet,
            policy.diffusion.view_noise_scheduler,
            policy.diffusion.view_action_dim,
            global_cond.shape[0],
            global_cond=global_cond,
            generator=generator,
        )
    # 通过统一接口组合普通双头或耦合双头的Arm/View动作。
    normalized_actions = policy.diffusion.combine_action_heads(
        arm_actions,
        view_head_actions,
    )
    env_actions = policy.unnormalize_outputs(              # 恢复为env.step可直接执行的物理动作量纲。
        {"action": normalized_actions}
    )["action"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - start_time) * 1000.0  # 完整双头采样耗时，单位ms。
    if normalized_actions.shape[0] != 1:
        raise ValueError("第一版DQN训练入口只支持单环境batch_size=1")
    return normalized_actions[0], env_actions[0], inference_ms


def linear_epsilon(
    step: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    """计算从start线性衰减到end的epsilon探索率。"""
    if int(decay_steps) <= 0:
        return float(end)
    fraction = min(                                       # 当前训练步在衰减区间中的归一化位置。
        1.0,
        max(0.0, float(step) / float(decay_steps)),
    )
    return float(start) + fraction * (float(end) - float(start))


def sample_warmup_decision(
    action_mask: torch.Tensor,
    continue_probability: float,
    rng: random.Random,
) -> int:
    """预热期优先继续执行，避免均匀随机导致约一半环境步重新推理。"""
    if action_mask.shape != (1, len(ReplanningDecision)):
        raise ValueError(f"单环境action_mask形状错误: {action_mask.shape}")
    valid_actions = (                                     # 当前状态所有合法的高层决策编号。
        torch.nonzero(action_mask[0], as_tuple=False).flatten().tolist()
    )
    if not valid_actions:
        raise ValueError("当前状态没有合法重规划决策")
    continue_id = int(ReplanningDecision.CONTINUE)        # 复用缓存动作对应的固定决策编号0。
    if (
        continue_id in valid_actions
        and rng.random() < float(continue_probability)
    ):
        return continue_id
    non_continue = [                                      # 未命中继续先验时可随机选择的重规划动作。
        action for action in valid_actions if action != continue_id
    ]
    return int(rng.choice(non_continue or valid_actions))


def create_dqn_components(
    cfg: DictConfig,
    visual_feature_dim: int,
    robot_state_dim: int,
    arm_action_dim: int,
    view_action_dim: int,
    horizon: int,
    device: torch.device,
):
    """根据首次真实特征形状创建DQN、目标网络、回放和收集器。"""
    dqn_config = ReplanningDQNConfig(                     # 网络结构、动作形状和TD优化配置。
        visual_feature_dim=int(visual_feature_dim),
        robot_state_dim=int(robot_state_dim),
        arm_action_dim=int(arm_action_dim),
        view_action_dim=int(view_action_dim),
        horizon=int(horizon),
        visual_embed_dim=int(cfg.dqn.visual_embed_dim),
        state_embed_dim=int(cfg.dqn.state_embed_dim),
        chunk_embed_dim=int(cfg.dqn.chunk_embed_dim),
        hidden_dim=int(cfg.dqn.hidden_dim),
        gamma=float(cfg.dqn.gamma),
        learning_rate=float(cfg.dqn.learning_rate),
        weight_decay=float(cfg.dqn.weight_decay),
        target_update_tau=float(cfg.dqn.target_update_tau),
        grad_clip_norm=float(cfg.dqn.grad_clip_norm),
    )
    reward_config = ReplanningRewardConfig(               # 环境奖励缩放与两类重规划成本。
        env_reward_scale=float(cfg.reward.env_reward_scale),
        view_only_replan_cost=float(cfg.reward.view_only_replan_cost),
        joint_replan_cost=float(cfg.reward.joint_replan_cost),
        arm_discontinuity_coef=float(cfg.reward.arm_discontinuity_coef),
    )
    online_network = ReplanningDuelingQNetwork(           # 实际参与选动作和梯度更新的在线Q网络。
        dqn_config
    ).to(device)
    trainer = DoubleDQNTrainer(online_network, dqn_config) # 同时创建冻结目标网络和AdamW优化器。
    replay_buffer = ReplanningReplayBuffer(               # 在CPU预分配的固定容量transition存储。
        capacity=int(cfg.training.replay_capacity),
        config=dqn_config,
    )
    collector = ReplanningDataCollector(                  # 负责奖励合成、动作掩码和transition写入。
        replay_buffer,
        reward_config,
    )
    return dqn_config, reward_config, trainer, replay_buffer, collector


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """恢复DQN检查点后把AdamW动量张量移动到参数设备。"""
    for state in optimizer.state.values():                # 每个可训练参数对应的AdamW内部状态。
        for key, value in state.items():                  # step、exp_avg和exp_avg_sq等具体字段。
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_dqn_checkpoint(
    checkpoint_path: str | Path | None,
    trainer: DoubleDQNTrainer,
    dqn_config: ReplanningDQNConfig,
    device: torch.device,
) -> tuple[int, int]:
    """可选恢复DQN网络、优化器、环境步和episode编号。"""
    if checkpoint_path is None or str(checkpoint_path).strip().lower() in {
        "",
        "none",
        "null",
    }:
        return 0, 0
    path = resolve_project_path(checkpoint_path)          # 用户传入的latest.pt或checkpoints目录。
    if path.is_dir():
        path = path / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"找不到DQN恢复文件: {path}")
    checkpoint = torch.load(                              # 先加载到CPU，避免恢复时占用额外显存峰值。
        path,
        map_location="cpu",
        weights_only=False,
    )
    stored_config = checkpoint.get("dqn_config", {})     # 保存时记录的网络输入和动作形状。
    for name in (
        "visual_feature_dim",
        "robot_state_dim",
        "arm_action_dim",
        "view_action_dim",
        "horizon",
    ):
        if name in stored_config and int(stored_config[name]) != int(
            getattr(dqn_config, name)
        ):
            raise ValueError(
                f"DQN恢复配置{name}不一致: "
                f"checkpoint={stored_config[name]}, current={getattr(dqn_config, name)}"
            )
    trainer.online_network.load_state_dict(checkpoint["online_network"])
    if "target_network" in checkpoint:
        trainer.target_network.load_state_dict(checkpoint["target_network"])
    else:
        trainer.hard_update_target()
    if "optimizer" in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(trainer.optimizer, device)
    total_env_steps = int(                                # 已完成的物理环境步数，用于继续调度。
        checkpoint.get("total_env_steps", 0)
    )
    episode = int(checkpoint.get("episode", 0))           # 已完成的训练episode数量。
    logging.info(
        "DQN检查点已恢复: %s, total_env_steps=%d, episode=%d",
        path,
        total_env_steps,
        episode,
    )
    return total_env_steps, episode


def save_dqn_checkpoint(
    output_dir: Path,
    trainer: DoubleDQNTrainer,
    dqn_config: ReplanningDQNConfig,
    reward_config: ReplanningRewardConfig,
    total_env_steps: int,
    episode: int,
    pretrained_model_dir: Path,
    *,
    save_step_copy: bool,
) -> Path:
    """保存DQN网络、优化器、配置和预训练策略来源。"""
    checkpoint_dir = output_dir / "checkpoints"          # 当前Hydra运行目录下的DQN快照目录。
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {                                           # 恢复DQN训练所需的全部轻量状态。
        "online_network": trainer.online_network.state_dict(),  # 在线Q网络参数。
        "target_network": trainer.target_network.state_dict(),  # 慢速目标Q网络参数。
        "optimizer": trainer.optimizer.state_dict(),            # AdamW动量与步数。
        "dqn_config": asdict(dqn_config),                        # 网络形状和TD超参数。
        "reward_config": asdict(reward_config),                  # 奖励缩放和重规划成本。
        "total_env_steps": int(total_env_steps),                 # 全局环境步计数器。
        "episode": int(episode),                                 # 全局episode计数器。
        "pretrained_policy_path": str(pretrained_model_dir),     # 冻结底层策略来源。
    }
    latest_path = checkpoint_dir / "latest.pt"           # 每次覆盖，作为默认恢复入口。
    torch.save(payload, latest_path)
    if save_step_copy:
        torch.save(payload, checkpoint_dir / f"step_{total_env_steps:08d}.pt")
    return latest_path


def append_json_metrics(path: Path, metrics: Mapping[str, float | int]) -> None:
    """把一行训练或评估指标追加到本地JSONL文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(metrics), ensure_ascii=False) + "\n")


def maybe_init_wandb(cfg: DictConfig, output_dir: Path):
    """按配置可选启动W&B；关闭时返回None。"""
    if not bool(cfg.wandb.enable):
        return None
    import wandb

    settings = wandb.Settings(                            # 关闭不需要的系统和机器信息采集。
        x_disable_stats=bool(cfg.wandb.disable_system_stats),
        x_disable_machine_info=bool(cfg.wandb.disable_machine_info),
    )
    return wandb.init(
        project=str(cfg.wandb.project),
        name=hydra.core.hydra_config.HydraConfig.get().job.name,
        notes=str(cfg.wandb.notes),
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=settings,
    )


def choose_dqn_decision(
    network: ReplanningDuelingQNetwork,
    state: Mapping[str, torch.Tensor],
    action_mask: torch.Tensor,
    replay_size: int,
    learning_starts: int,
    epsilon: float,
    warmup_continue_probability: float,
    rng: random.Random,
) -> tuple[int, torch.Tensor | None]:
    """预热期使用带先验随机策略，之后使用DQN epsilon-greedy策略。"""
    if replay_size < int(learning_starts):
        return (
            sample_warmup_decision(
                action_mask,
                warmup_continue_probability,
                rng,
            ),
            None,
        )
    action, q_values = network.select_action(             # 预热完成后的epsilon-greedy DQN决策。
        state,
        action_mask,
        epsilon=float(epsilon),
    )
    return int(action.item()), q_values


@torch.no_grad()
def evaluate_replanning_policy(
    policy,
    dqn_network: ReplanningDuelingQNetwork,
    env,
    collector: ReplanningDataCollector,
    dqn_config: ReplanningDQNConfig,
    cfg: DictConfig,
    device: torch.device,
    total_env_steps: int,
) -> dict[str, float]:
    """使用epsilon=0评估动态重规划成功率、回报和推理次数。"""
    dqn_network.eval()
    generator = make_diffusion_generator(                 # 与训练采样隔离的评估扩散随机流。
        device,
        int(cfg.eval.seed) + int(total_env_steps),
    )
    successes = []                                      # 各评估episode是否成功。
    returns = []                                        # 各评估episode原始环境回报。
    joint_replans = []                                  # 各评估episode联合推理次数。
    execution_lengths = []                              # 每次扩散推理平均支撑的环境步数。
    inference_times = []                                # 所有联合推理的实际耗时。

    for episode in range(int(cfg.eval.n_episodes)):
        episode_seed = int(cfg.eval.seed) + episode        # 当前评估回合的可复现派生种子。
        observation, _ = env.reset(seed=episode_seed)     # Gym原始首帧观测。
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(episode_seed)
        current = prepare_normalized_observation(         # 策略统计量归一化后的首帧观测。
            observation,
            policy,
            device,
        )
        history = NormalizedObservationHistory(           # 当前评估episode独立的观测历史。
            policy.config.n_obs_steps
        )
        history.reset(current)
        cache = ReplanningActionCache(dqn_config)         # 当前episode独立的完整动作块缓存。
        episode_return = 0.0                              # 当前episode未缩放环境累计奖励。
        episode_steps = 0                                 # 当前episode已执行的物理动作数。
        episode_replans = 0                               # 当前episode联合扩散推理次数。
        success = False                                   # 当前episode是否曾报告任务成功。

        while episode_steps < int(cfg.env.max_episode_steps):
            history_batch = history.stacked()             # [1,S,...]当前两帧归一化观测。
            state = build_replanning_state(               # DQN读取的视觉、状态和剩余动作组合。
                extract_visual_features(policy, history_batch),
                current_robot_state(history_batch),
                cache,
            )
            action_mask = collector.build_action_mask(    # 根据缓存和冷却时间屏蔽非法决策。
                cache,
                view_only_available=bool(cfg.training.view_only_available),
                min_steps_after_replan=int(cfg.training.min_steps_after_replan),
            )
            decision_tensor, _ = dqn_network.select_action(  # 评估阶段关闭epsilon随机探索。
                state,
                action_mask,
                epsilon=0.0,
            )
            decision = ReplanningDecision(                # 转换为具名的高层重规划决策。
                int(decision_tensor.item())
            )
            if decision == ReplanningDecision.VIEW_ONLY_REPLAN:
                raise NotImplementedError(
                    "View-only条件扩散尚未实现，请保持view_only_available=false"
                )
            if decision == ReplanningDecision.JOINT_REPLAN:
                normalized_chunk, env_chunk, inference_ms = infer_full_joint_chunk(  # 新的完整16步缓存。
                    policy,
                    history_batch,
                    generator,
                )
                cache.replace_joint(normalized_chunk, env_chunk)
                episode_replans += 1
                inference_times.append(inference_ms)

            env_action = (                                # 当前缓存中下一条可执行20维物理动作。
                cache.peek_env_action().detach().cpu().numpy()
            )
            next_observation, reward, terminated, truncated, info = env.step(
                env_action
            )
            cache.advance()
            episode_return += float(reward)
            episode_steps += 1
            success = success or bool(info.get("is_success", False))
            next_current = prepare_normalized_observation(  # 执行后真实新观测的归一化结果。
                next_observation,
                policy,
                device,
            )
            history.append(next_current)
            if terminated or truncated:
                break

        successes.append(float(success))
        returns.append(episode_return)
        joint_replans.append(float(episode_replans))
        execution_lengths.append(
            float(episode_steps) / float(max(1, episode_replans))
        )

    return {
        "eval/success_rate": float(np.mean(successes)),              # 评估任务成功率。
        "eval/average_return": float(np.mean(returns)),              # 平均原始环境累计奖励。
        "eval/joint_replans_per_episode": float(np.mean(joint_replans)),  # 每回合平均扩散调用次数。
        "eval/average_execution_length": float(np.mean(execution_lengths)),# 每次推理平均连续执行步数。
        "eval/average_inference_ms": (
            float(np.mean(inference_times)) if inference_times else 0.0
        ),                                                            # 单次联合扩散平均耗时。
    }


def train_replanning_dqn(
    cfg: DictConfig,
    output_dir: str | Path,
) -> None:
    """执行单环境在线数据收集、Double DQN更新、评估和保存。"""
    output_dir = Path(output_dir).resolve()                # 当前Hydra运行的日志与检查点根目录。
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("重规划DQN配置:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    validate_training_config(cfg)
    if cfg.pretrained_ckpt_path is None:
        raise ValueError(
            "必须设置pretrained_ckpt_path，例如 "
            "pretrained_ckpt_path=/path/to/checkpoint"
        )

    seed_everything(int(cfg.seed))
    device = resolve_device(str(cfg.device))              # DQN和冻结扩散策略共同使用的设备。
    policy, saved_policy_config, _, pretrained_model_dir = load_frozen_policy(  # 预训练策略及来源信息。
        cfg.pretrained_ckpt_path,
        device,
    )
    horizon = int(policy.config.horizon)                   # 预训练策略完整预测动作长度，计划值为16。
    arm_action_dim = int(policy.diffusion.arm_action_dim)  # 每步操作双臂动作维数，当前为14。
    view_action_dim = int(policy.diffusion.view_action_dim)# 每步主动视角动作维数，当前为6。
    robot_state_dim = int(                                # DQN接收的归一化机器人状态维数。
        policy.config.input_shapes["observation.state"][0]
    )
    if horizon != 16:
        logging.warning("当前预训练策略horizon=%d，不是计划中的16步。", horizon)

    train_env, env_id = make_training_env(                # 产生回放数据的单个MuJoCo训练环境。
        policy,
        saved_policy_config,
        cfg.env,
    )
    eval_env, _ = make_training_env(                      # 与训练episode隔离的确定性评估环境。
        policy,
        saved_policy_config,
        cfg.env,
    )

    probe_observation, _ = train_env.reset(               # 只用于探测冻结视觉编码器输出维数。
        seed=int(cfg.seed)
    )
    probe_current = prepare_normalized_observation(       # 带batch维的首帧归一化探测观测。
        probe_observation,
        policy,
        device,
    )
    probe_history = NormalizedObservationHistory(         # 按策略要求复制首帧填满历史。
        policy.config.n_obs_steps
    )
    probe_history.reset(probe_current)
    visual_feature_dim = int(                             # 两帧、多相机冻结视觉特征总维数。
        extract_visual_features(policy, probe_history.stacked()).shape[-1]
    )
    dqn_config, reward_config, trainer, replay_buffer, collector = (  # DQN训练的全部独立组件。
        create_dqn_components(
            cfg,
            visual_feature_dim,
            robot_state_dim,
            arm_action_dim,
            view_action_dim,
            horizon,
            device,
        )
    )
    total_env_steps, episode_index = restore_dqn_checkpoint(  # 从零开始或恢复全局训练进度。
        cfg.resume_dqn_path,
        trainer,
        dqn_config,
        device,
    )
    if bool(cfg.training.view_only_available):
        raise NotImplementedError(
            "训练入口尚未实现View-only条件扩散，请先设置"
            "training.view_only_available=false"
        )

    wandb_run = maybe_init_wandb(cfg, output_dir)          # 可选W&B运行对象；关闭时为None。
    metrics_path = output_dir / "metrics.jsonl"           # 始终保存的本地逐行JSON指标文件。
    train_generator = make_diffusion_generator(           # 训练rollout专用的扩散噪声随机流。
        device,
        int(cfg.seed) + 1,
    )
    exploration_rng = random.Random(int(cfg.seed) + 2)    # 预热决策和epsilon探索之外的Python随机流。
    last_train_metrics: dict[str, float] = {}             # 最近一次TD更新与Q值诊断结果。
    next_eval_step = (                                    # 大于当前恢复步数的下一个评估触发点。
        (total_env_steps // int(cfg.training.eval_freq_steps)) + 1
    ) * int(cfg.training.eval_freq_steps)
    next_save_step = (                                    # 大于当前恢复步数的下一个保存触发点。
        (total_env_steps // int(cfg.training.save_freq_steps)) + 1
    ) * int(cfg.training.save_freq_steps)

    logging.info(
        "开始训练重规划DQN: env=%s, visual_dim=%d, state_dim=%d, "
        "arm/view=%d/%d, horizon=%d, device=%s",
        env_id,
        visual_feature_dim,
        robot_state_dim,
        arm_action_dim,
        view_action_dim,
        horizon,
        device,
    )

    try:
        while total_env_steps < int(cfg.training.total_env_steps):
            episode_seed = int(cfg.seed) + episode_index  # 每个训练episode独立且可复现的派生种子。
            observation, _ = train_env.reset(             # 当前episode的Gym原始首帧。
                seed=episode_seed
            )
            if hasattr(train_env.action_space, "seed"):
                train_env.action_space.seed(episode_seed)
            current = prepare_normalized_observation(     # 送入冻结策略和DQN的归一化首帧。
                observation,
                policy,
                device,
            )
            history = NormalizedObservationHistory(      # 当前episode独立的两帧观测队列。
                policy.config.n_obs_steps
            )
            history.reset(current)
            cache = ReplanningActionCache(dqn_config)     # 当前episode独立的16步双头动作缓存。
            history_batch = history.stacked()             # 当前决策对应的两帧观测历史。
            state = build_replanning_state(               # 首个DQN状态；后续直接复用上一transition的next_state。
                extract_visual_features(policy, history_batch),
                current_robot_state(history_batch),
                cache,
            )
            episode_env_return = 0.0                      # 未缩放的环境累计奖励。
            episode_dqn_return = 0.0                      # 加入推理成本后的DQN累计奖励。
            episode_steps = 0                             # 当前episode已执行的物理环境步数。
            episode_replans = 0                           # 当前episode联合扩散推理次数。
            episode_inference_times = []                  # 当前episode每次扩散推理耗时列表。
            episode_success = False                       # 当前episode是否曾达到成功条件。

            while (
                episode_steps < int(cfg.env.max_episode_steps)
                and total_env_steps < int(cfg.training.total_env_steps)
            ):
                action_mask = collector.build_action_mask(  # [1,3]当前缓存状态允许的高层决策。
                    cache,
                    view_only_available=False,
                    min_steps_after_replan=int(
                        cfg.training.min_steps_after_replan
                    ),
                )
                epsilon = linear_epsilon(                  # 当前全局环境步对应的探索概率。
                    total_env_steps,
                    float(cfg.training.epsilon_start),
                    float(cfg.training.epsilon_end),
                    int(cfg.training.epsilon_decay_steps),
                )
                decision_id, q_values = choose_dqn_decision(  # 预热随机或DQN epsilon-greedy决策。
                    trainer.online_network,
                    state,
                    action_mask,
                    len(replay_buffer),
                    int(cfg.training.learning_starts),
                    epsilon,
                    float(cfg.training.warmup_continue_probability),
                    exploration_rng,
                )
                decision = ReplanningDecision(decision_id) # 具名的继续/单View/联合重规划动作。
                if decision == ReplanningDecision.VIEW_ONLY_REPLAN:
                    raise RuntimeError("动作掩码错误地开放了View-only重规划")

                previous_arm_action = None                 # 重规划前旧缓存的下一条归一化Arm动作。
                new_arm_action = None                      # 重规划后新缓存的第一条归一化Arm动作。
                if (
                    decision == ReplanningDecision.JOINT_REPLAN
                    and cache.has_remaining
                ):
                    previous_arm_action = (
                        cache.peek_normalized_arm_action().detach().clone()
                    )
                if decision == ReplanningDecision.JOINT_REPLAN:
                    normalized_chunk, env_chunk, inference_ms = (  # 新的归一化/物理16步动作及耗时。
                        infer_full_joint_chunk(
                            policy,
                            history_batch,
                            train_generator,
                        )
                    )
                    cache.replace_joint(normalized_chunk, env_chunk)
                    episode_replans += 1
                    episode_inference_times.append(inference_ms)
                    if previous_arm_action is not None:
                        new_arm_action = (
                            cache.peek_normalized_arm_action().detach().clone()
                        )

                env_action = (                            # 当前实际送入MuJoCo的20维物理动作。
                    cache.peek_env_action().detach().cpu().numpy()
                )
                next_observation, env_reward, terminated, truncated, info = (
                    train_env.step(env_action)
                )
                cache.advance()
                episode_steps += 1
                total_env_steps += 1
                forced_truncation = (                     # 训练入口自身达到episode步数上限。
                    episode_steps >= int(cfg.env.max_episode_steps)
                )
                done = bool(                              # DQN停止对下一状态进行价值自举的终止位。
                    terminated or truncated or forced_truncation
                )
                episode_env_return += float(env_reward)
                episode_success = episode_success or bool(
                    info.get("is_success", False)
                )

                next_current = prepare_normalized_observation(  # 执行动作后真实新观测的归一化结果。
                    next_observation,
                    policy,
                    device,
                )
                history.append(next_current)
                next_history_batch = history.stacked()    # 新观测加入后的[B,S,...]历史。
                next_state = build_replanning_state(      # transition中的DQN下一状态。
                    extract_visual_features(policy, next_history_batch),
                    current_robot_state(next_history_batch),
                    cache,
                )
                next_action_mask = collector.build_action_mask(  # 下一状态TD目标允许选择的动作。
                    cache,
                    view_only_available=False,
                    min_steps_after_replan=int(
                        cfg.training.min_steps_after_replan
                    ),
                )
                dqn_reward, _ = collector.add_step(       # 合成奖励并立即写入一条回放transition。
                    state=state,
                    decision=decision,
                    env_reward=float(env_reward),
                    next_state=next_state,
                    done=done,
                    next_action_mask=next_action_mask,
                    previous_arm_action=previous_arm_action,
                    new_arm_action=new_arm_action,
                )
                episode_dqn_return += dqn_reward

                ready_to_train = len(replay_buffer) >= max(  # 预热量和batch量均满足后才开始更新。
                    int(cfg.training.learning_starts),
                    int(cfg.training.batch_size),
                )
                if (
                    ready_to_train
                    and total_env_steps % int(cfg.training.train_frequency) == 0
                ):
                    for _ in range(int(cfg.training.updates_per_step)):
                        replay_batch = replay_buffer.sample(  # 从CPU回放均匀采样并搬到训练设备。
                            int(cfg.training.batch_size),
                            device=device,
                        )
                        last_train_metrics = trainer.train_step(replay_batch)

                history_batch = next_history_batch                # 下一步扩散重规划复用已堆叠的观测历史。
                state = next_state                                # 避免下一步重复提取相同RGB特征。
                if q_values is not None:
                    last_train_metrics.update(
                        {
                            "q_continue": float(
                                q_values[0, ReplanningDecision.CONTINUE]
                            ),
                            "q_view_only_replan": float(
                                q_values[0, ReplanningDecision.VIEW_ONLY_REPLAN]
                            ),
                            "q_joint_replan": float(
                                q_values[0, ReplanningDecision.JOINT_REPLAN]
                            ),
                        }
                    )
                if done:
                    break

            episode_index += 1
            decision_metrics = collector.decision_fractions()  # 从训练开始累计的三类决策比例。
            train_metrics = {                              # 当前episode、本地JSONL和W&B统一指标。
                "step": total_env_steps,                           # 全局物理环境步。
                "episode": episode_index,                          # 已完成训练回合数。
                "train/episode_success": float(episode_success),   # 当前回合是否成功。
                "train/episode_env_return": episode_env_return,    # 当前回合原始环境回报。
                "train/episode_dqn_return": episode_dqn_return,    # 扣除重规划成本后的回报。
                "train/episode_steps": episode_steps,              # 当前回合实际执行步数。
                "train/joint_replans": episode_replans,            # 当前回合联合重规划次数。
                "train/average_execution_length": (
                    float(episode_steps) / float(max(1, episode_replans))
                ),                                                   # 每次联合推理平均支撑步数。
                "train/average_inference_ms": (
                    float(np.mean(episode_inference_times))
                    if episode_inference_times
                    else 0.0
                ),                                                   # 当前回合平均扩散推理耗时。
                "train/epsilon": epsilon,                           # 当前epsilon探索率。
                "buffer/size": len(replay_buffer),                  # 当前有效回放transition数。
                **{
                    f"decision/{key}": value
                    for key, value in decision_metrics.items()
                },
                **{
                    f"dqn/{key}": value
                    for key, value in last_train_metrics.items()
                },
            }
            append_json_metrics(metrics_path, train_metrics)
            if wandb_run is not None:
                wandb_run.log(train_metrics, step=total_env_steps)
            if episode_index % int(cfg.training.log_freq_episodes) == 0:
                logging.info(
                    "episode=%d step=%d success=%s env_return=%.2f "
                    "dqn_return=%.3f replans=%d avg_chunk=%.2f "
                    "buffer=%d epsilon=%.3f loss=%s",
                    episode_index,
                    total_env_steps,
                    episode_success,
                    episode_env_return,
                    episode_dqn_return,
                    episode_replans,
                    train_metrics["train/average_execution_length"],
                    len(replay_buffer),
                    epsilon,
                    last_train_metrics.get("loss", "warmup"),
                )

            if total_env_steps >= next_eval_step:
                eval_metrics = evaluate_replanning_policy(  # 无epsilon探索的独立评估结果。
                    policy,
                    trainer.online_network,
                    eval_env,
                    collector,
                    dqn_config,
                    cfg,
                    device,
                    total_env_steps,
                )
                eval_record = {                            # 为本地JSONL和W&B补充全局训练坐标。
                    "step": total_env_steps,              # 触发本轮评估的全局环境步。
                    "episode": episode_index,             # 触发本轮评估的训练episode。
                    **eval_metrics,
                }
                append_json_metrics(metrics_path, eval_record)
                if wandb_run is not None:
                    wandb_run.log(eval_record, step=total_env_steps)
                logging.info("DQN评估结果: %s", eval_metrics)
                next_eval_step += int(cfg.training.eval_freq_steps)

            if total_env_steps >= next_save_step:
                saved_path = save_dqn_checkpoint(         # 周期性latest.pt和带步数快照路径。
                    output_dir,
                    trainer,
                    dqn_config,
                    reward_config,
                    total_env_steps,
                    episode_index,
                    pretrained_model_dir,
                    save_step_copy=True,
                )
                logging.info("DQN检查点已保存: %s", saved_path)
                next_save_step += int(cfg.training.save_freq_steps)

    finally:
        final_path = save_dqn_checkpoint(                 # 正常结束或异常退出时最后一次保存。
            output_dir,
            trainer,
            dqn_config,
            reward_config,
            total_env_steps,
            episode_index,
            pretrained_model_dir,
            save_step_copy=False,
        )
        logging.info("DQN最终状态已保存: %s", final_path)
        train_env.close()
        eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()


@hydra.main(
    version_base="1.2",
    config_path="../../configs/adaptive_replanning",
    config_name="default",
)
def train_cli(cfg: DictConfig) -> None:
    """Hydra命令行入口。"""
    output_dir = (                                        # Hydra为本次任务创建的唯一输出目录。
        hydra.core.hydra_config.HydraConfig.get().run.dir
    )
    train_replanning_dqn(cfg, output_dir)


if __name__ == "__main__":
    # 常用命令行参数集中放在这里；显式传入同名Hydra参数时，命令行值优先。
    default_args = [
        "pretrained_ckpt_path='outputs/2_pretrain/train/2026-07-15/12-01-50_InsertCylinder-3Arms-v0_pre_zed_coupled_dual_head_diffusion/checkpoints/030000_loss=0.0136_sr=2.0_ar=-154.44'",  # 冻结的底层双头策略。
        "device=cuda:0",                                 # 扩散策略和DQN共同使用的设备。
        "training.total_env_steps=200000",               # 正式训练采集的总物理环境步数。
        "training.replay_capacity=100000",               # CPU回放缓冲区最大transition数。
        "training.learning_starts=5000",                 # 开始Double DQN更新前的预热数据量。
        "training.batch_size=256",                       # 每次TD更新从回放采样的transition数。
        "training.epsilon_start=0.20",                   # 预热结束后的初始epsilon探索率。
        "training.epsilon_end=0.02",                     # 线性衰减完成后的最小epsilon。
        "training.epsilon_decay_steps=100000",           # epsilon线性衰减持续的环境步数。
        "training.min_steps_after_replan=2",             # 每次联合推理后强制继续执行的最少步数。
        "training.eval_freq_steps=10000",                # 动态重规划策略的评估间隔。
        "training.save_freq_steps=10000",                # DQN带步数检查点的保存间隔。
        "eval.n_episodes=10",                            # 每轮无探索评估的episode数量。
        "wandb.enable=true",                             # 是否上传训练和评估指标到W&B。
        "training.view_only_available=false",            # View-only条件扩散未实现前必须保持关闭。
        # 恢复DQN时取消下一行注释并填写已有latest.pt；回放缓冲区会重新预热。
        # "resume_dqn_path='outputs/4_replanning_dqn/train/.../checkpoints/latest.pt'",
    ]

    # 查看Hydra配置或帮助时不注入训练参数，避免仅检查配置却加载默认checkpoint。
    hydra_inspection_flags = {                            # 仅查看配置或帮助、不启动训练的命令参数。
        "--cfg",                                         # 打印Hydra组合后的配置。
        "--info",                                        # 打印Hydra运行信息。
        "--hydra-help",                                  # 打印Hydra专用帮助。
        "--help",                                        # 打印应用帮助。
        "-h",                                            # help的短参数形式。
        "--version",                                     # 打印版本信息。
    }
    if not any(flag in sys.argv for flag in hydra_inspection_flags):
        for arg in default_args:
            arg_key = arg.split("=", 1)[0]               # 当前默认参数的Hydra键名。
            if not any(
                sys_arg.split("=", 1)[0] == arg_key
                for sys_arg in sys.argv[1:]
            ):
                sys.argv.append(arg)

    train_cli()
