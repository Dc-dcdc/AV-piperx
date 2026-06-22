import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # 开启 PyTorch CUDA 显存动态分片 + 可扩展内存分配，不再一次性预占用大块连续显存
# 没有显示器时使用，比如在服务器上
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["EGL_DEVICE_ID"] = "0"
import sys
import math
import json
import torch
import numpy as np
import logging
import einops
import tempfile
import hydra
import yaml
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from lerobot.common.utils.utils import init_logging, set_global_seed
from pprint import pformat
from lerobot.common.logger import Logger
from tqdm import tqdm
from collections import deque
from lerobot.common.policies.factory import make_policy
# 路径处理：当前文件位于 train/finetune，项目根目录需要向上两级。
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import env.task.sim_envs
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
)
from train.finetune.critic import ImageCritic, SharedFeatureCritic
from train.pretrain.eval_train import custom_eval_policy, TopKCheckpointManager
from contextlib import nullcontext

import torch
from types import MethodType


def deep_update_dict(base: dict, override: dict) -> dict:
    """递归合并配置字典；override 中的值优先。"""
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def compute_value_diagnostics(values, returns, eps: float = 1e-8):
    """计算 critic 的解释方差和 value/return 相关性。"""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)

    if values.size == 0 or returns.size == 0:
        return float("nan"), float("nan")

    return_var = np.var(returns)
    if return_var < eps:
        explained_variance = float("nan")
    else:
        explained_variance = 1.0 - np.var(returns - values) / (return_var + eps)

    value_std = np.std(values)
    return_std = np.std(returns)
    if values.size < 2 or value_std < eps or return_std < eps:
        value_return_corr = float("nan")
    else:
        value_return_corr = float(np.corrcoef(values, returns)[0, 1])

    return float(explained_variance), value_return_corr


def init_logprob_advantage_stats():
    """初始化 PPO 更新方向诊断的累计量。"""
    return {
        "n": 0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
        "pos_n": 0,
        "pos_logratio_sum": 0.0,
        "neg_n": 0,
        "neg_logratio_sum": 0.0,
        "sign_n": 0,
        "sign_agree": 0,
    }


@torch.no_grad()
def update_logprob_advantage_stats(stats, log_ratio, advantages, eps: float = 1e-12):
    """累计 logprob 变化和 advantage 的相关性统计。"""
    x = log_ratio.detach().float().reshape(-1)
    y = advantages.detach().float().reshape(-1)
    finite_mask = torch.isfinite(x) & torch.isfinite(y)
    if not finite_mask.any():
        return

    x = x[finite_mask]
    y = y[finite_mask]
    stats["n"] += int(x.numel())
    stats["sum_x"] += float(x.sum().item())
    stats["sum_y"] += float(y.sum().item())
    stats["sum_x2"] += float((x * x).sum().item())
    stats["sum_y2"] += float((y * y).sum().item())
    stats["sum_xy"] += float((x * y).sum().item())

    pos_mask = y > eps
    if pos_mask.any():
        stats["pos_n"] += int(pos_mask.sum().item())
        stats["pos_logratio_sum"] += float(x[pos_mask].sum().item())

    neg_mask = y < -eps
    if neg_mask.any():
        stats["neg_n"] += int(neg_mask.sum().item())
        stats["neg_logratio_sum"] += float(x[neg_mask].sum().item())

    sign_mask = torch.abs(y) > eps
    if sign_mask.any():
        stats["sign_n"] += int(sign_mask.sum().item())
        stats["sign_agree"] += int(((x[sign_mask] * y[sign_mask]) > 0).sum().item())


def finalize_logprob_advantage_stats(stats, eps: float = 1e-12):
    """输出 PPO 更新方向诊断指标。"""
    n = stats["n"]
    if n < 2:
        corr = float("nan")
    else:
        mean_x = stats["sum_x"] / n
        mean_y = stats["sum_y"] / n
        var_x = stats["sum_x2"] / n - mean_x * mean_x
        var_y = stats["sum_y2"] / n - mean_y * mean_y
        cov_xy = stats["sum_xy"] / n - mean_x * mean_y
        if var_x <= eps or var_y <= eps:
            corr = float("nan")
        else:
            corr = cov_xy / math.sqrt(var_x * var_y)

    pos_mean = (
        stats["pos_logratio_sum"] / stats["pos_n"]
        if stats["pos_n"] > 0
        else float("nan")
    )
    neg_mean = (
        stats["neg_logratio_sum"] / stats["neg_n"]
        if stats["neg_n"] > 0
        else float("nan")
    )
    sign_agreement = (
        stats["sign_agree"] / stats["sign_n"]
        if stats["sign_n"] > 0
        else float("nan")
    )
    return corr, pos_mean, neg_mean, sign_agreement


class RunningMeanStd:
    """原版 DPPO 用的运行均值方差统计器。"""

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / (total_count - 1)
        self.count = total_count


def backward_discounted_sum(prevret, reward, first, gamma):
    """按原版 DPPO 的 RunningRewardScaler 方式累计折扣回报。"""
    assert first.ndim == 2
    _, nstep = reward.shape
    ret = np.zeros_like(reward)
    for t in range(nstep):
        prevret = ret[:, t] = reward[:, t] + (1 - first[:, t]) * gamma * prevret
    return ret


class RunningRewardScaler:
    """用折扣回报的运行方差缩放 reward，保持 PPO/Critic 目标量级稳定。"""

    def __init__(self, num_envs, cliprew=10.0, gamma=0.99, epsilon=1e-8, per_env=False):
        ret_rms_shape = (num_envs,) if per_env else ()
        self.ret_rms = RunningMeanStd(shape=ret_rms_shape)
        self.cliprew = cliprew
        self.ret = np.zeros(num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.per_env = per_env

    def __call__(self, reward, first):
        rets = backward_discounted_sum(
            prevret=self.ret,
            reward=reward,
            first=first,
            gamma=self.gamma,
        )
        self.ret = rets[:, -1]
        self.ret_rms.update(rets if self.per_env else rets.reshape(-1))
        return self.transform(reward)

    def transform(self, reward):
        return np.clip(
            reward / np.sqrt(self.ret_rms.var + self.epsilon),
            -self.cliprew,
            self.cliprew,
        )


@torch.no_grad()
def forward_dppo(self, cond: dict, return_chain=True):
    """
    专为 LeRobot DiffusionPolicy 定制的 DPPO 推理函数。
    cond 必须已经是 [B, n_obs_steps, ...] 的显式历史观测窗口。
    """
    self.eval()

    # ==========================================
    # 1. 观测输入归一化与相机堆叠
    # ==========================================
    batch = self.normalize_inputs(cond.copy())
    # 按照 LeRobot 源码将多个相机的图像堆叠在一个张量里
    if len(self.expected_image_keys) > 0:
        batch = dict(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

    stacked_batch = batch
    # 获取动态的 batch_size (即并行的 n_envs 个数) 和设备
    batch_size = next(iter(stacked_batch.values())).shape[0]
    device = get_device_from_parameters(self)
    dtype = get_dtype_from_parameters(self)

    # ==========================================
    # 2. 提取视觉和状态的全局条件 (Global Conditioning)
    # ==========================================
    # 🌟 修正点：直接调用 DiffusionModel 内置的完美处理函数
    global_cond = self.diffusion._prepare_global_conditioning(stacked_batch)

    # ==========================================
    # 3. 初始化纯噪声 [Batch, Horizon, Action_Dim]
    # ==========================================
    action_dim = self.config.output_shapes["action"][0]
    trajectory = torch.randn(
        size=(batch_size, self.config.horizon, action_dim),
        dtype=dtype,
        device=device
    )

    # ==========================================
    # 4. 配置 Scheduler 步数并计算截断起点
    # ==========================================
    num_inference_steps = self.diffusion.num_inference_steps
    self.diffusion.noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = self.diffusion.noise_scheduler.timesteps

    ft_denoising_steps = getattr(self.config, "ft_denoising_steps", 10)
    record_start_idx = len(timesteps) - ft_denoising_steps
    if record_start_idx < 0:
        record_start_idx = 0

    chains = []

    for i, t in enumerate(timesteps):
        # 🌟 保存当前的带噪状态 x_t
        if return_chain and i >= record_start_idx:
            chains.append(trajectory.clone())

        timestep_tensor = torch.full(trajectory.shape[:1], t, dtype=torch.long, device=device)
        model_output = self.diffusion.unet(
            trajectory,
            timestep_tensor,
            global_cond=global_cond
        )

        # 🌟 核心修复 1：抛弃黑盒 scheduler，使用与 get_logprobs 100% 对齐的 DDIM 数学公式！
        alphas_cumprod = self.diffusion.noise_scheduler.alphas_cumprod.to(device)
        alpha_prod_t = alphas_cumprod[t].view(-1, 1, 1)

        step_ratio = self.diffusion.noise_scheduler.config.num_train_timesteps // self.diffusion.num_inference_steps
        prev_t = t - step_ratio

        alpha_prod_t_prev = torch.where(
            prev_t >= 0,
            alphas_cumprod[torch.clamp(prev_t, min=0)],
            torch.tensor(1.0, device=device, dtype=trajectory.dtype)
        ).view(-1, 1, 1)

        # 严格计算 DDIM 均值 (mu)
        pred_original_sample = (trajectory - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        mu = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + torch.sqrt(1 - alpha_prod_t_prev) * model_output

        # 训练 rollout 保留中间去噪探索，但最终动作不再额外加噪声。
        # eval_train.py/select_action 走的是 scheduler 的最终输出，最后一步继续加噪会造成训练/评估分布不一致。
        if i == len(timesteps) - 1:
            trajectory = mu
        else:
            std = getattr(self.config, "min_sampling_denoising_std", 0.04)
            trajectory = mu + torch.randn_like(mu) * std

    if return_chain and len(chains) < ft_denoising_steps + 1:
        chains.append(trajectory.clone())

    # 堆叠维度：[Batch, ft_denoising_steps + 1, Horizon, Action_Dim]
    chains_tensor = torch.stack(chains, dim=1) if return_chain else None

    # ==========================================
    # 6. 反归一化并输出完整的 Horizon
    # ==========================================
    out_dict = self.unnormalize_outputs({"action": trajectory})
    final_actions = out_dict["action"]

    return {
        "actions": final_actions,       # 完整的预测动作序列 [Batch, Horizon, Action_Dim]
        "chains": chains_tensor         # 去噪链，+1是因为保存了纯噪声  [Batch, ft_denoising_steps + 1, Horizon, Action_Dim]
    }


# ==========================================
# 🌟  定义 PPO 概率计算函数
# ==========================================
def get_logprobs(self, cond: dict, x_t: torch.Tensor, x_t_1: torch.Tensor, timesteps: torch.Tensor, return_global_cond=False):
    """
    计算扩散模型从 x_t 转移到 x_{t-1} 的对数概率 (Log-Likelihood)。
    基于 DDIM (预测 Epsilon) 的数学展开，使用DDPM需要重新变化实现公式。
    """
    # 1. 提取条件特征 (复用 LeRobot 底层逻辑)，包括视觉和状态
    batch = self.normalize_inputs(cond.copy())
    # 堆叠后形状完美契合 LeRobot 底层要求: [Batch, Time, Num_Cams, C, H, W]
    if len(self.expected_image_keys) > 0:
        batch = dict(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
    # 这里的 global_cond 就是带着 Actor 视觉权重的特征
    global_cond = self.diffusion._prepare_global_conditioning(batch)

    # 2. 预测x_t中的噪声 (Epsilon)
    noise_pred = self.diffusion.unet(x_t, timesteps, global_cond=global_cond)

    # 3. 计算 DDIM 确定的均值 (mu)

    alphas_cumprod = self.diffusion.noise_scheduler.alphas_cumprod.to(x_t.device)

    # 3.1： 提取当前步的 alpha (注意对齐形状以支持广播)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1)

    # 3.2： 动态计算 DDIM 的真实“上一步”时间
    # 根据配置的 训练总步数 和 实际推理步数 算出跳跃步长
    scheduler = self.diffusion.noise_scheduler
    # 🌟 修复：防御 actor_ref 未初始化的 NoneType 报错
    if scheduler.num_inference_steps is None:
        scheduler.set_timesteps(self.diffusion.num_inference_steps)
    step_ratio = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timesteps = timesteps - step_ratio

    # 3.3： 提取上一步的 alpha
    # 注意细节：当 prev_timesteps < 0 时（也就是最后一步），意味着要抵达完全无噪的 x_0
    # 在数学上，x_0 的 alpha_cumprod 应该绝对等于 1.0
    alpha_prod_t_prev = torch.where(
        prev_timesteps >= 0,
        alphas_cumprod[torch.clamp(prev_timesteps, min=0)],
        torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)
    ).view(-1, 1, 1)

    # 4. DDIM 核心推导公式：
    # 步骤 A：预测出纯净的 x_0 (Pred Original Sample)
    pred_original_sample = (x_t - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)

    # 步骤 B：用 x_0 和 epsilon 重新组合出 DDIM 路径上的上一帧均值 (mu)
    mu = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + torch.sqrt(1 - alpha_prod_t_prev) * noise_pred

    # 5. 强制注入 RL 探索方差 (因为 DDIM 默认方差为0)
    std = getattr(self.config, "min_logprob_denoising_std", 0.04)
    var = std ** 2

    # 6. 高斯分布的对数概率公式: -0.5 * ((x - mu)/std)^2 - log(std) - 0.5 * log(2*pi)
    log_prob = -0.5 * ((x_t_1 - mu) ** 2) / var - math.log(std) - 0.5 * math.log(2 * math.pi)

    # ==========================================
    # 🌟 官方对齐 1：Action Chunking 真实执行步数截断
    # ==========================================
    act_steps = getattr(self.config, "n_action_steps", 8)
    n_obs_steps = getattr(self.config, "n_obs_steps", 2)
    action_start = n_obs_steps - 1
    action_end = action_start + act_steps

    log_prob = log_prob[:, action_start:action_end, :]

    # ==========================================
    # 🌟 官方对齐 2：概率限幅 (防止极端负数导致梯度崩塌)
    # ==========================================
    log_prob = torch.clamp(log_prob, min=-5.0, max=2.0)

    # ==========================================
    # 原版 DPPO 使用 action chunk 内所有维度 logprob 的均值，保持 PPO ratio 尺度稳定。
    # 如需做联合概率实验，可在配置里设为 "sum"。
    # ==========================================
    logprob_reduction = getattr(self.config, "logprob_reduction", "mean")
    if logprob_reduction == "sum":
        log_prob = log_prob.sum(dim=(-1, -2))
    elif logprob_reduction == "mean":
        log_prob = log_prob.mean(dim=(-1, -2))
    else:
        raise ValueError(f"未知 logprob_reduction={logprob_reduction!r}，可选: 'sum' 或 'mean'")

    # 按需返回特征，用于评价网络的resnet视觉底座
    if return_global_cond:
        return log_prob, global_cond
    return log_prob

def train_dppo_finetune(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    """
    DPPO 第二阶段：在预训练参数上采用PPO算法进行微调
    """
    # ==========================================
    # 1. 基础配置与日志初始化
    # ==========================================
    init_logging()
    logging.info("  启动 DPPO 微调程序...")
    logging.info(f"配置参数:\n{pformat(OmegaConf.to_container(cfg))}")

    # 初始化日志记录器与全局随机种子
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)
    set_global_seed(cfg.seed)

    # 获取设备
    device = get_safe_torch_device(cfg.device, log=True)
    logging.info(f"  运行设备已绑定: {device}")

    # ==========================================
    # 2. 权重路径检测与 Actor 网络加载
    # ==========================================
    ckpt_path = cfg.training.pretrained_ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 找不到权重路径: {ckpt_path}\n请检查路径是否正确。")

    # 自动探测 LeRobot 的 pretrained_model 子文件夹
    hf_model_dir = os.path.join(ckpt_path, "pretrained_model")
    if os.path.exists(hf_model_dir):
        logging.info(f"🔍 检测到 LeRobot 标准快照结构，将自动读取子目录: pretrained_model")
        load_dir = hf_model_dir
    else:
        load_dir = ckpt_path # 兼容其他保存格式
    logging.info(f"  正在从目录重建网络并加载权重: {load_dir}")

    hydra_cfg = None
    try:
        from pathlib import Path
        from lerobot.common.utils.utils import init_hydra_config

        # 1. 寻找 config.yaml 路径
        config_yaml_path = Path(load_dir) / "config.yaml"
        if not config_yaml_path.exists():
            config_yaml_path = Path(load_dir).parent / "config.yaml"

        if not config_yaml_path.exists():
            raise FileNotFoundError(f"找不到 config.yaml，无法初始化 hydra_cfg！")

        # 2. 根据 yaml 文件初始化 hydra_cfg 对象
        hydra_cfg = init_hydra_config(str(config_yaml_path))

        # 3. 🌟 核心：直接使用 make_policy，让框架接管底层张量与 EMA 加载
        actor = make_policy(
            hydra_cfg=hydra_cfg,
            pretrained_policy_name_or_path=str(load_dir)
        )

        logging.info("  成功使用 make_policy 加载策略！底层 Normalizer 与平滑权重已自动生效。")

        # 单独注入 DPPO 微调需要的超参数，保持采样分布和 logprob 分布一致。
        min_sampling_std = float(getattr(cfg.training, "min_sampling_denoising_std", 0.04))
        min_logprob_std = float(getattr(cfg.training, "min_logprob_denoising_std", min_sampling_std))
        logprob_reduction = str(getattr(cfg.training, "logprob_reduction", "mean"))
        actor.config.min_sampling_denoising_std = min_sampling_std
        actor.config.min_logprob_denoising_std = min_logprob_std
        actor.config.logprob_reduction = logprob_reduction
        actor.config.ft_denoising_steps = int(
            getattr(cfg.policy, "ft_denoising_steps", getattr(actor.config, "ft_denoising_steps", 5))
        )
        actor.config.n_action_steps = int(
            getattr(cfg.policy, "n_action_steps", getattr(actor.config, "n_action_steps", 8))
        )

        logging.info("  成功将微调配置 (YAML) 注入到 Actor 内部 config 中！")

        # 动态挂载 DPPO 专用的前向与概率计算函数
        actor.forward_dppo = MethodType(forward_dppo, actor) # 将forward_dppo绑定到实例中
        actor.get_logprobs = MethodType(get_logprobs, actor) # 将get_logprobs绑定到实例中
        logging.info("  成功加载 Actor (DiffusionPolicy) 并挂载 DPPO 专用接口！")

    except Exception as e:
        raise RuntimeError(f"❌ 权重加载失败！详细报错: {e}")

    # ==========================================
    # 3. 读取预训练配置文件中输入输出配置，保证与环境对齐
    # ==========================================
    ref_cams = [k.replace("observation.images.", "") for k in actor.config.input_shapes.keys() if "observation.images." in k]
    horizon_steps = getattr(actor.config, "horizon", None)
    action_dim = actor.config.output_shapes.get("action", [None])[-1]
    state_dim = actor.config.input_shapes.get("observation.state", [None])[0]
    # 优化 1：更严谨的校验，允许纯视觉策略 (state_dim=None)
    if not ref_cams or horizon_steps is None or action_dim is None:
        raise ValueError(f"❌ 严重冲突：模型快照中缺少关键参数 (ref_cams={ref_cams}, horizon={horizon_steps}, action_dim={action_dim})。")

    if state_dim is None:
        logging.warning("⚠️ 模型配置中未检测到 observation.state，如果这是纯视觉策略，请忽略此警告。")

    # 动态读取环境配置
    config_yaml_path = Path(load_dir) / "config.yaml"
    if config_yaml_path.exists():
        with open(config_yaml_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            # 从 YAML 字典中提取 env_name 和 env_task
            env_cfg = full_cfg.get("env", {})
            env_name = env_cfg.get("name")
            env_task = env_cfg.get("task")

            if not env_name or not env_task:
                raise ValueError("❌ config.yaml 中缺少环境的 name 或 task 字段！")
    else:
        raise ValueError(f"❌ 严重错误: 在 {load_dir} 中未找到 config.yaml，无法确定微调环境配置！")

    env_id = f"{env_name}/{env_task}"

    logging.info(f"🔄 准备通过 Gym 注册表构建环境: {env_id}")
    # ==========================================
    # 4. 初始化环境与 Critic
    # ==========================================

    # ---------------------------------------------------------
    # 4.1 提取并清洗相机参数 (防止单一字符串被误拆为字母列表)
    # ---------------------------------------------------------
    n_envs = getattr(cfg.env, "n_envs", 1)
    render_cams = getattr(cfg.eval, "render_camera", [])

    # 安全处理：如果是纯字符串 "top"，转换为 ["top"]；如果是列表则保持；None 则设为空列表
    if render_cams is None:
        render_cams = []
    elif isinstance(render_cams, str):
        render_cams = [render_cams]
    else:
        render_cams = list(render_cams)

    # 合并训练视角与渲染视角，并利用字典去重 (保留原始顺序)
    obs_cameras = list(dict.fromkeys(ref_cams + render_cams))
    logging.info(f"📷 最终绑定的环境相机视角: {obs_cameras}")

    # ---------------------------------------------------------
    # 4.2 启动 Gym 物理环境
    # ---------------------------------------------------------
    if n_envs > 1:
        # 使用 AsyncVectorEnv 自动拉起多个进程
        # 🌟 优化：使用 lambda 延迟初始化，保证每个进程拿到的都是绝对独立的环境实例
        env = gym.vector.AsyncVectorEnv(
            [lambda: gym.make(id=env_id, cameras=obs_cameras) for _ in range(n_envs)],
            shared_memory=True,  # 👈 开启官方内置的共享内存优化，防止传图像时卡顿
            context="spawn",     # 👈 强制安全启动子进程，防止 OpenGL/CUDA 崩溃
            autoreset_mode="SameStep",
        )
        logging.info(f"  成功启动 {n_envs} 个并行多进程环境 (AsyncVectorEnv) ...")
    else:
        env = gym.make(id=env_id, cameras=obs_cameras)
        logging.info("  成功启动单环境模式...")

    # ---------------------------------------------------------
    # 4.3 动态推断特征维度并初始化 Shared Critic
    # ---------------------------------------------------------
    # 利用预热的空输入，找 Actor 问一下它输出的 global_cond 是多少维
    with torch.no_grad():
        dummy_batch = {k: torch.zeros((1, 2, *v), device=device) for k, v in actor.config.input_shapes.items()}
        if len(actor.expected_image_keys) > 0:
            dummy_batch["observation.images"] = torch.stack([dummy_batch[k] for k in actor.expected_image_keys], dim=-4)
        dummy_cond = actor.diffusion._prepare_global_conditioning(dummy_batch)
        global_cond_dim = dummy_cond.shape[-1]

    logging.info(f"动态探测到 Actor 视觉底座输出特征维度为: {global_cond_dim}")

    actor.to(device)  # 在所有 Gym 子进程全部 Spawn 安全启动后再 手动推入 GPU
    critic = SharedFeatureCritic(global_cond_dim=global_cond_dim).to(device)
    logging.info("成功初始化 Shared Critic (完美视觉底座共享模式)！")
    import copy
    logging.info("正在克隆并冻结预训练参考模型 (Reference Model)...")
    actor_ref = copy.deepcopy(actor).to(device)
    actor_ref.forward_dppo = MethodType(forward_dppo, actor_ref)
    actor_ref.get_logprobs = MethodType(get_logprobs, actor_ref)
    actor_ref.eval()  # 永远保持 eval 模式
    for param in actor_ref.parameters():
        param.requires_grad = False  # 彻底锁死梯度
    # ---------------------------------------------------------
    # 4.4 初始化独立的评估环境与 Top-K 快照管理器
    # ---------------------------------------------------------
    logging.info("正在初始化独立的评估环境 (Eval Env)...")
    # 评估环境不需要向量化并发，只需一个单例环境即可
    eval_env = gym.make(id=env_id, cameras=obs_cameras)

    # 初始化 Top-K 快照管理器 (比如最多保留表现最好的 3 个模型)
    max_checkpoints = getattr(cfg.eval, "max_checkpoints", 5)
    records_resume = getattr(cfg.eval, "records_resume", True)
    checkpoint_metric = getattr(cfg.eval, "checkpoint_metric", "loss")
    manager = TopKCheckpointManager(out_dir=out_dir,
                                    max_keep=max_checkpoints,
                                    records_resume=records_resume,
                                    metric=checkpoint_metric)
    # ==========================================
    # 5. 初始化优化器与超参数
    # ==========================================
    def is_vision_encoder_param(param_name: str) -> bool:
        """只冻结视觉底座；不要误伤 UNet 内部的 cond_encoder / diffusion_step_encoder。"""
        name = param_name.lower()
        return (
            "rgb_encoder" in name
            or "image_encoder" in name
            or "vision_encoder" in name
            or "visual_encoder" in name
            or ("backbone" in name and "unet" not in name)
        )

    # 冻结视觉底座，只允许 RL 训练 UNet 动作去噪层。
    # 注意不要用泛化的 "encoder" 匹配，否则会冻结 UNet 的 diffusion_step_encoder 和 cond_encoder。
    frozen_vision_params = 0
    trainable_unet_params = 0
    trainable_other_params = 0
    for name, param in actor.named_parameters():
        if is_vision_encoder_param(name):
            param.requires_grad = False
            frozen_vision_params += param.numel()
        else:
            param.requires_grad = True
            if "unet" in name.lower():
                trainable_unet_params += param.numel()
            else:
                trainable_other_params += param.numel()

    # 过滤出 requires_grad=True 的参数传给优化器
    actor_trainable_params = [p for p in actor.parameters() if p.requires_grad]
    actor_optimizer = torch.optim.AdamW(actor_trainable_params, lr=cfg.training.actor_lr, weight_decay=getattr(cfg.training, "weight_decay", 1e-6))
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.training.critic_lr)
    logging.info(
        "  Actor冻结/训练参数统计: frozen_vision=%.2fM, trainable_unet=%.2fM, trainable_other=%.2fM",
        frozen_vision_params / 1e6,
        trainable_unet_params / 1e6,
        trainable_other_params / 1e6,
    )

    def snapshot_actor_state():
        return {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}

    def restore_actor_state(state_dict):
        actor.load_state_dict(state_dict, strict=True)
        actor.to(device)
        actor.reset()
        actor_optimizer.state.clear()

    best_actor_state = snapshot_actor_state()
    best_eval_reward = float("-inf")
    best_eval_success_rate = 0.0
    eval_collapse_count = 0

    # 在下方新增学习率热身调度器 (Warmup)
    from torch.optim.lr_scheduler import LinearLR
    # 前 5 轮迭代 (iters) 学习率从 10% 慢慢涨到 100%
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=0.1,
        total_iters=getattr(cfg.training, "actor_lr_warmup_iters", 5),
    )
    logging.info(f"  扩散模型调度器类型 (Scheduler Type): {actor.config.noise_scheduler_type}")
    logging.info(f"  预测目标类型 (Prediction Type): {actor.config.prediction_type}")
    # 从配置中提取 RL 收集参数 (提供后备默认值)
    n_steps = getattr(cfg.training, "rollout_steps", 300)   # 每次更新前收集的步数
    act_steps = getattr(cfg.policy, "n_action_steps", 8)
    denoising_steps = getattr(cfg.policy, "ft_denoising_steps", 10)
    n_obs_steps = getattr(actor.config, "n_obs_steps", 2)
    action_start = n_obs_steps - 1
    action_end = action_start + act_steps
    if action_end > horizon_steps:
        raise ValueError(
            f"动作切片越界: n_obs_steps={n_obs_steps}, act_steps={act_steps}, horizon={horizon_steps}"
        )
    critic_warmup_iters = getattr(cfg.training, "n_critic_warmup_itr", 5) # critic网络先默认热身 5 轮
    reward_scale_running = bool(getattr(cfg.training, "reward_scale_running", True))
    reward_scale_const = float(getattr(cfg.training, "reward_scale_const", 1.0))
    target_kl = getattr(cfg.training, "target_kl", 0.05)
    logging.info(
        "DPPO训练参数: rollout_steps=%s, act_steps=%s, action_slice=[%s:%s], ft_denoising_steps=%s, critic_warmup=%s, target_kl=%.4f, reward_scale_running=%s, reward_scale_const=%.3f",
        n_steps,
        act_steps,
        action_start,
        action_end,
        denoising_steps,
        critic_warmup_iters,
        target_kl,
        reward_scale_running,
        reward_scale_const,
    )

    # Gym 嵌套字典展平工具
    def flatten_lerobot_obs(obs_dict):
        """将环境吐出的嵌套字典翻译成 LeRobot 认识的扁平结构
            {
            "observation.images.zed": [96, 96, 3],
            "observation.images.wrist": [96, 96, 3],
            "observation.state": [0.1, 0.2, 0.5, ...]
            }
        """
        flat_obs = {}
        # 拆解图像字典
        if "pixels" in obs_dict:
            for cam_name, img_array in obs_dict["pixels"].items():
                flat_obs[f"observation.images.{cam_name}"] = img_array
        # 换名本体状态
        if "agent_pos" in obs_dict:
            flat_obs["observation.state"] = obs_dict["agent_pos"]
        # 保留环境可能吐出的其他一维信息（兜底）
        for k, v in obs_dict.items():
            if k not in ["pixels", "agent_pos"]:
                flat_obs[k] = v
        return flat_obs

    def clone_obs_value(v):
        if hasattr(v, "copy"):
            return v.copy()
        return v

    def reset_full_obs_queue(queue, obs):
        """用同一帧填满历史队列，通常用于 reset 后的第一步。"""
        for k, v in obs.items():
            if k not in queue:
                queue[k] = deque(maxlen=n_obs_steps)
            queue[k].clear()
            for _ in range(n_obs_steps):
                queue[k].append(clone_obs_value(v))

    def append_obs_queue(queue, obs):
        """每个真实 env.step 后调用一次，保证队列里的帧是相邻物理帧。"""
        for k, v in obs.items():
            if k not in queue:
                queue[k] = deque(maxlen=n_obs_steps)
                for _ in range(n_obs_steps - 1):
                    queue[k].append(clone_obs_value(v))
            queue[k].append(clone_obs_value(v))

    def reset_done_envs_in_obs_queue(queue, obs, done_mask):
        """某个环境刚 reset 时，只清洗该环境的历史帧，防止跨回合混入旧图像。"""
        done_mask = np.asarray(done_mask, dtype=bool)
        if not done_mask.any():
            return

        if n_envs == 1:
            reset_full_obs_queue(queue, obs)
            return

        for k, v in obs.items():
            if k not in queue:
                continue
            for env_idx in np.flatnonzero(done_mask):
                reset_frame = np.array(v[env_idx], copy=True)
                for q_idx in range(len(queue[k])):
                    queue[k][q_idx][env_idx] = reset_frame

    def stack_obs_queue(queue):
        """输出 [B, T, ...]，其中 T 维保存最近 n_obs_steps 个相邻观测。"""
        stacked_obs = {}
        for k, frames in queue.items():
            if len(frames) != n_obs_steps:
                raise RuntimeError(f"观测队列 {k} 长度异常: {len(frames)} != {n_obs_steps}")
            stacked_v = np.stack(list(frames), axis=0 if n_envs == 1 else 1)
            if n_envs == 1:
                stacked_v = np.expand_dims(stacked_v, axis=0)
            stacked_obs[k] = stacked_v
        return stacked_obs

    def build_select_action_obs(obs):
        """构造 select_action 需要的单帧 batch，历史帧和动作队列交给 policy 内部维护。"""
        batch = {}
        for k, v in obs.items():
            if k not in actor.config.input_shapes:
                continue
            tensor_v = torch.from_numpy(np.ascontiguousarray(v)).float().to(device)
            if n_envs == 1:
                tensor_v = tensor_v.unsqueeze(0)
            if "images" in k:
                tensor_v = tensor_v.permute(0, 3, 1, 2) / 255.0
            batch[k] = tensor_v
        return batch

    def info_success_mask(info, done_mask):
        """兼容单环境、VectorEnv普通info和SameStep final_info，提取每个完成回合是否成功。"""
        done_mask = np.asarray(done_mask, dtype=bool)
        success = np.zeros(n_envs, dtype=bool)

        def fill_success(raw_success, raw_mask):
            raw_success = np.asarray(raw_success)
            raw_mask = np.asarray(raw_mask, dtype=bool)
            if raw_success.shape == ():
                success[raw_mask] = bool(raw_success.item())
                return

            limit = min(len(raw_success), n_envs)
            valid_mask = raw_mask[:limit]
            success[:limit] = success[:limit] | (raw_success[:limit].astype(bool) & valid_mask)

        if isinstance(info, dict) and "final_info" in info:
            # SameStep autoreset 会先 step 到 done，再立刻 reset。
            # 此时终止步的真实 info 在 final_info 里；顶层 is_success 往往来自未结束环境/新 reset 信息。
            final_info = info["final_info"]
            final_mask = np.asarray(info.get("_final_info", done_mask), dtype=bool)

            # Gymnasium VectorEnv 会把 final_info 递归合并为 dict-of-arrays:
            # {"final_info": {"is_success": array([...]), "_is_success": mask, ...}}
            if isinstance(final_info, dict):
                if "is_success" in final_info:
                    raw_mask = final_info.get("_is_success", final_mask)
                    fill_success(final_info["is_success"], raw_mask)
            else:
                # 兼容 object array/list of dict 的旧格式。
                for env_idx in np.flatnonzero(done_mask):
                    try:
                        env_info = final_info[env_idx]
                        if isinstance(env_info, dict):
                            success[env_idx] = bool(env_info.get("is_success", False))
                    except Exception:
                        success[env_idx] = False
            if "is_success" in info:
                raw = np.asarray(info["is_success"])
                raw_mask = info.get("_is_success", done_mask)
                fill_success(raw, raw_mask)
        elif isinstance(info, dict) and "is_success" in info:
            raw = np.asarray(info["is_success"])
            raw_mask = info.get("_is_success", done_mask)
            fill_success(raw, raw_mask)

        return success & done_mask

    def run_select_action_rollout(prev_obs):
        """第一阶段排查：完全复用 eval 的 select_action，不训练，只验证采集链路。"""
        raw_obs, _ = env.reset()
        prev_obs = flatten_lerobot_obs(raw_obs)
        actor.eval()
        actor.reset()

        total_env_steps = n_steps * act_steps
        running_rewards = np.zeros(n_envs, dtype=np.float32)
        completed_rewards = []
        completed_successes = []
        logging.info(f"诊断rollout: policy.select_action, no update, physical_env_steps={total_env_steps}")

        for _ in tqdm(range(total_env_steps), desc="🧪 select_action rollout", leave=False):
            obs_tensor = build_select_action_obs(prev_obs)
            with torch.no_grad():
                action = actor.select_action(obs_tensor)
            action_np = action.cpu().numpy()
            action_to_step = action_np[0] if n_envs == 1 else action_np

            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = env.step(action_to_step)
            obs_venv = flatten_lerobot_obs(obs_venv)

            if n_envs == 1:
                reward_venv = np.array([reward_venv], dtype=np.float32)
                terminated_venv = np.array([terminated_venv], dtype=bool)
                truncated_venv = np.array([truncated_venv], dtype=bool)
            else:
                reward_venv = np.asarray(reward_venv, dtype=np.float32)
                terminated_venv = np.asarray(terminated_venv, dtype=bool)
                truncated_venv = np.asarray(truncated_venv, dtype=bool)

            done_mask = terminated_venv | truncated_venv
            running_rewards += reward_venv
            success_mask = info_success_mask(info_venv, done_mask)

            for env_idx in np.flatnonzero(done_mask):
                completed_rewards.append(float(running_rewards[env_idx]))
                completed_successes.append(bool(success_mask[env_idx]))
                running_rewards[env_idx] = 0.0

            if n_envs == 1 and done_mask[0]:
                reset_obs, _ = env.reset()
                obs_venv = flatten_lerobot_obs(reset_obs)

            if done_mask.any():
                actor.reset()

            prev_obs = obs_venv

        return prev_obs, completed_rewards, completed_successes

    # 在进入训练循环前，仅全局重置一次环境，保证后续 MDP (马尔可夫决策过程) 的连续性
    prev_obs, _ = env.reset()
    # 将初始画面展平
    prev_obs = flatten_lerobot_obs(prev_obs)
    # 记录当前每个环境正在跑的回合的累计分数
    running_ep_rewards = np.zeros(n_envs, dtype=np.float32)
    # ==========================================
    # 🌟 主循环：DPPO 强化学习全流程
    # ==========================================
    use_disk_cache = getattr(cfg.training, "use_disk_cache", False)
    rollout_policy = getattr(cfg.training, "rollout_policy", "dppo")
    skip_update = getattr(cfg.training, "skip_update", False)
    update_actor = getattr(cfg.training, "update_actor", True)
    valid_rollout_policies = {"dppo", "select_action"}
    if rollout_policy not in valid_rollout_policies:
        raise ValueError(f"training.rollout_policy 必须是 {valid_rollout_policies} 之一，当前是: {rollout_policy}")
    logging.info(f"Rollout模式: {rollout_policy}, skip_update={skip_update}, update_actor={update_actor}")
    if rollout_policy == "select_action" and n_envs > 1:
        logging.warning("⚠️ select_action的动作队列不能按单个子环境reset；严格对齐eval_train.py时建议设置 env.n_envs=1。")
    if not update_actor:
        logging.warning("  training.update_actor=false：本次只训练Critic，Actor参数和EMA不会更新。")
    running_reward_scaler = RunningRewardScaler(n_envs, gamma=float(getattr(cfg.training, "gamma", 0.99)))
    next_chunk_firsts = np.ones(n_envs, dtype=np.float32)
    raw_obs_queue = {k: deque(maxlen=n_obs_steps) for k in prev_obs.keys()}
    reset_full_obs_queue(raw_obs_queue, prev_obs)
    for itr in range(cfg.training.n_train_itr):
        logging.info(f"\n========== 第 {itr+1}/{cfg.training.n_train_itr} 轮迭代 ==========")

        if rollout_policy == "select_action":
            prev_obs, completed_ep_rewards, completed_successes = run_select_action_rollout(prev_obs)
            running_ep_rewards[:] = 0.0
            next_chunk_firsts[:] = 1.0
            reset_full_obs_queue(raw_obs_queue, prev_obs)

            if len(completed_ep_rewards) > 0:
                avg_ep_return = np.mean(completed_ep_rewards)
                max_ep_return = np.max(completed_ep_rewards)
                success_rate = np.mean(completed_successes) if completed_successes else 0.0
                logging.info(f"  select_action诊断第 {itr+1} 轮完成！")
                logging.info(f"-----完成回合数: {len(completed_ep_rewards)}")
                logging.info(f"-----成功率: {success_rate * 100:.1f}%")
                logging.info(f"-----平均回合总奖励 (Return): {avg_ep_return:.2f}")
                logging.info(f"-----最高回合奖励: {max_ep_return:.2f}")
            else:
                logging.info(f"select_action诊断第 {itr+1} 轮没有完成回合，请增加 training.rollout_steps")

            logging.info(f"select_action模式不产生 diffusion chain，本轮跳过 PPO 更新。")
            continue

        # ==========================================
        # 1. 初始化 DPPO Rollout 缓冲区 (Buffers)
        # ==========================================
        # raw_obs_queue 在每个真实 env.step 后更新一次；此处只复用，不重新创建。
        """
        {"observation.images.top": deque([], maxlen=2),
        "observation.images.wrist": deque([], maxlen=2),
        "observation.state": deque([], maxlen=2)}
        """
        obs_trajs = None

        if use_disk_cache:
            # 使用硬盘缓存
            temp_buffer_dir = tempfile.TemporaryDirectory()
            buffer_path = temp_buffer_dir.name
            logging.info(f"  [缓存模式] 已开启硬盘缓存 (Memmap)，临时目录: {buffer_path}")

            chains_trajs = np.memmap(
                os.path.join(buffer_path, 'chains_trajs.npy'),
                dtype=np.float32,
                mode='w+',
                shape=(n_steps, n_envs, denoising_steps + 1, horizon_steps, action_dim)
            )         # (步数，环境数，去噪步数，预测步数，动作维度)
        else:
            temp_buffer_dir = None
            logging.info("  [缓存模式] 使用纯内存 (RAM) 收集数据，追求极致速度...")
            # (步数，环境数，去噪步数，预测步数，动作维度)
            chains_trajs = np.zeros(
                (n_steps, n_envs, denoising_steps + 1, horizon_steps, action_dim),
                dtype=np.float32
            )

        reward_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
        terminated_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
        firsts_trajs = np.zeros((n_steps + 1, n_envs), dtype=np.float32)
        firsts_trajs[0] = next_chunk_firsts
        completed_ep_rewards = []                                # 存放所有【已经跑完】的回合的总分
        completed_ep_successes = []
        # ==========================================
        # 2. 开始收集环境交互数据 (Rollout Loop)
        # ==========================================
        actor.reset()
        logging.info(f"  开始进入数据收集循环 (共 {n_steps} 步)...")
        for step in tqdm(range(n_steps), leave=False):

            # 打包最近 n_obs_steps 个相邻物理帧；队列已在每个 env.step 后实时更新。
            stacked_raw_obs = stack_obs_queue(raw_obs_queue)

            # 使用包含了完整 T 维度的 stacked_raw_obs 初始化 obs_trajs
            if obs_trajs is None:
                obs_trajs = {}
                for k, v in stacked_raw_obs.items():
                    if use_disk_cache:
                        safe_k = k.replace(".", "_")
                        obs_trajs[k] = np.memmap(
                            os.path.join(buffer_path, f'obs_{safe_k}.npy'),
                            dtype=v.dtype,
                            mode='w+',
                            shape=(n_steps, *v.shape)
                        )
                    else:
                        obs_trajs[k] = np.zeros((n_steps, *v.shape), dtype=v.dtype)

            # 1. 格式化完整历史观测给 Actor 推理；forward_dppo 不再自己维护稀疏 queue
            batch_obs = {}
            for k, v in stacked_raw_obs.items():
                # 只提取模型 config 中真正需要的输入特征，防止 environment_state 引发报错
                if k not in actor.config.input_shapes:
                    continue
                safe_v = np.ascontiguousarray(v)
                tensor_v = torch.from_numpy(safe_v).float().to(device)

                # 此时 tensor_v 已经是带有时间维度的 5D 形状了：[B, T, H, W, C]
                if "images" in k:
                    # [Batch, Time, H, W, C] -> [Batch, Time, C, H, W]
                    tensor_v = tensor_v.permute(0, 1, 4, 2, 3) / 255.0

                batch_obs[k] = tensor_v

            # 2. 网络前向传播获取动作与去噪链
            with torch.no_grad():
                # ⚠️ 注意：此处需确保 actor 有返回 chains 的接口
                # 标准的 actor.select_action 不返回 chains，DPPO 通常需要调用底层的 forward 或特定生成函数
                # 此处仿照参考代码：返回 deterministic=False 的采样以及 return_chain=True
                samples = actor.forward_dppo(cond=batch_obs, return_chain=True)

                output_venv = samples["actions"].cpu().numpy()  # shape: [n_envs, horizon, action_dim]
                chains_venv = samples["chains"].cpu().numpy()   # shape: [n_envs, denoising_steps+1, horizon, action_dim]

            # 截取实际执行的动作长度 (Action Chunking)，与 LeRobot select_action 完全一致：
            # horizon 是从第一帧历史观测开始计数，当前观测对应 n_obs_steps - 1。
            # [B, act_steps, action_dim]
            action_venv = output_venv[:, action_start:action_end]

            # ==========================================
            # 3. 手动展开动作序列块 (Chunking Loop)
            # 网络一次预测了 8 步，我们必须让物理环境分 8 次真实执行
            # ==========================================
            chunk_reward = np.zeros(n_envs, dtype=np.float32)
            any_done_accum = np.zeros(n_envs, dtype=bool)   # 用于停止累加当前块的奖励
            true_term_accum = np.zeros(n_envs, dtype=bool)  # 用于告诉 GAE 抹除未来价值 (V=0)
            success_accum = np.zeros(n_envs, dtype=bool)

            # 🌟 用于存放每个环境真正的 "原地待命" 动作
            # 形状需要和动作维度一致
            safe_actions = np.zeros((n_envs, action_venv.shape[-1]), dtype=np.float32)

            for step_i in range(act_steps):
                curr_action = action_venv[:, step_i, :].copy() # 注意加 .copy()，防止修改原张量

                # 多环境动作冻结
                for env_idx in range(n_envs):
                    # 如果某个环境已经死亡并重置，我们不能给它喂后续的垃圾动作。
                    if any_done_accum[env_idx]:
                        # 策略：一直给它发送死亡前最后一刻的安全动作，让机器人在重置原点尽量保持静止
                        curr_action[env_idx] = safe_actions[env_idx]

                action_to_step = curr_action[0] if n_envs == 1 else curr_action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = env.step(action_to_step)
                # 环境交互后，需要将 obs 展平成 LeRobot 认识的扁平结构
                obs_venv = flatten_lerobot_obs(obs_venv)
                if n_envs == 1:
                    reward_venv = np.array([reward_venv])
                    terminated_venv = np.array([terminated_venv])
                    truncated_venv = np.array([truncated_venv])

                active_mask = ~any_done_accum
                chunk_reward += reward_venv * active_mask

                # 只在环境第一次真正终止时标记
                just_done = (terminated_venv | truncated_venv) & active_mask
                just_success = info_success_mask(info_venv, just_done)
                success_accum = success_accum | just_success
                true_term_accum = true_term_accum | (terminated_venv  & active_mask)

                if n_envs == 1 and just_done[0]:
                    reset_obs, _ = env.reset()
                    obs_venv = flatten_lerobot_obs(reset_obs)

                # 在环境重置的瞬间，提取它真实的初始位姿作为安全动作！ (仅针对多进程环境)
                for env_idx in range(n_envs):
                    if just_done[env_idx]:
                        # 将重置后状态对应的真实关节角度作为安全动作。如果有状态就提取，纯视觉无状态就用 0 填充保持原位
                        if "observation.state" in obs_venv:
                            state_data = obs_venv["observation.state"]
                            if n_envs == 1:
                                safe_actions[env_idx] = state_data[:action_venv.shape[-1]]
                            else:
                                safe_actions[env_idx] = state_data[env_idx][:action_venv.shape[-1]]
                        else:
                            safe_actions[env_idx] = np.zeros(action_venv.shape[-1], dtype=np.float32)

                append_obs_queue(raw_obs_queue, obs_venv)
                reset_done_envs_in_obs_queue(raw_obs_queue, obs_venv, just_done)
                any_done_accum = any_done_accum | terminated_venv | truncated_venv

            prev_obs = obs_venv

            # 4. 顺手统计回合总奖励 (用于日志打印)
            running_ep_rewards += chunk_reward
            for env_idx in range(n_envs):
                if any_done_accum[env_idx]: # 判断当前环境是否结束
                    completed_ep_rewards.append(running_ep_rewards[env_idx]) # 记录回合总奖励
                    completed_ep_successes.append(bool(success_accum[env_idx]))
                    running_ep_rewards[env_idx] = 0.0 # 重置回合总奖励，以便下一回合计算

            # 5. 写入轨迹 Buffer，将这 8 步累积的完整奖励和结束标志，交给外部的 PPO 缓冲区
            for k in obs_trajs:
                # 记录每一步的完整历史帧
                obs_trajs[k][step] = stacked_raw_obs[k]

            chains_trajs[step] = chains_venv             # [n_steps, n_envs, denoising_steps + 1, horizon_steps, action_dim]
            reward_trajs[step] = chunk_reward            # [n_steps, n_envs]
            terminated_trajs[step] = true_term_accum     # [n_steps, n_envs]
            firsts_trajs[step + 1] = any_done_accum      # [n_steps + 1, n_envs]

        logging.info("  数据收集 (Rollout) 完成，准备进入 PPO 网络更新阶段！")
        next_chunk_firsts = firsts_trajs[-1].copy()
        rollout_avg_return = np.mean(completed_ep_rewards) if completed_ep_rewards else float("-inf")
        rollout_success_rate = np.mean(completed_ep_successes) if completed_ep_successes else 0.0
        allow_actor_update_this_itr = True
        min_actor_success = getattr(cfg.training, "min_actor_update_success_rate", 0.0)
        min_actor_return = getattr(cfg.training, "min_actor_update_avg_return", float("-inf"))
        if (
            update_actor
            and itr >= critic_warmup_iters
            and rollout_success_rate <= min_actor_success
            and rollout_avg_return <= min_actor_return
        ):
            allow_actor_update_this_itr = False
            logging.warning(
                f"  本轮rollout已明显退化(success={rollout_success_rate * 100:.1f}%, "
                f"return={rollout_avg_return:.2f})，跳过Actor更新，只训练Critic。"
            )

        if skip_update:
            if len(completed_ep_rewards) > 0:
                avg_ep_return = np.mean(completed_ep_rewards)
                max_ep_return = np.max(completed_ep_rewards)
                success_rate = np.mean(completed_ep_successes) if completed_ep_successes else 0.0
                logging.info(f"  第 {itr+1} 轮DPPO采集诊断完成！")
                logging.info(f"     本轮共完成回合数: {len(completed_ep_rewards)}")
                logging.info(f"     成功率: {success_rate * 100:.1f}%")
                logging.info(f"     平均回合总奖励 (Return): {avg_ep_return:.2f}")
                logging.info(f"     最高回合奖励: {max_ep_return:.2f}")
            else:
                logging.info(f"⚠️ 第 {itr+1} 轮DPPO采集结束，但没有环境跑完一个完整回合")

            logging.info("🧪 training.skip_update=true，本轮只采集数据，不更新 Actor/Critic。")
            try:
                del chains_trajs, obs_trajs
                import gc
                gc.collect()
                if use_disk_cache and temp_buffer_dir is not None:
                    temp_buffer_dir.cleanup()
            except Exception:
                pass
            torch.cuda.empty_cache()
            continue

        # =========================================================
        # 5. 批量计算价值 (Values) 与 GAE 优势函数
        # =========================================================
        logging.info("🧠 计算状态价值 (Values) 与优势函数 (GAE)...")

        # 利用 reshape 虚拟展平 (不管底层是 RAM 还是 Disk，均不触发全量内存加载)
        # 将 Chains 展平为 [(步数*环境数), 去噪步数, 预测视野, 动作维度]
        chains_flat = chains_trajs.reshape(n_steps * n_envs, denoising_steps + 1, horizon_steps, action_dim)

        obs_flat = {}
        for k, v in obs_trajs.items():
            # [n_steps, n_envs, T, H, W, C] -> [(步数*环境数), T, H, W, C]
            obs_flat[k] = v.reshape(n_steps * n_envs, *v.shape[2:])

        with torch.no_grad():
            # 分批次 (Mini-batch) 计算 Critic 价值，防止显存爆炸
            total_samples = n_steps * n_envs
            val_batch_size = getattr(cfg.training, "batch_size", 32) * 2  # 评估不算梯度，batch 可以开大点
            values_flat = np.zeros(total_samples, dtype=np.float32)
            # 每次取 val_batch_size 个样本
            for i in range(0, total_samples, val_batch_size):
                end_i = min(i + val_batch_size, total_samples)

                # 临时切下一小块推入 GPU
                obs_chunk = {}
                for k, v in obs_flat.items():
                    tensor_v = torch.from_numpy(v[i:end_i]).float().to(device)
                    if "images" in k:
                        # [B, T, H, W, C] -> [B, T, C, H, W]
                        tensor_v = tensor_v.permute(0, 1, 4, 2, 3) / 255.0
                    obs_chunk[k] = tensor_v

                # 调用 Actor 的特征提取底座
                obs_chunk_norm = actor.normalize_inputs(obs_chunk) # 归一化输入
                if len(actor.expected_image_keys) > 0:
                    obs_chunk_norm["observation.images"] = torch.stack([obs_chunk_norm[k] for k in actor.expected_image_keys], dim=-4)

                # 获取融合特征并给 Critic 估值
                global_cond = actor.diffusion._prepare_global_conditioning(obs_chunk_norm)
                values_flat[i:end_i] = critic(global_cond.detach()).cpu().numpy().flatten()

            values_trajs = values_flat.reshape(n_steps, n_envs)
            # 计算最后一步的 Next Value (Bootstrap)
            # raw_obs_queue 已经在每个 env.step 后更新，这里直接使用最后的相邻历史窗口。
            last_stacked_raw_obs = stack_obs_queue(raw_obs_queue)
            last_obs_ts = {}
            for k, v in last_stacked_raw_obs.items():
                # 过滤不需要的键值
                if k not in actor.config.input_shapes:
                    continue
                tensor_v = torch.from_numpy(np.ascontiguousarray(v)).float().to(device)

                if "images" in k:
                    # 此时已经是 5D 张量: [Batch, Time, H, W, C] -> [Batch, Time, C, H, W]
                    tensor_v = tensor_v.permute(0, 1, 4, 2, 3) / 255.0

                last_obs_ts[k] = tensor_v

            # 2. 调用 Actor 逻辑进行统计学归一化
            last_obs_ts_norm = actor.normalize_inputs(last_obs_ts.copy())

            # 3. 按 LeRobot 的规范，将字典里的所有相机图像按 Num_Cams 维度堆叠
            if len(actor.expected_image_keys) > 0:
                last_obs_ts_norm["observation.images"] = torch.stack(
                    [last_obs_ts_norm[k] for k in actor.expected_image_keys], dim=-4
                )

            # 4. 调用共享底座提取终极特征 (global_cond)
            global_cond_last = actor.diffusion._prepare_global_conditioning(last_obs_ts_norm)

            # 5. 让盲人军师（Critic）对最后一步打分！
            next_values_last = critic(global_cond_last.detach()).cpu().numpy().flatten()


        # 使用原版 DPPO 的 RunningRewardScaler：根据折扣累计回报的运行方差缩放 reward。
        if reward_scale_running:
            reward_trajs = running_reward_scaler(
                reward=reward_trajs.T,
                first=firsts_trajs[:-1].T,
            ).T

        # 2. 计算 GAE (逆向时间推导)
        advantages_trajs = np.zeros_like(reward_trajs)
        last_gae_lam = 0
        gamma = getattr(cfg.training, "gamma", 0.99)
        gae_lambda = getattr(cfg.training, "gae_lambda", 0.95)

        for t in reversed(range(n_steps)):
            # 获取下一步的 观测评估价值
            next_val = (next_values_last if t == n_steps - 1 else values_trajs[t + 1])
            # 判断游戏是否结束，如果nonterminal为0，表示死亡或通关，那么未来价值都是0
            nonterminal = 1.0 - terminated_trajs[t]

            # 单步TD误差 = 第t步的奖励*缩放系数 + 折旧因子*下一步观测的价值 - 当前步观测的价值
            # TD 误差: δ = r + γ * V(s') * (1-done) - V(s)
            delta = reward_trajs[t] * reward_scale_const + gamma * next_val * nonterminal - values_trajs[t]

            # 优势值计算：t-1步的优势值 = TD误差 + 双重衰减系数*t-1步的优势值   一直迭代，一轮(n_steps步)就能算出所有步的优势值
            # 优势值: A_t = δ_t + γ * λ * (1-done) * A_{t+1}
            advantages_trajs[t] = last_gae_lam = delta + gamma * gae_lambda * nonterminal * last_gae_lam

        # Advantage = Return - Value,       优势 = 实际总回报 - 预期总回报
        # aritic网络输出的是当前画面对应的未来总回报  Return = Advantage + Value，
        returns_trajs = advantages_trajs + values_trajs # 用于训练critic网络
        critic_explained_variance, critic_value_return_corr = compute_value_diagnostics(
            values_trajs,
            returns_trajs,
        )
        logging.info(f"  [排查] Critic 预测 Value 均值: {values_trajs.mean():.4f}, 真实 Return 均值: {returns_trajs.mean():.4f}")
        logging.info(
            f"  [排查] Critic解释方差(EV): {critic_explained_variance:.4f}, "
            f"Value/Return相关性: {critic_value_return_corr:.4f}"
        )

        # =========================================================
        # 6. DPPO 多轮小批量更新 (Update Epochs)
        # =========================================================
        # 🌟 优化 1：统一提取 PPO 核心超参数，确保日志与实际运行绝对一致
        batch_size = getattr(cfg.training, "batch_size", 32)
        update_epochs = getattr(cfg.training, "update_epochs", 4)
        clip_ratio = getattr(cfg.training, "clip_ratio", 0.05)
        gamma_denoising = getattr(cfg.training, "gamma_denoising", 0.99)
        norm_adv = bool(getattr(cfg.training, "norm_adv", True))
        adv_lower_q = float(getattr(cfg.training, "clip_advantage_lower_quantile", 0.05))
        adv_upper_q = float(getattr(cfg.training, "clip_advantage_upper_quantile", 0.95))

        def normalize_clip_minibatch_advantage(advantages):
            """按原版 DPPO 口径在 minibatch 内归一化并裁剪 advantage。"""
            if norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            if adv_lower_q > 0.0 or adv_upper_q < 1.0:
                adv_lower = torch.quantile(advantages, adv_lower_q)
                adv_upper = torch.quantile(advantages, adv_upper_q)
                advantages = torch.clamp(advantages, min=adv_lower, max=adv_upper)
            return advantages

        logging.info(f"🔄 开始 PPO 网络更新 (Epochs: {getattr(cfg.training, 'update_epochs', 4)})...")


        # 1. 准备训练用的展平张量
        returns_k = torch.from_numpy(returns_trajs).float().to(device).reshape(-1)
        advantages_k = torch.from_numpy(advantages_trajs).float().to(device).reshape(-1)

        # total_steps 表示包含去噪步数的总状态转移次数 (例如：300 * 5 * 10)
        total_steps = n_steps * n_envs * denoising_steps

        # 获取与去噪步对应的真实 TimeSteps
        actor.diffusion.noise_scheduler.set_timesteps(actor.diffusion.num_inference_steps)
        all_timesteps = actor.diffusion.noise_scheduler.timesteps
        record_start_idx = max(0, len(all_timesteps) - denoising_steps) # 只保留最后 denoising_steps 步
        recorded_timesteps = all_timesteps[record_start_idx:].to(device)

        # =========================================================
        # 7. 预计算旧策略的对数概率 (Old Logprobs)
        # =========================================================
        # 预计算旧概率时，同样按需从 CPU 搬运数据
        logging.info("  正在预计算旧策略概率基准...")
        actor.eval()
        old_logprobs_k = torch.zeros(total_steps, device=device)
        with torch.no_grad():
            eval_batch_size = getattr(cfg.training, "batch_size", 32) * 2
            for i in range(0, total_steps, eval_batch_size):
                inds = torch.arange(i, min(i + eval_batch_size, total_steps), device=device)
                b_inds, d_inds = torch.unravel_index(inds, (n_steps * n_envs, denoising_steps))
                # ===================================================
                # 🌟 懒加载兼容修改 1：将 GPU 索引转为 CPU Numpy 索引，才能切片 memmap
                # ===================================================
                b_inds_np = b_inds.cpu().numpy()
                d_inds_np = d_inds.cpu().numpy()

                # 从硬盘/内存按需拉取画面到 GPU
                obs_eval = {}
                for k, v in obs_flat.items():
                    tensor_v = torch.from_numpy(v[b_inds_np]).float().to(device)
                    if "images" in k:
                        # [Batch, Time, H, W, C] -> [Batch, Time, C, H, W]
                        tensor_v = tensor_v.permute(0, 1, 4, 2, 3)/ 255.0
                    obs_eval[k] = tensor_v
                chains_prev_eval = torch.from_numpy(chains_flat[b_inds_np, d_inds_np]).float().to(device)
                chains_next_eval = torch.from_numpy(chains_flat[b_inds_np, d_inds_np + 1]).float().to(device)
                logprobs = actor.get_logprobs(
                    cond=obs_eval,
                    x_t=chains_prev_eval,
                    x_t_1=chains_next_eval,
                    timesteps=recorded_timesteps[d_inds],
                    return_global_cond=False
                )
                old_logprobs_k[inds] = logprobs

        actor_update_this_itr = allow_actor_update_this_itr and update_actor and itr >= critic_warmup_iters
        post_probe_n = 0
        post_probe_corr = float("nan")
        post_probe_pos_delta = float("nan")
        post_probe_neg_delta = float("nan")
        post_probe_sign_agreement = float("nan")
        post_probe_mean_delta = float("nan")
        probe_inds = None
        probe_logprobs_before = None
        probe_advantages = None
        probe_size = int(getattr(cfg.training, "post_update_probe_size", 256))
        if actor_update_this_itr and probe_size > 0:
            post_probe_n = min(probe_size, total_steps)
            probe_inds = torch.randperm(total_steps, device=device)[:post_probe_n]
            probe_batch_inds, probe_denoising_inds = torch.unravel_index(
                probe_inds,
                (n_steps * n_envs, denoising_steps),
            )
            probe_discount = gamma_denoising ** (denoising_steps - probe_denoising_inds - 1)
            probe_advantages = normalize_clip_minibatch_advantage(advantages_k[probe_batch_inds]) * probe_discount
            probe_advantages = probe_advantages.detach().clone()
            probe_logprobs_before = old_logprobs_k[probe_inds].detach().clone()
            logging.info(f"🧪 已固定 Post-update probe 样本: {post_probe_n}")

        bc_coef = float(getattr(cfg.training, "bc_loss_coef", 0.02))
        teacher_chains_flat = None
        if actor_update_this_itr and bc_coef > 0:
            logging.info("🧑‍🏫 正在预计算预训练老师策略的动作链，用于官方式 BC 正则...")
            teacher_chains_flat = np.empty(
                (total_samples, denoising_steps + 1, horizon_steps, action_dim),
                dtype=np.float32,
            )
            actor_ref.eval()
            with torch.no_grad():
                teacher_batch_size = getattr(cfg.training, "batch_size", 32) * 2
                for i in range(0, total_samples, teacher_batch_size):
                    end_i = min(i + teacher_batch_size, total_samples)
                    obs_teacher = {}
                    for k, v in obs_flat.items():
                        tensor_v = torch.from_numpy(v[i:end_i]).float().to(device)
                        if "images" in k:
                            tensor_v = tensor_v.permute(0, 1, 4, 2, 3) / 255.0
                        obs_teacher[k] = tensor_v

                    teacher_samples = actor_ref.forward_dppo(cond=obs_teacher, return_chain=True)
                    teacher_chains_flat[i:end_i] = teacher_samples["chains"].detach().cpu().numpy()
                    del obs_teacher, teacher_samples

        running_v_loss = []
        running_pg_loss = []
        running_kl = []
        running_bc_loss = []
        logprob_adv_stats = init_logprob_advantage_stats()

        # 2. 开始 Epoch 循环，PPO的数据集可以训练多轮，数据利用率高
        # PPO 的 old/new logprob 必须使用同一种观测预处理模式。
        # eval() 不会关闭梯度，但会关闭随机裁剪/Dropout/BN 统计漂移，避免冻结 actor 时 KL 仍然爆炸。
        actor.eval()
        critic.train()
        # 强制冻结视觉底座的 BatchNorm 层，防止小 Batch 引起的统计量震荡导致的虚假 KL 爆炸
        def freeze_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
                # 可选：如果完全不想更新底座特征，连梯度也关掉
                # m.weight.requires_grad = False
                # m.bias.requires_grad = False

        # 应用到 Actor 的视觉底座和 Critic 网络
        actor.apply(freeze_bn)
        critic.apply(freeze_bn)
        early_stop = False
        for epoch in tqdm(range(update_epochs), desc=f"  PPO 更新中 (Iter {itr+1})", leave=False):
            # 每一轮 Epoch 开始前检查，如果已熔断，彻底跳出 Epoch 循环，开启新的一轮迭代
            if early_stop:
                break
            # 打乱所有数据点 (不仅打乱时间步，还打乱去噪步)
            indices = torch.randperm(total_steps, device=device)
            num_batch = max(1, total_steps // batch_size) #分批次训练

            for batch_idx in tqdm(range(num_batch), desc=f"     Batch 更新", leave=False):
                start = batch_idx * batch_size
                end = start + batch_size
                inds_b = indices[start:end] #从打乱的总数据中提取一个batch的索引

                # 将inds_b索引值对应到哪批样本batch_inds_b 的 哪个去噪步denoising_inds_b
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b,
                    (n_steps * n_envs, denoising_steps) #n_steps * n_envs行，denoising_steps列
                )

                # ===================================================
                # 将 GPU 上的批量索引转为 CPU Numpy 格式
                # ===================================================
                b_inds_np = batch_inds_b.cpu().numpy()
                d_inds_np = denoising_inds_b.cpu().numpy()

                # 切片提取 Mini-batch
                obs_b = {}           # 给 Actor （带历史帧）

                for k, v in obs_flat.items():
                    tensor_v = torch.from_numpy(v[b_inds_np]).float().to(device)
                    if "images" in k:
                        # [Batch, Time, H, W, C] -> [Batch, Time, C, H, W]
                        tensor_v = tensor_v.permute(0, 1, 4, 2, 3)/ 255.0

                    # 完整数据给 Actor
                    obs_b[k] = tensor_v

                # ===================================================
                # 从硬盘 (或系统内存) 按需拉取这一小批动作切片到 GPU
                # ===================================================
                chains_prev_b = torch.from_numpy(chains_flat[b_inds_np, d_inds_np]).float().to(device)
                chains_next_b = torch.from_numpy(chains_flat[b_inds_np, d_inds_np + 1]).float().to(device)
                returns_b = returns_k[batch_inds_b]                           # 对应样本的总回报
                advantages_b = advantages_k[batch_inds_b]                     # 每一轮去噪步公用一个优势值
                timesteps_b = recorded_timesteps[denoising_inds_b]

                # ==========================================
                # 在 minibatch 内做 advantage normalize/clip，与原版 DPPO 的损失计算口径一致。
                # ==========================================
                advantages_b = normalize_clip_minibatch_advantage(advantages_b)

                # ==========================================
                # 去噪步骤的优势衰减折扣
                # 越接近输出端 (denoising_inds_b 越大)，折扣越接近 1.0
                # ==========================================
                # 将时刻t的优势值乘以折扣因子: gamma ** (ft_denoising_steps - i - 1)
                discount = gamma_denoising ** (denoising_steps - denoising_inds_b - 1)
                advantages_b = advantages_b * discount

                # 取出旧策略的对数概率
                old_logprobs_b = old_logprobs_k[inds_b]
                actor_update_enabled = actor_update_this_itr


                # ----------------------------------------------------
                # 🌟 网络 Loss 计算区 (共享计算底座)
                # ----------------------------------------------------
                # 1. 计算当前网络对动作的新概率预测，同时返回全局特征
                if actor_update_enabled:
                    new_logprobs_b, global_cond_b = actor.get_logprobs(
                        cond=obs_b,
                        x_t=chains_prev_b,
                        x_t_1=chains_next_b,
                        timesteps=timesteps_b,
                        return_global_cond=True
                    )
                else:
                    with torch.no_grad():
                        new_logprobs_b, global_cond_b = actor.get_logprobs(
                            cond=obs_b,
                            x_t=chains_prev_b,
                            x_t_1=chains_next_b,
                            timesteps=timesteps_b,
                            return_global_cond=True
                        )

                # 2. PPO 概率比截断，Ratio = exp(new_logprob - old_logprob)
                raw_log_ratio = new_logprobs_b - old_logprobs_b
                if actor_update_enabled:
                    update_logprob_advantage_stats(logprob_adv_stats, raw_log_ratio, advantages_b)
                log_ratio = torch.clamp(raw_log_ratio, min=-20.0, max=5.0)
                ratio = torch.exp(log_ratio)

                # 3. PPO 截断代理损失，把控单次更新范围
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_b
                pg_loss = -torch.min(surr1, surr2).mean()

                # 4. Critic 的打分损失 (直接使用 Smooth L1 Loss，无需截断，让其自由拟合真实价值)
                values_pred = critic(global_cond_b.detach()).squeeze(-1)
                v_loss = torch.nn.functional.smooth_l1_loss(values_pred, returns_b)

                # 5. 官方式 BC Loss：老师策略采样动作链，当前策略最大化老师动作链的概率。
                if actor_update_enabled and bc_coef > 0 and teacher_chains_flat is not None:
                    teacher_prev_b = torch.from_numpy(teacher_chains_flat[b_inds_np, d_inds_np]).float().to(device)
                    teacher_next_b = torch.from_numpy(teacher_chains_flat[b_inds_np, d_inds_np + 1]).float().to(device)
                    teacher_logprobs_b = actor.get_logprobs(
                        cond=obs_b,
                        x_t=teacher_prev_b,
                        x_t_1=teacher_next_b,
                        timesteps=timesteps_b,
                        return_global_cond=False
                    )
                    bc_loss = -teacher_logprobs_b.mean()
                else:
                    bc_loss = torch.zeros((), device=device)
                # 5. 总 Loss 汇总, 反向传播时，pg_loss流向策略网络，v_loss流向价值网络
                if actor_update_enabled:
                    loss = pg_loss + 0.5 * v_loss + bc_coef * bc_loss
                else:
                    loss = 0.5 * v_loss

                running_v_loss.append(v_loss.item())
                running_pg_loss.append(pg_loss.item())
                running_bc_loss.append(bc_loss.item())

                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                running_kl.append(approx_kl)
                # 早期熔断 (Early Stopping)，把控整体策略偏移量
                ref_bc_early_stop = getattr(cfg.training, "ref_bc_early_stop", None)
                bc_too_large = (
                    actor_update_enabled
                    and ref_bc_early_stop is not None
                    and ref_bc_early_stop > 0
                    and bc_loss.item() > ref_bc_early_stop
                )
                kl_too_large = actor_update_enabled and approx_kl > target_kl
                if kl_too_large or bc_too_large:
                    if kl_too_large:
                        logging.warning(f"⚠️ 策略偏离过大 (KL: {approx_kl:.4f} > {target_kl})，触发早期熔断！")
                    if bc_too_large:
                        logging.warning(
                            f"⚠️ Actor偏离预训练参考模型过大 "
                            f"(BC: {bc_loss.item():.4f} > {ref_bc_early_stop:.4f})，触发早期熔断！"
                        )
                    early_stop = True # 用于跳出当前epoch

                    # 🌟 修复 2：极其重要！必须手动销毁所有带梯度的局部变量，释放几 GB 的计算图！
                    try:
                        del loss, pg_loss, v_loss, new_logprobs_b, raw_log_ratio, log_ratio, ratio, surr1, surr2, values_pred
                    except Exception:
                        pass

                    # 清空优化器里的残余状态，并强制清空显存缓存
                    actor_optimizer.zero_grad(set_to_none=True)
                    critic_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    break # 跳出当前batch

                # ==========================================
                # 🌟 加入梯度累加 (Gradient Accumulation)
                # ==========================================
                grad_accum_steps = getattr(cfg.training, "grad_accumulate", 4) # 如果你显存小，设为4或8

                # 缩小当前 batch 的 loss 比例
                loss = loss / grad_accum_steps
                loss.backward()

                # 累计满足设定的步数，或者达到最后一个 batch 时，才真正更新权重
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1 == num_batch):
                    if actor_update_enabled:
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)

                    critic_optimizer.step()
                    if actor_update_enabled:
                        actor_optimizer.step()


                    # 更新完后务必清空梯度！
                    actor_optimizer.zero_grad(set_to_none=True)
                    critic_optimizer.zero_grad(set_to_none=True)

        # 固定样本 post-update probe：用同一批 rollout transition 比较更新前后的 logprob。
        # 这比训练过程中的 minibatch 统计更直接，用来确认 actor 是否真的按 advantage 方向改变概率。
        if probe_inds is not None and probe_logprobs_before is not None and probe_advantages is not None:
            actor.eval()
            probe_logprobs_after = torch.empty_like(probe_logprobs_before)
            probe_batch_size = int(
                getattr(cfg.training, "post_update_probe_batch_size", getattr(cfg.training, "batch_size", 32) * 2)
            )
            probe_batch_size = max(1, probe_batch_size)
            with torch.no_grad():
                for i in range(0, post_probe_n, probe_batch_size):
                    end_i = min(i + probe_batch_size, post_probe_n)
                    probe_inds_i = probe_inds[i:end_i]
                    probe_batch_inds, probe_denoising_inds = torch.unravel_index(
                        probe_inds_i,
                        (n_steps * n_envs, denoising_steps),
                    )
                    b_inds_np = probe_batch_inds.cpu().numpy()
                    d_inds_np = probe_denoising_inds.cpu().numpy()

                    obs_probe = {}
                    for k, v in obs_flat.items():
                        tensor_v = torch.from_numpy(v[b_inds_np]).float().to(device)
                        if "images" in k:
                            tensor_v = tensor_v.permute(0, 1, 4, 2, 3) / 255.0
                        obs_probe[k] = tensor_v

                    chains_prev_probe = torch.from_numpy(chains_flat[b_inds_np, d_inds_np]).float().to(device)
                    chains_next_probe = torch.from_numpy(chains_flat[b_inds_np, d_inds_np + 1]).float().to(device)
                    probe_logprobs_after[i:end_i] = actor.get_logprobs(
                        cond=obs_probe,
                        x_t=chains_prev_probe,
                        x_t_1=chains_next_probe,
                        timesteps=recorded_timesteps[probe_denoising_inds],
                        return_global_cond=False,
                    )

                    del obs_probe, chains_prev_probe, chains_next_probe

            probe_delta = probe_logprobs_after - probe_logprobs_before
            probe_stats = init_logprob_advantage_stats()
            update_logprob_advantage_stats(probe_stats, probe_delta, probe_advantages)
            post_probe_corr, post_probe_pos_delta, post_probe_neg_delta, post_probe_sign_agreement = (
                finalize_logprob_advantage_stats(probe_stats)
            )
            post_probe_mean_delta = float(probe_delta.mean().item())
            del probe_logprobs_after, probe_delta, probe_stats
        # 跑完一整轮环境交互 (Iteration)，让学习率步进一次
        if update_actor and itr >= critic_warmup_iters:
            actor_scheduler.step()
            # 同步更新模型的 EMA 平滑权重
            if hasattr(actor, "update"):
                with torch.no_grad():
                    actor.update()
        # =========================================================
        # 🌟 修复 4：PPO 阶段结束，清扫全部巨大的训练经验张量，让显存归还给评估和下一次 Rollout
        # =========================================================
        try:
            del loss, pg_loss, v_loss, new_logprobs_b, raw_log_ratio, log_ratio, ratio, surr1, surr2, values_pred

            # 🌟 终极修复：把本轮所有的巨型 Buffer 全部粉碎！释放物理内存与文件读写锁！
            del chains_flat, obs_flat, chains_trajs, obs_trajs, returns_k, advantages_k, old_logprobs_k
            if probe_inds is not None:
                del probe_inds, probe_logprobs_before, probe_advantages
            if teacher_chains_flat is not None:
                del teacher_chains_flat
            import gc
            gc.collect()
            # 必须在底层文件锁 (memmap) 被 del 释放之后，清理临时文件夹才能真正生效
            if use_disk_cache and temp_buffer_dir is not None:
                temp_buffer_dir.cleanup() # 如果开启了硬盘缓存，必须显式调用 cleanup() 删除物理文件
        except Exception:
            pass
        torch.cuda.empty_cache()

        # ------------------------------------------
        # 步骤 E：打印评估指标 (替代原论文代码)
        # ------------------------------------------
        if len(completed_ep_rewards) > 0:
            avg_ep_return = np.mean(completed_ep_rewards)
            max_ep_return = np.max(completed_ep_rewards)
            success_rate = np.mean(completed_ep_successes) if completed_ep_successes else 0.0
            avg_v_loss = np.mean(running_v_loss) if running_v_loss else 0.0
            avg_pg_loss = np.mean(running_pg_loss) if running_pg_loss else 0.0
            avg_kl = np.mean(running_kl) if running_kl else 0.0
            max_kl = np.max(running_kl) if running_kl else 0.0
            avg_bc_loss = np.mean(running_bc_loss) if running_bc_loss else 0.0
            logprob_adv_corr, pos_adv_logratio, neg_adv_logratio, adv_sign_agreement = (
                finalize_logprob_advantage_stats(logprob_adv_stats)
            )
            logging.info(f"  第 {itr+1} 轮完成！")
            logging.info(f"     本轮共完成回合数: {len(completed_ep_rewards)}")
            logging.info(f"     成功率: {success_rate * 100:.1f}%")
            logging.info(f"     平均回合总奖励 (Return): {avg_ep_return:.2f}")
            logging.info(f"     最高回合奖励: {max_ep_return:.2f}")
            # 🌟 打印平均 Loss
            logging.info(f"     Critic (Value) Loss: {avg_v_loss:.4f}")
            logging.info(f"     Actor (Policy) Loss: {avg_pg_loss:.4f}")
            logging.info(f"     PPO KL: avg={avg_kl:.3e}, max={max_kl:.3e}, BC_NLL={avg_bc_loss:.5f}")
            logging.info(
                f"     Critic诊断: EV={critic_explained_variance:.4f}, "
                f"Value/Return Corr={critic_value_return_corr:.4f}"
            )
            logging.info(
                f"     PPO方向诊断: corr(Δlogp, Adv)={logprob_adv_corr:.4f}, "
                f"Adv>0均值Δlogp={pos_adv_logratio:.3e}, "
                f"Adv<0均值Δlogp={neg_adv_logratio:.3e}, "
                f"符号一致率={adv_sign_agreement * 100:.1f}%"
            )
            logging.info(
                f"     Post-update Probe: n={post_probe_n}, "
                f"corr(Δlogp, Adv)={post_probe_corr:.4f}, "
                f"均值Δlogp={post_probe_mean_delta:.3e}, "
                f"Adv>0均值Δlogp={post_probe_pos_delta:.3e}, "
                f"Adv<0均值Δlogp={post_probe_neg_delta:.3e}, "
                f"符号一致率={post_probe_sign_agreement * 100:.1f}%"
            )

            # 如果配置了 WandB
            # logger.log_dict({"train/avg_return": avg_ep_return}, step=itr)
        else:
            logging.info(f"⚠️ 第 {itr+1} 轮结束，但没有环境跑完一个完整回合 (考虑增加 rollout_steps)")

        # ==========================================
        # 步骤 F：定期策略评估、录像与模型快照保存
        # ==========================================
        # 默认每 5 轮评估一次，最后一步强制评估
        eval_freq = getattr(cfg.eval, "eval_freq", 5)
        is_last_step = (itr + 1) == cfg.training.n_train_itr

        if ((itr + 1) > critic_warmup_iters) and ((itr + 1) % eval_freq == 0 or is_last_step):
            logging.info(f"\n  开始第 {itr+1} 轮的策略评估与录像...")

            # 1. 设定本轮视频的保存路径
            tmp_videos_dir = Path(out_dir) / "eval" / f"videos_{itr+1:06d}"

            # 2. 调用 eval_train.py 中的评估函数 (内部已自动处理 actor.eval() 和 actor.train() 切换)
            # 使用 getattr 安全获取 cfg.eval，如果 yaml 里没配就传一个空字典
            eval_cfg_node = getattr(cfg, "eval", OmegaConf.create())

            with torch.no_grad():
                with torch.autocast(device_type=device.type) if getattr(cfg, "use_amp", False) else nullcontext():
                    eval_info = custom_eval_policy(
                        env=eval_env,
                        policy=actor,
                        cfg_eval=eval_cfg_node,
                        videos_dir=tmp_videos_dir,  # 👈 先存到临时文件夹
                        device=device
                    )

            # 3. 提取测试成绩
            sr = eval_info["aggregated"]["success_rate"]
            ar = eval_info["aggregated"]["average_reward"]
            logging.info(f"  评估完成! 成功率: {sr*100:.1f}%, 平均奖励: {ar:.2f}")

            if ar > best_eval_reward:
                best_eval_reward = ar
                best_eval_success_rate = sr
                best_actor_state = snapshot_actor_state()
                eval_collapse_count = 0
                logging.info(
                    f"  刷新内存最佳Actor: success={best_eval_success_rate * 100:.1f}%, "
                    f"reward={best_eval_reward:.2f}"
                )
            else:
                rollback_enabled = getattr(cfg.training, "rollback_on_eval_collapse", True)
                rollback_sr = getattr(cfg.training, "rollback_success_rate", 0.1)
                rollback_reward = getattr(cfg.training, "rollback_reward", -100.0)
                rollback_patience = getattr(cfg.training, "rollback_patience", 1)
                # 评估塌陷判定：成功率和奖励同时低于设定阈值，才算真正的塌陷，触发计数
                eval_collapsed = sr <= rollback_sr and ar <= rollback_reward
                if rollback_enabled and eval_collapsed:
                    eval_collapse_count += 1
                    logging.warning(
                        f"  评估已塌陷(success={sr * 100:.1f}%, reward={ar:.2f})，"
                        f"计数 {eval_collapse_count}/{rollback_patience}。"
                    )
                    if eval_collapse_count >= rollback_patience:
                        restore_actor_state(best_actor_state)
                        eval_collapse_count = 0
                        logging.warning(
                            f"  已回滚到内存最佳Actor(success={best_eval_success_rate * 100:.1f}%, "
                            f"reward={best_eval_reward:.2f})，并清空Actor optimizer状态。"
                        )
                        if tmp_videos_dir.exists():
                            import shutil
                            shutil.rmtree(tmp_videos_dir, ignore_errors=True)
                        continue
                else:
                    eval_collapse_count = 0

            # 4. 保存模型权重快照 (LeRobot 标准格式)
            ckpt_name = f"{itr+1:06d}_sr={sr:.2f}_reward={ar:.2f}_Ploss={avg_pg_loss:.4f}_Vloss={avg_v_loss:.4f}"
            ckpt_path = Path(out_dir) / "checkpoints" / ckpt_name
            save_path = Path(out_dir) / "checkpoints" / ckpt_name / "pretrained_model"
            final_videos_dir = Path(ckpt_path) / "eval" / f"eval_videos"

            # 执行文件夹重命名 (把 tmp_videos_... 改成 videos_000005_sr=...)
            if tmp_videos_dir.exists() and tmp_videos_dir != final_videos_dir:
                import shutil
                # 使用 shutil.move 比 Path.rename 更安全，能兼容跨盘操作
                shutil.move(str(tmp_videos_dir), str(final_videos_dir))
                logging.info(f"  视频文件夹已重命名为: {final_videos_dir.name}")

            actor.save_pretrained(save_path)

            # save_pretrained 写出的 config.json 是 LeRobot policy 专用配置；
            # 这里额外重建完整 config.yaml，记录当前微调的训练/环境/eval 参数。
            save_path.mkdir(parents=True, exist_ok=True)
            config_out_path = save_path / "config.yaml"
            finetune_config_out_path = save_path / "finetune_config.yaml"

            current_ft_dict = OmegaConf.to_container(cfg, resolve=True)
            base_config_dict = OmegaConf.to_container(hydra_cfg, resolve=True) if hydra_cfg is not None else {}

            # 1. 以预训练配置为底，保留完整 policy 网络结构；再递归覆盖当前微调配置。
            final_config_dict = deep_update_dict(base_config_dict, current_ft_dict)

            # 2. training 节点必须使用当前微调配置，而不是预训练 offline training 配置。
            if "training" in current_ft_dict:
                final_config_dict["training"] = current_ft_dict["training"]

            # 3. policy 节点用三方合并：预训练 YAML 结构 + save_pretrained 的 config.json + 当前微调 policy 覆盖。
            saved_policy_json = {}
            policy_json_path = save_path / "config.json"
            if policy_json_path.exists():
                with open(policy_json_path, "r", encoding="utf-8") as f:
                    saved_policy_json = json.load(f)

            runtime_policy_overrides = {
                "ft_denoising_steps": int(getattr(actor.config, "ft_denoising_steps", getattr(cfg.policy, "ft_denoising_steps", 10))),
                "n_action_steps": int(getattr(actor.config, "n_action_steps", getattr(cfg.policy, "n_action_steps", 8))),
                "min_sampling_denoising_std": float(getattr(actor.config, "min_sampling_denoising_std", getattr(cfg.training, "min_sampling_denoising_std", 0.02))),
                "min_logprob_denoising_std": float(getattr(actor.config, "min_logprob_denoising_std", getattr(cfg.training, "min_logprob_denoising_std", 0.02))),
                "logprob_reduction": str(getattr(actor.config, "logprob_reduction", getattr(cfg.training, "logprob_reduction", "mean"))),
            }
            if hasattr(actor.config, "do_mask_loss_for_padding"):
                runtime_policy_overrides["do_mask_loss_for_padding"] = bool(actor.config.do_mask_loss_for_padding)

            final_policy = deep_update_dict(base_config_dict.get("policy", {}), saved_policy_json)
            final_policy = deep_update_dict(final_policy, current_ft_dict.get("policy", {}))
            final_policy = deep_update_dict(final_policy, runtime_policy_overrides)
            final_config_dict["policy"] = final_policy

            # 4. 记录 checkpoint 运行时指标，方便之后追溯这个快照来自哪一轮。
            final_config_dict["checkpoint"] = {
                "iteration": int(itr + 1),
                "success_rate": float(sr),
                "average_reward": float(ar),
                "avg_policy_loss": float(avg_pg_loss),
                "avg_value_loss": float(avg_v_loss),
                "critic_explained_variance": float(critic_explained_variance),
                "critic_value_return_correlation": float(critic_value_return_corr),
                "logprob_advantage_correlation": float(logprob_adv_corr),
                "positive_advantage_mean_logprob_delta": float(pos_adv_logratio),
                "negative_advantage_mean_logprob_delta": float(neg_adv_logratio),
                "logprob_advantage_sign_agreement": float(adv_sign_agreement),
                "post_update_probe_size": int(post_probe_n),
                "post_update_probe_logprob_delta_mean": float(post_probe_mean_delta),
                "post_update_probe_logprob_advantage_correlation": float(post_probe_corr),
                "post_update_probe_positive_advantage_mean_logprob_delta": float(post_probe_pos_delta),
                "post_update_probe_negative_advantage_mean_logprob_delta": float(post_probe_neg_delta),
                "post_update_probe_logprob_advantage_sign_agreement": float(post_probe_sign_agreement),
                "rollout_success_rate": float(success_rate) if len(completed_ep_rewards) > 0 else None,
                "rollout_average_return": float(avg_ep_return) if len(completed_ep_rewards) > 0 else None,
            }

            # 5. 写出完整混合配置和纯微调配置。config.json 保持 LeRobot 原生 policy 格式。
            with open(config_out_path, "w", encoding="utf-8") as f:
                yaml.dump(final_config_dict, f, allow_unicode=True, sort_keys=False)
            with open(finetune_config_out_path, "w", encoding="utf-8") as f:
                yaml.dump(current_ft_dict, f, allow_unicode=True, sort_keys=False)

            logging.info(f"  模型快照、完整 config.yaml 与 finetune_config.yaml 已保存至: {save_path}")


            # 5. 交给 TopKCheckpointManager 进行同步清理
            manager.update(step=itr+1, loss=avg_pg_loss,  ckpt_path=ckpt_path, reward=ar)
        elif (itr + 1) <= critic_warmup_iters:
            logging.info(f"Critic 正在预热阶段 ({itr+1}/{critic_warmup_iters})，本轮暂不评估 Actor 模型")
@hydra.main(version_base="1.2", config_name="ft_default", config_path="../../configs/finetune")
def train_cli(cfg: DictConfig):
    train_dppo_finetune(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,  # 获取当前训练运行的输出目录，用于保存训练输出的数据
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name, # 获取当前训练运行的作业名称，用于wandb
    )

if __name__ == "__main__":
    # 命令行参数注入
    default_args = [
        "policy=ft_zed_wrist_diffusion",
        "training.pretrained_ckpt_path='outputs/2_pretrain/train/2026-06-17/01-26-08_SewNeedle-3Arms-v0_pre_zed_wrist_diffusion/checkpoints/112000_loss=0.0025_sr=10.0_ar=18.69'",
        "env.n_envs=1",
        "training.rollout_steps=800",
        "training.batch_size=16",
        "training.update_epochs=1",
        "wandb.enable=false",
        "training.skip_update=false",
        "training.rollout_policy=dppo",
        # "training.update_actor=false"
    ]

    for arg in default_args:
        arg_key = arg.split("=")[0]
        if not any(arg_key in sys_arg for sys_arg in sys.argv):
            sys.argv.append(arg)

    train_cli()
