import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import copy
import json
import logging
import math
import sys
import tempfile
from collections import deque
from contextlib import contextmanager, nullcontext, redirect_stdout
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import DictConfig, OmegaConf
from pprint import pformat
from tqdm import tqdm

from lerobot.common.logger import Logger
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.utils import populate_queues
from lerobot.common.utils.utils import get_safe_torch_device, init_logging, set_global_seed

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # 确保 ROOT_DIR 是项目根目录
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import env.task.sim_envs  # noqa: F401
from train.s3_finetune.critic import SharedFeatureCritic
from train.s1_pretrain.eval.eval_train import TopKCheckpointManager, custom_eval_policy


def deep_update_dict(base: dict, override: dict) -> dict:
    """递归合并配置字典。"""
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def compute_value_diagnostics(values, returns, eps: float = 1e-8):
    """计算 Critic 的解释方差和相关性。"""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    if values.size == 0 or returns.size == 0:
        return float("nan"), float("nan")

    return_var = np.var(returns)
    explained_variance = (
        float("nan")
        if return_var < eps
        else 1.0 - np.var(returns - values) / (return_var + eps)
    )

    value_std = np.std(values)
    return_std = np.std(returns)
    value_return_corr = (
        float("nan")
        if values.size < 2 or value_std < eps or return_std < eps
        else float(np.corrcoef(values, returns)[0, 1])
    )
    return float(explained_variance), value_return_corr


class RunningMeanStd:
    """原版 DPPO 用的运行均值方差统计器。"""

    def __init__(self, epsilon=1e-4, shape=()):
        """初始化运行均值方差。"""
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        """用一个 batch 更新统计量。"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """用外部均值方差矩更新统计量。"""
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
        """初始化奖励缩放器。"""
        ret_rms_shape = (num_envs,) if per_env else ()
        self.ret_rms = RunningMeanStd(shape=ret_rms_shape)
        self.cliprew = cliprew
        self.ret = np.zeros(num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.per_env = per_env

    def __call__(self, reward, first):
        """更新折扣回报统计并返回缩放奖励。"""
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
        """按运行方差缩放并裁剪奖励。"""
        return np.clip(
            reward / np.sqrt(self.ret_rms.var + self.epsilon),
            -self.cliprew,
            self.cliprew,
        )


class SVMDiscriminator(nn.Module):
    """用成功/失败访问样本训练的 SVM 过程奖励判别器。"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, depth: int = 3):
        """构建二分类 MLP，输出成功访问的 logit。"""
        super().__init__()
        if input_dim <= 0:
            raise ValueError("SVMDiscriminator input_dim must be > 0")
        if depth < 1:
            raise ValueError("SVMDiscriminator depth must be >= 1")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(self.depth):
            layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """让初始 logit 接近 0，避免一开始过程奖励过大。"""
        linear_layers = [module for module in self.net.modules() if isinstance(module, nn.Linear)]
        for module in linear_layers[:-1]:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        last_linear = linear_layers[-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)
        nn.init.constant_(last_linear.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回每个访问特征属于成功分布的 logit。"""
        return self.net(features).squeeze(-1)


class SVMReplayBuffer:
    """分别缓存成功和失败 episode 的访问特征，用于平衡采样。"""

    def __init__(
        self,
        max_transitions: int,
        seed: int = 0,
        feature_dim: int | None = None,
        storage: str = "memory",
        storage_dir: str | Path | None = None,
        reset: bool = True,
    ):
        """初始化正负样本缓存，可选使用磁盘 memmap。"""
        if max_transitions <= 0:
            raise ValueError("SVMReplayBuffer max_transitions must be > 0")
        self.max_transitions = int(max_transitions)
        self.rng = np.random.default_rng(seed)
        storage = str(storage).lower()
        if storage in {"memmap", "mmap"}:
            storage = "disk"
        if storage in {"ram", "deque"}:
            storage = "memory"
        if storage not in {"memory", "disk"}:
            raise ValueError("SVMReplayBuffer storage must be 'memory' or 'disk'")
        self.storage = storage
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        self.metadata_path = None
        self._pos_count = 0
        self._neg_count = 0
        self._pos_write_index = 0
        self._neg_write_index = 0

        if self.storage == "memory":
            self.positive = deque(maxlen=self.max_transitions)
            self.negative = deque(maxlen=self.max_transitions)
            return

        if self.feature_dim is None or self.feature_dim <= 0:
            raise ValueError("feature_dim is required when SVMReplayBuffer storage='disk'")
        if self.storage_dir is None:
            raise ValueError("storage_dir is required when SVMReplayBuffer storage='disk'")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.storage_dir / "metadata.json"
        self.positive_path = self.storage_dir / "positive.dat"
        self.negative_path = self.storage_dir / "negative.dat"

        mode = "w+"
        if not reset and self.metadata_path.exists() and self.positive_path.exists() and self.negative_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if (
                    int(metadata.get("feature_dim", -1)) == self.feature_dim
                    and int(metadata.get("max_transitions", -1)) == self.max_transitions
                ):
                    self._pos_count = int(metadata.get("num_positive", 0))
                    self._neg_count = int(metadata.get("num_negative", 0))
                    self._pos_write_index = int(metadata.get("pos_write_index", 0))
                    self._neg_write_index = int(metadata.get("neg_write_index", 0))
                    mode = "r+"
            except Exception:
                mode = "w+"

        self.positive = np.memmap(
            self.positive_path,
            dtype=np.float32,
            mode=mode,
            shape=(self.max_transitions, self.feature_dim),
        )
        self.negative = np.memmap(
            self.negative_path,
            dtype=np.float32,
            mode=mode,
            shape=(self.max_transitions, self.feature_dim),
        )
        if mode == "w+":
            self._pos_count = 0
            self._neg_count = 0
            self._pos_write_index = 0
            self._neg_write_index = 0
            self._save_metadata()

    @property
    def num_positive(self) -> int:
        """成功访问样本数。"""
        return len(self.positive) if self.storage == "memory" else self._pos_count

    @property
    def num_negative(self) -> int:
        """失败访问样本数。"""
        return len(self.negative) if self.storage == "memory" else self._neg_count

    def _save_metadata(self):
        """保存磁盘缓存的计数和写入位置。"""
        if self.storage != "disk" or self.metadata_path is None:
            return
        metadata = {
            "storage": self.storage,
            "max_transitions": self.max_transitions,
            "feature_dim": self.feature_dim,
            "num_positive": self._pos_count,
            "num_negative": self._neg_count,
            "pos_write_index": self._pos_write_index,
            "neg_write_index": self._neg_write_index,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _append_disk(self, feature: np.ndarray, success: bool):
        """向磁盘环形缓存写入一个访问特征。"""
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        if feature.shape[0] != self.feature_dim:
            raise ValueError(
                f"SVM feature dim mismatch: got {feature.shape[0]}, expected {self.feature_dim}"
            )
        if success:
            self.positive[self._pos_write_index] = feature
            self._pos_write_index = (self._pos_write_index + 1) % self.max_transitions
            self._pos_count = min(self.max_transitions, self._pos_count + 1)
        else:
            self.negative[self._neg_write_index] = feature
            self._neg_write_index = (self._neg_write_index + 1) % self.max_transitions
            self._neg_count = min(self.max_transitions, self._neg_count + 1)

    def add_episode(self, features, success: bool):
        """把一个 episode 内的访问特征按最终标签加入缓存。"""
        if self.storage == "disk":
            for feature in features:
                self._append_disk(feature, bool(success))
            if success:
                self.positive.flush()
            else:
                self.negative.flush()
            self._save_metadata()
            return

        target = self.positive if success else self.negative
        for feature in features:
            target.append(np.asarray(feature, dtype=np.float32).copy())

    def can_train(self, min_positive: int, min_negative: int) -> bool:
        """检查是否已有足够正负样本训练判别器。"""
        return self.num_positive >= int(min_positive) and self.num_negative >= int(min_negative)

    def sample_balanced(self, batch_size: int):
        """正负各采一半，返回特征和二分类标签。"""
        if self.num_positive == 0 or self.num_negative == 0:
            raise RuntimeError("SVMReplayBuffer needs both positive and negative samples")
        pos_count = max(1, int(batch_size) // 2)
        neg_count = max(1, int(batch_size) - pos_count)
        pos_indices = self.rng.integers(0, self.num_positive, size=pos_count)
        neg_indices = self.rng.integers(0, self.num_negative, size=neg_count)
        if self.storage == "disk":
            pos_features = np.asarray(self.positive[pos_indices], dtype=np.float32)
            neg_features = np.asarray(self.negative[neg_indices], dtype=np.float32)
        else:
            pos_features = np.stack([self.positive[int(idx)] for idx in pos_indices], axis=0)
            neg_features = np.stack([self.negative[int(idx)] for idx in neg_indices], axis=0)
        features = np.concatenate([pos_features, neg_features], axis=0).astype(np.float32)
        labels = np.concatenate(
            [
                np.ones(pos_count, dtype=np.float32),
                np.zeros(neg_count, dtype=np.float32),
            ],
            axis=0,
        )
        order = self.rng.permutation(features.shape[0])
        return features[order], labels[order]


def svm_feature_dim(global_cond_dim: int, act_steps: int, action_dim: int, feature_mode: str) -> int:
    """按特征模式计算 SVM 判别器输入维度。"""
    feature_mode = str(feature_mode).lower()
    action_chunk_dim = int(act_steps) * int(action_dim)
    if feature_mode == "global":
        return int(global_cond_dim)
    if feature_mode == "global_action":
        return int(global_cond_dim) + action_chunk_dim
    if feature_mode in {"global_base_action", "global_base_residual"}:
        return int(global_cond_dim) + 2 * action_chunk_dim
    raise ValueError(
        "svm_feature_mode must be one of "
        "['global', 'global_action', 'global_base_action', 'global_base_residual']"
    )


def build_svm_features(
    global_cond: np.ndarray,
    base_actions: np.ndarray,
    action_chunk: np.ndarray,
    action_start: int,
    action_end: int,
    feature_mode: str,
) -> np.ndarray:
    """把 DP 条件、基础动作和执行动作拼成 SVM 访问特征。"""
    feature_mode = str(feature_mode).lower()
    n_envs = int(action_chunk.shape[0])
    global_part = np.asarray(global_cond, dtype=np.float32).reshape(n_envs, -1)
    if feature_mode == "global":
        return np.ascontiguousarray(global_part, dtype=np.float32)

    action_part = np.asarray(action_chunk, dtype=np.float32).reshape(n_envs, -1)
    if feature_mode == "global_action":
        return np.ascontiguousarray(np.concatenate([global_part, action_part], axis=-1))

    base_chunk = np.asarray(
        base_actions[:, action_start:action_end],
        dtype=np.float32,
    )
    base_part = base_chunk.reshape(n_envs, -1)
    if feature_mode == "global_base_action":
        return np.ascontiguousarray(
            np.concatenate([global_part, base_part, action_part], axis=-1),
            dtype=np.float32,
        )
    if feature_mode == "global_base_residual":
        residual_part = (np.asarray(action_chunk, dtype=np.float32) - base_chunk).reshape(n_envs, -1)
        return np.ascontiguousarray(
            np.concatenate([global_part, base_part, residual_part], axis=-1),
            dtype=np.float32,
        )
    raise ValueError(f"Unknown svm_feature_mode={feature_mode!r}")


@torch.no_grad()
def compute_svm_process_reward(
    discriminator: SVMDiscriminator,
    features: np.ndarray,
    device,
    coef: float,
    reward_clip: float,
) -> np.ndarray:
    """用判别器 logit 计算 SVM 过程奖励。"""
    feature_tensor = torch.from_numpy(np.ascontiguousarray(features)).float().to(device)
    logits = discriminator(feature_tensor)
    clipped_logits = torch.clamp(logits, min=-float(reward_clip), max=float(reward_clip))
    return (float(coef) * clipped_logits).detach().cpu().numpy().astype(np.float32)


def train_svm_discriminator(
    discriminator: SVMDiscriminator,
    replay_buffer: SVMReplayBuffer,
    optimizer: torch.optim.Optimizer,
    device,
    batch_size: int,
    updates: int,
):
    """用成功/失败访问样本训练 SVM 判别器。"""
    if updates <= 0:
        return {"loss": float("nan"), "acc": float("nan"), "updates": 0}

    discriminator.train()
    losses = []
    accuracies = []
    for _ in range(int(updates)):
        features_np, labels_np = replay_buffer.sample_balanced(batch_size)
        features = torch.from_numpy(features_np).float().to(device)
        labels = torch.from_numpy(labels_np).float().to(device)
        logits = discriminator(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            accuracy = (predictions == labels).float().mean()
        losses.append(float(loss.item()))
        accuracies.append(float(accuracy.item()))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(np.mean(accuracies)) if accuracies else float("nan"),
        "updates": int(updates),
    }


def log_box(title: str, rows: list[tuple[str, object]], width: int = 78):
    """用统一盒状格式写日志。"""
    key_width = max([len(str(key)) for key, _ in rows] + [0])
    line = "-" * width
    header = "=" * width
    body = [header, title, line]
    for key, value in rows:
        body.append(f"{str(key):<{key_width}} : {value}")
    body.append(header)
    logging.info("\n" + "\n".join(body))


def fmt_pct(value: float) -> str:
    """把比例格式化成百分比字符串。"""
    return f"{value * 100:.1f}%"


def fmt_float(value: float, digits: int = 4) -> str:
    """安全格式化浮点数。"""
    if not np.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}"


@contextmanager
def maybe_suppress_stdout(enabled: bool):
    """按开关临时屏蔽 stdout。"""
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull):
            yield


@contextmanager
def maybe_quiet_eval_progress(enabled: bool):
    """按开关关闭评估进度条。"""
    if not enabled:
        yield
        return

    original_tqdm = custom_eval_policy.__globals__.get("tqdm")
    custom_eval_policy.__globals__["tqdm"] = lambda iterable, *args, **kwargs: iterable
    try:
        yield
    finally:
        if original_tqdm is not None:
            custom_eval_policy.__globals__["tqdm"] = original_tqdm


class ResidualActionMLP(nn.Module):
    """针对一个动作切片预测残差，可选接入全局条件。"""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        extra_dim: int = 0,
    ):
        """构建零初始化输出层的残差 MLP。"""
        super().__init__()
        if action_dim <= 0:
            raise ValueError("ResidualActionMLP action_dim must be > 0")
        if depth < 1:
            raise ValueError("ResidualActionMLP depth must be >= 1")
        if extra_dim < 0:
            raise ValueError("ResidualActionMLP extra_dim must be >= 0")

        self.action_dim = int(action_dim)
        self.extra_dim = int(extra_dim)
        layers: list[nn.Module] = []
        in_dim = action_dim + self.extra_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """初始化隐藏层并让初始残差为 0。"""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        last_linear = self.net[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        actions: torch.Tensor,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """按最后一维动作切片输出同形状残差。"""
        orig_shape = actions.shape
        mlp_input = actions
        if self.extra_dim > 0:
            if global_cond is None:
                raise ValueError("global_cond is required when ResidualActionMLP extra_dim > 0")
            cond = global_cond
            while cond.dim() < actions.dim():
                cond = cond.unsqueeze(1)
            cond = cond.expand(*orig_shape[:-1], cond.shape[-1])
            mlp_input = torch.cat([actions, cond], dim=-1)
        flat_input = mlp_input.reshape(-1, mlp_input.shape[-1])
        return self.net(flat_input).reshape(*orig_shape[:-1], self.action_dim)


class FrozenDiffusionMLPPolicy(nn.Module):
    """
    冻结预训练 Diffusion，并用 residual MLP 调整动作。

    split 模式使用 MLP_arm/MLP_cam 分别调整动作切片；joint 模式使用单个
    20 维 MLP 同时调整完整动作。PPO 只更新 residual MLP 和动作方差。
    """

    def __init__(
        self,
        base_policy: nn.Module,
        action_dim: int,
        action_start: int,
        action_end: int,
        hidden_dim: int = 256,
        depth: int = 2,
        init_std: float = 0.02,
        learn_std: bool = True,
        logprob_reduction: str = "mean",
        arm_indices: list[int] | None = None,
        cam_indices: list[int] | None = None,
        lambda_arm: float = 1.0,
        lambda_cam: float = 1.0,
        arm_hidden_dim: int | None = None,
        cam_hidden_dim: int | None = None,
        arm_depth: int | None = None,
        cam_depth: int | None = None,
        max_delta: float = 0.0,
        residual_mode: str = "split",
        lambda_all: float = 1.0,
        all_hidden_dim: int | None = None,
        all_depth: int | None = None,
        global_cond_dim: int = 0,
        residual_condition: str = "action",
        residual_stepwise_obs: bool = False,
    ):
        """初始化冻结 DP 和可训练 residual MLP。"""
        super().__init__()
        self.base_policy = base_policy
        self.config = base_policy.config
        self.expected_image_keys = base_policy.expected_image_keys
        self.action_start = action_start
        self.action_end = action_end
        self.logprob_reduction = logprob_reduction
        self.action_dim = action_dim
        self.residual_mlp_hidden_dim = hidden_dim
        self.residual_mlp_depth = depth
        self.residual_mlp_learn_std = learn_std
        residual_mode = str(residual_mode).lower()
        if residual_mode in {"all", "single", "joint"}:
            residual_mode = "joint"
        elif residual_mode in {"split", "dual", "separate"}:
            residual_mode = "split"
        else:
            raise ValueError("residual_mode must be 'joint' or 'split'")
        self.residual_mlp_mode = residual_mode
        residual_condition = str(residual_condition).lower()
        if residual_condition in {"none", "action", "base_action"}:
            residual_condition = "action"
        elif residual_condition in {"global", "global_cond", "obs", "condition"}:
            residual_condition = "global_cond"
        else:
            raise ValueError("residual_condition must be 'action' or 'global_cond'")
        self.residual_mlp_condition = residual_condition
        self.residual_mlp_stepwise_obs = bool(residual_stepwise_obs)
        self.residual_global_cond_dim = (
            int(global_cond_dim) if self.residual_mlp_condition == "global_cond" else 0
        )
        if self.residual_global_cond_dim < 0:
            raise ValueError("global_cond_dim must be >= 0")
        if self.residual_mlp_condition == "global_cond" and self.residual_global_cond_dim == 0:
            raise ValueError("global_cond_dim must be > 0 when residual_condition is 'global_cond'")

        default_arm_indices = list(range(min(14, action_dim)))
        default_cam_indices = list(range(14, action_dim)) if action_dim > 14 else []
        self.arm_action_indices = self._validate_action_indices(
            arm_indices if arm_indices is not None else default_arm_indices,
            "arm_indices",
            action_dim,
        )
        self.cam_action_indices = self._validate_action_indices(
            cam_indices if cam_indices is not None else default_cam_indices,
            "cam_indices",
            action_dim,
        )
        self._validate_disjoint_action_indices(
            self.arm_action_indices,
            self.cam_action_indices,
        )
        self.lambda_arm = float(lambda_arm)
        self.lambda_cam = float(lambda_cam)
        self.lambda_all = float(lambda_all)
        self.all_hidden_dim = int(all_hidden_dim if all_hidden_dim is not None else hidden_dim)
        self.all_depth = int(all_depth if all_depth is not None else depth)
        self.arm_hidden_dim = int(arm_hidden_dim if arm_hidden_dim is not None else hidden_dim)
        self.cam_hidden_dim = int(cam_hidden_dim if cam_hidden_dim is not None else hidden_dim)
        self.arm_depth = int(arm_depth if arm_depth is not None else depth)
        self.cam_depth = int(cam_depth if cam_depth is not None else depth)
        self.residual_mlp_max_delta = float(max_delta)
        self.mlp_all = (
            ResidualActionMLP(
                action_dim=action_dim,
                hidden_dim=self.all_hidden_dim,
                depth=self.all_depth,
                extra_dim=self.residual_global_cond_dim,
            )
            if self.residual_mlp_mode == "joint"
            else nn.Identity()
        )
        self.mlp_arm = (
            ResidualActionMLP(
                action_dim=len(self.arm_action_indices),
                hidden_dim=self.arm_hidden_dim,
                depth=self.arm_depth,
                extra_dim=self.residual_global_cond_dim,
            )
            if self.residual_mlp_mode == "split" and self.arm_action_indices
            else nn.Identity()
        )
        self.mlp_cam = (
            ResidualActionMLP(
                action_dim=len(self.cam_action_indices),
                hidden_dim=self.cam_hidden_dim,
                depth=self.cam_depth,
                extra_dim=self.residual_global_cond_dim,
            )
            if self.residual_mlp_mode == "split" and self.cam_action_indices
            else nn.Identity()
        )

        self.register_buffer(
            "_arm_action_index_tensor",
            torch.as_tensor(self.arm_action_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_cam_action_index_tensor",
            torch.as_tensor(self.cam_action_indices, dtype=torch.long),
            persistent=False,
        )
        init_log_std = math.log(max(float(init_std), 1e-6))
        log_std = torch.full((action_dim,), init_log_std, dtype=torch.float32)
        if learn_std:
            self.log_std = nn.Parameter(log_std)
        else:
            self.register_buffer("log_std", log_std)

        self._queues = None
        self.freeze_base_policy()
        self.reset()

    @staticmethod
    def _validate_action_indices(indices, name: str, action_dim: int) -> list[int]:
        """检查动作索引是否唯一且在范围内。"""
        indices = [int(idx) for idx in ([] if indices is None else list(indices))]
        if len(indices) != len(set(indices)):
            raise ValueError(f"{name} contains duplicate indices: {indices}")
        invalid = [idx for idx in indices if idx < 0 or idx >= action_dim]
        if invalid:
            raise ValueError(
                f"{name} has out-of-range indices {invalid}; action_dim={action_dim}"
            )
        return indices

    @staticmethod
    def _validate_disjoint_action_indices(arm_indices, cam_indices):
        """检查双臂和相机臂索引没有重叠。"""
        overlap = sorted(set(arm_indices).intersection(cam_indices))
        if overlap:
            raise ValueError(f"arm_indices and cam_indices overlap: {overlap}")

    def freeze_base_policy(self):
        """冻结预训练 DP 参数并保持 eval 模式。"""
        self.base_policy.eval()
        for param in self.base_policy.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """切换训练状态，但强制 DP 仍处于 eval。"""
        super().train(mode)
        self.base_policy.eval()
        return self

    def reset(self):
        """重置观测和动作队列。"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if len(self.expected_image_keys) > 0:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if getattr(self.base_policy, "use_env_state", False):
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()

    @property
    def action_std(self) -> torch.Tensor:
        """返回当前动作采样标准差。"""
        return self.log_std.exp().clamp(min=1e-6)

    def adapter_parameters(self):
        """返回 PPO 需要更新的 residual 参数。"""
        return [p for p in self.parameters() if p.requires_grad]

    def normalize_history_batch(self, cond: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """按 DP 输入规范归一化历史观测。"""
        batch = self.base_policy.normalize_inputs(cond.copy())
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[k] for k in self.expected_image_keys],
                dim=-4,
            )
        return batch

    @torch.no_grad()
    def frozen_diffusion_actions_from_normalized_batch(
        self,
        batch: dict[str, torch.Tensor],
        return_global_cond: bool = False,
    ):
        """用冻结 DP 从已归一化观测采样基础动作。"""
        self.base_policy.eval()
        batch_size = next(iter(batch.values())).shape[0]
        global_cond = self.base_policy.diffusion._prepare_global_conditioning(batch)
        normalized_actions = self.base_policy.diffusion.conditional_sample(
            batch_size=batch_size,
            global_cond=global_cond,
        )
        actions = self.base_policy.unnormalize_outputs({"action": normalized_actions})["action"]
        if return_global_cond:
            return actions, global_cond
        return actions

    @torch.no_grad()
    def frozen_diffusion_actions(
        self,
        cond: dict[str, torch.Tensor],
        return_global_cond: bool = False,
    ):
        """归一化观测后调用冻结 DP 生成动作。"""
        batch = self.normalize_history_batch(cond)
        return self.frozen_diffusion_actions_from_normalized_batch(
            batch,
            return_global_cond=return_global_cond,
        )

    def mean_actions(
        self,
        base_actions: torch.Tensor,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """把 residual MLP 的输出加回 DP 动作。"""
        if self.residual_mlp_mode == "joint":
            delta_all = self.limit_residual_delta(
                self.mlp_all(base_actions, global_cond=global_cond)
            )
            return base_actions + self.lambda_all * delta_all

        mean_actions = base_actions.clone()
        if self.arm_action_indices:
            arm_idx = self._arm_action_index_tensor.to(base_actions.device)
            arm_base = base_actions.index_select(-1, arm_idx)
            delta_arm = self.limit_residual_delta(
                self.mlp_arm(arm_base, global_cond=global_cond)
            )
            mean_actions[..., arm_idx] = arm_base + self.lambda_arm * delta_arm
        if self.cam_action_indices:
            cam_idx = self._cam_action_index_tensor.to(base_actions.device)
            cam_base = base_actions.index_select(-1, cam_idx)
            delta_cam = self.limit_residual_delta(
                self.mlp_cam(cam_base, global_cond=global_cond)
            )
            mean_actions[..., cam_idx] = cam_base + self.lambda_cam * delta_cam
        return mean_actions

    def limit_residual_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """用平滑限幅约束 residual，防止 MLP 逐轮偏离冻结 Diffusion。"""
        if self.residual_mlp_max_delta <= 0.0:
            return delta
        max_delta = torch.as_tensor(
            self.residual_mlp_max_delta,
            dtype=delta.dtype,
            device=delta.device,
        )
        return torch.tanh(delta / max_delta) * max_delta

    def sample_from_mean(self, mean_actions: torch.Tensor, deterministic: bool):
        """从 residual 后的动作均值采样 PPO 动作。"""
        if deterministic:
            actions = mean_actions
        else:
            std = self.action_std.view(1, 1, -1)
            actions = mean_actions + torch.randn_like(mean_actions) * std
        log_probs = self.log_prob_from_mean(mean_actions, actions)
        return actions, log_probs

    def log_prob_from_mean(
        self,
        mean_actions: torch.Tensor,
        actions: torch.Tensor,
        action_start: int | None = None,
        action_end: int | None = None,
    ) -> torch.Tensor:
        """计算执行动作块在当前高斯策略下的 log_prob。"""
        start = self.action_start if action_start is None else action_start
        end = self.action_end if action_end is None else action_end
        mean_chunk = mean_actions[:, start:end]
        action_chunk = actions[:, start:end] if actions.shape[1] == mean_actions.shape[1] else actions

        std = self.action_std.view(1, 1, -1)
        var = std.square()
        log_probs = -0.5 * (action_chunk - mean_chunk).square() / var
        log_probs = log_probs - torch.log(std) - 0.5 * math.log(2 * math.pi)
        log_probs = torch.clamp(log_probs, min=-20.0, max=5.0)
        if self.logprob_reduction == "sum":
            return log_probs.sum(dim=(-1, -2))
        if self.logprob_reduction == "mean":
            return log_probs.mean(dim=(-1, -2))
        raise ValueError(f"Unknown logprob_reduction={self.logprob_reduction!r}")

    def log_prob(
        self,
        base_actions: torch.Tensor,
        executed_actions: torch.Tensor,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """从 DP 基础动作重算 residual 均值并求 log_prob。"""
        mean_actions = self.mean_actions(base_actions, global_cond=global_cond)
        return self.log_prob_from_mean(mean_actions, executed_actions)

    def forward_history(
        self,
        cond: dict[str, torch.Tensor],
        deterministic: bool = False,
        return_global_cond: bool = False,
    ):
        """完整前向：冻结 DP 生成动作，residual MLP 修正并采样。"""
        base_actions, global_cond = self.frozen_diffusion_actions(cond, return_global_cond=True)
        mean_actions = self.mean_actions(base_actions, global_cond=global_cond)
        actions, log_probs = self.sample_from_mean(mean_actions, deterministic=deterministic)
        result = {
            "base_actions": base_actions,
            "mean_actions": mean_actions,
            "actions": actions,
            "log_probs": log_probs,
        }
        if return_global_cond:
            result["global_cond"] = global_cond
        return result

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """评估时按 LeRobot 队列接口逐步取动作。"""
        batch = self.base_policy.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[k] for k in self.expected_image_keys],
                dim=-4,
            )

        self._queues = populate_queues(self._queues, batch)
        latest_global_cond = None
        if len(self._queues["action"]) == 0:
            history_batch = {
                k: torch.stack(list(self._queues[k]), dim=1)
                for k in batch
                if k in self._queues
            }
            base_actions, global_cond = self.frozen_diffusion_actions_from_normalized_batch(
                history_batch,
                return_global_cond=True,
            )
            if self.residual_mlp_stepwise_obs:
                action_chunk = base_actions[:, self.action_start:self.action_end]
                latest_global_cond = global_cond
            else:
                mean_actions = self.mean_actions(base_actions, global_cond=global_cond)
                action_chunk = mean_actions[:, self.action_start:self.action_end]
            self._queues["action"].extend(action_chunk.transpose(0, 1))

        queued_action = self._queues["action"].popleft()
        if not self.residual_mlp_stepwise_obs:
            return queued_action

        if latest_global_cond is None:
            history_batch = {
                k: torch.stack(list(self._queues[k]), dim=1)
                for k in batch
                if k in self._queues
            }
            latest_global_cond = self.base_policy.diffusion._prepare_global_conditioning(
                history_batch
            )
        return self.mean_actions(
            queued_action.unsqueeze(1),
            global_cond=latest_global_cond,
        ).squeeze(1)

    def save_pretrained(self, save_directory: str | Path):
        """保存冻结 DP 权重引用和 residual MLP 状态。"""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.base_policy.save_pretrained(save_directory)
        adapter_state = {
            "log_std": self.log_std.detach().cpu(),
            "action_start": self.action_start,
            "action_end": self.action_end,
            "logprob_reduction": self.logprob_reduction,
            "action_dim": self.action_dim,
            "hidden_dim": self.residual_mlp_hidden_dim,
            "depth": self.residual_mlp_depth,
            "learn_std": self.residual_mlp_learn_std,
            "residual_mlp_mode": self.residual_mlp_mode,
            "residual_mlp_condition": self.residual_mlp_condition,
            "residual_mlp_stepwise_obs": self.residual_mlp_stepwise_obs,
            "global_cond_dim": self.residual_global_cond_dim,
            "mlp_all": self.mlp_all.state_dict(),
            "mlp_arm": self.mlp_arm.state_dict(),
            "mlp_cam": self.mlp_cam.state_dict(),
            "arm_action_indices": self.arm_action_indices,
            "cam_action_indices": self.cam_action_indices,
            "lambda_all": self.lambda_all,
            "lambda_arm": self.lambda_arm,
            "lambda_cam": self.lambda_cam,
            "all_hidden_dim": self.all_hidden_dim,
            "all_depth": self.all_depth,
            "arm_hidden_dim": self.arm_hidden_dim,
            "cam_hidden_dim": self.cam_hidden_dim,
            "arm_depth": self.arm_depth,
            "cam_depth": self.cam_depth,
            "max_delta": self.residual_mlp_max_delta,
        }
        torch.save(adapter_state, save_directory / "residual_mlp.pt")
        adapter_config = {
            "wrapper": "FrozenDiffusionMLPPolicy",
            "action_start": self.action_start,
            "action_end": self.action_end,
            "logprob_reduction": self.logprob_reduction,
            "action_dim": self.action_dim,
            "hidden_dim": self.residual_mlp_hidden_dim,
            "depth": self.residual_mlp_depth,
            "learn_std": self.residual_mlp_learn_std,
            "residual_mlp_mode": self.residual_mlp_mode,
            "residual_mlp_condition": self.residual_mlp_condition,
            "residual_mlp_stepwise_obs": self.residual_mlp_stepwise_obs,
            "global_cond_dim": self.residual_global_cond_dim,
            "arm_action_indices": self.arm_action_indices,
            "cam_action_indices": self.cam_action_indices,
            "lambda_all": self.lambda_all,
            "lambda_arm": self.lambda_arm,
            "lambda_cam": self.lambda_cam,
            "all_hidden_dim": self.all_hidden_dim,
            "all_depth": self.all_depth,
            "arm_hidden_dim": self.arm_hidden_dim,
            "cam_hidden_dim": self.cam_hidden_dim,
            "arm_depth": self.arm_depth,
            "cam_depth": self.cam_depth,
            "max_delta": self.residual_mlp_max_delta,
        }
        with open(save_directory / "residual_mlp_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2, ensure_ascii=False)


def freeze_batch_norm(module: nn.Module):
    """冻结 BatchNorm 的运行状态。"""
    if "BatchNorm" in module.__class__.__name__:
        module.eval()


def flatten_lerobot_obs(obs_dict):
    """把环境观测展开成 LeRobot policy 使用的键。"""
    flat_obs = {}
    if "pixels" in obs_dict:
        for cam_name, img_array in obs_dict["pixels"].items():
            flat_obs[f"observation.images.{cam_name}"] = img_array
    if "agent_pos" in obs_dict:
        flat_obs["observation.state"] = obs_dict["agent_pos"]
    for key, value in obs_dict.items():
        if key not in ["pixels", "agent_pos"]:
            flat_obs[key] = value
    return flat_obs


def clone_obs_value(value):
    """复制观测值以避免队列中引用被复用。"""
    if hasattr(value, "copy"):
        return value.copy()
    return value


def reset_full_obs_queue(queue, obs, n_obs_steps):
    """用当前观测填满历史队列。"""
    for key, value in obs.items():
        if key not in queue:
            queue[key] = deque(maxlen=n_obs_steps)
        queue[key].clear()
        for _ in range(n_obs_steps):
            queue[key].append(clone_obs_value(value))


def append_obs_queue(queue, obs, n_obs_steps):
    """向历史观测队列追加一帧。"""
    for key, value in obs.items():
        if key not in queue:
            queue[key] = deque(maxlen=n_obs_steps)
            for _ in range(n_obs_steps - 1):
                queue[key].append(clone_obs_value(value))
        queue[key].append(clone_obs_value(value))


def reset_done_envs_in_obs_queue(queue, obs, done_mask, n_envs, n_obs_steps):
    """对已结束环境重置其历史观测。"""
    done_mask = np.asarray(done_mask, dtype=bool)
    if not done_mask.any():
        return
    if n_envs == 1:
        reset_full_obs_queue(queue, obs, n_obs_steps)
        return

    for key, value in obs.items():
        if key not in queue:
            continue
        for env_idx in np.flatnonzero(done_mask):
            reset_frame = np.array(value[env_idx], copy=True)
            for q_idx in range(len(queue[key])):
                queue[key][q_idx][env_idx] = reset_frame


def stack_obs_queue(queue, n_envs, n_obs_steps):
    """把历史观测队列堆成 batch。"""
    stacked_obs = {}
    for key, frames in queue.items():
        if len(frames) != n_obs_steps:
            raise RuntimeError(f"Observation queue {key} length {len(frames)} != {n_obs_steps}")
        stacked_value = np.stack(list(frames), axis=0 if n_envs == 1 else 1)
        if n_envs == 1:
            stacked_value = np.expand_dims(stacked_value, axis=0)
        stacked_obs[key] = stacked_value
    return stacked_obs


def info_success_mask(info, done_mask, n_envs):
    """从 Gym info 中提取每个环境的成功标记。"""
    done_mask = np.asarray(done_mask, dtype=bool)
    success = np.zeros(n_envs, dtype=bool)

    def fill_success(raw_success, raw_mask):
        """把不同形状的 success 字段写入统一数组。"""
        raw_success = np.asarray(raw_success)
        raw_mask = np.asarray(raw_mask, dtype=bool)
        if raw_success.shape == ():
            success[raw_mask] = bool(raw_success.item())
            return
        limit = min(len(raw_success), n_envs)
        valid_mask = raw_mask[:limit]
        success[:limit] = success[:limit] | (raw_success[:limit].astype(bool) & valid_mask)

    if isinstance(info, dict) and "final_info" in info:
        final_info = info["final_info"]
        final_mask = np.asarray(info.get("_final_info", done_mask), dtype=bool)
        if isinstance(final_info, dict):
            if "is_success" in final_info:
                fill_success(final_info["is_success"], final_info.get("_is_success", final_mask))
        else:
            for env_idx in np.flatnonzero(done_mask):
                try:
                    env_info = final_info[env_idx]
                    if isinstance(env_info, dict):
                        success[env_idx] = bool(env_info.get("is_success", False))
                except Exception:
                    success[env_idx] = False
        if "is_success" in info:
            fill_success(info["is_success"], info.get("_is_success", done_mask))
    elif isinstance(info, dict) and "is_success" in info:
        fill_success(info["is_success"], info.get("_is_success", done_mask))

    return success & done_mask


def success_label_chunk_rewards(
    done_mask,
    success_mask,
    success_reward: float,
    failure_reward: float,
    nonterminal_reward: float,
    success_time_bonus: float = 0.0,
    episode_steps=None,
    max_episode_steps: int = 0,
):
    """把 success/fail 标签转换成当前 action chunk 的训练 reward，可选成功时间引导。"""
    done_mask = np.asarray(done_mask, dtype=bool)
    success_mask = np.asarray(success_mask, dtype=bool)
    rewards = np.full(done_mask.shape, float(nonterminal_reward), dtype=np.float32)
    rewards[done_mask] = float(failure_reward)
    success_done = done_mask & success_mask
    rewards[success_done] = float(success_reward)
    if success_time_bonus > 0.0 and max_episode_steps > 0 and episode_steps is not None:
        elapsed = np.asarray(episode_steps, dtype=np.float32)
        time_fraction = np.clip(elapsed / float(max_episode_steps), 0.0, 1.0)
        bonus = float(success_time_bonus) * (1.0 - time_fraction)
        rewards[success_done] = float(success_reward) + bonus[success_done]
    return rewards


def build_history_batch(stacked_raw_obs, policy, device):
    """把 numpy 历史观测转为 policy 输入 tensor。"""
    batch_obs = {}
    for key, value in stacked_raw_obs.items():
        if key not in policy.config.input_shapes:
            continue
        tensor_value = torch.from_numpy(np.ascontiguousarray(value)).float().to(device)
        if "images" in key:
            tensor_value = tensor_value.permute(0, 1, 4, 2, 3) / 255.0
        batch_obs[key] = tensor_value
    return batch_obs


def global_cond_from_obs(policy, obs_batch):
    """从观测 batch 提取 DP 的全局条件特征。"""
    obs_norm = policy.base_policy.normalize_inputs(obs_batch.copy())
    if len(policy.expected_image_keys) > 0:
        obs_norm = dict(obs_norm)
        obs_norm["observation.images"] = torch.stack(
            [obs_norm[k] for k in policy.expected_image_keys],
            dim=-4,
        )
    return policy.base_policy.diffusion._prepare_global_conditioning(obs_norm)


def load_frozen_base_policy(cfg: DictConfig, device):
    """加载并冻结预训练 Diffusion policy。"""
    ckpt_path = cfg.training.pretrained_ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find pretrained checkpoint: {ckpt_path}")

    hf_model_dir = os.path.join(ckpt_path, "pretrained_model")
    load_dir = hf_model_dir if os.path.exists(hf_model_dir) else ckpt_path
    logging.info(f"Loading frozen diffusion policy from: {load_dir}")

    from lerobot.common.utils.utils import init_hydra_config

    config_yaml_path = Path(load_dir) / "config.yaml"
    if not config_yaml_path.exists():
        config_yaml_path = Path(load_dir).parent / "config.yaml"
    if not config_yaml_path.exists():
        raise FileNotFoundError(f"Cannot find config.yaml near {load_dir}")

    hydra_cfg = init_hydra_config(str(config_yaml_path))
    try:
        hydra_cfg.device = str(device)
    except Exception:
        pass

    base_policy = make_policy(
        hydra_cfg=hydra_cfg,
        pretrained_policy_name_or_path=str(load_dir),
    )
    base_policy.to(device)
    base_policy.eval()
    for param in base_policy.parameters():
        param.requires_grad = False

    if "policy" in cfg:
        base_policy.config.n_action_steps = int(
            getattr(cfg.policy, "n_action_steps", getattr(base_policy.config, "n_action_steps", 8))
        )

    return base_policy, hydra_cfg, Path(load_dir)


def infer_global_cond_dim(base_policy, device):
    """用 dummy 输入推断 Critic 需要的全局条件维度。"""
    n_obs_steps = int(getattr(base_policy.config, "n_obs_steps", 2))
    dummy_batch = {
        key: torch.zeros((1, n_obs_steps, *shape), device=device)
        for key, shape in base_policy.config.input_shapes.items()
    }
    if len(base_policy.expected_image_keys) > 0:
        dummy_batch["observation.images"] = torch.stack(
            [dummy_batch[k] for k in base_policy.expected_image_keys],
            dim=-4,
        )
    with torch.no_grad():
        dummy_cond = base_policy.diffusion._prepare_global_conditioning(dummy_batch)
    return dummy_cond.shape[-1]


def snapshot_trainable_state(policy: FrozenDiffusionMLPPolicy):
    """拷贝 residual MLP 和动作方差状态。"""
    state = {
        "log_std": policy.log_std.detach().cpu().clone(),
        "residual_mlp_mode": policy.residual_mlp_mode,
    }
    if policy.residual_mlp_mode == "joint":
        state["mlp_all"] = copy.deepcopy(policy.mlp_all.state_dict())
    else:
        state["mlp_arm"] = copy.deepcopy(policy.mlp_arm.state_dict())
        state["mlp_cam"] = copy.deepcopy(policy.mlp_cam.state_dict())
    return state


def restore_trainable_state(policy: FrozenDiffusionMLPPolicy, state, device):
    """恢复 residual MLP 和动作方差状态。"""
    if policy.residual_mlp_mode == "joint":
        policy.mlp_all.load_state_dict(state["mlp_all"], strict=True)
    else:
        policy.mlp_arm.load_state_dict(state["mlp_arm"], strict=True)
        policy.mlp_cam.load_state_dict(state["mlp_cam"], strict=True)
    with torch.no_grad():
        policy.log_std.copy_(state["log_std"].to(device))
    policy.to(device)


def train_mlp_finetune(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    """运行冻结 DP + residual MLP 的 PPO 微调。"""
    init_logging()
    log_box(
        "DP+MLP Finetune",
        [
            ("job", job_name or "-"),
            ("out_dir", out_dir or "-"),
            ("seed", cfg.seed),
            ("device", cfg.device),
        ],
    )
    if bool(getattr(cfg.training, "print_full_config", False)):
        logging.info(f"Config:\n{pformat(OmegaConf.to_container(cfg))}")

    Logger(cfg, out_dir, wandb_job_name=job_name)
    set_global_seed(cfg.seed)
    device = get_safe_torch_device(cfg.device, log=True)
    quiet_terminal = bool(getattr(cfg.training, "quiet_terminal", True))

    base_policy, hydra_cfg, load_dir = load_frozen_base_policy(cfg, device)
    action_dim = base_policy.config.output_shapes["action"][0]
    horizon_steps = int(base_policy.config.horizon)
    n_obs_steps = int(getattr(base_policy.config, "n_obs_steps", 2))
    act_steps = int(getattr(cfg.policy, "n_action_steps", getattr(base_policy.config, "n_action_steps", 8)))
    action_start = n_obs_steps - 1
    action_end = action_start + act_steps
    if action_end > horizon_steps:
        raise ValueError(
            f"Action slice is out of horizon: n_obs_steps={n_obs_steps}, "
            f"act_steps={act_steps}, horizon={horizon_steps}"
        )

    def cfg_int_list(key: str, default: list[int]) -> list[int]:
        """从配置读取整数列表。"""
        value = getattr(cfg.training, key, default)
        if value is None:
            return []
        return [int(item) for item in list(value)]

    default_arm_indices = list(range(min(14, action_dim)))
    default_cam_indices = list(range(14, action_dim)) if action_dim > 14 else []
    arm_action_indices = cfg_int_list("arm_action_indices", default_arm_indices)
    cam_action_indices = cfg_int_list("cam_action_indices", default_cam_indices)
    residual_hidden_dim = int(
        getattr(
            cfg.training,
            "residual_mlp_hidden_dim",
            256,
        )
    )
    residual_depth = int(
        getattr(
            cfg.training,
            "residual_mlp_depth",
            2,
        )
    )
    residual_std = float(
        getattr(
            cfg.training,
            "residual_mlp_std",
            getattr(cfg.training, "min_sampling_denoising_std", 0.02),
        )
    )
    residual_learn_std = bool(
        getattr(
            cfg.training,
            "residual_mlp_learn_std",
            True,
        )
    )
    lambda_arm = float(getattr(cfg.training, "lambda_arm", 1.0))
    lambda_cam = float(getattr(cfg.training, "lambda_cam", 1.0))
    lambda_all = float(getattr(cfg.training, "lambda_all", 1.0))
    residual_mlp_mode = str(getattr(cfg.training, "residual_mlp_mode", "split")).lower()
    residual_mlp_condition = str(
        getattr(cfg.training, "residual_mlp_condition", "action")
    ).lower()
    residual_mlp_stepwise_obs = bool(
        getattr(cfg.training, "residual_mlp_stepwise_obs", False)
    )
    ppo_control_step_training = bool(
        getattr(cfg.training, "ppo_control_step_training", residual_mlp_stepwise_obs)
    )
    if ppo_control_step_training and not residual_mlp_stepwise_obs:
        raise ValueError(
            "training.ppo_control_step_training=true requires "
            "training.residual_mlp_stepwise_obs=true"
        )
    residual_max_delta = float(getattr(cfg.training, "residual_mlp_max_delta", 0.0))
    all_hidden_dim = int(getattr(cfg.training, "residual_mlp_all_hidden_dim", residual_hidden_dim))
    arm_hidden_dim = int(getattr(cfg.training, "residual_mlp_arm_hidden_dim", residual_hidden_dim))
    cam_hidden_dim = int(getattr(cfg.training, "residual_mlp_cam_hidden_dim", residual_hidden_dim))
    all_depth = int(getattr(cfg.training, "residual_mlp_all_depth", residual_depth))
    arm_depth = int(getattr(cfg.training, "residual_mlp_arm_depth", residual_depth))
    cam_depth = int(getattr(cfg.training, "residual_mlp_cam_depth", residual_depth))

    ref_cams = [
        key.replace("observation.images.", "")
        for key in base_policy.config.input_shapes.keys()
        if "observation.images." in key
    ]
    if not ref_cams or action_dim is None:
        raise ValueError(f"Invalid policy snapshot: ref_cams={ref_cams}, action_dim={action_dim}")

    with open(load_dir / "config.yaml", "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    env_cfg = full_cfg.get("env", {})
    env_name = env_cfg.get("name")
    env_task = env_cfg.get("task")
    if not env_name or not env_task:
        raise ValueError("The pretrained config.yaml must contain env.name and env.task")
    env_id = f"{env_name}/{env_task}"

    n_envs = int(getattr(cfg.env, "n_envs", 1))
    render_cams = getattr(cfg.eval, "render_camera", [])
    if render_cams is None:
        render_cams = []
    elif isinstance(render_cams, str):
        render_cams = [render_cams]
    else:
        render_cams = list(render_cams)
    train_cameras = list(dict.fromkeys(ref_cams))
    eval_cameras = list(dict.fromkeys(ref_cams + render_cams))

    if n_envs > 1:
        env = gym.vector.AsyncVectorEnv(
            [lambda: gym.make(id=env_id, cameras=train_cameras) for _ in range(n_envs)],
            shared_memory=True,
            context="spawn",
            autoreset_mode="SameStep",
        )
    else:
        env = gym.make(id=env_id, cameras=train_cameras)
    eval_env = gym.make(id=env_id, cameras=eval_cameras)

    with maybe_suppress_stdout(quiet_terminal):
        global_cond_dim = infer_global_cond_dim(base_policy, device)
    critic = SharedFeatureCritic(global_cond_dim=global_cond_dim).to(device)

    policy = FrozenDiffusionMLPPolicy(
        base_policy=base_policy,
        action_dim=action_dim,
        action_start=action_start,
        action_end=action_end,
        hidden_dim=residual_hidden_dim,
        depth=residual_depth,
        init_std=residual_std,
        learn_std=residual_learn_std,
        logprob_reduction=str(getattr(cfg.training, "logprob_reduction", "mean")),
        arm_indices=arm_action_indices,
        cam_indices=cam_action_indices,
        lambda_arm=lambda_arm,
        lambda_cam=lambda_cam,
        arm_hidden_dim=arm_hidden_dim,
        cam_hidden_dim=cam_hidden_dim,
        arm_depth=arm_depth,
        cam_depth=cam_depth,
        max_delta=residual_max_delta,
        residual_mode=residual_mlp_mode,
        lambda_all=lambda_all,
        all_hidden_dim=all_hidden_dim,
        all_depth=all_depth,
        global_cond_dim=global_cond_dim,
        residual_condition=residual_mlp_condition,
        residual_stepwise_obs=residual_mlp_stepwise_obs,
    ).to(device)
    policy.freeze_base_policy()

    actor_optimizer = torch.optim.AdamW(
        policy.adapter_parameters(),
        lr=float(getattr(cfg.training, "actor_lr", 1e-5)),
        weight_decay=float(getattr(cfg.training, "weight_decay", 1e-6)),
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(getattr(cfg.training, "critic_lr", 3e-4)),
    )
    from torch.optim.lr_scheduler import LinearLR

    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=0.1,
        total_iters=int(getattr(cfg.training, "actor_lr_warmup_iters", 5)),
    )

    max_checkpoints = int(getattr(cfg.eval, "max_checkpoints", 5))
    checkpoint_metric = str(getattr(cfg.eval, "checkpoint_metric", "reward")).lower()
    success_metric_names = {"success", "success_rate", "sr"}
    manager_metric = "reward" if checkpoint_metric in success_metric_names else checkpoint_metric
    manager = TopKCheckpointManager(
        out_dir=out_dir,
        max_keep=max_checkpoints,
        records_resume=bool(getattr(cfg.eval, "records_resume", True)),
        metric=manager_metric,
    )

    n_steps = int(getattr(cfg.training, "rollout_steps", 300))
    critic_warmup_iters = int(getattr(cfg.training, "n_critic_warmup_itr", 5))
    batch_size = int(getattr(cfg.training, "batch_size", 32))
    update_epochs = int(getattr(cfg.training, "update_epochs", 4))
    clip_ratio = float(getattr(cfg.training, "clip_ratio", 0.1))
    target_kl = float(getattr(cfg.training, "target_kl", 0.03))
    bc_coef = float(getattr(cfg.training, "bc_loss_coef", 0.0))
    gamma = float(getattr(cfg.training, "gamma", 0.99))
    gae_lambda = float(getattr(cfg.training, "gae_lambda", 0.95))
    reward_scale_running = bool(getattr(cfg.training, "reward_scale_running", True))
    reward_scale_const = float(getattr(cfg.training, "reward_scale_const", 1.0))
    norm_adv = bool(getattr(cfg.training, "norm_adv", True))
    adv_lower_q = float(getattr(cfg.training, "clip_advantage_lower_quantile", 0.05))
    adv_upper_q = float(getattr(cfg.training, "clip_advantage_upper_quantile", 0.95))
    use_disk_cache = bool(getattr(cfg.training, "use_disk_cache", False))
    skip_update = bool(getattr(cfg.training, "skip_update", False))
    update_actor = bool(getattr(cfg.training, "update_actor", True))
    grad_accum_steps = max(1, int(getattr(cfg.training, "grad_accumulate", 1)))
    show_progress = bool(getattr(cfg.training, "show_progress", False))
    rollout_progress = bool(getattr(cfg.training, "rollout_progress", show_progress))
    rollout_log_interval = max(0, int(getattr(cfg.training, "rollout_log_interval", 100)))
    training_progress = bool(getattr(cfg.training, "training_progress", show_progress))
    training_log_interval = max(0, int(getattr(cfg.training, "training_log_interval", 100)))
    reward_source = str(getattr(cfg.training, "reward_source", "env")).lower()
    valid_reward_sources = {"env", "success_label"}
    if reward_source not in valid_reward_sources:
        raise ValueError(
            f"training.reward_source must be one of {sorted(valid_reward_sources)}, "
            f"got {reward_source!r}"
        )
    label_success_reward = float(getattr(cfg.training, "success_label_success_reward", 1.0))
    label_failure_reward = float(getattr(cfg.training, "success_label_failure_reward", -1.0))
    label_nonterminal_reward = float(getattr(cfg.training, "success_label_nonterminal_reward", 0.0))
    success_time_bonus = float(getattr(cfg.training, "success_time_bonus", 0.0))
    success_time_max_steps = int(
        getattr(cfg.training, "success_time_max_steps", getattr(cfg.eval, "max_steps", 0))
    )
    bootstrap_on_truncation = bool(
        getattr(cfg.training, "bootstrap_on_truncation", reward_source == "env")
    )
    svm_reward_enable = bool(getattr(cfg.training, "svm_reward_enable", False))
    svm_feature_mode = str(
        getattr(cfg.training, "svm_feature_mode", "global_base_residual")
    ).lower()
    svm_reward_coef = float(getattr(cfg.training, "svm_reward_coef", 0.05))
    svm_reward_clip = float(getattr(cfg.training, "svm_reward_clip", 5.0))
    svm_disc_hidden_dim = int(getattr(cfg.training, "svm_disc_hidden_dim", 256))
    svm_disc_depth = int(getattr(cfg.training, "svm_disc_depth", 3))
    svm_disc_lr = float(getattr(cfg.training, "svm_disc_lr", 1e-4))
    svm_disc_weight_decay = float(getattr(cfg.training, "svm_disc_weight_decay", 0.0))
    svm_disc_batch_size = int(getattr(cfg.training, "svm_disc_batch_size", 64))
    svm_disc_updates = int(getattr(cfg.training, "svm_disc_updates", 32))
    svm_buffer_max_transitions = int(getattr(cfg.training, "svm_buffer_max_transitions", 50000))
    svm_buffer_storage = str(getattr(cfg.training, "svm_buffer_storage", "disk")).lower()
    svm_buffer_dir = Path(str(getattr(cfg.training, "svm_buffer_dir", "outputs/buffer")))
    if not svm_buffer_dir.is_absolute():
        svm_buffer_dir = Path(ROOT_DIR) / svm_buffer_dir
    svm_buffer_reset = bool(getattr(cfg.training, "svm_buffer_reset", True))
    svm_min_positive = int(getattr(cfg.training, "svm_min_positive", 256))
    svm_min_negative = int(getattr(cfg.training, "svm_min_negative", 256))
    svm_feature_steps = 1 if ppo_control_step_training else act_steps
    svm_input_dim = svm_feature_dim(
        global_cond_dim=global_cond_dim,
        act_steps=svm_feature_steps,
        action_dim=action_dim,
        feature_mode=svm_feature_mode,
    )
    svm_discriminator = None
    svm_optimizer = None
    svm_replay_buffer = None
    svm_reward_ready = False
    svm_last_stats = {"loss": float("nan"), "acc": float("nan"), "updates": 0}
    if svm_reward_enable:
        svm_discriminator = SVMDiscriminator(
            input_dim=svm_input_dim,
            hidden_dim=svm_disc_hidden_dim,
            depth=svm_disc_depth,
        ).to(device)
        svm_optimizer = torch.optim.AdamW(
            svm_discriminator.parameters(),
            lr=svm_disc_lr,
            weight_decay=svm_disc_weight_decay,
        )
        svm_replay_buffer = SVMReplayBuffer(
            max_transitions=svm_buffer_max_transitions,
            seed=int(cfg.seed),
            feature_dim=svm_input_dim,
            storage=svm_buffer_storage,
            storage_dir=svm_buffer_dir,
            reset=svm_buffer_reset,
        )

    log_box(
        "Run Setup",
        [
            ("env", env_id),
            ("train_cameras", train_cameras),
            ("eval_cameras", eval_cameras),
            ("checkpoint", load_dir),
            ("action_dim", action_dim),
            ("horizon", horizon_steps),
            ("action_slice", f"[{action_start}:{action_end}]"),
            ("residual_mode", policy.residual_mlp_mode),
            ("residual_condition", policy.residual_mlp_condition),
            ("residual_stepwise_obs", residual_mlp_stepwise_obs),
            ("ppo_control_step_training", ppo_control_step_training),
            ("global_cond_dim", policy.residual_global_cond_dim),
            ("arm_indices", arm_action_indices),
            ("cam_indices", cam_action_indices),
            ("lambda_all", f"{lambda_all:.3f}"),
            ("lambda_arm/cam", f"{lambda_arm:.3f} / {lambda_cam:.3f}"),
            ("residual_max_delta", residual_max_delta),
            ("n_envs", n_envs),
            ("rollout_steps", n_steps),
            ("rollout_env_steps", n_steps * act_steps * n_envs),
            ("batch_size", batch_size),
            ("update_epochs", update_epochs),
            ("critic_warmup", critic_warmup_iters),
            ("actor_lr", f"{float(getattr(cfg.training, 'actor_lr', 1e-5)):.3e}"),
            ("critic_lr", f"{float(getattr(cfg.training, 'critic_lr', 3e-4)):.3e}"),
            ("clip_ratio", clip_ratio),
            ("target_kl", target_kl),
            ("reward_source", reward_source),
            ("success/fail reward", f"{label_success_reward:.2f} / {label_failure_reward:.2f}"),
            ("nonterminal reward", f"{label_nonterminal_reward:.2f}"),
            ("success_time_bonus", success_time_bonus),
            ("success_time_max_steps", success_time_max_steps),
            ("bootstrap_on_truncation", bootstrap_on_truncation),
            ("svm_reward_enable", svm_reward_enable),
            ("svm_feature_mode", svm_feature_mode),
            ("svm_feature_steps", svm_feature_steps if svm_reward_enable else "disabled"),
            ("svm_reward_coef/clip", f"{svm_reward_coef:.3f} / {svm_reward_clip:.2f}"),
            ("svm_buffer_storage", svm_buffer_storage if svm_reward_enable else "disabled"),
            ("svm_buffer_dir", str(svm_buffer_dir) if svm_reward_enable else "disabled"),
            ("svm_buffer_reset", svm_buffer_reset if svm_reward_enable else "disabled"),
            ("svm_min_pos/neg", f"{svm_min_positive} / {svm_min_negative}"),
            ("checkpoint_metric", checkpoint_metric),
            ("reward_scale_running", reward_scale_running),
            ("reward_scale_const", reward_scale_const),
            ("progress_bars", show_progress),
            ("rollout_progress", rollout_progress),
            ("rollout_log_interval", rollout_log_interval),
            ("training_progress", training_progress),
            ("training_log_interval", training_log_interval),
            ("quiet_terminal", quiet_terminal),
        ],
    )
    log_box(
        "Model Params",
        [
            ("frozen_diffusion", f"{sum(p.numel() for p in base_policy.parameters()) / 1e6:.2f}M"),
            ("trainable_residual_mlp", f"{sum(p.numel() for p in policy.adapter_parameters()) / 1e6:.2f}M"),
            ("critic", f"{sum(p.numel() for p in critic.parameters()) / 1e6:.2f}M"),
            ("svm_discriminator", f"{sum(p.numel() for p in svm_discriminator.parameters()) / 1e6:.2f}M" if svm_discriminator is not None else "disabled"),
            ("svm_input_dim", svm_input_dim if svm_reward_enable else "disabled"),
            ("all_mlp", f"mode={policy.residual_mlp_mode}, cond={policy.residual_mlp_condition}, dim={policy.action_dim}, hidden={policy.all_hidden_dim}, depth={policy.all_depth}"),
            ("arm_mlp", f"dim={len(policy.arm_action_indices)}, hidden={policy.arm_hidden_dim}, depth={policy.arm_depth}"),
            ("cam_mlp", f"dim={len(policy.cam_action_indices)}, hidden={policy.cam_hidden_dim}, depth={policy.cam_depth}"),
            ("action_std", f"{float(policy.action_std.mean().detach().cpu().item()):.5f}"),
        ],
    )

    prev_obs, _ = env.reset()
    prev_obs = flatten_lerobot_obs(prev_obs)
    raw_obs_queue = {key: deque(maxlen=n_obs_steps) for key in prev_obs.keys()}
    reset_full_obs_queue(raw_obs_queue, prev_obs, n_obs_steps)

    running_ep_train_rewards = np.zeros(n_envs, dtype=np.float32)
    running_ep_env_rewards = np.zeros(n_envs, dtype=np.float32)
    running_ep_steps = np.zeros(n_envs, dtype=np.int32)
    svm_episode_features = [[] for _ in range(n_envs)]
    running_reward_scaler = RunningRewardScaler(n_envs, gamma=gamma)
    next_chunk_firsts = np.ones(n_envs, dtype=np.float32)
    best_policy_state = snapshot_trainable_state(policy)
    best_eval_score = float("-inf")
    best_eval_reward = float("-inf")
    best_eval_success_rate = 0.0
    eval_collapse_count = 0

    for itr in range(int(cfg.training.n_train_itr)):
        actor_update_planned = bool(update_actor and itr >= critic_warmup_iters)
        log_box(
            f"Iteration {itr + 1}/{cfg.training.n_train_itr}",
            [
                ("phase", "actor+critic" if actor_update_planned else "critic warmup"),
                ("rollout_chunks", f"{n_steps} x {n_envs} env"),
                ("update_epochs", update_epochs),
                ("eval_due", (itr + 1) > critic_warmup_iters and (itr + 1) % int(getattr(cfg.eval, "eval_freq", 5)) == 0),
            ],
        )

        obs_trajs = None
        if use_disk_cache:
            temp_buffer_dir = tempfile.TemporaryDirectory()
            buffer_path = temp_buffer_dir.name
            base_action_trajs = np.memmap(
                os.path.join(buffer_path, "base_action_trajs.npy"),
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, n_envs, horizon_steps, action_dim),
            )
            action_trajs = np.memmap(
                os.path.join(buffer_path, "action_trajs.npy"),
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, n_envs, act_steps, action_dim),
            )
            global_cond_step_trajs = (
                np.memmap(
                    os.path.join(buffer_path, "global_cond_step_trajs.npy"),
                    dtype=np.float32,
                    mode="w+",
                    shape=(n_steps, n_envs, act_steps, global_cond_dim),
                )
                if residual_mlp_stepwise_obs
                else None
            )
        else:
            temp_buffer_dir = None
            buffer_path = None
            base_action_trajs = np.zeros(
                (n_steps, n_envs, horizon_steps, action_dim),
                dtype=np.float32,
            )
            action_trajs = np.zeros((n_steps, n_envs, act_steps, action_dim), dtype=np.float32)
            global_cond_step_trajs = (
                np.zeros(
                    (n_steps, n_envs, act_steps, global_cond_dim),
                    dtype=np.float32,
                )
                if residual_mlp_stepwise_obs
                else None
            )

        if ppo_control_step_training:
            old_logprob_trajs = np.zeros((n_steps, n_envs, act_steps), dtype=np.float32)
            reward_trajs = np.zeros((n_steps, n_envs, act_steps), dtype=np.float32)
            terminated_trajs = np.zeros((n_steps, n_envs, act_steps), dtype=np.float32)
            firsts_trajs = np.zeros((n_steps, n_envs, act_steps), dtype=np.float32)
            valid_step_trajs = np.zeros((n_steps, n_envs, act_steps), dtype=bool)
            chunk_start_firsts = next_chunk_firsts.copy()
        else:
            old_logprob_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
            reward_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
            terminated_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
            firsts_trajs = np.zeros((n_steps + 1, n_envs), dtype=np.float32)
            firsts_trajs[0] = next_chunk_firsts
            valid_step_trajs = None
        completed_ep_train_rewards = []
        completed_ep_env_rewards = []
        completed_ep_successes = []
        completed_ep_steps = []
        svm_rollout_rewards = []

        policy.eval()
        logging.info("Rollout | collecting frozen Diffusion + stochastic MLP actions")
        rollout_bar = tqdm(
            range(n_steps),
            desc=f"Rollout {itr + 1:04d}",
            leave=rollout_progress,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not rollout_progress,
        )
        for step in rollout_bar:
            stacked_raw_obs = stack_obs_queue(raw_obs_queue, n_envs, n_obs_steps)
            if obs_trajs is None:
                obs_trajs = {}
                for key, value in stacked_raw_obs.items():
                    if use_disk_cache:
                        safe_key = key.replace(".", "_")
                        obs_trajs[key] = np.memmap(
                            os.path.join(buffer_path, f"obs_{safe_key}.npy"),
                            dtype=value.dtype,
                            mode="w+",
                            shape=(n_steps, *value.shape),
                        )
                    else:
                        obs_trajs[key] = np.zeros((n_steps, *value.shape), dtype=value.dtype)

            batch_obs = build_history_batch(stacked_raw_obs, policy, device)
            with torch.no_grad():
                with maybe_suppress_stdout(quiet_terminal):
                    if residual_mlp_stepwise_obs:
                        base_actions_t, initial_global_cond_t = policy.frozen_diffusion_actions(
                            batch_obs,
                            return_global_cond=True,
                        )
                        base_venv = base_actions_t.cpu().numpy()
                        output_venv = base_venv.copy()
                        old_logprob_steps = np.zeros((act_steps, n_envs), dtype=np.float32)
                        step_global_cond_venv = np.zeros(
                            (n_envs, act_steps, global_cond_dim),
                            dtype=np.float32,
                        )
                        global_cond_venv = (
                            initial_global_cond_t.detach().cpu().numpy()
                            if svm_reward_enable
                            else None
                        )
                    else:
                        samples = policy.forward_history(
                            cond=batch_obs,
                            deterministic=False,
                            return_global_cond=svm_reward_enable,
                        )
                        output_venv = samples["actions"].cpu().numpy()
                        base_venv = samples["base_actions"].cpu().numpy()
                        old_logprob_venv = samples["log_probs"].cpu().numpy()
                        global_cond_venv = (
                            samples["global_cond"].detach().cpu().numpy()
                            if svm_reward_enable
                            else None
                        )

            action_venv = (
                np.zeros((n_envs, act_steps, action_dim), dtype=np.float32)
                if residual_mlp_stepwise_obs
                else output_venv[:, action_start:action_end]
            )
            svm_features_venv = None
            svm_reward_venv = np.zeros(n_envs, dtype=np.float32)
            chunk_env_reward = np.zeros(n_envs, dtype=np.float32)
            any_done_accum = np.zeros(n_envs, dtype=bool)
            true_term_accum = np.zeros(n_envs, dtype=bool)
            success_accum = np.zeros(n_envs, dtype=bool)
            safe_actions = np.zeros((n_envs, action_dim), dtype=np.float32)
            if ppo_control_step_training:
                step_reward_venv = np.zeros((n_envs, act_steps), dtype=np.float32)
                step_done_venv = np.zeros((n_envs, act_steps), dtype=np.float32)
                step_first_venv = np.zeros((n_envs, act_steps), dtype=np.float32)
                step_valid_venv = np.zeros((n_envs, act_steps), dtype=bool)
                step_svm_reward_venv = np.zeros((n_envs, act_steps), dtype=np.float32)
                current_firsts_venv = chunk_start_firsts.copy()

            for step_i in range(act_steps):
                if residual_mlp_stepwise_obs:
                    with torch.no_grad():
                        if step_i == 0:
                            global_cond_step_t = initial_global_cond_t
                        else:
                            step_stacked_raw_obs = stack_obs_queue(
                                raw_obs_queue,
                                n_envs,
                                n_obs_steps,
                            )
                            step_batch_obs = build_history_batch(
                                step_stacked_raw_obs,
                                policy,
                                device,
                            )
                            with maybe_suppress_stdout(quiet_terminal):
                                global_cond_step_t = global_cond_from_obs(
                                    policy,
                                    step_batch_obs,
                                )
                        base_step_t = base_actions_t[
                            :,
                            action_start + step_i : action_start + step_i + 1,
                        ]
                        mean_step_t = policy.mean_actions(
                            base_step_t,
                            global_cond=global_cond_step_t,
                        )
                        std = policy.action_std.view(1, 1, -1)
                        action_step_t = mean_step_t + torch.randn_like(mean_step_t) * std
                        logprob_step_t = policy.log_prob_from_mean(
                            mean_step_t,
                            action_step_t,
                            action_start=0,
                            action_end=1,
                        )
                        curr_action = action_step_t[:, 0, :].cpu().numpy()
                        action_venv[:, step_i, :] = curr_action
                        old_logprob_steps[step_i] = logprob_step_t.cpu().numpy()
                        step_global_cond_venv[:, step_i, :] = (
                            global_cond_step_t.detach().cpu().numpy()
                        )
                else:
                    curr_action = action_venv[:, step_i, :].copy()
                active_mask = ~any_done_accum
                if ppo_control_step_training and svm_reward_enable:
                    svm_step_features_venv = build_svm_features(
                        global_cond=step_global_cond_venv[:, step_i, :],
                        base_actions=base_venv,
                        action_chunk=action_venv[:, step_i : step_i + 1, :],
                        action_start=action_start + step_i,
                        action_end=action_start + step_i + 1,
                        feature_mode=svm_feature_mode,
                    )
                    if svm_reward_ready and svm_discriminator is not None:
                        svm_discriminator.eval()
                        step_svm_reward_venv[:, step_i] = compute_svm_process_reward(
                            discriminator=svm_discriminator,
                            features=svm_step_features_venv,
                            device=device,
                            coef=svm_reward_coef,
                            reward_clip=svm_reward_clip,
                        )
                    if active_mask.any():
                        svm_rollout_rewards.extend(
                            step_svm_reward_venv[active_mask, step_i].tolist()
                        )
                        for env_idx in np.flatnonzero(active_mask):
                            svm_episode_features[env_idx].append(
                                np.array(svm_step_features_venv[env_idx], copy=True)
                            )
                for env_idx in range(n_envs):
                    if any_done_accum[env_idx]:
                        curr_action[env_idx] = safe_actions[env_idx]

                action_to_step = curr_action[0] if n_envs == 1 else curr_action
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

                chunk_env_reward += reward_venv * active_mask
                running_ep_steps[active_mask] += 1
                just_done = (terminated_venv | truncated_venv) & active_mask
                just_success = info_success_mask(info_venv, just_done, n_envs)
                success_accum = success_accum | just_success
                true_term_accum = true_term_accum | (terminated_venv & active_mask)
                if ppo_control_step_training:
                    if reward_source == "success_label":
                        step_train_reward = success_label_chunk_rewards(
                            done_mask=just_done,
                            success_mask=just_success,
                            success_reward=label_success_reward,
                            failure_reward=label_failure_reward,
                            nonterminal_reward=label_nonterminal_reward,
                            success_time_bonus=success_time_bonus,
                            episode_steps=running_ep_steps,
                            max_episode_steps=success_time_max_steps,
                        )
                    else:
                        step_train_reward = reward_venv.copy()
                    if svm_reward_enable:
                        step_train_reward = (
                            step_train_reward + step_svm_reward_venv[:, step_i]
                        )
                    step_train_reward = step_train_reward * active_mask
                    step_reward_venv[:, step_i] = step_train_reward
                    step_done_venv[:, step_i] = (
                        (terminated_venv & active_mask)
                        if bootstrap_on_truncation
                        else just_done
                    ).astype(np.float32)
                    step_first_venv[:, step_i] = current_firsts_venv
                    step_valid_venv[:, step_i] = active_mask
                    current_firsts_venv = just_done.astype(np.float32)

                if n_envs == 1 and just_done[0]:
                    reset_obs, _ = env.reset()
                    obs_venv = flatten_lerobot_obs(reset_obs)

                for env_idx in range(n_envs):
                    if just_done[env_idx]:
                        if "observation.state" in obs_venv:
                            state_data = obs_venv["observation.state"]
                            safe_actions[env_idx] = (
                                state_data[:action_dim]
                                if n_envs == 1
                                else state_data[env_idx][:action_dim]
                            )
                        else:
                            safe_actions[env_idx] = np.zeros(action_dim, dtype=np.float32)

                append_obs_queue(raw_obs_queue, obs_venv, n_obs_steps)
                reset_done_envs_in_obs_queue(raw_obs_queue, obs_venv, just_done, n_envs, n_obs_steps)
                any_done_accum = any_done_accum | terminated_venv | truncated_venv

            if residual_mlp_stepwise_obs:
                if ppo_control_step_training:
                    invalid_step_mask = ~step_valid_venv
                    step_first_venv[invalid_step_mask] = 1.0
                    step_done_venv[invalid_step_mask] = 1.0
                    old_logprob_venv = old_logprob_steps.T
                elif policy.logprob_reduction == "sum":
                    old_logprob_venv = old_logprob_steps.sum(axis=0)
                else:
                    old_logprob_venv = old_logprob_steps.mean(axis=0)
                output_venv[:, action_start:action_end] = action_venv

            if svm_reward_enable and not ppo_control_step_training:
                svm_features_venv = build_svm_features(
                    global_cond=global_cond_venv,
                    base_actions=base_venv,
                    action_chunk=action_venv,
                    action_start=action_start,
                    action_end=action_end,
                    feature_mode=svm_feature_mode,
                )
                if svm_reward_ready and svm_discriminator is not None:
                    svm_discriminator.eval()
                    svm_reward_venv = compute_svm_process_reward(
                        discriminator=svm_discriminator,
                        features=svm_features_venv,
                        device=device,
                        coef=svm_reward_coef,
                        reward_clip=svm_reward_clip,
                    )
                svm_rollout_rewards.extend(svm_reward_venv.tolist())

            if ppo_control_step_training:
                chunk_reward = step_reward_venv.sum(axis=1)
            elif reward_source == "success_label":
                chunk_reward = success_label_chunk_rewards(
                    done_mask=any_done_accum,
                    success_mask=success_accum,
                    success_reward=label_success_reward,
                    failure_reward=label_failure_reward,
                    nonterminal_reward=label_nonterminal_reward,
                    success_time_bonus=success_time_bonus,
                    episode_steps=running_ep_steps,
                    max_episode_steps=success_time_max_steps,
                )
            else:
                chunk_reward = chunk_env_reward

            if svm_reward_enable and not ppo_control_step_training:
                chunk_reward = chunk_reward + svm_reward_venv

            prev_obs = obs_venv
            running_ep_train_rewards += chunk_reward
            running_ep_env_rewards += chunk_env_reward
            if svm_reward_enable and svm_features_venv is not None:
                for env_idx in range(n_envs):
                    svm_episode_features[env_idx].append(
                        np.array(svm_features_venv[env_idx], copy=True)
                    )
            for env_idx in range(n_envs):
                if any_done_accum[env_idx]:
                    if svm_replay_buffer is not None:
                        svm_replay_buffer.add_episode(
                            svm_episode_features[env_idx],
                            success=bool(success_accum[env_idx]),
                        )
                    svm_episode_features[env_idx].clear()
                    completed_ep_train_rewards.append(float(running_ep_train_rewards[env_idx]))
                    completed_ep_env_rewards.append(float(running_ep_env_rewards[env_idx]))
                    completed_ep_successes.append(bool(success_accum[env_idx]))
                    completed_ep_steps.append(int(running_ep_steps[env_idx]))
                    running_ep_train_rewards[env_idx] = 0.0
                    running_ep_env_rewards[env_idx] = 0.0
                    running_ep_steps[env_idx] = 0

            for key in obs_trajs:
                obs_trajs[key][step] = stacked_raw_obs[key]
            base_action_trajs[step] = base_venv
            action_trajs[step] = action_venv
            if residual_mlp_stepwise_obs and global_cond_step_trajs is not None:
                global_cond_step_trajs[step] = step_global_cond_venv
            if ppo_control_step_training:
                old_logprob_trajs[step] = old_logprob_venv
                reward_trajs[step] = step_reward_venv
                terminated_trajs[step] = step_done_venv
                firsts_trajs[step] = step_first_venv
                valid_step_trajs[step] = step_valid_venv
                chunk_start_firsts = any_done_accum.astype(np.float32)
            else:
                old_logprob_trajs[step] = old_logprob_venv
                reward_trajs[step] = chunk_reward
                terminated_trajs[step] = (
                    true_term_accum if bootstrap_on_truncation else any_done_accum
                )
                firsts_trajs[step + 1] = any_done_accum

            if rollout_progress and ((step + 1) % max(1, n_steps // 100) == 0 or step + 1 == n_steps):
                rollout_bar.set_postfix(
                    episodes=len(completed_ep_train_rewards),
                    success=fmt_pct(np.mean(completed_ep_successes))
                    if completed_ep_successes
                    else "n/a",
                    svm=fmt_float(float(np.mean(svm_rollout_rewards)), 3)
                    if svm_reward_enable and svm_rollout_rewards
                    else "n/a",
                )
            if rollout_log_interval > 0 and (
                (step + 1) % rollout_log_interval == 0 or step + 1 == n_steps
            ):
                logging.info(
                    "Rollout progress | "
                    f"chunk={step + 1}/{n_steps} | "
                    f"env_steps={(step + 1) * act_steps * n_envs}/{n_steps * act_steps * n_envs} | "
                    f"episodes={len(completed_ep_train_rewards)} | "
                    f"success={fmt_pct(np.mean(completed_ep_successes)) if completed_ep_successes else 'n/a'}"
                )

        next_chunk_firsts = (
            chunk_start_firsts.copy()
            if ppo_control_step_training
            else firsts_trajs[-1].copy()
        )
        rollout_avg_return = (
            np.mean(completed_ep_train_rewards) if completed_ep_train_rewards else float("-inf")
        )
        rollout_avg_env_return = (
            np.mean(completed_ep_env_rewards) if completed_ep_env_rewards else float("-inf")
        )
        rollout_avg_ep_steps = np.mean(completed_ep_steps) if completed_ep_steps else float("nan")
        rollout_success_rate = np.mean(completed_ep_successes) if completed_ep_successes else 0.0
        rollout_svm_reward_mean = (
            float(np.mean(svm_rollout_rewards)) if svm_rollout_rewards else 0.0
        )
        rollout_svm_reward_std = (
            float(np.std(svm_rollout_rewards)) if svm_rollout_rewards else 0.0
        )
        rollout_message = (
            "Rollout | "
            f"episodes={len(completed_ep_train_rewards):3d} | "
            f"success={fmt_pct(rollout_success_rate):>6} | "
            f"train_return={rollout_avg_return:8.2f} | "
            f"env_return={rollout_avg_env_return:8.2f} | "
            f"avg_steps={rollout_avg_ep_steps:6.1f}"
        )
        if svm_reward_enable:
            rollout_message += (
                f" | svm_reward={rollout_svm_reward_mean:7.4f}"
                f"±{rollout_svm_reward_std:.4f}"
            )
        logging.info(rollout_message)
        if reward_source == "success_label" and not completed_ep_successes:
            logging.warning(
                "Rollout produced no completed episodes, so success_label reward has no "
                "terminal labels in this batch. Increase training.rollout_steps or shorten episodes."
            )
            

        if skip_update:
            logging.info("training.skip_update=true; skipping MLP/Critic update.")
            try:
                del base_action_trajs, action_trajs, obs_trajs
                if global_cond_step_trajs is not None:
                    del global_cond_step_trajs
                if valid_step_trajs is not None:
                    del valid_step_trajs
                if use_disk_cache and temp_buffer_dir is not None:
                    temp_buffer_dir.cleanup()
            except Exception:
                pass
            torch.cuda.empty_cache()
            continue

        if svm_reward_enable and svm_replay_buffer is not None and svm_discriminator is not None:
            if svm_replay_buffer.can_train(svm_min_positive, svm_min_negative):
                svm_last_stats = train_svm_discriminator(
                    discriminator=svm_discriminator,
                    replay_buffer=svm_replay_buffer,
                    optimizer=svm_optimizer,
                    device=device,
                    batch_size=svm_disc_batch_size,
                    updates=svm_disc_updates,
                )
                svm_reward_ready = True
                logging.info(
                    "SVM     | "
                    f"updates={svm_last_stats['updates']} | "
                    f"loss={fmt_float(svm_last_stats['loss'])} | "
                    f"acc={fmt_pct(svm_last_stats['acc']) if np.isfinite(svm_last_stats['acc']) else 'nan'} | "
                    f"pos={svm_replay_buffer.num_positive} | "
                    f"neg={svm_replay_buffer.num_negative} | "
                    f"ready={svm_reward_ready}"
                )
            else:
                logging.info(
                    "SVM     | waiting for balanced labels | "
                    f"pos={svm_replay_buffer.num_positive}/{svm_min_positive} | "
                    f"neg={svm_replay_buffer.num_negative}/{svm_min_negative} | "
                    f"ready={svm_reward_ready}"
                )

        logging.info("Update  | computing values and GAE")
        if ppo_control_step_training:
            total_raw_samples = n_steps * n_envs * act_steps
            obs_flat = None
            base_actions_flat = base_action_trajs[
                :,
                :,
                action_start:action_end,
            ].reshape(total_raw_samples, 1, action_dim)
            actions_flat = action_trajs.reshape(total_raw_samples, 1, action_dim)
            global_cond_steps_flat = global_cond_step_trajs.reshape(
                total_raw_samples,
                global_cond_dim,
            )
            old_logprobs_flat = old_logprob_trajs.reshape(total_raw_samples)
            valid_flat = valid_step_trajs.reshape(total_raw_samples)
            valid_indices_np = np.flatnonzero(valid_flat)
            if valid_indices_np.size == 0:
                raise RuntimeError("No valid control-step samples were collected for PPO update.")

            with torch.no_grad():
                values_all_flat = np.zeros(total_raw_samples, dtype=np.float32)
                val_batch_size = batch_size * 4
                for i in range(0, total_raw_samples, val_batch_size):
                    end_i = min(i + val_batch_size, total_raw_samples)
                    global_cond = torch.from_numpy(
                        np.ascontiguousarray(global_cond_steps_flat[i:end_i])
                    ).float().to(device)
                    values_all_flat[i:end_i] = (
                        critic(global_cond.detach()).cpu().numpy().flatten()
                    )

                values_trajs = values_all_flat.reshape(n_steps, n_envs, act_steps)
                values_time = values_trajs.transpose(0, 2, 1).reshape(
                    n_steps * act_steps,
                    n_envs,
                )
                last_stacked_raw_obs = stack_obs_queue(raw_obs_queue, n_envs, n_obs_steps)
                last_obs = build_history_batch(last_stacked_raw_obs, policy, device)
                global_cond_last = global_cond_from_obs(policy, last_obs)
                next_values_last = critic(global_cond_last.detach()).cpu().numpy().flatten()

            rewards_time = reward_trajs.transpose(0, 2, 1).reshape(
                n_steps * act_steps,
                n_envs,
            )
            terminated_time = terminated_trajs.transpose(0, 2, 1).reshape(
                n_steps * act_steps,
                n_envs,
            )
            firsts_time = firsts_trajs.transpose(0, 2, 1).reshape(
                n_steps * act_steps,
                n_envs,
            )
            if reward_scale_running:
                scaled_rewards_time = running_reward_scaler(
                    reward=rewards_time.T,
                    first=firsts_time.T,
                ).T
            else:
                scaled_rewards_time = rewards_time

            advantages_time = np.zeros_like(scaled_rewards_time)
            last_gae_lam = np.zeros(n_envs, dtype=np.float32)
            for t in reversed(range(n_steps * act_steps)):
                next_val = next_values_last if t == n_steps * act_steps - 1 else values_time[t + 1]
                nonterminal = 1.0 - terminated_time[t]
                delta = (
                    scaled_rewards_time[t] * reward_scale_const
                    + gamma * next_val * nonterminal
                    - values_time[t]
                )
                last_gae_lam = delta + gamma * gae_lambda * nonterminal * last_gae_lam
                advantages_time[t] = last_gae_lam

            returns_time = advantages_time + values_time
            returns_trajs = returns_time.reshape(n_steps, act_steps, n_envs).transpose(0, 2, 1)
            advantages_trajs = advantages_time.reshape(n_steps, act_steps, n_envs).transpose(0, 2, 1)
            returns_flat = returns_trajs.reshape(total_raw_samples)
            advantages_flat = advantages_trajs.reshape(total_raw_samples)
            critic_ev, critic_corr = compute_value_diagnostics(
                values_all_flat[valid_indices_np],
                returns_flat[valid_indices_np],
            )

            base_actions_flat = base_actions_flat[valid_indices_np]
            actions_flat = actions_flat[valid_indices_np]
            global_cond_steps_flat = global_cond_steps_flat[valid_indices_np]
            old_logprobs_flat = old_logprobs_flat[valid_indices_np]
            returns_k = torch.from_numpy(returns_flat[valid_indices_np]).float().to(device)
            advantages_k = torch.from_numpy(advantages_flat[valid_indices_np]).float().to(device)
            old_logprobs_k = torch.from_numpy(old_logprobs_flat).float().to(device)
            total_samples = int(valid_indices_np.size)
        else:
            total_samples = n_steps * n_envs
            obs_flat = {
                key: value.reshape(total_samples, *value.shape[2:])
                for key, value in obs_trajs.items()
            }
            base_actions_flat = base_action_trajs.reshape(
                total_samples,
                horizon_steps,
                action_dim,
            )
            actions_flat = action_trajs.reshape(total_samples, act_steps, action_dim)
            global_cond_steps_flat = None
            old_logprobs_flat = old_logprob_trajs.reshape(total_samples)

            with torch.no_grad():
                values_flat = np.zeros(total_samples, dtype=np.float32)
                val_batch_size = batch_size * 2
                for i in range(0, total_samples, val_batch_size):
                    end_i = min(i + val_batch_size, total_samples)
                    obs_chunk = {}
                    for key, value in obs_flat.items():
                        tensor_value = torch.from_numpy(value[i:end_i]).float().to(device)
                        if "images" in key:
                            tensor_value = tensor_value.permute(0, 1, 4, 2, 3) / 255.0
                        obs_chunk[key] = tensor_value
                    global_cond = global_cond_from_obs(policy, obs_chunk)
                    values_flat[i:end_i] = critic(global_cond.detach()).cpu().numpy().flatten()

                values_trajs = values_flat.reshape(n_steps, n_envs)
                last_stacked_raw_obs = stack_obs_queue(raw_obs_queue, n_envs, n_obs_steps)
                last_obs = build_history_batch(last_stacked_raw_obs, policy, device)
                global_cond_last = global_cond_from_obs(policy, last_obs)
                next_values_last = critic(global_cond_last.detach()).cpu().numpy().flatten()

            if reward_scale_running:
                scaled_rewards = running_reward_scaler(
                    reward=reward_trajs.T,
                    first=firsts_trajs[:-1].T,
                ).T
            else:
                scaled_rewards = reward_trajs

            advantages_trajs = np.zeros_like(scaled_rewards)
            last_gae_lam = 0
            for t in reversed(range(n_steps)):
                next_val = next_values_last if t == n_steps - 1 else values_trajs[t + 1]
                nonterminal = 1.0 - terminated_trajs[t]
                delta = (
                    scaled_rewards[t] * reward_scale_const
                    + gamma * next_val * nonterminal
                    - values_trajs[t]
                )
                last_gae_lam = delta + gamma * gae_lambda * nonterminal * last_gae_lam
                advantages_trajs[t] = last_gae_lam

            returns_trajs = advantages_trajs + values_trajs
            critic_ev, critic_corr = compute_value_diagnostics(values_trajs, returns_trajs)
            returns_k = torch.from_numpy(returns_trajs.reshape(-1)).float().to(device)
            advantages_k = torch.from_numpy(advantages_trajs.reshape(-1)).float().to(device)
            old_logprobs_k = torch.from_numpy(old_logprobs_flat).float().to(device)

        def normalize_clip_minibatch_advantage(advantages):
            """按原版 DPPO 口径在 minibatch 内归一化并裁剪 advantage。"""
            if norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            if adv_lower_q > 0.0 or adv_upper_q < 1.0:
                adv_lower = torch.quantile(advantages, adv_lower_q)
                adv_upper = torch.quantile(advantages, adv_upper_q)
                advantages = torch.clamp(advantages, min=adv_lower, max=adv_upper)
            return advantages

        has_reward_signal = reward_source != "success_label" or len(completed_ep_successes) > 0
        actor_update_enabled = bool(
            update_actor and itr >= critic_warmup_iters and has_reward_signal
        )
        if reward_source == "success_label" and itr >= critic_warmup_iters and not has_reward_signal:
            logging.warning("Update  | actor update disabled because this batch has no success/fail labels.")
        policy.train()
        policy.apply(freeze_batch_norm)
        critic.train()
        critic.apply(freeze_batch_norm)
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)

        running_v_loss = []
        running_pg_loss = []
        running_kl = []
        running_bc_loss = []
        early_stop = False
        num_batches = max(1, math.ceil(total_samples / batch_size))
        total_update_batches = update_epochs * num_batches
        update_batches_done = 0
        logging.info(
            "Update  | "
            f"actor_update={actor_update_enabled} | "
            f"epochs={update_epochs} | "
            f"minibatches/epoch={num_batches} | "
            f"total_minibatches={total_update_batches}"
        )

        update_epoch_bar = tqdm(
            range(update_epochs),
            desc=f"PPO {itr + 1:04d}",
            leave=training_progress,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not training_progress,
        )
        for epoch in update_epoch_bar:
            if early_stop:
                break
            indices = torch.randperm(total_samples, device=device)
            minibatch_bar = tqdm(
                range(num_batches),
                desc=f"epoch {epoch + 1}",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
                disable=not training_progress,
            )
            for batch_idx in minibatch_bar:
                start = batch_idx * batch_size
                end = min(start + batch_size, total_samples)
                inds_b = indices[start:end]
                inds_np = inds_b.cpu().numpy()

                base_actions_b = torch.from_numpy(base_actions_flat[inds_np]).float().to(device)
                actions_b = torch.from_numpy(actions_flat[inds_np]).float().to(device)
                returns_b = returns_k[inds_b]
                advantages_b = advantages_k[inds_b]
                advantages_b = normalize_clip_minibatch_advantage(advantages_b)
                old_logprobs_b = old_logprobs_k[inds_b]

                if ppo_control_step_training:
                    global_cond_b = torch.from_numpy(
                        np.ascontiguousarray(global_cond_steps_flat[inds_np])
                    ).float().to(device)
                else:
                    obs_b = {}
                    for key, value in obs_flat.items():
                        tensor_value = torch.from_numpy(value[inds_np]).float().to(device)
                        if "images" in key:
                            tensor_value = tensor_value.permute(0, 1, 4, 2, 3) / 255.0
                        obs_b[key] = tensor_value
                    with torch.no_grad():
                        global_cond_b = global_cond_from_obs(policy, obs_b)

                if ppo_control_step_training:
                    actor_base_actions_b = base_actions_b
                    actor_global_cond_b = global_cond_b
                    actor_action_start = 0
                    actor_action_end = 1
                else:
                    actor_base_actions_b = base_actions_b
                    actor_global_cond_b = global_cond_b
                    actor_action_start = action_start
                    actor_action_end = action_end

                if actor_update_enabled:
                    mean_actions_b = policy.mean_actions(
                        actor_base_actions_b,
                        global_cond=actor_global_cond_b,
                    )
                    new_logprobs_b = policy.log_prob_from_mean(
                        mean_actions_b,
                        actions_b,
                        action_start=actor_action_start,
                        action_end=actor_action_end,
                    )
                    raw_log_ratio = new_logprobs_b - old_logprobs_b
                    log_ratio = torch.clamp(raw_log_ratio, min=-20.0, max=5.0)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * advantages_b
                    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_b
                    pg_loss = -torch.min(surr1, surr2).mean()
                    bc_target_b = (
                        actor_base_actions_b
                        if ppo_control_step_training
                        else actor_base_actions_b[:, action_start:action_end]
                    )
                    bc_pred_b = (
                        mean_actions_b
                        if ppo_control_step_training
                        else mean_actions_b[:, action_start:action_end]
                    )
                    bc_loss = F.mse_loss(bc_pred_b, bc_target_b)
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                else:
                    with torch.no_grad():
                        mean_actions_b = policy.mean_actions(
                            actor_base_actions_b,
                            global_cond=actor_global_cond_b,
                        )
                        new_logprobs_b = policy.log_prob_from_mean(
                            mean_actions_b,
                            actions_b,
                            action_start=actor_action_start,
                            action_end=actor_action_end,
                        )
                        raw_log_ratio = new_logprobs_b - old_logprobs_b
                        log_ratio = torch.clamp(raw_log_ratio, min=-20.0, max=5.0)
                        approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    pg_loss = torch.zeros((), device=device)
                    bc_loss = torch.zeros((), device=device)

                values_pred = critic(global_cond_b.detach()).squeeze(-1)
                v_loss = F.smooth_l1_loss(values_pred, returns_b)
                loss = 0.5 * v_loss
                if actor_update_enabled:
                    loss = loss + pg_loss + bc_coef * bc_loss

                running_v_loss.append(float(v_loss.item()))
                running_pg_loss.append(float(pg_loss.item()))
                running_bc_loss.append(float(bc_loss.item()))
                running_kl.append(float(approx_kl))
                update_batches_done += 1
                if training_progress:
                    minibatch_bar.set_postfix(
                        v=fmt_float(float(v_loss.item()), 3),
                        pg=fmt_float(float(pg_loss.item()), 3),
                        kl=f"{approx_kl:.2e}",
                    )
                if training_log_interval > 0 and (
                    update_batches_done % training_log_interval == 0
                    or update_batches_done == total_update_batches
                ):
                    logging.info(
                        "Update progress | "
                        f"minibatch={update_batches_done}/{total_update_batches} | "
                        f"epoch={epoch + 1}/{update_epochs} | "
                        f"batch={batch_idx + 1}/{num_batches} | "
                        f"value_loss={v_loss.item():.4f} | "
                        f"policy_loss={pg_loss.item():.4f} | "
                        f"kl={approx_kl:.3e}"
                    )

                if actor_update_enabled and approx_kl > target_kl:
                    logging.warning(
                        f"Early stop | KL {approx_kl:.4f} > target {target_kl:.4f} "
                        f"(epoch={epoch + 1}, batch={batch_idx + 1}/{num_batches})"
                    )
                    early_stop = True
                    actor_optimizer.zero_grad(set_to_none=True)
                    critic_optimizer.zero_grad(set_to_none=True)
                    break

                (loss / grad_accum_steps).backward()
                should_step = (
                    (batch_idx + 1) % grad_accum_steps == 0
                    or batch_idx + 1 == num_batches
                )
                if should_step:
                    if actor_update_enabled:
                        torch.nn.utils.clip_grad_norm_(policy.adapter_parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_optimizer.step()
                    if actor_update_enabled:
                        actor_optimizer.step()
                    actor_optimizer.zero_grad(set_to_none=True)
                    critic_optimizer.zero_grad(set_to_none=True)

        if actor_update_enabled:
            actor_scheduler.step()

        avg_v_loss = float(np.mean(running_v_loss)) if running_v_loss else 0.0
        avg_pg_loss = float(np.mean(running_pg_loss)) if running_pg_loss else 0.0
        avg_kl = float(np.mean(running_kl)) if running_kl else 0.0
        max_kl = float(np.max(running_kl)) if running_kl else 0.0
        avg_bc_loss = float(np.mean(running_bc_loss)) if running_bc_loss else 0.0

        summary_rows = [
            ("episodes", len(completed_ep_train_rewards)),
            ("rollout_success", fmt_pct(rollout_success_rate)),
            ("rollout_train_return", f"{rollout_avg_return:.2f}"),
            ("rollout_env_return", f"{rollout_avg_env_return:.2f}"),
            ("rollout_avg_steps", f"{rollout_avg_ep_steps:.1f}"),
            ("ppo_granularity", "control_step" if ppo_control_step_training else "chunk"),
            ("ppo_samples", total_samples),
            ("actor_update", actor_update_enabled),
            ("value_loss", fmt_float(avg_v_loss)),
            ("policy_loss", fmt_float(avg_pg_loss)),
            ("bc_loss", f"{avg_bc_loss:.5f}"),
            ("kl_avg/max", f"{avg_kl:.3e} / {max_kl:.3e}"),
            ("critic_ev", fmt_float(critic_ev)),
            ("value_return_corr", fmt_float(critic_corr)),
            ("action_std", f"{float(policy.action_std.mean().detach().cpu().item()):.5f}"),
        ]
        if svm_reward_enable and svm_replay_buffer is not None:
            summary_rows.extend(
                [
                    ("svm_ready", svm_reward_ready),
                    ("svm_reward_mean/std", f"{rollout_svm_reward_mean:.4f} / {rollout_svm_reward_std:.4f}"),
                    ("svm_loss", fmt_float(svm_last_stats["loss"])),
                    ("svm_acc", fmt_pct(svm_last_stats["acc"]) if np.isfinite(svm_last_stats["acc"]) else "nan"),
                    ("svm_pos/neg", f"{svm_replay_buffer.num_positive} / {svm_replay_buffer.num_negative}"),
                ]
            )
        log_box(f"Iteration {itr + 1} Summary", summary_rows)

        try:
            del (
                base_actions_flat,
                actions_flat,
                obs_flat,
                base_action_trajs,
                action_trajs,
                global_cond_step_trajs,
                global_cond_steps_flat,
                valid_step_trajs,
                obs_trajs,
                returns_k,
                advantages_k,
                old_logprobs_k,
            )
            import gc

            gc.collect()
            if use_disk_cache and temp_buffer_dir is not None:
                temp_buffer_dir.cleanup()
        except Exception:
            pass
        torch.cuda.empty_cache()

        eval_freq = int(getattr(cfg.eval, "eval_freq", 5))
        is_last_step = (itr + 1) == int(cfg.training.n_train_itr)
        if (itr + 1) > critic_warmup_iters and ((itr + 1) % eval_freq == 0 or is_last_step):
            logging.info(f"Eval    | running iteration {itr + 1}")
            tmp_videos_dir = Path(out_dir) / "eval" / f"videos_{itr + 1:06d}"
            eval_cfg_node = getattr(cfg, "eval", OmegaConf.create())
            with torch.no_grad():
                with (
                    torch.autocast(device_type=device.type)
                    if bool(getattr(cfg, "use_amp", False))
                    else nullcontext()
                ):
                    with maybe_quiet_eval_progress(quiet_terminal):
                        with maybe_suppress_stdout(quiet_terminal):
                            eval_info = custom_eval_policy(
                                env=eval_env,
                                policy=policy,
                                cfg_eval=eval_cfg_node,
                                videos_dir=tmp_videos_dir,
                                device=device,
                            )

            sr = eval_info["aggregated"]["success_rate"]
            ar = eval_info["aggregated"]["average_reward"]
            logging.info(
                "Eval    | "
                f"success={fmt_pct(sr):>6} | "
                f"avg_reward={ar:8.2f}"
            )
            eval_score = sr if checkpoint_metric in success_metric_names else ar

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_reward = ar
                best_eval_success_rate = sr
                best_policy_state = snapshot_trainable_state(policy)
                eval_collapse_count = 0
            else:
                rollback_enabled = bool(getattr(cfg.training, "rollback_on_eval_collapse", True))
                rollback_sr = float(getattr(cfg.training, "rollback_success_rate", 0.1))
                rollback_reward = float(getattr(cfg.training, "rollback_reward", -100.0))
                rollback_success_drop = float(getattr(cfg.training, "rollback_success_drop", 0.2))
                rollback_min_best_success = float(getattr(cfg.training, "rollback_min_best_success", 0.5))
                rollback_reward_gate = bool(
                    getattr(
                        cfg.training,
                        "rollback_reward_gate",
                        checkpoint_metric not in success_metric_names,
                    )
                )
                rollback_patience = int(getattr(cfg.training, "rollback_patience", 1))
                success_floor = rollback_sr
                if best_eval_success_rate >= rollback_min_best_success and rollback_success_drop > 0.0:
                    success_floor = max(success_floor, best_eval_success_rate - rollback_success_drop)
                success_collapsed = sr < success_floor
                reward_collapsed = ar <= rollback_reward
                if checkpoint_metric in success_metric_names:
                    eval_collapsed = success_collapsed and (
                        reward_collapsed if rollback_reward_gate else True
                    )
                else:
                    eval_collapsed = reward_collapsed
                if rollback_enabled and eval_collapsed:
                    eval_collapse_count += 1
                    logging.warning(
                        f"Eval collapse detected ({eval_collapse_count}/{rollback_patience}). "
                        f"success={sr * 100:.1f}% reward={ar:.2f} "
                        f"success_floor={success_floor * 100:.1f}% "
                        f"reward_gate={rollback_reward_gate}"
                    )
                    if eval_collapse_count >= rollback_patience:
                        restore_trainable_state(policy, best_policy_state, device)
                        actor_optimizer.state.clear()
                        eval_collapse_count = 0
                        logging.warning(
                            f"Rolled back to best adapter: success={best_eval_success_rate * 100:.1f}% "
                            f"reward={best_eval_reward:.2f}"
                        )
                        if tmp_videos_dir.exists():
                            import shutil

                            shutil.rmtree(tmp_videos_dir, ignore_errors=True)
                        continue
                else:
                    eval_collapse_count = 0

            ckpt_name = (
                f"{itr + 1:06d}_sr={sr:.2f}_reward={ar:.2f}"
                f"_MLPloss={avg_pg_loss:.4f}_Vloss={avg_v_loss:.4f}"
            )
            ckpt_path = Path(out_dir) / "checkpoints" / ckpt_name
            save_path = ckpt_path / "pretrained_model"
            final_videos_dir = ckpt_path / "eval" / "eval_videos"
            if tmp_videos_dir.exists() and tmp_videos_dir != final_videos_dir:
                import shutil

                final_videos_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_videos_dir), str(final_videos_dir))

            policy.save_pretrained(save_path)
            if svm_reward_enable and svm_discriminator is not None:
                torch.save(
                    {
                        "model": svm_discriminator.state_dict(),
                        "input_dim": int(svm_input_dim),
                        "feature_mode": str(svm_feature_mode),
                        "reward_coef": float(svm_reward_coef),
                        "reward_clip": float(svm_reward_clip),
                        "hidden_dim": int(svm_disc_hidden_dim),
                        "depth": int(svm_disc_depth),
                        "ready": bool(svm_reward_ready),
                        "last_stats": dict(svm_last_stats),
                        "buffer_storage": str(svm_buffer_storage),
                        "buffer_dir": str(svm_buffer_dir),
                        "buffer_reset": bool(svm_buffer_reset),
                        "num_positive": int(svm_replay_buffer.num_positive)
                        if svm_replay_buffer is not None
                        else 0,
                        "num_negative": int(svm_replay_buffer.num_negative)
                        if svm_replay_buffer is not None
                        else 0,
                    },
                    save_path / "svm_reward.pt",
                )

            current_ft_dict = OmegaConf.to_container(cfg, resolve=True)
            base_config_dict = OmegaConf.to_container(hydra_cfg, resolve=True) if hydra_cfg is not None else {}
            final_config_dict = deep_update_dict(base_config_dict, current_ft_dict)
            if "training" in current_ft_dict:
                final_config_dict["training"] = current_ft_dict["training"]

            saved_policy_json = {}
            policy_json_path = save_path / "config.json"
            if policy_json_path.exists():
                with open(policy_json_path, "r", encoding="utf-8") as f:
                    saved_policy_json = json.load(f)

            final_policy = deep_update_dict(base_config_dict.get("policy", {}), saved_policy_json)
            final_policy = deep_update_dict(final_policy, current_ft_dict.get("policy", {}))
            final_policy = deep_update_dict(
                final_policy,
                {
                    "wrapper": "FrozenDiffusionMLPPolicy",
                    "action_start": int(action_start),
                    "action_end": int(action_end),
                    "residual_mlp_checkpoint": "residual_mlp.pt",
                    "residual_mlp_mode": str(policy.residual_mlp_mode),
                    "residual_mlp_condition": str(policy.residual_mlp_condition),
                    "residual_mlp_stepwise_obs": bool(policy.residual_mlp_stepwise_obs),
                    "global_cond_dim": int(policy.residual_global_cond_dim),
                    "residual_mlp_hidden_dim": int(residual_hidden_dim),
                    "residual_mlp_depth": int(residual_depth),
                    "residual_mlp_learn_std": bool(residual_learn_std),
                    "arm_action_indices": list(policy.arm_action_indices),
                    "cam_action_indices": list(policy.cam_action_indices),
                    "lambda_all": float(policy.lambda_all),
                    "lambda_arm": float(policy.lambda_arm),
                    "lambda_cam": float(policy.lambda_cam),
                    "residual_mlp_max_delta": float(policy.residual_mlp_max_delta),
                    "all_hidden_dim": int(policy.all_hidden_dim),
                    "all_depth": int(policy.all_depth),
                    "arm_hidden_dim": int(policy.arm_hidden_dim),
                    "cam_hidden_dim": int(policy.cam_hidden_dim),
                    "arm_depth": int(policy.arm_depth),
                    "cam_depth": int(policy.cam_depth),
                },
            )
            final_config_dict["policy"] = final_policy
            final_config_dict["checkpoint"] = {
                "iteration": int(itr + 1),
                "success_rate": float(sr),
                "average_reward": float(ar),
                "selection_metric": checkpoint_metric,
                "selection_score": float(eval_score),
                "avg_policy_loss": float(avg_pg_loss),
                "avg_value_loss": float(avg_v_loss),
                "critic_explained_variance": float(critic_ev),
                "critic_value_return_correlation": float(critic_corr),
                "rollout_success_rate": float(rollout_success_rate),
                "rollout_average_return": float(rollout_avg_return),
                "rollout_average_env_return": float(rollout_avg_env_return),
                "rollout_average_steps": float(rollout_avg_ep_steps),
                "ppo_control_step_training": bool(ppo_control_step_training),
                "ppo_update_samples": int(total_samples),
                "reward_source": reward_source,
                "success_label_success_reward": float(label_success_reward),
                "success_label_failure_reward": float(label_failure_reward),
                "success_label_nonterminal_reward": float(label_nonterminal_reward),
                "success_time_bonus": float(success_time_bonus),
                "success_time_max_steps": int(success_time_max_steps),
                "residual_mlp_stepwise_obs": bool(residual_mlp_stepwise_obs),
                "svm_reward_enable": bool(svm_reward_enable),
                "svm_reward_ready": bool(svm_reward_ready),
                "svm_feature_mode": str(svm_feature_mode),
                "svm_reward_coef": float(svm_reward_coef),
                "svm_reward_clip": float(svm_reward_clip),
                "svm_reward_mean": float(rollout_svm_reward_mean),
                "svm_reward_std": float(rollout_svm_reward_std),
                "svm_buffer_storage": str(svm_buffer_storage),
                "svm_buffer_dir": str(svm_buffer_dir),
                "svm_buffer_reset": bool(svm_buffer_reset),
                "svm_disc_loss": float(svm_last_stats["loss"]),
                "svm_disc_acc": float(svm_last_stats["acc"]),
                "svm_buffer_positive": int(svm_replay_buffer.num_positive)
                if svm_replay_buffer is not None
                else 0,
                "svm_buffer_negative": int(svm_replay_buffer.num_negative)
                if svm_replay_buffer is not None
                else 0,
                "bootstrap_on_truncation": bool(bootstrap_on_truncation),
                "frozen_diffusion": True,
            }

            with open(save_path / "config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(final_config_dict, f, allow_unicode=True, sort_keys=False)
            with open(save_path / "finetune_config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(current_ft_dict, f, allow_unicode=True, sort_keys=False)

            logging.info(f"Save    | checkpoint={save_path}")
            manager.update(step=itr + 1, loss=avg_pg_loss, ckpt_path=ckpt_path, reward=eval_score)
        elif (itr + 1) <= critic_warmup_iters:
            logging.info(
                f"Eval    | skipped during critic warmup ({itr + 1}/{critic_warmup_iters})"
            )


@hydra.main(version_base="1.2", config_name="ft_default", config_path="../../configs/finetune")
def train_cli(cfg: DictConfig):
    """Hydra 命令行入口。"""
    train_mlp_finetune(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


if __name__ == "__main__":
    default_args = [
        "policy=ft_zed_diffusion_mlp",
        "training.pretrained_ckpt_path='outputs/2_pretrain/train/2026-06-26/17-07-44_InsertCylinder-3Arms-v0_pre_zed_diffusion/checkpoints/148000_loss=0.0026_sr=0.0_ar=387.35'",
        "env.n_envs=10",
        "training.rollout_steps=400",
        "training.batch_size=16",
        "training.update_epochs=2",
        "wandb.enable=false",
    ]

    for arg in default_args:
        arg_key = arg.split("=")[0].lstrip("+")
        if not any(sys_arg.split("=")[0].lstrip("+") == arg_key for sys_arg in sys.argv):
            sys.argv.append(arg)

    train_cli()
