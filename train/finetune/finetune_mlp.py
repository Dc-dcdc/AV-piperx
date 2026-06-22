import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from train.finetune.critic import SharedFeatureCritic
from train.pretrain.eval_train import TopKCheckpointManager, custom_eval_policy


def deep_update_dict(base: dict, override: dict) -> dict:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def compute_value_diagnostics(values, returns, eps: float = 1e-8):
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


def log_box(title: str, rows: list[tuple[str, object]], width: int = 78):
    key_width = max([len(str(key)) for key, _ in rows] + [0])
    line = "-" * width
    header = "=" * width
    body = [header, title, line]
    for key, value in rows:
        body.append(f"{str(key):<{key_width}} : {value}")
    body.append(header)
    logging.info("\n" + "\n".join(body))


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}"


@contextmanager
def maybe_suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull):
            yield


@contextmanager
def maybe_quiet_eval_progress(enabled: bool):
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


class ActionMLP(nn.Module):
    """Post-process one action vector while preserving input/output action_dim."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        residual: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("ActionMLP depth must be >= 1")

        layers: list[nn.Module] = []
        in_dim = action_dim
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
        self.residual = residual
        self.residual_scale = residual_scale
        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        last_linear = self.net[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        orig_shape = actions.shape
        flat_actions = actions.reshape(-1, orig_shape[-1])
        delta_or_action = self.net(flat_actions).reshape(orig_shape)
        if self.residual:
            return actions + self.residual_scale * delta_or_action
        return delta_or_action


class FrozenDiffusionMLPPolicy(nn.Module):
    """
    Frozen pretrained diffusion policy plus a trainable MLP action head.

    The diffusion model produces the full final action horizon. The MLP is applied
    to each action vector, so its input and output dimension are both action_dim.
    PPO log-probabilities are computed only on the executed action chunk.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        action_dim: int,
        action_start: int,
        action_end: int,
        hidden_dim: int = 256,
        depth: int = 2,
        residual: bool = True,
        residual_scale: float = 1.0,
        init_std: float = 0.02,
        learn_std: bool = True,
        logprob_reduction: str = "mean",
    ):
        super().__init__()
        self.base_policy = base_policy
        self.config = base_policy.config
        self.expected_image_keys = base_policy.expected_image_keys
        self.action_start = action_start
        self.action_end = action_end
        self.logprob_reduction = logprob_reduction
        self.action_dim = action_dim
        self.action_mlp_hidden_dim = hidden_dim
        self.action_mlp_depth = depth
        self.action_mlp_residual = residual
        self.action_mlp_residual_scale = residual_scale
        self.action_mlp_learn_std = learn_std
        self.action_mlp = ActionMLP(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            residual=residual,
            residual_scale=residual_scale,
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

    def freeze_base_policy(self):
        self.base_policy.eval()
        for param in self.base_policy.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_policy.eval()
        return self

    def reset(self):
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
        return self.log_std.exp().clamp(min=1e-6)

    def adapter_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def normalize_history_batch(self, cond: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
        batch = self.normalize_history_batch(cond)
        return self.frozen_diffusion_actions_from_normalized_batch(
            batch,
            return_global_cond=return_global_cond,
        )

    def mean_actions(self, base_actions: torch.Tensor) -> torch.Tensor:
        return self.action_mlp(base_actions)

    def sample_from_mean(self, mean_actions: torch.Tensor, deterministic: bool):
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
    ) -> torch.Tensor:
        mean_actions = self.mean_actions(base_actions)
        return self.log_prob_from_mean(mean_actions, executed_actions)

    def forward_history(
        self,
        cond: dict[str, torch.Tensor],
        deterministic: bool = False,
        return_global_cond: bool = False,
    ):
        base_actions, global_cond = self.frozen_diffusion_actions(cond, return_global_cond=True)
        mean_actions = self.mean_actions(base_actions)
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
        batch = self.base_policy.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[k] for k in self.expected_image_keys],
                dim=-4,
            )

        self._queues = populate_queues(self._queues, batch)
        if len(self._queues["action"]) == 0:
            history_batch = {
                k: torch.stack(list(self._queues[k]), dim=1)
                for k in batch
                if k in self._queues
            }
            base_actions = self.frozen_diffusion_actions_from_normalized_batch(history_batch)
            mean_actions = self.mean_actions(base_actions)
            action_chunk = mean_actions[:, self.action_start:self.action_end]
            self._queues["action"].extend(action_chunk.transpose(0, 1))

        return self._queues["action"].popleft()

    def save_pretrained(self, save_directory: str | Path):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.base_policy.save_pretrained(save_directory)
        torch.save(
            {
                "action_mlp": self.action_mlp.state_dict(),
                "log_std": self.log_std.detach().cpu(),
                "action_start": self.action_start,
                "action_end": self.action_end,
                "logprob_reduction": self.logprob_reduction,
                "action_dim": self.action_dim,
                "hidden_dim": self.action_mlp_hidden_dim,
                "depth": self.action_mlp_depth,
                "residual": self.action_mlp_residual,
                "residual_scale": self.action_mlp_residual_scale,
                "learn_std": self.action_mlp_learn_std,
            },
            save_directory / "action_mlp.pt",
        )
        with open(save_directory / "action_mlp_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "wrapper": "FrozenDiffusionMLPPolicy",
                    "action_start": self.action_start,
                    "action_end": self.action_end,
                    "logprob_reduction": self.logprob_reduction,
                    "action_dim": self.action_dim,
                    "hidden_dim": self.action_mlp_hidden_dim,
                    "depth": self.action_mlp_depth,
                    "residual": self.action_mlp_residual,
                    "residual_scale": self.action_mlp_residual_scale,
                    "learn_std": self.action_mlp_learn_std,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


def freeze_batch_norm(module: nn.Module):
    if "BatchNorm" in module.__class__.__name__:
        module.eval()


def flatten_lerobot_obs(obs_dict):
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
    if hasattr(value, "copy"):
        return value.copy()
    return value


def reset_full_obs_queue(queue, obs, n_obs_steps):
    for key, value in obs.items():
        if key not in queue:
            queue[key] = deque(maxlen=n_obs_steps)
        queue[key].clear()
        for _ in range(n_obs_steps):
            queue[key].append(clone_obs_value(value))


def append_obs_queue(queue, obs, n_obs_steps):
    for key, value in obs.items():
        if key not in queue:
            queue[key] = deque(maxlen=n_obs_steps)
            for _ in range(n_obs_steps - 1):
                queue[key].append(clone_obs_value(value))
        queue[key].append(clone_obs_value(value))


def reset_done_envs_in_obs_queue(queue, obs, done_mask, n_envs, n_obs_steps):
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


def build_history_batch(stacked_raw_obs, policy, device):
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
    obs_norm = policy.base_policy.normalize_inputs(obs_batch.copy())
    if len(policy.expected_image_keys) > 0:
        obs_norm = dict(obs_norm)
        obs_norm["observation.images"] = torch.stack(
            [obs_norm[k] for k in policy.expected_image_keys],
            dim=-4,
        )
    return policy.base_policy.diffusion._prepare_global_conditioning(obs_norm)


def load_frozen_base_policy(cfg: DictConfig, device):
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
    return {
        "action_mlp": copy.deepcopy(policy.action_mlp.state_dict()),
        "log_std": policy.log_std.detach().cpu().clone(),
    }


def restore_trainable_state(policy: FrozenDiffusionMLPPolicy, state, device):
    policy.action_mlp.load_state_dict(state["action_mlp"], strict=True)
    with torch.no_grad():
        policy.log_std.copy_(state["log_std"].to(device))
    policy.to(device)


def train_mlp_finetune(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
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
    obs_cameras = list(dict.fromkeys(ref_cams + render_cams))

    if n_envs > 1:
        env = gym.vector.AsyncVectorEnv(
            [lambda: gym.make(id=env_id, cameras=obs_cameras) for _ in range(n_envs)],
            shared_memory=True,
            context="spawn",
            autoreset_mode="SameStep",
        )
    else:
        env = gym.make(id=env_id, cameras=obs_cameras)
    eval_env = gym.make(id=env_id, cameras=obs_cameras)

    with maybe_suppress_stdout(quiet_terminal):
        global_cond_dim = infer_global_cond_dim(base_policy, device)
    critic = SharedFeatureCritic(global_cond_dim=global_cond_dim).to(device)

    policy = FrozenDiffusionMLPPolicy(
        base_policy=base_policy,
        action_dim=action_dim,
        action_start=action_start,
        action_end=action_end,
        hidden_dim=int(getattr(cfg.training, "action_mlp_hidden_dim", 256)),
        depth=int(getattr(cfg.training, "action_mlp_depth", 2)),
        residual=bool(getattr(cfg.training, "action_mlp_residual", True)),
        residual_scale=float(getattr(cfg.training, "action_mlp_residual_scale", 1.0)),
        init_std=float(
            getattr(
                cfg.training,
                "action_mlp_std",
                getattr(cfg.training, "min_sampling_denoising_std", 0.02),
            )
        ),
        learn_std=bool(getattr(cfg.training, "action_mlp_learn_std", True)),
        logprob_reduction=str(getattr(cfg.training, "logprob_reduction", "mean")),
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
    manager = TopKCheckpointManager(
        out_dir=out_dir,
        max_keep=max_checkpoints,
        records_resume=bool(getattr(cfg.eval, "records_resume", True)),
        metric=str(getattr(cfg.eval, "checkpoint_metric", "reward")),
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

    log_box(
        "Run Setup",
        [
            ("env", env_id),
            ("cameras", obs_cameras),
            ("checkpoint", load_dir),
            ("action_dim", action_dim),
            ("horizon", horizon_steps),
            ("action_slice", f"[{action_start}:{action_end}]"),
            ("n_envs", n_envs),
            ("rollout_steps", n_steps),
            ("batch_size", batch_size),
            ("update_epochs", update_epochs),
            ("critic_warmup", critic_warmup_iters),
            ("actor_lr", f"{float(getattr(cfg.training, 'actor_lr', 1e-5)):.3e}"),
            ("critic_lr", f"{float(getattr(cfg.training, 'critic_lr', 3e-4)):.3e}"),
            ("clip_ratio", clip_ratio),
            ("target_kl", target_kl),
            ("reward_scale_running", reward_scale_running),
            ("reward_scale_const", reward_scale_const),
            ("progress_bars", show_progress),
            ("quiet_terminal", quiet_terminal),
        ],
    )
    log_box(
        "Model Params",
        [
            ("frozen_diffusion", f"{sum(p.numel() for p in base_policy.parameters()) / 1e6:.2f}M"),
            ("trainable_mlp", f"{sum(p.numel() for p in policy.adapter_parameters()) / 1e6:.2f}M"),
            ("critic", f"{sum(p.numel() for p in critic.parameters()) / 1e6:.2f}M"),
            ("mlp_hidden", getattr(cfg.training, "action_mlp_hidden_dim", 256)),
            ("mlp_depth", getattr(cfg.training, "action_mlp_depth", 2)),
            ("mlp_residual", getattr(cfg.training, "action_mlp_residual", True)),
            ("action_std", f"{float(policy.action_std.mean().detach().cpu().item()):.5f}"),
        ],
    )

    prev_obs, _ = env.reset()
    prev_obs = flatten_lerobot_obs(prev_obs)
    raw_obs_queue = {key: deque(maxlen=n_obs_steps) for key in prev_obs.keys()}
    reset_full_obs_queue(raw_obs_queue, prev_obs, n_obs_steps)

    running_ep_rewards = np.zeros(n_envs, dtype=np.float32)
    running_reward_scaler = RunningRewardScaler(n_envs, gamma=gamma)
    next_chunk_firsts = np.ones(n_envs, dtype=np.float32)
    best_policy_state = snapshot_trainable_state(policy)
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
        else:
            temp_buffer_dir = None
            buffer_path = None
            base_action_trajs = np.zeros(
                (n_steps, n_envs, horizon_steps, action_dim),
                dtype=np.float32,
            )
            action_trajs = np.zeros((n_steps, n_envs, act_steps, action_dim), dtype=np.float32)

        old_logprob_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
        reward_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
        terminated_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
        firsts_trajs = np.zeros((n_steps + 1, n_envs), dtype=np.float32)
        firsts_trajs[0] = next_chunk_firsts
        completed_ep_rewards = []
        completed_ep_successes = []

        policy.eval()
        logging.info("Rollout | collecting frozen Diffusion + stochastic MLP actions")
        for step in tqdm(
            range(n_steps),
            desc=f"Rollout {itr + 1:04d}",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        ):
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
                    samples = policy.forward_history(
                        cond=batch_obs,
                        deterministic=False,
                        return_global_cond=False,
                    )
                output_venv = samples["actions"].cpu().numpy()
                base_venv = samples["base_actions"].cpu().numpy()
                old_logprob_venv = samples["log_probs"].cpu().numpy()

            action_venv = output_venv[:, action_start:action_end]
            chunk_reward = np.zeros(n_envs, dtype=np.float32)
            any_done_accum = np.zeros(n_envs, dtype=bool)
            true_term_accum = np.zeros(n_envs, dtype=bool)
            success_accum = np.zeros(n_envs, dtype=bool)
            safe_actions = np.zeros((n_envs, action_dim), dtype=np.float32)

            for step_i in range(act_steps):
                curr_action = action_venv[:, step_i, :].copy()
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

                active_mask = ~any_done_accum
                chunk_reward += reward_venv * active_mask
                just_done = (terminated_venv | truncated_venv) & active_mask
                just_success = info_success_mask(info_venv, just_done, n_envs)
                success_accum = success_accum | just_success
                true_term_accum = true_term_accum | (terminated_venv & active_mask)

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

            prev_obs = obs_venv
            running_ep_rewards += chunk_reward
            for env_idx in range(n_envs):
                if any_done_accum[env_idx]:
                    completed_ep_rewards.append(float(running_ep_rewards[env_idx]))
                    completed_ep_successes.append(bool(success_accum[env_idx]))
                    running_ep_rewards[env_idx] = 0.0

            for key in obs_trajs:
                obs_trajs[key][step] = stacked_raw_obs[key]
            base_action_trajs[step] = base_venv
            action_trajs[step] = action_venv
            old_logprob_trajs[step] = old_logprob_venv
            reward_trajs[step] = chunk_reward
            terminated_trajs[step] = true_term_accum
            firsts_trajs[step + 1] = any_done_accum

        next_chunk_firsts = firsts_trajs[-1].copy()
        rollout_avg_return = np.mean(completed_ep_rewards) if completed_ep_rewards else float("-inf")
        rollout_success_rate = np.mean(completed_ep_successes) if completed_ep_successes else 0.0
        logging.info(
            "Rollout | "
            f"episodes={len(completed_ep_rewards):3d} | "
            f"success={fmt_pct(rollout_success_rate):>6} | "
            f"avg_return={rollout_avg_return:8.2f}"
        )
            

        if skip_update:
            logging.info("training.skip_update=true; skipping MLP/Critic update.")
            try:
                del base_action_trajs, action_trajs, obs_trajs
                if use_disk_cache and temp_buffer_dir is not None:
                    temp_buffer_dir.cleanup()
            except Exception:
                pass
            torch.cuda.empty_cache()
            continue

        total_samples = n_steps * n_envs
        obs_flat = {
            key: value.reshape(total_samples, *value.shape[2:])
            for key, value in obs_trajs.items()
        }
        base_actions_flat = base_action_trajs.reshape(total_samples, horizon_steps, action_dim)
        actions_flat = action_trajs.reshape(total_samples, act_steps, action_dim)
        old_logprobs_flat = old_logprob_trajs.reshape(total_samples)

        logging.info("Update  | computing values and GAE")
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
            delta = scaled_rewards[t] * reward_scale_const + gamma * next_val * nonterminal - values_trajs[t]
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

        actor_update_enabled = bool(update_actor and itr >= critic_warmup_iters)
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
        logging.info(
            "Update  | "
            f"actor_update={actor_update_enabled} | "
            f"epochs={update_epochs} | "
            f"minibatches/epoch={max(1, math.ceil(total_samples / batch_size))}"
        )

        for epoch in tqdm(
            range(update_epochs),
            desc=f"PPO {itr + 1:04d}",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            if early_stop:
                break
            indices = torch.randperm(total_samples, device=device)
            num_batches = max(1, math.ceil(total_samples / batch_size))
            for batch_idx in tqdm(
                range(num_batches),
                desc=f"epoch {epoch + 1}",
                leave=False,
                dynamic_ncols=True,
                disable=not show_progress,
            ):
                start = batch_idx * batch_size
                end = min(start + batch_size, total_samples)
                inds_b = indices[start:end]
                inds_np = inds_b.cpu().numpy()

                obs_b = {}
                for key, value in obs_flat.items():
                    tensor_value = torch.from_numpy(value[inds_np]).float().to(device)
                    if "images" in key:
                        tensor_value = tensor_value.permute(0, 1, 4, 2, 3) / 255.0
                    obs_b[key] = tensor_value

                base_actions_b = torch.from_numpy(base_actions_flat[inds_np]).float().to(device)
                actions_b = torch.from_numpy(actions_flat[inds_np]).float().to(device)
                returns_b = returns_k[inds_b]
                advantages_b = advantages_k[inds_b]
                advantages_b = normalize_clip_minibatch_advantage(advantages_b)
                old_logprobs_b = old_logprobs_k[inds_b]

                with torch.no_grad():
                    global_cond_b = global_cond_from_obs(policy, obs_b)

                if actor_update_enabled:
                    mean_actions_b = policy.mean_actions(base_actions_b)
                    new_logprobs_b = policy.log_prob_from_mean(
                        mean_actions_b,
                        actions_b,
                        action_start=action_start,
                        action_end=action_end,
                    )
                    raw_log_ratio = new_logprobs_b - old_logprobs_b
                    log_ratio = torch.clamp(raw_log_ratio, min=-20.0, max=5.0)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * advantages_b
                    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_b
                    pg_loss = -torch.min(surr1, surr2).mean()
                    bc_loss = F.mse_loss(
                        mean_actions_b[:, action_start:action_end],
                        base_actions_b[:, action_start:action_end],
                    )
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                else:
                    with torch.no_grad():
                        mean_actions_b = policy.mean_actions(base_actions_b)
                        new_logprobs_b = policy.log_prob_from_mean(
                            mean_actions_b,
                            actions_b,
                            action_start=action_start,
                            action_end=action_end,
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

        log_box(
            f"Iteration {itr + 1} Summary",
            [
                ("episodes", len(completed_ep_rewards)),
                ("rollout_success", fmt_pct(rollout_success_rate)),
                ("rollout_return", f"{rollout_avg_return:.2f}"),
                ("actor_update", actor_update_enabled),
                ("value_loss", fmt_float(avg_v_loss)),
                ("policy_loss", fmt_float(avg_pg_loss)),
                ("bc_loss", f"{avg_bc_loss:.5f}"),
                ("kl_avg/max", f"{avg_kl:.3e} / {max_kl:.3e}"),
                ("critic_ev", fmt_float(critic_ev)),
                ("value_return_corr", fmt_float(critic_corr)),
                ("action_std", f"{float(policy.action_std.mean().detach().cpu().item()):.5f}"),
            ],
        )

        try:
            del (
                base_actions_flat,
                actions_flat,
                obs_flat,
                base_action_trajs,
                action_trajs,
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

            if ar > best_eval_reward:
                best_eval_reward = ar
                best_eval_success_rate = sr
                best_policy_state = snapshot_trainable_state(policy)
                eval_collapse_count = 0
            else:
                rollback_enabled = bool(getattr(cfg.training, "rollback_on_eval_collapse", True))
                rollback_sr = float(getattr(cfg.training, "rollback_success_rate", 0.1))
                rollback_reward = float(getattr(cfg.training, "rollback_reward", -100.0))
                rollback_patience = int(getattr(cfg.training, "rollback_patience", 1))
                eval_collapsed = sr <= rollback_sr and ar <= rollback_reward
                if rollback_enabled and eval_collapsed:
                    eval_collapse_count += 1
                    logging.warning(
                        f"Eval collapse detected ({eval_collapse_count}/{rollback_patience}). "
                        f"success={sr * 100:.1f}% reward={ar:.2f}"
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
                    "action_mlp_checkpoint": "action_mlp.pt",
                    "action_mlp_hidden_dim": int(getattr(cfg.training, "action_mlp_hidden_dim", 256)),
                    "action_mlp_depth": int(getattr(cfg.training, "action_mlp_depth", 2)),
                    "action_mlp_residual": bool(getattr(cfg.training, "action_mlp_residual", True)),
                    "action_mlp_residual_scale": float(
                        getattr(cfg.training, "action_mlp_residual_scale", 1.0)
                    ),
                    "action_mlp_learn_std": bool(getattr(cfg.training, "action_mlp_learn_std", True)),
                },
            )
            final_config_dict["policy"] = final_policy
            final_config_dict["checkpoint"] = {
                "iteration": int(itr + 1),
                "success_rate": float(sr),
                "average_reward": float(ar),
                "avg_policy_loss": float(avg_pg_loss),
                "avg_value_loss": float(avg_v_loss),
                "critic_explained_variance": float(critic_ev),
                "critic_value_return_correlation": float(critic_corr),
                "rollout_success_rate": float(rollout_success_rate),
                "rollout_average_return": float(rollout_avg_return),
                "frozen_diffusion": True,
            }

            with open(save_path / "config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(final_config_dict, f, allow_unicode=True, sort_keys=False)
            with open(save_path / "finetune_config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(current_ft_dict, f, allow_unicode=True, sort_keys=False)

            logging.info(f"Save    | checkpoint={save_path}")
            manager.update(step=itr + 1, loss=avg_pg_loss, ckpt_path=ckpt_path, reward=ar)
        elif (itr + 1) <= critic_warmup_iters:
            logging.info(
                f"Eval    | skipped during critic warmup ({itr + 1}/{critic_warmup_iters})"
            )


@hydra.main(version_base="1.2", config_name="ft_default", config_path="../../configs/finetune")
def train_cli(cfg: DictConfig):
    train_mlp_finetune(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


if __name__ == "__main__":
    default_args = [
        "policy=ft_wrist_diffusion_mlp",
        "training.pretrained_ckpt_path='outputs/1_hugging_model/pre_sim_sew_needle_2arms_wrist_diffusion'",
        "env.n_envs=1",
        "training.rollout_steps=400",
        "training.batch_size=16",
        "training.update_epochs=4",
        "+training.show_progress=false",
        "+training.quiet_terminal=true",
        "wandb.enable=false",
    ]

    for arg in default_args:
        arg_key = arg.split("=")[0].lstrip("+")
        if not any(sys_arg.split("=")[0].lstrip("+") == arg_key for sys_arg in sys.argv):
            sys.argv.append(arg)

    train_cli()
