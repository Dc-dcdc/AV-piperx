from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from collections import deque
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
import mujoco

_TORCHVISION_COMPAT_LIBS: list[Any] = []


def _register_torchvision_compat_ops() -> None:
    try:
        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return
    except RuntimeError as exc:
        if "does not exist" not in str(exc):
            return

    try:
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        _TORCHVISION_COMPAT_LIBS.append(lib)
    except RuntimeError as exc:
        if "already" not in str(exc) and "Only a single TORCH_LIBRARY" not in str(exc):
            raise


_register_torchvision_compat_ops()

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import env.mjlab  # noqa: F401
from env.constants import LEFT_JOINT_NAMES, MIDDLE_JOINT_NAMES, RIGHT_JOINT_NAMES
from env.mjlab.insert_cylinder_cfg import (
    ACTION_CLIP,
    ACTION_SCALE,
    ALL_ROBOT_JOINT_POS,
    TASK_ID,
    make_insert_cylinder_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from lerobot.common.policies.utils import populate_queues
from lerobot.common.utils.utils import get_safe_torch_device, init_logging, set_global_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional for training.
    SummaryWriter = None


DEFAULT_CONFIG = (
    ROOT_DIR / "configs" / "finetune" / "policy" / "ft_zed_diffusion_residual_mjlab.yaml"
)
ACTION_JOINT_NAMES = tuple(LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES + MIDDLE_JOINT_NAMES)
ACTION_DIM = len(ACTION_JOINT_NAMES)
GRIPPER_INDICES = (6, 13)
GRIPPER_JOINT_MAX = 0.05


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
    assert first.ndim == 2
    _, nstep = reward.shape
    ret = np.zeros_like(reward)
    for t in range(nstep):
        prevret = ret[:, t] = reward[:, t] + (1 - first[:, t]) * gamma * prevret
    return ret


class RunningRewardScaler:
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


@contextmanager
def maybe_suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull):
            yield


def layer_init(layer: nn.Module, nonlinearity: str = "ReLU", std: float = math.sqrt(2.0)):
    if isinstance(layer, nn.Linear):
        if nonlinearity in {"ReLU", "SiLU"}:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        else:
            nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    activation: str = "SiLU",
    output_std: float = 0.01,
    output_bias: bool = False,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    act_cls = getattr(nn, activation)
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for _ in range(int(depth)):
        layers.append(layer_init(nn.Linear(last_dim, int(hidden_dim)), nonlinearity=activation))
        layers.append(act_cls())
        last_dim = int(hidden_dim)
    out = nn.Linear(last_dim, int(output_dim), bias=output_bias)
    nn.init.orthogonal_(out.weight, gain=float(output_std))
    if out.bias is not None:
        nn.init.constant_(out.bias, 0.0)
    layers.append(out)
    return nn.Sequential(*layers)


class ResidualPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dim: int = 512,
        actor_depth: int = 2,
        critic_hidden_dim: int = 512,
        critic_depth: int = 2,
        activation: str = "SiLU",
        init_std: float = 0.02,
        learn_std: bool = True,
        action_head_std: float = 0.0,
        action_scale: float = 0.1,
        max_delta: float = 0.0,
        logprob_reduction: str = "sum",
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be > 0")
        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if init_std <= 0:
            raise ValueError("init_std must be > 0")
        logprob_reduction = str(logprob_reduction).lower()
        if logprob_reduction not in {"mean", "sum"}:
            raise ValueError("logprob_reduction must be 'mean' or 'sum'")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_scale = float(action_scale)
        self.max_delta = float(max_delta)
        self.logprob_reduction = logprob_reduction
        self.actor_mean = build_mlp(
            input_dim=self.obs_dim,
            hidden_dim=int(actor_hidden_dim),
            depth=int(actor_depth),
            output_dim=self.action_dim,
            activation=activation,
            output_std=float(action_head_std),
            output_bias=False,
        )
        self.critic = build_mlp(
            input_dim=self.obs_dim,
            hidden_dim=int(critic_hidden_dim),
            depth=int(critic_depth),
            output_dim=1,
            activation=activation,
            output_std=0.01,
            output_bias=True,
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, self.action_dim) * math.log(float(init_std)),
            requires_grad=bool(learn_std),
        )

    @property
    def action_std(self) -> torch.Tensor:
        return self.actor_logstd.exp().clamp(min=1e-6)

    @property
    def actor_parameters(self):
        return list(self.actor_mean.parameters()) + [self.actor_logstd]

    @property
    def critic_parameters(self):
        return list(self.critic.parameters())

    def residual_delta(self, residual_action: torch.Tensor) -> torch.Tensor:
        delta = residual_action * self.action_scale
        if self.max_delta > 0.0:
            max_delta = torch.as_tensor(self.max_delta, dtype=delta.dtype, device=delta.device)
            delta = torch.tanh(delta / max_delta) * max_delta
        return delta

    def get_value(self, residual_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(residual_obs).squeeze(-1)

    def _reduce_logprob(self, values: torch.Tensor) -> torch.Tensor:
        if self.logprob_reduction == "sum":
            return values.sum(dim=-1)
        return values.mean(dim=-1)

    def get_action_and_value(
        self,
        residual_obs: torch.Tensor,
        residual_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        mean = self.actor_mean(residual_obs)
        std = self.action_std.expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if residual_action is None:
            residual_action = mean if deterministic else dist.sample()
        logprob = self._reduce_logprob(dist.log_prob(residual_action))
        entropy = self._reduce_logprob(dist.entropy())
        value = self.get_value(residual_obs)
        return residual_action, logprob, entropy, value, mean


class FrozenDiffusionResidualPolicy(nn.Module):
    def __init__(
        self,
        base_policy: nn.Module,
        action_dim: int,
        action_start: int,
        action_end: int,
        global_cond_dim: int,
        actor_hidden_dim: int = 512,
        actor_depth: int = 2,
        critic_hidden_dim: int = 512,
        critic_depth: int = 2,
        activation: str = "SiLU",
        init_std: float = 0.02,
        learn_std: bool = True,
        action_head_std: float = 0.0,
        action_scale: float = 0.1,
        max_delta: float = 0.0,
        logprob_reduction: str = "sum",
        stepwise_obs: bool = True,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.config = base_policy.config
        self.expected_image_keys = list(getattr(base_policy, "expected_image_keys", []))
        self.action_dim = int(action_dim)
        self.action_start = int(action_start)
        self.action_end = int(action_end)
        self.global_cond_dim = int(global_cond_dim)
        self.stepwise_obs = bool(stepwise_obs)
        self.residual_obs_dim = self.global_cond_dim + self.action_dim
        self.residual_policy = ResidualPolicy(
            obs_dim=self.residual_obs_dim,
            action_dim=self.action_dim,
            actor_hidden_dim=actor_hidden_dim,
            actor_depth=actor_depth,
            critic_hidden_dim=critic_hidden_dim,
            critic_depth=critic_depth,
            activation=activation,
            init_std=init_std,
            learn_std=learn_std,
            action_head_std=action_head_std,
            action_scale=action_scale,
            max_delta=max_delta,
            logprob_reduction=logprob_reduction,
        )
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
        return self.residual_policy.action_std

    @property
    def actor_parameters(self):
        return self.residual_policy.actor_parameters

    @property
    def critic_parameters(self):
        return self.residual_policy.critic_parameters

    def adapter_parameters(self):
        return [p for p in self.residual_policy.parameters() if p.requires_grad]

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
        if return_global_cond:
            return normalized_actions, global_cond
        return normalized_actions

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

    def make_residual_obs(
        self,
        global_cond: torch.Tensor,
        base_action: torch.Tensor,
    ) -> torch.Tensor:
        if base_action.ndim == 3:
            base_action = base_action[:, 0, :]
        return torch.cat([global_cond.reshape(global_cond.shape[0], -1), base_action], dim=-1)

    def action_from_residual(
        self,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
    ) -> torch.Tensor:
        return base_action + self.residual_policy.residual_delta(residual_action)

    @torch.no_grad()
    def denormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        return self.base_policy.unnormalize_outputs({"action": normalized_action})["action"]

    def step_action(
        self,
        base_action: torch.Tensor,
        global_cond: torch.Tensor,
        residual_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        residual_obs = self.make_residual_obs(global_cond, base_action)
        residual_action, logprob, entropy, value, mean = self.residual_policy.get_action_and_value(
            residual_obs,
            residual_action=residual_action,
            deterministic=deterministic,
        )
        normalized_action = self.action_from_residual(base_action, residual_action)
        action = self.denormalize_action(normalized_action)
        return {
            "residual_obs": residual_obs,
            "residual_action": residual_action,
            "mean_residual_action": mean,
            "normalized_action": normalized_action,
            "action": action,
            "logprob": logprob,
            "entropy": entropy,
            "value": value,
        }

    def value_for_step(self, base_action: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        residual_obs = self.make_residual_obs(global_cond, base_action)
        return self.residual_policy.get_value(residual_obs)

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
            base_actions, global_cond = self.frozen_diffusion_actions_from_normalized_batch(
                history_batch,
                return_global_cond=True,
            )
            action_chunk = base_actions[:, self.action_start : self.action_end]
            if not self.stepwise_obs:
                flat_base = action_chunk.reshape(-1, self.action_dim)
                flat_cond = global_cond[:, None, :].expand(-1, action_chunk.shape[1], -1)
                flat_cond = flat_cond.reshape(-1, self.global_cond_dim)
                corrected = self.step_action(
                    base_action=flat_base,
                    global_cond=flat_cond,
                    deterministic=True,
                )["normalized_action"].reshape_as(action_chunk)
                action_chunk = corrected
            self._queues["action"].extend(action_chunk.transpose(0, 1))

        queued_action = self._queues["action"].popleft()
        if not self.stepwise_obs:
            return self.denormalize_action(queued_action)

        history_batch = {
            k: torch.stack(list(self._queues[k]), dim=1)
            for k in batch
            if k in self._queues
        }
        global_cond = self.base_policy.diffusion._prepare_global_conditioning(history_batch)
        return self.step_action(
            base_action=queued_action,
            global_cond=global_cond,
            deterministic=True,
        )["action"]

    def save_pretrained(self, save_directory: str | Path):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.base_policy.save_pretrained(save_directory)
        adapter_state = {
            "wrapper": "FrozenDiffusionResidualPolicy",
            "model": self.residual_policy.state_dict(),
            "action_dim": self.action_dim,
            "action_start": self.action_start,
            "action_end": self.action_end,
            "global_cond_dim": self.global_cond_dim,
            "residual_obs_dim": self.residual_obs_dim,
            "stepwise_obs": self.stepwise_obs,
            "action_scale": self.residual_policy.action_scale,
            "max_delta": self.residual_policy.max_delta,
            "logprob_reduction": self.residual_policy.logprob_reduction,
            "residual_action_space": "normalized",
        }
        torch.save(adapter_state, save_directory / "residual_policy.pt")
        adapter_config = {
            key: value
            for key, value in adapter_state.items()
            if key != "model"
        }
        with open(save_directory / "residual_policy_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2, ensure_ascii=False)


def normalize_clip_minibatch_advantage(
    advantages: torch.Tensor,
    norm_adv: bool,
    lower_q: float,
    upper_q: float,
) -> torch.Tensor:
    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    if lower_q > 0.0 or upper_q < 1.0:
        adv_lower = torch.quantile(advantages, lower_q)
        adv_upper = torch.quantile(advantages, upper_q)
        advantages = torch.clamp(advantages, adv_lower, adv_upper)
    return advantages


def load_frozen_base_policy(cfg, device):
    ckpt_path = cfg.training.pretrained_ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find pretrained checkpoint: {ckpt_path}")

    hf_model_dir = os.path.join(ckpt_path, "pretrained_model")
    load_dir = hf_model_dir if os.path.exists(hf_model_dir) else ckpt_path
    logging.info(f"Loading frozen diffusion policy from: {load_dir}")

    from lerobot.common.policies.factory import make_policy
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
    expected_image_keys = list(getattr(base_policy, "expected_image_keys", []))
    if len(expected_image_keys) > 0:
        dummy_batch["observation.images"] = torch.stack(
            [dummy_batch[k] for k in expected_image_keys],
            dim=-4,
        )
    with torch.no_grad():
        dummy_cond = base_policy.diffusion._prepare_global_conditioning(dummy_batch)
    return dummy_cond.shape[-1]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _get(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_device(raw_cfg: dict[str, Any], args: argparse.Namespace) -> str:
    if args.device:
        return str(args.device)

    gpu_ids = args.gpu_ids if args.gpu_ids is not None else str(raw_cfg.get("gpu_ids", "auto"))
    gpu_ids = str(gpu_ids).strip().lower()
    if gpu_ids in {"cpu", "none", ""}:
        return "cpu"
    if gpu_ids == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if gpu_ids == "all":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return f"cuda:{gpu_ids.split(',')[0].strip()}"


def _resolve_name_value(pattern_values: dict[str, Any], names: tuple[str, ...]) -> list[Any]:
    resolved = []
    for name in names:
        matched = None
        for pattern, value in pattern_values.items():
            if re.fullmatch(pattern, name):
                matched = value
                break
        if matched is None:
            raise KeyError(f"No pattern matched joint/action name: {name}")
        resolved.append(matched)
    return resolved


class MjlabDiffusionAdapter:
    """Translate between mjlab tensors and the frozen Diffusion policy interface."""

    def __init__(
        self,
        *,
        expected_state_dim: int,
        action_dim: int,
        state_mode: str,
        base_action_mode: str,
        env_action_clip: float | None,
        device: torch.device,
    ):
        self.expected_state_dim = int(expected_state_dim)
        self.action_dim = int(action_dim)
        self.state_mode = str(state_mode).lower()
        self.base_action_mode = str(base_action_mode).lower()
        self.env_action_clip = env_action_clip
        self.device = device
        self._resolved_state_mode: str | None = None

        default_joint_pos = [float(ALL_ROBOT_JOINT_POS[name]) for name in ACTION_JOINT_NAMES]
        self.default_joint_pos = torch.tensor(default_joint_pos, dtype=torch.float32, device=device)
        self.action_scale = torch.tensor(
            [float(v) for v in _resolve_name_value(ACTION_SCALE, ACTION_JOINT_NAMES)],
            dtype=torch.float32,
            device=device,
        )
        clip_pairs = _resolve_name_value(ACTION_CLIP, ACTION_JOINT_NAMES)
        self.target_min = torch.tensor([float(pair[0]) for pair in clip_pairs], device=device)
        self.target_max = torch.tensor([float(pair[1]) for pair in clip_pairs], device=device)

    @property
    def resolved_state_mode(self) -> str:
        return self._resolved_state_mode or self.state_mode

    def _choose_state_mode(self, actor_dim: int) -> str:
        if self.state_mode != "auto":
            return self.state_mode
        if self.expected_state_dim == self.action_dim:
            return "legacy_agent_pos"
        if self.expected_state_dim == actor_dim:
            return "actor"
        raise ValueError(
            "Cannot infer mjlab->Diffusion state mapping: "
            f"Diffusion expects state dim {self.expected_state_dim}, "
            f"mjlab actor obs dim is {actor_dim}, action dim is {self.action_dim}. "
            "Set env.state_mode to 'legacy_agent_pos', 'joint_pos_rel', or 'actor'."
        )

    def state_from_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        actor = obs["actor"].detach().to(self.device, dtype=torch.float32)
        mode = self._choose_state_mode(actor.shape[-1])
        self._resolved_state_mode = mode

        if mode == "actor":
            state = actor
        elif mode == "joint_pos_rel":
            state = actor[:, : self.action_dim]
        elif mode in {"legacy_agent_pos", "agent_pos", "legacy"}:
            rel_joint_pos = actor[:, : self.action_dim]
            state = rel_joint_pos + self.default_joint_pos
            for gripper_idx in GRIPPER_INDICES:
                state[:, gripper_idx] = (state[:, gripper_idx] / GRIPPER_JOINT_MAX).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unsupported env.state_mode={self.state_mode!r}")

        if state.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"State dim mismatch for state_mode={mode!r}: got {state.shape[-1]}, "
                f"but frozen Diffusion expects {self.expected_state_dim}."
            )
        return state

    def action_to_env(self, action: torch.Tensor) -> torch.Tensor:
        """Convert policy-space action to mjlab raw action."""
        if self.base_action_mode in {"mjlab_raw", "raw", "normalized"}:
            env_action = action
        elif self.base_action_mode in {"legacy_absolute", "legacy", "gym_absolute"}:
            target = action.clone()
            for gripper_idx in GRIPPER_INDICES:
                target[:, gripper_idx] = target[:, gripper_idx].clamp(0.0, 1.0) * GRIPPER_JOINT_MAX
            target = torch.max(torch.min(target, self.target_max), self.target_min)
            env_action = (target - self.default_joint_pos) / self.action_scale
        else:
            raise ValueError(f"Unsupported env.base_action_mode={self.base_action_mode!r}")

        if self.env_action_clip is not None and self.env_action_clip > 0:
            env_action = env_action.clamp(-float(self.env_action_clip), float(self.env_action_clip))
        return env_action


def _image_keys_from_policy(base_policy) -> list[str]:
    image_keys = [
        key
        for key in base_policy.config.input_shapes.keys()
        if key.startswith("observation.images.")
    ]
    expected_image_keys = list(getattr(base_policy, "expected_image_keys", []))
    return list(dict.fromkeys(image_keys + expected_image_keys))


def _camera_name_from_image_key(image_key: str) -> str:
    return image_key.removeprefix("observation.images.")


def _camera_map_from_cfg(raw_value: Any, image_keys: list[str]) -> dict[str, str]:
    if raw_value is None or (isinstance(raw_value, str) and raw_value.lower() in {"null", "none", ""}):
        raw_value = {}
    if not isinstance(raw_value, dict):
        raise ValueError("env.camera_map must be a mapping from image key/camera suffix to mjlab camera name.")

    camera_map: dict[str, str] = {}
    for image_key in image_keys:
        suffix = _camera_name_from_image_key(image_key)
        camera_map[image_key] = str(
            raw_value.get(image_key, raw_value.get(suffix, suffix))
        )
    return camera_map


class MjlabCameraRenderer:
    """Render mjlab RGB cameras into LeRobot image tensors."""

    def __init__(
        self,
        *,
        env: ManagerBasedRlEnv,
        image_shapes: dict[str, tuple[int, int, int]],
        camera_map: dict[str, str],
        device: torch.device,
    ):
        self.env = env
        self.image_shapes = image_shapes
        self.camera_map = camera_map
        self.device = device
        self.num_envs = int(env.num_envs)
        self.image_keys = list(image_shapes.keys())
        self._renderer: OffscreenRenderer | None = None

        if not self.image_keys:
            return

        unique_sizes = {(shape[1], shape[2]) for shape in image_shapes.values()}
        if len(unique_sizes) != 1:
            raise ValueError(
                "All image inputs for mjlab camera rendering must currently share one HxW. "
                f"Got: {image_shapes}"
            )
        for key, shape in image_shapes.items():
            if len(shape) != 3 or int(shape[0]) != 3:
                raise ValueError(f"{key} must have shape [3, H, W], got {shape}.")

        height, width = next(iter(unique_sizes))
        viewer_cfg = copy.deepcopy(env.cfg.viewer)
        viewer_cfg.height = int(height)
        viewer_cfg.width = int(width)
        viewer_cfg.max_extra_envs = 0
        self._renderer = OffscreenRenderer(
            model=env.sim.mj_model,
            cfg=viewer_cfg,
            scene=env.scene,
            sim_model=env.sim.model,
            expanded_fields=env.sim.expanded_fields,
        )
        self._renderer.initialize()
        self._validate_cameras()

    def _validate_cameras(self) -> None:
        if self._renderer is None:
            return
        missing = []
        available = [
            mujoco.mj_id2name(self.env.sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self.env.sim.mj_model.ncam)
        ]
        for image_key, camera_name in list(self.camera_map.items()):
            camera_id = mujoco.mj_name2id(
                self.env.sim.mj_model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )
            if camera_id < 0:
                suffix_matches = [
                    name for name in available if name == camera_name or name.endswith(f"/{camera_name}")
                ]
                if len(suffix_matches) == 1:
                    self.camera_map[image_key] = suffix_matches[0]
                else:
                    missing.append(camera_name)
        if missing:
            raise ValueError(
                f"mjlab cameras not found: {missing}. Available cameras: {available}"
            )

    @torch.no_grad()
    def render(self) -> dict[str, torch.Tensor]:
        if self._renderer is None:
            return {}

        rendered: dict[str, torch.Tensor] = {}
        for image_key in self.image_keys:
            _, height, width = self.image_shapes[image_key]
            camera_name = self.camera_map[image_key]
            frames = np.empty((self.num_envs, height, width, 3), dtype=np.uint8)
            for env_idx in range(self.num_envs):
                self._renderer._cfg.env_idx = int(env_idx)
                self._renderer.update(self.env.sim.data, camera=camera_name)
                frame = self._renderer.render()
                if frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        f"Rendered {camera_name} shape {frame.shape} does not match {(height, width, 3)}."
                    )
                frames[env_idx] = np.ascontiguousarray(frame[:, :, :3])
            tensor = torch.from_numpy(frames).to(self.device, dtype=torch.float32)
            tensor = tensor.permute(0, 3, 1, 2).contiguous().div_(255.0)
            rendered[image_key] = tensor
        return rendered

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class PolicyObsHistory:
    """Keep vectorized mjlab observations in LeRobot history shape."""

    def __init__(
        self,
        *,
        adapter: MjlabDiffusionAdapter,
        camera_renderer: MjlabCameraRenderer | None,
        image_keys: list[str],
        num_envs: int,
        n_obs_steps: int,
        device: torch.device,
    ):
        self.adapter = adapter
        self.camera_renderer = camera_renderer
        self.image_keys = list(image_keys)
        self.num_envs = int(num_envs)
        self.n_obs_steps = int(n_obs_steps)
        self.device = device
        self.state: torch.Tensor | None = None
        self.images: dict[str, torch.Tensor] = {}

    def reset(self, obs: dict[str, torch.Tensor]) -> None:
        state = self.adapter.state_from_obs(obs)
        self.state = state[:, None, :].repeat(1, self.n_obs_steps, 1).contiguous()
        if self.camera_renderer is not None and self.image_keys:
            images = self.camera_renderer.render()
            self.images = {
                key: images[key][:, None, :, :, :].repeat(1, self.n_obs_steps, 1, 1, 1).contiguous()
                for key in self.image_keys
            }

    def append(self, obs: dict[str, torch.Tensor], done: torch.Tensor) -> None:
        if self.state is None:
            self.reset(obs)
            return

        state = self.adapter.state_from_obs(obs)
        self.state = torch.roll(self.state, shifts=-1, dims=1)
        self.state[:, -1, :] = state
        images = self.camera_renderer.render() if self.camera_renderer is not None and self.image_keys else {}
        if torch.any(done):
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            self.state[done_ids] = state[done_ids, None, :].repeat(1, self.n_obs_steps, 1)
        for key in self.image_keys:
            if key not in images:
                raise KeyError(f"Camera renderer did not return required image key {key!r}.")
            image_history = self.images[key]
            image_history = torch.roll(image_history, shifts=-1, dims=1)
            image_history[:, -1] = images[key]
            if torch.any(done):
                image_history[done_ids] = images[key][done_ids, None, :, :, :].repeat(
                    1,
                    self.n_obs_steps,
                    1,
                    1,
                    1,
                )
            self.images[key] = image_history

    def batch(self) -> dict[str, torch.Tensor]:
        if self.state is None:
            raise RuntimeError("Observation history has not been initialized.")
        batch = {"observation.state": self.state}
        for key in self.image_keys:
            if key not in self.images:
                raise RuntimeError(f"Image history for {key!r} has not been initialized.")
            batch[key] = self.images[key]
        return batch


def _check_diffusion_inputs(base_policy) -> None:
    if "observation.state" not in base_policy.config.input_shapes:
        raise ValueError("Frozen Diffusion checkpoint must contain policy.input_shapes.observation.state.")
    for image_key in _image_keys_from_policy(base_policy):
        shape = tuple(int(dim) for dim in base_policy.config.input_shapes[image_key])
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(
                f"mjlab image rendering expects {image_key} shape [3, H, W], got {shape}."
            )


def _disable_lerobot_image_debug_saves(base_policy) -> int:
    disabled = 0
    modules = base_policy.modules() if isinstance(base_policy, nn.Module) else []
    for module in modules:
        if all(hasattr(module, attr) for attr in ("do_resize", "do_crop", "backbone")):
            module._debug_img_counter = max(int(getattr(module, "_debug_img_counter", 0)), 3)
            disabled += 1
    return disabled


def _make_env(num_envs: int, episode_length_s: float, device: str) -> ManagerBasedRlEnv:
    env_cfg = make_insert_cylinder_env_cfg(num_envs=num_envs)
    env_cfg.episode_length_s = float(episode_length_s)
    return ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)


def _open_rollout_arrays(
    *,
    use_disk_cache: bool,
    root: Path,
    keep_iterations: bool,
    iteration: int,
    shape_prefix: tuple[int, int, int],
    residual_obs_dim: int,
    action_dim: int,
):
    n_steps, n_envs, act_steps = shape_prefix
    if not use_disk_cache:
        return None, {
            "residual_obs": np.zeros((*shape_prefix, residual_obs_dim), dtype=np.float32),
            "residual_action": np.zeros((*shape_prefix, action_dim), dtype=np.float32),
            "old_logprob": np.zeros(shape_prefix, dtype=np.float32),
            "reward": np.zeros(shape_prefix, dtype=np.float32),
            "success": np.zeros(shape_prefix, dtype=np.float32),
            "terminated": np.zeros(shape_prefix, dtype=np.float32),
            "first": np.zeros(shape_prefix, dtype=np.float32),
            "valid": np.zeros(shape_prefix, dtype=bool),
        }

    buffer_path = root / (f"iteration_{iteration:06d}" if keep_iterations else "current")
    if buffer_path.exists():
        shutil.rmtree(buffer_path)
    buffer_path.mkdir(parents=True, exist_ok=True)
    arrays = {
        "residual_obs": np.lib.format.open_memmap(
            buffer_path / "residual_obs_trajs.npy",
            dtype=np.float32,
            mode="w+",
            shape=(*shape_prefix, residual_obs_dim),
        ),
        "residual_action": np.lib.format.open_memmap(
            buffer_path / "residual_action_trajs.npy",
            dtype=np.float32,
            mode="w+",
            shape=(*shape_prefix, action_dim),
        ),
        "old_logprob": np.lib.format.open_memmap(
            buffer_path / "old_logprob_trajs.npy",
            dtype=np.float32,
            mode="w+",
            shape=shape_prefix,
        ),
        "reward": np.lib.format.open_memmap(
            buffer_path / "reward_trajs.npy",
            dtype=np.float32,
            mode="w+",
            shape=shape_prefix,
        ),
        "success": np.lib.format.open_memmap(
            buffer_path / "success_trajs.npy",
            dtype=np.float32,
            mode="w+",
            shape=shape_prefix,
        ),
        "terminated": np.lib.format.open_memmap(
            buffer_path / "terminated_trajs.npy",
            dtype=np.float32,
            mode="w+",
            shape=shape_prefix,
        ),
        "first": np.lib.format.open_memmap(
            buffer_path / "firsts_trajs.npy",
            dtype=np.float32,
            mode="w+",
            shape=shape_prefix,
        ),
        "valid": np.lib.format.open_memmap(
            buffer_path / "valid_step_trajs.npy",
            dtype=bool,
            mode="w+",
            shape=shape_prefix,
        ),
    }
    _write_yaml(
        buffer_path / "metadata.yaml",
        {
            "iteration": int(iteration),
            "n_steps": int(n_steps),
            "n_envs": int(n_envs),
            "act_steps": int(act_steps),
            "action_dim": int(action_dim),
            "residual_obs_dim": int(residual_obs_dim),
        },
    )
    return buffer_path, arrays


def _flush_arrays(arrays: dict[str, np.ndarray]) -> None:
    for array in arrays.values():
        if hasattr(array, "flush"):
            array.flush()


def _save_checkpoint(
    *,
    policy: FrozenDiffusionResidualPolicy,
    log_dir: Path,
    raw_cfg: dict[str, Any],
    iteration: int,
    metrics: dict[str, Any],
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    actor_scheduler: Any | None = None,
    critic_scheduler: Any | None = None,
    global_step: int = 0,
    training_cum_time: float = 0.0,
    best_eval_success_rate: float | None = None,
    name: str | None = None,
) -> Path:
    if name is None:
        name = (
            f"{iteration:06d}_sr={metrics['rollout_success_rate']:.2f}"
            f"_return={metrics['rollout_avg_return']:.2f}"
        )
    ckpt_dir = log_dir / "checkpoints" / name
    save_path = ckpt_dir / "pretrained_model"
    policy.save_pretrained(save_path)
    _write_yaml(save_path / "finetune_mjlab_config.yaml", raw_cfg)
    with (ckpt_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    training_state = {
        "model_state_dict": policy.state_dict(),
        "residual_policy_state_dict": policy.residual_policy.state_dict(),
        "optimizer_actor_state_dict": (
            None if actor_optimizer is None else actor_optimizer.state_dict()
        ),
        "optimizer_critic_state_dict": (
            None if critic_optimizer is None else critic_optimizer.state_dict()
        ),
        "scheduler_actor_state_dict": None if actor_scheduler is None else actor_scheduler.state_dict(),
        "scheduler_critic_state_dict": None if critic_scheduler is None else critic_scheduler.state_dict(),
        "config": raw_cfg,
        "metrics": metrics,
        "success_rate": metrics.get("rollout_success_rate", 0.0),
        "best_eval_success_rate": best_eval_success_rate,
        "iteration": int(iteration),
        "global_step": int(global_step),
        "training_cum_time": float(training_cum_time),
    }
    torch.save(training_state, ckpt_dir / "training_state.pt")
    return save_path


def _resolve_training_state_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidate = path / "training_state.pt"
        if candidate.exists():
            return candidate
        candidate = path.parent / "training_state.pt"
        if candidate.exists():
            return candidate
        candidate = path / "residual_policy.pt"
        if candidate.exists():
            return candidate
    return path


def _load_training_state(
    path: str | Path,
    *,
    policy: FrozenDiffusionResidualPolicy,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    actor_scheduler: Any | None = None,
    critic_scheduler: Any | None = None,
    device: torch.device,
) -> dict[str, Any]:
    state_path = _resolve_training_state_path(path)
    if not state_path.exists():
        raise FileNotFoundError(f"Cannot find resume checkpoint: {state_path}")

    state = torch.load(state_path, map_location=device)
    if "model_state_dict" in state and state["model_state_dict"] is not None:
        policy.load_state_dict(state["model_state_dict"], strict=False)
    elif "residual_policy_state_dict" in state and state["residual_policy_state_dict"] is not None:
        policy.residual_policy.load_state_dict(state["residual_policy_state_dict"], strict=True)
    elif "model" in state:
        policy.residual_policy.load_state_dict(state["model"], strict=True)
    else:
        raise KeyError(f"Checkpoint {state_path} does not contain residual policy weights.")

    if actor_optimizer is not None and state.get("optimizer_actor_state_dict") is not None:
        actor_optimizer.load_state_dict(state["optimizer_actor_state_dict"])
    if critic_optimizer is not None and state.get("optimizer_critic_state_dict") is not None:
        critic_optimizer.load_state_dict(state["optimizer_critic_state_dict"])
    if actor_scheduler is not None and state.get("scheduler_actor_state_dict") is not None:
        actor_scheduler.load_state_dict(state["scheduler_actor_state_dict"])
    if critic_scheduler is not None and state.get("scheduler_critic_state_dict") is not None:
        critic_scheduler.load_state_dict(state["scheduler_critic_state_dict"])

    logging.info(f"Resume  | loaded residual PPO state from {state_path}")
    return state


def _make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    warmup_steps: int,
    total_steps: int,
):
    name = str(name).lower()
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    try:
        from diffusers.optimization import get_scheduler

        return get_scheduler(
            name=name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    except Exception:
        from torch.optim.lr_scheduler import LambdaLR, LinearLR

        if name in {"constant", "constant_with_warmup"} or warmup_steps == 0:
            return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        return LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )


def train_residual_mjlab(raw_cfg: dict[str, Any], args: argparse.Namespace) -> None:
    init_logging()

    seed = int(args.seed if args.seed is not None else _get(raw_cfg, "seed", 1000))
    set_global_seed(seed)

    device_str = _resolve_device(raw_cfg, args)
    device = get_safe_torch_device(device_str, log=True)
    num_envs = int(args.num_envs if args.num_envs is not None else _get(raw_cfg, "env.num_envs", 128))
    episode_length_s = float(_get(raw_cfg, "env.episode_length_s", 16.0))
    state_mode = str(args.state_mode or _get(raw_cfg, "env.state_mode", "auto"))
    base_action_mode = str(args.base_action_mode or _get(raw_cfg, "env.base_action_mode", "legacy_absolute"))
    raw_env_action_clip = args.env_action_clip if args.env_action_clip is not None else _get(
        raw_cfg, "env.env_action_clip", None
    )
    env_action_clip = None if raw_env_action_clip in {None, "null", "none"} else float(raw_env_action_clip)

    pretrained_ckpt_path = args.pretrained_ckpt_path or _get(raw_cfg, "training.pretrained_ckpt_path")
    if not pretrained_ckpt_path:
        raise ValueError(
            "Please set training.pretrained_ckpt_path in the config or pass --pretrained-ckpt-path."
        )

    policy_cfg = OmegaConf.create(
        {
            "seed": seed,
            "device": str(device),
            "training": {"pretrained_ckpt_path": str(pretrained_ckpt_path)},
            "policy": {
                "n_action_steps": int(args.act_steps or _get(raw_cfg, "policy.n_action_steps", 8))
            },
        }
    )
    base_policy, hydra_cfg, load_dir = load_frozen_base_policy(policy_cfg, device)
    n_debug_savers = _disable_lerobot_image_debug_saves(base_policy)
    if n_debug_savers > 0:
        logging.debug("Disabled LeRobot image debug saves on %d module(s).", n_debug_savers)
    _check_diffusion_inputs(base_policy)
    image_keys = _image_keys_from_policy(base_policy)
    image_shapes = {
        key: tuple(int(dim) for dim in base_policy.config.input_shapes[key])
        for key in image_keys
    }
    camera_map = _camera_map_from_cfg(_get(raw_cfg, "env.camera_map", {}), image_keys)

    action_dim = int(base_policy.config.output_shapes["action"][0])
    if action_dim != ACTION_DIM:
        raise ValueError(
            f"Frozen Diffusion action_dim={action_dim}, but mjlab PiperX action_dim={ACTION_DIM}."
        )
    horizon_steps = int(base_policy.config.horizon)
    n_obs_steps = int(getattr(base_policy.config, "n_obs_steps", 2))
    act_steps = int(getattr(policy_cfg.policy, "n_action_steps", getattr(base_policy.config, "n_action_steps", 8)))
    action_start = n_obs_steps - 1
    action_end = action_start + act_steps
    if action_end > horizon_steps:
        raise ValueError(
            f"Action slice [{action_start}:{action_end}] exceeds Diffusion horizon={horizon_steps}."
        )

    expected_state_shape = base_policy.config.input_shapes["observation.state"]
    expected_state_dim = int(np.prod([int(dim) for dim in expected_state_shape]))
    with maybe_suppress_stdout(_as_bool(_get(raw_cfg, "training.quiet_terminal", True), True)):
        global_cond_dim = infer_global_cond_dim(base_policy, device)

    env = _make_env(num_envs=num_envs, episode_length_s=episode_length_s, device=str(device))
    if env.action_manager.total_action_dim != action_dim:
        env.close()
        raise ValueError(
            f"mjlab env action_dim={env.action_manager.total_action_dim}, Diffusion action_dim={action_dim}."
        )

    adapter = MjlabDiffusionAdapter(
        expected_state_dim=expected_state_dim,
        action_dim=action_dim,
        state_mode=state_mode,
        base_action_mode=base_action_mode,
        env_action_clip=env_action_clip,
        device=device,
    )
    camera_renderer = (
        MjlabCameraRenderer(
            env=env,
            image_shapes=image_shapes,
            camera_map=camera_map,
            device=device,
        )
        if image_keys
        else None
    )
    history = PolicyObsHistory(
        adapter=adapter,
        camera_renderer=camera_renderer,
        image_keys=image_keys,
        num_envs=num_envs,
        n_obs_steps=n_obs_steps,
        device=device,
    )

    residual_action_space = str(_get(raw_cfg, "training.residual_action_space", "normalized")).lower()
    if residual_action_space != "normalized":
        raise ValueError(
            "robust-rearrangement residual PPO trains residuals in normalized action space. "
            "Set training.residual_action_space: normalized."
        )

    residual_hidden_dim = int(_get(raw_cfg, "training.residual_hidden_dim", 256))
    residual_depth = int(_get(raw_cfg, "training.residual_depth", 2))
    critic_hidden_dim = int(_get(raw_cfg, "training.residual_critic_hidden_dim", residual_hidden_dim))
    critic_depth = int(_get(raw_cfg, "training.residual_critic_depth", residual_depth))
    raw_residual_std = _get(raw_cfg, "training.residual_std", None)
    if raw_residual_std is None:
        residual_std = math.exp(float(_get(raw_cfg, "training.residual_init_logstd", -1.0)))
    else:
        residual_std = float(raw_residual_std)
    residual_learn_std = _as_bool(_get(raw_cfg, "training.residual_learn_std", False), False)
    residual_action_scale = float(_get(raw_cfg, "training.residual_action_scale", 0.1))
    residual_max_delta = float(_get(raw_cfg, "training.residual_max_delta", 0.0))
    residual_activation = str(_get(raw_cfg, "training.residual_activation", "ReLU"))
    action_head_std = float(_get(raw_cfg, "training.residual_action_head_std", 0.0))
    stepwise_obs = _as_bool(_get(raw_cfg, "training.residual_stepwise_obs", True), True)
    logprob_reduction = str(_get(raw_cfg, "training.logprob_reduction", "sum"))

    policy = FrozenDiffusionResidualPolicy(
        base_policy=base_policy,
        action_dim=action_dim,
        action_start=action_start,
        action_end=action_end,
        global_cond_dim=global_cond_dim,
        actor_hidden_dim=residual_hidden_dim,
        actor_depth=residual_depth,
        critic_hidden_dim=critic_hidden_dim,
        critic_depth=critic_depth,
        activation=residual_activation,
        init_std=residual_std,
        learn_std=residual_learn_std,
        action_head_std=action_head_std,
        action_scale=residual_action_scale,
        max_delta=residual_max_delta,
        logprob_reduction=logprob_reduction,
        stepwise_obs=stepwise_obs,
    ).to(device)

    n_train_itr = int(args.n_train_itr or args.max_iterations or _get(raw_cfg, "training.n_train_itr", 1000))

    actor_optimizer = torch.optim.AdamW(
        policy.actor_parameters,
        lr=float(_get(raw_cfg, "training.actor_lr", 3.0e-4)),
        betas=tuple(_get(raw_cfg, "training.optimizer_betas_actor", (0.9, 0.999))),
        eps=1e-5,
        weight_decay=float(_get(raw_cfg, "training.weight_decay", 1e-6)),
    )
    critic_optimizer = torch.optim.AdamW(
        policy.critic_parameters,
        lr=float(_get(raw_cfg, "training.critic_lr", 5.0e-3)),
        eps=1e-5,
        weight_decay=float(_get(raw_cfg, "training.critic_weight_decay", 1e-6)),
    )
    scheduler_name = str(_get(raw_cfg, "training.lr_scheduler", "cosine"))
    actor_scheduler = _make_lr_scheduler(
        actor_optimizer,
        name=scheduler_name,
        warmup_steps=int(_get(raw_cfg, "training.actor_lr_warmup_iters", 5)),
        total_steps=n_train_itr,
    )
    critic_scheduler = _make_lr_scheduler(
        critic_optimizer,
        name=scheduler_name,
        warmup_steps=int(_get(raw_cfg, "training.critic_lr_warmup_iters", 0)),
        total_steps=n_train_itr,
    )

    run_name = args.run_name or str(_get(raw_cfg, "training.run_name", "residual_ppo"))
    experiment_name = str(_get(raw_cfg, "experiment_name", "piperx_insert_cylinder_residual_mjlab"))
    log_root = Path(args.log_root or str(raw_cfg.get("log_root", "logs/rsl_rl")))
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        log_dir_name += f"_{run_name}"
    log_dir = (log_root / experiment_name / log_dir_name).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(log_dir / "params" / "config.yaml", raw_cfg)
    _write_yaml(log_dir / "params" / "base_policy_config.yaml", OmegaConf.to_container(hydra_cfg, resolve=True))
    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter is not None else None

    critic_warmup_iters = int(_get(raw_cfg, "training.n_critic_warmup_itr", 0))
    rollout_steps = int(args.rollout_steps or _get(raw_cfg, "training.rollout_steps", 32))
    num_minibatches = max(1, int(_get(raw_cfg, "training.num_minibatches", 1)))
    raw_minibatch_size = args.batch_size or _get(raw_cfg, "training.minibatch_size", None)
    if raw_minibatch_size in {None, "null", "none", ""}:
        batch_size = max(1, math.ceil((rollout_steps * num_envs * act_steps) / num_minibatches))
    else:
        batch_size = int(raw_minibatch_size)
    update_epochs = int(_get(raw_cfg, "training.update_epochs", 50))
    clip_ratio = float(_get(raw_cfg, "training.clip_ratio", 0.2))
    target_kl = float(_get(raw_cfg, "training.target_kl", 0.1))
    gamma = float(_get(raw_cfg, "training.gamma", 0.999))
    gae_lambda = float(_get(raw_cfg, "training.gae_lambda", 0.95))
    vf_coef = float(_get(raw_cfg, "training.vf_coef", 1.0))
    ent_coef = float(_get(raw_cfg, "training.ent_coef", 0.0))
    residual_l1 = float(_get(raw_cfg, "training.residual_l1", 0.0))
    residual_l2 = float(_get(raw_cfg, "training.residual_l2", 0.0))
    reward_scale_running = _as_bool(_get(raw_cfg, "training.reward_scale_running", True), True)
    reward_scale_const = float(_get(raw_cfg, "training.reward_scale_const", 1.0))
    reward_clip = float(_get(raw_cfg, "training.reward_clip", 5.0))
    reward_source = str(_get(raw_cfg, "training.reward_source", "env"))
    success_metric_source = str(_get(raw_cfg, "training.success_metric_source", "success_count")).lower()
    success_reward_count_threshold = int(_get(raw_cfg, "training.success_reward_count_threshold", 1))
    norm_adv = _as_bool(_get(raw_cfg, "training.norm_adv", True), True)
    adv_lower_q = float(_get(raw_cfg, "training.clip_advantage_lower_quantile", 0.0))
    adv_upper_q = float(_get(raw_cfg, "training.clip_advantage_upper_quantile", 1.0))
    clip_vloss = _as_bool(_get(raw_cfg, "training.clip_vloss", False), False)
    grad_accum_steps = max(1, int(_get(raw_cfg, "training.grad_accumulate", 1)))
    max_grad_norm = float(_get(raw_cfg, "training.max_grad_norm", 1.0))
    skip_update = bool(args.skip_update or _as_bool(_get(raw_cfg, "training.skip_update", False), False))
    update_actor = _as_bool(_get(raw_cfg, "training.update_actor", True), True)
    bootstrap_on_truncation = _as_bool(_get(raw_cfg, "training.bootstrap_on_truncation", False), False)
    save_interval = int(args.save_interval or _get(raw_cfg, "training.save_interval", 20))
    eval_interval = int(args.eval_interval if args.eval_interval is not None else _get(raw_cfg, "training.eval_interval", 5))
    eval_first = _as_bool(_get(raw_cfg, "training.eval_first", True), True)
    reset_every_iteration = _as_bool(_get(raw_cfg, "training.reset_every_iteration", True), True)
    resume_path = args.resume_path or _get(raw_cfg, "training.resume_path", None)
    use_disk_cache = _as_bool(_get(raw_cfg, "training.use_disk_cache", False), False)
    buffer_root = Path(str(_get(raw_cfg, "training.rollout_buffer_dir", "outputs/buffer/finetune_residual_mjlab")))
    if not buffer_root.is_absolute():
        buffer_root = ROOT_DIR / buffer_root
    buffer_keep_iterations = _as_bool(_get(raw_cfg, "training.rollout_buffer_keep_iterations", False), False)
    rollout_progress = _as_bool(_get(raw_cfg, "training.rollout_progress", True), True)
    training_progress = _as_bool(_get(raw_cfg, "training.training_progress", True), True)
    quiet_terminal = _as_bool(_get(raw_cfg, "training.quiet_terminal", True), True)

    start_itr = 0
    global_step = 0
    training_cum_time = 0.0
    best_eval_success_rate = 0.0
    if resume_path not in {None, "null", "none", ""}:
        resume_state = _load_training_state(
            resume_path,
            policy=policy,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_scheduler=actor_scheduler,
            critic_scheduler=critic_scheduler,
            device=device,
        )
        start_itr = int(resume_state.get("iteration", 0))
        global_step = int(resume_state.get("global_step", 0))
        training_cum_time = float(resume_state.get("training_cum_time", 0.0))
        best_eval_success_rate = float(
            resume_state.get(
                "best_eval_success_rate",
                resume_state.get("success_rate", 0.0),
            )
            or 0.0
        )

    obs, _ = env.reset(seed=seed)
    history.reset(obs)

    log_box(
        "Mjlab Frozen Diffusion + Residual PPO",
        [
            ("task", TASK_ID),
            ("log_dir", log_dir),
            ("device", device),
            ("num_envs", num_envs),
            ("pretrained", load_dir),
            ("state_mode", adapter.resolved_state_mode),
            ("base_action_mode", base_action_mode),
            ("residual_action_space", residual_action_space),
            ("image_keys", image_keys if image_keys else "none"),
            ("camera_map", camera_map if camera_map else "none"),
            ("env_action_clip", env_action_clip),
            ("action_slice", f"[{action_start}:{action_end}]"),
            ("act_steps", act_steps),
            ("rollout_control_steps", rollout_steps * num_envs * act_steps),
            ("minibatch_size", batch_size),
            ("eval_interval", eval_interval),
            ("reward_source", reward_source),
            ("success_metric_source", success_metric_source),
        ],
    )
    log_box(
        "Model Params",
        [
            ("frozen_diffusion", f"{sum(p.numel() for p in base_policy.parameters()) / 1e6:.2f}M"),
            ("residual_actor_critic", f"{sum(p.numel() for p in policy.adapter_parameters()) / 1e6:.2f}M"),
            ("global_cond_dim", global_cond_dim),
            ("residual_obs_dim", policy.residual_obs_dim),
            ("action_std", f"{float(policy.action_std.mean().detach().cpu().item()):.5f}"),
        ],
    )

    running_ep_train_rewards = np.zeros(num_envs, dtype=np.float32)
    running_ep_env_rewards = np.zeros(num_envs, dtype=np.float32)
    running_ep_steps = np.zeros(num_envs, dtype=np.int32)
    running_reward_scaler = RunningRewardScaler(num_envs, cliprew=reward_clip, gamma=gamma)
    chunk_start_firsts = np.ones(num_envs, dtype=np.float32)
    metrics_path = log_dir / "metrics.jsonl"

    try:
        for itr in range(start_itr, n_train_itr):
            iteration = itr + 1
            iteration_start_time = time.time()
            eval_mode = eval_interval > 0 and ((iteration - int(eval_first)) % eval_interval == 0)
            skip_update_this_itr = bool(skip_update or eval_mode)
            actor_update_planned = bool((not eval_mode) and update_actor and iteration > critic_warmup_iters)

            if eval_mode or reset_every_iteration:
                obs, _ = env.reset(seed=seed + iteration)
                history.reset(obs)
                policy.reset()
                running_ep_train_rewards.fill(0.0)
                running_ep_env_rewards.fill(0.0)
                running_ep_steps.fill(0)
                chunk_start_firsts = np.ones(num_envs, dtype=np.float32)

            shape_prefix = (rollout_steps, num_envs, act_steps)
            buffer_path, arrays = _open_rollout_arrays(
                use_disk_cache=use_disk_cache,
                root=buffer_root,
                keep_iterations=buffer_keep_iterations,
                iteration=iteration,
                shape_prefix=shape_prefix,
                residual_obs_dim=policy.residual_obs_dim,
                action_dim=action_dim,
            )
            if buffer_path is not None:
                logging.info(f"Rollout buffer | writing to {buffer_path}")

            completed_ep_train_rewards: list[float] = []
            completed_ep_env_rewards: list[float] = []
            completed_ep_successes: list[bool] = []
            completed_ep_steps: list[int] = []

            policy.eval()
            rollout_bar = tqdm(
                range(rollout_steps),
                desc=f"Mjlab Residual {'Eval' if eval_mode else 'Rollout'} {iteration:04d}",
                leave=rollout_progress,
                dynamic_ncols=True,
                mininterval=1.0,
                disable=not rollout_progress,
            )
            for step in rollout_bar:
                with torch.no_grad():
                    with maybe_suppress_stdout(quiet_terminal):
                        base_actions_t, initial_global_cond_t = policy.frozen_diffusion_actions(
                            history.batch(),
                            return_global_cond=True,
                        )

                chunk_env_reward = np.zeros(num_envs, dtype=np.float32)
                any_done_accum = torch.zeros(num_envs, device=device, dtype=torch.bool)
                success_accum = torch.zeros(num_envs, device=device, dtype=torch.bool)
                safe_env_action = torch.zeros((num_envs, action_dim), dtype=torch.float32, device=device)
                step_reward = np.zeros((num_envs, act_steps), dtype=np.float32)
                step_success = np.zeros((num_envs, act_steps), dtype=np.float32)
                step_done = np.zeros((num_envs, act_steps), dtype=np.float32)
                step_first = np.zeros((num_envs, act_steps), dtype=np.float32)
                step_valid = np.zeros((num_envs, act_steps), dtype=bool)
                current_firsts = chunk_start_firsts.copy()

                for step_i in range(act_steps):
                    if not eval_mode:
                        global_step += num_envs
                    with torch.no_grad():
                        if step_i == 0 or not stepwise_obs:
                            global_cond_step_t = initial_global_cond_t
                        else:
                            obs_norm = policy.normalize_history_batch(history.batch())
                            global_cond_step_t = policy.base_policy.diffusion._prepare_global_conditioning(obs_norm)

                        base_step_t = base_actions_t[:, action_start + step_i, :]
                        samples = policy.step_action(
                            base_action=base_step_t,
                            global_cond=global_cond_step_t,
                            deterministic=eval_mode,
                        )
                        env_action = adapter.action_to_env(samples["action"])
                        active_mask = ~any_done_accum
                        env_action = torch.where(active_mask[:, None], env_action, safe_env_action)

                        arrays["residual_obs"][step, :, step_i, :] = (
                            samples["residual_obs"].detach().cpu().numpy()
                        )
                        arrays["residual_action"][step, :, step_i, :] = (
                            samples["residual_action"].detach().cpu().numpy()
                        )
                        arrays["old_logprob"][step, :, step_i] = (
                            samples["logprob"].detach().cpu().numpy()
                        )

                    obs, reward, terminated, truncated, _ = env.step(env_action)
                    done = terminated | truncated
                    try:
                        success_mask = env.termination_manager.get_term("success").clone() & done
                    except Exception:
                        success_mask = terminated.clone() & done
                    success_accum |= success_mask & active_mask

                    active_np = active_mask.detach().cpu().numpy().astype(bool)
                    reward_np = reward.detach().cpu().numpy().astype(np.float32)
                    chunk_env_reward += reward_np * active_np
                    running_ep_steps[active_np] += 1

                    if reward_source in {"env", "dense", "mjlab"}:
                        train_reward_t = reward
                    elif reward_source in {"binary_success_label", "success", "success_label"}:
                        train_reward_t = success_mask.float()
                    else:
                        raise ValueError(
                            "training.reward_source must be 'binary_success_label' or 'env'."
                        )
                    train_reward_np = train_reward_t.detach().cpu().numpy().astype(np.float32) * active_np
                    done_for_gae = (terminated if bootstrap_on_truncation else done) & active_mask

                    step_reward[:, step_i] = train_reward_np
                    step_success[:, step_i] = (
                        success_mask.detach().cpu().numpy().astype(np.float32) * active_np
                    )
                    step_done[:, step_i] = done_for_gae.detach().cpu().numpy().astype(np.float32)
                    step_first[:, step_i] = current_firsts
                    step_valid[:, step_i] = active_np
                    current_firsts = done.detach().cpu().numpy().astype(np.float32)

                    history.append(obs, done)
                    any_done_accum |= done

                invalid_mask = ~step_valid
                step_first[invalid_mask] = 1.0
                step_done[invalid_mask] = 1.0
                step_success[invalid_mask] = 0.0
                arrays["reward"][step] = step_reward
                arrays["success"][step] = step_success
                arrays["terminated"][step] = step_done
                arrays["first"][step] = step_first
                arrays["valid"][step] = step_valid

                chunk_start_firsts = any_done_accum.detach().cpu().numpy().astype(np.float32)
                chunk_reward = step_reward.sum(axis=1)
                running_ep_train_rewards += chunk_reward
                running_ep_env_rewards += chunk_env_reward
                done_np = any_done_accum.detach().cpu().numpy().astype(bool)
                success_np = success_accum.detach().cpu().numpy().astype(bool)
                for env_idx in np.flatnonzero(done_np):
                    completed_ep_train_rewards.append(float(running_ep_train_rewards[env_idx]))
                    completed_ep_env_rewards.append(float(running_ep_env_rewards[env_idx]))
                    completed_ep_successes.append(bool(success_np[env_idx]))
                    completed_ep_steps.append(int(running_ep_steps[env_idx]))
                    running_ep_train_rewards[env_idx] = 0.0
                    running_ep_env_rewards[env_idx] = 0.0
                    running_ep_steps[env_idx] = 0

                if rollout_progress:
                    rollout_bar.set_postfix(
                        episodes=len(completed_ep_successes),
                        success=fmt_pct(
                            float(np.mean(completed_ep_successes))
                            if completed_ep_successes
                            else 0.0
                        ),
                    )

            _flush_arrays(arrays)
            rollout_avg_return = (
                float(np.mean(completed_ep_train_rewards)) if completed_ep_train_rewards else 0.0
            )
            rollout_avg_env_return = (
                float(np.mean(completed_ep_env_rewards)) if completed_ep_env_rewards else 0.0
            )
            rollout_avg_ep_steps = (
                float(np.mean(completed_ep_steps)) if completed_ep_steps else float("nan")
            )
            success_timesteps_share = 0.0
            mean_success_episode_length = 0.0
            max_success_episode_length = 0.0
            if success_metric_source in {"reward_count", "robust", "success_count", "success"}:
                success_time = arrays["success"].transpose(0, 2, 1).reshape(
                    rollout_steps * act_steps,
                    num_envs,
                )
                env_success = (
                    (success_time > 0).sum(axis=0) >= success_reward_count_threshold
                )
                rollout_success_rate = float(env_success.mean())
                if np.any(env_success):
                    success_reward_counts = np.cumsum(success_time[:, env_success] > 0, axis=0)
                    success_reached = success_reward_counts >= success_reward_count_threshold
                    last_reward_idx = np.argmax(success_reached, axis=0)
                    total_timesteps_in_success = int((last_reward_idx + 1).sum())
                    success_timesteps_share = float(total_timesteps_in_success / success_time.size)
                    mean_success_episode_length = float(total_timesteps_in_success / env_success.sum())
                    max_success_episode_length = float(last_reward_idx.max() + 1)
            else:
                rollout_success_rate = (
                    float(np.mean(completed_ep_successes)) if completed_ep_successes else 0.0
                )

            if skip_update_this_itr:
                avg_pg_loss = avg_v_loss = avg_entropy = avg_kl = avg_clipfrac = 0.0
                critic_ev = critic_corr = float("nan")
                total_samples = 0
            else:
                total_raw_samples = rollout_steps * num_envs * act_steps
                residual_obs_flat = arrays["residual_obs"].reshape(total_raw_samples, policy.residual_obs_dim)
                residual_action_flat = arrays["residual_action"].reshape(total_raw_samples, action_dim)
                old_logprobs_flat = arrays["old_logprob"].reshape(total_raw_samples)
                valid_flat = arrays["valid"].reshape(total_raw_samples)
                valid_indices_np = np.flatnonzero(valid_flat)
                if valid_indices_np.size == 0:
                    raise RuntimeError("No valid mjlab residual rollout samples were collected.")

                with torch.no_grad():
                    values_all_flat = np.zeros(total_raw_samples, dtype=np.float32)
                    val_batch_size = max(1, batch_size * 4)
                    for i in range(0, total_raw_samples, val_batch_size):
                        end_i = min(i + val_batch_size, total_raw_samples)
                        residual_obs_b = torch.from_numpy(
                            np.ascontiguousarray(residual_obs_flat[i:end_i])
                        ).float().to(device)
                        values_all_flat[i:end_i] = (
                            policy.residual_policy.get_value(residual_obs_b).cpu().numpy().flatten()
                        )

                    values_trajs = values_all_flat.reshape(rollout_steps, num_envs, act_steps)
                    values_time = values_trajs.transpose(0, 2, 1).reshape(
                        rollout_steps * act_steps, num_envs
                    )
                    with maybe_suppress_stdout(quiet_terminal):
                        last_base_actions_t, last_global_cond_t = policy.frozen_diffusion_actions(
                            history.batch(),
                            return_global_cond=True,
                        )
                    last_base_step_t = last_base_actions_t[:, action_start, :]
                    next_values_last = (
                        policy.value_for_step(last_base_step_t, last_global_cond_t)
                        .cpu()
                        .numpy()
                        .flatten()
                    )

                rewards_time = arrays["reward"].transpose(0, 2, 1).reshape(
                    rollout_steps * act_steps, num_envs
                )
                terminated_time = arrays["terminated"].transpose(0, 2, 1).reshape(
                    rollout_steps * act_steps, num_envs
                )
                firsts_time = arrays["first"].transpose(0, 2, 1).reshape(
                    rollout_steps * act_steps, num_envs
                )
                if reward_scale_running:
                    scaled_rewards_time = running_reward_scaler(
                        reward=rewards_time.T,
                        first=firsts_time.T,
                    ).T
                else:
                    scaled_rewards_time = rewards_time

                advantages_time = np.zeros_like(scaled_rewards_time)
                last_gae_lam = np.zeros(num_envs, dtype=np.float32)
                for t in reversed(range(rollout_steps * act_steps)):
                    next_val = next_values_last if t == rollout_steps * act_steps - 1 else values_time[t + 1]
                    nonterminal = 1.0 - terminated_time[t]
                    delta = (
                        scaled_rewards_time[t] * reward_scale_const
                        + gamma * next_val * nonterminal
                        - values_time[t]
                    )
                    last_gae_lam = delta + gamma * gae_lambda * nonterminal * last_gae_lam
                    advantages_time[t] = last_gae_lam

                returns_time = advantages_time + values_time
                returns_trajs = returns_time.reshape(rollout_steps, act_steps, num_envs).transpose(0, 2, 1)
                advantages_trajs = advantages_time.reshape(rollout_steps, act_steps, num_envs).transpose(0, 2, 1)
                returns_flat = returns_trajs.reshape(total_raw_samples)
                advantages_flat = advantages_trajs.reshape(total_raw_samples)
                critic_ev, critic_corr = compute_value_diagnostics(
                    values_all_flat[valid_indices_np],
                    returns_flat[valid_indices_np],
                )

                residual_obs_flat = residual_obs_flat[valid_indices_np]
                residual_action_flat = residual_action_flat[valid_indices_np]
                old_logprobs_k = torch.from_numpy(old_logprobs_flat[valid_indices_np]).float().to(device)
                values_k = torch.from_numpy(values_all_flat[valid_indices_np]).float().to(device)
                returns_k = torch.from_numpy(returns_flat[valid_indices_np]).float().to(device)
                advantages_k = torch.from_numpy(advantages_flat[valid_indices_np]).float().to(device)
                total_samples = int(valid_indices_np.size)

                avg_pg_loss = 0.0
                avg_v_loss = 0.0
                avg_entropy = 0.0
                avg_kl = 0.0
                avg_clipfrac = 0.0
                loss_count = 0
                indices = np.arange(total_samples)
                num_batches = max(1, math.ceil(total_samples / batch_size))
                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                ppo_bar = tqdm(
                    range(update_epochs),
                    desc=f"Mjlab Residual PPO {iteration:04d}",
                    leave=training_progress,
                    dynamic_ncols=True,
                    mininterval=1.0,
                    disable=not training_progress,
                )
                stop_update = False
                for epoch in ppo_bar:
                    np.random.shuffle(indices)
                    accum_counter = 0
                    for batch_idx in range(num_batches):
                        start = batch_idx * batch_size
                        end = min(start + batch_size, total_samples)
                        inds_np = indices[start:end]
                        mb_obs = torch.from_numpy(
                            np.ascontiguousarray(residual_obs_flat[inds_np])
                        ).float().to(device)
                        mb_actions = torch.from_numpy(
                            np.ascontiguousarray(residual_action_flat[inds_np])
                        ).float().to(device)
                        mb_old_logprobs = old_logprobs_k[inds_np]
                        mb_advantages = advantages_k[inds_np]
                        mb_returns = returns_k[inds_np]
                        mb_values = values_k[inds_np]

                        _, newlogprob, entropy, newvalue, action_mean = (
                            policy.residual_policy.get_action_and_value(
                                mb_obs,
                                residual_action=mb_actions,
                            )
                        )
                        logratio = newlogprob - mb_old_logprobs
                        ratio = logratio.exp()
                        with torch.no_grad():
                            approx_kl = ((ratio - 1.0) - logratio).mean()
                            clipfrac = ((ratio - 1.0).abs() > clip_ratio).float().mean()

                        mb_advantages = normalize_clip_minibatch_advantage(
                            mb_advantages,
                            norm_adv=norm_adv,
                            lower_q=adv_lower_q,
                            upper_q=adv_upper_q,
                        )
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio,
                            1.0 - clip_ratio,
                            1.0 + clip_ratio,
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        newvalue = newvalue.view(-1)
                        if clip_vloss:
                            v_loss_unclipped = (newvalue - mb_returns).square()
                            v_clipped = mb_values + torch.clamp(
                                newvalue - mb_values,
                                -clip_ratio,
                                clip_ratio,
                            )
                            v_loss_clipped = (v_clipped - mb_returns).square()
                            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        else:
                            v_loss = 0.5 * (newvalue - mb_returns).square().mean()
                        entropy_loss = entropy.mean()
                        residual_l1_loss = torch.abs(action_mean).mean()
                        residual_l2_loss = action_mean.square().mean()
                        if actor_update_planned:
                            actor_loss = (
                                pg_loss
                                - ent_coef * entropy_loss
                                + residual_l1 * residual_l1_loss
                                + residual_l2 * residual_l2_loss
                            )
                        else:
                            actor_loss = torch.zeros((), device=device)
                        loss = actor_loss + vf_coef * v_loss
                        (loss / grad_accum_steps).backward()
                        accum_counter += 1
                        is_last_batch = batch_idx == num_batches - 1
                        if accum_counter >= grad_accum_steps or is_last_batch:
                            nn.utils.clip_grad_norm_(policy.residual_policy.parameters(), max_grad_norm)
                            actor_optimizer.step()
                            critic_optimizer.step()
                            actor_optimizer.zero_grad(set_to_none=True)
                            critic_optimizer.zero_grad(set_to_none=True)
                            accum_counter = 0

                        avg_pg_loss += float(pg_loss.detach().cpu().item())
                        avg_v_loss += float(v_loss.detach().cpu().item())
                        avg_entropy += float(entropy_loss.detach().cpu().item())
                        avg_kl += float(approx_kl.detach().cpu().item())
                        avg_clipfrac += float(clipfrac.detach().cpu().item())
                        loss_count += 1
                        if actor_update_planned and target_kl > 0 and approx_kl > target_kl:
                            stop_update = True
                            break
                    if loss_count > 0:
                        ppo_bar.set_postfix(
                            pg=f"{avg_pg_loss / loss_count:.4f}",
                            vf=f"{avg_v_loss / loss_count:.4f}",
                            kl=f"{avg_kl / loss_count:.2e}",
                        )
                    if stop_update:
                        break

                actor_scheduler.step()
                critic_scheduler.step()
                if loss_count > 0:
                    avg_pg_loss /= loss_count
                    avg_v_loss /= loss_count
                    avg_entropy /= loss_count
                    avg_kl /= loss_count
                    avg_clipfrac /= loss_count

            if not eval_mode:
                training_cum_time += time.time() - iteration_start_time
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0
            metrics = {
                "iteration": int(iteration),
                "mode": "eval" if eval_mode else "train",
                "global_step": int(global_step),
                "sps": int(sps),
                "learning_rate_actor": float(actor_optimizer.param_groups[0]["lr"]),
                "learning_rate_critic": float(critic_optimizer.param_groups[0]["lr"]),
                "episodes": int(len(completed_ep_successes)),
                "rollout_success_rate": float(rollout_success_rate),
                "success_timesteps_share": float(success_timesteps_share),
                "mean_success_episode_length": float(mean_success_episode_length),
                "max_success_episode_length": float(max_success_episode_length),
                "rollout_avg_return": float(rollout_avg_return),
                "rollout_avg_env_return": float(rollout_avg_env_return),
                "rollout_avg_steps": float(rollout_avg_ep_steps),
                "ppo_samples": int(total_samples),
                "actor_update": bool(actor_update_planned and not skip_update_this_itr),
                "value_loss": float(avg_v_loss),
                "policy_loss": float(avg_pg_loss),
                "entropy": float(avg_entropy),
                "kl_avg": float(avg_kl),
                "clipfrac": float(avg_clipfrac),
                "critic_ev": float(critic_ev),
                "value_return_corr": float(critic_corr),
                "action_std": float(policy.action_std.mean().detach().cpu().item()),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            if writer is not None:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        writer.add_scalar(key, value, global_step if global_step > 0 else iteration)

            log_box(
                f"{'Eval' if eval_mode else 'Train'} Iteration {iteration}/{n_train_itr} Summary",
                [
                    ("episodes", metrics["episodes"]),
                    ("rollout_success", fmt_pct(rollout_success_rate)),
                    ("success_ts_share", fmt_pct(success_timesteps_share)),
                    ("rollout_train_return", f"{rollout_avg_return:.2f}"),
                    ("rollout_env_return", f"{rollout_avg_env_return:.2f}"),
                    ("rollout_avg_steps", f"{rollout_avg_ep_steps:.1f}"),
                    ("ppo_samples", total_samples),
                    ("actor_update", actor_update_planned and not skip_update_this_itr),
                    ("value_loss", f"{avg_v_loss:.4f}"),
                    ("policy_loss", f"{avg_pg_loss:.4f}"),
                    ("entropy", f"{avg_entropy:.4f}"),
                    ("kl_avg", f"{avg_kl:.3e}"),
                    ("critic_ev", f"{critic_ev:.4f}"),
                    ("action_std", f"{metrics['action_std']:.5f}"),
                    ("SPS", sps),
                ],
            )

            if eval_mode:
                if rollout_success_rate > best_eval_success_rate:
                    best_eval_success_rate = rollout_success_rate
                    save_path = _save_checkpoint(
                        policy=policy,
                        log_dir=log_dir,
                        raw_cfg=raw_cfg,
                        iteration=iteration,
                        metrics=metrics,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        actor_scheduler=actor_scheduler,
                        critic_scheduler=critic_scheduler,
                        global_step=global_step,
                        training_cum_time=training_cum_time,
                        best_eval_success_rate=best_eval_success_rate,
                        name="actor_chkpt_best_success_rate",
                    )
                    logging.info(f"Best    | eval_success={fmt_pct(best_eval_success_rate)} checkpoint={save_path}")
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            if save_interval > 0 and (iteration % save_interval == 0 or iteration == n_train_itr):
                save_path = _save_checkpoint(
                    policy=policy,
                    log_dir=log_dir,
                    raw_cfg=raw_cfg,
                    iteration=iteration,
                    metrics=metrics,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    actor_scheduler=actor_scheduler,
                    critic_scheduler=critic_scheduler,
                    global_step=global_step,
                    training_cum_time=training_cum_time,
                    best_eval_success_rate=best_eval_success_rate,
                )
                logging.info(f"Save    | checkpoint={save_path}")

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if writer is not None:
            writer.close()
        if camera_renderer is not None:
            camera_renderer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PiperX insert-cylinder with frozen Diffusion + residual PPO on mjlab."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    parser.add_argument("--pretrained-ckpt-path", default="outputs/2_pretrain/train/2026-06-26/17-07-44_InsertCylinder-3Arms-v0_pre_zed_diffusion/checkpoints/148000_loss=0.0026_sr=0.0_ar=387.35", help="Frozen Diffusion checkpoint dir.")
    parser.add_argument("--device", default=None, help="cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--gpu-ids", default=None, help="Compatibility alias: auto, cpu, all, or ids.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override env.num_envs.")
    parser.add_argument("--n-train-itr", type=int, default=None, help="Override residual PPO iterations.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Alias for --n-train-itr.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Diffusion chunks per iteration.")
    parser.add_argument("--act-steps", type=int, default=None, help="Actions executed per Diffusion chunk.")
    parser.add_argument("--batch-size", type=int, default=None, help="PPO minibatch size.")
    parser.add_argument("--run-name", default=None, help="Override training.run_name.")
    parser.add_argument("--log-root", default=None, help="Override log root.")
    parser.add_argument("--save-interval", type=int, default=None, help="Checkpoint interval in iterations.")
    parser.add_argument("--eval-interval", type=int, default=None, help="Evaluation interval in iterations; <=0 disables eval.")
    parser.add_argument("--resume-path", default=None, help="Resume from a training_state.pt or checkpoint directory.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument(
        "--state-mode",
        choices=["auto", "legacy_agent_pos", "joint_pos_rel", "actor"],
        default=None,
        help="How to map mjlab actor observation to Diffusion observation.state.",
    )
    parser.add_argument(
        "--base-action-mode",
        choices=["legacy_absolute", "mjlab_raw"],
        default=None,
        help="Interpret frozen Diffusion action as legacy Gym absolute action or mjlab raw action.",
    )
    parser.add_argument("--env-action-clip", type=float, default=None, help="Optional raw mjlab action clamp.")
    parser.add_argument("--skip-update", action="store_true", help="Collect rollout only, no PPO update.")
    args = parser.parse_args()

    raw_cfg = _load_yaml(Path(args.config))
    train_residual_mjlab(raw_cfg, args)


if __name__ == "__main__":
    main()
