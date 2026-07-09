import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import copy
import json
import logging
import math
import shutil
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig, OmegaConf
from pprint import pformat
from torch.distributions import Normal
from tqdm import tqdm

from lerobot.common.logger import Logger
from lerobot.common.policies.utils import populate_queues
from lerobot.common.utils.utils import get_safe_torch_device, init_logging, set_global_seed

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import env.task.sim_envs  # noqa: F401
from train.finetune.finetune_mlp import (
    RunningRewardScaler,
    build_history_batch,
    compute_value_diagnostics,
    deep_update_dict,
    flatten_lerobot_obs,
    fmt_float,
    fmt_pct,
    global_cond_from_obs,
    infer_global_cond_dim,
    info_success_mask,
    load_frozen_base_policy,
    log_box,
    maybe_quiet_eval_progress,
    maybe_suppress_stdout,
    reset_done_envs_in_obs_queue,
    reset_full_obs_queue,
    stack_obs_queue,
    append_obs_queue,
)
from train.pretrain.eval_train import TopKCheckpointManager, custom_eval_policy


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
    """
    PPO actor-critic used by residual RL.

    The input is the frozen diffusion feature for the latest observation plus the
    current base action. The actor outputs a latent residual action; the final
    environment action is base_action + scaled_residual.
    """

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
        logprob_reduction: str = "mean",
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
        dist = Normal(mean, std)
        if residual_action is None:
            residual_action = mean if deterministic else dist.rsample()
        logprob = self._reduce_logprob(dist.log_prob(residual_action))
        entropy = self._reduce_logprob(dist.entropy())
        value = self.get_value(residual_obs)
        return residual_action, logprob, entropy, value, mean


class FrozenDiffusionResidualPolicy(nn.Module):
    """
    Freeze a pretrained diffusion policy and train a step-wise residual PPO head.

    This follows the code structure of robust-rearrangement:
    base_action = frozen_policy(obs)
    residual_obs = concat(diffusion_global_condition, base_action_t)
    executed_action = base_action_t + residual_policy(residual_obs)
    """

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
        logprob_reduction: str = "mean",
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
        action = self.action_from_residual(base_action, residual_action)
        return {
            "residual_obs": residual_obs,
            "residual_action": residual_action,
            "mean_residual_action": mean,
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
                )["action"].reshape_as(action_chunk)
                action_chunk = corrected
            self._queues["action"].extend(action_chunk.transpose(0, 1))

        queued_action = self._queues["action"].popleft()
        if not self.stepwise_obs:
            return queued_action

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
        }
        torch.save(adapter_state, save_directory / "residual_policy.pt")
        adapter_config = {
            key: value
            for key, value in adapter_state.items()
            if key != "model"
        }
        with open(save_directory / "residual_policy_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2, ensure_ascii=False)


def snapshot_trainable_state(policy: FrozenDiffusionResidualPolicy):
    return copy.deepcopy(policy.residual_policy.state_dict())


def restore_trainable_state(policy: FrozenDiffusionResidualPolicy, state, device):
    policy.residual_policy.load_state_dict(state, strict=True)
    policy.to(device)


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


def make_vector_or_single_env(env_id: str, cameras: list[str], n_envs: int):
    if n_envs > 1:
        return gym.vector.AsyncVectorEnv(
            [lambda: gym.make(id=env_id, cameras=cameras) for _ in range(n_envs)],
            shared_memory=True,
            context="spawn",
            autoreset_mode="SameStep",
        )
    return gym.make(id=env_id, cameras=cameras)


def train_residual_finetune(
    cfg: DictConfig,
    out_dir: str | None = None,
    job_name: str | None = None,
):
    init_logging()
    out_dir = out_dir or os.getcwd()
    log_box(
        "DP+Residual PPO Finetune",
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
    action_dim = int(base_policy.config.output_shapes["action"][0])
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
    if not ref_cams:
        raise ValueError(f"Invalid policy snapshot: ref_cams={ref_cams}")

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
    env = make_vector_or_single_env(env_id, train_cameras, n_envs)
    eval_env = gym.make(id=env_id, cameras=eval_cameras)

    with maybe_suppress_stdout(quiet_terminal):
        global_cond_dim = infer_global_cond_dim(base_policy, device)

    residual_hidden_dim = int(
        getattr(cfg.training, "residual_hidden_dim", getattr(cfg.training, "residual_mlp_hidden_dim", 512))
    )
    residual_depth = int(
        getattr(cfg.training, "residual_depth", getattr(cfg.training, "residual_mlp_depth", 2))
    )
    critic_hidden_dim = int(getattr(cfg.training, "residual_critic_hidden_dim", residual_hidden_dim))
    critic_depth = int(getattr(cfg.training, "residual_critic_depth", residual_depth))
    residual_std = float(
        getattr(cfg.training, "residual_std", getattr(cfg.training, "residual_mlp_std", 0.02))
    )
    residual_learn_std = bool(
        getattr(cfg.training, "residual_learn_std", getattr(cfg.training, "residual_mlp_learn_std", True))
    )
    residual_action_scale = float(
        getattr(cfg.training, "residual_action_scale", getattr(cfg.training, "lambda_all", 0.1))
    )
    residual_max_delta = float(
        getattr(cfg.training, "residual_max_delta", getattr(cfg.training, "residual_mlp_max_delta", 0.0))
    )
    residual_activation = str(getattr(cfg.training, "residual_activation", "SiLU"))
    action_head_std = float(getattr(cfg.training, "residual_action_head_std", 0.0))
    stepwise_obs = bool(getattr(cfg.training, "residual_stepwise_obs", getattr(cfg.training, "residual_mlp_stepwise_obs", True)))
    if not stepwise_obs:
        logging.warning(
            "training.residual_stepwise_obs=false: residual will correct the whole DP chunk "
            "from the initial observation. Paper-style training usually keeps this true."
        )

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
        logprob_reduction=str(getattr(cfg.training, "logprob_reduction", "mean")),
        stepwise_obs=stepwise_obs,
    ).to(device)

    actor_optimizer = torch.optim.AdamW(
        policy.actor_parameters,
        lr=float(getattr(cfg.training, "actor_lr", 3e-4)),
        betas=tuple(getattr(cfg.training, "optimizer_betas_actor", (0.9, 0.999))),
        eps=1e-5,
        weight_decay=float(getattr(cfg.training, "weight_decay", 1e-6)),
    )
    critic_optimizer = torch.optim.AdamW(
        policy.critic_parameters,
        lr=float(getattr(cfg.training, "critic_lr", 5e-3)),
        eps=1e-5,
        weight_decay=float(getattr(cfg.training, "critic_weight_decay", 1e-6)),
    )
    from torch.optim.lr_scheduler import LinearLR

    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=0.1,
        total_iters=max(1, int(getattr(cfg.training, "actor_lr_warmup_iters", 5))),
    )

    max_checkpoints = int(getattr(cfg.eval, "max_checkpoints", 5))
    checkpoint_metric = str(getattr(cfg.eval, "checkpoint_metric", "success")).lower()
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
    batch_size = int(getattr(cfg.training, "batch_size", 256))
    update_epochs = int(getattr(cfg.training, "update_epochs", 4))
    clip_ratio = float(getattr(cfg.training, "clip_ratio", 0.2))
    target_kl = float(getattr(cfg.training, "target_kl", 0.1))
    gamma = float(getattr(cfg.training, "gamma", 0.999))
    gae_lambda = float(getattr(cfg.training, "gae_lambda", 0.95))
    vf_coef = float(getattr(cfg.training, "vf_coef", 1.0))
    ent_coef = float(getattr(cfg.training, "ent_coef", 0.0))
    residual_l1 = float(getattr(cfg.training, "residual_l1", 0.0))
    residual_l2 = float(getattr(cfg.training, "residual_l2", 0.0))
    reward_scale_running = bool(getattr(cfg.training, "reward_scale_running", True))
    reward_scale_const = float(getattr(cfg.training, "reward_scale_const", 1.0))
    norm_adv = bool(getattr(cfg.training, "norm_adv", True))
    adv_lower_q = float(getattr(cfg.training, "clip_advantage_lower_quantile", 0.05))
    adv_upper_q = float(getattr(cfg.training, "clip_advantage_upper_quantile", 0.95))
    grad_accum_steps = max(1, int(getattr(cfg.training, "grad_accumulate", 1)))
    max_grad_norm = float(getattr(cfg.training, "max_grad_norm", 1.0))
    use_disk_cache = bool(getattr(cfg.training, "use_disk_cache", False))
    rollout_buffer_root = Path(
        str(getattr(cfg.training, "rollout_buffer_dir", "outputs/buffer/finetune_residual"))
    )
    if not rollout_buffer_root.is_absolute():
        rollout_buffer_root = Path(ROOT_DIR) / rollout_buffer_root
    rollout_buffer_keep_iterations = bool(getattr(cfg.training, "rollout_buffer_keep_iterations", False))
    skip_update = bool(getattr(cfg.training, "skip_update", False))
    update_actor = bool(getattr(cfg.training, "update_actor", True))
    show_progress = bool(getattr(cfg.training, "show_progress", False))
    rollout_progress = bool(getattr(cfg.training, "rollout_progress", show_progress))
    training_progress = bool(getattr(cfg.training, "training_progress", show_progress))
    reward_source = "binary_success_label"
    bootstrap_on_truncation = bool(getattr(cfg.training, "bootstrap_on_truncation", False))

    log_box(
        "Run Config",
        [
            ("env_id", env_id),
            ("n_envs", n_envs),
            ("cameras", train_cameras),
            ("global_cond_dim", global_cond_dim),
            ("residual_obs_dim", policy.residual_obs_dim),
            ("action_dim", action_dim),
            ("action_slice", f"[{action_start}:{action_end}]"),
            ("act_steps", act_steps),
            ("rollout_control_steps", n_steps * n_envs * act_steps),
            ("gamma/gae_lambda", f"{gamma:.4f} / {gae_lambda:.3f}"),
            ("residual_scale/max_delta", f"{residual_action_scale:.4f} / {residual_max_delta:.5f}"),
            ("action_std", f"{float(policy.action_std.mean().detach().cpu().item()):.5f}"),
            ("reward", "binary_success_label: success=1, otherwise=0"),
            ("rollout_buffer", str(rollout_buffer_root) if use_disk_cache else "memory"),
        ],
    )
    log_box(
        "Model Params",
        [
            ("frozen_diffusion", f"{sum(p.numel() for p in base_policy.parameters()) / 1e6:.2f}M"),
            ("residual_actor_critic", f"{sum(p.numel() for p in policy.adapter_parameters()) / 1e6:.2f}M"),
        ],
    )

    prev_obs, _ = env.reset()
    prev_obs = flatten_lerobot_obs(prev_obs)
    raw_obs_queue = {key: deque(maxlen=n_obs_steps) for key in prev_obs.keys()}
    reset_full_obs_queue(raw_obs_queue, prev_obs, n_obs_steps)

    running_ep_train_rewards = np.zeros(n_envs, dtype=np.float32)
    running_ep_env_rewards = np.zeros(n_envs, dtype=np.float32)
    running_ep_steps = np.zeros(n_envs, dtype=np.int32)
    running_reward_scaler = RunningRewardScaler(n_envs, gamma=gamma)
    chunk_start_firsts = np.ones(n_envs, dtype=np.float32)
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
                ("control_step_samples", n_steps * n_envs * act_steps),
                ("update_epochs", update_epochs),
                ("eval_due", (itr + 1) > critic_warmup_iters and (itr + 1) % int(getattr(cfg.eval, "eval_freq", 5)) == 0),
            ],
        )

        rollout_buffer_path = None
        if use_disk_cache:
            rollout_buffer_path = rollout_buffer_root / (
                f"iteration_{itr + 1:06d}" if rollout_buffer_keep_iterations else "current"
            )
            if rollout_buffer_path.exists():
                shutil.rmtree(rollout_buffer_path)
            rollout_buffer_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Rollout buffer | writing to {rollout_buffer_path}")
            residual_obs_trajs = np.lib.format.open_memmap(
                rollout_buffer_path / "residual_obs_trajs.npy",
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, n_envs, act_steps, policy.residual_obs_dim),
            )
            residual_action_trajs = np.lib.format.open_memmap(
                rollout_buffer_path / "residual_action_trajs.npy",
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, n_envs, act_steps, action_dim),
            )
            base_action_trajs = np.lib.format.open_memmap(
                rollout_buffer_path / "base_action_trajs.npy",
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, n_envs, act_steps, action_dim),
            )
            old_logprob_trajs = np.lib.format.open_memmap(
                rollout_buffer_path / "old_logprob_trajs.npy",
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, n_envs, act_steps),
            )
            reward_trajs = np.lib.format.open_memmap(
                rollout_buffer_path / "reward_trajs.npy",
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, n_envs, act_steps),
            )
            terminated_trajs = np.lib.format.open_memmap(
                rollout_buffer_path / "terminated_trajs.npy",
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, n_envs, act_steps),
            )
            firsts_trajs = np.lib.format.open_memmap(
                rollout_buffer_path / "firsts_trajs.npy",
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, n_envs, act_steps),
            )
            valid_step_trajs = np.lib.format.open_memmap(
                rollout_buffer_path / "valid_step_trajs.npy",
                dtype=bool,
                mode="w+",
                shape=(n_steps, n_envs, act_steps),
            )
            with open(rollout_buffer_path / "metadata.yaml", "w", encoding="utf-8") as f:
                yaml.dump(
                    {
                        "iteration": int(itr + 1),
                        "reward_source": reward_source,
                        "reward_success": 1.0,
                        "reward_otherwise": 0.0,
                        "n_steps": int(n_steps),
                        "n_envs": int(n_envs),
                        "act_steps": int(act_steps),
                        "action_dim": int(action_dim),
                        "residual_obs_dim": int(policy.residual_obs_dim),
                    },
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                )
        else:
            residual_obs_trajs = np.zeros(
                (n_steps, n_envs, act_steps, policy.residual_obs_dim),
                dtype=np.float32,
            )
            residual_action_trajs = np.zeros(
                (n_steps, n_envs, act_steps, action_dim),
                dtype=np.float32,
            )
            base_action_trajs = np.zeros(
                (n_steps, n_envs, act_steps, action_dim),
                dtype=np.float32,
            )
            old_logprob_trajs = np.zeros((n_steps, n_envs, act_steps), dtype=np.float32)
            reward_trajs = np.zeros((n_steps, n_envs, act_steps), dtype=np.float32)
            terminated_trajs = np.zeros((n_steps, n_envs, act_steps), dtype=np.float32)
            firsts_trajs = np.zeros((n_steps, n_envs, act_steps), dtype=np.float32)
            valid_step_trajs = np.zeros((n_steps, n_envs, act_steps), dtype=bool)
        completed_ep_train_rewards = []
        completed_ep_env_rewards = []
        completed_ep_successes = []
        completed_ep_steps = []

        policy.eval()
        logging.info("Rollout | collecting frozen Diffusion + residual PPO actions")
        rollout_bar = tqdm(
            range(n_steps),
            desc=f"Residual Rollout {itr + 1:04d}",
            leave=rollout_progress,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not rollout_progress,
        )
        for step in rollout_bar:
            stacked_raw_obs = stack_obs_queue(raw_obs_queue, n_envs, n_obs_steps)
            batch_obs = build_history_batch(stacked_raw_obs, policy, device)
            with torch.no_grad():
                with maybe_suppress_stdout(quiet_terminal):
                    base_actions_t, initial_global_cond_t = policy.frozen_diffusion_actions(
                        batch_obs,
                        return_global_cond=True,
                    )

            chunk_env_reward = np.zeros(n_envs, dtype=np.float32)
            any_done_accum = np.zeros(n_envs, dtype=bool)
            success_accum = np.zeros(n_envs, dtype=bool)
            safe_actions = np.zeros((n_envs, action_dim), dtype=np.float32)
            step_reward_venv = np.zeros((n_envs, act_steps), dtype=np.float32)
            step_done_venv = np.zeros((n_envs, act_steps), dtype=np.float32)
            step_first_venv = np.zeros((n_envs, act_steps), dtype=np.float32)
            step_valid_venv = np.zeros((n_envs, act_steps), dtype=bool)
            current_firsts_venv = chunk_start_firsts.copy()

            for step_i in range(act_steps):
                with torch.no_grad():
                    if step_i == 0 or not stepwise_obs:
                        global_cond_step_t = initial_global_cond_t
                    else:
                        step_stacked_raw_obs = stack_obs_queue(raw_obs_queue, n_envs, n_obs_steps)
                        step_batch_obs = build_history_batch(step_stacked_raw_obs, policy, device)
                        with maybe_suppress_stdout(quiet_terminal):
                            global_cond_step_t = global_cond_from_obs(policy, step_batch_obs)

                    base_step_t = base_actions_t[:, action_start + step_i, :]
                    samples = policy.step_action(
                        base_action=base_step_t,
                        global_cond=global_cond_step_t,
                        deterministic=False,
                    )
                    curr_action = samples["action"].detach().cpu().numpy()
                    residual_obs_trajs[step, :, step_i, :] = samples["residual_obs"].detach().cpu().numpy()
                    residual_action_trajs[step, :, step_i, :] = samples["residual_action"].detach().cpu().numpy()
                    base_action_trajs[step, :, step_i, :] = base_step_t.detach().cpu().numpy()
                    old_logprob_trajs[step, :, step_i] = samples["logprob"].detach().cpu().numpy()

                active_mask = ~any_done_accum
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

                step_train_reward = just_success.astype(np.float32)
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

            invalid_step_mask = ~step_valid_venv
            step_first_venv[invalid_step_mask] = 1.0
            step_done_venv[invalid_step_mask] = 1.0
            reward_trajs[step] = step_reward_venv
            terminated_trajs[step] = step_done_venv
            firsts_trajs[step] = step_first_venv
            valid_step_trajs[step] = step_valid_venv
            chunk_start_firsts = any_done_accum.astype(np.float32)

            chunk_reward = step_reward_venv.sum(axis=1)
            running_ep_train_rewards += chunk_reward
            running_ep_env_rewards += chunk_env_reward
            for env_idx in range(n_envs):
                if any_done_accum[env_idx]:
                    completed_ep_train_rewards.append(float(running_ep_train_rewards[env_idx]))
                    completed_ep_env_rewards.append(float(running_ep_env_rewards[env_idx]))
                    completed_ep_successes.append(bool(success_accum[env_idx]))
                    completed_ep_steps.append(int(running_ep_steps[env_idx]))
                    running_ep_train_rewards[env_idx] = 0.0
                    running_ep_env_rewards[env_idx] = 0.0
                    running_ep_steps[env_idx] = 0

            if rollout_progress and ((step + 1) % max(1, n_steps // 100) == 0 or step + 1 == n_steps):
                rollout_bar.set_postfix(
                    episodes=len(completed_ep_successes),
                    success=fmt_pct(np.mean(completed_ep_successes) if completed_ep_successes else 0.0),
                )

        if use_disk_cache:
            for cached_array in (
                residual_obs_trajs,
                residual_action_trajs,
                base_action_trajs,
                old_logprob_trajs,
                reward_trajs,
                terminated_trajs,
                firsts_trajs,
                valid_step_trajs,
            ):
                cached_array.flush()

        rollout_avg_return = np.mean(completed_ep_train_rewards) if completed_ep_train_rewards else 0.0
        rollout_avg_env_return = np.mean(completed_ep_env_rewards) if completed_ep_env_rewards else 0.0
        rollout_avg_ep_steps = np.mean(completed_ep_steps) if completed_ep_steps else float("nan")
        rollout_success_rate = np.mean(completed_ep_successes) if completed_ep_successes else 0.0
        logging.info(
            "Rollout | "
            f"episodes={len(completed_ep_successes):3d} | "
            f"success={fmt_pct(rollout_success_rate):>6} | "
            f"train_return={rollout_avg_return:8.2f} | "
            f"env_return={rollout_avg_env_return:8.2f} | "
            f"avg_steps={rollout_avg_ep_steps:6.1f}"
        )

        if skip_update:
            logging.info("training.skip_update=true; skipping residual PPO update.")
            continue

        total_raw_samples = n_steps * n_envs * act_steps
        residual_obs_flat = residual_obs_trajs.reshape(total_raw_samples, policy.residual_obs_dim)
        residual_action_flat = residual_action_trajs.reshape(total_raw_samples, action_dim)
        old_logprobs_flat = old_logprob_trajs.reshape(total_raw_samples)
        valid_flat = valid_step_trajs.reshape(total_raw_samples)
        valid_indices_np = np.flatnonzero(valid_flat)
        if valid_indices_np.size == 0:
            raise RuntimeError("No valid control-step samples were collected for PPO update.")

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

            values_trajs = values_all_flat.reshape(n_steps, n_envs, act_steps)
            values_time = values_trajs.transpose(0, 2, 1).reshape(n_steps * act_steps, n_envs)
            last_stacked_raw_obs = stack_obs_queue(raw_obs_queue, n_envs, n_obs_steps)
            last_obs = build_history_batch(last_stacked_raw_obs, policy, device)
            with maybe_suppress_stdout(quiet_terminal):
                last_base_actions_t, last_global_cond_t = policy.frozen_diffusion_actions(
                    last_obs,
                    return_global_cond=True,
                )
            last_base_step_t = last_base_actions_t[:, action_start, :]
            next_values_last = (
                policy.value_for_step(last_base_step_t, last_global_cond_t).cpu().numpy().flatten()
            )

        rewards_time = reward_trajs.transpose(0, 2, 1).reshape(n_steps * act_steps, n_envs)
        terminated_time = terminated_trajs.transpose(0, 2, 1).reshape(n_steps * act_steps, n_envs)
        firsts_time = firsts_trajs.transpose(0, 2, 1).reshape(n_steps * act_steps, n_envs)
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

        residual_obs_flat = residual_obs_flat[valid_indices_np]
        residual_action_flat = residual_action_flat[valid_indices_np]
        old_logprobs_flat = old_logprobs_flat[valid_indices_np]
        returns_k = torch.from_numpy(returns_flat[valid_indices_np]).float().to(device)
        advantages_k = torch.from_numpy(advantages_flat[valid_indices_np]).float().to(device)
        old_logprobs_k = torch.from_numpy(old_logprobs_flat).float().to(device)
        total_samples = int(valid_indices_np.size)

        avg_pg_loss = 0.0
        avg_v_loss = 0.0
        avg_entropy = 0.0
        avg_kl = 0.0
        avg_clipfrac = 0.0
        loss_count = 0
        indices = np.arange(total_samples)
        num_batches = max(1, math.ceil(total_samples / batch_size))
        logging.info(
            f"Update  | actor_update={actor_update_planned} | epochs={update_epochs} | "
            f"minibatches/epoch={num_batches} | samples={total_samples}"
        )

        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        ppo_bar = tqdm(
            range(update_epochs),
            desc=f"Residual PPO {itr + 1:04d}",
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
                mb_obs = torch.from_numpy(np.ascontiguousarray(residual_obs_flat[inds_np])).float().to(device)
                mb_actions = torch.from_numpy(np.ascontiguousarray(residual_action_flat[inds_np])).float().to(device)
                mb_old_logprobs = old_logprobs_k[inds_np]
                mb_advantages = advantages_k[inds_np]
                mb_returns = returns_k[inds_np]

                _, newlogprob, entropy, newvalue, action_mean = policy.residual_policy.get_action_and_value(
                    mb_obs,
                    residual_action=mb_actions,
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
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
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
                    nn.utils.clip_grad_norm_(policy.critic_parameters, max_grad_norm)
                    critic_optimizer.step()
                    critic_optimizer.zero_grad(set_to_none=True)
                    if actor_update_planned:
                        nn.utils.clip_grad_norm_(policy.actor_parameters, max_grad_norm)
                        actor_optimizer.step()
                        actor_optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0

                avg_pg_loss += float(pg_loss.detach().cpu().item())
                avg_v_loss += float(v_loss.detach().cpu().item())
                avg_entropy += float(entropy_loss.detach().cpu().item())
                avg_kl += float(approx_kl.detach().cpu().item())
                avg_clipfrac += float(clipfrac.detach().cpu().item())
                loss_count += 1
                if actor_update_planned and target_kl > 0 and approx_kl > target_kl:
                    logging.info(
                        f"Update  | early stop at epoch={epoch} batch={batch_idx}: "
                        f"kl={float(approx_kl):.4e} > target_kl={target_kl:.4e}"
                    )
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

        if actor_update_planned:
            actor_scheduler.step()

        if loss_count > 0:
            avg_pg_loss /= loss_count
            avg_v_loss /= loss_count
            avg_entropy /= loss_count
            avg_kl /= loss_count
            avg_clipfrac /= loss_count

        log_box(
            f"Iteration {itr + 1} Summary",
            [
                ("episodes", len(completed_ep_successes)),
                ("rollout_success", fmt_pct(rollout_success_rate)),
                ("rollout_train_return", f"{rollout_avg_return:.2f}"),
                ("rollout_env_return", f"{rollout_avg_env_return:.2f}"),
                ("rollout_avg_steps", f"{rollout_avg_ep_steps:.1f}"),
                ("ppo_granularity", "control_step"),
                ("ppo_samples", total_samples),
                ("actor_update", actor_update_planned),
                ("value_loss", f"{avg_v_loss:.4f}"),
                ("policy_loss", f"{avg_pg_loss:.4f}"),
                ("entropy", f"{avg_entropy:.4f}"),
                ("kl_avg", f"{avg_kl:.3e}"),
                ("clipfrac", f"{avg_clipfrac:.3f}"),
                ("critic_ev", f"{critic_ev:.4f}"),
                ("value_return_corr", f"{critic_corr:.4f}"),
                ("action_std", f"{float(policy.action_std.mean().detach().cpu().item()):.5f}"),
            ],
        )

        try:
            del (
                residual_obs_flat,
                residual_action_flat,
                returns_k,
                advantages_k,
                old_logprobs_k,
                values_all_flat,
            )
            import gc

            gc.collect()
        except Exception:
            pass
        if device.type == "cuda":
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
            logging.info("Eval    | " f"success={fmt_pct(sr):>6} | " f"avg_reward={ar:8.2f}")
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
                eval_collapsed = (
                    success_collapsed and (reward_collapsed if rollback_reward_gate else True)
                    if checkpoint_metric in success_metric_names
                    else reward_collapsed
                )
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
                        critic_optimizer.state.clear()
                        eval_collapse_count = 0
                        logging.warning(
                            f"Rolled back to best residual policy: "
                            f"success={best_eval_success_rate * 100:.1f}% reward={best_eval_reward:.2f}"
                        )
                        continue
                else:
                    eval_collapse_count = 0

            ckpt_name = (
                f"{itr + 1:06d}_sr={sr:.2f}_reward={ar:.2f}"
                f"_Rloss={avg_pg_loss:.4f}_Vloss={avg_v_loss:.4f}"
            )
            ckpt_path = Path(out_dir) / "checkpoints" / ckpt_name
            save_path = ckpt_path / "pretrained_model"
            final_videos_dir = ckpt_path / "eval" / "eval_videos"
            if tmp_videos_dir.exists() and tmp_videos_dir != final_videos_dir:
                final_videos_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_videos_dir), str(final_videos_dir))

            policy.save_pretrained(save_path)
            current_ft_dict = OmegaConf.to_container(cfg, resolve=True)
            base_config_dict = OmegaConf.to_container(hydra_cfg, resolve=True) if hydra_cfg is not None else {}
            final_config_dict = deep_update_dict(base_config_dict, current_ft_dict)
            final_policy = deep_update_dict(base_config_dict.get("policy", {}), current_ft_dict.get("policy", {}))
            final_policy = deep_update_dict(
                final_policy,
                {
                    "wrapper": "FrozenDiffusionResidualPolicy",
                    "action_start": int(action_start),
                    "action_end": int(action_end),
                    "residual_policy_checkpoint": "residual_policy.pt",
                    "global_cond_dim": int(policy.global_cond_dim),
                    "residual_obs_dim": int(policy.residual_obs_dim),
                    "residual_stepwise_obs": bool(policy.stepwise_obs),
                    "residual_action_scale": float(policy.residual_policy.action_scale),
                    "residual_max_delta": float(policy.residual_policy.max_delta),
                    "residual_std": float(policy.action_std.mean().detach().cpu().item()),
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
                "ppo_control_step_training": True,
                "ppo_update_samples": int(total_samples),
                "reward_source": reward_source,
                "reward_success": 1.0,
                "reward_otherwise": 0.0,
                "rollout_buffer_dir": str(rollout_buffer_root) if use_disk_cache else None,
                "rollout_buffer_keep_iterations": bool(rollout_buffer_keep_iterations),
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
    train_residual_finetune(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


if __name__ == "__main__":
    default_args = [
        "policy=ft_zed_diffusion_residual",
    ]
    for arg in default_args:
        arg_key = arg.split("=")[0].lstrip("+")
        if not any(sys_arg.split("=")[0].lstrip("+") == arg_key for sys_arg in sys.argv):
            sys.argv.append(arg)
    train_cli()
