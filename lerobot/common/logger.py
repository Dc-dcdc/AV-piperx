#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Borrowed from https://github.com/fyhMer/fowm/blob/main/src/logger.py

# TODO(rcadene, alexander-soare): clean this file
"""

import logging
import os
import re
from pathlib import Path

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.utils.utils import get_global_random_state, set_global_random_state


def log_output_dir(out_dir):
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {out_dir}")


def cfg_to_group(cfg: DictConfig, return_list: bool = False) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.name}",
        f"dataset:{cfg.dataset_repo_id}",
        f"env:{cfg.env.name}",
        f"seed:{cfg.seed}",
    ]
    return lst if return_list else "-".join(lst)


def get_wandb_resume_info_from_filesystem(
    checkpoint_dir: Path,
) -> tuple[str, str | None, str | None]:
    """读取最近一次已成功初始化的本地 W&B run 信息。

    不能只读取 ``wandb/latest-run``：一次失败的恢复也会改写该软链接。优先从
    ``files/config.yaml`` 读取元数据；新版 W&B 未生成该文件时，仅接受日志能够证明
    初始化成功且包含匹配 run URL 的目录。恢复原 project/entity，避免同一 run ID
    被错误地放到另一个项目中。
    """
    wandb_dir = checkpoint_dir.parent / "wandb"
    run_dirs = sorted(
        wandb_dir.glob("run-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        run_files = list(run_dir.glob("run-*.wandb"))
        if len(run_files) != 1:
            continue
        match = re.fullmatch(r"run-(.+)\.wandb", run_files[0].name)
        if match is None:
            continue

        run_id = match.group(1)
        config_path = run_dir / "files" / "config.yaml"
        if config_path.is_file():
            local_cfg = OmegaConf.load(config_path)
            wandb_cfg = OmegaConf.select(local_cfg, "wandb.value", default={})
            project = OmegaConf.select(wandb_cfg, "project", default=None)
            entity = OmegaConf.select(wandb_cfg, "entity", default=None)
            return run_id, project, entity

        # W&B 0.25.x may not create files/config.yaml. In that case, require
        # evidence that wandb.init completed, then recover project/entity from
        # the URL emitted by this logger. Failed init attempts do not satisfy
        # both conditions and are therefore ignored.
        debug_path = run_dir / "logs" / "debug.log"
        output_path = run_dir / "files" / "output.log"
        if not debug_path.is_file() or not output_path.is_file():
            continue
        debug_log = debug_path.read_text(errors="replace")
        if "run started, returning control to user process" not in debug_log:
            continue
        output_log = output_path.read_text(errors="replace")
        url_match = re.search(
            rf"https?://wandb\.ai/([^/\s]+)/([^/\s]+)/runs/{re.escape(run_id)}(?:\s|$|\x1b)",
            output_log,
        )
        if url_match is not None:
            entity, project = url_match.groups()
            return run_id, project, entity

    raise RuntimeError(
        f"Couldn't get a successfully initialized previous W&B run from {wandb_dir}."
    )


def get_wandb_run_id_from_filesystem(checkpoint_dir: Path) -> str:
    """兼容旧调用，只返回最近一次成功初始化的 W&B run ID。"""
    return get_wandb_resume_info_from_filesystem(checkpoint_dir)[0]


class Logger:
    """Primary logger object. Logs either locally or using wandb.

    The logger creates the following directory structure:

    provided_log_dir
    ├── .hydra  # hydra's configuration cache
    ├── checkpoints
    │   ├── specific_checkpoint_name
    │   │   ├── pretrained_model  # Hugging Face pretrained model directory
    │   │   │   ├── ...
    │   │   └── training_state.pth  # optimizer, scheduler, and random states + training step
    |   ├── another_specific_checkpoint_name
    │   │   ├── ...
    |   ├── ...
    │   └── last  # a softlink to the last logged checkpoint
    """

    pretrained_model_dir_name = "pretrained_model"
    online_pretrained_model_dir_name = "online_pretrained_model"
    training_state_file_name = "training_state.pth"

    def __init__(self, cfg: DictConfig, log_dir: str, wandb_job_name: str | None = None):
        """
        Args:
            log_dir: The directory to save all logs and training outputs to.
            job_name: The WandB job name.
        """
        self._cfg = cfg
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.get_checkpoints_dir(log_dir)
        self.last_checkpoint_dir = self.get_last_checkpoint_dir(log_dir)
        self.last_pretrained_model_dir = self.get_last_pretrained_model_dir(log_dir)

        # Set up WandB.
        self._group = cfg_to_group(cfg)
        project = cfg.get("wandb", {}).get("project")
        entity = cfg.get("wandb", {}).get("entity")
        enable_wandb = cfg.get("wandb", {}).get("enable", False) #    
        wandb_run_id = None
        if enable_wandb and cfg.resume:
            wandb_run_id, previous_project, previous_entity = (
                get_wandb_resume_info_from_filesystem(self.checkpoints_dir)
            )
            if previous_project and previous_project != project:
                logging.warning(
                    "W&B续训项目与当前配置不一致："
                    f"current={project!r}, previous={previous_project!r}；"
                    "为恢复原run，将使用previous project。"
                )
                project = previous_project
            if previous_entity and previous_entity != entity:
                logging.warning(
                    "W&B续训entity与当前配置不一致："
                    f"current={entity!r}, previous={previous_entity!r}；"
                    "为恢复原run，将使用previous entity。"
                )
                entity = previous_entity
            # 让传给 W&B 的 config 与实际恢复目标保持一致，避免页面上仍显示错误项目。
            OmegaConf.update(self._cfg, "wandb.project", project, force_add=True)
            if entity is not None:
                OmegaConf.update(self._cfg, "wandb.entity", entity, force_add=True)

        run_offline = not enable_wandb or not project
        if run_offline:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
            self._wandb = None
        else:
            import wandb

            wandb.init(
                id=wandb_run_id,
                project=project,
                entity=entity,
                name=wandb_job_name,
                notes=cfg.get("wandb", {}).get("notes"),
                tags=cfg_to_group(cfg, return_list=True),
                dir=log_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
                # TODO(rcadene): try set to True
                save_code=False,
                # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
                job_type="train_eval",
                resume="must" if cfg.resume else None,
            )
            print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
            logging.info(f"Track this run --> {colored(wandb.run.url, 'yellow', attrs=['bold'])}")
            # 异步评估结果可能晚于对应checkpoint返回。W&B的全局step必须
            # 单调递增，因此用独立的checkpoint_step作为所有eval指标横轴。
            wandb.define_metric("eval/checkpoint_step")
            wandb.define_metric(
                "eval/*",
                step_metric="eval/checkpoint_step",
            )
            self._wandb = wandb

    @classmethod
    def get_checkpoints_dir(cls, log_dir: str | Path) -> Path:
        """Given the log directory, get the sub-directory in which checkpoints will be saved."""
        return Path(log_dir) / "checkpoints"

    @classmethod
    def get_last_checkpoint_dir(cls, log_dir: str | Path) -> Path:
        """Given the log directory, get the sub-directory in which the last checkpoint will be saved."""
        return cls.get_checkpoints_dir(log_dir) / "last"

    @classmethod
    def get_last_pretrained_model_dir(cls, log_dir: str | Path) -> Path:
        """
        Given the log directory, get the sub-directory in which the last checkpoint's pretrained weights will
        be saved.
        """
        return cls.get_last_checkpoint_dir(log_dir) / cls.pretrained_model_dir_name

    def save_model(self, save_dir: Path, policy: Policy, wandb_artifact_name: str | None = None):
        """Save the weights of the Policy model using PyTorchModelHubMixin.

        The weights are saved in a folder called "pretrained_model" under the checkpoint directory.

        Optionally also upload the model to WandB.
        """
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(save_dir)
        # Also save the full Hydra config for the env configuration.
        OmegaConf.save(self._cfg, save_dir / "config.yaml")
        if self._wandb and not self._cfg.wandb.disable_artifact:
            # note wandb artifact does not accept ":" or "/" in its name
            artifact = self._wandb.Artifact(wandb_artifact_name, type="model")
            artifact.add_file(save_dir / SAFETENSORS_SINGLE_FILE)
            self._wandb.log_artifact(artifact)
        if self.last_checkpoint_dir.exists():
            os.remove(self.last_checkpoint_dir)

    def save_training_state(
        self,
        save_dir: Path,
        train_step: int,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        *,
        grad_scaler=None,
        ema_state: dict | None = None,
    ):
        """Checkpoint the global training_step, optimizer state, scheduler state, and random state.

        All of these are saved as "training_state.pth" under the checkpoint directory.
        """
        training_state = {
            "step": train_step,
            "optimizer": optimizer.state_dict(),
            **get_global_random_state(),
        }
        if scheduler is not None:
            training_state["scheduler"] = scheduler.state_dict()
        if grad_scaler is not None and grad_scaler.is_enabled():
            training_state["grad_scaler"] = grad_scaler.state_dict()
        if ema_state is not None:
            training_state["ema"] = dict(ema_state)
        torch.save(training_state, save_dir / self.training_state_file_name)

    def save_checkpoint(
        self,
        train_step: int,
        policy: Policy,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        identifier: str,
        *,
        ema_policy: Policy | None = None,
        ema_state: dict | None = None,
        grad_scaler=None,
    ):
        """Checkpoint model weights and training state.

        启用EMA时，``pretrained_model``保存默认评估/部署的EMA权重，
        ``online_pretrained_model``保存与optimizer对应、用于resume的在线权重。
        """
        checkpoint_dir = self.checkpoints_dir / str(identifier)
        wandb_artifact_name = (
            None
            if self._wandb is None
            else f"{self._group.replace(':', '_').replace('/', '_')}-{self._cfg.seed}-{identifier}"
        )
        if ema_policy is None:
            self.save_model(
                checkpoint_dir / self.pretrained_model_dir_name,
                policy,
                wandb_artifact_name=wandb_artifact_name,
            )
        else:
            self.save_model(
                checkpoint_dir / self.pretrained_model_dir_name,
                ema_policy,
                wandb_artifact_name=wandb_artifact_name,
            )
            online_artifact_name = (
                None
                if wandb_artifact_name is None
                else f"{wandb_artifact_name}-online"
            )
            self.save_model(
                checkpoint_dir / self.online_pretrained_model_dir_name,
                policy,
                wandb_artifact_name=online_artifact_name,
            )
        self.save_training_state(
            checkpoint_dir,
            train_step,
            optimizer,
            scheduler,
            grad_scaler=grad_scaler,
            ema_state=ema_state,
        )
        os.symlink(checkpoint_dir.absolute(), self.last_checkpoint_dir)

    def load_last_training_state(self, optimizer: Optimizer, scheduler: LRScheduler | None) -> int:
        """
        Given the last checkpoint in the logging directory, load the optimizer state, scheduler state, and
        random state, and return the global training step.
        """
        training_state = torch.load(self.last_checkpoint_dir / self.training_state_file_name)
        optimizer.load_state_dict(training_state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(training_state["scheduler"])
        elif "scheduler" in training_state:
            raise ValueError(
                "The checkpoint contains a scheduler state_dict, but no LRScheduler was provided."
            )
        # Small hack to get the expected keys: use `get_global_random_state`.
        set_global_random_state({k: training_state[k] for k in get_global_random_state()})
        return training_state["step"]

    def log_dict(self, d, step, mode="train"):
        assert mode in {"train", "eval"}
        # TODO(alexander-soare): Add local text log.
        if self._wandb is not None:
            payload = {}
            for k, v in d.items():
                if not isinstance(v, (int, float, str)):
                    logging.warning(
                        f'WandB logging of key "{k}" was ignored as its type is not handled by this wrapper.'
                    )
                    continue
                payload[f"{mode}/{k}"] = v

            if mode == "eval":
                # 同步评估默认等于当前step；异步评估可在d中显式传入
                # checkpoint_step，从而与实际被评估的模型快照严格对齐。
                payload.setdefault("eval/checkpoint_step", int(step))
            if payload:
                self._wandb.log(payload, step=step)

    def log_video(
        self,
        video_path: str,
        step: int,
        mode: str = "train",
        checkpoint_step: int | None = None,
    ):
        assert mode in {"train", "eval"}
        assert self._wandb is not None
        # video_path 指向已编码的 MP4，实际帧率已写入文件元数据，W&B 会忽略额外的 fps 参数。
        wandb_video = self._wandb.Video(video_path, format="mp4")
        payload = {f"{mode}/video": wandb_video}
        if mode == "eval":
            payload["eval/checkpoint_step"] = int(
                step if checkpoint_step is None else checkpoint_step
            )
        self._wandb.log(payload, step=step)
