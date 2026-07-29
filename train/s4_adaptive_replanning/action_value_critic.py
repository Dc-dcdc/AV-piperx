"""离线动作价值 Critic、原始轨迹索引与 episode 级数据划分。

该模块与现有三决策 ``dqn.py`` 相互独立。Critic 只接收冻结视觉底座已经
编码好的两帧视觉特征、两帧归一化关节状态和一条归一化动作，并输出一个
标量 Q(s, a)。训练入口负责用数据集中真实的下一动作构造单步 SARSA 目标。
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ActionValueCriticConfig:
    """动作价值网络的固定输入维度和 MLP 结构。"""

    visual_feature_dim: int
    joint_history_dim: int
    action_dim: int
    visual_embed_dim: int = 256
    state_embed_dim: int = 128
    action_embed_dim: int = 128
    hidden_dims: tuple[int, ...] = (512, 256)
    output_activation: str = "sigmoid"
    initial_q: float = 0.05

    def __post_init__(self) -> None:
        positive_values = {
            "visual_feature_dim": self.visual_feature_dim,
            "joint_history_dim": self.joint_history_dim,
            "action_dim": self.action_dim,
            "visual_embed_dim": self.visual_embed_dim,
            "state_embed_dim": self.state_embed_dim,
            "action_embed_dim": self.action_embed_dim,
        }
        for name, value in positive_values.items():
            if int(value) <= 0:
                raise ValueError(f"{name} 必须大于0，当前为 {value}")
        if not self.hidden_dims or any(int(dim) <= 0 for dim in self.hidden_dims):
            raise ValueError(
                f"hidden_dims 必须包含正整数，当前为 {self.hidden_dims}"
            )
        if self.output_activation not in {"sigmoid", "identity"}:
            raise ValueError(
                "output_activation只支持sigmoid或identity，当前为"
                f"{self.output_activation}"
            )
        if self.output_activation == "sigmoid" and not (
            0.0 < float(self.initial_q) < 1.0
        ):
            raise ValueError("sigmoid Critic的initial_q必须位于(0,1)")


@dataclass(frozen=True)
class RelativeReplanningConfig:
    """基于Critic相对价值变化的通用重规划判定参数。"""

    gamma: float = 0.99
    anchor_ratio_threshold: float = 0.70
    local_drop_threshold: float = 0.15
    consecutive_bad_steps: int = 2
    ema_alpha: float = 0.20
    min_reference_q: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("gamma必须位于(0,1]")
        if not 0.0 < float(self.anchor_ratio_threshold) <= 1.0:
            raise ValueError("anchor_ratio_threshold必须位于(0,1]")
        if float(self.local_drop_threshold) < 0.0:
            raise ValueError("local_drop_threshold不能为负数")
        if int(self.consecutive_bad_steps) <= 0:
            raise ValueError("consecutive_bad_steps必须大于0")
        if not 0.0 < float(self.ema_alpha) <= 1.0:
            raise ValueError("ema_alpha必须位于(0,1]")
        if not 0.0 < float(self.min_reference_q) < 1.0:
            raise ValueError("min_reference_q必须位于(0,1)")


@dataclass(frozen=True)
class RelativeReplanningDecision:
    """一次Critic评分得到的相对价值诊断。"""

    should_replan: bool
    raw_q: float
    smoothed_q: float
    expected_q: float
    anchor_ratio: float
    normalized_td_change: float
    anchor_bad: bool
    local_bad: bool
    consecutive_bad_steps: int
    steps_since_inference: int


class RelativeValueReplanningDecider:
    """比较Critic连续输出，不依赖绝对时间步阈值。"""

    def __init__(self, config: RelativeReplanningConfig):
        self.config = config
        self.reset()

    @staticmethod
    def _validate_q(value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Critic Q必须为有限值，当前为{value}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"相对价值判定要求Q位于[0,1]，当前为{value}；"
                "请使用sigmoid Critic并重新训练"
            )
        return value

    def reset(self) -> None:
        """清除当前动作块的锚点和连续异常状态。"""
        self.anchor_q: float | None = None
        self.smoothed_q: float | None = None
        self.previous_smoothed_q: float | None = None
        self.steps_since_inference = 0
        self.bad_step_count = 0

    def start_chunk(self, q_value: float) -> RelativeReplanningDecision:
        """在扩散策略刚生成新动作块时记录Critic价值锚点。"""
        q_value = self._validate_q(q_value)
        self.anchor_q = q_value
        self.smoothed_q = q_value
        self.previous_smoothed_q = q_value
        self.steps_since_inference = 0
        self.bad_step_count = 0
        return RelativeReplanningDecision(
            should_replan=False,
            raw_q=q_value,
            smoothed_q=q_value,
            expected_q=q_value,
            anchor_ratio=1.0,
            normalized_td_change=0.0,
            anchor_bad=False,
            local_bad=False,
            consecutive_bad_steps=0,
            steps_since_inference=0,
        )

    def evaluate(
        self,
        q_value: float,
        *,
        previous_reward: float = 0.0,
        previous_done: bool = False,
    ) -> RelativeReplanningDecision:
        """比较当前Q与锚点预测和上一步Bellman关系。"""
        q_value = self._validate_q(q_value)
        if self.anchor_q is None or self.smoothed_q is None:
            return self.start_chunk(q_value)

        previous_q = float(self.smoothed_q)
        alpha = float(self.config.ema_alpha)
        smoothed_q = (1.0 - alpha) * previous_q + alpha * q_value
        self.steps_since_inference += 1
        expected_q = min(
            1.0,
            float(self.anchor_q)
            / float(self.config.gamma) ** self.steps_since_inference,
        )
        anchor_ratio = smoothed_q / max(
            expected_q,
            float(self.config.min_reference_q),
        )
        td_change = (
            float(previous_reward)
            + float(self.config.gamma)
            * (0.0 if previous_done else smoothed_q)
            - previous_q
        )
        normalized_td_change = td_change / max(
            abs(previous_q),
            float(self.config.min_reference_q),
        )
        anchor_bad = (
            anchor_ratio < float(self.config.anchor_ratio_threshold)
        )
        local_bad = (
            normalized_td_change
            < -float(self.config.local_drop_threshold)
        )
        if anchor_bad or local_bad:
            self.bad_step_count += 1
        else:
            self.bad_step_count = 0
        should_replan = (
            self.bad_step_count >= int(self.config.consecutive_bad_steps)
        )
        self.previous_smoothed_q = previous_q
        self.smoothed_q = smoothed_q
        return RelativeReplanningDecision(
            should_replan=should_replan,
            raw_q=q_value,
            smoothed_q=smoothed_q,
            expected_q=expected_q,
            anchor_ratio=anchor_ratio,
            normalized_td_change=normalized_td_change,
            anchor_bad=anchor_bad,
            local_bad=local_bad,
            consecutive_bad_steps=self.bad_step_count,
            steps_since_inference=self.steps_since_inference,
        )


def _make_projector(input_dim: int, output_dim: int) -> nn.Sequential:
    """构造视觉、状态或动作分支使用的轻量投影器。"""
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.SiLU(),
    )


class ActionValueCritic(nn.Module):
    """融合冻结视觉特征、两帧关节状态和下一动作并输出标量 Q 值。"""

    def __init__(self, config: ActionValueCriticConfig):
        super().__init__()
        self.config = config
        self.visual_projector = _make_projector(
            config.visual_feature_dim,
            config.visual_embed_dim,
        )
        self.state_projector = _make_projector(
            config.joint_history_dim,
            config.state_embed_dim,
        )
        self.action_projector = _make_projector(
            config.action_dim,
            config.action_embed_dim,
        )

        fusion_dim = (
            config.visual_embed_dim
            + config.state_embed_dim
            + config.action_embed_dim
        )
        layers: list[nn.Module] = []
        input_dim = fusion_dim
        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.SiLU(),
                ]
            )
            input_dim = int(hidden_dim)
        output_layer = nn.Linear(input_dim, 1)
        if config.output_activation == "sigmoid":
            nn.init.normal_(output_layer.weight, mean=0.0, std=1.0e-3)
            initial_logit = math.log(
                float(config.initial_q) / (1.0 - float(config.initial_q))
            )
            nn.init.constant_(output_layer.bias, initial_logit)
        layers.append(output_layer)
        self.fusion = nn.Sequential(*layers)

    @staticmethod
    def _validate_matrix(
        name: str,
        value: torch.Tensor,
        expected_dim: int,
    ) -> None:
        if not torch.is_tensor(value) or value.ndim != 2:
            shape = getattr(value, "shape", None)
            raise ValueError(f"{name} 必须是二维张量[B,D]，当前为 {shape}")
        if int(value.shape[-1]) != int(expected_dim):
            raise ValueError(
                f"{name} 最后一维应为 {expected_dim}，当前为 {value.shape[-1]}"
            )

    def forward(
        self,
        visual_features: torch.Tensor,
        joint_history: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """返回形状为 ``[B]`` 的标量Q；默认限制在 ``[0,1]``。"""
        self._validate_matrix(
            "visual_features",
            visual_features,
            self.config.visual_feature_dim,
        )
        self._validate_matrix(
            "joint_history",
            joint_history,
            self.config.joint_history_dim,
        )
        self._validate_matrix("action", action, self.config.action_dim)
        batch_sizes = {
            int(visual_features.shape[0]),
            int(joint_history.shape[0]),
            int(action.shape[0]),
        }
        if len(batch_sizes) != 1:
            raise ValueError("Critic 三类输入的 batch size 必须一致")

        fused = torch.cat(
            [
                self.visual_projector(visual_features),
                self.state_projector(joint_history),
                self.action_projector(action),
            ],
            dim=-1,
        )
        q_value = self.fusion(fused).squeeze(-1)
        if self.config.output_activation == "sigmoid":
            q_value = torch.sigmoid(q_value)
        return q_value


def make_target_critic(critic: ActionValueCritic) -> ActionValueCritic:
    """复制一个冻结的 Target Critic。"""
    target = copy.deepcopy(critic)
    target.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def polyak_update(
    online_critic: ActionValueCritic,
    target_critic: ActionValueCritic,
    tau: float,
) -> None:
    """按 ``target=(1-tau)*target+tau*online`` 软更新目标网络。"""
    tau = float(tau)
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau 必须位于(0,1]，当前为 {tau}")
    online_parameters = dict(online_critic.named_parameters())
    target_parameters = dict(target_critic.named_parameters())
    if online_parameters.keys() != target_parameters.keys():
        raise ValueError("Online Critic 与 Target Critic 参数结构不一致")
    for name, target_parameter in target_parameters.items():
        target_parameter.lerp_(online_parameters[name], tau)


@dataclass
class ActionValueEpisode:
    """一条已经完成基本一致性校验的原始采集 episode。"""

    path: Path
    episode_index: int
    execution_steps: int
    success: bool
    camera_names: tuple[str, ...]
    observation_state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    global_episode_index: int = -1

    @property
    def transition_count(self) -> int:
        return int(self.action.shape[0])


def _load_episode(
    episode_dir: Path,
    camera_names: Sequence[str],
) -> ActionValueEpisode:
    """读取一条 episode 的轻量数组，图像保持按需加载。"""
    info_path = episode_dir / "info.json"
    arrays_path = episode_dir / "arrays.npz"
    if not info_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(
            f"episode目录不完整，需要info.json和arrays.npz: {episode_dir}"
        )
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = {
            "observation_state",
            "action",
            "reward",
            "terminated",
            "truncated",
        }
        missing = required - set(arrays.files)
        if missing:
            raise KeyError(f"{arrays_path} 缺少字段: {sorted(missing)}")
        observation_state = np.asarray(
            arrays["observation_state"],
            dtype=np.float32,
        ).copy()
        action = np.asarray(arrays["action"], dtype=np.float32).copy()
        reward = np.asarray(arrays["reward"], dtype=np.float32).reshape(-1).copy()
        terminated = np.asarray(
            arrays["terminated"],
            dtype=np.bool_,
        ).reshape(-1).copy()
        truncated = np.asarray(
            arrays["truncated"],
            dtype=np.bool_,
        ).reshape(-1).copy()

    transition_count = int(action.shape[0])
    if action.ndim != 2:
        raise ValueError(f"{arrays_path}: action应为[T,D]，当前为{action.shape}")
    if observation_state.ndim != 2:
        raise ValueError(
            f"{arrays_path}: observation_state应为[T+1,D]，"
            f"当前为{observation_state.shape}"
        )
    if observation_state.shape[0] != transition_count + 1:
        raise ValueError(
            f"{arrays_path}: 观测数必须比动作数多1，"
            f"当前为{observation_state.shape[0]}和{transition_count}"
        )
    for name, values in {
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }.items():
        if values.shape[0] != transition_count:
            raise ValueError(
                f"{arrays_path}: {name}长度应为{transition_count}，"
                f"当前为{values.shape[0]}"
            )
    if transition_count <= 0:
        raise ValueError(f"{arrays_path}: episode中没有transition")
    if np.any((reward < 0.0) | (reward > 1.0)):
        raise ValueError(f"{arrays_path}: 稀疏reward必须位于[0,1]")
    if not bool(terminated[-1] or truncated[-1]):
        raise ValueError(
            f"{arrays_path}: 最后一个transition必须terminated或truncated"
        )

    expected_frames = transition_count + 1
    for camera in camera_names:
        camera_dir = episode_dir / "images" / camera
        if not camera_dir.is_dir():
            raise FileNotFoundError(f"缺少相机目录: {camera_dir}")
        first_path = camera_dir / "000000.jpg"
        last_path = camera_dir / f"{expected_frames - 1:06d}.jpg"
        if not first_path.is_file() or not last_path.is_file():
            raise FileNotFoundError(
                f"{camera_dir} 图像帧不完整，需要000000到"
                f"{expected_frames - 1:06d}.jpg"
            )

    return ActionValueEpisode(
        path=episode_dir,
        episode_index=int(info["episode_index"]),
        execution_steps=int(info["execution_steps"]),
        success=bool(info["success"]),
        camera_names=tuple(camera_names),
        observation_state=observation_state,
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        global_episode_index=int(
            info.get("global_episode_index", info["episode_index"])
        ),
    )


def load_action_value_episodes(
    dataset_dir: str | Path,
    execution_steps: int | Sequence[int] | None = None,
) -> tuple[list[ActionValueEpisode], dict]:
    """递归加载episode，并可选择一个或多个固定执行步长。"""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    metadata_path = dataset_dir / "metadata.json"
    episodes_dir = dataset_dir / "episodes"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"找不到动作价值数据集metadata: {metadata_path}")
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"找不到动作价值episodes目录: {episodes_dir}")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    camera_names = tuple(metadata.get("camera_names", ()))
    if not camera_names:
        raise ValueError(f"{metadata_path} 中没有camera_names")

    episodes = [
        _load_episode(path, camera_names)
        for path in sorted(
            episodes_dir.rglob(
                "episode_[0-9][0-9][0-9][0-9][0-9][0-9]"
            )
        )
        if path.is_dir()
    ]
    if not episodes:
        raise FileNotFoundError(f"{episodes_dir} 中没有完整episode")

    if execution_steps is not None:
        if isinstance(execution_steps, (int, np.integer)):
            requested_steps = [int(execution_steps)]
        else:
            requested_steps = [int(value) for value in execution_steps]
        requested_steps = list(dict.fromkeys(requested_steps))
        if not requested_steps or any(value <= 0 for value in requested_steps):
            raise ValueError(
                "训练选择的execution_steps必须包含至少一个正整数"
            )
        available_steps = sorted(
            {episode.execution_steps for episode in episodes}
        )
        missing_steps = sorted(set(requested_steps) - set(available_steps))
        if missing_steps:
            raise ValueError(
                f"数据集中不存在execution_steps={missing_steps}；"
                f"当前可选值为{available_steps}"
            )
        requested_set = set(requested_steps)
        episodes = [
            episode
            for episode in episodes
            if episode.execution_steps in requested_set
        ]
    return episodes, metadata


def split_episodes_by_execution_steps(
    episodes: Sequence[ActionValueEpisode],
    validation_ratio: float,
    seed: int,
) -> tuple[list[ActionValueEpisode], list[ActionValueEpisode]]:
    """按执行步长分组后进行 episode 级随机划分，不按成功率重采样。"""
    validation_ratio = float(validation_ratio)
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError(
            f"validation_ratio必须位于[0,1)，当前为{validation_ratio}"
        )
    groups: dict[int, list[ActionValueEpisode]] = {}
    for episode in episodes:
        groups.setdefault(episode.execution_steps, []).append(episode)

    rng = random.Random(int(seed))
    train_episodes: list[ActionValueEpisode] = []
    validation_episodes: list[ActionValueEpisode] = []
    for execution_steps in sorted(groups):
        group = list(groups[execution_steps])
        rng.shuffle(group)
        if validation_ratio == 0.0 or len(group) <= 1:
            validation_count = 0
        else:
            validation_count = max(1, round(len(group) * validation_ratio))
            validation_count = min(validation_count, len(group) - 1)
        validation_episodes.extend(group[:validation_count])
        train_episodes.extend(group[validation_count:])

    if not train_episodes:
        raise ValueError("episode划分后训练集为空")
    return train_episodes, validation_episodes


class ActionValueTransitionDataset(Dataset):
    """按 transition 随机读取三帧观测，供当前/下一两帧历史共享中间帧。"""

    def __init__(self, episodes: Iterable[ActionValueEpisode]):
        self.episodes = list(episodes)
        if not self.episodes:
            raise ValueError("ActionValueTransitionDataset需要至少一个episode")
        self.camera_names = self.episodes[0].camera_names
        self.sample_index = [
            (episode_position, transition_index)
            for episode_position, episode in enumerate(self.episodes)
            for transition_index in range(episode.transition_count)
        ]

    def __len__(self) -> int:
        return len(self.sample_index)

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        image = np.asarray(imageio.imread(path))
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"图像必须为HWC RGB格式，当前{path}为{image.shape}")
        return np.asarray(image, dtype=np.uint8)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_position, transition_index = self.sample_index[index]
        episode = self.episodes[episode_position]
        previous_index = max(0, transition_index - 1)
        frame_indices = (
            previous_index,
            transition_index,
            transition_index + 1,
        )

        frames = []
        for frame_index in frame_indices:
            camera_frames = [
                self._read_rgb(
                    episode.path
                    / "images"
                    / camera
                    / f"{frame_index:06d}.jpg"
                )
                for camera in self.camera_names
            ]
            frames.append(np.stack(camera_frames, axis=0))
        images = np.stack(frames, axis=0)  # [3,N,H,W,C]

        next_action_index = transition_index + 1
        if next_action_index < episode.transition_count:
            next_action = episode.action[next_action_index]
        else:
            next_action = np.zeros_like(episode.action[transition_index])
        done = bool(
            episode.terminated[transition_index]
            or episode.truncated[transition_index]
        )

        return {
            "images": torch.from_numpy(images),
            "joint_states": torch.from_numpy(
                episode.observation_state[list(frame_indices)]
            ),
            "action": torch.from_numpy(episode.action[transition_index]),
            "next_action": torch.from_numpy(next_action),
            "reward": torch.tensor(
                episode.reward[transition_index],
                dtype=torch.float32,
            ),
            "done": torch.tensor(done, dtype=torch.float32),
            "episode_success": torch.tensor(
                float(episode.success),
                dtype=torch.float32,
            ),
            "execution_steps": torch.tensor(
                episode.execution_steps,
                dtype=torch.int64,
            ),
        }
