#!/usr/bin/env python

"""Counterfactual none/Arm-to-View dataset primitives.

The collector branches the simulator from one physical state.  Both branches
receive candidates derived from one shared diffusion sample, then use the exact
same frozen ``none`` continuation policy.  Labels therefore describe the
effect of output correction rather than a change in diffusion noise.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from lerobot.common.policies.utils import populate_queues


ROUTER_CACHE_SCHEMA_VERSION = 1
ROUTER_CACHE_ARRAYS = {
    "global_condition": "global_condition.npy",
    "none_trajectory": "none_trajectory.npy",
    "arm_to_view_trajectory": "arm_to_view_trajectory.npy",
    "router_label": "router_label.npy",
    "sample_weight": "sample_weight.npy",
    "quality_none": "quality_none.npy",
    "quality_arm_to_view": "quality_arm_to_view.npy",
    "episode_seed": "episode_seed.npy",
    "decision_step": "decision_step.npy",
    "label_reason": "label_reason.npy",
}


@dataclass(frozen=True)
class BranchOutcome:
    success: bool
    max_stage: int
    reward: float
    steps: int

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [float(self.success), float(self.max_stage), self.reward],
            dtype=np.float32,
        )


@dataclass
class CandidateBatch:
    global_condition: Tensor
    none_trajectory: Tensor
    arm_to_view_trajectory: Tensor
    none_chunk: np.ndarray
    arm_to_view_chunk: np.ndarray


@dataclass
class EnvironmentSnapshot:
    physics_state: np.ndarray
    actuator_ctrl: np.ndarray
    current_step: int
    attributes: dict[str, Any]
    numpy_generator_state: dict[str, Any] | None


@dataclass
class RuntimeRngSnapshot:
    python_state: object
    numpy_state: tuple
    torch_cpu_state: Tensor
    torch_cuda_states: list[Tensor] | None


_TASK_STATE_FIELDS = (
    "terminated",
    "is_success",
    "right_has_grasped",
    "left_has_received",
    "right_has_released",
    "cylinder_was_grasped",
    "placement_checked",
    "_prev_metrics",
    "reward_debug",
)


def capture_environment_state(env) -> EnvironmentSnapshot:
    """Capture physics plus Python task state needed by shaped rewards."""
    unwrapped = env.unwrapped
    physics = unwrapped._physics
    attributes = {
        name: copy.deepcopy(getattr(unwrapped, name))
        for name in _TASK_STATE_FIELDS
        if hasattr(unwrapped, name)
    }
    np_generator = getattr(unwrapped, "np_random", None)
    numpy_generator_state = (
        copy.deepcopy(np_generator.bit_generator.state)
        if np_generator is not None
        else None
    )
    return EnvironmentSnapshot(
        physics_state=np.asarray(physics.get_state()).copy(),
        actuator_ctrl=np.asarray(physics.data.ctrl).copy(),
        current_step=int(getattr(unwrapped, "_current_step", 0)),
        attributes=attributes,
        numpy_generator_state=numpy_generator_state,
    )


def restore_environment_state(env, snapshot: EnvironmentSnapshot) -> dict:
    """Restore one branch point and return its pixel/state observation."""
    unwrapped = env.unwrapped
    physics = unwrapped._physics
    physics.set_state(snapshot.physics_state)
    physics.data.ctrl[:] = snapshot.actuator_ctrl
    unwrapped._current_step = int(snapshot.current_step)
    for name, value in snapshot.attributes.items():
        setattr(unwrapped, name, copy.deepcopy(value))
    if snapshot.numpy_generator_state is not None:
        unwrapped.np_random.bit_generator.state = copy.deepcopy(
            snapshot.numpy_generator_state
        )
    physics.forward()
    return unwrapped.get_obs()


def capture_runtime_rng() -> RuntimeRngSnapshot:
    return RuntimeRngSnapshot(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.random.get_rng_state(),
        torch_cuda_states=(
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    )


def restore_runtime_rng(snapshot: RuntimeRngSnapshot) -> None:
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.random.set_rng_state(snapshot.torch_cpu_state)
    if snapshot.torch_cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.torch_cuda_states)


def clone_policy_queues(policy) -> dict[str, deque]:
    return {
        key: deque(
            [
                value.detach().clone()
                if isinstance(value, Tensor)
                else copy.deepcopy(value)
                for value in queue
            ],
            maxlen=queue.maxlen,
        )
        for key, queue in policy._queues.items()
    }


def restore_policy_queues(policy, queues: dict[str, deque]) -> None:
    policy._queues = clone_queue_mapping(queues)


def clone_queue_mapping(queues: dict[str, deque]) -> dict[str, deque]:
    return {
        key: deque(
            [
                value.detach().clone()
                if isinstance(value, Tensor)
                else copy.deepcopy(value)
                for value in queue
            ],
            maxlen=queue.maxlen,
        )
        for key, queue in queues.items()
    }


def prepare_normalized_observation(policy, batch: dict[str, Tensor]):
    normalized = policy.normalize_inputs(batch)
    if policy.expected_image_keys:
        normalized = dict(normalized)
        normalized["observation.images"] = torch.stack(
            [normalized[key] for key in policy.expected_image_keys],
            dim=-4,
        )
    return normalized


def update_observation_queues(policy, batch: dict[str, Tensor]) -> None:
    """Advance only observation history without invoking diffusion."""
    normalized = prepare_normalized_observation(policy, batch)
    policy._queues = populate_queues(policy._queues, normalized)


def make_arm_to_view_candidate(
    diffusion,
    arm_trajectory: Tensor,
    view_trajectory: Tensor,
    *,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Apply only the trained Arm→View branch without mutating model config."""
    _, view_delta, _ = diffusion.output_corrector(
        arm_trajectory,
        view_trajectory,
        need_view_to_arm=False,
        need_arm_to_view=True,
    )
    corrected_view = view_trajectory + float(scale) * view_delta
    if diffusion.config.output_corrector_clamp_actions:
        limit = float(diffusion.config.clip_sample_range)
        corrected_view = corrected_view.clamp(-limit, limit)
    return arm_trajectory, corrected_view


@torch.inference_mode()
def next_none_action_and_candidates(
    policy,
    observation_batch: dict[str, Tensor],
    *,
    arm_to_view_scale: float,
) -> tuple[Tensor, CandidateBatch | None]:
    """Mirror ``select_action`` while exposing same-noise route candidates."""
    normalized = prepare_normalized_observation(policy, observation_batch)
    policy._queues = populate_queues(policy._queues, normalized)
    candidate_batch = None

    if len(policy._queues["action"]) == 0:
        stacked = {
            key: torch.stack(list(policy._queues[key]), dim=1)
            for key in normalized
            if key in policy._queues
        }
        diffusion = policy.diffusion
        global_condition = diffusion._prepare_global_conditioning(stacked)
        batch_size = stacked["observation.state"].shape[0]
        if batch_size != 1:
            raise ValueError("反事实Router采集器当前只支持单环境batch=1。")
        raw_arm, raw_view = diffusion.generate_baseline_full_trajectories(
            batch_size,
            global_cond=global_condition,
        )
        corrected_arm, corrected_view = make_arm_to_view_candidate(
            diffusion,
            raw_arm,
            raw_view,
            scale=arm_to_view_scale,
        )
        none_trajectory = diffusion.combine_action_heads(raw_arm, raw_view)
        corrected_trajectory = diffusion.combine_action_heads(
            corrected_arm,
            corrected_view,
        )
        start = int(policy.config.n_obs_steps) - 1
        end = start + int(policy.config.n_action_steps)
        none_chunk_normalized = none_trajectory[:, start:end]
        corrected_chunk_normalized = corrected_trajectory[:, start:end]
        none_chunk = policy.unnormalize_outputs(
            {"action": none_chunk_normalized}
        )["action"]
        corrected_chunk = policy.unnormalize_outputs(
            {"action": corrected_chunk_normalized}
        )["action"]
        policy._queues["action"].extend(none_chunk.transpose(0, 1))
        candidate_batch = CandidateBatch(
            global_condition=global_condition.detach().cpu(),
            none_trajectory=none_trajectory.detach().cpu(),
            arm_to_view_trajectory=corrected_trajectory.detach().cpu(),
            none_chunk=none_chunk[0].detach().cpu().numpy().copy(),
            arm_to_view_chunk=(
                corrected_chunk[0].detach().cpu().numpy().copy()
            ),
        )

    return policy._queues["action"].popleft(), candidate_batch


def extract_stage(info: dict) -> int:
    debug = info.get("reward_debug", {}) if isinstance(info, dict) else {}
    if isinstance(debug, dict):
        return int(debug.get("stage", 0))
    return 0


def make_counterfactual_label(
    none: BranchOutcome,
    arm_to_view: BranchOutcome,
    *,
    reward_margin: float,
) -> tuple[int, float, int]:
    """Return ``(label, weight, reason_code)``; -1 means ambiguous.

    Reason codes: 0 ambiguous, 1 success difference, 2 both-success none tie,
    3 task-stage difference, 4 return difference.
    """
    if none.success != arm_to_view.success:
        return (int(arm_to_view.success), 2.0, 1)
    if none.success and arm_to_view.success:
        return (0, 1.0, 2)
    if none.max_stage != arm_to_view.max_stage:
        return (int(arm_to_view.max_stage > none.max_stage), 1.5, 3)
    reward_difference = float(arm_to_view.reward - none.reward)
    if abs(reward_difference) > float(reward_margin):
        return (int(reward_difference > 0.0), 1.0, 4)
    return (-1, 0.0, 0)


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("schema_version") != ROUTER_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Router缓存版本不匹配: "
            f"{manifest.get('schema_version')} != "
            f"{ROUTER_CACHE_SCHEMA_VERSION}"
        )
    for filename in manifest["arrays"].values():
        if not (manifest_path.parent / filename).is_file():
            raise FileNotFoundError(
                f"Router缓存数组缺失: {manifest_path.parent / filename}"
            )
    return manifest_path, manifest


class CounterfactualRouterDataset(Dataset):
    """Memory or mmap-backed counterfactual route pairs."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        memory_limit_gb: float = 4.0,
        indices: np.ndarray | None = None,
        include_ambiguous: bool = False,
    ) -> None:
        self.manifest_path, self.manifest = _load_manifest(manifest_path)
        self.paths = {
            name: self.manifest_path.parent / filename
            for name, filename in self.manifest["arrays"].items()
        }
        total_bytes = sum(path.stat().st_size for path in self.paths.values())
        self.load_into_memory = total_bytes <= float(memory_limit_gb) * 1024**3
        mmap_mode = None if self.load_into_memory else "r"
        self.arrays = {
            name: np.load(path, mmap_mode=mmap_mode)
            for name, path in self.paths.items()
        }
        available = np.arange(int(self.manifest["num_samples"]))
        if not include_ambiguous:
            available = available[self.arrays["router_label"] >= 0]
        if indices is not None:
            requested = np.asarray(indices, dtype=np.int64)
            allowed = np.zeros(int(self.manifest["num_samples"]), dtype=bool)
            allowed[available] = True
            available = requested[allowed[requested]]
        self.indices = available
        if len(self.indices) == 0:
            raise ValueError("Router数据集没有可用的已标注样本。")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        index = int(self.indices[item])
        return {
            "global_condition": torch.from_numpy(
                np.asarray(
                    self.arrays["global_condition"][index],
                    dtype=np.float32,
                ).copy()
            ),
            "none_trajectory": torch.from_numpy(
                np.asarray(
                    self.arrays["none_trajectory"][index],
                    dtype=np.float32,
                ).copy()
            ),
            "arm_to_view_trajectory": torch.from_numpy(
                np.asarray(
                    self.arrays["arm_to_view_trajectory"][index],
                    dtype=np.float32,
                ).copy()
            ),
            "router_label": torch.tensor(
                float(self.arrays["router_label"][index]),
                dtype=torch.float32,
            ),
            "sample_weight": torch.tensor(
                float(self.arrays["sample_weight"][index]),
                dtype=torch.float32,
            ),
            "quality_none": torch.from_numpy(
                np.asarray(
                    self.arrays["quality_none"][index],
                    dtype=np.float32,
                ).copy()
            ),
            "quality_arm_to_view": torch.from_numpy(
                np.asarray(
                    self.arrays["quality_arm_to_view"][index],
                    dtype=np.float32,
                ).copy()
            ),
            "episode_seed": torch.tensor(
                int(self.arrays["episode_seed"][index]),
                dtype=torch.int64,
            ),
            "decision_step": torch.tensor(
                int(self.arrays["decision_step"][index]),
                dtype=torch.int64,
            ),
        }


def split_indices_by_episode_seed(
    manifest_path: str | Path,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole episodes; adjacent states never cross the boundary."""
    _, manifest = _load_manifest(manifest_path)
    path = Path(manifest_path).expanduser().resolve().parent
    labels = np.load(path / manifest["arrays"]["router_label"], mmap_mode="r")
    episode_seeds = np.load(
        path / manifest["arrays"]["episode_seed"],
        mmap_mode="r",
    )
    valid = np.flatnonzero(labels >= 0)
    unique_seeds = np.unique(episode_seeds[valid])
    if len(unique_seeds) < 2:
        raise ValueError("按episode划分训练/验证集至少需要2个episode seed。")
    rng = np.random.default_rng(int(seed))
    shuffled = unique_seeds.copy()
    rng.shuffle(shuffled)
    n_validation = max(
        1,
        min(
            len(shuffled) - 1,
            int(round(len(shuffled) * float(validation_fraction))),
        ),
    )
    validation_seeds = set(int(value) for value in shuffled[:n_validation])
    validation_mask = np.asarray(
        [int(value) in validation_seeds for value in episode_seeds],
        dtype=bool,
    )
    return (
        valid[~validation_mask[valid]],
        valid[validation_mask[valid]],
    )


def stable_cache_hash(payload: dict[str, Any], length: int = 12) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]
