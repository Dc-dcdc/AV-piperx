"""为每次专家/人在环采集 attempt 分配派生 seed 或评估失败 seed。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAX_UINT32_SEED = int(np.iinfo(np.uint32).max)
ENVIRONMENT_SEED_STRATEGY = "numpy_seed_sequence_v1"
REPLAY_ENVIRONMENT_SEED_STRATEGY = "direct_eval_seed_replay_v1"
SEED_MODE_SEQUENTIAL = "sequential"
SEED_MODE_REPLAY = "replay"
REPLAY_EXHAUSTED_STOP = "stop"
REPLAY_EXHAUSTED_LOOP = "loop"


class ReplaySeedsExhausted(RuntimeError):
    """失败 seed 队列已消费完毕。"""


@dataclass(frozen=True)
class ReplaySeedRecord:
    """从评估明细或简单 seed 列表中读取的一条重放记录。"""

    seed: int
    source_index: int
    evaluation_episode: int | None = None
    evaluation_success: bool | None = None
    evaluation_reward: float | None = None
    evaluation_steps: int | None = None


@dataclass(frozen=True)
class EpisodeSeedAssignment:
    """一次采集 attempt 实际使用的环境和策略 seed。"""

    attempt_index: int
    environment_seed: int | None
    policy_seed: int | None
    seed_source: str
    environment_seed_strategy: str
    replay_batch_index: int | None = None
    replay_index: int | None = None
    replay_record_index: int | None = None
    replay_repeat_index: int | None = None
    replay_cycle: int | None = None
    evaluation_source_index: int | None = None
    evaluation_episode: int | None = None
    evaluation_success: bool | None = None
    evaluation_reward: float | None = None
    evaluation_steps: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        """转为可直接写入 JSON 的元数据。"""
        return {
            "attempt_index": int(self.attempt_index),
            "environment_seed": self.environment_seed,
            "policy_seed": self.policy_seed,
            "seed_source": self.seed_source,
            "environment_seed_strategy": self.environment_seed_strategy,
            "replay_batch_index": self.replay_batch_index,
            "replay_index": self.replay_index,
            "replay_record_index": self.replay_record_index,
            "replay_repeat_index": self.replay_repeat_index,
            "replay_cycle": self.replay_cycle,
            "evaluation_source_index": self.evaluation_source_index,
            "evaluation_episode": self.evaluation_episode,
            "evaluation_success": self.evaluation_success,
            "evaluation_reward": self.evaluation_reward,
            "evaluation_steps": self.evaluation_steps,
        }


def normalize_base_seed(seed: int | None) -> int | None:
    """校验并规范化进程基础 seed；``None`` 表示不固定。"""
    if seed is None:
        return None
    normalized = int(seed)
    if not 0 <= normalized <= MAX_UINT32_SEED:
        raise ValueError(
            f"random_seed must be between 0 and {MAX_UINT32_SEED}, got {normalized}."
        )
    return normalized


def normalize_seed_mode(mode: str) -> str:
    """校验采集 seed 模式。"""
    normalized = str(mode).strip().lower()
    if normalized not in (SEED_MODE_SEQUENTIAL, SEED_MODE_REPLAY):
        raise ValueError(
            "seed_mode must be 'sequential' or 'replay', "
            f"got {mode!r}."
        )
    return normalized


def normalize_replay_on_exhausted(value: str) -> str:
    """校验失败 seed 用尽后的行为。"""
    normalized = str(value).strip().lower()
    if normalized not in (REPLAY_EXHAUSTED_STOP, REPLAY_EXHAUSTED_LOOP):
        raise ValueError(
            "replay_on_exhausted must be 'stop' or 'loop', "
            f"got {value!r}."
        )
    return normalized


def _replay_items(payload: Any, path: Path) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("episodes", "seeds"):
            items = payload.get(key)
            if isinstance(items, list):
                return items
    raise ValueError(
        "Replay seed file must contain a JSON list, or an object with an "
        f"'episodes'/'seeds' list: {path}"
    )


def load_replay_seed_records(
    path: str | Path,
    *,
    failed_only: bool = True,
) -> list[ReplaySeedRecord]:
    """读取 ``episode_results.json`` 或简单 seed JSON 列表。"""
    replay_path = Path(path)
    try:
        with replay_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Replay seed file does not exist: {replay_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid replay seed JSON: {replay_path}: {exc}") from exc

    records: list[ReplaySeedRecord] = []
    for source_index, item in enumerate(_replay_items(payload, replay_path)):
        if isinstance(item, bool):
            raise ValueError(
                f"Replay seed at index {source_index} must be an integer, not bool."
            )

        if isinstance(item, (int, np.integer)):
            raw_seed = item
            episode = None
            success = None
            reward = None
            steps = None
        elif isinstance(item, dict):
            if "seed" not in item:
                raise ValueError(
                    f"Replay record at index {source_index} is missing 'seed'."
                )
            raw_seed = item["seed"]
            if isinstance(raw_seed, bool):
                raise ValueError(
                    f"Replay seed at index {source_index} must be an integer, not bool."
                )
            success = item.get("success")
            if success is not None and not isinstance(success, (bool, np.bool_)):
                raise ValueError(
                    f"Replay record success at index {source_index} must be boolean."
                )
            if failed_only and success is True:
                continue
            episode = item.get("episode")
            reward = item.get("reward")
            steps = item.get("steps")
        else:
            raise ValueError(
                f"Unsupported replay record at index {source_index}: {item!r}."
            )

        normalized_seed = normalize_base_seed(raw_seed)
        if normalized_seed is None:
            raise ValueError(f"Replay seed at index {source_index} cannot be null.")
        records.append(
            ReplaySeedRecord(
                seed=normalized_seed,
                source_index=int(source_index),
                evaluation_episode=None if episode is None else int(episode),
                evaluation_success=None if success is None else bool(success),
                evaluation_reward=None if reward is None else float(reward),
                evaluation_steps=None if steps is None else int(steps),
            )
        )

    if not records:
        filter_description = " failed" if failed_only else ""
        raise ValueError(f"Replay seed file contains no{filter_description} seeds: {replay_path}")
    return records


def replay_seed_digest(records: list[ReplaySeedRecord]) -> str:
    """生成所选 seed 顺序的稳定摘要，用于保护 append 重放游标。"""
    payload = [
        {
            "seed": record.seed,
            "source_index": record.source_index,
            "evaluation_episode": record.evaluation_episode,
            "evaluation_success": record.evaluation_success,
        }
        for record in records
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_environment_seed(base_seed: int | None, attempt_index: int) -> int | None:
    """根据基础 seed 和 attempt 编号派生稳定的 uint32 环境 seed。"""
    normalized_seed = normalize_base_seed(base_seed)
    normalized_index = int(attempt_index)
    if normalized_index < 0:
        raise ValueError(f"attempt_index must be non-negative, got {normalized_index}.")
    if normalized_seed is None:
        return None

    seed_sequence = np.random.SeedSequence([normalized_seed, normalized_index])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def validate_append_seed_configuration(
    *,
    existing_metadata: dict,
    base_seed: int | None,
    saved_episode_count: int,
) -> dict | None:
    """校验追加采集的 seed 配置，并保留旧版固定 seed 记录。"""
    if not existing_metadata:
        return None

    normalized_seed = normalize_base_seed(base_seed)
    previous_seed_marker = object()
    previous_seed = existing_metadata.get(
        "base_random_seed",
        existing_metadata.get("random_seed", previous_seed_marker),
    )
    if previous_seed is not previous_seed_marker:
        normalized_previous_seed = normalize_base_seed(previous_seed)
        if normalized_previous_seed != normalized_seed:
            raise ValueError(
                "Cannot append with a different base random seed: "
                f"existing={normalized_previous_seed}, requested={normalized_seed}."
            )

    expected_strategy = (
        ENVIRONMENT_SEED_STRATEGY if normalized_seed is not None else "unseeded"
    )
    previous_strategy = existing_metadata.get(
        "environment_seed_strategy",
        existing_metadata.get("episode_seed_strategy"),
    )
    if previous_strategy is not None and str(previous_strategy) != expected_strategy:
        raise ValueError(
            "Cannot append with a different environment seed strategy: "
            f"existing={previous_strategy!r}, requested={expected_strategy!r}."
        )

    legacy_record = existing_metadata.get("legacy_seed_configuration")
    if previous_strategy is None and legacy_record is None:
        legacy_record = {
            "strategy": (
                "legacy_fixed_reset_seed"
                if bool(existing_metadata.get("fixed_reset_seed", False))
                else "legacy_unknown_reset_seed"
            ),
            "base_random_seed": (
                None
                if previous_seed is previous_seed_marker
                else normalize_base_seed(previous_seed)
            ),
            "saved_episode_count": int(saved_episode_count),
            "attempt_history": "unavailable; next cursor recovered as a lower bound",
        }
    return legacy_record


def resolve_next_attempt_index(
    *,
    episode_index: int,
    existing_metadata: dict,
    existing_infos: list[dict],
) -> int:
    """综合元数据和已保存轨迹，恢复下一个 attempt 编号。"""
    candidates = [int(episode_index)]

    for key in ("next_attempt_index", "attempted_episodes"):
        value = existing_metadata.get(key)
        if value is None:
            continue
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"metadata.{key} must be non-negative, got {parsed}.")
        candidates.append(parsed)

    for info in existing_infos:
        value = info.get("attempt_index")
        if value is None:
            continue
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"episode attempt_index must be non-negative, got {parsed}.")
        candidates.append(parsed + 1)

    return max(candidates)


@dataclass
class EpisodeSeedStream:
    """按 attempt 顺序分配派生环境 seed。"""

    base_seed: int | None
    next_attempt_index: int = 0

    def __post_init__(self) -> None:
        """规范化基础 seed 和恢复后的 attempt 游标。"""
        self.base_seed = normalize_base_seed(self.base_seed)
        self.next_attempt_index = int(self.next_attempt_index)
        if self.next_attempt_index < 0:
            raise ValueError(
                "next_attempt_index must be non-negative, "
                f"got {self.next_attempt_index}."
            )

    def take(self) -> tuple[int, int | None]:
        """返回当前 attempt 及其环境 seed，然后推进游标。"""
        attempt_index = self.next_attempt_index
        environment_seed = derive_environment_seed(self.base_seed, attempt_index)
        self.next_attempt_index += 1
        return attempt_index, environment_seed

    def take_assignment(self) -> EpisodeSeedAssignment:
        """返回兼容旧顺序采集行为的 seed 分配记录。"""
        attempt_index, environment_seed = self.take()
        return EpisodeSeedAssignment(
            attempt_index=attempt_index,
            environment_seed=environment_seed,
            policy_seed=None,
            seed_source="sequential_derived" if environment_seed is not None else "unseeded",
            environment_seed_strategy=(
                ENVIRONMENT_SEED_STRATEGY if environment_seed is not None else "unseeded"
            ),
        )


@dataclass
class ReplaySeedStream:
    """按评估失败 seed 列表直接分配环境 seed。"""

    records: list[ReplaySeedRecord]
    next_attempt_index: int = 0
    next_replay_index: int = 0
    replay_batch_index: int = 0
    repeat: int = 1
    on_exhausted: str = REPLAY_EXHAUSTED_STOP
    replay_policy_rng: bool = True

    def __post_init__(self) -> None:
        self.records = list(self.records)
        if not self.records:
            raise ValueError("ReplaySeedStream requires at least one seed record.")
        self.next_attempt_index = int(self.next_attempt_index)
        self.next_replay_index = int(self.next_replay_index)
        self.replay_batch_index = int(self.replay_batch_index)
        self.repeat = int(self.repeat)
        self.on_exhausted = normalize_replay_on_exhausted(self.on_exhausted)
        self.replay_policy_rng = bool(self.replay_policy_rng)
        if self.next_attempt_index < 0:
            raise ValueError(
                "next_attempt_index must be non-negative, "
                f"got {self.next_attempt_index}."
            )
        if self.next_replay_index < 0:
            raise ValueError(
                "next_replay_index must be non-negative, "
                f"got {self.next_replay_index}."
            )
        if self.replay_batch_index < 0:
            raise ValueError(
                "replay_batch_index must be non-negative, "
                f"got {self.replay_batch_index}."
            )
        if self.repeat < 1:
            raise ValueError(f"replay_repeat must be at least 1, got {self.repeat}.")

    @property
    def allocations_per_cycle(self) -> int:
        return len(self.records) * self.repeat

    @property
    def exhausted(self) -> bool:
        return (
            self.on_exhausted == REPLAY_EXHAUSTED_STOP
            and self.next_replay_index >= self.allocations_per_cycle
        )

    def take_assignment(self) -> EpisodeSeedAssignment:
        """领取一个直接重放 seed；stop 模式用尽后抛出明确异常。"""
        if self.exhausted:
            raise ReplaySeedsExhausted(
                f"Replay seed queue exhausted after {self.next_replay_index} attempts."
            )

        replay_index = self.next_replay_index
        position = replay_index % self.allocations_per_cycle
        record_index = position // self.repeat
        repeat_index = position % self.repeat
        cycle = replay_index // self.allocations_per_cycle
        record = self.records[record_index]
        attempt_index = self.next_attempt_index

        self.next_attempt_index += 1
        self.next_replay_index += 1
        return EpisodeSeedAssignment(
            attempt_index=attempt_index,
            environment_seed=record.seed,
            policy_seed=record.seed if self.replay_policy_rng else None,
            seed_source="eval_failure_replay",
            environment_seed_strategy=REPLAY_ENVIRONMENT_SEED_STRATEGY,
            replay_batch_index=self.replay_batch_index,
            replay_index=replay_index,
            replay_record_index=record_index,
            replay_repeat_index=repeat_index,
            replay_cycle=cycle,
            evaluation_source_index=record.source_index,
            evaluation_episode=record.evaluation_episode,
            evaluation_success=record.evaluation_success,
            evaluation_reward=record.evaluation_reward,
            evaluation_steps=record.evaluation_steps,
        )
