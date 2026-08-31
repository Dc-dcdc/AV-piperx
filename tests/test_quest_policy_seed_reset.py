import unittest
from unittest.mock import patch

from data_collect.expert_data_collection.episode_seeding import (
    ENVIRONMENT_SEED_STRATEGY,
    REPLAY_ENVIRONMENT_SEED_STRATEGY,
    EpisodeSeedAssignment,
)
from data_collect.expert_data_collection.quest_policy_collect import (
    reset_control_state,
    resolve_append_replay_state,
)


class _RecordingSpace:
    def __init__(self, name: str, calls: list[tuple]) -> None:
        self.name = name
        self.calls = calls

    def seed(self, seed: int) -> None:
        self.calls.append((self.name, seed))


class _RecordingEnv:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls
        self.action_space = _RecordingSpace("action_space", calls)
        self.observation_space = _RecordingSpace("observation_space", calls)

    def reset(self, *, seed=None):
        self.calls.append(("env.reset", seed))
        return {"agent_pos": [0.0]}, {}


class _RecordingReset:
    def __init__(self, name: str, calls: list[tuple]) -> None:
        self.name = name
        self.calls = calls

    def reset(self, *args, **kwargs) -> None:
        self.calls.append((self.name, args, kwargs))


class QuestPolicySeedResetTest(unittest.TestCase):
    def test_replay_matches_single_environment_evaluation_seed_order(self) -> None:
        calls = []
        assignment = EpisodeSeedAssignment(
            attempt_index=0,
            environment_seed=106,
            policy_seed=106,
            seed_source="eval_failure_replay",
            environment_seed_strategy=REPLAY_ENVIRONMENT_SEED_STRATEGY,
        )
        env = _RecordingEnv(calls)

        with patch(
            "data_collect.expert_data_collection.quest_policy_collect.set_random_seed",
            side_effect=lambda seed: calls.append(("runtime", seed)),
        ):
            reset_control_state(
                env,
                _RecordingReset("policy.reset", calls),
                _RecordingReset("quest.reset", calls),
                _RecordingReset("ik.reset", calls),
                seed_assignment=assignment,
            )

        self.assertEqual(
            calls,
            [
                ("runtime", 106),
                ("action_space", 106),
                ("observation_space", 106),
                ("env.reset", 106),
                ("policy.reset", (), {}),
                ("runtime", 106),
                ("quest.reset", (), {}),
                ("ik.reset", (), {"active": False}),
            ],
        )

    def test_exhausted_replay_queue_can_start_a_new_batch_in_same_run(self) -> None:
        state = resolve_append_replay_state(
            existing_metadata={
                "replay_batch_index": 0,
                "replay_batch_start_attempt_index": 0,
                "replay_seed_file": "/tmp/old.json",
                "replay_seed_digest": "old",
                "replay_failed_only": True,
                "replay_repeat": 1,
                "replay_policy_rng": True,
                "replay_on_exhausted": "stop",
                "replay_seed_count": 2,
                "replay_seeds": [10, 11],
                "next_replay_index": 2,
                "next_attempt_index": 2,
            },
            seed_digest="new",
            failed_only=True,
            repeat=1,
            on_exhausted="stop",
            replay_policy_rng=True,
        )

        self.assertTrue(state["source_changed"])
        self.assertEqual(state["next_replay_index"], 0)
        self.assertEqual(state["replay_batch_index"], 1)
        self.assertEqual(state["replay_batch_start_attempt_index"], 2)
        self.assertEqual(len(state["replay_seed_history"]), 1)
        self.assertEqual(state["replay_seed_history"][0]["replay_seed_digest"], "old")

    def test_replay_source_change_rejects_an_unfinished_queue(self) -> None:
        with self.assertRaisesRegex(ValueError, "before the current queue is exhausted"):
            resolve_append_replay_state(
                existing_metadata={
                    "replay_seed_digest": "old",
                    "replay_failed_only": True,
                    "replay_repeat": 1,
                    "replay_policy_rng": True,
                    "replay_on_exhausted": "stop",
                    "replay_seed_count": 3,
                    "next_replay_index": 2,
                },
                seed_digest="new",
                failed_only=True,
                repeat=1,
                on_exhausted="stop",
                replay_policy_rng=True,
            )

    def test_sequential_mode_preserves_continuous_policy_rng(self) -> None:
        calls = []
        assignment = EpisodeSeedAssignment(
            attempt_index=0,
            environment_seed=123,
            policy_seed=None,
            seed_source="sequential_derived",
            environment_seed_strategy=ENVIRONMENT_SEED_STRATEGY,
        )

        with patch(
            "data_collect.expert_data_collection.quest_policy_collect.set_random_seed",
            side_effect=lambda seed: calls.append(("runtime", seed)),
        ):
            reset_control_state(
                _RecordingEnv(calls),
                _RecordingReset("policy.reset", calls),
                _RecordingReset("quest.reset", calls),
                _RecordingReset("ik.reset", calls),
                seed_assignment=assignment,
            )

        self.assertEqual(
            calls,
            [
                ("env.reset", 123),
                ("policy.reset", (), {}),
                ("quest.reset", (), {}),
                ("ik.reset", (), {"active": False}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
