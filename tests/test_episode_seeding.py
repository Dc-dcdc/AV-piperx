import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_collect.episode_seeding import (
    MAX_UINT32_SEED,
    EpisodeSeedStream,
    ReplaySeedStream,
    ReplaySeedsExhausted,
    derive_environment_seed,
    load_replay_seed_records,
    normalize_base_seed,
    replay_seed_digest,
    resolve_next_attempt_index,
    validate_append_seed_configuration,
)


class EpisodeSeedingTest(unittest.TestCase):
    def test_seed_is_deterministic_for_same_base_and_attempt(self) -> None:
        first = derive_environment_seed(100, 7)
        second = derive_environment_seed(100, 7)

        self.assertEqual(first, second)
        self.assertIsInstance(first, int)
        self.assertGreaterEqual(first, 0)
        self.assertLessEqual(first, MAX_UINT32_SEED)

    def test_attempts_receive_distinct_seeds(self) -> None:
        seeds = [derive_environment_seed(100, attempt_index) for attempt_index in range(32)]

        self.assertEqual(len(seeds), len(set(seeds)))

    def test_seed_matches_numpy_seed_sequence_derivation(self) -> None:
        expected = int(
            np.random.SeedSequence([100, 3]).generate_state(1, dtype=np.uint32)[0]
        )

        self.assertEqual(derive_environment_seed(100, 3), expected)

    def test_unseeded_stream_still_advances_attempt_index(self) -> None:
        stream = EpisodeSeedStream(base_seed=None, next_attempt_index=4)

        self.assertEqual(stream.take(), (4, None))
        self.assertEqual(stream.take(), (5, None))
        self.assertEqual(stream.next_attempt_index, 6)

    def test_stream_can_resume_from_persisted_cursor(self) -> None:
        stream = EpisodeSeedStream(base_seed=100, next_attempt_index=9)

        attempt_index, environment_seed = stream.take()

        self.assertEqual(attempt_index, 9)
        self.assertEqual(environment_seed, derive_environment_seed(100, 9))
        self.assertEqual(stream.next_attempt_index, 10)

    def test_persisted_attempt_cursor_wins_over_saved_episode_index(self) -> None:
        next_index = resolve_next_attempt_index(
            episode_index=3,
            existing_metadata={"next_attempt_index": 11, "attempted_episodes": 10},
            existing_infos=[{"episode": 2, "attempt_index": 7}],
        )

        self.assertEqual(next_index, 11)

    def test_saved_attempt_index_recovers_cursor_when_metadata_is_old(self) -> None:
        next_index = resolve_next_attempt_index(
            episode_index=3,
            existing_metadata={},
            existing_infos=[{"episode": 2, "attempt_index": 8}],
        )

        self.assertEqual(next_index, 9)

    def test_old_metadata_falls_back_to_next_saved_episode_index(self) -> None:
        next_index = resolve_next_attempt_index(
            episode_index=5,
            existing_metadata={},
            existing_infos=[{"episode": 4}],
        )

        self.assertEqual(next_index, 5)

    def test_legacy_fixed_seed_metadata_is_preserved_during_migration(self) -> None:
        legacy = validate_append_seed_configuration(
            existing_metadata={"random_seed": 100, "fixed_reset_seed": True},
            base_seed=100,
            saved_episode_count=2,
        )

        self.assertEqual(legacy["strategy"], "legacy_fixed_reset_seed")
        self.assertEqual(legacy["base_random_seed"], 100)
        self.assertEqual(legacy["saved_episode_count"], 2)

    def test_append_rejects_changed_base_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "different base random seed"):
            validate_append_seed_configuration(
                existing_metadata={
                    "base_random_seed": 100,
                    "environment_seed_strategy": "numpy_seed_sequence_v1",
                },
                base_seed=101,
                saved_episode_count=2,
            )

    def test_append_rejects_changed_seed_strategy(self) -> None:
        with self.assertRaisesRegex(ValueError, "different environment seed strategy"):
            validate_append_seed_configuration(
                existing_metadata={
                    "base_random_seed": 100,
                    "environment_seed_strategy": "unknown_strategy",
                },
                base_seed=100,
                saved_episode_count=2,
            )

    def test_invalid_seed_or_attempt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_base_seed(-1)
        with self.assertRaises(ValueError):
            normalize_base_seed(MAX_UINT32_SEED + 1)
        with self.assertRaises(ValueError):
            derive_environment_seed(100, -1)
        with self.assertRaises(ValueError):
            EpisodeSeedStream(base_seed=100, next_attempt_index=-1)
        with self.assertRaises(ValueError):
            resolve_next_attempt_index(
                episode_index=0,
                existing_metadata={"next_attempt_index": -1},
                existing_infos=[],
            )

    def test_episode_results_loader_filters_successful_records(self) -> None:
        payload = [
            {"episode": 0, "seed": 100, "success": True, "reward": 10.0, "steps": 5},
            {"episode": 1, "seed": 101, "success": False, "reward": 2.5, "steps": 20},
            {"episode": 2, "seed": 102, "success": False, "reward": 3.5, "steps": 20},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "episode_results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            records = load_replay_seed_records(path, failed_only=True)

        self.assertEqual([record.seed for record in records], [101, 102])
        self.assertEqual(records[0].source_index, 1)
        self.assertEqual(records[0].evaluation_episode, 1)
        self.assertFalse(records[0].evaluation_success)
        self.assertEqual(records[0].evaluation_reward, 2.5)
        self.assertEqual(records[0].evaluation_steps, 20)

    def test_simple_seed_list_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "seeds.json"
            path.write_text(json.dumps([7, 9]), encoding="utf-8")

            records = load_replay_seed_records(path)

        self.assertEqual([record.seed for record in records], [7, 9])
        self.assertEqual(records[0].evaluation_success, None)

    def test_replay_stream_repeats_each_seed_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "seeds.json"
            path.write_text(json.dumps([11, 22]), encoding="utf-8")
            records = load_replay_seed_records(path)

        stream = ReplaySeedStream(
            records=records,
            next_attempt_index=4,
            replay_batch_index=3,
            repeat=2,
            on_exhausted="stop",
            replay_policy_rng=True,
        )
        assignments = [stream.take_assignment() for _ in range(4)]

        self.assertEqual([item.attempt_index for item in assignments], [4, 5, 6, 7])
        self.assertEqual([item.replay_batch_index for item in assignments], [3, 3, 3, 3])
        self.assertEqual([item.environment_seed for item in assignments], [11, 11, 22, 22])
        self.assertEqual([item.policy_seed for item in assignments], [11, 11, 22, 22])
        self.assertEqual([item.replay_repeat_index for item in assignments], [0, 1, 0, 1])
        self.assertEqual([item.evaluation_source_index for item in assignments], [0, 0, 1, 1])
        self.assertTrue(stream.exhausted)
        with self.assertRaises(ReplaySeedsExhausted):
            stream.take_assignment()

    def test_replay_stream_can_loop_without_reusing_attempt_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "seeds.json"
            path.write_text(json.dumps({"seeds": [31, 32]}), encoding="utf-8")
            records = load_replay_seed_records(path)

        stream = ReplaySeedStream(records=records, on_exhausted="loop")
        assignments = [stream.take_assignment() for _ in range(5)]

        self.assertEqual([item.environment_seed for item in assignments], [31, 32, 31, 32, 31])
        self.assertEqual([item.attempt_index for item in assignments], [0, 1, 2, 3, 4])
        self.assertEqual([item.replay_cycle for item in assignments], [0, 0, 1, 1, 2])

    def test_replay_policy_rng_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "seeds.json"
            path.write_text(json.dumps([42]), encoding="utf-8")
            records = load_replay_seed_records(path)

        assignment = ReplaySeedStream(
            records=records,
            replay_policy_rng=False,
        ).take_assignment()

        self.assertEqual(assignment.environment_seed, 42)
        self.assertIsNone(assignment.policy_seed)

    def test_replay_digest_depends_on_selected_seed_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_path = Path(tmp_dir) / "first.json"
            second_path = Path(tmp_dir) / "second.json"
            first_path.write_text(json.dumps([1, 2]), encoding="utf-8")
            second_path.write_text(json.dumps([2, 1]), encoding="utf-8")
            first = load_replay_seed_records(first_path)
            second = load_replay_seed_records(second_path)

        self.assertNotEqual(replay_seed_digest(first), replay_seed_digest(second))


if __name__ == "__main__":
    unittest.main()
