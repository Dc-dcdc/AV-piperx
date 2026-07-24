import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from train.replanning_dqn.eval_dynamic_steps import (
    checkpoint_identity_keys,
    configure_policy_action_steps,
    resolve_action_step_values,
    row_identity_keys,
    write_eval_summary,
)


class _FakePolicy:
    def __init__(self, **config_values) -> None:
        self.config = SimpleNamespace(**config_values)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class EvalActionStepsTest(unittest.TestCase):
    def test_action_step_sweep_is_deduplicated_in_user_order(self) -> None:
        cfg = SimpleNamespace(action_steps=[1, 2, 4, 2, 8])
        self.assertEqual(resolve_action_step_values(cfg), [1, 2, 4, 8])

    def test_native_checkpoint_step_is_supported(self) -> None:
        cfg = SimpleNamespace(action_steps=None, n_action_steps=None)
        self.assertEqual(resolve_action_step_values(cfg), [None])

    def test_diffusion_action_steps_update_config_and_reset_queue(self) -> None:
        policy = _FakePolicy(
            n_action_steps=8,
            n_obs_steps=2,
            horizon=16,
            coupling_mode="full",
            temporal_ensemble_coeff=None,
        )

        checkpoint_steps, active_steps = configure_policy_action_steps(policy, 4)

        self.assertEqual((checkpoint_steps, active_steps), (8, 4))
        self.assertEqual(policy.config.n_action_steps, 4)
        self.assertEqual(policy.reset_calls, 1)

    def test_action_steps_cannot_exceed_prediction_slice(self) -> None:
        policy = _FakePolicy(
            n_action_steps=8,
            n_obs_steps=2,
            horizon=16,
            coupling_mode="full",
            temporal_ensemble_coeff=None,
        )

        with self.assertRaisesRegex(ValueError, "超过模型允许上限"):
            configure_policy_action_steps(policy, 16)

    def test_resume_identity_distinguishes_action_steps(self) -> None:
        checkpoint = Path("/tmp/checkpoints/1000")
        step_one_keys = checkpoint_identity_keys(checkpoint, 1)
        step_eight_keys = checkpoint_identity_keys(checkpoint, 8)
        self.assertTrue(step_one_keys.isdisjoint(step_eight_keys))

        row_keys = row_identity_keys(
            {
                "checkpoint_name": checkpoint.name,
                "checkpoint_path": str(checkpoint),
                "n_action_steps": 8,
            }
        )
        self.assertTrue(row_keys & step_eight_keys)
        self.assertFalse(row_keys & step_one_keys)

    def test_summary_contains_step_comparison_csv(self) -> None:
        cfg = SimpleNamespace(
            checkpoint_source="all",
            mode="fast_repro",
            seed=100,
            n_episodes=10,
            batch_size=2,
            max_steps=400,
            action_steps=[1, 8],
        )
        rows = [
            {
                "status": "ok",
                "checkpoint_name": "1000",
                "checkpoint_key": "/tmp/checkpoints/1000",
                "checkpoint_path": "/tmp/checkpoints/1000",
                "step": 1000,
                "n_action_steps": action_steps,
                "checkpoint_n_action_steps": 8,
                "success_rate": success_rate,
                "success_rate_percent": success_rate * 100.0,
                "average_reward": 1.0,
            }
            for action_steps, success_rate in ((1, 0.5), (8, 0.8))
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            write_eval_summary(Path(tmp_dir), rows, cfg, source_path="/tmp/checkpoints")
            comparison_path = Path(tmp_dir) / "action_step_comparison.csv"
            self.assertTrue(comparison_path.exists())
            comparison = comparison_path.read_text(encoding="utf-8")

        self.assertIn("n_action_steps", comparison)
        self.assertIn("0.5", comparison)
        self.assertIn("0.8", comparison)


if __name__ == "__main__":
    unittest.main()
