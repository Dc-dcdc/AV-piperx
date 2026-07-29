import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from train.s4_adaptive_replanning.action_value_critic import (
    RelativeReplanningConfig,
)
from train.s4_adaptive_replanning.eval_dynamic_steps import (
    checkpoint_identity_keys,
    configure_policy_for_critic,
    custom_eval_policy,
    row_identity_keys,
    write_eval_summary,
)


class _FakePolicy:
    def __init__(self, **config_values) -> None:
        self.config = SimpleNamespace(**config_values)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _DynamicFakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            input_shapes={"observation.state": (2,)},
            n_action_steps=3,
        )
        self.expected_image_keys = []
        self.generated_chunks = 0
        self.reset()

    def eval(self):
        return self

    def reset(self) -> None:
        self._queues = {"action": deque(maxlen=3)}

    def select_action(self, observation):
        del observation
        if not self._queues["action"]:
            self.generated_chunks += 1
            first_value = 0.9 if self.generated_chunks == 1 else 0.8
            actions = (
                (first_value, 0.0),
                (0.2, 0.0),
                (0.2, 0.0),
            )
            self._queues["action"].extend(
                torch.tensor([action], dtype=torch.float32)
                for action in actions
            )
        return self._queues["action"].popleft()


class _FakeCritic:
    def __init__(self, training_execution_steps=()) -> None:
        self.training_execution_steps = tuple(training_execution_steps)

    def eval(self):
        return self


class _TwoStepEnv:
    def __init__(self, terminal_after: int = 2) -> None:
        self.actions = []
        self.step_index = 0
        self.terminal_after = int(terminal_after)

    def reset(self, seed=None):
        del seed
        self.step_index = 0
        return {"agent_pos": np.zeros(2, dtype=np.float32)}, {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.step_index += 1
        return (
            {"agent_pos": np.full(2, self.step_index, dtype=np.float32)},
            0.0,
            False,
            self.step_index >= self.terminal_after,
            {"is_success": False},
        )


class EvalDynamicCriticTest(unittest.TestCase):
    def test_execution_bounds_update_action_queue_capacity(self) -> None:
        policy = _FakePolicy(
            n_action_steps=16,
            n_obs_steps=2,
            horizon=16,
            coupling_mode="full",
            temporal_ensemble_coeff=None,
        )

        checkpoint_limit, min_steps, active_limit = (
            configure_policy_for_critic(
            policy,
            _FakeCritic(training_execution_steps=[8]),
            min_execution_steps=2,
            max_execution_steps=8,
            )
        )

        self.assertEqual(
            (checkpoint_limit, min_steps, active_limit),
            (16, 2, 8),
        )
        self.assertEqual(policy.config.n_action_steps, 8)
        self.assertEqual(policy.reset_calls, 1)

    def test_critic_chunk_capacity_cannot_exceed_model_horizon(self) -> None:
        policy = _FakePolicy(
            n_action_steps=8,
            n_obs_steps=2,
            horizon=16,
            coupling_mode="full",
            temporal_ensemble_coeff=None,
        )

        with self.assertRaisesRegex(ValueError, "超过模型允许上限"):
            configure_policy_for_critic(
                policy,
                _FakeCritic(training_execution_steps=[16]),
                min_execution_steps=2,
                max_execution_steps=16,
            )

    def test_resume_identity_distinguishes_critic_configuration(self) -> None:
        checkpoint = Path("/tmp/checkpoints/1000")
        critic_off = checkpoint_identity_keys(
            checkpoint,
            "critic=off",
        )
        critic_on = checkpoint_identity_keys(
            checkpoint,
            "critic=on::ckpt=/tmp/critic/best.pt",
        )
        self.assertTrue(critic_off.isdisjoint(critic_on))
        row_keys = row_identity_keys(
            {
                "checkpoint_name": checkpoint.name,
                "checkpoint_path": str(checkpoint),
                "critic_variant": "critic=on::ckpt=/tmp/critic/best.pt",
            }
        )
        self.assertTrue(row_keys & critic_on)

    def test_critic_drop_clears_chunk_and_replans(self) -> None:
        policy = _DynamicFakePolicy()
        env = _TwoStepEnv()
        cfg = SimpleNamespace(
            n_episodes=1,
            max_episodes_rendered=0,
            fps=25,
            max_steps=2,
            render_camera=["overhead_cam"],
            seed=100,
            min_execution_steps=1,
            max_execution_steps=3,
        )

        def fake_score(policy, critic, previous, current, action):
            del policy, critic, previous, current
            return float(action[0, 0])

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "train.s4_adaptive_replanning.eval_dynamic_steps."
            "score_action_with_critic",
            side_effect=fake_score,
        ):
            result = custom_eval_policy(
                env,
                policy,
                cfg,
                tmp_dir,
                torch.device("cpu"),
                critic=_FakeCritic(),
                replanning_config=RelativeReplanningConfig(
                    anchor_ratio_threshold=0.70,
                    local_drop_threshold=0.15,
                    consecutive_bad_steps=1,
                    ema_alpha=1.0,
                ),
            )

        self.assertEqual(policy.generated_chunks, 2)
        self.assertAlmostEqual(float(env.actions[0][0]), 0.9, places=6)
        self.assertAlmostEqual(float(env.actions[1][0]), 0.8, places=6)
        self.assertEqual(
            result["aggregated"]["critic_total_replans"],
            1,
        )
        self.assertEqual(
            result["episodes"][0]["critic_trigger_steps"],
            [1],
        )
        self.assertEqual(
            result["episodes"][0]["critic_trace"][1]["action_source"],
            "critic_replanned",
        )
        self.assertEqual(
            result["aggregated"]["mean_policy_inference_interval"],
            1.0,
        )

    def test_minimum_steps_block_early_critic_trigger(self) -> None:
        policy = _DynamicFakePolicy()
        env = _TwoStepEnv()
        cfg = SimpleNamespace(
            n_episodes=1,
            max_episodes_rendered=0,
            fps=25,
            max_steps=2,
            render_camera=["overhead_cam"],
            seed=100,
            min_execution_steps=2,
            max_execution_steps=3,
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "train.s4_adaptive_replanning.eval_dynamic_steps."
            "score_action_with_critic",
            side_effect=lambda policy, critic, previous, current, action: (
                float(action[0, 0])
            ),
        ):
            result = custom_eval_policy(
                env,
                policy,
                cfg,
                tmp_dir,
                torch.device("cpu"),
                critic=_FakeCritic(),
                replanning_config=RelativeReplanningConfig(
                    consecutive_bad_steps=1,
                    ema_alpha=1.0,
                ),
            )

        self.assertEqual(policy.generated_chunks, 1)
        self.assertEqual(
            result["aggregated"]["critic_total_replans"],
            0,
        )

    def test_maximum_steps_force_policy_inference(self) -> None:
        policy = _DynamicFakePolicy()
        env = _TwoStepEnv(terminal_after=4)
        cfg = SimpleNamespace(
            n_episodes=1,
            max_episodes_rendered=0,
            fps=25,
            max_steps=4,
            render_camera=["overhead_cam"],
            seed=100,
            min_execution_steps=1,
            max_execution_steps=3,
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "train.s4_adaptive_replanning.eval_dynamic_steps."
            "score_action_with_critic",
            return_value=0.9,
        ):
            result = custom_eval_policy(
                env,
                policy,
                cfg,
                tmp_dir,
                torch.device("cpu"),
                critic=_FakeCritic(),
                replanning_config=RelativeReplanningConfig(
                    anchor_ratio_threshold=0.01,
                    local_drop_threshold=10.0,
                    consecutive_bad_steps=1,
                    ema_alpha=1.0,
                ),
            )

        self.assertEqual(policy.generated_chunks, 2)
        self.assertEqual(
            result["aggregated"]["max_forced_replans"],
            1,
        )
        self.assertEqual(
            result["episodes"][0]["max_forced_replan_steps"],
            [3],
        )
        self.assertEqual(
            result["aggregated"]["mean_policy_inference_interval"],
            3.0,
        )

    def test_summary_contains_only_critic_outputs(self) -> None:
        cfg = SimpleNamespace(
            checkpoint_source="all",
            mode="fast_repro",
            seed=100,
            n_episodes=10,
            max_steps=400,
            critic_ckpt_path="/tmp/critic/best.pt",
            min_execution_steps=2,
            max_execution_steps=8,
        )
        rows = [
            {
                "status": "ok",
                "checkpoint_name": "1000",
                "checkpoint_key": "/tmp/checkpoints/1000",
                "checkpoint_path": "/tmp/checkpoints/1000",
                "step": 1000,
                "action_chunk_capacity": 8,
                "checkpoint_action_chunk_capacity": 16,
                "success_rate": success_rate,
                "success_rate_percent": success_rate * 100.0,
                "average_reward": 1.0,
                "critic_variant": "critic=on",
            }
            for success_rate in (0.5, 0.8)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            write_eval_summary(Path(tmp_dir), rows, cfg, source_path="/tmp/checkpoints")
            json_path = Path(tmp_dir) / "critic_eval_summary.json"
            csv_path = Path(tmp_dir) / "critic_eval_summary.csv"
            comparison_path = Path(tmp_dir) / "action_step_comparison.csv"

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertFalse(comparison_path.exists())
            comparison = csv_path.read_text(encoding="utf-8")

        self.assertIn("action_chunk_capacity", comparison)
        self.assertIn("0.8", comparison)


if __name__ == "__main__":
    unittest.main()
