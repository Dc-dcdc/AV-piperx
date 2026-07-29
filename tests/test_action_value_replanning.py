"""有界动作价值Critic和相对价值重规划判定测试。"""

import tempfile
import unittest
from pathlib import Path

import torch

from train.s4_adaptive_replanning.action_value_critic import (
    ActionValueCritic,
    ActionValueCriticConfig,
    RelativeReplanningConfig,
    RelativeValueReplanningDecider,
    make_target_critic,
)
from train.s4_adaptive_replanning.train_action_value_critic import (
    save_critic_checkpoint,
)


class ActionValueReplanningTest(unittest.TestCase):
    def test_sigmoid_critic_output_is_bounded(self):
        critic = ActionValueCritic(
            ActionValueCriticConfig(
                visual_feature_dim=4,
                joint_history_dim=3,
                action_dim=2,
                visual_embed_dim=4,
                state_embed_dim=4,
                action_embed_dim=4,
                hidden_dims=(8,),
                output_activation="sigmoid",
                initial_q=0.05,
            )
        )
        values = critic(
            torch.randn(5, 4),
            torch.randn(5, 3),
            torch.randn(5, 2),
        )
        self.assertTrue(torch.all(values >= 0.0))
        self.assertTrue(torch.all(values <= 1.0))

    def test_bellman_expected_growth_does_not_trigger(self):
        config = RelativeReplanningConfig(ema_alpha=1.0)
        decider = RelativeValueReplanningDecider(config)
        decider.start_chunk(0.2)
        decision = decider.evaluate(0.2 / config.gamma)
        self.assertAlmostEqual(decision.anchor_ratio, 1.0, places=6)
        self.assertAlmostEqual(
            decision.normalized_td_change,
            0.0,
            places=6,
        )
        self.assertFalse(decision.should_replan)

    def test_two_consecutive_relative_drops_trigger(self):
        config = RelativeReplanningConfig(
            ema_alpha=1.0,
            consecutive_bad_steps=2,
        )
        decider = RelativeValueReplanningDecider(config)
        decider.start_chunk(0.5)
        first = decider.evaluate(0.2)
        second = decider.evaluate(0.2)
        self.assertFalse(first.should_replan)
        self.assertTrue(second.should_replan)
        self.assertTrue(second.anchor_bad)

    def test_epoch_checkpoint_filename_contains_loss(self):
        config = ActionValueCriticConfig(
            visual_feature_dim=4,
            joint_history_dim=3,
            action_dim=2,
            visual_embed_dim=4,
            state_embed_dim=4,
            action_embed_dim=4,
            hidden_dims=(8,),
        )
        critic = ActionValueCritic(config)
        target = make_target_critic(critic)
        optimizer = torch.optim.AdamW(critic.parameters())
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_critic_checkpoint(
                output_dir,
                critic,
                target,
                optimizer,
                config,
                epoch=3,
                global_step=12,
                checkpoint_loss=0.1234567,
                best_validation_loss=0.1234567,
                pretrained_model_dir=output_dir / "policy",
                dataset_dir=output_dir / "dataset",
                camera_names=["left", "right"],
                training_execution_steps=[8],
                save_epoch_copy=True,
                is_best=True,
            )
            checkpoint_dir = output_dir / "checkpoints"
            self.assertTrue((checkpoint_dir / "latest.pt").is_file())
            self.assertTrue((checkpoint_dir / "best.pt").is_file())
            self.assertTrue(
                (
                    checkpoint_dir
                    / "epoch_000003_loss=0.123457.pt"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
