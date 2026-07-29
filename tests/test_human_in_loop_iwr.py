import unittest

import numpy as np
import torch

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DualHeadDiffusionModel,
)
from train.s1_pretrain.train.train_human_in_loop import (
    classify_iwr_sample_indices,
    prepare_intervention_batch,
)
from omegaconf import OmegaConf


class HumanInLoopIWRTest(unittest.TestCase):
    def test_classify_keeps_robot_horizon_clear_of_intervention(self) -> None:
        # episode frames 0..7, intervention at 4 and 5, horizon=3
        is_intervention = np.asarray([0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
        weights = np.zeros((8, 2), dtype=np.float32)
        weights[4:6, 0] = 1.0

        intervention, robot = classify_iwr_sample_indices(
            episode_from=np.asarray([0]),
            episode_to=np.asarray([8]),
            is_intervention=is_intervention,
            intervention_action_weight=weights,
            horizon=3,
            min_intervention_weight=1.0e-6,
        )

        self.assertEqual(intervention, [4, 5])
        # 2 and 3 are excluded because their future action chunk reaches D_I.
        self.assertEqual(robot, [0, 1, 6, 7])

    def test_prepare_intervention_batch_constructs_full_action_target(self) -> None:
        executed = torch.arange(40, dtype=torch.float32).reshape(1, 2, 20)
        teleop = -torch.ones(1, 2, 20)
        weights = torch.zeros(1, 2, 20)
        weights[:, 0, :7] = 1.0
        weights[:, 1, :7] = 0.5
        batch = {
            "action": executed,
            "teleop_action": teleop,
            "teleop_action_available": torch.tensor([[True, False]]),
            "intervention_action_weight": weights,
        }
        cfg = OmegaConf.create({"training": {"iwr": {"prefer_teleop_action": True}}})

        prepared = prepare_intervention_batch(batch, cfg)

        # raw teleop 只覆盖第一帧人工控制的左臂 7 维。
        torch.testing.assert_close(prepared["action"][:, 0, :7], teleop[:, 0, :7])
        torch.testing.assert_close(prepared["action"][:, 0, 7:], executed[:, 0, 7:])
        # 第二帧 raw teleop 不可用，完整 target 回退到 executed action。
        torch.testing.assert_close(prepared["action"][:, 1], executed[:, 1])
        # 默认 human_action_weight=1，20 维全部等权监督。
        torch.testing.assert_close(prepared["loss_mask"], torch.ones_like(executed))

    def test_prepare_intervention_batch_can_boost_human_dimensions(self) -> None:
        executed = torch.zeros(1, 2, 20)
        weights = torch.zeros_like(executed)
        weights[:, 0, :7] = 1.0
        weights[:, 1, :7] = 0.5
        cfg = OmegaConf.create(
            {
                "training": {
                    "iwr": {
                        "prefer_teleop_action": False,
                        "human_action_weight": 2.0,
                    }
                }
            }
        )

        prepared = prepare_intervention_batch(
            {
                "action": executed,
                "intervention_action_weight": weights,
            },
            cfg,
        )

        torch.testing.assert_close(
            prepared["loss_mask"][:, 0, :7],
            torch.full((1, 7), 2.0),
        )
        torch.testing.assert_close(
            prepared["loss_mask"][:, 1, :7],
            torch.full((1, 7), 1.5),
        )
        torch.testing.assert_close(
            prepared["loss_mask"][..., 7:],
            torch.ones(1, 2, 13),
        )

    def test_dual_head_loss_masks_uncontrolled_view_dimensions(self) -> None:
        config = DiffusionConfig(
            n_obs_steps=2,
            horizon=8,
            n_action_steps=4,
            input_shapes={
                "observation.state": [2],
                "observation.environment_state": [3],
            },
            output_shapes={"action": [20]},
            input_normalization_modes={
                "observation.state": "mean_std",
                "observation.environment_state": "mean_std",
            },
            output_normalization_modes={"action": "min_max"},
            crop_shape=None,
            down_dims=(16, 32),
            n_groups=4,
            diffusion_step_embed_dim=16,
            noise_scheduler_type="DDIM",
            num_train_timesteps=20,
            num_inference_steps=2,
            arm_action_dim=14,
            view_action_dim=6,
            do_mask_loss_for_padding=True,
        )
        model = DualHeadDiffusionModel(config)
        loss_mask = torch.zeros(2, 8, 20)
        loss_mask[..., :7] = 1.0  # only the left arm is controlled by the human
        output = model.compute_loss(
            {
                "observation.state": torch.randn(2, 2, 2),
                "observation.environment_state": torch.randn(2, 2, 3),
                "action": torch.randn(2, 8, 20),
                "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
                "loss_mask": loss_mask,
            }
        )

        self.assertGreater(float(output["arm_loss"]), 0.0)
        self.assertEqual(float(output["view_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
