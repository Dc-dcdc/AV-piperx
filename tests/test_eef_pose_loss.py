import unittest

import torch
from omegaconf import OmegaConf

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.eef_pose_loss import (
    PIPER_HAND_HOME,
    PIPER_VIEW_HOME,
    PiperEndEffectorPoseLoss,
)
from lerobot.common.policies.diffusion.modeling_coupled_dual_head_diffusion import (
    CoupledDualHeadDiffusionPolicy,
)
from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DualHeadDiffusionModel,
    DualHeadDiffusionPolicy,
)
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.factory import (
    _validate_eef_pose_loss_policy_support,
)


def make_config(
    *,
    position_weight: float = 0.0,
    rotation_weight: float = 0.0,
    coupled: bool = False,
) -> DiffusionConfig:
    return DiffusionConfig(
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
        view_loss_weight=0.2,
        coupling_num_heads=4 if coupled else 8,
        eef_pose_position_loss_weight=position_weight,
        eef_pose_rotation_loss_weight=rotation_weight,
        eef_pose_loss_max_timestep=None,
    )


def make_dataset_stats() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "observation.state": {
            "mean": torch.zeros(2),
            "std": torch.ones(2),
        },
        "observation.environment_state": {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
        },
        "action": {
            "min": -torch.ones(20),
            "max": torch.ones(20),
        },
    }


def make_batch() -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.randn(2, 2, 2),
        "observation.environment_state": torch.randn(2, 2, 3),
        "action": torch.randn(2, 8, 20).clamp(-1.0, 1.0),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
    }


class PiperEndEffectorPoseLossTest(unittest.TestCase):
    def test_zero_joint_pose_matches_extracted_home_transforms(self):
        module = PiperEndEffectorPoseLoss()
        pose = module._forward_kinematics(torch.zeros(3, 6))

        expected = torch.tensor(
            (PIPER_HAND_HOME, PIPER_HAND_HOME, PIPER_VIEW_HOME),
            dtype=torch.float32,
        )
        torch.testing.assert_close(pose, expected, rtol=1e-6, atol=1e-6)

    def test_identical_actions_have_zero_loss_and_grippers_are_ignored(self):
        module = PiperEndEffectorPoseLoss()
        target = torch.randn(2, 5, 20)
        predicted = target.clone()
        predicted[..., 6] += 10.0
        predicted[..., 13] -= 10.0

        losses = module(predicted, target)

        torch.testing.assert_close(
            losses["eef_position_loss"],
            torch.tensor(0.0),
            rtol=0,
            atol=1e-8,
        )
        torch.testing.assert_close(
            losses["eef_rotation_loss"],
            torch.tensor(0.0),
            rtol=0,
            atol=1e-8,
        )

    def test_pose_loss_backpropagates_finite_joint_gradients(self):
        module = PiperEndEffectorPoseLoss()
        target = torch.zeros(2, 5, 20)
        predicted = (torch.randn(2, 5, 20) * 0.1).requires_grad_()

        losses = module(predicted, target)
        total = losses["eef_position_loss"] + losses["eef_rotation_loss"]
        total.backward()

        self.assertIsNotNone(predicted.grad)
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(predicted.grad[..., :6].abs().sum().item(), 0.0)
        self.assertEqual(predicted.grad[..., 6].abs().sum().item(), 0.0)
        self.assertEqual(predicted.grad[..., 13].abs().sum().item(), 0.0)

    def test_epsilon_reconstruction_recovers_clean_action(self):
        model = DualHeadDiffusionModel(make_config())
        scheduler = model.arm_noise_scheduler
        clean = torch.randn(3, 8, 14)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([0, 7, 19], dtype=torch.long)
        noisy = scheduler.add_noise(clean, noise, timesteps)

        reconstructed = model._prediction_to_x0(
            noisy,
            noise,
            timesteps,
            scheduler,
        )

        torch.testing.assert_close(reconstructed, clean, rtol=1e-4, atol=1e-4)

    def test_zero_weights_do_not_create_fk_or_x0_intermediates(self):
        policy = DualHeadDiffusionPolicy(make_config(), make_dataset_stats())
        self.assertFalse(policy.eef_pose_loss_enabled)
        self.assertIsNone(policy.eef_pose_loss_module)

        output = policy(make_batch())

        self.assertEqual(set(output), {"loss", "arm_loss", "view_loss"})

    def test_zero_weights_keep_single_head_output_unchanged(self):
        policy = DiffusionPolicy(make_config(), make_dataset_stats())
        self.assertFalse(policy.eef_pose_loss_enabled)
        self.assertIsNone(policy.eef_pose_loss_module)

        output = policy(make_batch())

        self.assertEqual(set(output), {"loss"})

    def test_enabled_single_head_policy_adds_pose_metrics_and_gradients(self):
        torch.manual_seed(3)
        policy = DiffusionPolicy(
            make_config(
                position_weight=2.0,
                rotation_weight=0.5,
            ),
            make_dataset_stats(),
        )

        output = policy(make_batch())
        output["loss"].backward()

        self.assertTrue(torch.isfinite(output["loss"]))
        for key in (
            "eef_position_loss",
            "eef_rotation_loss",
            "eef_position_error_m",
            "eef_rotation_error_rad",
            "eef_position_weighted_loss",
            "eef_rotation_weighted_loss",
            "eef_pose_active_role_fraction",
        ):
            self.assertIn(key, output)
            self.assertTrue(torch.isfinite(output[key]))
        self.assertGreater(
            policy.diffusion.unet.final_conv[1].weight.grad.abs().sum().item(),
            0.0,
        )

    def test_enabled_coupled_policy_adds_pose_metrics_and_gradients(self):
        torch.manual_seed(4)
        policy = CoupledDualHeadDiffusionPolicy(
            make_config(
                position_weight=2.0,
                rotation_weight=0.5,
                coupled=True,
            ),
            make_dataset_stats(),
        )

        output = policy(make_batch())
        output["loss"].backward()

        self.assertTrue(torch.isfinite(output["loss"]))
        for key in (
            "eef_position_loss",
            "eef_rotation_loss",
            "eef_position_error_m",
            "eef_rotation_error_rad",
            "eef_position_weighted_loss",
            "eef_rotation_weighted_loss",
            "eef_pose_active_role_fraction",
        ):
            self.assertIn(key, output)
            self.assertTrue(torch.isfinite(output[key]))
        self.assertGreater(
            policy.diffusion.arm_unet.final_conv[1].weight.grad.abs().sum().item(),
            0.0,
        )

    def test_invalid_pose_loss_configuration_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "20D joint layout"):
            DiffusionConfig(
                input_shapes={"observation.environment_state": [3]},
                output_shapes={"action": [14]},
                input_normalization_modes={
                    "observation.environment_state": "mean_std"
                },
                output_normalization_modes={"action": "min_max"},
                crop_shape=None,
                arm_action_dim=8,
                view_action_dim=6,
                eef_pose_position_loss_weight=1.0,
            )

    def test_policy_support_guard_accepts_diffusion_and_rejects_other_policy(self):
        supported_single_head = OmegaConf.create(
            {
                "policy": {
                    "name": "diffusion",
                    "eef_pose_position_loss_weight": 1.0,
                    "eef_pose_rotation_loss_weight": 0.0,
                }
            }
        )
        _validate_eef_pose_loss_policy_support(supported_single_head)

        unsupported = OmegaConf.create(
            {
                "policy": {
                    "name": "act",
                    "eef_pose_position_loss_weight": 1.0,
                    "eef_pose_rotation_loss_weight": 0.1,
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "当前仅支持"):
            _validate_eef_pose_loss_policy_support(unsupported)


if __name__ == "__main__":
    unittest.main()
