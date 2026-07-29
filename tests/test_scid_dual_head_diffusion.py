import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import datasets
import torch
from omegaconf import OmegaConf

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DualHeadDiffusionModel,
    DualHeadDiffusionPolicy,
)
from lerobot.common.policies.diffusion.modeling_scid_dual_head_diffusion import (
    SCIDDualHeadDiffusionModel,
    SCIDDualHeadDiffusionPolicy,
)
from lerobot.common.policies.factory import get_policy_and_config_classes, make_policy
from train.s3_finetune.finetune_dppo_dual_head import forward_dppo_from_global_cond
from train.s1_pretrain.train.scid_transform import (
    fit_scid_transform,
    initialize_scid_transform_from_dataset,
)
from train.s4_adaptive_replanning.train_replanning_dqn import (
    infer_full_joint_chunk,
)


def make_small_config(*, clamp: bool = True) -> DiffusionConfig:
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
        scid_ridge=1e-3,
        scid_residual_eps=1e-6,
        scid_clamp_reconstructed_view=clamp,
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


def make_transform() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = torch.arange(84, dtype=torch.float32).reshape(6, 14) / 840.0
    bias = torch.linspace(-0.1, 0.1, 6)
    scale = torch.linspace(0.2, 0.7, 6)
    return matrix, bias, scale


def make_model_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.randn(batch_size, 2, 2),
        "observation.environment_state": torch.randn(batch_size, 2, 3),
        "action": torch.empty(batch_size, 8, 20).uniform_(-0.8, 0.8),
        "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
    }


class SCIDDualHeadDiffusionTest(unittest.TestCase):
    def test_encode_decode_round_trip_without_clamp(self):
        model = SCIDDualHeadDiffusionModel(make_small_config(clamp=False))
        model.set_scid_transform(*make_transform())
        arm = torch.randn(3, 8, 14)
        view = torch.randn(3, 8, 6)

        innovation = model.encode_view_innovation(arm, view)
        reconstructed = model.decode_view_innovation(arm, innovation, clamp=False)

        torch.testing.assert_close(reconstructed, view, rtol=1e-5, atol=1e-6)

    def test_unfitted_transform_fails_fast(self):
        model = SCIDDualHeadDiffusionModel(make_small_config())
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            model.compute_loss(make_model_batch())
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            model.generate_actions(
                {
                    "observation.state": torch.randn(1, 2, 2),
                    "observation.environment_state": torch.randn(1, 2, 3),
                }
            )

    def test_transform_validation_and_config_validation(self):
        model = SCIDDualHeadDiffusionModel(make_small_config())
        matrix, bias, scale = make_transform()
        with self.assertRaisesRegex(ValueError, "matrix"):
            model.set_scid_transform(matrix[:, :-1], bias, scale)
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid_matrix = matrix.clone()
            invalid_matrix[0, 0] = float("nan")
            model.set_scid_transform(invalid_matrix, bias, scale)
        with self.assertRaisesRegex(ValueError, "residual scale"):
            model.set_scid_transform(matrix, bias, torch.zeros_like(scale))
        with self.assertRaisesRegex(ValueError, "scid_ridge"):
            DiffusionConfig(**{**make_small_config().__dict__, "scid_ridge": -1.0})

    def test_loss_uses_innovation_target_and_keeps_compatible_keys(self):
        torch.manual_seed(7)
        model = SCIDDualHeadDiffusionModel(make_small_config())
        model.set_scid_transform(*make_transform())

        losses = model.compute_loss(make_model_batch())
        losses["loss"].backward()

        self.assertEqual(
            set(losses),
            {"loss", "arm_loss", "view_loss", "innovation_loss"},
        )
        torch.testing.assert_close(losses["innovation_loss"], losses["view_loss"])
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIsNotNone(next(model.arm_unet.parameters()).grad)
        self.assertIsNotNone(next(model.view_unet.parameters()).grad)
        self.assertNotIn("scid_matrix", dict(model.named_parameters()))
        self.assertIsNone(model.scid_matrix.grad)

    def test_sampler_reconstructs_view_and_preserves_original_slice(self):
        model = SCIDDualHeadDiffusionModel(make_small_config(clamp=False))
        matrix, bias, scale = make_transform()
        model.set_scid_transform(matrix, bias, scale)
        arm = torch.linspace(-0.2, 0.2, 2 * 8 * 14).reshape(2, 8, 14)
        innovation = torch.linspace(-0.5, 0.5, 2 * 8 * 6).reshape(2, 8, 6)

        with patch.object(model, "conditional_sample", side_effect=[arm, innovation]):
            actions = model.generate_actions(
                {
                    "observation.state": torch.randn(2, 2, 2),
                    "observation.environment_state": torch.randn(2, 2, 3),
                }
            )

        expected_view = torch.nn.functional.linear(arm, matrix, bias) + innovation * scale
        expected = torch.cat([arm, expected_view], dim=-1)[:, 1:5]
        self.assertEqual(tuple(actions.shape), (2, 4, 20))
        torch.testing.assert_close(actions, expected)

    def test_zero_transform_exactly_matches_raw_dual_head(self):
        torch.manual_seed(11)
        raw = DualHeadDiffusionModel(make_small_config()).eval()
        scid = SCIDDualHeadDiffusionModel(make_small_config()).eval()
        incompatible = scid.load_state_dict(raw.state_dict(), strict=False)
        self.assertEqual(
            set(incompatible.missing_keys),
            {
                "scid_matrix",
                "scid_bias",
                "scid_residual_scale",
                "scid_transform_fitted",
            },
        )
        self.assertEqual(incompatible.unexpected_keys, [])
        scid.set_scid_transform(torch.zeros(6, 14), torch.zeros(6), torch.ones(6))
        batch = {
            "observation.state": torch.randn(2, 2, 2),
            "observation.environment_state": torch.randn(2, 2, 3),
        }

        torch.manual_seed(23)
        raw_actions = raw.generate_actions(batch)
        torch.manual_seed(23)
        scid_actions = scid.generate_actions(batch)

        torch.testing.assert_close(scid_actions, raw_actions, rtol=0, atol=0)

    def test_checkpoint_round_trip_preserves_transform_strictly(self):
        policy = SCIDDualHeadDiffusionPolicy(make_small_config(), make_dataset_stats())
        policy.diffusion.set_scid_transform(*make_transform())
        hydra_config = OmegaConf.create(
            {
                "device": "cpu",
                "policy": {
                    "name": "scid_dual_head_diffusion",
                    **make_small_config().__dict__,
                },
            }
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            policy.save_pretrained(directory)
            restored = SCIDDualHeadDiffusionPolicy.from_pretrained(
                directory,
                strict=True,
            )
            factory_restored = make_policy(
                hydra_config,
                pretrained_policy_name_or_path=directory,
            )

        self.assertTrue(restored.diffusion.is_scid_transform_fitted)
        self.assertTrue(factory_restored.diffusion.is_scid_transform_fitted)
        torch.testing.assert_close(restored.diffusion.scid_matrix, policy.diffusion.scid_matrix)
        torch.testing.assert_close(restored.diffusion.scid_bias, policy.diffusion.scid_bias)
        torch.testing.assert_close(
            restored.diffusion.scid_residual_scale,
            policy.diffusion.scid_residual_scale,
        )

    def test_factory_controlled_raw_dual_migration(self):
        raw_policy = DualHeadDiffusionPolicy(make_small_config(), make_dataset_stats())
        target_config = OmegaConf.create(
            {
                "device": "cpu",
                "policy": {
                    "name": "scid_dual_head_diffusion",
                    **make_small_config().__dict__,
                },
            }
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            raw_policy.save_pretrained(directory)
            with self.assertRaises(RuntimeError):
                make_policy(
                    target_config,
                    pretrained_policy_name_or_path=directory,
                )
            migrated = make_policy(
                target_config,
                pretrained_policy_name_or_path=directory,
                allow_scid_dual_init=True,
            )

        self.assertIsInstance(migrated, SCIDDualHeadDiffusionPolicy)
        self.assertFalse(migrated.diffusion.is_scid_transform_fitted)
        raw_parameters = dict(raw_policy.named_parameters())
        migrated_parameters = dict(migrated.named_parameters())
        self.assertEqual(set(raw_parameters), set(migrated_parameters))
        for name in raw_parameters:
            torch.testing.assert_close(raw_parameters[name], migrated_parameters[name])

    def test_closed_form_fit_and_fresh_initialization(self):
        torch.manual_seed(31)
        arm = torch.empty(512, 14).uniform_(-0.8, 0.8)
        true_matrix = torch.randn(6, 14) * 0.15
        true_bias = torch.linspace(-0.1, 0.1, 6)
        view = torch.nn.functional.linear(arm, true_matrix, true_bias)
        view += 0.01 * torch.randn_like(view)
        actions = torch.cat([arm, view], dim=-1)
        leaf = SimpleNamespace(
            hf_dataset=datasets.Dataset.from_dict({"action": actions.numpy()})
        )

        fit = fit_scid_transform(
            leaf,
            lambda values: values,
            arm_action_dim=14,
            view_action_dim=6,
            ridge=1e-8,
            batch_size=97,
        )

        torch.testing.assert_close(fit.matrix, true_matrix, rtol=0.03, atol=0.01)
        torch.testing.assert_close(fit.bias, true_bias, rtol=0.03, atol=0.01)
        self.assertGreater(fit.diagnostics["view_r2_mean"], 0.99)
        self.assertLess(
            fit.diagnostics["residual_cross_corr_norm"],
            fit.diagnostics["raw_cross_corr_norm"],
        )

        policy = SCIDDualHeadDiffusionPolicy(make_small_config(), make_dataset_stats())
        initialized = initialize_scid_transform_from_dataset(
            policy,
            leaf,
            resume=False,
            batch_size=113,
        )
        self.assertIsNotNone(initialized)
        self.assertTrue(policy.diffusion.is_scid_transform_fitted)
        self.assertIsNone(
            initialize_scid_transform_from_dataset(
                policy,
                leaf,
                resume=True,
            )
        )

    def test_dppo_keeps_latent_chain_but_decodes_environment_action(self):
        policy = SCIDDualHeadDiffusionPolicy(
            make_small_config(clamp=False),
            make_dataset_stats(),
        )
        policy.diffusion.set_scid_transform(*make_transform())
        arm = torch.randn(2, 8, 14) * 0.1
        innovation = torch.randn(2, 8, 6) * 0.1
        arm_chain = torch.randn(2, 3, 8, 14)
        innovation_chain = torch.randn(2, 3, 8, 6)

        samples = [(arm, arm_chain), (innovation, innovation_chain)]
        with patch(
            "train.s3_finetune.finetune_dppo_dual_head._sample_head_dppo",
            side_effect=samples,
        ):
            result = forward_dppo_from_global_cond(
                policy,
                torch.randn(2, 10),
                return_chain=True,
            )

        expected_actions = policy.diffusion.combine_action_heads(arm, innovation)
        expected_chain = torch.cat([arm_chain, innovation_chain], dim=-1)
        torch.testing.assert_close(result["actions"], expected_actions)
        torch.testing.assert_close(result["chains"], expected_chain)

    def test_replanning_sampler_decodes_scid_innovation(self):
        policy = SCIDDualHeadDiffusionPolicy(
            make_small_config(clamp=False),
            make_dataset_stats(),
        )
        policy.diffusion.set_scid_transform(*make_transform())
        arm = torch.randn(1, 8, 14) * 0.1
        innovation = torch.randn(1, 8, 6) * 0.1
        history = {
            "observation.state": torch.randn(1, 2, 2),
            "observation.environment_state": torch.randn(1, 2, 3),
        }

        with patch.object(
            policy.diffusion,
            "conditional_sample",
            side_effect=[arm, innovation],
        ):
            normalized, environment, _ = infer_full_joint_chunk(
                policy,
                history,
                torch.Generator().manual_seed(0),
            )

        expected = policy.diffusion.combine_action_heads(arm, innovation)[0]
        torch.testing.assert_close(normalized, expected)
        # Unit-test action stats are [-1, 1], so unnormalization is identity.
        torch.testing.assert_close(environment, expected)

    def test_factory_registration(self):
        policy_class, config_class = get_policy_and_config_classes(
            "scid_dual_head_diffusion"
        )
        self.assertIs(policy_class, SCIDDualHeadDiffusionPolicy)
        self.assertIs(config_class, DiffusionConfig)


if __name__ == "__main__":
    unittest.main()
