import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lerobot.common.policies.diffusion.configuration_routed_post_diffusion_output_corrector import (
    RoutedPostDiffusionOutputCorrectorConfig,
)
from lerobot.common.policies.diffusion.modeling_post_diffusion_output_corrector import (
    PostDiffusionOutputCorrectorPolicy,
)
from lerobot.common.policies.diffusion.modeling_routed_post_diffusion_output_corrector import (
    RoutedPostDiffusionOutputCorrectorPolicy,
)
from lerobot.common.policies.factory import get_policy_and_config_classes
from train.s2_incremental.router.counterfactual_dataset import (
    CounterfactualRouterDataset,
    BranchOutcome,
    make_counterfactual_label,
    split_indices_by_episode_seed,
)
from train.s2_incremental.train.train_none_av_router import (
    configure_router_only,
    migrate_corrector_into_router,
)


def make_config() -> RoutedPostDiffusionOutputCorrectorConfig:
    return RoutedPostDiffusionOutputCorrectorConfig(
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
        output_corrector_type="bipartite_attention",
        output_corrector_direction="arm_to_view",
        output_corrector_d_model=16,
        output_corrector_num_heads=4,
        output_corrector_residual_limit=0.1,
        view_to_arm_output_scale=0.0,
        arm_to_view_output_scale=1.0,
        router_d_model=16,
        router_num_heads=4,
        router_num_layers=1,
        router_ffn_dim=32,
        router_dropout=0.0,
        router_threshold=0.7,
    )


def make_stats():
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


class NoneArmToViewRouterTest(unittest.TestCase):
    def test_pairwise_scores_and_router_only_gradients(self):
        policy = RoutedPostDiffusionOutputCorrectorPolicy(
            make_config(),
            make_stats(),
        )
        scope = configure_router_only(policy)
        batch_size = 5
        global_condition = torch.randn(batch_size, 10)
        none = torch.randn(batch_size, 8, 20)
        corrected = none.clone()
        corrected[..., 14:] += 0.1
        label = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32)

        result = policy.diffusion.compute_router_loss(
            global_condition,
            none,
            corrected,
            label,
        )
        result["loss"].backward()

        self.assertGreater(scope["parameter_count"], 0)
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in policy.diffusion.output_router.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in policy.named_parameters()
                if not name.startswith("diffusion.output_router.")
            )
        )

    def test_none_mode_is_bitwise_frozen_source(self):
        config = make_config()
        source_config = dict(config.__dict__)
        for name in tuple(source_config):
            if name.startswith("router_"):
                source_config.pop(name)
        source = PostDiffusionOutputCorrectorPolicy(
            source_config,
            make_stats(),
        )
        target = RoutedPostDiffusionOutputCorrectorPolicy(
            config,
            make_stats(),
        )
        migrate_corrector_into_router(source, target)
        source.diffusion.set_output_correction_scales(
            view_to_arm=0.0,
            arm_to_view=0.0,
        )
        target.diffusion.set_router_mode("none")
        batch = {
            "observation.state": torch.randn(2, 2, 2),
            "observation.environment_state": torch.randn(2, 2, 3),
        }

        torch.manual_seed(91)
        source_action = source.diffusion.generate_actions(batch)
        torch.manual_seed(91)
        target_action = target.diffusion.generate_actions(batch)

        torch.testing.assert_close(
            target_action,
            source_action,
            rtol=0,
            atol=0,
        )

    def test_factory_and_checkpoint_round_trip(self):
        policy = RoutedPostDiffusionOutputCorrectorPolicy(
            make_config(),
            make_stats(),
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            policy.save_pretrained(directory)
            restored = (
                RoutedPostDiffusionOutputCorrectorPolicy.from_pretrained(
                    directory,
                    strict=True,
                )
            )
        policy_class, config_class = get_policy_and_config_classes(
            "routed_post_diffusion_output_corrector"
        )
        self.assertIs(policy_class, RoutedPostDiffusionOutputCorrectorPolicy)
        self.assertIs(config_class, RoutedPostDiffusionOutputCorrectorConfig)
        self.assertIsInstance(
            restored,
            RoutedPostDiffusionOutputCorrectorPolicy,
        )

    def test_counterfactual_label_priorities(self):
        failure = BranchOutcome(False, 2, 50.0, 100)
        success = BranchOutcome(True, 3, 20.0, 100)
        self.assertEqual(
            make_counterfactual_label(
                failure,
                success,
                reward_margin=10.0,
            ),
            (1, 2.0, 1),
        )
        self.assertEqual(
            make_counterfactual_label(
                success,
                success,
                reward_margin=10.0,
            ),
            (0, 1.0, 2),
        )
        ambiguous = BranchOutcome(False, 2, 55.0, 100)
        self.assertEqual(
            make_counterfactual_label(
                failure,
                ambiguous,
                reward_margin=10.0,
            ),
            (-1, 0.0, 0),
        )

    def test_cache_split_keeps_episode_seeds_disjoint(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            arrays = {
                "global_condition": np.zeros((6, 10), dtype=np.float16),
                "none_trajectory": np.zeros((6, 8, 20), dtype=np.float16),
                "arm_to_view_trajectory": np.ones(
                    (6, 8, 20),
                    dtype=np.float16,
                ),
                "router_label": np.asarray(
                    [0, 1, -1, 1, 0, 1],
                    dtype=np.int8,
                ),
                "sample_weight": np.ones(6, dtype=np.float32),
                "quality_none": np.zeros((6, 3), dtype=np.float32),
                "quality_arm_to_view": np.ones((6, 3), dtype=np.float32),
                "episode_seed": np.asarray(
                    [10, 10, 10, 20, 20, 20],
                    dtype=np.int64,
                ),
                "decision_step": np.arange(6, dtype=np.int32),
                "label_reason": np.ones(6, dtype=np.int8),
            }
            filenames = {}
            for name, array in arrays.items():
                filename = f"{name}.npy"
                np.save(root / filename, array)
                filenames[name] = filename
            manifest = {
                "schema_version": 1,
                "num_samples": 6,
                "arrays": filenames,
            }
            with open(
                root / "manifest.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(manifest, file)

            train_indices, validation_indices = split_indices_by_episode_seed(
                root / "manifest.json",
                validation_fraction=0.5,
                seed=7,
            )
            train_seeds = set(arrays["episode_seed"][train_indices])
            validation_seeds = set(
                arrays["episode_seed"][validation_indices]
            )
            self.assertFalse(train_seeds & validation_seeds)
            dataset = CounterfactualRouterDataset(
                root / "manifest.json",
                indices=train_indices,
            )
            self.assertGreater(len(dataset), 0)
            self.assertEqual(
                tuple(dataset[0]["none_trajectory"].shape),
                (8, 20),
            )
