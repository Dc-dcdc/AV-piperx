import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.configuration_post_diffusion_output_corrector import (
    PostDiffusionOutputCorrectorConfig,
)
from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DualHeadDiffusionPolicy,
)
from lerobot.common.policies.diffusion.modeling_post_diffusion_output_corrector import (
    PostDiffusionOutputCorrectorPolicy,
)
from lerobot.common.policies.factory import get_policy_and_config_classes, make_policy
from train.s2_incremental.train.train_post_diffusion_output_corrector import (
    CombinedCorrectionTrajectoryCacheDataset,
    CorrectionTrajectoryCacheDataset,
    _resolve_output_cache_layout,
    configure_trainable_parameters,
    migrate_dual_into_output_corrector,
    resolve_output_cache_sampling_seeds,
    run_scale_zero_equivalence_check,
    validate_training_config,
)


def make_base_config() -> DiffusionConfig:
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
    )


def make_corrector_config(
    *,
    corrector_type: str = "bipartite_attention",
    direction: str = "arm_to_view",
) -> PostDiffusionOutputCorrectorConfig:
    base = dict(make_base_config().__dict__)
    scales = {
        "view_to_arm": (1.0, 0.0),
        "arm_to_view": (0.0, 1.0),
        "bidirectional": (1.0, 1.0),
    }[direction]
    return PostDiffusionOutputCorrectorConfig(
        **base,
        output_corrector_type=corrector_type,
        output_corrector_direction=direction,
        output_corrector_d_model=16,
        output_corrector_num_heads=4,
        output_corrector_residual_limit=0.1,
        view_to_arm_output_scale=scales[0],
        arm_to_view_output_scale=scales[1],
    )


def make_stats() -> dict[str, dict[str, torch.Tensor]]:
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


class PostDiffusionOutputCorrectorTest(unittest.TestCase):
    @staticmethod
    def make_training_validation_config():
        return OmegaConf.create(
            {
                "resume": False,
                "init_policy_path": "baseline-checkpoint",
                "policy": {
                    "name": "post_diffusion_output_corrector",
                },
                "output_cache": {
                    "sampling_seeds": [20260727, 20260728],
                    "sampling_seed": 20260727,
                },
                "training": {
                    "lr": 1e-4,
                    "output_corrector_lr": 1e-4,
                    "save_checkpoint": True,
                    "save_freq": 2000,
                    "eval_freq": 2000,
                    "image_transforms": {"enable": False},
                },
                "eval": {
                    "async_enabled": True,
                    "skip_if_busy": False,
                },
            }
        )

    def test_every_checkpoint_evaluation_config_is_enforced(self):
        cfg = self.make_training_validation_config()
        validate_training_config(cfg)

        cfg.eval.skip_if_busy = True
        with self.assertRaisesRegex(ValueError, "skip_if_busy=false"):
            validate_training_config(cfg)

        cfg.eval.skip_if_busy = False
        cfg.training.eval_freq = 4000
        with self.assertRaisesRegex(ValueError, "save_freq和training.eval_freq"):
            validate_training_config(cfg)

    def test_shared_attention_graph_is_14_by_6_and_zero_initialized(self):
        policy = PostDiffusionOutputCorrectorPolicy(
            make_corrector_config(direction="bidirectional"),
            make_stats(),
        )
        arm = torch.randn(3, 8, 14).clamp(-1, 1)
        view = torch.randn(3, 8, 6).clamp(-1, 1)

        corrected_arm, corrected_view, diagnostics = (
            policy.diffusion.apply_output_correction(arm, view)
        )

        self.assertEqual(tuple(diagnostics["shared_graph"].shape), (3, 14, 6))
        self.assertEqual(
            tuple(diagnostics["shared_affinity"].shape),
            (3, 14, 6),
        )
        self.assertEqual(
            tuple(diagnostics["arm_to_view_graph"].shape),
            (3, 6, 14),
        )
        torch.testing.assert_close(corrected_arm, arm, rtol=0, atol=0)
        torch.testing.assert_close(corrected_view, view, rtol=0, atol=0)

    def test_direction_only_changes_requested_target(self):
        policy = PostDiffusionOutputCorrectorPolicy(
            make_corrector_config(direction="arm_to_view"),
            make_stats(),
        )
        corrector = policy.diffusion.output_corrector
        with torch.no_grad():
            corrector.view_output.bias.fill_(0.5)
        arm = torch.zeros(2, 8, 14)
        view = torch.zeros(2, 8, 6)

        corrected_arm, corrected_view, _ = (
            policy.diffusion.apply_output_correction(arm, view)
        )

        torch.testing.assert_close(corrected_arm, arm, rtol=0, atol=0)
        self.assertGreater((corrected_view - view).abs().sum().item(), 0)
        self.assertLessEqual(
            (corrected_view - view).abs().max().item(),
            policy.config.output_corrector_residual_limit,
        )

    def test_scale_zero_is_bitwise_equal_after_complete_sampling(self):
        torch.manual_seed(11)
        source = DualHeadDiffusionPolicy(make_base_config(), make_stats())
        target = PostDiffusionOutputCorrectorPolicy(
            make_corrector_config(),
            make_stats(),
        )
        migrate_dual_into_output_corrector(source, target)

        report = run_scale_zero_equivalence_check(
            source,
            target,
            seed=23,
        )

        self.assertEqual(report["max_abs_error"], 0.0)

    def test_cached_loss_updates_only_corrector_parameters(self):
        torch.manual_seed(31)
        policy = PostDiffusionOutputCorrectorPolicy(
            make_corrector_config(corrector_type="linear"),
            make_stats(),
        )
        scope = configure_trainable_parameters(policy)
        frozen_before = {
            name: tensor.detach().clone()
            for name, tensor in policy.named_parameters()
            if not tensor.requires_grad
        }
        batch = {
            "baseline_action_trajectory": torch.zeros(4, 8, 20),
            "target_action_trajectory": torch.randn(4, 8, 20).clamp(-1, 1),
            "action_is_pad": torch.zeros(4, 8, dtype=torch.bool),
        }

        losses = policy(batch)
        losses["loss"].backward()

        self.assertGreater(scope["trainable_parameter_count"], 0)
        self.assertIsNotNone(policy.diffusion.output_corrector.affinity.grad)
        for name, tensor in policy.named_parameters():
            if name in frozen_before:
                torch.testing.assert_close(
                    tensor,
                    frozen_before[name],
                    rtol=0,
                    atol=0,
                )
        self.assertIn("baseline_arm_loss", losses)
        self.assertIn("smoothness_loss", losses)

    def test_checkpoint_and_factory_round_trip(self):
        policy = PostDiffusionOutputCorrectorPolicy(
            make_corrector_config(),
            make_stats(),
        )
        hydra_config = OmegaConf.create(
            {
                "device": "cpu",
                "policy": {
                    "name": "post_diffusion_output_corrector",
                    **make_corrector_config().__dict__,
                },
            }
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            policy.save_pretrained(directory)
            restored = PostDiffusionOutputCorrectorPolicy.from_pretrained(
                directory,
                strict=True,
            )
            factory_restored = make_policy(
                hydra_config,
                pretrained_policy_name_or_path=directory,
                strict_pretrained_loading=True,
            )

        policy_class, config_class = get_policy_and_config_classes(
            "post_diffusion_output_corrector"
        )
        self.assertIs(policy_class, PostDiffusionOutputCorrectorPolicy)
        self.assertIs(config_class, PostDiffusionOutputCorrectorConfig)
        self.assertIsInstance(restored, PostDiffusionOutputCorrectorPolicy)
        self.assertIsInstance(factory_restored, PostDiffusionOutputCorrectorPolicy)

    def test_cache_dataset_supports_memory_and_mmap(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            cache_dir = Path(directory)
            np.save(
                cache_dir / "baseline.npy",
                np.zeros((5, 8, 20), dtype=np.float32),
            )
            np.save(
                cache_dir / "target.npy",
                np.ones((5, 8, 20), dtype=np.float32),
            )
            np.save(
                cache_dir / "padding.npy",
                np.zeros((5, 8), dtype=np.bool_),
            )
            manifest = {
                "schema_version": 1,
                "cache_key": "unit-test",
                "num_samples": 5,
                "num_episodes": 2,
                "episode_data_index": {},
                "arrays": {
                    "baseline": "baseline.npy",
                    "target": "target.npy",
                    "padding": "padding.npy",
                },
            }
            with open(
                cache_dir / "manifest.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(manifest, file)

            memory_dataset = CorrectionTrajectoryCacheDataset(
                cache_dir / "manifest.json",
                memory_limit_gb=1.0,
            )
            mmap_dataset = CorrectionTrajectoryCacheDataset(
                cache_dir / "manifest.json",
                memory_limit_gb=0.0,
            )

            self.assertTrue(memory_dataset.load_into_memory)
            self.assertFalse(mmap_dataset.load_into_memory)
            self.assertEqual(
                tuple(mmap_dataset[0]["baseline_action_trajectory"].shape),
                (8, 20),
            )
            torch.testing.assert_close(
                memory_dataset[0]["target_action_trajectory"],
                torch.ones(8, 20),
            )

    def test_output_cache_layout_groups_multiple_sampling_seeds(self):
        root = Path("/tmp/output-cache-layout")
        shared_payload = {
            "schema_version": 1,
            "dataset": {"source": "unit-test"},
            "source_checkpoint_fingerprint": "checkpoint",
            "precompute_batch_size": 30,
            "horizon": 16,
            "action_dim": 20,
        }
        first = _resolve_output_cache_layout(
            root,
            {**shared_payload, "sampling_seed": 20260727},
        )
        second = _resolve_output_cache_layout(
            root,
            {**shared_payload, "sampling_seed": 20260728},
        )

        self.assertEqual(first["cache_group_key"], second["cache_group_key"])
        self.assertEqual(first["cache_group_dir"], second["cache_group_dir"])
        self.assertNotEqual(first["cache_key"], second["cache_key"])
        self.assertEqual(first["cache_dir"].name, "seed=20260727")
        self.assertEqual(second["cache_dir"].name, "seed=20260728")
        self.assertRegex(
            first["cache_group_dir"].name,
            r"^dual_trajectory_[0-9a-f]{12}$",
        )
        self.assertEqual(
            first["legacy_cache_dir"].name,
            f"dual_trajectory_{first['cache_key'][:20]}",
        )

    def test_sampling_seed_list_and_legacy_fallback(self):
        multi_seed_cfg = OmegaConf.create(
            {
                "output_cache": {
                    "sampling_seeds": [20260727, 20260728],
                    "sampling_seed": 1,
                }
            }
        )
        legacy_cfg = OmegaConf.create(
            {"output_cache": {"sampling_seed": 20260729}}
        )

        self.assertEqual(
            resolve_output_cache_sampling_seeds(multi_seed_cfg),
            (20260727, 20260728),
        )
        self.assertEqual(
            resolve_output_cache_sampling_seeds(legacy_cfg),
            (20260729,),
        )
        multi_seed_cfg.output_cache.sampling_seeds = [7, 7]
        with self.assertRaisesRegex(ValueError, "重复"):
            resolve_output_cache_sampling_seeds(multi_seed_cfg)

    def test_combined_cache_dataset_concatenates_seed_trajectories(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            manifest_paths = []
            for seed, baseline_value in (
                (20260727, 1.0),
                (20260728, 2.0),
            ):
                seed_dir = root / f"seed={seed}"
                seed_dir.mkdir()
                np.save(
                    seed_dir / "baseline.npy",
                    np.full((2, 8, 20), baseline_value, dtype=np.float32),
                )
                np.save(
                    seed_dir / "target.npy",
                    np.zeros((2, 8, 20), dtype=np.float32),
                )
                np.save(
                    seed_dir / "padding.npy",
                    np.zeros((2, 8), dtype=np.bool_),
                )
                manifest = {
                    "schema_version": 1,
                    "cache_key": f"seed-{seed}",
                    "cache_group_key": "shared-group",
                    "sampling_seed": seed,
                    "num_samples": 2,
                    "num_episodes": 1,
                    "episode_data_index": {
                        "from": [0],
                        "to": [2],
                    },
                    "arrays": {
                        "baseline": "baseline.npy",
                        "target": "target.npy",
                        "padding": "padding.npy",
                    },
                }
                manifest_path = seed_dir / "manifest.json"
                with open(
                    manifest_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(manifest, file)
                manifest_paths.append(manifest_path)

            dataset = CombinedCorrectionTrajectoryCacheDataset(
                manifest_paths,
                memory_limit_gb=1.0,
            )

            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.num_samples, 4)
            self.assertEqual(dataset.num_episodes, 2)
            self.assertEqual(dataset.sampling_seeds, (20260727, 20260728))
            self.assertEqual(
                dataset.episode_data_index["from"].tolist(),
                [0, 2],
            )
            self.assertEqual(
                dataset.episode_data_index["to"].tolist(),
                [2, 4],
            )
            torch.testing.assert_close(
                dataset[0]["baseline_action_trajectory"],
                torch.ones(8, 20),
            )
            torch.testing.assert_close(
                dataset[2]["baseline_action_trajectory"],
                torch.full((8, 20), 2.0),
            )


if __name__ == "__main__":
    unittest.main()
