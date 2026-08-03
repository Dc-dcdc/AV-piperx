import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import datasets
import torch
from omegaconf import OmegaConf

from lerobot.common.datasets.view_delta_stats import (
    load_or_compute_view_delta_stats,
)
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_coupled_dual_head_diffusion import (
    CoupledDualHeadDiffusionPolicy,
)
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DualHeadDiffusionPolicy,
)
from lerobot.common.policies.diffusion.view_action_representation import (
    VIEW_ACTION_DELTA_STATS_KEY,
    decode_actions_delta_from_current,
    encode_actions_delta_from_current,
)
from lerobot.common.policies.factory import make_policy


def make_config(*, coupled: bool = False, representation: str = "delta_from_current"):
    return DiffusionConfig(
        n_obs_steps=2,
        horizon=8,
        n_action_steps=2,
        input_shapes={
            "observation.state": [20],
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
        view_action_representation=representation,
        coupling_num_heads=4 if coupled else 8,
    )


def make_dataset_stats(*, with_view_delta: bool = True):
    stats = {
        "observation.state": {
            "mean": torch.zeros(20),
            "std": torch.ones(20),
        },
        "observation.environment_state": {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
        },
        "action": {
            "mean": torch.zeros(20),
            "std": torch.ones(20),
            "min": -2.0 * torch.ones(20),
            "max": 2.0 * torch.ones(20),
        },
    }
    if with_view_delta:
        stats[VIEW_ACTION_DELTA_STATS_KEY] = {
            "mean": torch.zeros(6),
            "std": 0.2 * torch.ones(6),
            "min": -0.5 * torch.ones(6),
            "max": 0.5 * torch.ones(6),
        }
    return stats


class ViewActionRepresentationTest(unittest.TestCase):
    def test_horizon_delta_stats_are_computed_without_rewriting_dataset(self):
        states = torch.zeros(3, 20)
        actions = torch.zeros(3, 20)
        states[:, 14:] = torch.tensor([0.0, 1.0, 3.0])[:, None]
        actions[:, 14:] = torch.tensor([0.2, 1.2, 3.2])[:, None]
        hf_dataset = datasets.Dataset.from_dict(
            {
                "timestamp": [0.0, 0.04, 0.08],
                "episode_index": [0, 0, 0],
                "observation.state": states.tolist(),
                "action": actions.tolist(),
            }
        )
        dataset = SimpleNamespace(
            hf_dataset=hf_dataset,
            episode_data_index={
                "from": torch.tensor([0]),
                "to": torch.tensor([3]),
            },
            fps=25,
            repo_id="synthetic",
            root=None,
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            stats, cache_path, metadata = load_or_compute_view_delta_stats(
                dataset,
                action_delta_timestamps=[0.0, 0.04],
                arm_action_dim=14,
                view_action_dim=6,
                include_padding=False,
                cache_dir=directory,
            )
            cached_stats, cached_path, cached_metadata = (
                load_or_compute_view_delta_stats(
                    dataset,
                    action_delta_timestamps=[0.0, 0.04],
                    arm_action_dim=14,
                    view_action_dim=6,
                    include_padding=False,
                    cache_dir=directory,
                )
            )

        self.assertEqual(metadata["valid_or_included_count"], 5)
        self.assertEqual(metadata["padding_count"], 1)
        torch.testing.assert_close(
            stats["mean"],
            0.8 * torch.ones(6),
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(stats["min"], -stats["max"])
        self.assertEqual(cache_path, cached_path)
        self.assertEqual(metadata, cached_metadata)
        for name in stats:
            torch.testing.assert_close(stats[name], cached_stats[name])

    def test_round_trip_and_fixed_offset_invariance(self):
        torch.manual_seed(7)
        anchor = torch.randn(3, 6)
        actions = torch.randn(3, 8, 20)
        offset = torch.randn(3, 6)

        encoded = encode_actions_delta_from_current(
            actions,
            anchor,
            arm_action_dim=14,
            view_action_dim=6,
        )
        shifted_actions = actions.clone()
        shifted_actions[..., 14:] += offset[:, None, :]
        shifted_anchor = anchor + offset
        shifted_encoded = encode_actions_delta_from_current(
            shifted_actions,
            shifted_anchor,
            arm_action_dim=14,
            view_action_dim=6,
        )
        decoded = decode_actions_delta_from_current(
            encoded,
            anchor,
            arm_action_dim=14,
            view_action_dim=6,
        )

        torch.testing.assert_close(
            decoded[..., :14],
            actions[..., :14],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(decoded, actions, rtol=0, atol=2e-7)
        torch.testing.assert_close(
            shifted_encoded,
            encoded,
            rtol=0,
            atol=3e-7,
        )

    def test_delta_mode_uses_mixed_action_stats_for_all_diffusion_policies(self):
        stats = make_dataset_stats()
        for policy_class, coupled in (
            (DiffusionPolicy, False),
            (DualHeadDiffusionPolicy, False),
            (CoupledDualHeadDiffusionPolicy, True),
        ):
            with self.subTest(policy=policy_class.__name__):
                policy = policy_class(make_config(coupled=coupled), stats)
                target_buffer = policy.normalize_targets.buffer_action
                output_buffer = policy.unnormalize_outputs.buffer_action
                torch.testing.assert_close(
                    target_buffer["min"][:14],
                    stats["action"]["min"][:14],
                )
                torch.testing.assert_close(
                    target_buffer["min"][14:],
                    stats[VIEW_ACTION_DELTA_STATS_KEY]["min"],
                )
                torch.testing.assert_close(
                    output_buffer["max"][14:],
                    stats[VIEW_ACTION_DELTA_STATS_KEY]["max"],
                )

    def test_fresh_delta_policy_requires_derived_stats(self):
        for policy_class in (DiffusionPolicy, DualHeadDiffusionPolicy):
            with self.subTest(policy=policy_class.__name__):
                with self.assertRaisesRegex(KeyError, "View增量统计量"):
                    policy_class(
                        make_config(),
                        make_dataset_stats(with_view_delta=False),
                    )

    def test_action_queue_keeps_one_anchor_for_whole_chunk(self):
        def observation(view_value: float):
            state = torch.zeros(1, 20)
            state[:, 14:] = view_value
            return {
                "observation.state": state,
                "observation.environment_state": torch.zeros(1, 3),
            }

        for policy_class in (DiffusionPolicy, DualHeadDiffusionPolicy):
            with self.subTest(policy=policy_class.__name__):
                policy = policy_class(
                    make_config(),
                    make_dataset_stats(),
                ).eval()
                first_chunk = torch.zeros(1, 2, 20)
                first_chunk[:, 1, 14:] = 0.4  # 反归一化后是+0.2rad。
                second_chunk = torch.zeros(1, 2, 20)

                with patch.object(
                    policy.diffusion,
                    "generate_actions",
                    side_effect=(first_chunk, second_chunk),
                ) as generate:
                    action_0 = policy.select_action(observation(1.0))
                    # 队列尚未耗尽；新的真实状态不能改变已解码动作块的锚点。
                    action_1 = policy.select_action(observation(9.0))
                    # 队列耗尽后，新动作块才以当前真实状态重新建立锚点。
                    action_2 = policy.select_action(observation(3.0))

                torch.testing.assert_close(
                    action_0[:, 14:],
                    torch.ones(1, 6),
                )
                torch.testing.assert_close(
                    action_1[:, 14:],
                    1.2 * torch.ones(1, 6),
                )
                torch.testing.assert_close(
                    action_2[:, 14:],
                    3.0 * torch.ones(1, 6),
                )
                self.assertEqual(generate.call_count, 2)

    def test_absolute_mode_does_not_require_or_replace_delta_stats(self):
        stats = make_dataset_stats(with_view_delta=False)
        for policy_class in (DiffusionPolicy, DualHeadDiffusionPolicy):
            with self.subTest(policy=policy_class.__name__):
                policy = policy_class(
                    make_config(representation="absolute"),
                    stats,
                )
                torch.testing.assert_close(
                    policy.normalize_targets.buffer_action["min"],
                    stats["action"]["min"],
                )
                torch.testing.assert_close(
                    policy.unnormalize_outputs.buffer_action["max"],
                    stats["action"]["max"],
                )

    def test_representation_switch_does_not_change_state_dict_structure(self):
        for policy_class in (DiffusionPolicy, DualHeadDiffusionPolicy):
            with self.subTest(policy=policy_class.__name__):
                absolute_policy = policy_class(
                    make_config(representation="absolute"),
                    make_dataset_stats(with_view_delta=False),
                )
                delta_policy = policy_class(
                    make_config(representation="delta_from_current"),
                    make_dataset_stats(),
                )
                absolute_state = absolute_policy.state_dict()
                delta_state = delta_policy.state_dict()
                self.assertEqual(set(absolute_state), set(delta_state))
                self.assertEqual(
                    {
                        key: tuple(value.shape)
                        for key, value in absolute_state.items()
                    },
                    {
                        key: tuple(value.shape)
                        for key, value in delta_state.items()
                    },
                )

    def test_delta_mode_reconstructs_absolute_actions_for_fk_loss(self):
        torch.manual_seed(17)
        for policy_class, coupled in (
            (DiffusionPolicy, False),
            (DualHeadDiffusionPolicy, False),
            (CoupledDualHeadDiffusionPolicy, True),
        ):
            with self.subTest(policy=policy_class.__name__):
                config = make_config(coupled=coupled)
                config.eef_pose_position_loss_weight = 1.0
                config.eef_pose_rotation_loss_weight = 0.01
                policy = policy_class(config, make_dataset_stats())
                batch = {
                    "observation.state": 0.1 * torch.randn(2, 2, 20),
                    "observation.environment_state": torch.randn(2, 2, 3),
                    "action": 0.1 * torch.randn(2, 8, 20),
                    "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
                }

                output = policy(batch)
                output["loss"].backward()

                self.assertIn("eef_position_loss", output)
                self.assertIn("eef_rotation_loss", output)
                self.assertTrue(torch.isfinite(output["loss"]))
                final_conv = (
                    policy.diffusion.unet.final_conv
                    if policy_class is DiffusionPolicy
                    else policy.diffusion.arm_unet.final_conv
                )
                self.assertGreater(
                    final_conv[1].weight.grad.abs().sum().item(),
                    0.0,
                )

    def test_factory_accepts_single_head_delta_policy(self):
        config = make_config(representation="delta_from_current")
        policy = DiffusionPolicy(config, make_dataset_stats())
        hydra_config = OmegaConf.create(
            {
                "device": "cpu",
                "policy": {
                    "name": "diffusion",
                    **config.__dict__,
                },
            }
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            policy.save_pretrained(directory)
            restored = make_policy(
                hydra_config,
                pretrained_policy_name_or_path=directory,
                strict_pretrained_loading=True,
            )

        self.assertEqual(
            restored.view_action_representation,
            "delta_from_current",
        )
        torch.testing.assert_close(
            restored.unnormalize_outputs.buffer_action["min"],
            policy.unnormalize_outputs.buffer_action["min"],
        )

    def test_factory_rejects_absolute_checkpoint_as_delta_policy(self):
        absolute_config = make_config(representation="absolute")
        absolute_policy = DualHeadDiffusionPolicy(
            absolute_config,
            make_dataset_stats(with_view_delta=False),
        )
        delta_config = make_config(representation="delta_from_current")
        hydra_config = OmegaConf.create(
            {
                "device": "cpu",
                "policy": {
                    "name": "dual_head_diffusion",
                    **delta_config.__dict__,
                },
            }
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            absolute_policy.save_pretrained(directory)
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint的View动作表示",
            ):
                make_policy(
                    hydra_config,
                    pretrained_policy_name_or_path=directory,
                    strict_pretrained_loading=True,
                )

    def test_delta_checkpoint_round_trip_preserves_semantics_and_stats(self):
        config = make_config(representation="delta_from_current")
        policy = DualHeadDiffusionPolicy(config, make_dataset_stats())
        hydra_config = OmegaConf.create(
            {
                "device": "cpu",
                "policy": {
                    "name": "dual_head_diffusion",
                    **config.__dict__,
                },
            }
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            policy.save_pretrained(directory)
            restored = make_policy(
                hydra_config,
                pretrained_policy_name_or_path=directory,
                strict_pretrained_loading=True,
            )

        self.assertEqual(
            restored.view_action_representation,
            "delta_from_current",
        )
        torch.testing.assert_close(
            restored.unnormalize_outputs.buffer_action["min"],
            policy.unnormalize_outputs.buffer_action["min"],
        )


if __name__ == "__main__":
    unittest.main()
