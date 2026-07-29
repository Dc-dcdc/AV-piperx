import unittest

import torch
from omegaconf import OmegaConf

from lerobot.common.policies.diffusion.modeling_coupled_dual_head_diffusion import (
    CoupledDualHeadDiffusionPolicy,
)
from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DualHeadDiffusionPolicy,
)
from tests.test_coupled_dual_head_diffusion import (
    make_dataset_stats,
    make_small_config,
)
from train.s2_incremental.train.train_coupling_incremental import (
    CouplingDiagnostics,
    configure_incremental_trainable_parameters,
    install_frozen_module_mode_guard,
    make_incremental_optimizer_and_scheduler,
    migrate_dual_state_into_coupled,
    run_zero_gate_equivalence_check,
    update_incremental_policy,
)


class CouplingIncrementalTrainTest(unittest.TestCase):
    def test_dual_migration_only_leaves_new_coupling_state(self):
        config = make_small_config(
            "bidirectional_prefix_to_suffix",
            coupling_block_type="role_adaln_zero",
            coupling_use_temporal_pos_emb=True,
            coupling_use_ffn=True,
        )
        stats = make_dataset_stats()
        source = DualHeadDiffusionPolicy(config, stats)
        target = CoupledDualHeadDiffusionPolicy(config, stats)

        report = migrate_dual_state_into_coupled(source, target)
        equivalence = run_zero_gate_equivalence_check(
            source,
            target,
            atol=0.0,
            rtol=0.0,
        )

        self.assertGreater(report["shared_tensor_count"], 0)
        self.assertGreater(report["new_coupling_tensor_count"], 0)
        self.assertEqual(equivalence["max_abs_error"], 0.0)

    def test_single_direction_scalar_scope_freezes_entire_baseline(self):
        config = make_small_config(
            "bidirectional_prefix_to_suffix",
            coupling_block_type="scalar_gate",
        )
        config.view_to_arm_coupling_scale = 0.1
        config.arm_to_view_coupling_scale = 0.0
        policy = CoupledDualHeadDiffusionPolicy(config, make_dataset_stats())

        report = configure_incremental_trainable_parameters(policy)
        trainable_names = set(report["trainable_names"])

        self.assertTrue(
            any("view_to_arm_attention" in name for name in trainable_names)
        )
        self.assertFalse(
            any("arm_to_view_attention" in name for name in trainable_names)
        )
        self.assertTrue(
            any("coupling_timestep_encoder" in name for name in trainable_names)
        )
        self.assertTrue(any("coupling_norm" in name for name in trainable_names))
        self.assertFalse(any("arm_unet" in name for name in trainable_names))
        self.assertFalse(any("view_unet" in name for name in trainable_names))

    def test_role_adaln_ffn_scope_obeys_enabled_direction(self):
        config = make_small_config(
            "bidirectional_prefix_to_suffix",
            coupling_block_type="role_adaln_zero",
            coupling_use_ffn=True,
        )
        config.view_to_arm_coupling_scale = 0.1
        config.arm_to_view_coupling_scale = 0.0
        policy = CoupledDualHeadDiffusionPolicy(config, make_dataset_stats())

        report = configure_incremental_trainable_parameters(policy)
        trainable_names = set(report["trainable_names"])

        self.assertTrue(any(".arm_ffn." in name for name in trainable_names))
        self.assertFalse(any(".view_ffn." in name for name in trainable_names))
        self.assertTrue(
            any(".attention_modulation." in name for name in trainable_names)
        )
        self.assertTrue(any(".ffn_modulation." in name for name in trainable_names))

    def test_one_update_changes_only_coupling_parameters_and_logs_diagnostics(self):
        torch.manual_seed(71)
        config = make_small_config(
            "bidirectional_prefix_to_suffix",
            coupling_block_type="scalar_gate",
        )
        config.view_to_arm_coupling_scale = 0.1
        config.arm_to_view_coupling_scale = 0.0
        policy = CoupledDualHeadDiffusionPolicy(config, make_dataset_stats())
        configure_incremental_trainable_parameters(policy)
        install_frozen_module_mode_guard(policy)
        policy._incremental_init_report = {
            "equivalence": {"max_abs_error": 0.0}
        }
        policy._incremental_scope_report = {
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in policy.parameters()
                if parameter.requires_grad
            )
        }
        policy._coupling_diagnostics = CouplingDiagnostics(policy)

        frozen_before = {
            name: parameter.detach().clone()
            for name, parameter in policy.named_parameters()
            if not parameter.requires_grad
        }
        trainable_before = {
            name: parameter.detach().clone()
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        }
        cfg = OmegaConf.create(
            {
                "training": {
                    "lr": 1e-3,
                    "coupling_lr": 1e-3,
                    "coupling_structural_weight_decay": 1e-6,
                    "adam_betas": [0.95, 0.999],
                    "adam_eps": 1e-8,
                    "lr_scheduler": "constant",
                    "lr_warmup_steps": 0,
                    "offline_steps": 2,
                }
            }
        )
        optimizer, scheduler = make_incremental_optimizer_and_scheduler(cfg, policy)
        batch = {
            "observation.state": torch.randn(2, 2, 2),
            "observation.environment_state": torch.randn(2, 2, 3),
            "action": torch.randn(2, 8, 20).clamp(-1, 1),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        }

        info = update_incremental_policy(
            policy,
            batch,
            optimizer,
            10.0,
            grad_scaler=torch.amp.GradScaler("cuda", enabled=False),
            lr_scheduler=scheduler,
            use_amp=False,
            collect_metrics=True,
        )

        self.assertIn("coupling/gate/view_to_arm_mean", info)
        self.assertIn("coupling/residual/effective_arm_rms", info)
        self.assertIn("coupling/grad_norm/attention_gate", info)
        self.assertTrue(
            any(
                not torch.equal(trainable_before[name], parameter)
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad
            )
        )
        for name, parameter in policy.named_parameters():
            if name in frozen_before:
                torch.testing.assert_close(
                    parameter,
                    frozen_before[name],
                    rtol=0,
                    atol=0,
                )
        self.assertFalse(policy.diffusion.arm_unet.training)
        self.assertFalse(policy.diffusion.view_unet.training)


if __name__ == "__main__":
    unittest.main()
