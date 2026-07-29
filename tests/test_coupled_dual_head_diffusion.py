import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_coupled_dual_head_diffusion import (
    CoupledDualHeadDiffusionModel,
    CoupledDualHeadDiffusionPolicy,
)
from train.s3_finetune.finetune_dppo_dual_head import (
    forward_dppo_from_global_cond,
    get_logprobs_from_global_cond,
)
from train.s1_pretrain.eval.coupling_ablation import (
    apply_coupling_ablation_overrides,
    coupling_ablation_tag,
)


def make_small_config(
    coupling_mode: str = "full",
    *,
    coupling_block_type: str = "scalar_gate",
    coupling_use_temporal_pos_emb: bool = False,
    coupling_use_ffn: bool = False,
    coupling_active_max_timestep: int | None = None,
) -> DiffusionConfig:
    """构造无需图像编码器的小尺寸耦合双头测试配置。"""
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
        coupling_num_heads=4,
        coupling_dropout=0.0,
        coupling_mode=coupling_mode,
        coupling_block_type=coupling_block_type,
        coupling_use_temporal_pos_emb=coupling_use_temporal_pos_emb,
        coupling_use_ffn=coupling_use_ffn,
        coupling_ffn_ratio=2.0,
        coupling_active_max_timestep=coupling_active_max_timestep,
    )


def make_dataset_stats() -> dict[str, dict[str, torch.Tensor]]:
    """构造策略归一化和反归一化所需的单位测试统计量。"""
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


class CoupledDualHeadDiffusionTest(unittest.TestCase):
    def test_legacy_action_start_is_ignored_when_loading_config(self):
        """验证旧checkpoint中的action_start字段不会再改变原版动作切片。"""
        legacy_values = dict(make_small_config().__dict__)
        legacy_values["action_start"] = 0

        restored = DiffusionConfig.from_dict(legacy_values)

        self.assertFalse(hasattr(restored, "action_start"))
        self.assertEqual(restored.n_obs_steps - 1, 1)

    def test_all_coupling_modes_are_checkpoint_compatible(self):
        """验证六种模式只改变特征路由，不改变可学习参数名称或形状。"""
        torch.manual_seed(3)
        full_model = CoupledDualHeadDiffusionModel(make_small_config("full"))
        rbac_model = CoupledDualHeadDiffusionModel(make_small_config("rbac"))
        balanced_model = CoupledDualHeadDiffusionModel(
            make_small_config("balanced_lookahead")
        )
        rcla_model = CoupledDualHeadDiffusionModel(make_small_config("rcla"))
        prefix_to_suffix_model = CoupledDualHeadDiffusionModel(
            make_small_config("bidirectional_prefix_to_suffix")
        )
        half_prefix_to_suffix_model = CoupledDualHeadDiffusionModel(
            make_small_config("bidirectional_half_prefix_to_suffix")
        )

        rbac_incompatible = rbac_model.load_state_dict(
            full_model.state_dict(), strict=True
        )
        balanced_incompatible = balanced_model.load_state_dict(
            full_model.state_dict(), strict=True
        )
        rcla_incompatible = rcla_model.load_state_dict(
            full_model.state_dict(), strict=True
        )
        prefix_to_suffix_incompatible = prefix_to_suffix_model.load_state_dict(
            full_model.state_dict(), strict=True
        )
        half_prefix_to_suffix_incompatible = (
            half_prefix_to_suffix_model.load_state_dict(
                full_model.state_dict(),
                strict=True,
            )
        )

        self.assertEqual(rbac_incompatible.missing_keys, [])
        self.assertEqual(rbac_incompatible.unexpected_keys, [])
        self.assertEqual(balanced_incompatible.missing_keys, [])
        self.assertEqual(balanced_incompatible.unexpected_keys, [])
        self.assertEqual(rcla_incompatible.missing_keys, [])
        self.assertEqual(rcla_incompatible.unexpected_keys, [])
        self.assertEqual(prefix_to_suffix_incompatible.missing_keys, [])
        self.assertEqual(prefix_to_suffix_incompatible.unexpected_keys, [])
        self.assertEqual(half_prefix_to_suffix_incompatible.missing_keys, [])
        self.assertEqual(half_prefix_to_suffix_incompatible.unexpected_keys, [])

    def test_temporal_position_embedding_marks_each_bottleneck_token(self):
        """验证无参数时间编码具有[1,T,C]形状且不同位置可区分。"""
        model = CoupledDualHeadDiffusionModel(make_small_config())

        embedding = model._build_temporal_sinusoidal_embedding(
            4,
            32,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(tuple(embedding.shape), (1, 4, 32))
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertGreater((embedding[:, 0] - embedding[:, 1]).abs().sum().item(), 0.0)

    def test_role_adaln_zero_initially_matches_independent_heads(self):
        """验证adaLN、位置编码和FFN在零门控初始化时不改变原双UNet输出。"""
        torch.manual_seed(53)
        config = make_small_config(
            "bidirectional_prefix_to_suffix",
            coupling_block_type="role_adaln_zero",
            coupling_use_temporal_pos_emb=True,
            coupling_use_ffn=True,
        )
        model = CoupledDualHeadDiffusionModel(config).eval()
        arm_sample = torch.randn(2, 8, 14)
        view_sample = torch.randn(2, 8, 6)
        timesteps = torch.tensor([3, 11])
        global_cond = torch.randn(2, 10)

        with torch.no_grad():
            expected_arm = model.arm_unet(
                arm_sample, timesteps, global_cond=global_cond
            )
            expected_view = model.view_unet(
                view_sample, timesteps, global_cond=global_cond
            )
            actual_arm, actual_view = model.predict_noise(
                arm_sample, view_sample, timesteps, global_cond
            )

        torch.testing.assert_close(actual_arm, expected_arm, rtol=0, atol=0)
        torch.testing.assert_close(actual_view, expected_view, rtol=0, atol=0)

    def test_role_adaln_ffn_respects_prefix_to_suffix_write_mask(self):
        """验证开启adaLN与FFN后仍只修改双方最后一个token。"""
        torch.manual_seed(59)
        model = CoupledDualHeadDiffusionModel(
            make_small_config(
                "bidirectional_prefix_to_suffix",
                coupling_block_type="role_adaln_zero",
                coupling_use_temporal_pos_emb=True,
                coupling_use_ffn=True,
            )
        ).eval()
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))
            ffn_bias = model.role_adaln_coupling.ffn_modulation[-1].bias
            dim = model.role_adaln_coupling.bottleneck_dim
            ffn_bias[2 * dim] = 0.5
            ffn_bias[4 * dim + 1] = 0.5

        arm = torch.randn(2, 32, 4)
        view = torch.randn(2, 32, 4)
        with torch.no_grad():
            coupled_arm, coupled_view = model._couple_bottlenecks(
                arm, view, torch.tensor([3, 7])
            )

        torch.testing.assert_close(coupled_arm[..., :3], arm[..., :3], rtol=0, atol=0)
        torch.testing.assert_close(coupled_view[..., :3], view[..., :3], rtol=0, atol=0)
        self.assertGreater((coupled_arm[..., 3:] - arm[..., 3:]).abs().sum().item(), 0.0)
        self.assertGreater((coupled_view[..., 3:] - view[..., 3:]).abs().sum().item(), 0.0)

    def test_role_adaln_and_ffn_receive_gradients(self):
        """验证新时间调制、交叉注意力与FFN都接入联合损失计算图。"""
        torch.manual_seed(61)
        model = CoupledDualHeadDiffusionModel(
            make_small_config(
                "bidirectional_prefix_to_suffix",
                coupling_block_type="role_adaln_zero",
                coupling_use_temporal_pos_emb=True,
                coupling_use_ffn=True,
            )
        )
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))
            ffn_bias = model.role_adaln_coupling.ffn_modulation[-1].bias
            dim = model.role_adaln_coupling.bottleneck_dim
            ffn_bias[2 * dim] = 0.5
            ffn_bias[4 * dim + 1] = 0.5

        coupled_arm, coupled_view = model._couple_bottlenecks(
            torch.randn(2, 32, 4),
            torch.randn(2, 32, 4),
            torch.tensor([2, 9]),
        )
        (coupled_arm.square().mean() + coupled_view.square().mean()).backward()

        role_block = model.role_adaln_coupling
        self.assertGreater(
            role_block.attention_modulation[-1].weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            role_block.ffn_modulation[-1].weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(role_block.arm_ffn[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(role_block.view_ffn[0].weight.grad.abs().sum().item(), 0.0)

    def test_scalar_gate_rejects_coupling_ffn(self):
        """验证FFN只能与role_adaln_zero组合，避免配置被静默忽略。"""
        with self.assertRaisesRegex(ValueError, "role_adaln_zero"):
            make_small_config(coupling_use_ffn=True)

    def test_low_noise_coupling_mask_supports_both_block_types(self):
        """验证高噪声样本严格关闭Attention/FFN，低噪声样本仍可被修正。"""
        for block_type, use_ffn in (
            ("scalar_gate", False),
            ("role_adaln_zero", True),
        ):
            with self.subTest(block_type=block_type):
                torch.manual_seed(71)
                model = CoupledDualHeadDiffusionModel(
                    make_small_config(
                        "bidirectional_prefix_to_suffix",
                        coupling_block_type=block_type,
                        coupling_use_ffn=use_ffn,
                        coupling_active_max_timestep=4,
                    )
                ).train()
                with torch.no_grad():
                    model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))
                    if use_ffn:
                        ffn_bias = model.role_adaln_coupling.ffn_modulation[-1].bias
                        dim = model.role_adaln_coupling.bottleneck_dim
                        ffn_bias[2 * dim] = 0.5
                        ffn_bias[4 * dim + 1] = 0.5

                arm = torch.randn(2, 32, 4)
                view = torch.randn(2, 32, 4)
                coupled_arm, coupled_view = model._couple_bottlenecks(
                    arm,
                    view,
                    torch.tensor([4, 5]),
                )

                self.assertGreater(
                    (coupled_arm[0] - arm[0]).abs().sum().item(),
                    0.0,
                )
                self.assertGreater(
                    (coupled_view[0] - view[0]).abs().sum().item(),
                    0.0,
                )
                torch.testing.assert_close(
                    coupled_arm[1],
                    arm[1],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    coupled_view[1],
                    view[1],
                    rtol=0,
                    atol=0,
                )

    def test_low_noise_coupling_mask_skips_high_noise_attention_at_inference(self):
        """验证推理前段全部高噪声时不执行交叉注意力。"""
        model = CoupledDualHeadDiffusionModel(
            make_small_config(coupling_active_max_timestep=4)
        ).eval()
        arm = torch.randn(2, 32, 4)
        view = torch.randn(2, 32, 4)

        with (
            mock.patch.object(
                model.view_to_arm_attention,
                "forward",
                wraps=model.view_to_arm_attention.forward,
            ) as view_to_arm_forward,
            mock.patch.object(
                model.arm_to_view_attention,
                "forward",
                wraps=model.arm_to_view_attention.forward,
            ) as arm_to_view_forward,
            torch.no_grad(),
        ):
            coupled_arm, coupled_view = model._couple_bottlenecks(
                arm,
                view,
                torch.tensor([5, 19]),
            )

        view_to_arm_forward.assert_not_called()
        arm_to_view_forward.assert_not_called()
        torch.testing.assert_close(coupled_arm, arm, rtol=0, atol=0)
        torch.testing.assert_close(coupled_view, view, rtol=0, atol=0)

    def test_low_noise_schedule_does_not_change_checkpoint_parameters(self):
        """验证新增阈值不增加参数，旧的全时间步checkpoint仍可严格加载。"""
        unrestricted = CoupledDualHeadDiffusionModel(make_small_config())
        low_noise_only = CoupledDualHeadDiffusionModel(
            make_small_config(coupling_active_max_timestep=4)
        )

        incompatible = low_noise_only.load_state_dict(
            unrestricted.state_dict(),
            strict=True,
        )

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(
            set(low_noise_only.state_dict()),
            set(unrestricted.state_dict()),
        )

    def test_coupling_active_max_timestep_validation(self):
        """验证低噪声阈值只能取合法训练时间步，且不把bool误当整数。"""
        for invalid_value in (-1, 20, True, 1.5):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(
                    ValueError,
                    "coupling_active_max_timestep",
                ):
                    DiffusionConfig(
                        **{
                            **make_small_config().__dict__,
                            "coupling_active_max_timestep": invalid_value,
                        }
                    )

    def test_rcla_builds_role_causal_lag_masks(self):
        """验证4-token RCLA掩码与角色因果时滞交互图严格一致。"""
        model = CoupledDualHeadDiffusionModel(make_small_config("rcla"))

        view_to_arm_mask, arm_to_view_mask = model._build_rcla_attention_masks(
            token_count=4, device=torch.device("cpu")
        )

        expected_view_to_arm = torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
                [True, False, False, True],
                [True, True, False, False],
            ]
        )
        expected_arm_to_view = torch.tensor(
            [
                [False, False, True, True],
                [True, False, False, True],
                [True, True, False, False],
                [True, True, True, False],
            ]
        )
        torch.testing.assert_close(view_to_arm_mask, expected_view_to_arm)
        torch.testing.assert_close(arm_to_view_mask, expected_arm_to_view)

    def test_rcla_routes_gradients_to_both_attention_directions(self):
        """验证RCLA两个局部交互方向均可由联合损失训练。"""
        torch.manual_seed(37)
        model = CoupledDualHeadDiffusionModel(make_small_config("rcla"))
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))

        coupled_arm, coupled_view = model._couple_bottlenecks(
            torch.randn(2, 32, 4),
            torch.randn(2, 32, 4),
            torch.tensor([2, 9]),
        )
        (coupled_arm.square().mean() + coupled_view.square().mean()).backward()

        view_to_arm_grad = model.view_to_arm_attention.in_proj_weight.grad
        arm_to_view_grad = model.arm_to_view_attention.in_proj_weight.grad
        self.assertIsNotNone(view_to_arm_grad)
        self.assertIsNotNone(arm_to_view_grad)
        self.assertGreater(view_to_arm_grad.abs().sum().item(), 0.0)
        self.assertGreater(arm_to_view_grad.abs().sum().item(), 0.0)

    def test_rcla_runs_end_to_end_noise_prediction(self):
        """验证RCLA可通过完整双UNet编码、局部耦合与解码路径。"""
        torch.manual_seed(41)
        model = CoupledDualHeadDiffusionModel(make_small_config("rcla")).eval()

        with torch.no_grad():
            arm_noise, view_noise = model.predict_noise(
                torch.randn(2, 8, 14),
                torch.randn(2, 8, 6),
                torch.tensor([3, 7]),
                torch.randn(2, 10),
            )

        self.assertEqual(tuple(arm_noise.shape), (2, 8, 14))
        self.assertEqual(tuple(view_noise.shape), (2, 8, 6))
        self.assertTrue(torch.isfinite(arm_noise).all())
        self.assertTrue(torch.isfinite(view_noise).all())

    def test_rbac_projects_execution_boundary_to_bottleneck_tokens(self):
        """验证原版起点1将执行区间投影为前三个瓶颈token。"""
        model = CoupledDualHeadDiffusionModel(make_small_config("rbac"))

        view_prefix, arm_future = model._resolve_rbac_token_slices(token_count=4)

        self.assertEqual((view_prefix.start, view_prefix.stop), (0, 3))
        self.assertEqual((arm_future.start, arm_future.stop), (3, 4))

    def test_rbac_only_updates_executed_view_and_future_arm_tokens(self):
        """验证RBAC不会把未来View注入当前Arm，也不会修改未执行View后缀。"""
        torch.manual_seed(5)
        model = CoupledDualHeadDiffusionModel(make_small_config("rbac")).eval()
        # 绕过零初始化门控，仅测试耦合路由本身。
        gate = model.coupling_timestep_encoder[-1]
        with torch.no_grad():
            gate.bias.fill_(math.atanh(0.5))

        arm = torch.randn(2, 32, 4)
        view = torch.randn(2, 32, 4)
        with torch.no_grad():
            coupled_arm, coupled_view = model._couple_bottlenecks(
                arm, view, torch.tensor([3, 7])
            )

        # Arm执行前缀[0:3]与View预测后缀[3:4]保持逐bit不变。
        torch.testing.assert_close(coupled_arm[..., :3], arm[..., :3], rtol=0, atol=0)
        torch.testing.assert_close(coupled_view[..., 3:], view[..., 3:], rtol=0, atol=0)
        self.assertGreater((coupled_arm[..., 3:] - arm[..., 3:]).abs().sum().item(), 0.0)
        self.assertGreater((coupled_view[..., :3] - view[..., :3]).abs().sum().item(), 0.0)

    def test_rbac_routes_gradients_to_both_attention_directions(self):
        """验证切片写回完整token后，两个RBAC注意力方向仍保留反向传播图。"""
        torch.manual_seed(6)
        model = CoupledDualHeadDiffusionModel(make_small_config("rbac"))
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))

        coupled_arm, coupled_view = model._couple_bottlenecks(
            torch.randn(2, 32, 4),
            torch.randn(2, 32, 4),
            torch.tensor([2, 9]),
        )
        (coupled_arm.square().mean() + coupled_view.square().mean()).backward()

        view_to_arm_grad = model.view_to_arm_attention.in_proj_weight.grad
        arm_to_view_grad = model.arm_to_view_attention.in_proj_weight.grad
        self.assertIsNotNone(view_to_arm_grad)
        self.assertIsNotNone(arm_to_view_grad)
        self.assertGreater(view_to_arm_grad.abs().sum().item(), 0.0)
        self.assertGreater(arm_to_view_grad.abs().sum().item(), 0.0)

    def test_bidirectional_prefix_to_suffix_only_updates_both_suffixes(self):
        """验证两路前3个token只修正对方最后1个token。"""
        torch.manual_seed(43)
        model = CoupledDualHeadDiffusionModel(
            make_small_config("bidirectional_prefix_to_suffix")
        ).eval()
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))

        arm = torch.randn(2, 32, 4)
        view = torch.randn(2, 32, 4)
        with torch.no_grad():
            coupled_arm, coupled_view = model._couple_bottlenecks(
                arm, view, torch.tensor([3, 7])
            )

        torch.testing.assert_close(coupled_arm[..., :3], arm[..., :3], rtol=0, atol=0)
        torch.testing.assert_close(coupled_view[..., :3], view[..., :3], rtol=0, atol=0)
        self.assertGreater((coupled_arm[..., 3:] - arm[..., 3:]).abs().sum().item(), 0.0)
        self.assertGreater((coupled_view[..., 3:] - view[..., 3:]).abs().sum().item(), 0.0)

    def test_bidirectional_prefix_to_suffix_routes_both_gradients(self):
        """验证两个prefix-to-suffix注意力方向均能获得梯度。"""
        torch.manual_seed(47)
        model = CoupledDualHeadDiffusionModel(
            make_small_config("bidirectional_prefix_to_suffix")
        )
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))

        coupled_arm, coupled_view = model._couple_bottlenecks(
            torch.randn(2, 32, 4),
            torch.randn(2, 32, 4),
            torch.tensor([2, 9]),
        )
        (coupled_arm.square().mean() + coupled_view.square().mean()).backward()

        view_to_arm_grad = model.view_to_arm_attention.in_proj_weight.grad
        arm_to_view_grad = model.arm_to_view_attention.in_proj_weight.grad
        self.assertIsNotNone(view_to_arm_grad)
        self.assertIsNotNone(arm_to_view_grad)
        self.assertGreater(view_to_arm_grad.abs().sum().item(), 0.0)
        self.assertGreater(arm_to_view_grad.abs().sum().item(), 0.0)

    def test_bidirectional_half_prefix_to_suffix_uses_exact_two_by_two_mapping(self):
        """验证Arm/View后缀[2,3]分别只读取对方前缀[0,1]。"""
        torch.manual_seed(67)
        model = CoupledDualHeadDiffusionModel(
            make_small_config("bidirectional_half_prefix_to_suffix")
        ).eval()
        normalized_arm = torch.randn(2, 4, 32)
        normalized_view = torch.randn(2, 4, 32)

        with (
            mock.patch.object(
                model.view_to_arm_attention,
                "forward",
                wraps=model.view_to_arm_attention.forward,
            ) as view_to_arm_forward,
            mock.patch.object(
                model.arm_to_view_attention,
                "forward",
                wraps=model.arm_to_view_attention.forward,
            ) as arm_to_view_forward,
            torch.no_grad(),
        ):
            view_context, arm_context = model._compute_coupling_contexts(
                normalized_arm,
                normalized_view,
            )

        view_to_arm_call = view_to_arm_forward.call_args.kwargs
        arm_to_view_call = arm_to_view_forward.call_args.kwargs
        torch.testing.assert_close(
            view_to_arm_call["query"], normalized_arm[:, 2:4]
        )
        torch.testing.assert_close(
            view_to_arm_call["key"], normalized_view[:, 0:2]
        )
        torch.testing.assert_close(
            view_to_arm_call["value"], normalized_view[:, 0:2]
        )
        torch.testing.assert_close(
            arm_to_view_call["query"], normalized_view[:, 2:4]
        )
        torch.testing.assert_close(
            arm_to_view_call["key"], normalized_arm[:, 0:2]
        )
        torch.testing.assert_close(
            arm_to_view_call["value"], normalized_arm[:, 0:2]
        )
        torch.testing.assert_close(
            view_context[:, :2],
            torch.zeros_like(view_context[:, :2]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            arm_context[:, :2],
            torch.zeros_like(arm_context[:, :2]),
            rtol=0,
            atol=0,
        )

    def test_bidirectional_half_prefix_to_suffix_only_updates_both_suffix_halves(self):
        """验证开启AdaLN/FFN后仍只修改双方后缀[2,3]。"""
        torch.manual_seed(71)
        model = CoupledDualHeadDiffusionModel(
            make_small_config(
                "bidirectional_half_prefix_to_suffix",
                coupling_block_type="role_adaln_zero",
                coupling_use_temporal_pos_emb=True,
                coupling_use_ffn=True,
            )
        ).eval()
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))
            ffn_bias = model.role_adaln_coupling.ffn_modulation[-1].bias
            dim = model.role_adaln_coupling.bottleneck_dim
            ffn_bias[2 * dim] = 0.5
            ffn_bias[4 * dim + 1] = 0.5

        arm = torch.randn(2, 32, 4)
        view = torch.randn(2, 32, 4)
        with torch.no_grad():
            coupled_arm, coupled_view = model._couple_bottlenecks(
                arm,
                view,
                torch.tensor([3, 7]),
            )

        torch.testing.assert_close(
            coupled_arm[..., :2], arm[..., :2], rtol=0, atol=0
        )
        torch.testing.assert_close(
            coupled_view[..., :2], view[..., :2], rtol=0, atol=0
        )
        self.assertGreater(
            (coupled_arm[..., 2:] - arm[..., 2:]).abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            (coupled_view[..., 2:] - view[..., 2:]).abs().sum().item(),
            0.0,
        )

    def test_bidirectional_half_prefix_to_suffix_routes_both_gradients(self):
        """验证新模式两个2×2交叉注意力方向都能获得梯度。"""
        torch.manual_seed(73)
        model = CoupledDualHeadDiffusionModel(
            make_small_config("bidirectional_half_prefix_to_suffix")
        )
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))

        coupled_arm, coupled_view = model._couple_bottlenecks(
            torch.randn(2, 32, 4),
            torch.randn(2, 32, 4),
            torch.tensor([2, 9]),
        )
        (coupled_arm.square().mean() + coupled_view.square().mean()).backward()

        view_to_arm_grad = model.view_to_arm_attention.in_proj_weight.grad
        arm_to_view_grad = model.arm_to_view_attention.in_proj_weight.grad
        self.assertIsNotNone(view_to_arm_grad)
        self.assertIsNotNone(arm_to_view_grad)
        self.assertGreater(view_to_arm_grad.abs().sum().item(), 0.0)
        self.assertGreater(arm_to_view_grad.abs().sum().item(), 0.0)

    def test_bidirectional_half_prefix_to_suffix_runs_end_to_end(self):
        """验证新模式可通过完整双UNet编码、耦合与解码路径。"""
        torch.manual_seed(79)
        model = CoupledDualHeadDiffusionModel(
            make_small_config("bidirectional_half_prefix_to_suffix")
        ).eval()

        with torch.no_grad():
            arm_noise, view_noise = model.predict_noise(
                torch.randn(2, 8, 14),
                torch.randn(2, 8, 6),
                torch.tensor([3, 7]),
                torch.randn(2, 10),
            )

        self.assertEqual(tuple(arm_noise.shape), (2, 8, 14))
        self.assertEqual(tuple(view_noise.shape), (2, 8, 6))
        self.assertTrue(torch.isfinite(arm_noise).all())
        self.assertTrue(torch.isfinite(view_noise).all())

    def test_balanced_lookahead_splits_four_tokens_into_equal_halves(self):
        """验证当前4-token瓶颈被确定地切分为[0:2]与[2:4]。"""
        model = CoupledDualHeadDiffusionModel(
            make_small_config("balanced_lookahead")
        )

        current, future = model._resolve_balanced_lookahead_token_slices(4)

        self.assertEqual((current.start, current.stop), (0, 2))
        self.assertEqual((future.start, future.stop), (2, 4))

    def test_balanced_lookahead_only_updates_current_half_of_both_heads(self):
        """验证两路future后缀只修正对方current前缀，future瓶颈保持不变。"""
        torch.manual_seed(23)
        model = CoupledDualHeadDiffusionModel(
            make_small_config("balanced_lookahead")
        ).eval()
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))

        arm = torch.randn(2, 32, 4)
        view = torch.randn(2, 32, 4)
        with torch.no_grad():
            coupled_arm, coupled_view = model._couple_bottlenecks(
                arm, view, torch.tensor([3, 7])
            )

        torch.testing.assert_close(coupled_arm[..., 2:], arm[..., 2:], rtol=0, atol=0)
        torch.testing.assert_close(coupled_view[..., 2:], view[..., 2:], rtol=0, atol=0)
        self.assertGreater((coupled_arm[..., :2] - arm[..., :2]).abs().sum().item(), 0.0)
        self.assertGreater((coupled_view[..., :2] - view[..., :2]).abs().sum().item(), 0.0)

    def test_balanced_lookahead_routes_gradients_to_both_directions(self):
        """验证两个2×2 cross-attention路径均可由联合损失训练。"""
        torch.manual_seed(29)
        model = CoupledDualHeadDiffusionModel(
            make_small_config("balanced_lookahead")
        )
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))

        coupled_arm, coupled_view = model._couple_bottlenecks(
            torch.randn(2, 32, 4),
            torch.randn(2, 32, 4),
            torch.tensor([2, 9]),
        )
        (coupled_arm.square().mean() + coupled_view.square().mean()).backward()

        view_to_arm_grad = model.view_to_arm_attention.in_proj_weight.grad
        arm_to_view_grad = model.arm_to_view_attention.in_proj_weight.grad
        self.assertIsNotNone(view_to_arm_grad)
        self.assertIsNotNone(arm_to_view_grad)
        self.assertGreater(view_to_arm_grad.abs().sum().item(), 0.0)
        self.assertGreater(arm_to_view_grad.abs().sum().item(), 0.0)

    def test_balanced_lookahead_rejects_unbalanced_token_count(self):
        """验证奇数token不会被静默切成不对称注意力。"""
        model = CoupledDualHeadDiffusionModel(
            make_small_config("balanced_lookahead")
        )

        with self.assertRaisesRegex(ValueError, "偶数"):
            model._resolve_balanced_lookahead_token_slices(3)

    def test_balanced_lookahead_runs_end_to_end_noise_prediction(self):
        """验证新路由可通过完整双UNet编码、耦合与解码路径。"""
        torch.manual_seed(31)
        model = CoupledDualHeadDiffusionModel(
            make_small_config("balanced_lookahead")
        ).eval()

        with torch.no_grad():
            arm_noise, view_noise = model.predict_noise(
                torch.randn(2, 8, 14),
                torch.randn(2, 8, 6),
                torch.tensor([3, 7]),
                torch.randn(2, 10),
            )

        self.assertEqual(tuple(arm_noise.shape), (2, 8, 14))
        self.assertEqual(tuple(view_noise.shape), (2, 8, 6))
        self.assertTrue(torch.isfinite(arm_noise).all())
        self.assertTrue(torch.isfinite(view_noise).all())

    def test_directional_scales_disable_only_the_named_coupling_path(self):
        """验证v2a只修改Arm、a2v只修改View，避免实验标签与门方向写反。"""
        torch.manual_seed(19)
        model = CoupledDualHeadDiffusionModel(make_small_config("rbac")).eval()
        with torch.no_grad():
            model.coupling_timestep_encoder[-1].bias.fill_(math.atanh(0.5))

        arm = torch.randn(2, 32, 4)
        view = torch.randn(2, 32, 4)
        timesteps = torch.tensor([3, 7])

        model.set_coupling_scales(view_to_arm=0.0, arm_to_view=1.0)
        with torch.no_grad():
            arm_without_v2a, view_with_a2v = model._couple_bottlenecks(
                arm, view, timesteps
            )
        torch.testing.assert_close(arm_without_v2a, arm, rtol=0, atol=0)
        self.assertGreater((view_with_a2v - view).abs().sum().item(), 0.0)

        model.set_coupling_scales(view_to_arm=1.0, arm_to_view=0.0)
        with torch.no_grad():
            arm_with_v2a, view_without_a2v = model._couple_bottlenecks(
                arm, view, timesteps
            )
        self.assertGreater((arm_with_v2a - arm).abs().sum().item(), 0.0)
        torch.testing.assert_close(view_without_a2v, view, rtol=0, atol=0)

    def test_inference_ablation_override_updates_loaded_model_without_parameters(self):
        """验证评估配置可在加载权重后覆盖标量，且不会改变state_dict结构。"""
        model = CoupledDualHeadDiffusionModel(make_small_config("rbac"))
        policy = SimpleNamespace(diffusion=model, config=model.config)
        before_keys = set(model.state_dict())
        eval_cfg = SimpleNamespace(
            view_to_arm_coupling_scale=0.0,
            arm_to_view_coupling_scale=0.25,
        )

        active = apply_coupling_ablation_overrides(policy, eval_cfg)

        self.assertEqual(
            active,
            {
                "view_to_arm_coupling_scale": 0.0,
                "arm_to_view_coupling_scale": 0.25,
            },
        )
        self.assertEqual(coupling_ablation_tag(eval_cfg), "_v2a=0_a2v=0.25")
        self.assertEqual(set(model.state_dict()), before_keys)

    def test_coupling_scale_validation_rejects_unsafe_values(self):
        """验证方向缩放只允许有限的[0,1]区间。"""
        with self.assertRaisesRegex(ValueError, "view_to_arm_coupling_scale"):
            DiffusionConfig(
                **{
                    **make_small_config().__dict__,
                    "view_to_arm_coupling_scale": -0.1,
                }
            )
        model = CoupledDualHeadDiffusionModel(make_small_config())
        with self.assertRaisesRegex(ValueError, "arm_to_view_coupling_scale"):
            model.set_coupling_scales(arm_to_view=float("nan"))

    def test_rbac_requires_unexecuted_prediction_suffix(self):
        """验证执行完整horizon时拒绝启用没有未来意图后缀的RBAC。"""
        with self.assertRaisesRegex(ValueError, "unexecuted action suffix"):
            DiffusionConfig(
                **{
                    **make_small_config().__dict__,
                    "coupling_mode": "rbac",
                    "n_action_steps": 7,
                }
            )

    def test_zero_initialized_coupling_matches_independent_heads(self):
        """验证初始门控为零时，联合预测严格退化为原独立双头预测。"""
        torch.manual_seed(7)
        model = CoupledDualHeadDiffusionModel(make_small_config()).eval()
        arm_sample = torch.randn(2, 8, 14)
        view_sample = torch.randn(2, 8, 6)
        timesteps = torch.tensor([3, 11])
        global_cond = torch.randn(2, 10)

        with torch.no_grad():
            expected_arm = model.arm_unet(
                arm_sample, timesteps, global_cond=global_cond
            )
            expected_view = model.view_unet(
                view_sample, timesteps, global_cond=global_cond
            )
            actual_arm, actual_view = model.predict_noise(
                arm_sample, view_sample, timesteps, global_cond
            )

        torch.testing.assert_close(actual_arm, expected_arm, rtol=0, atol=0)
        torch.testing.assert_close(actual_view, expected_view, rtol=0, atol=0)

    def test_joint_loss_trains_coupling_gate(self):
        """验证联合预训练损失能向时间步耦合门控传播有效梯度。"""
        torch.manual_seed(11)
        model = CoupledDualHeadDiffusionModel(make_small_config())
        batch = {
            "observation.state": torch.randn(2, 2, 2),
            "observation.environment_state": torch.randn(2, 2, 3),
            "action": torch.randn(2, 8, 20),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        }

        losses = model.compute_loss(batch)
        losses["loss"].backward()
        gate_output = model.coupling_timestep_encoder[-1]

        self.assertEqual(set(losses), {"loss", "arm_loss", "view_loss"})
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIsNotNone(gate_output.weight.grad)
        self.assertGreater(gate_output.weight.grad.abs().sum().item(), 0.0)

    def test_joint_sampler_uses_synchronized_head_shapes(self):
        """验证两个动作头在同一去噪循环中输出各自的完整轨迹。"""
        torch.manual_seed(13)
        model = CoupledDualHeadDiffusionModel(make_small_config()).eval()
        with torch.no_grad():
            arm_actions, view_actions = model.conditional_sample_coupled(
                batch_size=2,
                global_cond=torch.randn(2, 10),
            )

        self.assertEqual(tuple(arm_actions.shape), (2, 8, 14))
        self.assertEqual(tuple(view_actions.shape), (2, 8, 6))
        torch.testing.assert_close(
            model.arm_noise_scheduler.timesteps,
            model.view_noise_scheduler.timesteps,
        )

    def test_dppo_chain_and_likelihood_share_coupled_prediction(self):
        """验证耦合DPPO采样链可由同一联合模型计算有限的转移likelihood。"""
        torch.manual_seed(17)
        config = make_small_config()
        config.ft_denoising_steps = 2
        config.ddim_eta = 0.0
        config.min_sampling_denoising_std = 0.01
        config.min_logprob_denoising_std = 0.01
        config.randn_clip_value = 3.0
        config.final_action_clip_value = None
        config.denoised_clip_value = None
        config.eps_clip_value = None
        config.logprob_reduction = "mean"
        policy = CoupledDualHeadDiffusionPolicy(config, make_dataset_stats())
        global_cond = torch.randn(2, 10)

        rollout = forward_dppo_from_global_cond(
            policy, global_cond, return_chain=True
        )
        chain = rollout["chains"]
        timestep = policy.diffusion.arm_noise_scheduler.timesteps[0]
        timestep_batch = torch.full((2,), int(timestep.item()), dtype=torch.long)
        logprobs = get_logprobs_from_global_cond(
            policy,
            global_cond,
            chain[:, 0],
            chain[:, 1],
            timestep_batch,
        )

        self.assertEqual(tuple(rollout["actions"].shape), (2, 8, 20))
        self.assertEqual(tuple(chain.shape), (2, 3, 8, 20))
        self.assertEqual(set(logprobs), {"arm", "view", "joint"})
        self.assertTrue(all(torch.isfinite(value).all() for value in logprobs.values()))


if __name__ == "__main__":
    unittest.main()
