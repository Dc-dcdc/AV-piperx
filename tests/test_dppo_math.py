import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from train.finetune.dppo_math import (
    clipped_ppo_loss,
    combine_head_logprobs,
    dppo_ddim_mean_std,
    finalize_ppo_ratio_stats,
    init_ppo_ratio_stats,
    resolve_action_slice,
    summarize_ppo_ratio,
    update_ppo_ratio_stats,
)
from train.finetune.finetune_dppo_dual_head import (
    _head_ddim_mean_std,
    _head_logprob,
    _sample_head_dppo,
    get_logprobs_from_global_cond,
)


class FakeScheduler:
    """提供 DPPO 数学测试所需的最小 scheduler 接口。"""

    def __init__(self, num_train_timesteps=100, num_inference_steps=10):
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps)
        self.num_inference_steps = num_inference_steps
        self.alphas_cumprod = torch.linspace(0.999, 0.1, num_train_timesteps)

    def set_timesteps(self, num_inference_steps):
        """生成足够测试最终转移的简化 timestep 序列。"""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(
            self.config.num_train_timesteps - step_ratio,
            -1,
            -step_ratio,
            dtype=torch.long,
        )


class ZeroUNet(torch.nn.Module):
    """返回零 epsilon 的最小可调用 UNet 替身。"""

    def forward(self, sample, timesteps, global_cond=None):
        return torch.zeros_like(sample)


class FakePolicy(torch.nn.Module):
    """提供双头 DPPO helper 所需属性的最小策略替身。"""

    def __init__(self, *, min_std=0.05, horizon=2, inference_steps=1):
        super().__init__()
        self.dtype_anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            horizon=horizon,
            ft_denoising_steps=inference_steps,
            min_sampling_denoising_std=min_std,
            min_logprob_denoising_std=min_std,
            ddim_eta=1.0,
            randn_clip_value=None,
            final_action_clip_value=None,
            denoised_clip_value=None,
            eps_clip_value=None,
            prediction_type="epsilon",
            n_obs_steps=1,
            n_action_steps=horizon,
            logprob_reduction="mean",
        )
        self.diffusion = SimpleNamespace(num_inference_steps=inference_steps)


class DPPOActionSliceTest(unittest.TestCase):
    def test_action_slice_starts_at_last_observation(self):
        """验证两帧历史观测按原版规则从动作索引1开始执行。"""
        config = SimpleNamespace(n_obs_steps=2, n_action_steps=8)
        self.assertEqual(resolve_action_slice(config, horizon=16), (1, 9))

    def test_one_observation_starts_at_zero(self):
        """验证只有一帧观测时动作仍从索引0开始执行。"""
        config = SimpleNamespace(n_obs_steps=1, n_action_steps=4)
        self.assertEqual(resolve_action_slice(config, horizon=8), (0, 4))

    def test_action_slice_rejects_horizon_overflow(self):
        """验证动作切片超过预测 horizon 时会立即报错。"""
        config = SimpleNamespace(n_obs_steps=4, n_action_steps=6)
        with self.assertRaisesRegex(ValueError, "动作切片越界"):
            resolve_action_slice(config, horizon=8)


def reference_ddim_mean_std(sample, model_output, timesteps, scheduler, eta, min_std):
    """用直接展开的公式生成独立参考结果。"""
    timesteps = torch.as_tensor(timesteps, device=sample.device, dtype=torch.long)
    alphas = scheduler.alphas_cumprod.to(sample)
    alpha_t = alphas[timesteps].view(-1, 1, 1)
    step_ratio = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    previous = timesteps - step_ratio
    alpha_previous = torch.where(
        previous >= 0,
        alphas[previous.clamp(min=0)],
        torch.ones_like(previous, dtype=sample.dtype),
    ).view(-1, 1, 1)
    predicted_sample = (
        sample - torch.sqrt((1 - alpha_t).clamp(min=1e-12)) * model_output
    ) / torch.sqrt(alpha_t.clamp(min=1e-12))
    sigma = eta * torch.sqrt(
        (
            ((1 - alpha_previous) / (1 - alpha_t).clamp(min=1e-12))
            * (1 - alpha_t / alpha_previous.clamp(min=1e-12))
        ).clamp(min=0)
    )
    sigma = sigma.clamp(min=1e-10)
    mean = (
        torch.sqrt(alpha_previous.clamp(min=0)) * predicted_sample
        + torch.sqrt((1 - alpha_previous - sigma**2).clamp(min=0))
        * model_output
    )
    return mean, sigma.clamp(min=min_std)


class DDPODDIMMeanStdTest(unittest.TestCase):
    def test_scalar_and_batched_timesteps_match_reference_formula(self):
        """验证标量和批量 timestep 均与原 DPPO DDIM 公式一致。"""
        scheduler = FakeScheduler()
        for timesteps, batch_size in ((torch.tensor(70), 2), (torch.tensor([70, 30]), 2)):
            sample = torch.linspace(-0.6, 0.8, batch_size * 4 * 3).reshape(
                batch_size, 4, 3
            )
            model_output = sample.flip(-1) * 0.2
            actual_mean, actual_std = dppo_ddim_mean_std(
                sample,
                model_output,
                timesteps,
                scheduler,
                eta=0.7,
                min_std=0.05,
            )
            expected_mean, expected_std = reference_ddim_mean_std(
                sample,
                model_output,
                timesteps,
                scheduler,
                eta=0.7,
                min_std=0.05,
            )
            torch.testing.assert_close(actual_mean, expected_mean)
            torch.testing.assert_close(actual_std, expected_std)

    def test_last_timestep_uses_minimum_stochastic_std(self):
        """验证最终去噪步仍具有最小采样方差，而不是退化为确定性。"""
        scheduler = FakeScheduler()
        sample = torch.zeros(2, 3, 4)
        model_output = torch.zeros_like(sample)
        mean, std = dppo_ddim_mean_std(
            sample,
            model_output,
            torch.tensor(0),
            scheduler,
            eta=1.0,
            min_std=0.05,
        )

        torch.testing.assert_close(std, torch.full_like(std, 0.05))
        transition_noise = torch.ones_like(mean)
        sampled_next = mean + std * transition_noise
        torch.testing.assert_close(sampled_next - mean, torch.full_like(mean, 0.05))

    def test_gaussian_logprob_matches_sampled_transition(self):
        """验证 mean/std 对应的解析高斯 likelihood 数值。"""
        scheduler = FakeScheduler()
        sample = torch.zeros(1, 2, 3)
        model_output = torch.zeros_like(sample)
        mean, std = dppo_ddim_mean_std(
            sample,
            model_output,
            torch.tensor(0),
            scheduler,
            eta=1.0,
            min_std=0.2,
        )
        next_sample = mean + 0.5 * std
        logprob = (
            -0.5 * ((next_sample - mean) ** 2) / (std**2)
            - torch.log(std)
            - 0.5 * math.log(2 * math.pi)
        )
        expected = -0.5 * 0.5**2 - math.log(0.2) - 0.5 * math.log(2 * math.pi)
        torch.testing.assert_close(logprob, torch.full_like(logprob, expected))

    def test_dual_head_sampler_keeps_final_transition_stochastic(self):
        """验证双头 sampler 最后一步实际执行 mean+std*noise。"""
        policy = FakePolicy(min_std=0.05, horizon=2, inference_steps=1)
        scheduler = FakeScheduler(num_train_timesteps=10, num_inference_steps=1)
        global_cond = torch.zeros(2, 3)
        unet = ZeroUNet()

        with patch(
            "train.finetune.finetune_dppo_dual_head.torch.randn_like",
            side_effect=lambda tensor: torch.ones_like(tensor),
        ):
            _, chain = _sample_head_dppo(
                policy,
                global_cond,
                unet,
                scheduler,
                action_dim=4,
                return_chain=True,
            )

        self.assertEqual(tuple(chain.shape), (2, 2, 2, 4))
        mean, std = _head_ddim_mean_std(
            policy,
            chain[:, 0],
            torch.zeros_like(chain[:, 0]),
            scheduler.timesteps[-1],
            scheduler,
            "min_sampling_denoising_std",
        )
        torch.testing.assert_close(chain[:, 1], mean + std)

    def test_dual_head_logprob_uses_same_transition_distribution(self):
        """验证双头 likelihood 与 sampler 共用的 mean/std 解析分布。"""
        policy = FakePolicy(min_std=0.2, horizon=2, inference_steps=1)
        scheduler = FakeScheduler(num_train_timesteps=10, num_inference_steps=1)
        scheduler.set_timesteps(1)
        unet = ZeroUNet()
        x_t = torch.zeros(1, 2, 4)
        timesteps = torch.tensor([0])
        mean, std = _head_ddim_mean_std(
            policy,
            x_t,
            torch.zeros_like(x_t),
            timesteps,
            scheduler,
            "min_logprob_denoising_std",
        )
        x_t_1 = mean + 0.5 * std
        reduced_logprob = _head_logprob(
            policy,
            torch.zeros(1, 3),
            x_t,
            x_t_1,
            timesteps,
            unet,
            scheduler,
        )
        expected = -0.5 * 0.5**2 - math.log(0.2) - 0.5 * math.log(2 * math.pi)
        torch.testing.assert_close(
            reduced_logprob,
            torch.full_like(reduced_logprob, expected),
        )

    def test_dual_head_likelihood_always_covers_full_joint_action(self):
        """验证 train_head 不会让严格联合 likelihood 静默漏掉冻结头。"""
        policy = FakePolicy(min_std=0.2, horizon=2, inference_steps=1)
        arm_scheduler = FakeScheduler(num_train_timesteps=10, num_inference_steps=1)
        view_scheduler = FakeScheduler(num_train_timesteps=10, num_inference_steps=1)
        arm_scheduler.set_timesteps(1)
        view_scheduler.set_timesteps(1)
        policy.config.dppo_trainable_heads = ("arm",)
        policy.diffusion = SimpleNamespace(
            num_inference_steps=1,
            arm_action_dim=14,
            view_action_dim=6,
            arm_unet=ZeroUNet(),
            view_unet=ZeroUNet(),
            arm_noise_scheduler=arm_scheduler,
            view_noise_scheduler=view_scheduler,
        )
        x_t = torch.zeros(1, 2, 20)
        x_t_1 = torch.full_like(x_t, 0.05)
        logprobs = get_logprobs_from_global_cond(
            policy,
            torch.zeros(1, 3),
            x_t,
            x_t_1,
            torch.tensor([0]),
        )

        self.assertEqual(set(logprobs), {"arm", "view", "joint"})
        expected_joint = 0.7 * logprobs["arm"] + 0.3 * logprobs["view"]
        torch.testing.assert_close(logprobs["joint"], expected_joint)

    def test_dual_head_joint_mode_rejects_sum_reduction(self):
        """验证联合复现模式拒绝会改变 ratio 尺度的 sum reduction。"""
        policy = FakePolicy(min_std=0.2, horizon=2, inference_steps=1)
        policy.config.logprob_reduction = "sum"
        scheduler = FakeScheduler(num_train_timesteps=10, num_inference_steps=1)
        scheduler.set_timesteps(1)
        with self.assertRaisesRegex(ValueError, "仅支持 logprob_reduction='mean'"):
            _head_logprob(
                policy,
                torch.zeros(1, 3),
                torch.zeros(1, 2, 4),
                torch.zeros(1, 2, 4),
                torch.tensor([0]),
                ZeroUNet(),
                scheduler,
            )


class JointHeadLogprobTest(unittest.TestCase):
    def test_joint_mean_equals_concatenated_action_mean(self):
        """验证 14:6 加权结果等价于拼接 20 维动作后统一求 mean。"""
        torch.manual_seed(7)
        arm_elements = torch.randn(5, 8, 14)
        view_elements = torch.randn(5, 8, 6)
        head_logprobs = {
            "arm": arm_elements.mean(dim=(-1, -2)),
            "view": view_elements.mean(dim=(-1, -2)),
        }
        joint = combine_head_logprobs(
            head_logprobs,
            {"arm": 14, "view": 6},
        )
        expected = torch.cat([arm_elements, view_elements], dim=-1).mean(
            dim=(-1, -2)
        )
        torch.testing.assert_close(joint, expected)

    def test_joint_gradient_uses_action_dimension_weights(self):
        """验证联合 logprob 对 arm/view 的梯度权重分别为 0.7 和 0.3。"""
        arm = torch.tensor([0.4], requires_grad=True)
        view = torch.tensor([-0.2], requires_grad=True)
        joint = combine_head_logprobs(
            {"arm": arm, "view": view},
            {"arm": 14, "view": 6},
        )
        joint.sum().backward()

        torch.testing.assert_close(arm.grad, torch.tensor([0.7]))
        torch.testing.assert_close(view.grad, torch.tensor([0.3]))

    def test_opposite_head_changes_cancel_in_joint_ratio(self):
        """验证两头反向变化先合并再计算唯一 ratio。"""
        old = {"arm": torch.tensor([0.0]), "view": torch.tensor([0.0])}
        new = {"arm": torch.tensor([0.3]), "view": torch.tensor([-0.7])}
        dims = {"arm": 14, "view": 6}
        log_ratio = combine_head_logprobs(new, dims) - combine_head_logprobs(old, dims)

        torch.testing.assert_close(log_ratio, torch.zeros_like(log_ratio), atol=1e-7, rtol=0)
        torch.testing.assert_close(torch.exp(log_ratio), torch.ones_like(log_ratio))

    def test_identical_old_and_new_has_identity_ratio_and_zero_kl(self):
        """验证新旧策略相同时联合 ratio=1 且 k3 KL=0。"""
        old_joint = torch.tensor([0.2, -0.1])
        new_joint = old_joint.detach().clone().requires_grad_(True)
        _, log_ratio, ratio, approx_kl = clipped_ppo_loss(
            new_joint,
            old_joint,
            torch.ones_like(new_joint),
            torch.full_like(new_joint, 0.1),
        )

        torch.testing.assert_close(ratio, torch.ones_like(ratio))
        torch.testing.assert_close(approx_kl, torch.zeros_like(approx_kl))

    def test_joint_ppo_applies_one_clip_after_combining_heads(self):
        """验证联合 logprob 合并后只执行一次 PPO clip。"""
        old_heads = {"arm": torch.tensor([0.0]), "view": torch.tensor([0.0])}
        new_heads = {"arm": torch.tensor([0.4]), "view": torch.tensor([0.4])}
        dims = {"arm": 14, "view": 6}
        old_joint = combine_head_logprobs(old_heads, dims)
        new_joint = combine_head_logprobs(new_heads, dims).requires_grad_(True)
        loss, log_ratio, ratio, _ = clipped_ppo_loss(
            new_joint,
            old_joint,
            torch.ones_like(new_joint),
            torch.full_like(new_joint, 0.1),
        )

        torch.testing.assert_close(log_ratio, torch.tensor([0.4]))
        torch.testing.assert_close(ratio, torch.exp(torch.tensor([0.4])))
        torch.testing.assert_close(loss, torch.tensor(-1.1))

    def test_joint_mode_requires_both_likelihood_heads(self):
        """验证严格联合模式不会因 train_head 设置而静默漏掉一个头。"""
        with self.assertRaisesRegex(KeyError, "view"):
            combine_head_logprobs(
                {"arm": torch.tensor([0.0])},
                {"arm": 14, "view": 6},
            )


class PPORatioDiagnosticsTest(unittest.TestCase):
    def test_summary_distinguishes_outside_and_objective_clip_fraction(self):
        """验证 ratio 越界率与真正进入裁剪代理目标的比例采用不同口径。"""
        ratio = torch.tensor([1.20, 0.80, 1.15, 0.85, 1.00])
        clip_coef = torch.full_like(ratio, 0.10)
        advantages = torch.tensor([1.0, -1.0, -1.0, 1.0, 0.0])
        denoising_indices = torch.tensor([0, 0, 1, 1, 1])

        summary = summarize_ppo_ratio(
            ratio,
            clip_coef,
            advantages,
            denoising_indices,
            denoising_steps=2,
        )

        self.assertEqual(summary["count"], 5)
        self.assertAlmostEqual(summary["mean"], 1.0, places=6)
        self.assertAlmostEqual(summary["std"], math.sqrt(0.025), places=6)
        self.assertAlmostEqual(summary["outside_clip_fraction"], 0.8)
        self.assertAlmostEqual(summary["objective_clip_fraction"], 0.4)
        self.assertAlmostEqual(summary["upper_clip_fraction"], 0.4)
        self.assertAlmostEqual(summary["lower_clip_fraction"], 0.4)
        self.assertAlmostEqual(summary["p05"], 0.81, places=6)
        self.assertAlmostEqual(summary["p50"], 1.0, places=6)
        self.assertAlmostEqual(summary["p95"], 1.19, places=6)
        self.assertEqual(summary["per_step_outside_clip_fraction"], [1.0, 2 / 3])
        self.assertEqual(summary["per_step_objective_clip_fraction"], [1.0, 0.0])

    def test_online_accumulation_matches_single_tensor_summary(self):
        """验证跨 minibatch 在线累计与一次性汇总得到相同的 ratio 指标。"""
        ratio = torch.tensor([0.7, 0.95, 1.0, 1.08, 1.3, 1.01])
        clip_coef = torch.tensor([0.1, 0.02, 0.01, 0.05, 0.1, 0.02])
        advantages = torch.tensor([-1.0, 1.0, 0.5, 1.0, -1.0, -0.5])
        denoising_indices = torch.tensor([0, 1, 2, 1, 0, 2])
        stats = init_ppo_ratio_stats(denoising_steps=3)

        for batch_slice in (slice(0, 2), slice(2, None)):
            update_ppo_ratio_stats(
                stats,
                ratio[batch_slice],
                clip_coef[batch_slice],
                advantages[batch_slice],
                denoising_indices[batch_slice],
            )

        online = finalize_ppo_ratio_stats(stats)
        combined = summarize_ppo_ratio(
            ratio,
            clip_coef,
            advantages,
            denoising_indices,
            denoising_steps=3,
        )
        for key in (
            "count",
            "mean",
            "std",
            "min",
            "max",
            "outside_clip_fraction",
            "objective_clip_fraction",
            "upper_clip_fraction",
            "lower_clip_fraction",
        ):
            self.assertAlmostEqual(online[key], combined[key], places=6)
        self.assertEqual(
            online["per_step_outside_clip_fraction"],
            combined["per_step_outside_clip_fraction"],
        )
        self.assertEqual(
            online["per_step_objective_clip_fraction"],
            combined["per_step_objective_clip_fraction"],
        )


if __name__ == "__main__":
    unittest.main()
