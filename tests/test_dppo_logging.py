import unittest
from types import SimpleNamespace

from omegaconf import OmegaConf

from train.finetune.dppo_logging import (
    add_wandb_parameter_tags,
    build_dppo_eval_metrics,
    build_dppo_train_metrics,
)


def make_ratio_summary(*, denoising_steps: int, include_quantiles: bool) -> dict:
    """构造日志模块测试使用的最小 ratio 汇总结果。"""
    summary = {
        "count": 12,
        "mean": 1.01,
        "std": 0.03,
        "min": 0.92,
        "max": 1.08,
        "outside_clip_fraction": 0.25,
        "objective_clip_fraction": 0.125,
        "upper_clip_fraction": 0.15,
        "lower_clip_fraction": 0.10,
        "per_step_outside_clip_fraction": [
            0.1 * (step + 1) for step in range(denoising_steps)
        ],
        "per_step_objective_clip_fraction": [
            0.05 * (step + 1) for step in range(denoising_steps)
        ],
    }
    if include_quantiles:
        summary.update({"p05": 0.95, "p50": 1.0, "p95": 1.06})
    return summary


def make_train_metrics(**overrides) -> dict:
    """用稳定默认值调用训练指标构造器，并允许测试覆盖局部参数。"""
    arguments = {
        "completed_episode_count": 4,
        "rollout_success_rate": 0.5,
        "rollout_average_return": 12.0,
        "rollout_action_chunks": 64,
        "rollout_env_steps": 512,
        "critic_loss": 0.4,
        "actor_loss": -0.2,
        "bc_loss": 0.03,
        "avg_kl": 0.004,
        "max_kl": 0.009,
        "ratio_summary": make_ratio_summary(
            denoising_steps=2,
            include_quantiles=False,
        ),
        "critic_explained_variance": 0.6,
        "critic_value_return_correlation": 0.7,
        "logprob_advantage_correlation": 0.2,
        "positive_advantage_mean_logprob_delta": 0.01,
        "negative_advantage_mean_logprob_delta": -0.02,
        "logprob_advantage_sign_agreement": 0.65,
        "post_probe_size": 128,
        "post_probe_logprob_delta_mean": 0.002,
        "post_probe_logprob_advantage_correlation": 0.3,
        "post_probe_positive_advantage_mean_logprob_delta": 0.02,
        "post_probe_negative_advantage_mean_logprob_delta": -0.01,
        "post_probe_logprob_advantage_sign_agreement": 0.7,
        "post_probe_ratio_summary": make_ratio_summary(
            denoising_steps=2,
            include_quantiles=True,
        ),
        "actor_update_enabled": True,
        "early_stop": False,
        "denoising_steps": 2,
    }
    arguments.update(overrides)
    return build_dppo_train_metrics(**arguments)


class WandbParameterTagsTest(unittest.TestCase):
    def test_parameter_tags_merge_config_and_effective_values(self):
        """验证手动标签、参数标签去重后按原顺序写入 run。"""
        run = SimpleNamespace(tags=("existing", "manual"))
        logger = SimpleNamespace(_wandb=SimpleNamespace(run=run))
        cfg = OmegaConf.create(
            {
                "device": "cuda",
                "env": {"n_envs": 8},
                "training": {
                    "actor_lr": 1e-5,
                    "train_vision_encoder": False,
                },
                "policy": {"ft_denoising_steps": 10},
                "wandb": {"tags": ["manual", "experiment_a"]},
            }
        )

        add_wandb_parameter_tags(logger, cfg)

        self.assertEqual(run.tags[:3], ("existing", "manual", "experiment_a"))
        self.assertIn("device:cuda", run.tags)
        self.assertIn("n_envs:8", run.tags)
        self.assertIn("actor_lr:1e-05", run.tags)
        self.assertIn("denoise_steps:10", run.tags)
        self.assertIn("train_vision:false", run.tags)
        self.assertEqual(run.tags.count("manual"), 1)

    def test_missing_wandb_run_is_a_noop(self):
        """验证关闭 W&B 或没有活动 run 时标签函数安全返回。"""
        cfg = OmegaConf.create({"wandb": {"tags": ["unused"]}})
        add_wandb_parameter_tags(SimpleNamespace(_wandb=None), cfg)


class DPPOTrainMetricsTest(unittest.TestCase):
    def test_train_metrics_preserve_scalar_and_per_step_names(self):
        """验证训练字段名称、数值类型及逐去噪步指标保持稳定。"""
        metrics = make_train_metrics()

        self.assertEqual(metrics["rollout_completed_episodes"], 4)
        self.assertEqual(metrics["rollout_action_chunks"], 64)
        self.assertEqual(metrics["ppo_ratio_sample_count"], 12)
        self.assertAlmostEqual(metrics["ppo_ratio_mean"], 1.01)
        self.assertAlmostEqual(metrics["post_update_probe_ratio_p95"], 1.06)
        self.assertEqual(metrics["actor_update_enabled"], 1)
        self.assertEqual(metrics["early_stop"], 0)
        self.assertAlmostEqual(
            metrics["ppo_ratio_outside_clip_fraction_denoising_step_1"],
            0.2,
        )
        self.assertAlmostEqual(
            metrics[
                "post_update_probe_objective_clip_fraction_denoising_step_1"
            ],
            0.1,
        )

    def test_train_metrics_reject_mismatched_denoising_steps(self):
        """验证逐步统计长度错误时在上传前给出明确异常。"""
        ratio_summary = make_ratio_summary(
            denoising_steps=1,
            include_quantiles=False,
        )
        with self.assertRaisesRegex(ValueError, "denoising_steps"):
            make_train_metrics(ratio_summary=ratio_summary)


class DPPOEvalMetricsTest(unittest.TestCase):
    def test_eval_metrics_add_rollback_baseline_only_when_available(self):
        """验证无回滚基线时省略字段，有完整基线时统一追加。"""
        arguments = {
            "success_rate": 0.6,
            "average_reward": 20.0,
            "best_success_rate": 0.7,
            "best_average_reward": 24.0,
            "new_best_actor": False,
            "eval_collapsed": False,
            "candidate_eligible": True,
            "rollback_triggered": False,
        }
        without_rollback = build_dppo_eval_metrics(**arguments)
        self.assertNotIn("rollback_best_success_rate", without_rollback)
        self.assertEqual(without_rollback["candidate_eligible"], 1)

        with_rollback = build_dppo_eval_metrics(
            **arguments,
            rollback_best_success_rate=0.7,
            rollback_best_average_reward=24.0,
            rollback_best_policy_loss=-0.3,
        )
        self.assertAlmostEqual(with_rollback["rollback_best_success_rate"], 0.7)
        self.assertAlmostEqual(with_rollback["rollback_best_policy_loss"], -0.3)

    def test_eval_metrics_reject_partial_rollback_baseline(self):
        """验证回滚基线不完整时拒绝生成含义不一致的日志。"""
        with self.assertRaisesRegex(ValueError, "必须同时提供"):
            build_dppo_eval_metrics(
                success_rate=0.6,
                average_reward=20.0,
                best_success_rate=0.7,
                best_average_reward=24.0,
                new_best_actor=False,
                eval_collapsed=False,
                candidate_eligible=True,
                rollback_triggered=False,
                rollback_best_success_rate=0.7,
            )


if __name__ == "__main__":
    unittest.main()
