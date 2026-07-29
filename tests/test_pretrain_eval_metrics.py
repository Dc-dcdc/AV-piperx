import unittest

from train.s1_pretrain.eval.eval_train import build_eval_log_metrics


class PretrainEvalMetricsTest(unittest.TestCase):
    def test_build_eval_log_metrics_includes_reward_distribution(self):
        """验证成功率、奖励分布、回合长度和推理耗时都会写入评估指标。"""
        eval_info = {
            "aggregated": {
                "success_rate": 0.5,
                "average_reward": 7.0,
                "avg_inference_ms": 12.5,
                "max_inference_ms": 20.0,
            },
            "episodes": [
                {"success": True, "reward": 3.0, "steps": 10},
                {"success": False, "reward": 11.0, "steps": 20},
            ],
        }

        metrics = build_eval_log_metrics(eval_info)

        self.assertEqual(metrics["success_rate"], 0.5)
        self.assertEqual(metrics["success_rate_percent"], 50.0)
        self.assertEqual(metrics["average_reward"], 7.0)
        self.assertEqual(metrics["reward_std"], 4.0)
        self.assertEqual(metrics["minimum_reward"], 3.0)
        self.assertEqual(metrics["maximum_reward"], 11.0)
        self.assertEqual(metrics["average_episode_steps"], 15.0)
        self.assertEqual(metrics["successful_episodes"], 1)
        self.assertEqual(metrics["num_episodes"], 2)
        self.assertEqual(metrics["avg_inference_ms"], 12.5)
        self.assertEqual(metrics["max_inference_ms"], 20.0)

    def test_build_eval_log_metrics_handles_missing_episode_records(self):
        """验证没有逐回合记录时仍能上传稳定且有限的聚合指标。"""
        eval_info = {
            "aggregated": {
                "success_rate": 0.25,
                "average_reward": -2.0,
            }
        }

        metrics = build_eval_log_metrics(eval_info)

        self.assertEqual(metrics["num_episodes"], 0)
        self.assertEqual(metrics["reward_std"], 0.0)
        self.assertEqual(metrics["minimum_reward"], -2.0)
        self.assertEqual(metrics["maximum_reward"], -2.0)


if __name__ == "__main__":
    unittest.main()
