import unittest

from train.s3_finetune.eval_selection import (
    canonical_checkpoint_metric,
    is_better_eval_candidate,
    is_eval_candidate_eligible,
    update_historical_eval_best,
)


class HistoricalEvalBestTest(unittest.TestCase):
    def test_success_and_reward_history_are_independently_monotonic(self):
        """验证奖励创新高但成功率降低时，历史最高成功率不会下降。"""
        best_success = 0.0
        best_reward = float("-inf")
        observed_best = []
        for success_rate, average_reward in (
            (0.8, 500.0),
            (0.6, 600.0),
            (0.7, 400.0),
        ):
            best_success, best_reward = update_historical_eval_best(
                best_success_rate=best_success,
                best_average_reward=best_reward,
                success_rate=success_rate,
                average_reward=average_reward,
            )
            observed_best.append((best_success, best_reward))

        self.assertEqual(
            observed_best,
            [(0.8, 500.0), (0.8, 600.0), (0.8, 600.0)],
        )


class EvalCandidateSelectionTest(unittest.TestCase):
    def test_collapse_candidate_is_excluded_from_all_best_selection(self):
        """验证启用回滚时，塌陷候选不能参与内存最佳或磁盘 Top-K。"""
        self.assertFalse(
            is_eval_candidate_eligible(
                rollback_enabled=True,
                eval_collapsed=True,
            )
        )
        self.assertTrue(
            is_eval_candidate_eligible(
                rollback_enabled=True,
                eval_collapsed=False,
            )
        )
        self.assertTrue(
            is_eval_candidate_eligible(
                rollback_enabled=False,
                eval_collapsed=True,
            )
        )

    def test_success_metric_uses_reward_only_as_tiebreaker(self):
        """验证 success 模式优先比较成功率，再用奖励打破并列。"""
        self.assertFalse(
            is_better_eval_candidate(
                metric="success",
                candidate_loss=0.0,
                candidate_reward=700.0,
                candidate_success_rate=0.7,
                best_loss=1.0,
                best_reward=500.0,
                best_success_rate=0.8,
            )
        )
        self.assertTrue(
            is_better_eval_candidate(
                metric="success_rate",
                candidate_loss=0.0,
                candidate_reward=600.0,
                candidate_success_rate=0.8,
                best_loss=1.0,
                best_reward=500.0,
                best_success_rate=0.8,
            )
        )
        self.assertFalse(
            is_better_eval_candidate(
                metric="success",
                candidate_loss=0.0,
                candidate_reward=500.0,
                candidate_success_rate=0.8,
                best_loss=1.0,
                best_reward=500.0,
                best_success_rate=0.8,
            )
        )

    def test_reward_and_loss_metrics_follow_topk_direction(self):
        """验证 reward 越大越好、loss 越小越好的 TopK 方向。"""
        common = {
            "candidate_success_rate": 0.1,
            "best_success_rate": 0.9,
        }
        self.assertTrue(
            is_better_eval_candidate(
                metric="reward",
                candidate_loss=2.0,
                candidate_reward=6.0,
                best_loss=1.0,
                best_reward=5.0,
                **common,
            )
        )
        self.assertTrue(
            is_better_eval_candidate(
                metric="loss",
                candidate_loss=0.5,
                candidate_reward=1.0,
                best_loss=1.0,
                best_reward=5.0,
                **common,
            )
        )

    def test_metric_aliases_match_topk_manager(self):
        """验证成功率别名以及未知指标回退到 loss。"""
        self.assertEqual(canonical_checkpoint_metric("sr"), "success")
        self.assertEqual(canonical_checkpoint_metric("success_rate"), "success")
        self.assertEqual(canonical_checkpoint_metric("reward"), "reward")
        self.assertEqual(canonical_checkpoint_metric("unknown"), "loss")


if __name__ == "__main__":
    unittest.main()
