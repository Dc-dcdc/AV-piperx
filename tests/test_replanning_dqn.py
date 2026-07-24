"""三决策主动视觉重规划 DQN 的独立单元测试。"""

import unittest

import torch

from train.replanning_dqn import (
    CHUNK_PROGRESS,
    REMAINING_ACTION_MASK,
    REMAINING_ARM_ACTIONS,
    REMAINING_VIEW_ACTIONS,
    ROBOT_STATE,
    VISUAL_FEATURES,
    DoubleDQNTrainer,
    ReplanningActionCache,
    ReplanningDataCollector,
    ReplanningDecision,
    ReplanningDQNConfig,
    ReplanningDuelingQNetwork,
    ReplanningReplayBuffer,
    ReplanningRewardConfig,
    ReplanningTransitionBatch,
    build_replanning_action_mask,
    build_replanning_state,
    compute_replanning_reward,
    pad_remaining_action_chunk,
)


def make_state(config, batch_size=4):
    """生成满足网络接口的随机批量状态。"""
    return {
        VISUAL_FEATURES: torch.randn(batch_size, config.visual_feature_dim),
        ROBOT_STATE: torch.randn(batch_size, config.robot_state_dim),
        REMAINING_ARM_ACTIONS: torch.randn(
            batch_size,
            config.horizon,
            config.arm_action_dim,
        ),
        REMAINING_VIEW_ACTIONS: torch.randn(
            batch_size,
            config.horizon,
            config.view_action_dim,
        ),
        REMAINING_ACTION_MASK: torch.ones(
            batch_size,
            config.horizon,
            dtype=torch.bool,
        ),
        CHUNK_PROGRESS: torch.rand(batch_size, 1),
    }


class ReplanningDQNTest(unittest.TestCase):
    """验证动作补齐、动作约束和 Double DQN 更新的核心行为。"""

    def setUp(self):
        """创建尺寸较小但保持 Arm/View 维度一致的测试网络。"""
        torch.manual_seed(7)
        self.config = ReplanningDQNConfig(
            visual_feature_dim=12,
            robot_state_dim=20,
            visual_embed_dim=16,
            state_embed_dim=16,
            chunk_embed_dim=16,
            hidden_dim=32,
        )

    def test_pad_remaining_action_chunk(self):
        """验证可变长度动作被正确补齐，并为真实剩余位置生成掩码。"""
        arm = torch.randn(5, self.config.arm_action_dim)
        view = torch.randn(5, self.config.view_action_dim)
        padded_arm, padded_view, mask = pad_remaining_action_chunk(
            arm,
            view,
            horizon=self.config.horizon,
        )

        self.assertEqual(
            tuple(padded_arm.shape),
            (self.config.horizon, self.config.arm_action_dim),
        )
        self.assertEqual(
            tuple(padded_view.shape),
            (self.config.horizon, self.config.view_action_dim),
        )
        self.assertEqual(mask.sum().item(), 5)
        torch.testing.assert_close(padded_arm[:5], arm)
        torch.testing.assert_close(padded_view[:5], view)
        self.assertTrue((padded_arm[5:] == 0).all())

    def test_masked_padding_does_not_change_q_values(self):
        """验证无效补齐位置的任意数值不会影响三个决策的 Q 值。"""
        network = ReplanningDuelingQNetwork(self.config).eval()
        state = make_state(self.config, batch_size=2)
        state[REMAINING_ACTION_MASK][:, 6:] = False
        changed_state = {key: value.clone() for key, value in state.items()}
        changed_state[REMAINING_ARM_ACTIONS][:, 6:] = 1000.0
        changed_state[REMAINING_VIEW_ACTIONS][:, 6:] = -1000.0

        with torch.no_grad():
            original_q = network(state)
            changed_q = network(changed_state)

        torch.testing.assert_close(original_q, changed_q)

    def test_action_mask_forces_safe_joint_replan(self):
        """验证无缓存和安全接管状态只允许联合重规划。"""
        network = ReplanningDuelingQNetwork(self.config).eval()
        state = make_state(self.config, batch_size=2)
        mask = build_replanning_action_mask(
            has_cached_plan=torch.tensor([False, True]),
            view_only_available=True,
            force_joint_replan=torch.tensor([False, True]),
        )
        actions, q_values = network.select_action(state, mask, epsilon=1.0)

        self.assertEqual(tuple(q_values.shape), (2, len(ReplanningDecision)))
        self.assertTrue(
            (actions == int(ReplanningDecision.JOINT_REPLAN)).all()
        )

    def test_replay_and_double_dqn_train_step(self):
        """验证回放缓冲区采样后可以完成一次有限的 Double DQN 更新。"""
        network = ReplanningDuelingQNetwork(self.config)
        trainer = DoubleDQNTrainer(network, self.config)
        state = make_state(self.config, batch_size=6)
        next_state = make_state(self.config, batch_size=6)
        transition = ReplanningTransitionBatch(
            state=state,
            action=torch.tensor([0, 1, 2, 0, 1, 2]),
            reward=torch.linspace(-1.0, 1.0, 6),
            next_state=next_state,
            done=torch.tensor([False, False, False, False, True, True]),
            next_action_mask=build_replanning_action_mask(
                has_cached_plan=torch.ones(6, dtype=torch.bool),
                view_only_available=True,
            ),
        )
        replay = ReplanningReplayBuffer(capacity=16, config=self.config)
        replay.add_batch(transition)

        metrics = trainer.train_step(replay.sample(4))

        self.assertEqual(len(replay), 6)
        for value in metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertGreaterEqual(metrics["loss"], 0.0)


class ReplanningDataCollectionTest(unittest.TestCase):
    """验证显式动作缓存、状态构造、奖励和transition写入。"""

    def setUp(self):
        """创建适配16步、Arm 14维和View 6维的数据收集配置。"""
        torch.manual_seed(11)
        self.config = ReplanningDQNConfig(
            visual_feature_dim=12,
            robot_state_dim=20,
            visual_embed_dim=16,
            state_embed_dim=16,
            chunk_embed_dim=16,
            hidden_dim=32,
        )
        self.cache = ReplanningActionCache(self.config)

    def test_action_cache_and_state_follow_remaining_chunk(self):
        """验证缓存推进后DQN状态只保留真实的剩余Arm/View动作。"""
        normalized = torch.randn(self.config.horizon, 20)
        env_actions = torch.randn(self.config.horizon, 20)
        self.cache.replace_joint(normalized, env_actions)
        for _ in range(5):
            self.cache.advance()

        state = build_replanning_state(
            torch.randn(self.config.visual_feature_dim),
            torch.randn(self.config.robot_state_dim),
            self.cache,
        )

        self.assertEqual(self.cache.remaining_steps, 11)
        self.assertEqual(state[REMAINING_ACTION_MASK].sum().item(), 11)
        self.assertAlmostEqual(state[CHUNK_PROGRESS].item(), 5 / 16)
        torch.testing.assert_close(
            state[REMAINING_ARM_ACTIONS][0, :11],
            normalized[5:, :14],
        )
        torch.testing.assert_close(
            state[REMAINING_VIEW_ACTIONS][0, :11],
            normalized[5:, 14:],
        )

    def test_view_only_replan_preserves_remaining_arm(self):
        """验证只替换View缓存不会改变尚未执行的Arm动作。"""
        normalized = torch.randn(self.config.horizon, 20)
        env_actions = torch.randn(self.config.horizon, 20)
        self.cache.replace_joint(normalized, env_actions)
        self.cache.advance()
        old_arm = self.cache.normalized_actions[1:, :14].clone()
        new_view = torch.randn(15, 6)
        new_env_view = torch.randn(15, 6)

        self.cache.replace_remaining_view(new_view, new_env_view)

        torch.testing.assert_close(
            self.cache.normalized_actions[1:, :14],
            old_arm,
        )
        torch.testing.assert_close(
            self.cache.normalized_actions[1:, 14:],
            new_view,
        )

    def test_reward_applies_replan_and_discontinuity_costs(self):
        """验证联合重规划奖励包含环境缩放、推理成本和Arm跳变惩罚。"""
        reward_config = ReplanningRewardConfig(
            env_reward_scale=0.01,
            joint_replan_cost=0.02,
            arm_discontinuity_coef=0.1,
        )
        reward, components = compute_replanning_reward(
            env_reward=10.0,
            decision=ReplanningDecision.JOINT_REPLAN,
            config=reward_config,
            previous_arm_action=torch.zeros(14),
            new_arm_action=torch.ones(14),
        )

        self.assertAlmostEqual(reward, -0.02)
        self.assertAlmostEqual(components["scaled_env_reward"], 0.1)
        self.assertAlmostEqual(components["arm_discontinuity"], 1.0)

    def test_collector_adds_one_transition_and_tracks_decision(self):
        """验证收集器把一个环境步写入回放并更新决策比例。"""
        normalized = torch.randn(self.config.horizon, 20)
        env_actions = torch.randn(self.config.horizon, 20)
        self.cache.replace_joint(normalized, env_actions)
        state = build_replanning_state(
            torch.randn(self.config.visual_feature_dim),
            torch.randn(self.config.robot_state_dim),
            self.cache,
        )
        self.cache.advance()
        next_state = build_replanning_state(
            torch.randn(self.config.visual_feature_dim),
            torch.randn(self.config.robot_state_dim),
            self.cache,
        )
        replay = ReplanningReplayBuffer(capacity=8, config=self.config)
        collector = ReplanningDataCollector(
            replay,
            ReplanningRewardConfig(),
        )
        next_mask = collector.build_action_mask(
            self.cache,
            view_only_available=False,
        )

        collector.add_step(
            state=state,
            decision=ReplanningDecision.CONTINUE,
            env_reward=1.0,
            next_state=next_state,
            done=False,
            next_action_mask=next_mask,
        )

        self.assertEqual(len(replay), 1)
        self.assertEqual(collector.total_transitions, 1)
        self.assertAlmostEqual(
            collector.decision_fractions()["continue_fraction"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
