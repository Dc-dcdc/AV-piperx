"""重规划DQN训练入口中纯函数和完整动作采样接口的测试。"""

import random
import unittest
from types import SimpleNamespace

import torch

from train.s4_adaptive_replanning.dqn import ReplanningDecision
from train.s4_adaptive_replanning.train_replanning_dqn import (
    NormalizedObservationHistory,
    extract_visual_features,
    infer_full_joint_chunk,
    linear_epsilon,
    sample_warmup_decision,
)


class FakeRGBEncoder(torch.nn.Module):
    """把每张图像压缩成均值和最大值两个冻结特征。"""

    def forward(self, images):
        """返回形状为[B,2]的确定性图像特征。"""
        flattened = images.flatten(start_dim=1)
        return torch.stack(
            [flattened.mean(dim=1), flattened.max(dim=1).values],
            dim=1,
        )


class FakeCoupledDiffusion:
    """提供完整动作采样辅助函数所需的最小双头接口。"""

    def __init__(self):
        self.rgb_encoder = FakeRGBEncoder()

    def _prepare_global_conditioning(self, batch):
        """用状态历史展平结果模拟扩散模型的全局条件。"""
        return batch["observation.state"].flatten(start_dim=1)

    def conditional_sample_coupled(self, batch_size, global_cond, generator):
        """返回4步、Arm 2维和View 1维的确定性归一化动作。"""
        del global_cond, generator
        arm = torch.full((batch_size, 4, 2), 0.25)
        view = torch.full((batch_size, 4, 1), -0.5)
        return arm, view

    def combine_action_heads(self, arm, view):
        """耦合基线的两个头直接还原为完整环境动作。"""
        return torch.cat([arm, view], dim=-1)


class FakePolicy:
    """模拟训练入口所需的冻结双头策略。"""

    def __init__(self):
        self.diffusion = FakeCoupledDiffusion()
        self.config = SimpleNamespace(horizon=4)

    def unnormalize_outputs(self, batch):
        """用简单平移验证环境动作与归一化动作被分别保留。"""
        return {"action": batch["action"] + 1.0}


class ReplanningDQNTrainHelperTest(unittest.TestCase):
    """验证观测历史、探索率、预热决策和完整动作采样。"""

    def test_observation_history_fills_first_frame_and_appends_once(self):
        """验证首帧复制填充，追加后最旧帧被正确移除。"""
        history = NormalizedObservationHistory(n_obs_steps=2)
        history.reset({"observation.state": torch.tensor([[1.0, 2.0]])})
        first = history.stacked()["observation.state"]
        history.append({"observation.state": torch.tensor([[3.0, 4.0]])})
        updated = history.stacked()["observation.state"]

        torch.testing.assert_close(
            first,
            torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]),
        )
        torch.testing.assert_close(
            updated,
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        )

    def test_visual_features_flatten_history_and_camera_dimensions(self):
        """验证两帧两相机特征最终合并为一条DQN视觉向量。"""
        policy = FakePolicy()
        images = torch.arange(1 * 2 * 2 * 3 * 2 * 2, dtype=torch.float32)
        images = images.reshape(1, 2, 2, 3, 2, 2)

        features = extract_visual_features(
            policy,
            {"observation.images": images},
        )

        self.assertEqual(tuple(features.shape), (1, 8))

    def test_linear_epsilon_reaches_configured_end(self):
        """验证epsilon在衰减区间内线性变化并在终点后保持不变。"""
        self.assertAlmostEqual(linear_epsilon(0, 0.2, 0.02, 100), 0.2)
        self.assertAlmostEqual(linear_epsilon(50, 0.2, 0.02, 100), 0.11)
        self.assertAlmostEqual(linear_epsilon(200, 0.2, 0.02, 100), 0.02)

    def test_warmup_decision_respects_forced_joint_mask(self):
        """验证没有动作缓存时预热策略一定选择联合重规划。"""
        mask = torch.tensor([[False, False, True]])
        decision = sample_warmup_decision(
            mask,
            continue_probability=1.0,
            rng=random.Random(0),
        )
        self.assertEqual(decision, int(ReplanningDecision.JOINT_REPLAN))

    def test_joint_sampler_returns_full_normalized_and_env_chunks(self):
        """验证训练入口绕过动作队列并保留完整双头动作块。"""
        policy = FakePolicy()
        history_batch = {
            "observation.state": torch.zeros(1, 2, 3),
        }
        normalized, env_actions, inference_ms = infer_full_joint_chunk(
            policy,
            history_batch,
            torch.Generator().manual_seed(0),
        )

        self.assertEqual(tuple(normalized.shape), (4, 3))
        torch.testing.assert_close(env_actions, normalized + 1.0)
        self.assertGreaterEqual(inference_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
