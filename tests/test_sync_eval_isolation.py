import copy
import random
import tempfile
import unittest
from collections import deque
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from train.s1_pretrain.eval.eval_train import (
    evaluate_and_checkpoint_if_needed,
    isolate_synchronous_evaluation,
)


class QueuePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self._queues = None
        self.reset()

    def reset(self):
        self._queues = {
            "observation.state": deque(maxlen=2),
            "action": deque(maxlen=4),
        }


def assert_numpy_random_states_equal(test_case, actual, expected):
    test_case.assertEqual(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    test_case.assertEqual(actual[2:], expected[2:])


class SynchronousEvaluationIsolationTest(unittest.TestCase):
    def setUp(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)

    def test_restores_rng_mode_and_action_queues(self):
        policy = QueuePolicy()
        policy.train()
        policy._queues["observation.state"].append(torch.tensor([3.0]))
        policy._queues["action"].append(torch.tensor([4.0]))

        expected_queues = copy.deepcopy(policy._queues)
        expected_python_state = random.getstate()
        expected_numpy_state = np.random.get_state()
        expected_torch_state = torch.random.get_rng_state().clone()

        with isolate_synchronous_evaluation(policy, torch.device("cpu")):
            self.assertFalse(policy.training)
            random.random()
            np.random.rand()
            torch.rand(8)
            policy._queues["action"].append(torch.tensor([99.0]))
            # 模拟评估内部意外改变模型模式，隔离层仍应恢复真实模式。
            policy.train()

        self.assertTrue(policy.training)
        self.assertEqual(random.getstate(), expected_python_state)
        assert_numpy_random_states_equal(
            self,
            np.random.get_state(),
            expected_numpy_state,
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(),
            expected_torch_state,
        )
        self.assertEqual(
            list(policy._queues["observation.state"]),
            list(expected_queues["observation.state"]),
        )
        self.assertEqual(
            list(policy._queues["action"]),
            list(expected_queues["action"]),
        )

    def test_restores_state_when_evaluation_raises(self):
        policy = QueuePolicy()
        policy.eval()
        policy._queues["action"].append(torch.tensor([5.0]))

        expected_python_state = random.getstate()
        expected_numpy_state = np.random.get_state()
        expected_torch_state = torch.random.get_rng_state().clone()

        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            with isolate_synchronous_evaluation(policy, torch.device("cpu")):
                random.random()
                np.random.rand()
                torch.rand(4)
                policy.train()
                raise RuntimeError("evaluation failed")

        self.assertFalse(policy.training)
        self.assertEqual(random.getstate(), expected_python_state)
        assert_numpy_random_states_equal(
            self,
            np.random.get_state(),
            expected_numpy_state,
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(),
            expected_torch_state,
        )
        self.assertEqual(
            list(policy._queues["action"]),
            [torch.tensor([5.0])],
        )

    def test_training_evaluation_entrypoint_is_isolated(self):
        policy = QueuePolicy()
        policy.train()
        policy._queues["action"].append(torch.tensor([7.0]))
        expected_queues = copy.deepcopy(policy._queues)
        expected_python_state = random.getstate()
        expected_numpy_state = np.random.get_state()
        expected_torch_state = torch.random.get_rng_state().clone()

        cfg = OmegaConf.create(
            {
                "use_amp": False,
                "training": {
                    "offline_steps": 3,
                    "eval_freq": 1,
                    "save_checkpoint": False,
                    "save_freq": 10,
                },
                "eval": {},
                "wandb": {"enable": False},
            }
        )

        class RandomConsumingLogger:
            def log_dict(self, metrics, step, mode):
                random.random()
                np.random.rand()
                torch.rand(2)

        def fake_eval(**kwargs):
            evaluated_policy = kwargs["policy"]
            random.random()
            np.random.rand()
            torch.rand(3)
            evaluated_policy._queues["action"].append(torch.tensor([99.0]))
            evaluated_policy.train()
            return {
                "aggregated": {
                    "success_rate": 0.5,
                    "average_reward": 1.0,
                    "avg_inference_ms": 2.0,
                    "max_inference_ms": 3.0,
                },
                "video_paths": [],
                "episodes": [],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch(
                "train.s1_pretrain.eval.eval_train.custom_eval_policy",
                side_effect=fake_eval,
            ):
                evaluate_and_checkpoint_if_needed(
                    step=1,
                    policy=policy,
                    optimizer=None,
                    lr_scheduler=None,
                    logger=RandomConsumingLogger(),
                    cfg=cfg,
                    device=torch.device("cpu"),
                    out_dir=temp_dir,
                    eval_env=object(),
                    train_loss=0.1,
                )

        self.assertTrue(policy.training)
        self.assertEqual(random.getstate(), expected_python_state)
        assert_numpy_random_states_equal(
            self,
            np.random.get_state(),
            expected_numpy_state,
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(),
            expected_torch_state,
        )
        self.assertEqual(
            list(policy._queues["action"]),
            list(expected_queues["action"]),
        )


if __name__ == "__main__":
    unittest.main()
