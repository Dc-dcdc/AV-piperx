import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from lerobot.common.logger import Logger
from train.s1_pretrain.train.ema import PolicyEMA


class TinyPolicy(torch.nn.Module):
    def __init__(
        self,
        weight: float = 0.0,
        *,
        use_ema: bool = True,
        decay: float = 0.5,
        update_after_step: int = 2,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([weight]))
        self.frozen = torch.nn.Parameter(
            torch.tensor([weight + 10.0]),
            requires_grad=False,
        )
        self.register_buffer("running", torch.tensor([weight + 20.0]))
        self.config = SimpleNamespace(
            use_ema=use_ema,
            ema_decay=decay,
            ema_update_after_step=update_after_step,
        )

    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")


class PolicyEMATest(unittest.TestCase):
    def test_delayed_initialization_then_exponential_average(self):
        online = TinyPolicy(weight=1.0, decay=0.5, update_after_step=2)
        ema = PolicyEMA.from_policy_config(online)
        self.assertIsNotNone(ema)

        with torch.no_grad():
            online.weight.fill_(3.0)
            online.frozen.fill_(13.0)
            online.running.fill_(23.0)

        self.assertFalse(ema.update(online, step=1))
        self.assertFalse(ema.ready)
        self.assertIs(ema.evaluation_policy(online), online)

        # 第一次开放EMA时必须精确复制online，不能立刻使用衰减平均。
        self.assertTrue(ema.update(online, step=2))
        torch.testing.assert_close(ema.ema_policy.weight, torch.tensor([3.0]))
        self.assertEqual(ema.num_updates, 1)

        with torch.no_grad():
            online.weight.fill_(5.0)
            online.frozen.fill_(17.0)
            online.running.fill_(29.0)
        self.assertTrue(ema.update(online, step=3))

        # 只有可训练浮点参数做EMA；冻结参数和buffer始终精确同步。
        torch.testing.assert_close(ema.ema_policy.weight, torch.tensor([4.0]))
        torch.testing.assert_close(ema.ema_policy.frozen, torch.tensor([17.0]))
        torch.testing.assert_close(ema.ema_policy.running, torch.tensor([29.0]))
        self.assertFalse(any(p.requires_grad for p in ema.ema_policy.parameters()))
        self.assertEqual(ema.num_updates, 2)
        self.assertEqual(ema.last_step, 3)

    def test_restore_recovers_weights_and_progress_strictly(self):
        online = TinyPolicy(weight=1.0, decay=0.9, update_after_step=1)
        ema = PolicyEMA.from_policy_config(online)
        with torch.no_grad():
            online.weight.fill_(7.0)
        ema.update(online, step=1)

        restored = PolicyEMA.from_policy_config(
            TinyPolicy(weight=-1.0, decay=0.9, update_after_step=1)
        )
        restored.restore(ema.ema_policy, ema.metadata())

        torch.testing.assert_close(
            restored.ema_policy.weight,
            ema.ema_policy.weight,
        )
        self.assertEqual(restored.metadata(), ema.metadata())

        incompatible = PolicyEMA.from_policy_config(
            TinyPolicy(weight=0.0, decay=0.99, update_after_step=1)
        )
        with self.assertRaisesRegex(ValueError, "EMA配置与checkpoint不一致"):
            incompatible.restore(ema.ema_policy, ema.metadata())

    def test_disabled_config_does_not_create_shadow_policy(self):
        self.assertIsNone(PolicyEMA.from_policy_config(TinyPolicy(use_ema=False)))


class EMACheckpointTest(unittest.TestCase):
    def test_checkpoint_saves_deploy_and_online_weights_separately(self):
        cfg = OmegaConf.create(
            {
                "policy": {"name": "tiny"},
                "dataset_repo_id": "local/test",
                "env": {"name": "test"},
                "seed": 1,
                "resume": False,
                "wandb": {"enable": False},
            }
        )
        online = TinyPolicy(weight=5.0)
        deploy = TinyPolicy(weight=4.0)
        optimizer = torch.optim.SGD(online.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(cfg, temp_dir)
            logger.save_checkpoint(
                7,
                online,
                optimizer,
                None,
                identifier="000007",
                ema_policy=deploy,
                ema_state={
                    "ready": True,
                    "decay": 0.5,
                    "update_after_step": 2,
                    "num_updates": 6,
                    "last_step": 7,
                },
            )

            checkpoint_dir = Path(temp_dir) / "checkpoints" / "000007"
            deploy_state = torch.load(
                checkpoint_dir / "pretrained_model" / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            online_state = torch.load(
                checkpoint_dir / "online_pretrained_model" / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            training_state = torch.load(
                checkpoint_dir / "training_state.pth",
                map_location="cpu",
                weights_only=False,
            )

            torch.testing.assert_close(deploy_state["weight"], torch.tensor([4.0]))
            torch.testing.assert_close(online_state["weight"], torch.tensor([5.0]))
            self.assertEqual(training_state["ema"]["num_updates"], 6)
            self.assertTrue((Path(temp_dir) / "checkpoints" / "last").is_symlink())


if __name__ == "__main__":
    unittest.main()
