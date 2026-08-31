import unittest
from types import SimpleNamespace

from omegaconf import OmegaConf

from train.s1_pretrain.train.wandb_logging import (
    WANDB_TAG_MAX_LENGTH,
    _sanitize_wandb_tag,
    add_wandb_parameter_tags,
    log_train_info,
)


class PretrainWandbLoggingTest(unittest.TestCase):
    def test_overlong_tag_keeps_suffix_within_wandb_limit(self):
        raw_tag = (
            "quest_teleop_SewNeedle-3Arms-v0_rgb_"
            "arm_view_mixed_random_recovery_3+0"
        )

        tag = _sanitize_wandb_tag(raw_tag)

        self.assertEqual(len(tag), WANDB_TAG_MAX_LENGTH)
        self.assertTrue(tag.startswith("..."))
        self.assertTrue(tag.endswith("arm_view_mixed_random_recovery_3+0"))

    def test_add_wandb_parameter_tags_appends_configured_and_parameter_tags(self):
        run = SimpleNamespace(tags=("existing",))
        logger = SimpleNamespace(_wandb=SimpleNamespace(run=run))
        cfg = OmegaConf.create(
            {
                "wandb": {"tags": ["manual"]},
                "dataset_local_dir": (
                    "outputs/5_hf_datasets/InsertCylinder-3Arms/"
                    "quest_teleop_InsertCylinder-3Arms-v0_rgb_view_random_recovery"
                ),
                "dataset_repo_id": "iantc104/gv_quest_teleop_insert_cylinder_3arms",
                "device": "cuda:0",
                "use_amp": True,
                "training": {
                    "batch_size": 32,
                    "lr": 1e-4,
                    "lr_backbone": 1e-5,
                    "offline_steps": 200000,
                    "lr_scheduler": "cosine",
                    "lr_warmup_steps": 5000,
                    "weight_decay": 1e-6,
                    "grad_clip_norm": 10,
                    "image_transforms": {"enable": False},
                },
                "policy": {
                    "n_obs_steps": 2,
                    "horizon": 16,
                    "n_action_steps": 8,
                    "down_dims": [128, 256, 512],
                    "n_groups": 8,
                    "noise_scheduler_type": "DDIM",
                    "num_train_timesteps": 100,
                    "num_inference_steps": 10,
                    "prediction_type": "epsilon",
                    "use_ema": True,
                    "ema_decay": 0.999,
                    "arm_action_dim": 14,
                    "view_action_dim": 6,
                    "view_loss_weight": 0.5,
                    "coupling_mode": "rbac",
                    "coupling_block_type": "role_adaln_zero",
                    "coupling_use_temporal_pos_emb": True,
                    "coupling_use_ffn": True,
                    "coupling_ffn_ratio": 2.0,
                    "view_to_arm_coupling_scale": 0.5,
                    "arm_to_view_coupling_scale": 0.5,
                },
            }
        )

        add_wandb_parameter_tags(logger, cfg)

        self.assertIn("existing", run.tags)
        self.assertIn("manual", run.tags)
        self.assertIn(
            "quest_teleop_InsertCylinder-3Arms-v0_rgb_view_random_recovery",
            run.tags,
        )
        self.assertNotIn(
            "outputs/5_hf_datasets/InsertCylinder-3Arms/"
            "quest_teleop_InsertCylinder-3Arms-v0_rgb_view_random_recovery",
            run.tags,
        )
        self.assertIn("batch:32", run.tags)
        self.assertIn("coupling:rbac", run.tags)
        self.assertIn("down_dims:128-256-512", run.tags)
        self.assertIn("coupling_ffn:true", run.tags)

    def test_log_train_info_uploads_progress_and_loss_metrics(self):
        class CaptureLogger:
            def __init__(self):
                self.calls = []

            def log_dict(self, payload, step, mode):
                self.calls.append((dict(payload), step, mode))

        logger = CaptureLogger()
        cfg = OmegaConf.create({"training": {"batch_size": 4}})
        dataset = SimpleNamespace(num_samples=100, num_episodes=10)
        info = {
            "loss": 1.25,
            "arm_loss": 1.0,
            "view_loss": 0.5,
            "grad_norm": 0.75,
            "lr": 1e-4,
            "update_s": 0.2,
            "dataloading_s": 0.03,
        }

        log_train_info(logger, info, step=24, cfg=cfg, dataset=dataset)

        self.assertEqual(len(logger.calls), 1)
        payload, step, mode = logger.calls[0]
        self.assertEqual(step, 24)
        self.assertEqual(mode, "train")
        self.assertEqual(payload["num_samples"], 100)
        self.assertEqual(payload["num_episodes"], 10)
        self.assertEqual(payload["num_epochs"], 1)
        self.assertEqual(payload["arm_loss"], 1.0)
        self.assertEqual(payload["view_loss"], 0.5)

    def test_log_train_info_uses_resolved_sampler_epoch_length(self):
        class CaptureLogger:
            def __init__(self):
                self.calls = []

            def log_dict(self, payload, step, mode):
                self.calls.append((dict(payload), step, mode))

        logger = CaptureLogger()
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 4,
                    "effective_num_samples": 12,
                    "steps_per_epoch": 3,
                }
            }
        )
        dataset = SimpleNamespace(num_samples=20, num_episodes=4)
        info = {
            "loss": 1.0,
            "grad_norm": 0.5,
            "lr": 1e-4,
            "update_s": 0.2,
            "dataloading_s": 0.03,
        }

        log_train_info(logger, info, step=5, cfg=cfg, dataset=dataset)

        payload, _, _ = logger.calls[0]
        self.assertEqual(payload["num_epochs"], 2.0)
        self.assertEqual(payload["num_episodes"], 8.0)

    def test_epoch_progress_counts_the_final_small_batch_exactly(self):
        class CaptureLogger:
            def __init__(self):
                self.calls = []

            def log_dict(self, payload, step, mode):
                self.calls.append((dict(payload), step, mode))

        logger = CaptureLogger()
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 4,
                    "effective_num_samples": 10,
                    "steps_per_epoch": 3,
                    "resolved_drop_last": False,
                }
            }
        )
        dataset = SimpleNamespace(num_samples=10, num_episodes=2)
        info = {
            "loss": 1.0,
            "grad_norm": 0.5,
            "lr": 1e-4,
            "update_s": 0.2,
            "dataloading_s": 0.03,
        }

        log_train_info(logger, info, step=2, cfg=cfg, dataset=dataset)

        payload, _, _ = logger.calls[0]
        self.assertEqual(payload["num_samples"], 10)
        self.assertEqual(payload["num_epochs"], 1.0)


if __name__ == "__main__":
    unittest.main()
