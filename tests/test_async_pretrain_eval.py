import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from train.s1_pretrain.eval.async_eval import (
    AsyncEvalController,
    finalize_async_eval_result,
    make_evaluated_checkpoint_identifier,
    migrate_evaluated_checkpoint_names,
    rename_evaluated_checkpoint,
)
from train.s1_pretrain.eval.eval_train import (
    TopKCheckpointManager,
    make_checkpoint_identifier,
)


class _RecordingLogger:
    def __init__(self):
        self.dict_calls = []
        self.video_calls = []

    def log_dict(self, metrics, step, mode):
        self.dict_calls.append((metrics, step, mode))

    def log_video(self, video_path, step, mode, checkpoint_step=None):
        self.video_calls.append(
            (
                video_path,
                step,
                mode,
                checkpoint_step,
                Path(video_path).exists(),
            )
        )


class AsyncPretrainEvalTest(unittest.TestCase):
    def test_worker_reports_invalid_eval_gpu_during_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = AsyncEvalController(
                policy_name="coupled_dual_head_diffusion",
                env_id="unused/unused",
                obs_cameras=[],
                eval_cfg=OmegaConf.create(
                    {
                        "n_episodes": 1,
                        "batch_size": 1,
                        "max_steps": 1,
                    }
                ),
                eval_device="cuda:999999",
                use_amp=False,
                amp_dtype="bfloat16",
                out_dir=directory,
                startup_timeout_s=30,
            )
            with self.assertRaisesRegex(RuntimeError, "异步评估进程初始化失败"):
                controller.start()

    def test_checkpoint_identifier_does_not_depend_on_delayed_eval_metrics(self):
        self.assertEqual(
            make_checkpoint_identifier(25, 200_000, 0.123456),
            "000025_loss=0.1235",
        )

    def test_evaluated_checkpoint_identifier_matches_synchronous_format(self):
        self.assertEqual(
            make_evaluated_checkpoint_identifier(
                "000025_loss=0.1235",
                success_rate=0.875,
                average_reward=751.504,
            ),
            "000025_loss=0.1235_sr=87.5_ar=751.50",
        )
        self.assertEqual(
            make_evaluated_checkpoint_identifier(
                "000025_loss=0.1235_sr=10.0_ar=1.00",
                success_rate=0.875,
                average_reward=751.504,
            ),
            "000025_loss=0.1235_sr=87.5_ar=751.50",
        )

    def test_renaming_old_checkpoint_does_not_replace_newer_last_link(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoints_dir = Path(directory) / "checkpoints"
            old_checkpoint = checkpoints_dir / "000100_loss=0.2000"
            new_checkpoint = checkpoints_dir / "000200_loss=0.1000"
            old_checkpoint.mkdir(parents=True)
            new_checkpoint.mkdir()
            last_link = checkpoints_dir / "last"
            last_link.symlink_to(new_checkpoint.absolute(), target_is_directory=True)

            renamed = rename_evaluated_checkpoint(
                old_checkpoint,
                success_rate=0.9,
                average_reward=10.0,
            )

            self.assertEqual(
                renamed.name,
                "000100_loss=0.2000_sr=90.0_ar=10.00",
            )
            self.assertEqual(last_link.resolve(), new_checkpoint.resolve())

    def test_historical_checkpoint_migration_updates_records_and_last_link(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoints_dir = Path(directory) / "checkpoints"
            checkpoint = checkpoints_dir / "000100_loss=0.2000"
            checkpoint.mkdir(parents=True)
            (checkpoint / "eval_metrics.json").write_text(
                json.dumps(
                    {
                        "aggregated": {
                            "success_rate": 0.75,
                            "average_reward": 123.456,
                        }
                    }
                ),
                encoding="utf-8",
            )
            last_link = checkpoints_dir / "last"
            last_link.symlink_to(checkpoint.absolute(), target_is_directory=True)
            records_path = checkpoints_dir / "top_k_records.json"
            records_path.write_text(
                json.dumps(
                    {
                        "latest": str(checkpoint.absolute()),
                        "latest_step": 100,
                        "top_k": [
                            {
                                "step": 100,
                                "path": str(checkpoint.absolute()),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            renamed_paths = migrate_evaluated_checkpoint_names(checkpoints_dir)

            renamed = checkpoint.with_name(
                "000100_loss=0.2000_sr=75.0_ar=123.46"
            )
            self.assertEqual(renamed_paths[str(checkpoint.absolute())], str(renamed))
            self.assertTrue(renamed.is_dir())
            self.assertEqual(last_link.resolve(), renamed.resolve())
            records = json.loads(records_path.read_text(encoding="utf-8"))
            self.assertEqual(records["latest"], str(renamed))
            self.assertEqual(records["top_k"][0]["path"], str(renamed))

    def test_pending_checkpoint_is_protected_and_old_result_cannot_replace_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            manager = TopKCheckpointManager(
                out_dir=out_dir,
                max_keep=1,
                metric="success",
                records_resume=False,
            )
            old_pending = out_dir / "checkpoints" / "000100_loss=0.2000"
            newer = out_dir / "checkpoints" / "000200_loss=0.1500"
            old_pending.mkdir(parents=True)
            newer.mkdir(parents=True)

            manager.protect(old_pending)
            manager.update(
                200,
                0.15,
                newer,
                reward=-10.0,
                success_rate=0.1,
            )
            self.assertTrue(old_pending.exists())

            manager.release(old_pending)
            manager.update(
                100,
                0.2,
                old_pending,
                reward=10.0,
                success_rate=0.9,
            )

            self.assertEqual(manager.latest_step, 200)
            self.assertEqual(manager.latest_path, newer)
            self.assertEqual(manager.top_k[0]["path"], old_pending)
            records = json.loads(manager.records_file.read_text())
            self.assertEqual(records["latest_step"], 200)

    def test_unassessed_success_checkpoint_updates_latest_but_not_top_k(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            manager = TopKCheckpointManager(
                out_dir=out_dir,
                max_keep=2,
                metric="success",
                records_resume=False,
            )
            evaluated = out_dir / "checkpoints" / "000100_loss=0.2000"
            unassessed = out_dir / "checkpoints" / "000200_loss=0.1500"
            evaluated.mkdir(parents=True)

            manager.update(
                100,
                0.2,
                evaluated,
                reward=10.0,
                success_rate=0.8,
            )
            unassessed.mkdir()
            manager.update(200, 0.15, unassessed)

            self.assertEqual(manager.latest_path, unassessed.absolute())
            self.assertEqual(
                [item["step"] for item in manager.top_k],
                [100],
            )
            self.assertTrue(evaluated.exists())
            self.assertTrue(unassessed.exists())
            raw_records = manager.records_file.read_text(encoding="utf-8")
            records = json.loads(
                raw_records,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(value)
                ),
            )
            self.assertNotIn("Infinity", raw_records)
            self.assertEqual(len(records["top_k"]), 1)
            self.assertTrue(Path(records["latest"]).is_absolute())
            self.assertTrue(Path(records["top_k"][0]["path"]).is_absolute())

    def test_loss_top_k_serializes_missing_eval_metrics_as_json_null(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            manager = TopKCheckpointManager(
                out_dir=out_dir,
                max_keep=1,
                metric="loss",
                records_resume=False,
            )
            checkpoint = out_dir / "checkpoints" / "000100_loss=0.2000"
            checkpoint.mkdir(parents=True)

            manager.update(100, 0.2, checkpoint)

            raw_records = manager.records_file.read_text(encoding="utf-8")
            records = json.loads(
                raw_records,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(value)
                ),
            )
            self.assertIsNone(records["top_k"][0]["reward"])
            self.assertIsNone(records["top_k"][0]["success_rate"])

    def test_resume_normalizes_legacy_infinity_records(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            checkpoints_dir = out_dir / "checkpoints"
            evaluated = checkpoints_dir / "000100_loss=0.2000"
            unassessed = checkpoints_dir / "000200_loss=0.1500"
            evaluated.mkdir(parents=True)
            unassessed.mkdir()
            records_path = checkpoints_dir / "top_k_records.json"
            records_path.write_text(
                json.dumps(
                    {
                        "latest": str(unassessed),
                        "latest_step": 200,
                        "top_k": [
                            {
                                "step": 100,
                                "loss": 0.2,
                                "reward": 10.0,
                                "success_rate": 0.8,
                                "path": str(evaluated),
                            },
                            {
                                "step": 200,
                                "loss": 0.15,
                                "reward": -float("inf"),
                                "success_rate": -float("inf"),
                                "path": str(unassessed),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manager = TopKCheckpointManager(
                out_dir=out_dir,
                max_keep=2,
                metric="success",
                records_resume=True,
            )

            self.assertEqual(
                [item["step"] for item in manager.top_k],
                [100],
            )
            normalized_records = records_path.read_text(encoding="utf-8")
            json.loads(
                normalized_records,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(value)
                ),
            )
            self.assertNotIn("Infinity", normalized_records)

    def test_async_result_logs_at_current_step_and_archives_by_checkpoint_step(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            checkpoint = out_dir / "checkpoints" / "000100_loss=0.2500"
            checkpoint.mkdir(parents=True)
            last_link = checkpoint.parent / "last"
            last_link.symlink_to(checkpoint.absolute(), target_is_directory=True)
            videos_dir = out_dir / "eval" / "videos_000100_loss=0.2500"
            videos_dir.mkdir(parents=True)
            video_path = videos_dir / "episode.mp4"
            video_path.write_bytes(b"video")

            manager = TopKCheckpointManager(
                out_dir=out_dir,
                max_keep=2,
                metric="success",
                records_resume=False,
            )
            manager.protect(checkpoint)
            logger = _RecordingLogger()
            cfg = SimpleNamespace(wandb=SimpleNamespace(enable=True))
            result = {
                "kind": "eval_result",
                "ok": True,
                "request": {
                    "task_id": "100-test",
                    "step": 100,
                    "train_loss": 0.25,
                    "base_identifier": "000100_loss=0.2500",
                    "snapshot_path": str(checkpoint / "pretrained_model"),
                    "videos_dir": str(videos_dir),
                    "checkpoint_path": str(checkpoint),
                    "cleanup_snapshot_dir": None,
                },
                "eval_info": {
                    "aggregated": {
                        "success_rate": 0.8,
                        "average_reward": 12.0,
                        "avg_inference_ms": 5.0,
                        "max_inference_ms": 8.0,
                    },
                    "video_paths": [str(video_path)],
                    "episodes": [
                        {
                            "episode": 0,
                            "seed": 100,
                            "success": True,
                            "reward": 12.0,
                            "steps": 20,
                        }
                    ],
                },
            }

            metrics = finalize_async_eval_result(
                result,
                logger=logger,
                cfg=cfg,
                manager=manager,
                logging_step=150,
            )

            self.assertEqual(metrics["checkpoint_step"], 100)
            self.assertEqual(metrics["evaluation_lag_steps"], 50)
            self.assertEqual(logger.dict_calls[0][1:], (150, "eval"))
            self.assertEqual(
                logger.video_calls[0][1:],
                (150, "eval", 100, True),
            )
            renamed_checkpoint = checkpoint.with_name(
                "000100_loss=0.2500_sr=80.0_ar=12.00"
            )
            self.assertFalse(checkpoint.exists())
            self.assertTrue(
                (renamed_checkpoint / "eval_videos" / "episode.mp4").is_file()
            )
            self.assertEqual(last_link.resolve(), renamed_checkpoint.resolve())
            saved_metrics = json.loads(
                (renamed_checkpoint / "eval_metrics.json").read_text()
            )
            self.assertEqual(saved_metrics["checkpoint_step"], 100)
            self.assertEqual(manager.top_k[0]["success_rate"], 0.8)
            self.assertEqual(manager.top_k[0]["path"], renamed_checkpoint)


if __name__ == "__main__":
    unittest.main()
