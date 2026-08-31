import unittest

from omegaconf import OmegaConf

from lerobot.common.logger import Logger


class _FakeWandb:
    def __init__(self):
        self.log_calls = []

    def log(self, payload, step):
        self.log_calls.append((payload, step))

    @staticmethod
    def Video(path, format):
        return {"path": path, "format": format}


class LoggerEvalStepTest(unittest.TestCase):
    def make_logger(self):
        logger = Logger.__new__(Logger)
        logger._wandb = _FakeWandb()
        logger._cfg = OmegaConf.create(
            {"training": {"steps_per_epoch": 50}}
        )
        return logger

    def test_async_eval_uses_checkpoint_step_as_custom_axis(self):
        logger = self.make_logger()

        logger.log_dict(
            {
                "success_rate": 0.8,
                "checkpoint_step": 100,
                "evaluation_lag_steps": 50,
            },
            step=150,
            mode="eval",
        )

        self.assertEqual(len(logger._wandb.log_calls), 1)
        payload, wandb_step = logger._wandb.log_calls[0]
        self.assertEqual(wandb_step, 150)
        self.assertEqual(payload["eval/checkpoint_step"], 100)
        self.assertEqual(payload["eval/success_rate"], 0.8)

    def test_synchronous_eval_defaults_checkpoint_step_to_current_step(self):
        logger = self.make_logger()

        logger.log_dict({"success_rate": 0.9}, step=200, mode="eval")

        payload, wandb_step = logger._wandb.log_calls[0]
        self.assertEqual(wandb_step, 200)
        self.assertEqual(payload["eval/checkpoint_step"], 200)

    def test_success_rate_is_also_logged_against_evaluated_epoch(self):
        logger = self.make_logger()

        logger.log_dict(
            {
                "success_rate_percent": 80.0,
                "checkpoint_step": 99,
                "evaluation_lag_steps": 51,
            },
            step=150,
            mode="eval",
        )

        payload, wandb_step = logger._wandb.log_calls[0]
        self.assertEqual(wandb_step, 150)
        self.assertEqual(payload["eval/checkpoint_step"], 99)
        self.assertEqual(payload["eval_by_epoch/epoch"], 2.0)
        self.assertEqual(
            payload["eval_by_epoch/success_rate_percent"],
            80.0,
        )

    def test_epoch_success_metric_is_skipped_without_epoch_budget(self):
        logger = self.make_logger()
        logger._cfg = OmegaConf.create({"training": {}})

        logger.log_dict(
            {"success_rate_percent": 80.0},
            step=99,
            mode="eval",
        )

        payload, _ = logger._wandb.log_calls[0]
        self.assertNotIn("eval_by_epoch/epoch", payload)
        self.assertNotIn(
            "eval_by_epoch/success_rate_percent",
            payload,
        )

    def test_async_video_uses_evaluated_checkpoint_step(self):
        logger = self.make_logger()

        logger.log_video(
            "/tmp/eval.mp4",
            step=150,
            mode="eval",
            checkpoint_step=100,
        )

        payload, wandb_step = logger._wandb.log_calls[0]
        self.assertEqual(wandb_step, 150)
        self.assertEqual(payload["eval/checkpoint_step"], 100)
        self.assertEqual(payload["eval/video"]["path"], "/tmp/eval.mp4")


if __name__ == "__main__":
    unittest.main()
