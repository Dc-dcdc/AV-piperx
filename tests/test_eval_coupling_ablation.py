import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from train.s2_incremental.eval.eval_coupling_ablation import (
    _output_root,
    make_preset_eval_cfg,
    main,
    normalize_ablation_presets,
)


class EvalCouplingAblationTest(unittest.TestCase):
    def test_output_root_automatically_appends_seed_and_episode_count(self):
        checkpoint = Path("/tmp/checkpoint")
        requested = Path("/tmp/custom_ablation")

        self.assertEqual(
            _output_root(checkpoint, None, seed=100, n_episodes=200),
            checkpoint / "coupling_ablation" / "eval_seed=100_ep=200",
        )
        self.assertEqual(
            _output_root(
                checkpoint,
                str(requested),
                seed=1000,
                n_episodes=300,
            ),
            requested / "eval_seed=1000_ep=300",
        )
        self.assertEqual(
            _output_root(
                checkpoint,
                str(requested / "eval_seed=1000_ep=300"),
                seed=1000,
                n_episodes=300,
            ),
            requested / "eval_seed=1000_ep=300",
        )

    def test_checkpoint_config_preserves_checkpoint_scales(self):
        presets = normalize_ablation_presets(
            [("checkpoint_config", None, None), ("uncoupled", 0, 0)]
        )

        self.assertEqual(
            presets,
            [
                ("checkpoint_config", None, None),
                ("uncoupled", 0.0, 0.0),
            ],
        )

    def test_invalid_presets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能重复"):
            normalize_ablation_presets([("same", 0, 0), ("same", 1, 1)])
        with self.assertRaisesRegex(ValueError, "view_to_arm_coupling_scale"):
            normalize_ablation_presets([("bad_scale", 1.1, 0)])
        with self.assertRaisesRegex(ValueError, "单层非空目录名"):
            normalize_ablation_presets([("../bad_name", 0, 0)])

    def test_preset_config_clones_common_fields_and_disables_pruning(self):
        source = SimpleNamespace(
            seed=100,
            render_camera=["zed_cam_left"],
            prune_checkpoints=True,
            prune_eval_outputs=True,
        )

        result = make_preset_eval_cfg(
            source,
            checkpoint=Path("/tmp/checkpoint"),
            preset_output=Path("/tmp/output"),
            view_to_arm_scale=0.5,
            arm_to_view_scale=0.0,
        )

        self.assertEqual(result.seed, 100)
        self.assertEqual(result.ckpt_path, "/tmp/checkpoint")
        self.assertEqual(result.eval_output_dir, "/tmp/output")
        self.assertEqual(result.view_to_arm_coupling_scale, 0.5)
        self.assertEqual(result.arm_to_view_coupling_scale, 0.0)
        self.assertFalse(result.prune_checkpoints)
        self.assertFalse(result.prune_eval_outputs)
        self.assertIsNot(result.render_camera, source.render_camera)

    def test_main_runs_all_configured_presets_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint"
            output = root / "output"
            checkpoint.mkdir()
            eval_cfg = SimpleNamespace(
                ckpt_path=str(checkpoint),
                eval_output_dir=str(output),
                seed=100,
                n_episodes=2,
                ablation_presets=[
                    ("trained", 0.5, 0.0),
                    ("uncoupled", 0.0, 0.0),
                ],
                render_camera=["overhead_cam"],
            )

            def fake_evaluate(preset_cfg):
                return [
                    {
                        "status": "ok",
                        "checkpoint_name": "checkpoint",
                        "eval_output_dir": preset_cfg.eval_output_dir,
                    }
                ]

            with patch(
                "train.s2_incremental.eval.eval_coupling_ablation.evaluate_policy",
                side_effect=fake_evaluate,
            ) as evaluate_mock:
                rows = main(eval_cfg)

            self.assertEqual(evaluate_mock.call_count, 2)
            self.assertEqual([row["preset"] for row in rows], ["trained", "uncoupled"])
            self.assertEqual(rows[0]["requested_view_to_arm_scale"], 0.5)
            self.assertEqual(rows[1]["requested_view_to_arm_scale"], 0.0)

            summary_path = (
                output
                / "eval_seed=100_ep=2"
                / "coupling_ablation_summary.json"
            )
            self.assertTrue(summary_path.is_file())
            saved_rows = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_rows, rows)


if __name__ == "__main__":
    unittest.main()
