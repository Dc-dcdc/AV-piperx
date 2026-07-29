import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from train.s2_incremental.eval.eval_output_corrector_ablation import (
    _output_root,
    main,
    make_preset_eval_cfg,
    normalize_ablation_presets,
)
from train.s1_pretrain.eval.output_corrector_ablation import (
    apply_output_corrector_ablation_overrides,
    output_corrector_ablation_tag,
)


class _FakeOutputCorrector:
    def __init__(self):
        self.view_to_arm_output_scale = 0.5
        self.arm_to_view_output_scale = 0.5

    def set_output_correction_scales(
        self,
        *,
        view_to_arm=None,
        arm_to_view=None,
    ):
        if view_to_arm is not None:
            self.view_to_arm_output_scale = float(view_to_arm)
        if arm_to_view is not None:
            self.arm_to_view_output_scale = float(arm_to_view)
        return {
            "view_to_arm_output_scale": self.view_to_arm_output_scale,
            "arm_to_view_output_scale": self.arm_to_view_output_scale,
        }


class EvalOutputCorrectorAblationTest(unittest.TestCase):
    def test_output_root_automatically_appends_seed_and_episode_count(self):
        checkpoint = Path("/tmp/checkpoint")
        requested = Path("/tmp/custom_ablation")

        self.assertEqual(
            _output_root(checkpoint, None, seed=1000, n_episodes=200),
            checkpoint
            / "output_corrector_ablation"
            / "eval_seed=1000_ep=200",
        )
        self.assertEqual(
            _output_root(
                checkpoint,
                str(requested),
                seed=2000,
                n_episodes=300,
            ),
            requested / "eval_seed=2000_ep=300",
        )

    def test_presets_validate_output_scales_and_names(self):
        self.assertEqual(
            normalize_ablation_presets(
                [("baseline", 0, 0), ("arm_to_view", 0, 0.5)]
            ),
            [
                ("baseline", 0.0, 0.0),
                ("arm_to_view", 0.0, 0.5),
            ],
        )
        with self.assertRaisesRegex(ValueError, "不能重复"):
            normalize_ablation_presets([("same", 0, 0), ("same", 0, 1)])
        with self.assertRaisesRegex(ValueError, "arm_to_view_output_scale"):
            normalize_ablation_presets([("bad", 0, 1.1)])
        with self.assertRaisesRegex(ValueError, "单层非空目录名"):
            normalize_ablation_presets([("../bad", 0, 0)])

    def test_output_scale_override_calls_post_diffusion_setter(self):
        policy = SimpleNamespace(
            diffusion=_FakeOutputCorrector(),
            config=SimpleNamespace(),
        )
        eval_cfg = SimpleNamespace(
            view_to_arm_output_scale=0.0,
            arm_to_view_output_scale=0.25,
        )

        active = apply_output_corrector_ablation_overrides(policy, eval_cfg)

        self.assertEqual(
            active,
            {
                "view_to_arm_output_scale": 0.0,
                "arm_to_view_output_scale": 0.25,
            },
        )
        self.assertEqual(policy.config.view_to_arm_output_scale, 0.0)
        self.assertEqual(policy.config.arm_to_view_output_scale, 0.25)
        self.assertEqual(
            output_corrector_ablation_tag(eval_cfg),
            "_out_v2a=0_out_a2v=0.25",
        )

    def test_output_override_rejects_non_output_corrector_policy(self):
        policy = SimpleNamespace(diffusion=SimpleNamespace())
        eval_cfg = SimpleNamespace(
            view_to_arm_output_scale=0.0,
            arm_to_view_output_scale=1.0,
        )
        with self.assertRaisesRegex(
            TypeError,
            "post_diffusion_output_corrector",
        ):
            apply_output_corrector_ablation_overrides(policy, eval_cfg)

    def test_preset_config_disables_coupling_and_pruning(self):
        source = SimpleNamespace(
            seed=1000,
            render_camera=["zed_cam_left"],
            prune_checkpoints=True,
            prune_eval_outputs=True,
        )

        result = make_preset_eval_cfg(
            source,
            checkpoint=Path("/tmp/checkpoint"),
            preset_output=Path("/tmp/output"),
            view_to_arm_scale=0.0,
            arm_to_view_scale=0.75,
        )

        self.assertEqual(result.view_to_arm_output_scale, 0.0)
        self.assertEqual(result.arm_to_view_output_scale, 0.75)
        self.assertIsNone(result.view_to_arm_coupling_scale)
        self.assertIsNone(result.arm_to_view_coupling_scale)
        self.assertFalse(result.prune_checkpoints)
        self.assertFalse(result.prune_eval_outputs)
        self.assertIsNot(result.render_camera, source.render_camera)

    def test_main_runs_all_presets_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint"
            output = root / "output"
            checkpoint.mkdir()
            eval_cfg = SimpleNamespace(
                ckpt_path=str(checkpoint),
                eval_output_dir=str(output),
                seed=1000,
                n_episodes=2,
                ablation_presets=[
                    ("baseline", 0.0, 0.0),
                    ("arm_to_view", 0.0, 0.5),
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
                "train.s2_incremental.eval.eval_output_corrector_ablation.evaluate_policy",
                side_effect=fake_evaluate,
            ) as evaluate_mock:
                rows = main(eval_cfg)

            self.assertEqual(evaluate_mock.call_count, 2)
            self.assertEqual(
                [row["preset"] for row in rows],
                ["baseline", "arm_to_view"],
            )
            self.assertEqual(rows[0]["requested_arm_to_view_output_scale"], 0.0)
            self.assertEqual(rows[1]["requested_arm_to_view_output_scale"], 0.5)

            summary_path = (
                output
                / "eval_seed=1000_ep=2"
                / "output_corrector_ablation_summary.json"
            )
            self.assertTrue(summary_path.is_file())
            saved_rows = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_rows, rows)


if __name__ == "__main__":
    unittest.main()
