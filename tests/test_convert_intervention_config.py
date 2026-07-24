import unittest
from unittest.mock import patch

from hugging_face import convert_intervention_to_hf as converter


class ConvertInterventionConfigTest(unittest.TestCase):
    def test_common_cli_defaults_match_base_converter(self) -> None:
        intervention_args = converter.build_arg_parser().parse_args([])
        base_args = converter.base.build_arg_parser().parse_args([])

        for name in (
            "overwrite",
            "max_episodes",
            "cameras",
            "fps",
            "max_image_stat_frames",
            "action_key",
            "http_proxy",
            "https_proxy",
        ):
            self.assertEqual(getattr(intervention_args, name), getattr(base_args, name))

    def test_python_configuration_is_forwarded_as_namespace(self) -> None:
        with (
            patch.object(converter.base, "init_logging"),
            patch.object(converter, "run_from_args") as run_from_args,
        ):
            converter.convert_intervention_folder_to_hf(
                raw_dir="raw/run",
                output_dir="output/dataset",
                dataset_mode="full_rollout",
                overwrite=False,
                max_episodes=20,
                cameras="zed_cam_left,zed_cam_right",
                fps=25,
                max_image_stat_frames=100,
                action_key="joint_action",
                filter_mode="saved_segments",
                min_blend_weight=0.25,
                merge_gap_frames=2,
                pre_context_frames=3,
                post_context_frames=4,
                min_segment_frames=16,
                http_proxy="http://proxy",
                https_proxy="https://proxy",
            )

        run_from_args.assert_called_once()
        args = run_from_args.call_args.args[0]
        self.assertEqual(vars(args), {
            "raw_dir": "raw/run",
            "output_dir": "output/dataset",
            "dataset_mode": "full_rollout",
            "overwrite": False,
            "max_episodes": 20,
            "cameras": "zed_cam_left,zed_cam_right",
            "fps": 25,
            "max_image_stat_frames": 100,
            "action_key": "joint_action",
            "filter_mode": "saved_segments",
            "min_blend_weight": 0.25,
            "merge_gap_frames": 2,
            "pre_context_frames": 3,
            "post_context_frames": 4,
            "min_segment_frames": 16,
            "http_proxy": "http://proxy",
            "https_proxy": "https://proxy",
        })

    def test_full_rollout_mode_and_action_mask_expansion(self) -> None:
        args = converter.build_arg_parser().parse_args(["--dataset-mode", "full_rollout"])
        self.assertEqual(args.dataset_mode, converter.DATASET_MODE_FULL_ROLLOUT)

        arm_values = converter.np.asarray([[True, False, True]], dtype=bool)
        expanded = converter.expand_arm_values_to_action(arm_values, action_dim=20)
        self.assertEqual(expanded.shape, (1, 20))
        self.assertTrue(expanded[0, :7].all())
        self.assertFalse(expanded[0, 7:14].any())
        self.assertTrue(expanded[0, 14:].all())


if __name__ == "__main__":
    unittest.main()
