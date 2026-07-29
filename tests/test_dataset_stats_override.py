import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from lerobot.common.datasets.factory import apply_dataset_stats_overrides


class DatasetStatsOverrideTest(unittest.TestCase):
    def setUp(self):
        self.dataset = SimpleNamespace(
            stats={
                "observation.images.zed_cam_left": {
                    "mean": torch.full((3, 1, 1), 0.5),
                    "std": torch.full((3, 1, 1), 0.25),
                    "min": torch.zeros((3, 1, 1)),
                    "max": torch.ones((3, 1, 1)),
                }
            }
        )

    def test_image_mean_std_are_replaced_exactly(self):
        overrides = OmegaConf.create(
            {
                "observation.images.zed_cam_left": {
                    "mean": [[[0.485]], [[0.456]], [[0.406]]],
                    "std": [[[0.229]], [[0.224]], [[0.225]]],
                }
            }
        )

        applied = apply_dataset_stats_overrides(self.dataset, overrides)

        torch.testing.assert_close(
            self.dataset.stats["observation.images.zed_cam_left"]["mean"],
            torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
        )
        torch.testing.assert_close(
            self.dataset.stats["observation.images.zed_cam_left"]["std"],
            torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
        )
        self.assertEqual(
            set(applied["observation.images.zed_cam_left"]),
            {"mean", "std"},
        )

    def test_shape_mismatch_fails_fast(self):
        overrides = {
            "observation.images.zed_cam_left": {
                "mean": [0.485, 0.456, 0.406],
            }
        }

        with self.assertRaisesRegex(ValueError, "形状错误"):
            apply_dataset_stats_overrides(self.dataset, overrides)

    def test_nonpositive_std_fails_fast(self):
        overrides = {
            "observation.images.zed_cam_left": {
                "std": [[[0.229]], [[0.0]], [[0.225]]],
            }
        }

        with self.assertRaisesRegex(ValueError, "必须全部大于0"):
            apply_dataset_stats_overrides(self.dataset, overrides)

    def test_unknown_feature_fails_fast(self):
        overrides = {
            "observation.images.unknown": {
                "mean": [[[0.5]], [[0.5]], [[0.5]]],
            }
        }

        with self.assertRaisesRegex(KeyError, "数据集不存在"):
            apply_dataset_stats_overrides(self.dataset, overrides)


if __name__ == "__main__":
    unittest.main()
