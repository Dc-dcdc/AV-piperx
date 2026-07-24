import math
import unittest

import numpy as np

from data_collect.quest_pose_filter import (
    QuestPoseActionFilter,
    QuestPoseFilterConfig,
    quaternion_angle_rad,
)


def _pose_action() -> np.ndarray:
    action = np.zeros(23, dtype=np.float64)
    action[3] = 1.0
    action[11] = 1.0
    action[19] = 1.0
    return action


class QuestPoseActionFilterTest(unittest.TestCase):
    def test_first_frame_initializes_without_changing_anchor_pose(self) -> None:
        pose_filter = QuestPoseActionFilter(QuestPoseFilterConfig())
        action = _pose_action()
        action[0:3] = [0.1, -0.2, 0.3]
        action[7] = 0.25

        filtered = pose_filter.filter(action, timestamp=1.0)

        np.testing.assert_allclose(filtered, action)

    def test_position_one_euro_filter_and_step_limit_reject_large_jump(self) -> None:
        pose_filter = QuestPoseActionFilter(
            QuestPoseFilterConfig(max_position_step=0.01)
        )
        pose_filter.filter(_pose_action(), timestamp=1.0)
        jumped = _pose_action()
        jumped[0] = 1.0

        filtered = pose_filter.filter(jumped, timestamp=1.04)

        self.assertGreater(filtered[0], 0.0)
        self.assertLessEqual(float(np.linalg.norm(filtered[0:3])), 0.01 + 1.0e-12)

    def test_rotation_uses_shortest_slerp_path_and_step_limit(self) -> None:
        pose_filter = QuestPoseActionFilter(
            QuestPoseFilterConfig(
                rotation_alpha=0.35,
                max_rotation_step_deg=5.0,
            )
        )
        initial = _pose_action()
        pose_filter.filter(initial, timestamp=1.0)
        rotated = _pose_action()
        half_angle = math.radians(90.0) / 2.0
        rotated[3:7] = [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]

        filtered = pose_filter.filter(rotated, timestamp=1.04)

        angle_deg = math.degrees(quaternion_angle_rad(initial[3:7], filtered[3:7]))
        self.assertAlmostEqual(angle_deg, 5.0, places=7)

        pose_filter.reset()
        pose_filter.filter(initial, timestamp=2.0)
        same_rotation_opposite_sign = _pose_action()
        same_rotation_opposite_sign[3:7] = [-1.0, 0.0, 0.0, 0.0]
        filtered = pose_filter.filter(same_rotation_opposite_sign, timestamp=2.04)
        self.assertAlmostEqual(
            quaternion_angle_rad(initial[3:7], filtered[3:7]),
            0.0,
            places=12,
        )

    def test_gripper_deadband_and_exponential_smoothing(self) -> None:
        pose_filter = QuestPoseActionFilter(
            QuestPoseFilterConfig(
                gripper_alpha=0.4,
                gripper_deadband=0.02,
            )
        )
        initial = _pose_action()
        initial[7] = 0.5
        pose_filter.filter(initial, timestamp=1.0)

        small_jitter = _pose_action()
        small_jitter[7] = 0.51
        filtered_jitter = pose_filter.filter(small_jitter, timestamp=1.04)
        self.assertEqual(filtered_jitter[7], 0.5)

        close_command = _pose_action()
        close_command[7] = 1.0
        filtered_close = pose_filter.filter(close_command, timestamp=1.08)
        self.assertAlmostEqual(filtered_close[7], 0.7)

    def test_packet_gap_resets_velocity_without_disabling_step_limit(self) -> None:
        pose_filter = QuestPoseActionFilter(
            QuestPoseFilterConfig(
                max_position_step=0.01,
                tracking_reset_gap=0.25,
            )
        )
        pose_filter.filter(_pose_action(), timestamp=1.0)
        jumped = _pose_action()
        jumped[0] = 1.0
        before_gap = pose_filter.filter(jumped, timestamp=1.04)

        after_gap = pose_filter.filter(jumped, timestamp=1.50)

        step_after_recovery = float(np.linalg.norm(after_gap[0:3] - before_gap[0:3]))
        self.assertLessEqual(step_after_recovery, 0.01 + 1.0e-12)

    def test_explicit_reset_clears_episode_history(self) -> None:
        pose_filter = QuestPoseActionFilter(
            QuestPoseFilterConfig(max_position_step=0.01)
        )
        pose_filter.filter(_pose_action(), timestamp=1.0)
        pose_filter.reset()
        new_anchor = _pose_action()
        new_anchor[0] = 0.4

        filtered = pose_filter.filter(new_anchor, timestamp=2.0)

        np.testing.assert_allclose(filtered, new_anchor)

    def test_disabled_filter_is_exact_passthrough(self) -> None:
        pose_filter = QuestPoseActionFilter(
            QuestPoseFilterConfig(enabled=False)
        )
        action = np.arange(23, dtype=np.float64)

        filtered = pose_filter.filter(action, timestamp=1.0)

        np.testing.assert_array_equal(filtered, action)
        self.assertIsNot(filtered, action)


if __name__ == "__main__":
    unittest.main()
