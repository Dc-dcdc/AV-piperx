import os
import unittest
from unittest.mock import patch

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym

import env as _registered_env  # noqa: F401 触发项目环境注册。


class SewNeedleEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make(
            "guided_vision/SewNeedle-3Arms-v0",
            cameras=[],
            disable_env_checker=True,
            enable_reward_debug=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_registered_env_resets_and_steps_with_piper_dimensions(self):
        observation, _ = self.env.reset(seed=123)

        self.assertEqual(observation["agent_pos"].shape, (20,))
        self.assertEqual(self.env.action_space.shape, (20,))
        self.assertTrue(self.env.observation_space.contains(observation))

        action = observation["agent_pos"].astype(np.float32)
        _, reward, terminated, truncated, info = self.env.step(action)

        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reward_debug"]["reward_total"], reward)
        self.assertIn("reward_terms", info["reward_debug"])
        self.assertIn("step_penalty", info["reward_debug"]["reward_terms"])

    def test_reset_seed_reproduces_task_object_positions(self):
        self.env.reset(seed=456)
        unwrapped = self.env.unwrapped
        first_needle = unwrapped._physics.bind(unwrapped._needle_joint).qpos.copy()
        first_wall = unwrapped._physics.bind(unwrapped._wall_joint).qpos.copy()

        self.env.reset(seed=456)
        second_needle = unwrapped._physics.bind(unwrapped._needle_joint).qpos.copy()
        second_wall = unwrapped._physics.bind(unwrapped._wall_joint).qpos.copy()

        np.testing.assert_allclose(first_needle, second_needle)
        np.testing.assert_allclose(first_wall, second_wall)

    def test_unnamed_finger_geom_falls_back_to_piper_body_name(self):
        self.env.reset(seed=789)
        unwrapped = self.env.unwrapped
        finger_geom_ids = [
            geom_id
            for geom_id in range(unwrapped._physics.model.ngeom)
            if (
                unwrapped._physics.model.id2name(
                    int(unwrapped._physics.model.geom_bodyid[geom_id]), "body"
                )
                or ""
            ).startswith("left_left_finger")
        ]

        self.assertTrue(finger_geom_ids)
        resolved_name = unwrapped._geom_or_body_name(finger_geom_ids[0])
        self.assertEqual(resolved_name, "left_left_finger_link")

    def test_stationary_near_needle_cannot_farm_positive_reward(self):
        self.env.reset(seed=101)
        unwrapped = self.env.unwrapped
        metrics = unwrapped._calculate_distances()
        metrics["dist_right_to_needle"] = 0.1
        unwrapped._prev_dists = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }

        with (
            patch.object(unwrapped, "_calculate_distances", return_value=metrics),
            patch.object(unwrapped, "_get_gripper_contact_flags", return_value=(False, False)),
        ):
            reward = unwrapped.get_reward()

        self.assertAlmostEqual(reward, -0.1)
        self.assertLess(reward, 0.0)
        self.assertNotIn(
            "right_approach_distance",
            unwrapped.reward_debug["reward_terms"],
        )

    def test_drop_after_handover_uses_strong_terminal_penalty(self):
        self.env.reset(seed=102)
        unwrapped = self.env.unwrapped
        metrics = unwrapped._calculate_distances()
        metrics["needle_z"] = metrics["hole_z"] - 0.04
        unwrapped._prev_dists = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }
        unwrapped.needle_was_grasped = True
        unwrapped.right_has_grasped = True
        unwrapped.left_has_grasped = True

        with (
            patch.object(unwrapped, "_calculate_distances", return_value=metrics),
            patch.object(unwrapped, "_get_gripper_contact_flags", return_value=(False, False)),
        ):
            reward = unwrapped.get_reward()

        self.assertAlmostEqual(reward, -500.1)
        self.assertTrue(unwrapped.terminated)
        self.assertFalse(unwrapped.is_success)
        self.assertEqual(
            unwrapped.reward_debug["reward_terms"]["penalty_drop_after_handover"],
            -500.0,
        )

    def test_head_entry_requires_negative_crossing_inside_aperture(self):
        self.env.reset(seed=105)
        unwrapped = self.env.unwrapped
        metrics = unwrapped._calculate_distances()
        entrance_pos = metrics["entrance_pos"].copy()
        metrics["head_pos"] = entrance_pos + np.array([-0.001, 0.03, 0.0])
        previous = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }
        previous["head_pos"] = entrance_pos + np.array([0.001, 0.03, 0.0])
        unwrapped._prev_dists = previous
        unwrapped.needle_was_grasped = True
        unwrapped.right_has_grasped = True

        with (
            patch.object(unwrapped, "_calculate_distances", return_value=metrics),
            patch.object(unwrapped, "_get_gripper_contact_flags", return_value=(False, True)),
        ):
            unwrapped.get_reward()

        self.assertFalse(unwrapped.needle_start_through)

        metrics["head_pos"] = entrance_pos + np.array([-0.001, 0.0, 0.0])
        previous["head_pos"] = entrance_pos + np.array([0.001, 0.0, 0.0])
        unwrapped._prev_dists = previous
        with (
            patch.object(unwrapped, "_calculate_distances", return_value=metrics),
            patch.object(unwrapped, "_get_gripper_contact_flags", return_value=(False, True)),
        ):
            unwrapped.get_reward()

        self.assertTrue(unwrapped.needle_start_through)
        self.assertIn(
            "needle_entered_hole",
            unwrapped.reward_debug["reward_terms"],
        )

    def test_tail_crossing_requires_left_only_handover_state(self):
        self.env.reset(seed=103)
        unwrapped = self.env.unwrapped
        metrics = unwrapped._calculate_distances()
        exit_pos = metrics["exit_pos"].copy()
        metrics["tail_pos"] = exit_pos + np.array([-0.001, 0.0, 0.0])
        metrics["dist_tail_to_exit"] = 0.001
        previous = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }
        previous["tail_pos"] = exit_pos + np.array([0.001, 0.0, 0.0])
        previous["dist_tail_to_exit"] = 0.001

        unwrapped.needle_was_grasped = True
        unwrapped.right_has_grasped = True
        unwrapped.needle_start_through = True
        unwrapped.needle_reached_exit = True
        unwrapped.left_has_grasped = True
        unwrapped._prev_dists = previous

        with (
            patch.object(unwrapped, "_calculate_distances", return_value=metrics),
            patch.object(unwrapped, "_get_gripper_contact_flags", return_value=(True, True)),
        ):
            unwrapped.get_reward()

        self.assertFalse(unwrapped.right_released_after_handover)
        self.assertFalse(unwrapped.needle_completely_through)

        unwrapped._prev_dists = previous
        with (
            patch.object(unwrapped, "_calculate_distances", return_value=metrics),
            patch.object(unwrapped, "_get_gripper_contact_flags", return_value=(True, False)),
        ):
            unwrapped.get_reward()

        self.assertTrue(unwrapped.right_released_after_handover)
        self.assertTrue(unwrapped.needle_completely_through)

    def test_success_requires_xyz_alignment_right_release_and_stability(self):
        self.env.reset(seed=104)
        unwrapped = self.env.unwrapped
        metrics = unwrapped._calculate_distances()
        metrics["x_error"] = 0.0
        metrics["y_error"] = 0.5
        metrics["z_error"] = 0.0
        metrics["composite_error_dist"] = 0.5

        unwrapped.needle_was_grasped = True
        unwrapped.right_has_grasped = True
        unwrapped.needle_start_through = True
        unwrapped.needle_reached_exit = True
        unwrapped.left_has_grasped = True
        unwrapped.right_released_after_handover = True
        unwrapped.needle_completely_through = True
        unwrapped._prev_dists = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }

        with (
            patch.object(unwrapped, "_calculate_distances", return_value=metrics),
            patch.object(unwrapped, "_get_gripper_contact_flags", return_value=(True, False)),
        ):
            unwrapped.get_reward()

        self.assertFalse(unwrapped.is_success)
        self.assertEqual(unwrapped.success_stable_count, 0)

        metrics["y_error"] = 0.0
        metrics["composite_error_dist"] = 0.0
        unwrapped._prev_dists = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }
        with (
            patch.object(unwrapped, "_calculate_distances", return_value=metrics),
            patch.object(unwrapped, "_get_gripper_contact_flags", return_value=(True, True)),
        ):
            unwrapped.get_reward()

        self.assertFalse(unwrapped.is_success)
        self.assertEqual(unwrapped.success_stable_count, 0)

        with (
            patch.object(unwrapped, "_calculate_distances", return_value=metrics),
            patch.object(unwrapped, "_get_gripper_contact_flags", return_value=(True, False)),
        ):
            for _ in range(unwrapped.success_stable_steps - 1):
                unwrapped.get_reward()
                self.assertFalse(unwrapped.is_success)
            final_reward = unwrapped.get_reward()

        self.assertTrue(unwrapped.is_success)
        self.assertTrue(unwrapped.terminated)
        self.assertIn("task_success", unwrapped.reward_debug["reward_terms"])
        self.assertGreater(final_reward, 400.0)


if __name__ == "__main__":
    unittest.main()
