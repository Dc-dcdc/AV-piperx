import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np

import env  # noqa: F401  注册guided_vision环境
from env.task.insert_peg_env import InsertPegEnv


class InsertPegEnvTest(unittest.TestCase):
    def make_env(self, **kwargs) -> InsertPegEnv:
        return InsertPegEnv(
            cameras=[],
            episode_length=10,
            **kwargs,
        )

    def test_registered_environment_uses_project_dimensions(self):
        environment = gym.make(
            "guided_vision/InsertPeg-3Arms-v0",
            disable_env_checker=True,
            cameras=[],
            episode_length=2,
        )
        try:
            observation, info = environment.reset(seed=7)
            self.assertEqual(observation["agent_pos"].shape, (20,))
            self.assertEqual(environment.action_space.shape, (20,))
            self.assertEqual(observation["pixels"], {})
            self.assertFalse(info["is_success"])

            action = observation["agent_pos"].astype(np.float32)
            _, reward, terminated, truncated, step_info = (
                environment.step(action)
            )
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertFalse(step_info["is_success"])
        finally:
            environment.close()

    def test_reset_seed_reproduces_both_object_poses(self):
        environment = self.make_env()
        try:
            _, first_info = environment.reset(seed=1234)
            first_peg_qpos = np.asarray(
                environment._physics.bind(environment._peg_joint).qpos
            ).copy()
            first_hole_qpos = np.asarray(
                environment._physics.bind(environment._hole_joint).qpos
            ).copy()
            environment._physics.bind(environment._peg_joint).qvel[:] = 1
            environment._physics.bind(environment._hole_joint).qvel[:] = 1

            _, second_info = environment.reset(seed=1234)
            second_peg_qpos = np.asarray(
                environment._physics.bind(environment._peg_joint).qpos
            ).copy()
            second_hole_qpos = np.asarray(
                environment._physics.bind(environment._hole_joint).qpos
            ).copy()
            peg_qvel = np.asarray(
                environment._physics.bind(environment._peg_joint).qvel
            ).copy()
            hole_qvel = np.asarray(
                environment._physics.bind(environment._hole_joint).qvel
            ).copy()
        finally:
            environment.close()

        np.testing.assert_array_equal(
            first_info["peg_position"],
            second_info["peg_position"],
        )
        np.testing.assert_array_equal(
            first_info["hole_position"],
            second_info["hole_position"],
        )
        np.testing.assert_array_equal(first_peg_qpos, second_peg_qpos)
        np.testing.assert_array_equal(first_hole_qpos, second_hole_qpos)
        np.testing.assert_array_equal(peg_qvel, np.zeros(6))
        np.testing.assert_array_equal(hole_qvel, np.zeros(6))

    def test_xy_insertion_succeeds_after_five_release_steps(self):
        environment = self.make_env(enable_reward_debug=True)
        try:
            environment.reset(seed=5)
            inserted_position = np.array([0.0, 0.0, 0.10])
            environment._physics.bind(
                environment._hole_joint
            ).qpos = np.concatenate(
                (
                    inserted_position,
                    np.array([1.0, 0.0, 0.0, 0.0]),
                )
            )
            # Peg绕Z轴旋转90度且具有较大速度，X/Y位置达标仍应成功。
            environment._physics.bind(
                environment._peg_joint
            ).qpos = np.concatenate(
                (
                    inserted_position,
                    np.array(
                        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
                    ),
                )
            )
            environment._physics.bind(
                environment._peg_joint
            ).qvel = np.full(6, 10.0)
            environment._physics.bind(
                environment._hole_joint
            ).qvel = np.full(6, -10.0)
            environment._physics.forward()

            # 模拟右手此前已经抓住过Peg，随后完成释放。
            environment.right_has_grasped = True
            rewards = [
                environment.get_reward()
                for _ in range(
                    environment.success_release_stable_steps
                )
            ]
            metrics = environment._calculate_metrics()
        finally:
            environment.close()

        self.assertTrue(metrics["inserted"])
        self.assertNotIn(4.0, rewards[:-1])
        self.assertEqual(rewards[-1], 4.0)
        self.assertTrue(environment.is_success)
        self.assertTrue(environment.terminated)
        self.assertEqual(environment.reward_debug["stage"], 4)
        self.assertEqual(
            environment.reward_debug["release_stable_count"],
            5,
        )

    def test_position_ranges_reject_reversed_bounds(self):
        with self.assertRaisesRegex(ValueError, "min<=max"):
            self.make_env(
                peg_position_ranges=[
                    [0.20, 0.10],
                    [-0.10, 0.10],
                    [0.01, 0.01],
                ]
            )

    def test_right_recontact_resets_release_counter(self):
        environment = self.make_env()
        try:
            environment.reset(seed=9)
            environment.right_has_grasped = True
            environment._release_stable_count = 4
            contact_flags = {
                "left_hole": True,
                "right_peg": True,
                "peg_table": False,
                "hole_table": False,
                "peg_hole": True,
            }
            with patch.object(
                environment,
                "_get_contact_flags",
                return_value=contact_flags,
            ):
                environment.get_reward()
        finally:
            environment.close()

        self.assertEqual(environment._release_stable_count, 0)
        self.assertFalse(environment.is_success)

    def test_no_contact_but_insufficient_clearance_cannot_succeed(self):
        environment = self.make_env()
        try:
            environment.reset(seed=10)
            environment.right_has_grasped = True
            environment._release_stable_count = 4
            contact_flags = {
                "left_hole": True,
                "right_peg": False,
                "peg_table": False,
                "hole_table": False,
                "peg_hole": True,
            }
            metrics = {
                "peg_pos": np.array([0.0, 0.0, 0.10]),
                "hole_pos": np.array([0.0, 0.0, 0.10]),
                "right_gripper_pos": np.array([0.0, 0.0, 0.10]),
                "right_to_peg_distance": (
                    environment.success_right_release_distance - 0.001
                ),
                "x_error": 0.0,
                "y_error": 0.0,
                "inserted": True,
            }
            with (
                patch.object(
                    environment,
                    "_get_contact_flags",
                    return_value=contact_flags,
                ),
                patch.object(
                    environment,
                    "_calculate_metrics",
                    return_value=metrics,
                ),
            ):
                environment.get_reward()
        finally:
            environment.close()

        self.assertEqual(environment._release_stable_count, 0)
        self.assertFalse(environment.is_success)


if __name__ == "__main__":
    unittest.main()
