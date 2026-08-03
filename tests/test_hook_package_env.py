import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np

import env  # noqa: F401  注册guided_vision环境
from env.task.hook_package_env import HookPackageEnv


class HookPackageEnvTest(unittest.TestCase):
    def make_env(self, **kwargs) -> HookPackageEnv:
        return HookPackageEnv(
            cameras=[],
            episode_length=10,
            success_stable_steps=3,
            **kwargs,
        )

    def test_registered_environment_has_project_action_observation_shapes(self):
        environment = gym.make(
            "guided_vision/HookPackage-3Arms-v0",
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

    def test_reset_seed_reproduces_hook_and_package_initial_state(self):
        environment = self.make_env()
        try:
            _, first_info = environment.reset(seed=1234)
            first_package_qpos = np.asarray(
                environment._physics.bind(
                    environment._package_joint
                ).qpos
            ).copy()
            _, second_info = environment.reset(seed=1234)
            second_package_qpos = np.asarray(
                environment._physics.bind(
                    environment._package_joint
                ).qpos
            ).copy()
        finally:
            environment.close()

        np.testing.assert_allclose(
            first_info["hook_position"],
            second_info["hook_position"],
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            first_package_qpos,
            second_package_qpos,
            rtol=0,
            atol=0,
        )

    def test_success_requires_stable_geometric_hook_engagement(self):
        environment = self.make_env(enable_reward_debug=True)
        try:
            environment.reset(seed=5)
            hook_center = environment._physics.named.data.geom_xpos[
                "pin-hook"
            ].copy()
            # identity姿态下，pin-package相对package body位于[0, 0, 0.11]。
            package_position = hook_center - np.array([0.0, 0.0, 0.11])
            environment._physics.bind(
                environment._package_joint
            ).qpos = np.concatenate(
                (package_position, np.array([1.0, 0.0, 0.0, 0.0]))
            )
            environment._physics.bind(
                environment._package_joint
            ).qvel = np.zeros(6)
            environment._physics.forward()

            rewards = [
                environment.get_reward()
                for _ in range(environment.success_stable_steps)
            ]
            metrics = environment._calculate_metrics()
        finally:
            environment.close()

        self.assertTrue(metrics["hook_engaged"])
        self.assertEqual(rewards[:-1], [0.0, 0.0])
        self.assertEqual(rewards[-1], 4.0)
        self.assertTrue(environment.is_success)
        self.assertTrue(environment.terminated)
        self.assertEqual(environment.reward_debug["stage"], 4)

    def test_static_hook_is_declared_for_exact_trajectory_replay(self):
        environment = self.make_env()
        try:
            self.assertEqual(
                environment.replay_model_body_names,
                ("hook",),
            )
            self.assertIsNone(
                environment._mjcf_root.find("joint", "hook_joint")
            )
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
