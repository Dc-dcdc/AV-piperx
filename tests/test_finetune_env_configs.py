import os
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir


class FinetuneEnvConfigTest(unittest.TestCase):
    def test_all_policies_compose_with_each_task_environment(self):
        config_root = Path(__file__).resolve().parents[1] / "configs" / "finetune"
        policies = sorted(path.stem for path in (config_root / "policy").glob("*.yaml"))
        environments = {
            "sim_insert_cylinder_3arms": (
                "InsertCylinder-3Arms-v0",
                "av_piper_insert_cylinder_3arms_online_rollout",
            ),
            "sim_sew_needle_3arms": (
                "SewNeedle-3Arms-v0",
                "av_piper_sew_needle_3arms_online_rollout",
            ),
        }

        with initialize_config_dir(
            config_dir=os.fspath(config_root),
            version_base=None,
        ):
            for policy in policies:
                for env_name, (task, dataset_repo_id) in environments.items():
                    with self.subTest(policy=policy, env=env_name):
                        cfg = compose(
                            config_name="ft_default",
                            overrides=[f"env={env_name}", f"policy={policy}"],
                        )
                        self.assertEqual(cfg.env.name, "guided_vision")
                        self.assertEqual(cfg.env.task, task)
                        self.assertEqual(cfg.env.state_dim, 20)
                        self.assertEqual(cfg.env.action_dim, 20)
                        self.assertEqual(cfg.env.fps, 25)
                        self.assertEqual(cfg.env.episode_length, 400)
                        self.assertEqual(cfg.eval.max_steps, 400)
                        self.assertGreater(int(cfg.env.n_envs), 0)
                        self.assertEqual(cfg.dataset_repo_id, dataset_repo_id)


if __name__ == "__main__":
    unittest.main()
