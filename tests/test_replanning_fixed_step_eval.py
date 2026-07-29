"""固定推理执行步长评估中纯函数和报告生成的测试。"""

import tempfile
import unittest
from pathlib import Path

from train.s4_adaptive_replanning.eval_fixed_steps import (
    build_markdown_report,
    choose_recommended_summary,
    default_execution_lengths,
    extract_episode_success,
    normalize_model_input_path,
    summarize_episodes,
    validate_execution_lengths,
    write_evaluation_report,
)


class FixedExecutionStepEvaluationTest(unittest.TestCase):
    """验证步长解析、成功标记、指标聚合和报告输出。"""

    def test_default_lengths_include_actual_horizon(self):
        """验证默认使用2的幂，且非2的幂horizon也不会丢失。"""
        self.assertEqual(default_execution_lengths(16), [1, 2, 4, 8, 16])
        self.assertEqual(default_execution_lengths(12), [1, 2, 4, 8, 12])

    def test_custom_lengths_are_unique_sorted_and_bounded(self):
        """验证自定义步长去重排序，并拒绝超过horizon的数值。"""
        self.assertEqual(validate_execution_lengths([8, 1, 2, 2], 16), [1, 2, 8])
        with self.assertRaises(ValueError):
            validate_execution_lengths([1, 17], 16)

    def test_model_file_path_is_normalized_to_model_directory(self):
        """验证直接传入model.safetensors时能自动转换为父目录。"""
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "model.safetensors"
            model_file.touch()
            self.assertEqual(normalize_model_input_path(model_file), model_file.parent)

    def test_success_extraction_supports_final_info(self):
        """验证成功标记可以从Gymnasium final_info列表中读取。"""
        info = {"final_info": [None, {"is_success": True}]}
        self.assertTrue(extract_episode_success(info))
        self.assertFalse(extract_episode_success({"success": False}))

    def test_summary_computes_success_return_and_inference_efficiency(self):
        """验证聚合结果包含成功率、回报和实际推理效率。"""
        records = [
            {
                "success": True,
                "return": 10.0,
                "env_steps": 8,
                "inference_count": 2,
                "average_inference_ms": 5.0,
                "total_inference_ms": 10.0,
            },
            {
                "success": False,
                "return": 6.0,
                "env_steps": 4,
                "inference_count": 1,
                "average_inference_ms": 7.0,
                "total_inference_ms": 7.0,
            },
        ]
        summary = summarize_episodes(4, records)

        self.assertEqual(summary["success_count"], 1)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["average_return"], 8.0)
        self.assertAlmostEqual(summary["average_effective_steps_per_inference"], 4.0)
        self.assertAlmostEqual(summary["average_inference_ms"], 6.0)

    def test_recommendation_prioritizes_success_then_return(self):
        """验证自动推荐首先保留成功率更高的执行步长。"""
        summaries = [
            {
                "execution_length": 2,
                "success_rate": 0.8,
                "average_return": 20.0,
                "average_inference_count": 10.0,
            },
            {
                "execution_length": 8,
                "success_rate": 0.9,
                "average_return": 10.0,
                "average_inference_count": 3.0,
            },
        ]
        self.assertEqual(choose_recommended_summary(summaries)["execution_length"], 8)

    def test_report_writer_creates_all_formats(self):
        """验证一次写入会生成JSON、两份CSV和Markdown报告。"""
        summary = {
            "execution_length": 4,
            "n_episodes": 1,
            "success_count": 1,
            "success_rate": 1.0,
            "success_rate_percent": 100.0,
            "success_ci95_low": 0.2,
            "success_ci95_high": 1.0,
            "average_return": 10.0,
            "return_std": 0.0,
            "return_ci95_low": 10.0,
            "return_ci95_high": 10.0,
            "average_episode_steps": 4.0,
            "average_inference_count": 1.0,
            "average_effective_steps_per_inference": 4.0,
            "average_inference_ms": 5.0,
            "average_total_inference_ms": 5.0,
            "inferences_per_100_env_steps": 25.0,
            "total_env_steps": 4,
        }
        episode = {
            "execution_length": 4,
            "episode_index": 0,
            "seed": 100,
            "success": True,
            "return": 10.0,
            "env_steps": 4,
            "inference_count": 1,
            "effective_steps_per_inference": 4.0,
            "average_inference_ms": 5.0,
            "total_inference_ms": 5.0,
            "wall_time_seconds": 0.1,
        }
        report = {
            "metadata": {
                "model_dir": "/tmp/model",
                "env_id": "guided_vision/Test-v0",
                "horizon": 16,
                "n_episodes": 1,
                "max_episode_steps": 400,
                "seed": 100,
                "device": "cpu",
                "status": "completed",
            },
            "summaries": [summary],
            "episodes": [episode],
            "recommendation": summary,
        }

        with tempfile.TemporaryDirectory() as directory:
            paths = write_evaluation_report(Path(directory), report)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            markdown = build_markdown_report(report)
            self.assertIn("推荐固定执行步长：**4**", markdown)


if __name__ == "__main__":
    unittest.main()
