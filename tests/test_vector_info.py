import unittest

import numpy as np

from train.pretrain.vector_info import as_bool_array, extract_info_bool


class AsBoolArrayTest(unittest.TestCase):
    def test_scalar_and_short_inputs_are_normalized(self):
        """验证标量广播以及短数组按默认值补齐。"""
        np.testing.assert_array_equal(
            as_bool_array(True, n_envs=3),
            np.array([True, True, True]),
        )
        np.testing.assert_array_equal(
            as_bool_array([True], n_envs=3, default=False),
            np.array([True, False, False]),
        )


class ExtractInfoBoolTest(unittest.TestCase):
    def test_new_dict_of_arrays_format_honors_both_masks(self):
        """验证新版 dict-of-arrays 同时应用字段和 final_info 掩码。"""
        info = {
            "is_success": np.array([False, False, True, True]),
            "_is_success": np.array([True, True, False, False]),
            "final_info": {
                "is_success": np.array([True, True, False, True]),
                "_is_success": np.array([True, False, True, True]),
            },
            "_final_info": np.array([True, True, True, False]),
        }

        np.testing.assert_array_equal(
            extract_info_bool(info, "is_success", n_envs=4),
            np.array([True, False, False, False]),
        )

    def test_legacy_list_of_dicts_format_is_supported(self):
        """验证兼容旧版 final_info 字典列表及其中的 None 占位。"""
        info = {
            "final_info": [
                {"is_success": True},
                None,
                {"is_success": False},
            ],
            "_final_info": np.array([True, False, True]),
        }

        np.testing.assert_array_equal(
            extract_info_bool(info, "is_success", n_envs=3),
            np.array([True, False, False]),
        )

    def test_top_level_mask_rejects_placeholder_values(self):
        """验证顶层字段掩码会忽略无效占位值。"""
        info = {
            "is_success": np.array([False, True, True]),
            "_is_success": np.array([True, False, True]),
        }

        np.testing.assert_array_equal(
            extract_info_bool(info, "is_success", n_envs=3),
            np.array([False, False, True]),
        )

    def test_invalid_info_returns_requested_default(self):
        """验证 info 无效时为所有环境返回调用方指定的默认值。"""
        np.testing.assert_array_equal(
            extract_info_bool(None, "is_success", n_envs=2, default=True),
            np.array([True, True]),
        )


if __name__ == "__main__":
    unittest.main()
