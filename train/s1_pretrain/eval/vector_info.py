"""统一解析 Gymnasium VectorEnv 返回的 ``info``。

新版 Gymnasium 使用“字段到数组”的字典，并通过 ``_字段名`` 掩码标记有效值；
旧版则可能返回字典列表或 object 数组。集中兼容两种格式，避免各评估入口重复实现。
"""

from collections.abc import Mapping
from typing import Any

import numpy as np


def as_bool_array(value: Any, n_envs: int, default: bool = False) -> np.ndarray:
    """把标量或数组规范为长度固定的布尔数组。"""
    # 缺失字段用调用方指定的默认值填充全部环境。
    if value is None:
        return np.full(n_envs, default, dtype=bool)

    array = np.asarray(value)
    # 标量需要广播到每个并行环境。
    if array.shape == ():
        return np.full(n_envs, bool(array), dtype=bool)

    array = array.astype(bool).reshape(-1)
    # 输入过短时补默认值；过长时在函数末尾截断，保证输出长度恒为 n_envs。
    if array.shape[0] < n_envs:
        padded = np.full(n_envs, default, dtype=bool)
        padded[: array.shape[0]] = array
        return padded
    return array[:n_envs]


def extract_info_bool(
    info: Any,
    key: str,
    n_envs: int,
    default: bool = False,
) -> np.ndarray:
    """从当前步和终止步的 VectorEnv info 中提取布尔字段。

    同时支持新版 dict-of-arrays ``final_info`` 和旧版字典列表/object 数组。
    ``_final_info`` 与 ``_<key>`` 等有效性掩码会被严格应用，避免占位值覆盖
    对应环境的真实结果。
    """
    values = np.full(n_envs, default, dtype=bool)
    if not isinstance(info, Mapping):
        return values

    def merge(raw_value: Any, raw_mask: Any) -> None:
        # 只合并掩码为 True 的位置，False 位置继续保留已有值。
        parsed = as_bool_array(raw_value, n_envs, default=default)
        mask = as_bool_array(raw_mask, n_envs, default=True)
        values[mask] = parsed[mask]

    # 先读取顶层当前步字段；终止环境随后可由 final_info 覆盖。
    if key in info:
        merge(
            info[key],
            info.get(f"_{key}", np.ones(n_envs, dtype=bool)),
        )

    final_infos = info.get("final_info")
    if final_infos is None:
        return values

    # _final_info 指出哪些并行环境在本步确实产生了终止信息。
    final_mask = as_bool_array(
        info.get("_final_info", np.ones(n_envs, dtype=bool)),
        n_envs,
        default=True,
    )

    # 新版格式：final_info 本身仍是“字段到数组”的字典。
    if isinstance(final_infos, Mapping):
        if key in final_infos:
            key_mask = as_bool_array(
                final_infos.get(f"_{key}", np.ones(n_envs, dtype=bool)),
                n_envs,
                default=True,
            )
            merge(final_infos[key], final_mask & key_mask)
        return values

    # 旧版格式：每个环境对应一个终止 info 字典，也可能是 None 占位。
    for index, final_info in enumerate(list(final_infos)[:n_envs]):
        if (
            final_mask[index]
            and isinstance(final_info, Mapping)
            and key in final_info
        ):
            values[index] = bool(final_info[key])

    return values
