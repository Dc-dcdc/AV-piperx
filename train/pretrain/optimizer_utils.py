"""预训练优化器的参数分组辅助函数。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar


ParameterT = TypeVar("ParameterT")


_COUPLING_MODULE_NAMES = frozenset(
    {
        "view_to_arm_attention",
        "arm_to_view_attention",
        "arm_coupling_norm",
        "view_coupling_norm",
        "coupling_timestep_encoder",
        "role_adaln_coupling",
    }
)


def is_coupling_parameter(parameter_name: str) -> bool:
    """识别交叉注意力、AdaLN、FFN及其门控/时间编码参数。"""
    return not _COUPLING_MODULE_NAMES.isdisjoint(parameter_name.split("."))


def partition_optimizer_parameters(
    named_parameters: Iterable[tuple[str, ParameterT]],
    *,
    is_backbone_parameter: Callable[[str], bool],
) -> tuple[list[ParameterT], list[ParameterT], list[ParameterT]]:
    """把可训练参数无遗漏、无重复地拆成主网络、耦合分支和视觉Backbone。"""
    named_parameters = list(named_parameters)
    main_parameters = [
        parameter
        for name, parameter in named_parameters
        if not is_backbone_parameter(name) and not is_coupling_parameter(name)
    ]
    coupling_parameters = [
        parameter
        for name, parameter in named_parameters
        if is_coupling_parameter(name)
    ]
    backbone_parameters = [
        parameter
        for name, parameter in named_parameters
        if is_backbone_parameter(name) and not is_coupling_parameter(name)
    ]

    grouped_parameters = [
        *main_parameters,
        *coupling_parameters,
        *backbone_parameters,
    ]
    grouped_parameter_ids = [id(parameter) for parameter in grouped_parameters]
    expected_parameter_ids = [id(parameter) for _, parameter in named_parameters]
    if len(grouped_parameter_ids) != len(set(grouped_parameter_ids)):
        raise RuntimeError("优化器参数分组存在重复，请检查参数分类规则。")
    if set(grouped_parameter_ids) != set(expected_parameter_ids):
        raise RuntimeError("优化器参数分组存在遗漏，请检查参数分类规则。")

    return main_parameters, coupling_parameters, backbone_parameters
