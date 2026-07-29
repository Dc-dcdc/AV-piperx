#!/usr/bin/env python
"""冻结双扩散头基线、只训练瓶颈耦合模块的独立实验入口。

该入口复用 ``train_pretrain.py`` 的数据、评估、checkpoint 与 W&B 流程，但不
修改原训练脚本。首次训练使用 ``init_policy_path`` 将普通双头 checkpoint
严格迁移到 Coupled 模型；断点续训使用 ``resume_path`` 恢复本实验的完整状态。
"""

from __future__ import annotations

import logging
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from lerobot.common.policies.diffusion.modeling_coupled_dual_head_diffusion import (
    CoupledDualHeadDiffusionPolicy,
)
from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DualHeadDiffusionPolicy,
)
from lerobot.common.policies.factory import (
    _policy_cfg_from_hydra_cfg,
    get_policy_and_config_classes,
)
from lerobot.common.utils.utils import get_safe_torch_device
from train.s1_pretrain.train import train_pretrain as base_train
from train.s1_pretrain.train.optimizer_utils import is_coupling_parameter


_ORIGINAL_MAKE_POLICY = base_train.make_policy
_ORIGINAL_UPDATE_POLICY = base_train.update_policy

_BASELINE_COMPATIBILITY_FIELDS = (
    "n_obs_steps",
    "horizon",
    "n_action_steps",
    "input_shapes",
    "output_shapes",
    "input_normalization_modes",
    "output_normalization_modes",
    "vision_backbone",
    "resize_shape",
    "crop_shape",
    "crop_is_random",
    "pretrained_backbone_weights",
    "use_group_norm",
    "spatial_softmax_num_keypoints",
    "down_dims",
    "kernel_size",
    "n_groups",
    "diffusion_step_embed_dim",
    "use_film_scale_modulation",
    "noise_scheduler_type",
    "num_train_timesteps",
    "beta_schedule",
    "beta_start",
    "beta_end",
    "prediction_type",
    "clip_sample",
    "clip_sample_range",
    "num_inference_steps",
    "do_mask_loss_for_padding",
    "arm_action_dim",
    "view_action_dim",
    "view_loss_weight",
)


def _plain_config_value(value: Any) -> Any:
    """把配置值规范化成可稳定比较的普通Python对象。"""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _plain_config_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain_config_value(child) for child in value)
    return value


def validate_incremental_config(cfg: DictConfig) -> None:
    """拒绝会破坏冻结增量实验语义的配置组合。"""
    if str(cfg.policy.name) != "coupled_dual_head_diffusion":
        raise ValueError(
            "增量入口只支持 policy.name='coupled_dual_head_diffusion'，当前为"
            f"{cfg.policy.name!r}。"
        )

    view_to_arm = float(cfg.policy.view_to_arm_coupling_scale)
    arm_to_view = float(cfg.policy.arm_to_view_coupling_scale)
    if view_to_arm <= 0.0 and arm_to_view <= 0.0:
        raise ValueError(
            "至少启用一个耦合方向；两个 coupling scale 同时为0时没有可测试的增量模块。"
        )

    if (
        bool(cfg.policy.coupling_use_ffn)
        and str(cfg.policy.coupling_block_type) != "role_adaln_zero"
    ):
        raise ValueError(
            "coupling_use_ffn=true 只允许与 coupling_block_type=role_adaln_zero 配合。"
        )

    if not bool(cfg.resume) and base_train.clean_optional_path(
        cfg.get("init_policy_path", None)
    ) is None:
        raise ValueError(
            "首次增量训练必须提供普通双扩散头 checkpoint 的 init_policy_path。"
        )

    coupling_lr = float(
        OmegaConf.select(cfg, "training.coupling_lr", default=cfg.training.lr)
    )
    if not math.isfinite(coupling_lr) or coupling_lr <= 0:
        raise ValueError(f"training.coupling_lr必须是有限正数，当前为{coupling_lr}。")


def validate_baseline_config_compatibility(source_config, target_config) -> None:
    """确保除Coupling以外的模型与训练目标均和基线checkpoint一致。"""
    mismatches = []
    for field_name in _BASELINE_COMPATIBILITY_FIELDS:
        source_value = _plain_config_value(getattr(source_config, field_name))
        target_value = _plain_config_value(getattr(target_config, field_name))
        if source_value != target_value:
            mismatches.append(
                f"{field_name}: checkpoint={source_value!r}, target={target_value!r}"
            )
    if mismatches:
        raise ValueError(
            "目标Coupled配置与双头基线不兼容；冻结增量实验不允许同时改变以下字段:\n"
            + "\n".join(f"  - {item}" for item in mismatches)
        )


def migrate_dual_state_into_coupled(
    source_policy: DualHeadDiffusionPolicy,
    target_policy: CoupledDualHeadDiffusionPolicy,
) -> dict[str, int]:
    """复制全部基线状态，并严格限定缺失项只能来自新增Coupling模块。"""
    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()

    unexpected_source_keys = sorted(set(source_state).difference(target_state))
    if unexpected_source_keys:
        raise RuntimeError(
            "双头checkpoint包含Coupled模型无法识别的状态:\n"
            + "\n".join(f"  - {key}" for key in unexpected_source_keys)
        )

    shape_mismatches = []
    for key, source_tensor in source_state.items():
        target_tensor = target_state[key]
        if source_tensor.shape != target_tensor.shape:
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(source_tensor.shape)}, "
                f"target={tuple(target_tensor.shape)}"
            )
    if shape_mismatches:
        raise RuntimeError(
            "双头checkpoint与Coupled模型存在形状不一致:\n"
            + "\n".join(f"  - {item}" for item in shape_mismatches)
        )

    incompatible = target_policy.load_state_dict(source_state, strict=False)
    expected_missing = {
        key for key in target_state if is_coupling_parameter(key)
    }
    actual_missing = set(incompatible.missing_keys)
    if actual_missing != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "不安全的dual→coupled迁移: "
            f"missing={sorted(actual_missing)}, "
            f"expected_missing={sorted(expected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )

    copied_state = target_policy.state_dict()
    unequal_keys = [
        key
        for key, source_tensor in source_state.items()
        if not torch.equal(source_tensor, copied_state[key])
    ]
    if unequal_keys:
        raise RuntimeError(
            "共有状态复制后未保持逐元素一致:\n"
            + "\n".join(f"  - {key}" for key in unequal_keys)
        )

    return {
        "shared_tensor_count": len(source_state),
        "new_coupling_tensor_count": len(expected_missing),
    }


def _global_condition_dim(policy: CoupledDualHeadDiffusionPolicy) -> int:
    config = policy.config
    dimension = int(config.input_shapes["observation.state"][0])
    image_keys = [
        key for key in config.input_shapes if key.startswith("observation.image")
    ]
    if image_keys:
        dimension += int(policy.diffusion.rgb_encoder.feature_dim) * len(image_keys)
    if "observation.environment_state" in config.input_shapes:
        dimension += int(config.input_shapes["observation.environment_state"][0])
    return dimension * int(config.n_obs_steps)


def run_zero_gate_equivalence_check(
    source_policy: DualHeadDiffusionPolicy,
    target_policy: CoupledDualHeadDiffusionPolicy,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    """比较原UNet前向与Coupled零门控拆分前向，失败时阻止训练。"""
    if atol < 0 or rtol < 0:
        raise ValueError("等价性检查的atol/rtol必须非负。")

    source_was_training = source_policy.training
    target_was_training = target_policy.training
    source_policy.eval()
    target_policy.eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260727)
    config = target_policy.config
    arm_sample = torch.randn(
        1,
        int(config.horizon),
        int(target_policy.diffusion.arm_action_dim),
        generator=generator,
    )
    view_sample = torch.randn(
        1,
        int(config.horizon),
        int(target_policy.diffusion.view_action_dim),
        generator=generator,
    )
    global_cond = torch.randn(
        1,
        _global_condition_dim(target_policy),
        generator=generator,
    )
    timesteps = torch.tensor(
        [int(config.num_train_timesteps) - 1],
        dtype=torch.long,
    )

    with torch.inference_mode():
        expected_arm = source_policy.diffusion.arm_unet(
            arm_sample, timesteps, global_cond=global_cond
        )
        expected_view = source_policy.diffusion.view_unet(
            view_sample, timesteps, global_cond=global_cond
        )
        actual_arm, actual_view = target_policy.diffusion.predict_noise(
            arm_sample,
            view_sample,
            timesteps,
            global_cond,
        )

    arm_max_abs_error = float((actual_arm - expected_arm).abs().max())
    view_max_abs_error = float((actual_view - expected_view).abs().max())
    arm_close = torch.allclose(actual_arm, expected_arm, atol=atol, rtol=rtol)
    view_close = torch.allclose(actual_view, expected_view, atol=atol, rtol=rtol)

    source_policy.train(source_was_training)
    target_policy.train(target_was_training)
    if not arm_close or not view_close:
        raise RuntimeError(
            "零门控等价性检查失败，拒绝开始增量训练: "
            f"arm_max_abs_error={arm_max_abs_error:.3e}, "
            f"view_max_abs_error={view_max_abs_error:.3e}, "
            f"atol={atol:.3e}, rtol={rtol:.3e}"
        )

    return {
        "arm_max_abs_error": arm_max_abs_error,
        "view_max_abs_error": view_max_abs_error,
        "max_abs_error": max(arm_max_abs_error, view_max_abs_error),
    }


def _enable_module_parameters(module: nn.Module | None) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def configure_incremental_trainable_parameters(
    policy: CoupledDualHeadDiffusionPolicy,
) -> dict[str, Any]:
    """冻结基线并只开放当前配置实际使用的Coupling参数。"""
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    diffusion = policy.diffusion
    view_to_arm_enabled = float(diffusion.view_to_arm_coupling_scale) > 0.0
    arm_to_view_enabled = float(diffusion.arm_to_view_coupling_scale) > 0.0

    if view_to_arm_enabled:
        _enable_module_parameters(diffusion.view_to_arm_attention)
    if arm_to_view_enabled:
        _enable_module_parameters(diffusion.arm_to_view_attention)

    # 两个方向的标量gate共享前面的时间编码层；关闭方向对应的输出行不会收到梯度。
    _enable_module_parameters(diffusion.coupling_timestep_encoder)

    if diffusion.coupling_block_type == "scalar_gate":
        # 单向Attention仍同时需要Query角色和Key/Value角色的归一化。
        _enable_module_parameters(diffusion.arm_coupling_norm)
        _enable_module_parameters(diffusion.view_coupling_norm)
    elif diffusion.coupling_block_type == "role_adaln_zero":
        role_block = diffusion.role_adaln_coupling
        if role_block is None:
            raise RuntimeError("role_adaln_zero配置缺少role_adaln_coupling模块。")
        _enable_module_parameters(role_block.timestep_encoder)
        _enable_module_parameters(role_block.attention_modulation)
        if diffusion.coupling_use_ffn:
            _enable_module_parameters(role_block.ffn_modulation)
            if view_to_arm_enabled:
                _enable_module_parameters(role_block.arm_ffn)
            if arm_to_view_enabled:
                _enable_module_parameters(role_block.view_ffn)
    else:
        raise ValueError(f"未知coupling_block_type={diffusion.coupling_block_type!r}")

    trainable = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    invalid_names = [
        name for name, _ in trainable if not is_coupling_parameter(name)
    ]
    if invalid_names:
        raise RuntimeError(
            "严格白名单失败，发现非Coupling可训练参数:\n"
            + "\n".join(f"  - {name}" for name in invalid_names)
        )
    if not trainable:
        raise RuntimeError("冻结后没有任何可训练参数。")

    frozen_parameter_count = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if not parameter.requires_grad
    )
    trainable_parameter_count = sum(
        parameter.numel() for _, parameter in trainable
    )
    return {
        "view_to_arm_enabled": view_to_arm_enabled,
        "arm_to_view_enabled": arm_to_view_enabled,
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": frozen_parameter_count,
        "trainable_names": [name for name, _ in trainable],
    }


def install_frozen_module_mode_guard(
    policy: CoupledDualHeadDiffusionPolicy,
) -> None:
    """让冻结模块在外层反复调用policy.train()后仍保持eval模式。"""
    frozen_modules = [
        policy.diffusion.arm_unet,
        policy.diffusion.view_unet,
    ]
    if getattr(policy.diffusion, "_use_images", False):
        frozen_modules.append(policy.diffusion.rgb_encoder)

    original_train = policy.train

    def guarded_train(self, mode: bool = True):
        result = original_train(mode)
        if mode:
            for module in frozen_modules:
                module.eval()
        return result

    policy.train = types.MethodType(guarded_train, policy)
    policy.train(True)


def _parameter_diagnostic_group(parameter_name: str) -> str:
    if ".view_to_arm_attention." in parameter_name:
        return "grad_norm/view_to_arm_attention"
    if ".arm_to_view_attention." in parameter_name:
        return "grad_norm/arm_to_view_attention"
    if ".coupling_timestep_encoder." in parameter_name:
        return "grad_norm/attention_gate"
    if ".arm_ffn." in parameter_name:
        return "grad_norm/arm_ffn"
    if ".view_ffn." in parameter_name:
        return "grad_norm/view_ffn"
    if ".ffn_modulation." in parameter_name:
        return "grad_norm/ffn_modulation"
    if ".role_adaln_coupling." in parameter_name:
        return "grad_norm/role_adaln"
    if "coupling_norm" in parameter_name:
        return "grad_norm/coupling_layernorm"
    return "grad_norm/other_coupling"


class CouplingDiagnostics:
    """收集增量实验所需的gate、残差和分模块梯度诊断。"""

    def __init__(self, policy: CoupledDualHeadDiffusionPolicy):
        self.policy = policy
        self.enabled = False
        self.forward_values: dict[str, Tensor] = {}
        self.gradient_squares: dict[str, Tensor] = {}
        self._install_parameter_hooks()
        self._install_forward_hooks()

    @staticmethod
    def _rms(tensor: Tensor) -> Tensor:
        return tensor.detach().float().square().mean().sqrt()

    def _install_parameter_hooks(self) -> None:
        for parameter_name, parameter in self.policy.named_parameters():
            if not parameter.requires_grad:
                continue
            group_name = _parameter_diagnostic_group(parameter_name)

            def capture_gradient(
                gradient: Tensor,
                *,
                diagnostic_group: str = group_name,
            ) -> Tensor:
                if self.enabled:
                    square_sum = gradient.detach().float().square().sum()
                    previous = self.gradient_squares.get(diagnostic_group)
                    self.gradient_squares[diagnostic_group] = (
                        square_sum if previous is None else previous + square_sum
                    )
                return gradient

            parameter.register_hook(capture_gradient)

    def _install_forward_hooks(self) -> None:
        diffusion = self.policy.diffusion

        def attention_hook(metric_name: str):
            def capture(_module, _inputs, output):
                if self.enabled:
                    attention_output = output[0] if isinstance(output, tuple) else output
                    self.forward_values[metric_name] = self._rms(attention_output)

            return capture

        if float(diffusion.view_to_arm_coupling_scale) > 0.0:
            diffusion.view_to_arm_attention.register_forward_hook(
                attention_hook("residual/raw_view_to_arm_attention_rms")
            )
        if float(diffusion.arm_to_view_coupling_scale) > 0.0:
            diffusion.arm_to_view_attention.register_forward_hook(
                attention_hook("residual/raw_arm_to_view_attention_rms")
            )

        original_couple = diffusion._couple_bottlenecks

        def monitored_couple(
            _diffusion,
            arm_bottleneck: Tensor,
            view_bottleneck: Tensor,
            timesteps: Tensor,
        ):
            coupled_arm, coupled_view = original_couple(
                arm_bottleneck, view_bottleneck, timesteps
            )
            if self.enabled:
                arm_residual = coupled_arm - arm_bottleneck
                view_residual = coupled_view - view_bottleneck
                arm_input_rms = self._rms(arm_bottleneck).clamp_min(1e-12)
                view_input_rms = self._rms(view_bottleneck).clamp_min(1e-12)
                arm_residual_rms = self._rms(arm_residual)
                view_residual_rms = self._rms(view_residual)
                self.forward_values["residual/effective_arm_rms"] = arm_residual_rms
                self.forward_values["residual/effective_view_rms"] = view_residual_rms
                self.forward_values["residual/arm_to_input_ratio"] = (
                    arm_residual_rms / arm_input_rms
                )
                self.forward_values["residual/view_to_input_ratio"] = (
                    view_residual_rms / view_input_rms
                )
            return coupled_arm, coupled_view

        diffusion._couple_bottlenecks = types.MethodType(
            monitored_couple,
            diffusion,
        )

        role_block = diffusion.role_adaln_coupling
        if diffusion.coupling_use_ffn and role_block is not None:
            original_ffn = role_block.compute_ffn_deltas

            def monitored_ffn(
                _role_block,
                arm_tokens: Tensor,
                view_tokens: Tensor,
                condition: Tensor,
            ):
                arm_delta, view_delta = original_ffn(
                    arm_tokens, view_tokens, condition
                )
                if self.enabled:
                    self.forward_values["residual/gated_arm_ffn_rms"] = self._rms(
                        arm_delta
                    )
                    self.forward_values["residual/gated_view_ffn_rms"] = self._rms(
                        view_delta
                    )
                return arm_delta, view_delta

            role_block.compute_ffn_deltas = types.MethodType(
                monitored_ffn,
                role_block,
            )

    def begin_step(self, enabled: bool) -> None:
        self.enabled = enabled
        self.forward_values = {}
        self.gradient_squares = {}

    def _gate_metrics(self) -> dict[str, float]:
        diffusion = self.policy.diffusion
        parameter = next(diffusion.coupling_timestep_encoder.parameters())
        timestep_count = int(diffusion.config.num_train_timesteps)
        timesteps = torch.arange(timestep_count, device=parameter.device)
        with torch.no_grad():
            gates = torch.tanh(diffusion.coupling_timestep_encoder(timesteps)).float()

        metrics: dict[str, float] = {}
        directions = (
            ("view_to_arm", 0, float(diffusion.view_to_arm_coupling_scale)),
            ("arm_to_view", 1, float(diffusion.arm_to_view_coupling_scale)),
        )
        for direction_name, column, scale in directions:
            values = gates[:, column]
            metrics[f"gate/{direction_name}_mean"] = float(values.mean())
            metrics[f"gate/{direction_name}_abs_mean"] = float(values.abs().mean())
            metrics[f"gate/{direction_name}_min"] = float(values.min())
            metrics[f"gate/{direction_name}_max"] = float(values.max())
            metrics[f"gate/{direction_name}_effective_abs_mean"] = float(
                values.abs().mean() * scale
            )
            sample_timesteps = sorted(
                {
                    0,
                    timestep_count // 4,
                    timestep_count // 2,
                    (3 * timestep_count) // 4,
                    timestep_count - 1,
                }
            )
            for timestep in sample_timesteps:
                metrics[f"gate/{direction_name}_t{timestep:03d}"] = float(
                    values[timestep]
                )
        return metrics

    def _role_adaln_metrics(self) -> dict[str, float]:
        diffusion = self.policy.diffusion
        role_block = diffusion.role_adaln_coupling
        if role_block is None:
            return {}

        parameter = next(role_block.parameters())
        timesteps = torch.arange(
            int(diffusion.config.num_train_timesteps),
            device=parameter.device,
        )
        with torch.no_grad():
            condition = role_block.timestep_encoder(timesteps)
            modulation = role_block.attention_modulation(condition)
            arm_shift, arm_scale, view_shift, view_scale = modulation.chunk(4, dim=-1)

        metrics = {
            "adaln/arm_shift_abs_mean": float(arm_shift.float().abs().mean()),
            "adaln/arm_scale_abs_mean": float(arm_scale.float().abs().mean()),
            "adaln/view_shift_abs_mean": float(view_shift.float().abs().mean()),
            "adaln/view_scale_abs_mean": float(view_scale.float().abs().mean()),
        }
        if diffusion.coupling_use_ffn:
            with torch.no_grad():
                ffn_modulation = role_block.ffn_modulation(condition)
                dim = role_block.bottleneck_dim
                pieces = torch.split(
                    ffn_modulation,
                    [dim, dim, 1, dim, dim, 1],
                    dim=-1,
                )
                arm_ffn_gate = pieces[2]
                view_ffn_gate = pieces[5]
            metrics.update(
                {
                    "ffn_gate/arm_mean": float(arm_ffn_gate.float().mean()),
                    "ffn_gate/arm_abs_mean": float(
                        arm_ffn_gate.float().abs().mean()
                    ),
                    "ffn_gate/view_mean": float(view_ffn_gate.float().mean()),
                    "ffn_gate/view_abs_mean": float(
                        view_ffn_gate.float().abs().mean()
                    ),
                }
            )
        return metrics

    def collect(self) -> dict[str, float | int]:
        if not self.enabled:
            return {}
        metrics = {
            f"coupling/{name}": float(value)
            for name, value in self.forward_values.items()
        }
        for group_name, square_sum in self.gradient_squares.items():
            metrics[f"coupling/{group_name}"] = float(square_sum.sqrt())
        metrics.update(
            {
                f"coupling/{name}": value
                for name, value in self._gate_metrics().items()
            }
        )
        metrics.update(
            {
                f"coupling/{name}": value
                for name, value in self._role_adaln_metrics().items()
            }
        )

        report = getattr(self.policy, "_incremental_init_report", {})
        equivalence = report.get("equivalence", {})
        if equivalence:
            metrics["coupling/init_equivalence_max_abs_error"] = float(
                equivalence["max_abs_error"]
            )
        scope = getattr(self.policy, "_incremental_scope_report", {})
        if scope:
            metrics["coupling/trainable_parameter_count"] = int(
                scope["trainable_parameter_count"]
            )
        return metrics


def _is_zero_decay_parameter(parameter_name: str, parameter: nn.Parameter) -> bool:
    """控制层、bias和归一化参数不做weight decay。"""
    if parameter.ndim <= 1 or parameter_name.endswith(".bias"):
        return True
    zero_initialized_control_outputs = (
        ".coupling_timestep_encoder.3.",
        ".role_adaln_coupling.attention_modulation.1.",
        ".role_adaln_coupling.ffn_modulation.1.",
    )
    return any(marker in parameter_name for marker in zero_initialized_control_outputs)


def make_incremental_optimizer_and_scheduler(cfg: DictConfig, policy):
    """为冻结Adapter构造AdamW；控制输出零衰减，结构矩阵弱衰减。"""
    trainable = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("增量优化器没有收到可训练参数。")
    invalid = [name for name, _ in trainable if not is_coupling_parameter(name)]
    if invalid:
        raise RuntimeError(f"优化器发现非Coupling参数: {invalid}")

    decay_parameters = [
        parameter
        for name, parameter in trainable
        if not _is_zero_decay_parameter(name, parameter)
    ]
    no_decay_parameters = [
        parameter
        for name, parameter in trainable
        if _is_zero_decay_parameter(name, parameter)
    ]
    coupling_lr = float(
        OmegaConf.select(cfg, "training.coupling_lr", default=cfg.training.lr)
    )
    structural_weight_decay = float(
        OmegaConf.select(
            cfg,
            "training.coupling_structural_weight_decay",
            default=1e-6,
        )
    )
    optimizer_groups = []
    if decay_parameters:
        optimizer_groups.append(
            {
                "name": "coupling_structure",
                "params": decay_parameters,
                "lr": coupling_lr,
                "weight_decay": structural_weight_decay,
            }
        )
    if no_decay_parameters:
        optimizer_groups.append(
            {
                "name": "coupling_control_no_decay",
                "params": no_decay_parameters,
                "lr": coupling_lr,
                "weight_decay": 0.0,
            }
        )

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=coupling_lr,
        betas=tuple(cfg.training.adam_betas),
        eps=float(cfg.training.adam_eps),
    )
    from diffusers.optimization import get_scheduler

    scheduler = get_scheduler(
        str(cfg.training.lr_scheduler),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=int(cfg.training.offline_steps),
    )
    logging.info(
        "冻结增量优化器: AdamW, lr=%g, structure=%d tensors (wd=%g), "
        "control/norm/bias=%d tensors (wd=0)",
        coupling_lr,
        len(decay_parameters),
        structural_weight_decay,
        len(no_decay_parameters),
    )
    return optimizer, scheduler


def make_incremental_policy(
    hydra_cfg: DictConfig,
    pretrained_policy_name_or_path: str | None = None,
    dataset_stats=None,
    *,
    allow_scid_dual_init: bool = False,
    strict_pretrained_loading: bool = False,
):
    """为原训练循环提供严格的增量Policy构造函数。"""
    del allow_scid_dual_init, strict_pretrained_loading
    if pretrained_policy_name_or_path is None:
        raise ValueError(
            "增量入口不允许随机初始化；请提供init_policy_path或有效resume_path。"
        )
    if dataset_stats is not None:
        raise ValueError("从checkpoint初始化时不应传入新的dataset_stats。")

    if bool(hydra_cfg.resume):
        policy = _ORIGINAL_MAKE_POLICY(
            hydra_cfg=hydra_cfg,
            pretrained_policy_name_or_path=pretrained_policy_name_or_path,
            dataset_stats=None,
            strict_pretrained_loading=True,
        )
        init_report: dict[str, Any] = {"mode": "resume"}
    else:
        policy_class, policy_config_class = get_policy_and_config_classes(
            hydra_cfg.policy.name
        )
        if policy_class is not CoupledDualHeadDiffusionPolicy:
            raise TypeError("增量目标必须是CoupledDualHeadDiffusionPolicy。")
        target_config = _policy_cfg_from_hydra_cfg(
            policy_config_class,
            hydra_cfg,
        )
        source_policy = DualHeadDiffusionPolicy.from_pretrained(
            pretrained_policy_name_or_path,
            strict=True,
        )
        validate_baseline_config_compatibility(
            source_policy.config,
            target_config,
        )
        policy = CoupledDualHeadDiffusionPolicy(target_config)
        migration = migrate_dual_state_into_coupled(source_policy, policy)

        check_enabled = bool(
            OmegaConf.select(
                hydra_cfg,
                "incremental.equivalence_check.enabled",
                default=True,
            )
        )
        equivalence = {}
        if check_enabled:
            equivalence = run_zero_gate_equivalence_check(
                source_policy,
                policy,
                atol=float(
                    OmegaConf.select(
                        hydra_cfg,
                        "incremental.equivalence_check.atol",
                        default=1e-6,
                    )
                ),
                rtol=float(
                    OmegaConf.select(
                        hydra_cfg,
                        "incremental.equivalence_check.rtol",
                        default=1e-5,
                    )
                ),
            )
        init_report = {
            "mode": "dual_checkpoint_migration",
            "migration": migration,
            "equivalence": equivalence,
        }
        del source_policy

    scope_report = configure_incremental_trainable_parameters(policy)
    install_frozen_module_mode_guard(policy)
    policy._incremental_init_report = init_report
    policy._incremental_scope_report = scope_report
    policy._coupling_diagnostics = CouplingDiagnostics(policy)
    policy.to(get_safe_torch_device(hydra_cfg.device))

    logging.info(
        "增量参数白名单: trainable=%d tensors/%d params, frozen=%d params, "
        "View->Arm=%s, Arm->View=%s",
        scope_report["trainable_tensor_count"],
        scope_report["trainable_parameter_count"],
        scope_report["frozen_parameter_count"],
        scope_report["view_to_arm_enabled"],
        scope_report["arm_to_view_enabled"],
    )
    logging.info("可训练参数:\n%s", "\n".join(scope_report["trainable_names"]))
    if init_report.get("equivalence"):
        logging.info(
            "零门控等价性检查通过: max_abs_error=%.3e",
            init_report["equivalence"]["max_abs_error"],
        )
    return policy


def update_incremental_policy(*args, **kwargs):
    """复用原更新逻辑，并在需要记录日志的step追加Coupling诊断。"""
    policy = args[0] if args else kwargs["policy"]
    collect_metrics = bool(kwargs.get("collect_metrics", True))
    diagnostics = getattr(policy, "_coupling_diagnostics", None)
    if diagnostics is not None:
        diagnostics.begin_step(collect_metrics)
    # 同步评估会调用policy.train()；每步再次约束可防止冻结BN状态漂移。
    policy.train(True)
    info = _ORIGINAL_UPDATE_POLICY(*args, **kwargs)
    if info and diagnostics is not None:
        info.update(diagnostics.collect())
    return info


def install_runtime_patches() -> None:
    """只在本独立进程中把可复用训练循环替换为增量构造/更新策略。"""
    base_train.make_policy = make_incremental_policy
    base_train.make_optimizer_and_scheduler = make_incremental_optimizer_and_scheduler
    base_train.update_policy = update_incremental_policy


def _save_effective_resume_config(
    cfg: DictConfig,
    out_dir: str,
    snapshot_config_path: Path | None,
) -> None:
    if snapshot_config_path is None:
        return
    effective_config_path = Path(out_dir) / ".hydra" / "config.yaml"
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, effective_config_path)
    logging.info(
        "严格恢复增量实验配置: snapshot=%s, effective=%s",
        snapshot_config_path,
        effective_config_path,
    )


@hydra.main(
    version_base="1.2",
    config_name="pre_default",
    config_path="../../../configs/pretrain",
)
def train_cli(cfg: DictConfig) -> None:
    """Hydra独立启动入口。"""
    hydra_runtime = hydra.core.hydra_config.HydraConfig.get()
    out_dir = hydra_runtime.run.dir
    cfg, snapshot_config_path = base_train.build_resume_config(cfg)
    validate_incremental_config(cfg)
    _save_effective_resume_config(cfg, out_dir, snapshot_config_path)
    install_runtime_patches()
    base_train.train_dppo_pretrain(
        cfg,
        out_dir=out_dir,
        job_name=hydra_runtime.job.name,
    )


def _prepare_resume_cli(user_cli_args: tuple[str, ...]) -> None:
    is_resume = (
        str(base_train.get_cli_override_value(sys.argv, "resume")).lower() == "true"
    )
    resume_path = base_train.get_cli_override_value(sys.argv, "resume_path")
    if not is_resume or not resume_path or resume_path.lower() in {"none", "null", ""}:
        return

    # 续训只能从Coupling实验checkpoint恢复，不能再次加载原始双头基线。
    base_train.replace_cli_override(sys.argv, "init_policy_path", "null")
    base_train.restore_resume_hydra_choices(
        sys.argv,
        user_cli_args,
        resume_path,
    )
    original_run_dir = base_train.get_resume_run_dir(resume_path)
    if original_run_dir is None:
        checkpoint_dir = base_train.get_resume_checkpoint_dir(resume_path)
        original_run_dir = checkpoint_dir.parent.parent
    base_train.replace_cli_override(
        sys.argv,
        "hydra.run.dir",
        f'"{original_run_dir.absolute()}"',
    )
    print(
        "🔄 [增量实验恢复] 输出目录已重定向至原运行:\n"
        f"   👉 {original_run_dir.absolute()}"
    )


if __name__ == "__main__":
    # 先保存用户在命令行中显式传入的参数。断点续训时据此判断用户是否
    # 主动修改了env/policy等训练语义配置，避免被下面的默认值干扰。
    explicit_cli_args = tuple(sys.argv[1:])

    # 可直接启动的本地默认配置。命令行显式提供同名参数时，用户值优先。
    default_args = [
        "init_policy_path='outputs/2_pretrain/train/2026-07-16/20-59-53_InsertCylinder-3Arms-v0_pre_zed_dual_head_diffusion/checkpoints/162000_loss=0.0051_sr=87.0_ar=751.50'",
        # 本地LeRobot格式专家数据集目录；目录有效时优先离线读取，不访问Hub。
        "dataset_local_dir=outputs/5_hf_datasets/quest_teleop_insert_cylinder_3arms_rgb_joint",
        # 本地目录未提供时使用的Hugging Face数据集仓库ID。
        "dataset_repo_id=Dc-dc/quest_teleop_insert_cylinder_3arms_rgb_joint",
        # Hydra环境配置组，决定任务名称、状态/动作维度及评估环境。
        "env=sim_insert_cylinder_3arms",
        # 冻结双头基线、只训练Coupling模块的专用策略配置组。
        "policy=pre_zed_coupling_incremental",
        # false表示创建新实验；true表示从resume_path恢复完整增量训练状态。
        "resume=false",
        # 增量实验checkpoint目录；首次训练时保持null并使用init_policy_path。
        "resume_path=null",
        # DataLoader并行工作进程数，可根据服务器CPU和内存容量调整。
        "training.num_workers=5",
        # 是否启用W&B实验指标、评估结果及Coupling诊断数据上传。
        "wandb.enable=true",
    ]
    for argument in default_args:
        argument_key = argument.split("=", 1)[0]
        if base_train.get_cli_override_value(sys.argv, argument_key) is None:
            sys.argv.append(argument)

    _prepare_resume_cli(explicit_cli_args)
    train_cli()
