#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""去噪过程耦合的双头 Diffusion Policy。

该模型保留独立双头策略的动作拆分和共享观测编码器，但在每个扩散时间步同步处理
操作头与视角头，并在两个 UNet 的瓶颈层进行双向交叉注意力。交叉注意力通过
随扩散时间步变化的零初始化门控残差注入，因此初始化时严格退化为独立双头，
训练后再逐渐学习操作动作与主动视角动作之间的协调关系。
"""

from __future__ import annotations

import math
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DiffusionSinusoidalPosEmb,
    DualHeadDiffusionModel,
    DualHeadDiffusionPolicy,
)
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import get_device_from_parameters, get_dtype_from_parameters


class RoleAdaLNZeroCouplingBlock(nn.Module):
    """DiT式角色条件调制与可选FFN，不改变跨头注意力的时间路由。"""

    def __init__(
        self,
        *,
        bottleneck_dim: int,
        timestep_embed_dim: int,
        use_ffn: bool,
        ffn_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.bottleneck_dim = int(bottleneck_dim)
        self.use_ffn = bool(use_ffn)

        # 与DiT相同，先把标量扩散时间步编码成一个条件向量。
        self.timestep_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim * 2, timestep_embed_dim),
        )
        self.arm_attention_norm = nn.LayerNorm(
            bottleneck_dim, elementwise_affine=False, eps=1e-6
        )
        self.view_attention_norm = nn.LayerNorm(
            bottleneck_dim, elementwise_affine=False, eps=1e-6
        )
        # 输出Arm/View各自的shift和scale，共4C维。
        self.attention_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, bottleneck_dim * 4),
        )
        nn.init.zeros_(self.attention_modulation[-1].weight)
        nn.init.zeros_(self.attention_modulation[-1].bias)

        if self.use_ffn:
            ffn_hidden_dim = max(1, int(round(bottleneck_dim * ffn_ratio)))
            self.arm_ffn_norm = nn.LayerNorm(
                bottleneck_dim, elementwise_affine=False, eps=1e-6
            )
            self.view_ffn_norm = nn.LayerNorm(
                bottleneck_dim, elementwise_affine=False, eps=1e-6
            )
            self.arm_ffn = nn.Sequential(
                nn.Linear(bottleneck_dim, ffn_hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden_dim, bottleneck_dim),
            )
            self.view_ffn = nn.Sequential(
                nn.Linear(bottleneck_dim, ffn_hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden_dim, bottleneck_dim),
            )
            # 每个角色输出FFN的shift(C)、scale(C)和一个可解释标量gate。
            self.ffn_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_embed_dim, bottleneck_dim * 4 + 2),
            )
            nn.init.zeros_(self.ffn_modulation[-1].weight)
            nn.init.zeros_(self.ffn_modulation[-1].bias)

    @staticmethod
    def _modulate(tokens: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        """按DiT约定执行(1+scale)*LayerNorm(x)+shift。"""
        return tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def modulate_attention_inputs(
        self,
        arm_tokens: Tensor,
        view_tokens: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """生成角色独立的adaLN参数并返回调制后的交叉注意力输入。"""
        condition = self.timestep_encoder(timesteps)
        arm_shift, arm_scale, view_shift, view_scale = self.attention_modulation(
            condition
        ).chunk(4, dim=-1)
        arm_tokens = self._modulate(
            self.arm_attention_norm(arm_tokens), arm_shift, arm_scale
        )
        view_tokens = self._modulate(
            self.view_attention_norm(view_tokens), view_shift, view_scale
        )
        return arm_tokens, view_tokens, condition

    def compute_ffn_deltas(
        self,
        arm_tokens: Tensor,
        view_tokens: Tensor,
        condition: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """计算零门控角色FFN残差；具体写回token由外部路由掩码决定。"""
        if not self.use_ffn:
            raise RuntimeError("未启用coupling FFN，不能计算FFN残差")

        dim = self.bottleneck_dim
        modulation = self.ffn_modulation(condition)
        (
            arm_shift,
            arm_scale,
            arm_gate,
            view_shift,
            view_scale,
            view_gate,
        ) = torch.split(modulation, [dim, dim, 1, dim, dim, 1], dim=-1)
        arm_delta = arm_gate.unsqueeze(1) * self.arm_ffn(
            self._modulate(self.arm_ffn_norm(arm_tokens), arm_shift, arm_scale)
        )
        view_delta = view_gate.unsqueeze(1) * self.view_ffn(
            self._modulate(self.view_ffn_norm(view_tokens), view_shift, view_scale)
        )
        return arm_delta, view_delta


class CoupledDualHeadDiffusionPolicy(DualHeadDiffusionPolicy):
    """使用瓶颈双向注意力协调操作头和视角头的 Diffusion Policy。"""

    name = "coupled_dual_head_diffusion"

    def __init__(
        self,
        config: DiffusionConfig | dict | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        # 不调用父类构造函数，避免先创建一套即将被丢弃的独立双头 UNet。
        nn.Module.__init__(self)
        if config is None:
            config = DiffusionConfig()
        elif isinstance(config, dict):
            # huggingface_hub.from_pretrained 会从 config.json 传入普通字典。
            config = DiffusionConfig.from_dict(config)
        self.config = config
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self._queues = None
        self.diffusion = CoupledDualHeadDiffusionModel(config)
        self.expected_image_keys = [
            key for key in config.input_shapes if key.startswith("observation.image")
        ]
        self.use_env_state = "observation.environment_state" in config.input_shapes
        self.reset()

    def reset(self):
        """清空历史观测和待执行动作队列。"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.expected_image_keys:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.use_env_state:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)


class CoupledDualHeadDiffusionModel(DualHeadDiffusionModel):
    """在每个去噪时间步联合预测两个动作头噪声的双头扩散模型。"""

    def __init__(self, config: DiffusionConfig):
        super().__init__(config)

        bottleneck_dim = int(config.down_dims[-1])
        num_heads = int(getattr(config, "coupling_num_heads", 8))
        dropout = float(getattr(config, "coupling_dropout", 0.0))
        self.coupling_mode = str(getattr(config, "coupling_mode", "full"))
        self.coupling_block_type = str(
            getattr(config, "coupling_block_type", "scalar_gate")
        )
        self.coupling_use_temporal_pos_emb = bool(
            getattr(config, "coupling_use_temporal_pos_emb", False)
        )
        self.coupling_use_ffn = bool(getattr(config, "coupling_use_ffn", False))
        coupling_ffn_ratio = float(getattr(config, "coupling_ffn_ratio", 2.0))
        self.view_to_arm_coupling_scale = 1.0
        self.arm_to_view_coupling_scale = 1.0
        self.set_coupling_scales(
            view_to_arm=float(
                getattr(config, "view_to_arm_coupling_scale", 1.0)
            ),
            arm_to_view=float(
                getattr(config, "arm_to_view_coupling_scale", 1.0)
            ),
        )
        if num_heads <= 0:
            raise ValueError(f"耦合注意力头数必须为正数，当前为{num_heads}")
        if bottleneck_dim % num_heads != 0:
            raise ValueError(
                "耦合注意力头数必须整除UNet瓶颈维度: "
                f"bottleneck_dim={bottleneck_dim}, coupling_num_heads={num_heads}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"耦合dropout必须位于[0, 1)，当前为{dropout}")
        if self.coupling_mode not in {
            "full",
            "rbac",
            "balanced_lookahead",
            "rcla",
            "bidirectional_prefix_to_suffix",
        }:
            raise ValueError(
                "coupling_mode必须为'full'、'rbac'、'balanced_lookahead'、'rcla'或"
                "'bidirectional_prefix_to_suffix'，当前为"
                f"{self.coupling_mode!r}"
            )
        if self.coupling_block_type not in {"scalar_gate", "role_adaln_zero"}:
            raise ValueError(
                "coupling_block_type必须为'scalar_gate'或'role_adaln_zero'，当前为"
                f"{self.coupling_block_type!r}"
            )
        if self.coupling_use_ffn and self.coupling_block_type != "role_adaln_zero":
            raise ValueError(
                "coupling_use_ffn=True要求coupling_block_type='role_adaln_zero'"
            )
        if not math.isfinite(coupling_ffn_ratio) or coupling_ffn_ratio <= 0:
            raise ValueError(
                f"coupling_ffn_ratio必须为有限正数，当前为{coupling_ffn_ratio}"
            )

        # 两个方向使用独立注意力参数，允许操作和视角学习不对称的信息需求。
        self.view_to_arm_attention = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.arm_to_view_attention = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.arm_coupling_norm = nn.LayerNorm(bottleneck_dim)
        self.view_coupling_norm = nn.LayerNorm(bottleneck_dim)
        self.coupling_dropout = nn.Dropout(dropout)

        # 输出两个随时间步变化的门值；最后一层零初始化保证初始模型等价于独立双头。
        gate_hidden_dim = int(config.diffusion_step_embed_dim)
        self.coupling_timestep_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(gate_hidden_dim),
            nn.Linear(gate_hidden_dim, gate_hidden_dim * 2),
            nn.Mish(),
            nn.Linear(gate_hidden_dim * 2, 2),
        )
        nn.init.zeros_(self.coupling_timestep_encoder[-1].weight)
        nn.init.zeros_(self.coupling_timestep_encoder[-1].bias)

        # 仅在新模式下创建额外参数，使默认scalar_gate仍可严格加载旧checkpoint。
        self.role_adaln_coupling = (
            RoleAdaLNZeroCouplingBlock(
                bottleneck_dim=bottleneck_dim,
                timestep_embed_dim=gate_hidden_dim,
                use_ffn=self.coupling_use_ffn,
                ffn_ratio=coupling_ffn_ratio,
                dropout=dropout,
            )
            if self.coupling_block_type == "role_adaln_zero"
            else None
        )

    @staticmethod
    def _validate_coupling_scale(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}必须是[0, 1]内的有限数，当前为{value}")
        return value

    def set_coupling_scales(
        self,
        *,
        view_to_arm: float | None = None,
        arm_to_view: float | None = None,
    ) -> dict[str, float]:
        """设置两个耦合方向的外部缩放，用于同一checkpoint上的推理消融。

        ``view_to_arm``控制注入Arm流的View上下文；``arm_to_view``控制
        注入View流的Arm上下文。该接口只修改无参数标量，不改变state_dict。
        """
        if view_to_arm is not None:
            self.view_to_arm_coupling_scale = self._validate_coupling_scale(
                "view_to_arm_coupling_scale", view_to_arm
            )
        if arm_to_view is not None:
            self.arm_to_view_coupling_scale = self._validate_coupling_scale(
                "arm_to_view_coupling_scale", arm_to_view
            )

        # 同步配置，确保保存新checkpoint时保留实际生效的实验设置。
        self.config.view_to_arm_coupling_scale = self.view_to_arm_coupling_scale
        self.config.arm_to_view_coupling_scale = self.arm_to_view_coupling_scale
        return {
            "view_to_arm_coupling_scale": self.view_to_arm_coupling_scale,
            "arm_to_view_coupling_scale": self.arm_to_view_coupling_scale,
        }

    def _resolve_rbac_token_slices(self, token_count: int) -> tuple[slice, slice]:
        """把完整动作时间边界投影到UNet瓶颈token，并返回View前缀与Arm后缀。

        RBAC使用本轮真实执行的View动作准备下一次观测，并把本轮不执行的Arm
        预测后缀作为未来操作意图。向下取整/向上取整可在边界不与UNet降采样
        对齐时保留覆盖执行区间的token；两个区间始终互不重叠。
        """
        if token_count <= 1:
            raise ValueError(f"RBAC至少需要2个瓶颈token，当前为{token_count}")

        horizon = int(self.config.horizon)
        action_start = int(self.config.n_obs_steps) - 1
        boundary = action_start + int(self.config.n_action_steps)
        if boundary >= horizon:
            raise ValueError(
                "RBAC要求重规划边界之后存在Arm预测后缀，当前为"
                f"boundary={boundary}, horizon={horizon}"
            )

        view_start = math.floor(action_start * token_count / horizon)  # view_start = floor(1 * 4 / 16) = 0
        boundary_token = math.ceil(boundary * token_count / horizon)   # boundary = action_start + n_action_steps = 9
        view_start = min(max(view_start, 0), token_count - 1)          # view_start = floor(1 * 4 / 16) = 0
        boundary_token = min(max(boundary_token, view_start + 1), token_count - 1)
        return slice(view_start, boundary_token), slice(boundary_token, token_count)  # boundary_token = ceil(9 * 4 / 16) = 3

    @staticmethod
    def _resolve_balanced_lookahead_token_slices(
        token_count: int,
    ) -> tuple[slice, slice]:
        """把瓶颈token等分为当前前缀和未来后缀。

        等长切分让两个方向都是N×N注意力，避免RBAC在4-token瓶颈上退化为
        单个future key；future仅作为前瞻信息，耦合残差只写回current前缀。
        """
        if token_count < 2 or token_count % 2 != 0:
            raise ValueError(
                "balanced_lookahead要求不少于2个且数量为偶数的瓶颈token，当前为"
                f"{token_count}"
            )
        split = token_count // 2
        return slice(0, split), slice(split, token_count)   # token_count=4

    @staticmethod
    def _build_rcla_attention_masks(
        token_count: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """构造角色不对称的局部时滞掩码；True表示禁止该Query读取Key。

        View->Arm允许Arm token i读取View token {i-1, i}，使操作参考当前及
        一步滞后的视角信息；Arm->View允许View token i读取Arm token {i, i+1}，
        使主动视角参考当前及一步前瞻的操作意图。
        """
        if token_count <= 0:
            raise ValueError(f"RCLA至少需要1个瓶颈token，当前为{token_count}")

        view_to_arm_mask = torch.ones(
            token_count, token_count, dtype=torch.bool, device=device
        )
        arm_to_view_mask = torch.ones_like(view_to_arm_mask)
        token_indices = torch.arange(token_count, device=device)

        # 对角线：两个方向都允许同时间token交互。
        view_to_arm_mask[token_indices, token_indices] = False
        arm_to_view_mask[token_indices, token_indices] = False
        if token_count > 1:
            # 下副对角线：Arm(i) <- View(i-1)。
            view_to_arm_mask[token_indices[1:], token_indices[:-1]] = False
            # 上副对角线：View(i) <- Arm(i+1)。
            arm_to_view_mask[token_indices[:-1], token_indices[1:]] = False

        return view_to_arm_mask, arm_to_view_mask

    @staticmethod
    def _build_temporal_sinusoidal_embedding(
        token_count: int,
        embedding_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """构造无参数的一维时间位置编码，形状为[1,T,C]。"""
        if token_count <= 0:
            raise ValueError(f"时间位置编码至少需要1个token，当前为{token_count}")
        if embedding_dim <= 0:
            raise ValueError(f"时间位置编码维度必须为正数，当前为{embedding_dim}")

        half_dim = embedding_dim // 2
        if half_dim == 0:
            return torch.zeros(
                1, token_count, embedding_dim, device=device, dtype=dtype
            )
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / half_dim
        )
        positions = torch.arange(token_count, device=device, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if embedding_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding.unsqueeze(0).to(dtype=dtype)

    def _resolve_coupling_write_masks(
        self,
        token_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """返回Arm/View允许写入耦合残差的位置，形状均为[1,T,1]。"""
        arm_mask = torch.zeros(1, token_count, 1, device=device, dtype=dtype)
        view_mask = torch.zeros_like(arm_mask)

        if self.coupling_mode in {"full", "rcla"}:
            arm_mask.fill_(1)
            view_mask.fill_(1)
            return arm_mask, view_mask

        if self.coupling_mode == "rbac":
            view_prefix, arm_future = self._resolve_rbac_token_slices(token_count)
            arm_mask[:, arm_future] = 1
            view_mask[:, view_prefix] = 1
            return arm_mask, view_mask

        if self.coupling_mode == "bidirectional_prefix_to_suffix":
            _, suffix = self._resolve_rbac_token_slices(token_count)
            arm_mask[:, suffix] = 1
            view_mask[:, suffix] = 1
            return arm_mask, view_mask

        current, _ = self._resolve_balanced_lookahead_token_slices(token_count)
        arm_mask[:, current] = 1
        view_mask[:, current] = 1
        return arm_mask, view_mask

    def _compute_coupling_contexts(
        self,
        normalized_arm: Tensor,
        normalized_view: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """按配置计算注入Arm和View流的上下文，输出形状保持完整token长度。"""
        if self.coupling_mode == "full":
            view_context, _ = self.view_to_arm_attention(
                query=normalized_arm,
                key=normalized_view,
                value=normalized_view,
                need_weights=False,
            )
            arm_context, _ = self.arm_to_view_attention(
                query=normalized_view,
                key=normalized_arm,
                value=normalized_arm,
                need_weights=False,
            )
            return view_context, arm_context

        if self.coupling_mode == "rbac":
            view_prefix, arm_future = self._resolve_rbac_token_slices(
                normalized_arm.shape[1]
            )
            # Arm后缀读取将被执行的View前缀，使未来意图与本轮视角计划一致。
            future_arm_context, _ = self.view_to_arm_attention(
                query=normalized_arm[:, arm_future],   # arm_future  # token [3]
                key=normalized_view[:, view_prefix],   # view_prefix token [0, 1, 2]
                value=normalized_view[:, view_prefix],
                need_weights=False,
            )
            # View前缀读取不会被执行的Arm后缀，把后缀变成下一阶段的操作意图预览。
            executed_view_context, _ = self.arm_to_view_attention(
                query=normalized_view[:, view_prefix],
                key=normalized_arm[:, arm_future],
                value=normalized_arm[:, arm_future],
                need_weights=False,
            )

            view_context = torch.zeros_like(normalized_arm)
            arm_context = torch.zeros_like(normalized_view)
            view_context[:, arm_future] = future_arm_context
            arm_context[:, view_prefix] = executed_view_context
            return view_context, arm_context

        if self.coupling_mode == "bidirectional_prefix_to_suffix":
            prefix, suffix = self._resolve_rbac_token_slices(
                normalized_arm.shape[1]
            )
            # View前缀[0:3]形成视角条件，只修正Arm未来后缀[3:4]。
            future_arm_context, _ = self.view_to_arm_attention(
                query=normalized_arm[:, suffix],
                key=normalized_view[:, prefix],
                value=normalized_view[:, prefix],
                need_weights=False,
            )
            # 对称地由Arm前缀[0:3]形成操作条件，只修正View未来后缀[3:4]。
            future_view_context, _ = self.arm_to_view_attention(
                query=normalized_view[:, suffix],
                key=normalized_arm[:, prefix],
                value=normalized_arm[:, prefix],
                need_weights=False,
            )

            view_context = torch.zeros_like(normalized_arm)
            arm_context = torch.zeros_like(normalized_view)
            view_context[:, suffix] = future_arm_context
            arm_context[:, suffix] = future_view_context
            return view_context, arm_context

        if self.coupling_mode == "rcla":
            view_to_arm_mask, arm_to_view_mask = self._build_rcla_attention_masks(
                normalized_arm.shape[1], normalized_arm.device
            )
            # Arm(i)仅读取View(i-1, i)：视角信息先于或同步约束操作。
            view_context, _ = self.view_to_arm_attention(
                query=normalized_arm,
                key=normalized_view,
                value=normalized_view,
                attn_mask=view_to_arm_mask,
                need_weights=False,
            )
            # View(i)仅读取Arm(i, i+1)：视角根据当前与下一步操作意图前瞻调整。
            arm_context, _ = self.arm_to_view_attention(
                query=normalized_view,
                key=normalized_arm,
                value=normalized_arm,
                attn_mask=arm_to_view_mask,
                need_weights=False,
            )
            return view_context, arm_context

        current, future = self._resolve_balanced_lookahead_token_slices(
            normalized_arm.shape[1]
        )
        # View未来后缀只修正Arm当前前缀，不向Arm未来瓶颈直接写入耦合残差。
        # View → Arm：
        # Arm当前token [0,1] 作为Query，
        # 读取View未来token [2,3]，得到对Arm当前动作的修正信息。
        #
        # 映射不是固定的 0→2、1→3，而是：
        # Arm token 0 → 加权读取 View token {2,3}
        # Arm token 1 → 加权读取 View token {2,3}
        current_arm_context, _ = self.view_to_arm_attention(
            query=normalized_arm[:, current],
            key=normalized_view[:, future],
            value=normalized_view[:, future],
            need_weights=False,
        )
        # 对称地以Arm未来后缀修正View当前前缀；两个方向读取同一份未融合快照。
        # Arm → View：
        # View当前token [0,1] 作为Query，
        # 读取Arm未来token [2,3]，得到对View当前动作的修正信息。
        #
        # 映射同样是：
        # View token 0 → 加权读取 Arm token {2,3}
        # View token 1 → 加权读取 Arm token {2,3}
        current_view_context, _ = self.arm_to_view_attention(
            query=normalized_view[:, current],  # [B, 2, C]，View当前视角意图
            key=normalized_arm[:, future],      # [B, 2, C]，Arm未来操作意图
            value=normalized_arm[:, future],    # [B, 2, C]，Arm未来操作信息
            need_weights=False,
        )

        view_context = torch.zeros_like(normalized_arm)
        arm_context = torch.zeros_like(normalized_view)
        view_context[:, current] = current_arm_context
        arm_context[:, current] = current_view_context
        return view_context, arm_context

    @staticmethod
    def _as_timestep_batch(timestep: Tensor | int, sample: Tensor) -> Tensor:
        """把标量时间步转换为与 batch 对齐的一维 LongTensor。"""
        if isinstance(timestep, int):
            return torch.full(
                sample.shape[:1], timestep, dtype=torch.long, device=sample.device
            )
        timestep = timestep.to(device=sample.device, dtype=torch.long)
        if timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0])
        if timestep.shape != sample.shape[:1]:
            raise ValueError(
                f"扩散时间步形状必须为{sample.shape[:1]}，当前为{tuple(timestep.shape)}"
            )
        return timestep

    @staticmethod
    def _encode_unet_bottleneck(
        unet: nn.Module,
        sample: Tensor,
        timesteps: Tensor,
        global_cond: Tensor | None,
    ) -> tuple[Tensor, list[Tensor]]:
        """运行一个动作头的UNet编码器和中间层，返回瓶颈及跳连特征。"""
        feature = einops.rearrange(sample, "b t d -> b d t")
        timestep_feature = unet.diffusion_step_encoder(timesteps)
        condition = (
            torch.cat([timestep_feature, global_cond], dim=-1)
            if global_cond is not None
            else timestep_feature
        )

        skip_features: list[Tensor] = []
        for residual, residual2, downsample in unet.down_modules:
            feature = residual(feature, condition)
            feature = residual2(feature, condition)
            skip_features.append(feature)
            feature = downsample(feature)

        for mid_module in unet.mid_modules:
            feature = mid_module(feature, condition)
        return feature, skip_features

    @staticmethod
    def _decode_unet_bottleneck(
        unet: nn.Module,
        bottleneck: Tensor,
        skip_features: list[Tensor],
        timesteps: Tensor,
        global_cond: Tensor | None,
    ) -> Tensor:
        """从耦合后的瓶颈恢复当前动作头的噪声预测。"""
        timestep_feature = unet.diffusion_step_encoder(timesteps)
        condition = (
            torch.cat([timestep_feature, global_cond], dim=-1)
            if global_cond is not None
            else timestep_feature
        )

        feature = bottleneck
        remaining_skips = list(skip_features)
        for residual, residual2, upsample in unet.up_modules:
            feature = torch.cat((feature, remaining_skips.pop()), dim=1)
            feature = residual(feature, condition)
            feature = residual2(feature, condition)
            feature = upsample(feature)

        feature = unet.final_conv(feature)
        return einops.rearrange(feature, "b d t -> b t d")

    def _couple_bottlenecks(
        self,
        arm_bottleneck: Tensor,
        view_bottleneck: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """用双向交叉注意力和时间步门控交换两个动作头的瓶颈特征。"""
        arm_tokens = einops.rearrange(arm_bottleneck, "b c t -> b t c")
        view_tokens = einops.rearrange(view_bottleneck, "b c t -> b t c")
        if arm_tokens.shape != view_tokens.shape:
            raise ValueError(
                "两个动作头的瓶颈形状必须一致，当前为"
                f"arm={tuple(arm_tokens.shape)}, view={tuple(view_tokens.shape)}"
            )

        # 位置编码只进入耦合分支；零门控时不会改变两个UNet的原始瓶颈。
        coupling_arm_tokens = arm_tokens
        coupling_view_tokens = view_tokens
        if self.coupling_use_temporal_pos_emb:
            position_embedding = self._build_temporal_sinusoidal_embedding(
                arm_tokens.shape[1],
                arm_tokens.shape[2],
                device=arm_tokens.device,
                dtype=arm_tokens.dtype,
            )
            coupling_arm_tokens = coupling_arm_tokens + position_embedding
            coupling_view_tokens = coupling_view_tokens + position_embedding

        # 两个方向都读取未融合的快照，避免一次去噪内部出现先后更新偏差。
        role_condition = None
        if self.coupling_block_type == "role_adaln_zero":
            if self.role_adaln_coupling is None:
                raise RuntimeError("role_adaln_zero模式缺少对应耦合模块")
            normalized_arm, normalized_view, role_condition = (
                self.role_adaln_coupling.modulate_attention_inputs(
                    coupling_arm_tokens,
                    coupling_view_tokens,
                    timesteps,
                )
            )
        else:
            normalized_arm = self.arm_coupling_norm(coupling_arm_tokens)
            normalized_view = self.view_coupling_norm(coupling_view_tokens)
        view_context, arm_context = self._compute_coupling_contexts(
            normalized_arm, normalized_view
        )

        gates = torch.tanh(self.coupling_timestep_encoder(timesteps))
        arm_gate = gates[:, 0].view(-1, 1, 1)
        view_gate = gates[:, 1].view(-1, 1, 1)
        arm_tokens = arm_tokens + (
            self.view_to_arm_coupling_scale
            * arm_gate
            * self.coupling_dropout(view_context)
        )
        view_tokens = view_tokens + (
            self.arm_to_view_coupling_scale
            * view_gate
            * self.coupling_dropout(arm_context)
        )

        # 可选DiT式FFN只写回当前路由允许修改的token，保持路由语义可解释。
        if self.coupling_use_ffn:
            if self.role_adaln_coupling is None or role_condition is None:
                raise RuntimeError("启用coupling FFN时必须使用role_adaln_zero")
            arm_ffn_delta, view_ffn_delta = (
                self.role_adaln_coupling.compute_ffn_deltas(
                    arm_tokens,
                    view_tokens,
                    role_condition,
                )
            )
            arm_write_mask, view_write_mask = self._resolve_coupling_write_masks(
                arm_tokens.shape[1],
                device=arm_tokens.device,
                dtype=arm_tokens.dtype,
            )
            arm_tokens = arm_tokens + (
                self.view_to_arm_coupling_scale
                * arm_write_mask
                * self.coupling_dropout(arm_ffn_delta)
            )
            view_tokens = view_tokens + (
                self.arm_to_view_coupling_scale
                * view_write_mask
                * self.coupling_dropout(view_ffn_delta)
            )

        return (
            einops.rearrange(arm_tokens, "b t c -> b c t"),
            einops.rearrange(view_tokens, "b t c -> b c t"),
        )

    def predict_noise(
        self,
        arm_sample: Tensor,
        view_sample: Tensor,
        timestep: Tensor | int,
        global_cond: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """联合预测同一扩散时间步下的操作头与视角头噪声。"""
        if arm_sample.shape[:2] != view_sample.shape[:2]:
            raise ValueError(
                "两个动作头的batch和horizon必须一致，当前为"
                f"arm={tuple(arm_sample.shape)}, view={tuple(view_sample.shape)}"
            )
        if arm_sample.shape[-1] != self.arm_action_dim:
            raise ValueError(
                f"操作头动作维度应为{self.arm_action_dim}，当前为{arm_sample.shape[-1]}"
            )
        if view_sample.shape[-1] != self.view_action_dim:
            raise ValueError(
                f"视角头动作维度应为{self.view_action_dim}，当前为{view_sample.shape[-1]}"
            )

        timesteps = self._as_timestep_batch(timestep, arm_sample)
        arm_bottleneck, arm_skips = self._encode_unet_bottleneck(
            self.arm_unet, arm_sample, timesteps, global_cond
        )
        view_bottleneck, view_skips = self._encode_unet_bottleneck(
            self.view_unet, view_sample, timesteps, global_cond
        )
        arm_bottleneck, view_bottleneck = self._couple_bottlenecks(
            arm_bottleneck, view_bottleneck, timesteps
        )
        arm_prediction = self._decode_unet_bottleneck(
            self.arm_unet, arm_bottleneck, arm_skips, timesteps, global_cond
        )
        view_prediction = self._decode_unet_bottleneck(
            self.view_unet, view_bottleneck, view_skips, timesteps, global_cond
        )
        return arm_prediction, view_prediction

    def _set_synchronized_inference_timesteps(self) -> Tensor:
        """设置两个调度器的推理时间步并验证完全一致。"""
        self.arm_noise_scheduler.set_timesteps(self.num_inference_steps)
        self.view_noise_scheduler.set_timesteps(self.num_inference_steps)
        arm_timesteps = self.arm_noise_scheduler.timesteps
        view_timesteps = self.view_noise_scheduler.timesteps
        if not torch.equal(arm_timesteps, view_timesteps):
            raise RuntimeError("耦合双头要求两个noise scheduler使用相同的推理timesteps")
        return arm_timesteps

    def conditional_sample_coupled(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """从独立高斯先验开始，在一个同步循环中联合去噪两个动作头。"""
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        arm_sample = torch.randn(
            (batch_size, self.config.horizon, self.arm_action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        view_sample = torch.randn(
            (batch_size, self.config.horizon, self.view_action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        for timestep in self._set_synchronized_inference_timesteps():
            timestep_batch = torch.full(
                arm_sample.shape[:1],
                int(timestep.item()),
                dtype=torch.long,
                device=device,
            )
            arm_prediction, view_prediction = self.predict_noise(
                arm_sample, view_sample, timestep_batch, global_cond
            )
            # 两个预测都基于同一份 x_t，随后才同步推进到 x_(t-1)。
            arm_sample = self.arm_noise_scheduler.step(
                arm_prediction, timestep, arm_sample, generator=generator
            ).prev_sample
            view_sample = self.view_noise_scheduler.step(
                view_prediction, timestep, view_sample, generator=generator
            ).prev_sample

        return arm_sample, view_sample

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """根据共享观测条件生成并截取待执行的耦合双头动作块。"""
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(
                f"观测历史长度应为{self.config.n_obs_steps}，当前为{n_obs_steps}"
            )
        global_cond = self._prepare_global_conditioning(batch)
        arm_actions, view_actions = self.conditional_sample_coupled(
            batch_size, global_cond=global_cond
        )
        actions = torch.cat([arm_actions, view_actions], dim=-1)
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        if start < 0 or end > actions.shape[1]:
            raise ValueError(f"动作切片[{start}:{end}]越界，horizon={actions.shape[1]}")
        return actions[:, start:end]

    @staticmethod
    def _masked_mse(
        prediction: Tensor,
        target: Tensor,
        action_is_pad: Tensor,
        mask_padding: bool,
    ) -> Tensor:
        """计算一个动作头的MSE，并按配置屏蔽episode末尾填充动作。"""
        loss = F.mse_loss(prediction, target, reduction="none")
        if mask_padding:
            if action_is_pad is None:
                raise ValueError("启用padding loss mask时必须提供action_is_pad")
            loss = loss * (~action_is_pad).unsqueeze(-1)
        return loss.mean()

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """使用相同扩散时间步联合训练两个动作头及其耦合模块。"""
        required_keys = {"observation.state", "action", "action_is_pad"}
        if not set(batch).issuperset(required_keys):
            raise KeyError(f"训练batch缺少字段: {required_keys.difference(batch)}")
        if "observation.images" not in batch and "observation.environment_state" not in batch:
            raise KeyError("训练batch必须包含图像或environment_state")
        if batch["observation.state"].shape[1] != self.config.n_obs_steps:
            raise ValueError("观测历史长度与配置不一致")
        if batch["action"].shape[1] != self.config.horizon:
            raise ValueError("动作horizon与配置不一致")
        if batch["action"].shape[-1] != self.action_dim:
            raise ValueError(
                f"完整动作维度应为{self.action_dim}，当前为{batch['action'].shape[-1]}"
            )

        global_cond = self._prepare_global_conditioning(batch)
        arm_trajectory = batch["action"][..., : self.arm_action_dim]
        view_trajectory = batch["action"][..., self.arm_action_dim :]
        arm_noise = torch.randn_like(arm_trajectory)
        view_noise = torch.randn_like(view_trajectory)

        # 同一个timestep保证两个头交换的是同等噪声水平下的动作特征。
        timesteps = torch.randint(
            low=0,
            high=self.arm_noise_scheduler.config.num_train_timesteps,
            size=(arm_trajectory.shape[0],),
            device=arm_trajectory.device,
        ).long()
        noisy_arm = self.arm_noise_scheduler.add_noise(
            arm_trajectory, arm_noise, timesteps
        )
        noisy_view = self.view_noise_scheduler.add_noise(
            view_trajectory, view_noise, timesteps
        )
        arm_prediction, view_prediction = self.predict_noise(
            noisy_arm, noisy_view, timesteps, global_cond
        )

        if self.config.prediction_type == "epsilon":
            arm_target, view_target = arm_noise, view_noise
        elif self.config.prediction_type == "sample":
            arm_target, view_target = arm_trajectory, view_trajectory
        else:
            raise ValueError(f"不支持prediction_type={self.config.prediction_type!r}")

        arm_loss = self._masked_mse(
            arm_prediction,
            arm_target,
            batch["action_is_pad"],
            self.config.do_mask_loss_for_padding,
        )
        view_loss = self._masked_mse(
            view_prediction,
            view_target,
            batch["action_is_pad"],
            self.config.do_mask_loss_for_padding,
        )
        loss = arm_loss + self.view_loss_weight * view_loss
        return {
            "loss": loss,
            "arm_loss": arm_loss.detach(),
            "view_loss": view_loss.detach(),
        }
