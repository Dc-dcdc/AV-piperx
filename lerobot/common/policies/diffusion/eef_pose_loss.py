#!/usr/bin/env python

"""Piper三臂关节动作的可微末端位姿监督。

训练数据和策略仍使用20维关节动作：

``left(6 joint + gripper) + right(6 joint + gripper) + middle(6 joint)``。

本模块只在训练时把三个6轴关节子向量送入可微PoE正运动学。固定运动学参数
来自 ``env/assets/piperx_sim.xml`` 的各机械臂基座局部坐标系；基座在世界坐标
中的平移和旋转不会改变预测/目标之间的欧氏位置距离与旋转测地距离。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


PIPER_JOINT_AXES = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.086981968, 0.0, -0.996209886),
    (0.996209888, 0.0, 0.086981942),
)
PIPER_JOINT_ANCHORS = (
    (0.0, 0.0, 0.123),
    (0.0, 0.0, 0.123),
    (-0.282406341, 0.0, 0.161584444),
    (-0.012606357, 0.0, 0.207266150),
    (0.061770673, 0.0, 0.213760221),
    (0.096638019, 0.0, 0.216804590),
)
PIPER_HAND_HOME = (
    (0.086981942, 0.0, 0.996209888, 0.238597928),
    (0.0, 1.0, 0.0, 0.0),
    (-0.996209888, 0.0, 0.086981942, 0.229199517),
    (0.0, 0.0, 0.0, 1.0),
)
PIPER_VIEW_HOME = (
    (0.000000737, 0.242091646, -0.970253387, 0.134733106),
    (-1.0, 0.0, -0.000000736, 0.030000296),
    (0.0, 0.970253387, 0.242091646, 0.288443567),
    (0.0, 0.0, 0.0, 1.0),
)


def _skew(vector: Tensor) -> Tensor:
    """把最后一维三向量转换为反对称矩阵。"""
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


class PiperEndEffectorPoseLoss(nn.Module):
    """由20维关节轨迹计算三条机械臂的末端位姿损失。"""

    action_dim = 20
    joint_slices = ((0, 6), (7, 13), (14, 20))
    num_roles = 3

    def __init__(self) -> None:
        super().__init__()
        # 这些是固定机器人参数而非模型状态；persistent=False保证旧checkpoint
        # 与开启位姿监督的新checkpoint仍保持严格兼容。
        self.register_buffer(
            "joint_axes",
            torch.tensor(PIPER_JOINT_AXES, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "joint_anchors",
            torch.tensor(PIPER_JOINT_ANCHORS, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "home_transforms",
            torch.tensor(
                (PIPER_HAND_HOME, PIPER_HAND_HOME, PIPER_VIEW_HOME),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _forward_kinematics(self, joint_positions: Tensor) -> Tensor:
        """计算 ``(..., 3 roles, 6 joints) -> (..., 3, 4, 4)`` 的PoE FK。"""
        if joint_positions.shape[-2:] != (self.num_roles, 6):
            raise ValueError(
                "Piper FK输入最后两维必须为(3, 6)，"
                f"当前为{tuple(joint_positions.shape[-2:])}。"
            )

        leading_shape = joint_positions.shape[:-1]
        identity3 = torch.eye(
            3,
            dtype=joint_positions.dtype,
            device=joint_positions.device,
        )
        identity4 = torch.eye(
            4,
            dtype=joint_positions.dtype,
            device=joint_positions.device,
        )
        transform = identity4.expand(leading_shape + (4, 4))

        axes = self.joint_axes.to(
            device=joint_positions.device,
            dtype=joint_positions.dtype,
        )
        anchors = self.joint_anchors.to(
            device=joint_positions.device,
            dtype=joint_positions.dtype,
        )

        for joint_index in range(6):
            theta = joint_positions[..., joint_index]
            axis = axes[joint_index]
            axis_skew = _skew(axis)
            axis_skew_squared = axis_skew @ axis_skew
            sin_theta = torch.sin(theta)[..., None, None]
            cos_theta = torch.cos(theta)[..., None, None]
            rotation = (
                identity3
                + sin_theta * axis_skew
                + (1.0 - cos_theta) * axis_skew_squared
            )
            translation = (
                (identity3 - rotation) @ anchors[joint_index][..., None]
            ).squeeze(-1)
            upper = torch.cat((rotation, translation[..., None]), dim=-1)
            bottom = torch.zeros(
                leading_shape + (1, 4),
                dtype=joint_positions.dtype,
                device=joint_positions.device,
            )
            bottom[..., 0, 3] = 1.0
            joint_transform = torch.cat((upper, bottom), dim=-2)
            transform = transform @ joint_transform

        home = self.home_transforms.to(
            device=joint_positions.device,
            dtype=joint_positions.dtype,
        )
        return transform @ home

    @classmethod
    def _select_arm_joints(cls, actions: Tensor) -> Tensor:
        """从20维动作中去掉左右夹爪并整理为三组6轴关节。"""
        if actions.shape[-1] != cls.action_dim:
            raise ValueError(
                f"末端位姿监督要求{cls.action_dim}维Piper三臂关节动作，"
                f"当前为{actions.shape[-1]}维。"
            )
        return torch.stack(
            [actions[..., start:end] for start, end in cls.joint_slices],
            dim=-2,
        )

    @staticmethod
    def _rotation_geodesic_angle(
        predicted_rotation: Tensor,
        target_rotation: Tensor,
    ) -> Tensor:
        """使用atan2稳定计算SO(3)测地角，返回弧度。"""
        relative = predicted_rotation.transpose(-1, -2) @ target_rotation
        skew_vector = torch.stack(
            (
                relative[..., 2, 1] - relative[..., 1, 2],
                relative[..., 0, 2] - relative[..., 2, 0],
                relative[..., 1, 0] - relative[..., 0, 1],
            ),
            dim=-1,
        )
        sin_angle = 0.5 * torch.linalg.vector_norm(skew_vector, dim=-1)
        cos_angle = 0.5 * (
            relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0
        )
        return torch.atan2(sin_angle, cos_angle.clamp(min=-1.0, max=1.0))

    @staticmethod
    def _masked_mean(values: Tensor, valid_roles: Tensor | None) -> Tensor:
        if valid_roles is None:
            return values.mean()
        weights = valid_roles.to(device=values.device, dtype=values.dtype)
        if weights.shape != values.shape:
            raise ValueError(
                "末端位姿loss掩码形状必须与[B, horizon, 3]误差一致，"
                f"mask={tuple(weights.shape)}，error={tuple(values.shape)}。"
            )
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(
        self,
        predicted_actions: Tensor,
        target_actions: Tensor,
        valid_roles: Tensor | None = None,
        *,
        compute_position: bool = True,
        compute_rotation: bool = True,
    ) -> dict[str, Tensor]:
        """计算位置/旋转平方损失及便于日志解释的物理量误差。"""
        if not compute_position and not compute_rotation:
            raise ValueError("末端位姿loss至少需要启用位置或旋转分支之一。")
        if predicted_actions.shape != target_actions.shape:
            raise ValueError(
                "预测动作与目标动作形状必须一致，"
                f"pred={tuple(predicted_actions.shape)}，"
                f"target={tuple(target_actions.shape)}。"
            )

        # FK和SO(3)损失固定使用float32；即使外层开启AMP，也避免半精度三角
        # 函数和矩阵连乘放大数值误差，同时梯度仍可传回原始预测张量。
        with torch.autocast(
            device_type=predicted_actions.device.type,
            enabled=False,
        ):
            predicted_joints = self._select_arm_joints(predicted_actions.float())
            target_joints = self._select_arm_joints(target_actions.float())
            predicted_pose = self._forward_kinematics(predicted_joints)
            target_pose = self._forward_kinematics(target_joints)

            losses = {}
            if compute_position:
                position_delta = (
                    predicted_pose[..., :3, 3] - target_pose[..., :3, 3]
                )
                position_squared_error = position_delta.square().sum(dim=-1)
                losses["eef_position_loss"] = self._masked_mean(
                    position_squared_error,
                    valid_roles,
                )
                losses["eef_position_error_m"] = self._masked_mean(
                    torch.sqrt(position_squared_error.clamp_min(0.0)),
                    valid_roles,
                ).detach()
            if compute_rotation:
                rotation_angle = self._rotation_geodesic_angle(
                    predicted_pose[..., :3, :3],
                    target_pose[..., :3, :3],
                )
                losses["eef_rotation_loss"] = self._masked_mean(
                    rotation_angle.square(),
                    valid_roles,
                )
                losses["eef_rotation_error_rad"] = self._masked_mean(
                    rotation_angle,
                    valid_roles,
                ).detach()

        return losses
