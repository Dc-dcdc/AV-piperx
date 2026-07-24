"""Quest 末端位姿动作的低延迟平滑与突跳保护。

输入、输出均采用 ``QuestControl.run()`` 的 23 维布局：

``[left xyz, left quat(wxyz), left gripper,
   right xyz, right quat(wxyz), right gripper,
   middle xyz, middle quat(wxyz)]``。
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class QuestPoseFilterConfig:
    """Quest 位姿滤波参数。"""

    enabled: bool = True
    fps: float = 25.0
    position_min_cutoff: float = 1.5
    position_beta: float = 0.08
    position_d_cutoff: float = 1.0
    rotation_alpha: float = 0.35
    max_position_step: float = 0.010
    max_rotation_step_deg: float = 5.0
    gripper_alpha: float = 0.4
    gripper_deadband: float = 0.02
    tracking_reset_gap: float = 0.25

    @classmethod
    def from_mapping(
        cls,
        values: Mapping | None,
        *,
        fps: float,
    ) -> "QuestPoseFilterConfig":
        values = values or {}
        return cls(
            enabled=bool(values.get("enabled", True)),
            fps=float(fps),
            position_min_cutoff=float(values.get("position_min_cutoff", 1.5)),
            position_beta=float(values.get("position_beta", 0.08)),
            position_d_cutoff=float(values.get("position_d_cutoff", 1.0)),
            rotation_alpha=float(values.get("rotation_alpha", 0.35)),
            max_position_step=float(values.get("max_position_step", 0.010)),
            max_rotation_step_deg=float(values.get("max_rotation_step_deg", 5.0)),
            gripper_alpha=float(values.get("gripper_alpha", 0.4)),
            gripper_deadband=float(values.get("gripper_deadband", 0.02)),
            tracking_reset_gap=float(values.get("tracking_reset_gap", 0.25)),
        )

    def __post_init__(self) -> None:
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}.")
        if self.position_min_cutoff <= 0.0:
            raise ValueError("position_min_cutoff must be positive.")
        if self.position_beta < 0.0:
            raise ValueError("position_beta must be non-negative.")
        if self.position_d_cutoff <= 0.0:
            raise ValueError("position_d_cutoff must be positive.")
        if not 0.0 < self.rotation_alpha <= 1.0:
            raise ValueError("rotation_alpha must be in (0, 1].")
        if self.max_position_step < 0.0:
            raise ValueError("max_position_step must be non-negative.")
        if self.max_rotation_step_deg < 0.0:
            raise ValueError("max_rotation_step_deg must be non-negative.")
        if not 0.0 < self.gripper_alpha <= 1.0:
            raise ValueError("gripper_alpha must be in (0, 1].")
        if not 0.0 <= self.gripper_deadband <= 1.0:
            raise ValueError("gripper_deadband must be in [0, 1].")
        if self.tracking_reset_gap < 0.0:
            raise ValueError("tracking_reset_gap must be non-negative.")

    def to_dict(self) -> dict[str, bool | float]:
        return asdict(self)


def _smoothing_alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _unit_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quaternion_angle_rad(first: np.ndarray, second: np.ndarray) -> float:
    """返回两个 wxyz 四元数代表的最短旋转夹角。"""

    first = _unit_quat_wxyz(first)
    second = _unit_quat_wxyz(second)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def quaternion_slerp_wxyz(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """沿最短路径对 wxyz 四元数做球面线性插值。"""

    first = _unit_quat_wxyz(first)
    second = _unit_quat_wxyz(second)
    fraction = float(np.clip(fraction, 0.0, 1.0))

    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        return _unit_quat_wxyz(first + fraction * (second - first))

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    first_weight = math.sin((1.0 - fraction) * theta) / sin_theta
    second_weight = math.sin(fraction * theta) / sin_theta
    return _unit_quat_wxyz(first_weight * first + second_weight * second)


class _OneEuroVectorFilter:
    """对一个固定长度向量使用共享自适应截止频率的 One Euro Filter。"""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.raw_value: np.ndarray | None = None
        self.filtered_value: np.ndarray | None = None
        self.filtered_derivative: np.ndarray | None = None

    def reset(self) -> None:
        self.raw_value = None
        self.filtered_value = None
        self.filtered_derivative = None

    def reset_derivative(self, raw_value: np.ndarray) -> None:
        """断流恢复时清除旧速度，但保留滤波输出作为安全限幅基准。"""

        self.raw_value = np.asarray(raw_value, dtype=np.float64).copy()
        if self.filtered_value is None:
            self.filtered_value = self.raw_value.copy()
        self.filtered_derivative = np.zeros_like(self.raw_value)

    def filter(self, value: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.raw_value is None or self.filtered_value is None:
            self.raw_value = value.copy()
            self.filtered_value = value.copy()
            self.filtered_derivative = np.zeros_like(value)
            return value.copy()

        derivative = (value - self.raw_value) / dt
        derivative_alpha = _smoothing_alpha(self.d_cutoff, dt)
        self.filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self.filtered_derivative
        )
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.filtered_derivative))
        value_alpha = _smoothing_alpha(cutoff, dt)
        self.filtered_value = (
            value_alpha * value
            + (1.0 - value_alpha) * self.filtered_value
        )
        self.raw_value = value.copy()
        return self.filtered_value.copy()

    def replace_filtered_value(self, value: np.ndarray) -> None:
        self.filtered_value = np.asarray(value, dtype=np.float64).copy()


class QuestPoseActionFilter:
    """平滑 QuestControl 的 23 维位姿动作，并限制相邻控制帧突跳。"""

    POSE_LAYOUT = {
        "left": (slice(0, 3), slice(3, 7), 7),
        "right": (slice(8, 11), slice(11, 15), 15),
        "middle": (slice(16, 19), slice(19, 23), None),
    }

    def __init__(self, config: QuestPoseFilterConfig) -> None:
        self.config = config
        self._nominal_dt = 1.0 / config.fps
        self._position_filters = {
            name: _OneEuroVectorFilter(
                config.position_min_cutoff,
                config.position_beta,
                config.position_d_cutoff,
            )
            for name in self.POSE_LAYOUT
        }
        self._rotation_values: dict[str, np.ndarray] = {}
        self._gripper_values: dict[str, float] = {}
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        """清空一条 episode 的全部历史状态。"""

        for position_filter in self._position_filters.values():
            position_filter.reset()
        self._rotation_values.clear()
        self._gripper_values.clear()
        self._last_timestamp = None

    def filter(self, pose_action: np.ndarray, *, timestamp: float | None = None) -> np.ndarray:
        pose_action = np.asarray(pose_action, dtype=np.float64).reshape(-1)
        if pose_action.size != 23:
            raise ValueError(f"pose_action must contain 23 values, got {pose_action.size}.")
        if not np.all(np.isfinite(pose_action)):
            raise ValueError("pose_action contains NaN or infinite values.")

        output = pose_action.copy()
        if not self.config.enabled:
            return output

        now = time.monotonic() if timestamp is None else float(timestamp)
        dt = self._nominal_dt
        tracking_recovered = False
        if self._last_timestamp is not None:
            elapsed = now - self._last_timestamp
            if elapsed > 0.0:
                dt = elapsed
            tracking_recovered = (
                self.config.tracking_reset_gap > 0.0
                and elapsed > self.config.tracking_reset_gap
            )
        self._last_timestamp = now

        # 避免系统时钟异常或长时间断流让一次滤波跨越过大的 dt。
        if not np.isfinite(dt) or dt <= 0.0 or tracking_recovered:
            dt = self._nominal_dt

        for name, (position_slice, rotation_slice, gripper_index) in self.POSE_LAYOUT.items():
            raw_position = pose_action[position_slice]
            position_filter = self._position_filters[name]
            previous_position = (
                None
                if position_filter.filtered_value is None
                else position_filter.filtered_value.copy()
            )
            if tracking_recovered:
                position_filter.reset_derivative(raw_position)
            filtered_position = position_filter.filter(raw_position, dt)
            filtered_position = self._limit_position_step(previous_position, filtered_position)
            position_filter.replace_filtered_value(filtered_position)
            output[position_slice] = filtered_position

            raw_rotation = _unit_quat_wxyz(pose_action[rotation_slice])
            previous_rotation = self._rotation_values.get(name)
            if previous_rotation is None:
                filtered_rotation = raw_rotation
            else:
                filtered_rotation = quaternion_slerp_wxyz(
                    previous_rotation,
                    raw_rotation,
                    self.config.rotation_alpha,
                )
                filtered_rotation = self._limit_rotation_step(
                    previous_rotation,
                    filtered_rotation,
                )
            self._rotation_values[name] = filtered_rotation.copy()
            output[rotation_slice] = filtered_rotation

            if gripper_index is not None:
                raw_gripper = float(np.clip(pose_action[gripper_index], 0.0, 1.0))
                previous_gripper = self._gripper_values.get(name)
                if previous_gripper is None:
                    filtered_gripper = raw_gripper
                elif abs(raw_gripper - previous_gripper) < self.config.gripper_deadband:
                    filtered_gripper = previous_gripper
                else:
                    filtered_gripper = (
                        self.config.gripper_alpha * raw_gripper
                        + (1.0 - self.config.gripper_alpha) * previous_gripper
                    )
                self._gripper_values[name] = filtered_gripper
                output[gripper_index] = filtered_gripper

        return output

    def _limit_position_step(
        self,
        previous: np.ndarray | None,
        target: np.ndarray,
    ) -> np.ndarray:
        if previous is None or self.config.max_position_step <= 0.0:
            return target
        delta = target - previous
        distance = float(np.linalg.norm(delta))
        if distance <= self.config.max_position_step:
            return target
        return previous + delta * (self.config.max_position_step / max(distance, 1.0e-12))

    def _limit_rotation_step(
        self,
        previous: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        max_angle = math.radians(self.config.max_rotation_step_deg)
        if max_angle <= 0.0:
            return target
        angle = quaternion_angle_rad(previous, target)
        if angle <= max_angle:
            return target
        return quaternion_slerp_wxyz(previous, target, max_angle / max(angle, 1.0e-12))


__all__ = [
    "QuestPoseActionFilter",
    "QuestPoseFilterConfig",
    "quaternion_angle_rad",
    "quaternion_slerp_wxyz",
]
