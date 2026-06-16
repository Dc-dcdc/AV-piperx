"""QuestControl 位姿动作到 MuJoCo 关节动作的 IK 求解器。

左右臂使用 GradIK，中间臂使用 DiffIK。外部主要调用
``PoseActionIKSolver.pose2joint()``，将 ``QuestControl.run()`` 返回的
23 维位姿动作转换为 ``env.step()`` 需要的关节动作。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
ENV_DIR = ROOT_DIR / "env"

for path in (ROOT_DIR, CURRENT_DIR, ENV_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/dppo_numba_cache")

from headset_utils import HeadsetData
from env.constants import LEFT_EEF_SITE, MIDDLE_ARM_POSE, MIDDLE_EEF_SITE, RIGHT_EEF_SITE, SIM_DT
from env.diff_ik import DiffIK
from env.grad_ik import GradIK
from transform_utils import mat2quat, xyzw_to_wxyz


@dataclass
class ArmIKState:
    name: str
    joints: list
    eef_site: object
    ik: DiffIK | GradIK
    action_slice: slice
    position_scale: float
    max_delta: float
    eef_anchor_pos: np.ndarray
    eef_anchor_quat: np.ndarray
    target_pos: np.ndarray
    target_quat: np.ndarray
    active: bool = False
    gripper_index: int | None = None
    last_q_current: np.ndarray | None = None
    last_q_target: np.ndarray | None = None
    last_q_error: np.ndarray | None = None


class PoseActionIKSolver:
    """Convert QuestControl 23-D pose actions to env.step joint actions."""

    POSE_LAYOUT = {
        "left": (slice(0, 3), slice(3, 7), 7),         # 左臂位置、四元素、夹爪切片
        "right": (slice(8, 11), slice(11, 15), 15),
        "middle": (slice(16, 19), slice(19, 23), None),
    }
    QUEST_SOURCE_BY_ARM = {"left": "left", "right": "right", "middle": "head"}
    HAND_IK_POSITION_WEIGHT = 500.0
    HAND_IK_ROTATION_WEIGHT = 100.0
    HAND_IK_JOINT_DISPLACEMENT_WEIGHT = 50.0
    HAND_IK_MAX_ROT_DIFF = 0.8
    HAND_IK_JOINT_P = 0.9
    HAND_IK_MAX_ITERATIONS = 50

    # 初始化 IK 求解器、末端 site 和各臂控制器。
    def __init__(
        self,
        sim_env,
        *,
        head_control: bool = True,
        lock_roll: bool = True,
        lock_pitch: bool = False,
        hand_position_scale: float = 1.0,
        hand_max_delta: float = 1.0,
        head_position_scale: float = 1.0,
        head_max_delta: float = 1.0,
        workspace_low: np.ndarray | list[float] | None = None,
        workspace_high: np.ndarray | list[float] | None = None,
        active_on_reset: bool = False,
    ) -> None:
        self.sim_env = sim_env
        self.physics = sim_env._physics
        self.head_control = bool(head_control)
        self.lock_roll = bool(lock_roll)
        self.lock_pitch = bool(lock_pitch)
        self.hand_position_scale = float(hand_position_scale)
        self.hand_max_delta = float(hand_max_delta)
        self.head_position_scale = float(head_position_scale)
        self.head_max_delta = float(head_max_delta)
        self.workspace_low = None if workspace_low is None else np.asarray(workspace_low, dtype=np.float64).reshape(3)
        self.workspace_high = None if workspace_high is None else np.asarray(workspace_high, dtype=np.float64).reshape(3)

        self._left_eef_site = self._find_site(LEFT_EEF_SITE)
        self._right_eef_site = self._find_site(RIGHT_EEF_SITE)
        self._middle_eef_site = self._find_site(MIDDLE_EEF_SITE)

        self._left_controller = GradIK(
            physics=self.physics,
            joints=sim_env._left_joints[:6],
            actuators=sim_env._left_actuators[:6],
            eef_site=self._left_eef_site,
            step_size=0.0001,
            min_cost_delta=1.0e-12,
            max_iterations=self.HAND_IK_MAX_ITERATIONS,
            position_weight=self.HAND_IK_POSITION_WEIGHT,
            rotation_weight=self.HAND_IK_ROTATION_WEIGHT,
            joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0], dtype=np.float64),
            joint_displacement_weight=np.array(6 * [self.HAND_IK_JOINT_DISPLACEMENT_WEIGHT], dtype=np.float64),
            position_threshold=0.001,
            rotation_threshold=0.001,
            max_pos_diff=0.1,
            max_rot_diff=self.HAND_IK_MAX_ROT_DIFF,
            joint_p=self.HAND_IK_JOINT_P,
        )
        self._right_controller = GradIK(
            physics=self.physics,
            joints=sim_env._right_joints[:6],
            actuators=sim_env._right_actuators[:6],
            eef_site=self._right_eef_site,
            step_size=0.0001,
            min_cost_delta=1.0e-12,
            max_iterations=self.HAND_IK_MAX_ITERATIONS,
            position_weight=self.HAND_IK_POSITION_WEIGHT,
            rotation_weight=self.HAND_IK_ROTATION_WEIGHT,
            joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0], dtype=np.float64),
            joint_displacement_weight=np.array(6 * [self.HAND_IK_JOINT_DISPLACEMENT_WEIGHT], dtype=np.float64),
            position_threshold=0.001,
            rotation_threshold=0.001,
            max_pos_diff=0.1,
            max_rot_diff=self.HAND_IK_MAX_ROT_DIFF,
            joint_p=self.HAND_IK_JOINT_P,
        )

        self._middle_controller = None
        if self.head_control:
            middle_joints = sim_env._middle_joints[:7]
            middle_q0 = np.asarray(MIDDLE_ARM_POSE[:7], dtype=np.float64)
            if middle_q0.shape != (len(middle_joints),):
                raise ValueError(f"Middle IK q0 length mismatch: got {middle_q0.shape[0]}, expected {len(middle_joints)}")

            k_null_template = np.asarray([20.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=np.float64)
            if len(middle_joints) > k_null_template.shape[0]:
                raise ValueError(f"Middle IK null gain supports {k_null_template.shape[0]} DoF, got {len(middle_joints)}")

            self._middle_controller = DiffIK(
                physics=self.physics,
                joints=middle_joints,
                actuators=sim_env._middle_actuators[:7],
                eef_site=self._middle_eef_site,
                k_pos=0.30,
                k_ori=0.20,
                damping=1.0e-4,
                k_null=k_null_template[: len(middle_joints)].copy(),
                q0=middle_q0,
                max_angvel=3.14,
                integration_dt=SIM_DT,
                iterations=10,
            )

        self.states = self._make_arm_states()
        self.eef_sites = {
            "left": self._left_eef_site,
            "right": self._right_eef_site,
            "middle": self._middle_eef_site,
        }
        self.reset(active=active_on_reset)

    # 查找 MuJoCo 模型中的指定末端 site，用来读取末端位姿
    def _find_site(self, site_name: str):
        site = self.sim_env._mjcf_root.find("site", site_name)
        if site is None:
            raise RuntimeError(f"Cannot find EEF site: {site_name}")
        return site

    # 读取末端 site 在世界系下的位置和四元数。
    def _read_eef_pose(self, eef_site) -> tuple[np.ndarray, np.ndarray]:
        site = self.physics.bind(eef_site)
        pos = site.xpos.copy().astype(np.float64)
        quat = xyzw_to_wxyz(mat2quat(site.xmat.reshape(3, 3))).astype(np.float64)
        return pos, quat

    # 构建每条机械臂的 IK 状态配置。
    def _make_arm_states(self) -> list[ArmIKState]:
        left_pos, left_quat = self._read_eef_pose(self._left_eef_site)
        right_pos, right_quat = self._read_eef_pose(self._right_eef_site)
        states = [
            ArmIKState(
                name="left",  
                joints=self.sim_env._left_joints[:6],   # 左臂参与 IK 的 6 个关节。
                eef_site=self._left_eef_site,           # 末端 site。
                ik=self._left_controller,               # 使用的 GradIK 求解器。
                action_slice=slice(0, 6),               # 关节结果写入 action 的范围。
                position_scale=self.hand_position_scale,# 位移缩放系数。
                max_delta=self.hand_max_delta,          # 相对锚点的最大位移。
                eef_anchor_pos=left_pos.copy(),         # 末端锚点位置。
                eef_anchor_quat=left_quat.copy(),       # 末端锚点姿态。
                target_pos=left_pos.copy(),             # 当前目标位置。
                target_quat=left_quat.copy(),           # 当前目标姿态。
                gripper_index=6,                        # 夹爪写入 action 的下标。
            ),
            ArmIKState(
                name="right",
                joints=self.sim_env._right_joints[:6],
                eef_site=self._right_eef_site,
                ik=self._right_controller,
                action_slice=slice(7, 13),
                position_scale=self.hand_position_scale,
                max_delta=self.hand_max_delta,
                eef_anchor_pos=right_pos.copy(),
                eef_anchor_quat=right_quat.copy(),
                target_pos=right_pos.copy(),
                target_quat=right_quat.copy(),
                gripper_index=13,
            ),
        ]

        if self.head_control and self._middle_controller is not None:
            middle_pos, middle_quat = self._read_eef_pose(self._middle_eef_site)
            states.append(
                ArmIKState(
                    name="middle",
                    joints=self.sim_env._middle_joints[:7],
                    eef_site=self._middle_eef_site,
                    ik=self._middle_controller,
                    action_slice=slice(14, 21),
                    position_scale=self.head_position_scale,
                    max_delta=self.head_max_delta,
                    eef_anchor_pos=middle_pos.copy(),
                    eef_anchor_quat=middle_quat.copy(),
                    target_pos=middle_pos.copy(),
                    target_quat=middle_quat.copy(),
                    gripper_index=None,
                )
            )
        return states

    # 重置各臂锚点、目标位姿和激活状态。
    def reset(self, *, active: bool = False) -> None:
        for state in self.states:
            pos, quat = self._read_eef_pose(state.eef_site)
            state.eef_anchor_pos = pos.copy()
            state.eef_anchor_quat = quat.copy()
            state.target_pos = pos.copy()
            state.target_quat = quat.copy()
            state.active = bool(active)
            state.last_q_current = None
            state.last_q_target = None
            state.last_q_error = None

    # 判断当前 Quest 数据是否足够用于开始锚定。
    def can_anchor_from_data(self, data: HeadsetData, *, allow_partial: bool = False) -> bool:
        readiness = [self._quest_source_ready(data, self.QUEST_SOURCE_BY_ARM[state.name]) for state in self.states]
        return any(readiness) if allow_partial else all(readiness)

    # 根据可用 Quest 数据激活对应机械臂并刷新锚点。
    def activate_from_data(self, data: HeadsetData, *, require_all: bool = True) -> int:
        missing = sorted(
            {
                self.QUEST_SOURCE_BY_ARM[state.name]
                for state in self.states
                if not self._quest_source_ready(data, self.QUEST_SOURCE_BY_ARM[state.name])
            }
        )
        if missing and require_all:
            details = []
            for source in missing:
                pos, quat = self._quest_pose(data, source)
                details.append(
                    f"{source}: pos={np.round(pos, 4).tolist()} "
                    f"quat={np.round(quat, 4).tolist()} pos_norm={np.linalg.norm(pos):.4f}"
                )
            print(f"Cannot anchor yet: missing usable pose for {', '.join(missing)}. {' | '.join(details)}")
            return 0

        active_count = 0
        for state in self.states:
            source = self.QUEST_SOURCE_BY_ARM[state.name]
            if not self._quest_source_ready(data, source):
                state.active = False
                continue

            eef_pos, eef_quat = self._read_eef_pose(state.eef_site)
            state.eef_anchor_pos = eef_pos.copy()
            state.eef_anchor_quat = eef_quat.copy()
            state.target_pos = eef_pos.copy()
            state.target_quat = eef_quat.copy()
            state.active = True
            active_count += 1

        if active_count:
            summary = ", ".join(f"{state.name}:{np.round(state.eef_anchor_pos, 4).tolist()}" for state in self.states if state.active)
            print(f"Quest IK anchored {active_count} arm(s). EEF anchors: {summary}")
        return active_count

    # 读取当前三条机械臂的末端位姿。
    def current_three_arm_poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        poses = []
        for name in ("left", "right", "middle"):
            pos, quat = self._read_eef_pose(self.eef_sites[name])
            poses.append(np.concatenate([pos, quat]).astype(np.float64))
        return poses[0], poses[1], poses[2]

    # 计算各激活机械臂目标姿态与当前实际 site 姿态之间的角度误差。
    def rotation_error_degrees(self) -> dict[str, float]:
        errors = {}
        for state in self.states:
            if not state.active:
                continue
            _actual_pos, actual_quat = self._read_eef_pose(state.eef_site)
            target_quat = self._unit_quat(state.target_quat, default_w_last=False)
            actual_quat = self._unit_quat(actual_quat, default_w_last=False)
            quat_dot = abs(float(np.dot(target_quat, actual_quat)))
            quat_dot = float(np.clip(quat_dot, -1.0, 1.0))
            errors[state.name] = float(np.degrees(2.0 * np.arccos(quat_dot)))
        return errors

    # 拆分姿态误差：目标到实际、目标到 IK 解、IK 解到实际。
    def rotation_error_breakdown_degrees(self) -> dict[str, dict[str, float]]:
        errors = {}
        for state in self.states:
            if not state.active:
                continue

            _actual_pos, actual_quat = self._read_eef_pose(state.eef_site)
            target_quat = self._unit_quat(state.target_quat, default_w_last=False)
            actual_quat = self._unit_quat(actual_quat, default_w_last=False)
            values = {
                "target_actual": self._quat_angle_degrees(target_quat, actual_quat),
            }

            if state.last_q_target is not None and hasattr(state.ik, "fk_fn"):
                command_mat = state.ik.fk_fn(state.last_q_target.astype(np.float64))
                command_quat = xyzw_to_wxyz(mat2quat(command_mat[:3, :3])).astype(np.float64)
                command_quat = self._unit_quat(command_quat, default_w_last=False)
                values["target_ik"] = self._quat_angle_degrees(target_quat, command_quat)
                values["ik_actual"] = self._quat_angle_degrees(command_quat, actual_quat)

            errors[state.name] = values
        return errors

    # 将 QuestControl 输出的位姿动作转换为 env.step 的关节动作。
    def pose2joint(
        self,
        pose_action: np.ndarray,
        obs: dict | None = None,
        *,
        current_action: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        if obs is not None:
            action = np.asarray(obs["agent_pos"], dtype=np.float64).copy()
        elif current_action is not None:
            action = np.asarray(current_action, dtype=np.float64).copy()
        else:
            action = self._current_agent_pos()

        pose_action = np.asarray(pose_action, dtype=np.float64).reshape(-1)
        if pose_action.size < 23:
            raise ValueError(f"pose_action length must be at least 23, got {pose_action.size}")

        active_count = 0
        for state in self.states:
            if not state.active:
                continue

            pos_slice, quat_slice, gripper_pose_index = self.POSE_LAYOUT[state.name]
            target_pos = np.asarray(pose_action[pos_slice], dtype=np.float64)
            target_delta = (target_pos - state.eef_anchor_pos) * state.position_scale
            delta_norm = float(np.linalg.norm(target_delta))
            if state.max_delta > 0.0 and delta_norm > state.max_delta:
                target_delta *= state.max_delta / max(delta_norm, 1e-8)
            state.target_pos = self._apply_workspace(state.eef_anchor_pos + target_delta)

            state.target_quat = self._unit_quat(np.asarray(pose_action[quat_slice], dtype=np.float64), default_w_last=False)

            # 中间臂可按配置锁定头显相对初始姿态中的 pitch/roll。
            if state.name == "middle" and (self.lock_roll or self.lock_pitch):
                state.target_quat = self._middle_quat_with_locked_axes(state.target_quat, state.eef_anchor_quat)

            if state.gripper_index is not None and gripper_pose_index is not None:
                close_value = float(np.clip(pose_action[gripper_pose_index], 0.0, 1.0))
                action[state.gripper_index] = 1.0 - close_value

            q_current = self.physics.bind(state.joints).qpos.copy().astype(np.float64)
            q_target = state.ik.run(
                q_current,
                state.target_pos.astype(np.float64),
                state.target_quat.astype(np.float64),
            )
            state.last_q_current = q_current.copy()
            state.last_q_target = q_target.copy()
            state.last_q_error = q_target - q_current
            action[state.action_slice] = q_target
            active_count += 1

        return action, active_count

    # 将目标位置限制在配置的工作空间范围内。
    def _apply_workspace(self, target_pos: np.ndarray) -> np.ndarray:
        if self.workspace_low is None or self.workspace_high is None:
            return target_pos
        return np.clip(target_pos, self.workspace_low, self.workspace_high)

    # 读取当前环境关节状态作为动作基准。
    def _current_agent_pos(self) -> np.ndarray:
        left_qpos = self.physics.bind(self.sim_env._left_joints).qpos.copy()
        right_qpos = self.physics.bind(self.sim_env._right_joints).qpos.copy()
        if hasattr(self.sim_env, "left_gripper_norm_fn"):
            left_qpos[6] = self.sim_env.left_gripper_norm_fn(left_qpos[6])
        if hasattr(self.sim_env, "right_gripper_norm_fn"):
            right_qpos[6] = self.sim_env.right_gripper_norm_fn(right_qpos[6])

        if self.head_control:
            middle_qpos = self.physics.bind(self.sim_env._middle_joints).qpos.copy()
            return np.concatenate([left_qpos, right_qpos, middle_qpos]).astype(np.float64)
        return np.concatenate([left_qpos, right_qpos]).astype(np.float64)

    # 取出指定 Quest 源的位姿。
    @staticmethod
    def _quest_pose(data: HeadsetData, source: str) -> tuple[np.ndarray, np.ndarray]:
        if source == "head":
            return data.h_pos.copy(), data.h_quat.copy()
        if source == "left":
            return data.l_pos.copy(), data.l_quat.copy()
        if source == "right":
            return data.r_pos.copy(), data.r_quat.copy()
        raise ValueError(f"Unsupported Quest source: {source}")

    # 判断指定 Quest 源是否有可用位姿。
    @classmethod
    def _quest_source_ready(cls, data: HeadsetData, source: str) -> bool:
        pos, quat = cls._quest_pose(data, source)
        return bool(np.linalg.norm(pos) > 1e-6 and np.linalg.norm(quat) > 1e-6)

    # 归一化四元数，输入无效时返回默认值。
    @staticmethod
    def _unit_quat(quat: np.ndarray, *, default_w_last: bool) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0] if default_w_last else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm

    # 计算两个 wxyz 四元数之间的最短旋转角，单位为度。
    @staticmethod
    def _quat_angle_degrees(quat_a_wxyz: np.ndarray, quat_b_wxyz: np.ndarray) -> float:
        quat_dot = abs(float(np.dot(quat_a_wxyz, quat_b_wxyz)))
        quat_dot = float(np.clip(quat_dot, -1.0, 1.0))
        return float(np.degrees(2.0 * np.arccos(quat_dot)))

    # 根据配置锁定中间臂相对初始姿态的 pitch/roll。
    def _middle_quat_with_locked_axes(self, target_quat_wxyz: np.ndarray, anchor_quat_wxyz: np.ndarray) -> np.ndarray:
        target_quat_wxyz = self._unit_quat(target_quat_wxyz, default_w_last=False)
        anchor_quat_wxyz = self._unit_quat(anchor_quat_wxyz, default_w_last=False)

        target_rot = R.from_quat(
            np.array([target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]], dtype=np.float64)
        )
        anchor_rot = R.from_quat(
            np.array([anchor_quat_wxyz[1], anchor_quat_wxyz[2], anchor_quat_wxyz[3], anchor_quat_wxyz[0]], dtype=np.float64)
        )
        yaw, pitch, roll = (anchor_rot.inv() * target_rot).as_euler("zyx", degrees=False)
        if self.lock_pitch:
            pitch = 0.0
        if self.lock_roll:
            roll = 0.0
        return xyzw_to_wxyz((anchor_rot * R.from_euler("zyx", [yaw, pitch, roll], degrees=False)).as_quat().astype(np.float64))
