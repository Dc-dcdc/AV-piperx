from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/av_piper_numba_cache")

from headset_utils import HeadsetData, HeadsetFeedback, convert_right_to_left_coordinates
from transform_utils import (
    align_rotation_to_z_axis, # 将旋转矩阵对齐到Z轴方向
    mat2pose,                 # 把 4x4 齐次变换矩阵转换为 position + quaternion
    pose2mat,                 # 把 position + quaternion 转换为 4x4 齐次变换矩阵
    transform_coordinates,    # 坐标系变换
    within_pose_threshold,    # 判断两个位姿误差是否在阈值内
    wxyz_to_xyzw,             # 把四元数从 MuJoCo 常用的 [w, x, y, z] 转成 Unity/Scipy 常见的 [x, y, z, w]
    xyzw_to_wxyz,             # 把四元数从 [x, y, z, w] 转回 [w, x, y, z]
)


__all__ = ["QuestControl", "QuestHeadControl"]


class QuestControl:
    """把 Quest3 位姿数据转换成三臂 MuJoCo 环境可执行的 action。

    输入的机械臂位姿使用环境里的格式: [x, y, z, qw, qx, qy, qz]。
    输出 action 共 23 维，拼接格式:
        left xyz(3), left quat wxyz(4), left gripper(1),
        right xyz(3), right quat wxyz(4), right gripper(1),
        middle xyz(3), middle quat wxyz(4)
    """

    def __init__(
        self,
        start_ctrl_position_threshold: float = 0.06,     # 启动阶段，控制器对应机械臂允许的位置误差阈值
        start_ctrl_rotation_threshold: float = 0.4,      # 启动阶段允许的旋转误差
        start_head_position_threshold: float = 0.03,     # 启动阶段头显位置误差
        start_head_rotation_threshold: float = 0.2,      # 启动阶段头显旋转误差
        ctrl_position_threshold: float = 0.04,           # 运行阶段机械臂允许的误差
        ctrl_rotation_threshold: float = 0.3,
        head_position_threshold: float = 0.05,
        head_rotation_threshold: float = 0.3,
        use_head_control: bool = True,
        use_individual_hand_anchors: bool = True,       # True 时左右臂分别使用各自手柄初始位姿作为相对控制锚点。
    ):
        self.start_headset_mat: np.ndarray | None = None
        self.start_middle_arm_mat: np.ndarray | None = None
        self.start_left_ctrl_mat: np.ndarray | None = None
        self.start_right_ctrl_mat: np.ndarray | None = None
        self.start_left_arm_mat: np.ndarray | None = None
        self.start_right_arm_mat: np.ndarray | None = None
        self.started = False

        self.start_ctrl_position_threshold = start_ctrl_position_threshold
        self.start_ctrl_rotation_threshold = start_ctrl_rotation_threshold
        self.start_head_position_threshold = start_head_position_threshold
        self.start_head_rotation_threshold = start_head_rotation_threshold
        self.ctrl_position_threshold = ctrl_position_threshold
        self.ctrl_rotation_threshold = ctrl_rotation_threshold
        self.head_position_threshold = head_position_threshold
        self.head_rotation_threshold = head_rotation_threshold

        self.use_head_control = bool(use_head_control)
        self.use_individual_hand_anchors = bool(use_individual_hand_anchors)

    def reset(self):
        self.start_headset_mat = None
        self.start_middle_arm_mat = None
        self.start_left_ctrl_mat = None
        self.start_right_ctrl_mat = None
        self.start_left_arm_mat = None
        self.start_right_arm_mat = None
        self.started = False

    def is_running(self) -> bool:
        return self.started

    @staticmethod
    def should_start(data: HeadsetData) -> bool:
        # Quest A 或 X 按键。
        return bool(data.r_button_one or data.l_button_one)

    @staticmethod
    def should_reset(data: HeadsetData) -> bool:
        # Quest B 或 Y 按键。
        return bool(data.r_button_two or data.l_button_two)

    # 记录所有设备锚点位置
    def start(
        self,
        data: HeadsetData,
        middle_arm_pose: np.ndarray,
        left_arm_pose: np.ndarray | None = None,
        right_arm_pose: np.ndarray | None = None,
    ):
        if self.started:
            return

        self.start_headset_mat = pose2mat(data.h_pos, data.h_quat).copy()
        self.start_middle_arm_mat = _robot_pose_to_mat(middle_arm_pose).copy()
        if self.use_individual_hand_anchors:
            if left_arm_pose is None or right_arm_pose is None:
                raise ValueError(
                    "use_individual_hand_anchors=True 时，start() 必须传入 left_arm_pose 和 right_arm_pose。"
            )

            self.start_left_ctrl_mat = pose2mat(data.l_pos, data.l_quat).copy()
            self.start_right_ctrl_mat = pose2mat(data.r_pos, data.r_quat).copy()
            self.start_left_arm_mat = _robot_pose_to_mat(left_arm_pose).copy()
            self.start_right_arm_mat = _robot_pose_to_mat(right_arm_pose).copy()

        self.started = True

    # 将quest设备的位姿从unity映射到mujoco中
    def run(
        self,
        data: HeadsetData,
        left_arm_pose: np.ndarray,
        right_arm_pose: np.ndarray,
        middle_arm_pose: np.ndarray,
    ) -> tuple[np.ndarray, HeadsetFeedback]:
        left_arm_mat = _robot_pose_to_mat(left_arm_pose)
        right_arm_mat = _robot_pose_to_mat(right_arm_pose)
        middle_arm_mat = _robot_pose_to_mat(middle_arm_pose)

        headset_mat = pose2mat(data.h_pos, data.h_quat)
        left_ctrl_mat = pose2mat(data.l_pos, data.l_quat)
        right_ctrl_mat = pose2mat(data.r_pos, data.r_quat)
        left_gripper = float(data.l_index_trigger)
        right_gripper = float(data.r_index_trigger)

        # 获取初始位姿，start后将不再变化，可进行reset
        start_headset_mat, start_middle_arm_mat = self._get_anchor_frames(headset_mat, middle_arm_mat)

        # 左右手柄分别相对各自启动位姿的变化，叠加到各自机械臂启动位姿。
        if self.use_individual_hand_anchors:
            if (
                self.start_left_ctrl_mat is None
                or self.start_right_ctrl_mat is None
                or self.start_left_arm_mat is None
                or self.start_right_arm_mat is None
            ):
                self.start_left_ctrl_mat = np.asarray(left_ctrl_mat, dtype=np.float64).copy()
                self.start_right_ctrl_mat = np.asarray(right_ctrl_mat, dtype=np.float64).copy()
                self.start_left_arm_mat = np.asarray(left_arm_mat, dtype=np.float64).copy()
                self.start_right_arm_mat = np.asarray(right_arm_mat, dtype=np.float64).copy()
            new_left_arm_mat = transform_coordinates(left_ctrl_mat, self.start_left_ctrl_mat, self.start_left_arm_mat)
            new_right_arm_mat = transform_coordinates(right_ctrl_mat, self.start_right_ctrl_mat, self.start_right_arm_mat)
        else: # 左右手柄相对一开始头显的位姿，映射到中间臂锚点。
            new_left_arm_mat = transform_coordinates(left_ctrl_mat, start_headset_mat, start_middle_arm_mat)
            new_right_arm_mat = transform_coordinates(right_ctrl_mat, start_headset_mat, start_middle_arm_mat)
        if self.use_head_control:
            new_middle_arm_mat = transform_coordinates(headset_mat, start_headset_mat, start_middle_arm_mat)
        else:
            new_middle_arm_mat = middle_arm_mat

        # 将新的中间臂位姿和左右臂位姿转换为位置和四元数的形式
        new_middle_arm_pos, new_middle_arm_quat = mat2pose(new_middle_arm_mat)
        new_left_arm_pos, new_left_arm_quat = mat2pose(new_left_arm_mat)
        new_right_arm_pos, new_right_arm_quat = mat2pose(new_right_arm_mat)
        # 将四元数从xyzw格式转换为wxyz格式，以便与mujoco环境中的位姿表示一致
        new_middle_arm_quat = xyzw_to_wxyz(new_middle_arm_quat)
        new_left_arm_quat = xyzw_to_wxyz(new_left_arm_quat)
        new_right_arm_quat = xyzw_to_wxyz(new_right_arm_quat)
        # 读取左右手控制器的扳机值作为夹持器的状态
        new_left_gripper = np.array([np.clip(float(left_gripper), 0.0, 1.0)], dtype=np.float64)
        new_right_gripper = np.array([np.clip(float(right_gripper), 0.0, 1.0)], dtype=np.float64)

        # 将所有的动作拼接成一个23维度的数组
        action = np.concatenate([
            new_left_arm_pos, new_left_arm_quat, new_left_gripper,
            new_right_arm_pos, new_right_arm_quat, new_right_gripper,
            new_middle_arm_pos, new_middle_arm_quat
        ])

        feedback = self._make_feedback(
            left_arm_mat=left_arm_mat,
            right_arm_mat=right_arm_mat,
            middle_arm_mat=middle_arm_mat,
            new_left_arm_mat=new_left_arm_mat,
            new_right_arm_mat=new_right_arm_mat,
            new_middle_arm_mat=new_middle_arm_mat,
            start_headset_mat=start_headset_mat,
            start_middle_arm_mat=start_middle_arm_mat,
        )
        return action, feedback

    def get_action(
        self,
        data: HeadsetData,
        left_arm_pose: np.ndarray,
        right_arm_pose: np.ndarray,
        middle_arm_pose: np.ndarray,
    ) -> np.ndarray:
        action, _feedback = self.run(data, left_arm_pose, right_arm_pose, middle_arm_pose)
        return action

    def _get_anchor_frames(
        self,
        headset_mat: np.ndarray,
        middle_arm_mat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 如果已经开始执行轨迹，则使用保存的起始位姿
        if self.started and self.start_headset_mat is not None and self.start_middle_arm_mat is not None:
            return self.start_headset_mat, self.start_middle_arm_mat

        # 否则将当前状态保存为起始位姿，计算增量时就一直为0，用户可以在开始控制前调整到合适的位姿
        aligned_headset_mat = np.eye(4)
        aligned_headset_mat[:3, :3] = align_rotation_to_z_axis(headset_mat[:3, :3])
        aligned_headset_mat[:3, 3] = headset_mat[:3, 3]

        aligned_middle_arm_mat = np.eye(4)
        aligned_middle_arm_mat[:3, :3] = align_rotation_to_z_axis(middle_arm_mat[:3, :3])
        aligned_middle_arm_mat[:3, 3] = middle_arm_mat[:3, 3]
        return aligned_headset_mat, aligned_middle_arm_mat

    def _make_feedback(
        self,
        left_arm_mat: np.ndarray,
        right_arm_mat: np.ndarray,
        middle_arm_mat: np.ndarray,
        new_left_arm_mat: np.ndarray,
        new_right_arm_mat: np.ndarray,
        new_middle_arm_mat: np.ndarray,
        start_headset_mat: np.ndarray,
        start_middle_arm_mat: np.ndarray,
    ) -> HeadsetFeedback:
        unity_left_arm_mat = transform_coordinates(left_arm_mat, start_middle_arm_mat, start_headset_mat)
        unity_right_arm_mat = transform_coordinates(right_arm_mat, start_middle_arm_mat, start_headset_mat)
        unity_middle_arm_mat = transform_coordinates(middle_arm_mat, start_middle_arm_mat, start_headset_mat)

        unity_left_arm_pos, unity_left_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_left_arm_mat))
        unity_right_arm_pos, unity_right_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_right_arm_mat))
        unity_middle_arm_pos, unity_middle_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_middle_arm_mat))

        feedback = HeadsetFeedback()
        feedback.info = ""
        feedback.head_out_of_sync = self._is_out_of_sync(
            middle_arm_mat,
            new_middle_arm_mat,
            self.head_position_threshold if self.started else self.start_head_position_threshold,
            self.head_rotation_threshold if self.started else self.start_head_rotation_threshold,
        )
        feedback.left_out_of_sync = self._is_out_of_sync(
            left_arm_mat,
            new_left_arm_mat,
            self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
            self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold,
        )
        feedback.right_out_of_sync = self._is_out_of_sync(
            right_arm_mat,
            new_right_arm_mat,
            self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
            self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold,
        )
        feedback.left_arm_position = unity_left_arm_pos
        feedback.left_arm_rotation = unity_left_arm_quat
        feedback.right_arm_position = unity_right_arm_pos
        feedback.right_arm_rotation = unity_right_arm_quat
        feedback.middle_arm_position = unity_middle_arm_pos
        feedback.middle_arm_rotation = unity_middle_arm_quat
        return feedback

    @staticmethod
    def _is_out_of_sync(
        current_mat: np.ndarray,
        target_mat: np.ndarray,
        position_threshold: float,
        rotation_threshold: float,
    ) -> bool:
        return not within_pose_threshold(
            current_mat[:3, 3],
            current_mat[:3, :3],
            target_mat[:3, 3],
            target_mat[:3, :3],
            position_threshold,
            rotation_threshold,
        )


class QuestHeadControl:
    """只用头显控制中间臂，输出 [middle xyz, middle quat wxyz]。"""


# 将位姿转为齐次矩阵
def _robot_pose_to_mat(robot_pose: np.ndarray) -> np.ndarray:
    robot_pose = np.asarray(robot_pose, dtype=np.float64).reshape(-1)
    if robot_pose.size < 7:
        raise ValueError("robot_pose 需要至少包含 [x, y, z, qw, qx, qy, qz] 7 个值。")
    return pose2mat(robot_pose[:3], wxyz_to_xyzw(robot_pose[3:7]))
