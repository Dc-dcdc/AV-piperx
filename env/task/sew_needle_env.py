import os

import numpy as np

from env.constants import XML_DIR
from env.task.sim_envs import GuidedVisionEnv


class SewNeedleEnv(GuidedVisionEnv):
    """PiperX 三臂缝合针穿引任务环境。"""

    def __init__(self, **kwargs):
        self.needle_position_ranges = self._validate_reset_position_ranges("needle_position_ranges", kwargs.pop("needle_position_ranges", [[0.15, 0.20], [0.00, 0.20], [0.00, 0.00]]))  # 针初始XYZ范围，单位m。
        self.wall_position_ranges = self._validate_reset_position_ranges("wall_position_ranges", kwargs.pop("wall_position_ranges", [[-0.025, 0.025], [-0.01, 0.11], [0.00, 0.00]]))  # 孔墙初始XYZ范围，单位m。
        self.enable_reward_debug = bool(kwargs.pop("enable_reward_debug", False))  # 是否记录奖励阶段、接触和距离信息。
        self.hole_y_threshold = float(kwargs.pop("hole_y_threshold", 0.018))  # 针穿孔时允许的Y轴孔径误差，单位m。
        self.hole_z_threshold = float(kwargs.pop("hole_z_threshold", 0.018))  # 针穿孔时允许的Z轴孔径误差，单位m。
        self.left_grasp_mark_threshold = float(kwargs.pop("left_grasp_mark_threshold", 0.06))  # 左夹爪到接针标记的最大距离，单位m。
        self.success_x_threshold = float(kwargs.pop("success_x_threshold", 0.015))  # 最终针中心允许的X轴误差，单位m。
        self.success_y_threshold = float(kwargs.pop("success_y_threshold", 0.05))  # 最终针中心允许的Y轴误差，单位m。
        self.success_z_clearance = float(kwargs.pop("success_z_clearance", 0.012))  # 最终针中心高于孔出口的安全余量，单位m。
        self.success_stable_steps = int(kwargs.pop("success_stable_steps", 5))  # 最终成功姿态需要连续保持的仿真步数。
        if self.success_stable_steps < 1:
            raise ValueError("success_stable_steps 必须大于等于 1")
        for name in (
            "hole_y_threshold",
            "hole_z_threshold",
            "left_grasp_mark_threshold",
            "success_x_threshold",
            "success_y_threshold",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.success_z_clearance < 0:
            raise ValueError("success_z_clearance 必须大于等于 0")
        xml_path = kwargs.pop("xml_path", os.path.join(XML_DIR, "task_sew_needle.xml"))  # MuJoCo任务XML路径。
        super().__init__(xml_path, **kwargs)

        self._needle_joint = self._mjcf_root.find("joint", "needle_joint")
        self._wall_joint = self._mjcf_root.find("joint", "wall_joint")
        self._left_gripper_site = self._mjcf_root.find("site", "left_gripper_control")
        self._right_gripper_site = self._mjcf_root.find("site", "right_gripper_control")
        required_elements = (
            ("needle_joint", self._needle_joint),
            ("wall_joint", self._wall_joint),
            ("left_gripper_control", self._left_gripper_site),
            ("right_gripper_control", self._right_gripper_site),
            ("needle", self._mjcf_root.find("geom", "needle")),
            ("needle_head", self._mjcf_root.find("geom", "needle_head")),
            ("needle_tail", self._mjcf_root.find("geom", "needle_tail")),
            ("needle_mark_1_4", self._mjcf_root.find("geom", "needle_mark_1_4")),
            ("needle_mark_3_4", self._mjcf_root.find("geom", "needle_mark_3_4")),
            ("hole_entrance", self._mjcf_root.find("geom", "hole_entrance")),
            ("hole_exit", self._mjcf_root.find("geom", "hole_exit")),
        )
        for name, element in required_elements:
            if element is None:
                raise ValueError(f"PiperX sew needle XML 缺少必要元素: {name}")

        self._prev_dists = {}
        self.reward_debug = None

    def _geom_or_body_name(self, geom_id: int) -> str:
        """匿名碰撞 geom 回退到所属 body，兼容 PiperX 手指模型。"""
        geom_name = self._physics.model.id2name(geom_id, "geom")
        if geom_name and not geom_name.startswith("//unnamed_geom_"):
            return geom_name
        body_id = int(self._physics.model.geom_bodyid[geom_id])
        body_name = self._physics.model.id2name(body_id, "body")
        return body_name or ""

    def _get_gripper_contact_flags(self) -> tuple[bool, bool]:
        """判断针是否同时接触左右夹爪各自的两根手指。"""
        objects_touching_needle = set()
        for i_contact in range(self._physics.data.ncon):
            geom_id_1 = self._physics.data.contact[i_contact].geom1
            geom_id_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._geom_or_body_name(geom_id_1)
            geom2 = self._geom_or_body_name(geom_id_2)
            if geom1 == "needle":
                objects_touching_needle.add(geom2)
            elif geom2 == "needle":
                objects_touching_needle.add(geom1)

        touch_left = (
            any(name.startswith("left_left") for name in objects_touching_needle)
            and any(name.startswith("left_right") for name in objects_touching_needle)
        )
        touch_right = (
            any(name.startswith("right_left") for name in objects_touching_needle)
            and any(name.startswith("right_right") for name in objects_touching_needle)
        )
        return touch_left, touch_right

    def _set_reward_debug(
        self,
        reward,
        curr_dists,
        touch_left,
        touch_right,
        reward_terms,
    ):
        if not self.enable_reward_debug:
            self.reward_debug = None
            return

        if self.is_success:
            stage = 6
            stage_name = "success"
        elif self.needle_completely_through:
            stage = 5
            stage_name = "needle_completely_through"
        elif self.right_released_after_handover:
            stage = 4
            stage_name = "right_released_after_handover"
        elif self.left_has_grasped:
            stage = 3
            stage_name = "left_has_grasped"
        elif self.needle_reached_exit:
            stage = 2
            stage_name = "needle_reached_exit"
        elif self.needle_start_through:
            stage = 1
            stage_name = "needle_start_through"
        else:
            stage = 0
            stage_name = "approach"
        self.reward_debug = {
            "stage": stage,
            "stage_name": stage_name,
            "reward_total": float(reward),
            "reward_terms": {
                key: float(value) for key, value in reward_terms.items()
            },
            "needle_start_through": bool(self.needle_start_through),
            "needle_reached_exit": bool(self.needle_reached_exit),
            "left_has_grasped": bool(self.left_has_grasped),
            "right_released_after_handover": bool(
                self.right_released_after_handover
            ),
            "needle_completely_through": bool(self.needle_completely_through),
            "needle_was_grasped": bool(self.needle_was_grasped),
            "right_has_grasped": bool(self.right_has_grasped),
            "success_stable_count": int(self.success_stable_count),
            "success_stable_steps": int(self.success_stable_steps),
            "is_success": bool(self.is_success),
            "touch_left": bool(touch_left),
            "touch_right": bool(touch_right),
            "right_to_needle_dist": float(curr_dists["dist_right_to_needle"]),
            "head_to_entrance_dist": float(curr_dists["dist_head_to_entrance"]),
            "head_to_exit_dist": float(curr_dists["dist_head_to_exit"]),
            "left_to_mark_dist": float(curr_dists["dist_left_to_mark"]),
            "tail_to_exit_dist": float(curr_dists["dist_tail_to_exit"]),
            "x_error": float(curr_dists["x_error"]),
            "y_error": float(curr_dists["y_error"]),
            "z_error": float(curr_dists["z_error"]),
            "head_entrance_y_error": float(curr_dists["head_entrance_y_error"]),
            "head_entrance_z_error": float(curr_dists["head_entrance_z_error"]),
            "head_exit_y_error": float(curr_dists["head_exit_y_error"]),
            "head_exit_z_error": float(curr_dists["head_exit_z_error"]),
            "tail_exit_y_error": float(curr_dists["tail_exit_y_error"]),
            "tail_exit_z_error": float(curr_dists["tail_exit_z_error"]),
        }

    @staticmethod
    def _crossed_negative_x_plane(prev_point, point, prev_plane, plane) -> bool:
        """判断关键点是否在相邻控制帧间沿负 X 方向穿过目标平面。"""

        prev_side = float(prev_point[0] - prev_plane[0])
        curr_side = float(point[0] - plane[0])
        return prev_side > 0.0 and curr_side <= 0.0

    def _inside_hole_aperture(self, point, hole) -> bool:
        """用独立的 Y/Z 阈值约束关键点确实位于孔径范围内。"""

        return bool(
            abs(float(point[1] - hole[1])) <= self.hole_y_threshold
            and abs(float(point[2] - hole[2])) <= self.hole_z_threshold
        )

    def _calculate_distances(self):
        """🌟 辅助函数：统一计算所有关键点的坐标与距离"""
        # 1. 提取物体坐标
        needle_head = self._physics.named.data.geom_xpos['needle_head']                #针头
        needle_tail = self._physics.named.data.geom_xpos['needle_tail']                #针尾
        needle_left_pos = self._physics.named.data.geom_xpos['needle_mark_1_4']        #左臂抓取标记点 1/4处
        needle_right_pos = self._physics.named.data.geom_xpos['needle_mark_3_4']       #右臂抓取标记点 3/4处

        hole_entrance = self._physics.named.data.geom_xpos['hole_entrance']            #进洞口
        hole_exit = self._physics.named.data.geom_xpos['hole_exit']                    #出洞口

        # 2. 计算复合中心点
        left_gripper_center = self._physics.bind(self._left_gripper_site).xpos.copy()
        right_gripper_center = self._physics.bind(self._right_gripper_site).xpos.copy()
        needle_center = (needle_head + needle_tail) / 2.0
        wall_center_x = (hole_entrance[0] + hole_exit[0]) / 2.0

        # 右臂初期靠近奖励使用“到整根针线段的距离”，避免只靠近针体但没靠近固定标记点时奖励反向。
        needle_vec = needle_tail - needle_head
        needle_len_sq = float(np.dot(needle_vec, needle_vec))
        if needle_len_sq > 1e-12:
            t = np.clip(
                np.dot(right_gripper_center - needle_head, needle_vec) / needle_len_sq,
                0.0,
                1.0,
            )
            closest_needle_pos = needle_head + t * needle_vec
        else:
            closest_needle_pos = needle_center
        dist_right_to_needle = np.linalg.norm(right_gripper_center - closest_needle_pos)

        # 3. 终极抬举误差解耦计算
        target_z_target = hole_exit[2] + self.success_z_clearance
        z_error = max(0.0, target_z_target - needle_center[2])
        x_error = abs(needle_center[0] - wall_center_x)
        y_error = abs(needle_center[1] - hole_exit[1])
        composite_error_dist = np.sqrt(x_error**2 + y_error**2 + z_error**2)

        # 4. 返回所有距离信息
        return {
            'head_pos': needle_head.copy(),
            'tail_pos': needle_tail.copy(),
            'entrance_pos': hole_entrance.copy(),
            'exit_pos': hole_exit.copy(),
            'needle_z': needle_center[2],
            'hole_z': hole_exit[2],

            # 强化学习需要计算差分的绝对距离
            'dist_right_to_mark': np.linalg.norm(right_gripper_center - needle_right_pos),
            'dist_right_to_needle': dist_right_to_needle,
            'dist_head_to_entrance': np.linalg.norm(needle_head - hole_entrance),
            'dist_head_to_exit': np.linalg.norm(needle_head - hole_exit),
            'dist_left_to_mark': np.linalg.norm(left_gripper_center - needle_left_pos),
            'dist_tail_to_exit': np.linalg.norm(needle_tail - hole_exit),
            'composite_error_dist': composite_error_dist,

            # 孔径约束使用分轴误差，避免仅凭较宽的三维球形距离误触发阶段。
            'head_entrance_y_error': abs(needle_head[1] - hole_entrance[1]),
            'head_entrance_z_error': abs(needle_head[2] - hole_entrance[2]),
            'head_exit_y_error': abs(needle_head[1] - hole_exit[1]),
            'head_exit_z_error': abs(needle_head[2] - hole_exit[2]),
            'tail_exit_y_error': abs(needle_tail[1] - hole_exit[1]),
            'tail_exit_z_error': abs(needle_tail[2] - hole_exit[2]),

            # 保存误差用于通关判定
            'x_error': x_error,
            'y_error': y_error,
            'z_error': z_error
        }

    def reset(self, seed=None, options=None) -> tuple:
        """重置机器人、针和孔墙的初始位置。"""
        super().reset(seed=seed, options=options)

        needle_position = self._sample_reset_position(
            self.needle_position_ranges
        )
        wall_position = self._sample_reset_position(
            self.wall_position_ranges
        )

        self._set_free_joint_reset_pose(
            self._needle_joint,
            needle_position,
        )
        self._set_free_joint_reset_pose(
            self._wall_joint,
            wall_position,
        )
        self._physics.forward()
        self.needle_reached_exit = False                 # 针头穿墙标志位
        self.left_has_grasped = False                    # 左臂成功接针标志位
        self.needle_was_grasped = False                  # 针是否曾经被任一夹爪抓住过
        self.right_has_grasped = False                   # 右臂是否曾经可靠抓住针
        self.right_released_after_handover = False       # 左手接针后右手是否完成释放
        self.needle_completely_through = False           # 针完全过孔标志位
        self.needle_start_through = False                # 针开始过孔标志位
        self.success_stable_count = 0                    # 最终安全姿态连续满足帧数
        self.reward_debug = None
        self._prev_dists = self._calculate_distances()   # 记录物理引擎第一帧的距离，作为差分计算的起点
        observation = self.get_obs()
        info = {"message": "SewNeedle env reset."}
        return observation, info

    def get_reward(self):
        touch_left_gripper, touch_right_gripper = self._get_gripper_contact_flags()
        curr_dists = self._calculate_distances()
        prev_dists = self._prev_dists or curr_dists
        reward = 0.0
        reward_terms = {}

        def add_reward(name, value):
            nonlocal reward
            value = float(value)
            reward += value
            if self.enable_reward_debug:
                reward_terms[name] = reward_terms.get(name, 0.0) + value

        def add_progress(name, distance_key):
            progress = float(
                prev_dists[distance_key] - curr_dists[distance_key]
            )
            add_reward(name, 100.0 * progress)

        # 所有静止状态都应承担轻微时间成本；连续 shaping 只奖励真实距离进展。
        add_reward("step_penalty", -0.1)

        needle_is_grasped = touch_left_gripper or touch_right_gripper
        if needle_is_grasped:
            self.needle_was_grasped = True
        if touch_right_gripper and not self.right_has_grasped:
            self.right_has_grasped = True
            add_reward("right_first_grasp", 20.0)

        # 掉落失败只在此处处理，避免后半程的强惩罚被前置 return 吞掉。
        if (
            self.needle_was_grasped
            and not needle_is_grasped
            and curr_dists['needle_z'] < (curr_dists['hole_z'] - 0.03)
        ):
            drop_penalty = -500.0 if self.left_has_grasped else -100.0
            add_reward(
                "penalty_drop_after_handover"
                if self.left_has_grasped
                else "penalty_drop_before_handover",
                drop_penalty,
            )
            self.is_success = False
            self.terminated = True
            self.success_stable_count = 0
            self._prev_dists = curr_dists
            self._set_reward_debug(
                reward,
                curr_dists,
                touch_left_gripper,
                touch_right_gripper,
                reward_terms,
            )
            return float(reward)

        # 阶段 1：右手抓针后，针头沿负 X 方向穿过入口平面且位于孔径内。
        if (
            not self.needle_start_through
            and self.right_has_grasped
            and touch_right_gripper
            and self._crossed_negative_x_plane(
                prev_dists['head_pos'],
                curr_dists['head_pos'],
                prev_dists['entrance_pos'],
                curr_dists['entrance_pos'],
            )
            and self._inside_hole_aperture(
                curr_dists['head_pos'], curr_dists['entrance_pos']
            )
        ):
            self.needle_start_through = True
            add_reward("needle_entered_hole", 25.0)

        # 阶段 2：继续由右手抓持，针头沿负 X 方向穿过出口平面。
        if (
            self.needle_start_through
            and not self.needle_reached_exit
            and touch_right_gripper
            and self._crossed_negative_x_plane(
                prev_dists['head_pos'],
                curr_dists['head_pos'],
                prev_dists['exit_pos'],
                curr_dists['exit_pos'],
            )
            and self._inside_hole_aperture(
                curr_dists['head_pos'], curr_dists['exit_pos']
            )
        ):
            self.needle_reached_exit = True
            add_reward("needle_head_reached_exit", 50.0)

        # 阶段 3：针头露出后，左手必须在预定接针区域可靠抓住针。
        if (
            self.needle_reached_exit
            and not self.left_has_grasped
            and touch_left_gripper
            and curr_dists['dist_left_to_mark'] <= self.left_grasp_mark_threshold
        ):
            self.left_has_grasped = True
            add_reward("left_handover_grasp", 75.0)

        # 阶段 4：左手保持抓取且右手确实释放，记录一次性交接事件。
        if (
            self.left_has_grasped
            and not self.right_released_after_handover
            and touch_left_gripper
            and not touch_right_gripper
        ):
            self.right_released_after_handover = True
            add_reward("right_release_after_handover", 50.0)

        # 阶段 5：交接完成后，左手把针尾沿负 X 方向拉过出口平面。
        if (
            self.right_released_after_handover
            and not self.needle_completely_through
            and touch_left_gripper
            and not touch_right_gripper
            and self._crossed_negative_x_plane(
                prev_dists['tail_pos'],
                curr_dists['tail_pos'],
                prev_dists['exit_pos'],
                curr_dists['exit_pos'],
            )
            and self._inside_hole_aperture(
                curr_dists['tail_pos'], curr_dists['exit_pos']
            )
        ):
            self.needle_completely_through = True
            add_reward("needle_completely_through", 200.0)

        # 纯差分 shaping：静止或来回振荡不能持续累积正奖励。
        if not self.right_has_grasped:
            add_progress("right_approach_progress", "dist_right_to_needle")
        elif not self.needle_start_through:
            if touch_right_gripper:
                add_progress("head_to_entrance_progress", "dist_head_to_entrance")
            else:
                add_progress("right_regrasp_progress", "dist_right_to_needle")
        elif not self.needle_reached_exit:
            if touch_right_gripper:
                add_progress("head_to_exit_progress", "dist_head_to_exit")
            else:
                add_progress("right_regrasp_progress", "dist_right_to_needle")
        elif not self.left_has_grasped:
            add_progress("left_approach_progress", "dist_left_to_mark")
        elif not self.right_released_after_handover:
            if not touch_left_gripper:
                add_progress("left_regrasp_progress", "dist_left_to_mark")
        elif not self.needle_completely_through:
            if touch_left_gripper:
                add_progress("tail_to_exit_progress", "dist_tail_to_exit")
            else:
                add_progress("left_regrasp_progress", "dist_left_to_mark")
        elif touch_left_gripper:
            add_progress("safe_pose_progress", "composite_error_dist")
        else:
            add_progress("left_regrasp_progress", "dist_left_to_mark")

        if self.left_has_grasped and touch_right_gripper:
            add_reward("penalty_right_still_touching", -0.5)

        # 最终成功必须左手单独稳定抓持，并在 X/Y/Z 三个方向进入安全区域。
        safe_final_pose = bool(
            self.needle_completely_through
            and self.right_released_after_handover
            and touch_left_gripper
            and not touch_right_gripper
            and curr_dists['x_error'] < self.success_x_threshold
            and curr_dists['y_error'] < self.success_y_threshold
            and curr_dists['z_error'] == 0.0
        )
        if safe_final_pose:
            self.success_stable_count += 1
            if self.success_stable_count >= self.success_stable_steps:
                add_reward("task_success", 500.0)
                self.is_success = True
                self.terminated = True
        else:
            self.success_stable_count = 0

        self._prev_dists = curr_dists
        self._set_reward_debug(
            reward,
            curr_dists,
            touch_left_gripper,
            touch_right_gripper,
            reward_terms,
        )
        return float(reward)
