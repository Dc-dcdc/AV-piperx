import os

import numpy as np

from env.constants import XML_DIR
from env.task.sim_envs import GuidedVisionEnv


class InsertPegEnv(GuidedVisionEnv):
    """双臂分别抓取Peg和插座并完成稳定对准插入的三臂主动视觉任务。"""

    def __init__(self, **kwargs):
        self.peg_position_ranges = self._validate_reset_position_ranges("peg_position_ranges", kwargs.pop("peg_position_ranges", [[0.10, 0.20], [-0.05, 0.15], [0.01, 0.01]]))  # Peg初始XYZ范围，单位m。
        self.hole_position_ranges = self._validate_reset_position_ranges("hole_position_ranges", kwargs.pop("hole_position_ranges", [[-0.20, -0.10], [-0.05, 0.15], [0.021, 0.021]]))  # 插座初始XYZ范围，单位m。
        self.enable_reward_debug = bool(kwargs.pop("enable_reward_debug", False))  # 是否记录奖励、接触和几何误差。
        self.success_x_threshold = float(kwargs.pop("success_x_threshold", 0.010))  # Peg与插座中心允许的最大X轴误差，单位m。
        self.success_y_threshold = float(kwargs.pop("success_y_threshold", 0.008))  # Peg与插座中心允许的最大Y轴误差，单位m。
        self.success_right_release_distance = float(kwargs.pop("success_right_release_distance", 0.10))  # 右夹爪离开Peg表面的最小距离，单位m。
        self.success_release_stable_steps = int(kwargs.pop("success_release_stable_steps", 5))  # 右手释放Peg后需要连续保持的控制步数。
        self.failure_z_threshold = float(kwargs.pop("failure_z_threshold", -0.08))  # 任一物体低于该Z坐标时判定失败，单位m。

        for name in (
            "success_x_threshold",
            "success_y_threshold",
            "success_right_release_distance",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}必须大于等于0。")
        if self.success_release_stable_steps < 1:
            raise ValueError("success_release_stable_steps必须大于等于1。")

        xml_path = kwargs.pop("xml_path", os.path.join(XML_DIR, "task_insert_peg.xml"))  # MuJoCo任务XML路径。
        super().__init__(xml_path, **kwargs)

        self.max_reward = 4
        self._peg_body = self._mjcf_root.find("body", "peg")
        self._hole_body = self._mjcf_root.find("body", "hole")
        self._peg_joint = self._mjcf_root.find("joint", "peg_joint")
        self._hole_joint = self._mjcf_root.find("joint", "hole_joint")
        self._right_gripper_site = self._mjcf_root.find(
            "site",
            "right_gripper_control",
        )
        required_elements = (
            ("peg", self._peg_body),
            ("hole", self._hole_body),
            ("peg_joint", self._peg_joint),
            ("hole_joint", self._hole_joint),
            ("right_gripper_control", self._right_gripper_site),
            ("peg_geom", self._mjcf_root.find("geom", "peg")),
            ("hole-1", self._mjcf_root.find("geom", "hole-1")),
            ("hole-2", self._mjcf_root.find("geom", "hole-2")),
            ("hole-3", self._mjcf_root.find("geom", "hole-3")),
            ("hole-4", self._mjcf_root.find("geom", "hole-4")),
            ("table", self._mjcf_root.find("geom", "table")),
        )
        for name, element in required_elements:
            if element is None:
                raise ValueError(f"Piper InsertPeg XML缺少必要元素: {name}")

        self.right_has_grasped = False
        self._release_stable_count = 0
        self.reward_debug = None

    def _geom_or_body_name(self, geom_id: int) -> str:
        """匿名Piper碰撞geom回退到所属body名称。"""
        geom_name = self._physics.model.id2name(geom_id, "geom")
        if geom_name and not geom_name.startswith("//unnamed_geom_"):
            return geom_name
        body_id = int(self._physics.model.geom_bodyid[geom_id])
        body_name = self._physics.model.id2name(body_id, "body")
        return body_name or ""

    @staticmethod
    def _finger_side(name: str) -> str | None:
        if name.startswith(("left_left_finger", "left_right_finger")):
            return "left"
        if name.startswith(("right_left_finger", "right_right_finger")):
            return "right"
        return None

    def _get_contact_flags(self) -> dict[str, bool]:
        flags = {
            "left_hole": False,
            "right_peg": False,
            "peg_table": False,
            "hole_table": False,
            "peg_hole": False,
        }
        hole_names = {"hole-1", "hole-2", "hole-3", "hole-4"}
        for i_contact in range(self._physics.data.ncon):
            contact = self._physics.data.contact[i_contact]
            name1 = self._geom_or_body_name(int(contact.geom1))
            name2 = self._geom_or_body_name(int(contact.geom2))
            if not name1 or not name2:
                continue

            pair = {name1, name2}
            if "peg" in pair and pair.intersection(hole_names):
                flags["peg_hole"] = True

            for object_name, other_name in ((name1, name2), (name2, name1)):
                finger_side = self._finger_side(other_name)
                if object_name == "peg":
                    if other_name == "table":
                        flags["peg_table"] = True
                    elif finger_side == "right":
                        flags["right_peg"] = True
                elif object_name in hole_names:
                    if other_name == "table":
                        flags["hole_table"] = True
                    elif finger_side == "left":
                        flags["left_hole"] = True
        return flags

    def _calculate_metrics(self) -> dict[str, float | bool | np.ndarray]:
        """计算X/Y对接误差和右夹爪到Peg外表面的最短距离。"""
        peg_pos = self._physics.bind(self._peg_body).xpos.copy()
        hole_pos = self._physics.bind(self._hole_body).xpos.copy()
        peg_xmat = self._physics.named.data.geom_xmat["peg"].reshape(3, 3)
        peg_half_size = self._physics.named.model.geom_size["peg"].copy()
        right_gripper_pos = self._physics.bind(
            self._right_gripper_site
        ).xpos.copy()

        # 转入Peg局部坐标系，计算夹爪中心到Peg有向包围盒的最短距离。
        gripper_in_peg = peg_xmat.T @ (right_gripper_pos - peg_pos)
        outside_offset = np.maximum(
            np.abs(gripper_in_peg) - peg_half_size,
            0.0,
        )
        right_to_peg_distance = float(np.linalg.norm(outside_offset))
        x_error = abs(float(peg_pos[0] - hole_pos[0]))
        y_error = abs(float(peg_pos[1] - hole_pos[1]))
        inserted = (
            x_error <= self.success_x_threshold
            and y_error <= self.success_y_threshold
        )
        return {
            "peg_pos": peg_pos,
            "hole_pos": hole_pos,
            "right_gripper_pos": right_gripper_pos,
            "right_to_peg_distance": right_to_peg_distance,
            "x_error": x_error,
            "y_error": y_error,
            "inserted": bool(inserted),
        }

    def reset(self, seed=None, options=None) -> tuple:
        """按Gym seed可复现地重置Peg和插座初始位置。"""
        super().reset(seed=seed, options=options)

        peg_position = self._sample_reset_position(
            self.peg_position_ranges
        )
        hole_position = self._sample_reset_position(
            self.hole_position_ranges
        )
        self._set_free_joint_reset_pose(
            self._peg_joint,
            peg_position,
        )
        self._set_free_joint_reset_pose(
            self._hole_joint,
            hole_position,
        )
        self._physics.forward()

        self.terminated = False
        self.is_success = False
        self.right_has_grasped = False
        self._release_stable_count = 0
        self.reward_debug = None

        observation = self.get_obs()
        info = {
            "message": "InsertPeg env reset.",
            "is_success": False,
            "peg_position": peg_position.copy(),
            "hole_position": hole_position.copy(),
        }
        return observation, info

    def get_reward(self) -> float:
        """返回0～4阶段奖励，并在X/Y位置满足插入阈值后终止。"""
        if self.is_success:
            return float(self.max_reward)

        contacts = self._get_contact_flags()
        metrics = self._calculate_metrics()
        objects_off_table = (
            not contacts["peg_table"]
            and not contacts["hole_table"]
        )
        if contacts["right_peg"]:
            self.right_has_grasped = True

        right_clear_of_peg = (
            not contacts["right_peg"]
            and metrics["right_to_peg_distance"]
            >= self.success_right_release_distance
        )
        # 释放候选：右手曾抓住Peg，对接后退到安全距离且物体仍在空中。
        success_candidate = (
            self.right_has_grasped
            and right_clear_of_peg
            and bool(metrics["inserted"])
            and objects_off_table
        )
        # 释放后必须连续保持设定步数；重新接触或对接失效时立即清零。
        if success_candidate:
            self._release_stable_count += 1
        else:
            self._release_stable_count = 0

        # 阶段0：尚未同时抓住Peg和插座，奖励保持为0。
        reward = 0
        stage = 0

        # 阶段1：左手接触插座、右手接触Peg，完成双物体抓取。
        if contacts["left_hole"] and contacts["right_peg"]:
            reward = 1
            stage = 1

        # 阶段2：保持双手抓取，并将Peg和插座都抬离桌面。
        if (
            contacts["left_hole"]
            and contacts["right_peg"]
            and objects_off_table
        ):
            reward = 2
            stage = 2

        # 阶段3：两个物体离桌后，Peg开始接触插座，进入对接阶段。
        if contacts["peg_hole"] and objects_off_table:
            reward = 3
            stage = 3

        # 阶段4：右手释放Peg后稳定保持足够步数，判定成功并终止。
        if self._release_stable_count >= self.success_release_stable_steps:
            reward = self.max_reward
            stage = 4
            self.is_success = True
            self.terminated = True

        # 失败终止：任一物体掉到最低安全高度以下。
        if (
            metrics["peg_pos"][2] < self.failure_z_threshold
            or metrics["hole_pos"][2] < self.failure_z_threshold
        ):
            self.is_success = False
            self.terminated = True

        if self.enable_reward_debug:
            self.reward_debug = {
                "stage": int(stage),
                "reward_total": float(reward),
                "success_candidate": bool(success_candidate),
                "right_has_grasped": bool(self.right_has_grasped),
                "right_clear_of_peg": bool(right_clear_of_peg),
                "release_stable_count": int(self._release_stable_count),
                "success_release_stable_steps": int(
                    self.success_release_stable_steps
                ),
                "left_gripper_touch_hole": bool(contacts["left_hole"]),
                "right_gripper_touch_peg": bool(contacts["right_peg"]),
                "peg_touch_table": bool(contacts["peg_table"]),
                "hole_touch_table": bool(contacts["hole_table"]),
                "peg_touch_hole": bool(contacts["peg_hole"]),
                "inserted": bool(metrics["inserted"]),
                "x_error_m": float(metrics["x_error"]),
                "y_error_m": float(metrics["y_error"]),
                "right_to_peg_distance_m": float(
                    metrics["right_to_peg_distance"]
                ),
            }
        return float(reward)
