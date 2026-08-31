import os

import numpy as np

from env.constants import XML_DIR
from env.task.sim_envs import GuidedVisionEnv


class OpenDrawerRetrieveEnv(GuidedVisionEnv):
    """左手开抽屉、右手取物并放入右侧托盘的三臂主动视觉任务。"""

    def __init__(self, **kwargs):
        # 目标物初始位置的XYZ随机范围，格式为[[x_min, x_max], [y_min, y_max], [z_min, z_max]]，单位m。
        self.object_position_ranges = self._validate_reset_position_ranges("object_position_ranges", kwargs.pop("object_position_ranges", [[-0.06, 0.06], [0.05, 0.15], [0.103, 0.103]]))
        # 目标托盘位置的XYZ随机范围，格式为[[x_min, x_max], [y_min, y_max], [z_min, z_max]]，单位m。
        self.target_position_ranges = self._validate_reset_position_ranges("target_position_ranges", kwargs.pop("target_position_ranges", [[-0.08, 0.08], [0.10, 0.10], [0.16, 0.16]]))
        # 判定抽屉已经充分打开所需的最小滑动距离，单位m。
        self.drawer_open_threshold = float(kwargs.pop("drawer_open_threshold", 0.24))
        # 判定目标物已完全取出时，目标物底部高于抽屉侧板顶部的最小间隙，单位m。
        self.retrieval_clearance = float(kwargs.pop("retrieval_clearance", 0.01))
        # 判定目标物进入托盘时允许的最大X轴位置误差，单位m。
        self.target_x_threshold = float(kwargs.pop("target_x_threshold", 0.045))
        # 判定目标物进入托盘时允许的最大Y轴位置误差，单位m。
        self.target_y_threshold = float(kwargs.pop("target_y_threshold", 0.030))
        # 判定目标物进入托盘时允许的中心高度范围[z_min, z_max]，单位m。
        self.target_z_range = np.asarray(kwargs.pop("target_z_range", [0.18, 0.22]), dtype=np.float64)
        # 判定右手已经释放目标物所需的夹爪到物体表面的最小距离，单位m。
        self.success_release_distance = float(kwargs.pop("success_release_distance", 0.06))
        # 成功状态下目标物允许的最大线速度，单位m/s。
        self.success_linear_speed = float(kwargs.pop("success_linear_speed", 0.08))
        # 成功状态下目标物允许的最大角速度，单位rad/s。
        self.success_angular_speed = float(kwargs.pop("success_angular_speed", 0.8))
        # 成功条件需要连续保持的环境控制步数。
        self.success_stable_steps = int(kwargs.pop("success_stable_steps", 5))
        # 目标物低于该世界坐标Z值时判定任务失败，单位m。
        self.failure_z_threshold = float(kwargs.pop("failure_z_threshold", -0.05))
        # 是否在info中记录奖励阶段、接触状态和几何误差等调试信息。
        self.enable_reward_debug = bool(kwargs.pop("enable_reward_debug", False))

        nonnegative_parameters = (
            "drawer_open_threshold",
            "retrieval_clearance",
            "target_x_threshold",
            "target_y_threshold",
            "success_release_distance",
            "success_linear_speed",
            "success_angular_speed",
        )
        for name in nonnegative_parameters:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name}必须大于等于0。")
        if self.target_z_range.shape != (2,):
            raise ValueError("target_z_range必须包含[z_min, z_max]。")
        if not np.isfinite(self.target_z_range).all():
            raise ValueError("target_z_range包含NaN或Inf。")
        if self.target_z_range[0] > self.target_z_range[1]:
            raise ValueError("target_z_range要求z_min<=z_max。")
        if self.success_stable_steps < 1:
            raise ValueError("success_stable_steps必须大于等于1。")

        # MuJoCo任务场景XML文件路径。
        xml_path = kwargs.pop("xml_path", os.path.join(XML_DIR, "task_open_drawer_retrieve.xml"))
        super().__init__(xml_path, **kwargs)

        self.max_reward = 4
        self._drawer_joint = self._mjcf_root.find(
            "joint", "drawer_slide_joint"
        )
        self._object_joint = self._mjcf_root.find(
            "joint", "drawer_object_joint"
        )
        self._object_body = self._mjcf_root.find("body", "drawer_object")
        self._target_body = self._mjcf_root.find(
            "body", "retrieval_target"
        )
        self._target_site = self._mjcf_root.find(
            "site", "retrieval_target_site"
        )
        self._left_gripper_site = self._mjcf_root.find(
            "site", "left_gripper_control"
        )
        self._right_gripper_site = self._mjcf_root.find(
            "site", "right_gripper_control"
        )
        required_elements = (
            ("drawer_slide_joint", self._drawer_joint),
            ("drawer_object_joint", self._object_joint),
            ("drawer_object", self._object_body),
            ("retrieval_target", self._target_body),
            ("retrieval_target_site", self._target_site),
            ("left_gripper_control", self._left_gripper_site),
            ("right_gripper_control", self._right_gripper_site),
            (
                "drawer_object_geom",
                self._mjcf_root.find("geom", "drawer_object_geom"),
            ),
            (
                "drawer_right_side",
                self._mjcf_root.find("geom", "drawer_right_side"),
            ),
            ("drawer_handle", self._mjcf_root.find("geom", "drawer_handle")),
        )
        for name, element in required_elements:
            if element is None:
                raise ValueError(
                    f"OpenDrawerRetrieve XML缺少必要元素: {name}"
                )

        # 托盘为无关节静态body，其随机位置不在data.qpos中，需要随轨迹保存。
        self.replay_model_body_names = ("retrieval_target",)
        self.left_has_contacted_handle = False
        self.drawer_has_opened = False
        self.right_has_grasped = False
        self.object_was_retrieved = False
        self._success_stable_count = 0
        self.reward_debug = None

    def _geom_or_body_name(self, geom_id: int) -> str:
        """匿名机械臂碰撞geom回退到所属body名称。"""
        geom_name = self._physics.model.id2name(geom_id, "geom")
        if geom_name and not geom_name.startswith("//unnamed_geom_"):
            return geom_name
        body_id = int(self._physics.model.geom_bodyid[geom_id])
        body_name = self._physics.model.id2name(body_id, "body")
        return body_name or ""

    @staticmethod
    def _finger_role(name: str) -> str | None:
        if name.startswith(("left_left_finger", "left_right_finger")):
            return "left"
        if name.startswith(("right_left_finger", "right_right_finger")):
            return "right"
        return None

    def _get_contact_flags(self) -> dict[str, bool]:
        """读取左夹爪与把手、右夹爪与目标物之间的接触。"""
        flags = {
            "left_handle": False,
            "right_object": False,
            "object_table": False,
            "object_cabinet_top": False,
            "object_target_rim": False,
        }
        handle_names = {
            "drawer_handle",
            "drawer_handle_left_mount",
            "drawer_handle_right_mount",
        }
        target_rim_names = {
            "retrieval_target_back_rim",
            "retrieval_target_front_rim",
            "retrieval_target_left_rim",
            "retrieval_target_right_rim",
        }

        for i_contact in range(self._physics.data.ncon):
            contact = self._physics.data.contact[i_contact]
            name1 = self._geom_or_body_name(int(contact.geom1))
            name2 = self._geom_or_body_name(int(contact.geom2))
            if not name1 or not name2:
                continue

            pair = {name1, name2}
            if "drawer_object_geom" in pair and "table" in pair:
                flags["object_table"] = True
            if "drawer_object_geom" in pair and "cabinet_top" in pair:
                flags["object_cabinet_top"] = True
            if (
                "drawer_object_geom" in pair
                and pair.intersection(target_rim_names)
            ):
                flags["object_target_rim"] = True

            for task_geom, other_name in ((name1, name2), (name2, name1)):
                finger_role = self._finger_role(other_name)
                if task_geom in handle_names and finger_role == "left":
                    flags["left_handle"] = True
                elif (
                    task_geom == "drawer_object_geom"
                    and finger_role == "right"
                ):
                    flags["right_object"] = True
        return flags

    def _calculate_metrics(self) -> dict[str, float | bool | np.ndarray]:
        """计算抽屉开度、取出高度、目标误差及物体稳定性。"""
        drawer_qpos = float(
            np.ravel(self._physics.bind(self._drawer_joint).qpos)[0]
        )
        object_pos = self._physics.bind(self._object_body).xpos.copy()
        target_pos = self._physics.bind(self._target_site).xpos.copy()
        right_gripper_pos = self._physics.bind(
            self._right_gripper_site
        ).xpos.copy()

        object_xmat = self._physics.named.data.geom_xmat[
            "drawer_object_geom"
        ].reshape(3, 3)
        object_half_size = self._physics.named.model.geom_size[
            "drawer_object_geom"
        ].copy()
        object_vertical_extent = float(
            np.abs(object_xmat[2]) @ object_half_size
        )
        object_bottom_z = float(object_pos[2] - object_vertical_extent)
        drawer_side_top_z = float(
            self._physics.named.data.geom_xpos["drawer_right_side", "z"]
            + self._physics.named.model.geom_size[
                "drawer_right_side", 2
            ]
        )

        gripper_in_object = object_xmat.T @ (
            right_gripper_pos - object_pos
        )
        outside_offset = np.maximum(
            np.abs(gripper_in_object) - object_half_size,
            0.0,
        )
        right_to_object_distance = float(np.linalg.norm(outside_offset))

        object_qvel = np.asarray(
            self._physics.bind(self._object_joint).qvel
        ).copy()
        linear_speed = float(np.linalg.norm(object_qvel[:3]))
        angular_speed = float(np.linalg.norm(object_qvel[3:]))
        target_x_error = abs(float(object_pos[0] - target_pos[0]))
        target_y_error = abs(float(object_pos[1] - target_pos[1]))
        object_retrieved = (
            object_bottom_z
            >= drawer_side_top_z + self.retrieval_clearance
        )
        object_in_target = (
            target_x_error <= self.target_x_threshold
            and target_y_error <= self.target_y_threshold
            and self.target_z_range[0]
            <= object_pos[2]
            <= self.target_z_range[1]
        )
        object_stable = (
            linear_speed <= self.success_linear_speed
            and angular_speed <= self.success_angular_speed
        )
        return {
            "drawer_qpos": drawer_qpos,
            "object_pos": object_pos,
            "object_bottom_z": object_bottom_z,
            "drawer_side_top_z": drawer_side_top_z,
            "target_pos": target_pos,
            "right_gripper_pos": right_gripper_pos,
            "right_to_object_distance": right_to_object_distance,
            "target_x_error": target_x_error,
            "target_y_error": target_y_error,
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "object_retrieved": bool(object_retrieved),
            "object_in_target": bool(object_in_target),
            "object_stable": bool(object_stable),
        }

    def reset(self, seed=None, options=None) -> tuple:
        """关闭抽屉，并按Gym seed重置目标物和托盘位置。"""
        super().reset(seed=seed, options=options)

        object_position = self._sample_reset_position(
            self.object_position_ranges
        )
        target_position = self._sample_reset_position(
            self.target_position_ranges
        )
        self._physics.bind(self._drawer_joint).qpos = 0.0
        self._physics.bind(self._drawer_joint).qvel = 0.0
        self._set_free_joint_reset_pose(
            self._object_joint,
            object_position,
        )
        self._set_body_reset_position(
            self._target_body,
            target_position,
        )
        self._physics.forward()

        self.terminated = False
        self.is_success = False
        self.left_has_contacted_handle = False
        self.drawer_has_opened = False
        self.right_has_grasped = False
        self.object_was_retrieved = False
        self._success_stable_count = 0
        self.reward_debug = None

        observation = self.get_obs()
        info = {
            "message": "OpenDrawerRetrieve env reset.",
            "is_success": False,
            "object_position": object_position.copy(),
            "target_position": target_position.copy(),
        }
        return observation, info

    def get_reward(self) -> float:
        """返回0～4阶段奖励，并在右手稳定释放目标物后终止。"""
        if self.is_success:
            return float(self.max_reward)

        contacts = self._get_contact_flags()
        metrics = self._calculate_metrics()

        if contacts["left_handle"]:
            self.left_has_contacted_handle = True
        if (
            self.left_has_contacted_handle
            and metrics["drawer_qpos"] >= self.drawer_open_threshold
        ):
            self.drawer_has_opened = True
        if self.drawer_has_opened and contacts["right_object"]:
            self.right_has_grasped = True
        if (
            self.right_has_grasped
            and bool(metrics["object_retrieved"])
        ):
            self.object_was_retrieved = True

        right_clear_of_object = (
            not contacts["right_object"]
            and metrics["right_to_object_distance"]
            >= self.success_release_distance
        )
        success_candidate = (
            self.drawer_has_opened
            and self.right_has_grasped
            and self.object_was_retrieved
            and bool(metrics["object_in_target"])
            and bool(metrics["object_stable"])
            and (
                contacts["object_table"]
                or contacts["object_cabinet_top"]
                or contacts["object_target_rim"]
            )
            and right_clear_of_object
        )
        if success_candidate:
            self._success_stable_count += 1
        else:
            self._success_stable_count = 0

        reward = 0
        stage = 0
        if self.left_has_contacted_handle:
            reward = 1
            stage = 1
        if self.drawer_has_opened:
            reward = 2
            stage = 2
        if self.object_was_retrieved:
            reward = 3
            stage = 3
        if self._success_stable_count >= self.success_stable_steps:
            reward = self.max_reward
            stage = 4
            self.is_success = True
            self.terminated = True

        if metrics["object_pos"][2] < self.failure_z_threshold:
            self.is_success = False
            self.terminated = True

        if self.enable_reward_debug:
            self.reward_debug = {
                "stage": int(stage),
                "reward_total": float(reward),
                "success_candidate": bool(success_candidate),
                "success_stable_count": int(self._success_stable_count),
                "left_has_contacted_handle": bool(
                    self.left_has_contacted_handle
                ),
                "drawer_has_opened": bool(self.drawer_has_opened),
                "right_has_grasped": bool(self.right_has_grasped),
                "object_was_retrieved": bool(self.object_was_retrieved),
                "left_gripper_touch_handle": bool(
                    contacts["left_handle"]
                ),
                "right_gripper_touch_object": bool(
                    contacts["right_object"]
                ),
                "object_touch_target_rim": bool(
                    contacts["object_target_rim"]
                ),
                "object_touch_table": bool(contacts["object_table"]),
                "object_touch_cabinet_top": bool(
                    contacts["object_cabinet_top"]
                ),
                "right_clear_of_object": bool(right_clear_of_object),
                "object_in_target": bool(metrics["object_in_target"]),
                "object_stable": bool(metrics["object_stable"]),
                "drawer_qpos_m": float(metrics["drawer_qpos"]),
                "object_bottom_z_m": float(metrics["object_bottom_z"]),
                "drawer_side_top_z_m": float(
                    metrics["drawer_side_top_z"]
                ),
                "target_x_error_m": float(metrics["target_x_error"]),
                "target_y_error_m": float(metrics["target_y_error"]),
                "right_to_object_distance_m": float(
                    metrics["right_to_object_distance"]
                ),
                "object_linear_speed_m_s": float(
                    metrics["linear_speed"]
                ),
                "object_angular_speed_rad_s": float(
                    metrics["angular_speed"]
                ),
            }
        return float(reward)
