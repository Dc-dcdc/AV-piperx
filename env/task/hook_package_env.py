import os

import numpy as np

from env.constants import XML_DIR
from env.task.sim_envs import GuidedVisionEnv


class HookPackageEnv(GuidedVisionEnv):
    """双臂抬起包裹并将顶部挂孔稳定挂到墙面Hook上的三臂主动视觉任务。"""

    def __init__(self, **kwargs):
        self.hook_position_ranges = self._validate_reset_position_ranges("hook_position_ranges", kwargs.pop("hook_position_ranges", [[-0.10, 0.10], [0.40, 0.40], [0.20, 0.30]]))  # Hook初始XYZ范围，单位m。
        self.package_position_ranges = self._validate_reset_position_ranges("package_position_ranges", kwargs.pop("package_position_ranges", [[-0.10, 0.10], [-0.025, 0.125], [0.00, 0.00]]))  # 包裹初始XYZ范围，单位m。
        self.enable_reward_debug = bool(kwargs.pop("enable_reward_debug", False))  # 是否记录奖励、接触和几何误差。
        self.hook_radial_threshold = float(kwargs.pop("hook_radial_threshold", 0.022))  # 挂孔到Hook轴线的最大径向误差，单位m。
        self.hook_axial_margin = float(kwargs.pop("hook_axial_margin", 0.012))  # 挂孔超出Hook轴段的允许余量，单位m。
        self.success_stable_steps = int(kwargs.pop("success_stable_steps", 5))  # 成功条件需要连续保持的仿真步数。
        self.success_max_linear_speed = float(kwargs.pop("success_max_linear_speed", 0.08))  # 成功时最大线速度，单位m/s。
        self.success_max_angular_speed = float(kwargs.pop("success_max_angular_speed", 1.0))  # 成功时最大角速度，单位rad/s。
        self.require_release_for_success = bool(kwargs.pop("require_release_for_success", True))  # 是否要求释放包裹后才成功。
        self.failure_z_threshold = float(kwargs.pop("failure_z_threshold", -0.08))  # 包裹低于该Z坐标时判定失败，单位m。
        for name in (
            "hook_radial_threshold",
            "hook_axial_margin",
            "success_max_linear_speed",
            "success_max_angular_speed",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}必须大于等于0。")
        if self.success_stable_steps < 1:
            raise ValueError("success_stable_steps必须大于等于1。")

        xml_path = kwargs.pop("xml_path", os.path.join(XML_DIR, "task_hook_package.xml"))  # MuJoCo任务XML路径。
        super().__init__(xml_path, **kwargs)

        self.max_reward = 4
        self._hook_body = self._mjcf_root.find("body", "hook")
        self._package_body = self._mjcf_root.find("body", "package")
        self._package_joint = self._mjcf_root.find(
            "joint",
            "package_joint",
        )
        required_elements = (
            ("hook", self._hook_body),
            ("package", self._package_body),
            ("package_joint", self._package_joint),
            ("table", self._mjcf_root.find("geom", "table")),
            ("hook_geom", self._mjcf_root.find("geom", "hook_geom")),
            ("hook_axis_marker", self._mjcf_root.find("geom", "pin-hook")),
            (
                "package_hole_marker",
                self._mjcf_root.find("geom", "pin-package"),
            ),
        )
        for name, element in required_elements:
            if element is None:
                raise ValueError(
                    f"Piper HookPackage XML缺少必要元素: {name}"
                )

        # Hook是reset时随机移动的无关节静态body，其位置不在data.qpos中。
        # 采集和增强脚本必须额外保存/恢复MuJoCo model body初态。
        self.replay_model_body_names = ("hook",)
        self._success_stable_count = 0
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
    def _finger_role(name: str) -> str | None:
        if name.startswith("left_left_finger"):
            return "left_left"
        if name.startswith("left_right_finger"):
            return "left_right"
        if name.startswith("right_left_finger"):
            return "right_left"
        if name.startswith("right_right_finger"):
            return "right_right"
        return None

    def _get_contact_flags(self) -> dict[str, bool]:
        flags = {
            "left_left": False,
            "left_right": False,
            "right_left": False,
            "right_right": False,
            "package_table": False,
            "package_hook": False,
        }
        package_names = {
            "package-1",
            "package-2",
            "package-3",
            "package-4",
        }
        for i_contact in range(self._physics.data.ncon):
            contact = self._physics.data.contact[i_contact]
            name1 = self._geom_or_body_name(int(contact.geom1))
            name2 = self._geom_or_body_name(int(contact.geom2))
            if not name1 or not name2:
                continue
            if name1 in package_names:
                package_name, other_name = name1, name2
            elif name2 in package_names:
                package_name, other_name = name2, name1
            else:
                continue
            if not package_name:
                continue

            finger_role = self._finger_role(other_name)
            if finger_role is not None:
                flags[finger_role] = True
            elif other_name == "table":
                flags["package_table"] = True
            elif other_name == "hook_geom":
                flags["package_hook"] = True

        flags["left_gripper"] = flags["left_left"] or flags["left_right"]
        flags["right_gripper"] = (
            flags["right_left"] or flags["right_right"]
        )
        return flags

    def _calculate_metrics(self) -> dict[str, float | bool | np.ndarray]:
        """计算包裹挂孔相对有限Hook轴段的几何误差。"""
        package_pin_pos = self._physics.named.data.geom_xpos[
            "pin-package"
        ].copy()
        hook_center = self._physics.named.data.geom_xpos[
            "pin-hook"
        ].copy()
        hook_xmat = self._physics.named.data.geom_xmat[
            "pin-hook"
        ].reshape(3, 3)
        hook_axis = hook_xmat @ np.array([0.0, 0.0, 1.0])
        hook_axis /= max(float(np.linalg.norm(hook_axis)), 1e-12)
        hook_half_length = float(
            self._physics.named.model.geom_size["pin-hook"][1]
        )

        offset = package_pin_pos - hook_center
        axial_position = float(np.dot(offset, hook_axis))
        clamped_axial = float(
            np.clip(
                axial_position,
                -hook_half_length,
                hook_half_length,
            )
        )
        nearest_hook_axis_pos = hook_center + clamped_axial * hook_axis
        radial_error = float(
            np.linalg.norm(package_pin_pos - nearest_hook_axis_pos)
        )
        axial_overrun = max(
            0.0,
            abs(axial_position) - hook_half_length,
        )
        package_qvel = np.asarray(
            self._physics.bind(self._package_joint).qvel,
            dtype=np.float64,
        )
        package_pos = self._physics.bind(self._package_body).xpos.copy()
        hook_engaged = (
            radial_error <= self.hook_radial_threshold
            and axial_overrun <= self.hook_axial_margin
        )
        return {
            "package_pos": package_pos,
            "package_pin_pos": package_pin_pos,
            "hook_center": hook_center,
            "nearest_hook_axis_pos": nearest_hook_axis_pos,
            "hook_axial_position": axial_position,
            "hook_axial_overrun": axial_overrun,
            "hook_radial_error": radial_error,
            "package_linear_speed": float(
                np.linalg.norm(package_qvel[:3])
            ),
            "package_angular_speed": float(
                np.linalg.norm(package_qvel[3:])
            ),
            "hook_engaged": bool(hook_engaged),
        }

    def reset(self, seed=None, options=None) -> tuple:
        """按Gym seed可复现地随机化Hook高度和包裹初始平面位置。"""
        super().reset(seed=seed, options=options)

        hook_position = self._sample_reset_position(
            self.hook_position_ranges
        )
        package_position = self._sample_reset_position(
            self.package_position_ranges
        )

        self._set_body_reset_position(
            self._hook_body,
            hook_position,
        )
        self._set_free_joint_reset_pose(
            self._package_joint,
            package_position,
        )
        self._physics.forward()

        self.terminated = False
        self.is_success = False
        self._success_stable_count = 0
        self.reward_debug = None

        observation = self.get_obs()
        info = {
            "message": "HookPackage env reset.",
            "is_success": False,
            "hook_position": hook_position.copy(),
            "package_position": package_position.copy(),
        }
        return observation, info

    def get_reward(self) -> float:
        """返回0～4阶段奖励，并在稳定挂接完成后终止episode。"""
        if self.is_success:
            return float(self.max_reward)

        contacts = self._get_contact_flags()
        metrics = self._calculate_metrics()
        left_touch = bool(contacts["left_gripper"])
        right_touch = bool(contacts["right_gripper"])
        package_off_table = not bool(contacts["package_table"])
        grippers_released = not left_touch and not right_touch
        speed_is_stable = (
            metrics["package_linear_speed"]
            <= self.success_max_linear_speed
            and metrics["package_angular_speed"]
            <= self.success_max_angular_speed
        )
        success_candidate = (
            bool(metrics["hook_engaged"])
            and package_off_table
            and speed_is_stable
            and (
                grippers_released
                or not self.require_release_for_success
            )
        )
        if success_candidate:
            self._success_stable_count += 1
        else:
            self._success_stable_count = 0

        reward = 0
        stage = 0
        if left_touch and right_touch:
            reward = 1
            stage = 1
        if left_touch and right_touch and package_off_table:
            reward = 2
            stage = 2
        if contacts["package_hook"] and package_off_table:
            reward = 3
            stage = 3
        if self._success_stable_count >= self.success_stable_steps:
            reward = self.max_reward
            stage = 4
            self.is_success = True
            self.terminated = True

        if metrics["package_pos"][2] < self.failure_z_threshold:
            self.is_success = False
            self.terminated = True

        if self.enable_reward_debug:
            self.reward_debug = {
                "stage": int(stage),
                "reward_total": float(reward),
                "success_candidate": bool(success_candidate),
                "success_stable_count": int(self._success_stable_count),
                "left_gripper_touch": left_touch,
                "right_gripper_touch": right_touch,
                "package_touch_table": bool(contacts["package_table"]),
                "package_touch_hook": bool(contacts["package_hook"]),
                "grippers_released": bool(grippers_released),
                "hook_engaged": bool(metrics["hook_engaged"]),
                "hook_radial_error_m": float(
                    metrics["hook_radial_error"]
                ),
                "hook_axial_overrun_m": float(
                    metrics["hook_axial_overrun"]
                ),
                "package_linear_speed_m_s": float(
                    metrics["package_linear_speed"]
                ),
                "package_angular_speed_rad_s": float(
                    metrics["package_angular_speed"]
                ),
            }
        return float(reward)
