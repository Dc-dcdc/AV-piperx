import os

import numpy as np

from env.constants import XML_DIR
from env.task.sim_envs import GuidedVisionEnv


class InsertCylinderEnv(GuidedVisionEnv):
    """挡板遮挡下的圆柱放入容器任务环境。"""

    def __init__(self, **kwargs):
        self.success_xy_threshold = float(kwargs.pop("success_xy_threshold", 0.012))
        self.success_z_threshold = float(kwargs.pop("success_z_threshold", 0.025))
        self.cylinder_target_z = float(kwargs.pop("cylinder_target_z", 0.04))
        self.enable_reward_debug = bool(kwargs.pop("enable_reward_debug", False))

        xml_path = os.path.join(XML_DIR, "task_insert_cylinder.xml")
        super().__init__(xml_path, **kwargs)

        self._cylinder_joint = self._mjcf_root.find("joint", "insert_cylinder_joint")
        self._container_body = self._mjcf_root.find("body", "cylinder_container")
        self._prev_metrics = {}
        self.right_has_grasped = False
        self.left_has_received = False
        self.right_has_released = False
        self.cylinder_was_grasped = False
        self.placement_checked = False
        self.reward_debug = None

    def _get_contact_flags(self):
        """读取圆柱是否被左右夹爪稳定夹住。"""
        touch_left_left = False
        touch_left_right = False
        touch_right_left = False
        touch_right_right = False
        touch_container = False

        for i_contact in range(self._physics.data.ncon):
            geom_id_1 = self._physics.data.contact[i_contact].geom1
            geom_id_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(geom_id_1, "geom")
            geom2 = self._physics.model.id2name(geom_id_2, "geom")
            if not geom1 or not geom2:
                continue

            pair = {geom1, geom2}
            if "insert_cylinder_geom" not in pair:
                continue

            other = geom2 if geom1 == "insert_cylinder_geom" else geom1
            if other.startswith("left_left"):
                touch_left_left = True
            elif other.startswith("left_right"):
                touch_left_right = True
            elif other.startswith("right_left"):
                touch_right_left = True
            elif other.startswith("right_right"):
                touch_right_right = True
            elif other.startswith("container_"):
                touch_container = True

        touch_left_gripper = touch_left_left and touch_left_right
        touch_right_gripper = touch_right_left and touch_right_right
        return touch_left_gripper, touch_right_gripper, touch_container

    def _calculate_metrics(self):
        """计算圆柱、容器和夹爪之间的关键距离。"""
        cylinder_pos = self._physics.named.data.geom_xpos["insert_cylinder_geom"].copy()
        target_pos = self._physics.named.data.xpos["cylinder_container"].copy()
        target_pos[2] = self.cylinder_target_z

        left_left_finger = self._physics.named.data.geom_xpos["left_left_g2"]
        left_right_finger = self._physics.named.data.geom_xpos["left_right_g2"]
        right_left_finger = self._physics.named.data.geom_xpos["right_left_g2"]
        right_right_finger = self._physics.named.data.geom_xpos["right_right_g2"]

        left_gripper_center = (left_left_finger + left_right_finger) / 2.0
        right_gripper_center = (right_left_finger + right_right_finger) / 2.0

        left_dist = np.linalg.norm(left_gripper_center - cylinder_pos)
        right_dist = np.linalg.norm(right_gripper_center - cylinder_pos)
        closest_gripper_dist = min(left_dist, right_dist)

        cylinder_xmat = self._physics.named.data.geom_xmat["insert_cylinder_geom"].reshape(3, 3)
        cylinder_axis = cylinder_xmat @ np.array([0.0, 0.0, 1.0])
        cylinder_upright_cos = abs(float(cylinder_axis[2]))
        cylinder_half_height = float(self._physics.named.model.geom_size["insert_cylinder_geom"][1])
        cylinder_bottom_pos = cylinder_pos - cylinder_axis * cylinder_half_height

        xy_error = np.linalg.norm(cylinder_pos[:2] - target_pos[:2])
        bottom_xy_error = np.linalg.norm(cylinder_bottom_pos[:2] - target_pos[:2])
        z_error = abs(cylinder_pos[2] - target_pos[2])
        target_dist = np.linalg.norm(
            np.array([cylinder_pos[0] - target_pos[0], cylinder_pos[1] - target_pos[1], cylinder_pos[2] - target_pos[2]])
        )

        return {
            "cylinder_pos": cylinder_pos,
            "cylinder_bottom_pos": cylinder_bottom_pos,
            "target_pos": target_pos,
            "left_gripper_pos": left_gripper_center.copy(),
            "right_gripper_pos": right_gripper_center.copy(),
            "closest_gripper_dist": closest_gripper_dist,
            "left_dist": left_dist,
            "right_dist": right_dist,
            "target_dist": target_dist,
            "xy_error": xy_error,
            "bottom_xy_error": bottom_xy_error,
            "z_error": z_error,
            "cylinder_upright_cos": cylinder_upright_cos,
        }

    def _set_reward_debug(self, reward, stage, reward_terms, contact_flags, curr_metrics):
        """整理当前步奖励调试信息，供数据采集脚本单独保存。"""
        if not self.enable_reward_debug:
            return
        touch_left, touch_right, touch_container = contact_flags
        cabinet_top_z = (
            self._physics.named.data.geom_xpos["middle_view_cabinet_top"][2]
            + self._physics.named.model.geom_size["middle_view_cabinet_top"][2]
        )
        left_clear_z = cabinet_top_z + 0.02
        left_clearance = curr_metrics["left_gripper_pos"][2] - left_clear_z
        placed_now = (
            curr_metrics["bottom_xy_error"] < self.success_xy_threshold
            and curr_metrics["cylinder_upright_cos"] > 0.9
        )
        self.reward_debug = {
            "stage": int(stage),
            "reward_total": float(reward),
            "reward_terms": {key: float(value) for key, value in reward_terms.items()},
            "right_has_grasped": bool(self.right_has_grasped),
            "left_has_received": bool(self.left_has_received),
            "right_has_released": bool(self.right_has_released),
            "placement_checked": bool(self.placement_checked),
            "cylinder_was_grasped": bool(self.cylinder_was_grasped),
            "touch_left": bool(touch_left),
            "touch_right": bool(touch_right),
            "touch_container": bool(touch_container),
            "placed_now": bool(placed_now),
            "left_above_cabinet": bool(left_clearance > 0.0),
            "right_to_cylinder_dist": float(curr_metrics["right_dist"]),
            "left_to_cylinder_dist": float(curr_metrics["left_dist"]),
            "cylinder_to_target_xy_dist": float(curr_metrics["xy_error"]),
            "cylinder_bottom_to_target_xy_dist": float(curr_metrics["bottom_xy_error"]),
            "cylinder_z_error": float(curr_metrics["z_error"]),
            "cylinder_upright_cos": float(curr_metrics["cylinder_upright_cos"]),
            "left_eef_height": float(curr_metrics["left_gripper_pos"][2]),
            "left_clearance": float(left_clearance),
        }

    def reset(self, seed=None, options=None) -> tuple:
        """重置机器人、圆柱和容器初始位置。"""
        super().reset(seed=seed, options=options)
        rng = self.np_random

        x_range = [0.045, 0.045]
        y_range = [0.10, 0.30]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        cylinder_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        cylinder_quat = np.array([1, 0, 0, 0])

        self._physics.bind(self._cylinder_joint).qpos = np.concatenate([cylinder_position, cylinder_quat])
        self._physics.bind(self._cylinder_joint).qvel = np.zeros(6)

        container_position = np.array([
            rng.uniform(-0.045, -0.045),
            rng.uniform(0.10, 0.30),
            0.0,
        ], dtype=np.float64)
        self._physics.bind(self._container_body).pos = container_position


        self._physics.forward()

        self.terminated = False
        self.is_success = False
        self.right_has_grasped = False
        self.left_has_received = False
        self.right_has_released = False
        self.cylinder_was_grasped = False
        self.placement_checked = False
        self.reward_debug = None
        self._prev_metrics = self._calculate_metrics()

        observation = self.get_obs()
        info = {"message": "InsertCylinder env reset."}
        return observation, info

    def get_reward(self):
        """按右抓、左接、右松、放置四个阶段计算奖励。"""
        touch_left, touch_right, touch_container = self._get_contact_flags()
        curr_metrics = self._calculate_metrics()

        reward = -0.5
        stage = 0
        reward_terms = {"step_penalty": -0.5} if self.enable_reward_debug else None
        dist_scale = 100.0
        target_scale = 250.0
        retreat_scale = 80.0

        def add_reward(name, value):
            nonlocal reward
            value = float(value)
            reward += value
            if reward_terms is not None:
                reward_terms[name] = reward_terms.get(name, 0.0) + value

        # 任一夹爪抓住过圆柱后，才启用后续掉落判定。
        if touch_left or touch_right:
            self.cylinder_was_grasped = True

        # 圆柱曾被抓起后无人抓住且掉到桌面下方，判定失败。
        if self.cylinder_was_grasped and not touch_left and not touch_right and curr_metrics["cylinder_pos"][2] < -0.03:
            stage = -1
            add_reward("penalty_drop", -100.0)
            self.is_success = False
            self.terminated = True
            self._prev_metrics = curr_metrics
            self._set_reward_debug(
                reward,
                stage,
                reward_terms,
                (touch_left, touch_right, touch_container),
                curr_metrics,
            )
            return float(reward)

        # 阶段 1：右臂尚未抓住圆柱时，引导右臂靠近并抓取。
        if not self.right_has_grasped:
            stage = 1
            progress = self._prev_metrics["right_dist"] - curr_metrics["right_dist"]
            add_reward("right_approach_progress", dist_scale * progress)
            add_reward("right_approach_distance", 0.5 * max(0.0, 1.0 - curr_metrics["right_dist"] / 0.18))
            # 右臂稳定接触圆柱后，进入左臂接应阶段。
            if touch_right:
                self.right_has_grasped = True
                self.cylinder_was_grasped = True
                add_reward("right_grasp", 40.0)

        # 阶段 2：右臂已抓住圆柱，引导左臂接近并接住。
        elif not self.left_has_received:
            stage = 2
            # 右臂保持抓取给小奖励，提前松开则扣分。
            if touch_right:
                add_reward("right_hold", 0.2)
            else:
                add_reward("penalty_right_release_early", -2.0)
            progress = self._prev_metrics["left_dist"] - curr_metrics["left_dist"]
            add_reward("left_approach_progress", dist_scale * progress)
            add_reward("left_approach_distance", 0.5 * max(0.0, 1.0 - curr_metrics["left_dist"] / 0.18))
            # 左臂稳定接触圆柱后，进入右臂松手阶段。
            if touch_left:
                self.left_has_received = True
                add_reward("left_receive", 50.0)

        # 阶段 3：左臂已接住圆柱，要求右臂松手完成交接。
        elif not self.right_has_released:
            stage = 3
            # 左臂保持抓取给小奖励，丢失圆柱则扣分。
            if touch_left:
                add_reward("left_hold", 1.0)
            else:
                add_reward("penalty_left_lost", -5.0)

            # 右臂继续抓住会扣分，松手后进入放置阶段。
            if touch_right:
                add_reward("penalty_right_still_touching", -1.0)
            else:
                self.right_has_released = True
                add_reward("right_release", 35.0)

        # 阶段 4/5：右臂已松手，左臂负责放置圆柱并抬离柜子。
        else:
            # 放置是否合格：圆柱底部中心接近目标，且最终姿态基本直立。
            placed_now = (
                curr_metrics["bottom_xy_error"] < self.success_xy_threshold
                and curr_metrics["cylinder_upright_cos"] > 0.9
            )

            # 阶段 4：左臂尚未松手时，引导圆柱底部靠近圆环目标。
            if not self.placement_checked:
                stage = 4
                progress = self._prev_metrics["bottom_xy_error"] - curr_metrics["bottom_xy_error"]
                add_reward("place_progress", target_scale * progress)
                # 只给很小的靠近提示，避免在目标附近持续停留刷分。
                add_reward("place_close_hint", 0.2 * max(0.0, 1.0 - curr_metrics["bottom_xy_error"] / 0.08))

                # 左臂仍抓着圆柱时保持当前放置阶段。
                if touch_left:
                    add_reward("left_hold_for_place", 0.05)
                # 左臂松手时完成一次放置质量评分，并进入阶段 5。
                else:
                    self.placement_checked = True
                    placement_score = max(0.0, 1.0 - curr_metrics["bottom_xy_error"] / 0.12)
                    add_reward("place_score", 250.0 * placement_score)
                    # 最终放置时圆柱不够直立，只在松手时扣一次分。
                    if curr_metrics["cylinder_upright_cos"] < 0.9:
                        add_reward("penalty_final_tilt", -50.0 * (0.9 - curr_metrics["cylinder_upright_cos"]))
                    # 放置合格给额外阶段奖励。
                    if placed_now:
                        add_reward("place_success", 200.0)
            # 阶段 5：放置检查后，等待左臂松手并抬到柜子高度以上。
            else:
                stage = 5
                # 阶段 5 中如果左臂又碰到圆柱，说明尚未完全撤离。
                if touch_left:
                    add_reward("penalty_touch_after_place", -10.0)
                    # 明显碰倒时给重罚，但不直接终止，最终成功条件会自动失效。
                    if curr_metrics["cylinder_upright_cos"] < 0.5:
                        add_reward("penalty_knock_over_after_place", -100.0)

                # 计算柜子顶面高度，作为左臂撤离的高度阈值。
                cabinet_top_z = (
                    self._physics.named.data.geom_xpos["middle_view_cabinet_top"][2]
                    + self._physics.named.model.geom_size["middle_view_cabinet_top"][2]
                )
                left_clear_z = cabinet_top_z + 0.02
                left_clearance = curr_metrics["left_gripper_pos"][2] - left_clear_z
                prev_left_clearance = self._prev_metrics["left_gripper_pos"][2] - left_clear_z
                # 撤离阶段奖励抬高手臂的进度，不再因为高度不足每步持续大额扣分。
                add_reward("left_retreat_progress", retreat_scale * (left_clearance - prev_left_clearance))
                if left_clearance > 0.0:
                    add_reward("left_clear_bonus", 2.0)

                # 圆柱放置合格、左臂高于柜子且没有接触圆柱，任务完成。
                left_above_cabinet = left_clearance > 0.0
                if placed_now and left_above_cabinet and not touch_left:
                    add_reward("task_success", 300.0)
                    self.is_success = True
                    self.terminated = True

        self._prev_metrics = curr_metrics
        self._set_reward_debug(
            reward,
            stage,
            reward_terms,
            (touch_left, touch_right, touch_container),
            curr_metrics,
        )
        return float(reward)
