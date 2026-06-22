import os

import numpy as np

from env.constants import XML_DIR
from env.task.sim_envs import GuidedVisionEnv


class SewNeedleEnv(GuidedVisionEnv):
    """
    缝合针穿引任务专用环境
    """
    def __init__(self, **kwargs):
        xml_path = os.path.join(XML_DIR, 'task_sew_needle.xml')
        super().__init__(xml_path, **kwargs)

        self._needle_joint = self._mjcf_root.find('joint', 'needle_joint')
        self._wall_joint = self._mjcf_root.find('joint', 'wall_joint')
        self._threaded_needle = False
        self._prev_dists = {} #  用于存储上一步的距离字典

    def _calculate_distances(self):
        """🌟 辅助函数：统一计算所有关键点的坐标与距离"""
        # 1. 提取物体坐标
        needle_head = self._physics.named.data.geom_xpos['needle_head']                #针头
        needle_tail = self._physics.named.data.geom_xpos['needle_tail']                #针尾
        needle_left_pos = self._physics.named.data.geom_xpos['needle_mark_1_4']        #左臂抓取标记点 1/4处
        needle_right_pos = self._physics.named.data.geom_xpos['needle_mark_3_4']       #右臂抓取标记点 3/4处

        left_left_finger = self._physics.named.data.geom_xpos['left_left_g2']
        left_right_finger = self._physics.named.data.geom_xpos['left_right_g2']
        right_left_finger = self._physics.named.data.geom_xpos['right_left_g2']        #右臂左指尖
        right_right_finger = self._physics.named.data.geom_xpos['right_right_g2']      #右臂右指尖

        hole_entrance = self._physics.named.data.geom_xpos['hole_entrance']            #出洞口
        hole_exit = self._physics.named.data.geom_xpos['hole_exit']                    #出洞口

        # 2. 计算复合中心点
        left_gripper_center = (left_left_finger + left_right_finger) / 2.0
        right_gripper_center = (right_left_finger + right_right_finger) / 2.0
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
        target_z_target = hole_exit[2] + 0.012
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

            # 保存误差用于通关判定
            'x_error': x_error,
            'y_error': y_error,
            'z_error': z_error
        }

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        rng = self.np_random
        # 随机化针的位置
        x_range = [0.15, 0.2]
        y_range = [-.025, 0.1]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        needle_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        needle_quat = np.array([1, 0, 0, 0])

        # 随机化墙(洞)的位置
        x_range = [-0.025, 0.025]
        y_range = [-.025, 0.1]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        wall_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        wall_quat = np.array([1, 0, 0, 0])

        self._physics.bind(self._needle_joint).qpos = np.concatenate([needle_position, needle_quat])
        self._physics.bind(self._wall_joint).qpos = np.concatenate([wall_position, wall_quat])

        self._physics.forward()
        self._threaded_needle = False
        self.needle_reached_exit = False                 # 针头穿墙标志位
        self.left_has_grasped = False                    # 左臂成功接针标志位
        self.needle_was_grasped = False                  # 针是否曾经被任一夹爪抓住过
        self.needle_completely_through = False           # 针完全过孔标志位
        self.needle_start_through = False                # 针开始过孔标志位
        self._prev_dists = self._calculate_distances()   # 记录物理引擎第一帧的距离，作为差分计算的起点
        observation = self.get_obs()
        info = {"message": "SewNeedle env reset."}
        return observation, info

    def get_reward(self):
        touch_left_gripper = False
        touch_right_gripper = False
        objects_touching_needle = set()     # 存放当前帧里所有碰到针的物体名字

        # 遍历底层物理引擎计算出的所有有效接触点 (Contacts)
        for i_contact in range(self._physics.data.ncon):
            # 提取发生碰撞的两个物体的底层 ID
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2

            # 将底层 ID 翻译成你在 XML 环境文件中定义的名字 (如 "peg", "right_finger_1")
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')

            # 只有两个物体都有名字时，才加入判定列表(双向记录)
            if geom1 and geom2:
                # 2. 只要有一方是针，就把另一方的名字扔进篮子
                if geom1 == "needle":
                    objects_touching_needle.add(geom2)
                elif geom2 == "needle":
                    objects_touching_needle.add(geom1)

        # 检查左夹爪是否成功捏住：左半边手指碰到了，且右半边手指也碰到了left_left中包含主模型right_right_finger和g0、g1、g2
        touched_left_left = any(obj.startswith("left_left") for obj in objects_touching_needle)
        touched_left_right = any(obj.startswith("left_right") for obj in objects_touching_needle)
        if touched_left_left and touched_left_right:
            touch_left_gripper = True
        # 检查右夹爪是否成功捏住
        touched_right_left = any(obj.startswith("right_left") for obj in objects_touching_needle)
        touched_right_right = any(obj.startswith("right_right") for obj in objects_touching_needle)
        if touched_right_left and touched_right_right:
            touch_right_gripper = True

        # ==========================================
        # 🌟 提取当前帧的距离字典
        # ==========================================
        curr_dists = self._calculate_distances()

        # ==========================================
        # 阶段流转奖励逻辑 (稀疏事件奖励保留)
        # ==========================================
        reward = -0.8 # 存活的时间惩罚，逼迫其快速完成任务
        needle_is_grasped = touch_left_gripper or touch_right_gripper
        if needle_is_grasped:
            self.needle_was_grasped = True

        # 如果针曾经被抓起过，但当前左右臂都没抓住且针明显掉到洞口下方，则判定掉落失败。
        if (
            self.needle_was_grasped
            and not needle_is_grasped
            and curr_dists['needle_z'] < (curr_dists['hole_z'] - 0.03)
        ):
            reward -= 100.0
            self.is_success = False
            self.terminated = True
            self._prev_dists = curr_dists
            return float(reward)

        # 触发 1：针头到达入口 (进洞)
        if (
            not self.needle_start_through
            and curr_dists['head_pos'][0] - curr_dists['entrance_pos'][0] < 0.001
            and curr_dists['dist_head_to_entrance'] < 0.03
        ):
            self.needle_start_through = True
            reward += 25.0
        # print("当前阶段状态：", {
        #     "needle_start_through": self.needle_start_through,})
        # 触发 2：针头到达出口 (露头)
        # 触发 2：针头到达出口 (露头)，必须先完成进洞阶段，避免从后续几何位置误触发。
        if (
            self.needle_start_through                                 # 必须先完成进洞阶段
            and not self.needle_reached_exit                          # 针头还没有到达出口
            and curr_dists['head_pos'][0] < curr_dists['exit_pos'][0] # 针头必须在出口的负方向（洞口的另一侧）
            and curr_dists['dist_head_to_exit'] < 0.03                # 针头必须足够接近出口（防止从墙外面绕过）
        ):
            self.needle_reached_exit = True
            reward += 50.0

        # 触发 3：左手成功接应
        if (
            self.needle_reached_exit            # 必须先完成露头阶段，避免从前面几何位置误触发。
            and not self.left_has_grasped       # 左手还没有成功接应过
            and touch_left_gripper              # 左手夹爪确实接触到了针
        ):
            self.left_has_grasped = True
            reward += 75.0

        # 触发 4：针尾完全拔出
        # 穿孔方向为沿着 X 轴向负方向移动，针尾小于出口 且 距离小于设定值（防止从墙外面绕过）
        if (
            self.left_has_grasped                                       # 必须先完成左手接应阶段，避免从前面几何位置误触发。
            and not self.needle_completely_through                      # 针尾还没有完全拔出
            and curr_dists['tail_pos'][0] < curr_dists['exit_pos'][0]   # 针尾必须在出口的负方向（洞口的另一侧）
            and curr_dists['dist_tail_to_exit'] < 0.03                  # 针尾部与出口距离
        ):
            self.needle_completely_through = True
            reward += 500.0  # 穿针完成是任务核心，给最高阶段奖励

        # ==========================================
        # 🌟 差分连续奖励 (Dense Differential Rewards)
        # ==========================================
        # 差分倍率：例如设为 150，意味着机器人每靠近目标 1 厘米(0.01m)，就能获得 +1.5 分。
        # 如果原地不动，差分为 0，就会实打实地吃满 -1 的时间惩罚！
        diff_scale = 150.0

        if not self.left_has_grasped:
            # --- 前半场：右臂主导 ---
            if not touch_right_gripper:
                # 引导右臂接近整根针；同时给当前距离势能，避免慢速靠近时被时间惩罚完全盖住。
                progress = self._prev_dists['dist_right_to_needle'] - curr_dists['dist_right_to_needle']
                reward += diff_scale * progress
                right_approach_radius = 0.25
                right_near_reward = 2.0 * max(
                    0.0,
                    1.0 - curr_dists['dist_right_to_needle'] / right_approach_radius,
                )
                reward += right_near_reward
            else:
                reward += 0.8 # 保持抓取的微弱奖励
                if not self.needle_start_through:
                    # 没到达洞口，引导针头到达入口
                    progress = self._prev_dists['dist_head_to_entrance'] - curr_dists['dist_head_to_entrance']
                    reward += diff_scale * progress
                elif not self.needle_reached_exit:
                    # 到达洞口，引导针头到达出口
                    progress = self._prev_dists['dist_head_to_exit'] - curr_dists['dist_head_to_exit']
                    reward += diff_scale * progress
                else:
                    # 引导左臂接近露出的针头
                    progress = self._prev_dists['dist_left_to_mark'] - curr_dists['dist_left_to_mark']
                    reward += diff_scale * progress

        else:
            # --- 后半场：左臂主导 ---
            reward += 0.25
            if touch_right_gripper:
                reward -= 0.5 # 惩罚右手不松开

            if not touch_left_gripper:
                #  如果左手脱靶导致针掉落（跌落到洞口下方 3 厘米以上）
                if (
                    not touch_right_gripper
                    and curr_dists['needle_z'] < (curr_dists['hole_z'] - 0.03)
                ):
                    reward -= 500.0
                    self.is_success = False
                    self.terminated = True # 判负，任务彻底失败，重新开局
                else:
                    # 还在半空中，引导左臂快速抓回
                    progress = self._prev_dists['dist_left_to_mark'] - curr_dists['dist_left_to_mark']
                    reward += diff_scale * progress
            else:
                reward += 0.25
                if not self.needle_completely_through:
                    # 引导左臂往外拉
                    progress = self._prev_dists['dist_tail_to_exit'] - curr_dists['dist_tail_to_exit']
                    reward += diff_scale * progress
                else:
                    #  确保安全停稳在墙正上方
                    is_above_wall = (curr_dists['z_error'] == 0.0) and (curr_dists['x_error'] < 0.015) #and (curr_dists['y_error'] < 0.05)

                    if not is_above_wall:
                        # 引导向上抬举并对中
                        progress = self._prev_dists['composite_error_dist'] - curr_dists['composite_error_dist']
                        reward += diff_scale * progress
                    else:
                        reward += 100.0  # 回归到终点只是穿针后的附属奖励
                        self.is_success = True
                        self.terminated = True

        # ==========================================
        # 🌟 更新历史距离状态，为下一帧做准备
        # ==========================================
        self._prev_dists = curr_dists
        return float(reward)
