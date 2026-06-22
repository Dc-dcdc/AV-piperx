import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import mjcf
import mujoco.viewer
from lerobot.common.envs.utils import preprocess_observation

from env.constants import (
    XML_DIR,
    SIM_DT, SIM_PHYSICS_DT, SIM_PHYSICS_ENV_STEP_RATIO,
    LEFT_ARM_POSE, RIGHT_ARM_POSE, MIDDLE_ARM_POSE,
    LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES,
    LEFT_ACTUATOR_NAMES, RIGHT_ACTUATOR_NAMES, MIDDLE_ACTUATOR_NAMES,
    LEFT_EEF_SITE, RIGHT_EEF_SITE, MIDDLE_EEF_SITE, MIDDLE_BASE_LINK,
    LEFT_GRIPPER_JOINT_NAMES, RIGHT_GRIPPER_JOINT_NAMES
)

CAMERAS = ['zed_cam_left', 'zed_cam_right', 'wrist_cam_left', 'wrist_cam_right', 'overhead_cam', 'worms_eye_cam']

class GuidedVisionEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1/SIM_DT}

    def __init__(self,
            xml_path: str,
            num_arms: int = 3,
            episode_length: int = 300,
            cameras: list[str] = CAMERAS,
            observation_height: int = 480,
            observation_width: int = 640,
            left_arm_pose: list[float] | None = None,
            right_arm_pose: list[float] | None = None,
            middle_arm_pose: list[float] | None = None,
            left_base_pos: list[float] | None = None,
            right_base_pos: list[float] | None = None,
            middle_base_pos: list[float] | None = None,
        ):
        super().__init__()
        assert num_arms in [2, 3], f"Invalid number of arms: {num_arms}"
        assert all([camera in CAMERAS for camera in cameras]), f"Invalid camera names: {cameras}"
        # self.num_envs = 1
        # ==========================================
        # 🌟 1. 加载物理模型
        # ==========================================
        self.cameras = cameras # 使用的摄像头列表
        self.num_arms = num_arms
        self.left_arm_pose = np.asarray(left_arm_pose if left_arm_pose is not None else LEFT_ARM_POSE, dtype=np.float64)
        self.right_arm_pose = np.asarray(right_arm_pose if right_arm_pose is not None else RIGHT_ARM_POSE, dtype=np.float64)
        self.middle_arm_pose = np.asarray(middle_arm_pose if middle_arm_pose is not None else MIDDLE_ARM_POSE, dtype=np.float64)
        self._mjcf_root = mjcf.from_path(xml_path)                       # 加载 MJCF 模型
        for body_name, base_pos in (
            ("left_base_link", left_base_pos),
            ("right_base_link", right_base_pos),
            (MIDDLE_BASE_LINK, middle_base_pos),
        ):
            if base_pos is None:
                continue
            body = self._mjcf_root.find("body", body_name)
            if body is None:
                raise ValueError(f"找不到需要设置初始位置的 body: {body_name}")
            body.pos = np.asarray(base_pos, dtype=np.float64)
        self._physics = mjcf.Physics.from_mjcf_model(self._mjcf_root)    # 构建物理引擎
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._mjcf_root.option.timestep = SIM_PHYSICS_DT                 

        self.episode_length = episode_length
        self._middle_base_link = self._mjcf_root.find('body', MIDDLE_BASE_LINK)
        self._middle_base_link_init_pos = self._middle_base_link.pos.copy()

        if self.num_arms == 2:
            self.hide_middle_arm() # HACK, 隐藏中央机械臂
            self.num_joints = 14
        elif self.num_arms == 3:
            self.num_joints = 21
        # ==========================================
        # 🌟 2. 构建观察空间 (Observation Space)
        # ==========================================
        """
        {
        "pixels": {
            "cam_1": Box(...),
            "cam_2": Box(...)
        },
        "agent_pos": Box(...)
        }
        """
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        camera : spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.observation_height, self.observation_width, 3),
                            dtype=np.uint8,
                        )
                        for camera in self.cameras
                    }
                ),
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_joints,),
                    dtype=np.float64,
                ),
            }
        )
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints,), dtype=np.float32)

        # ==========================================
        # 🌟 4. 寻址与绑定 MJCF 节点，和底层的 MuJoCo XML 物理模型之间建立连接
        # ==========================================

        # 绑定关节，用于读取关节角度
        self._left_joints = [self._mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
        self._right_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self._middle_joints = [self._mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
        # 绑定动作，用于发送控制指令
        self._left_actuators = [self._mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
        self._right_actuators = [self._mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
        self._middle_actuators = [self._mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
        # 绑定夹爪关节进行单独控制
        self._left_gripper_joints = [self._mjcf_root.find('joint', name) for name in LEFT_GRIPPER_JOINT_NAMES]
        self._right_gripper_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_GRIPPER_JOINT_NAMES]

        # ==========================================
        # 🌟 5. 夹爪归一化函数
        # ==========================================
        # 读取真实物理引擎中夹爪电机的控制限位
        self.left_gripper_range = self._physics.bind(self._left_actuators[-1]).ctrlrange
        self.right_gripper_range = self._physics.bind(self._right_actuators[-1]).ctrlrange
        # 归一化：(x - min) / (max - min)
        self.left_gripper_norm_fn = lambda x: (x - self.left_gripper_range[0]) / (self.left_gripper_range[1] - self.left_gripper_range[0])
        self.right_gripper_norm_fn = lambda x: (x - self.right_gripper_range[0]) / (self.right_gripper_range[1] - self.right_gripper_range[0])
        # 反归一化:将模型输出映射到实际控制量 y = x * (max - min) + min
        self.left_gripper_unnorm_fn = lambda x: x * (self.left_gripper_range[1] - self.left_gripper_range[0]) + self.left_gripper_range[0]
        self.right_gripper_unnorm_fn = lambda x: x * (self.right_gripper_range[1] - self.right_gripper_range[0]) + self.right_gripper_range[0]

        self._viewer = None

    def get_obs(self) -> dict:
        """获取并格式化模型输入所需的字典"""
        # ==========================================
        # 🌟 1. 提取本体感知状态 (Proprioceptive State)
        # ==========================================
        left_qpos = self._physics.bind(self._left_joints).qpos.copy()
        left_qpos[6] = self.left_gripper_norm_fn(left_qpos[6]) # 夹爪数据归一化

        right_qpos = self._physics.bind(self._right_joints).qpos.copy()
        right_qpos[6] = self.right_gripper_norm_fn(right_qpos[6])

        middle_qpos = self._physics.bind(self._middle_joints).qpos.copy()

        if self.num_arms == 2:
            agent_pos = np.concatenate([left_qpos, right_qpos]).astype(np.float64)
        elif self.num_arms == 3:
            agent_pos = np.concatenate([left_qpos, right_qpos, middle_qpos]).astype(np.float64)
        # state_21d = np.concatenate([left_qpos, right_qpos, middle_qpos]).astype(np.float32)

        return {
            'pixels': {
                camera: self._physics.render(
                    height=self.observation_height,
                    width=self.observation_width,
                    camera_id=camera
                )
                for camera in self.cameras
            },
            'agent_pos': agent_pos,
        }
        # ==========================================
        # 🌟 2. 渲染相机图像并转换通道为 (C, H, W)
        # ==========================================
        # 2. 准备返回的字典
        # obs_dict = {'observation.state': agent_pos}
        # for cam_name in self.cameras:
        #     # 注意：这里的 camera_id 必须和你的 XML 文件里 <camera name="..."> 的名字完全一致！
        #     try:
        #         img = self._physics.render(height=self.observation_height, width=self.observation_width, camera_id=cam_name)
        #         img = np.transpose(img, (2, 0, 1))# 转换通道 (H, W, C) -> (C, H, W)
        #         # 存入字典，键名严格对齐
        #         obs_dict[f'observation.images.{cam_name}'] = img
        #     except Exception as e:
        #         raise ValueError(f"❌ 渲染相机 '{cam_name}' 失败！请检查 XML 文件中是否有这个名字的相机。报错详情: {e}")

        # return obs_dict

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        self._physics.reset()
        # 🌟 新增：重置回合内部的步数计数器
        self._current_step = 0
        # 恢复默认位姿
        self._physics.bind(self._left_joints).qpos = self.left_arm_pose
        self._physics.bind(self._left_gripper_joints).qpos = self.left_gripper_unnorm_fn(1) # 夹爪张开到最大
        self._physics.bind(self._right_joints).qpos = self.right_arm_pose
        self._physics.bind(self._right_gripper_joints).qpos = self.right_gripper_unnorm_fn(1)
        self._physics.bind(self._middle_joints).qpos = self.middle_arm_pose
        # 初始化控制器
        self._physics.bind(self._left_actuators).ctrl = self.left_arm_pose
        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(1)
        self._physics.bind(self._right_actuators).ctrl = self.right_arm_pose
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(1)
        self._physics.bind(self._middle_actuators).ctrl = self.middle_arm_pose
        # 强制物理引擎进行一次正向运动学计算
        self._physics.forward()
        self.terminated = False
        self.is_success = False
        # 读取当前环境的观测
        observation = self.get_obs()
        info = {"message": "Environment reset successfully."}
        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        """Gymnasium 标准步进函数"""
        # 1. 引擎防爆护盾 (拦截 NaN 和无穷大)
        if np.isnan(action).any() or np.isinf(action).any():
            print("⚠️ 警告：检测到非法动作 (NaN/Inf)，已启动安全降级为全 0 动作！")
            action = np.zeros_like(action)

        # 2. 动作拆包
        left_joints = action[:6]
        # left_gripper = action[6]
        left_gripper = np.clip(action[6], 0.0, 1.0) # 0.0 到 1.0 之间的归一化值
        right_joints = action[7:13]
        # right_gripper = action[13]
        right_gripper = np.clip(action[13], 0.0, 1.0)
        if self.num_arms == 3:
            middle_joints = action[14:21]
            self._physics.bind(self._middle_actuators).ctrl = middle_joints

        # 3. 映射到物理引擎执行器
        self._physics.bind(self._left_actuators[:6]).ctrl = left_joints
        self._physics.bind(self._right_actuators[:6]).ctrl = right_joints

        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(left_gripper)
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(right_gripper)

        # 4. 步进物理引擎
        for _ in range(SIM_PHYSICS_ENV_STEP_RATIO): self._physics.step()
        self._current_step += 1   # 步数追踪

        # 5. 获取观察与奖励
        observation = self.get_obs()
        reward = self.get_reward() if hasattr(self, 'get_reward') else 0.0

        # 6. 判断终止条件
        truncated = bool(self._current_step >= self.episode_length) # 超出最大步数

        info = {
            "is_success": bool(getattr(self, "is_success", False)),
            "reward": reward,
            "step": self._current_step,
        }
        reward_debug = getattr(self, "reward_debug", None)
        if reward_debug is not None:
            info["reward_debug"] = reward_debug

        return observation, float(reward),  self.terminated, truncated, info

    def render(self,render_camera):
        """
        Gymnasium 标准渲染接口，供 LeRobot 等高级框架录制视频时调用。
        必须返回 (H, W, C) 维度的 numpy 图像矩阵。
        """
        # 选择一个最好的相机视角用来生成测试录像（比如用左目相机）
        # 这里默认使用 self.cameras 列表里的第一个相机
        # render_cam = self.cameras[0] if len(self.cameras) > 0 else 'zed_cam_left'
        render_cam = render_camera[0] if len(render_camera) > 0 else 'overhead_cam'

        try:
            # MuJoCo 的 render 默认输出的就是标准的 (H, W, C) rgb_array
            img = self._physics.render(height=self.observation_height, width=self.observation_width, camera_id=render_cam)
            return img
        except Exception as e:
            # 防止万一没找到相机导致崩溃
            print(f"⚠️ 渲染视频帧失败: {e}")
            return np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)

    def render_viewer(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr, self._physics.data.ptr,
                show_left_ui=True, show_right_ui=True,
            )
        self._viewer.sync()

    # 将中间臂隐藏（移出视角外）
    def hide_middle_arm(self):
        self._physics.bind(self._middle_base_link).pos = np.array([0, -2.4, -0.4]) # HACK


    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()


class SlotInsertionEnv(GuidedVisionEnv):
    def __init__(self, **kwargs):
        xml = os.path.join(XML_DIR, 'task_slot_insertion.xml')
        super().__init__(xml, **kwargs)

        self.max_reward = 4

        self._slot_joint = self._mjcf_root.find('joint', 'slot_joint')
        self._stick_joint = self._mjcf_root.find('joint', 'stick_joint')

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed, options=options)
        rng = self.np_random

        # reset physics
        x_range = [-0.05, 0.05]
        y_range = [0.1, 0.15]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        slot_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        slot_quat = np.array([1, 0, 0, 0])


        peg_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        peg_quat = np.array([1, 0, 0, 0])

        x_range = [-0.08, 0.08]
        y_range = [-0.1, 0.0]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        stick_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        stick_quat = np.array([1, 0, 0, 0])

        self._physics.bind(self._slot_joint).qpos = np.concatenate([slot_position, slot_quat])
        self._physics.bind(self._stick_joint).qpos = np.concatenate([stick_position, stick_quat])

        self._physics.forward()

        observation = self.get_obs()
        info = {"is_success": False}

        return observation, info


    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        stick_touch_table = False
        stick_touch_slot = False
        pins_touch = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "stick" and geom2.startswith("right"):
                touch_right_gripper = True

            if geom1 == "stick" and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "stick":
                stick_touch_table = True

            if geom1 == "stick" and geom2.startswith("slot-"):
                stick_touch_slot = True

            if geom1 == "pin-stick" and geom2 == "pin-slot":
                pins_touch = True

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not stick_touch_table): # grasp stick
            reward = 2
        if stick_touch_slot and (not stick_touch_table): # peg and socket touching
            reward = 3
        if pins_touch: # successful insertion
            reward = 4
        return reward


if __name__ == "__main__":
    import argparse

    import cv2

    parser = argparse.ArgumentParser(description="加载指定 Gym 任务环境或 MuJoCo XML，并拼接显示指定相机画面。")
    parser.add_argument(
        "--env-id",
        default="guided_vision/InsertCylinder-3Arms-v0",
        help="要加载的 Gym 环境 ID。默认加载插圆柱任务；传空字符串并指定 --xml 可直接查看 XML。",
    )
    parser.add_argument(
        "--xml",
        default=None,
        help="要加载的 MJCF/XML 文件路径。相对路径会按当前工作目录解析。",
    )
    parser.add_argument("--num-arms", type=int, default=3, choices=[2, 3], help="使用 2 臂或 3 臂模型。")
    parser.add_argument("--height", type=int, default=480, help="相机渲染高度。")
    parser.add_argument("--width", type=int, default=640, help="相机渲染宽度。")
    parser.add_argument("--max-cols", type=int, default=2, help="OpenCV 拼接窗口中每行最多显示几个相机。")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        help="需要拼接显示的相机名称。",
    )
    args = parser.parse_args()

    xml_path = args.xml

    viewer_cmd = {"reset": False, "quit": False}

    def key_callback(keycode):
        if keycode == 32:  # Space
            viewer_cmd["reset"] = True
        elif keycode in (81, 113, 256):  # Q/q/Esc
            viewer_cmd["quit"] = True

    def make_camera_grid(frames_bgr, max_cols):
        max_cols = max(1, int(max_cols))
        grid_rows = []
        for i in range(0, len(frames_bgr), max_cols):
            row_frames = frames_bgr[i:i + max_cols]
            while len(row_frames) < max_cols:
                row_frames.append(np.zeros_like(frames_bgr[0]))
            grid_rows.append(np.hstack(row_frames))
        return np.vstack(grid_rows)

    print(f"显示相机: {args.cameras}")
    if args.env_id:
        import env  # 注册 Gym 环境

        print(f"加载 Gym 环境: {args.env_id}")
        sim_env = gym.make(
            args.env_id,
            disable_env_checker=True,
            num_arms=args.num_arms,
            episode_length=10**9,
            cameras=args.cameras,
            observation_height=args.height,
            observation_width=args.width,
        ).unwrapped
        display_name = args.env_id
    else:
        if xml_path is None:
            xml_path = os.path.join(XML_DIR, "task_insert_cylinder.xml")
        if not os.path.isabs(xml_path):
            xml_path = os.path.abspath(xml_path)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"找不到 MuJoCo XML 文件: {xml_path}")
        print(f"加载 MuJoCo XML: {xml_path}")
        sim_env = GuidedVisionEnv(
            xml_path=xml_path,
            num_arms=args.num_arms,
            episode_length=10**9,
            cameras=args.cameras,
            observation_height=args.height,
            observation_width=args.width,
        )
        display_name = os.path.basename(xml_path)
    sim_env.reset()

    viewer = mujoco.viewer.launch_passive(
        sim_env._physics.model.ptr,
        sim_env._physics.data.ptr,
        show_left_ui=True,
        show_right_ui=True,
        key_callback=key_callback,
    )

    window_name = "MuJoCo Camera Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width * args.max_cols, args.height * 2)

    print("\n控制说明:")
    print("  Space : 重置模型")
    print("  Q/Esc : 退出")

    step_count = 0
    try:
        while viewer.is_running() and not viewer_cmd["quit"]:
            step_start = time.time()

            if viewer_cmd["reset"]:
                sim_env.reset()
                step_count = 0
                viewer_cmd["reset"] = False

            frames_bgr = []
            for camera_name in args.cameras:
                img_rgb = sim_env._physics.render(
                    height=args.height,
                    width=args.width,
                    camera_id=camera_name,
                )
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    img_bgr,
                    camera_name,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2,
                )
                frames_bgr.append(img_bgr)

            if frames_bgr:
                combined_img = make_camera_grid(frames_bgr, args.max_cols)
                h, _ = combined_img.shape[:2]
                cv2.putText(
                    combined_img,
                    f"{display_name} | step={step_count}",
                    (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow(window_name, combined_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                viewer_cmd["reset"] = True
            elif key in (ord("q"), ord("Q"), 27):
                viewer_cmd["quit"] = True

            for _ in range(SIM_PHYSICS_ENV_STEP_RATIO):
                sim_env._physics.step()
            step_count += 1
            viewer.sync()

            time_until_next_step = SIM_DT - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    finally:
        cv2.destroyAllWindows()
        viewer.close()
        sim_env.close()

    # 窗口关闭后清理资源
    cv2.destroyAllWindows()
    env.close()
