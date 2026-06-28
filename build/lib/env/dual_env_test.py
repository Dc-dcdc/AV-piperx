from dm_control import mjcf
from constants import (
    XML_DIR,  
    SIM_DT, SIM_PHYSICS_DT, SIM_PHYSICS_ENV_STEP_RATIO,
    LEFT_ARM_POSE, RIGHT_ARM_POSE, MIDDLE_ARM_POSE,
    LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES,
    LEFT_ACTUATOR_NAMES, RIGHT_ACTUATOR_NAMES, MIDDLE_ACTUATOR_NAMES,
    LEFT_EEF_SITE, RIGHT_EEF_SITE, MIDDLE_EEF_SITE, MIDDLE_BASE_LINK,
    LEFT_GRIPPER_JOINT_NAMES, RIGHT_GRIPPER_JOINT_NAMES
)
import mujoco.viewer
import time
import os
from transform_utils import mat2quat, xyzw_to_wxyz
import cv2
import pygame #手柄库
from diff_ik import DiffIK
from grad_ik import GradIK
import numpy as np

if __name__ == '__main__':

    MOCAP_NAME = "left_mocap"
    PHYSICS_DT=0.002  
    DT = 0.04
    PHYSICS_ENV_STEP_RATIO = int(DT/PHYSICS_DT)
    DT = PHYSICS_DT * PHYSICS_ENV_STEP_RATIO
    render_width = 720
    render_height = 720

    xml_path = os.path.join(XML_DIR, f'dual_arm.xml')
    mjcf_root = mjcf.from_path(xml_path)  
    mjcf_root.option.timestep = PHYSICS_DT  # 仿真时间步长，即控制频率
    
    physics = mjcf.Physics.from_mjcf_model(mjcf_root) # 创建Mujoco物理引擎实例，加载机器人模型并设置时间步长

    left_joints = [mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
    right_joints = [mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
    left_actuators = [mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
    right_actuators = [mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
    left_eef_site = [mjcf_root.find('site', LEFT_EEF_SITE)]
    right_eef_site = [mjcf_root.find('site', RIGHT_EEF_SITE)]
    left_gripper_joints = [mjcf_root.find('joint', name) for name in LEFT_GRIPPER_JOINT_NAMES]
    right_gripper_joints = [mjcf_root.find('joint', name) for name in RIGHT_GRIPPER_JOINT_NAMES]
    mocap_left = mjcf_root.find('body', "left_mocap")
    mocap_right = mjcf_root.find('body', "right_mocap")
    # 读取真实物理引擎中夹爪电机的控制限位
    left_gripper_range = physics.bind(left_actuators[-1]).ctrlrange # 
    right_gripper_range = physics.bind(right_actuators[-1]).ctrlrange
    # 归一化：(x - min) / (max - min)
    left_gripper_norm_fn = lambda x: (x - left_gripper_range[0]) / (left_gripper_range[1] - left_gripper_range[0])
    right_gripper_norm_fn = lambda x: (x - right_gripper_range[0]) / (right_gripper_range[1] - right_gripper_range[0])
    # 反归一化:将模型输出映射到实际控制量 y = x * (max - min) + min
    left_gripper_unnorm_fn = lambda x: x * (left_gripper_range[1] - left_gripper_range[0]) + left_gripper_range[0]
    right_gripper_unnorm_fn = lambda x: x * (right_gripper_range[1] - right_gripper_range[0]) + right_gripper_range[0]
    # 速度归一化
    left_gripper_vel_norm_fn = lambda x: x / (left_gripper_range[1] - left_gripper_range[0])
    right_gripper_vel_norm_fn = lambda x: x / (right_gripper_range[1] - right_gripper_range[0])
    # 速度反归一化：将模型输出的比例速度放大回物理速度
    left_gripper_vel_unnorm_fn = lambda x: x * (left_gripper_range[1] - left_gripper_range[0])
    right_gripper_vel_unnorm_fn = lambda x: x * (right_gripper_range[1] - right_gripper_range[0])
    # 左臂控制器，采用梯度下降逆解法
    left_controller = GradIK(
        physics=physics,
        joints = left_joints[:6],
        actuators=left_actuators[:6],
        eef_site=left_eef_site,
        step_size=0.0001, 
        min_cost_delta=1.0e-12, 
        max_iterations=50, 
        position_weight=500.0,
        rotation_weight=100.0,
        joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0]),
        joint_displacement_weight=np.array(6*[50.0]),
        position_threshold=0.001,
        rotation_threshold=0.001,
        max_pos_diff=0.1,
        max_rot_diff=0.3,
        joint_p = 0.9,
    )
    right_controller = GradIK(
        physics=physics,
        joints=right_joints[:6],
        actuators=right_actuators[:6],
        eef_site=right_eef_site,
        step_size=0.0001, 
        min_cost_delta=1.0e-12, 
        max_iterations=50, 
        position_weight=500.0,
        rotation_weight=100.0,
        joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0]),
        joint_displacement_weight=np.array(6*[50.0]),
        position_threshold=0.001,
        rotation_threshold=0.001,
        max_pos_diff=0.1,
        max_rot_diff=0.3,
        joint_p = 0.9,
    )
    physics.bind(left_joints).qpos = LEFT_ARM_POSE # 初始化左臂的关节角度
    physics.bind(left_gripper_joints).qpos = left_gripper_unnorm_fn(1) # 夹爪张开到最大
    physics.bind(right_joints).qpos = RIGHT_ARM_POSE
    physics.bind(right_gripper_joints).qpos = right_gripper_unnorm_fn(1)

    physics.bind(left_actuators).ctrl = LEFT_ARM_POSE # 初始化左臂的控制器，和左臂的关节角度一样
    physics.bind(left_actuators[6]).ctrl = left_gripper_unnorm_fn(1) #初始化左臂的夹爪控制器
    physics.bind(right_actuators).ctrl = RIGHT_ARM_POSE
    physics.bind(right_actuators[6]).ctrl = right_gripper_unnorm_fn(1)

    physics.bind(mocap_left).mocap_pos = physics.bind(left_eef_site).xpos
    physics.bind(mocap_left).mocap_quat = xyzw_to_wxyz(mat2quat(physics.bind(left_eef_site).xmat.reshape(3,3)))
    physics.bind(mocap_right).mocap_pos = physics.bind(right_eef_site).xpos
    physics.bind(mocap_right).mocap_quat = xyzw_to_wxyz(mat2quat(physics.bind(right_eef_site).xmat.reshape(3,3)))
    
    # ==========================================
    # 🎮 初始化 Pygame 手柄控制器
    # ==========================================
    pygame.init()
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0: # 检测到至少一个手柄连接，获取第一个手柄并初始化
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"\n[SUCCESS] 成功连接手柄: {joystick.get_name()}\n")
    else:
        print("\n[WARNING] 未检测到手柄，请通过蓝牙或 USB 连接 Xbox/PS 手柄！\n")
    # ==========================================
    with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
        while viewer.is_running():
            step_start = time.time()
            # ==========================================
            # 🕹️ 读取手柄摇杆输入并移动 Mocap 目标点
            # ==========================================
            if joystick is not None:
                pygame.event.pump() # 强制刷新手柄内部事件队列
                
                DEADZONE = 0.05      # 摇杆死区（防止手柄老化导致的自动漂移）
                MAX_SPEED = 0.005    # 每次循环的最大移动步长（单位：米，可根据手感调节）
                
                # 读取摇杆轴数据 (-1.0 到 1.0 的模拟量)
                # 注意：不同品牌手柄的轴映射可能不同，这里以标准 Xbox 布局为例
                left_y = joystick.get_axis(0)  # 左摇杆上下 (控制 X 轴前后)
                left_x = joystick.get_axis(1)  # 左摇杆左右 (控制 Y 轴左右)
                right_y = joystick.get_axis(4) # 右摇杆上下 (控制 Z 轴上下)
                lt_raw = joystick.get_axis(2) # 左摇杆触发器 (控制夹爪)
                rt_raw = joystick.get_axis(5) # 右摇杆触发器 (控制夹爪)
                # 过滤死区并计算增量 (根据你的视角习惯，正负号可能需要微调)
                dx = left_y * MAX_SPEED if abs(left_y) > DEADZONE else 0.0
                dy = -left_x * MAX_SPEED if abs(left_x) > DEADZONE else 0.0
                dz = -right_y * MAX_SPEED if abs(right_y) > DEADZONE else 0.0
                
                # 更新 mocap 目标位置
                current_pos = physics.bind(mocap_left).mocap_pos.copy()
                current_pos[0] += dx
                current_pos[1] += dy
                current_pos[2] += dz
                physics.bind(mocap_left).mocap_pos = current_pos #更新mocap位置
                # ----------------- 新增：计算并发送夹爪指令 -----------------
                # 1. 扳机原始值 -1.0 到 1.0 -> 转换为按压比例 0.0 到 1.0
                lt_pressed_ratio = (lt_raw + 1.0) / 2.0
                # 2. 映射到夹爪状态：不按(0.0)->张开(1.0)；按到底(1.0)->闭合(0.0)
                left_gripper_target = 1.0 - lt_pressed_ratio
                rt_pressed_ratio = (rt_raw + 1.0) / 2.0
                right_gripper_target = 1.0 - rt_pressed_ratio
                # 3. 反归一化并发送给夹爪电机 (列表中索引为6的actuator)
                physics.bind(left_actuators[6]).ctrl = left_gripper_unnorm_fn(left_gripper_target)
                physics.bind(right_actuators[6]).ctrl = right_gripper_unnorm_fn(right_gripper_target)
            # ==========================================
            mocap_pos_left = physics.bind(mocap_left).mocap_pos # 获取动捕mocap的位置信息，作为末端执行器的目标位置
            mocap_quat_left = physics.bind(mocap_left).mocap_quat
            start = time.time()
            physics.bind(left_actuators[:6]).ctrl = left_controller.run(physics.bind(left_joints).qpos[:6], mocap_pos_left, mocap_quat_left)
            # print("Time taken: ", time.time() - start)
            physics.step(nstep=PHYSICS_ENV_STEP_RATIO)

            # ==========================================
            # 📸 双目相机截取与拼接模块
            # ==========================================
            # 1. 分别向 physics 索要左右相机的 RGB 画面 (假设分辨率 720x720)
            img_left_rgb = physics.render(height=render_height, width=render_width, camera_id="zed_cam_left")
            img_right_rgb = physics.render(height=render_height, width=render_width, camera_id="zed_cam_right")

            # 2. 将 RGB 转换为 OpenCV 的 BGR 格式
            img_left_bgr = cv2.cvtColor(img_left_rgb, cv2.COLOR_RGB2BGR)
            img_right_bgr = cv2.cvtColor(img_right_rgb, cv2.COLOR_RGB2BGR)

            # 3. 核心魔法：使用 numpy.hstack 将两张图水平拼接 (变成 720 x 1440 的宽图)
            # 如果你想上下拼接，可以使用 np.vstack
            stereo_img = np.hstack((img_left_bgr, img_right_bgr)) 
            
            # 4. 用 OpenCV 显示拼接后的宽屏画面
            cv2.imshow("ZED Stereo Camera (Left & Right)", stereo_img)
            cv2.waitKey(1) # 必须有这一句，否则窗口会卡死
            # ==========================================
            viewer.sync() # 同步渲染器和物理引擎

            time_until_next_step = DT - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)  