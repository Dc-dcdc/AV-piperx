import numpy as np
from data_collect.transform_utils import angular_error, wxyz_to_xyzw, quat2mat
from kinematics import create_fk_fn, create_safety_fn, create_jac_fn
from numba import jit
import mujoco

# 基于差分逆运动学的控制器，使用雅可比矩阵和阻尼最小二乘法来计算关节速度，以使末端执行器达到目标位姿，同时考虑了关节限制和冗余度
class DiffIK():
    def __init__(
        self, 
        physics, # mujoco引擎实例，包含机器人真实状态信息
        joints, # 机器人关节列表
        actuators, # 机器人执行器列表，对应于关节的控制输入
        eef_site, # 末端执行器位姿
        k_pos, # 位置控制增益
        k_ori, # 方向控制增益
        damping, # 阻尼系数
        k_null, # 虚拟关节控制增益
        q0, # 关节初始位置
        max_angvel, # 最大关节速度
        integration_dt, # 积分时间步长
        iterations, # 迭代次数，用于提高控制精度
    ):
        self.physics = physics # mujoco引擎实例，包含机器人真实状态信息
        self.joints = joints # 机器人关节列表
        self.actuators = actuators # 机器人执行器列表，对应于关节的控制输入
        self.eef_site = eef_site # 末端执行器位姿
        self.k_pos = k_pos # 位置控制增益
        self.k_ori = k_ori # 方向控制增益
        self.damping = damping # 阻尼系数
        self.k_null = k_null # 虚拟关节控制增益
        self.q0 = q0 # 关节初始位置
        self.max_angvel = max_angvel # 最大关节速度
        self.integration_dt = integration_dt # 积分时间步长
        self.iterations = iterations # 迭代次数，用于提高控制精度

        self.diff_ik_fn = self.make_diff_ik_fn()

    def make_diff_ik_fn(self): #只需要调用一次，返回逆运动学函数，后续每次控制时直接调用这个函数即可，避免重复计算一些不变的量，提高效率
        integration_dt = self.integration_dt
        k_pos = self.k_pos
        k_ori = self.k_ori
        fk_fn = create_fk_fn(self.physics, self.joints, self.eef_site) # 返回正向运动学计算函数，可以输入关节角度，输出末端执行器的位姿矩阵
        self.fk_fn = fk_fn
        jac_fn = create_jac_fn(self.physics, self.joints, self.eef_site) # 返回末端 site 在关节空间中的雅可比矩阵
        diag = np.ascontiguousarray(self.damping * np.eye(6)) # 阻尼矩阵，6×6 的对角矩阵，阻尼系数乘以单位矩阵，用于阻尼最小二乘法计算关节速度时增加数值稳定性
        eye = np.ascontiguousarray(np.eye(len(self.joints))) # 单位矩阵，大小为关节数量×关节数量，用于计算冗余度控制的投影矩阵
        k_null = self.k_null
        q0 = self.q0
        max_angvel = self.max_angvel
        joint_limits = self.physics.bind(self.joints).range.copy()

        '''
        带有 @jit 的函数第一次被调用时，Numba 会启动编译器，把这段 Python 代码翻译成底层的 C/机器码，
        之后的调用会直接使用编译好的机器码，极大提高运行速度
         '''
        @jit(nopython=True, fastmath=True, cache=False) # 使用Numba加速计算     
        def diff_ik(q, target_pos, target_quat, iterations=1):
            '''
            1.计算位姿误差和目标速度
            2.获取当前关节角计算雅可比矩阵
            3.计算关节角速度并优化
            4.输出目标关节角度
            '''
            for _ in range(iterations):
                current_pose = fk_fn(q) #输入关节角度，计算当前末端执行器的位姿矩阵
                current_pos = current_pose[:3, 3]
                current_mat = current_pose[:3, :3]

                target_pos = target_pos
                target_mat = quat2mat(wxyz_to_xyzw(target_quat))

                twist = np.zeros(6)
                dx = target_pos - current_pos # 位置误差
                twist[:3] = k_pos * dx / integration_dt # 位置误差乘以位置增益，除以积分时间步长，得到位置控制的线速度分量
                dr = angular_error(target_mat, current_mat) # 旋转误差
                twist[3:] = k_ori *dr / integration_dt # 旋转误差乘以方向增益，除以积分时间步长，得到方向控制的角速度分量

                # Jacobian.
                jac = jac_fn(q) # 输入关节角度，计算末端执行器在关节空间中的雅可比矩阵

                # Damped least squares. 
                # 使用阻尼最小二乘法计算关节速度，避免雅可比矩阵奇异时的数值不稳定问题，diag是一个对角矩阵，增加了一个小的正数来确保矩阵的可逆性
                dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)# \Delta q = J^T (J J^T + \lambda I)^{-1} V_{twist}

                # Null space control.
                # 计算冗余度控制的关节速度分量，将 k_null * (q0 - q) 投影到雅可比矩阵的零空间中，
                # 使得在满足末端执行器控制的同时，关节也朝着初始位置 q0 的方向移动，增加了冗余度控制的效果
                # 例如夹爪在没有指令的时候，零空间会牵引夹爪保持在默认位置，防止它因为重力或惯性乱晃
                dq += (eye - np.linalg.pinv(jac) @ jac) @ (k_null * (q0 - q))

                # Limit joint velocity. 
                dq = np.clip(dq, -max_angvel, max_angvel) # 限制关节速度在最大速度范围内

                # integrate
                q = q + dq * integration_dt # 根据计算得到的关节速度 dq 和积分时间步长 integration_dt，更新关节位置 q

                # Limit joint position.
                q = np.clip(q, joint_limits[:,0], joint_limits[:,1]) # 限制关节位置在关节限制范围内
            
            return q
        
        return diff_ik

    def run(self, q, target_pos, target_quat):
        return self.diff_ik_fn(q, target_pos, target_quat, iterations=self.iterations)

if __name__ == '__main__':
    from dm_control import mjcf
    from constants import XML_DIR, MIDDLE_ACTUATOR_NAMES, MIDDLE_ARM_POSE, MIDDLE_JOINT_NAMES, MIDDLE_EEF_SITE
    import mujoco.viewer
    import time
    import os
    from data_collect.transform_utils import mat2quat, xyzw_to_wxyz
    import cv2
    import pygame #手柄库
    MOCAP_NAME = "target"
    PHYSICS_DT=0.002
    DT = 0.04
    PHYSICS_ENV_STEP_RATIO = int(DT/PHYSICS_DT)
    DT = PHYSICS_DT * PHYSICS_ENV_STEP_RATIO
    render_width = 720
    render_height = 720
    '''    
    相比于传统的 mujoco.MjModel 和 mujoco.MjData，即：
    model = mujoco.MjModel.from_xml_path(xml_path) #模型一旦加载就不可修改，必须重新加载才能改变参数
    data = mujoco.MjData(model) #数据结构，包含了模型的状态信息，可以在仿真过程中修改和访问，但无法直接修改模型参数
    mjcf 模块提供了更高层次的接口，将XML转换为可编辑的mjcf_root对象树，可以进行参数修改和各个xml文件的组合，
    最后通过mjcf.Physics.from_mjcf_model(mjcf_root) 创建物理引擎实例，加载机器人模型并设置时间步长。
    '''
    xml_path = os.path.join(XML_DIR, f'single_arm.xml')
    mjcf_root = mjcf.from_path(xml_path)  
    mjcf_root.option.timestep = PHYSICS_DT  # 仿真时间步长，即控制频率
    
    physics = mjcf.Physics.from_mjcf_model(mjcf_root) # 创建Mujoco物理引擎实例，加载机器人模型并设置时间步长

    left_joints = [mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
    left_actuators = [mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
    left_eef_site = mjcf_root.find('site', MIDDLE_EEF_SITE)
    mocap = mjcf_root.find('body', MOCAP_NAME)

    # set up controllers
    left_controller = DiffIK(
        physics=physics, # mujoco引擎实例，包含机器人真实状态信息
        joints = left_joints, # 机器人关节列表
        actuators=left_actuators, # 机器人执行器列表，对应于关节的控制输入
        eef_site=left_eef_site, # 末端执行器位姿
        k_pos=0.3, # 位置控制增益
        k_ori=0.3, # 方向控制增益
        damping=1.0e-4, # 阻尼系数
        k_null=np.array([20.0, 10.0, 10.0, 10.0, 5.0, 5.0]), # 虚拟关节控制增益
        q0=np.array(MIDDLE_ARM_POSE), # 关节初始位置
        max_angvel=3.14, # 最大关节速度
        integration_dt=DT, # 积分时间步长
        iterations=10 # 迭代次数，用于提高控制精度
    )

    physics.bind(left_joints).qpos = MIDDLE_ARM_POSE
    physics.bind(left_actuators).ctrl = MIDDLE_ARM_POSE
    physics.bind(mocap).mocap_pos = physics.bind(left_eef_site).xpos 
    physics.bind(mocap).mocap_quat = xyzw_to_wxyz(mat2quat(physics.bind(left_eef_site).xmat.reshape(3,3)))
    
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
                left_y = joystick.get_axis(1)  # 左摇杆上下 (控制 X 轴前后)
                left_x = joystick.get_axis(0)  # 左摇杆左右 (控制 Y 轴左右)
                right_y = joystick.get_axis(3) # 右摇杆上下 (控制 Z 轴上下)
                
                # 过滤死区并计算增量 (根据你的视角习惯，正负号可能需要微调)
                dx = -left_y * MAX_SPEED if abs(left_y) > DEADZONE else 0.0
                dy = -left_x * MAX_SPEED if abs(left_x) > DEADZONE else 0.0
                dz = -right_y * MAX_SPEED if abs(right_y) > DEADZONE else 0.0
                
                # 更新 mocap 目标位置
                current_pos = physics.bind(mocap).mocap_pos.copy()
                current_pos[0] += dx
                current_pos[1] += dy
                current_pos[2] += dz
                physics.bind(mocap).mocap_pos = current_pos
            # ==========================================
            mocap_pos = physics.bind(mocap).mocap_pos # 获取动捕mocap的位置信息，作为末端执行器的目标位置
            mocap_quat = physics.bind(mocap).mocap_quat
            start = time.time()
            physics.bind(left_actuators).ctrl = left_controller.run(physics.bind(left_joints).qpos, mocap_pos, mocap_quat)
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
