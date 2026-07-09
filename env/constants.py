"""集中管理整个项目中不变的全局变量"""
import pathlib
import os
# task parameters
#获取当前运行文件所在文件夹的绝对路径
XML_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), 'assets', 'task_insert_cylinder_piper.xml') #/脚本所在目录/assets/task_insert_cylinder_piper.xml
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets' #/脚本所在目录/assets

"""
MuJoCo内部物理频率: 500 Hz
env.step动作执行频率: 25 Hz 40ms
Diffusion重新推理频率: 3.125 Hz(包含8个动作步骤，每个动作步长40ms，总共320ms)，即每320ms重新推理一次动作块
每次推理执行动作块长度: 8步 = 0.32秒
"""
# control parameters
#上位机（主控电脑）每秒钟 50 次去读取 VR 头显和手柄的位姿（Pose），进行逆运动学（IK）解算，并将计算出的关节目标角度下发给真实的机械臂底层电机
REAL_DT = 0.02 # 控制回路（发送指令给电机、读取传感器）的运行频率是 50 Hz

# physics parameters
SIM_PHYSICS_DT=0.002 # 物理引擎（计算刚体动力学、碰撞、摩擦力等）每 0.002 秒（500 Hz）进行一次演算。
SIM_DT = 0.04  #读取操作员动捕设备或模型推理动作的频率：25HZ, 40ms
SIM_PHYSICS_ENV_STEP_RATIO = int(SIM_DT/SIM_PHYSICS_DT) #AI 每输出一个动作，仿真器会在底层保持这个动作不变，连续运行 20 次物理演算（每次 0.002 秒），然后再把第 20 次演算后的最新状态返回给 AI。
SIM_DT = SIM_PHYSICS_DT * SIM_PHYSICS_ENV_STEP_RATIO # 强制确保最终的 SIM_DT 绝对是底层物理步长的整数倍，这样可以保证仿真器的稳定性

# PiperX robot parameters
LEFT_ARM_POSE = [0.0, 1.60, -0.30, -1.16, 0, 0.0, 0.027] # 左臂默认初始姿态
RIGHT_ARM_POSE = [0.0, 1.60, -0.30, -1.16, 0, 0.0, 0.027] # 右臂默认初始姿态
MIDDLE_ARM_POSE = [0.0, 1.1, -1.1, 0.37, 0.0, 0.0] # 中臂初始姿态，6 轴
LEFT_BASE_POS = [-0.55, 0.15, 0.0] # 左臂默认底座位置
RIGHT_BASE_POS = [0.55, 0.15, 0.0] # 右臂默认底座位置
MIDDLE_BASE_POS = [0.0, -0.65, 0.0] # 中臂默认底座位置

# 按任务管理机器人初始配置；env/__init__.py 会读取并传入对应环境。
ROBOT_INIT_CONFIGS = {
    "default": {
        "left_arm_pose": LEFT_ARM_POSE,
        "right_arm_pose": RIGHT_ARM_POSE,
        "middle_arm_pose": MIDDLE_ARM_POSE,
        "left_base_pos": LEFT_BASE_POS,
        "right_base_pos": RIGHT_BASE_POS,
        "middle_base_pos": MIDDLE_BASE_POS,
    },
    "insert_cylinder": {
        "left_arm_pose": LEFT_ARM_POSE,
        "right_arm_pose": RIGHT_ARM_POSE,
        "middle_arm_pose": MIDDLE_ARM_POSE,
        "left_base_pos": LEFT_BASE_POS,
        "right_base_pos": RIGHT_BASE_POS,
        "middle_base_pos": MIDDLE_BASE_POS,
    },
}
LEFT_JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_left_finger", #夹爪的左指关节
]
RIGHT_JOINT_NAMES = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_right_finger", #夹爪的右指关节
]
MIDDLE_JOINT_NAMES = [
    "middle_waist",
    "middle_shoulder",
    "middle_elbow",
    "middle_forearm_roll",
    "middle_wrist_1_joint",
    "middle_wrist_2_joint",
]
LEFT_ACTUATOR_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
]
RIGHT_ACTUATOR_NAMES = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
MIDDLE_ACTUATOR_NAMES = [
    "middle_waist",
    "middle_shoulder",
    "middle_elbow",
    "middle_forearm_roll",
    "middle_wrist_1_joint",
    "middle_wrist_2_joint",
]
LEFT_ARM_ACTION_DIM = len(LEFT_JOINT_NAMES)
RIGHT_ARM_ACTION_DIM = len(RIGHT_JOINT_NAMES)
MIDDLE_ARM_ACTION_DIM = len(MIDDLE_JOINT_NAMES)
TWO_ARM_ACTION_DIM = LEFT_ARM_ACTION_DIM + RIGHT_ARM_ACTION_DIM
THREE_ARM_ACTION_DIM = TWO_ARM_ACTION_DIM + MIDDLE_ARM_ACTION_DIM
MIDDLE_ARM_ACTION_START = TWO_ARM_ACTION_DIM
LEFT_EEF_SITE = "left_gripper_control"
RIGHT_EEF_SITE = "right_gripper_control"
MIDDLE_EEF_SITE = "middle_zed_camera_center"
MIDDLE_BASE_LINK = "middle_base_link"
LEFT_GRIPPER_JOINT_NAMES = ["left_left_finger", "left_right_finger"]
RIGHT_GRIPPER_JOINT_NAMES = ["right_left_finger", "right_right_finger"]

# Gripper joint limits (qpos[6])
LEFT_GRIPPER_JOINT_OPEN = 0.05982525274157524
LEFT_GRIPPER_JOINT_CLOSE = -0.99055535531044006
RIGHT_GRIPPER_JOINT_OPEN =   0.11044661700725555
RIGHT_GRIPPER_JOINT_CLOSE = -1.0139613151550293

# TODO: ANDREW SET THESE VALUES
LEFT_MASTER_GRIPPER_JOINT_OPEN = 0.6596117615699768
LEFT_MASTER_GRIPPER_JOINT_CLOSE = -0.1672039031982422
RIGHT_MASTER_GRIPPER_JOINT_OPEN = 0.7240389585494995
RIGHT_MASTER_GRIPPER_JOINT_CLOSE = -0.07976700365543365

############################ Helper functions ############################
LEFT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - LEFT_GRIPPER_JOINT_CLOSE) / (LEFT_GRIPPER_JOINT_OPEN - LEFT_GRIPPER_JOINT_CLOSE)
LEFT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (LEFT_GRIPPER_JOINT_OPEN - LEFT_GRIPPER_JOINT_CLOSE) + LEFT_GRIPPER_JOINT_CLOSE
RIGHT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - RIGHT_GRIPPER_JOINT_CLOSE) / (RIGHT_GRIPPER_JOINT_OPEN - RIGHT_GRIPPER_JOINT_CLOSE)
RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (RIGHT_GRIPPER_JOINT_OPEN - RIGHT_GRIPPER_JOINT_CLOSE) + RIGHT_GRIPPER_JOINT_CLOSE
LEFT_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (LEFT_GRIPPER_JOINT_OPEN - LEFT_GRIPPER_JOINT_CLOSE)
RIGHT_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (RIGHT_GRIPPER_JOINT_OPEN - RIGHT_GRIPPER_JOINT_CLOSE)

LEFT_MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - LEFT_MASTER_GRIPPER_JOINT_CLOSE) / (LEFT_MASTER_GRIPPER_JOINT_OPEN - LEFT_MASTER_GRIPPER_JOINT_CLOSE)
LEFT_MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (LEFT_MASTER_GRIPPER_JOINT_OPEN - LEFT_MASTER_GRIPPER_JOINT_CLOSE) + LEFT_MASTER_GRIPPER_JOINT_CLOSE
RIGHT_MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - RIGHT_MASTER_GRIPPER_JOINT_CLOSE) / (RIGHT_MASTER_GRIPPER_JOINT_OPEN - RIGHT_MASTER_GRIPPER_JOINT_CLOSE)
RIGHT_MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (RIGHT_MASTER_GRIPPER_JOINT_OPEN - RIGHT_MASTER_GRIPPER_JOINT_CLOSE) + RIGHT_MASTER_GRIPPER_JOINT_CLOSE
