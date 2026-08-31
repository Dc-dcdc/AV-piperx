import numpy as np
from numba import jit, float64
from numba.types import UniTuple
from scipy.spatial.transform import Rotation as R
from data_collect.transform_utils import (
    pose2mat,
    mat2pose,
)

TRANSFORM_TO_WORLD = np.ascontiguousarray(np.eye(4))
TRANSFORM_TO_WORLD[:3, :3] = R.from_euler('xyz', [-90, 0, -90], degrees=True).as_matrix()
WORLD_TO_TRANSFORM = np.ascontiguousarray(np.linalg.inv(TRANSFORM_TO_WORLD))

# 定义一个类来存储头戴设备的数据
class HeadsetData:
    h_pos = np.zeros(3) #头部位置
    h_quat = np.zeros(4) #头部旋转
    l_pos = np.zeros(3) #左臂位置
    l_quat = np.zeros(4) #左臂旋转
    l_thumbstick_x = 0 #左手柄X轴
    l_thumbstick_y = 0 #左手柄Y轴
    l_index_trigger = 0 #左手食指触发器
    l_hand_trigger = 0 #左手握持触发器
    l_button_one = False #左手按钮一
    l_button_two = False #左手按钮二
    l_button_thumbstick = False #左手拇指按键
    r_pos = np.zeros(3) #右臂位置
    r_quat = np.zeros(4) #右臂旋转
    r_thumbstick_x = 0 #右手柄X轴
    r_thumbstick_y = 0 #右手柄Y轴
    r_index_trigger = 0 #右手食指触发器
    r_hand_trigger = 0 #右手握持触发器
    r_button_one = False #右手按钮一
    r_button_two = False #右手按钮二
    r_button_thumbstick = False #右手拇指按键

# 定义一个类来存储头戴设备的反馈信息，将信息发送给头戴设备
class HeadsetFeedback:
    head_out_of_sync = False #表示头部是否不同步
    left_out_of_sync = False #表示左臂是否不同步
    right_out_of_sync = False #表示右臂是否不同步
    info = ""
    left_arm_position = np.zeros(3)
    left_arm_rotation = np.zeros(4)
    right_arm_position = np.zeros(3)
    right_arm_rotation = np.zeros(4)
    middle_arm_position = np.zeros(3)
    middle_arm_rotation = np.zeros(4)

"""
机器人世界（MuJoCo / ROS）: 默认使用右手坐标系 (Right-Handed)。
VR 游戏世界（Unity / Meta Quest）： 默认使用左手坐标系 (Left-Handed)。
因此，在将头戴设备的数据转换为机器人世界的坐标系时，需要进行坐标系转换。
具体来说，左手坐标系中的 y 轴需要翻转（乘以 -1），同时旋转也需要进行相应的调整（例如，绕 x 轴旋转 -90 度，绕 z 轴旋转 -90 度）。同样地，在将机器人世界的坐标系转换回左手坐标系时，也需要进行相应的调整。
"""
# 定义一个函数来将左臂的位姿转换为右臂的位姿，使用numba进行加速
@jit(UniTuple(float64[:], 2)(float64[:], float64[:]), nopython=True, fastmath=True, cache=True)
def convert_left_to_right_coordinates(left_pos, left_quat):

    x = left_pos[0]
    y = -left_pos[1] # flip y from left to right
    z = left_pos[2]
    qx = -left_quat[0] # flip rotation from left to right
    qy = left_quat[1]
    qz = -left_quat[2] # flip rotation from left to right
    qw = left_quat[3]

    transform = pose2mat(np.array([x, y, z]), np.array([qx, qy, qz, qw]))

    transform = np.ascontiguousarray(transform)

    transform = TRANSFORM_TO_WORLD @ transform

    right_pos, right_quat = mat2pose(transform)

    return right_pos, right_quat

# 定义一个函数来将右臂的位姿转换为左臂的位姿，使用numba进行加速 
@jit(UniTuple(float64[:], 2)(float64[:], float64[:]), nopython=True, fastmath=True, cache=True)
def convert_right_to_left_coordinates(right_pos, right_quat):

    transform = pose2mat(right_pos, right_quat)

    transform = np.ascontiguousarray(transform)

    transform = WORLD_TO_TRANSFORM @ transform

    pos, quat = mat2pose(transform)

    x = pos[0]
    y = -pos[1] # flip y from right to left
    z = pos[2]
    qx = -quat[0] # flip rotation from right to left
    qy = quat[1]
    qz = -quat[2] # flip rotation from right to left
    qw = quat[3]

    return np.array([x, y, z]), np.array([qx, qy, qz, qw])
