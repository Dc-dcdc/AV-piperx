import numpy as np
from numba import jit, float64, boolean
from numba.types import UniTuple
import math

PI = np.pi
EPS = np.finfo(float).eps * 4.0

@jit(float64[:](float64[:,:]), nopython=True, fastmath=True, cache=True)
def mat2quat(rmat):
    # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/transform_utils.py
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = rmat

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


@jit(float64[:,:](float64[:]), nopython=True, fastmath=True, cache=True)
def quat2mat(quaternion):
    # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/transform_utils.py
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


@jit(float64[:](float64[:]), nopython=True, fastmath=True, cache=True)
def quat2axisangle(quat):
    # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/transform_utils.py
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

@jit(float64[:](float64[:]), nopython=True, fastmath=True, cache=True)
def axisangle2quat(vec):
    # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/transform_utils.py
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if np.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


@jit(float64[:,:](float64[:], float64[:]), nopython=True, fastmath=True, cache=True)
def pose2mat(pos, quat):
    # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/transform_utils.py
    """
    Converts pose to homogeneous matrix.

    Args:
        pose (x,y,z,qx,qy,qz,qw): 7x1 pose vector

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.eye(4)
    homo_pose_mat[:3, :3] = quat2mat(quat)
    homo_pose_mat[:3, 3] = pos
    return homo_pose_mat

@jit(UniTuple(float64[:], 2)(float64[:,:]), nopython=True, fastmath=True, cache=True)
def mat2pose(homo_pose_mat):
    pos = homo_pose_mat[:3, 3]
    quat = mat2quat(homo_pose_mat[:3, :3])
    return pos, quat

@jit(nopython=True, fastmath=True, cache=True)
def xyzw_to_wxyz(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])

@jit(nopython=True, fastmath=True, cache=True)
def wxyz_to_xyzw(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])

# 
@jit(float64[:,:](float64[:,:]), nopython=True, fastmath=True, cache=True)
def align_rotation_to_z_axis(matrix):     
    # Get the current z-axis of the quaternion
    current_z_axis = matrix[:, 2]
    
    # Calculate the rotation needed to align the z-axis with the global z-axis
    rotation_needed = quat2mat(axisangle2quat(np.cross(current_z_axis, [0, 0, 1])))

    matrix = np.ascontiguousarray(matrix)
    rotation_needed = np.ascontiguousarray(rotation_needed)
    
    # Combine the original rotation with the rotation needed to align the z-axis
    aligned_rotation = rotation_needed @ matrix

    return aligned_rotation

@jit(float64[:](float64[:,:], float64[:,:]), nopython=True, fastmath=True, cache=True)
def angular_error(desired, current):
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    return error

@jit(boolean(float64[:], float64[:,:], float64[:], float64[:,:], float64, float64), nopython=True, fastmath=True, cache=True)
def within_pose_threshold(current_xpos, current_xmat, target_xpos, target_xmat, position_threshold=0.001, rotation_threshold=0.001):
    pos_err = np.linalg.norm(target_xpos - current_xpos)
    rot_err = np.linalg.norm(angular_error(target_xmat, current_xmat))

    return pos_err < position_threshold and rot_err < rotation_threshold

# 将当前位姿从当前坐标系转换到目标坐标系
# @jit(float64[:,:](float64[:,:],float64[:,:],float64[:,:]), nopython=True, fastmath=True, cache=True)
# def transform_coordinates(current_pose, current_frame, target_frame):
#     current_frame = np.ascontiguousarray(current_frame)
#     current_pose = np.ascontiguousarray(current_pose)
#     target_frame = np.ascontiguousarray(target_frame)
#     # Unity 当前位姿相对起始位姿 = Unity 起始位姿矩阵的逆 · Unity 当前位姿矩阵的逆
#     offset_matrix = np.dot(np.linalg.inv(current_frame), current_pose) 
#     # mujoco中目标位姿 = mujoco初始位姿矩阵 · Unity 当前位姿相对起始位姿
#     new_pose = np.dot(target_frame, offset_matrix)
#     return new_pose

# unity初始位姿标定 + unity中相对位姿变化 + 转换为mujoco坐标系 + 叠加到机械臂初始位姿
# def transform_coordinates(
#     current_unity_mat: np.ndarray,
#     start_unity_mat: np.ndarray,
#     start_mujoco_mat: np.ndarray,
# ) -> np.ndarray:
#     S = np.array([
#         [0.0,  0.0, 1.0],   # MuJoCo x = Unity z
#         [-1.0, 0.0, 0.0],   # MuJoCo y = -Unity x
#         [0.0,  -1.0, 0.0],  # MuJoCo z = -Unity y
#     ], dtype=np.float64)

#     T_S = np.eye(4)
#     T_S[:3, :3] = S

#     # 计算手柄相对初始变化 = 初始点相对于unity坐标系的逆 x 当前点相对于unity坐标系
#     delta_u = np.linalg.inv(start_unity_mat) @ current_unity_mat
#     # 坐标系 unity ->  MuJoCo
#     delta_m = T_S @ delta_u @ np.linalg.inv(T_S)

#     # 叠加机械臂末端初始位姿
#     T_m_target = start_mujoco_mat @ delta_m

#     return T_m_target

"""
将 Unity 手柄当前位姿映射为 MuJoCo 机械臂末端目标位姿。

设计目标：
- 三个机械臂末端初始坐标系可以不同；
- 手柄在 Unity 中的同一方向运动，映射到 MuJoCo 后保持一致的世界方向；
- 避免 start_mujoco_mat @ delta_m 导致平移被末端初始姿态旋转。

current_unity_mat:
    Unity 当前手柄位姿，{}^{U_o}T_{A_U}

start_unity_mat:
    Unity 初始手柄位姿，{}^{U_o}T_{B_U}

start_mujoco_mat:
    MuJoCo 机械臂初始末端位姿，{}^{M_o}T_{B_M}
"""
def transform_coordinates(
    current_unity_mat: np.ndarray,
    start_unity_mat: np.ndarray,
    start_mujoco_mat: np.ndarray,
) -> np.ndarray:
    S = np.array([
        [1.0,  0.0, 0.0],   # MuJoCo x = Unity x
        [0.0, 0.0, 1.0],    # MuJoCo y = Unity z
        [0.0,  1.0, 0.0],   # MuJoCo z =  Unity y
    ], dtype=np.float64)
    S_inv = np.linalg.inv(S)

    T_S = np.eye(4)
    T_S[:3, :3] = S

    T_S_inv = np.eye(4)
    T_S_inv[:3, :3] = S_inv

    # Unity 中手柄相对于启动时刻的完整相对位姿：
    # ΔT_U = (T_U_start)^-1 T_U_current
    delta_u = np.linalg.inv(start_unity_mat) @ current_unity_mat

    # 将 Unity 相对位姿换基到 MuJoCo：
    # ΔT_M = T_S ΔT_U T_S^-1
    delta_m = T_S @ delta_u @ T_S_inv

    # ---------- 位置 ----------
    # 只取换基后的平移增量，并直接加到 MuJoCo 世界坐标下的初始末端位置。
    # 关键点：不写 start_mujoco_mat @ delta_m。
    # 否则 delta_m 的平移会被 start_mujoco_mat 的旋转部分再次旋转，
    # 三个初始末端坐标系不同的机械臂就会向不同方向运动。
    p_m_start = start_mujoco_mat[:3, 3]
    p_m_target = p_m_start + delta_m[:3, 3]

    # ---------- 姿态 ----------
    # 手柄相对启动姿态的局部旋转增量。
    R_u_start = start_unity_mat[:3, :3]
    R_u_current = current_unity_mat[:3, :3]
    R_delta_u = R_u_start.T @ R_u_current

    # 将旋转增量换基到 MuJoCo。
    R_delta_m = S @ R_delta_u @ S_inv

    # 将旋转增量作为 MuJoCo 世界旋转增量作用到初始末端姿态。
    R_m_start = start_mujoco_mat[:3, :3]
    R_m_target = R_delta_m @ R_m_start

    T_m_target = start_mujoco_mat.copy()
    T_m_target[:3, 3] = p_m_target
    T_m_target[:3, :3] = R_m_target

    return T_m_target





@jit(nopython=True, fastmath=True, cache=True)
def skew_sym(x):
    """ convert 3D vector to skew-symmetric matrix form """
    x1, x2, x3 = x.ravel()
    return np.array([
        [0, -x3, x2],
        [x3, 0, -x1],
        [-x2, x1, 0]
    ])

@jit(nopython=True, fastmath=True, cache=True)
def exp2rot(w, theta):
    """Matrix exponential of rotations (Rodrigues' Formula)

    Convert exponential coordinates to rotation matrix 
    """
    ss_w = skew_sym(w)
    R = np.eye(3) + np.sin(theta) * ss_w + (1-np.cos(theta)) * np.dot(ss_w, ss_w)
    return R

@jit(nopython=True, fastmath=True, cache=True)
def exp2mat(w, v, theta): 
    """Matrix exponential of rigid-body motions
    
    Convert exponential coordinates to transformation matrix
    """
    
    w_norm = np.linalg.norm(w)
    v_norm = np.linalg.norm(v)
    
    if np.isclose(w_norm, 0):
        assert np.isclose(v_norm, 1), 'norm(v) must be 1'
        new_v = v.ravel() * theta
        return np.vstack((
            np.hstack((
                exp2rot(w, theta), new_v[:,np.newaxis]
            )),
            np.array([[0, 0, 0, 1]]),
        ))
    
    assert np.isclose(w_norm, 1), 'norm(w) must be 1'
    
    ss_w = skew_sym(w)
    new_v = (np.eye(3)*theta + (1-np.cos(theta))*ss_w + (theta-np.sin(theta))*np.dot(ss_w, ss_w)).dot(v) 
    return np.vstack((
        np.hstack((
            exp2rot(w, theta), new_v[:,np.newaxis]
        )),
        np.array([[0, 0, 0, 1]]),
    ))

@jit(nopython=True, fastmath=True, cache=True)
def limit_pose(current_xpos, current_xmat, target_xpos, target_xmat, max_pos_diff=0.1, max_rot_diff=0.1):
    # Calculate position difference
    pos_diff = target_xpos - current_xpos
    pos_diff_norm = np.linalg.norm(pos_diff)

    # Limit position difference
    if pos_diff_norm > max_pos_diff:
        pos_diff = (pos_diff / pos_diff_norm) * max_pos_diff

    limited_xpos = current_xpos + pos_diff

    # Calculate rotation difference
    relative_xmat = np.ascontiguousarray(target_xmat) @ np.linalg.inv(np.ascontiguousarray(current_xmat))
    relative_rotvec = quat2axisangle(mat2quat(relative_xmat))
    rot_diff_angle = np.linalg.norm(relative_rotvec)

    # Limit rotation difference
    if rot_diff_angle > max_rot_diff:
        limited_relative_xmat = quat2mat(axisangle2quat(relative_rotvec * (max_rot_diff / rot_diff_angle)))
        limited_xmat = np.ascontiguousarray(limited_relative_xmat) @ np.ascontiguousarray(current_xmat)
    else:
        limited_xmat = target_xmat

    return limited_xpos, limited_xmat

@jit(nopython=True, fastmath=True, cache=True)
def adjoint(T):
    """ adjoint representation of transformation """
    R = T[:3, :3]
    p = T[:3, 3]
    pR = np.dot(skew_sym(p), np.ascontiguousarray(R))

    ret = np.zeros((6, 6))
    ret[:3, :3] = R
    ret[3:, 3:] = R
    ret[3:, :3] = pR

    return ret
