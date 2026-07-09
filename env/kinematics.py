
import numpy as np
import mujoco
from data_collect.transform_utils import exp2mat, adjoint, within_pose_threshold, pose2mat, wxyz_to_xyzw
from numba import jit, prange

# 基于PoE的正运动学，比自带的mujoco.mj_kinematics更快，且可以直接输出位姿矩阵，方便后续计算雅可比矩阵和差分逆运动学
def create_fk_fn(physics, joints, eef_site):
    qpos_before = physics.data.qpos.copy()
    physics.bind(joints).qpos = np.zeros(len(joints)) # 将关节位置设置为零，以计算初始位姿
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr) # 调用Mujoco的正运动学函数，计算末端执行器的位姿
    w0 = physics.bind(joints).xaxis.copy() # 获取每个关节的旋转轴向量
    p0 = physics.bind(joints).xanchor.copy() # 获取每个关节的锚点位置，可以理解为关节的旋转中心位置（在世界坐标系里的物理中心点）
    v0 = -np.cross(w0, p0) # 计算每个关节的线速度分量，得到完整的螺旋轴表示
    site0 = np.eye(4) # 4×4 的单位矩阵（对角线为 1，其余全 0），用于存储末端执行器的初始位姿（旋转矩阵和平移向量）
    site0[:3, :3] = physics.bind(eef_site).xmat.reshape(3,3).copy() # 提取末端执行器的旋转矩阵
    site0[:3, 3] = physics.bind(eef_site).xpos.copy() # 提取末端执行器的位置
    physics.data.qpos[:] = qpos_before
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
    '''
    带有 @jit 的函数第一次被调用时，Numba 会启动编译器，把这段 Python 代码翻译成底层的 C/机器码，
    之后的调用会直接使用编译好的机器码，极大提高运行速度
    '''
    @jit(nopython=True, fastmath=True, cache=False)  # 这是 Numba 里给函数加速的装饰器
    def forward_kinematics(theta): # 输入关节角度，输出末端执行器的位姿矩阵
        M = np.eye(4)
        M[:,:] = site0
        for i in prange(len(theta)-1, -1, -1): 
            T = exp2mat(w0[i], v0[i], theta[i]) # 计算每个关节的变换矩阵
            M = np.dot(T, M) # 从末端执行器开始，依次右乘每个关节的变换矩阵，得到最终的位姿矩阵
        return M
    
    return forward_kinematics #返回编译的机器码函数，可以直接调用，速度非常快，输入关节角度，输出末端执行器的位姿矩阵
# 雅可比矩阵的计算，基于PoE的正运动学实现，可以直接输出末端执行器在关节空间中的雅可比矩阵，方便后续计算差分逆运动学
def create_jac_fn(physics, joints, eef_site=None):
    qpos_before = physics.data.qpos.copy()
    physics.bind(joints).qpos = np.zeros(len(joints)) #设置关节角度为初始位置
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr) #计算此时末端执行器位姿
    w0 = physics.bind(joints).xaxis.copy() # 各关节的旋转轴向量
    p0 = physics.bind(joints).xanchor.copy() # 各关节的锚点位置（关节在世界坐标系里的物理中心点）
    v0 = -np.cross(w0, p0) # 各关节的线速度分量
    S = np.ascontiguousarray(np.hstack((w0, v0)))
    site0 = np.eye(4)
    if eef_site is not None:
        site0[:3, :3] = physics.bind(eef_site).xmat.reshape(3, 3).copy()
        site0[:3, 3] = physics.bind(eef_site).xpos.copy()
    physics.data.qpos[:] = qpos_before
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
    '''
    带有 @jit 的函数第一次被调用时，Numba 会启动编译器，把这段 Python 代码翻译成底层的 C/机器码，
    之后的调用会直接使用编译好的机器码，极大提高运行速度
    '''
    if eef_site is not None:
        @jit(nopython=True, fastmath=True, cache=False)
        def site_jacobian(theta): # 输入关节角度，输出末端 site 在关节空间中的雅可比矩阵
            J_twist = np.zeros((6, len(theta)))
            Ts = np.eye(4)

            for i in range(len(theta)):
                J_twist[:, i] = adjoint(Ts) @ S[i, :]
                Ts = np.dot(Ts, exp2mat(w0[i], v0[i], theta[i]))

            site_pose = np.dot(Ts, site0)
            site_pos = site_pose[:3, 3]
            J = np.zeros((6, len(theta)))
            for i in range(len(theta)):
                w = J_twist[:3, i]
                v = J_twist[3:, i]
                J[:3, i] = v + np.cross(w, site_pos)
                J[3:, i] = w

            return J

        return site_jacobian

    @jit(nopython=True, fastmath=True, cache=False)
    def jacobian(theta): # 输入关节角度，输出空间 twist 雅可比矩阵
        # screw axis at rest place
        J = np.zeros((6, len(theta))) # 6行，len(theta)列的零矩阵，用于存储雅可比矩阵，前3行对应位置雅可比，后3行对应旋转雅可比
        Ts = np.eye(4) # 4×4 的单位矩阵，用于存储当前的变换矩阵

        # compute each column of the Jacobian
        for i in prange(len(theta)):
            
            J[:, i] = adjoint(Ts) @ S[i,:] # PoE法计算雅可比矩阵的第i列
            Ts = np.dot(Ts, exp2mat(w0[i], v0[i], theta[i])) # 从末端执行器开始，依次右乘每个关节的变换矩阵，得到当前的变换矩阵，然后计算雅可比矩阵的每一列

        # swap jacp and jacr
        J = J[np.array((3,4,5,0,1,2)),:] # 将雅可比矩阵的前3行和后3行交换，[wx, wy, wz, vx, vy, vz] -> [vx, vy, vz, wx, wy, wz]

        return J
    
    return jacobian #返回编译的机器码函数，可以直接调用，速度非常快，输入关节角度，输出末端执行器在关节空间中的雅可比矩阵

@jit(nopython=True, fastmath=True)
def safety(
    qpos, 
    ctrl, 
    Taction,
    fk_fn,
    joint_limits,
    xyz_bounds,
    joint_tracking_safety_margin,
    eef_pos_tracking_safety_margin,
    eef_rot_tracking_safety_margin,
):
    # check if difference of any qpos and ctrl is too large
    if np.any(np.abs(qpos - ctrl) > joint_tracking_safety_margin):
        return False, "Joint tracking safety margin exceeded"
    
    # check that not near joint limits
    if np.any(qpos < joint_limits[:,0]) or np.any(qpos > joint_limits[:,1]):
        return False, "Joint limit safety margin exceeded"
    
    # check that not near boundaries of workspace
    Tqpos = fk_fn(qpos)
    if np.any(Tqpos[:3, 3] < xyz_bounds[:,0]) or np.any(Tqpos[:3, 3] > xyz_bounds[:,1]):
        return False, "End effector position outside bounds"
    
    if Taction is not None:
        if np.any(Taction[:3,3] < xyz_bounds[:,0]) or np.any(Taction[:3,3] > xyz_bounds[:,1]):
            return False, "End effector action position outside bounds"
                    
        if not within_pose_threshold(
            Tqpos[:3, 3],
            Tqpos[:3, :3],
            Taction[:3, 3],
            Taction[:3, :3],
            eef_pos_tracking_safety_margin, 
            eef_rot_tracking_safety_margin):
            return False, "End effector pose tracking safety margin exceeded"

    return True, ""


# check that ctrl and qpos are close to each other
# check that not near joint limits
# check that not near boundaries of workspace
# check that action is reasonably close to current position
def create_safety_fn(
    physics,
    joints,
    eef_site,
    xyz_bounds,
    joint_limit_safety_margin=0.01,
    joint_tracking_safety_margin=1.0,
    eef_pos_tracking_safety_margin=0.2,
    eef_rot_tracking_safety_margin=3.0):
    
    joint_limit_safety_margin = joint_limit_safety_margin
    xyz_bounds = np.array(xyz_bounds)
    joint_tracking_safety_margin = joint_tracking_safety_margin
    eef_pos_tracking_safety_margin = eef_pos_tracking_safety_margin
    eef_rot_tracking_safety_margin = eef_rot_tracking_safety_margin
    fk_fn = create_fk_fn(physics, joints, eef_site)
    joint_limits = physics.bind(joints).range.copy()
    # check safety margin is not larger than joint limit half range
    assert np.all(joint_limit_safety_margin < (joint_limits[:,1] - joint_limits[:,0])/2)
    # add some safety margin to joint limits
    joint_limits[:,0] += joint_limit_safety_margin
    joint_limits[:,1] -= joint_limit_safety_margin

    def safety_fn(qpos, ctrl, Taction=None):
        return safety(
            qpos,
            ctrl,
            Taction,
            fk_fn,
            joint_limits,
            xyz_bounds,
            joint_tracking_safety_margin,
            eef_pos_tracking_safety_margin,
            eef_rot_tracking_safety_margin
        )
    
    return safety_fn



if __name__ == "__main__":
    from constants import XML_DIR, LEFT_JOINT_NAMES, LEFT_ACTUATOR_NAMES, LEFT_EEF_SITE
    from dm_control import mjcf
    from data_collect.transform_utils import mat2quat
    import os

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    
    # set some random pose
    LEFT_ARM_POSE = np.array([1,0,0,-1,0,1])

    # setup mujoco 
    mjcf_root = mjcf.from_path(os.path.join(XML_DIR, 'piperx_sim.xml'))
    physics = mjcf.Physics.from_mjcf_model(mjcf_root) 
    left_joints = [mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
    left_actuators = [mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
    left_eef_site = mjcf_root.find('site', LEFT_EEF_SITE)
    left_eef_site_id = physics.bind(left_eef_site).element_id
    jnt_dof_ids = physics.bind(left_joints[:6]).dofadr

    # test forward kinematics in mujoco
    physics.bind(left_joints[:6]).qpos = LEFT_ARM_POSE[:6]
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
    left_xpos = physics.data.site_xpos[left_eef_site_id]
    left_xmat = physics.data.site_xmat[left_eef_site_id].reshape(3,3)
    left_quat = mat2quat(left_xmat)
    print("Mujoco FK left arm pose: ", left_xpos, left_quat)

    # test forward kinematics in custom implementation
    fk_fn = create_fk_fn(physics, left_joints[:6], left_eef_site)
    theta = np.array(LEFT_ARM_POSE[:6]).copy()
    M = fk_fn(theta)
    print("PoE FK left arm pose: ", M[:3, 3], mat2quat(M[:3, :3]))

    # test jacobian in mujoco
    physics.bind(left_joints[:6]).qpos = LEFT_ARM_POSE[:6]
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
    jac = np.zeros((6, physics.model.nv))
    mujoco.mj_jacBody(physics.model.ptr, physics.data.ptr, jac[:3], jac[3:], physics.bind(left_eef_site).bodyid)
    jac = jac[:,jnt_dof_ids]
    print("Mujoco Jacobian: ", jac)

    # test jacobian in custom implementation
    jac_fn = create_jac_fn(physics, left_joints[:6])
    theta = np.array(LEFT_ARM_POSE[:6]).copy()
    J = jac_fn(theta)
    print("PoE Jacobian: ", J)
