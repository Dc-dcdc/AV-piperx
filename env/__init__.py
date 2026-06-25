from gymnasium.envs.registration import register

from env.constants import ROBOT_INIT_CONFIGS

# ==========================================
# 🌟 环境配置字典
# ==========================================
ENVS = {
    # ==========================================
    #  穿针任务 (3臂)
    # ==========================================
    "guided_vision/SewNeedle-3Arms-v0": {   
        "entry_point": "env.task.sew_needle_env:SewNeedleEnv",  # 指向你的文件路径和类名
        "num_arms": 3,
        "episode_length": 400,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
        "init_config": "sew_needle",
    },
    # 穿针任务 (2臂)
    "guided_vision/SewNeedle-2Arms-v0": {
        "entry_point": "env.task.sew_needle_env:SewNeedleEnv",
        "num_arms": 2,
        "episode_length": 400,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
        "init_config": "sew_needle",
    },


    # ==========================================
    #  插槽插入任务 (3臂)
    # ==========================================
    "guided_vision/SlotInsertion-3Arms-v0": {   
        "entry_point": "env.task.sim_envs:SlotInsertionEnv",  # 指向你的文件路径和类名
        "num_arms": 3,
        "episode_length": 400,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
        "init_config": "slot_insertion",
    },
    # ==========================================
    #  松灵 Piper 挡板遮挡圆柱插入容器任务 (3臂主入口)
    # ==========================================
    "guided_vision/InsertCylinder-3Arms-v0": {
        "entry_point": "env.task.insert_cylinder_env:InsertCylinderEnv",
        "num_arms": 3,
        "episode_length": 400,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
        "init_config": "insert_cylinder",
    },
    # ==========================================
    #  兼容旧命令的 Piper 圆柱插入别名，配置同 InsertCylinder-3Arms-v0
    # ==========================================
    "guided_vision/InsertCylinder-Piper3Arms-v0": {
        "entry_point": "env.task.insert_cylinder_env:InsertCylinderEnv",
        "num_arms": 3,
        "episode_length": 400,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
        "init_config": "insert_cylinder",
    },
    # 💡 如果你有其他的任务（比如插孔），可以继续在这里添加：
    # "guided_vision/InsertPeg-3Arms-v0": {
    #     "entry_point": "env.task.sim_envs:InsertPegEnv",
    #     "num_arms": 3,
    #     "cameras": ["zed_cam_left", "zed_cam_right", "overhead_cam"],
    #     "observation_height": 480,
    #     "observation_width": 640,
    # },
}

# ==========================================
# 🚀 批量注册环境
# ==========================================
for env_id, env_kwargs in ENVS.items():
    init_config_name = env_kwargs.get("init_config", "default")
    init_config = ROBOT_INIT_CONFIGS[init_config_name]
    kwargs = {
        "num_arms": env_kwargs["num_arms"],
        "cameras": env_kwargs["cameras"],
        "episode_length": env_kwargs["episode_length"],
        "observation_height": env_kwargs["observation_height"],
        "observation_width": env_kwargs["observation_width"],
    }
    kwargs.update(init_config)
    for key in (
        "left_arm_pose",
        "right_arm_pose",
        "middle_arm_pose",
        "left_base_pos",
        "right_base_pos",
        "middle_base_pos",
    ):
        if key in env_kwargs:
            kwargs[key] = env_kwargs[key]

    register(
        id=env_id,
        entry_point=env_kwargs["entry_point"],
        # nondeterministic=True 告诉 Gym 这个环境的渲染/物理可能不是绝对确定的，防止 check_env 测试报错
        nondeterministic=True, 
        # kwargs 里的参数会直接传递给你 SewNeedleEnv 类的 __init__ 方法
        kwargs=kwargs
    )
