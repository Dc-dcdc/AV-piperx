from gymnasium.envs.registration import register

from env.constants import ROBOT_INIT_CONFIGS


ENVS = {
    "guided_vision/InsertCylinder-3Arms-v0": {
        "entry_point": "env.task.insert_cylinder_env:InsertCylinderEnv",
        "num_arms": 3,
        "episode_length": 400,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
        "init_config": "insert_cylinder",
    },
    "guided_vision/SewNeedle-3Arms-v0": {
        "entry_point": "env.task.sew_needle_env:SewNeedleEnv",
        "num_arms": 3,
        "episode_length": 350,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
        "init_config": "sew_needle",
    },
    "guided_vision/HookPackage-3Arms-v0": {
        "entry_point": "env.task.hook_package_env:HookPackageEnv",
        "num_arms": 3,
        "episode_length": 400,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
        "init_config": "hook_package",
    },
    "guided_vision/InsertPeg-3Arms-v0": {
        "entry_point": "env.task.insert_peg_env:InsertPegEnv",
        "num_arms": 3,
        "episode_length": 400,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
        "init_config": "insert_peg",
    },
    "guided_vision/OpenDrawerRetrieve-3Arms-v0": {
        "entry_point": "env.task.open_drawer_retrieve_env:OpenDrawerRetrieveEnv",
        "num_arms": 3,
        "episode_length": 400,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
        "init_config": "open_drawer_retrieve",
    },
}


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
        nondeterministic=True,
        kwargs=kwargs,
    )
