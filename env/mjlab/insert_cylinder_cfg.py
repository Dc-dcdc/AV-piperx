from __future__ import annotations

from pathlib import Path

import mujoco

from env.constants import (
    LEFT_JOINT_NAMES,
    MIDDLE_JOINT_NAMES,
    RIGHT_JOINT_NAMES,
    XML_DIR,
)
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
    action_rate_l2,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    reset_scene_to_default,
    time_out,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from env.mjlab import events, observations, rewards, terminations

TASK_ID = "Mjlab-Insert-Cylinder-PiperX"
ROBOT_ENTITY = "robot"
CYLINDER_ENTITY = "insert_cylinder"
CONTAINER_ENTITY = "cylinder_container"
BAFFLE_ENTITY = "baffle"

ACTION_JOINT_NAMES = tuple(LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES + MIDDLE_JOINT_NAMES)
ALL_ROBOT_JOINT_POS = {
    "left_waist": 0.0,
    "left_shoulder": 1.60,
    "left_elbow": -0.30,
    "left_forearm_roll": -1.16,
    "left_wrist_angle": 0.0,
    "left_wrist_rotate": 0.0,
    "left_left_finger": 0.027,
    "left_right_finger": 0.027,
    "right_waist": 0.0,
    "right_shoulder": 1.60,
    "right_elbow": -0.30,
    "right_forearm_roll": -1.16,
    "right_wrist_angle": 0.0,
    "right_wrist_rotate": 0.0,
    "right_right_finger": 0.027,
    "right_left_finger": 0.027,
    "middle_waist": 0.0,
    "middle_shoulder": 1.1,
    "middle_elbow": -1.1,
    "middle_forearm_roll": 0.37,
    "middle_wrist_1_joint": 0.0,
    "middle_wrist_2_joint": 0.0,
}

ACTION_SCALE = {
    ".*waist": 0.7,
    ".*shoulder": 0.7,
    ".*elbow": 0.7,
    ".*forearm_roll": 0.7,
    ".*wrist.*": 0.7,
    ".*finger": 0.03,
}
ACTION_CLIP = {
    ".*waist": (-2.6179938, 2.6179938),
    ".*shoulder": (0.0, 3.1415926),
    ".*elbow": (-2.9670597, 0.0),
    ".*forearm_roll": (-1.553343, 1.553343),
    ".*wrist_angle": (-1.553343, 1.553343),
    ".*wrist_1_joint": (-1.553343, 1.553343),
    ".*wrist_rotate": (-2.0943951, 2.0943951),
    ".*wrist_2_joint": (-2.0943951, 2.0943951),
    ".*finger": (0.0, 0.05),
}


def _xml_path(filename: str) -> Path:
    return Path(XML_DIR) / filename


def get_robot_scene_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_xml_path("piperx_scene.xml")))


def get_insert_cylinder_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(
        """
<mujoco model="insert_cylinder_entity">
  <asset>
    <material name="cylinder_mat" rgba="0.05 0.35 0.95 1"/>
  </asset>
  <worldbody>
    <body name="insert_cylinder" pos="0.045 0.15 0.01">
      <joint name="insert_cylinder_joint" type="free"/>
      <geom name="insert_cylinder_geom" type="cylinder" pos="0 0 0.061"
            size="0.02 0.06" material="cylinder_mat" friction="1.0"
            solref="0.01 1" mass="0.08"/>
    </body>
  </worldbody>
</mujoco>
"""
    )


def get_container_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(
        """
<mujoco model="cylinder_container_entity">
  <asset>
    <material name="container_mat" rgba="1.00 0.68 0.12 1"/>
    <material name="baffle_mat" rgba="0.20 0.22 0.24 1"/>
  </asset>
  <worldbody>
    <body name="cylinder_container" pos="-0.045 0.15 0">
      <geom name="container_ring_outer" type="cylinder" pos="0 0 0.012"
            size="0.032 0.0001" material="container_mat"
            contype="0" conaffinity="0"/>
      <geom name="container_ring_inner_cutout" type="cylinder" pos="0 0 0.012"
            size="0.024 0.00011" material="baffle_mat"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
    )


def get_baffle_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(
        """
<mujoco model="middle_view_baffle_entity">
  <asset>
    <material name="baffle_mat" rgba="0.20 0.22 0.24 1"/>
  </asset>
  <worldbody>
    <body name="middle_view_baffle" pos="0 0 0">
      <geom name="middle_view_cabinet_divider" type="box" pos="0 0.15 0.10"
            size="0.006 0.20 0.10" material="baffle_mat"
            friction="0.8" mass="5"/>
      <geom name="middle_view_cabinet_top" type="box" pos="0 0.15 0.20"
            size="0.08 0.2 0.006" material="baffle_mat"
            friction="0.8" mass="5"/>
      <geom name="middle_view_cabinet_bottom" type="box" pos="0 0.15 0.006"
            size="0.08 0.2 0.006" material="baffle_mat"
            friction="0.8" mass="5"/>
      <geom name="middle_view_cabinet_front" type="box" pos="0 -0.05 0.10"
            size="0.08 0.006 0.1" material="baffle_mat"
            friction="0.8" mass="5"/>
      <geom name="middle_view_cabinet_back" type="box" pos="0 0.35 0.10"
            size="0.08 0.006 0.1" material="baffle_mat"
            friction="0.8" mass="5"/>
    </body>
  </worldbody>
</mujoco>
"""
    )


def make_robot_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_robot_scene_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=ACTION_JOINT_NAMES),)
        ),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos=ALL_ROBOT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
    )


def make_insert_cylinder_env_cfg(num_envs: int = 128, play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_sites = SceneEntityCfg(
        ROBOT_ENTITY,
        site_names=("left_gripper_control", "right_gripper_control"),
        preserve_order=True,
    )
    robot_joints = SceneEntityCfg(
        ROBOT_ENTITY,
        joint_names=ACTION_JOINT_NAMES,
        preserve_order=True,
    )
    cylinder_geom = SceneEntityCfg(CYLINDER_ENTITY, geom_names=("insert_cylinder_geom",))
    container_geom = SceneEntityCfg(CONTAINER_ENTITY, geom_names=("container_ring_outer",))

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": robot_joints},
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": robot_joints},
            scale=0.1,
        ),
        "task_state": ObservationTermCfg(
            func=observations.task_state,
            params={
                "robot_cfg": robot_sites,
                "cylinder_cfg": cylinder_geom,
                "container_cfg": container_geom,
            },
        ),
        "actions": ObservationTermCfg(func=last_action),
    }
    observations_cfg = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

    actions_cfg = {
        "joint_pos": JointPositionActionCfg(
            entity_name=ROBOT_ENTITY,
            actuator_names=ACTION_JOINT_NAMES,
            scale=ACTION_SCALE,
            clip=ACTION_CLIP,
            use_default_offset=True,
        )
    }

    common_reward_params = {
        "robot_cfg": robot_sites,
        "cylinder_cfg": cylinder_geom,
        "container_cfg": container_geom,
    }
    rewards_cfg = {
        "right_reach": RewardTermCfg(
            func=rewards.right_gripper_reach,
            weight=0.5,
            params={**common_reward_params, "std": 0.12},
        ),
        "left_reach": RewardTermCfg(
            func=rewards.left_gripper_reach,
            weight=0.5,
            params={**common_reward_params, "std": 0.12},
        ),
        "place": RewardTermCfg(
            func=rewards.cylinder_to_target,
            weight=5.0,
            params={**common_reward_params, "xy_std": 0.06, "z_std": 0.05},
        ),
        "upright": RewardTermCfg(
            func=rewards.upright_bonus,
            weight=0.1,
            params=common_reward_params,
        ),
        "success": RewardTermCfg(
            func=rewards.success_bonus,
            weight=25.0,
            params=common_reward_params,
        ),
        "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
    }

    terminations_cfg = {
        "success": TerminationTermCfg(
            func=terminations.task_success,
            params=common_reward_params,
        ),
        "drop": TerminationTermCfg(
            func=terminations.cylinder_dropped,
            params={"cylinder_cfg": cylinder_geom, "min_z": -0.03},
        ),
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }

    events_cfg = {
        "reset_scene_to_default": EventTermCfg(
            func=reset_scene_to_default,
            mode="reset",
        ),
        "reset_insert_cylinder_task": EventTermCfg(
            func=events.reset_insert_cylinder_task,
            mode="reset",
            params={
                "cylinder_name": CYLINDER_ENTITY,
                "container_name": CONTAINER_ENTITY,
            },
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1 if play else int(num_envs),
            env_spacing=1.0,
            entities={
                ROBOT_ENTITY: make_robot_entity_cfg(),
                BAFFLE_ENTITY: EntityCfg(spec_fn=get_baffle_spec),
                CYLINDER_ENTITY: EntityCfg(spec_fn=get_insert_cylinder_spec),
                CONTAINER_ENTITY: EntityCfg(spec_fn=get_container_spec),
            },
            extent=1.2,
        ),
        observations=observations_cfg,
        actions=actions_cfg,
        events=events_cfg,
        rewards=rewards_cfg,
        terminations=terminations_cfg,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name=ROBOT_ENTITY,
            body_name="middle_link4",
            distance=1.5,
            elevation=-20.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=2048,
            mujoco=MujocoCfg(
                timestep=0.002,
                integrator="implicitfast",
                iterations=10,
                ls_iterations=20,
                impratio=10.0,
                cone="elliptic",
            ),
        ),
        decimation=20,
        episode_length_s=16.0,
        scale_rewards_by_dt=False,
    )
    if play:
        cfg.episode_length_s = 1e9
        cfg.observations["actor"].enable_corruption = False
    return cfg


def make_insert_cylinder_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.7,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="piperx_insert_cylinder_mjlab",
        run_name="ppo_state",
        logger="tensorboard",
        upload_model=False,
        save_interval=100,
        num_steps_per_env=32,
        max_iterations=3000,
        clip_actions=1.0,
    )
