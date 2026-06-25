"""策略推理为主、Quest 遥操接管为辅的连续轨迹采集脚本。

运行后会立即用策略控制环境并连续录制轨迹；后台持续监听 Quest 手柄。
第一次按下右手柄 A 键后暂停策略推理并保持当前关节目标，第二次按下 A 键后
以当前机械臂末端位姿作为锚点切换到遥操控制。同一个 episode 不会中断，
保存的数据中会记录每一步动作来源。
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from lerobot.common.envs.utils import preprocess_observation
from omegaconf import open_dict


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

for path in (ROOT_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/dppo_numba_cache")

import env as _registered_env  # noqa: F401  注册 Gym 环境
from quest_control import QuestControl
from quest_receive import QuestReceive
from data_collect.robot_ik_solver import PoseActionIKSolver


ACTION_SOURCE_POLICY = 0
ACTION_SOURCE_TELEOP = 1
ACTION_SOURCE_HOLD = 2


@dataclass
class PolicyTeleopConfig:
    ckpt_path: str
    output_dir: str = "outputs/4_data_collect/policy_teleop_takeover"
    run_name: str | None = None
    device: str = "cuda"
    max_steps_per_episode: int = 400
    control_hz: float = 25.0
    save_unfinished_on_quit: bool = True

    host: str = "0.0.0.0"
    port: int = 5005
    quest_timeout: float = 0.0

    head_control: bool = True
    lock_roll: bool = True
    lock_pitch: bool = True
    hand_position_scale: float = 1.0
    hand_max_delta: float = 1.0
    head_position_scale: float = 1.0
    head_max_delta: float = 1.0

    show_mujoco_viewer: bool = True
    print_hz: float = 2.0


@dataclass
class EpisodeBuffer:
    index: int
    tmp_dir: Path
    final_dir: Path
    initial_time: float | None = None
    initial_qpos: np.ndarray | None = None
    initial_qvel: np.ndarray | None = None
    initial_ctrl: np.ndarray | None = None
    initial_act: np.ndarray | None = None
    initial_mocap_pos: np.ndarray | None = None
    initial_mocap_quat: np.ndarray | None = None
    joint_actions: list[np.ndarray] = field(default_factory=list)
    pose_actions: list[np.ndarray] = field(default_factory=list)
    observation_states: list[np.ndarray] = field(default_factory=list)
    action_sources: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    cumulative_rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.joint_actions)


@dataclass
class RunState:
    reset: bool = False
    quit: bool = False


# 把相对路径解析到项目根目录。
def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


# 解析 LeRobot checkpoint 的权重目录和配置文件。
def resolve_checkpoint_dirs(ckpt_path: str | Path) -> tuple[Path, Path, Path]:
    ckpt_root = resolve_path(ckpt_path)
    if not ckpt_root.exists():
        raise FileNotFoundError(f"找不到权重路径: {ckpt_root}")

    if (ckpt_root / "pretrained_model").is_dir():
        load_dir = ckpt_root / "pretrained_model"
        root_dir = ckpt_root
    elif ckpt_root.name == "pretrained_model":
        load_dir = ckpt_root
        root_dir = ckpt_root.parent
    else:
        load_dir = ckpt_root
        root_dir = ckpt_root

    config_path = load_dir / "config.yaml"
    if not config_path.exists():
        config_path = root_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint 配置文件: {load_dir} 或 {root_dir}")
    return root_dir, load_dir, config_path


# 加载策略，并强制使用当前脚本配置的推理设备。
def load_policy(cfg: PolicyTeleopConfig):
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.utils.utils import init_hydra_config

    root_dir, load_dir, config_path = resolve_checkpoint_dirs(cfg.ckpt_path)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    hydra_cfg = init_hydra_config(str(config_path))
    with open_dict(hydra_cfg):
        hydra_cfg.device = str(device)

    print(f"加载策略权重: {load_dir}")
    print(f"推理设备: {device}")
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(load_dir))
    policy.to(device)
    policy.eval()

    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}
    return policy, full_cfg, device, root_dir, load_dir


# 根据策略输入相机和 checkpoint 环境配置创建 Gym 环境。
def make_env(policy, full_cfg: dict, cfg: PolicyTeleopConfig):
    input_keys = policy.config.input_shapes.keys()
    cameras = [
        key.removeprefix("observation.images.")
        for key in input_keys
        if key.startswith("observation.images.")
    ]
    cameras = list(dict.fromkeys(cameras))

    env_cfg = full_cfg.get("env", {})
    env_id = f"{env_cfg.get('name', 'guided_vision')}/{env_cfg.get('task', 'SewNeedle-3Arms-v0')}"
    print(f"初始化环境: {env_id}")
    print(f"策略观测相机: {cameras}")

    env = gym.make(
        id=env_id,
        disable_env_checker=True,
        cameras=cameras,
        episode_length=cfg.max_steps_per_episode,
    )
    return env, cameras, env_id


# 只把策略需要的观测键转成 torch tensor。
def prepare_obs_for_policy(obs: dict, policy, device) -> dict[str, torch.Tensor]:
    def add_batch_dim(obj):
        if isinstance(obj, dict):
            return {key: add_batch_dim(value) for key, value in obj.items()}
        if hasattr(obj, "copy"):
            return np.expand_dims(obj.copy(), axis=0).copy()
        return obj

    batch = preprocess_observation(add_batch_dim(obs))
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in policy.config.input_shapes
    }


# 用策略输出下一步关节动作。
def policy_action(policy, obs: dict, device) -> np.ndarray:
    batch = prepare_obs_for_policy(obs, policy, device)
    with torch.no_grad():
        action_tensor = policy.select_action(batch)
    return action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)


# 尝试读取最新 Quest 数据；没有新包时返回上一帧。
def poll_latest_quest(receiver: QuestReceive, latest_data):
    try:
        return receiver.receive_latest_data()
    except socket.timeout:
        return latest_data
    except BlockingIOError:
        return latest_data
    except json.JSONDecodeError as exc:
        print(f"Invalid Quest packet: {exc}")
        return latest_data


# 启动 MuJoCo viewer，并监听重置和退出按键。
def launch_viewer(sim_env, state: RunState, enabled: bool):
    if not enabled:
        return None

    import mujoco.viewer

    def key_callback(keycode):
        if keycode == 32:
            state.reset = True
        elif keycode in (ord("q"), ord("Q"), 27):
            state.quit = True

    return mujoco.viewer.launch_passive(
        sim_env._physics.model.ptr,
        sim_env._physics.data.ptr,
        show_left_ui=True,
        show_right_ui=True,
        key_callback=key_callback,
    )


# 创建新的 episode 临时目录。
def start_episode(run_dir: Path, episode_index: int, physics) -> EpisodeBuffer:
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = episodes_dir / f"episode_{episode_index:06d}.tmp"
    final_dir = episodes_dir / f"episode_{episode_index:06d}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    buffer = EpisodeBuffer(index=episode_index, tmp_dir=tmp_dir, final_dir=final_dir)
    buffer.initial_time = float(physics.data.time)
    buffer.initial_qpos = physics.data.qpos.copy()
    buffer.initial_qvel = physics.data.qvel.copy()
    buffer.initial_ctrl = physics.data.ctrl.copy()
    buffer.initial_act = physics.data.act.copy()
    buffer.initial_mocap_pos = physics.data.mocap_pos.copy()
    buffer.initial_mocap_quat = physics.data.mocap_quat.copy()
    return buffer


# 记录单步 transition。
def record_step(
    buffer: EpisodeBuffer,
    *,
    obs_before: dict,
    joint_action: np.ndarray,
    pose_action: np.ndarray,
    source: int,
    reward: float,
    cumulative_reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
) -> None:
    buffer.observation_states.append(np.asarray(obs_before["agent_pos"], dtype=np.float32).copy())
    buffer.joint_actions.append(np.asarray(joint_action, dtype=np.float32).copy())
    buffer.pose_actions.append(np.asarray(pose_action, dtype=np.float32).copy())
    buffer.action_sources.append(int(source))
    buffer.rewards.append(float(reward))
    buffer.cumulative_rewards.append(float(cumulative_reward))
    buffer.terminated.append(bool(terminated))
    buffer.truncated.append(bool(truncated))
    buffer.infos.append(
        {
            "is_success": bool(info.get("is_success", False)),
            "reward": float(reward),
            "source": int(source),
        }
    )


# 保存 episode 数组和说明文件。
def save_episode(buffer: EpisodeBuffer, metadata: dict, *, reason: str) -> dict | None:
    if buffer.steps <= 0:
        shutil.rmtree(buffer.tmp_dir, ignore_errors=True)
        return None

    arrays = {
        "joint_action": np.asarray(buffer.joint_actions, dtype=np.float32),
        "pose_action": np.asarray(buffer.pose_actions, dtype=np.float32),
        "action_source": np.asarray(buffer.action_sources, dtype=np.int8),
        "observation_state": np.asarray(buffer.observation_states, dtype=np.float32),
        "reward": np.asarray(buffer.rewards, dtype=np.float32),
        "cumulative_reward": np.asarray(buffer.cumulative_rewards, dtype=np.float32),
        "terminated": np.asarray(buffer.terminated, dtype=np.bool_),
        "truncated": np.asarray(buffer.truncated, dtype=np.bool_),
    }
    optional_arrays = {
        "initial_qpos": buffer.initial_qpos,
        "initial_qvel": buffer.initial_qvel,
        "initial_ctrl": buffer.initial_ctrl,
        "initial_act": buffer.initial_act,
        "initial_mocap_pos": buffer.initial_mocap_pos,
        "initial_mocap_quat": buffer.initial_mocap_quat,
    }
    for key, value in optional_arrays.items():
        if value is not None:
            arrays[key] = np.asarray(value)
    if buffer.initial_time is not None:
        arrays["initial_time"] = np.asarray([buffer.initial_time], dtype=np.float64)

    np.savez_compressed(buffer.tmp_dir / "arrays.npz", **arrays)
    info = {
        "episode": int(buffer.index),
        "steps": int(buffer.steps),
        "reason": reason,
        "success": bool(buffer.infos[-1].get("is_success", False)) if buffer.infos else False,
        "action_source_map": {
            "0": "policy",
            "1": "teleop",
            "2": "hold",
        },
        "policy_steps": int(np.sum(arrays["action_source"] == ACTION_SOURCE_POLICY)),
        "teleop_steps": int(np.sum(arrays["action_source"] == ACTION_SOURCE_TELEOP)),
        "hold_steps": int(np.sum(arrays["action_source"] == ACTION_SOURCE_HOLD)),
        "final_cumulative_reward": float(buffer.cumulative_rewards[-1]) if buffer.cumulative_rewards else 0.0,
    }
    with open(buffer.tmp_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    if buffer.final_dir.exists():
        shutil.rmtree(buffer.final_dir)
    buffer.tmp_dir.rename(buffer.final_dir)

    info["path"] = str(buffer.final_dir.relative_to(Path(metadata["run_dir"])))
    metadata["episodes"].append(info)
    metadata["saved_episodes"] = len(metadata["episodes"])
    with open(Path(metadata["run_dir"]) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(
        f"Saved episode {buffer.index:06d}: steps={buffer.steps}, "
        f"policy={info['policy_steps']}, teleop={info['teleop_steps']}, reason={reason}"
    )
    return info


# 根据已有 episode 计算下一个编号。
def next_episode_index(run_dir: Path) -> int:
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.exists():
        return 0
    indices = []
    for path in episodes_dir.glob("episode_*"):
        if path.is_dir() and path.name.startswith("episode_") and not path.name.endswith(".tmp"):
            try:
                indices.append(int(path.name.split("_")[-1]))
            except ValueError:
                pass
    return max(indices, default=-1) + 1


# 构建 run 输出目录。
def make_run_dir(cfg: PolicyTeleopConfig, env_id: str) -> Path:
    if cfg.run_name:
        run_name = cfg.run_name
    else:
        task_name = env_id.split("/")[-1]
        run_name = f"policy_teleop_{task_name}"
    run_dir = resolve_path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    return run_dir


# 重置环境、策略和遥操状态。
def reset_control_state(env, policy, quest_control: QuestControl, ik_solver: PoseActionIKSolver):
    obs, _ = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    quest_control.reset()
    ik_solver.reset(active=False)
    return obs


# 主循环：默认策略推理，第一次 A 暂停，第二次 A 切换遥操并连续录制。
def run(cfg: PolicyTeleopConfig) -> None:
    policy, full_cfg, device, _root_dir, load_dir = load_policy(cfg)
    env, _policy_cameras, env_id = make_env(policy, full_cfg, cfg)
    sim_env = env.unwrapped
    physics = sim_env._physics

    run_dir = make_run_dir(cfg, env_id)
    episode_index = next_episode_index(run_dir)
    metadata = {
        "run_dir": str(run_dir),
        "ckpt_path": str(resolve_path(cfg.ckpt_path)),
        "load_dir": str(load_dir),
        "env_id": env_id,
        "control_hz": float(cfg.control_hz),
        "action_source_map": {"0": "policy", "1": "teleop", "2": "hold"},
        "saved_episodes": episode_index,
        "episodes": [],
    }
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                old_metadata = json.load(f)
            metadata["episodes"] = old_metadata.get("episodes", [])
            metadata["saved_episodes"] = len(metadata["episodes"])
        except Exception:
            pass

    receiver = QuestReceive(host=cfg.host, port=cfg.port, timeout=cfg.quest_timeout)
    quest_control = QuestControl(use_head_control=cfg.head_control, use_individual_hand_anchors=True)
    ik_solver = PoseActionIKSolver(
        sim_env,
        head_control=cfg.head_control,
        lock_roll=cfg.lock_roll,
        lock_pitch=cfg.lock_pitch,
        hand_position_scale=cfg.hand_position_scale,
        hand_max_delta=cfg.hand_max_delta,
        head_position_scale=cfg.head_position_scale,
        head_max_delta=cfg.head_max_delta,
    )

    obs = reset_control_state(env, policy, quest_control, ik_solver)
    buffer = start_episode(run_dir, episode_index, physics)
    cumulative_reward = 0.0
    control_mode = "policy"
    latest_data = None
    prev_a_button = False
    prev_reset_button = False
    last_print = 0.0

    state = RunState()
    viewer = launch_viewer(sim_env, state, cfg.show_mujoco_viewer)
    dt = 1.0 / float(cfg.control_hz)

    print("\n" + "=" * 70)
    print("Policy + Quest Teleop takeover")
    print("默认执行策略推理并连续录制。右手 A 键第一次: 暂停推理；第二次: 人工接管。")
    print("B/Y 或 Space: 保存当前段并重置。Q/Esc: 退出。")
    print(f"输出目录: {run_dir}")
    print("=" * 70 + "\n")

    try:
        while not state.quit:
            loop_start = time.time()
            latest_data = poll_latest_quest(receiver, latest_data)

            a_button = bool(latest_data.r_button_one) if latest_data is not None else False
            reset_button = (
                bool(latest_data.r_button_two or latest_data.l_button_two)
                if latest_data is not None
                else False
            )

            if a_button and not prev_a_button:
                if control_mode == "policy":
                    control_mode = "paused"
                    print(f"Policy paused at step {buffer.steps}. Press A again to start Quest takeover.")
                elif control_mode == "paused" and latest_data is not None:
                    if not ik_solver.can_anchor_from_data(latest_data, allow_partial=False):
                        print("Quest A pressed, but headset/controller poses are not ready for takeover.")
                    else:
                        left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
                        quest_control.start(latest_data, middle_pose, left_pose, right_pose)
                        active_count = ik_solver.activate_from_data(latest_data, require_all=True)
                        if active_count > 0:
                            control_mode = "teleop"
                            print(f"Quest takeover started at step {buffer.steps}.")
                        else:
                            print("Quest A pressed, but no IK arm was activated.")
                elif control_mode == "paused":
                    print("Quest A pressed, but no Quest packet has arrived yet.")

            if state.reset or (reset_button and not prev_reset_button):
                save_episode(buffer, metadata, reason="manual_reset")
                episode_index += 1
                obs = reset_control_state(env, policy, quest_control, ik_solver)
                buffer = start_episode(run_dir, episode_index, physics)
                cumulative_reward = 0.0
                control_mode = "policy"
                state.reset = False
                print(f"Reset env. Started episode {episode_index:06d}.")
                prev_a_button = a_button
                prev_reset_button = reset_button
                continue

            obs_before = obs
            pose_action = np.full(23, np.nan, dtype=np.float32)

            if control_mode == "teleop":
                if latest_data is None:
                    joint_action = np.asarray(obs_before["agent_pos"], dtype=np.float32).copy()
                    source = ACTION_SOURCE_HOLD
                else:
                    left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
                    pose_action, _feedback = quest_control.run(latest_data, left_pose, right_pose, middle_pose)
                    joint_action, _active_count = ik_solver.pose2joint(pose_action, obs_before)
                    joint_action = np.asarray(joint_action, dtype=np.float32)
                    pose_action = np.asarray(pose_action, dtype=np.float32)
                    source = ACTION_SOURCE_TELEOP
            elif control_mode == "paused":
                joint_action = np.asarray(obs_before["agent_pos"], dtype=np.float32).copy()
                source = ACTION_SOURCE_HOLD
            else:
                joint_action = policy_action(policy, obs_before, device)
                source = ACTION_SOURCE_POLICY

            obs, reward, terminated, truncated, info = env.step(joint_action)
            cumulative_reward += float(reward)
            record_step(
                buffer,
                obs_before=obs_before,
                joint_action=joint_action,
                pose_action=pose_action,
                source=source,
                reward=float(reward),
                cumulative_reward=cumulative_reward,
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
            )

            if terminated or truncated or buffer.steps >= cfg.max_steps_per_episode:
                reason = "terminated" if terminated else "truncated" if truncated else "max_steps"
                save_episode(buffer, metadata, reason=reason)
                episode_index += 1
                obs = reset_control_state(env, policy, quest_control, ik_solver)
                buffer = start_episode(run_dir, episode_index, physics)
                cumulative_reward = 0.0
                control_mode = "policy"
                print(f"Started episode {episode_index:06d}.")

            now = time.time()
            if cfg.print_hz > 0 and now - last_print >= 1.0 / cfg.print_hz:
                print(
                    f"[{control_mode}] episode={buffer.index:06d} steps={buffer.steps} "
                    f"saved={metadata['saved_episodes']} reward={cumulative_reward:.2f}"
                )
                last_print = now

            if viewer is not None:
                viewer.sync()

            prev_a_button = a_button
            prev_reset_button = reset_button

            sleep_time = dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        if cfg.save_unfinished_on_quit and buffer.steps > 0:
            save_episode(buffer, metadata, reason="quit")
        elif buffer.tmp_dir.exists():
            shutil.rmtree(buffer.tmp_dir, ignore_errors=True)

        receiver.close()
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    CONFIG = PolicyTeleopConfig(
        ckpt_path=(
            "outputs/2_pretrain/train/2026-06-22/"
            "20-18-10_SewNeedle-3Arms-v0_pre_zed_diffusion/"
            "checkpoints/086000_loss=0.0063_sr=0.0_ar=-260.74"
        ),
        max_steps_per_episode=400,
        control_hz=25.0,
        device="cuda",
        show_mujoco_viewer=True,
        head_control=True,
        lock_roll=True,
        lock_pitch=True,
    )
    run(CONFIG)
