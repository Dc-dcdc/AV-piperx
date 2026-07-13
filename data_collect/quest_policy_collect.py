"""策略推理与 Quest 分区遥操接管的连续轨迹采集脚本。

支持按部位独立接管：
1. ``policy``：头部/中间臂和左右臂全部由策略自主推理控制。
2. ``right_teleop``：右臂由 Quest 右手柄遥操，其他机械臂仍由策略控制。
3. ``left_teleop``：左臂由 Quest 左手柄遥操，其他机械臂仍由策略控制。
4. ``head_teleop``：头部/中间臂由 Quest 头显遥操，左右臂仍由策略控制。

A/X/B 可以独立切换右臂、左臂和头部接管，Y 会丢弃当前轨迹并重置环境。
每一步都会先计算完整的 policy action，再用 QuestControl + IK 计算被接管机械臂的候选动作，
最终通过分臂 blend 权重平滑混合，避免接管和恢复策略时出现大的动作跳变。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import gymnasium as gym
import hydra
import numpy as np
import torch
import yaml
from lerobot.common.envs.utils import preprocess_observation
from omegaconf import DictConfig, open_dict


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

for path in (ROOT_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/av_piper_numba_cache")

import env as _registered_env  # noqa: F401  注册 Gym 环境
from quest_control import QuestControl
from quest_receive import QuestReceive
from quest_send import UnityImageStreamer
from data_collect.robot_ik_solver import PoseActionIKSolver
from data_collect.quest_teleop_collect import (
    AsyncEpisodeVideoWriter,
    flatten_numeric_obs,
    json_safe,
    list_existing_episode_dirs,
    load_existing_episode_infos,
    replay_episode_videos,
    save_episode_reward_debug,
    stack_trace,
    write_metadata,
)


ARM_LEFT = "left"
ARM_RIGHT = "right"
ARM_MIDDLE = "middle"
ARM_ORDER = (ARM_LEFT, ARM_RIGHT, ARM_MIDDLE)

MODE_POLICY = "policy"
MODE_RIGHT_TELEOP = "right_teleop"
MODE_LEFT_TELEOP = "left_teleop"
MODE_HEAD_TELEOP = "head_teleop"
MODE_HANDS_TELEOP = "hands_teleop"
MODE_RIGHT_HEAD_TELEOP = "right_head_teleop"
MODE_LEFT_HEAD_TELEOP = "left_head_teleop"
MODE_ALL_TELEOP = "all_teleop"

MODE_ACTIVE_ARMS = {
    MODE_POLICY: (),
    MODE_RIGHT_TELEOP: (ARM_RIGHT,),
    MODE_LEFT_TELEOP: (ARM_LEFT,),
    MODE_HEAD_TELEOP: (ARM_MIDDLE,),
    MODE_HANDS_TELEOP: (ARM_LEFT, ARM_RIGHT),
    MODE_RIGHT_HEAD_TELEOP: (ARM_RIGHT, ARM_MIDDLE),
    MODE_LEFT_HEAD_TELEOP: (ARM_LEFT, ARM_MIDDLE),
    MODE_ALL_TELEOP: (ARM_LEFT, ARM_RIGHT, ARM_MIDDLE),
}
MODE_CODE = {mode: code for code, mode in enumerate(MODE_ACTIVE_ARMS)}
CODE_MODE = {code: mode for mode, code in MODE_CODE.items()}
ACTIVE_ARMS_TO_MODE = {tuple(sorted(active_arms)): mode for mode, active_arms in MODE_ACTIVE_ARMS.items()}
MODE_LABEL = {
    MODE_POLICY: "策略自主推理",
    MODE_RIGHT_TELEOP: "右臂遥操 + 其他策略",
    MODE_LEFT_TELEOP: "左臂遥操 + 其他策略",
    MODE_HEAD_TELEOP: "头部遥操 + 双臂策略",
    MODE_HANDS_TELEOP: "双臂遥操 + 头部策略",
    MODE_RIGHT_HEAD_TELEOP: "右臂和头部遥操 + 左臂策略",
    MODE_LEFT_HEAD_TELEOP: "左臂和头部遥操 + 右臂策略",
    MODE_ALL_TELEOP: "左臂、右臂和头部全部遥操",
}


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
    obs_traces: dict[str, list[np.ndarray]] = field(default_factory=dict)
    depth_traces: dict[str, list[np.ndarray]] = field(default_factory=dict)
    joint_actions: list[np.ndarray] = field(default_factory=list)
    policy_actions: list[np.ndarray] = field(default_factory=list)
    pose_actions: list[np.ndarray] = field(default_factory=list)
    control_modes: list[int] = field(default_factory=list)
    teleop_applied: list[bool] = field(default_factory=list)
    blend_weights: list[np.ndarray] = field(default_factory=list)
    teleop_masks: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    cumulative_rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)
    reward_debug: list[dict] = field(default_factory=list)
    video_writer: AsyncEpisodeVideoWriter | None = None
    video_paths: dict[str, Path] = field(default_factory=dict)
    final_info: dict = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return len(self.joint_actions)


@dataclass
class RunState:
    reset: bool = False
    quit: bool = False
    requested_mode: str | None = None


class TakeoverBlender:
    """按机械臂平滑混合策略动作和遥操动作。"""

    def __init__(self, *, blend_steps: int, arm_names: tuple[str, ...] = (ARM_LEFT, ARM_RIGHT, ARM_MIDDLE)) -> None:
        self.arm_names = tuple(arm_names)
        self.blend_steps = max(1, int(blend_steps))
        self.value = {name: 0.0 for name in self.arm_names}
        self.start = {name: 0.0 for name in self.arm_names}
        self.target = {name: 0.0 for name in self.arm_names}
        self.progress = {name: self.blend_steps for name in self.arm_names}
        self.release_action: np.ndarray | None = None

    # 重置所有接管权重。
    def reset(self) -> None:
        for name in self.arm_names:
            self.value[name] = 0.0
            self.start[name] = 0.0
            self.target[name] = 0.0
            self.progress[name] = self.blend_steps
        self.release_action = None

    # 设置新的接管目标，退出接管时保存当前动作作为平滑回退起点。
    def set_mode(self, mode: str, *, current_action: np.ndarray | None = None) -> None:
        active_arms = set(MODE_ACTIVE_ARMS.get(mode, ()))
        releasing = False
        for name in self.arm_names:
            new_target = 1.0 if name in active_arms else 0.0
            if np.isclose(new_target, self.target[name]):
                continue
            if new_target == 0.0 and self.value[name] > 1.0e-6:
                releasing = True
            self.start[name] = self.value[name]
            self.target[name] = new_target
            self.progress[name] = 0
        if releasing and current_action is not None:
            self.release_action = np.asarray(current_action, dtype=np.float32).copy()

    # 推进一步平滑权重。
    def step(self) -> dict[str, float]:
        for name in self.arm_names:
            if self.progress[name] >= self.blend_steps:
                self.value[name] = self.target[name]
                continue
            self.progress[name] += 1
            alpha = self.progress[name] / self.blend_steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            self.value[name] = self.start[name] + (self.target[name] - self.start[name]) * smooth

        if all(self.value[name] <= 1.0e-6 and self.target[name] <= 1.0e-6 for name in self.arm_names):
            self.release_action = None
        return self.weights()

    # 当前每条机械臂的平滑权重。
    def weights(self) -> dict[str, float]:
        return {name: float(self.value[name]) for name in self.arm_names}

    # 当前目标接管掩码。
    def target_mask(self) -> dict[str, bool]:
        return {name: bool(self.target[name] > 0.5) for name in self.arm_names}

    # 以固定顺序导出权重数组。
    def weights_array(self) -> np.ndarray:
        return np.asarray([self.value[name] for name in self.arm_names], dtype=np.float32)

    # 以固定顺序导出接管掩码数组。
    def target_mask_array(self) -> np.ndarray:
        return np.asarray([self.target[name] > 0.5 for name in self.arm_names], dtype=np.bool_)

    # 是否仍处在策略/遥操混合过渡中。
    def is_blending(self) -> bool:
        return any(self.value[name] > 1.0e-6 for name in self.arm_names)


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


# 加载策略，并把 checkpoint 中的 device 覆盖为当前脚本配置。
def load_policy(cfg: DictConfig):
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
def make_env(policy, full_cfg: dict, cfg: DictConfig):
    input_keys = policy.config.input_shapes.keys()
    cameras = [
        key.removeprefix("observation.images.")
        for key in input_keys
        if key.startswith("observation.images.")
    ]
    cameras = list(dict.fromkeys(cameras))

    env_cfg = full_cfg.get("env", {})
    checkpoint_env_id = f"{env_cfg.get('name', 'guided_vision')}/{env_cfg.get('task', 'InsertCylinder-3Arms-v0')}"
    env_id = str(cfg.env_id) if cfg.get("env_id") else checkpoint_env_id
    print(f"初始化环境: {env_id}")
    print(f"策略观测相机: {cameras}")

    env_kwargs = {
        "disable_env_checker": True,
        "cameras": cameras,
        "episode_length": int(cfg.episode_length),
        "observation_height": int(cfg.render_height),
        "observation_width": int(cfg.render_width),
    }
    if env_id == "guided_vision/InsertCylinder-3Arms-v0":
        env_kwargs["enable_reward_debug"] = bool(cfg.get("save_reward_debug", False))
    env = gym.make(id=env_id, **env_kwargs)
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


# 用策略输出完整关节动作。
def policy_action(policy, obs: dict, device) -> np.ndarray:
    batch = prepare_obs_for_policy(obs, policy, device)
    with torch.no_grad():
        action_tensor = policy.select_action(batch)
    return action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)


# 将 float 深度图转换成便于查看的伪彩色 BGR 图像。
def depth_to_colormap_bgr(depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    if depth_max <= depth_min:
        depth_max = depth_min + 1e-6
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=depth_max, posinf=depth_max, neginf=depth_min)
    depth = np.clip(depth, depth_min, depth_max)
    depth_u8 = ((depth - depth_min) / (depth_max - depth_min + 1e-8) * 255.0).astype(np.uint8)
    depth_u8 = 255 - depth_u8
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


# 在 BGR 图像上叠加多行状态文字。
def draw_text_lines_bgr(frame_bgr: np.ndarray, lines: list[str], x: int = 14, y: int = 26) -> np.ndarray:
    output = frame_bgr.copy()
    text_y = y
    for line in lines:
        cv2.putText(output, line, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(output, line, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 245, 255), 1, cv2.LINE_AA)
        text_y += 24
    return output


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


# 根据接管机械臂集合生成控制模式。
def mode_from_active_arms(active_arms) -> str:
    return ACTIVE_ARMS_TO_MODE[tuple(sorted(set(active_arms)))]


# 根据 Quest 按键边沿切换接管部位；A+X 仍用于完成后的保存确认。
def quest_button_request(
    data,
    prev_buttons: dict[str, bool],
    current_mode: str,
) -> tuple[str | None, bool, bool]:
    if data is None:
        return None, False, False

    buttons = {
        "a": bool(data.r_button_one),       # 右手 A: 右臂接管；完成后与 X 同按保存
        "x": bool(data.l_button_one),       # 左手 X: 左臂接管；完成后与 A 同按保存
        "b": bool(data.r_button_two),       # 右手 B: 切换头部/中间臂接管
        "y": bool(data.l_button_two),       # 左手 Y: 丢弃当前 episode 并重置环境
    }
    edges = {name: pressed and not prev_buttons.get(name, False) for name, pressed in buttons.items()}

    both_ax = buttons["a"] and buttons["x"]
    prev_both_ax = prev_buttons.get("a", False) and prev_buttons.get("x", False)
    save_confirm = bool(both_ax and not prev_both_ax)

    reset_requested = edges["y"]

    requested_mode = None
    if reset_requested:
        pass
    else:
        active_arms = set(MODE_ACTIVE_ARMS.get(current_mode, ()))
        if edges["a"]:
            if ARM_RIGHT in active_arms:
                active_arms.remove(ARM_RIGHT)
            else:
                active_arms.add(ARM_RIGHT)
        if edges["x"]:
            if ARM_LEFT in active_arms:
                active_arms.remove(ARM_LEFT)
            else:
                active_arms.add(ARM_LEFT)
        if edges["b"]:
            if ARM_MIDDLE in active_arms:
                active_arms.remove(ARM_MIDDLE)
            else:
                active_arms.add(ARM_MIDDLE)
        if edges["a"] or edges["x"] or edges["b"]:
            requested_mode = mode_from_active_arms(active_arms)

    prev_buttons.clear()
    prev_buttons.update(buttons)
    return requested_mode, reset_requested, save_confirm


# 启动 MuJoCo viewer，并监听键盘模式切换。
def launch_viewer(sim_env, state: RunState, enabled: bool):
    if not enabled:
        return None

    import mujoco.viewer

    def key_callback(keycode):
        if keycode == ord("1"):
            state.requested_mode = MODE_POLICY
        elif keycode == ord("2"):
            state.requested_mode = MODE_RIGHT_TELEOP
        elif keycode == ord("3"):
            state.requested_mode = MODE_LEFT_TELEOP
        elif keycode == ord("4"):
            state.requested_mode = MODE_HEAD_TELEOP
        elif keycode == ord("5"):
            state.requested_mode = MODE_HANDS_TELEOP
        elif keycode == ord("6"):
            state.requested_mode = MODE_ALL_TELEOP
        elif keycode in (ord("y"), ord("Y")):
            state.reset = True
        elif keycode == 32:
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
    policy_joint_action: np.ndarray,
    pose_action: np.ndarray,
    control_mode: str,
    teleop_applied: bool,
    blend_weights: np.ndarray,
    teleop_mask: np.ndarray,
    reward: float,
    cumulative_reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    record_cameras: list[str],
    depth_frames: dict[str, np.ndarray] | None,
    save_reward_debug: bool,
) -> None:
    step_index = buffer.steps
    for raw_key, value in flatten_numeric_obs(obs_before).items():
        buffer.obs_traces.setdefault(raw_key, []).append(np.asarray(value).copy())
    if depth_frames:
        for camera in record_cameras:
            if camera in depth_frames:
                buffer.depth_traces.setdefault(camera, []).append(
                    np.asarray(depth_frames[camera], dtype=np.float32).copy()
                )

    buffer.joint_actions.append(np.asarray(joint_action, dtype=np.float32).copy())
    buffer.policy_actions.append(np.asarray(policy_joint_action, dtype=np.float32).copy())
    buffer.pose_actions.append(np.asarray(pose_action, dtype=np.float32).copy())
    buffer.control_modes.append(int(MODE_CODE[control_mode]))
    buffer.teleop_applied.append(bool(teleop_applied))
    buffer.blend_weights.append(np.asarray(blend_weights, dtype=np.float32).copy())
    buffer.teleop_masks.append(np.asarray(teleop_mask, dtype=np.bool_).copy())
    buffer.rewards.append(float(reward))
    buffer.cumulative_rewards.append(float(cumulative_reward))
    buffer.terminated.append(bool(terminated))
    buffer.truncated.append(bool(truncated))
    buffer.infos.append(
        {
            "is_success": bool(info.get("is_success", False)),
            "reward": float(reward),
            "mode": control_mode,
            "teleop_applied": bool(teleop_applied),
            "blend_weight": np.asarray(blend_weights, dtype=np.float32).round(4).tolist(),
            "teleop_mask": np.asarray(teleop_mask, dtype=np.bool_).tolist(),
        }
    )
    buffer.final_info = dict(info or {})
    if save_reward_debug:
        reward_item = {
            "step": int(step_index),
            "reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "is_success": bool(info.get("is_success", False)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        env_reward_debug = info.get("reward_debug")
        if isinstance(env_reward_debug, dict):
            reward_item.update(json_safe(env_reward_debug))
        buffer.reward_debug.append(reward_item)


# 将标准轨迹字段和策略接管扩展字段写入 arrays.npz。
def save_episode_arrays(buffer: EpisodeBuffer, cfg: DictConfig) -> dict[str, dict[str, str]]:
    mode_arr = np.asarray(buffer.control_modes, dtype=np.int8)
    arrays = {
        "joint_action": np.asarray(buffer.joint_actions, dtype=np.float32),
        "policy_action": np.asarray(buffer.policy_actions, dtype=np.float32),
        "control_mode": mode_arr,
        "teleop_applied": np.asarray(buffer.teleop_applied, dtype=np.bool_),
        "blend_weight": np.asarray(buffer.blend_weights, dtype=np.float32),
        "teleop_mask": np.asarray(buffer.teleop_masks, dtype=np.bool_),
        "reward": np.asarray(buffer.rewards, dtype=np.float32),
        "cumulative_reward": np.asarray(buffer.cumulative_rewards, dtype=np.float32),
        "terminated": np.asarray(buffer.terminated, dtype=np.bool_),
        "truncated": np.asarray(buffer.truncated, dtype=np.bool_),
    }
    if cfg.save_pose_action:
        arrays["pose_action"] = np.asarray(buffer.pose_actions, dtype=np.float32)

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
        arrays["initial_time"] = np.asarray(buffer.initial_time, dtype=np.float64)

    depth_key_map = {}
    for camera, values in sorted(buffer.depth_traces.items()):
        if values:
            npz_key = f"depth__{camera}"
            depth_key_map[camera] = npz_key
            arrays[npz_key] = stack_trace(values).astype(np.float32, copy=False)

    obs_key_map = {}
    for raw_key, values in sorted(buffer.obs_traces.items()):
        if not values:
            continue
        if raw_key in ("agent_pos", "observation.state"):
            npz_key = "observation_state"
        else:
            npz_key = f"obs__{raw_key.replace('.', '__').replace('/', '__')}"
        obs_key_map[raw_key] = npz_key
        arrays[npz_key] = stack_trace(values)

    np.savez_compressed(buffer.tmp_dir / "arrays.npz", **arrays)
    return {"observation": obs_key_map, "depth": depth_key_map}


# 从当前 episode 中生成轻量的人类干预片段索引。
def build_intervention_index(buffer: EpisodeBuffer, cfg: DictConfig) -> dict:
    steps = int(buffer.steps)
    fps = int(cfg.fps)
    arm_order = list(ARM_ORDER)

    teleop_applied = np.asarray(buffer.teleop_applied, dtype=np.bool_)
    if teleop_applied.shape[0] != steps:
        teleop_applied = np.zeros(steps, dtype=np.bool_)

    teleop_mask = np.asarray(buffer.teleop_masks, dtype=np.bool_)
    if teleop_mask.ndim != 2 or teleop_mask.shape[0] != steps:
        teleop_mask = np.zeros((steps, len(arm_order)), dtype=np.bool_)

    blend_weight = np.asarray(buffer.blend_weights, dtype=np.float32)
    if blend_weight.ndim != 2 or blend_weight.shape[0] != steps:
        blend_weight = np.zeros((steps, len(arm_order)), dtype=np.float32)

    control_modes = np.asarray(buffer.control_modes, dtype=np.int64)
    if control_modes.shape[0] != steps:
        control_modes = np.zeros(steps, dtype=np.int64)

    segments = []
    start = None
    for idx, active in enumerate(teleop_applied.tolist()):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, steps))

    segment_items = []
    for segment_id, (start_frame, stop_frame) in enumerate(segments):
        segment_mask = teleop_mask[start_frame:stop_frame]
        segment_blend = blend_weight[start_frame:stop_frame]
        if segment_mask.size:
            target_arm_flags = np.any(segment_mask, axis=0)
        else:
            target_arm_flags = np.zeros(len(arm_order), dtype=np.bool_)
        if segment_blend.size:
            blend_arm_flags = np.max(segment_blend, axis=0) > 1.0e-6
            max_blend_weight = np.max(segment_blend, axis=0)
            mean_blend_weight = np.mean(segment_blend, axis=0)
        else:
            blend_arm_flags = np.zeros(len(arm_order), dtype=np.bool_)
            max_blend_weight = np.zeros(len(arm_order), dtype=np.float32)
            mean_blend_weight = np.zeros(len(arm_order), dtype=np.float32)

        mode_slice = control_modes[start_frame:stop_frame]
        mode_steps = {}
        if mode_slice.size:
            unique_modes, counts = np.unique(mode_slice, return_counts=True)
            mode_steps = {
                CODE_MODE.get(int(mode_code), str(int(mode_code))): int(count)
                for mode_code, count in zip(unique_modes, counts)
            }

        target_arms = [
            arm for arm, active in zip(arm_order, target_arm_flags.tolist()) if bool(active)
        ]
        influenced_arms = [
            arm
            for arm, active in zip(arm_order, np.logical_or(target_arm_flags, blend_arm_flags).tolist())
            if bool(active)
        ]

        segment_items.append(
            {
                "id": int(segment_id),
                "start_frame": int(start_frame),
                "stop_frame": int(stop_frame),
                "num_frames": int(stop_frame - start_frame),
                "start_timestamp": float(start_frame / fps),
                "stop_timestamp_exclusive": float(stop_frame / fps),
                "target_arms": target_arms,
                "influenced_arms": influenced_arms,
                "mode_steps": mode_steps,
                "max_blend_weight": max_blend_weight.astype(np.float32).round(6).tolist(),
                "mean_blend_weight": mean_blend_weight.astype(np.float32).round(6).tolist(),
            }
        )

    return {
        "schema": "av_piper_intervention_segments_v1",
        "episode": int(buffer.index),
        "fps": fps,
        "steps": steps,
        "mask_source": "teleop_applied",
        "arm_order": arm_order,
        "segment_count": int(len(segment_items)),
        "intervention_frames": int(np.sum(teleop_applied)),
        "segments": segment_items,
    }


# 将人类干预片段索引写入单独 JSON；完整训练数据仍以 arrays.npz 为准。
def save_intervention_index(buffer: EpisodeBuffer, cfg: DictConfig) -> dict:
    index = build_intervention_index(buffer, cfg)
    path = buffer.tmp_dir / "intervention_segments.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return index


# 按标准成功确认规则保存或丢弃 episode。
def finish_episode(
    *,
    buffer: EpisodeBuffer,
    cfg: DictConfig,
    metadata: dict,
    keep: bool = True,
    reason: str = "",
) -> dict | None:
    success = bool(buffer.final_info.get("is_success", False))
    confirmed = bool(buffer.final_info.get("success_confirmed", False))
    if buffer.steps < int(cfg.min_steps_to_save) or not success or not confirmed:
        keep = False

    if not keep:
        shutil.rmtree(buffer.tmp_dir, ignore_errors=True)
        discard_reasons = []
        if buffer.steps < int(cfg.min_steps_to_save):
            discard_reasons.append(f"steps<{cfg.min_steps_to_save}")
        if not success:
            discard_reasons.append("not_success")
        if not confirmed:
            discard_reasons.append("not_confirmed")
        print(
            f"Discarded episode {buffer.index:06d}: steps={buffer.steps}, "
            f"reason={','.join(discard_reasons) or reason or 'discarded'}"
        )
        return None

    array_key_maps = save_episode_arrays(buffer, cfg)
    intervention_index = save_intervention_index(buffer, cfg)
    reward_debug_path = save_episode_reward_debug(buffer, cfg)
    mode_arr = np.asarray(buffer.control_modes, dtype=np.int8)
    info = {
        "episode": int(buffer.index),
        "steps": int(buffer.steps),
        "reason": reason,
        "success": success,
        "fps": int(cfg.fps),
        "save_rgb": bool(cfg.save_rgb),
        "save_videos": bool(cfg.save_rgb and cfg.save_videos and buffer.video_paths),
        "save_depth": bool(cfg.save_depth),
        "save_reward_debug": bool(cfg.get("save_reward_debug", False)),
        "reward_debug_path": reward_debug_path,
        "observation_npz_keys": array_key_maps["observation"],
        "depth_npz_keys": array_key_maps["depth"],
        "video_paths": {
            f"pixels.{camera}": f"videos/{camera}.mp4"
            for camera in sorted(buffer.video_paths)
        },
        "intervention_segments_path": "intervention_segments.json",
        "intervention_segment_count": int(intervention_index["segment_count"]),
        "intervention_frames": int(intervention_index["intervention_frames"]),
        "control_mode_map": {str(code): mode for mode, code in MODE_CODE.items()},
        "blend_weight_order": list(ARM_ORDER),
        "mode_steps": {mode: int(np.sum(mode_arr == code)) for mode, code in MODE_CODE.items()},
        "final_cumulative_reward": float(buffer.cumulative_rewards[-1]) if buffer.cumulative_rewards else 0.0,
        "final_info": json_safe(buffer.final_info),
    }
    with open(buffer.tmp_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    if buffer.final_dir.exists():
        raise FileExistsError(f"Episode already exists: {buffer.final_dir}")
    buffer.tmp_dir.rename(buffer.final_dir)

    info["path"] = str(buffer.final_dir.relative_to(Path(metadata["run_dir"])))
    metadata["episodes"].append(info)
    metadata["saved_episodes"] = len(metadata["episodes"])
    metadata["successful_episodes"] = sum(
        1 for episode_info in metadata["episodes"] if bool(episode_info.get("success", False))
    )
    write_metadata(Path(metadata["run_dir"]), metadata)

    print(f"Saved episode {buffer.index:06d}: steps={buffer.steps}, success={success}, modes={info['mode_steps']}")
    return info


# 根据已有 episode 计算下一个编号。
def next_episode_index(run_dir: Path) -> int:
    indices = []
    for path in list_existing_episode_dirs(run_dir):
        match = re.search(r"episode_(\d+)$", path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


# 按任务和数据模态创建或复用 run 输出目录。
def make_run_dir(cfg: DictConfig, env_id: str) -> Path:
    modes = []
    if cfg.save_rgb:
        modes.append("rgb")
    if cfg.save_depth:
        modes.append("depth")
    suffix = "_".join(modes) if modes else "state"
    if cfg.run_name is not None:
        base_run_name = str(cfg.run_name)
        run_name = base_run_name if base_run_name.endswith(f"_{suffix}") else f"{base_run_name}_{suffix}"
    else:
        task_name = re.sub(r"[^A-Za-z0-9._=-]+", "_", env_id.split("/")[-1]).strip("_")
        run_name = f"quest_policy_{task_name}_{suffix}"
    run_dir = resolve_path(cfg.output_dir) / run_name
    if run_dir.exists() and not cfg.append:
        raise FileExistsError(f"Run directory already exists: {run_dir}. Set append=true or change run_name.")
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    return run_dir


# 固定 Python 数值库和策略采样用到的随机数。
def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# 重置环境、策略和遥操状态。
def reset_control_state(
    env,
    policy,
    quest_control: QuestControl,
    ik_solver: PoseActionIKSolver,
    *,
    seed: int | None = None,
):
    if seed is None:
        obs, _ = env.reset()
    else:
        set_random_seed(seed)
        obs, _ = env.reset(seed=int(seed))
    if hasattr(policy, "reset"):
        policy.reset()
    quest_control.reset()
    ik_solver.reset(active=False)
    return obs


# 检查当前 Quest 数据是否满足某个遥操模式所需的输入源。
def quest_ready_for_mode(ik_solver: PoseActionIKSolver, data, mode: str) -> bool:
    active_arms = set(MODE_ACTIVE_ARMS[mode])
    for state in ik_solver.states:
        if state.name not in active_arms:
            continue
        source = ik_solver.QUEST_SOURCE_BY_ARM[state.name]
        if not ik_solver._quest_source_ready(data, source):
            return False
    return True


# 切换控制模式，并为需要遥操的机械臂重新锚定当前末端位姿。
def switch_control_mode(
    mode: str,
    *,
    latest_data,
    policy,
    quest_control: QuestControl,
    ik_solver: PoseActionIKSolver,
) -> str | None:
    if mode not in MODE_ACTIVE_ARMS:
        print(f"Unsupported control mode: {mode}")
        return None

    switch_start = time.time()
    active_names = set(MODE_ACTIVE_ARMS[mode])
    if active_names:
        if latest_data is None:
            print(f"Cannot switch to {mode}: no Quest packet has arrived yet.")
            return None
        if not quest_ready_for_mode(ik_solver, latest_data, mode):
            print(f"Cannot switch to {mode}: required Quest pose is not ready.")
            return None

    # 不在模式切换时 reset policy。Diffusion/ACT 策略内部有 action chunk 队列，
    # 中途清空队列会重新采样动作块，容易导致策略控制部分出现抖动。
    quest_control.reset()
    ik_solver.reset(active=False)

    if not active_names:
        print(f"Control mode -> {MODE_LABEL[mode]} switch_ms={(time.time() - switch_start) * 1000.0:.1f}")
        return mode

    left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
    quest_control.start(latest_data, middle_pose, left_pose, right_pose)
    ik_solver.activate_from_data(latest_data, require_all=False)

    for arm_state in ik_solver.states:
        arm_state.active = arm_state.name in active_names

    active_summary = ", ".join(sorted(active_names))
    print(f"Control mode -> {MODE_LABEL[mode]} ({active_summary}) switch_ms={(time.time() - switch_start) * 1000.0:.1f}")
    return mode


# 根据当前模式计算 Quest 遥操候选动作。
def mixed_action(
    *,
    control_mode: str,
    policy_joint_action: np.ndarray,
    obs_before: dict,
    latest_data,
    quest_control: QuestControl,
    ik_solver: PoseActionIKSolver,
) -> tuple[np.ndarray, np.ndarray, bool]:
    pose_action = np.full(23, np.nan, dtype=np.float32)
    if control_mode == MODE_POLICY or latest_data is None:
        return policy_joint_action.copy(), pose_action, False

    left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
    pose_action, _feedback = quest_control.run(latest_data, left_pose, right_pose, middle_pose)
    teleop_joint_action, active_count = ik_solver.pose2joint(
        pose_action,
        obs=None,
        current_action=policy_joint_action,
    )
    return (
        np.asarray(teleop_joint_action, dtype=np.float32),
        np.asarray(pose_action, dtype=np.float32),
        bool(active_count > 0),
    )


# 按每条机械臂的 blend 权重平滑混合策略动作和遥操动作。
def blend_joint_action(
    *,
    policy_action: np.ndarray,
    teleop_action: np.ndarray,
    blender: TakeoverBlender,
    ik_solver: PoseActionIKSolver,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = blender.step()
    final_action = np.asarray(policy_action, dtype=np.float32).copy()
    teleop_action = np.asarray(teleop_action, dtype=np.float32)
    release_action = blender.release_action

    for arm_state in ik_solver.states:
        weight = float(weights.get(arm_state.name, 0.0))
        if weight <= 1.0e-6:
            continue

        source_action = teleop_action if blender.target.get(arm_state.name, 0.0) > 0.5 else release_action
        if source_action is None:
            source_action = teleop_action
        source_action = np.asarray(source_action, dtype=np.float32)

        final_action[arm_state.action_slice] = (
            (1.0 - weight) * final_action[arm_state.action_slice]
            + weight * source_action[arm_state.action_slice]
        )
        if arm_state.gripper_index is not None:
            idx = int(arm_state.gripper_index)
            final_action[idx] = (1.0 - weight) * final_action[idx] + weight * source_action[idx]

    return final_action, blender.weights_array(), blender.target_mask_array()


# 处理本地 OpenCV 相机窗口按键。
def handle_camera_window_key(key: int, state: RunState) -> None:
    if key in (-1, 255):
        return
    if key == ord("1"):
        state.requested_mode = MODE_POLICY
    elif key == ord("2"):
        state.requested_mode = MODE_RIGHT_TELEOP
    elif key == ord("3"):
        state.requested_mode = MODE_LEFT_TELEOP
    elif key == ord("4"):
        state.requested_mode = MODE_HEAD_TELEOP
    elif key == ord("5"):
        state.requested_mode = MODE_HANDS_TELEOP
    elif key == ord("6"):
        state.requested_mode = MODE_ALL_TELEOP
    elif key in (ord("y"), ord("Y")):
        state.reset = True
    elif key in (ord(" "), ord("r"), ord("R")):
        state.reset = True
    elif key in (ord("q"), ord("Q"), 27):
        state.quit = True


# 在本地窗口显示指定 ZED 相机画面。
def show_camera_window(
    *,
    physics,
    cfg: DictConfig,
    state: RunState,
    overlay_lines: list[str],
) -> None:
    if not cfg.camera_window:
        return
    rgb = physics.render(height=cfg.render_height, width=cfg.render_width, camera_id=cfg.display_camera)
    frame = draw_text_lines_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), overlay_lines)
    cv2.imshow(cfg.window_name, frame)
    handle_camera_window_key(cv2.waitKey(1) & 0xFF, state)


# 渲染一帧并异步发送到 Quest/Unity。
def send_unity_image(
    *,
    physics,
    cfg: DictConfig,
    image_streamer: UnityImageStreamer | None,
    overlay_lines: list[str],
) -> None:
    if image_streamer is None:
        return
    if cfg.unity_image_source not in ("rgb", "depth"):
        return

    if cfg.unity_image_source == "rgb":
        if cfg.unity_image_stereo:
            frames = []
            for camera in (cfg.unity_left_camera, cfg.unity_right_camera):
                rgb = physics.render(height=cfg.render_height, width=cfg.render_width, camera_id=camera)
                frames.append(draw_text_lines_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), overlay_lines))
            image_streamer.maybe_send_bgr(np.concatenate(frames, axis=1))
        else:
            rgb = physics.render(height=cfg.render_height, width=cfg.render_width, camera_id=cfg.display_camera)
            frame = draw_text_lines_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), overlay_lines)
            image_streamer.maybe_send_bgr(frame)
        return

    depth_min = float(cfg.depth_vis_min)
    depth_max = float(cfg.depth_vis_max)
    if cfg.unity_image_stereo:
        frames = []
        for camera in (cfg.unity_left_camera, cfg.unity_right_camera):
            depth = physics.render(height=cfg.render_height, width=cfg.render_width, camera_id=camera, depth=True)
            frames.append(draw_text_lines_bgr(depth_to_colormap_bgr(depth, depth_min, depth_max), overlay_lines))
        image_streamer.maybe_send_bgr(np.concatenate(frames, axis=1))
    else:
        depth = physics.render(height=cfg.render_height, width=cfg.render_width, camera_id=cfg.display_camera, depth=True)
        frame = draw_text_lines_bgr(depth_to_colormap_bgr(depth, depth_min, depth_max), overlay_lines)
        image_streamer.maybe_send_bgr(frame)


# 主循环：每步先算策略动作，再按模式覆盖头部或双臂。
def run(cfg: DictConfig) -> None:
    if float(cfg.fps) <= 0:
        raise ValueError(f"fps must be positive, got {cfg.fps}.")
    if cfg.unity_image_source not in ("rgb", "depth"):
        raise ValueError(f"unity_image_source must be rgb or depth, got {cfg.unity_image_source!r}.")
    if not cfg.ckpt_path:
        raise ValueError("ckpt_path must point to a pretrained policy checkpoint.")

    if cfg.mujoco_gl != "auto":
        os.environ["MUJOCO_GL"] = str(cfg.mujoco_gl)
    elif cfg.viewer and os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "glfw"
    elif not cfg.viewer and not cfg.camera_window and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    set_random_seed(cfg.random_seed)
    policy, full_cfg, device, _root_dir, load_dir = load_policy(cfg)
    env, policy_cameras, env_id = make_env(policy, full_cfg, cfg)
    sim_env = env.unwrapped
    physics = sim_env._physics
    record_cameras = list(cfg.record_cameras)

    run_dir = make_run_dir(cfg, env_id)
    existing_infos = load_existing_episode_infos(run_dir) if cfg.append else []
    episode_index = next_episode_index(run_dir)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "ckpt_path": str(resolve_path(cfg.ckpt_path)),
        "load_dir": str(load_dir),
        "env_id": env_id,
        "fps": int(cfg.fps),
        "policy_cameras": policy_cameras,
        "record_cameras": record_cameras,
        "save_rgb": bool(cfg.save_rgb),
        "save_videos": bool(cfg.save_rgb and cfg.save_videos),
        "save_depth": bool(cfg.save_depth),
        "save_reward_debug": bool(cfg.get("save_reward_debug", False)),
        "video_save_mode": "replay_after_success_confirm",
        "render_width": int(cfg.render_width),
        "render_height": int(cfg.render_height),
        "takeover_blend_steps": int(cfg.takeover_blend_steps),
        "blend_weight_order": list(ARM_ORDER),
        "random_seed": None if cfg.random_seed is None else int(cfg.random_seed),
        "fixed_reset_seed": cfg.random_seed is not None,
        "episode_initial_mode": MODE_POLICY,
        "control_mode_map": {str(code): mode for mode, code in MODE_CODE.items()},
        "camera_window": bool(cfg.camera_window),
        "display_camera": str(cfg.display_camera),
        "unity_image_stream": bool(cfg.unity_image_stream),
        "unity_image_source": str(cfg.unity_image_source),
        "unity_image_stereo": bool(cfg.unity_image_stereo),
        "saved_episodes": len(existing_infos),
        "successful_episodes": sum(1 for info in existing_infos if bool(info.get("success", False))),
        "episodes": existing_infos,
    }
    write_metadata(run_dir, metadata)

    receiver = QuestReceive(host=cfg.host, port=cfg.port, timeout=cfg.timeout)
    image_streamer = (
        UnityImageStreamer(
            host=cfg.unity_image_host,
            port=cfg.unity_image_port,
            send_hz=cfg.unity_image_hz,
            jpeg_quality=cfg.unity_image_jpeg_quality,
            chunk_size=cfg.unity_image_chunk_size,
            log_interval=cfg.unity_image_log_interval,
        )
        if cfg.unity_image_stream
        else None
    )
    if cfg.save_depth or cfg.camera_window or image_streamer is not None:
        warmup_camera = cfg.unity_left_camera if cfg.unity_image_stereo else cfg.display_camera
        physics.render(
            height=cfg.render_height,
            width=cfg.render_width,
            camera_id=warmup_camera,
            depth=bool(cfg.unity_image_source == "depth"),
        )
    if cfg.camera_window:
        cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.window_name, cfg.render_width, cfg.render_height)
    quest_control = QuestControl(
        use_head_control=cfg.head_control,
        use_individual_hand_anchors=cfg.individual_hand_anchors,
    )
    ik_solver = PoseActionIKSolver(
        sim_env,
        head_control=cfg.head_control,
        lock_roll=cfg.lock_roll,
        lock_pitch=cfg.lock_pitch,
        hand_position_scale=cfg.hand_position_scale,
        hand_max_delta=cfg.hand_max_delta,
        head_position_scale=cfg.head_position_scale,
        head_max_delta=cfg.head_max_delta,
        workspace_low=cfg.workspace_low,
        workspace_high=cfg.workspace_high,
    )

    print("预热 IK/Numba，避免第一次接管时卡顿...")
    warmup_start = time.time()
    ik_solver.warmup()
    print(f"IK warmup done: {(time.time() - warmup_start) * 1000.0:.1f} ms")

    obs = reset_control_state(env, policy, quest_control, ik_solver, seed=cfg.random_seed)
    takeover_blender = TakeoverBlender(blend_steps=cfg.takeover_blend_steps)
    last_joint_action = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
    buffer = start_episode(run_dir, episode_index, physics)
    cumulative_reward = 0.0
    latest_data = None
    prev_buttons: dict[str, bool] = {}
    last_print = 0.0
    waiting_save_confirm = False
    completion_reason = ""

    control_mode = MODE_POLICY
    state = RunState()
    viewer = launch_viewer(sim_env, state, cfg.viewer)
    dt = 1.0 / float(cfg.fps)

    print("\n" + "=" * 78)
    print("Policy + partitioned Quest teleop takeover")
    print("Quest 按键:")
    print("  A: 切换右臂接管/策略控制")
    print("  X: 切换左臂接管/策略控制")
    print("  B: 切换头部/中间臂接管/策略控制")
    print("  Y: 放弃当前 episode 并重置环境")
    print("  A+X: 轨迹完成后确认保存")
    print("本地窗口/Viewer 快捷键:")
    print("  1: 全部策略 | 2: 右臂接管 | 3: 左臂接管 | 4: 头部接管 | 5: 双臂接管 | 6: 全部接管")
    print("  Space/R: 放弃当前 episode 并重置 | Q/Esc: 退出")
    print(f"固定随机种子: {cfg.random_seed if cfg.random_seed is not None else '关闭'}")
    print(f"记录相机: {record_cameras}")
    print(f"保存 RGB 视频: {'开启' if cfg.save_rgb and cfg.save_videos else '关闭'}")
    print(f"保存深度: {'开启' if cfg.save_depth else '关闭'}")
    print(f"输出目录: {run_dir}")
    print("=" * 78 + "\n")

    switch_control_mode(
        MODE_POLICY,
        latest_data=latest_data,
        policy=policy,
        quest_control=quest_control,
        ik_solver=ik_solver,
    )
    control_mode = MODE_POLICY
    takeover_blender.set_mode(control_mode, current_action=last_joint_action)

    try:
        while not state.quit:
            loop_start = time.time()
            latest_data = poll_latest_quest(receiver, latest_data)
            if image_streamer is not None and receiver.latest_address is not None:
                image_streamer.update_auto_host(receiver.latest_address[0])

            quest_mode_request, quest_reset, save_confirm = quest_button_request(latest_data, prev_buttons, control_mode)
            requested_mode = state.requested_mode or quest_mode_request
            state.requested_mode = None
            if save_confirm and waiting_save_confirm:
                buffer.final_info["is_success"] = True
                buffer.final_info["success_confirmed"] = True
                buffer.final_info["success_confirm_method"] = "A+X"
                if cfg.save_rgb and cfg.save_videos:
                    print(f"Replaying episode {buffer.index:06d} to render videos...")
                    try:
                        replay_episode_videos(
                            buffer=buffer,
                            env_obj=env,
                            physics=physics,
                            cfg=cfg,
                            record_cameras=record_cameras,
                        )
                    except Exception as exc:
                        buffer.final_info["video_replay_error"] = repr(exc)
                        print(f"Episode {buffer.index:06d} video replay failed; arrays will still be saved: {exc}")
                saved_info = finish_episode(
                    buffer=buffer,
                    cfg=cfg,
                    metadata=metadata,
                    keep=True,
                    reason=completion_reason or "success_confirmed",
                )
                episode_index = next_episode_index(run_dir)
                obs = reset_control_state(env, policy, quest_control, ik_solver, seed=cfg.random_seed)
                takeover_blender.reset()
                last_joint_action = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
                buffer = start_episode(run_dir, episode_index, physics)
                cumulative_reward = 0.0
                waiting_save_confirm = False
                completion_reason = ""
                switch_control_mode(
                    MODE_POLICY,
                    latest_data=latest_data,
                    policy=policy,
                    quest_control=quest_control,
                    ik_solver=ik_solver,
                )
                control_mode = MODE_POLICY
                takeover_blender.set_mode(control_mode, current_action=last_joint_action)
                result = "saved" if saved_info is not None else "not saved"
                print(f"A+X confirmed. Episode {result}; started episode {episode_index:06d}.")
                continue

            if requested_mode is not None and not waiting_save_confirm:
                switched = switch_control_mode(
                    requested_mode,
                    latest_data=latest_data,
                    policy=policy,
                    quest_control=quest_control,
                    ik_solver=ik_solver,
                )
                if switched is not None:
                    control_mode = switched
                    takeover_blender.set_mode(control_mode, current_action=last_joint_action)

            if state.reset or quest_reset:
                buffer.final_info.setdefault("manual_abort", True)
                finish_episode(
                    buffer=buffer,
                    cfg=cfg,
                    metadata=metadata,
                    keep=False,
                    reason="manual_reset",
                )
                episode_index = next_episode_index(run_dir)
                obs = reset_control_state(env, policy, quest_control, ik_solver, seed=cfg.random_seed)
                takeover_blender.reset()
                last_joint_action = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
                buffer = start_episode(run_dir, episode_index, physics)
                cumulative_reward = 0.0
                waiting_save_confirm = False
                completion_reason = ""
                state.reset = False
                switch_control_mode(
                    MODE_POLICY,
                    latest_data=latest_data,
                    policy=policy,
                    quest_control=quest_control,
                    ik_solver=ik_solver,
                )
                control_mode = MODE_POLICY
                takeover_blender.set_mode(control_mode, current_action=last_joint_action)
                print(f"Discarded current episode. Started episode {episode_index:06d}.")
                continue

            overlay_lines = [
                f"mode: {control_mode}",
                f"steps: {buffer.steps}",
                f"complete: {'YES' if waiting_save_confirm else 'NO'}",
                f"saved episodes: {metadata['saved_episodes']}",
                f"blend L/R/M: {np.round(takeover_blender.weights_array(), 2).tolist()}",
            ]
            if waiting_save_confirm:
                overlay_lines.append("Press A+X to save")
                show_camera_window(
                    physics=physics,
                    cfg=cfg,
                    state=state,
                    overlay_lines=overlay_lines,
                )
                send_unity_image(
                    physics=physics,
                    cfg=cfg,
                    image_streamer=image_streamer,
                    overlay_lines=overlay_lines,
                )
                if viewer is not None:
                    viewer.sync()
                sleep_time = dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            obs_before = obs
            policy_joint_action = policy_action(policy, obs_before, device)
            teleop_joint_action, pose_action, teleop_target_ready = mixed_action(
                control_mode=control_mode,
                policy_joint_action=policy_joint_action,
                obs_before=obs_before,
                latest_data=latest_data,
                quest_control=quest_control,
                ik_solver=ik_solver,
            )
            joint_action, blend_weights, teleop_mask = blend_joint_action(
                policy_action=policy_joint_action,
                teleop_action=teleop_joint_action,
                blender=takeover_blender,
                ik_solver=ik_solver,
            )
            teleop_applied = bool(teleop_target_ready or np.any(blend_weights > 1.0e-6))

            depth_frames = None
            if cfg.save_depth and record_cameras:
                depth_frames = {
                    camera: physics.render(
                        height=int(cfg.render_height),
                        width=int(cfg.render_width),
                        camera_id=camera,
                        depth=True,
                    ).astype(np.float32, copy=False)
                    for camera in record_cameras
                }

            obs, reward, terminated, truncated, info = env.step(joint_action)
            last_joint_action = np.asarray(joint_action, dtype=np.float32).copy()
            cumulative_reward += float(reward)
            record_step(
                buffer,
                obs_before=obs_before,
                joint_action=joint_action,
                policy_joint_action=policy_joint_action,
                pose_action=pose_action,
                control_mode=control_mode,
                teleop_applied=teleop_applied,
                blend_weights=blend_weights,
                teleop_mask=teleop_mask,
                reward=float(reward),
                cumulative_reward=cumulative_reward,
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
                record_cameras=record_cameras,
                depth_frames=depth_frames,
                save_reward_debug=bool(cfg.get("save_reward_debug", False)),
            )

            if bool(info.get("is_success", False)):
                completion_reason = "success"
                buffer.final_info["is_success"] = True
                waiting_save_confirm = True
                print("Task success detected. Paused. Press A+X together to save, or Y/Space/R to discard.")
            elif terminated or truncated or buffer.steps >= cfg.max_steps_per_episode:
                if terminated:
                    completion_reason = "terminated"
                elif truncated:
                    completion_reason = "truncated"
                else:
                    completion_reason = "max_steps"
                buffer.final_info["completion_reason"] = completion_reason
                finish_episode(
                    buffer=buffer,
                    cfg=cfg,
                    metadata=metadata,
                    keep=False,
                    reason=completion_reason,
                )
                episode_index = next_episode_index(run_dir)
                obs = reset_control_state(env, policy, quest_control, ik_solver, seed=cfg.random_seed)
                takeover_blender.reset()
                last_joint_action = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
                buffer = start_episode(run_dir, episode_index, physics)
                cumulative_reward = 0.0
                waiting_save_confirm = False
                completion_reason = ""
                switch_control_mode(
                    MODE_POLICY,
                    latest_data=latest_data,
                    policy=policy,
                    quest_control=quest_control,
                    ik_solver=ik_solver,
                )
                control_mode = MODE_POLICY
                takeover_blender.set_mode(control_mode, current_action=last_joint_action)
                print(f"Episode ended without success. Started episode {episode_index:06d}.")
                continue

            now = time.time()
            if cfg.status_hz_interval > 0 and now - last_print >= float(cfg.status_hz_interval):
                print(
                    f"[{control_mode}] episode={buffer.index:06d} steps={buffer.steps} "
                    f"saved={metadata['saved_episodes']} reward={cumulative_reward:.2f} "
                    f"teleop={int(teleop_applied)} blend={np.round(blend_weights, 2).tolist()}"
                )
                last_print = now

            overlay_lines = [
                f"mode: {control_mode}",
                f"steps: {buffer.steps}",
                f"complete: {'YES' if waiting_save_confirm else 'NO'}",
                f"saved episodes: {metadata['saved_episodes']}",
                f"blend L/R/M: {np.round(takeover_blender.weights_array(), 2).tolist()}",
            ]
            if waiting_save_confirm:
                overlay_lines.append("Press A+X to save")
            show_camera_window(
                physics=physics,
                cfg=cfg,
                state=state,
                overlay_lines=overlay_lines,
            )
            send_unity_image(
                physics=physics,
                cfg=cfg,
                image_streamer=image_streamer,
                overlay_lines=overlay_lines,
            )

            if viewer is not None:
                viewer.sync()

            sleep_time = dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        if buffer.tmp_dir.exists():
            shutil.rmtree(buffer.tmp_dir, ignore_errors=True)

        receiver.close()
        if image_streamer is not None:
            image_streamer.close()
        if viewer is not None:
            viewer.close()
        if cfg.camera_window:
            cv2.destroyAllWindows()
        env.close()


# Hydra 命令行入口，配置格式与 quest_teleop_collect.py 保持一致。
@hydra.main(version_base="1.2", config_name="quest_policy_collect", config_path="../configs/data_collect")
def quest_policy_collect_cli(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    default_args = [
        (
            "ckpt_path='outputs/2_pretrain/train/2026-06-30/"
            "23-42-18_InsertCylinder-3Arms-v0_pre_zed_two_model_diffusion/"
            "checkpoints/176000_loss=0.0031_sr=60.0_ar=577.94'"
        ),                                                  # 预训练策略 checkpoint
        "device=cuda",                                      # 策略推理设备
        "env_id=guided_vision/InsertCylinder-3Arms-v0",     # 仿真任务；设为 null 时读取 checkpoint 配置
        "max_steps_per_episode=600",                        # 单条轨迹最大步数
        "fps=25",                                           # 控制、记录与视频帧率
        "takeover_blend_steps=15",                          # 策略与人工接管的平滑过渡步数
        "head_control=true",                                # 是否允许头显接管中间臂
        "lock_pitch=false",                                 # 是否锁定中间臂 pitch
        "lock_roll=true",                                   # 是否锁定中间臂 roll
        "save_pose_action=true",                            # 是否保存 Quest 末端位姿动作
        "save_rgb=true",                                    # 成功确认后是否回放并渲染 RGB
        "save_videos=true",                                 # 是否将回放 RGB 保存为 mp4
        "save_depth=false",                                 # 是否保存各记录相机的逐像素深度
        "camera_window=false",                              # 是否打开本地 RGB 窗口
        "viewer=true",                                      # 是否打开 MuJoCo viewer
        "unity_image_stream=true",                          # 是否向 Quest/Unity 发送画面
        "unity_image_source=rgb",                           # Quest/Unity 画面类型：rgb/depth
        "unity_image_stereo=true",                          # 是否发送左右眼 side-by-side 画面
    ]

    for arg in default_args:
        arg_key = arg.split("=", 1)[0]
        if not any(sys_arg.split("=", 1)[0] == arg_key for sys_arg in sys.argv):
            sys.argv.append(arg)

    quest_policy_collect_cli()
