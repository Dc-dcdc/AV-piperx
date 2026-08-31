#!/home/dc/miniforge3/envs/AV-piper/bin/python
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import socket
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import gymnasium as gym
import hydra
import imageio.v2 as imageio
import mujoco.viewer
import numpy as np
from omegaconf import DictConfig


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_COLLECT_DIR = ROOT_DIR / "data_collect"
ENV_DIR = ROOT_DIR / "env"

for path in (ROOT_DIR, DATA_COLLECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_collect.expert_data_collection.quest_receive import QuestReceive
from data_collect.expert_data_collection.quest_send import UnityImageStreamer
from data_collect.expert_data_collection.quest_control import QuestControl
from data_collect.expert_data_collection.quest_pose_filter import QuestPoseActionFilter, QuestPoseFilterConfig
from data_collect.expert_data_collection.robot_ik_solver import PoseActionIKSolver
from data_collect.expert_data_collection.headset_utils import HeadsetData

if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

import env as _register_guided_vision_envs  # noqa: F401


@dataclass
class EpisodeBuffer:
    index: int
    tmp_dir: Path
    final_dir: Path
    obs_traces: dict[str, list[np.ndarray]] = field(default_factory=dict)
    joint_actions: list[np.ndarray] = field(default_factory=list)
    pose_actions: list[np.ndarray] = field(default_factory=list)
    cumulative_rewards: list[float] = field(default_factory=list)
    reward_debug: list[dict] = field(default_factory=list)
    depth_traces: dict[str, list[np.ndarray]] = field(default_factory=dict)
    initial_time: float | None = None
    initial_qpos: np.ndarray | None = None
    initial_qvel: np.ndarray | None = None
    initial_ctrl: np.ndarray | None = None
    initial_act: np.ndarray | None = None
    initial_mocap_pos: np.ndarray | None = None
    initial_mocap_quat: np.ndarray | None = None
    # MuJoCo 无关节静态 body（例如 InsertCylinder 容器）的位置不在
    # data.qpos 中，需要额外保存 model 侧初态才能精确离线重放。
    initial_model_body_pos: np.ndarray | None = None
    initial_model_body_quat: np.ndarray | None = None
    video_writer: AsyncEpisodeVideoWriter | None = None
    video_paths: dict[str, Path] = field(default_factory=dict)
    final_info: dict = field(default_factory=dict)

    # 返回当前 episode 已记录的步数。
    @property
    def steps(self) -> int:
        return len(self.joint_actions)


# 将图像整理成 uint8 HWC 格式。
def image_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


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


# 计算深度图分布统计，便于判断远处背景是否主导数据。
def depth_stats(depth: np.ndarray, vis_max: float) -> dict[str, float] | None:
    values = np.asarray(depth, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "over_vis_max_pct": float(np.mean(values > vis_max) * 100.0),
    }


# 在 BGR 图像上叠加多行状态文字。
def draw_text_lines_bgr(frame_bgr: np.ndarray, lines: list[str], x: int = 14, y: int = 26) -> np.ndarray:
    output = frame_bgr.copy()
    text_y = y
    for line in lines:
        cv2.putText(output, line, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(output, line, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 245, 255), 1, cv2.LINE_AA)
        text_y += 24
    return output


# 后台异步写入单条 episode 的多相机视频。
class AsyncEpisodeVideoWriter:
    def __init__(self, videos_dir: Path, fps: int, max_queue_size: int):
        self.videos_dir = videos_dir
        self.fps = int(fps)
        self.paths: dict[str, Path] = {}
        self._writers: dict[str, object] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=max(0, int(max_queue_size)))
        self._closed = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="episode-video-writer", daemon=True)
        self._thread.start()

    # 将一帧相机图像加入后台写入队列。
    def enqueue(self, camera: str, frame: np.ndarray) -> Path:
        if self._closed:
            raise RuntimeError("Video writer is already closed.")
        if self._error is not None:
            raise RuntimeError("Video writer failed.") from self._error

        self.videos_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.setdefault(camera, self.videos_dir / f"{camera}.mp4")
        self._queue.put((camera, np.asarray(frame).copy()))
        return path

    # 关闭后台写入线程，并按需丢弃未写入帧。
    def close(self, *, discard_pending: bool = False) -> dict[str, Path]:
        if self._closed:
            return self.paths

        if discard_pending:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()

        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._closed = True

        if self._error is not None:
            raise RuntimeError("Video writer failed.") from self._error
        return self.paths

    # 后台线程循环，从队列中取帧并写入对应相机视频。
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    break

                camera, frame = item
                if camera not in self._writers:
                    self._writers[camera] = imageio.get_writer(
                        str(self.paths[camera]),
                        fps=self.fps,
                        macro_block_size=1,
                    )
                self._writers[camera].append_data(image_to_uint8_hwc(frame))
            except BaseException as exc:
                self._error = exc
            finally:
                self._queue.task_done()

        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
        self._writers.clear()


# 展平观测中的数值字段，跳过图像数据。
def flatten_numeric_obs(obs, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(obs, dict):
        flattened = {}
        for key, value in obs.items():
            raw_key = f"{prefix}.{key}" if prefix else str(key)
            if raw_key == "pixels" or raw_key.startswith("observation.images"):
                continue
            flattened.update(flatten_numeric_obs(value, raw_key))
        return flattened

    try:
        array = np.asarray(obs)
    except Exception:
        return {}

    if array.dtype.kind not in "biufc":
        return {}
    return {prefix: array.astype(np.float32, copy=False)}


# 将同一字段的多步数据堆叠成数组。
def stack_trace(values: list[np.ndarray]) -> np.ndarray:
    try:
        return np.stack(values, axis=0)
    except ValueError:
        return np.asarray(values, dtype=object)


# 将 numpy 类型转换成 json 可以直接保存的 Python 类型。
def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


# 列出 run 目录下已有的有效 episode 目录。
def list_existing_episode_dirs(run_dir: Path) -> list[Path]:
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.exists():
        return []
    return sorted(
        path
        for path in episodes_dir.glob("episode_*")
        if path.is_dir() and not path.name.endswith(".tmp") and (path / "arrays.npz").exists()
    )


# 从 episode 目录名中解析 episode 编号。
def episode_index_from_dir(episode_dir: Path) -> int:
    match = re.search(r"episode_(\d+)$", episode_dir.name)
    return int(match.group(1)) if match else -1


# 计算下一个可写入的 episode 编号。
def next_episode_index(run_dir: Path) -> int:
    indices = [episode_index_from_dir(path) for path in list_existing_episode_dirs(run_dir)]
    indices = [index for index in indices if index >= 0]
    return max(indices, default=-1) + 1


# 读取已有 episode 的 info 信息，用于追加采集。
def load_existing_episode_infos(run_dir: Path) -> list[dict]:
    infos = []
    for episode_dir in list_existing_episode_dirs(run_dir):
        info_path = episode_dir / "info.json"
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
        else:
            info = {"episode": episode_index_from_dir(episode_dir), "success": True}
        info["path"] = str(episode_dir.relative_to(run_dir))
        infos.append(info)
    return infos


# 根据本次采集模态生成目录后缀。
def data_mode_suffix(cfg: DictConfig) -> str:
    modes = []
    if cfg.save_rgb:
        modes.append("rgb")
    if cfg.save_depth:
        modes.append("depth")
    return "_".join(modes) if modes else "state"


# 创建或复用本次采集的 run 目录。
def make_run_dir(cfg: DictConfig) -> Path:
    output_dir = Path(cfg.output_dir).expanduser()
    output_dir = output_dir if output_dir.is_absolute() else ROOT_DIR / output_dir
    suffix = data_mode_suffix(cfg)
    if cfg.run_name is None:
        task_name = str(cfg.env_id).split("/", 1)[-1]
        safe_task_name = re.sub(r"[^A-Za-z0-9._=-]+", "_", task_name).strip("_") or "unknown"
        run_name = f"quest_teleop_{safe_task_name}_{suffix}"
    else:
        base_run_name = str(cfg.run_name)
        run_name = base_run_name if base_run_name.endswith(f"_{suffix}") else f"{base_run_name}_{suffix}"
    run_dir = output_dir / str(run_name)
    if run_dir.exists() and not cfg.append:
        raise FileExistsError(f"Run directory already exists: {run_dir}. Set append=true or change run_name.")
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    return run_dir


# 写入 run 级别的 metadata.json。
def write_metadata(run_dir: Path, metadata: dict) -> None:
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# 将当前 episode 的基础数值轨迹保存为 arrays.npz。
def save_episode_arrays(buffer: EpisodeBuffer, cfg: DictConfig) -> dict[str, dict[str, str]]:
    obs_key_map = {}
    arrays = {"joint_action": np.asarray(buffer.joint_actions, dtype=np.float32)}
    if buffer.initial_time is not None:
        arrays["initial_time"] = np.asarray(buffer.initial_time, dtype=np.float64)
    if buffer.initial_qpos is not None:
        arrays["initial_qpos"] = np.asarray(buffer.initial_qpos, dtype=np.float64)
    if buffer.initial_qvel is not None:
        arrays["initial_qvel"] = np.asarray(buffer.initial_qvel, dtype=np.float64)
    if buffer.initial_ctrl is not None:
        arrays["initial_ctrl"] = np.asarray(buffer.initial_ctrl, dtype=np.float64)
    if buffer.initial_act is not None:
        arrays["initial_act"] = np.asarray(buffer.initial_act, dtype=np.float64)
    if buffer.initial_mocap_pos is not None:
        arrays["initial_mocap_pos"] = np.asarray(buffer.initial_mocap_pos, dtype=np.float64)
    if buffer.initial_mocap_quat is not None:
        arrays["initial_mocap_quat"] = np.asarray(buffer.initial_mocap_quat, dtype=np.float64)
    if buffer.initial_model_body_pos is not None:
        arrays["initial_model_body_pos"] = np.asarray(
            buffer.initial_model_body_pos,
            dtype=np.float64,
        )
    if buffer.initial_model_body_quat is not None:
        arrays["initial_model_body_quat"] = np.asarray(
            buffer.initial_model_body_quat,
            dtype=np.float64,
        )
    if cfg.save_pose_action:
        arrays["pose_action"] = np.asarray(buffer.pose_actions, dtype=np.float32)
    depth_key_map = {}
    for camera, values in sorted(buffer.depth_traces.items()):
        if not values:
            continue
        npz_key = f"depth__{camera}"
        depth_key_map[camera] = npz_key
        arrays[npz_key] = stack_trace(values).astype(np.float32, copy=False)

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


# 将 reward 调试信息单独保存，避免污染训练用 arrays.npz。
def save_episode_reward_debug(buffer: EpisodeBuffer, cfg: DictConfig) -> str | None:
    if not bool(cfg.get("save_reward_debug", False)) or not buffer.reward_debug:
        return None

    reward_debug_path = buffer.tmp_dir / "reward_debug.jsonl"
    with reward_debug_path.open("w", encoding="utf-8") as f:
        for item in buffer.reward_debug:
            json.dump(json_safe(item), f, ensure_ascii=False)
            f.write("\n")
    return reward_debug_path.name


# 关闭当前 episode 打开的相机视频写入器。
def close_episode_videos(
    buffer: EpisodeBuffer,
    *,
    discard_pending: bool = False,
    raise_errors: bool = True,
) -> None:
    if buffer.video_writer is None:
        return

    try:
        buffer.video_paths.update(
            buffer.video_writer.close(discard_pending=discard_pending)
        )
    except RuntimeError as exc:
        if raise_errors:
            raise
        print(f"Video writer closed with error while discarding episode {buffer.index:06d}: {exc}")
    finally:
        buffer.video_writer = None


# 创建一个临时 episode 缓冲目录。
def start_episode(run_dir: Path, episode_index: int) -> EpisodeBuffer:
    tmp_dir = run_dir / "episodes" / f"episode_{episode_index:06d}.tmp"
    final_dir = run_dir / "episodes" / f"episode_{episode_index:06d}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if final_dir.exists():
        raise FileExistsError(f"Episode already exists: {final_dir}")
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return EpisodeBuffer(index=episode_index, tmp_dir=tmp_dir, final_dir=final_dir)


# 记录单步 transition 到 episode 缓冲区。
def record_transition(
    *,
    buffer: EpisodeBuffer,
    obs,
    joint_action: np.ndarray,
    pose_action: np.ndarray,
    reward: float,
    info: dict,
    terminated: bool,
    truncated: bool,
    record_cameras: list[str],
    save_depth: bool,
    save_reward_debug: bool,
    depth_frames: dict[str, np.ndarray] | None,
) -> None:
    step_index = buffer.steps

    for raw_key, value in flatten_numeric_obs(obs).items():
        buffer.obs_traces.setdefault(raw_key, []).append(value)

    if save_depth and depth_frames:
        for camera in record_cameras:
            if camera not in depth_frames:
                continue
            buffer.depth_traces.setdefault(camera, []).append(
                np.asarray(depth_frames[camera], dtype=np.float32).copy()
            )

    buffer.joint_actions.append(np.asarray(joint_action, dtype=np.float32))
    buffer.pose_actions.append(np.asarray(pose_action, dtype=np.float32))
    if save_reward_debug:
        previous_reward = buffer.cumulative_rewards[-1] if buffer.cumulative_rewards else 0.0
        cumulative_reward = float(previous_reward + float(reward))
        buffer.cumulative_rewards.append(cumulative_reward)
        reward_item = {
            "step": int(step_index),
            "reward": float(reward),
            "cumulative_reward": cumulative_reward,
            "is_success": bool((info or {}).get("is_success", False)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        env_reward_debug = (info or {}).get("reward_debug")
        if isinstance(env_reward_debug, dict):
            reward_item.update(json_safe(env_reward_debug))
        buffer.reward_debug.append(reward_item)
    buffer.final_info = dict(info or {})


# 将 MuJoCo 恢复到 episode 开始录制时的物理状态。
def restore_episode_initial_state(buffer: EpisodeBuffer, physics) -> bool:
    if buffer.initial_qpos is None or buffer.initial_qvel is None:
        return False

    model_body_values = (
        buffer.initial_model_body_pos,
        buffer.initial_model_body_quat,
    )
    if any(value is None for value in model_body_values) and any(
        value is not None for value in model_body_values
    ):
        raise ValueError(
            "initial_model_body_pos和initial_model_body_quat必须同时存在。"
        )
    if buffer.initial_model_body_pos is not None:
        if physics.model.body_pos.shape != buffer.initial_model_body_pos.shape:
            raise ValueError(
                "initial_model_body_pos形状与当前MuJoCo模型不一致: "
                f"model={physics.model.body_pos.shape}, "
                f"saved={buffer.initial_model_body_pos.shape}"
            )
        if physics.model.body_quat.shape != buffer.initial_model_body_quat.shape:
            raise ValueError(
                "initial_model_body_quat形状与当前MuJoCo模型不一致: "
                f"model={physics.model.body_quat.shape}, "
                f"saved={buffer.initial_model_body_quat.shape}"
            )
        physics.model.body_pos[:] = buffer.initial_model_body_pos
        physics.model.body_quat[:] = buffer.initial_model_body_quat

    physics.data.qpos[:] = buffer.initial_qpos
    physics.data.qvel[:] = buffer.initial_qvel
    if buffer.initial_time is not None:
        physics.data.time = float(buffer.initial_time)
    if buffer.initial_ctrl is not None and physics.data.ctrl.shape == buffer.initial_ctrl.shape:
        physics.data.ctrl[:] = buffer.initial_ctrl
    if buffer.initial_act is not None and physics.data.act.shape == buffer.initial_act.shape:
        physics.data.act[:] = buffer.initial_act
    if (
        buffer.initial_mocap_pos is not None
        and physics.data.mocap_pos.shape == buffer.initial_mocap_pos.shape
    ):
        physics.data.mocap_pos[:] = buffer.initial_mocap_pos
    if (
        buffer.initial_mocap_quat is not None
        and physics.data.mocap_quat.shape == buffer.initial_mocap_quat.shape
    ):
        physics.data.mocap_quat[:] = buffer.initial_mocap_quat
    physics.forward()
    return True


# 保存确认后回放关节动作，并离线渲染多相机 RGB 视频。
def replay_episode_videos(
    *,
    buffer: EpisodeBuffer,
    env_obj,
    physics,
    cfg: DictConfig,
    record_cameras: list[str],
) -> bool:
    if not cfg.save_rgb or not cfg.save_videos or not record_cameras:
        return False
    if not buffer.joint_actions:
        return False
    if not restore_episode_initial_state(buffer, physics):
        print(f"Episode {buffer.index:06d} has no initial state. Skip replay video rendering.")
        return False

    writer = AsyncEpisodeVideoWriter(
        buffer.tmp_dir / "videos",
        fps=int(cfg.fps),
        max_queue_size=int(cfg.video_queue_size),
    )
    try:
        for action in buffer.joint_actions:
            env_obj.step(np.asarray(action, dtype=np.float64))
            for camera in record_cameras:
                frame = physics.render(
                    height=int(cfg.render_height),
                    width=int(cfg.render_width),
                    camera_id=camera,
                )
                buffer.video_paths[camera] = writer.enqueue(camera, frame)
    finally:
        buffer.video_paths.update(writer.close(discard_pending=False))

    buffer.final_info["video_replay_rendered"] = True
    buffer.final_info["video_replay_cameras"] = list(record_cameras)
    return True


# 完成当前 episode，满足成功确认条件时落盘保存。
def finish_episode(
    *,
    buffer: EpisodeBuffer,
    cfg: DictConfig,
    metadata: dict,
    keep: bool = True,
) -> dict | None:
    success = bool(buffer.final_info.get("is_success", False))
    confirmed = bool(buffer.final_info.get("success_confirmed", False))

    if buffer.steps < cfg.min_steps_to_save:
        keep = False
    if not success:
        keep = False
    if not confirmed:
        keep = False

    close_episode_videos(
        buffer,
        discard_pending=not keep,
        raise_errors=keep,
    )

    if not keep:
        shutil.rmtree(buffer.tmp_dir, ignore_errors=True)
        reason = []
        if buffer.steps < cfg.min_steps_to_save:
            reason.append(f"steps<{cfg.min_steps_to_save}")
        if not success:
            reason.append("not_success")
        if not confirmed:
            reason.append("not_confirmed")
        print(f"Discarded episode {buffer.index:06d}: steps={buffer.steps}, reason={','.join(reason) or 'discarded'}")
        return None

    array_key_maps = save_episode_arrays(buffer, cfg)
    reward_debug_path = save_episode_reward_debug(buffer, cfg)
    episode_info = {
        "episode": int(buffer.index),
        "success": success,
        "steps": int(buffer.steps),
        "fps": int(cfg.fps),
        "path": str(buffer.final_dir.relative_to(metadata["run_dir"])),
        "save_rgb": bool(cfg.save_rgb),
        "save_videos": bool(cfg.save_rgb and cfg.save_videos and buffer.video_paths),
        "save_depth": bool(cfg.save_depth),
        "save_reward_debug": bool(cfg.get("save_reward_debug", False)),
        "reward_debug_path": reward_debug_path,
        "final_cumulative_reward": (
            float(buffer.cumulative_rewards[-1])
            if bool(cfg.get("save_reward_debug", False)) and buffer.cumulative_rewards
            else None
        ),
        "observation_npz_keys": array_key_maps["observation"],
        "depth_npz_keys": array_key_maps["depth"],
        "video_paths": {
            f"pixels.{camera}": f"videos/{camera}.mp4"
            for camera in sorted(buffer.video_paths)
        },
        "final_info": buffer.final_info,
    }

    with (buffer.tmp_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(episode_info, f, indent=2, ensure_ascii=False)

    buffer.tmp_dir.rename(buffer.final_dir)

    metadata["episodes"].append(episode_info)
    metadata["saved_episodes"] = len(metadata["episodes"])
    metadata["successful_episodes"] = sum(1 for info in metadata["episodes"] if bool(info.get("success", False)))
    write_metadata(Path(metadata["run_dir"]), metadata)
    print(f"Saved episode {buffer.index:06d}: steps={buffer.steps}, success={success}")
    return episode_info


# 打印采集脚本启动信息和按键说明。
def _print_header(cfg: DictConfig, run_dir: Path, record_cameras: list[str]) -> None:
    pose_filter_cfg = cfg.get("pose_filter", {})
    print("\nQuest3 -> MuJoCo teleop data collection")
    print("-" * 78)
    print("Config:        configs/data_collect/quest_teleop_collect.yaml")
    print(f"Output:        {run_dir}")
    print(f"UDP:           {cfg.host}:{cfg.port}")
    print(f"Env:           {cfg.env_id}")
    print(f"Record cams:   {record_cameras}")
    print(f"Save RGB:      {'on' if cfg.save_rgb else 'off'}")
    video_mode = "on, replay after A+X" if cfg.save_rgb and cfg.save_videos else "off"
    print(f"Save videos:   {video_mode}")
    print(f"Save depth:    {'on' if cfg.save_depth else 'off'}")
    print(f"Reward debug:  {'on' if cfg.get('save_reward_debug', False) else 'off'}")
    print(f"Depth window:  {'on' if cfg.depth_window else 'off'}")
    print(f"Pose filter:   {'on' if pose_filter_cfg.get('enabled', True) else 'off'}")
    print(f"FPS:           {cfg.fps}")
    print(f"Loop limit:    {1000.0 / float(cfg.fps):.1f} ms")
    print(f"Unity stream:  {'on' if cfg.unity_image_stream else 'off'}")
    print(f"Unity source:  {cfg.unity_image_source}")
    if cfg.unity_image_stream:
        stereo_text = "on" if cfg.unity_image_stereo else "off"
        print(f"Unity stereo:  {stereo_text} ({cfg.unity_left_camera} | {cfg.unity_right_camera})")
    print("-" * 78)
    print("Controls:")
    print("  A or X or P: anchor valid Quest poses and start recording")
    print("  A+X together: save after SUCCESS is shown")
    print("  B/Y or R/Space: abort current episode without saving, reset env, and pause")
    print("  Q or Esc: quit")
    print("-" * 78)


# 非阻塞丢弃 UDP 队列中的旧包，返回当前可读到的最新 Quest 数据。
def _refresh_latest_quest_data(receiver: QuestReceive, fallback: HeadsetData | None) -> HeadsetData | None:
    latest = fallback
    old_timeout = receiver.sock.gettimeout()
    receiver.sock.settimeout(0.0)
    try:
        while True:
            latest = receiver.receive_data()
    except (BlockingIOError, socket.timeout):
        pass
    finally:
        receiver.sock.settimeout(old_timeout)
    return latest


# 运行 Quest 遥操采集主循环。
def run(cfg: DictConfig) -> None:
    if float(cfg.fps) <= 0:
        raise ValueError(f"fps must be positive, got {cfg.fps}.")
    pose_filter_config = QuestPoseFilterConfig.from_mapping(
        cfg.get("pose_filter", {}),
        fps=float(cfg.fps),
    )
    pose_action_filter = QuestPoseActionFilter(pose_filter_config)
    loop_period = 1.0 / float(cfg.fps)
    max_loop_ms = loop_period * 1000.0  # 实际最大的loop时长
    loop_timeout_margin_ms = 20.0       # 给予一定的缓冲时长
    slow_loop_limit_ms = max_loop_ms + loop_timeout_margin_ms  # 最终要求每一轮时长限制
    max_consecutive_slow_loops = 5
    loop_timeout_grace_steps = max(1, int(round(float(cfg.fps) * 0.25)))
    if cfg.unity_image_source not in ("rgb", "depth"):
        raise ValueError(f"unity_image_source must be rgb or depth, got {cfg.unity_image_source!r}.")

    if cfg.mujoco_gl != "auto":
        os.environ["MUJOCO_GL"] = cfg.mujoco_gl
    elif cfg.viewer and os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "glfw"
    elif not cfg.viewer and not cfg.camera_window and not cfg.depth_window and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    run_dir = make_run_dir(cfg)
    record_cameras = list(cfg.record_cameras)
    # 实时遥操阶段不让 env.step 渲染多相机图像，保存确认后再回放生成视频。
    env_cameras = []
    _print_header(cfg, run_dir, record_cameras)
    print(f"Loop startup grace: {loop_timeout_grace_steps} step(s)")
    print(
        "Loop timeout: "
        f">{slow_loop_limit_ms:.1f} ms "
        f"for {max_consecutive_slow_loops} consecutive recorded step(s)"
    )

    existing_infos = load_existing_episode_infos(run_dir) if cfg.append else []
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "env_id": str(cfg.env_id),
        "fps": int(cfg.fps),
        "record_cameras": record_cameras,
        "save_rgb": bool(cfg.save_rgb),
        "save_videos": bool(cfg.save_rgb and cfg.save_videos),
        "save_depth": bool(cfg.save_depth),
        "save_reward_debug": bool(cfg.get("save_reward_debug", False)),
        "pose_action_semantics": "filtered_quest_target_pose",
        "pose_filter": pose_filter_config.to_dict(),
        "video_save_mode": "replay_after_success_confirm",
        "render_width": int(cfg.render_width),
        "render_height": int(cfg.render_height),
        "saved_episodes": len(existing_infos),
        "successful_episodes": sum(1 for info in existing_infos if bool(info.get("success", False))),
        "episodes": existing_infos,
    }
    write_metadata(run_dir, metadata)
    episode_index = next_episode_index(run_dir)

    env_make_kwargs = {
        "disable_env_checker": True,
        "cameras": env_cameras,
        "episode_length": cfg.episode_length,
        "observation_height": cfg.render_height,
        "observation_width": cfg.render_width,
    }
    if bool(cfg.get("save_reward_debug", False)):
        env_make_kwargs["enable_reward_debug"] = True

    env_obj = gym.make(cfg.env_id, **env_make_kwargs)
    sim_env = env_obj.unwrapped
    obs, _ = env_obj.reset()
    physics = sim_env._physics

    if cfg.save_depth or cfg.depth_window or cfg.camera_window or cfg.unity_image_stream:
        warmup_camera = cfg.depth_display_camera if cfg.depth_display_camera is not None else cfg.display_camera
        physics.render(
            height=cfg.render_height,
            width=cfg.render_width,
            camera_id=warmup_camera,
            depth=bool(cfg.save_depth or cfg.depth_window or cfg.unity_image_source == "depth"),
        )
        if cfg.unity_image_stream and cfg.unity_image_stereo:
            for camera in dict.fromkeys([cfg.unity_left_camera, cfg.unity_right_camera]):
                physics.render(
                    height=cfg.render_height,
                    width=cfg.render_width,
                    camera_id=camera,
                    depth=bool(cfg.unity_image_source == "depth"),
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
    quest_control = QuestControl(
        use_head_control=cfg.head_control,
        use_individual_hand_anchors=cfg.individual_hand_anchors,
    )
    print("Warming up IK before waiting for Quest anchor...")
    warmup_t0 = time.perf_counter()
    ik_solver.warmup()
    print(f"IK warmup done in {(time.perf_counter() - warmup_t0):.2f}s.")

    command = {"anchor": False, "reset": False, "quit": False, "save_success": False}

    # 处理 MuJoCo viewer 的键盘控制。
    def key_callback(keycode: int) -> None:
        if keycode in (ord("p"), ord("P")):
            command["anchor"] = True
        elif keycode in (ord("r"), ord("R"), 32):
            command["reset"] = True
        elif keycode in (ord("q"), ord("Q"), 256):
            command["quit"] = True

    viewer_cm = (
        mujoco.viewer.launch_passive(
            physics.model.ptr,
            physics.data.ptr,
            show_left_ui=True,
            show_right_ui=True,
            key_callback=key_callback,
        )
        if cfg.viewer
        else nullcontext(None)
    )

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

    if cfg.camera_window:
        cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.window_name, cfg.render_width, cfg.render_height)
    if cfg.depth_window:
        cv2.namedWindow(cfg.depth_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.depth_window_name, cfg.render_width, cfg.render_height)

    recording = False
    waiting_success_confirm = False
    skip_recording_this_loop = False
    consecutive_slow_loops = 0
    episode_buffer: EpisodeBuffer | None = None
    latest_data: HeadsetData | None = None
    latest_feedback = None
    prev_start_button = False
    prev_reset_button = False
    prev_success_confirm = False
    last_status_t = 0.0
    last_success_notice = ""
    last_success_notice_t = 0.0
    latest_timing_ms: dict[str, float] = {}
    latest_depth_stats: dict[str, dict[str, float]] = {}

    try:
        with viewer_cm as viewer:
            while True:
                loop_start = time.time()
                loop_start_perf = time.perf_counter()
                timing_updated = False

                if viewer is not None and not viewer.is_running():
                    break
                if command["quit"]:
                    break

                try:
                    latest_data = receiver.receive_latest_data()
                except socket.timeout:
                    latest_data = None
                except json.JSONDecodeError as exc:
                    print(f"Invalid Quest JSON packet: {exc}")
                    latest_data = None

                if latest_data is not None:
                    if image_streamer is not None and receiver.latest_address is not None:
                        image_streamer.update_auto_host(receiver.latest_address[0])

                    success_button = bool(latest_data.r_button_one and latest_data.l_button_one)
                    start_button = quest_control.should_start(latest_data)
                    reset_button = quest_control.should_reset(latest_data)
                    if success_button and not prev_success_confirm and waiting_success_confirm:
                        command["save_success"] = True
                    elif start_button and not prev_start_button and not recording and episode_buffer is None:
                        command["anchor"] = True
                    if reset_button and not prev_reset_button:
                        command["reset"] = True
                    prev_start_button = start_button
                    prev_reset_button = reset_button
                    prev_success_confirm = success_button

                if command["save_success"]:
                    if episode_buffer is None:
                        print("No active episode to save.")
                    elif not waiting_success_confirm:
                        print("Task is not waiting for success confirmation.")
                    else:
                        finished_episode_index = episode_buffer.index
                        episode_buffer.final_info["is_success"] = True
                        episode_buffer.final_info["success_confirmed"] = True
                        episode_buffer.final_info["success_confirm_method"] = "A+X"
                        if cfg.save_rgb and cfg.save_videos:
                            print(f"Replaying episode {finished_episode_index:06d} to render videos...")
                            try:
                                replay_episode_videos(
                                    buffer=episode_buffer,
                                    env_obj=env_obj,
                                    physics=physics,
                                    cfg=cfg,
                                    record_cameras=record_cameras,
                                )
                            except Exception as exc:
                                episode_buffer.final_info["video_replay_error"] = repr(exc)
                                print(
                                    f"Episode {finished_episode_index:06d} video replay failed. "
                                    f"Arrays will still be saved: {exc}"
                                )
                        saved_info = finish_episode(buffer=episode_buffer, cfg=cfg, metadata=metadata, keep=True)
                        if saved_info is not None:
                            episode_index += 1
                            last_success_notice = f"episode {finished_episode_index:06d}: SUCCESS saved"
                        else:
                            last_success_notice = f"episode {finished_episode_index:06d}: SUCCESS not saved"
                        last_success_notice_t = time.time()
                        episode_buffer = None

                        obs, _ = env_obj.reset()
                        ik_solver.reset()
                        quest_control.reset()
                        pose_action_filter.reset()
                        recording = False
                        waiting_success_confirm = False
                        skip_recording_this_loop = False
                        consecutive_slow_loops = 0
                        latest_feedback = None
                        print("Env reset, waiting for a new QuestControl anchor.")
                    command["save_success"] = False

                if command["reset"]:
                    if episode_buffer is not None:
                        episode_buffer.final_info.setdefault("manual_abort", True)
                        finish_episode(buffer=episode_buffer, cfg=cfg, metadata=metadata, keep=False)
                        episode_buffer = None

                    obs, _ = env_obj.reset()
                    ik_solver.reset()
                    quest_control.reset()
                    pose_action_filter.reset()
                    recording = False
                    waiting_success_confirm = False
                    skip_recording_this_loop = False
                    consecutive_slow_loops = 0
                    latest_feedback = None
                    command["reset"] = False
                    print("Reset MuJoCo env. Waiting for a new QuestControl anchor.")

                can_auto_anchor = latest_data is not None and (
                    ik_solver.can_anchor_from_data(latest_data, allow_partial=cfg.allow_partial_anchor)
                )
                if cfg.start_on_first_packet and can_auto_anchor and not recording and episode_buffer is None:
                    command["anchor"] = True

                if command["anchor"]:
                    if recording:
                        print("Already recording. Wait for SUCCESS, then press A+X together to save.")
                    elif waiting_success_confirm:
                        print("Task succeeded. Press A+X together to save, or reset to abort.")
                    elif latest_data is None:
                        print("Cannot anchor yet: no Quest data.")
                    else:
                        try:
                            latest_data = _refresh_latest_quest_data(receiver, latest_data)
                        except json.JSONDecodeError as exc:
                            print(f"Invalid Quest JSON packet while anchoring: {exc}")
                        if latest_data is None:
                            print("Cannot anchor yet: no Quest data.")
                            command["anchor"] = False
                            continue
                        active_count = ik_solver.activate_from_data(
                            latest_data,
                            require_all=not cfg.allow_partial_anchor,
                        )
                        if active_count > 0:
                            left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
                            quest_control.start(latest_data, middle_pose, left_pose, right_pose)
                            pose_action_filter.reset()
                            episode_buffer = start_episode(run_dir, episode_index)
                            episode_buffer.initial_time = float(physics.data.time)
                            # 从专家轨迹中获取初始状态
                            episode_buffer.initial_qpos = physics.data.qpos.copy()
                            episode_buffer.initial_qvel = physics.data.qvel.copy()
                            episode_buffer.initial_ctrl = physics.data.ctrl.copy()
                            episode_buffer.initial_act = physics.data.act.copy()
                            episode_buffer.initial_mocap_pos = physics.data.mocap_pos.copy()
                            episode_buffer.initial_mocap_quat = physics.data.mocap_quat.copy()
                            episode_buffer.initial_model_body_pos = (
                                physics.model.body_pos.copy()
                            )
                            episode_buffer.initial_model_body_quat = (
                                physics.model.body_quat.copy()
                            )
                            recording = True
                            waiting_success_confirm = False
                            skip_recording_this_loop = True
                            consecutive_slow_loops = 0
                            print(f"Started episode {episode_index:06d}.")
                    command["anchor"] = False

                if recording and latest_data is not None and episode_buffer is not None and not skip_recording_this_loop:
                    obs_before = obs

                    timing_t0 = time.perf_counter()
                    left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
                    timing_t1 = time.perf_counter()
                    pose_action, latest_feedback = quest_control.run(latest_data, left_pose, right_pose, middle_pose)
                    pose_action = pose_action_filter.filter(
                        pose_action,
                        timestamp=time.monotonic(),
                    )
                    timing_t2 = time.perf_counter()
                    joint_action, active_count = ik_solver.pose2joint(pose_action, obs_before)
                    timing_t3 = time.perf_counter()
                    latest_timing_ms = {
                        "pose": (timing_t1 - timing_t0) * 1000.0,
                        "map": (timing_t2 - timing_t1) * 1000.0,
                        "ik": (timing_t3 - timing_t2) * 1000.0,
                        "control": (timing_t3 - timing_t0) * 1000.0,
                    }
                    timing_updated = True

                    if active_count > 0:
                        depth_frames = None
                        timing_t4 = time.perf_counter()
                        # 收集深度信息
                        if cfg.save_depth and record_cameras:
                            depth_frames = {}
                            for camera in record_cameras:
                                depth_frames[camera] = physics.render(
                                    height=cfg.render_height,
                                    width=cfg.render_width,
                                    camera_id=camera,
                                    depth=True,
                                ).astype(np.float32, copy=False)
                            latest_depth_stats.update(
                                {
                                    camera: stats
                                    for camera, depth in depth_frames.items()
                                    if (stats := depth_stats(depth, float(cfg.depth_vis_max))) is not None
                                }
                            )
                        timing_t5 = time.perf_counter()
                        obs, _reward, terminated, truncated, info = env_obj.step(joint_action)
                        timing_t6 = time.perf_counter()
                        record_transition(
                            buffer=episode_buffer,
                            obs=obs_before,
                            joint_action=joint_action,
                            pose_action=pose_action,
                            reward=float(_reward),
                            info=info,
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            record_cameras=record_cameras,
                            save_depth=bool(cfg.save_depth),
                            save_reward_debug=bool(cfg.get("save_reward_debug", False)),
                            depth_frames=depth_frames,
                        )
                        timing_t7 = time.perf_counter()
                        latest_timing_ms.update(
                            {
                                "depth": (timing_t5 - timing_t4) * 1000.0,
                                "env": (timing_t6 - timing_t5) * 1000.0,
                                "rec": (timing_t7 - timing_t6) * 1000.0,
                                "control": (timing_t7 - timing_t0) * 1000.0,
                            }
                        )

                        task_success = bool(info.get("is_success", False))
                        if task_success:
                            episode_buffer.final_info["is_success"] = True
                            recording = False
                            waiting_success_confirm = True
                            print("Task success detected. Paused. Press A+X together to save.")
                        elif terminated or truncated or episode_buffer.steps >= cfg.max_steps_per_episode:
                            saved_info = finish_episode(buffer=episode_buffer, cfg=cfg, metadata=metadata, keep=True)
                            if saved_info is not None:
                                episode_index += 1
                            episode_buffer = None
                            obs, _ = env_obj.reset()
                            ik_solver.reset()
                            quest_control.reset()
                            pose_action_filter.reset()
                            recording = False
                            waiting_success_confirm = False
                            skip_recording_this_loop = False
                            consecutive_slow_loops = 0
                            latest_feedback = None
                            print("Episode ended. Env reset, waiting for a new QuestControl anchor.")

                status_frame = None
                depth_frame = None
                quest_steps = episode_buffer.steps if episode_buffer is not None else 0
                quest_success = bool(
                    waiting_success_confirm
                    or (
                        episode_buffer is not None
                        and episode_buffer.final_info.get("is_success", False)
                    )
                )
                quest_overlay_lines = [
                    f"steps: {quest_steps}",
                    f"success: {'YES' if quest_success else 'NO'}",
                    f"saved episodes: {metadata['saved_episodes']}",
                ]

                if cfg.camera_window:
                    frame_rgb = physics.render(
                        height=cfg.render_height,
                        width=cfg.render_width,
                        camera_id=cfg.display_camera,
                    )
                    status_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(cfg.window_name, status_frame)

                if image_streamer is not None and cfg.unity_image_source == "rgb":
                    if cfg.unity_image_stereo:
                        left_rgb = physics.render(
                            height=cfg.render_height,
                            width=cfg.render_width,
                            camera_id=cfg.unity_left_camera,
                        )
                        right_rgb = physics.render(
                            height=cfg.render_height,
                            width=cfg.render_width,
                            camera_id=cfg.unity_right_camera,
                        )
                        left_frame = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
                        right_frame = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
                        left_frame = draw_text_lines_bgr(left_frame, quest_overlay_lines)
                        right_frame = draw_text_lines_bgr(right_frame, quest_overlay_lines)
                        image_streamer.maybe_send_bgr(np.concatenate([left_frame, right_frame], axis=1))
                    else:
                        if status_frame is None:
                            frame_rgb = physics.render(
                                height=cfg.render_height,
                                width=cfg.render_width,
                                camera_id=cfg.display_camera,
                            )
                            status_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        image_streamer.maybe_send_bgr(draw_text_lines_bgr(status_frame, quest_overlay_lines))

                depth_min = float(cfg.depth_vis_min)
                depth_max = float(cfg.depth_vis_max)

                if cfg.depth_window:
                    depth_camera = cfg.depth_display_camera if cfg.depth_display_camera is not None else cfg.display_camera
                    depth = physics.render(
                        height=cfg.render_height,
                        width=cfg.render_width,
                        camera_id=depth_camera,
                        depth=True,
                    )
                    depth_frame = depth_to_colormap_bgr(depth, depth_min, depth_max)
                    stats = depth_stats(depth, depth_max)
                    if stats is not None:
                        latest_depth_stats[depth_camera] = stats
                        depth_text = (
                            f"{depth_camera} depth "
                            f"min={stats['min']:.3f} "
                            f"p50={stats['p50']:.3f} "
                            f"p95={stats['p95']:.3f} "
                            f"max={stats['max']:.3f} "
                            f"vis=[{depth_min:.2f},{depth_max:.2f}]"
                        )
                    else:
                        depth_text = f"{depth_camera} depth invalid"
                    depth_frame = draw_text_lines_bgr(depth_frame, [depth_text], y=28)
                    cv2.imshow(cfg.depth_window_name, depth_frame)

                if image_streamer is not None and cfg.unity_image_source == "depth":
                    if cfg.unity_image_stereo:
                        stereo_depth_frames = []
                        for label, camera in (("L", cfg.unity_left_camera), ("R", cfg.unity_right_camera)):
                            depth = physics.render(
                                height=cfg.render_height,
                                width=cfg.render_width,
                                camera_id=camera,
                                depth=True,
                            )
                            stereo_depth_frame = depth_to_colormap_bgr(depth, depth_min, depth_max)
                            stats = depth_stats(depth, depth_max)
                            if stats is not None:
                                latest_depth_stats[camera] = stats
                                depth_text = (
                                    f"{label}: {camera} depth "
                                    f"min={stats['min']:.3f} "
                                    f"p50={stats['p50']:.3f} "
                                    f"p95={stats['p95']:.3f} "
                                    f"max={stats['max']:.3f}"
                                )
                            else:
                                depth_text = f"{label}: {camera} depth invalid"
                            stereo_depth_frame = draw_text_lines_bgr(stereo_depth_frame, quest_overlay_lines)
                            stereo_depth_frames.append(stereo_depth_frame)
                        image_streamer.maybe_send_bgr(np.concatenate(stereo_depth_frames, axis=1))
                    else:
                        if depth_frame is None:
                            depth_camera = cfg.depth_display_camera if cfg.depth_display_camera is not None else cfg.display_camera
                            depth = physics.render(
                                height=cfg.render_height,
                                width=cfg.render_width,
                                camera_id=depth_camera,
                                depth=True,
                            )
                            depth_frame = depth_to_colormap_bgr(depth, depth_min, depth_max)
                            stats = depth_stats(depth, depth_max)
                            if stats is not None:
                                latest_depth_stats[depth_camera] = stats
                                depth_text = (
                                    f"{depth_camera} depth "
                                    f"min={stats['min']:.3f} "
                                    f"p50={stats['p50']:.3f} "
                                    f"p95={stats['p95']:.3f} "
                                    f"max={stats['max']:.3f} "
                                    f"vis=[{depth_min:.2f},{depth_max:.2f}]"
                                )
                            else:
                                depth_text = f"{depth_camera} depth invalid"
                            depth_frame = draw_text_lines_bgr(depth_frame, [depth_text], y=28)
                        image_streamer.maybe_send_bgr(draw_text_lines_bgr(depth_frame, quest_overlay_lines, y=52))

                if cfg.camera_window or cfg.depth_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key in (ord("p"), ord("P")):
                        command["anchor"] = True
                    if key in (ord("r"), ord("R"), 32):
                        command["reset"] = True

                if viewer is not None:
                    viewer.sync()

                if timing_updated:
                    latest_timing_ms["loop"] = (time.perf_counter() - loop_start_perf) * 1000.0
                    if (
                        recording
                        and episode_buffer is not None
                        and episode_buffer.steps > loop_timeout_grace_steps
                    ):
                        if latest_timing_ms["loop"] > slow_loop_limit_ms:
                            consecutive_slow_loops += 1
                        else:
                            consecutive_slow_loops = 0

                        if consecutive_slow_loops >= max_consecutive_slow_loops:
                            delayed_loop_ms = latest_timing_ms["loop"]
                            episode_buffer.final_info["loop_timeout_ms"] = delayed_loop_ms
                            episode_buffer.final_info["loop_target_ms"] = max_loop_ms
                            episode_buffer.final_info["loop_limit_ms"] = slow_loop_limit_ms
                            episode_buffer.final_info["loop_timeout_margin_ms"] = loop_timeout_margin_ms
                            episode_buffer.final_info["loop_timeout_consecutive_count"] = consecutive_slow_loops
                            finish_episode(buffer=episode_buffer, cfg=cfg, metadata=metadata, keep=False)
                            episode_buffer = None
                            obs, _ = env_obj.reset()
                            ik_solver.reset()
                            quest_control.reset()
                            pose_action_filter.reset()
                            recording = False
                            waiting_success_confirm = False
                            skip_recording_this_loop = False
                            consecutive_slow_loops = 0
                            latest_feedback = None
                            latest_depth_stats = {}
                            last_success_notice = (
                                f"loop timeout {delayed_loop_ms:.1f}>{slow_loop_limit_ms:.1f} ms "
                                f"x{max_consecutive_slow_loops}, episode reset"
                            )
                            last_success_notice_t = time.time()
                            print(last_success_notice)
                    else:
                        consecutive_slow_loops = 0

                now = time.time()
                if now - last_status_t >= cfg.status_hz_interval:
                    packet_state = "data" if latest_data is not None else "no-data"
                    run_state = "success" if waiting_success_confirm else "recording" if recording else "paused"
                    steps = episode_buffer.steps if episode_buffer is not None else 0
                    timing_text = ""
                    if latest_timing_ms:
                        timing_text = (
                            " timing_ms="
                            f"map:{latest_timing_ms.get('map', 0.0):.1f} "
                            f"ik:{latest_timing_ms.get('ik', 0.0):.1f} "
                            f"depth:{latest_timing_ms.get('depth', 0.0):.1f} "
                            f"env:{latest_timing_ms.get('env', 0.0):.1f} "
                            f"rec:{latest_timing_ms.get('rec', 0.0):.1f} "
                            f"loop:{latest_timing_ms.get('loop', 0.0):.1f}"
                        )
                    depth_stats_text = ""
                    if latest_depth_stats:
                        depth_camera = cfg.depth_display_camera if cfg.depth_display_camera is not None else cfg.display_camera
                        if depth_camera not in latest_depth_stats:
                            depth_camera = next(iter(latest_depth_stats))
                        stats = latest_depth_stats[depth_camera]
                        depth_stats_text = (
                            f" depth[{depth_camera}]="
                            f"min:{stats['min']:.3f} "
                            f"p50:{stats['p50']:.3f} "
                            f"p95:{stats['p95']:.3f} "
                            f"p99:{stats['p99']:.3f} "
                            f"max:{stats['max']:.3f} "
                            f"mean:{stats['mean']:.3f} "
                            f">vis:{stats['over_vis_max_pct']:.1f}%"
                        )
                    print(
                        f"[{run_state:9s}] {packet_state:9s} "
                        f"saved episode={metadata['saved_episodes']} steps={steps}"
                        f"{timing_text}"
                        f"{depth_stats_text}"
                    )
                    last_status_t = now

                sleep_t = loop_period - (time.time() - loop_start)
                skip_recording_this_loop = False
                if sleep_t > 0:
                    time.sleep(sleep_t)
    finally:
        if episode_buffer is not None:
            close_episode_videos(episode_buffer, discard_pending=True, raise_errors=False)
            shutil.rmtree(episode_buffer.tmp_dir, ignore_errors=True)
            print(f"Discarded unfinished episode {episode_buffer.index:06d}.")
        receiver.close()
        if image_streamer is not None:
            image_streamer.close()
        env_obj.close()
        if cfg.camera_window or cfg.depth_window:
            cv2.destroyAllWindows()

# Hydra 命令行入口，加载配置后启动采集。
@hydra.main(version_base="1.2", config_name="quest_teleop_collect", config_path="../../configs/data_collect")
def quest_teleop_collect_cli(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    default_args = [
        "max_steps_per_episode=350",                  # 最大步长
        "head_control=true",                          # 是否使用头显控制中间臂
        "lock_pitch=False",                            # 是否锁定中间臂 pitch 角，true 时禁用抬头低头
        "lock_roll=true",                             # 是否锁定中间臂 link6 转轴对应的 roll
        "save_pose_action=true",                      # 是否额外保存 Quest 映射后的末端位姿动作
        "save_rgb=true",                              # 是否在 A+X 确认后回放轨迹并保存 RGB 视频
        "save_depth=false",                           # 是否保存 record_cameras 中每个相机的逐像素深度图
        "depth_window=false",                         # 是否打开本地 OpenCV 深度图窗口
        "camera_window=false",                       # 是否打开本地 OpenCV RGB 窗口，关闭可减少一次额外渲染
        "unity_image_source=rgb",                     # 默认发送 RGB 图到 Quest
        "unity_image_stereo=true",                   # 是否按左右眼 side-by-side 发送到 Quest
        # 不在这里设置 env_id；任务场景只通过 yaml 或命令行 env_id 切换。
        # "unity_image_stream=false",
    ]

    for arg in default_args:
        arg_key = arg.split("=", 1)[0]
        if not any(sys_arg.split("=", 1)[0] == arg_key for sys_arg in sys.argv):
            sys.argv.append(arg)

    quest_teleop_collect_cli()
