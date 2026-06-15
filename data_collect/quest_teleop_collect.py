#!/home/dc/miniforge3/envs/DPPO/bin/python
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sys
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


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_COLLECT_DIR = ROOT_DIR / "data_collect"
ENV_DIR = ROOT_DIR / "env"

for path in (ROOT_DIR, DATA_COLLECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quest_receive import QuestReceive
from quest_send import UnityImageStreamer
from quest_control import QuestControl
from data_collect.robot_ik_solver import PoseActionIKSolver
from headset_utils import HeadsetData

if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

import env as _register_guided_vision_envs  # noqa: F401
from env.constants import SIM_DT


@dataclass
class EpisodeBuffer:
    index: int
    tmp_dir: Path
    final_dir: Path
    obs_traces: dict[str, list[np.ndarray]] = field(default_factory=dict)
    joint_actions: list[np.ndarray] = field(default_factory=list)
    pose_actions: list[np.ndarray] = field(default_factory=list)
    video_writers: dict[str, object] = field(default_factory=dict)
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


# 创建或复用本次采集的 run 目录。
def make_run_dir(cfg: DictConfig) -> Path:
    output_dir = Path(cfg.output_dir).expanduser()
    output_dir = output_dir if output_dir.is_absolute() else ROOT_DIR / output_dir
    if cfg.run_name is None:
        task_name = str(cfg.env_id).split("/", 1)[-1]
        safe_task_name = re.sub(r"[^A-Za-z0-9._=-]+", "_", task_name).strip("_") or "unknown"
        run_name = f"quest_teleop_{safe_task_name}"
    else:
        run_name = cfg.run_name
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
def save_episode_arrays(buffer: EpisodeBuffer, cfg: DictConfig) -> dict[str, str]:
    obs_key_map = {}
    arrays = {"joint_action": np.asarray(buffer.joint_actions, dtype=np.float32)}
    if cfg.save_pose_action:
        arrays["pose_action"] = np.asarray(buffer.pose_actions, dtype=np.float32)

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
    return obs_key_map


# 关闭当前 episode 打开的相机视频写入器。
def close_episode_videos(buffer: EpisodeBuffer) -> None:
    for writer in buffer.video_writers.values():
        writer.close()
    buffer.video_writers.clear()


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
    info: dict,
    record_cameras: list[str],
    save_videos: bool,
    fps: int,
) -> None:
    for raw_key, value in flatten_numeric_obs(obs).items():
        buffer.obs_traces.setdefault(raw_key, []).append(value)

    if save_videos and record_cameras:
        if "pixels" in obs:
            pixels = {key: np.asarray(value) for key, value in obs["pixels"].items()}
        else:
            image_prefix = "observation.images."
            pixels = {
                key.replace(image_prefix, ""): np.asarray(value)
                for key, value in obs.items()
                if key.startswith(image_prefix)
            }

        videos_dir = buffer.tmp_dir / "videos"
        for camera in record_cameras:
            if camera not in pixels:
                continue
            if camera not in buffer.video_writers:
                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"{camera}.mp4"
                buffer.video_paths[camera] = video_path
                buffer.video_writers[camera] = imageio.get_writer(
                    str(video_path),
                    fps=fps,
                    macro_block_size=1,
                )
            buffer.video_writers[camera].append_data(image_to_uint8_hwc(pixels[camera]))

    buffer.joint_actions.append(np.asarray(joint_action, dtype=np.float32))
    buffer.pose_actions.append(np.asarray(pose_action, dtype=np.float32))
    buffer.final_info = dict(info or {})


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
    close_episode_videos(buffer)

    if buffer.steps < cfg.min_steps_to_save:
        keep = False
    if not success:
        keep = False
    if not confirmed:
        keep = False

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

    obs_key_map = save_episode_arrays(buffer, cfg)
    episode_info = {
        "episode": int(buffer.index),
        "success": success,
        "steps": int(buffer.steps),
        "fps": int(cfg.fps),
        "path": str(buffer.final_dir.relative_to(metadata["run_dir"])),
        "observation_npz_keys": obs_key_map,
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
    print("\nQuest3 -> MuJoCo teleop data collection")
    print("-" * 78)
    print("Config:        configs/data_collect/quest_teleop_collect.yaml")
    print(f"Output:        {run_dir}")
    print(f"UDP:           {cfg.host}:{cfg.port}")
    print(f"Env:           {cfg.env_id}")
    print(f"Record cams:   {record_cameras}")
    print(f"FPS:           {cfg.fps}")
    print(f"Unity stream:  {'on' if cfg.unity_image_stream else 'off'}")
    print("-" * 78)
    print("Controls:")
    print("  A or X or P: anchor valid Quest poses and start recording")
    print("  A+X together: save after SUCCESS is shown")
    print("  B/Y or R/Space: abort current episode without saving, reset env, and pause")
    print("  Q or Esc: quit")
    print("-" * 78)


# 运行 Quest 遥操采集主循环。
def run(cfg: DictConfig) -> None:
    if cfg.mujoco_gl != "auto":
        os.environ["MUJOCO_GL"] = cfg.mujoco_gl
    elif not cfg.viewer and not cfg.camera_window and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    run_dir = make_run_dir(cfg)
    record_cameras = list(cfg.record_cameras)
    env_cameras = record_cameras if record_cameras else [cfg.display_camera]
    _print_header(cfg, run_dir, record_cameras)

    existing_infos = load_existing_episode_infos(run_dir) if cfg.append else []
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "env_id": str(cfg.env_id),
        "fps": int(cfg.fps),
        "record_cameras": record_cameras,
        "render_width": int(cfg.render_width),
        "render_height": int(cfg.render_height),
        "saved_episodes": len(existing_infos),
        "successful_episodes": sum(1 for info in existing_infos if bool(info.get("success", False))),
        "episodes": existing_infos,
    }
    write_metadata(run_dir, metadata)
    episode_index = next_episode_index(run_dir)

    env_obj = gym.make(
        cfg.env_id,
        disable_env_checker=True,
        cameras=env_cameras,
        episode_length=cfg.episode_length,
        observation_height=cfg.render_height,
        observation_width=cfg.render_width,
    )
    sim_env = env_obj.unwrapped
    obs, _ = env_obj.reset()
    physics = sim_env._physics

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

    recording = False
    waiting_success_confirm = False
    episode_buffer: EpisodeBuffer | None = None
    latest_data: HeadsetData | None = None
    latest_feedback = None
    prev_start_button = False
    prev_reset_button = False
    prev_success_confirm = False
    last_status_t = 0.0
    last_success_notice = ""
    last_success_notice_t = 0.0

    try:
        with viewer_cm as viewer:
            while True:
                loop_start = time.time()

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
                        recording = False
                        waiting_success_confirm = False
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
                    recording = False
                    waiting_success_confirm = False
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
                        active_count = ik_solver.activate_from_data(
                            latest_data,
                            require_all=not cfg.allow_partial_anchor,
                        )
                        if active_count > 0:
                            left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
                            quest_control.start(latest_data, middle_pose, left_pose, right_pose)
                            episode_buffer = start_episode(run_dir, episode_index)
                            recording = True
                            waiting_success_confirm = False
                            print(f"Started episode {episode_index:06d}.")
                    command["anchor"] = False

                if recording and latest_data is not None and episode_buffer is not None:
                    obs_before = obs
                    left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
                    pose_action, latest_feedback = quest_control.run(latest_data, left_pose, right_pose, middle_pose)
                    joint_action, active_count = ik_solver.pose2joint(pose_action, obs_before)
                    if active_count > 0:
                        obs, _reward, terminated, truncated, info = env_obj.step(joint_action)
                        record_transition(
                            buffer=episode_buffer,
                            obs=obs_before,
                            joint_action=joint_action,
                            pose_action=pose_action,
                            info=info,
                            record_cameras=record_cameras,
                            save_videos=bool(cfg.save_videos),
                            fps=int(cfg.fps),
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
                            recording = False
                            waiting_success_confirm = False
                            latest_feedback = None
                            print("Episode ended. Env reset, waiting for a new QuestControl anchor.")

                if cfg.camera_window or image_streamer is not None:
                    frame_rgb = physics.render(
                        height=cfg.render_height,
                        width=cfg.render_width,
                        camera_id=cfg.display_camera,
                    )
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    if waiting_success_confirm:
                        status = "SUCCESS"
                    elif recording:
                        status = "RECORDING"
                    else:
                        status = "PAUSED"
                    steps = episode_buffer.steps if episode_buffer is not None else 0
                    success_text = "YES" if episode_buffer is not None and episode_buffer.final_info.get("is_success", False) else "NO"
                    active_text = " ".join(f"{state.name[0].upper()}:{'on' if state.active else 'off'}" for state in ik_solver.states)
                    lines = [
                        f"{status} | A/X/P start | A+X save after success | B/Y/R abort | Q quit",
                        f"episode: {episode_index:06d} | steps: {steps} | success: {success_text} | saved: {metadata['saved_episodes']}",
                        f"active: {active_text}",
                    ]
                    if waiting_success_confirm:
                        lines.insert(0, "SUCCESS! Press A+X together to save")
                    if latest_feedback is not None:
                        lines.append(
                            f"sync: H={int(latest_feedback.head_out_of_sync)} L={int(latest_feedback.left_out_of_sync)} R={int(latest_feedback.right_out_of_sync)}"
                        )
                    if last_success_notice and time.time() - last_success_notice_t < 3.0:
                        lines.append(last_success_notice)

                    status_frame = frame_bgr.copy()
                    y = 26
                    for line in lines:
                        cv2.putText(status_frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 3, cv2.LINE_AA)
                        cv2.putText(status_frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 245, 255), 1, cv2.LINE_AA)
                        y += 24

                    if image_streamer is not None:
                        image_streamer.maybe_send_bgr(status_frame)

                    if cfg.camera_window:
                        cv2.imshow(cfg.window_name, status_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        if key in (ord("p"), ord("P")):
                            command["anchor"] = True
                        if key in (ord("r"), ord("R"), 32):
                            command["reset"] = True

                if viewer is not None:
                    viewer.sync()

                now = time.time()
                if now - last_status_t >= cfg.status_hz_interval:
                    packet_state = "data" if latest_data is not None else "no-data"
                    run_state = "success" if waiting_success_confirm else "recording" if recording else "paused"
                    steps = episode_buffer.steps if episode_buffer is not None else 0
                    print(
                        f"[{run_state:9s}] {packet_state:9s} episode={episode_index:06d} "
                        f"steps={steps} saved={metadata['saved_episodes']}"
                    )
                    last_status_t = now

                sleep_t = SIM_DT - (time.time() - loop_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    finally:
        if episode_buffer is not None:
            close_episode_videos(episode_buffer)
            shutil.rmtree(episode_buffer.tmp_dir, ignore_errors=True)
            print(f"Discarded unfinished episode {episode_buffer.index:06d}.")
        receiver.close()
        if image_streamer is not None:
            image_streamer.close()
        env_obj.close()
        if cfg.camera_window:
            cv2.destroyAllWindows()

# Hydra 命令行入口，加载配置后启动采集。
@hydra.main(version_base="1.2", config_name="quest_teleop_collect", config_path="../configs/data_collect")
def quest_teleop_collect_cli(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    default_args = [
        "env_id=guided_vision/SewNeedle-3Arms-v0",    # 仿真环境 ID
        "max_steps_per_episode=500",                 # 最大步长
        "head_control=true",                          # 是否使用头显控制中间臂
        "lock_pitch=false",                           # 是否锁定中间臂 pitch 角，true 时禁用抬头低头
        "lock_roll=true",                             # 是否锁定中间臂 roll 角, true 时保持头部水平
        "save_pose_action=true",                      # 是否额外保存 Quest 映射后的末端位姿动作
        # "camera_window=false",
        # "unity_image_stream=false",
    ]

    for arg in default_args:
        arg_key = arg.split("=")[0]
        if not any(arg_key in sys_arg for sys_arg in sys.argv):
            sys.argv.append(arg)

    quest_teleop_collect_cli()
