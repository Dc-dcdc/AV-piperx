#!/home/dc/miniforge3/envs/DPPO/bin/python
from __future__ import annotations

import json
import os
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import gymnasium as gym
import hydra
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


# 打印配置参数和控制说明。
def _print_header(args) -> None:
    print("\nQuest3 -> MuJoCo three-arm teleop test")
    print("-" * 78)
    print("Config:        configs/data_collect/quest_mujoco_test.yaml")
    print(f"UDP:           {args.host}:{args.port}")
    print(f"Env:           {args.env_id}")
    mapping = "left controller -> left arm | right controller -> right arm"
    mapping += " | head -> middle arm" if args.head_control else " | middle arm fixed"
    print(f"Pipeline:      QuestReceive -> QuestControl -> PoseActionIKSolver -> env.step")
    print(f"Mapping:       {mapping}")
    print(f"Hand anchors:  {args.individual_hand_anchors}")
    print(f"Hand scale:    {args.hand_position_scale:.3f}, max delta {args.hand_max_delta:.3f} m")
    print(f"Head control:  {args.head_control}")
    if args.head_control:
        print(f"Head scale:    {args.head_position_scale:.3f}, max delta {args.head_max_delta:.3f} m")
    print(f"Middle roll:   {args.lock_roll}")
    print(f"Camera:        {args.display_camera}")
    print(f"Unity stream:  {'on' if args.unity_image_stream else 'off'}")
    if args.unity_image_stream:
        host_text = str(args.unity_image_host).strip().lower()
        if host_text == "auto":
            target = f"broadcast:{args.unity_image_port} -> auto from Quest pose packet"
        elif host_text in {"broadcast", "255.255.255.255"}:
            target = f"broadcast:{args.unity_image_port}"
        else:
            target = f"{args.unity_image_host}:{args.unity_image_port}"
        print(f"Unity target:  {target}, {args.unity_image_hz:.1f} Hz, JPEG q={args.unity_image_jpeg_quality}")
    print(f"Partial anchor:{args.allow_partial_anchor}")
    print("-" * 78)
    print("Controls:")
    print("  A/X or P: anchor valid Quest poses and start")
    print("  B/Y or R/Space: reset MuJoCo env and pause")
    print("  Controller trigger/grip: close that side gripper")
    print("  Q or Esc: quit")
    print("-" * 78)


def run(cfg: DictConfig) -> None:
    args = cfg

    if args.mujoco_gl != "auto":
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    elif not args.viewer and not args.camera_window and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    _print_header(args)

    env_obj = gym.make(
        args.env_id,
        disable_env_checker=True,
        cameras=[args.display_camera],
        episode_length=args.episode_length,
        observation_height=args.render_height,
        observation_width=args.render_width,
    )
    sim_env = env_obj.unwrapped
    obs, _ = env_obj.reset()

    physics = sim_env._physics
    ik_solver = PoseActionIKSolver(
        sim_env,
        head_control=args.head_control,
        lock_roll=args.lock_roll,
        hand_position_scale=args.hand_position_scale,
        hand_max_delta=args.hand_max_delta,
        head_position_scale=args.head_position_scale,
        head_max_delta=args.head_max_delta,
        workspace_low=args.workspace_low,
        workspace_high=args.workspace_high,
    )

    quest_control = QuestControl(
        use_head_control=args.head_control,
        use_individual_hand_anchors=args.individual_hand_anchors,
    )

    command = {"anchor": False, "reset": False, "quit": False}

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
        if args.viewer
        else nullcontext(None)
    )

    receiver = QuestReceive(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )
    image_streamer = (
        UnityImageStreamer(
            host=args.unity_image_host,
            port=args.unity_image_port,
            send_hz=args.unity_image_hz,
            jpeg_quality=args.unity_image_jpeg_quality,
            chunk_size=args.unity_image_chunk_size,
            log_interval=args.unity_image_log_interval,
        )
        if args.unity_image_stream
        else None
    )

    if args.camera_window:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, args.render_width, args.render_height)

    started = False
    latest_data: HeadsetData | None = None
    latest_feedback = None
    prev_start_button = False
    prev_reset_button = False
    last_status_t = 0.0
    step_count = 0

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

                    start_button = quest_control.should_start(latest_data)
                    reset_button = quest_control.should_reset(latest_data)
                    if start_button and not prev_start_button:
                        command["anchor"] = True
                    if reset_button and not prev_reset_button:
                        command["reset"] = True
                    prev_start_button = start_button
                    prev_reset_button = reset_button

                if command["reset"]:
                    obs, _ = env_obj.reset()
                    ik_solver.reset()
                    quest_control.reset()
                    started = False
                    latest_feedback = None
                    step_count = 0
                    command["reset"] = False
                    print("Reset MuJoCo env. Waiting for a new QuestControl anchor.")

                can_auto_anchor = latest_data is not None and (
                    ik_solver.can_anchor_from_data(latest_data, allow_partial=args.allow_partial_anchor)
                )
                if args.start_on_first_packet and can_auto_anchor and not started:
                    command["anchor"] = True

                if command["anchor"]:
                    if latest_data is None:
                        print("Cannot anchor yet: no Quest data.")
                    else:
                        active_count = ik_solver.activate_from_data(
                            latest_data,
                            require_all=not args.allow_partial_anchor,
                        )
                        if active_count > 0:
                            left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
                            quest_control.start(latest_data, middle_pose, left_pose, right_pose)
                            started = True
                    command["anchor"] = False

                if started and latest_data is not None:
                    left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
                    pose_action, latest_feedback = quest_control.run(latest_data, left_pose, right_pose, middle_pose)
                    action, active_count = ik_solver.pose2joint(pose_action, obs)
                    if active_count > 0:
                        obs, _, terminated, truncated, _ = env_obj.step(action)
                        step_count += 1

                        if terminated or truncated:
                            obs, _ = env_obj.reset()
                            ik_solver.reset()
                            quest_control.reset()
                            started = False
                            latest_feedback = None
                            step_count = 0
                            print("Episode ended. Env reset, waiting for a new QuestControl anchor.")

                if args.camera_window or image_streamer is not None:
                    frame_rgb = physics.render(
                        height=args.render_height,
                        width=args.render_width,
                        camera_id=args.display_camera,
                    )
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    if image_streamer is not None:
                        image_streamer.maybe_send_bgr(frame_bgr)

                    
                    if args.camera_window:
                        # 相机窗口显示的内容
                        status = "RUNNING" if started else "PAUSED"
                        active_text = " ".join(f"{state.name[0].upper()}:{'on' if state.active else 'off'}" for state in ik_solver.states)
                        lines = [
                            f"{status} | A/X/P anchor | B/Y/R reset | Q quit",
                            f"active: {active_text} | steps: {step_count}",
                        ]
                        if latest_feedback is not None:
                            lines.append(
                                f"sync: H={int(latest_feedback.head_out_of_sync)} L={int(latest_feedback.left_out_of_sync)} R={int(latest_feedback.right_out_of_sync)}"
                            )
                        for state in ik_solver.states:
                            label = state.name[0].upper()
                            lines.append(
                                f"{label} target: {state.target_pos[0]:+.3f} {state.target_pos[1]:+.3f} {state.target_pos[2]:+.3f}"
                            )
                        # 显示图像
                        status_frame = frame_bgr.copy()
                        y = 26
                        for line in lines:
                            cv2.putText(
                                status_frame,
                                line,
                                (14, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.58,
                                (20, 20, 20),
                                3,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                status_frame,
                                line,
                                (14, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.58,
                                (235, 245, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            y += 24
                        cv2.imshow(args.window_name, status_frame)
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
                if now - last_status_t >= args.status_hz_interval:
                    packet_state = "data" if latest_data is not None else "no-data"
                    run_state = "running" if started else "paused"
                    labels = {"left": "L", "right": "R", "middle": "M"}
                    target_summary = " ".join(
                        f"{labels[state.name]}=({state.target_pos[0]:+.3f},{state.target_pos[1]:+.3f},{state.target_pos[2]:+.3f})"
                        for state in ik_solver.states
                    )
                    print(f"[{run_state:7s}] {packet_state:9s} {target_summary} steps={step_count}")
                    last_status_t = now

                sleep_t = SIM_DT - (time.time() - loop_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    finally:
        receiver.close()
        if image_streamer is not None:
            image_streamer.close()
        env_obj.close()
        if args.camera_window:
            cv2.destroyAllWindows()

@hydra.main(version_base="1.2", config_name="quest_mujoco_test", config_path="../configs/data_collect")
def quest_mujoco_cli(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    # 命令行参数注入
    default_args = [
        "env_id=guided_vision/InsertCylinder-3Arms-v0",
        "display_camera=zed_cam_left",
        "lock_roll=True",                            # 是否锁定中间臂 roll 角
        "render_width=640",                          # 渲染图像宽度
        "render_height=480",                         # 渲染图像高度
        "unity_image_stream=true",                   # 是否向 Unity/Quest 发送渲染图像
        "unity_image_hz=25.0",                       # 图像发送帧率
    ]

    for arg in default_args:
        arg_key = arg.split("=")[0]
        if not any(arg_key in sys_arg for sys_arg in sys.argv):
            sys.argv.append(arg)

    quest_mujoco_cli()
