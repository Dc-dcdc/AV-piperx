"""Quest3 位姿映射可视化工具。

该脚本用于接收 Quest3 头显和手柄位姿，并通过 QuestControl 或原始位姿映射
转换到 MuJoCo mocap 方块上进行可视化。主要用于在真实遥操作前检查坐标系转换、
锚点逻辑和 Quest/MuJoCo 位姿对齐是否正确。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for path in (ROOT_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/av_piper_numba_cache")

from quest_control import QuestControl
from quest_receive import QuestReceive
from headset_utils import HeadsetData
from transform_utils import (
    align_rotation_to_z_axis,
    mat2pose,
    pose2mat,
    transform_coordinates,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


BODY_SOURCE_MAP = {
    "quest_head": ("head", "h_pos", "h_quat"),
    "quest_left": ("left", "l_pos", "l_quat"),
    "quest_right": ("right", "r_pos", "r_quat"),
}

QUEST_CONTROL_MOCAP_BODIES = ("quest_head", "quest_left", "quest_right")

QUEST_CONTROL_ACTION_LAYOUT = {
    "quest_left": (slice(0, 3), slice(3, 7), slice(7, 8)),
    "quest_right": (slice(8, 11), slice(11, 15), slice(15, 16)),
    "quest_head": (slice(16, 19), slice(19, 23), None),
}


class QuestPoseMapper:
    """把 QuestReceive 得到的 HeadsetData 映射到 test_quest.xml 的三个 mocap 方块。"""

    def __init__(
        self,
        use_head_frame: bool = True,
        allow_partial_anchor: bool = False,
        position_scale: float = 1.0,
        target_frame_body: str = "quest_head",
    ):
        self.use_head_frame = bool(use_head_frame)
        self.allow_partial_anchor = bool(allow_partial_anchor)
        self.position_scale = float(position_scale)
        self.target_frame_body = target_frame_body
        self.start_source_frame: np.ndarray | None = None
        self.start_target_frame: np.ndarray | None = None
        self.started = False

    def reset(self):
        self.start_source_frame = None
        self.start_target_frame = None
        self.started = False

    def is_running(self) -> bool:
        return self.started

    def start(self, data: HeadsetData, target_poses: dict[str, np.ndarray]) -> list[str]:
        if self.target_frame_body not in target_poses:
            raise RuntimeError(f"找不到目标锚定方块: {self.target_frame_body}")

        missing = [source for source in self._required_sources() if not _pose_is_usable(data, source)]
        if missing and not self.allow_partial_anchor:
            raise RuntimeError("缺少可用 Quest 位姿: " + ", ".join(missing))

        anchored = [body for body, (source, _, _) in BODY_SOURCE_MAP.items() if _pose_is_usable(data, source)]
        if not anchored:
            raise RuntimeError("head/left/right 都没有可用位姿")

        self.start_source_frame = _aligned_head_pose(data) if self.use_head_frame else np.eye(4, dtype=np.float64)
        self.start_target_frame = np.asarray(target_poses[self.target_frame_body], dtype=np.float64).copy()
        self.started = True
        return anchored

    def map_relative(self, data: HeadsetData) -> tuple[dict[str, np.ndarray], list[str]]:
        if not self.started:
            raise RuntimeError("需要先调用 start() 完成锚定。")
        assert self.start_source_frame is not None
        assert self.start_target_frame is not None

        targets: dict[str, np.ndarray] = {}
        skipped: list[str] = []
        for body_name, (source, pos_attr, quat_attr) in BODY_SOURCE_MAP.items():
            if not _pose_is_usable(data, source):
                skipped.append(body_name)
                continue

            source_pose = _pose_xyzw_to_mat(getattr(data, pos_attr), getattr(data, quat_attr))
            target_pose = transform_coordinates(source_pose, self.start_source_frame, self.start_target_frame)
            if self.position_scale != 1.0:
                target_pose = np.asarray(target_pose, dtype=np.float64).copy()
                target_pose[:3, 3] = (
                    self.start_target_frame[:3, 3]
                    + (target_pose[:3, 3] - self.start_target_frame[:3, 3]) * self.position_scale
                )
            targets[body_name] = target_pose
        return targets, skipped

    def map_absolute(self, data: HeadsetData, scale: float, offset: np.ndarray) -> tuple[dict[str, np.ndarray], list[str]]:
        targets: dict[str, np.ndarray] = {}
        skipped: list[str] = []
        for body_name, (source, pos_attr, quat_attr) in BODY_SOURCE_MAP.items():
            if not _pose_is_usable(data, source):
                skipped.append(body_name)
                continue
            pose = _pose_xyzw_to_mat(getattr(data, pos_attr), getattr(data, quat_attr))
            pose = np.asarray(pose, dtype=np.float64).copy()
            pose[:3, 3] = np.asarray(offset, dtype=np.float64) + pose[:3, 3] * float(scale)
            targets[body_name] = pose
        return targets, skipped

    def _required_sources(self) -> list[str]:
        sources = [source for source, _, _ in BODY_SOURCE_MAP.values()]
        if self.use_head_frame and "head" not in sources:
            sources.append("head")
        return sources


def _str_to_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on", "是", "开", "开启"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off", "否", "关", "关闭"}:
        return False
    raise argparse.ArgumentTypeError("布尔参数请填写 true/false、1/0、yes/no、on/off、是/否")


def _normalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat_xyzw))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat_xyzw / norm


def _pose_is_usable(data: HeadsetData, source: str) -> bool:
    pos, quat = _source_pose(data, source)
    return bool(np.linalg.norm(pos) > 1e-6 and np.linalg.norm(quat) > 1e-6)


def _source_pose(data: HeadsetData, source: str) -> tuple[np.ndarray, np.ndarray]:
    if source == "head":
        return data.h_pos, data.h_quat
    if source == "left":
        return data.l_pos, data.l_quat
    if source == "right":
        return data.r_pos, data.r_quat
    raise ValueError(f"未知 Quest 数据源: {source}")


def _pose_xyzw_to_mat(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    return pose2mat(np.asarray(pos, dtype=np.float64), _normalize_quat(quat_xyzw))


def _mocap_pose_to_mat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    return pose2mat(np.asarray(pos, dtype=np.float64), wxyz_to_xyzw(np.asarray(quat_wxyz, dtype=np.float64)))


def _mat_to_mocap_pose(pose_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos, quat_xyzw = mat2pose(np.asarray(pose_mat, dtype=np.float64))
    return np.asarray(pos, dtype=np.float64), xyzw_to_wxyz(quat_xyzw)


def _aligned_head_pose(data: HeadsetData) -> np.ndarray:
    pose = _pose_xyzw_to_mat(data.h_pos, data.h_quat)
    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, :3] = align_rotation_to_z_axis(pose[:3, :3])
    aligned[:3, 3] = pose[:3, 3]
    return aligned


def _mocap_id(model, mujoco_module, body_name: str) -> int:
    body_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"MuJoCo body not found: {body_name}")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError(f"MuJoCo body is not mocap-enabled: {body_name}")
    return mocap_id


def _current_mocap_pose_mats(mj_data, mocap_ids: dict[str, int]) -> dict[str, np.ndarray]:
    return {
        body_name: _mocap_pose_to_mat(mj_data.mocap_pos[mocap_id], mj_data.mocap_quat[mocap_id])
        for body_name, mocap_id in mocap_ids.items()
    }


def _apply_pose_mats_to_mocap(mj_data, mocap_ids: dict[str, int], pose_mats: dict[str, np.ndarray]) -> None:
    for body_name, pose_mat in pose_mats.items():
        mocap_id = mocap_ids[body_name]
        pos, quat_wxyz = _mat_to_mocap_pose(pose_mat)
        mj_data.mocap_pos[mocap_id] = pos
        mj_data.mocap_quat[mocap_id] = quat_wxyz


def _mat_to_robot_pose(pose_mat: np.ndarray) -> np.ndarray:
    pos, quat_wxyz = _mat_to_mocap_pose(pose_mat)
    return np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat_wxyz, dtype=np.float64)])


def _current_robot_poses_from_mocaps(mj_data, mocap_ids: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_mats = _current_mocap_pose_mats(mj_data, mocap_ids)
    return (
        _mat_to_robot_pose(pose_mats["quest_left"]),
        _mat_to_robot_pose(pose_mats["quest_right"]),
        _mat_to_robot_pose(pose_mats["quest_head"]),
    )


def _quest_action_to_pose_mats(action: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.size < 23:
        raise ValueError(f"QuestControl.run 返回的 action 长度不足，当前为 {action.size}，期望至少 23。")

    pose_mats: dict[str, np.ndarray] = {}
    for body_name, (pos_slice, quat_slice, _gripper_slice) in QUEST_CONTROL_ACTION_LAYOUT.items():
        pos = np.asarray(action[pos_slice], dtype=np.float64)
        quat_wxyz = _normalize_quat(np.asarray(action[quat_slice], dtype=np.float64))
        pose_mats[body_name] = pose2mat(pos, wxyz_to_xyzw(quat_wxyz))
    return pose_mats


def _quest_action_summary(action: np.ndarray) -> str:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    return (
        f"L pos={_fmt_vec(action[0:3])} quat={_fmt_vec(action[3:7])} g={float(action[7]):.3f} | "
        f"R pos={_fmt_vec(action[8:11])} quat={_fmt_vec(action[11:15])} g={float(action[15]):.3f} | "
        f"M pos={_fmt_vec(action[16:19])} quat={_fmt_vec(action[19:23])}"
    )


def _fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x): .4f}" for x in vec) + "]"

# 打印接受到的数据
def format_headset_data(data: HeadsetData, address: tuple[str, int] | None = None, output_format: str = "default") -> str:
    if output_format == "raw":
        return "raw 输出请直接运行 quest_receive.py；quest3_receiver.py 当前只显示解析后的 HeadsetData。"

    prefix = "-" * 86
    source = "-" if address is None else f"{address[0]}:{address[1]}"
    lines = [
        prefix,
        f"from={source}  recv_time={time.time():.3f}",
        f"head  pos={_fmt_vec(data.h_pos)} quat_xyzw={_fmt_vec(data.h_quat)}",
        f"left  pos={_fmt_vec(data.l_pos)} quat_xyzw={_fmt_vec(data.l_quat)} trig={data.l_index_trigger:.3f} grip={data.l_hand_trigger:.3f} stick=({data.l_thumbstick_x:.3f},{data.l_thumbstick_y:.3f}) btn1={int(data.l_button_one)} btn2={int(data.l_button_two)} stick_btn={int(data.l_button_thumbstick)}",
        f"right pos={_fmt_vec(data.r_pos)} quat_xyzw={_fmt_vec(data.r_quat)} trig={data.r_index_trigger:.3f} grip={data.r_hand_trigger:.3f} stick=({data.r_thumbstick_x:.3f},{data.r_thumbstick_y:.3f}) btn1={int(data.r_button_one)} btn2={int(data.r_button_two)} stick_btn={int(data.r_button_thumbstick)}",
    ]
    return "\n".join(lines)


def run_mujoco_viewer_loop(args) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("MuJoCo viewer mode requires the mujoco Python package.") from exc

    xml_path = Path(args.xml_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(model)
    mujoco.mj_forward(model, mj_data)

    mocap_ids = {body_name: _mocap_id(model, mujoco, body_name) for body_name in BODY_SOURCE_MAP}
    initial_mocap_pos = mj_data.mocap_pos.copy()
    initial_mocap_quat = mj_data.mocap_quat.copy()
    offset = np.asarray(args.viewer_offset, dtype=np.float64).reshape(3)

    mapper = QuestPoseMapper(
        use_head_frame=args.use_head_frame,
        allow_partial_anchor=args.allow_partial_anchor,
        position_scale=args.viewer_scale,
        target_frame_body=args.target_frame_body,
    )
    control_started = not args.start_on_button
    reset_was_pressed = False
    recv_timeout = min(float(args.timeout), max(1.0e-3, float(args.viewer_dt)))

    receiver = QuestReceive(
        host=args.host,
        port=args.port,
        timeout=recv_timeout,
    )
    try:
        print(f"Listening for Quest3 UDP JSON on {args.host}:{args.port}")
        print(f"MuJoCo viewer XML: {xml_path}")
        print("Pipeline: QuestReceive -> QuestPoseMapper -> MuJoCo mocap cubes")
        print(f"Relative pose: {args.use_relative_pose}; head frame: {args.use_head_frame}; target frame: {args.target_frame_body}")
        print("Start: A/X or first packet, depending on --start-on-button.")
        print("Reset: B/Y returns cubes to XML initial pose.")
        print("Press Ctrl+C or close the MuJoCo viewer to stop.")

        last_print = 0.0
        last_status = 0.0
        with mujoco.viewer.launch_passive(model, mj_data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = np.array([0.0, 0.0, 1.15], dtype=np.float64)
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -25.0
            try:
                while viewer.is_running():
                    try:
                        data = receiver.receive_latest_data()
                    except socket.timeout:
                        viewer.sync()
                        time.sleep(float(args.viewer_dt))
                        continue
                    except json.JSONDecodeError as exc:
                        print(f"Invalid JSON packet: {exc}")
                        viewer.sync()
                        time.sleep(float(args.viewer_dt))
                        continue

                    now = time.time()
                    reset_pressed = QuestControl.should_reset(data)
                    if reset_pressed and not reset_was_pressed:
                        mj_data.mocap_pos[:] = initial_mocap_pos
                        mj_data.mocap_quat[:] = initial_mocap_quat
                        mujoco.mj_forward(model, mj_data)
                        mapper.reset()
                        control_started = not args.start_on_button
                        print("MuJoCo reset to XML initial pose by B/Y.")
                        reset_was_pressed = reset_pressed
                        viewer.sync()
                        time.sleep(float(args.viewer_dt))
                        continue
                    reset_was_pressed = reset_pressed

                    if not control_started:
                        if args.print_hz <= 0 or now - last_print >= 1.0 / args.print_hz:
                            print(format_headset_data(data, receiver.latest_address, args.format))
                            print("Waiting for A/X to start MuJoCo visualization...")
                            last_print = now
                        if not QuestControl.should_start(data):
                            viewer.sync()
                            time.sleep(float(args.viewer_dt))
                            continue
                        control_started = True
                        mapper.reset()
                        print("MuJoCo visualization start requested by A/X.")

                    if args.use_relative_pose:
                        if not mapper.is_running():
                            try:
                                anchored = mapper.start(data, _current_mocap_pose_mats(mj_data, mocap_ids))
                                print("Relative anchors set: " + ", ".join(anchored))
                            except RuntimeError as exc:
                                if now - last_status > 1.0:
                                    print(f"Cannot anchor yet: {exc}")
                                    last_status = now
                                viewer.sync()
                                time.sleep(float(args.viewer_dt))
                                continue
                        pose_mats, skipped = mapper.map_relative(data)
                    else:
                        pose_mats, skipped = mapper.map_absolute(data, scale=args.viewer_scale, offset=offset)

                    _apply_pose_mats_to_mocap(mj_data, mocap_ids, pose_mats)
                    if skipped and now - last_status > 1.0:
                        print("Skipped unusable pose: " + ", ".join(skipped))
                        last_status = now

                    mujoco.mj_forward(model, mj_data)
                    if args.print_hz <= 0 or now - last_print >= 1.0 / args.print_hz:
                        print(format_headset_data(data, receiver.latest_address, args.format))
                        last_print = now

                    viewer.sync()
                    time.sleep(float(args.viewer_dt))
            except KeyboardInterrupt:
                print("\nStopped Quest3 MuJoCo viewer.")
    finally:
        receiver.close()


def run_quest_control_viewer_loop(args) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("QuestControl 测试模式需要安装 mujoco Python 包。") from exc

    xml_path = Path(args.xml_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(model)
    mujoco.mj_forward(model, mj_data)

    mocap_ids = {body_name: _mocap_id(model, mujoco, body_name) for body_name in QUEST_CONTROL_MOCAP_BODIES}
    initial_mocap_pos = mj_data.mocap_pos.copy()
    initial_mocap_quat = mj_data.mocap_quat.copy()
    quest_control = QuestControl(
        use_individual_hand_anchors=args.individual_hand_anchors,
    )

    control_started = not args.start_on_button
    reset_was_pressed = False
    recv_timeout = min(float(args.timeout), max(1.0e-3, float(args.viewer_dt)))

    receiver = QuestReceive(
        host=args.host,
        port=args.port,
        timeout=recv_timeout,
    )
    try:
        print(f"Listening for Quest3 UDP JSON on {args.host}:{args.port}")
        print(f"MuJoCo viewer XML: {xml_path}")
        print("Pipeline: QuestReceive -> QuestControl.run -> MuJoCo mocap cubes")
        print("QuestControl mode: three mocap cubes directly visualize QuestControl.run() output.")
        print(f"Individual hand anchors: {args.individual_hand_anchors}")
        print("Start: A/X or first packet, depending on --start-on-button.")
        print("Reset: B/Y returns cubes to XML initial pose.")
        print("Press Ctrl+C or close the MuJoCo viewer to stop.")

        last_print = 0.0
        with mujoco.viewer.launch_passive(model, mj_data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = np.array([0.0, 0.0, 1.15], dtype=np.float64)
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -25.0
            try:
                while viewer.is_running():
                    try:
                        data = receiver.receive_latest_data()
                    except socket.timeout:
                        viewer.sync()
                        time.sleep(float(args.viewer_dt))
                        continue
                    except json.JSONDecodeError as exc:
                        print(f"Invalid JSON packet: {exc}")
                        viewer.sync()
                        time.sleep(float(args.viewer_dt))
                        continue

                    now = time.time()
                    reset_pressed = QuestControl.should_reset(data)
                    if reset_pressed and not reset_was_pressed:
                        mj_data.mocap_pos[:] = initial_mocap_pos
                        mj_data.mocap_quat[:] = initial_mocap_quat
                        mujoco.mj_forward(model, mj_data)
                        quest_control.reset()
                        control_started = not args.start_on_button
                        print("MuJoCo reset to XML initial pose by B/Y.")
                        reset_was_pressed = reset_pressed
                        viewer.sync()
                        time.sleep(float(args.viewer_dt))
                        continue
                    reset_was_pressed = reset_pressed

                    if not control_started:
                        if args.print_hz <= 0 or now - last_print >= 1.0 / args.print_hz:
                            print(format_headset_data(data, receiver.latest_address, args.format))
                            print("Waiting for A/X to start QuestControl testing...")
                            last_print = now
                        if not QuestControl.should_start(data):
                            viewer.sync()
                            time.sleep(float(args.viewer_dt))
                            continue
                        control_started = True
                        print("QuestControl test start requested by A/X.")

                    if not quest_control.is_running():
                        left_pose, right_pose, middle_pose = _current_robot_poses_from_mocaps(mj_data, mocap_ids)
                        quest_control.start(data, middle_pose, left_pose, right_pose)
                        print("QuestControl anchor set from current middle cube pose.")

                    left_pose, right_pose, middle_pose = _current_robot_poses_from_mocaps(mj_data, mocap_ids)
                    pose_action, latest_feedback = quest_control.run(data, left_pose, right_pose, middle_pose)
                    pose_mats = _quest_action_to_pose_mats(pose_action)
                    _apply_pose_mats_to_mocap(mj_data, mocap_ids, pose_mats)
                    mujoco.mj_forward(model, mj_data)

                    if args.print_hz <= 0 or now - last_print >= 1.0 / args.print_hz:
                        print(format_headset_data(data, receiver.latest_address, args.format))
                        print(
                            "QuestControl action -> "
                            f"{_quest_action_summary(pose_action)} | "
                            f"sync(head/left/right)="
                            f"{int(not latest_feedback.head_out_of_sync)}/"
                            f"{int(not latest_feedback.left_out_of_sync)}/"
                            f"{int(not latest_feedback.right_out_of_sync)}"
                        )
                        last_print = now

                    viewer.sync()
                    time.sleep(float(args.viewer_dt))
            except KeyboardInterrupt:
                print("\nStopped QuestControl MuJoCo viewer.")
    finally:
        receiver.close()


def run_print_loop(args) -> None:
    receiver = QuestReceive(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )
    try:
        print(f"Listening for Quest3 UDP JSON on {args.host}:{args.port}")
        print("Pipeline: QuestReceive -> terminal print")
        print("Press Ctrl+C to stop.")
        last_print = 0.0
        try:
            while True:
                try:
                    data = receiver.receive_latest_data()
                except socket.timeout:
                    print(f"No packet received within {args.timeout:.1f}s...")
                    continue
                except json.JSONDecodeError as exc:
                    print(f"Invalid JSON packet: {exc}")
                    continue

                now = time.time()
                if args.print_hz <= 0 or now - last_print >= 1.0 / args.print_hz:
                    print(format_headset_data(data, receiver.latest_address, args.format))
                    last_print = now
        except KeyboardInterrupt:
            print("\nStopped Quest3 receiver.")
    finally:
        receiver.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过 QuestReceive 接收 Meta Quest 3 位姿；可直接测试 QuestControl.run() 输出或查看原始 Quest 位姿。")
    parser.add_argument("--host", default="0.0.0.0", help="UDP 监听地址；0.0.0.0 表示监听电脑所有网卡。")
    parser.add_argument("--port", type=int, default=5005, help="UDP 监听端口，需要和 Unity 发送端 receiverPort 一致。")
    parser.add_argument("--timeout", type=float, default=1.0, help="等待 UDP 数据的超时时间，单位秒。")
    parser.add_argument("--print-hz", type=float, default=30.0, help="终端最大打印频率；填 0 表示每个包都打印。")
    parser.add_argument("--format", choices=("default", "raw", "both"), default="default", help="终端输出格式；raw/both 仅兼容保留，当前显示解析后的 HeadsetData。")
    parser.add_argument("--use-relative-pose", type=_str_to_bool, default=True, help="是否使用头显锚定后的相对位姿映射，填写 true/false。")
    parser.add_argument("--use-head-frame", type=_str_to_bool, default=True, help="是否用开始时的头显 yaw 作为控制坐标系，填写 true/false。")
    parser.add_argument("--allow-partial-anchor", type=_str_to_bool, default=False, help="是否允许只有部分 Quest 位姿有效时开始锚定，填写 true/false。")
    parser.add_argument("--target-frame-body", choices=tuple(BODY_SOURCE_MAP.keys()), default="quest_head", help="相对映射使用的 MuJoCo 目标坐标系。")
    parser.add_argument("--individual-hand-anchors", type=_str_to_bool, default=True, help="quest-control 模式中左右臂是否使用各自手柄初始位姿作为锚点，填写 true/false。")
    parser.add_argument("--start-on-button", type=_str_to_bool, default=True, help="是否等待 A/X 后才开始驱动 MuJoCo，填写 true/false。")
    parser.add_argument("--viewer", type=_str_to_bool, default=True, help="是否打开 MuJoCo viewer，填写 true/false。")
    parser.add_argument("--mode", choices=("quest-control", "raw-pose"), default="quest-control", help="选择测试模式：quest-control 直接测试 QuestControl.run() 输出，raw-pose 只看原始 Quest 位姿。")
    parser.add_argument("--xml-path", default=str(ROOT_DIR / "env" / "assets" / "test_quest.xml"), help="MuJoCo 可视化 XML 文件路径。")
    parser.add_argument("--viewer-scale", type=float, default=1.0, help="相对映射时缩放位置变化量；绝对映射时缩放收到的位置。")
    parser.add_argument("--viewer-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"), help="绝对映射时的位置偏移，单位米。")
    parser.add_argument("--viewer-dt", type=float, default=1.0 / 60.0, help="MuJoCo viewer 循环间隔，单位秒。")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.mode == "quest-control":
        run_quest_control_viewer_loop(args) if args.viewer else run_print_loop(args)
    else:
        run_mujoco_viewer_loop(args) if args.viewer else run_print_loop(args)
