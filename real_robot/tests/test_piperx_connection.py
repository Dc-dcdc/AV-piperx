#!/usr/bin/env python3
"""PiperX CAN指令收发与状态反馈诊断（只读测试）。

测试目的：
    验证CAN接口、固件查询、六关节角、法兰位姿、机械臂状态反馈频率，
    并统计CAN错误和丢包。该脚本不包含使能、运动或夹爪控制指令。

运行位置：
    以下命令均在项目根目录 ``/home/dc/dc_project/AV-piper`` 执行。

首次连接或USB-CAN重连后，先激活CAN接口：
    bash /home/dc/dc_project/piperx/pyAgxArm/pyAgxArm/scripts/ubuntu/can_activate.sh \
        can_piperx 1000000

执行15秒只读测试：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_piperx_connection.py

临时延长到30秒且不保存逐帧反馈：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_piperx_connection.py \
        --duration-seconds 30 --no-save-samples

结果目录：
    ``outputs/6_real_robot_eval/piperx_connection_test/<时间戳>/``。
    ``summary.json``保存结论，``samples.jsonl``保存逐次反馈（若启用）。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("outputs/6_real_robot_eval/piperx_connection_test")


@dataclass
class FeedbackStatistics:
    """一次PiperX只读诊断的累计统计。"""

    poll_iterations: int = 0
    joint_available_polls: int = 0
    flange_available_polls: int = 0
    status_available_polls: int = 0
    joint_none_polls: int = 0
    flange_none_polls: int = 0
    status_none_polls: int = 0
    new_joint_samples: int = 0
    new_flange_samples: int = 0
    new_status_samples: int = 0
    joint_timestamp_stale_polls: int = 0
    flange_timestamp_stale_polls: int = 0
    status_timestamp_stale_polls: int = 0
    joint_timestamp_nonmonotonic: int = 0
    flange_timestamp_nonmonotonic: int = 0
    status_timestamp_nonmonotonic: int = 0
    invalid_numeric_samples: int = 0
    abnormal_status_samples: int = 0
    elapsed_seconds: float = 0.0
    achieved_poll_hz: float = 0.0
    observed_joint_sample_hz: float = 0.0
    observed_flange_sample_hz: float = 0.0
    observed_status_sample_hz: float = 0.0
    mean_reported_joint_hz: float = 0.0
    min_reported_joint_hz: float = 0.0
    max_reported_joint_hz: float = 0.0
    mean_reported_flange_hz: float = 0.0
    mean_reported_status_hz: float = 0.0
    max_joint_step_rad: float = 0.0
    max_feedback_age_ms: float = 0.0


def _load_pyagxarm():
    try:
        from pyAgxArm import (
            AgxArmFactory,
            ArmModel,
            PiperFW,
            create_agx_arm_config,
            resolve_firmware_profile,
        )
    except ImportError as exc:
        raise RuntimeError(
            "未找到pyAgxArm。请先在当前环境执行: "
            "python -m pip install -e /home/dc/dc_project/piperx/pyAgxArm"
        ) from exc
    return (
        AgxArmFactory,
        ArmModel,
        PiperFW,
        create_agx_arm_config,
        resolve_firmware_profile,
    )


def _enum_dict(value: Any) -> dict[str, Any]:
    """将SDK的状态枚举转成适合JSON保存的字典。"""

    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        numeric_value = None
    return {
        "name": getattr(value, "name", str(value)),
        "value": numeric_value,
        "display": str(value),
    }


def _status_dict(status_msg: Any) -> dict[str, Any]:
    err = status_msg.err_status
    angle_limit = [
        bool(getattr(err, f"joint_{index}_angle_limit"))
        for index in range(1, 7)
    ]
    communication_error = [
        bool(getattr(err, f"communication_status_joint_{index}"))
        for index in range(1, 7)
    ]
    return {
        "ctrl_mode": _enum_dict(status_msg.ctrl_mode),
        "arm_status": _enum_dict(status_msg.arm_status),
        "mode_feedback": _enum_dict(status_msg.mode_feedback),
        "teach_status": _enum_dict(status_msg.teach_status),
        "motion_status": _enum_dict(status_msg.motion_status),
        "trajectory_num": int(status_msg.trajectory_num),
        "err_code": int(status_msg.err_code),
        "joint_angle_limit": angle_limit,
        "joint_communication_error": communication_error,
        "repr": repr(status_msg),
    }


def _is_status_normal(status: dict[str, Any]) -> bool:
    return (
        status["arm_status"]["value"] == 0
        and status["err_code"] == 0
        and not any(status["joint_angle_limit"])
        and not any(status["joint_communication_error"])
    )


def _finite_vector(value: Any, expected_length: int) -> list[float] | None:
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(vector) != expected_length or not all(math.isfinite(item) for item in vector):
        return None
    return vector


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _observed_rate(timestamps: list[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return 0.0
    return (len(timestamps) - 1) / elapsed


def _can_interface_details(channel: str) -> dict[str, Any]:
    """只读查询SocketCAN接口状态，不修改网络配置。"""

    sysfs_path = Path("/sys/class/net") / channel
    if not sysfs_path.exists():
        raise RuntimeError(
            f"CAN接口{channel!r}不存在。请先运行can_activate.sh。"
        )
    result = subprocess.run(
        ["ip", "-details", "-statistics", "link", "show", channel],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"无法读取CAN接口{channel!r}: {result.stderr.strip()}"
        )
    text = result.stdout
    return {
        "channel": channel,
        "is_up": "state UP" in text or "<UP," in text,
        "is_lower_up": "LOWER_UP" in text,
        "error_active": "can state ERROR-ACTIVE" in text,
        "details": text.rstrip(),
    }


def _can_network_counters(channel: str) -> dict[str, int]:
    """读取Linux网络设备统计，用测试前后差值判断新增丢包。"""

    statistics_dir = Path("/sys/class/net") / channel / "statistics"
    names = (
        "rx_packets",
        "rx_bytes",
        "rx_errors",
        "rx_dropped",
        "rx_missed_errors",
        "tx_packets",
        "tx_bytes",
        "tx_errors",
        "tx_dropped",
    )
    counters: dict[str, int] = {}
    for name in names:
        path = statistics_dir / name
        try:
            counters[name] = int(path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError(f"无法读取CAN计数器{path}: {exc}") from exc
    return counters


def _counter_delta(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, int]:
    return {name: after[name] - before[name] for name in before}


def _acquire_channel_lock(channel: str):
    """防止同时启动多个本诊断脚本占用同一CAN通道。"""

    safe_channel = "".join(char if char.isalnum() else "_" for char in channel)
    lock_path = Path("/tmp") / f"av_piperx_{safe_channel}.lock"
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(
            f"另一个PiperX诊断进程正在使用{channel!r}。"
        ) from exc
    lock_file.write(f"pid={os.getpid()}\n")
    lock_file.flush()
    return lock_file


def _create_config(
    *,
    create_agx_arm_config: Any,
    robot_model: str,
    firmware_profile: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return create_agx_arm_config(
        robot=robot_model,
        firmeware_version=firmware_profile,
        interface=args.interface,
        channel=args.channel,
        bitrate=int(args.bitrate),
        enable_check_can=True,
        auto_connect=True,
        timeout=float(args.can_timeout_seconds),
        receive_own_messages=False,
        local_loopback=False,
        log_level=args.sdk_log_level,
    )


def _discover_firmware(
    *,
    factory: Any,
    create_agx_arm_config: Any,
    robot_model: str,
    default_profile: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """用默认协议发送固件查询，不使能机械臂。"""

    config = _create_config(
        create_agx_arm_config=create_agx_arm_config,
        robot_model=robot_model,
        firmware_profile=default_profile,
        args=args,
    )
    robot = factory.create_arm(config)
    try:
        robot.connect()
        deadline = time.monotonic() + args.startup_timeout_seconds
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            firmware = robot.get_firmware(
                timeout=min(1.0, remaining),
                min_interval=0.2,
            )
            if firmware is not None:
                return dict(firmware)
            time.sleep(0.05)
    finally:
        robot.disconnect()
    raise TimeoutError(
        f"{args.startup_timeout_seconds:.1f}秒内未收到PiperX固件信息。"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not args.channel:
        raise ValueError("--channel不能为空。")
    if args.bitrate <= 0:
        raise ValueError("--bitrate必须大于0。")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds必须大于0。")
    if args.poll_hz <= 0:
        raise ValueError("--poll-hz必须大于0。")
    if args.startup_timeout_seconds <= 0:
        raise ValueError("--startup-timeout-seconds必须大于0。")
    if args.feedback_warmup_seconds < 0:
        raise ValueError("--feedback-warmup-seconds不能小于0。")
    if args.can_timeout_seconds <= 0:
        raise ValueError("--can-timeout-seconds必须大于0。")
    if args.min_feedback_hz <= 0:
        raise ValueError("--min-feedback-hz必须大于0。")
    if args.report_interval_seconds <= 0:
        raise ValueError("--report-interval-seconds必须大于0。")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="测试单台PiperX的CAN固件查询和状态反馈，不使能也不运动。"
    )
    parser.add_argument("--channel", default="can_piperx")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    parser.add_argument("--poll-hz", type=float, default=100.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--feedback-warmup-seconds", type=float, default=0.5)
    parser.add_argument("--can-timeout-seconds", type=float, default=1.0)
    parser.add_argument("--min-feedback-hz", type=float, default=150.0)
    parser.add_argument("--report-interval-seconds", type=float, default=2.0)
    parser.add_argument("--sdk-log-level", default="WARNING")
    parser.add_argument(
        "--save-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存逐次新反馈到samples.jsonl。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认写入outputs/6_real_robot_eval/piperx_connection_test/时间戳。",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    (
        factory,
        ArmModel,
        PiperFW,
        create_agx_arm_config,
        resolve_firmware_profile,
    ) = _load_pyagxarm()

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_name)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lock_file = _acquire_channel_lock(args.channel)
    samples_file = None
    robot = None
    try:
        can_details = _can_interface_details(args.channel)
        if not can_details["is_up"]:
            raise RuntimeError(f"CAN接口{args.channel!r}存在，但尚未启用。")
        can_counters_before = _can_network_counters(args.channel)

        print("PiperX CAN只读通信测试")
        print("-" * 72)
        print(f"CAN channel:     {args.channel}")
        print(f"CAN interface:   {args.interface}")
        print(f"CAN bitrate:     {args.bitrate}")
        print(f"Duration:        {args.duration_seconds:.1f} s")
        print(f"Poll rate:       {args.poll_hz:.1f} Hz")
        print(f"Output:          {output_dir}")
        print("Safety:          read-only; NEVER enable or send motion commands")
        print("-" * 72)

        firmware = _discover_firmware(
            factory=factory,
            create_agx_arm_config=create_agx_arm_config,
            robot_model=ArmModel.PIPER_X,
            default_profile=PiperFW.DEFAULT,
            args=args,
        )
        software_version = str(firmware["software_version"])
        firmware_profile = resolve_firmware_profile(
            ArmModel.PIPER_X, software_version
        )
        print(
            f"Firmware:        {software_version} "
            f"(profile={firmware_profile})"
        )

        runtime_config = _create_config(
            create_agx_arm_config=create_agx_arm_config,
            robot_model=ArmModel.PIPER_X,
            firmware_profile=firmware_profile,
            args=args,
        )
        robot = factory.create_arm(runtime_config)
        robot.connect()
        # Piper关节角由3类CAN子帧组合而成。刚连接时可能只收到
        # 部分子帧，SDK尚未更新的维度会暂时保持0。先等待缓存完整，
        # 再开始正式统计，避免将启动期部分帧误判为关节突跳。
        if args.feedback_warmup_seconds > 0:
            print(
                "Feedback warmup: "
                f"{args.feedback_warmup_seconds:.2f} s "
                "(not included in statistics)"
            )
            time.sleep(args.feedback_warmup_seconds)

        if args.save_samples:
            samples_file = (output_dir / "samples.jsonl").open(
                "w", encoding="utf-8"
            )

        stats = FeedbackStatistics()
        poll_period = 1.0 / args.poll_hz
        start_monotonic = time.monotonic()
        end_monotonic = start_monotonic + args.duration_seconds
        next_poll = start_monotonic
        last_report = start_monotonic

        last_joint_timestamp: float | None = None
        last_flange_timestamp: float | None = None
        last_status_timestamp: float | None = None
        last_joint_vector: list[float] | None = None
        latest_joint: list[float] | None = None
        latest_flange: list[float] | None = None
        latest_status: dict[str, Any] | None = None
        joint_min = [math.inf] * 6
        joint_max = [-math.inf] * 6
        joint_timestamps: list[float] = []
        flange_timestamps: list[float] = []
        status_timestamps: list[float] = []
        joint_reported_hz: list[float] = []
        flange_reported_hz: list[float] = []
        status_reported_hz: list[float] = []

        while time.monotonic() < end_monotonic:
            now_monotonic = time.monotonic()
            if now_monotonic < next_poll:
                time.sleep(next_poll - now_monotonic)
            next_poll += poll_period
            stats.poll_iterations += 1
            local_time_ns = time.time_ns()

            joint_message = robot.get_joint_angles()
            flange_message = robot.get_flange_pose()
            status_message = robot.get_arm_status()

            new_joint = False
            new_flange = False
            new_status = False

            if joint_message is None:
                stats.joint_none_polls += 1
            else:
                stats.joint_available_polls += 1
                timestamp = float(joint_message.timestamp)
                if last_joint_timestamp is not None and timestamp < last_joint_timestamp:
                    stats.joint_timestamp_nonmonotonic += 1
                if last_joint_timestamp is None or timestamp > last_joint_timestamp:
                    vector = _finite_vector(joint_message.msg, 6)
                    if vector is None:
                        stats.invalid_numeric_samples += 1
                    else:
                        new_joint = True
                        stats.new_joint_samples += 1
                        latest_joint = vector
                        joint_timestamps.append(timestamp)
                        joint_reported_hz.append(float(joint_message.hz))
                        for index, value in enumerate(vector):
                            joint_min[index] = min(joint_min[index], value)
                            joint_max[index] = max(joint_max[index], value)
                        if last_joint_vector is not None:
                            step = max(
                                abs(current - previous)
                                for current, previous in zip(vector, last_joint_vector)
                            )
                            stats.max_joint_step_rad = max(
                                stats.max_joint_step_rad, step
                            )
                        last_joint_vector = vector
                        feedback_age_ms = max(
                            0.0, time.time() - timestamp
                        ) * 1000.0
                        stats.max_feedback_age_ms = max(
                            stats.max_feedback_age_ms, feedback_age_ms
                        )
                    last_joint_timestamp = timestamp
                else:
                    stats.joint_timestamp_stale_polls += 1

            if flange_message is None:
                stats.flange_none_polls += 1
            else:
                stats.flange_available_polls += 1
                timestamp = float(flange_message.timestamp)
                if last_flange_timestamp is not None and timestamp < last_flange_timestamp:
                    stats.flange_timestamp_nonmonotonic += 1
                if last_flange_timestamp is None or timestamp > last_flange_timestamp:
                    vector = _finite_vector(flange_message.msg, 6)
                    if vector is None:
                        stats.invalid_numeric_samples += 1
                    else:
                        new_flange = True
                        stats.new_flange_samples += 1
                        latest_flange = vector
                        flange_timestamps.append(timestamp)
                        flange_reported_hz.append(float(flange_message.hz))
                    last_flange_timestamp = timestamp
                else:
                    stats.flange_timestamp_stale_polls += 1

            if status_message is None:
                stats.status_none_polls += 1
            else:
                stats.status_available_polls += 1
                timestamp = float(status_message.timestamp)
                if last_status_timestamp is not None and timestamp < last_status_timestamp:
                    stats.status_timestamp_nonmonotonic += 1
                if last_status_timestamp is None or timestamp > last_status_timestamp:
                    new_status = True
                    stats.new_status_samples += 1
                    latest_status = _status_dict(status_message.msg)
                    status_timestamps.append(timestamp)
                    status_reported_hz.append(float(status_message.hz))
                    if not _is_status_normal(latest_status):
                        stats.abnormal_status_samples += 1
                    last_status_timestamp = timestamp
                else:
                    stats.status_timestamp_stale_polls += 1

            if samples_file is not None and (new_joint or new_flange or new_status):
                record = {
                    "poll_index": stats.poll_iterations - 1,
                    "local_time_ns": local_time_ns,
                    "joint_timestamp": last_joint_timestamp,
                    "joint_angles_rad": latest_joint,
                    "flange_timestamp": last_flange_timestamp,
                    "flange_pose_m_rad": latest_flange,
                    "status_timestamp": last_status_timestamp,
                    "arm_status": latest_status,
                }
                samples_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            now_monotonic = time.monotonic()
            if now_monotonic - last_report >= args.report_interval_seconds:
                elapsed = max(now_monotonic - start_monotonic, 1e-9)
                print(
                    f"polls={stats.poll_iterations}, "
                    f"poll_hz={stats.poll_iterations / elapsed:.1f}, "
                    f"joint={stats.new_joint_samples}, "
                    f"flange={stats.new_flange_samples}, "
                    f"status={stats.new_status_samples}, "
                    f"reported_joint_hz={joint_reported_hz[-1] if joint_reported_hz else 0:.1f}"
                )
                last_report = now_monotonic

        stats.elapsed_seconds = max(time.monotonic() - start_monotonic, 0.0)
        if stats.elapsed_seconds > 0:
            stats.achieved_poll_hz = stats.poll_iterations / stats.elapsed_seconds
        stats.observed_joint_sample_hz = _observed_rate(joint_timestamps)
        stats.observed_flange_sample_hz = _observed_rate(flange_timestamps)
        stats.observed_status_sample_hz = _observed_rate(status_timestamps)
        stats.mean_reported_joint_hz = _mean(joint_reported_hz)
        stats.min_reported_joint_hz = min(joint_reported_hz, default=0.0)
        stats.max_reported_joint_hz = max(joint_reported_hz, default=0.0)
        stats.mean_reported_flange_hz = _mean(flange_reported_hz)
        stats.mean_reported_status_hz = _mean(status_reported_hz)

        has_comm_error = bool(robot.has_comm_error())
        comm_error = robot.get_comm_error()
        can_counters_after = _can_network_counters(args.channel)
        can_counter_delta = _counter_delta(
            can_counters_before, can_counters_after
        )
        checks = {
            "can_interface_up": bool(can_details["is_up"]),
            "can_error_active": bool(can_details["error_active"]),
            "firmware_detected": bool(firmware),
            "joint_feedback_received": stats.new_joint_samples > 0,
            "flange_feedback_received": stats.new_flange_samples > 0,
            "status_feedback_received": stats.new_status_samples > 0,
            "reported_joint_hz_sufficient": (
                stats.mean_reported_joint_hz >= args.min_feedback_hz
            ),
            "reported_flange_hz_sufficient": (
                stats.mean_reported_flange_hz >= args.min_feedback_hz
            ),
            "reported_status_hz_sufficient": (
                stats.mean_reported_status_hz >= args.min_feedback_hz
            ),
            "timestamps_monotonic": (
                stats.joint_timestamp_nonmonotonic == 0
                and stats.flange_timestamp_nonmonotonic == 0
                and stats.status_timestamp_nonmonotonic == 0
            ),
            "numeric_feedback_valid": stats.invalid_numeric_samples == 0,
            "arm_status_normal": stats.abnormal_status_samples == 0,
            "communication_error_absent": not has_comm_error,
            "no_new_can_rx_errors": can_counter_delta["rx_errors"] == 0,
            "no_new_can_rx_drops": can_counter_delta["rx_dropped"] == 0,
            "no_new_can_rx_missed": (
                can_counter_delta["rx_missed_errors"] == 0
            ),
            "no_new_can_tx_errors": can_counter_delta["tx_errors"] == 0,
            "no_new_can_tx_drops": can_counter_delta["tx_dropped"] == 0,
        }
        passed = all(checks.values())

        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "test_type": "piperx_read_only_connection",
            "safety": {
                "read_only": True,
                "firmware_query_sent": True,
                "arm_enabled": False,
                "motion_command_sent": False,
                "effector_command_sent": False,
            },
            "configuration": {
                "robot_model": ArmModel.PIPER_X,
                "channel": args.channel,
                "interface": args.interface,
                "bitrate": int(args.bitrate),
                "duration_seconds": float(args.duration_seconds),
                "poll_hz": float(args.poll_hz),
                "feedback_warmup_seconds": float(
                    args.feedback_warmup_seconds
                ),
                "minimum_feedback_hz": float(args.min_feedback_hz),
                "save_samples": bool(args.save_samples),
            },
            "can_interface": {
                **can_details,
                "counters_before": can_counters_before,
                "counters_after": can_counters_after,
                "counter_delta": can_counter_delta,
            },
            "firmware": firmware,
            "firmware_profile": firmware_profile,
            "statistics": asdict(stats),
            "latest_feedback": {
                "joint_angles_rad": latest_joint,
                "joint_min_rad": (
                    None if latest_joint is None else joint_min
                ),
                "joint_max_rad": (
                    None if latest_joint is None else joint_max
                ),
                "flange_pose_m_rad": latest_flange,
                "arm_status": latest_status,
            },
            "communication": {
                "has_comm_error": has_comm_error,
                "comm_error": None if comm_error is None else repr(comm_error),
            },
            "checks": checks,
            "passed": passed,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("-" * 72)
        print(f"固件版本:        {software_version} ({firmware_profile})")
        print(f"关节反馈:        {stats.mean_reported_joint_hz:.1f} Hz")
        print(f"法兰反馈:        {stats.mean_reported_flange_hz:.1f} Hz")
        print(f"状态反馈:        {stats.mean_reported_status_hz:.1f} Hz")
        print(f"数值异常:        {stats.invalid_numeric_samples}")
        print(f"异常状态:        {stats.abnormal_status_samples}")
        print(f"通信错误:        {has_comm_error}")
        print(
            "CAN新增丢包:      "
            f"rx={can_counter_delta['rx_dropped']}, "
            f"tx={can_counter_delta['tx_dropped']}"
        )
        print(f"测试结果:        {'PASS' if passed else 'FAIL'}")
        print(f"统计文件:        {summary_path}")
        if args.save_samples:
            print(f"反馈样本:        {output_dir / 'samples.jsonl'}")
        return 0 if passed else 3
    finally:
        if samples_file is not None:
            samples_file.close()
        if robot is not None:
            robot.disconnect()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def main(config_defaults: dict[str, Any] | None = None) -> int:
    parser = _build_parser()
    if config_defaults:
        valid_destinations = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_defaults) - valid_destinations)
        if unknown_keys:
            raise ValueError(f"入口配置包含未知参数: {unknown_keys}")
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    try:
        return run(args)
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # =====================================================================
    # PiperX只读通信测试配置区：日常使用时直接修改下列参数。
    # 命令行显式参数仍可临时覆盖这些默认值。
    # =====================================================================
    piperx_test_config = {
        # 单台PiperX的SocketCAN接口名称。
        "channel": "can_piperx",
        # Ubuntu使用socketcan后端。
        "interface": "socketcan",
        # PiperX官方CAN模块波特率为1 Mbps。
        "bitrate": 1_000_000,
        # 连续状态反馈测试时长，单位为秒。
        "duration_seconds": 15.0,
        # Python主线程读取SDK最新缓存的频率，不是CAN底层上报频率。
        "poll_hz": 100.0,
        # 首次固件查询允许的最长等待时间。
        "startup_timeout_seconds": 5.0,
        # 重连后等待所有关节/法兰CAN子帧填充完整，不计入统计。
        "feedback_warmup_seconds": 0.5,
        # SDK CAN请求的超时时间。
        "can_timeout_seconds": 1.0,
        # 判定关节、法兰和状态反馈正常的最低SDK报告频率。
        "min_feedback_hz": 150.0,
        # 终端打印当前诊断进度的间隔。
        "report_interval_seconds": 2.0,
        # pyAgxArm内部日志级别。
        "sdk_log_level": "WARNING",
        # True保存逐次新反馈，用于检查时间戳和数值稳定性。
        "save_samples": True,
        # None表示按时间戳自动创建输出目录；也可设为Path("...")。
        "output_dir": None,
    }

    raise SystemExit(main(config_defaults=piperx_test_config))
