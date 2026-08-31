#!/usr/bin/env python3
"""PiperX AGX夹爪回零、使能与初始化宽度一体化测试。

测试目的：
    读取AGX夹爪通信、反馈、故障、标定记录与使能状态；必要时在人工确认
    后完成零点标定，并以低力分段张开至指定初始化宽度。

运行位置：
    以下命令均在项目根目录 ``/home/dc/dc_project/AV-piper`` 执行。

第一步，仅执行只读预检（默认不标定、不使能、不运动）：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/setup_piperx_gripper.py

第二步，允许交互式初始化；程序仍会要求在终端输入现场确认词：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/setup_piperx_gripper.py --allow-setup

仅在夹爪机械安装、零点或传动结构发生变化后重新标定：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/setup_piperx_gripper.py \
        --allow-setup --force-recalibration

启用 ``allow_setup`` 后：

1. 已回零时跳过标定；
2. 未回零时要求人工确认夹爪空载且已经完全闭合，再将当前位置设为零点；
3. 以低力、分段小行程张开到初始化宽度，并检查反馈、故障和机械臂漂移。

安全说明：
    标定前必须确保夹爪内无物体、手指和线缆，并按提示将夹爪轻柔闭合。
    脚本不会在缺少现场确认时运动，也不会自动复位夹爪故障。结果保存在
    ``outputs/6_real_robot_eval/piperx_gripper_setup/<时间戳>/``。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from test_piperx_connection import (
    _acquire_channel_lock,
    _can_interface_details,
    _can_network_counters,
    _counter_delta,
    _create_config,
    _discover_firmware,
    _finite_vector,
    _load_pyagxarm,
    _status_dict,
)
from test_piperx_motion import (
    _append_event,
    _read_preflight,
    _validate_preflight,
    _write_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/6_real_robot_eval/piperx_gripper_setup"
DEFAULT_CALIBRATION_RECORD = DEFAULT_OUTPUT_ROOT / "calibration_record.json"
CALIBRATION_RECORD_SCHEMA_VERSION = 1
CALIBRATION_CONFIRMATION = "I_CONFIRM_GRIPPER_IS_EMPTY_AND_FULLY_CLOSED"
MOTION_CONFIRMATION = "I_CONFIRM_GRIPPER_IS_EMPTY_AND_MAY_MOVE"
DISABLE_CONFIRMATION = "I_CONFIRM_GRIPPER_MAY_BE_RELEASED"

_FOC_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "sensor_status",
    "driver_error_status",
    "driver_enable_status",
    "homing_status",
)
_FAULT_FIELDS = _FOC_FIELDS[:6]


def _gripper_status_dict(message: Any) -> dict[str, Any]:
    value = float(message.value)
    force = float(message.force)
    if not math.isfinite(value) or not math.isfinite(force):
        raise RuntimeError("夹爪反馈包含NaN或Inf。")
    foc = message.foc_status
    return {
        "value": value,
        "force_n": force,
        "mode": str(message.mode),
        "status_code": int(message.status_code),
        "foc_status": {
            field: bool(getattr(foc, field)) for field in _FOC_FIELDS
        },
        "repr": repr(message),
    }


def _gripper_parameters_dict(message: Any) -> dict[str, Any]:
    max_range = message.max_range_config
    return {
        "teaching_range_per": (
            None
            if message.teaching_range_per is None
            else int(message.teaching_range_per)
        ),
        "max_range_config_m": (
            None if max_range is None else float(max_range)
        ),
        "teaching_friction": (
            None
            if message.teaching_friction is None
            else int(message.teaching_friction)
        ),
        "repr": repr(message),
    }


def _fault_names(status: dict[str, Any]) -> list[str]:
    foc = status["foc_status"]
    return [name for name in _FAULT_FIELDS if foc[name]]


def _wait_initial_gripper_feedback(
    gripper: Any,
    timeout_seconds: float,
) -> tuple[Any, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        message = gripper.get_gripper_status()
        if message is not None:
            return message, _gripper_status_dict(message.msg)
        time.sleep(0.02)
    raise TimeoutError(
        f"{timeout_seconds:.1f}秒内未收到AGX夹爪反馈；"
        "请检查夹爪供电、CAN接线和型号。"
    )


def _build_width_ramp(
    current_width_m: float,
    target_width_m: float,
    max_step_m: float,
) -> list[float]:
    """构造相邻变化不超过max_step_m且包含最终目标的宽度序列。"""

    delta = target_width_m - current_width_m
    if math.isclose(delta, 0.0, abs_tol=1e-9):
        return [target_width_m]
    step_count = max(1, math.ceil(abs(delta) / max_step_m))
    return [
        current_width_m + delta * index / step_count
        for index in range(1, step_count + 1)
    ]


def _wait_for_gripper_state(
    gripper: Any,
    *,
    timeout_seconds: float,
    require_homed: bool | None = None,
    require_enabled: bool | None = None,
) -> dict[str, Any]:
    """等待夹爪反馈达到指定回零/使能状态。"""

    deadline = time.monotonic() + timeout_seconds
    latest: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        message = gripper.get_gripper_status()
        if message is None:
            time.sleep(0.02)
            continue
        latest = _gripper_status_dict(message.msg)
        foc = latest["foc_status"]
        homed_ok = require_homed is None or foc["homing_status"] == require_homed
        enabled_ok = (
            require_enabled is None
            or foc["driver_enable_status"] == require_enabled
        )
        if homed_ok and enabled_ok:
            return latest
        time.sleep(0.02)
    raise TimeoutError(
        f"{timeout_seconds:.1f}秒内夹爪状态未达到要求："
        f"homed={require_homed}, enabled={require_enabled}；最后反馈={latest}。"
    )


def _wait_for_arm_p_control_mode(
    robot: Any,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """等待机械臂进入CAN指令控制和MOVE P模式。"""

    deadline = time.monotonic() + timeout_seconds
    latest: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        message = robot.get_arm_status()
        if message is not None:
            latest = _status_dict(message.msg)
            if (
                latest["ctrl_mode"]["value"] == 1
                and latest["mode_feedback"]["value"] == 0
            ):
                return latest
        time.sleep(0.02)
    raise TimeoutError(
        f"{timeout_seconds:.1f}秒内机械臂未进入CAN控制/MOVE P模式；"
        f"最后状态={latest}。"
    )


def _require_terminal_confirmation(prompt: str, expected: str) -> None:
    """要求操作者在当前终端现场输入完整确认词。"""

    if not sys.stdin.isatty():
        raise RuntimeError("当前不是交互式终端，拒绝执行夹爪标定或运动。")
    print(prompt, flush=True)
    entered = input(f"请输入 {expected}：\n> ").strip()
    if entered != expected:
        raise RuntimeError("现场确认词不正确，未执行后续硬件操作。")


def _device_identity(
    firmware: dict[str, Any],
    firmware_profile: str,
) -> dict[str, str]:
    """提取用于约束本地夹爪标定记录的机械臂身份。"""

    return {
        "hardware_version": str(firmware.get("hardware_version", "")),
        "software_version": str(firmware.get("software_version", "")),
        "production_date": str(firmware.get("production_date", "")),
        "motor_ratio_and_batch": str(
            firmware.get("motor_ratio_and_batch", "")
        ),
        "node_type": str(firmware.get("node_type", "")),
        "node_number": str(firmware.get("node_number", "")),
        "firmware_profile": str(firmware_profile),
    }


def _load_matching_calibration_record(
    path: Path,
    device_identity: dict[str, str],
) -> tuple[dict[str, Any] | None, str]:
    """读取与当前机械臂完全匹配的本地标定记录。"""

    if not path.is_file():
        return None, "missing"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid: {exc}"
    if record.get("schema_version") != CALIBRATION_RECORD_SCHEMA_VERSION:
        return None, "unsupported_schema"
    if record.get("device_identity") != device_identity:
        return None, "device_mismatch"
    if record.get("calibration_acknowledged") is not True:
        return None, "ack_missing"
    return record, "matched"


def _write_calibration_record(path: Path, record: dict[str, Any]) -> None:
    """原子保存夹爪标定记录，供后续运行跳过重复置零。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _wait_for_zero_feedback(
    gripper: Any,
    *,
    newer_than_timestamp: float,
    tolerance_m: float,
    stable_feedback_count: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """用标定命令之后的连续零宽度反馈确认零点已生效。"""

    deadline = time.monotonic() + timeout_seconds
    stable_count = 0
    last_timestamp = newer_than_timestamp
    latest: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        message = gripper.get_gripper_status()
        if message is None or float(message.timestamp) <= last_timestamp:
            time.sleep(0.02)
            continue
        last_timestamp = float(message.timestamp)
        latest = _gripper_status_dict(message.msg)
        if _fault_names(latest):
            raise RuntimeError(f"标定后夹爪出现故障: {_fault_names(latest)}")
        if latest["mode"] != "width":
            raise RuntimeError("标定后夹爪反馈不再是width模式。")
        if latest["foc_status"]["driver_enable_status"]:
            raise RuntimeError("标定验证期间夹爪意外处于使能状态。")
        stable_count = (
            stable_count + 1
            if abs(float(latest["value"])) <= tolerance_m
            else 0
        )
        if stable_count >= stable_feedback_count:
            return latest
        time.sleep(0.02)
    raise TimeoutError(
        f"{timeout_seconds:.1f}秒内未获得连续{stable_feedback_count}帧"
        f"零宽度反馈；最后反馈={latest}。"
    )


def _collect_read_only_feedback(
    gripper: Any,
    *,
    duration_seconds: float,
    poll_hz: float,
    samples_file: Any,
) -> dict[str, Any]:
    period = 1.0 / poll_hz
    deadline = time.monotonic() + duration_seconds
    last_timestamp: float | None = None
    timestamps: list[float] = []
    reported_hz: list[float] = []
    latest: dict[str, Any] | None = None
    values: list[float] = []
    stale_polls = 0
    nonmonotonic = 0
    poll_count = 0

    while time.monotonic() < deadline:
        loop_start = time.monotonic()
        poll_count += 1
        message = gripper.get_gripper_status()
        if message is not None:
            timestamp = float(message.timestamp)
            if last_timestamp is not None and timestamp < last_timestamp:
                nonmonotonic += 1
            if last_timestamp is None or timestamp > last_timestamp:
                latest = _gripper_status_dict(message.msg)
                timestamps.append(timestamp)
                reported_hz.append(float(message.hz))
                values.append(float(latest["value"]))
                last_timestamp = timestamp
                if samples_file is not None:
                    samples_file.write(
                        json.dumps(
                            {
                                "phase": "read_only",
                                "local_time_ns": time.time_ns(),
                                "feedback_timestamp": timestamp,
                                "reported_hz": float(message.hz),
                                "gripper_status": latest,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            else:
                stale_polls += 1
        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))

    observed_hz = 0.0
    if len(timestamps) >= 2 and timestamps[-1] > timestamps[0]:
        observed_hz = (len(timestamps) - 1) / (
            timestamps[-1] - timestamps[0]
        )
    return {
        "poll_count": poll_count,
        "new_feedback_count": len(timestamps),
        "stale_poll_count": stale_polls,
        "timestamp_nonmonotonic_count": nonmonotonic,
        "observed_feedback_hz": observed_hz,
        "mean_reported_feedback_hz": (
            sum(reported_hz) / len(reported_hz) if reported_hz else 0.0
        ),
        "min_value": min(values) if values else None,
        "max_value": max(values) if values else None,
        "latest_status": latest,
    }


def _wait_for_width(
    gripper: Any,
    robot: Any,
    *,
    phase: str,
    target_width_m: float,
    arm_reference_rad: list[float],
    force_n: float,
    tolerance_m: float,
    max_arm_joint_drift_rad: float,
    stable_feedback_count: int,
    timeout_seconds: float,
    poll_hz: float,
    trajectory_file: Any,
) -> dict[str, Any]:
    period = 1.0 / poll_hz
    deadline = time.monotonic() + timeout_seconds
    stable_count = 0
    last_timestamp: float | None = None
    latest: dict[str, Any] | None = None
    latest_error: float | None = None
    max_error = 0.0
    max_arm_drift = 0.0
    samples = 0

    while time.monotonic() < deadline:
        loop_start = time.monotonic()
        message = gripper.get_gripper_status()
        joint_message = robot.get_joint_angles()
        if message is None or joint_message is None:
            time.sleep(period)
            continue

        timestamp = float(message.timestamp)
        if last_timestamp is not None and timestamp <= last_timestamp:
            time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
            continue
        last_timestamp = timestamp
        latest = _gripper_status_dict(message.msg)
        if latest["mode"] != "width":
            raise RuntimeError(
                f"{phase}期间夹爪反馈模式变为{latest['mode']!r}，拒绝继续。"
            )
        faults = _fault_names(latest)
        if faults:
            raise RuntimeError(f"{phase}期间夹爪故障: {faults}")

        arm_joints = _finite_vector(joint_message.msg, 6)
        if arm_joints is None:
            raise RuntimeError(f"{phase}期间机械臂关节反馈无效。")
        arm_drift = max(
            abs(current - reference)
            for current, reference in zip(arm_joints, arm_reference_rad)
        )
        max_arm_drift = max(max_arm_drift, arm_drift)
        if arm_drift > max_arm_joint_drift_rad:
            raise RuntimeError(
                f"{phase}期间机械臂关节漂移{arm_drift:.6f} rad超过上限"
                f"{max_arm_joint_drift_rad:.6f} rad。"
            )

        latest_error = abs(float(latest["value"]) - target_width_m)
        max_error = max(max_error, latest_error)
        stable_count = stable_count + 1 if latest_error <= tolerance_m else 0
        samples += 1
        trajectory_file.write(
            json.dumps(
                {
                    "phase": phase,
                    "local_time_ns": time.time_ns(),
                    "feedback_timestamp": timestamp,
                    "target_width_m": target_width_m,
                    "command_force_n": force_n,
                    "width_error_m": latest_error,
                    "arm_joint_angles_rad": arm_joints,
                    "arm_max_drift_rad": arm_drift,
                    "gripper_status": latest,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        trajectory_file.flush()

        if stable_count >= stable_feedback_count:
            return {
                "reached": True,
                "target_width_m": target_width_m,
                "final_width_m": float(latest["value"]),
                "final_error_m": latest_error,
                "max_error_m": max_error,
                "max_arm_joint_drift_rad": max_arm_drift,
                "samples": samples,
                "final_status": latest,
            }
        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))

    raise TimeoutError(
        f"{phase}未在{timeout_seconds:.1f}秒内稳定到达"
        f"{target_width_m:.4f} m；最后反馈={latest}，误差={latest_error}。"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PiperX AGX夹爪回零、使能和初始化宽度一体化向导；默认仅预检。"
    )
    parser.add_argument("--channel", default="can_piperx")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--startup-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--feedback-warmup-seconds", type=float, default=0.5)
    parser.add_argument("--can-timeout-seconds", type=float, default=1.0)
    parser.add_argument("--sdk-log-level", default="WARNING")
    parser.add_argument("--read-duration-seconds", type=float, default=2.0)
    parser.add_argument("--poll-hz", type=float, default=100.0)
    parser.add_argument("--min-feedback-hz", type=float, default=50.0)
    parser.add_argument("--target-width-m", type=float, default=0.030)
    parser.add_argument("--force-n", type=float, default=0.5)
    parser.add_argument("--hard-max-width-m", type=float, default=0.070)
    parser.add_argument("--max-width-step-m", type=float, default=0.010)
    parser.add_argument("--target-tolerance-m", type=float, default=0.002)
    parser.add_argument("--max-arm-joint-drift-rad", type=float, default=0.010)
    parser.add_argument("--stable-feedback-count", type=int, default=5)
    parser.add_argument("--motion-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--calibration-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--state-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--step-hold-seconds", type=float, default=0.25)
    parser.add_argument("--countdown-seconds", type=int, default=5)
    parser.add_argument(
        "--calibration-record",
        type=Path,
        default=None,
        help="本地标定记录路径；默认保存在夹爪初始化输出根目录。",
    )
    parser.add_argument(
        "--force-recalibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="忽略匹配的本地记录并重新标定；仅在机械结构变化后使用。",
    )
    parser.add_argument(
        "--allow-setup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="允许现场交互式回零、使能和真实运动；默认只读。",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.channel:
        raise ValueError("--channel不能为空。")
    if args.bitrate <= 0:
        raise ValueError("--bitrate必须大于0。")
    for name in (
        "startup_timeout_seconds",
        "can_timeout_seconds",
        "read_duration_seconds",
        "poll_hz",
        "min_feedback_hz",
        "hard_max_width_m",
        "max_width_step_m",
        "target_tolerance_m",
        "max_arm_joint_drift_rad",
        "motion_timeout_seconds",
        "calibration_timeout_seconds",
        "state_timeout_seconds",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')}必须大于0。")
    if args.feedback_warmup_seconds < 0 or args.step_hold_seconds < 0:
        raise ValueError("反馈预热和保持时间不能小于0。")
    if not 0.1 <= args.force_n <= 1.0:
        raise ValueError("安全初始化限制：--force-n必须在0.1到1.0 N之间。")
    if args.hard_max_width_m > 0.10:
        raise ValueError("安全初始化限制：硬行程上限不能超过0.10 m。")
    if not 0.0 < args.target_width_m <= args.hard_max_width_m:
        raise ValueError("--target-width-m必须大于0且不超过硬行程上限。")
    if args.max_width_step_m > 0.020:
        raise ValueError("安全初始化限制：分段宽度变化不能超过0.020 m。")
    if args.stable_feedback_count <= 0:
        raise ValueError("--stable-feedback-count必须大于0。")
    if args.countdown_seconds < 0:
        raise ValueError("--countdown-seconds不能小于0。")


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
    robot = None
    gripper = None
    samples_file = None
    trajectory_file = None
    motion_started = False
    motion_completed = False
    latest_status: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_type": "piperx_agx_gripper_setup",
        "configuration": vars(args).copy(),
        "events": events,
        "passed": False,
    }
    for path_key in ("output_dir", "calibration_record"):
        if isinstance(summary["configuration"].get(path_key), Path):
            summary["configuration"][path_key] = str(
                summary["configuration"][path_key]
            )

    try:
        print("PiperX AGX夹爪一体化初始化：正在预检……", flush=True)
        can_details = _can_interface_details(args.channel)
        if not can_details["is_up"]:
            raise RuntimeError(f"CAN接口{args.channel!r}尚未启用。")
        counters_before = _can_network_counters(args.channel)

        firmware = _discover_firmware(
            factory=factory,
            create_agx_arm_config=create_agx_arm_config,
            robot_model=ArmModel.PIPER_X,
            default_profile=PiperFW.DEFAULT,
            args=args,
        )
        profile = resolve_firmware_profile(
            ArmModel.PIPER_X, str(firmware["software_version"])
        )
        identity = _device_identity(firmware, profile)
        calibration_record_path = (
            DEFAULT_CALIBRATION_RECORD
            if args.calibration_record is None
            else Path(args.calibration_record).expanduser().resolve()
        )
        calibration_record, calibration_record_status = (
            _load_matching_calibration_record(
                calibration_record_path,
                identity,
            )
        )
        runtime_config = _create_config(
            create_agx_arm_config=create_agx_arm_config,
            robot_model=ArmModel.PIPER_X,
            firmware_profile=profile,
            args=args,
        )
        robot = factory.create_arm(runtime_config)
        # 必须在connect前初始化末端执行器，让读线程注册夹爪反馈解析器。
        gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        robot.connect()
        time.sleep(args.feedback_warmup_seconds)

        preflight = _read_preflight(robot, args.startup_timeout_seconds)
        _validate_preflight(preflight, min_feedback_hz=150.0)
        _, initial_status = _wait_initial_gripper_feedback(
            gripper, args.startup_timeout_seconds
        )
        samples_file = (output_dir / "samples.jsonl").open(
            "w", encoding="utf-8"
        )
        read_result = _collect_read_only_feedback(
            gripper,
            duration_seconds=args.read_duration_seconds,
            poll_hz=args.poll_hz,
            samples_file=samples_file,
        )
        latest_status = read_result["latest_status"] or initial_status
        params_message = gripper.get_gripper_teaching_pendant_param(
            timeout=args.can_timeout_seconds,
            min_interval=0.0,
        )
        parameters = (
            None
            if params_message is None
            else _gripper_parameters_dict(params_message.msg)
        )
        reported_max = (
            None if parameters is None else parameters["max_range_config_m"]
        )
        effective_max = args.hard_max_width_m
        if reported_max is not None and reported_max > 0:
            effective_max = min(effective_max, reported_max)

        faults = _fault_names(latest_status)
        communication_ok = bool(gripper.is_ok())
        mean_feedback_hz = float(read_result["mean_reported_feedback_hz"])
        controller_homing_signal = bool(
            latest_status["foc_status"]["homing_status"]
        )
        if args.force_recalibration:
            calibration_known = False
            calibration_source = "forced_recalibration"
        elif calibration_record is not None:
            calibration_known = True
            calibration_source = "matching_local_record"
        elif controller_homing_signal:
            calibration_known = True
            calibration_source = "controller_homing_signal"
        else:
            calibration_known = False
            calibration_source = "not_verified"
        setup_rejections: list[str] = []
        if faults:
            setup_rejections.append(f"夹爪存在故障标志: {faults}")
        if latest_status["mode"] != "width":
            setup_rejections.append(
                f"当前夹爪模式为{latest_status['mode']!r}，不是width。"
            )
        if not communication_ok:
            setup_rejections.append("gripper.is_ok()为False。")
        if mean_feedback_hz < args.min_feedback_hz:
            setup_rejections.append(
                f"夹爪反馈频率{mean_feedback_hz:.1f} Hz低于"
                f"{args.min_feedback_hz:.1f} Hz。"
            )
        if args.target_width_m > effective_max:
            setup_rejections.append(
                f"初始化宽度{args.target_width_m:.4f} m超过有效行程"
                f"[0, {effective_max:.4f}] m。"
            )
        if calibration_known:
            current_width = float(latest_status["value"])
            if not 0.0 <= current_width <= effective_max:
                setup_rejections.append(
                    f"已回零夹爪的当前宽度{current_width:.4f} m不在安全行程"
                    f"[0, {effective_max:.4f}] m内，拒绝自动移动。"
                )

        summary.update(
            {
                "firmware": firmware,
                "firmware_profile": profile,
                "can_interface": can_details,
                "arm_preflight": preflight,
                "initial_gripper_status": initial_status,
                "read_only_result": read_result,
                "gripper_parameters": parameters,
                "effective_max_width_m": effective_max,
                "controller_homing_signal": controller_homing_signal,
                "calibration_record_path": str(calibration_record_path),
                "calibration_record_status": calibration_record_status,
                "calibration_record": calibration_record,
                "calibration_known": calibration_known,
                "calibration_source": calibration_source,
                "setup_rejections": setup_rejections,
            }
        )
        _append_event(events, "read_only_preflight_completed")

        print("-" * 72)
        print(f"Firmware:        {firmware['software_version']} ({profile})")
        print(f"CAN channel:     {args.channel}")
        print(f"Communication:   {communication_ok}")
        print(f"Feedback:        {mean_feedback_hz:.1f} Hz")
        print(f"Mode:            {latest_status['mode']}")
        print(f"Current value:   {latest_status['value']:.6f}")
        print(f"Current force:   {latest_status['force_n']:.3f} N")
        print(f"Homing signal:   {controller_homing_signal}")
        print(f"Calibration:     {calibration_source}")
        print(f"Record:          {calibration_record_status}")
        print(f"Enabled:         {latest_status['foc_status']['driver_enable_status']}")
        print(f"Faults:          {faults or 'none'}")
        print(f"Reported range:  {reported_max}")
        print(f"Effective range: 0.0 .. {effective_max:.4f} m")
        print(f"Target width:    {args.target_width_m:.4f} m")
        print(f"Force:           {args.force_n:.2f} N")
        for rejection in setup_rejections:
            print(f"Setup reject:    {rejection}")
        print(f"Output:          {output_dir}")
        print("Safety:          标定与运动均需当前终端现场确认")
        print("-" * 72)

        if not args.allow_setup:
            checks = {
                "communication_ok": communication_ok,
                "feedback_received": read_result["new_feedback_count"] > 0,
                "feedback_rate_sufficient": (
                    mean_feedback_hz >= args.min_feedback_hz
                ),
                "timestamps_monotonic": (
                    read_result["timestamp_nonmonotonic_count"] == 0
                ),
                "fault_free": not faults,
            }
            summary.update(
                {
                    "mode": "read_only_preflight",
                    "checks": checks,
                    "passed": all(checks.values()),
                    "ready_for_setup": not setup_rejections,
                    "message": "只读预检完成，未发送失能、标定或运动指令。",
                }
            )
            path = _write_summary(output_dir, summary)
            print(
                f"只读结果:        {'PASS' if summary['passed'] else 'FAIL'}"
            )
            print(
                "初始化准备:      "
                f"{'READY' if not setup_rejections else 'NOT READY'}"
            )
            print("硬件操作:        未执行")
            print(f"统计文件:        {path}")
            return 0 if summary["passed"] else 2

        if setup_rejections:
            summary.update(
                {
                    "mode": "setup_preflight_rejected",
                    "message": "初始化预检未通过，未发送失能、标定或运动指令。",
                }
            )
            path = _write_summary(output_dir, summary)
            print("初始化预检:      REJECTED（未操作硬件）")
            print(f"统计文件:        {path}")
            return 2

        calibrated_this_run = False
        status_before_motion = latest_status
        if not calibration_known:
            if latest_status["foc_status"]["driver_enable_status"]:
                _require_terminal_confirmation(
                    "夹爪当前已使能。继续会先失能，夹爪可能因外力发生位移；"
                    "请托住机构并确保释放安全。",
                    DISABLE_CONFIRMATION,
                )
            disable_result = bool(gripper.disable_gripper())
            _append_event(
                events,
                "gripper_disable_requested",
                sdk_result=disable_result,
            )
            _wait_for_gripper_state(
                gripper,
                timeout_seconds=args.state_timeout_seconds,
                require_enabled=False,
            )
            _require_terminal_confirmation(
                "夹爪尚未回零。请确认夹爪内无物体、手指和线缆，"
                "并在失能状态下用手轻柔地将夹爪完全闭合。"
                "当前位置将被设为零点。",
                CALIBRATION_CONFIRMATION,
            )
            pre_calibration_message = gripper.get_gripper_status()
            if pre_calibration_message is None:
                raise RuntimeError("标定前无法读取夹爪反馈时间戳。")
            pre_calibration_timestamp = float(
                pre_calibration_message.timestamp
            )
            calibration_ok = bool(
                gripper.calibrate_gripper(
                    timeout=args.calibration_timeout_seconds
                )
            )
            _append_event(
                events,
                "gripper_calibration_requested",
                sdk_result=calibration_ok,
            )
            if not calibration_ok:
                raise RuntimeError("SDK未确认夹爪零点标定成功，拒绝运动。")
            status_before_motion = _wait_for_zero_feedback(
                gripper,
                newer_than_timestamp=pre_calibration_timestamp,
                tolerance_m=args.target_tolerance_m,
                stable_feedback_count=args.stable_feedback_count,
                timeout_seconds=args.state_timeout_seconds,
            )
            calibrated_this_run = True
            calibration_known = True
            calibration_source = "new_calibration_ack_and_zero_feedback"
            calibration_record = {
                "schema_version": CALIBRATION_RECORD_SCHEMA_VERSION,
                "calibrated_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "device_identity": identity,
                "calibration_acknowledged": True,
                "verified_zero_width_m": float(
                    status_before_motion["value"]
                ),
                "controller_homing_signal": bool(
                    status_before_motion["foc_status"]["homing_status"]
                ),
                "source_output_dir": str(output_dir),
            }
            _write_calibration_record(
                calibration_record_path,
                calibration_record,
            )
            calibration_record_status = "written"
            _append_event(
                events,
                "calibration_record_written",
                path=str(calibration_record_path),
            )
            print("零点标定:        PASS")
        else:
            print(f"零点标定:        已验证（{calibration_source}），自动跳过")

        current_width = float(status_before_motion["value"])
        width_ramp = _build_width_ramp(
            current_width,
            args.target_width_m,
            args.max_width_step_m,
        )
        if any(not 0.0 <= width <= effective_max for width in width_ramp):
            raise RuntimeError(
                f"生成的宽度序列超出安全行程[0, {effective_max:.4f}] m："
                f"{width_ramp}。"
            )
        summary.update(
            {
                "calibrated_this_run": calibrated_this_run,
                "calibration_known": calibration_known,
                "calibration_source": calibration_source,
                "calibration_record_status": calibration_record_status,
                "calibration_record": calibration_record,
                "status_before_motion": status_before_motion,
                "width_ramp_m": width_ramp,
            }
        )
        _require_terminal_confirmation(
            "夹爪将以低力分段张开。请再次确认夹爪内无物体、手指和线缆，"
            "物理急停可立即触发。",
            MOTION_CONFIRMATION,
        )

        print(f"Width ramp:      {[round(value, 6) for value in width_ramp]}")
        for remaining in range(args.countdown_seconds, 0, -1):
            print(f"{remaining}...", flush=True)
            time.sleep(1.0)

        # AGX官方示例要求夹爪位置控制前先让PiperX进入CAN/MOVE P模式。
        # 该调用不会使能六个机械臂关节，也不会发送机械臂位姿目标。
        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
        _append_event(events, "arm_move_p_mode_requested")
        arm_mode_before_gripper_motion = _wait_for_arm_p_control_mode(
            robot,
            timeout_seconds=args.state_timeout_seconds,
        )
        _append_event(
            events,
            "arm_move_p_mode_confirmed",
            arm_status=arm_mode_before_gripper_motion,
        )
        summary["arm_mode_before_gripper_motion"] = (
            arm_mode_before_gripper_motion
        )
        print("Arm control:     CAN_CTRL / MOVE_P")

        trajectory_file = (output_dir / "trajectory.jsonl").open(
            "w", encoding="utf-8"
        )
        arm_reference = list(preflight["joint_angles_rad"])
        motion_results: list[dict[str, Any]] = []
        motion_started = True
        for index, target_width in enumerate(width_ramp, start=1):
            phase = f"setup_ramp_{index}"
            gripper.move_gripper_m(
                value=float(target_width), force=args.force_n
            )
            _append_event(
                events,
                "gripper_command_sent",
                phase_name=phase,
                target_width_m=float(target_width),
                force_n=args.force_n,
            )
            result = _wait_for_width(
                gripper,
                robot,
                phase=phase,
                target_width_m=float(target_width),
                arm_reference_rad=arm_reference,
                force_n=args.force_n,
                tolerance_m=args.target_tolerance_m,
                max_arm_joint_drift_rad=args.max_arm_joint_drift_rad,
                stable_feedback_count=args.stable_feedback_count,
                timeout_seconds=args.motion_timeout_seconds,
                poll_hz=args.poll_hz,
                trajectory_file=trajectory_file,
            )
            motion_results.append(result)
            _append_event(events, "gripper_target_reached", result=result)
            if args.step_hold_seconds > 0:
                time.sleep(args.step_hold_seconds)
        motion_completed = True

        final_message, final_status = _wait_initial_gripper_feedback(
            gripper, args.startup_timeout_seconds
        )
        del final_message
        counters_after = _can_network_counters(args.channel)
        counter_delta = _counter_delta(counters_before, counters_after)
        final_arm_message = robot.get_arm_status()
        final_arm_status = (
            None
            if final_arm_message is None
            else _status_dict(final_arm_message.msg)
        )
        checks = {
            "all_targets_reached": all(
                bool(result["reached"]) for result in motion_results
            ),
            "final_fault_free": not _fault_names(final_status),
            "driver_enabled": final_status["foc_status"]["driver_enable_status"],
            "zero_reference_verified": calibration_known,
            "arm_remains_in_can_move_p": (
                final_arm_status is not None
                and final_arm_status["ctrl_mode"]["value"] == 1
                and final_arm_status["mode_feedback"]["value"] == 0
            ),
            "final_width_reached": (
                abs(float(final_status["value"]) - args.target_width_m)
                <= args.target_tolerance_m
            ),
            "arm_drift_within_limit": all(
                result["max_arm_joint_drift_rad"]
                <= args.max_arm_joint_drift_rad
                for result in motion_results
            ),
            "no_new_can_rx_errors": counter_delta["rx_errors"] == 0,
            "no_new_can_rx_drops": counter_delta["rx_dropped"] == 0,
            "no_new_can_tx_errors": counter_delta["tx_errors"] == 0,
            "no_new_can_tx_drops": counter_delta["tx_dropped"] == 0,
        }
        summary.update(
            {
                "mode": "setup_completed",
                "calibrated_this_run": calibrated_this_run,
                "calibration_source": calibration_source,
                "calibration_record_path": str(calibration_record_path),
                "calibration_record_status": calibration_record_status,
                "width_ramp_m": width_ramp,
                "motion_results": motion_results,
                "final_gripper_status": final_status,
                "final_arm_status": final_arm_status,
                "can_counters_before": counters_before,
                "can_counters_after": counters_after,
                "can_counter_delta": counter_delta,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        path = _write_summary(output_dir, summary)
        print("-" * 72)
        print(f"Final width:     {final_status['value']:.6f} m")
        print(
            "Driver enabled:  "
            f"{final_status['foc_status']['driver_enable_status']}"
        )
        print(
            "Homing signal:   "
            f"{final_status['foc_status']['homing_status']}（仅记录，不作为持久标定依据）"
        )
        print(f"Calibration:     {calibration_source}")
        print(
            "最大机械臂漂移: "
            f"{max(r['max_arm_joint_drift_rad'] for r in motion_results):.6f} rad"
        )
        print(
            "CAN新增丢包:      "
            f"rx={counter_delta['rx_dropped']}, tx={counter_delta['tx_dropped']}"
        )
        print(f"初始化结果:      {'PASS' if summary['passed'] else 'FAIL'}")
        print(f"统计文件:        {path}")
        print(f"运动轨迹:        {output_dir / 'trajectory.jsonl'}")
        return 0 if summary["passed"] else 3
    except KeyboardInterrupt:
        summary["error"] = "用户中断夹爪初始化。"
        _append_event(events, "interrupted")
        print("\n用户中断夹爪初始化。", file=sys.stderr)
        return 130
    except Exception as exc:
        summary["error"] = repr(exc)
        _append_event(events, "failed", error=repr(exc))
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        if (
            robot is not None
            and gripper is not None
            and motion_started
            and not motion_completed
            and latest_status is not None
        ):
            try:
                current_message = gripper.get_gripper_status()
                if current_message is not None:
                    current_status = _gripper_status_dict(current_message.msg)
                    current_faults = _fault_names(current_status)
                    current_enabled = current_status["foc_status"][
                        "driver_enable_status"
                    ]
                    if current_faults:
                        summary["failure_hold_skipped"] = (
                            f"夹爪存在故障标志: {current_faults}"
                        )
                        print(
                            "检测到夹爪故障，未发送额外保持指令；"
                            "请使用物理急停并检查设备。",
                            file=sys.stderr,
                        )
                    elif not current_enabled:
                        summary["failure_hold_skipped"] = "夹爪已失能"
                        print(
                            "夹爪已失能，未发送会重新使能的保持指令。",
                            file=sys.stderr,
                        )
                    elif current_status["mode"] == "width":
                        current_width = float(current_status["value"])
                        gripper.move_gripper_m(
                            value=current_width, force=args.force_n
                        )
                        _append_event(
                            events,
                            "failure_width_hold_sent",
                            width_m=current_width,
                            force_n=args.force_n,
                        )
                        print(
                            "已发送当前夹爪宽度保持指令；请检查现场。",
                            file=sys.stderr,
                        )
            except Exception as hold_exc:
                summary["failure_hold_error"] = repr(hold_exc)
                print(
                    f"无法发送夹爪异常保持指令: {hold_exc}；"
                    "请使用物理急停。",
                    file=sys.stderr,
                )
        if samples_file is not None:
            samples_file.close()
        if trajectory_file is not None:
            trajectory_file.close()
        if robot is not None:
            robot.disconnect()
        _write_summary(output_dir, summary)
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
    return run(parser.parse_args())


if __name__ == "__main__":
    # =====================================================================
    # PiperX AGX夹爪一体化初始化配置区。
    # 默认只读取反馈，不会失能、标定、使能或运动。
    # =====================================================================
    piperx_gripper_setup_config = {
        # 已激活的PiperX SocketCAN接口。
        "channel": "can_piperx",
        "interface": "socketcan",
        "bitrate": 1_000_000,
        # 固件查询、反馈预热和CAN请求超时。
        "startup_timeout_seconds": 5.0,
        "feedback_warmup_seconds": 0.5,
        "can_timeout_seconds": 1.0,
        "sdk_log_level": "WARNING",
        # 预检阶段持续2秒，以100 Hz轮询，要求SDK报告至少50 Hz反馈。
        "read_duration_seconds": 2.0,
        "poll_hz": 100.0,
        "min_feedback_hz": 50.0,
        # 完成回零后，将夹爪分段张开到30 mm初始化宽度。
        "target_width_m": 0.010,
        # 初始化仅使用0.5 N，程序强制限制在0.1到1.0 N。
        "force_n": 0.5,
        # 初始化硬行程上限70 mm，且相邻指令最多变化10 mm。
        "hard_max_width_m": 0.070,
        "max_width_step_m": 0.010,
        # 连续5帧宽度误差不超过2 mm，才判定分段目标到达。
        "target_tolerance_m": 0.002,
        "stable_feedback_count": 5,
        # 初始化期间机械臂任一关节漂移不得超过0.01 rad。
        "max_arm_joint_drift_rad": 0.010,
        # 每个分段最多等待5秒；标定响应和状态切换采用独立超时。
        "motion_timeout_seconds": 5.0,
        "calibration_timeout_seconds": 2.0,
        "state_timeout_seconds": 3.0,
        # 每个分段到达后短暂停留，再执行下一分段。
        "step_hold_seconds": 0.25,
        "countdown_seconds": 5,
        # None使用输出根目录下的calibration_record.json记录一次性标定结果。
        "calibration_record": None,
        # 机械结构或夹爪安装发生变化时才设为True并重新标定。
        "force_recalibration": False,
        # False仅预检；改为True后仍需在当前终端手动输入现场确认词。
        "allow_setup": False,
        # None保存到outputs/6_real_robot_eval/piperx_gripper_setup/时间戳。
        "output_dir": None,
    }

    raise SystemExit(main(config_defaults=piperx_gripper_setup_config))
