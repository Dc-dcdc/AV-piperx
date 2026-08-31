#!/usr/bin/env python3
"""PiperX六关节低速、小幅度顺序运动测试。

测试目的：
    一次连接后依次测试J1至J6。每次只移动一个关节，达到目标后返回共同
    初始姿态，再测试下一关节；不会让六个关节同时执行测试偏移。

运行位置：
    以下命令均在项目根目录 ``/home/dc/dc_project/AV-piper`` 执行。

第一步，仅执行预检（默认不使能、不运动）：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_piperx_motion.py

第二步，只有预检PASS且现场安全时才允许真实运动：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_piperx_motion.py \
        --allow-motion \
        --confirmation I_UNDERSTAND_PIPERX_WILL_MOVE

可选，仅测试J6并向负方向偏移0.02 rad：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_piperx_motion.py \
        --joint-index 6 --joint-offset-rad -0.02

安全说明：
    默认只执行预检。六轴使能状态不一致、机械臂状态异常、目标接近限位
    或任一关节在运动中失能时，程序会拒绝或立即终止后续测试。真实运动
    前必须固定底座、清空工作区，并确保物理急停可立即触发。

真实运动流程：

1. 检查CAN、固件、机械臂状态和初始关节角。
2. 使用平滑的 ``move_j`` 位置-速度模式保持初始位姿。
3. 按配置顺序逐个关节执行小幅偏移，绝不同时测试多个关节。
4. 每个关节稳定到达后返回共同初始位，再测试下一个关节。
5. 仅当测试前所有关节未使能且成功返回时，才恢复为未使能状态。

结果保存在 ``outputs/6_real_robot_eval/piperx_motion_test/<时间戳>/``。
本脚本仅使用平滑的 ``move_j``，不使用 ``move_js``/MIT模式。
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
    _is_status_normal,
    _load_pyagxarm,
    _status_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/6_real_robot_eval/piperx_motion_test"
)
MOTION_CONFIRMATION = "I_UNDERSTAND_PIPERX_WILL_MOVE"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PiperX低速顺序关节小幅运动测试；默认仅预检。"
    )
    parser.add_argument("--channel", default="can_piperx")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--startup-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--feedback-warmup-seconds", type=float, default=0.5)
    parser.add_argument("--can-timeout-seconds", type=float, default=1.0)
    parser.add_argument("--sdk-log-level", default="WARNING")
    parser.add_argument(
        "--joint-indices",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
        help="按顺序测试的关节编号。",
    )
    parser.add_argument(
        "--joint-offsets-rad",
        nargs="+",
        type=float,
        default=[0.03, 0.03, -0.03, 0.03, 0.03, 0.03],
        help="与joint-indices逐项对应的关节偏移。",
    )
    # 保留旧单关节CLI兼容性；只要指定其中一个，就必须同时指定二者。
    parser.add_argument("--joint-index", type=int, default=None)
    parser.add_argument("--joint-offset-rad", type=float, default=None)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--limit-margin-rad", type=float, default=0.05)
    parser.add_argument(
        "--max-non-target-clamp-rad",
        type=float,
        default=0.001,
        help=(
            "SDK对非目标关节进行限幅时允许的最大修正量；超过后预检失败。"
        ),
    )
    parser.add_argument("--target-tolerance-rad", type=float, default=0.005)
    parser.add_argument("--max-other-joint-drift-rad", type=float, default=0.02)
    parser.add_argument("--stable-feedback-count", type=int, default=5)
    parser.add_argument("--motion-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--monitor-hz", type=float, default=100.0)
    parser.add_argument("--countdown-seconds", type=int, default=5)
    parser.add_argument(
        "--restore-initial-enable-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="成功返回后恢复测试前的使能状态。",
    )
    parser.add_argument(
        "--allow-motion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="允许实际使能和运动；仍需提供完整确认词。",
    )
    parser.add_argument(
        "--confirmation",
        default="",
        help=f"运动确认词：{MOTION_CONFIRMATION}",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.channel:
        raise ValueError("--channel不能为空。")
    if args.bitrate <= 0:
        raise ValueError("--bitrate必须大于0。")
    if args.startup_timeout_seconds <= 0:
        raise ValueError("--startup-timeout-seconds必须大于0。")
    if args.feedback_warmup_seconds < 0:
        raise ValueError("--feedback-warmup-seconds不能小于0。")
    if args.can_timeout_seconds <= 0:
        raise ValueError("--can-timeout-seconds必须大于0。")
    joint_tests = _resolve_joint_tests(args)
    if not joint_tests:
        raise ValueError("至少需要配置一个待测试关节。")
    indices = [index for index, _ in joint_tests]
    if len(set(indices)) != len(indices):
        raise ValueError("同一次顺序测试中不能重复关节编号。")
    for joint_index, joint_offset in joint_tests:
        if joint_index not in range(1, 7):
            raise ValueError("关节编号必须在1到6之间。")
        if not math.isfinite(joint_offset) or joint_offset == 0:
            raise ValueError("关节偏移必须是非0有限数。")
        if abs(joint_offset) > 0.1:
            raise ValueError("安全限制：关节偏移绝对值不能超过0.1 rad。")
    if not 1 <= args.speed_percent <= 10:
        raise ValueError("安全限制：--speed-percent必须在1到10之间。")
    if args.limit_margin_rad < 0.02:
        raise ValueError("--limit-margin-rad不得小于0.02 rad。")
    if not 0 <= args.max_non_target_clamp_rad <= 0.005:
        raise ValueError(
            "安全限制：--max-non-target-clamp-rad必须在0到0.005 rad之间。"
        )
    if args.target_tolerance_rad <= 0:
        raise ValueError("--target-tolerance-rad必须大于0。")
    if args.max_other_joint_drift_rad <= 0:
        raise ValueError("--max-other-joint-drift-rad必须大于0。")
    if args.stable_feedback_count <= 0:
        raise ValueError("--stable-feedback-count必须大于0。")
    if args.motion_timeout_seconds <= 0:
        raise ValueError("--motion-timeout-seconds必须大于0。")
    if args.hold_seconds < 0:
        raise ValueError("--hold-seconds不能小于0。")
    if args.monitor_hz <= 0:
        raise ValueError("--monitor-hz必须大于0。")
    if args.countdown_seconds < 0:
        raise ValueError("--countdown-seconds不能小于0。")


def _resolve_joint_tests(args: argparse.Namespace) -> list[tuple[int, float]]:
    """解析新版多关节配置，并兼容旧版单关节参数。"""

    legacy_index = args.joint_index
    legacy_offset = args.joint_offset_rad
    if legacy_index is not None or legacy_offset is not None:
        if legacy_index is None or legacy_offset is None:
            raise ValueError(
                "旧版--joint-index和--joint-offset-rad必须同时提供。"
            )
        return [(int(legacy_index), float(legacy_offset))]
    if len(args.joint_indices) != len(args.joint_offsets_rad):
        raise ValueError(
            "--joint-indices与--joint-offsets-rad的数量必须一致。"
        )
    return [
        (int(index), float(offset))
        for index, offset in zip(
            args.joint_indices,
            args.joint_offsets_rad,
        )
    ]


def _read_preflight(robot: Any, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        joints_message = robot.get_joint_angles()
        flange_message = robot.get_flange_pose()
        status_message = robot.get_arm_status()
        if (
            joints_message is not None
            and flange_message is not None
            and status_message is not None
        ):
            joints = _finite_vector(joints_message.msg, 6)
            flange = _finite_vector(flange_message.msg, 6)
            status = _status_dict(status_message.msg)
            if joints is not None and flange is not None:
                return {
                    "joint_angles_rad": joints,
                    "flange_pose_m_rad": flange,
                    "arm_status": status,
                    "joint_feedback_hz": float(joints_message.hz),
                    "flange_feedback_hz": float(flange_message.hz),
                    "status_feedback_hz": float(status_message.hz),
                    "joint_enable_status": [
                        bool(value)
                        for value in robot.get_joints_enable_status_list()
                    ],
                }
        time.sleep(0.02)
    raise TimeoutError("未在限定时间内获得完整PiperX预检反馈。")


def _validate_preflight(preflight: dict[str, Any], min_feedback_hz: float) -> None:
    if not _is_status_normal(preflight["arm_status"]):
        raise RuntimeError(
            "机械臂存在异常状态，拒绝运动: "
            f"{preflight['arm_status']['repr']}"
        )
    for key in ("joint_feedback_hz", "flange_feedback_hz", "status_feedback_hz"):
        if preflight[key] < min_feedback_hz:
            raise RuntimeError(
                f"反馈频率过低: {key}={preflight[key]:.1f} Hz < "
                f"{min_feedback_hz:.1f} Hz"
            )
    enable_status = preflight["joint_enable_status"]
    if any(enable_status) and not all(enable_status):
        disabled_joints = [
            index + 1 for index, enabled in enumerate(enable_status) if not enabled
        ]
        raise RuntimeError(
            f"六个关节使能状态不一致，拒绝运动: {enable_status}；"
            f"失能关节={disabled_joints}。这通常表示对应驱动器触发保护。"
            "禁止直接重跑或由本测试自动重新使能；请先检查机械限位和现场，"
            "再按厂商流程清错/复位，使六轴恢复为一致状态。"
        )


def _clamp_joint_vector(
    joints: list[float],
    joint_limits: dict[str, list[float]],
) -> tuple[list[float], list[float]]:
    """复现pyAgxArm move_j启用关节限位时的逐关节截断。"""
    clamped: list[float] = []
    adjustments: list[float] = []
    for index, value in enumerate(joints, start=1):
        lower, upper = (float(v) for v in joint_limits[f"joint{index}"])
        command_value = min(max(float(value), lower), upper)
        clamped.append(command_value)
        adjustments.append(command_value - float(value))
    return clamped, adjustments


def _build_command_plan(
    *,
    initial: list[float],
    requested_target: list[float],
    moved_joint_index: int,
    joint_limits: dict[str, list[float]],
    margin: float,
    max_non_target_clamp_rad: float,
) -> dict[str, Any]:
    """生成SDK实际发送的目标，并拒绝显著的非目标关节限幅运动。"""
    initial_command, initial_adjustments = _clamp_joint_vector(
        initial, joint_limits
    )
    target_command, target_adjustments = _clamp_joint_vector(
        requested_target, joint_limits
    )
    warnings: list[str] = []
    violations: list[str] = []
    for index, (initial_value, requested_target_value) in enumerate(
        zip(initial, requested_target), start=1
    ):
        lower, upper = joint_limits[f"joint{index}"]
        initial_adjustment = initial_adjustments[index - 1]
        target_adjustment = target_adjustments[index - 1]
        max_adjustment = max(abs(initial_adjustment), abs(target_adjustment))
        if index != moved_joint_index:
            if not math.isclose(
                initial_value, requested_target_value, abs_tol=1e-12
            ):
                raise RuntimeError(f"非目标关节J{index}的目标值被改变。")
            if max_adjustment > 0:
                message = (
                    f"J{index}当前反馈{initial_value:.6f} rad超出SDK限位"
                    f"[{float(lower):.6f}, {float(upper):.6f}]，move_j将发送"
                    f"{initial_command[index - 1]:.6f} rad，修正"
                    f"{initial_adjustment:+.6f} rad。"
                )
                if max_adjustment > max_non_target_clamp_rad:
                    violations.append(
                        message
                        + f"超过允许值{max_non_target_clamp_rad:.6f} rad。"
                    )
                else:
                    warnings.append(
                        message
                        + "修正量处于允许容差内，但该关节可能发生微小运动。"
                    )
            continue
        safe_lower = float(lower) + margin
        safe_upper = float(upper) - margin
        if not safe_lower <= initial_value <= safe_upper:
            raise RuntimeError(
                f"J{index}初始角{initial_value:.5f} rad距离限位过近，"
                f"安全区间为[{safe_lower:.5f}, {safe_upper:.5f}]。"
            )
        if not safe_lower <= requested_target_value <= safe_upper:
            raise RuntimeError(
                f"J{index}目标角{requested_target_value:.5f} rad超出安全区间"
                f"[{safe_lower:.5f}, {safe_upper:.5f}]。"
            )
    return {
        "initial_command": initial_command,
        "target_command": target_command,
        "initial_adjustments_rad": initial_adjustments,
        "target_adjustments_rad": target_adjustments,
        "warnings": warnings,
        "violations": violations,
        "motion_safe": not violations,
    }


def _wait_enable_state(
    robot: Any,
    *,
    enabled: bool,
    timeout_seconds: float,
) -> list[bool]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if enabled:
            robot.enable()
        else:
            robot.disable()
        time.sleep(0.05)
        states = [bool(value) for value in robot.get_joints_enable_status_list()]
        if all(states) if enabled else not any(states):
            return states
    target_state_name = "使能" if enabled else "失能"
    raise TimeoutError(
        f"机械臂未在{timeout_seconds:.1f}秒内{target_state_name}成功。"
    )


def _append_event(events: list[dict[str, Any]], phase: str, **values: Any) -> None:
    events.append(
        {
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "monotonic_seconds": time.monotonic(),
            "phase": phase,
            **values,
        }
    )


def _wait_for_target(
    robot: Any,
    *,
    phase: str,
    target: list[float],
    reference: list[float],
    moved_joint_index: int,
    tolerance_rad: float,
    max_other_joint_drift_rad: float,
    stable_feedback_count: int,
    timeout_seconds: float,
    monitor_hz: float,
    trajectory_file: Any,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    period = 1.0 / monitor_hz
    stable_count = 0
    max_error = 0.0
    max_other_drift = 0.0
    latest_moved_joint_error: float | None = None
    latest_other_command_error: float | None = None
    samples = 0
    last_timestamp: float | None = None
    latest: list[float] | None = None

    while time.monotonic() < deadline:
        loop_start = time.monotonic()
        joints_message = robot.get_joint_angles()
        status_message = robot.get_arm_status()
        if joints_message is None or status_message is None:
            time.sleep(period)
            continue

        timestamp = float(joints_message.timestamp)
        if last_timestamp is not None and timestamp <= last_timestamp:
            time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
            continue
        last_timestamp = timestamp
        latest = _finite_vector(joints_message.msg, 6)
        if latest is None:
            raise RuntimeError(f"{phase}期间收到非有限关节角。")

        status = _status_dict(status_message.msg)
        if not _is_status_normal(status):
            raise RuntimeError(f"{phase}期间机械臂状态异常: {status['repr']}")

        enable_status = [
            bool(value) for value in robot.get_joints_enable_status_list()
        ]
        if not all(enable_status):
            disabled_joints = [
                index + 1
                for index, enabled in enumerate(enable_status)
                if not enabled
            ]
            raise RuntimeError(
                f"{phase}期间检测到关节失能: {enable_status}；"
                f"失能关节={disabled_joints}。立即终止后续测试，"
                "禁止自动清错或重新使能。"
            )

        errors = [abs(value - goal) for value, goal in zip(latest, target)]
        current_error = max(errors)
        moved_joint_error = errors[moved_joint_index - 1]
        other_command_error = max(
            (
                error
                for index, error in enumerate(errors)
                if index != moved_joint_index - 1
            ),
            default=0.0,
        )
        latest_moved_joint_error = moved_joint_error
        latest_other_command_error = other_command_error
        max_error = max(max_error, current_error)
        other_drift = max(
            (
                abs(value - reference[index])
                for index, value in enumerate(latest)
                if index != moved_joint_index - 1
            ),
            default=0.0,
        )
        max_other_drift = max(max_other_drift, other_drift)
        if other_drift > max_other_joint_drift_rad:
            raise RuntimeError(
                f"{phase}期间非目标关节漂移{other_drift:.5f} rad超过"
                f"上限{max_other_joint_drift_rad:.5f} rad。"
            )

        samples += 1
        if current_error <= tolerance_rad:
            stable_count += 1
        else:
            stable_count = 0

        trajectory_file.write(
            json.dumps(
                {
                    "phase": phase,
                    "local_time_ns": time.time_ns(),
                    "feedback_timestamp": timestamp,
                    "joint_angles_rad": latest,
                    "target_joint_angles_rad": target,
                    "max_target_error_rad": current_error,
                    "moved_joint_error_rad": moved_joint_error,
                    "max_non_target_command_error_rad": other_command_error,
                    "max_other_joint_drift_rad": other_drift,
                    "joint_enable_status": enable_status,
                    "arm_status": status,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        if stable_count >= stable_feedback_count:
            return {
                "reached": True,
                "samples": samples,
                "latest_joint_angles_rad": latest,
                "final_max_error_rad": current_error,
                "final_moved_joint_error_rad": moved_joint_error,
                "final_max_non_target_command_error_rad": (
                    other_command_error
                ),
                "max_error_rad": max_error,
                "max_other_joint_drift_rad": max_other_drift,
            }
        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))

    raise TimeoutError(
        f"{phase}未在{timeout_seconds:.1f}秒内稳定到达目标；"
        f"最后关节角={latest}，目标关节误差={latest_moved_joint_error}，"
        f"非目标关节最大指令误差={latest_other_command_error}。"
    )


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    path = output_dir / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    joint_tests = _resolve_joint_tests(args)
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
    trajectory_file = None
    motion_started = False
    returned_to_initial = False
    initial_enable_status: list[bool] | None = None
    joint_limits: dict[str, list[float]] | None = None
    events: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_type": "piperx_low_speed_joint_motion",
        "configuration": vars(args).copy(),
        "events": events,
        "passed": False,
    }
    if isinstance(summary["configuration"].get("output_dir"), Path):
        summary["configuration"]["output_dir"] = str(
            summary["configuration"]["output_dir"]
        )

    try:
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
        runtime_config = _create_config(
            create_agx_arm_config=create_agx_arm_config,
            robot_model=ArmModel.PIPER_X,
            firmware_profile=profile,
            args=args,
        )
        joint_limits = runtime_config["joint_limits"]
        robot = factory.create_arm(runtime_config)
        robot.connect()
        time.sleep(args.feedback_warmup_seconds)

        preflight = _read_preflight(robot, args.startup_timeout_seconds)
        _validate_preflight(preflight, min_feedback_hz=150.0)
        initial = list(preflight["joint_angles_rad"])
        initial_enable_status = list(preflight["joint_enable_status"])
        command_plans: list[dict[str, Any]] = []
        joint_limit_warnings: list[str] = []
        joint_limit_violations: list[str] = []
        for joint_index, joint_offset in joint_tests:
            requested_target = list(initial)
            requested_target[joint_index - 1] += joint_offset
            try:
                plan = _build_command_plan(
                    initial=initial,
                    requested_target=requested_target,
                    moved_joint_index=joint_index,
                    joint_limits=joint_limits,
                    margin=args.limit_margin_rad,
                    max_non_target_clamp_rad=args.max_non_target_clamp_rad,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"J{joint_index}顺序测试预检失败：{exc} "
                    "请先运行test_piperx_safe_pose.py进入安全初始姿态。"
                ) from exc
            plan.update(
                {
                    "joint_index": joint_index,
                    "joint_offset_rad": joint_offset,
                    "requested_target": requested_target,
                }
            )
            command_plans.append(plan)
            joint_limit_warnings.extend(
                f"J{joint_index}测试: {warning}"
                for warning in plan["warnings"]
            )
            joint_limit_violations.extend(
                f"J{joint_index}测试: {violation}"
                for violation in plan["violations"]
            )
        initial_command = list(command_plans[0]["initial_command"])
        summary.update(
            {
                "firmware": firmware,
                "firmware_profile": profile,
                "can_interface": can_details,
                "preflight": preflight,
                "initial_joint_angles_rad": initial,
                "command_initial_joint_angles_rad": initial_command,
                "joint_tests": [
                    {
                        "joint_index": plan["joint_index"],
                        "joint_offset_rad": plan["joint_offset_rad"],
                        "requested_target_joint_angles_rad": plan[
                            "requested_target"
                        ],
                        "command_target_joint_angles_rad": plan[
                            "target_command"
                        ],
                        "initial_adjustments_rad": plan[
                            "initial_adjustments_rad"
                        ],
                        "target_adjustments_rad": plan[
                            "target_adjustments_rad"
                        ],
                    }
                    for plan in command_plans
                ],
                "initial_joint_enable_status": initial_enable_status,
                "joint_limit_warnings": joint_limit_warnings,
                "joint_limit_violations": joint_limit_violations,
                "command_clamp": {
                    "max_non_target_clamp_rad": (
                        args.max_non_target_clamp_rad
                    ),
                    "initial_adjustments_rad": command_plans[0][
                        "initial_adjustments_rad"
                    ],
                    "motion_safe": not joint_limit_violations,
                },
                "safety": {
                    "move_api": "move_j",
                    "move_js_or_mit_used": False,
                    "joint_indices": [index for index, _ in joint_tests],
                    "joint_offsets_rad": [offset for _, offset in joint_tests],
                    "sequential_single_joint_motion": True,
                    "return_to_common_initial_between_joints": True,
                    "speed_percent": args.speed_percent,
                    "motion_allowed": bool(args.allow_motion),
                    "confirmation_valid": (
                        args.confirmation == MOTION_CONFIRMATION
                    ),
                },
            }
        )

        print("PiperX低速顺序关节运动测试")
        print("-" * 72)
        print(f"Firmware:        {firmware['software_version']} ({profile})")
        print(f"CAN channel:     {args.channel}")
        print(f"Initial joints:  {[round(value, 5) for value in initial]}")
        print(
            "SDK command init: "
            f"{[round(value, 5) for value in initial_command]}"
        )
        print(f"Speed:           {args.speed_percent}%")
        for plan in command_plans:
            print(
                f"J{plan['joint_index']} target:       "
                f"offset={plan['joint_offset_rad']:+.5f} rad, "
                f"target={plan['target_command'][plan['joint_index'] - 1]:.5f} rad"
            )
        print(f"Initial enabled: {initial_enable_status}")
        for warning in joint_limit_warnings:
            print(f"Limit warning:   {warning}")
        for violation in joint_limit_violations:
            print(f"Limit violation: {violation}")
        print(f"Output:          {output_dir}")
        print("-" * 72)

        if joint_limit_violations:
            summary["mode"] = "preflight_rejected"
            summary["passed"] = False
            summary["message"] = (
                "非目标关节所需的SDK限幅修正超过安全容差，"
                "未使能且未发送运动指令。"
            )
            _append_event(
                events,
                "preflight_rejected",
                violations=joint_limit_violations,
            )
            path = _write_summary(output_dir, summary)
            print("预检结果:        FAIL（未运动）")
            print(f"统计文件:        {path}")
            return 2

        _append_event(events, "preflight_passed")

        if not args.allow_motion:
            summary["mode"] = "preflight_only"
            summary["passed"] = True
            summary["message"] = (
                "预检通过，allow_motion=False，未使能也未发送运动指令。"
            )
            path = _write_summary(output_dir, summary)
            print("预检结果:        PASS（未运动）")
            print(f"统计文件:        {path}")
            return 0

        if args.confirmation != MOTION_CONFIRMATION:
            raise RuntimeError(
                "allow_motion=True，但确认词不正确。拒绝使能和运动。"
            )

        print("警告：机械臂即将使能并产生真实运动。")
        print("确保工作空间无人、底座已固定且物理急停可立即触发。")
        for remaining in range(args.countdown_seconds, 0, -1):
            print(f"{remaining}...")
            time.sleep(1.0)

        trajectory_file = (output_dir / "trajectory.jsonl").open(
            "w", encoding="utf-8"
        )
        _append_event(events, "enable_started")
        _wait_enable_state(
            robot,
            enabled=True,
            timeout_seconds=args.startup_timeout_seconds,
        )
        motion_started = True
        _append_event(events, "enabled")

        robot.set_speed_percent(args.speed_percent)
        robot.move_j(initial_command)
        hold_result = _wait_for_target(
            robot,
            phase="initial_hold",
            target=initial_command,
            # 漂移必须相对真实初始反馈计算，不能相对SDK限幅目标计算。
            reference=initial,
            moved_joint_index=joint_tests[0][0],
            tolerance_rad=args.target_tolerance_rad,
            max_other_joint_drift_rad=args.max_other_joint_drift_rad,
            stable_feedback_count=args.stable_feedback_count,
            timeout_seconds=args.motion_timeout_seconds,
            monitor_hz=args.monitor_hz,
            trajectory_file=trajectory_file,
        )
        _append_event(events, "initial_hold_reached", result=hold_result)

        joint_motion_results: list[dict[str, Any]] = []
        for plan in command_plans:
            joint_index = int(plan["joint_index"])
            target_command = list(plan["target_command"])
            returned_to_initial = False
            robot.move_j(target_command)
            outward_result = _wait_for_target(
                robot,
                phase=f"j{joint_index}_outward",
                target=target_command,
                reference=initial,
                moved_joint_index=joint_index,
                tolerance_rad=args.target_tolerance_rad,
                max_other_joint_drift_rad=args.max_other_joint_drift_rad,
                stable_feedback_count=args.stable_feedback_count,
                timeout_seconds=args.motion_timeout_seconds,
                monitor_hz=args.monitor_hz,
                trajectory_file=trajectory_file,
            )
            _append_event(
                events,
                "joint_outward_target_reached",
                joint_index=joint_index,
                result=outward_result,
            )
            if args.hold_seconds > 0:
                time.sleep(args.hold_seconds)

            robot.move_j(initial_command)
            return_result = _wait_for_target(
                robot,
                phase=f"j{joint_index}_return",
                target=initial_command,
                reference=initial,
                moved_joint_index=joint_index,
                tolerance_rad=args.target_tolerance_rad,
                max_other_joint_drift_rad=args.max_other_joint_drift_rad,
                stable_feedback_count=args.stable_feedback_count,
                timeout_seconds=args.motion_timeout_seconds,
                monitor_hz=args.monitor_hz,
                trajectory_file=trajectory_file,
            )
            returned_to_initial = True
            result = {
                "joint_index": joint_index,
                "joint_offset_rad": float(plan["joint_offset_rad"]),
                "target_joint_angles_rad": target_command,
                "outward": outward_result,
                "return": return_result,
            }
            joint_motion_results.append(result)
            _append_event(
                events,
                "joint_returned_to_initial",
                joint_index=joint_index,
                result=return_result,
            )

        final_enable_status = [
            bool(value) for value in robot.get_joints_enable_status_list()
        ]
        if args.restore_initial_enable_state and not any(initial_enable_status):
            # 只在已成功返回初始角度时恢复原本的未使能状态。
            # 若机械臂处于悬空负载位姿，用户应将该配置设为False。
            final_enable_status = _wait_enable_state(
                robot,
                enabled=False,
                timeout_seconds=args.startup_timeout_seconds,
            )
            _append_event(events, "initial_disabled_state_restored")

        counters_after = _can_network_counters(args.channel)
        counter_delta = _counter_delta(counters_before, counters_after)
        final_status_message = robot.get_arm_status()
        final_status = (
            None
            if final_status_message is None
            else _status_dict(final_status_message.msg)
        )
        checks = {
            "preflight_normal": _is_status_normal(preflight["arm_status"]),
            "all_outward_targets_reached": all(
                bool(result["outward"]["reached"])
                for result in joint_motion_results
            ),
            "all_returns_reached": all(
                bool(result["return"]["reached"])
                for result in joint_motion_results
            ),
            "all_requested_joint_tests_completed": (
                len(joint_motion_results) == len(joint_tests)
            ),
            "final_status_normal": (
                final_status is not None and _is_status_normal(final_status)
            ),
            "no_new_can_rx_errors": counter_delta["rx_errors"] == 0,
            "no_new_can_rx_drops": counter_delta["rx_dropped"] == 0,
            "no_new_can_tx_errors": counter_delta["tx_errors"] == 0,
            "no_new_can_tx_drops": counter_delta["tx_dropped"] == 0,
        }
        summary.update(
            {
                "mode": "motion",
                "motion_results": {
                    "initial_hold": hold_result,
                    "joints": joint_motion_results,
                },
                "final_joint_enable_status": final_enable_status,
                "final_arm_status": final_status,
                "can_counters_before": counters_before,
                "can_counters_after": counters_after,
                "can_counter_delta": counter_delta,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        path = _write_summary(output_dir, summary)
        print("-" * 72)
        for result in joint_motion_results:
            max_non_target_drift = max(
                result["outward"]["max_other_joint_drift_rad"],
                result["return"]["max_other_joint_drift_rad"],
            )
            print(
                f"J{result['joint_index']}: 外移误差="
                f"{result['outward']['final_moved_joint_error_rad']:.6f} rad, "
                f"返回误差={result['return']['final_moved_joint_error_rad']:.6f} rad, "
                f"非目标最大漂移={max_non_target_drift:.6f} rad"
            )
        print(f"最终使能状态:  {final_enable_status}")
        print(
            "CAN新增丢包:      "
            f"rx={counter_delta['rx_dropped']}, tx={counter_delta['tx_dropped']}"
        )
        print(f"测试结果:        {'PASS' if summary['passed'] else 'FAIL'}")
        print(f"统计文件:        {path}")
        print(f"运动轨迹:        {output_dir / 'trajectory.jsonl'}")
        return 0 if summary["passed"] else 3
    except KeyboardInterrupt:
        error = "用户中断测试。"
        summary["error"] = error
        _append_event(events, "interrupted")
        print(f"\n{error}", file=sys.stderr)
        return 130
    except Exception as exc:
        summary["error"] = repr(exc)
        _append_event(events, "failed", error=repr(exc))
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None and motion_started and not returned_to_initial:
            # 异常时将当前反馈角度发回给平滑位置控制器，覆盖旧目标。
            # 不在不明负载姿态下自动失能，避免机械臂突然下落。
            try:
                current_message = robot.get_joint_angles()
                current = (
                    None
                    if current_message is None
                    else _finite_vector(current_message.msg, 6)
                )
                if current is not None and joint_limits is not None:
                    hold_command, hold_adjustments = _clamp_joint_vector(
                        current, joint_limits
                    )
                    robot.set_speed_percent(1)
                    robot.move_j(hold_command)
                    _append_event(
                        events,
                        "failure_hold_command_sent",
                        requested_joints=current,
                        command_joints=hold_command,
                        clamp_adjustments_rad=hold_adjustments,
                    )
                    print(
                        "已发送显式限幅后的当前角度保持指令；"
                        "机械臂保持使能，"
                        "请检查现场并使用物理急停。",
                        file=sys.stderr,
                    )
            except Exception as hold_exc:
                summary["failure_hold_error"] = repr(hold_exc)
                print(
                    f"无法发送异常保持指令: {hold_exc}；请立即按下物理急停。",
                    file=sys.stderr,
                )
        if trajectory_file is not None:
            trajectory_file.close()
        if robot is not None:
            robot.disconnect()
        # 无论成功与否都再写一次，确保中断/异常也有记录。
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
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    # =====================================================================
    # PiperX低速运动测试配置区。默认只预检，不会使能或运动。
    # 启用运动前，必须亲自检查现场并同时修改allow_motion和confirmation。
    # =====================================================================
    piperx_motion_config = {
        # 已激活的PiperX SocketCAN接口。
        "channel": "can_piperx",
        "interface": "socketcan",
        "bitrate": 1_000_000,
        # 固件/反馈启动参数。
        "startup_timeout_seconds": 5.0,
        "feedback_warmup_seconds": 0.5,
        "can_timeout_seconds": 1.0,
        "sdk_log_level": "WARNING",
        # 依次测试J1到J6；每个关节返回共同初始位后再测试下一轴。
        "joint_indices": [1, 2, 3, 4, 5, 6],
        # 偏移幅度均为0.03 rad；J3与J6使用负向以远离当前正向边界。
        # 本机J6在约2.94 rad处继续正向运动曾触发驱动器保护，禁止再向正向测试。
        "joint_offsets_rad": [0.03, 0.03, -0.03, 0.03, 0.03, -0.03],
        # 旧单关节参数保留为None；两者同时填写时覆盖上面的多轴配置。
        "joint_index": None,
        "joint_offset_rad": None,
        # move_j平滑位置-速度控制的速度比例；程序强制1%~10%。
        "speed_percent": 5,
        # 目标角必须与SDK关节限位保持的最小距离。
        "limit_margin_rad": 0.05,
        # 非目标关节被SDK限幅时允许的最大微调量；超过即预检失败。
        # 默认0.001 rad，约0.057°，禁止用它掩盖明显的关节漂移。
        "max_non_target_clamp_rad": 0.001,
        # 连续多帧最大关节误差低于该值才判定到达。
        "target_tolerance_rad": 0.005,
        "stable_feedback_count": 5,
        # 运动期间任一非目标关节相对初始值的最大允许漂移。
        "max_other_joint_drift_rad": 0.005,
        "motion_timeout_seconds": 10.0,
        "hold_seconds": 1.0,
        "monitor_hz": 100.0,
        # 发送真实使能指令前的倒计时，可用Ctrl+C取消。
        "countdown_seconds": 5,
        # 测试前若六个关节全未使能，成功返回后恢复为未使能。
        # 若机械臂携带负载且失能后可能下落，必须改为False。
        "restore_initial_enable_state": False,
        # 安全门：默认False时只预检，绝不运动。
        "allow_motion": False,
        # 第二道安全门：必须完整填写I_UNDERSTAND_PIPERX_WILL_MOVE。
        "confirmation": "I_UNDERSTAND_PIPERX_WILL_MOVE",
        # None表示保存到AV-piper/outputs/6_real_robot_eval/时间戳目录。
        "output_dir": None,
    }

    raise SystemExit(main(config_defaults=piperx_motion_config))
