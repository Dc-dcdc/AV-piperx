#!/usr/bin/env python3
"""将PiperX低速移动到可继续测试的安全初始关节姿态。

测试目的：
    在当前J2、J3或J4靠近/超出SDK安全区间、无法直接执行普通运动测试
    时，将机械臂低速移动到预设安全姿态。它不是日常运动测试脚本。

运行位置：
    以下命令均在项目根目录 ``/home/dc/dc_project/AV-piper`` 执行。

第一步，仅检查当前位置、目标、限位和路径（默认不使能、不运动）：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_piperx_safe_pose.py

第二步，只有预检PASS且完整路径无碰撞时才允许真实运动：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_piperx_safe_pose.py \
        --allow-motion \
        --confirmation I_UNDERSTAND_PIPERX_WILL_MOVE_TO_SAFE_POSE

安全说明：
    六轴使能状态不一致时禁止运行，应先检查触发保护的关节并按厂家流程
    复位。安全姿态到达后默认保持使能，不返回原来的越界姿态。执行前必须
    固定底座、清空工作区、支撑可能下落的连杆，并确保物理急停可立即触发。

真实运动必须同时满足：

1. ``allow_motion=True``；
2. 确认词完全正确；
3. 每个目标关节都位于带安全余量的软件限位内；
4. 任一关节变化量不超过程序设定的硬上限。

结果保存在 ``outputs/6_real_robot_eval/piperx_safe_pose_test/<时间戳>/``。
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
from test_piperx_motion import (
    _append_event,
    _clamp_joint_vector,
    _read_preflight,
    _validate_preflight,
    _wait_enable_state,
    _write_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/6_real_robot_eval/piperx_safe_pose_test"
)
MOTION_CONFIRMATION = "I_UNDERSTAND_PIPERX_WILL_MOVE_TO_SAFE_POSE"


def _optional_joint_value(text: str) -> float | None:
    if text.strip().lower() in {"keep", "none", "null"}:
        return None
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("关节目标必须是有限数或keep。")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PiperX低速安全初始姿态测试；默认仅预检。"
    )
    parser.add_argument("--channel", default="can_piperx")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--startup-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--feedback-warmup-seconds", type=float, default=0.5)
    parser.add_argument("--can-timeout-seconds", type=float, default=1.0)
    parser.add_argument("--sdk-log-level", default="WARNING")
    parser.add_argument(
        "--target-joint-angles-rad",
        nargs=6,
        type=_optional_joint_value,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=[None, 0.10, -0.08, -1.48, None, None],
        help="六关节安全目标；keep/none表示保持当前反馈。",
    )
    parser.add_argument("--speed-percent", type=int, default=3)
    parser.add_argument(
        "--mode-settle-seconds",
        type=float,
        default=0.5,
        help="首次move_j预保持完成后，等待控制模式稳定的时间。",
    )
    parser.add_argument("--limit-margin-rad", type=float, default=0.05)
    parser.add_argument("--max-joint-change-rad", type=float, default=0.15)
    parser.add_argument("--target-tolerance-rad", type=float, default=0.005)
    parser.add_argument("--path-slack-rad", type=float, default=0.02)
    parser.add_argument("--stable-feedback-count", type=int, default=5)
    parser.add_argument("--motion-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--monitor-hz", type=float, default=100.0)
    parser.add_argument("--countdown-seconds", type=int, default=5)
    parser.add_argument(
        "--disable-after-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="成功后是否失能；默认保持使能以防机械臂下垂。",
    )
    parser.add_argument(
        "--allow-motion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="允许真实使能和运动；仍需填写完整确认词。",
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
    if len(args.target_joint_angles_rad) != 6:
        raise ValueError("--target-joint-angles-rad必须包含六项。")
    if not 1 <= args.speed_percent <= 3:
        raise ValueError("安全限制：--speed-percent必须在1到3之间。")
    if not 0.1 <= args.mode_settle_seconds <= 2.0:
        raise ValueError("--mode-settle-seconds必须在0.1到2.0秒之间。")
    if not 0.02 <= args.limit_margin_rad <= 0.15:
        raise ValueError("--limit-margin-rad必须在0.02到0.15 rad之间。")
    if not 0 < args.max_joint_change_rad <= 0.20:
        raise ValueError(
            "安全限制：--max-joint-change-rad必须大于0且不超过0.20 rad。"
        )
    if args.target_tolerance_rad <= 0:
        raise ValueError("--target-tolerance-rad必须大于0。")
    if not 0 <= args.path_slack_rad <= 0.05:
        raise ValueError("--path-slack-rad必须在0到0.05 rad之间。")
    if args.stable_feedback_count <= 0:
        raise ValueError("--stable-feedback-count必须大于0。")
    if args.motion_timeout_seconds <= 0:
        raise ValueError("--motion-timeout-seconds必须大于0。")
    if args.monitor_hz <= 0:
        raise ValueError("--monitor-hz必须大于0。")
    if args.countdown_seconds < 0:
        raise ValueError("--countdown-seconds不能小于0。")


def _build_safe_target(
    *,
    initial: list[float],
    requested: list[float | None],
    joint_limits: dict[str, list[float]],
    margin_rad: float,
    max_joint_change_rad: float,
) -> dict[str, Any]:
    target = [
        initial[index] if value is None else float(value)
        for index, value in enumerate(requested)
    ]
    changed_indices: list[int] = []
    changes: list[float] = []
    for index, (start, goal) in enumerate(zip(initial, target), start=1):
        lower, upper = (float(v) for v in joint_limits[f"joint{index}"])
        safe_lower = lower + margin_rad
        safe_upper = upper - margin_rad
        if safe_lower > safe_upper:
            raise RuntimeError(f"J{index}安全余量导致有效区间为空。")
        if not safe_lower <= goal <= safe_upper:
            raise RuntimeError(
                f"J{index}安全目标{goal:.6f} rad不在带余量区间"
                f"[{safe_lower:.6f}, {safe_upper:.6f}]内。"
            )
        change = goal - start
        changes.append(change)
        if not math.isclose(change, 0.0, abs_tol=1e-12):
            changed_indices.append(index)
        if abs(change) > max_joint_change_rad:
            raise RuntimeError(
                f"J{index}需要移动{change:+.6f} rad，超过单次安全上限"
                f"{max_joint_change_rad:.6f} rad。"
            )

    if not changed_indices:
        raise RuntimeError("安全目标与当前反馈相同，无需执行初始化运动。")

    clamped_target, clamp_adjustments = _clamp_joint_vector(
        target, joint_limits
    )
    if any(abs(value) > 1e-12 for value in clamp_adjustments):
        raise RuntimeError(
            "安全目标仍会被SDK限幅，拒绝运动: "
            f"adjustments={clamp_adjustments}"
        )
    return {
        "target": clamped_target,
        "changes_rad": changes,
        "changed_joint_indices": changed_indices,
    }


def _wait_for_safe_target(
    robot: Any,
    *,
    initial: list[float],
    target: list[float],
    tolerance_rad: float,
    path_slack_rad: float,
    stable_feedback_count: int,
    timeout_seconds: float,
    monitor_hz: float,
    trajectory_file: Any,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    period = 1.0 / monitor_hz
    stable_count = 0
    samples = 0
    last_timestamp: float | None = None
    latest: list[float] | None = None
    maximum_error = 0.0

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
            raise RuntimeError("安全姿态运动期间收到非有限关节反馈。")

        status = _status_dict(status_message.msg)
        if not _is_status_normal(status):
            raise RuntimeError(
                f"安全姿态运动期间机械臂状态异常: {status['repr']}"
            )

        for index, value in enumerate(latest):
            path_lower = min(initial[index], target[index]) - path_slack_rad
            path_upper = max(initial[index], target[index]) + path_slack_rad
            if not path_lower <= value <= path_upper:
                raise RuntimeError(
                    f"J{index + 1}反馈{value:.6f} rad超出预期运动包络"
                    f"[{path_lower:.6f}, {path_upper:.6f}]。"
                )

        errors = [abs(value - goal) for value, goal in zip(latest, target)]
        current_error = max(errors)
        maximum_error = max(maximum_error, current_error)
        samples += 1
        stable_count = stable_count + 1 if current_error <= tolerance_rad else 0

        trajectory_file.write(
            json.dumps(
                {
                    "local_time_ns": time.time_ns(),
                    "feedback_timestamp": timestamp,
                    "joint_angles_rad": latest,
                    "target_joint_angles_rad": target,
                    "joint_errors_rad": errors,
                    "max_target_error_rad": current_error,
                    "arm_status": status,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        trajectory_file.flush()

        if stable_count >= stable_feedback_count:
            return {
                "reached": True,
                "samples": samples,
                "latest_joint_angles_rad": latest,
                "final_joint_errors_rad": errors,
                "final_max_error_rad": current_error,
                "max_error_rad": maximum_error,
            }
        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))

    raise TimeoutError(
        f"未在{timeout_seconds:.1f}秒内到达安全姿态；最后反馈={latest}。"
    )


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    print("PiperX安全初始姿态测试：正在初始化……", flush=True)
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
    reached_safe_pose = False
    joint_limits: dict[str, list[float]] | None = None
    events: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_type": "piperx_low_speed_safe_initial_pose",
        "configuration": vars(args).copy(),
        "events": events,
        "passed": False,
    }
    if isinstance(summary["configuration"].get("output_dir"), Path):
        summary["configuration"]["output_dir"] = str(
            summary["configuration"]["output_dir"]
        )

    try:
        print("检查CAN接口……", flush=True)
        can_details = _can_interface_details(args.channel)
        if not can_details["is_up"]:
            raise RuntimeError(f"CAN接口{args.channel!r}尚未启用。")
        counters_before = _can_network_counters(args.channel)

        print("查询PiperX固件并等待反馈……", flush=True)
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
        plan = _build_safe_target(
            initial=initial,
            requested=list(args.target_joint_angles_rad),
            joint_limits=joint_limits,
            margin_rad=args.limit_margin_rad,
            max_joint_change_rad=args.max_joint_change_rad,
        )
        target = list(plan["target"])
        changes = list(plan["changes_rad"])
        startup_hold_target, startup_hold_adjustments = _clamp_joint_vector(
            initial, joint_limits
        )

        summary.update(
            {
                "firmware": firmware,
                "firmware_profile": profile,
                "can_interface": can_details,
                "preflight": preflight,
                "initial_joint_angles_rad": initial,
                "safe_target_joint_angles_rad": target,
                "startup_hold_target_joint_angles_rad": startup_hold_target,
                "startup_hold_adjustments_rad": startup_hold_adjustments,
                "joint_changes_rad": changes,
                "changed_joint_indices": plan["changed_joint_indices"],
                "initial_joint_enable_status": initial_enable_status,
                "safety": {
                    "move_api": "move_j",
                    "speed_percent": args.speed_percent,
                    "mode_settle_seconds": args.mode_settle_seconds,
                    "motion_allowed": bool(args.allow_motion),
                    "confirmation_valid": (
                        args.confirmation == MOTION_CONFIRMATION
                    ),
                    "returns_to_original_pose": False,
                },
            }
        )
        _append_event(events, "preflight_passed")

        print("-" * 72)
        print(f"Firmware:        {firmware['software_version']} ({profile})")
        print(f"CAN channel:     {args.channel}")
        print(f"Initial joints:  {[round(v, 6) for v in initial]}")
        print(f"Safe target:     {[round(v, 6) for v in target]}")
        print(f"Joint changes:   {[round(v, 6) for v in changes]}")
        print(f"Changed joints:  {plan['changed_joint_indices']}")
        print(f"Speed:           {args.speed_percent}%")
        print(f"Initial enabled: {initial_enable_status}")
        print(f"Output:          {output_dir}")
        print("Return behavior: 不返回原越界姿态，成功后默认保持使能")
        print("-" * 72)

        if not args.allow_motion:
            summary["mode"] = "preflight_only"
            summary["passed"] = True
            summary["message"] = (
                "安全姿态预检通过，allow_motion=False，未使能且未运动。"
            )
            path = _write_summary(output_dir, summary)
            print("预检结果:        PASS（未运动）")
            print(f"统计文件:        {path}")
            return 0

        if args.confirmation != MOTION_CONFIRMATION:
            raise RuntimeError(
                "allow_motion=True，但安全姿态确认词不正确；拒绝运动。"
            )

        print("警告：机械臂将低速使能并移动到安全初始姿态。")
        print("确认路径无碰撞、工作区无人且物理急停可立即触发。")
        for remaining in range(args.countdown_seconds, 0, -1):
            print(f"{remaining}...", flush=True)
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
        # 第一次move_j用于把控制器从STANDBY/MOVE_P稳定切换到MOVE_J。
        # 直接在使能后立刻发送最终目标，部分固件可能只完成模式切换而
        # 忽略同一时刻到达的首个关节目标。这里先发送限幅后的当前姿态，
        # 同时把启动时轻微越界的关节安全带回SDK边界内。
        robot.move_j(startup_hold_target)
        startup_hold_result = _wait_for_safe_target(
            robot,
            initial=initial,
            target=startup_hold_target,
            tolerance_rad=args.target_tolerance_rad,
            path_slack_rad=args.path_slack_rad,
            stable_feedback_count=args.stable_feedback_count,
            timeout_seconds=args.motion_timeout_seconds,
            monitor_hz=args.monitor_hz,
            trajectory_file=trajectory_file,
        )
        _append_event(
            events,
            "startup_hold_reached",
            result=startup_hold_result,
        )
        time.sleep(args.mode_settle_seconds)
        _append_event(events, "move_j_mode_settled")

        robot.move_j(target)
        result = _wait_for_safe_target(
            robot,
            initial=list(startup_hold_result["latest_joint_angles_rad"]),
            target=target,
            tolerance_rad=args.target_tolerance_rad,
            path_slack_rad=args.path_slack_rad,
            stable_feedback_count=args.stable_feedback_count,
            timeout_seconds=args.motion_timeout_seconds,
            monitor_hz=args.monitor_hz,
            trajectory_file=trajectory_file,
        )
        reached_safe_pose = True
        _append_event(events, "safe_pose_reached", result=result)

        final_enable_status = [
            bool(value) for value in robot.get_joints_enable_status_list()
        ]
        if args.disable_after_success:
            print(
                "警告：配置要求成功后失能，机械臂可能因重力下落。",
                file=sys.stderr,
            )
            final_enable_status = _wait_enable_state(
                robot,
                enabled=False,
                timeout_seconds=args.startup_timeout_seconds,
            )
            _append_event(events, "disabled_after_success")

        counters_after = _can_network_counters(args.channel)
        counter_delta = _counter_delta(counters_before, counters_after)
        final_status_message = robot.get_arm_status()
        final_status = (
            None
            if final_status_message is None
            else _status_dict(final_status_message.msg)
        )
        checks = {
            "safe_pose_reached": bool(result["reached"]),
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
                "startup_hold_result": startup_hold_result,
                "motion_result": result,
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
        print(f"最终最大误差:  {result['final_max_error_rad']:.6f} rad")
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
        summary["error"] = "用户中断安全姿态测试。"
        _append_event(events, "interrupted")
        print("\n用户中断安全姿态测试。", file=sys.stderr)
        return 130
    except Exception as exc:
        summary["error"] = repr(exc)
        _append_event(events, "failed", error=repr(exc))
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None and motion_started and not reached_safe_pose:
            try:
                current_message = robot.get_joint_angles()
                current = (
                    None
                    if current_message is None
                    else _finite_vector(current_message.msg, 6)
                )
                if current is not None and joint_limits is not None:
                    hold_command, adjustments = _clamp_joint_vector(
                        current, joint_limits
                    )
                    robot.set_speed_percent(1)
                    robot.move_j(hold_command)
                    _append_event(
                        events,
                        "failure_hold_command_sent",
                        requested_joints=current,
                        command_joints=hold_command,
                        clamp_adjustments_rad=adjustments,
                    )
                    print(
                        "已发送显式限幅后的保持指令；机械臂保持使能，"
                        "请检查现场并准备物理急停。",
                        file=sys.stderr,
                    )
            except Exception as hold_exc:
                summary["failure_hold_error"] = repr(hold_exc)
                print(
                    f"无法发送异常保持指令: {hold_exc}；"
                    "请立即使用物理急停。",
                    file=sys.stderr,
                )
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
    # PiperX安全初始姿态配置区。默认只预检，不会使能或运动。
    # 启用前必须确认从当前姿态到目标姿态的路径不会发生碰撞。
    # =====================================================================
    piperx_safe_pose_config = {
        # 已激活的PiperX SocketCAN接口。
        "channel": "can_piperx",
        "interface": "socketcan",
        "bitrate": 1_000_000,
        # 固件查询和反馈等待参数。
        "startup_timeout_seconds": 5.0,
        "feedback_warmup_seconds": 0.5,
        "can_timeout_seconds": 1.0,
        "sdk_log_level": "WARNING",
        # 六关节目标：None表示保持启动时反馈。
        # J2=0.10、J3=-0.08、J4=-1.48 rad，避免贴近软件限位。
        "target_joint_angles_rad": [None, 0.10, -0.08, -1.48, None, None],
        # 安全姿态只允许1%~3%速度；3%仍为低速，也更容易克服静摩擦。
        "speed_percent": 3,
        # 首次move_j预保持完成后等待控制模式稳定，再发送安全目标。
        "mode_settle_seconds": 0.5,
        # 所有关节目标与SDK限位至少保留0.05 rad安全余量。
        "limit_margin_rad": 0.05,
        # 任一关节单次变化不得超过0.15 rad，程序硬上限为0.20 rad。
        "max_joint_change_rad": 0.15,
        # 连续5帧全部关节误差不超过0.005 rad才判定到达。
        "target_tolerance_rad": 0.005,
        "stable_feedback_count": 5,
        # 反馈允许超出初始值与目标值形成区间的最大瞬时余量。
        "path_slack_rad": 0.02,
        "motion_timeout_seconds": 15.0,
        "monitor_hz": 100.0,
        # 真实使能前倒计时，可随时按Ctrl+C取消。
        "countdown_seconds": 5,
        # 默认成功后保持使能，避免再次下垂到越界姿态。
        "disable_after_success": False,
        # 第一安全门：False时只预检，绝不运动。
        "allow_motion": False,
        # 第二安全门：真实运动时必须完整填写下面的确认词。
        "confirmation": "I_UNDERSTAND_PIPERX_WILL_MOVE_TO_SAFE_POSE",
        # None表示保存到outputs/6_real_robot_eval/piperx_safe_pose_test。
        "output_dir": None,
    }

    raise SystemExit(main(config_defaults=piperx_safe_pose_config))
