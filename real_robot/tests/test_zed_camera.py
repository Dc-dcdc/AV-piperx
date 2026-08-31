#!/usr/bin/env python3
"""ZED Mini双目RGB取流、录像与预处理诊断。

测试目的：
    从同一次ZED ``grab`` 读取左右校正RGB图像，中心裁剪为4:3并缩放到
    640x480，检查实际FPS、左右同步、取流失败、预处理耗时和录像完整性。

运行位置：
    以下命令均在项目根目录 ``/home/dc/dc_project/AV-piper`` 执行。

无GUI、无录像运行15秒，只检查稳定取流和掉帧：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_zed_camera.py \
        --duration-seconds 15 --no-display --no-record-mp4 --no-record-svo

显示左右目预览；按q/Esc退出，按s保存当前左右图：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_zed_camera.py --display --no-record-mp4

录制模型输入尺寸的左右MP4：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_zed_camera.py \
        --duration-seconds 30 --record-mp4 --no-display

同时保存ZED原生双目SVO2：
    /home/dc/miniforge3/envs/AV-piper/bin/python \
        real_robot/tests/test_zed_camera.py \
        --duration-seconds 30 --record-mp4 --record-svo --no-display

配置说明：
    文件底部当前配置为HD720、30 FPS、运行5秒、开启预览和MP4；命令行
    参数可临时覆盖这些值。结果保存在
    ``outputs/6_real_robot_eval/zed_camera_test/<时间戳>/``。

脚本有意延迟导入pyzed和Tk，使未安装相机SDK的训练/测试进程仍可导入
本模块。图像裁剪、缩放、显示和PNG保存均不依赖OpenCV，避免pyzed 5.3
所需NumPy 2与旧版OpenCV wheel之间的ABI冲突。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_OUTPUT_ROOT = Path("outputs/6_real_robot_eval/zed_camera_test")
RESOLUTION_NAMES = ("HD2K", "HD1080", "HD720", "VGA")


@dataclass
class CaptureStatistics:
    """一次相机测试的累计统计。"""

    requested_fps: int
    attempted_grabs: int = 0
    successful_grabs: int = 0
    failed_grabs: int = 0
    timestamp_gap_events: int = 0
    estimated_missing_frames: int = 0
    first_image_timestamp_ns: int | None = None
    last_image_timestamp_ns: int | None = None
    elapsed_seconds: float = 0.0
    achieved_fps: float = 0.0
    mean_preprocess_ms: float = 0.0
    max_preprocess_ms: float = 0.0
    sdk_dropped_frames: int | None = None
    mp4_frames_written: int = 0


def _load_pyzed():
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise RuntimeError(
            "未找到pyzed。请在当前Python环境安装与ZED SDK及Python版本匹配的wheel。"
        ) from exc
    return sl


class _TkPreview:
    """不依赖OpenCV的轻量双目预览窗口。"""

    def __init__(self) -> None:
        try:
            import tkinter as tk
            from PIL import ImageTk
        except ImportError as exc:
            raise RuntimeError(
                "--display需要Tk支持；请安装python3-tk，或去掉--display运行。"
            ) from exc

        self._tk = tk
        self._image_tk = ImageTk
        self._root = tk.Tk()
        self._root.title("ZED Mini | LEFT                           RIGHT")
        self._label = tk.Label(self._root)
        self._label.pack()
        self._photo = None
        self._quit_requested = False
        self._save_requested = False
        self._root.bind("q", self._request_quit)
        self._root.bind("<Escape>", self._request_quit)
        self._root.bind("s", self._request_save)
        self._root.protocol("WM_DELETE_WINDOW", self._request_quit)

    def _request_quit(self, _event=None) -> None:
        self._quit_requested = True

    def _request_save(self, _event=None) -> None:
        self._save_requested = True

    def show(
        self,
        stereo_rgb: np.ndarray,
        *,
        running_fps: float,
        failed_grabs: int,
        estimated_missing_frames: int,
    ) -> tuple[bool, bool]:
        image = Image.fromarray(stereo_rgb)
        draw = ImageDraw.Draw(image)
        text = (
            f"FPS {running_fps:.1f} | fail {failed_grabs} | "
            f"gap {estimated_missing_frames}"
        )
        draw.rectangle((6, 5, 390, 32), fill=(0, 0, 0))
        draw.text((12, 10), text, fill=(0, 255, 0))
        self._photo = self._image_tk.PhotoImage(image=image)
        self._label.configure(image=self._photo)
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._quit_requested = True
        save_requested = self._save_requested
        self._save_requested = False
        return self._quit_requested, save_requested

    def close(self) -> None:
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass


def _zed_bgra_to_rgb(image: np.ndarray) -> np.ndarray:
    """将ZED返回的BGRA/BGR数组转换成连续RGB uint8数组。"""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"ZED图像应为H×W×3/4，实际为{array.shape}。")
    if array.dtype != np.uint8:
        raise ValueError(f"ZED彩色图像应为uint8，实际为{array.dtype}。")
    return np.ascontiguousarray(array[:, :, :3][:, :, ::-1])


def _center_crop_and_resize_rgb(
    image_rgb: np.ndarray,
    *,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """保持比例中心裁剪，再缩放到模型输入尺寸。

    返回 ``(processed_rgb, crop_box)``，crop_box采用Pillow的
    ``(left, top, right, bottom)`` 约定。
    """

    if target_width <= 0 or target_height <= 0:
        raise ValueError("目标图像宽高必须为正整数。")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"输入必须为H×W×3 RGB图像，实际为{image_rgb.shape}。")

    source_height, source_width = image_rgb.shape[:2]
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height

    if source_ratio > target_ratio:
        crop_height = source_height
        crop_width = max(1, round(source_height * target_ratio))
        left = (source_width - crop_width) // 2
        top = 0
    else:
        crop_width = source_width
        crop_height = max(1, round(source_width / target_ratio))
        left = 0
        top = (source_height - crop_height) // 2

    right = left + crop_width
    bottom = top + crop_height
    pil_image = Image.fromarray(image_rgb, mode="RGB")
    processed = pil_image.crop((left, top, right, bottom)).resize(
        (target_width, target_height),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(processed, dtype=np.uint8), (left, top, right, bottom)


def _save_pair(
    *,
    output_dir: Path,
    frame_index: int,
    timestamp_ns: int,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{frame_index:06d}_ts={timestamp_ns}"
    left_path = output_dir / f"{stem}_left.png"
    right_path = output_dir / f"{stem}_right.png"
    Image.fromarray(left_rgb, mode="RGB").save(left_path)
    Image.fromarray(right_rgb, mode="RGB").save(right_path)
    return left_path, right_path


class _FFmpegVideoWriter:
    """通过FFmpeg stdin将RGB帧编码为H.264 MP4。

    不使用OpenCV，避免pyzed 5.3所需NumPy 2与旧版OpenCV wheel的
    ABI冲突。
    """

    def __init__(
        self,
        *,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        ffmpeg_bin: str,
        preset: str,
        crf: int,
    ) -> None:
        executable = shutil.which(ffmpeg_bin)
        if executable is None:
            raise RuntimeError(
                f"未找到FFmpeg可执行文件 {ffmpeg_bin!r}，无法录制MP4。"
            )
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.frames_written = 0
        self._closed = False
        self._log_path = output_path.with_suffix(".ffmpeg.log")
        self._log_file = self._log_path.open("wb")
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(int(fps)),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._log_file,
            )
        except Exception:
            self._log_file.close()
            raise

    def write(self, frame_rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError(f"视频编码器已关闭: {self.output_path}")
        frame = np.asarray(frame_rgb)
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape or frame.dtype != np.uint8:
            raise ValueError(
                f"MP4帧必须为{expected_shape}的uint8 RGB，"
                f"实际为{frame.shape}/{frame.dtype}。"
            )
        if self._process.poll() is not None or self._process.stdin is None:
            raise RuntimeError(
                f"FFmpeg编码器提前退出: {self.output_path}；"
                f"请查看{self._log_path}。"
            )
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(
                f"FFmpeg编码管道已断开: {self.output_path}；"
                f"请查看{self._log_path}。"
            ) from exc
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            return_code = self._process.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._log_file.close()
            raise RuntimeError(f"FFmpeg结束超时: {self.output_path}") from exc
        self._log_file.close()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg编码失败(returncode={return_code}): "
                f"{self.output_path}；请查看{self._log_path}。"
            )


class _StereoMP4Recorder:
    """将同一grab的左右预处理图像写入两个独立MP4。"""

    def __init__(
        self,
        *,
        output_dir: Path,
        width: int,
        height: int,
        fps: int,
        ffmpeg_bin: str,
        preset: str,
        crf: int,
    ) -> None:
        self.left = _FFmpegVideoWriter(
            output_path=output_dir / "left.mp4",
            width=width,
            height=height,
            fps=fps,
            ffmpeg_bin=ffmpeg_bin,
            preset=preset,
            crf=crf,
        )
        try:
            self.right = _FFmpegVideoWriter(
                output_path=output_dir / "right.mp4",
                width=width,
                height=height,
                fps=fps,
                ffmpeg_bin=ffmpeg_bin,
                preset=preset,
                crf=crf,
            )
        except Exception:
            self.left.close()
            raise

    @property
    def frames_written(self) -> int:
        if self.left.frames_written != self.right.frames_written:
            raise RuntimeError(
                "左右MP4帧数不一致: "
                f"left={self.left.frames_written}, right={self.right.frames_written}"
            )
        return self.left.frames_written

    def write(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> None:
        self.left.write(left_rgb)
        self.right.write(right_rgb)

    def close(self) -> None:
        errors: list[Exception] = []
        for writer in (self.left, self.right):
            try:
                writer.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))


def _enum_member(enum_type: Any, name: str, *, kind: str) -> Any:
    try:
        return getattr(enum_type, name)
    except AttributeError as exc:
        raise ValueError(f"当前ZED SDK不支持{kind}={name!r}。") from exc


def _camera_information(camera: Any) -> dict[str, Any]:
    info = camera.get_camera_information()
    configuration = info.camera_configuration
    resolution = configuration.resolution
    return {
        "model": str(info.camera_model),
        "serial_number": int(info.serial_number),
        "firmware_version": str(info.camera_configuration.firmware_version),
        "native_width": int(resolution.width),
        "native_height": int(resolution.height),
        "configured_fps": float(configuration.fps),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="测试ZED Mini左右目同步RGB取流、预处理、FPS和掉帧。"
    )
    parser.add_argument(
        "--resolution",
        choices=RESOLUTION_NAMES,
        default="HD720",
        help="ZED原生采集分辨率，默认HD720。",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="相机采集帧率，ZED Mini的HD720建议30或60，默认30。",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=15.0,
        help="测试时长；0表示一直运行到Ctrl+C或显示窗口中按q/Esc。",
    )
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="使用Pillow/Tk显示左右预处理图；默认关闭。",
    )
    parser.add_argument(
        "--save-first-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="自动保存第一对成功帧，默认开启。",
    )
    parser.add_argument(
        "--record-mp4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="录制预处理后的left.mp4/right.mp4，默认关闭。",
    )
    parser.add_argument(
        "--record-svo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="通过ZED SDK录制原生capture.svo2，默认关闭。",
    )
    parser.add_argument(
        "--svo-compression",
        choices=("H264", "H264_LOSSLESS", "H265", "H265_LOSSLESS", "LOSSLESS"),
        default="H264",
        help="SVO2压缩模式，默认H264。",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="FFmpeg可执行文件名称或路径，默认ffmpeg。",
    )
    parser.add_argument(
        "--mp4-preset",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
        ),
        default="veryfast",
        help="libx264编码预设，越快则CPU开销越低，默认veryfast。",
    )
    parser.add_argument(
        "--mp4-crf",
        type=int,
        default=18,
        help="H.264 CRF质量，越小质量越高，默认18。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果目录；默认写入outputs/6_real_robot_eval/zed_camera_test/时间戳。",
    )
    parser.add_argument(
        "--report-interval-seconds",
        type=float,
        default=2.0,
        help="终端进度打印间隔，默认2秒。",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps必须大于0。")
    if args.duration_seconds < 0:
        raise ValueError("--duration-seconds不能小于0。")
    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("目标宽高必须大于0。")
    if args.report_interval_seconds <= 0:
        raise ValueError("--report-interval-seconds必须大于0。")
    if not 0 <= args.mp4_crf <= 51:
        raise ValueError("--mp4-crf必须在0到51之间。")
    if args.record_mp4 and shutil.which(args.ffmpeg_bin) is None:
        raise ValueError(
            f"--record-mp4需要FFmpeg，但未找到{args.ffmpeg_bin!r}。"
        )


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    sl = _load_pyzed()
    preview = None

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_name)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = _enum_member(
        sl.RESOLUTION, args.resolution, kind="resolution"
    )
    init.camera_fps = int(args.fps)
    init.depth_mode = sl.DEPTH_MODE.NONE

    print("ZED Mini双目相机测试")
    print("-" * 72)
    print(f"SDK:             {sl.Camera.get_sdk_version()}")
    print(f"Capture:         {args.resolution} @ {args.fps} FPS")
    print(f"Model input:     {args.target_width}x{args.target_height} RGB")
    print(f"Display:         {'on' if args.display else 'off'}")
    print(f"Record MP4:      {'on' if args.record_mp4 else 'off'}")
    print(f"Record SVO2:     {'on' if args.record_svo else 'off'}")
    print(f"Output:          {output_dir}")
    print("-" * 72)

    open_status = camera.open(init)
    if open_status != sl.ERROR_CODE.SUCCESS:
        print(f"打开ZED失败: {open_status}", file=sys.stderr)
        camera.close()
        return 2

    svo_recording_enabled = False
    if args.record_svo:
        svo_path = output_dir / "capture.svo2"
        compression_mode = _enum_member(
            sl.SVO_COMPRESSION_MODE,
            args.svo_compression,
            kind="svo_compression",
        )
        recording_parameters = sl.RecordingParameters(
            str(svo_path), compression_mode
        )
        recording_status = camera.enable_recording(recording_parameters)
        if recording_status != sl.ERROR_CODE.SUCCESS:
            camera.close()
            print(f"启用SVO2录制失败: {recording_status}", file=sys.stderr)
            return 2
        svo_recording_enabled = True

    mp4_recorder: _StereoMP4Recorder | None = None
    timestamp_file = None
    try:
        preview = _TkPreview() if args.display else None
        if args.record_mp4:
            mp4_recorder = _StereoMP4Recorder(
                output_dir=output_dir,
                width=args.target_width,
                height=args.target_height,
                fps=args.fps,
                ffmpeg_bin=args.ffmpeg_bin,
                preset=args.mp4_preset,
                crf=args.mp4_crf,
            )
        if args.record_mp4 or args.record_svo:
            timestamp_file = (output_dir / "frame_timestamps.jsonl").open(
                "w", encoding="utf-8"
            )
    except Exception:
        if mp4_recorder is not None:
            mp4_recorder.close()
        if timestamp_file is not None:
            timestamp_file.close()
        if svo_recording_enabled:
            camera.disable_recording()
        camera.close()
        raise

    left_mat = sl.Mat()
    right_mat = sl.Mat()
    stats = CaptureStatistics(requested_fps=int(args.fps))
    preprocess_times_ms: list[float] = []
    previous_timestamp_ns: int | None = None
    expected_period_ns = 1_000_000_000.0 / float(args.fps)
    first_pair_saved = False
    latest_pair: tuple[np.ndarray, np.ndarray, int, int] | None = None
    start_time = time.perf_counter()
    last_report_time = start_time

    try:
        info = _camera_information(camera)
        print(
            "Camera:          "
            f"{info['model']}, SN={info['serial_number']}, "
            f"FW={info['firmware_version']}"
        )
        print(
            "Native stream:   "
            f"{info['native_width']}x{info['native_height']} "
            f"@ {info['configured_fps']:.1f} FPS"
        )
        print("按Ctrl+C退出；显示模式下按q/Esc退出，按s保存当前帧。")

        while True:
            now = time.perf_counter()
            if args.duration_seconds > 0 and now - start_time >= args.duration_seconds:
                break

            stats.attempted_grabs += 1
            grab_status = camera.grab()
            if grab_status != sl.ERROR_CODE.SUCCESS:
                stats.failed_grabs += 1
                if now - last_report_time >= args.report_interval_seconds:
                    print(f"等待图像失败: {grab_status}")
                    last_report_time = now
                continue

            left_status = camera.retrieve_image(left_mat, sl.VIEW.LEFT)
            right_status = camera.retrieve_image(right_mat, sl.VIEW.RIGHT)
            if (
                left_status != sl.ERROR_CODE.SUCCESS
                or right_status != sl.ERROR_CODE.SUCCESS
            ):
                stats.failed_grabs += 1
                print(
                    f"读取左右图失败: left={left_status}, right={right_status}",
                    file=sys.stderr,
                )
                continue

            preprocess_start = time.perf_counter()
            left_rgb = _zed_bgra_to_rgb(left_mat.get_data())
            right_rgb = _zed_bgra_to_rgb(right_mat.get_data())
            left_processed, crop_box_left = _center_crop_and_resize_rgb(
                left_rgb,
                target_width=args.target_width,
                target_height=args.target_height,
            )
            right_processed, crop_box_right = _center_crop_and_resize_rgb(
                right_rgb,
                target_width=args.target_width,
                target_height=args.target_height,
            )
            if crop_box_left != crop_box_right:
                raise RuntimeError(
                    "左右目原始尺寸不同，无法保证使用相同裁剪区域: "
                    f"left={crop_box_left}, right={crop_box_right}"
                )
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
            preprocess_times_ms.append(preprocess_ms)

            timestamp_ns = int(
                camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            )
            if stats.first_image_timestamp_ns is None:
                stats.first_image_timestamp_ns = timestamp_ns
            stats.last_image_timestamp_ns = timestamp_ns
            if previous_timestamp_ns is not None:
                delta_ns = timestamp_ns - previous_timestamp_ns
                estimated_intervals = max(1, round(delta_ns / expected_period_ns))
                missing = max(0, estimated_intervals - 1)
                if missing:
                    stats.timestamp_gap_events += 1
                    stats.estimated_missing_frames += int(missing)
            previous_timestamp_ns = timestamp_ns

            stats.successful_grabs += 1
            frame_index = stats.successful_grabs - 1
            latest_pair = (
                left_processed,
                right_processed,
                frame_index,
                timestamp_ns,
            )

            if mp4_recorder is not None:
                mp4_recorder.write(left_processed, right_processed)
                stats.mp4_frames_written = mp4_recorder.frames_written
            if timestamp_file is not None:
                timestamp_record = {
                    "frame_index": frame_index,
                    "image_timestamp_ns": timestamp_ns,
                    "relative_timestamp_seconds": (
                        timestamp_ns - stats.first_image_timestamp_ns
                    )
                    / 1_000_000_000.0,
                }
                timestamp_file.write(
                    json.dumps(timestamp_record, ensure_ascii=False) + "\n"
                )

            if args.save_first_frame and not first_pair_saved:
                left_path, right_path = _save_pair(
                    output_dir=output_dir,
                    frame_index=frame_index,
                    timestamp_ns=timestamp_ns,
                    left_rgb=left_processed,
                    right_rgb=right_processed,
                )
                print(f"已保存首帧: {left_path.name}, {right_path.name}")
                print(f"Center crop:     {crop_box_left}")
                first_pair_saved = True

            if preview is not None:
                stereo_rgb = np.concatenate(
                    (left_processed, right_processed), axis=1
                )
                elapsed = max(time.perf_counter() - start_time, 1e-9)
                running_fps = stats.successful_grabs / elapsed
                quit_requested, save_requested = preview.show(
                    stereo_rgb,
                    running_fps=running_fps,
                    failed_grabs=stats.failed_grabs,
                    estimated_missing_frames=stats.estimated_missing_frames,
                )
                if quit_requested:
                    break
                if save_requested and latest_pair is not None:
                    pair_left, pair_right, pair_index, pair_timestamp = latest_pair
                    left_path, right_path = _save_pair(
                        output_dir=output_dir,
                        frame_index=pair_index,
                        timestamp_ns=pair_timestamp,
                        left_rgb=pair_left,
                        right_rgb=pair_right,
                    )
                    print(f"已保存: {left_path.name}, {right_path.name}")

            now = time.perf_counter()
            if now - last_report_time >= args.report_interval_seconds:
                elapsed = max(now - start_time, 1e-9)
                running_fps = stats.successful_grabs / elapsed
                print(
                    f"frames={stats.successful_grabs}, fps={running_fps:.2f}, "
                    f"grab_fail={stats.failed_grabs}, "
                    f"estimated_missing={stats.estimated_missing_frames}, "
                    f"preprocess={preprocess_ms:.2f} ms"
                )
                last_report_time = now
    except KeyboardInterrupt:
        print("\n收到Ctrl+C，结束测试。")
    finally:
        stats.elapsed_seconds = max(time.perf_counter() - start_time, 0.0)
        if stats.elapsed_seconds > 0:
            stats.achieved_fps = stats.successful_grabs / stats.elapsed_seconds
        if preprocess_times_ms:
            stats.mean_preprocess_ms = float(np.mean(preprocess_times_ms))
            stats.max_preprocess_ms = float(np.max(preprocess_times_ms))
        try:
            stats.sdk_dropped_frames = int(camera.get_frame_dropped_count())
        except Exception:
            stats.sdk_dropped_frames = None
        recording_errors: list[Exception] = []
        if timestamp_file is not None:
            timestamp_file.close()
        if mp4_recorder is not None:
            try:
                mp4_recorder.close()
            except Exception as exc:
                recording_errors.append(exc)
        if svo_recording_enabled:
            try:
                camera.disable_recording()
            except Exception as exc:
                recording_errors.append(exc)
        camera.close()
        if preview is not None:
            preview.close()
        if recording_errors:
            raise RuntimeError(
                "关闭录像时出错: "
                + "; ".join(str(error) for error in recording_errors)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "zed_sdk_version": str(sl.Camera.get_sdk_version()),
        "camera": info,
        "configuration": {
            "resolution": args.resolution,
            "fps": int(args.fps),
            "target_width": int(args.target_width),
            "target_height": int(args.target_height),
            "center_crop": True,
            "display": bool(args.display),
        },
        "recording": {
            "mp4_enabled": bool(args.record_mp4),
            "mp4_codec": "libx264" if args.record_mp4 else None,
            "mp4_pixel_format": "yuv420p" if args.record_mp4 else None,
            "mp4_preset": args.mp4_preset if args.record_mp4 else None,
            "mp4_crf": int(args.mp4_crf) if args.record_mp4 else None,
            "left_mp4": str(output_dir / "left.mp4") if args.record_mp4 else None,
            "right_mp4": str(output_dir / "right.mp4") if args.record_mp4 else None,
            "svo_enabled": bool(args.record_svo),
            "svo_compression": args.svo_compression if args.record_svo else None,
            "svo_path": str(output_dir / "capture.svo2") if args.record_svo else None,
            "timestamps_path": (
                str(output_dir / "frame_timestamps.jsonl")
                if args.record_mp4 or args.record_svo
                else None
            ),
        },
        "statistics": asdict(stats),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("-" * 72)
    print(f"成功帧:          {stats.successful_grabs}")
    print(f"失败grab:        {stats.failed_grabs}")
    print(f"实际FPS:         {stats.achieved_fps:.2f}")
    print(f"时间戳估计丢帧:  {stats.estimated_missing_frames}")
    print(f"SDK dropped:     {stats.sdk_dropped_frames}")
    print(f"平均预处理:      {stats.mean_preprocess_ms:.2f} ms")
    print(f"最大预处理:      {stats.max_preprocess_ms:.2f} ms")
    if args.record_mp4:
        print(f"MP4录制帧数:     {stats.mp4_frames_written}")
        print(f"MP4视频:         {output_dir / 'left.mp4'}")
        print(f"                 {output_dir / 'right.mp4'}")
    if args.record_svo:
        print(f"SVO2录像:        {output_dir / 'capture.svo2'}")
    if args.record_mp4 or args.record_svo:
        print(f"帧时间戳:        {output_dir / 'frame_timestamps.jsonl'}")
    print(f"统计文件:        {summary_path}")

    if stats.successful_grabs == 0:
        return 3
    return 0


def main(config_defaults: dict[str, Any] | None = None) -> int:
    parser = _build_parser()
    if config_defaults:
        valid_destinations = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_defaults) - valid_destinations)
        if unknown_keys:
            raise ValueError(f"入口配置包含未知参数: {unknown_keys}")
        # 脚本底部配置作为默认值，命令行显式参数仍可临时覆盖。
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    try:
        return run(args)
    except (RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # =====================================================================
    # ZED Mini测试配置区：日常使用时直接修改下列参数。
    # 命令行参数的优先级更高，例如 --duration-seconds 300 会覆盖此处设置。
    # =====================================================================
    camera_test_config = {
        # ZED原始取流分辨率；ZED Mini建议使用HD720。
        "resolution": "HD720",
        # 相机采集帧率；需与后续实机采集频率匹配。
        "fps": 30,
        # 测试时长，单位为秒；设为0时持续运行到手动退出。
        "duration_seconds": 5.0,

        # 左右目中心裁剪和缩放后的模型输入宽度。
        "target_width": 640,
        # 左右目中心裁剪和缩放后的模型输入高度。
        "target_height": 480,

        # True打开左右目实时预览；无显示器或长时测试时建议False。
        "display": True,
        # True自动保存第一对处理后的左右RGB图像。
        "save_first_frame": True,

        # True分别录制处理后的left.mp4和right.mp4。
        "record_mp4": True,
        # True通过ZED SDK录制包含原始双目数据的capture.svo2。
        "record_svo": False,
        # SVO2压缩模式；H264通用且文件较小，LOSSLESS质量最高但占用空间大。
        "svo_compression": "H264",

        # FFmpeg可执行文件名称或绝对路径。
        "ffmpeg_bin": "ffmpeg",
        # libx264编码速度预设；veryfast较适合实时双目录制。
        "mp4_preset": "veryfast",
        # MP4画质参数，越小质量越高且文件越大；建议18。
        "mp4_crf": 18,

        # None表示按时间戳自动创建输出目录；也可改为Path("...")。
        "output_dir": None,
        # 终端输出取流速度、丢帧和预处理耗时的时间间隔。
        "report_interval_seconds": 2.0,
    }

    raise SystemExit(main(config_defaults=camera_test_config))
