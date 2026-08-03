#!/usr/bin/env python

"""将主动视觉关节轨迹通过MuJoCo正向运动学绘制为真实3D相机轨迹。

输入是 ``augment_view_joint_trajectories.py`` 生成的raw run。脚本按
``info.json`` 中的 ``source_episode`` 和 ``variant_index`` 自动匹配原始
轨迹与固定View关节偏移变体，并输出：

* ``view_camera_trajectories_3d.png``：双目中心3D轨迹、相机光轴及笛卡尔偏差；
* ``view_joint_offsets.png``：六个View关节相对原轨迹的偏移；
* ``trajectory_points.csv``：逐帧相机世界坐标与相对误差；
* ``trajectory_summary.csv/json``：每条变体的汇总统计。

固定的是关节偏移 ``q_aug(t)-q_src(t)=delta``。由于机械臂雅可比随姿态
变化，相机笛卡尔偏移通常不是固定平移，这正是本脚本需要展示的现象。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Matplotlib只用于高质量可选路径。训练服务器通常没有安装它，因此不能在
# 模块导入阶段形成硬依赖；缺失时下方会自动使用Pillow绘制同名PNG。
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]  # noqa: E402
except Exception as matplotlib_error:  # pragma: no cover - 具体异常取决于安装状态。
    plt = None
    _MATPLOTLIB_IMPORT_ERROR: Exception | None = matplotlib_error
else:
    _MATPLOTLIB_IMPORT_ERROR = None


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_collect.augment_view_joint_trajectories import (  # noqa: E402
    _configure_mujoco_runtime,
    _make_environment,
    _restore_initial_state,
)


LEFT_CAMERA = "zed_cam_left"
RIGHT_CAMERA = "zed_cam_right"
VIEW_JOINT_NAMES = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_1",
    "wrist_2",
)


@dataclass(frozen=True)
class EpisodeData:
    directory: Path
    info: dict[str, Any]
    arrays: dict[str, np.ndarray]
    states: np.ndarray
    source_episode: int
    variant_index: int
    offset_rad: np.ndarray
    fps: float

    @property
    def name(self) -> str:
        return self.directory.name


@dataclass(frozen=True)
class CameraTrajectory:
    episode: EpisodeData
    frame_indices: np.ndarray
    time_s: np.ndarray
    view_q: np.ndarray
    left_position: np.ndarray
    right_position: np.ndarray
    center_position: np.ndarray
    center_rotation: np.ndarray
    forward: np.ndarray
    stereo_baseline_m: np.ndarray


def _absolute_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"JSON根节点必须是对象: {path}")
    return value


def _scan_episode_groups(
    run_dir: Path,
) -> dict[int, dict[int, tuple[Path, dict[str, Any]]]]:
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"找不到episodes目录: {episodes_dir}")

    groups: dict[int, dict[int, tuple[Path, dict[str, Any]]]] = {}
    for info_path in sorted(episodes_dir.glob("*/info.json")):
        info = _read_json(info_path)
        if "source_episode" not in info or "variant_index" not in info:
            logging.warning("忽略缺少source_episode/variant_index的episode: %s", info_path.parent)
            continue
        source_episode = int(info["source_episode"])
        variant_index = int(info["variant_index"])
        source_group = groups.setdefault(source_episode, {})
        if variant_index in source_group:
            raise ValueError(
                "同一source存在重复variant: "
                f"source={source_episode}, variant={variant_index}, "
                f"paths={[source_group[variant_index][0], info_path.parent]}"
            )
        source_group[variant_index] = (info_path.parent, info)

    if not groups:
        raise RuntimeError(f"没有找到可用的增强episode: {episodes_dir}")
    return groups


def _resolve_source_episode(
    groups: dict[int, dict[int, tuple[Path, dict[str, Any]]]],
    configured_source: Any,
) -> int:
    if configured_source is not None:
        source_episode = int(configured_source)
        if source_episode not in groups:
            raise KeyError(
                f"source_episode={source_episode}不存在；可用编号为{sorted(groups)}"
            )
        if -1 not in groups[source_episode]:
            raise KeyError(f"source_episode={source_episode}缺少原始variant=-1。")
        return source_episode

    candidates = [
        source
        for source, variants in groups.items()
        if -1 in variants and any(index >= 0 for index in variants)
    ]
    if not candidates:
        raise RuntimeError("没有同时包含原始轨迹和增强变体的source_episode。")
    return min(candidates)


def _resolve_variants(
    group: dict[int, tuple[Path, dict[str, Any]]],
    configured_variants: Any,
) -> list[int]:
    available = sorted(index for index in group if index >= 0)
    if configured_variants is None:
        selected = available
    else:
        selected = [int(index) for index in OmegaConf.to_container(configured_variants)]
    if not selected:
        raise ValueError("至少需要选择一个增强variant。")
    if len(set(selected)) != len(selected):
        raise ValueError(f"variant_indices包含重复值: {selected}")
    missing = [index for index in selected if index not in group]
    if missing:
        raise KeyError(f"缺少variant={missing}；可用增强variant为{available}")
    return selected


def _load_episode(
    directory: Path,
    info: dict[str, Any],
    *,
    state_key: str,
    view_start: int,
    view_dim: int,
    default_fps: float,
) -> EpisodeData:
    arrays_path = directory / "arrays.npz"
    if not arrays_path.is_file():
        raise FileNotFoundError(f"找不到arrays.npz: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if state_key not in arrays:
        raise KeyError(f"{arrays_path}缺少状态字段{state_key!r}，现有字段={sorted(arrays)}")

    states = np.asarray(arrays[state_key], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < view_start + view_dim:
        raise ValueError(
            f"{arrays_path}:{state_key}形状无法读取View切片"
            f"[{view_start}:{view_start + view_dim}]，当前={states.shape}"
        )
    if not np.isfinite(states).all():
        raise ValueError(f"{arrays_path}:{state_key}包含NaN或Inf。")

    offset = np.asarray(info.get("view_joint_offset_rad", np.zeros(view_dim)), dtype=np.float64)
    if offset.shape != (view_dim,) or not np.isfinite(offset).all():
        raise ValueError(
            f"{directory.name}的view_joint_offset_rad应为({view_dim},)，当前={offset}"
        )

    return EpisodeData(
        directory=directory,
        info=info,
        arrays=arrays,
        states=states,
        source_episode=int(info["source_episode"]),
        variant_index=int(info["variant_index"]),
        offset_rad=offset,
        fps=float(info.get("fps", default_fps)),
    )


def _validate_paired_episodes(
    source: EpisodeData,
    variants: list[EpisodeData],
    *,
    view_start: int,
    view_dim: int,
) -> None:
    if source.variant_index != -1:
        raise ValueError(f"原始轨迹variant_index必须为-1，当前={source.variant_index}")
    if not np.allclose(source.offset_rad, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"原始轨迹记录了非零View偏移: {source.offset_rad}")

    source_view = source.states[:, view_start : view_start + view_dim]
    for variant in variants:
        if variant.source_episode != source.source_episode:
            raise ValueError(
                f"{variant.name}的source_episode={variant.source_episode}，"
                f"与原始轨迹{source.source_episode}不一致。"
            )
        if variant.states.shape != source.states.shape:
            raise ValueError(
                f"{variant.name}与原始轨迹状态形状不一致: "
                f"{variant.states.shape} vs {source.states.shape}"
            )
        measured_offset = (
            variant.states[:, view_start : view_start + view_dim] - source_view
        )
        max_error = float(np.max(np.abs(measured_offset - variant.offset_rad[None, :])))
        if max_error > 1e-5:
            raise ValueError(
                f"{variant.name}不是严格固定View偏移，"
                f"相对info记录的最大误差={max_error:.6g}rad。"
            )


def _frame_indices(frame_count: int, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError(f"frame_stride必须为正整数，当前={stride}")
    indices = np.arange(0, frame_count, stride, dtype=np.int64)
    if indices.size == 0 or indices[-1] != frame_count - 1:
        indices = np.append(indices, frame_count - 1)
    return indices


def _compute_camera_trajectory(
    env_obj,
    episode: EpisodeData,
    *,
    view_start: int,
    view_dim: int,
    frame_stride: int,
) -> CameraTrajectory:
    physics = env_obj._physics
    left_id = physics.model.name2id(LEFT_CAMERA, "camera")
    right_id = physics.model.name2id(RIGHT_CAMERA, "camera")
    if left_id < 0 or right_id < 0:
        raise KeyError(f"MuJoCo模型缺少{LEFT_CAMERA}/{RIGHT_CAMERA}。")

    indices = _frame_indices(len(episode.states), frame_stride)
    view_q = episode.states[indices, view_start : view_start + view_dim]
    count = len(indices)
    left_position = np.empty((count, 3), dtype=np.float64)
    right_position = np.empty((count, 3), dtype=np.float64)
    center_position = np.empty((count, 3), dtype=np.float64)
    center_rotation = np.empty((count, 3, 3), dtype=np.float64)
    forward = np.empty((count, 3), dtype=np.float64)
    stereo_baseline = np.empty(count, dtype=np.float64)

    bound_joints = physics.bind(env_obj._middle_joints)
    bound_actuators = physics.bind(env_obj._middle_actuators)
    for output_index, q_view in enumerate(view_q):
        bound_joints.qpos = q_view
        bound_actuators.ctrl = q_view
        physics.forward()

        left_pos = np.asarray(physics.data.cam_xpos[left_id], dtype=np.float64).copy()
        right_pos = np.asarray(physics.data.cam_xpos[right_id], dtype=np.float64).copy()
        left_rot = np.asarray(
            physics.data.cam_xmat[left_id], dtype=np.float64
        ).reshape(3, 3).copy()
        right_rot = np.asarray(
            physics.data.cam_xmat[right_id], dtype=np.float64
        ).reshape(3, 3).copy()

        left_position[output_index] = left_pos
        right_position[output_index] = right_pos
        center_position[output_index] = 0.5 * (left_pos + right_pos)
        center_rotation[output_index] = left_rot
        direction = -0.5 * (left_rot[:, 2] + right_rot[:, 2])
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise RuntimeError(f"{episode.name} frame={indices[output_index]}相机光轴无效。")
        forward[output_index] = direction / norm
        stereo_baseline[output_index] = np.linalg.norm(left_pos - right_pos)

    return CameraTrajectory(
        episode=episode,
        frame_indices=indices,
        time_s=indices.astype(np.float64) / episode.fps,
        view_q=view_q,
        left_position=left_position,
        right_position=right_position,
        center_position=center_position,
        center_rotation=center_rotation,
        forward=forward,
        stereo_baseline_m=stereo_baseline,
    )


def _rotation_error_deg(
    reference_rotation: np.ndarray,
    target_rotation: np.ndarray,
) -> np.ndarray:
    relative = np.matmul(np.swapaxes(reference_rotation, -1, -2), target_rotation)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _path_length_m(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _trajectory_metrics(
    trajectory: CameraTrajectory,
    source: CameraTrajectory,
) -> dict[str, Any]:
    delta = trajectory.center_position - source.center_position
    translation_m = np.linalg.norm(delta, axis=1)
    rotation_deg = _rotation_error_deg(
        source.center_rotation,
        trajectory.center_rotation,
    )
    measured_joint_offset = trajectory.view_q - source.view_q
    offset_error = measured_joint_offset - trajectory.episode.offset_rad[None, :]
    return {
        "episode_name": trajectory.episode.name,
        "source_episode": trajectory.episode.source_episode,
        "variant_index": trajectory.episode.variant_index,
        "view_joint_offset_rad": trajectory.episode.offset_rad.tolist(),
        "view_joint_offset_l2_rad": float(np.linalg.norm(trajectory.episode.offset_rad)),
        "fixed_joint_offset_max_error_rad": float(np.max(np.abs(offset_error))),
        "frames_used": int(len(trajectory.frame_indices)),
        "camera_path_length_m": _path_length_m(trajectory.center_position),
        "translation_start_cm": float(translation_m[0] * 100.0),
        "translation_mean_cm": float(translation_m.mean() * 100.0),
        "translation_rms_cm": float(np.sqrt(np.mean(np.square(translation_m))) * 100.0),
        "translation_max_cm": float(translation_m.max() * 100.0),
        "cartesian_delta_std_cm": (delta.std(axis=0) * 100.0).tolist(),
        "rotation_start_deg": float(rotation_deg[0]),
        "rotation_mean_deg": float(rotation_deg.mean()),
        "rotation_max_deg": float(rotation_deg.max()),
        "stereo_baseline_mean_cm": float(trajectory.stereo_baseline_m.mean() * 100.0),
        "stereo_baseline_std_mm": float(trajectory.stereo_baseline_m.std() * 1000.0),
    }


def _set_equal_3d_axes(axis, points: np.ndarray) -> float:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    span = float(np.max(maximum - minimum))
    span = max(span, 0.05)
    radius = 0.56 * span
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    return span


def _sample_plot_indices(length: int, count: int) -> np.ndarray:
    count = max(0, min(int(count), length))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, count, dtype=np.int64))


_PILLOW_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)


def _load_pillow_modules():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        detail = (
            "Matplotlib不可用，同时也没有安装Pillow，无法输出PNG。"
            "请安装matplotlib或Pillow。"
        )
        if _MATPLOTLIB_IMPORT_ERROR is not None:
            detail += f" Matplotlib导入错误: {_MATPLOTLIB_IMPORT_ERROR}"
        raise RuntimeError(detail) from error
    return Image, ImageDraw, ImageFont


def _pillow_font(ImageFont, size: int, *, bold: bool = False):
    font_names = (
        ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")
    )
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, max(8, int(size)))
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=max(8, int(size)))
    except TypeError:  # Pillow<10不支持load_default(size=...)。
        return ImageFont.load_default()


def _blend_with_white(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return tuple(
        int(round(alpha * channel + (1.0 - alpha) * 255.0))
        for channel in color
    )


def _draw_centered_text(draw, xy, text: str, *, font, fill=(20, 20, 20)) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor="mm")


def _draw_dashed_line(
    draw,
    start: np.ndarray,
    end: np.ndarray,
    *,
    fill,
    width: int,
    dash_px: float,
) -> None:
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return
    direction = delta / length
    position = 0.0
    while position < length:
        segment_end = min(position + dash_px, length)
        p0 = np.asarray(start, dtype=np.float64) + position * direction
        p1 = np.asarray(start, dtype=np.float64) + segment_end * direction
        draw.line(
            [tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int))],
            fill=fill,
            width=width,
        )
        position += 2.0 * dash_px


def _draw_arrow_2d(
    draw,
    start: np.ndarray,
    end: np.ndarray,
    *,
    fill,
    width: int,
    head_px: float,
) -> None:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-6:
        return
    draw.line(
        [tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int))],
        fill=fill,
        width=width,
    )
    direction = delta / length
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float64)
    head = min(float(head_px), max(3.0, 0.42 * length))
    base = end - head * direction
    wing = 0.46 * head
    draw.polygon(
        [
            tuple(np.rint(end).astype(int)),
            tuple(np.rint(base + wing * perpendicular).astype(int)),
            tuple(np.rint(base - wing * perpendicular).astype(int)),
        ],
        fill=fill,
    )


def _draw_star(draw, center: np.ndarray, radius: float, *, fill, outline) -> None:
    center = np.asarray(center, dtype=np.float64)
    points = []
    for index in range(10):
        angle = -0.5 * math.pi + index * math.pi / 5.0
        current_radius = radius if index % 2 == 0 else 0.43 * radius
        points.append(
            (
                int(round(center[0] + current_radius * math.cos(angle))),
                int(round(center[1] + current_radius * math.sin(angle))),
            )
        )
    draw.polygon(points, fill=fill, outline=outline)


def _orthographic_projector(
    points: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    elevation_deg: float,
    azimuth_deg: float,
) -> Callable[[np.ndarray], np.ndarray]:
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    screen_x_axis = np.array(
        [-math.sin(azimuth), math.cos(azimuth), 0.0],
        dtype=np.float64,
    )
    screen_y_axis = np.array(
        [
            -math.cos(azimuth) * math.sin(elevation),
            -math.sin(azimuth) * math.sin(elevation),
            math.cos(elevation),
        ],
        dtype=np.float64,
    )

    def raw_project(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.stack(
            (values @ screen_x_axis, values @ screen_y_axis),
            axis=-1,
        )

    projected = raw_project(points)
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    left, top, right, bottom = (float(value) for value in box)
    scale = min((right - left) / span[0], (bottom - top) / span[1])
    projected_center = 0.5 * (minimum + maximum)
    image_center = np.array(
        [0.5 * (left + right), 0.5 * (top + bottom)],
        dtype=np.float64,
    )

    def project(values: np.ndarray) -> np.ndarray:
        output = raw_project(values)
        output = (output - projected_center) * scale
        output[..., 1] *= -1.0
        return output + image_center

    return project


def _equal_cube(points: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    span = max(float(np.max(maximum - minimum)), 0.05)
    radius = 0.56 * span
    low = center - radius
    high = center + radius
    corners = np.asarray(
        [
            [
                high[0] if mask & 1 else low[0],
                high[1] if mask & 2 else low[1],
                high[2] if mask & 4 else low[2],
            ]
            for mask in range(8)
        ],
        dtype=np.float64,
    )
    edges = [
        (first, second)
        for first in range(8)
        for second in range(first + 1, 8)
        if bin(first ^ second).count("1") == 1
    ]
    return corners, edges


def _format_tick(value: float) -> str:
    absolute = abs(float(value))
    if absolute >= 100.0:
        return f"{value:.0f}"
    if absolute >= 10.0:
        return f"{value:.1f}"
    if absolute >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _draw_pillow_line_chart(
    draw,
    box: tuple[int, int, int, int],
    series: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    unit: float,
    ImageFont,
    y_limits: tuple[float, float] | None = None,
) -> None:
    left, top, right, bottom = box
    margin_left = int(64 * unit)
    margin_right = int(18 * unit)
    margin_top = int(38 * unit)
    margin_bottom = int(48 * unit)
    plot_left = left + margin_left
    plot_right = right - margin_right
    plot_top = top + margin_top
    plot_bottom = bottom - margin_bottom

    x_values = np.concatenate([values[0] for values in series])
    y_values = np.concatenate([values[1] for values in series])
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_limits is None:
        y_min = min(0.0, float(y_values.min()))
        y_max = max(0.0, float(y_values.max()))
        padding = max(0.05 * (y_max - y_min), 1e-4)
        y_min -= padding
        y_max += padding
    else:
        y_min, y_max = (float(value) for value in y_limits)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def to_pixels(x_values_: np.ndarray, y_values_: np.ndarray) -> np.ndarray:
        x_pixel = plot_left + (x_values_ - x_min) / (x_max - x_min) * (
            plot_right - plot_left
        )
        y_pixel = plot_bottom - (y_values_ - y_min) / (y_max - y_min) * (
            plot_bottom - plot_top
        )
        return np.stack((x_pixel, y_pixel), axis=-1)

    tick_font = _pillow_font(ImageFont, int(10 * unit))
    label_font = _pillow_font(ImageFont, int(12 * unit))
    title_font = _pillow_font(ImageFont, int(14 * unit), bold=True)
    grid_color = (220, 224, 230)
    for tick_index in range(6):
        fraction = tick_index / 5.0
        x_pixel = plot_left + fraction * (plot_right - plot_left)
        y_pixel = plot_bottom - fraction * (plot_bottom - plot_top)
        draw.line(
            [(int(x_pixel), plot_top), (int(x_pixel), plot_bottom)],
            fill=grid_color,
            width=max(1, int(unit)),
        )
        draw.line(
            [(plot_left, int(y_pixel)), (plot_right, int(y_pixel))],
            fill=grid_color,
            width=max(1, int(unit)),
        )
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        draw.text(
            (int(x_pixel), plot_bottom + int(7 * unit)),
            _format_tick(x_value),
            fill=(65, 65, 65),
            font=tick_font,
            anchor="ma",
        )
        draw.text(
            (plot_left - int(7 * unit), int(y_pixel)),
            _format_tick(y_value),
            fill=(65, 65, 65),
            font=tick_font,
            anchor="rm",
        )

    if y_min < 0.0 < y_max:
        zero_y = int(to_pixels(np.asarray([x_min]), np.asarray([0.0]))[0, 1])
        draw.line(
            [(plot_left, zero_y), (plot_right, zero_y)],
            fill=(120, 120, 120),
            width=max(1, int(1.4 * unit)),
        )
    draw.rectangle(
        [plot_left, plot_top, plot_right, plot_bottom],
        outline=(95, 100, 110),
        width=max(1, int(unit)),
    )
    for x_values_, y_values_, color in series:
        pixels = to_pixels(x_values_, y_values_)
        draw.line(
            [tuple(np.rint(point).astype(int)) for point in pixels],
            fill=color,
            width=max(2, int(1.8 * unit)),
            joint="curve",
        )

    _draw_centered_text(
        draw,
        (0.5 * (left + right), top + int(13 * unit)),
        title,
        font=title_font,
    )
    _draw_centered_text(
        draw,
        (0.5 * (plot_left + plot_right), bottom - int(10 * unit)),
        x_label,
        font=label_font,
        fill=(55, 55, 55),
    )
    draw.text(
        (left + int(3 * unit), plot_top - int(2 * unit)),
        y_label,
        font=label_font,
        fill=(55, 55, 55),
        anchor="la",
    )


def _plot_camera_trajectories_pillow(
    output_path: Path,
    source: CameraTrajectory,
    variants: list[CameraTrajectory],
    cfg_plot: DictConfig,
    env_id: str,
) -> None:
    Image, ImageDraw, ImageFont = _load_pillow_modules()
    dpi = max(72, int(cfg_plot.dpi))
    unit = max(1.0, dpi / 120.0)
    width = int(round(16.5 * dpi))
    height = int(round(8.5 * dpi))
    image = Image.new("RGB", (width, height), (250, 251, 253))
    draw = ImageDraw.Draw(image)

    coordinate_frame = str(cfg_plot.coordinate_frame)
    if coordinate_frame not in {"world", "source_start"}:
        raise ValueError(
            "plot.coordinate_frame必须是world或source_start，"
            f"当前={coordinate_frame!r}"
        )
    origin = (
        source.center_position[0]
        if coordinate_frame == "source_start"
        else np.zeros(3, dtype=np.float64)
    )
    entries: list[
        tuple[CameraTrajectory, tuple[int, int, int], str, int]
    ] = [(source, (15, 15, 18), "original", max(3, int(2.7 * unit)))]
    entries.extend(
        (
            trajectory,
            _PILLOW_COLORS[index % len(_PILLOW_COLORS)],
            (
                f"aug_{trajectory.episode.variant_index:02d} "
                f"(|dq|={np.linalg.norm(trajectory.episode.offset_rad):.3f} rad)"
            ),
            max(2, int(1.8 * unit)),
        )
        for index, trajectory in enumerate(variants)
    )

    centers = [
        trajectory.center_position - origin
        for trajectory, _, _, _ in entries
    ]
    center_points = np.concatenate(centers, axis=0)
    spatial_span = max(
        float(np.max(center_points.max(axis=0) - center_points.min(axis=0))),
        0.05,
    )
    configured_arrow_length = cfg_plot.get("arrow_length_m")
    arrow_length = (
        max(0.012, 0.10 * spatial_span)
        if configured_arrow_length is None
        else float(configured_arrow_length)
    )
    if arrow_length <= 0:
        raise ValueError("plot.arrow_length_m必须为正数或null。")

    extent_parts = list(centers)
    arrow_count = int(cfg_plot.arrow_count)
    show_stereo = bool(cfg_plot.show_stereo_rails)
    for trajectory, _, _, _ in entries:
        arrow_indices = _sample_plot_indices(
            len(trajectory.center_position),
            arrow_count,
        )
        extent_parts.append(
            trajectory.center_position[arrow_indices]
            - origin
            + arrow_length * trajectory.forward[arrow_indices]
        )
        if show_stereo:
            extent_parts.extend(
                (
                    trajectory.left_position - origin,
                    trajectory.right_position - origin,
                )
            )
    cube, cube_edges = _equal_cube(np.concatenate(extent_parts, axis=0))

    title_height = int(58 * unit)
    left_box = (
        int(26 * unit),
        title_height + int(20 * unit),
        int(0.64 * width) - int(14 * unit),
        height - int(24 * unit),
    )
    right_left = int(0.65 * width)
    right_box_top = (
        right_left,
        title_height + int(16 * unit),
        width - int(24 * unit),
        int(0.51 * height),
    )
    right_box_bottom = (
        right_left,
        int(0.52 * height),
        width - int(24 * unit),
        height - int(24 * unit),
    )
    projection = _orthographic_projector(
        cube,
        left_box,
        elevation_deg=float(cfg_plot.elevation_deg),
        azimuth_deg=float(cfg_plot.azimuth_deg),
    )

    title_font = _pillow_font(ImageFont, int(18 * unit), bold=True)
    subtitle_font = _pillow_font(ImageFont, int(11 * unit))
    legend_font = _pillow_font(ImageFont, int(10 * unit))
    axis_font = _pillow_font(ImageFont, int(11 * unit), bold=True)
    _draw_centered_text(
        draw,
        (0.5 * width, int(17 * unit)),
        (
            "Fixed View-joint Offset Effect | "
            f"source={source.episode.source_episode:06d} | {env_id}"
        ),
        font=title_font,
    )
    _draw_centered_text(
        draw,
        (0.32 * width, title_height),
        "Orthographic camera trajectories (circle=start, star=end, arrows=optical axis)",
        font=subtitle_font,
        fill=(65, 65, 65),
    )

    projected_cube = projection(cube)
    for first, second in cube_edges:
        draw.line(
            [
                tuple(np.rint(projected_cube[first]).astype(int)),
                tuple(np.rint(projected_cube[second]).astype(int)),
            ],
            fill=(205, 210, 218),
            width=max(1, int(unit)),
        )
    low_corner = cube[0]
    for endpoint_index, label, color in (
        (1, "X", (180, 45, 45)),
        (2, "Y", (45, 145, 55)),
        (4, "Z", (45, 85, 185)),
    ):
        start_pixel = projected_cube[0]
        end_pixel = projected_cube[endpoint_index]
        _draw_arrow_2d(
            draw,
            start_pixel,
            end_pixel,
            fill=_blend_with_white(color, 0.65),
            width=max(1, int(1.2 * unit)),
            head_px=8 * unit,
        )
        draw.text(
            tuple(np.rint(end_pixel).astype(int)),
            label,
            fill=color,
            font=axis_font,
            anchor="mm",
        )
    del low_corner

    connector_indices = _sample_plot_indices(
        len(source.center_position),
        int(cfg_plot.connector_count),
    )
    source_center = source.center_position - origin
    for variant_index, trajectory in enumerate(variants):
        variant_center = trajectory.center_position - origin
        color = _PILLOW_COLORS[variant_index % len(_PILLOW_COLORS)]
        for index in connector_indices:
            endpoints = projection(
                np.stack((source_center[index], variant_center[index]))
            )
            _draw_dashed_line(
                draw,
                endpoints[0],
                endpoints[1],
                fill=_blend_with_white(color, 0.48),
                width=max(1, int(unit)),
                dash_px=5.0 * unit,
            )

    for trajectory, color, _, line_width in entries:
        center = trajectory.center_position - origin
        arrow_indices = _sample_plot_indices(len(center), arrow_count)
        if show_stereo:
            left = trajectory.left_position - origin
            right = trajectory.right_position - origin
            rail_color = _blend_with_white(color, 0.36)
            draw.line(
                [tuple(np.rint(point).astype(int)) for point in projection(left)],
                fill=rail_color,
                width=max(1, int(0.7 * unit)),
                joint="curve",
            )
            draw.line(
                [tuple(np.rint(point).astype(int)) for point in projection(right)],
                fill=rail_color,
                width=max(1, int(0.7 * unit)),
                joint="curve",
            )
            for index in arrow_indices:
                baseline = projection(np.stack((left[index], right[index])))
                draw.line(
                    [
                        tuple(np.rint(baseline[0]).astype(int)),
                        tuple(np.rint(baseline[1]).astype(int)),
                    ],
                    fill=rail_color,
                    width=max(1, int(0.7 * unit)),
                )

        center_pixels = projection(center)
        draw.line(
            [tuple(np.rint(point).astype(int)) for point in center_pixels],
            fill=color,
            width=line_width,
            joint="curve",
        )
        marker_radius = 4.0 * unit
        start = center_pixels[0]
        draw.ellipse(
            [
                int(start[0] - marker_radius),
                int(start[1] - marker_radius),
                int(start[0] + marker_radius),
                int(start[1] + marker_radius),
            ],
            fill=color,
            outline=(255, 255, 255),
            width=max(1, int(unit)),
        )
        _draw_star(
            draw,
            center_pixels[-1],
            7.0 * unit,
            fill=color,
            outline=(255, 255, 255),
        )
        for index in arrow_indices:
            arrow_world = np.stack(
                (
                    center[index],
                    center[index] + arrow_length * trajectory.forward[index],
                )
            )
            arrow_pixels = projection(arrow_world)
            _draw_arrow_2d(
                draw,
                arrow_pixels[0],
                arrow_pixels[1],
                fill=_blend_with_white(color, 0.80),
                width=max(1, int(1.1 * unit)),
                head_px=7.0 * unit,
            )

    legend_x = left_box[0] + int(13 * unit)
    legend_y = left_box[1] + int(12 * unit)
    legend_width = int(240 * unit)
    legend_height = int((13 + 19 * len(entries)) * unit)
    draw.rounded_rectangle(
        [
            legend_x,
            legend_y,
            legend_x + legend_width,
            legend_y + legend_height,
        ],
        radius=int(5 * unit),
        fill=(248, 249, 251),
        outline=(190, 195, 205),
        width=max(1, int(unit)),
    )
    for index, (_, color, label, line_width) in enumerate(entries):
        row_y = legend_y + int((13 + 19 * index) * unit)
        draw.line(
            [
                (legend_x + int(8 * unit), row_y),
                (legend_x + int(31 * unit), row_y),
            ],
            fill=color,
            width=line_width,
        )
        draw.text(
            (legend_x + int(37 * unit), row_y),
            label,
            fill=(25, 25, 25),
            font=legend_font,
            anchor="lm",
        )

    translation_series = []
    rotation_series = []
    for variant_index, trajectory in enumerate(variants):
        color = _PILLOW_COLORS[variant_index % len(_PILLOW_COLORS)]
        delta = trajectory.center_position - source.center_position
        translation_series.append(
            (
                trajectory.time_s,
                np.linalg.norm(delta, axis=1) * 100.0,
                color,
            )
        )
        rotation_series.append(
            (
                trajectory.time_s,
                _rotation_error_deg(
                    source.center_rotation,
                    trajectory.center_rotation,
                ),
                color,
            )
        )
    _draw_pillow_line_chart(
        draw,
        right_box_top,
        translation_series,
        title="Cartesian translation from original",
        x_label="Time (s)",
        y_label="Translation (cm)",
        unit=unit,
        ImageFont=ImageFont,
    )
    _draw_pillow_line_chart(
        draw,
        right_box_bottom,
        rotation_series,
        title="Camera orientation from original",
        x_label="Time (s)",
        y_label="SO(3) angle (deg)",
        unit=unit,
        ImageFont=ImageFont,
    )
    image.save(output_path, format="PNG", optimize=True)


def _plot_joint_offsets_pillow(
    output_path: Path,
    source: CameraTrajectory,
    variants: list[CameraTrajectory],
    dpi: int,
) -> None:
    Image, ImageDraw, ImageFont = _load_pillow_modules()
    dpi = max(72, int(dpi))
    unit = max(1.0, dpi / 120.0)
    width = int(round(15.5 * dpi))
    height = int(round(7.8 * dpi))
    image = Image.new("RGB", (width, height), (250, 251, 253))
    draw = ImageDraw.Draw(image)
    title_font = _pillow_font(ImageFont, int(18 * unit), bold=True)
    legend_font = _pillow_font(ImageFont, int(10 * unit))
    _draw_centered_text(
        draw,
        (0.5 * width, int(18 * unit)),
        "View joint offsets relative to the original trajectory",
        font=title_font,
    )
    _draw_centered_text(
        draw,
        (0.5 * width, int(39 * unit)),
        "Horizontal lines verify that each episode uses a fixed joint-space offset",
        font=legend_font,
        fill=(65, 65, 65),
    )

    legend_y = int(58 * unit)
    legend_total_width = min(width - int(40 * unit), int(122 * unit * len(variants)))
    legend_start = 0.5 * (width - legend_total_width)
    legend_step = legend_total_width / max(1, len(variants))
    for index, trajectory in enumerate(variants):
        color = _PILLOW_COLORS[index % len(_PILLOW_COLORS)]
        x = legend_start + (index + 0.5) * legend_step
        draw.line(
            [
                (int(x - 22 * unit), legend_y),
                (int(x + 2 * unit), legend_y),
            ],
            fill=color,
            width=max(2, int(1.8 * unit)),
        )
        draw.text(
            (int(x + 7 * unit), legend_y),
            f"aug_{trajectory.episode.variant_index:02d}",
            fill=(30, 30, 30),
            font=legend_font,
            anchor="lm",
        )

    all_offsets = np.concatenate(
        [
            trajectory.view_q - source.view_q
            for trajectory in variants
        ],
        axis=0,
    )
    y_radius = max(float(np.max(np.abs(all_offsets))) * 1.12, 0.01)
    outer_left = int(24 * unit)
    outer_right = width - int(20 * unit)
    outer_top = int(76 * unit)
    outer_bottom = height - int(14 * unit)
    horizontal_gap = int(12 * unit)
    vertical_gap = int(12 * unit)
    panel_width = (outer_right - outer_left - 2 * horizontal_gap) // 3
    panel_height = (outer_bottom - outer_top - vertical_gap) // 2
    for joint_index, joint_name in enumerate(VIEW_JOINT_NAMES):
        row = joint_index // 3
        column = joint_index % 3
        left = outer_left + column * (panel_width + horizontal_gap)
        top = outer_top + row * (panel_height + vertical_gap)
        box = (left, top, left + panel_width, top + panel_height)
        series = []
        for variant_index, trajectory in enumerate(variants):
            series.append(
                (
                    trajectory.time_s,
                    trajectory.view_q[:, joint_index]
                    - source.view_q[:, joint_index],
                    _PILLOW_COLORS[
                        variant_index % len(_PILLOW_COLORS)
                    ],
                )
            )
        _draw_pillow_line_chart(
            draw,
            box,
            series,
            title=joint_name,
            x_label="Time (s)",
            y_label="Offset (rad)",
            unit=unit,
            ImageFont=ImageFont,
            y_limits=(-y_radius, y_radius),
        )
    image.save(output_path, format="PNG", optimize=True)


def _plot_camera_trajectories(
    output_path: Path,
    source: CameraTrajectory,
    variants: list[CameraTrajectory],
    cfg_plot: DictConfig,
    env_id: str,
) -> None:
    if plt is None:
        logging.info(
            "Matplotlib不可用，使用Pillow输出静态正交投影PNG: %s",
            _MATPLOTLIB_IMPORT_ERROR,
        )
        _plot_camera_trajectories_pillow(
            output_path,
            source,
            variants,
            cfg_plot,
            env_id,
        )
        return

    figure = plt.figure(figsize=(16.5, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.65, 1.0))
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_translation = figure.add_subplot(grid[0, 1])
    axis_rotation = figure.add_subplot(grid[1, 1])

    coordinate_frame = str(cfg_plot.coordinate_frame)
    if coordinate_frame not in {"world", "source_start"}:
        raise ValueError(
            "plot.coordinate_frame必须是world或source_start，"
            f"当前={coordinate_frame!r}"
        )
    origin = (
        source.center_position[0]
        if coordinate_frame == "source_start"
        else np.zeros(3, dtype=np.float64)
    )

    colors = plt.get_cmap("tab10")
    entries: list[tuple[CameraTrajectory, Any, str, float]] = [
        (source, "black", "original", 2.8)
    ]
    entries.extend(
        (
            trajectory,
            colors(index % 10),
            (
                f"aug_{trajectory.episode.variant_index:02d} "
                f"(|dq|={np.linalg.norm(trajectory.episode.offset_rad):.3f} rad)"
            ),
            1.8,
        )
        for index, trajectory in enumerate(variants)
    )

    all_plot_points = np.concatenate(
        [trajectory.center_position - origin for trajectory, _, _, _ in entries],
        axis=0,
    )
    span = _set_equal_3d_axes(axis_3d, all_plot_points)
    configured_arrow_length = cfg_plot.get("arrow_length_m")
    arrow_length = (
        max(0.012, 0.10 * span)
        if configured_arrow_length is None
        else float(configured_arrow_length)
    )
    if arrow_length <= 0:
        raise ValueError("plot.arrow_length_m必须为正数或null。")

    arrow_count = int(cfg_plot.arrow_count)
    show_stereo = bool(cfg_plot.show_stereo_rails)
    for trajectory, color, label, line_width in entries:
        center = trajectory.center_position - origin
        axis_3d.plot(
            center[:, 0],
            center[:, 1],
            center[:, 2],
            color=color,
            linewidth=line_width,
            label=label,
        )
        axis_3d.scatter(
            *center[0],
            color=color,
            s=42,
            marker="o",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        axis_3d.scatter(
            *center[-1],
            color=color,
            s=70,
            marker="*",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )

        arrow_indices = _sample_plot_indices(len(center), arrow_count)
        for index in arrow_indices:
            direction = trajectory.forward[index]
            axis_3d.quiver(
                center[index, 0],
                center[index, 1],
                center[index, 2],
                direction[0],
                direction[1],
                direction[2],
                length=arrow_length,
                normalize=True,
                color=color,
                linewidth=1.0,
                alpha=0.78,
                arrow_length_ratio=0.24,
            )

        if show_stereo:
            left = trajectory.left_position - origin
            right = trajectory.right_position - origin
            axis_3d.plot(
                left[:, 0], left[:, 1], left[:, 2],
                color=color, linewidth=0.65, linestyle=":", alpha=0.45,
            )
            axis_3d.plot(
                right[:, 0], right[:, 1], right[:, 2],
                color=color, linewidth=0.65, linestyle=":", alpha=0.45,
            )
            for index in arrow_indices:
                axis_3d.plot(
                    [left[index, 0], right[index, 0]],
                    [left[index, 1], right[index, 1]],
                    [left[index, 2], right[index, 2]],
                    color=color,
                    linewidth=0.7,
                    alpha=0.45,
                )

    connector_indices = _sample_plot_indices(
        len(source.center_position),
        int(cfg_plot.connector_count),
    )
    source_center = source.center_position - origin
    for variant_index, trajectory in enumerate(variants):
        variant_center = trajectory.center_position - origin
        color = colors(variant_index % 10)
        for index in connector_indices:
            axis_3d.plot(
                [source_center[index, 0], variant_center[index, 0]],
                [source_center[index, 1], variant_center[index, 1]],
                [source_center[index, 2], variant_center[index, 2]],
                color=color,
                linewidth=0.8,
                linestyle="--",
                alpha=0.42,
            )

    prefix = "Delta " if coordinate_frame == "source_start" else "World "
    axis_3d.set_xlabel(f"{prefix}X (m)")
    axis_3d.set_ylabel(f"{prefix}Y (m)")
    axis_3d.set_zlabel(f"{prefix}Z (m)")
    axis_3d.set_title(
        "Stereo-center camera trajectories\n"
        "circle=start, star=end, arrows=optical axis"
    )
    axis_3d.view_init(
        elev=float(cfg_plot.elevation_deg),
        azim=float(cfg_plot.azimuth_deg),
    )
    axis_3d.legend(loc="upper left", fontsize=8)
    axis_3d.grid(True, alpha=0.28)

    for variant_index, trajectory in enumerate(variants):
        color = colors(variant_index % 10)
        delta = trajectory.center_position - source.center_position
        translation_cm = np.linalg.norm(delta, axis=1) * 100.0
        rotation_deg = _rotation_error_deg(
            source.center_rotation,
            trajectory.center_rotation,
        )
        label = f"aug_{trajectory.episode.variant_index:02d}"
        axis_translation.plot(
            trajectory.time_s,
            translation_cm,
            color=color,
            linewidth=1.7,
            label=label,
        )
        axis_rotation.plot(
            trajectory.time_s,
            rotation_deg,
            color=color,
            linewidth=1.7,
            label=label,
        )

    axis_translation.set_title("Cartesian translation from original")
    axis_translation.set_ylabel("Translation (cm)")
    axis_translation.grid(True, alpha=0.3)
    axis_translation.legend(ncol=3, fontsize=8)
    axis_rotation.set_title("Camera orientation from original")
    axis_rotation.set_xlabel("Time (s)")
    axis_rotation.set_ylabel("SO(3) angle (deg)")
    axis_rotation.grid(True, alpha=0.3)

    figure.suptitle(
        f"Fixed View-joint Offset Effect | source={source.episode.source_episode:06d}"
        f" | {env_id}",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=int(cfg_plot.dpi), bbox_inches="tight")
    plt.close(figure)


def _plot_joint_offsets(
    output_path: Path,
    source: CameraTrajectory,
    variants: list[CameraTrajectory],
    dpi: int,
) -> None:
    if plt is None:
        _plot_joint_offsets_pillow(
            output_path,
            source,
            variants,
            dpi,
        )
        return

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 7.8),
        sharex=True,
        constrained_layout=True,
    )
    colors = plt.get_cmap("tab10")
    for joint_index, axis in enumerate(axes.flat):
        for variant_index, trajectory in enumerate(variants):
            offset = trajectory.view_q[:, joint_index] - source.view_q[:, joint_index]
            axis.plot(
                trajectory.time_s,
                offset,
                color=colors(variant_index % 10),
                linewidth=1.6,
                label=f"aug_{trajectory.episode.variant_index:02d}",
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(VIEW_JOINT_NAMES[joint_index])
        axis.set_ylabel("Offset (rad)")
        axis.grid(True, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=len(variants), fontsize=9)
    figure.suptitle(
        "View joint offsets relative to the original trajectory\n"
        "horizontal lines verify that each episode uses a fixed joint-space offset",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _write_trajectory_points(
    output_path: Path,
    source: CameraTrajectory,
    trajectories: list[CameraTrajectory],
) -> None:
    fields = [
        "episode_name",
        "source_episode",
        "variant_index",
        "frame_index",
        "time_s",
        *[f"view_q_{index}" for index in range(6)],
        "center_x_m",
        "center_y_m",
        "center_z_m",
        "left_x_m",
        "left_y_m",
        "left_z_m",
        "right_x_m",
        "right_y_m",
        "right_z_m",
        "forward_x",
        "forward_y",
        "forward_z",
        "delta_x_m",
        "delta_y_m",
        "delta_z_m",
        "translation_from_original_cm",
        "rotation_from_original_deg",
        "stereo_baseline_cm",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for trajectory in trajectories:
            delta = trajectory.center_position - source.center_position
            translation_cm = np.linalg.norm(delta, axis=1) * 100.0
            rotation_deg = _rotation_error_deg(
                source.center_rotation,
                trajectory.center_rotation,
            )
            for index in range(len(trajectory.frame_indices)):
                row: dict[str, Any] = {
                    "episode_name": trajectory.episode.name,
                    "source_episode": trajectory.episode.source_episode,
                    "variant_index": trajectory.episode.variant_index,
                    "frame_index": int(trajectory.frame_indices[index]),
                    "time_s": float(trajectory.time_s[index]),
                    "center_x_m": float(trajectory.center_position[index, 0]),
                    "center_y_m": float(trajectory.center_position[index, 1]),
                    "center_z_m": float(trajectory.center_position[index, 2]),
                    "left_x_m": float(trajectory.left_position[index, 0]),
                    "left_y_m": float(trajectory.left_position[index, 1]),
                    "left_z_m": float(trajectory.left_position[index, 2]),
                    "right_x_m": float(trajectory.right_position[index, 0]),
                    "right_y_m": float(trajectory.right_position[index, 1]),
                    "right_z_m": float(trajectory.right_position[index, 2]),
                    "forward_x": float(trajectory.forward[index, 0]),
                    "forward_y": float(trajectory.forward[index, 1]),
                    "forward_z": float(trajectory.forward[index, 2]),
                    "delta_x_m": float(delta[index, 0]),
                    "delta_y_m": float(delta[index, 1]),
                    "delta_z_m": float(delta[index, 2]),
                    "translation_from_original_cm": float(translation_cm[index]),
                    "rotation_from_original_deg": float(rotation_deg[index]),
                    "stereo_baseline_cm": float(
                        trajectory.stereo_baseline_m[index] * 100.0
                    ),
                }
                row.update(
                    {
                        f"view_q_{joint_index}": float(
                            trajectory.view_q[index, joint_index]
                        )
                        for joint_index in range(trajectory.view_q.shape[1])
                    }
                )
                writer.writerow(row)


def _write_summary_csv(output_path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "episode_name",
        "source_episode",
        "variant_index",
        *[f"offset_q{index}_rad" for index in range(6)],
        "view_joint_offset_l2_rad",
        "fixed_joint_offset_max_error_rad",
        "frames_used",
        "camera_path_length_m",
        "translation_start_cm",
        "translation_mean_cm",
        "translation_rms_cm",
        "translation_max_cm",
        "rotation_start_deg",
        "rotation_mean_deg",
        "rotation_max_deg",
        "stereo_baseline_mean_cm",
        "stereo_baseline_std_mm",
        "cartesian_delta_std_x_cm",
        "cartesian_delta_std_y_cm",
        "cartesian_delta_std_z_cm",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {key: summary.get(key) for key in fields}
            for index, value in enumerate(summary["view_joint_offset_rad"]):
                row[f"offset_q{index}_rad"] = value
            for index, axis_name in enumerate(("x", "y", "z")):
                row[f"cartesian_delta_std_{axis_name}_cm"] = summary[
                    "cartesian_delta_std_cm"
                ][index]
            writer.writerow(row)


def visualize_view_camera_trajectories(cfg: DictConfig) -> Path:
    run_dir = _absolute_path(cfg.input_run_dir)
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"找不到增强run的metadata.json: {metadata_path}")
    metadata = _read_json(metadata_path)
    metadata_env_id = str(metadata.get("env_id", "")).strip()
    configured_env_id = (
        "" if cfg.get("env_id") is None else str(cfg.env_id).strip()
    )
    if configured_env_id and metadata_env_id and configured_env_id != metadata_env_id:
        raise ValueError(
            f"配置env_id={configured_env_id!r}与metadata env_id={metadata_env_id!r}不一致。"
        )
    env_id = configured_env_id or metadata_env_id
    if not env_id:
        raise ValueError("metadata和配置中都没有提供env_id。")

    groups = _scan_episode_groups(run_dir)
    source_episode = _resolve_source_episode(groups, cfg.get("source_episode"))
    group = groups[source_episode]
    variant_indices = _resolve_variants(group, cfg.get("variant_indices"))
    default_fps = float(metadata.get("fps", 25.0))
    view_start = int(cfg.view_action_start)
    view_dim = int(cfg.view_action_dim)
    if view_dim != len(VIEW_JOINT_NAMES):
        raise ValueError(
            f"当前绘图器要求View维度={len(VIEW_JOINT_NAMES)}，配置为{view_dim}。"
        )

    source_dir, source_info = group[-1]
    source_episode_data = _load_episode(
        source_dir,
        source_info,
        state_key=str(cfg.state_key),
        view_start=view_start,
        view_dim=view_dim,
        default_fps=default_fps,
    )
    variant_episode_data = [
        _load_episode(
            group[index][0],
            group[index][1],
            state_key=str(cfg.state_key),
            view_start=view_start,
            view_dim=view_dim,
            default_fps=default_fps,
        )
        for index in variant_indices
    ]
    _validate_paired_episodes(
        source_episode_data,
        variant_episode_data,
        view_start=view_start,
        view_dim=view_dim,
    )

    _configure_mujoco_runtime(cfg)
    env_obj = _make_environment(
        env_id=env_id,
        cameras=(),
        render_height=64,
        render_width=64,
    )
    try:
        _restore_initial_state(env_obj, source_episode_data.arrays)
        source_trajectory = _compute_camera_trajectory(
            env_obj,
            source_episode_data,
            view_start=view_start,
            view_dim=view_dim,
            frame_stride=int(cfg.frame_stride),
        )
        variant_trajectories = [
            _compute_camera_trajectory(
                env_obj,
                episode,
                view_start=view_start,
                view_dim=view_dim,
                frame_stride=int(cfg.frame_stride),
            )
            for episode in variant_episode_data
        ]
    finally:
        env_obj.close()

    output_root = _absolute_path(cfg.output_dir)
    output_dir = (
        output_root
        / run_dir.name
        / f"source_episode={source_episode:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_3d_path = output_dir / "view_camera_trajectories_3d.png"
    joint_plot_path = output_dir / "view_joint_offsets.png"
    points_path = output_dir / "trajectory_points.csv"
    summary_csv_path = output_dir / "trajectory_summary.csv"
    summary_json_path = output_dir / "trajectory_summary.json"

    _plot_camera_trajectories(
        plot_3d_path,
        source_trajectory,
        variant_trajectories,
        cfg.plot,
        env_id,
    )
    _plot_joint_offsets(
        joint_plot_path,
        source_trajectory,
        variant_trajectories,
        dpi=int(cfg.plot.dpi),
    )
    all_trajectories = [source_trajectory, *variant_trajectories]
    _write_trajectory_points(points_path, source_trajectory, all_trajectories)
    summaries = [
        _trajectory_metrics(trajectory, source_trajectory)
        for trajectory in all_trajectories
    ]
    _write_summary_csv(summary_csv_path, summaries)
    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "input_run_dir": str(run_dir),
                "env_id": env_id,
                "source_episode": source_episode,
                "variant_indices": variant_indices,
                "frame_stride": int(cfg.frame_stride),
                "coordinate_frame": str(cfg.plot.coordinate_frame),
                "trajectories": summaries,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    logging.info("3D相机轨迹图: %s", plot_3d_path)
    logging.info("View关节偏移图: %s", joint_plot_path)
    logging.info("逐帧轨迹数据: %s", points_path)
    for summary in summaries[1:]:
        logging.info(
            "%s: |dq|=%.4frad, translation mean/max=%.2f/%.2fcm, "
            "rotation mean/max=%.2f/%.2fdeg",
            summary["episode_name"],
            summary["view_joint_offset_l2_rad"],
            summary["translation_mean_cm"],
            summary["translation_max_cm"],
            summary["rotation_mean_deg"],
            summary["rotation_max_deg"],
        )
    return output_dir


@hydra.main(
    version_base="1.2",
    config_path="../configs/data_collect",
    config_name="view_camera_trajectory_visualization",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output_dir = visualize_view_camera_trajectories(cfg)
    logging.info("可视化完成: %s", output_dir)


if __name__ == "__main__":
    main()
