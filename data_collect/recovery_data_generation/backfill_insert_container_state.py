#!/usr/bin/env python

"""从原始双目视频恢复旧 InsertCylinder episode 的容器初始位置。

旧版 Quest 轨迹只保存 ``data.qpos``，但 InsertCylinder 的
``cylinder_container`` 没有 joint；它的随机位置位于
``model.body_pos``，因此无法从 ``initial_qpos`` 直接恢复。

本脚本使用以下只读恢复流程：

1. 从左右 ZED 视频后半段自动选择黄色圆环清晰可见的帧；
2. 按旧采集时序重放专家动作，保存对应帧的 MuJoCo 状态；
3. 在容器 y 轴随机区间内做粗到细一维搜索；
4. 将候选位置的 MuJoCo geom 分割掩码与原视频黄色圆环匹配；
5. 输出置信度报告和诊断图；只有显式开启时才创建回填后的新 raw run。

源 run 永远不会被修改。默认配置只处理前 10 条并生成报告。
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import av
import cv2
import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig, OmegaConf


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_collect.recovery_data_generation.trajectory_replay_common import (  # noqa: E402
    MODEL_BODY_INITIAL_KEYS,
    _apply_original_action,
    _configure_mujoco_runtime,
    _make_environment,
    _read_agent_state,
    _validate_source_arrays,
)


RECOVERY_SCHEMA_VERSION = 1
EXPECTED_ENV_ID = "guided_vision/InsertCylinder-3Arms-v0"
EPISODE_PATTERN = re.compile(r"^episode_(?P<episode>\d{6,})$")
CAMERAS = ("zed_cam_left", "zed_cam_right")
CONTAINER_BODY_NAME = "cylinder_container"
CONTAINER_OUTER_GEOM_NAME = "container_ring_outer"
MJOBJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


@dataclass(frozen=True)
class SourceEpisode:
    episode_number: int
    directory: Path
    info: dict[str, Any]


@dataclass(frozen=True)
class RingCandidate:
    frame_index: int
    component_score: float
    bbox_xywh: tuple[int, int, int, int]
    area_px: int
    aspect_ratio: float
    fill_ratio: float


@dataclass(frozen=True)
class RingObservation:
    camera: str
    frame_index: int
    component_score: float
    bbox_xywh: tuple[int, int, int, int]
    area_px: int
    mask: np.ndarray
    rgb: np.ndarray


@dataclass(frozen=True)
class ReplaySnapshot:
    physics_state: np.ndarray
    actuator_ctrl: np.ndarray


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(value), file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON根节点必须是对象: {path}")
    return value


def _episode_number(path: Path) -> int:
    match = EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"无法解析原始episode目录名: {path}")
    return int(match.group("episode"))


def _load_sources(
    input_run_dir: Path,
    source_episode_indices: list[int] | None,
    max_source_episodes: int | None,
) -> list[SourceEpisode]:
    episodes_dir = input_run_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"输入run缺少episodes目录: {episodes_dir}")

    requested = (
        None
        if source_episode_indices is None
        else {int(value) for value in source_episode_indices}
    )
    directories = sorted(
        (
            path
            for path in episodes_dir.iterdir()
            if path.is_dir()
            and EPISODE_PATTERN.fullmatch(path.name)
            and (path / "arrays.npz").is_file()
        ),
        key=_episode_number,
    )
    if requested is not None:
        existing = {_episode_number(path) for path in directories}
        missing = sorted(requested - existing)
        if missing:
            raise FileNotFoundError(f"找不到指定的源episode: {missing}")
        directories = [
            path for path in directories if _episode_number(path) in requested
        ]
    if max_source_episodes is not None:
        if int(max_source_episodes) <= 0:
            raise ValueError("max_source_episodes必须为正整数或null。")
        directories = directories[: int(max_source_episodes)]
    if not directories:
        raise RuntimeError(f"没有可处理的源episode: {input_run_dir}")

    sources = []
    for directory in directories:
        info_path = directory / "info.json"
        sources.append(
            SourceEpisode(
                episode_number=_episode_number(directory),
                directory=directory,
                info=_load_json(info_path) if info_path.is_file() else {},
            )
        )
    return sources


def _validate_fraction(name: str, value: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name}必须位于[0,1]，当前为{value!r}。")
    return numeric


def _validate_config(cfg: DictConfig) -> None:
    if int(cfg.fps) <= 0:
        raise ValueError("fps必须为正整数。")
    if tuple(str(camera) for camera in cfg.cameras) != CAMERAS:
        raise ValueError(f"当前回填固定使用双目相机{CAMERAS}。")

    y_min = float(cfg.search.y_min_m)
    y_max = float(cfg.search.y_max_m)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min >= y_max:
        raise ValueError("search.y_min_m必须小于search.y_max_m。")
    for name in ("coarse_step_m", "fine_radius_m", "fine_step_m"):
        value = float(cfg.search[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"search.{name}必须为有限正数。")
    if float(cfg.search.fine_step_m) >= float(cfg.search.coarse_step_m):
        raise ValueError("fine_step_m必须小于coarse_step_m。")

    tail_start = _validate_fraction(
        "search.tail_start_fraction",
        cfg.search.tail_start_fraction,
    )
    tail_end = _validate_fraction(
        "search.tail_end_fraction",
        cfg.search.tail_end_fraction,
    )
    if tail_start >= tail_end:
        raise ValueError("tail_start_fraction必须小于tail_end_fraction。")
    if int(cfg.search.observations_per_camera) <= 0:
        raise ValueError("observations_per_camera必须为正整数。")
    if int(cfg.confidence.min_observations_per_camera) <= 0:
        raise ValueError("min_observations_per_camera必须为正整数。")

    for name in ("hsv_lower", "hsv_upper"):
        value = np.asarray(cfg.mask[name], dtype=np.int64)
        if value.shape != (3,) or np.any(value < 0) or np.any(value > 255):
            raise ValueError(f"mask.{name}必须是3个0～255整数。")

    weights = np.asarray(
        [
            cfg.matching.iou_weight,
            cfg.matching.centroid_weight,
            cfg.matching.area_weight,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("matching权重必须是有限非负数且总和大于0。")

    video_mode = str(cfg.output.video_copy_mode)
    if video_mode not in {"hardlink", "symlink", "copy"}:
        raise ValueError("output.video_copy_mode必须为hardlink/symlink/copy。")
    if bool(cfg.output.write_recovered_run):
        input_dir = _resolve_path(cfg.input_run_dir)
        output_dir = _resolve_path(cfg.output.recovered_run_dir)
        if input_dir == output_dir:
            raise ValueError("回填输出目录不能与源run相同。")


def _best_ring_component(
    rgb: np.ndarray,
    cfg: DictConfig,
    *,
    frame_index: int,
    return_mask: bool,
) -> tuple[RingCandidate, np.ndarray | None] | None:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"视频帧必须是uint8 HWC RGB，当前{rgb.shape}/{rgb.dtype}。")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower = np.asarray(cfg.mask.hsv_lower, dtype=np.uint8)
    upper = np.asarray(cfg.mask.hsv_upper, dtype=np.uint8)
    threshold = cv2.inRange(hsv, lower, upper)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        threshold,
        connectivity=8,
    )

    height, width = threshold.shape
    best: tuple[float, int, RingCandidate] | None = None
    for component_id in range(1, count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[component_id]
        )
        aspect_ratio = box_width / max(box_height, 1)
        fill_ratio = area / max(box_width * box_height, 1)
        if not int(cfg.mask.min_area_px) <= area <= int(cfg.mask.max_area_px):
            continue
        if not int(cfg.mask.min_width_px) <= box_width <= int(cfg.mask.max_width_px):
            continue
        if not int(cfg.mask.min_height_px) <= box_height <= int(cfg.mask.max_height_px):
            continue
        if not float(cfg.mask.min_aspect_ratio) <= aspect_ratio <= float(
            cfg.mask.max_aspect_ratio
        ):
            continue
        if not float(cfg.mask.min_fill_ratio) <= fill_ratio <= float(
            cfg.mask.max_fill_ratio
        ):
            continue
        border = int(cfg.mask.border_margin_px)
        if (
            x <= border
            or y <= border
            or x + box_width >= width - border
            or y + box_height >= height - border
        ):
            continue

        expected_aspect = float(cfg.mask.expected_aspect_ratio)
        expected_fill = float(cfg.mask.expected_fill_ratio)
        aspect_factor = np.exp(
            -abs(np.log(max(aspect_ratio, 1e-12) / expected_aspect))
        )
        fill_factor = np.exp(
            -float(cfg.mask.fill_score_scale) * abs(fill_ratio - expected_fill)
        )
        component_score = float(area * aspect_factor * fill_factor)
        candidate = RingCandidate(
            frame_index=int(frame_index),
            component_score=component_score,
            bbox_xywh=(x, y, box_width, box_height),
            area_px=area,
            aspect_ratio=float(aspect_ratio),
            fill_ratio=float(fill_ratio),
        )
        if best is None or component_score > best[0]:
            best = (component_score, component_id, candidate)

    if best is None:
        return None
    mask = labels == best[1] if return_mask else None
    return best[2], mask


def _select_temporally_separated(
    candidates: list[RingCandidate],
    *,
    count: int,
    min_separation_frames: int,
) -> list[RingCandidate]:
    selected = []
    for candidate in sorted(
        candidates,
        key=lambda item: item.component_score,
        reverse=True,
    ):
        if all(
            abs(candidate.frame_index - existing.frame_index)
            >= min_separation_frames
            for existing in selected
        ):
            selected.append(candidate)
        if len(selected) >= count:
            break
    return sorted(selected, key=lambda item: item.frame_index)


def _decode_selected_observations(
    video_path: Path,
    *,
    camera: str,
    expected_frames: int,
    cfg: DictConfig,
) -> list[RingObservation]:
    if not video_path.is_file():
        raise FileNotFoundError(f"缺少源视频: {video_path}")
    start_frame = int(np.floor(expected_frames * float(cfg.search.tail_start_fraction)))
    end_frame = min(
        expected_frames - 1,
        int(np.ceil(expected_frames * float(cfg.search.tail_end_fraction))) - 1,
    )

    candidates = []
    decoded_frames = 0
    with av.open(str(video_path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            decoded_frames = frame_index + 1
            if frame_index < start_frame or frame_index > end_frame:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            result = _best_ring_component(
                rgb,
                cfg,
                frame_index=frame_index,
                return_mask=False,
            )
            if result is not None:
                candidates.append(result[0])
    if decoded_frames != expected_frames:
        raise ValueError(
            f"视频帧数与动作数不一致: video={decoded_frames}, "
            f"actions={expected_frames}, path={video_path}"
        )

    min_separation = max(
        1,
        int(
            round(
                expected_frames
                * float(cfg.search.min_frame_separation_fraction)
            )
        ),
    )
    selected = _select_temporally_separated(
        candidates,
        count=int(cfg.search.observations_per_camera),
        min_separation_frames=min_separation,
    )
    selected_indices = {item.frame_index: item for item in selected}
    observations = []
    with av.open(str(video_path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index not in selected_indices:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            result = _best_ring_component(
                rgb,
                cfg,
                frame_index=frame_index,
                return_mask=True,
            )
            if result is None or result[1] is None:
                raise RuntimeError(
                    f"第二次解码未能复现黄色圆环候选: {video_path}, "
                    f"frame={frame_index}"
                )
            candidate, mask = result
            observations.append(
                RingObservation(
                    camera=camera,
                    frame_index=frame_index,
                    component_score=candidate.component_score,
                    bbox_xywh=candidate.bbox_xywh,
                    area_px=candidate.area_px,
                    mask=np.asarray(mask, dtype=np.bool_),
                    rgb=np.asarray(rgb, dtype=np.uint8),
                )
            )
    return observations


def _restore_data_state(env_obj, arrays: dict[str, np.ndarray]) -> None:
    physics = env_obj._physics
    env_obj.reset(seed=0)
    present_model_keys = [
        key for key in MODEL_BODY_INITIAL_KEYS if key in arrays
    ]
    if present_model_keys and len(present_model_keys) != len(
        MODEL_BODY_INITIAL_KEYS
    ):
        raise KeyError(
            "源episode的initial_model_body_pos/body_quat字段不完整。"
        )
    if present_model_keys:
        body_pos = np.asarray(arrays["initial_model_body_pos"])
        body_quat = np.asarray(arrays["initial_model_body_quat"])
        if body_pos.shape != physics.model.body_pos.shape:
            raise ValueError(
                "initial_model_body_pos与当前模型形状不一致: "
                f"{body_pos.shape} != {physics.model.body_pos.shape}"
            )
        if body_quat.shape != physics.model.body_quat.shape:
            raise ValueError(
                "initial_model_body_quat与当前模型形状不一致: "
                f"{body_quat.shape} != {physics.model.body_quat.shape}"
            )
        physics.model.body_pos[:] = body_pos
        physics.model.body_quat[:] = body_quat

    physics.data.qpos[:] = arrays["initial_qpos"]
    physics.data.qvel[:] = arrays["initial_qvel"]
    physics.data.time = float(arrays["initial_time"])
    physics.data.ctrl[:] = arrays["initial_ctrl"]
    physics.data.act[:] = arrays["initial_act"]
    physics.data.mocap_pos[:] = arrays["initial_mocap_pos"]
    physics.data.mocap_quat[:] = arrays["initial_mocap_quat"]
    physics.forward()


def _capture_replay_snapshots(
    env_obj,
    arrays: dict[str, np.ndarray],
    frame_indices: set[int],
    *,
    max_agent_state_abs_error: float,
) -> tuple[dict[int, ReplaySnapshot], float]:
    _restore_data_state(env_obj, arrays)
    physics = env_obj._physics
    actions = arrays["joint_action"]
    recorded_states = arrays["observation_state"]
    snapshots = {}
    max_state_error = 0.0

    for frame_index, action in enumerate(actions):
        replayed_state = _read_agent_state(env_obj)
        state_error = float(
            np.max(
                np.abs(
                    replayed_state
                    - recorded_states[frame_index].astype(np.float64)
                )
            )
        )
        max_state_error = max(max_state_error, state_error)
        if state_error > max_agent_state_abs_error:
            raise RuntimeError(
                f"专家轨迹重放误差过大: frame={frame_index}, "
                f"state_error={state_error:.6g} > "
                f"{max_agent_state_abs_error:.6g}"
            )

        # 旧采集视频先执行action[k]再渲染，因此目标视频帧对应步进后的状态。
        _apply_original_action(env_obj, action)
        if frame_index in frame_indices:
            snapshots[frame_index] = ReplaySnapshot(
                physics_state=np.asarray(physics.get_state()).copy(),
                actuator_ctrl=physics.data.ctrl.copy(),
            )

    missing = sorted(frame_indices - set(snapshots))
    if missing:
        raise RuntimeError(f"没有捕获到目标重放帧: {missing}")
    return snapshots, max_state_error


def _inclusive_grid(lower: float, upper: float, step: float) -> np.ndarray:
    if not lower <= upper:
        raise ValueError("搜索下界不能大于上界。")
    count = int(np.floor((upper - lower) / step + 1e-12))
    values = lower + np.arange(count + 1, dtype=np.float64) * step
    if values[-1] < upper - step * 1e-6:
        values = np.concatenate((values, np.asarray([upper], dtype=np.float64)))
    return np.clip(values, lower, upper)


def _mask_similarity(
    observed: np.ndarray,
    predicted: np.ndarray,
    cfg: DictConfig,
) -> float:
    observed = np.asarray(observed, dtype=np.bool_)
    predicted = np.asarray(predicted, dtype=np.bool_)
    if observed.shape != predicted.shape:
        raise ValueError(
            f"观测与预测mask形状不一致: {observed.shape} != {predicted.shape}"
        )
    if not observed.any() or not predicted.any():
        return 0.0

    radius = int(cfg.matching.mask_dilation_radius_px)
    if radius < 0:
        raise ValueError("mask_dilation_radius_px不能为负数。")
    if radius:
        kernel_size = radius * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        observed_soft = cv2.dilate(
            observed.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)
        predicted_soft = cv2.dilate(
            predicted.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)
    else:
        observed_soft = observed
        predicted_soft = predicted

    intersection = int(np.logical_and(observed_soft, predicted_soft).sum())
    union = int(np.logical_or(observed_soft, predicted_soft).sum())
    iou = intersection / max(union, 1)

    observed_center = np.argwhere(observed).mean(axis=0)
    predicted_center = np.argwhere(predicted).mean(axis=0)
    center_distance = float(np.linalg.norm(observed_center - predicted_center))
    centroid_similarity = float(
        np.exp(
            -center_distance
            / max(float(cfg.matching.centroid_scale_px), 1e-12)
        )
    )
    area_similarity = float(
        np.exp(
            -abs(
                np.log(
                    (float(predicted.sum()) + 1.0)
                    / (float(observed.sum()) + 1.0)
                )
            )
        )
    )

    weights = np.asarray(
        [
            cfg.matching.iou_weight,
            cfg.matching.centroid_weight,
            cfg.matching.area_weight,
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return float(
        weights[0] * iou
        + weights[1] * centroid_similarity
        + weights[2] * area_similarity
    )


def _render_container_mask(
    env_obj,
    snapshot: ReplaySnapshot,
    *,
    camera: str,
    container_y: float,
    height: int,
    width: int,
    container_geom_id: int,
) -> np.ndarray:
    physics = env_obj._physics
    physics.set_state(snapshot.physics_state)
    physics.data.ctrl[:] = snapshot.actuator_ctrl
    physics.bind(env_obj._container_body).pos = np.asarray(
        [-0.045, container_y, 0.0],
        dtype=np.float64,
    )
    physics.forward()
    segmentation = physics.render(
        height=height,
        width=width,
        camera_id=camera,
        segmentation=True,
    )
    return (
        (segmentation[..., 0] == container_geom_id)
        & (segmentation[..., 1] == MJOBJ_GEOM)
    )


def _evaluate_grid(
    env_obj,
    observations: list[RingObservation],
    snapshots: dict[int, ReplaySnapshot],
    y_values: np.ndarray,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    physics = env_obj._physics
    container_geom_id = int(
        physics.model.name2id(CONTAINER_OUTER_GEOM_NAME, "geom")
    )
    height = int(cfg.render_height)
    width = int(cfg.render_width)
    results = []

    for container_y in y_values:
        observation_scores = {}
        for observation in observations:
            predicted = _render_container_mask(
                env_obj,
                snapshots[observation.frame_index],
                camera=observation.camera,
                container_y=float(container_y),
                height=height,
                width=width,
                container_geom_id=container_geom_id,
            )
            key = f"{observation.camera}:frame={observation.frame_index}"
            observation_scores[key] = _mask_similarity(
                observation.mask,
                predicted,
                cfg,
            )

        camera_scores = {}
        for camera in CAMERAS:
            values = [
                score
                for key, score in observation_scores.items()
                if key.startswith(f"{camera}:")
            ]
            camera_scores[camera] = (
                float(np.mean(values)) if values else 0.0
            )
        results.append(
            {
                "y": float(container_y),
                "score": float(np.mean(list(observation_scores.values()))),
                "camera_scores": camera_scores,
                "observation_scores": observation_scores,
            }
        )
    return results


def _best_by_score(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("搜索结果为空。")
    return max(results, key=lambda item: float(item["score"]))


def _render_diagnostic(
    env_obj,
    observation: RingObservation,
    snapshot: ReplaySnapshot,
    *,
    recovered_y: float,
    cfg: DictConfig,
) -> tuple[np.ndarray, float]:
    physics = env_obj._physics
    geom_id = int(physics.model.name2id(CONTAINER_OUTER_GEOM_NAME, "geom"))
    predicted = _render_container_mask(
        env_obj,
        snapshot,
        camera=observation.camera,
        container_y=recovered_y,
        height=int(cfg.render_height),
        width=int(cfg.render_width),
        container_geom_id=geom_id,
    )
    rendered = physics.render(
        height=int(cfg.render_height),
        width=int(cfg.render_width),
        camera_id=observation.camera,
    )

    original_overlay = observation.rgb.copy()
    rendered_overlay = rendered.copy()
    observed_contours, _ = cv2.findContours(
        observation.mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    predicted_contours, _ = cv2.findContours(
        predicted.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(original_overlay, observed_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(original_overlay, predicted_contours, -1, (255, 0, 255), 2)
    cv2.drawContours(rendered_overlay, predicted_contours, -1, (255, 0, 255), 2)

    label = (
        f"{observation.camera} frame={observation.frame_index} "
        f"y={recovered_y:.5f}"
    )
    for image in (original_overlay, rendered_overlay):
        cv2.putText(
            image,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    exclusion_radius = int(cfg.diagnostics.rgb_exclusion_radius_px)
    kernel_size = exclusion_radius * 2 + 1
    exclusion = cv2.dilate(
        np.logical_or(observation.mask, predicted).astype(np.uint8),
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    background = ~exclusion
    rgb_mae = float(
        np.abs(
            observation.rgb.astype(np.float32)
            - rendered.astype(np.float32)
        )[background].mean()
        / 255.0
    )
    diagnostic = np.concatenate((original_overlay, rendered_overlay), axis=1)
    return diagnostic, rgb_mae


def _recover_episode(
    env_obj,
    source: SourceEpisode,
    arrays: dict[str, np.ndarray],
    cfg: DictConfig,
    diagnostic_root: Path,
) -> dict[str, Any]:
    actions = arrays["joint_action"]
    observations = []
    for camera in CAMERAS:
        observations.extend(
            _decode_selected_observations(
                source.directory / "videos" / f"{camera}.mp4",
                camera=camera,
                expected_frames=len(actions),
                cfg=cfg,
            )
        )
    if not observations:
        raise RuntimeError("左右视频中均未检测到可用黄色圆环。")

    frame_indices = {item.frame_index for item in observations}
    snapshots, max_replay_error = _capture_replay_snapshots(
        env_obj,
        arrays,
        frame_indices,
        max_agent_state_abs_error=float(
            cfg.replay.max_agent_state_abs_error
        ),
    )

    coarse_values = _inclusive_grid(
        float(cfg.search.y_min_m),
        float(cfg.search.y_max_m),
        float(cfg.search.coarse_step_m),
    )
    coarse_results = _evaluate_grid(
        env_obj,
        observations,
        snapshots,
        coarse_values,
        cfg,
    )
    coarse_best = _best_by_score(coarse_results)
    fine_lower = max(
        float(cfg.search.y_min_m),
        float(coarse_best["y"]) - float(cfg.search.fine_radius_m),
    )
    fine_upper = min(
        float(cfg.search.y_max_m),
        float(coarse_best["y"]) + float(cfg.search.fine_radius_m),
    )
    fine_values = _inclusive_grid(
        fine_lower,
        fine_upper,
        float(cfg.search.fine_step_m),
    )
    fine_results = _evaluate_grid(
        env_obj,
        observations,
        snapshots,
        fine_values,
        cfg,
    )
    best = _best_by_score(fine_results)
    recovered_y = float(best["y"])

    camera_best_y = {}
    for camera in CAMERAS:
        camera_best = max(
            fine_results,
            key=lambda item: float(item["camera_scores"][camera]),
        )
        camera_best_y[camera] = float(camera_best["y"])
    camera_disagreement = abs(
        camera_best_y[CAMERAS[0]] - camera_best_y[CAMERAS[1]]
    )

    observation_best = {}
    for observation in observations:
        key = f"{observation.camera}:frame={observation.frame_index}"
        item = max(
            fine_results,
            key=lambda result: float(result["observation_scores"][key]),
        )
        observation_best[key] = {
            "best_y": float(item["y"]),
            "best_score": float(item["observation_scores"][key]),
        }
    frame_best_values = np.asarray(
        [item["best_y"] for item in observation_best.values()],
        dtype=np.float64,
    )
    frame_std = float(frame_best_values.std()) if len(frame_best_values) else float("inf")

    separation = float(cfg.confidence.score_separation_m)
    far_results = [
        result
        for result in (*coarse_results, *fine_results)
        if abs(float(result["y"]) - recovered_y) >= separation
    ]
    far_best_score = (
        max(float(result["score"]) for result in far_results)
        if far_results
        else 0.0
    )
    score_margin = float(best["score"]) - far_best_score

    diagnostic_dir = diagnostic_root / f"episode_{source.episode_number:06d}"
    if bool(cfg.diagnostics.save_images):
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
    rgb_mae_values = []
    observation_descriptions = []
    for observation in observations:
        diagnostic, rgb_mae = _render_diagnostic(
            env_obj,
            observation,
            snapshots[observation.frame_index],
            recovered_y=recovered_y,
            cfg=cfg,
        )
        rgb_mae_values.append(rgb_mae)
        observation_descriptions.append(
            {
                "camera": observation.camera,
                "frame_index": observation.frame_index,
                "component_score": observation.component_score,
                "bbox_xywh": list(observation.bbox_xywh),
                "area_px": observation.area_px,
                "match_score": best["observation_scores"][
                    f"{observation.camera}:frame={observation.frame_index}"
                ],
                "background_rgb_mae": rgb_mae,
            }
        )
        if bool(cfg.diagnostics.save_images):
            output_path = (
                diagnostic_dir
                / f"{observation.camera}_frame={observation.frame_index:06d}.png"
            )
            if not cv2.imwrite(
                str(output_path),
                cv2.cvtColor(diagnostic, cv2.COLOR_RGB2BGR),
            ):
                raise RuntimeError(f"诊断图写入失败: {output_path}")

    mean_rgb_mae = (
        float(np.mean(rgb_mae_values)) if rgb_mae_values else float("inf")
    )
    per_camera_counts = {
        camera: sum(item.camera == camera for item in observations)
        for camera in CAMERAS
    }
    rejection_reasons = []
    for camera, count in per_camera_counts.items():
        if count < int(cfg.confidence.min_observations_per_camera):
            rejection_reasons.append(
                f"{camera}_observations={count}<"
                f"{int(cfg.confidence.min_observations_per_camera)}"
            )
    if float(best["score"]) < float(cfg.confidence.min_match_score):
        rejection_reasons.append(
            f"match_score={float(best['score']):.4f}<"
            f"{float(cfg.confidence.min_match_score):.4f}"
        )
    if camera_disagreement > float(
        cfg.confidence.max_camera_disagreement_m
    ):
        rejection_reasons.append(
            f"camera_disagreement={camera_disagreement:.6f}>"
            f"{float(cfg.confidence.max_camera_disagreement_m):.6f}"
        )
    if frame_std > float(cfg.confidence.max_frame_std_m):
        rejection_reasons.append(
            f"frame_std={frame_std:.6f}>"
            f"{float(cfg.confidence.max_frame_std_m):.6f}"
        )
    if score_margin < float(cfg.confidence.min_score_margin):
        rejection_reasons.append(
            f"score_margin={score_margin:.4f}<"
            f"{float(cfg.confidence.min_score_margin):.4f}"
        )
    if mean_rgb_mae > float(cfg.confidence.max_background_rgb_mae):
        rejection_reasons.append(
            f"background_rgb_mae={mean_rgb_mae:.4f}>"
            f"{float(cfg.confidence.max_background_rgb_mae):.4f}"
        )

    if bool(cfg.diagnostics.save_search_scores):
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            diagnostic_dir / "search_scores.json",
            {
                "coarse": coarse_results,
                "fine": fine_results,
            },
        )

    return {
        "source_episode": int(source.episode_number),
        "status": "completed",
        "confidence": "high" if not rejection_reasons else "low",
        "rejection_reasons": rejection_reasons,
        "recovered_container_position": [-0.045, recovered_y, 0.0],
        "recovered_container_quaternion": [1.0, 0.0, 0.0, 0.0],
        "coarse_best_y": float(coarse_best["y"]),
        "coarse_best_score": float(coarse_best["score"]),
        "recovered_y": recovered_y,
        "match_score": float(best["score"]),
        "camera_scores": best["camera_scores"],
        "camera_best_y": camera_best_y,
        "camera_disagreement_m": camera_disagreement,
        "observation_best": observation_best,
        "frame_best_y_std_m": frame_std,
        "far_best_score": far_best_score,
        "score_margin": score_margin,
        "mean_background_rgb_mae": mean_rgb_mae,
        "max_replay_agent_state_abs_error": max_replay_error,
        "observations": observation_descriptions,
        "source_hashes": {
            "arrays.npz": _sha256_file(source.directory / "arrays.npz"),
            **{
                f"{camera}.mp4": _sha256_file(
                    source.directory / "videos" / f"{camera}.mp4"
                )
                for camera in CAMERAS
            },
        },
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _copy_or_link_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            logging.warning(
                "源视频与输出目录位于不同文件系统，硬链接不可用，"
                "自动复制: %s",
                source,
            )
            shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"未知文件复制模式: {mode}")


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temporary, path)


def _build_model_body_initial_state(
    env_obj,
    arrays: dict[str, np.ndarray],
    recovered_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    _restore_data_state(env_obj, arrays)
    physics = env_obj._physics
    physics.bind(env_obj._container_body).pos = np.asarray(
        [-0.045, recovered_y, 0.0],
        dtype=np.float64,
    )
    physics.bind(env_obj._container_body).quat = np.asarray(
        [1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    physics.forward()
    return (
        physics.model.body_pos.copy().astype(np.float64),
        physics.model.body_quat.copy().astype(np.float64),
    )


def _write_recovered_episode(
    env_obj,
    source: SourceEpisode,
    arrays: dict[str, np.ndarray],
    result: dict[str, Any],
    recovered_run_dir: Path,
    cfg: DictConfig,
) -> bool:
    final_dir = (
        recovered_run_dir
        / "episodes"
        / f"episode_{source.episode_number:06d}"
    )
    if final_dir.is_dir():
        arrays_path = final_dir / "arrays.npz"
        if bool(cfg.resume) and arrays_path.is_file():
            with np.load(arrays_path, allow_pickle=False) as archive:
                if all(key in archive.files for key in MODEL_BODY_INITIAL_KEYS):
                    return False
        raise FileExistsError(f"回填输出episode已存在或不完整: {final_dir}")

    temporary = final_dir.with_name(final_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        body_pos, body_quat = _build_model_body_initial_state(
            env_obj,
            arrays,
            float(result["recovered_y"]),
        )
        output_arrays = {
            key: np.asarray(value).copy() for key, value in arrays.items()
        }
        output_arrays["initial_model_body_pos"] = body_pos
        output_arrays["initial_model_body_quat"] = body_quat
        _write_npz_atomic(temporary / "arrays.npz", output_arrays)

        output_info = dict(source.info)
        output_info.update(
            {
                "episode": int(source.episode_number),
                "path": f"episodes/episode_{source.episode_number:06d}",
                "container_state_backfill": {
                    "schema_version": RECOVERY_SCHEMA_VERSION,
                    "source_run_dir": str(
                        _resolve_path(cfg.input_run_dir)
                    ),
                    "recovered_y": float(result["recovered_y"]),
                    "confidence": result["confidence"],
                    "match_score": float(result["match_score"]),
                    "camera_disagreement_m": float(
                        result["camera_disagreement_m"]
                    ),
                    "frame_best_y_std_m": float(
                        result["frame_best_y_std_m"]
                    ),
                },
            }
        )
        _write_json_atomic(temporary / "info.json", output_info)

        source_videos = source.directory / "videos"
        if source_videos.is_dir():
            for video_path in sorted(source_videos.iterdir()):
                if video_path.is_file():
                    _copy_or_link_file(
                        video_path,
                        temporary / "videos" / video_path.name,
                        str(cfg.output.video_copy_mode),
                    )
        temporary.rename(final_dir)
        return True
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _record_recovered_metadata(
    source_metadata: dict[str, Any],
    recovered_run_dir: Path,
    input_run_dir: Path,
    report_path: Path,
) -> None:
    episodes = []
    episodes_dir = recovered_run_dir / "episodes"
    if episodes_dir.is_dir():
        for directory in sorted(
            (
                path
                for path in episodes_dir.iterdir()
                if path.is_dir() and EPISODE_PATTERN.fullmatch(path.name)
            ),
            key=_episode_number,
        ):
            info_path = directory / "info.json"
            arrays_path = directory / "arrays.npz"
            if not info_path.is_file() or not arrays_path.is_file():
                raise RuntimeError(f"发现不完整回填episode: {directory}")
            episodes.append(_load_json(info_path))

    metadata = dict(source_metadata)
    metadata.update(
        {
            "run_dir": str(recovered_run_dir),
            "source_run_dir": str(input_run_dir),
            "container_state_backfill_schema_version": (
                RECOVERY_SCHEMA_VERSION
            ),
            "container_state_backfill_report": str(report_path),
            "saved_episodes": len(episodes),
            "successful_episodes": sum(
                bool(item.get("success", False)) for item in episodes
            ),
            "episodes": episodes,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    recovered_run_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(recovered_run_dir / "metadata.json", metadata)


def _semantic_config(
    cfg: DictConfig,
    input_run_dir: Path,
    sources: list[SourceEpisode],
) -> dict[str, Any]:
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "input_run_dir": str(input_run_dir),
        "source_episode_indices": [
            source.episode_number for source in sources
        ],
        "fps": int(cfg.fps),
        "cameras": list(CAMERAS),
        "render_height": int(cfg.render_height),
        "render_width": int(cfg.render_width),
        "search": OmegaConf.to_container(cfg.search, resolve=True),
        "mask": OmegaConf.to_container(cfg.mask, resolve=True),
        "matching": OmegaConf.to_container(cfg.matching, resolve=True),
        "confidence": OmegaConf.to_container(
            cfg.confidence,
            resolve=True,
        ),
        "replay": OmegaConf.to_container(cfg.replay, resolve=True),
    }


def _fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _is_safe_scope_expansion(
    existing_config: Any,
    current_config: dict[str, Any],
) -> bool:
    """只允许在恢复算法配置不变时扩大待处理episode集合。"""
    if not isinstance(existing_config, dict):
        return False
    existing = _json_safe(existing_config)
    current = _json_safe(current_config)
    existing_indices = existing.pop("source_episode_indices", None)
    current_indices = current.pop("source_episode_indices", None)
    if not isinstance(existing_indices, list) or not isinstance(
        current_indices,
        list,
    ):
        return False
    try:
        existing_set = {int(value) for value in existing_indices}
        current_set = {int(value) for value in current_indices}
    except (TypeError, ValueError):
        return False
    return existing == current and existing_set < current_set


def recover_insert_container_states(cfg: DictConfig) -> None:
    _validate_config(cfg)
    input_run_dir = _resolve_path(cfg.input_run_dir)
    report_dir = _resolve_path(cfg.report_dir)
    report_path = report_dir / "recovery_report.json"
    source_metadata_path = input_run_dir / "metadata.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(
            f"输入run缺少metadata.json: {source_metadata_path}"
        )
    source_metadata = _load_json(source_metadata_path)
    env_id = str(source_metadata.get("env_id", ""))
    if env_id != EXPECTED_ENV_ID:
        raise ValueError(
            f"该脚本仅支持{EXPECTED_ENV_ID}，当前env_id={env_id!r}。"
        )
    if int(source_metadata.get("fps", cfg.fps)) != int(cfg.fps):
        raise ValueError(
            f"源fps={source_metadata.get('fps')}与配置fps={cfg.fps}不一致。"
        )

    source_indices = (
        None
        if cfg.source_episode_indices is None
        else [int(value) for value in cfg.source_episode_indices]
    )
    sources = _load_sources(
        input_run_dir,
        source_indices,
        (
            None
            if cfg.max_source_episodes is None
            else int(cfg.max_source_episodes)
        ),
    )
    semantic_config = _semantic_config(cfg, input_run_dir, sources)
    fingerprint = _fingerprint(semantic_config)
    report_dir.mkdir(parents=True, exist_ok=True)

    if report_path.is_file():
        if not bool(cfg.resume):
            raise FileExistsError(
                f"恢复报告已存在且resume=false: {report_path}"
            )
        report = _load_json(report_path)
        if report.get("config_fingerprint") != fingerprint:
            if _is_safe_scope_expansion(
                report.get("config"),
                semantic_config,
            ):
                previous_count = len(
                    report["config"]["source_episode_indices"]
                )
                report.setdefault("scope_expansions", []).append(
                    {
                        "time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "previous_fingerprint": report.get(
                            "config_fingerprint"
                        ),
                        "current_fingerprint": fingerprint,
                        "previous_episode_count": previous_count,
                        "current_episode_count": len(sources),
                    }
                )
                report["config"] = semantic_config
                report["config_fingerprint"] = fingerprint
                report["updated_at"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                _write_json_atomic(report_path, report)
                logging.info(
                    "检测到处理范围由%d条安全扩展到%d条，"
                    "复用已有结果并继续回填。",
                    previous_count,
                    len(sources),
                )
            else:
                raise ValueError(
                    "当前回填算法配置与已有报告不一致，请使用新的"
                    "report_dir；只有episode范围扩大时允许原目录续跑。"
                    f" existing={report.get('config_fingerprint')!r}, "
                    f"current={fingerprint!r}"
                )
    else:
        report = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_run_dir": str(input_run_dir),
            "config_fingerprint": fingerprint,
            "config": semantic_config,
            "episodes": [],
            "failures": [],
        }
        _write_json_atomic(report_path, report)

    existing_results = {
        int(item["source_episode"]): item
        for item in report.get("episodes", [])
        if item.get("status") == "completed"
    }
    recovered_run_dir = _resolve_path(cfg.output.recovered_run_dir)
    env_obj = _make_environment(
        env_id,
        (),
        int(cfg.render_height),
        int(cfg.render_width),
    )
    try:
        for source in sources:
            result = existing_results.get(source.episode_number)
            arrays_path = source.directory / "arrays.npz"
            try:
                arrays = _validate_source_arrays(arrays_path)
                if all(key in arrays for key in MODEL_BODY_INITIAL_KEYS):
                    if bool(cfg.skip_existing_model_state):
                        logging.info(
                            "跳过已有model body初态: source=%06d",
                            source.episode_number,
                        )
                        continue
                    raise ValueError(
                        f"source={source.episode_number:06d}已经包含"
                        f"{list(MODEL_BODY_INITIAL_KEYS)}。"
                    )

                if result is None:
                    logging.info(
                        "开始恢复容器位置: source=%06d",
                        source.episode_number,
                    )
                    result = _recover_episode(
                        env_obj,
                        source,
                        arrays,
                        cfg,
                        report_dir / "diagnostics",
                    )
                    report["episodes"] = [
                        item
                        for item in report.get("episodes", [])
                        if int(item.get("source_episode", -1))
                        != source.episode_number
                    ]
                    report["episodes"].append(result)
                    report["episodes"].sort(
                        key=lambda item: int(item["source_episode"])
                    )
                    report["failures"] = [
                        item
                        for item in report.get("failures", [])
                        if int(item.get("source_episode", -1))
                        != source.episode_number
                    ]
                    report["updated_at"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    _write_json_atomic(report_path, report)
                    logging.info(
                        "恢复完成: source=%06d y=%.5f score=%.4f "
                        "confidence=%s camera_diff=%.6f frame_std=%.6f",
                        source.episode_number,
                        float(result["recovered_y"]),
                        float(result["match_score"]),
                        result["confidence"],
                        float(result["camera_disagreement_m"]),
                        float(result["frame_best_y_std_m"]),
                    )
                else:
                    expected_hash = result.get("source_hashes", {}).get(
                        "arrays.npz"
                    )
                    current_hash = _sha256_file(arrays_path)
                    if expected_hash != current_hash:
                        raise ValueError(
                            "源arrays.npz自恢复报告生成后发生变化: "
                            f"source={source.episode_number:06d}"
                        )
                    logging.info(
                        "复用已有恢复报告: source=%06d y=%.5f "
                        "confidence=%s",
                        source.episode_number,
                        float(result["recovered_y"]),
                        result["confidence"],
                    )

                should_write = bool(cfg.output.write_recovered_run)
                if (
                    should_write
                    and bool(cfg.output.only_high_confidence)
                    and result["confidence"] != "high"
                ):
                    logging.warning(
                        "低置信度结果不写入回填run: source=%06d reasons=%s",
                        source.episode_number,
                        result["rejection_reasons"],
                    )
                    should_write = False
                if should_write:
                    written = _write_recovered_episode(
                        env_obj,
                        source,
                        arrays,
                        result,
                        recovered_run_dir,
                        cfg,
                    )
                    logging.info(
                        "%s回填episode: source=%06d output=%s",
                        "已写入" if written else "已存在，跳过",
                        source.episode_number,
                        recovered_run_dir,
                    )
                failures_before = len(report.get("failures", []))
                report["failures"] = [
                    item
                    for item in report.get("failures", [])
                    if int(item.get("source_episode", -1))
                    != source.episode_number
                ]
                if len(report["failures"]) != failures_before:
                    report["updated_at"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    _write_json_atomic(report_path, report)
            except Exception as exc:
                failure = {
                    "source_episode": int(source.episode_number),
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                report["failures"] = [
                    item
                    for item in report.get("failures", [])
                    if int(item.get("source_episode", -1))
                    != source.episode_number
                ]
                report["failures"].append(failure)
                report["updated_at"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                _write_json_atomic(report_path, report)
                logging.exception(
                    "容器位置恢复失败: source=%06d",
                    source.episode_number,
                )
                if not bool(cfg.continue_on_error):
                    raise
    finally:
        env_obj.close()

    if bool(cfg.output.write_recovered_run):
        _record_recovered_metadata(
            source_metadata,
            recovered_run_dir,
            input_run_dir,
            report_path,
        )
    logging.info(
        "回填实验结束: completed=%d failures=%d report=%s",
        len(report.get("episodes", [])),
        len(report.get("failures", [])),
        report_path,
    )


@hydra.main(
    version_base="1.2",
    config_path="../../configs/data_collect",
    config_name="insert_container_state_backfill",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _configure_mujoco_runtime(cfg)
    recover_insert_container_states(cfg)


if __name__ == "__main__":
    main()
