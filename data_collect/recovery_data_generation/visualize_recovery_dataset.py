#!/usr/bin/env python

"""可视化专家数据与Arm/View恢复数据的任务空间覆盖范围。

该脚本读取 ``generate_contractive_*_recovery_trajectories.py`` 生成的raw
数据，通过MuJoCo正向运动学将20维关节状态转换为：

* Arm恢复：实际受扰机械臂的两指尖中心世界坐标；
* View恢复：ZED双目几何中心、相机旋转和光轴方向。

输出的主图同时显示完整专家轨迹、恢复轨迹、恢复起点、恢复完成点以及
代表性局部恢复过程。定量结果包含三维体素覆盖率、恢复数据新增体素比例、
等量抽样覆盖率对照、相对同帧专家状态的离轨距离，以及View方向误差。

默认输入是SewNeedle的Arm/View ``random_recovery_6+0`` 数据。示例：

.. code-block:: bash

   /home/dc/miniforge3/envs/AV-piper/bin/python \
       data_collect/recovery_data_generation/visualize_recovery_dataset.py

脚本只读取数据，并将结果写入 ``--output-dir``。绘图使用Pillow，无需安装
matplotlib；PNG按双栏论文宽图生成，CSV/JSON可用于后续统计和重绘。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_collect.recovery_data_generation.trajectory_replay_common import _make_environment  # noqa: E402

EPISODE_PATTERN = re.compile(
    r"^episode_(?P<source>\d{6,})(?:_aug_(?P<variant>\d{2,}))?$"
)
LEFT_CAMERA = "zed_cam_left"
RIGHT_CAMERA = "zed_cam_right"
LEFT_FINGERTIP_BODIES = ("left_left_finger_link", "left_right_finger_link")
RIGHT_FINGERTIP_BODIES = ("right_left_finger_link", "right_right_finger_link")

EXPERT_COLOR = (70, 78, 92, 175)
EXPERT_DENSITY_COLOR = (115, 124, 139)
ARM_COLOR = (230, 119, 43, 185)
ARM_DENSITY_COLOR = (230, 119, 43)
VIEW_COLOR = (0, 119, 187, 185)
VIEW_DENSITY_COLOR = (0, 119, 187)
START_COLOR = (204, 51, 17, 255)
COMPLETE_COLOR = (0, 153, 136, 255)
GRID_COLOR = (220, 224, 230, 255)
AXIS_COLOR = (30, 39, 56, 255)
BACKGROUND_COLOR = (255, 255, 255, 255)


@dataclass(frozen=True)
class RawEpisode:
    """一条raw轨迹及其恢复身份信息。"""

    directory: Path
    info: dict[str, Any]
    states: np.ndarray
    source_indices: np.ndarray
    source_episode: int
    variant_index: int
    is_augmented: bool

    @property
    def name(self) -> str:
        return self.directory.name


@dataclass(frozen=True)
class KinematicTrajectory:
    """一条轨迹的任务空间正向运动学结果。"""

    left_eef: np.ndarray
    right_eef: np.ndarray
    camera_center: np.ndarray
    camera_rotation: np.ndarray
    camera_forward: np.ndarray


@dataclass(frozen=True)
class RecoveryTrajectory:
    """一条恢复分支与同帧专家参考的任务空间配对。"""

    recovery_type: str
    episode_name: str
    source_episode: int
    variant_index: int
    perturbed_role: str
    points: np.ndarray
    reference_points: np.ndarray
    source_indices: np.ndarray
    completion_index: int
    rotations: np.ndarray | None = None
    reference_rotations: np.ndarray | None = None
    forwards: np.ndarray | None = None
    reference_forwards: np.ndarray | None = None

    @property
    def aligned_distance_m(self) -> np.ndarray:
        return np.linalg.norm(self.points - self.reference_points, axis=1)

    @property
    def initial_distance_m(self) -> float:
        return float(self.aligned_distance_m[0])

    @property
    def peak_distance_m(self) -> float:
        return float(self.aligned_distance_m.max())


@dataclass(frozen=True)
class PlotSeries:
    """二维面板中的一条折线。"""

    points: np.ndarray
    color: tuple[int, int, int, int]
    width: int = 2


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"JSON根节点必须为对象: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _parse_episode_name(name: str) -> tuple[int, int, bool]:
    match = EPISODE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"无法解析episode名称: {name}")
    variant = match.group("variant")
    return int(match.group("source")), (-1 if variant is None else int(variant)), variant is not None


def _load_episode(directory: Path, state_key: str) -> RawEpisode:
    source_from_name, variant_index, is_augmented = _parse_episode_name(directory.name)
    info_path = directory / "info.json"
    arrays_path = directory / "arrays.npz"
    if not info_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"episode缺少info.json或arrays.npz: {directory}")
    info = _read_json(info_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        if state_key not in archive:
            raise KeyError(f"{arrays_path}缺少{state_key!r}，现有字段={archive.files}")
        states = np.asarray(archive[state_key], dtype=np.float64).copy()
        if "source_frame_index" in archive:
            source_indices = np.asarray(archive["source_frame_index"], dtype=np.int64).copy()
        else:
            source_indices = np.arange(len(states), dtype=np.int64)
    if states.ndim != 2 or states.shape[1] < 20:
        raise ValueError(f"{arrays_path}:{state_key}应至少为(N,20)，当前={states.shape}")
    if len(source_indices) != len(states):
        raise ValueError(f"{arrays_path}的source_frame_index与状态长度不一致。")
    if not np.isfinite(states).all():
        raise ValueError(f"{arrays_path}:{state_key}包含NaN或Inf。")
    source_episode = int(info.get("source_episode", source_from_name))
    return RawEpisode(
        directory=directory,
        info=info,
        states=states[:, :20],
        source_indices=source_indices,
        source_episode=source_episode,
        variant_index=variant_index,
        is_augmented=is_augmented,
    )


def _scan_run(run_dir: Path, state_key: str) -> tuple[dict[int, RawEpisode], list[RawEpisode], dict[str, Any]]:
    metadata_path = run_dir / "metadata.json"
    episodes_dir = run_dir / "episodes"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"找不到metadata.json: {metadata_path}")
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"找不到episodes目录: {episodes_dir}")
    originals: dict[int, RawEpisode] = {}
    augmented: list[RawEpisode] = []
    for directory in sorted(path for path in episodes_dir.iterdir() if path.is_dir()):
        if directory.name.endswith(".tmp") or EPISODE_PATTERN.fullmatch(directory.name) is None:
            continue
        episode = _load_episode(directory, state_key)
        if episode.is_augmented:
            augmented.append(episode)
        else:
            if episode.source_episode in originals:
                raise ValueError(f"存在重复原始source_episode={episode.source_episode}")
            originals[episode.source_episode] = episode
    if not originals or not augmented:
        raise RuntimeError(
            f"{run_dir}需要同时包含原始与恢复episode，"
            f"当前original={len(originals)}, augmented={len(augmented)}。"
        )
    for episode in augmented:
        if episode.source_episode not in originals:
            raise KeyError(f"{episode.name}找不到source_episode={episode.source_episode}的原始轨迹。")
    return originals, augmented, _read_json(metadata_path)


class ForwardKinematics:
    """将20维状态批量映射为两指尖中心和双目相机位姿。"""

    def __init__(self, env_id: str, mujoco_gl: str, render_device: int | None):
        backend = str(mujoco_gl).strip().lower()
        if backend not in {"auto", "glfw", "egl", "osmesa"}:
            raise ValueError("mujoco_gl必须是auto/glfw/egl/osmesa。")
        if backend == "auto":
            os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") else "egl")
        else:
            os.environ["MUJOCO_GL"] = backend
        if render_device is not None:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(render_device))

        self.env = _make_environment(env_id, (), 64, 64)
        self.physics = self.env._physics
        self.left_joint_binding = self.physics.bind(self.env._left_joints[:6])
        self.right_joint_binding = self.physics.bind(self.env._right_joints[:6])
        self.view_joint_binding = self.physics.bind(self.env._middle_joints)
        self.left_gripper_joint_binding = self.physics.bind(self.env._left_gripper_joints)
        self.right_gripper_joint_binding = self.physics.bind(self.env._right_gripper_joints)
        left_fingertips = [
            self.env._mjcf_root.find("body", name) for name in LEFT_FINGERTIP_BODIES
        ]
        right_fingertips = [
            self.env._mjcf_root.find("body", name) for name in RIGHT_FINGERTIP_BODIES
        ]
        missing_fingertips = [
            name
            for name, body in zip(
                (*LEFT_FINGERTIP_BODIES, *RIGHT_FINGERTIP_BODIES),
                (*left_fingertips, *right_fingertips),
                strict=True,
            )
            if body is None
        ]
        if missing_fingertips:
            raise KeyError(f"MuJoCo模型缺少夹爪指尖body: {missing_fingertips}。")
        self.left_fingertip_binding = self.physics.bind(left_fingertips)
        self.right_fingertip_binding = self.physics.bind(right_fingertips)
        self.left_camera_id = self.physics.model.name2id(LEFT_CAMERA, "camera")
        self.right_camera_id = self.physics.model.name2id(RIGHT_CAMERA, "camera")
        if self.left_camera_id < 0 or self.right_camera_id < 0:
            raise KeyError(f"MuJoCo模型缺少{LEFT_CAMERA}/{RIGHT_CAMERA}。")

    def close(self) -> None:
        self.env.close()

    def compute(self, states: np.ndarray) -> KinematicTrajectory:
        states = np.asarray(states, dtype=np.float64)
        count = len(states)
        left_eef = np.empty((count, 3), dtype=np.float64)
        right_eef = np.empty((count, 3), dtype=np.float64)
        camera_center = np.empty((count, 3), dtype=np.float64)
        camera_rotation = np.empty((count, 3, 3), dtype=np.float64)
        camera_forward = np.empty((count, 3), dtype=np.float64)
        for index, state in enumerate(states):
            self.left_joint_binding.qpos = state[:6]
            self.right_joint_binding.qpos = state[7:13]
            self.view_joint_binding.qpos = state[14:20]
            left_gripper_qpos = float(
                self.env.left_gripper_unnorm_fn(np.clip(state[6], 0.0, 1.0))
            )
            right_gripper_qpos = float(
                self.env.right_gripper_unnorm_fn(np.clip(state[13], 0.0, 1.0))
            )
            self.left_gripper_joint_binding.qpos = left_gripper_qpos
            self.right_gripper_joint_binding.qpos = right_gripper_qpos
            self.physics.forward()

            # 手指body原点位于各自指尖；取两指尖中点作为夹爪TCP，避免误用腕部/法兰位置。
            left_eef[index] = np.asarray(
                self.left_fingertip_binding.xpos, dtype=np.float64
            ).mean(axis=0)
            right_eef[index] = np.asarray(
                self.right_fingertip_binding.xpos, dtype=np.float64
            ).mean(axis=0)
            left_pos = np.asarray(self.physics.data.cam_xpos[self.left_camera_id])
            right_pos = np.asarray(self.physics.data.cam_xpos[self.right_camera_id])
            left_rot = np.asarray(
                self.physics.data.cam_xmat[self.left_camera_id], dtype=np.float64
            ).reshape(3, 3)
            right_rot = np.asarray(
                self.physics.data.cam_xmat[self.right_camera_id], dtype=np.float64
            ).reshape(3, 3)
            camera_center[index] = 0.5 * (left_pos + right_pos)
            camera_rotation[index] = left_rot
            forward = -0.5 * (left_rot[:, 2] + right_rot[:, 2])
            norm = float(np.linalg.norm(forward))
            if norm <= 1e-12:
                raise RuntimeError(f"frame={index}相机光轴无效。")
            camera_forward[index] = forward / norm
        return KinematicTrajectory(
            left_eef=left_eef,
            right_eef=right_eef,
            camera_center=camera_center,
            camera_rotation=camera_rotation,
            camera_forward=camera_forward,
        )


def _infer_env_id(
    configured_env_id: str | None,
    arm_metadata: dict[str, Any],
    view_metadata: dict[str, Any],
) -> str:
    candidates = [
        str(value).strip()
        for value in (configured_env_id, arm_metadata.get("env_id"), view_metadata.get("env_id"))
        if value is not None and str(value).strip()
    ]
    if not candidates:
        raise ValueError("--env-id和两份metadata均未提供env_id。")
    canonical = candidates[0]
    if any(value != canonical for value in candidates[1:]):
        raise ValueError(f"Arm/View数据的env_id不一致: {candidates}")
    return canonical


def _completion_index(episode: RawEpisode) -> int:
    achieved = episode.info.get("recovery_achieved_source_frame")
    if achieved is not None:
        matches = np.flatnonzero(episode.source_indices == int(achieved))
        if matches.size:
            return int(matches[-1])
    actual_steps = int(episode.info.get("actual_recovery_steps", len(episode.states) - 1))
    return int(np.clip(actual_steps, 0, len(episode.states) - 1))


def _validate_originals(
    arm_originals: dict[int, RawEpisode],
    view_originals: dict[int, RawEpisode],
) -> None:
    if set(arm_originals) != set(view_originals):
        missing_arm = sorted(set(view_originals) - set(arm_originals))
        missing_view = sorted(set(arm_originals) - set(view_originals))
        raise ValueError(
            f"Arm/View原始source集合不一致，Arm缺少={missing_arm}，View缺少={missing_view}。"
        )
    for source in sorted(arm_originals):
        arm_states = arm_originals[source].states
        view_states = view_originals[source].states
        if arm_states.shape != view_states.shape or not np.allclose(
            arm_states, view_states, atol=1e-7, rtol=0.0
        ):
            raise ValueError(f"source_episode={source}在Arm/View目录中的原始状态不一致。")


def _make_arm_recovery(
    episode: RawEpisode,
    trajectory: KinematicTrajectory,
    source_trajectory: KinematicTrajectory,
) -> RecoveryTrajectory:
    if np.any(episode.source_indices < 0) or np.any(
        episode.source_indices >= len(source_trajectory.left_eef)
    ):
        raise IndexError(f"{episode.name}的source_frame_index越界。")
    role = str(episode.info.get("perturbed_arm", "")).strip().lower()
    if role == "left":
        points = trajectory.left_eef
        reference = source_trajectory.left_eef[episode.source_indices]
    elif role == "right":
        points = trajectory.right_eef
        reference = source_trajectory.right_eef[episode.source_indices]
    else:
        raise ValueError(f"{episode.name}的perturbed_arm必须为left/right，当前={role!r}。")
    return RecoveryTrajectory(
        recovery_type="arm",
        episode_name=episode.name,
        source_episode=episode.source_episode,
        variant_index=episode.variant_index,
        perturbed_role=role,
        points=points,
        reference_points=reference,
        source_indices=episode.source_indices,
        completion_index=_completion_index(episode),
    )


def _make_view_recovery(
    episode: RawEpisode,
    trajectory: KinematicTrajectory,
    source_trajectory: KinematicTrajectory,
) -> RecoveryTrajectory:
    if np.any(episode.source_indices < 0) or np.any(
        episode.source_indices >= len(source_trajectory.camera_center)
    ):
        raise IndexError(f"{episode.name}的source_frame_index越界。")
    indices = episode.source_indices
    return RecoveryTrajectory(
        recovery_type="view",
        episode_name=episode.name,
        source_episode=episode.source_episode,
        variant_index=episode.variant_index,
        perturbed_role="view",
        points=trajectory.camera_center,
        reference_points=source_trajectory.camera_center[indices],
        source_indices=indices,
        completion_index=_completion_index(episode),
        rotations=trajectory.camera_rotation,
        reference_rotations=source_trajectory.camera_rotation[indices],
        forwards=trajectory.camera_forward,
        reference_forwards=source_trajectory.camera_forward[indices],
    )


def _rotation_error_deg(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.matmul(np.swapaxes(reference, -1, -2), target)
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _voxel_keys(points: np.ndarray, voxel_size_m: float) -> set[tuple[int, int, int]]:
    quantized = np.floor(np.asarray(points, dtype=np.float64) / voxel_size_m).astype(np.int64)
    return {tuple(int(value) for value in row) for row in quantized}


def _pose_keys(
    points: np.ndarray,
    rotations: np.ndarray,
    voxel_size_m: float,
    angle_bin_deg: float,
) -> set[tuple[int, int, int, int, int, int]]:
    positions = np.floor(np.asarray(points) / voxel_size_m).astype(np.int64)
    rotation_vectors_deg = np.degrees(
        Rotation.from_matrix(np.asarray(rotations, dtype=np.float64)).as_rotvec()
    )
    orientation_bins = np.floor(
        (rotation_vectors_deg + 180.0) / angle_bin_deg
    ).astype(np.int64)
    return {
        (int(p[0]), int(p[1]), int(p[2]), int(r[0]), int(r[1]), int(r[2]))
        for p, r in zip(positions, orientation_bins, strict=True)
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    levels = (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)
    result = np.quantile(values, levels)
    return {
        name: float(value)
        for name, value in zip(("min", "p25", "median", "p75", "p95", "max"), result, strict=True)
    }


def _balanced_coverage_bootstrap(
    expert_points: np.ndarray,
    recovery_points: np.ndarray,
    voxel_size_m: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """等量比较“追加专家点”和“追加恢复点”产生的覆盖增益。"""

    expert_points = np.asarray(expert_points)
    recovery_points = np.asarray(recovery_points)
    add_count = min(len(recovery_points), len(expert_points) // 2)
    if add_count <= 0 or repeats <= 0:
        return {"repeats": 0, "sample_count_per_component": 0}
    rng = np.random.default_rng(seed)
    control_counts: list[int] = []
    recovery_counts: list[int] = []
    relative_gains: list[float] = []
    for _ in range(repeats):
        expert_indices = rng.choice(len(expert_points), size=2 * add_count, replace=False)
        recovery_indices = rng.choice(
            len(recovery_points), size=add_count, replace=len(recovery_points) < add_count
        )
        base = expert_points[expert_indices[:add_count]]
        extra_expert = expert_points[expert_indices[add_count:]]
        extra_recovery = recovery_points[recovery_indices]
        control = len(_voxel_keys(np.concatenate((base, extra_expert)), voxel_size_m))
        treatment = len(_voxel_keys(np.concatenate((base, extra_recovery)), voxel_size_m))
        control_counts.append(control)
        recovery_counts.append(treatment)
        relative_gains.append((treatment - control) / max(control, 1))
    gain = np.asarray(relative_gains, dtype=np.float64) * 100.0
    return {
        "repeats": int(repeats),
        "sample_count_per_component": int(add_count),
        "expert_plus_expert_voxels_mean": float(np.mean(control_counts)),
        "expert_plus_recovery_voxels_mean": float(np.mean(recovery_counts)),
        "relative_gain_percent_mean": float(gain.mean()),
        "relative_gain_percent_ci95": [
            float(np.quantile(gain, 0.025)),
            float(np.quantile(gain, 0.975)),
        ],
    }


def _coverage_metrics(
    name: str,
    expert_points: np.ndarray,
    recovery_points: np.ndarray,
    voxel_size_m: float,
    tube_radius_m: float,
    bootstrap_repeats: int,
    seed: int,
    expert_rotations: np.ndarray | None = None,
    recovery_rotations: np.ndarray | None = None,
    angle_bin_deg: float = 5.0,
) -> dict[str, Any]:
    expert_voxels = _voxel_keys(expert_points, voxel_size_m)
    recovery_voxels = _voxel_keys(recovery_points, voxel_size_m)
    union_voxels = expert_voxels | recovery_voxels
    novel_voxels = recovery_voxels - expert_voxels
    nearest_distance = cKDTree(expert_points).query(recovery_points, workers=-1)[0]
    result: dict[str, Any] = {
        "space": name,
        "expert_points": int(len(expert_points)),
        "recovery_points": int(len(recovery_points)),
        "voxel_size_mm": float(voxel_size_m * 1000.0),
        "expert_occupied_voxels": int(len(expert_voxels)),
        "recovery_occupied_voxels": int(len(recovery_voxels)),
        "expert_plus_recovery_occupied_voxels": int(len(union_voxels)),
        "new_voxels_from_recovery": int(len(novel_voxels)),
        "coverage_gain_percent": float(
            (len(union_voxels) - len(expert_voxels)) / max(len(expert_voxels), 1) * 100.0
        ),
        "recovery_voxel_novelty_percent": float(
            len(novel_voxels) / max(len(recovery_voxels), 1) * 100.0
        ),
        "nearest_expert_distance_cm": {
            key: value * 100.0 for key, value in _quantiles(nearest_distance).items()
        },
        "outside_expert_tube_percent": float(np.mean(nearest_distance > tube_radius_m) * 100.0),
        "expert_tube_radius_cm": float(tube_radius_m * 100.0),
        "balanced_bootstrap": _balanced_coverage_bootstrap(
            expert_points,
            recovery_points,
            voxel_size_m,
            bootstrap_repeats,
            seed,
        ),
    }
    if expert_rotations is not None and recovery_rotations is not None:
        expert_pose = _pose_keys(
            expert_points, expert_rotations, voxel_size_m, angle_bin_deg
        )
        recovery_pose = _pose_keys(
            recovery_points, recovery_rotations, voxel_size_m, angle_bin_deg
        )
        pose_union = expert_pose | recovery_pose
        result["view_pose_bins"] = {
            "orientation_bin_deg": float(angle_bin_deg),
            "expert_bins": int(len(expert_pose)),
            "recovery_bins": int(len(recovery_pose)),
            "expert_plus_recovery_bins": int(len(pose_union)),
            "new_bins_from_recovery": int(len(recovery_pose - expert_pose)),
            "coverage_gain_percent": float(
                (len(pose_union) - len(expert_pose)) / max(len(expert_pose), 1) * 100.0
            ),
        }
    return result


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _nice_ticks(low: float, high: float, count: int = 5) -> np.ndarray:
    if not np.isfinite([low, high]).all() or high <= low:
        return np.asarray([low])
    raw_step = (high - low) / max(count - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(max(raw_step, 1e-12)))
    normalized = raw_step / magnitude
    step = (1.0 if normalized <= 1.0 else 2.0 if normalized <= 2.0 else 5.0 if normalized <= 5.0 else 10.0) * magnitude
    start = math.ceil(low / step) * step
    ticks = np.arange(start, high + step * 1e-6, step)
    return ticks[ticks <= high + step * 1e-6]


def _panel_limits(point_groups: Sequence[np.ndarray], axes: tuple[int, int]) -> tuple[float, float, float, float]:
    points = np.concatenate([np.asarray(group)[:, axes] for group in point_groups if len(group)], axis=0) * 100.0
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = np.maximum(high - low, 1.0)
    margin = np.maximum(span * 0.08, 0.5)
    return float(low[0] - margin[0]), float(high[0] + margin[0]), float(low[1] - margin[1]), float(high[1] + margin[1])


def _map_points(
    points: np.ndarray,
    axes: tuple[int, int],
    limits: tuple[float, float, float, float],
    box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    x0, x1, y0, y1 = limits
    left, top, right, bottom = box
    values = np.asarray(points)[:, axes] * 100.0
    px = left + (values[:, 0] - x0) / max(x1 - x0, 1e-12) * (right - left)
    py = bottom - (values[:, 1] - y0) / max(y1 - y0, 1e-12) * (bottom - top)
    return [(int(round(x)), int(round(y))) for x, y in zip(px, py, strict=True)]


def _draw_density(
    image: Image.Image,
    points: np.ndarray,
    axes: tuple[int, int],
    limits: tuple[float, float, float, float],
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    bins: int = 90,
) -> None:
    values = np.asarray(points)[:, axes] * 100.0
    x0, x1, y0, y1 = limits
    hist, _, _ = np.histogram2d(
        values[:, 0], values[:, 1], bins=bins, range=((x0, x1), (y0, y1))
    )
    if not np.any(hist):
        return
    scaled = np.log1p(hist) / np.log1p(hist.max())
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    left, top, right, bottom = box
    cell_w = (right - left) / bins
    cell_h = (bottom - top) / bins
    for ix, iy in np.argwhere(hist > 0):
        alpha = int(18 + 82 * scaled[ix, iy])
        xa = int(left + ix * cell_w)
        xb = int(left + (ix + 1) * cell_w + 1)
        ya = int(bottom - (iy + 1) * cell_h)
        yb = int(bottom - iy * cell_h + 1)
        draw.rectangle((xa, ya, xb, yb), fill=(*color, alpha))
    image.alpha_composite(overlay)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    limits: tuple[float, float, float, float],
    x_label: str,
    y_label: str,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=AXIS_COLOR, width=2)
    tick_font = _font(22)
    label_font = _font(25)
    x0, x1, y0, y1 = limits
    for value in _nice_ticks(x0, x1):
        x = int(left + (value - x0) / max(x1 - x0, 1e-12) * (right - left))
        draw.line((x, top, x, bottom), fill=GRID_COLOR, width=1)
        label = f"{value:g}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 7), label, font=tick_font, fill=AXIS_COLOR)
    for value in _nice_ticks(y0, y1):
        y = int(bottom - (value - y0) / max(y1 - y0, 1e-12) * (bottom - top))
        draw.line((left, y, right, y), fill=GRID_COLOR, width=1)
        label = f"{value:g}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - (bbox[2] - bbox[0]) - 10, y - 12), label, font=tick_font, fill=AXIS_COLOR)
    bbox = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(
        ((left + right - (bbox[2] - bbox[0])) / 2, bottom + 42),
        x_label,
        font=label_font,
        fill=AXIS_COLOR,
    )
    y_image = Image.new("RGBA", (220, 44), (0, 0, 0, 0))
    y_draw = ImageDraw.Draw(y_image)
    y_draw.text((0, 4), y_label, font=label_font, fill=AXIS_COLOR)
    y_image = y_image.crop(y_image.getbbox()).rotate(90, expand=True)
    # Pillow坐标原点位于左上；旋转文字贴到纵轴左侧。
    draw._image.alpha_composite(y_image, (left - 85, int((top + bottom - y_image.height) / 2)))


def _draw_marker(draw: ImageDraw.ImageDraw, point: tuple[int, int], kind: str, radius: int = 7) -> None:
    x, y = point
    if kind == "start":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=START_COLOR, outline=(255, 255, 255, 255), width=2)
    else:
        polygon = ((x, y - radius - 1), (x + radius + 1, y), (x, y + radius + 1), (x - radius - 1, y))
        draw.polygon(polygon, fill=COMPLETE_COLOR)
        draw.line((*polygon, polygon[0]), fill=(255, 255, 255, 255), width=2)


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int, int], width: int = 4) -> None:
    draw.line((start, end), fill=color, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm < 2:
        return
    ux, uy = dx / norm, dy / norm
    size = 10
    left = (int(end[0] - size * ux + 0.55 * size * uy), int(end[1] - size * uy - 0.55 * size * ux))
    right = (int(end[0] - size * ux - 0.55 * size * uy), int(end[1] - size * uy + 0.55 * size * ux))
    draw.polygon((end, left, right), fill=color)


def _draw_panel(
    image: Image.Image,
    panel_rect: tuple[int, int, int, int],
    title: str,
    axes: tuple[int, int],
    expert_points: np.ndarray,
    recovery_points: np.ndarray,
    expert_lines: Sequence[np.ndarray],
    recovery_trajectories: Sequence[RecoveryTrajectory],
    recovery_color: tuple[int, int, int, int],
    recovery_density_color: tuple[int, int, int],
    x_label: str,
    y_label: str,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    px0, py0, px1, py1 = panel_rect
    plot_box = (px0 + 105, py0 + 66, px1 - 28, py1 - 86)
    limits = _panel_limits((expert_points, recovery_points), axes)
    _draw_density(image, expert_points, axes, limits, plot_box, EXPERT_DENSITY_COLOR)
    _draw_density(image, recovery_points, axes, limits, plot_box, recovery_density_color)
    draw = ImageDraw.Draw(image, "RGBA")
    for line in expert_lines:
        mapped = _map_points(line, axes, limits, plot_box)
        if len(mapped) >= 2:
            draw.line(mapped, fill=EXPERT_COLOR, width=2)
    for recovery in recovery_trajectories:
        mapped = _map_points(recovery.points, axes, limits, plot_box)
        if len(mapped) >= 2:
            draw.line(mapped, fill=recovery_color, width=3)
            completion = int(np.clip(recovery.completion_index, 0, len(mapped) - 1))
            _draw_marker(draw, mapped[0], "start", radius=4)
            _draw_marker(draw, mapped[completion], "complete", radius=4)
    _draw_axes(draw, plot_box, limits, x_label, y_label)
    draw.text((px0 + 18, py0 + 14), title, font=_font(31, bold=True), fill=AXIS_COLOR)


def _draw_local_panel(
    image: Image.Image,
    panel_rect: tuple[int, int, int, int],
    title: str,
    recovery: RecoveryTrajectory,
    axes: tuple[int, int],
    color: tuple[int, int, int, int],
    x_label: str,
    y_label: str,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    px0, py0, px1, py1 = panel_rect
    plot_box = (px0 + 105, py0 + 66, px1 - 28, py1 - 116)
    limits = _panel_limits((recovery.reference_points, recovery.points), axes)
    ref = _map_points(recovery.reference_points, axes, limits, plot_box)
    rec = _map_points(recovery.points, axes, limits, plot_box)
    draw.line(ref, fill=(65, 73, 86, 255), width=6)
    draw.line(rec, fill=color, width=7)
    completion = int(np.clip(recovery.completion_index, 0, len(rec) - 1))
    marker_indices = np.unique(np.linspace(0, completion, num=min(6, completion + 1), dtype=np.int64))
    for first, second in zip(marker_indices[:-1], marker_indices[1:], strict=True):
        _draw_arrow(draw, rec[int(first)], rec[int(second)], color, width=4)
    _draw_marker(draw, rec[0], "start", radius=10)
    _draw_marker(draw, rec[completion], "complete", radius=10)
    annotation_font = _font(22, bold=True)
    draw.text((rec[0][0] + 10, rec[0][1] - 30), "r0", font=annotation_font, fill=START_COLOR)
    draw.text((rec[completion][0] + 10, rec[completion][1] - 30), "r*", font=annotation_font, fill=COMPLETE_COLOR)
    _draw_axes(draw, plot_box, limits, x_label, y_label)
    draw.text((px0 + 18, py0 + 14), title, font=_font(31, bold=True), fill=AXIS_COLOR)
    detail = (
        f"{recovery.episode_name}, {recovery.perturbed_role}; "
        f"initial offset={recovery.initial_distance_m * 100:.2f} cm"
    )
    draw.text((px0 + 105, py1 - 34), detail, font=_font(20), fill=AXIS_COLOR)


def _draw_legend(image: Image.Image, y: int) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(22)
    entries = [
        (EXPERT_COLOR, "Expert trajectory", "line"),
        (ARM_COLOR, "Arm recovery (gripper tip)", "line"),
        (VIEW_COLOR, "View recovery", "line"),
        (START_COLOR, "Perturbed start", "circle"),
        (COMPLETE_COLOR, "Recovery completion", "diamond"),
    ]
    x = 80
    for color, label, kind in entries:
        if kind == "line":
            draw.line((x, y + 12, x + 48, y + 12), fill=color, width=6)
        elif kind == "circle":
            _draw_marker(draw, (x + 24, y + 12), "start", radius=8)
        else:
            _draw_marker(draw, (x + 24, y + 12), "complete", radius=8)
        draw.text((x + 58, y), label, font=font, fill=AXIS_COLOR)
        x += 58 + draw.textlength(label, font=font) + 42


def _evenly_subsample(values: Sequence[Any], maximum: int) -> list[Any]:
    """确定性均匀抽取绘图曲线；覆盖统计仍使用全部数据。"""

    if len(values) <= maximum:
        return list(values)
    indices = np.unique(np.linspace(0, len(values) - 1, num=maximum, dtype=np.int64))
    return [values[int(index)] for index in indices]


def _plot_coverage_overview(
    output_path: Path,
    arm_expert_points: np.ndarray,
    arm_recovery_points: np.ndarray,
    arm_expert_lines: Sequence[np.ndarray],
    arm_recoveries: Sequence[RecoveryTrajectory],
    view_expert_points: np.ndarray,
    view_recovery_points: np.ndarray,
    view_expert_lines: Sequence[np.ndarray],
    view_recoveries: Sequence[RecoveryTrajectory],
    arm_metrics: dict[str, Any],
    view_metrics: dict[str, Any],
    max_plotted_trajectories: int,
) -> None:
    width, height = 3000, 2070
    image = Image.new("RGBA", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((80, 35), "Task-space coverage induced by recovery trajectories", font=_font(43, bold=True), fill=AXIS_COLOR)
    _draw_legend(image, 96)

    margin, gap = 55, 30
    top = 160
    panel_width = (width - 2 * margin - 2 * gap) // 3
    panel_height = 820
    row_gap = 75
    rects: list[tuple[int, int, int, int]] = []
    for row in range(2):
        for column in range(3):
            x0 = margin + column * (panel_width + gap)
            y0 = top + row * (panel_height + row_gap)
            rect = (x0, y0, x0 + panel_width, y0 + panel_height)
            rects.append(rect)
            draw.rounded_rectangle(rect, radius=18, outline=(24, 48, 82, 255), width=4)

    plotted_arm_expert = _evenly_subsample(arm_expert_lines, max_plotted_trajectories)
    plotted_view_expert = _evenly_subsample(view_expert_lines, max_plotted_trajectories)
    plotted_arm_recovery = _evenly_subsample(arm_recoveries, max_plotted_trajectories)
    plotted_view_recovery = _evenly_subsample(view_recoveries, max_plotted_trajectories)

    _draw_panel(
        image, rects[0], "(a) Gripper-tip coverage, top view", (0, 1),
        arm_expert_points, arm_recovery_points, plotted_arm_expert,
        plotted_arm_recovery, ARM_COLOR, ARM_DENSITY_COLOR,
        "World x (cm)", "World y (cm)",
    )
    _draw_panel(
        image, rects[1], "(b) Gripper-tip coverage, side view", (1, 2),
        arm_expert_points, arm_recovery_points, plotted_arm_expert,
        plotted_arm_recovery, ARM_COLOR, ARM_DENSITY_COLOR,
        "World y (cm)", "World z (cm)",
    )
    representative_arm = max(arm_recoveries, key=lambda item: item.initial_distance_m)
    _draw_local_panel(
        image, rects[2], "(c) Representative gripper-tip recovery", representative_arm,
        (0, 2), ARM_COLOR, "World x (cm)", "World z (cm)",
    )
    _draw_panel(
        image, rects[3], "(d) View coverage, top view", (0, 1),
        view_expert_points, view_recovery_points, plotted_view_expert,
        plotted_view_recovery, VIEW_COLOR, VIEW_DENSITY_COLOR,
        "World x (cm)", "World y (cm)",
    )
    _draw_panel(
        image, rects[4], "(e) View coverage, side view", (1, 2),
        view_expert_points, view_recovery_points, plotted_view_expert,
        plotted_view_recovery, VIEW_COLOR, VIEW_DENSITY_COLOR,
        "World y (cm)", "World z (cm)",
    )
    representative_view = max(view_recoveries, key=lambda item: item.initial_distance_m)
    _draw_local_panel(
        image, rects[5], "(f) Representative View recovery", representative_view,
        (0, 2), VIEW_COLOR, "World x (cm)", "World z (cm)",
    )

    summary_font = _font(21)
    summary_y = height - 73
    arm_text = (
        f"Gripper tip: +{arm_metrics['coverage_gain_percent']:.1f}% occupied 3D voxels; "
        f"{arm_metrics['outside_expert_tube_percent']:.1f}% recovery states outside "
        f"the {arm_metrics['expert_tube_radius_cm']:.1f} cm expert tube."
    )
    view_text = (
        f"View: +{view_metrics['coverage_gain_percent']:.1f}% position voxels; "
        f"+{view_metrics['view_pose_bins']['coverage_gain_percent']:.1f}% camera pose bins."
    )
    draw.text((80, summary_y), arm_text, font=summary_font, fill=AXIS_COLOR)
    draw.text((1610, summary_y), view_text, font=summary_font, fill=AXIS_COLOR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG", dpi=(300, 300), optimize=True)


def _write_recovery_summary(
    path: Path,
    arm_recoveries: Sequence[RecoveryTrajectory],
    view_recoveries: Sequence[RecoveryTrajectory],
) -> None:
    fields = (
        "recovery_type", "episode_name", "source_episode", "variant_index",
        "perturbed_role", "frames", "completion_index", "initial_distance_cm",
        "mean_aligned_distance_cm", "peak_aligned_distance_cm",
        "initial_rotation_error_deg", "mean_rotation_error_deg", "peak_rotation_error_deg",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for recovery in [*arm_recoveries, *view_recoveries]:
            distances = recovery.aligned_distance_m * 100.0
            if recovery.rotations is not None and recovery.reference_rotations is not None:
                rotation = _rotation_error_deg(recovery.reference_rotations, recovery.rotations)
                initial_rotation, mean_rotation, peak_rotation = (
                    float(rotation[0]), float(rotation.mean()), float(rotation.max())
                )
            else:
                initial_rotation = mean_rotation = peak_rotation = float("nan")
            writer.writerow(
                {
                    "recovery_type": recovery.recovery_type,
                    "episode_name": recovery.episode_name,
                    "source_episode": recovery.source_episode,
                    "variant_index": recovery.variant_index,
                    "perturbed_role": recovery.perturbed_role,
                    "frames": len(recovery.points),
                    "completion_index": recovery.completion_index,
                    "initial_distance_cm": f"{distances[0]:.8f}",
                    "mean_aligned_distance_cm": f"{distances.mean():.8f}",
                    "peak_aligned_distance_cm": f"{distances.max():.8f}",
                    "initial_rotation_error_deg": f"{initial_rotation:.8f}",
                    "mean_rotation_error_deg": f"{mean_rotation:.8f}",
                    "peak_rotation_error_deg": f"{peak_rotation:.8f}",
                }
            )


def _write_trajectory_points(
    path: Path,
    expert_kinematics: dict[int, KinematicTrajectory],
    arm_recoveries: Sequence[RecoveryTrajectory],
    view_recoveries: Sequence[RecoveryTrajectory],
) -> None:
    fields = (
        "data_role", "recovery_type", "episode_name", "source_episode", "variant_index",
        "perturbed_role", "frame_index", "source_frame_index", "x_m", "y_m", "z_m",
        "is_recovery_start", "is_recovery_complete",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source, trajectory in sorted(expert_kinematics.items()):
            for role, points in (
                ("left", trajectory.left_eef),
                ("right", trajectory.right_eef),
                ("view", trajectory.camera_center),
            ):
                recovery_type = "view" if role == "view" else "arm"
                for index, point in enumerate(points):
                    writer.writerow(
                        {
                            "data_role": "expert", "recovery_type": recovery_type,
                            "episode_name": f"episode_{source:06d}", "source_episode": source,
                            "variant_index": -1, "perturbed_role": role, "frame_index": index,
                            "source_frame_index": index, "x_m": f"{point[0]:.10f}",
                            "y_m": f"{point[1]:.10f}", "z_m": f"{point[2]:.10f}",
                            "is_recovery_start": False, "is_recovery_complete": False,
                        }
                    )
        for recovery in [*arm_recoveries, *view_recoveries]:
            for index, (source_index, point) in enumerate(
                zip(recovery.source_indices, recovery.points, strict=True)
            ):
                writer.writerow(
                    {
                        "data_role": "recovery", "recovery_type": recovery.recovery_type,
                        "episode_name": recovery.episode_name,
                        "source_episode": recovery.source_episode,
                        "variant_index": recovery.variant_index,
                        "perturbed_role": recovery.perturbed_role, "frame_index": index,
                        "source_frame_index": int(source_index), "x_m": f"{point[0]:.10f}",
                        "y_m": f"{point[1]:.10f}", "z_m": f"{point[2]:.10f}",
                        "is_recovery_start": index == 0,
                        "is_recovery_complete": index == recovery.completion_index,
                    }
                )


def _write_metrics_csv(path: Path, metrics: dict[str, dict[str, Any]]) -> None:
    fields = (
        "space", "expert_points", "recovery_points", "voxel_size_mm",
        "expert_occupied_voxels", "expert_plus_recovery_occupied_voxels",
        "new_voxels_from_recovery", "coverage_gain_percent",
        "recovery_voxel_novelty_percent", "outside_expert_tube_percent",
        "balanced_gain_percent_mean", "balanced_gain_ci95_low", "balanced_gain_ci95_high",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in ("arm", "view"):
            value = metrics[key]
            bootstrap = value["balanced_bootstrap"]
            ci = bootstrap.get("relative_gain_percent_ci95", [float("nan"), float("nan")])
            writer.writerow(
                {
                    "space": value["space"],
                    "expert_points": value["expert_points"],
                    "recovery_points": value["recovery_points"],
                    "voxel_size_mm": value["voxel_size_mm"],
                    "expert_occupied_voxels": value["expert_occupied_voxels"],
                    "expert_plus_recovery_occupied_voxels": value["expert_plus_recovery_occupied_voxels"],
                    "new_voxels_from_recovery": value["new_voxels_from_recovery"],
                    "coverage_gain_percent": f"{value['coverage_gain_percent']:.8f}",
                    "recovery_voxel_novelty_percent": f"{value['recovery_voxel_novelty_percent']:.8f}",
                    "outside_expert_tube_percent": f"{value['outside_expert_tube_percent']:.8f}",
                    "balanced_gain_percent_mean": f"{bootstrap.get('relative_gain_percent_mean', float('nan')):.8f}",
                    "balanced_gain_ci95_low": f"{ci[0]:.8f}",
                    "balanced_gain_ci95_high": f"{ci[1]:.8f}",
                }
            )


def _caption_text(metrics: dict[str, dict[str, Any]]) -> str:
    arm = metrics["arm"]
    view = metrics["view"]
    return (
        "Figure X\n\n"
        "Task-space coverage induced by recovery trajectory augmentation\n\n"
        "Note. Gray curves denote expert demonstrations. Orange and blue curves denote "
        "Arm and View recovery trajectories, respectively. Red circles indicate perturbed "
        "initial states, and teal diamonds indicate the first state at which the configured "
        "recovery criterion was satisfied for three consecutive frames. All density layers and "
        "reported metrics use every state; overlaid curves are a deterministic subset for visual "
        "clarity. Recovery augmentation increased occupied "
        f"{arm['voxel_size_mm']:.1f}-mm Arm workspace voxels by "
        f"{arm['coverage_gain_percent']:.1f}% and View position voxels by "
        f"{view['coverage_gain_percent']:.1f}%. View position-orientation coverage increased "
        f"by {view['view_pose_bins']['coverage_gain_percent']:.1f}% under "
        f"{view['view_pose_bins']['orientation_bin_deg']:.1f}-degree orientation bins."
    )


def _rounded_points_cm(points: np.ndarray) -> list[list[float]]:
    """压缩交互HTML中的坐标体积，同时保留0.01 mm显示精度。"""

    return np.round(np.asarray(points, dtype=np.float64) * 100.0, 3).tolist()


def _interactive_dataset_payload(
    expert_kinematics: dict[int, KinematicTrajectory],
    recoveries: Sequence[RecoveryTrajectory],
    recovery_type: str,
    maximum: int,
) -> dict[str, Any]:
    expert_lines: list[dict[str, Any]] = []
    if recovery_type == "arm":
        for source, trajectory in sorted(expert_kinematics.items()):
            expert_lines.extend(
                (
                    {
                        "name": f"episode_{source:06d}: left",
                        "role": "left",
                        "points": _rounded_points_cm(trajectory.left_eef),
                    },
                    {
                        "name": f"episode_{source:06d}: right",
                        "role": "right",
                        "points": _rounded_points_cm(trajectory.right_eef),
                    },
                )
            )
    else:
        for source, trajectory in sorted(expert_kinematics.items()):
            expert_lines.append(
                {
                    "name": f"episode_{source:06d}: view",
                    "role": "view",
                    "points": _rounded_points_cm(trajectory.camera_center),
                }
            )

    sampled_expert = _evenly_subsample(expert_lines, maximum)
    sampled_recovery = _evenly_subsample(recoveries, maximum)
    recovery_lines = []
    for recovery in sampled_recovery:
        completion = int(np.clip(recovery.completion_index, 0, len(recovery.points) - 1))
        recovery_lines.append(
            {
                "name": recovery.episode_name,
                "role": recovery.perturbed_role,
                "points": _rounded_points_cm(recovery.points),
                "reference": _rounded_points_cm(recovery.reference_points),
                "completion": completion,
            }
        )
    return {
        "expert": sampled_expert,
        "recovery": recovery_lines,
        "total_expert": len(expert_lines),
        "total_recovery": len(recoveries),
    }


def _write_interactive_3d_html(
    output_path: Path,
    expert_kinematics: dict[int, KinematicTrajectory],
    arm_recoveries: Sequence[RecoveryTrajectory],
    view_recoveries: Sequence[RecoveryTrajectory],
    maximum: int,
) -> None:
    """写入不依赖网络或第三方JavaScript库的可拖动3D轨迹查看器。"""

    payload = {
        "arm": _interactive_dataset_payload(
            expert_kinematics, arm_recoveries, "arm", maximum
        ),
        "view": _interactive_dataset_payload(
            expert_kinematics, view_recoveries, "view", maximum
        ),
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # 防止极端episode名称意外提前结束script标签。
    payload_json = payload_json.replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Interactive recovery trajectory coverage</title>
<style>
  :root { color-scheme: light; --navy:#182738; --line:#dce3eb; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:Arial,"Noto Sans SC",sans-serif; color:var(--navy); background:#f4f7fa; }
  header { padding:16px 22px 10px; background:white; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 7px; font-size:22px; }
  .hint { color:#536273; font-size:13px; }
  .toolbar { display:flex; gap:14px; flex-wrap:wrap; align-items:center; padding:10px 22px; background:white; border-bottom:1px solid var(--line); }
  .group { display:flex; gap:9px; align-items:center; padding-right:14px; border-right:1px solid var(--line); }
  button, select { border:1px solid #aebaca; border-radius:6px; padding:6px 10px; color:var(--navy); background:white; cursor:pointer; }
  button.active { background:#183a65; color:white; border-color:#183a65; }
  label { font-size:13px; user-select:none; }
  input { vertical-align:middle; }
  #stage { position:relative; width:100%; height:calc(100vh - 138px); min-height:520px; background:white; overflow:hidden; }
  canvas { width:100%; height:100%; display:block; touch-action:none; cursor:grab; }
  canvas.dragging { cursor:grabbing; }
  #status { position:absolute; left:16px; bottom:14px; padding:7px 10px; border-radius:6px; color:#34465a; background:rgba(255,255,255,.9); border:1px solid var(--line); font-size:12px; pointer-events:none; }
  #legend { position:absolute; right:16px; top:14px; padding:9px 11px; border-radius:7px; background:rgba(255,255,255,.92); border:1px solid var(--line); font-size:12px; line-height:1.8; }
  .swatch { display:inline-block; width:24px; height:3px; margin-right:7px; vertical-align:middle; }
</style>
</head>
<body>
<header>
  <h1>Interactive 3D recovery trajectory coverage</h1>
  <div class="hint">左键拖动旋转 · 滚轮缩放 · 右键拖动平移 · 双击恢复默认视角。坐标单位为厘米。</div>
</header>
<div class="toolbar">
  <div class="group"><button id="armTab" class="active">Arm workspace</button><button id="viewTab">View workspace</button></div>
  <div class="group">
    <label><input id="showExpert" type="checkbox" checked> Expert</label>
    <label><input id="showRecovery" type="checkbox" checked> Recovery</label>
    <label><input id="showReference" type="checkbox"> Aligned expert reference</label>
    <label><input id="showStart" type="checkbox" checked> Start</label>
    <label><input id="showComplete" type="checkbox" checked> Completion</label>
  </div>
  <div class="group"><button id="resetView">Reset view</button></div>
</div>
<div id="stage">
  <canvas id="canvas"></canvas>
  <div id="legend"><div><span class="swatch" style="background:#46505f"></span>Expert</div><div><span id="recoverySwatch" class="swatch" style="background:#e6772b"></span>Recovery</div><div>● Perturbed start</div><div>◆ Recovery completion</div></div>
  <div id="status"></div>
</div>
<script>
const DATA = __PAYLOAD__;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const stage = document.getElementById('stage');
const controls = ['showExpert','showRecovery','showReference','showStart','showComplete'];
let mode='arm', yaw=-0.72, pitch=0.58, zoom=1, panX=0, panY=0;
let dragging=false, dragButton=0, lastX=0, lastY=0;

function resetCamera(){ yaw=-0.72; pitch=0.58; zoom=1; panX=0; panY=0; draw(); }
function visible(id){ return document.getElementById(id).checked; }
function lines(){ return DATA[mode]; }
function allPoints(){
  const d=lines(), out=[];
  for(const line of d.expert) for(const p of line.points) out.push(p);
  for(const line of d.recovery) for(const p of line.points) out.push(p);
  return out;
}
function bounds(){
  const pts=allPoints(); let lo=[Infinity,Infinity,Infinity], hi=[-Infinity,-Infinity,-Infinity];
  for(const p of pts) for(let i=0;i<3;i++){ lo[i]=Math.min(lo[i],p[i]); hi[i]=Math.max(hi[i],p[i]); }
  const center=lo.map((v,i)=>(v+hi[i])/2), span=Math.max(...lo.map((v,i)=>hi[i]-v),1);
  return {lo,hi,center,span};
}
function rotate(p,b){
  let x=p[0]-b.center[0], y=p[1]-b.center[1], z=p[2]-b.center[2];
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const x1=cy*x-sy*y, y1=sy*x+cy*y;
  return [x1, cp*z-sp*y1, sp*z+cp*y1];
}
function project(p,b){
  const r=rotate(p,b), s=Math.min(canvas.clientWidth,canvas.clientHeight)*0.72/b.span*zoom;
  return [canvas.clientWidth/2+panX+r[0]*s, canvas.clientHeight/2+panY-r[1]*s, r[2]];
}
function drawLine(points,b,color,width,alpha=1){
  if(points.length<2) return;
  ctx.beginPath(); const p0=project(points[0],b); ctx.moveTo(p0[0],p0[1]);
  for(let i=1;i<points.length;i++){ const p=project(points[i],b); ctx.lineTo(p[0],p[1]); }
  ctx.strokeStyle=color; ctx.globalAlpha=alpha; ctx.lineWidth=width; ctx.stroke(); ctx.globalAlpha=1;
}
function marker(p,b,color,diamond=false){
  const q=project(p,b); ctx.beginPath();
  if(diamond){ ctx.moveTo(q[0],q[1]-5); ctx.lineTo(q[0]+5,q[1]); ctx.lineTo(q[0],q[1]+5); ctx.lineTo(q[0]-5,q[1]); ctx.closePath(); }
  else ctx.arc(q[0],q[1],4.2,0,Math.PI*2);
  ctx.fillStyle=color; ctx.fill(); ctx.strokeStyle='white'; ctx.lineWidth=1.2; ctx.stroke();
}
function drawAxes(b){
  const origin=b.center, length=b.span*0.22;
  const axes=[[[origin[0]+length,origin[1],origin[2]],'#cc3311','X'],[[origin[0],origin[1]+length,origin[2]],'#009988','Y'],[[origin[0],origin[1],origin[2]+length],'#0077bb','Z']];
  const o=project(origin,b);
  ctx.font='bold 13px Arial';
  for(const [end,color,label] of axes){ const e=project(end,b); ctx.beginPath();ctx.moveTo(o[0],o[1]);ctx.lineTo(e[0],e[1]);ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.stroke();ctx.fillStyle=color;ctx.fillText(label,e[0]+4,e[1]-4); }
}
function drawBox(b){
  const [x0,y0,z0]=b.lo,[x1,y1,z1]=b.hi;
  const c=[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]];
  const edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  ctx.strokeStyle='#d8e0e8';ctx.lineWidth=1;
  for(const [a,d] of edges){ const p=project(c[a],b),q=project(c[d],b);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke(); }
}
function resize(){ const dpr=window.devicePixelRatio||1,w=stage.clientWidth,h=stage.clientHeight; canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);canvas.style.width=w+'px';canvas.style.height=h+'px';ctx.setTransform(dpr,0,0,dpr,0,0);draw(); }
function draw(){
  const w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);ctx.fillStyle='white';ctx.fillRect(0,0,w,h);
  const d=lines(),b=bounds(),recoveryColor=mode==='arm'?'#e6772b':'#0077bb'; drawBox(b);
  if(visible('showExpert')) for(const line of d.expert) drawLine(line.points,b,'#46505f',1.1,.55);
  if(visible('showReference')) for(const line of d.recovery) drawLine(line.reference,b,'#009988',1.1,.5);
  if(visible('showRecovery')) for(const line of d.recovery) drawLine(line.points,b,recoveryColor,1.8,.82);
  for(const line of d.recovery){
    if(visible('showStart')) marker(line.points[0],b,'#cc3311',false);
    if(visible('showComplete')) marker(line.points[Math.min(line.completion,line.points.length-1)],b,'#009988',true);
  }
  drawAxes(b);
  document.getElementById('status').textContent=`${mode.toUpperCase()} · displayed ${d.expert.length}/${d.total_expert} expert and ${d.recovery.length}/${d.total_recovery} recovery trajectories · x/y/z range ${b.span.toFixed(1)} cm`;
}
function setMode(next){ mode=next;document.getElementById('armTab').classList.toggle('active',mode==='arm');document.getElementById('viewTab').classList.toggle('active',mode==='view');document.getElementById('recoverySwatch').style.background=mode==='arm'?'#e6772b':'#0077bb';resetCamera(); }
document.getElementById('armTab').onclick=()=>setMode('arm'); document.getElementById('viewTab').onclick=()=>setMode('view'); document.getElementById('resetView').onclick=resetCamera;
for(const id of controls) document.getElementById(id).onchange=draw;
canvas.oncontextmenu=e=>e.preventDefault();
canvas.onpointerdown=e=>{dragging=true;dragButton=e.button;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging');};
canvas.onpointermove=e=>{if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;if(dragButton===2){panX+=dx;panY+=dy;}else{yaw+=dx*.009;pitch=Math.max(-1.48,Math.min(1.48,pitch+dy*.009));}draw();};
canvas.onpointerup=e=>{dragging=false;canvas.releasePointerCapture(e.pointerId);canvas.classList.remove('dragging');};
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.15,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));draw();};
canvas.ondblclick=resetCamera; window.addEventListener('resize',resize); resize();
</script>
</body>
</html>
'''
    output_path.write_text(
        template.replace("__PAYLOAD__", payload_json), encoding="utf-8"
    )


def visualize_recovery_dataset(args: argparse.Namespace) -> Path:
    arm_run_dir = _resolve_path(args.arm_run_dir)
    view_run_dir = _resolve_path(args.view_run_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("读取Arm恢复数据: %s", arm_run_dir)
    arm_originals, arm_augmented, arm_metadata = _scan_run(arm_run_dir, args.state_key)
    logging.info("读取View恢复数据: %s", view_run_dir)
    view_originals, view_augmented, view_metadata = _scan_run(view_run_dir, args.state_key)
    _validate_originals(arm_originals, view_originals)
    if args.max_branches is not None:
        arm_augmented = arm_augmented[: int(args.max_branches)]
        view_augmented = view_augmented[: int(args.max_branches)]
    env_id = _infer_env_id(args.env_id, arm_metadata, view_metadata)

    logging.info(
        "正向运动学: env=%s, originals=%d, arm=%d, view=%d",
        env_id, len(arm_originals), len(arm_augmented), len(view_augmented),
    )
    fk = ForwardKinematics(env_id, args.mujoco_gl, args.render_device)
    try:
        expert_kinematics: dict[int, KinematicTrajectory] = {}
        for index, (source, episode) in enumerate(sorted(arm_originals.items()), start=1):
            expert_kinematics[source] = fk.compute(episode.states[:: args.frame_stride])
            if index % 20 == 0 or index == len(arm_originals):
                logging.info("已完成专家轨迹FK: %d/%d", index, len(arm_originals))

        arm_recoveries: list[RecoveryTrajectory] = []
        for index, episode in enumerate(arm_augmented, start=1):
            arm_recoveries.append(
                _make_arm_recovery(
                    episode,
                    fk.compute(episode.states),
                    expert_kinematics[episode.source_episode],
                )
            )
            if index % 50 == 0 or index == len(arm_augmented):
                logging.info("已完成Arm恢复FK: %d/%d", index, len(arm_augmented))

        view_recoveries: list[RecoveryTrajectory] = []
        for index, episode in enumerate(view_augmented, start=1):
            view_recoveries.append(
                _make_view_recovery(
                    episode,
                    fk.compute(episode.states),
                    expert_kinematics[episode.source_episode],
                )
            )
            if index % 50 == 0 or index == len(view_augmented):
                logging.info("已完成View恢复FK: %d/%d", index, len(view_augmented))
    finally:
        fk.close()

    arm_expert_lines = [
        points
        for trajectory in expert_kinematics.values()
        for points in (trajectory.left_eef, trajectory.right_eef)
    ]
    view_expert_lines = [trajectory.camera_center for trajectory in expert_kinematics.values()]
    arm_expert_points = np.concatenate(arm_expert_lines, axis=0)
    view_expert_points = np.concatenate(view_expert_lines, axis=0)
    view_expert_rotations = np.concatenate(
        [trajectory.camera_rotation for trajectory in expert_kinematics.values()], axis=0
    )
    arm_recovery_points = np.concatenate([item.points for item in arm_recoveries], axis=0)
    view_recovery_points = np.concatenate([item.points for item in view_recoveries], axis=0)
    view_recovery_rotations = np.concatenate(
        [item.rotations for item in view_recoveries if item.rotations is not None], axis=0
    )

    voxel_size_m = float(args.voxel_size_mm) / 1000.0
    metrics = {
        "arm": _coverage_metrics(
            "arm_gripper_fingertip_center",
            arm_expert_points,
            arm_recovery_points,
            voxel_size_m,
            float(args.arm_tube_cm) / 100.0,
            args.bootstrap_repeats,
            args.seed,
        ),
        "view": _coverage_metrics(
            "view_camera_pose",
            view_expert_points,
            view_recovery_points,
            voxel_size_m,
            float(args.view_tube_cm) / 100.0,
            args.bootstrap_repeats,
            args.seed + 1,
            expert_rotations=view_expert_rotations,
            recovery_rotations=view_recovery_rotations,
            angle_bin_deg=float(args.view_angle_bin_deg),
        ),
    }

    aligned_arm = np.concatenate([item.aligned_distance_m for item in arm_recoveries])
    aligned_view = np.concatenate([item.aligned_distance_m for item in view_recoveries])
    view_rotation = np.concatenate(
        [
            _rotation_error_deg(item.reference_rotations, item.rotations)
            for item in view_recoveries
            if item.rotations is not None and item.reference_rotations is not None
        ]
    )
    metrics["arm"]["temporally_aligned_distance_cm"] = {
        key: value * 100.0 for key, value in _quantiles(aligned_arm).items()
    }
    metrics["view"]["temporally_aligned_distance_cm"] = {
        key: value * 100.0 for key, value in _quantiles(aligned_view).items()
    }
    metrics["view"]["temporally_aligned_rotation_error_deg"] = _quantiles(view_rotation)
    metrics["dataset"] = {
        "env_id": env_id,
        "arm_run_dir": str(arm_run_dir),
        "view_run_dir": str(view_run_dir),
        "original_episodes": len(expert_kinematics),
        "arm_recovery_episodes": len(arm_recoveries),
        "view_recovery_episodes": len(view_recoveries),
        "max_plotted_trajectories_per_role": args.max_plotted_trajectories,
        "state_key": args.state_key,
        "arm_tracking_point": "midpoint_of_two_fingertip_body_origins",
    }

    figure_path = output_dir / "workspace_coverage.png"
    _plot_coverage_overview(
        figure_path,
        arm_expert_points,
        arm_recovery_points,
        arm_expert_lines,
        arm_recoveries,
        view_expert_points,
        view_recovery_points,
        view_expert_lines,
        view_recoveries,
        metrics["arm"],
        metrics["view"],
        args.max_plotted_trajectories,
    )
    interactive_path = output_dir / "workspace_coverage_3d.html"
    if not args.skip_interactive_html:
        _write_interactive_3d_html(
            interactive_path,
            expert_kinematics,
            arm_recoveries,
            view_recoveries,
            args.max_plotted_trajectories,
        )
    _write_json(output_dir / "coverage_metrics.json", metrics)
    _write_metrics_csv(output_dir / "coverage_metrics.csv", metrics)
    _write_recovery_summary(output_dir / "recovery_trajectory_summary.csv", arm_recoveries, view_recoveries)
    if not args.skip_points_csv:
        _write_trajectory_points(
            output_dir / "trajectory_points.csv",
            expert_kinematics,
            arm_recoveries,
            view_recoveries,
        )
    (output_dir / "figure_caption.txt").write_text(
        _caption_text(metrics) + "\n", encoding="utf-8"
    )
    representatives = {
        "arm": max(arm_recoveries, key=lambda item: item.initial_distance_m).episode_name,
        "view": max(view_recoveries, key=lambda item: item.initial_distance_m).episode_name,
    }
    _write_json(output_dir / "representative_recoveries.json", representatives)

    logging.info("覆盖轨迹图: %s", figure_path)
    if not args.skip_interactive_html:
        logging.info("交互式3D轨迹: %s", interactive_path)
    logging.info("覆盖率统计: %s", output_dir / "coverage_metrics.json")
    logging.info("逐分支统计: %s", output_dir / "recovery_trajectory_summary.csv")
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将Arm/View恢复数据投影到真实任务空间，并比较专家与恢复数据覆盖范围。"
    )
    parser.add_argument("--arm-run-dir", default="outputs/4_data_collect/quest_teleop_recovery/SewNeedle-3Arms/quest_teleop_SewNeedle-3Arms-v0_rgb_arm_random_recovery_3+0", help="Arm恢复raw run目录。")
    parser.add_argument("--view-run-dir", default="outputs/4_data_collect/quest_teleop_recovery/SewNeedle-3Arms/quest_teleop_SewNeedle-3Arms-v0_rgb_view_random_recovery_3+0", help="View恢复raw run目录。")
    parser.add_argument("--output-dir", default="outputs/4_data_collect/recovery_dataset_visualization/SewNeedle-3Arms_3+0", help="可视化输出目录。")
    parser.add_argument("--env-id", default=None, help="MuJoCo环境ID；默认从metadata推断。")
    parser.add_argument("--state-key", default="observation_state", help="arrays.npz中的20维状态字段。")
    parser.add_argument("--voxel-size-mm", type=float, default=5.0, help="三维覆盖率体素边长，单位mm。")
    parser.add_argument("--arm-tube-cm", type=float, default=1.0, help="Arm专家轨迹管道半径，单位cm。")
    parser.add_argument("--view-tube-cm", type=float, default=1.0, help="View专家位置管道半径，单位cm。")
    parser.add_argument("--view-angle-bin-deg", type=float, default=5.0, help="View姿态覆盖的方向角分箱宽度。")
    parser.add_argument("--bootstrap-repeats", type=int, default=100, help="等量抽样覆盖率对照重复次数。")
    parser.add_argument("--seed", type=int, default=20260812, help="等量抽样随机种子。")
    parser.add_argument("--max-branches", type=int, default=None, help="每类最多分析多少恢复分支；默认全部。")
    parser.add_argument(
        "--max-plotted-trajectories",
        type=int,
        default=300,
        help="每类最多叠加多少条折线；密度和统计始终使用全部数据。",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="FK帧步长；当前同帧配准要求为1。")
    parser.add_argument("--skip-points-csv", action="store_true", help="不输出体积较大的trajectory_points.csv。")
    parser.add_argument(
        "--skip-interactive-html",
        action="store_true",
        help="不输出可离线拖动的workspace_coverage_3d.html。",
    )
    parser.add_argument("--mujoco-gl", default="egl", choices=("auto", "glfw", "egl", "osmesa"), help="MuJoCo图形后端。")
    parser.add_argument("--render-device", type=int, default=None, help="可选EGL设备编号。")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not np.isfinite(args.voxel_size_mm) or args.voxel_size_mm <= 0:
        raise ValueError("--voxel-size-mm必须为正数。")
    for name in ("arm_tube_cm", "view_tube_cm", "view_angle_bin_deg"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')}必须为正数。")
    if args.bootstrap_repeats < 0:
        raise ValueError("--bootstrap-repeats不能为负数。")
    if args.max_branches is not None and args.max_branches <= 0:
        raise ValueError("--max-branches必须为正整数。")
    if args.max_plotted_trajectories <= 0:
        raise ValueError("--max-plotted-trajectories必须为正整数。")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride必须为正整数。")
    if args.frame_stride != 1:
        raise ValueError("当前同帧恢复配准要求--frame-stride=1。")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args()
    _validate_args(args)
    output_dir = visualize_recovery_dataset(args)
    logging.info("恢复数据可视化完成: %s", output_dir)


if __name__ == "__main__":
    main()
