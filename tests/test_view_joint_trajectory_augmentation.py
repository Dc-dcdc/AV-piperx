import json
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from data_collect.augment_view_joint_trajectories import (
    ACTION_DIM,
    EPISODE_NAMING,
    MODEL_BODY_INITIAL_KEYS,
    VIEW_DIM,
    VIEW_SLICE,
    SourceEpisode,
    _build_variants,
    _feasible_offset_bounds,
    _migrate_legacy_episode_names,
    _output_episode_name,
    _parse_output_episode_name,
    _replay_error_groups,
    _replay_validation_limits,
    _replay_validation_violation,
    _restore_initial_model_body_state,
    _sample_offset,
    _validate_source_arrays,
)


def _source_arrays() -> dict[str, np.ndarray]:
    states = np.zeros((4, ACTION_DIM), dtype=np.float64)
    actions = np.zeros_like(states)
    states[:, VIEW_SLICE] = np.asarray(
        [
            [-0.4, -0.2, 0.0, 0.1, 0.2, 0.3],
            [-0.3, -0.1, 0.1, 0.2, 0.3, 0.4],
            [-0.2, 0.0, 0.2, 0.3, 0.4, 0.5],
            [-0.1, 0.1, 0.3, 0.4, 0.5, 0.6],
        ]
    )
    actions[:, VIEW_SLICE] = states[:, VIEW_SLICE] + 0.05
    initial_ctrl = np.zeros(ACTION_DIM, dtype=np.float64)
    initial_ctrl[VIEW_SLICE] = states[0, VIEW_SLICE]
    return {
        "observation_state": states,
        "joint_action": actions,
        "initial_ctrl": initial_ctrl,
    }


def _complete_source_arrays() -> dict[str, np.ndarray]:
    arrays = _source_arrays()
    arrays.update(
        {
            "initial_time": np.asarray(0.0, dtype=np.float64),
            "initial_qpos": np.zeros(3, dtype=np.float64),
            "initial_qvel": np.zeros(3, dtype=np.float64),
            "initial_act": np.zeros(0, dtype=np.float64),
            "initial_mocap_pos": np.zeros((0, 3), dtype=np.float64),
            "initial_mocap_quat": np.zeros((0, 4), dtype=np.float64),
        }
    )
    return arrays


def test_static_model_body_state_is_required_and_restored():
    with tempfile.TemporaryDirectory() as temporary_root:
        arrays_path = Path(temporary_root) / "arrays.npz"
        arrays = _complete_source_arrays()
        np.savez_compressed(arrays_path, **arrays)

        try:
            _validate_source_arrays(
                arrays_path,
                required_model_body_names=("cylinder_container",),
            )
        except KeyError as exc:
            assert "cylinder_container" in str(exc)
            assert MODEL_BODY_INITIAL_KEYS[0] in str(exc)
        else:
            raise AssertionError(
                "缺少model body初态的InsertCylinder源数据必须被拒绝。"
            )

        saved_body_pos = np.asarray(
            [[0.0, 0.0, 0.0], [-0.045, 0.137, 0.0]],
            dtype=np.float64,
        )
        saved_body_quat = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        arrays["initial_model_body_pos"] = saved_body_pos
        arrays["initial_model_body_quat"] = saved_body_quat
        np.savez_compressed(arrays_path, **arrays)
        loaded = _validate_source_arrays(
            arrays_path,
            required_model_body_names=("cylinder_container",),
        )

        class FakeModel:
            body_pos = np.full_like(saved_body_pos, 9.0)
            body_quat = np.full_like(saved_body_quat, 9.0)

        class FakePhysics:
            model = FakeModel()

        class FakeEnvironment:
            _physics = FakePhysics()
            replay_model_body_names = ("cylinder_container",)

        environment = FakeEnvironment()
        _restore_initial_model_body_state(environment, loaded)
        np.testing.assert_array_equal(
            environment._physics.model.body_pos,
            saved_body_pos,
        )
        np.testing.assert_array_equal(
            environment._physics.model.body_quat,
            saved_body_quat,
        )


def test_feasible_bounds_keep_entire_state_and_action_trajectory_in_range():
    arrays = _source_arrays()
    joint_ranges = np.tile(np.asarray([[-1.0, 1.0]]), (VIEW_DIM, 1))
    max_abs = np.full(VIEW_DIM, 0.5)

    lower, upper = _feasible_offset_bounds(
        arrays,
        joint_ranges,
        max_abs,
        margin=0.1,
    )

    values = np.concatenate(
        (
            arrays["observation_state"][:, VIEW_SLICE],
            arrays["joint_action"][:, VIEW_SLICE],
            arrays["initial_ctrl"][None, VIEW_SLICE],
        ),
        axis=0,
    )
    assert np.all(values + lower >= -0.9 - 1e-12)
    assert np.all(values + upper <= 0.9 + 1e-12)
    assert np.all(lower >= -max_abs)
    assert np.all(upper <= max_abs)


def test_offset_sampling_is_deterministic_and_within_all_bounds():
    kwargs = {
        "seed": 20260730,
        "source_episode": 7,
        "variant_index": 2,
        "std": np.full(VIEW_DIM, 0.04),
        "max_abs": np.full(VIEW_DIM, 0.1),
        "feasible_lower": np.full(VIEW_DIM, -0.08),
        "feasible_upper": np.full(VIEW_DIM, 0.07),
        "min_normalized_l2": 0.5,
        "max_attempts": 1000,
    }

    raw_a, actual_a = _sample_offset(**kwargs)
    raw_b, actual_b = _sample_offset(**kwargs)

    np.testing.assert_array_equal(raw_a, raw_b)
    np.testing.assert_array_equal(actual_a, actual_b)
    assert np.all(np.abs(raw_a) <= kwargs["max_abs"])
    assert np.all(actual_a >= kwargs["feasible_lower"])
    assert np.all(actual_a <= kwargs["feasible_upper"])


def test_replay_validation_retries_with_group_specific_limits():
    cfg = OmegaConf.create(
        {
            "replay_validation": {
                "max_state_abs_error": 1.0e-4,
                "retry_max_arm_joint_abs_error": 2.0e-3,
                "retry_max_gripper_abs_error": 5.0e-3,
                "retry_max_view_joint_abs_error": 1.0e-5,
            }
        }
    )
    error = np.zeros(ACTION_DIM, dtype=np.float64)
    error[4] = 7.5e-4
    error[6] = 3.0e-3
    error[15] = 5.0e-7
    groups = _replay_error_groups(error)

    strict_violation = _replay_validation_violation(
        groups,
        _replay_validation_limits(cfg, "strict"),
    )
    fallback_violation = _replay_validation_violation(
        groups,
        _replay_validation_limits(cfg, "fallback"),
    )

    assert strict_violation == ("state", 3.0e-3, 1.0e-4)
    assert fallback_violation is None

    error[15] = 2.0e-5
    view_groups = _replay_error_groups(error)
    assert _replay_validation_violation(
        view_groups,
        _replay_validation_limits(cfg, "fallback"),
    ) == ("view_joint", 2.0e-5, 1.0e-5)


def test_variant_output_indices_are_contiguous_and_original_is_first():
    cfg = OmegaConf.create(
        {
            "include_original": True,
            "variants_per_episode": 2,
            "seed": 20260730,
            "view_joint_noise": {
                "distribution": "truncated_gaussian",
                "std_rad": [0.02] * VIEW_DIM,
                "max_abs_rad": [0.08] * VIEW_DIM,
                "joint_limit_margin_rad": 0.01,
                "min_normalized_l2": 0.5,
                "max_sampling_attempts": 1000,
            },
        }
    )
    source = SourceEpisode(
        source_index=1,
        episode_number=42,
        directory=Path("episode_000042"),
        info={},
    )

    variants = _build_variants(
        source=source,
        source_position=1,
        arrays=_source_arrays(),
        joint_ranges=np.tile(np.asarray([[-1.0, 1.0]]), (VIEW_DIM, 1)),
        cfg=cfg,
    )

    assert [variant.output_index for variant in variants] == [3, 4, 5]
    assert [variant.variant_index for variant in variants] == [-1, 0, 1]
    assert [variant.is_augmented for variant in variants] == [False, True, True]
    np.testing.assert_array_equal(variants[0].actual_offset, np.zeros(VIEW_DIM))


def test_output_episode_name_preserves_source_and_augmented_variant():
    assert _output_episode_name(3, -1) == "episode_000003"
    assert _output_episode_name(3, 0) == "episode_000003_aug_00"
    assert _output_episode_name(128, 4) == "episode_000128_aug_04"
    assert _parse_output_episode_name("episode_000003") == (3, -1)
    assert _parse_output_episode_name("episode_000003_aug_10") == (3, 10)


def test_legacy_contiguous_names_are_migrated_without_collision():
    cameras = ("zed_cam_left", "zed_cam_right")
    with tempfile.TemporaryDirectory() as temporary_root:
        output_run_dir = Path(temporary_root)
        episodes_dir = output_run_dir / "episodes"
        variants = (-1, 0, 1, 2, 3, 4)
        for output_index, variant_index in enumerate(variants):
            directory = episodes_dir / f"episode_{output_index:06d}"
            videos_dir = directory / "videos"
            videos_dir.mkdir(parents=True)
            (directory / "arrays.npz").write_bytes(b"arrays")
            for camera in cameras:
                (videos_dir / f"{camera}.mp4").write_bytes(b"video")
            (directory / "info.json").write_text(
                json.dumps(
                    {
                        "episode": output_index,
                        "source_episode": 3,
                        "variant_index": variant_index,
                        "is_augmented": variant_index >= 0,
                    }
                ),
                encoding="utf-8",
            )

        migrated = _migrate_legacy_episode_names(output_run_dir, cameras)

        assert migrated == 6
        expected_names = {
            "episode_000003",
            "episode_000003_aug_00",
            "episode_000003_aug_01",
            "episode_000003_aug_02",
            "episode_000003_aug_03",
            "episode_000003_aug_04",
        }
        assert {path.name for path in episodes_dir.iterdir()} == expected_names
        for directory in episodes_dir.iterdir():
            info = json.loads((directory / "info.json").read_text())
            assert info["episode_name"] == directory.name
            assert info["episode_naming"] == EPISODE_NAMING
            assert info["path"] == f"episodes/{directory.name}"
            assert info["output_index"] == info["episode"]

        assert _migrate_legacy_episode_names(output_run_dir, cameras) == 0


def test_hf_converter_sorts_source_and_variant_semantically():
    from hugging_face.convert_data_to_hf import list_episode_dirs

    with tempfile.TemporaryDirectory() as temporary_root:
        raw_dir = Path(temporary_root)
        episodes_dir = raw_dir / "episodes"
        names = (
            "episode_000004",
            "episode_000003_aug_100",
            "episode_000003_aug_02",
            "episode_000003",
            "episode_000003_aug_10",
        )
        for name in names:
            directory = episodes_dir / name
            directory.mkdir(parents=True)
            (directory / "arrays.npz").write_bytes(b"arrays")

        assert [
            path.name for path in list_episode_dirs(raw_dir)
        ] == [
            "episode_000003",
            "episode_000003_aug_02",
            "episode_000003_aug_10",
            "episode_000003_aug_100",
            "episode_000004",
        ]
