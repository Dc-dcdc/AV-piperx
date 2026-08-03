import numpy as np
import pytest

from data_collect.augment_view_joint_trajectories import (
    ACTION_DIM,
    VIEW_DIM,
    VIEW_SLICE,
)
from data_collect.generate_contractive_view_recovery_trajectories import (
    _adaptive_recovery_steps,
    _quintic_remaining_fraction,
    _recovery_action,
    _sample_injection_frames,
    _sample_recovery_offset,
    _source_marked_success,
)


def test_quintic_remaining_fraction_has_exact_endpoints_and_is_monotone():
    progress = np.linspace(0.0, 1.0, 1001)
    remaining = _quintic_remaining_fraction(progress)

    assert remaining[0] == 1.0
    assert remaining[-1] == 0.0
    assert np.all(np.diff(remaining) <= 1.0e-12)
    assert np.all((remaining >= 0.0) & (remaining <= 1.0))


@pytest.mark.parametrize("progress", [-0.01, 1.01, np.nan, np.inf])
def test_quintic_remaining_fraction_rejects_invalid_progress(progress):
    with pytest.raises(ValueError):
        _quintic_remaining_fraction(progress)


def test_adaptive_recovery_steps_uses_quintic_peak_velocity_and_clamps():
    fps = 25
    velocity_limit = np.ones(VIEW_DIM, dtype=np.float64)

    assert _adaptive_recovery_steps(
        np.zeros(VIEW_DIM), fps, velocity_limit, min_steps=20, max_steps=40
    ) == 20

    # 1.875 * 0.6 / 1.0 s * 25 Hz = 28.125，向上取整为29步。
    moderate_offset = np.zeros(VIEW_DIM, dtype=np.float64)
    moderate_offset[2] = 0.6
    assert _adaptive_recovery_steps(
        moderate_offset, fps, velocity_limit, min_steps=20, max_steps=40
    ) == 29

    assert _adaptive_recovery_steps(
        np.full(VIEW_DIM, 10.0),
        fps,
        velocity_limit,
        min_steps=20,
        max_steps=40,
    ) == 40


def test_injection_frame_sampling_is_deterministic_bounded_and_spaced():
    kwargs = {
        "num_frames": 320,
        "seed": 20260801,
        "source_episode": 17,
        "probability": 0.08,
        "min_interval_steps": 35,
        "min_events": 3,
        "max_events": 5,
        "exclude_initial_steps": 16,
        "required_tail_steps": 70,
    }

    frames_a = _sample_injection_frames(**kwargs)
    frames_b = _sample_injection_frames(**kwargs)
    other_source_frames = _sample_injection_frames(
        **{**kwargs, "source_episode": kwargs["source_episode"] + 1}
    )

    assert frames_a == frames_b
    assert frames_a != other_source_frames
    assert kwargs["min_events"] <= len(frames_a) <= kwargs["max_events"]
    assert frames_a == sorted(frames_a)
    assert all(
        kwargs["exclude_initial_steps"]
        <= frame
        <= kwargs["num_frames"] - kwargs["required_tail_steps"]
        for frame in frames_a
    )
    assert all(
        right - left >= kwargs["min_interval_steps"]
        for left, right in zip(frames_a, frames_a[1:])
    )


def test_injection_frame_sampling_probability_zero_still_meets_minimum():
    frames = _sample_injection_frames(
        num_frames=300,
        seed=7,
        source_episode=4,
        probability=0.0,
        min_interval_steps=40,
        min_events=3,
        max_events=3,
        exclude_initial_steps=20,
        required_tail_steps=60,
    )

    assert len(frames) == 3
    assert all(
        right - left >= 40 for left, right in zip(frames, frames[1:])
    )


def test_injection_fallback_does_not_fail_due_to_unlucky_greedy_middle_point():
    # 0..100中先选到50会让普通贪心无法再满足interval=60；压缩坐标
    # fallback只要数学上存在可行组合，就必须稳定生成两帧。
    for seed in range(200):
        frames = _sample_injection_frames(
            num_frames=101,
            seed=seed,
            source_episode=0,
            probability=0.0,
            min_interval_steps=60,
            min_events=2,
            max_events=2,
            exclude_initial_steps=0,
            required_tail_steps=1,
        )
        assert len(frames) == 2
        assert frames[1] - frames[0] >= 60


def test_injection_frame_sampling_rejects_impossible_event_spacing():
    with pytest.raises(ValueError, match="无法满足最少恢复事件数"):
        _sample_injection_frames(
            num_frames=100,
            seed=1,
            source_episode=1,
            probability=0.0,
            min_interval_steps=80,
            min_events=3,
            max_events=3,
            exclude_initial_steps=10,
            required_tail_steps=20,
        )


def test_recovery_offset_is_reproducible_and_retry_uses_independent_seed():
    kwargs = {
        "seed": 20260801,
        "source_episode": 11,
        "variant_index": 2,
        "std": np.full(VIEW_DIM, 0.04),
        "max_abs": np.full(VIEW_DIM, 0.12),
        "feasible_lower": np.full(VIEW_DIM, -0.1),
        "feasible_upper": np.full(VIEW_DIM, 0.09),
        "min_normalized_l2": 0.1,
        "max_sampling_attempts": 1000,
    }

    first_a = _sample_recovery_offset(attempt=0, **kwargs)
    first_b = _sample_recovery_offset(attempt=0, **kwargs)
    retry = _sample_recovery_offset(attempt=1, **kwargs)

    np.testing.assert_array_equal(first_a, first_b)
    assert not np.array_equal(first_a, retry)
    for offset in (first_a, retry):
        assert np.all(np.abs(offset) <= kwargs["max_abs"])
        assert np.all(offset >= kwargs["feasible_lower"])
        assert np.all(offset <= kwargs["feasible_upper"])
        assert (
            np.linalg.norm(offset / kwargs["max_abs"])
            >= kwargs["min_normalized_l2"]
        )


def test_recovery_action_changes_only_view_and_does_not_mutate_expert():
    expert = np.linspace(-1.0, 1.0, ACTION_DIM, dtype=np.float64)
    expert_before = expert.copy()
    offset = np.linspace(0.01, 0.06, VIEW_DIM, dtype=np.float64)

    recovered = _recovery_action(expert, offset)

    np.testing.assert_array_equal(expert, expert_before)
    np.testing.assert_array_equal(
        recovered[: VIEW_SLICE.start], expert[: VIEW_SLICE.start]
    )
    np.testing.assert_allclose(recovered[VIEW_SLICE], expert[VIEW_SLICE] + offset)


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"success": True}, True),
        ({"success": np.bool_(True)}, True),
        ({"success": False}, False),
        ({"success": 1}, False),
        ({"success": "true"}, False),
        ({"final_info": {"is_success": True}}, True),
        ({"final_info": {"is_success": "true"}}, False),
        ({"success": False, "final_info": {"is_success": True}}, False),
        ({}, False),
    ],
)
def test_source_success_requires_explicit_boolean_marker(info, expected):
    assert _source_marked_success(info) is expected
