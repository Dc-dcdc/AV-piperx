import numpy as np
import pytest

from data_collect.augment_view_joint_trajectories import ACTION_DIM, VIEW_SLICE
from data_collect.generate_contractive_arm_recovery_trajectories import (
    ARM_DIM,
    GRIPPER_INDICES,
    LEFT_ARM_SLICE,
    RIGHT_ARM_SLICE,
    _arm_recovery_action,
    _arm_velocity_percentile,
    _assign_balanced_arms,
    _local_arm_feasible_offset_bounds,
    _sample_arm_recovery_offset,
)


@pytest.mark.parametrize(
    ("side", "selected_slice", "other_slice"),
    [
        ("left", LEFT_ARM_SLICE, RIGHT_ARM_SLICE),
        ("right", RIGHT_ARM_SLICE, LEFT_ARM_SLICE),
    ],
)
def test_arm_recovery_action_only_changes_selected_six_joints(
    side, selected_slice, other_slice
):
    expert = np.linspace(-1.0, 1.0, ACTION_DIM, dtype=np.float64)
    original = expert.copy()
    offset = np.linspace(0.01, 0.06, ARM_DIM, dtype=np.float64)

    action = _arm_recovery_action(expert, offset, side)

    np.testing.assert_array_equal(expert, original)
    np.testing.assert_allclose(action[selected_slice], expert[selected_slice] + offset)
    np.testing.assert_array_equal(action[other_slice], expert[other_slice])
    np.testing.assert_array_equal(action[GRIPPER_INDICES], expert[GRIPPER_INDICES])
    np.testing.assert_array_equal(action[VIEW_SLICE], expert[VIEW_SLICE])


def test_arm_recovery_action_rejects_unknown_side():
    with pytest.raises(ValueError, match="left.*right"):
        _arm_recovery_action(
            np.zeros(ACTION_DIM, dtype=np.float64),
            np.zeros(ARM_DIM, dtype=np.float64),
            "middle",
        )


def test_balanced_arm_assignment_is_deterministic_and_globally_balanced():
    identities = [(episode, variant) for episode in range(7) for variant in range(5)]

    first = _assign_balanced_arms(identities, seed=20260802)
    repeated = _assign_balanced_arms(list(reversed(identities)), seed=20260802)

    assert first == repeated
    counts = {side: list(first.values()).count(side) for side in ("left", "right")}
    assert abs(counts["left"] - counts["right"]) <= 1
    ordered_sides = [first[key] for key in sorted(first)]
    assert all(left != right for left, right in zip(ordered_sides, ordered_sides[1:]))


def test_arm_velocity_percentile_combines_both_arms_without_episode_boundaries():
    fps = 25
    actions_a = np.zeros((3, ACTION_DIM), dtype=np.float64)
    actions_b = np.zeros((2, ACTION_DIM), dtype=np.float64)
    actions_a[1:, LEFT_ARM_SLICE] = [[0.04] * ARM_DIM, [0.12] * ARM_DIM]
    actions_a[1:, RIGHT_ARM_SLICE] = [[0.08] * ARM_DIM, [0.16] * ARM_DIM]
    actions_b[1, LEFT_ARM_SLICE] = 0.20
    actions_b[1, RIGHT_ARM_SLICE] = 0.24

    velocity, sample_count = _arm_velocity_percentile(
        [actions_a, actions_b], fps=fps, percentile=100.0
    )

    # 最大单步增量为0.24 rad；不能在两个episode边界之间额外做差分。
    np.testing.assert_allclose(velocity, np.full(ARM_DIM, 0.24 * fps))
    assert sample_count == 2 * ((len(actions_a) - 1) + (len(actions_b) - 1))


def test_arm_offset_sampling_is_reproducible_and_arm_specific():
    kwargs = {
        "seed": 20260802,
        "source_episode": 9,
        "variant_index": 1,
        "attempt": 0,
        "std": np.full(ARM_DIM, 0.04),
        "max_abs": np.full(ARM_DIM, 0.12),
        "feasible_lower": np.full(ARM_DIM, -0.10),
        "feasible_upper": np.full(ARM_DIM, 0.09),
        "min_normalized_l2": 0.1,
        "max_sampling_attempts": 1000,
    }

    left_a = _sample_arm_recovery_offset(perturbed_arm="left", **kwargs)
    left_b = _sample_arm_recovery_offset(perturbed_arm="left", **kwargs)
    right = _sample_arm_recovery_offset(perturbed_arm="right", **kwargs)

    np.testing.assert_array_equal(left_a, left_b)
    assert not np.array_equal(left_a, right)
    for offset in (left_a, right):
        assert np.all(np.abs(offset) <= kwargs["max_abs"])
        assert np.all(offset >= kwargs["feasible_lower"])
        assert np.all(offset <= kwargs["feasible_upper"])


@pytest.mark.parametrize("side", ["left", "right"])
def test_local_feasible_bounds_use_only_selected_arm(side):
    states = np.zeros((10, ACTION_DIM), dtype=np.float64)
    actions = np.zeros((10, ACTION_DIM), dtype=np.float64)
    selected_slice = LEFT_ARM_SLICE if side == "left" else RIGHT_ARM_SLICE
    other_slice = RIGHT_ARM_SLICE if side == "left" else LEFT_ARM_SLICE
    states[:, selected_slice] = 0.8
    actions[:, selected_slice] = 0.7
    # 另一只手臂即使远超虚拟限位，也不应影响所选Arm的可行域。
    states[:, other_slice] = 100.0
    actions[:, other_slice] = 100.0
    source_arrays = {"observation_state": states, "joint_action": actions}

    lower, upper = _local_arm_feasible_offset_bounds(
        source_arrays,
        event_frame=2,
        horizon_steps=5,
        control_ranges=np.tile(np.asarray([[-1.0, 1.0]]), (ARM_DIM, 1)),
        max_abs=np.full(ARM_DIM, 0.5),
        margin=0.05,
        perturbed_arm=side,
    )

    np.testing.assert_allclose(lower, np.full(ARM_DIM, -0.5))
    np.testing.assert_allclose(upper, np.full(ARM_DIM, 0.15))
