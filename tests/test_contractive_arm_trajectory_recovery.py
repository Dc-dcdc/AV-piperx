import numpy as np
import pytest
from omegaconf import OmegaConf

from data_collect.recovery_data_generation.trajectory_replay_common import (
    ACTION_DIM,
    VIEW_SLICE,
    build_static_anchor_reference_state,
    resolve_recovery_base_action,
    resolve_recovery_timeline_step,
)
from data_collect.recovery_data_generation.arm_recovery_trajectories import (
    ARM_DIM,
    ArmRecoveryBranchError,
    GRIPPER_INDICES,
    LEFT_ARM_SLICE,
    RIGHT_ARM_SLICE,
    _arm_recovery_action,
    _arm_velocity_percentile,
    _build_arm_recovery_candidates,
    _local_arm_feasible_offset_bounds,
    _sample_arm_recovery_offset,
    _select_active_arm,
    _validate_recovery_unperturbed_roles,
)
from data_collect.recovery_data_generation.view_recovery_trajectories import ModelRiskAnchor


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


def test_static_hold_drift_is_only_validated_at_recovery_end():
    cfg = OmegaConf.create(
        {
            "validation": {
                "branch_max_other_arm_joint_abs_error": 0.002,
                "branch_max_gripper_abs_error": 0.020,
                "branch_max_view_joint_abs_error": 0.0001,
                "static_hold_max_other_arm_joint_drift_rad": 0.010,
                "static_hold_max_gripper_drift": 0.030,
                "static_hold_max_view_joint_drift_rad": 0.006,
            }
        }
    )
    transient_errors = {
        "other_arm": 0.020,
        "gripper": 0.040,
        "view": 0.010,
    }

    _validate_recovery_unperturbed_roles(
        transient_errors,
        cfg,
        trajectory_alignment_mode="static_anchor_wait",
        final=False,
    )

    with pytest.raises(ArmRecoveryBranchError, match="Arm 恢复结束.*other_arm"):
        _validate_recovery_unperturbed_roles(
            transient_errors,
            cfg,
            trajectory_alignment_mode="static_anchor_wait",
            final=True,
        )


def test_moving_expert_still_validates_every_recovery_step():
    cfg = OmegaConf.create(
        {
            "validation": {
                "branch_max_other_arm_joint_abs_error": 0.002,
                "branch_max_gripper_abs_error": 0.020,
                "branch_max_view_joint_abs_error": 0.0001,
            }
        }
    )
    errors = {"other_arm": 0.003, "gripper": 0.0, "view": 0.0}

    with pytest.raises(ArmRecoveryBranchError, match="Arm 恢复.*other_arm"):
        _validate_recovery_unperturbed_roles(
            errors,
            cfg,
            trajectory_alignment_mode="moving_expert",
            final=False,
        )


@pytest.mark.parametrize(
    ("side", "selected_slice", "other_slice"),
    [
        ("left", LEFT_ARM_SLICE, RIGHT_ARM_SLICE),
        ("right", RIGHT_ARM_SLICE, LEFT_ARM_SLICE),
    ],
)
def test_static_anchor_arm_action_holds_every_unperturbed_channel(
    side, selected_slice, other_slice
):
    source_actions = np.stack(
        [np.full(ACTION_DIM, frame + 100, dtype=np.float64) for frame in range(50)]
    )
    source_states = np.stack(
        [np.full(ACTION_DIM, frame, dtype=np.float64) for frame in range(50)]
    )
    reference, _ = resolve_recovery_timeline_step(
        "static_anchor_wait", recovery_anchor_frame=12, source_frame=12
    )
    static_reference = build_static_anchor_reference_state(
        expert_state=source_states[reference],
        actual_state=np.full(ACTION_DIM, -3.0),
        perturbed_indices=selected_slice,
    )
    base_action = resolve_recovery_base_action(
        "static_anchor_wait",
        expert_action=source_actions[reference],
        expert_state=static_reference,
    )
    recovered = _arm_recovery_action(
        base_action, np.full(ARM_DIM, 0.1), side
    )

    np.testing.assert_allclose(recovered[selected_slice], 12.1)
    np.testing.assert_allclose(recovered[other_slice], -3.0)
    np.testing.assert_allclose(recovered[GRIPPER_INDICES], -3.0)
    np.testing.assert_allclose(recovered[VIEW_SLICE], -3.0)


def test_requested_active_arm_is_honored_but_must_have_local_motion():
    states = np.zeros((50, ACTION_DIM), dtype=np.float64)
    states[:, LEFT_ARM_SLICE] = np.arange(50)[:, None] * 0.02
    states[:, RIGHT_ARM_SLICE] = np.arange(50)[:, None] * 0.005

    selected = _select_active_arm(
        states=states,
        event_frame=10,
        fps=25,
        lookback_steps=0,
        lookahead_steps=10,
        min_rms_velocity_rad_s=0.02,
        requested_side="right",
    )
    assert selected is not None
    assert selected.side == "right"

    rejected = _select_active_arm(
        states=states,
        event_frame=10,
        fps=25,
        lookback_steps=0,
        lookahead_steps=10,
        min_rms_velocity_rad_s=0.2,
        requested_side="right",
    )
    assert rejected is None


def test_arm_model_risk_candidates_honor_manifest_side_and_local_interval():
    states = np.zeros((100, ACTION_DIM), dtype=np.float64)
    states[:, LEFT_ARM_SLICE] = np.arange(100)[:, None] * 0.02
    states[:, RIGHT_ARM_SLICE] = np.arange(100)[:, None] * 0.005
    cfg = OmegaConf.create(
        {
            "seed": 20260802,
            "fps": 25,
            "event_sampling": {
                "mode": "model_risk",
                "exclude_initial_steps": 16,
                "fallback_radius_steps": 2,
                "fallback_to_random": False,
                "min_injection_interval_steps": 5,
                "score_key": "arm_joint_score_smoothed",
                "normalized_regions": [],
            },
            "arm_selection": {
                "lookback_steps": 0,
                "lookahead_steps": 10,
                "min_rms_velocity_rad_s": 0.02,
            },
        }
    )
    anchors = [
        ModelRiskAnchor(
            2, 40, 0.8, "right", 100, 0, "model_risk", "0" * 64
        )
    ]

    candidates, inactive = _build_arm_recovery_candidates(
        states=states,
        source_episode=2,
        setup_steps=10,
        required_tail_steps=20,
        cfg=cfg,
        model_risk_anchors=anchors,
    )

    assert inactive == []
    assert candidates[0].event.frame == 40
    assert {candidate.event.frame for candidate in candidates} == set(range(38, 43))
    assert all(candidate.motion_selection.side == "right" for candidate in candidates)
    assert len({candidate.domain_key for candidate in candidates}) == 1


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
