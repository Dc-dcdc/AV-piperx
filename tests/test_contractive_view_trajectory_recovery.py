import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from data_collect.recovery_data_generation.trajectory_replay_common import (
    ACTION_DIM,
    VIEW_DIM,
    VIEW_SLICE,
    build_static_anchor_reference_state,
    recovery_suffix_start_frame,
    resolve_recovery_base_action,
    resolve_recovery_timeline_step,
    resolve_trajectory_alignment_mode,
)
from data_collect.recovery_data_generation.view_recovery_trajectories import (
    ModelRiskAnchor,
    _adaptive_recovery_steps,
    _build_view_recovery_candidates,
    _neighbor_view_candidates,
    _load_model_risk_manifest,
    _quintic_remaining_fraction,
    _recovery_action,
    _resolve_recovery_branch_targets,
    _sample_recovery_offset,
    _source_marked_success,
    _view_candidate_is_spaced_from_successes,
)


def test_model_risk_targets_follow_manifest_counts_and_allow_zero():
    anchors = {
        1: [
            ModelRiskAnchor(1, 20, 0.8, None, 100, 0, "model_risk", "0" * 64),
            ModelRiskAnchor(1, 50, 0.7, None, 100, 1, "model_risk", "0" * 64),
        ],
        3: [
            ModelRiskAnchor(3, 30, 0.9, None, 100, 2, "model_risk", "0" * 64),
        ],
    }

    targets = _resolve_recovery_branch_targets(
        [1, 2, 3],
        sampling_mode="model_risk",
        configured_branches_per_source=3,
        model_risk_anchors_by_episode=anchors,
    )
    fixed_targets = _resolve_recovery_branch_targets(
        [1, 2, 3],
        sampling_mode="random",
        configured_branches_per_source=3,
        model_risk_anchors_by_episode=anchors,
    )

    assert targets == {1: 2, 2: 0, 3: 1}
    assert fixed_targets == {1: 3, 2: 3, 3: 3}


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


def _candidate_cfg(*, mode="random", weight=0.08, interval=35, regions=None):
    return OmegaConf.create(
        {
            "seed": 20260801,
            "event_sampling": {
                "mode": mode,
                "injection_probability_per_frame": weight,
                "min_injection_interval_steps": interval,
                "exclude_initial_steps": 16,
                "normalized_regions": [] if regions is None else regions,
                "risk_manifest_path": None,
                "score_key": "view_score_smoothed",
                "fallback_radius_steps": 2,
                "fallback_to_random": False,
            },
        }
    )


def test_model_risk_manifest_filters_role_and_sorts_score(tmp_path):
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    manifest = tmp_path / "anchors.jsonl"
    records = [
        {
            "schema_version": 1,
            "source_raw_dir": str(source_dir.resolve()),
            "raw_episode_name": "episode_000003",
            "source_episode": 3,
            "frame_index": 70,
            "selection_role": "view",
            "selection_source": "model_risk",
            "score_key": "view_score_smoothed",
            "score": 0.4,
            "action_sha256": "0" * 64,
        },
        {
            "schema_version": 1,
            "source_raw_dir": str(source_dir.resolve()),
            "raw_episode_name": "episode_000003",
            "source_episode": 3,
            "frame_index": 20,
            "selection_role": "view",
            "selection_source": "model_risk",
            "score_key": "view_score_smoothed",
            "score": 0.9,
            "action_sha256": "0" * 64,
        },
        {
            "schema_version": 1,
            "source_raw_dir": str(source_dir.resolve()),
            "raw_episode_name": "episode_000003",
            "source_episode": 3,
            "frame_index": 50,
            "selection_role": "arm",
            "selection_source": "model_risk",
            "score_key": "arm_joint_score_smoothed",
            "score": 1.2,
            "action_sha256": "0" * 64,
        },
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    loaded = _load_model_risk_manifest(
        manifest_path=manifest,
        input_run_dir=source_dir,
        score_key="view_score_smoothed",
        selection_role="view",
    )

    assert [anchor.frame for anchor in loaded[3]] == [20, 70]
    assert [anchor.score for anchor in loaded[3]] == [0.9, 0.4]


@pytest.mark.parametrize("bad_score", [-0.1, float("nan"), float("inf")])
def test_model_risk_manifest_rejects_invalid_score(tmp_path, bad_score):
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    manifest = tmp_path / "anchors.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "source_raw_dir": str(source_dir.resolve()),
                    "raw_episode_name": "episode_000000",
                    "source_episode": 0,
                    "frame_index": 2,
                    "selection_role": "view",
                    "selection_source": "model_risk",
                    "score_key": "view_score_smoothed",
                    "score": bad_score,
                    "action_sha256": "0" * 64,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="有限非负数"):
        _load_model_risk_manifest(
            manifest_path=manifest,
            input_run_dir=source_dir,
            score_key="view_score_smoothed",
            selection_role="view",
        )


def test_model_risk_manifest_rejects_different_source_run(tmp_path):
    source_dir = tmp_path / "raw"
    other_dir = tmp_path / "other_raw"
    source_dir.mkdir()
    other_dir.mkdir()
    manifest = tmp_path / "anchors.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_raw_dir": str(other_dir.resolve()),
                "source_episode": 0,
                "frame_index": 2,
                "selection_role": "view",
                "selection_source": "model_risk",
                "score_key": "view_score_smoothed",
                "score": 0.4,
                "action_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="来自不同源数据"):
        _load_model_risk_manifest(
            manifest_path=manifest,
            input_run_dir=source_dir,
            score_key="view_score_smoothed",
            selection_role="view",
        )


def test_model_risk_candidates_are_score_ordered_and_fallback_is_local():
    cfg = _candidate_cfg(mode="model_risk", interval=3)
    anchors = [
        ModelRiskAnchor(1, 70, 0.4, None, 100, 0, "model_risk", "0" * 64),
        ModelRiskAnchor(1, 30, 0.9, None, 100, 1, "model_risk", "0" * 64),
        # 在尾部有效范围之外，必须通过与valid range取交集而被排除。
        ModelRiskAnchor(1, 90, 2.0, None, 100, 2, "model_risk", "0" * 64),
    ]

    candidates = _build_view_recovery_candidates(
        num_frames=100,
        source_episode=1,
        setup_steps=10,
        required_tail_steps=20,
        cfg=cfg,
        model_risk_anchors=anchors,
    )

    assert candidates[0].event.frame == 30
    assert candidates[0].event.model_risk_score == 0.9
    assert all(candidate.event.frame <= 80 for candidate in candidates)
    primary = candidates[0]
    neighbors = _neighbor_view_candidates(
        primary=primary,
        all_candidates=candidates,
        unavailable_frames=set(),
        successful_candidates=[],
    )
    assert {candidate.event.frame for candidate in neighbors} == set(range(28, 33))
    assert all(candidate.domain_key == primary.domain_key for candidate in neighbors)


def test_candidate_pool_is_deterministic_complete_and_bounded():
    cfg = _candidate_cfg()
    kwargs = {
        "num_frames": 320,
        "source_episode": 17,
        "setup_steps": 10,
        "required_tail_steps": 70,
        "cfg": cfg,
    }
    candidates_a = _build_view_recovery_candidates(**kwargs)
    candidates_b = _build_view_recovery_candidates(**kwargs)
    candidates_other_source = _build_view_recovery_candidates(
        **{**kwargs, "source_episode": 18}
    )
    frames_a = [candidate.event.frame for candidate in candidates_a]
    frames_b = [candidate.event.frame for candidate in candidates_b]
    other_frames = [
        candidate.event.frame for candidate in candidates_other_source
    ]

    assert frames_a == frames_b
    assert frames_a != other_frames
    assert set(frames_a) == set(range(16, 251))
    assert len(frames_a) == len(set(frames_a))


def test_successful_candidate_spacing_uses_configured_interval():
    candidates = _build_view_recovery_candidates(
        num_frames=160,
        source_episode=1,
        setup_steps=10,
        required_tail_steps=40,
        cfg=_candidate_cfg(interval=20),
    )
    successful = [candidates[0]]
    anchor = successful[0].event.frame

    for candidate in candidates[1:]:
        expected = abs(candidate.event.frame - anchor) >= 20
        assert _view_candidate_is_spaced_from_successes(
            candidate, successful
        ) is expected


def test_neighbor_fallback_stays_in_same_normalized_region_and_is_nearest_first():
    cfg = _candidate_cfg(
        mode="specified_region",
        regions=[
            {
                "start": 0.20,
                "end": 0.35,
                "injection_probability_per_frame": 0.1,
                "min_injection_interval_steps": 5,
            },
            {
                "start": 0.60,
                "end": 0.75,
                "injection_probability_per_frame": 0.2,
                "min_injection_interval_steps": 5,
            },
        ],
    )
    candidates = _build_view_recovery_candidates(
        num_frames=200,
        source_episode=4,
        setup_steps=10,
        required_tail_steps=20,
        cfg=cfg,
    )
    primary = next(candidate for candidate in candidates if candidate.domain_key == "region:0")
    neighbors = _neighbor_view_candidates(
        primary=primary,
        all_candidates=candidates,
        unavailable_frames=set(),
        successful_candidates=[],
    )
    distances = [
        abs(candidate.event.frame - primary.event.frame) for candidate in neighbors
    ]

    assert neighbors[0].event.frame == primary.event.frame
    assert all(candidate.domain_key == "region:0" for candidate in neighbors)
    assert distances == sorted(distances)


def test_candidate_pool_rejects_zero_weight_domain():
    with pytest.raises(ValueError, match="采样权重大于0"):
        _build_view_recovery_candidates(
            num_frames=100,
            source_episode=1,
            setup_steps=10,
            required_tail_steps=20,
            cfg=_candidate_cfg(weight=0.0),
        )


def test_recovery_offset_is_reproducible_and_retry_uses_independent_seed():
    kwargs = {
        "seed": 20260801,
        "source_episode": 11,
        "variant_index": 2,
        "anchor_frame": 83,
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
    np.testing.assert_allclose(
        recovered[: VIEW_SLICE.start], expert[: VIEW_SLICE.start]
    )
    np.testing.assert_allclose(recovered[VIEW_SLICE], expert[VIEW_SLICE] + offset)


def test_moving_expert_timeline_advances_reference_and_suffix():
    reference, next_source = resolve_recovery_timeline_step(
        "moving_expert", recovery_anchor_frame=40, source_frame=47
    )

    assert (reference, next_source) == (47, 48)
    assert recovery_suffix_start_frame(
        "moving_expert", recovery_anchor_frame=40, source_frame=48
    ) == 48
    np.testing.assert_array_equal(
        resolve_recovery_base_action(
            "moving_expert",
            expert_action=np.full(ACTION_DIM, 7.0),
            expert_state=np.full(ACTION_DIM, 3.0),
        ),
        7.0,
    )


def test_static_anchor_timeline_holds_reference_then_resumes_anchor_action():
    source_frame = 40
    references = []
    for _ in range(5):
        reference, source_frame = resolve_recovery_timeline_step(
            "static_anchor_wait",
            recovery_anchor_frame=40,
            source_frame=source_frame,
        )
        references.append(reference)

    assert references == [40] * 5
    assert source_frame == 40
    assert recovery_suffix_start_frame(
        "static_anchor_wait", recovery_anchor_frame=40, source_frame=source_frame
    ) == 40


def test_static_anchor_view_action_holds_all_unperturbed_channels():
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
        perturbed_indices=VIEW_SLICE,
    )
    base_action = resolve_recovery_base_action(
        "static_anchor_wait",
        expert_action=source_actions[reference],
        expert_state=static_reference,
    )
    recovered = _recovery_action(
        base_action, np.full(VIEW_DIM, 0.1, dtype=np.float64)
    )

    np.testing.assert_allclose(recovered[: VIEW_SLICE.start], -3.0)
    np.testing.assert_allclose(recovered[VIEW_SLICE], 12.1)


def test_trajectory_alignment_mode_rejects_unknown_value():
    cfg = OmegaConf.create(
        {"recovery": {"trajectory_alignment_mode": "hold_other_only"}}
    )

    with pytest.raises(ValueError, match="trajectory_alignment_mode"):
        resolve_trajectory_alignment_mode(cfg)


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
