from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from data_collect.backfill_insert_container_state import (
    RingCandidate,
    _best_ring_component,
    _inclusive_grid,
    _is_safe_scope_expansion,
    _mask_similarity,
    _select_temporally_separated,
)


def _config():
    return OmegaConf.load(
        Path("configs/data_collect/insert_container_state_backfill.yaml")
    )


def test_yellow_ring_is_detected_as_single_container_candidate():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.ellipse(
        rgb,
        center=(80, 60),
        axes=(20, 12),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=(255, 255, 0),
        thickness=4,
    )

    result = _best_ring_component(
        rgb,
        _config(),
        frame_index=17,
        return_mask=True,
    )

    assert result is not None
    candidate, mask = result
    assert candidate.frame_index == 17
    assert candidate.area_px > 0
    assert mask is not None
    assert mask.dtype == np.bool_
    assert mask.sum() == candidate.area_px


def test_temporal_selection_prefers_score_but_keeps_frame_separation():
    candidates = [
        RingCandidate(index, score, (1, 1, 8, 6), 20, 1.3, 0.4)
        for index, score in [(10, 10.0), (11, 9.0), (20, 8.0), (30, 7.0)]
    ]

    selected = _select_temporally_separated(
        candidates,
        count=3,
        min_separation_frames=5,
    )

    assert [candidate.frame_index for candidate in selected] == [10, 20, 30]


def test_search_grid_includes_upper_bound_and_identical_masks_score_one():
    grid = _inclusive_grid(0.05, 0.0565, 0.002)
    np.testing.assert_allclose(grid, [0.05, 0.052, 0.054, 0.056, 0.0565])

    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[7:12, 8:14] = True
    assert np.isclose(_mask_similarity(mask, mask, _config()), 1.0)


def test_resume_allows_only_episode_scope_expansion():
    existing = {
        "source_episode_indices": [0, 1],
        "search": {"fine_step_m": 0.0001},
    }
    expanded = {
        "source_episode_indices": [0, 1, 2],
        "search": {"fine_step_m": 0.0001},
    }
    changed_algorithm = {
        "source_episode_indices": [0, 1, 2],
        "search": {"fine_step_m": 0.00005},
    }

    assert _is_safe_scope_expansion(existing, expanded)
    assert not _is_safe_scope_expansion(existing, existing)
    assert not _is_safe_scope_expansion(existing, changed_algorithm)
