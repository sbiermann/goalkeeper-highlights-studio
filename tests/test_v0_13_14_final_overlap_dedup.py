from goalkeeper_highlights.detection import merge_overlapping_final_clips
from goalkeeper_highlights.models import Candidate


def _cfg(**overrides):
    cfg = {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "continuation_gap_seconds": 0.0,
        "phase_merge_gap_seconds": 0.0,
        "phase_merge_duration_tolerance": 0.08,
        "phase_merge_min_pre_roll_seconds": 2.0,
        "phase_merge_min_post_roll_seconds": 2.0,
        "max_dynamic_clip_seconds": 45.0,
        "minimum_clip_seconds": 1.0,
        "final_keeper_contact_tail_seconds": 0.0,
        "final_overlap_merge_enabled": True,
        "final_overlap_merge_min_ratio": 0.60,
        "final_overlap_merge_max_gap_seconds": 1.0,
        "final_overlap_merge_require_same_keeper": True,
        "interaction_validation": {"enabled": False},
        "controlled_release_enabled": False,
        "recovery_distribution_continuation_enabled": False,
        "recovery_window_tail_fallback_enabled": False,
    }
    cfg.update(overrides)
    return cfg


def _candidate(candidate_id: str, start: float, end: float, *, accepted: bool = True, keeper: str = "Keeper #1", category: str = "save_or_deflection", clip_end_reason: str = "timeout") -> Candidate:
    return Candidate(
        start=start,
        end=end,
        trigger_time=(start + end) / 2.0,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        accepted=accepted,
        category=category,
        keeper_label=keeper,
        candidate_id=candidate_id,
        action_start=start,
        action_end=end,
        clip_end_reason=clip_end_reason,
        score_breakdown={},
    )


def test_strong_overlap_same_keeper_is_deduplicated():
    c1 = _candidate("raw-0008", 218.24, 263.24, clip_end_reason="controlled_release")
    c2 = _candidate("raw-0009", 236.72, 273.00, clip_end_reason="recovery_window_tail")

    result = merge_overlapping_final_clips([c1, c2], 2000.0, _cfg(max_dynamic_clip_seconds=80.0))

    assert len(result) == 1
    merged = result[0]
    assert merged.start == 218.24
    assert merged.end == 273.00
    assert "raw-0009" in merged.merged_from
    assert merged.score_breakdown.get("final_overlap_merge_applied") == 1.0


def test_real_overlap_case_uses_context_trimming_and_merges():
    c1 = _candidate("raw-0008", 218.24, 263.24, clip_end_reason="controlled_release")
    c1.action_start = 228.24
    c1.action_end = 263.24
    c2 = _candidate("raw-0009", 236.72, 273.00, clip_end_reason="recovery_window_tail")
    c2.action_start = 240.72
    c2.action_end = 264.00

    result = merge_overlapping_final_clips(
        [c1, c2],
        3000.0,
        _cfg(
            max_dynamic_clip_seconds=45.0,
            phase_merge_duration_tolerance=0.08,
            phase_merge_min_pre_roll_seconds=2.0,
            phase_merge_min_post_roll_seconds=2.0,
        ),
    )

    assert len(result) == 1
    merged = result[0]
    assert merged.action_start == 228.24
    assert merged.action_end == 264.00
    assert merged.start <= merged.action_start
    assert merged.end >= merged.action_end
    assert (merged.end - merged.start) <= (45.0 * 1.08)
    assert merged.score_breakdown.get("final_overlap_context_trimmed") == 1.0
    assert "raw-0009" in merged.merged_from


def test_union_within_limit_merges_without_context_trimming():
    c1 = _candidate("raw-0001", 100.0, 130.0)
    c1.action_start = 105.0
    c1.action_end = 128.0
    c2 = _candidate("raw-0002", 112.0, 141.0)
    c2.action_start = 116.0
    c2.action_end = 139.0

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(max_dynamic_clip_seconds=45.0))

    assert len(result) == 1
    assert result[0].score_breakdown.get("final_overlap_context_trimmed") == 0.0


def test_union_too_long_but_action_with_min_context_fits_merges():
    c1 = _candidate("raw-0001", 100.0, 140.0)
    c1.action_start = 108.0
    c1.action_end = 130.0
    c2 = _candidate("raw-0002", 115.0, 170.0)
    c2.action_start = 118.0
    c2.action_end = 145.0

    result = merge_overlapping_final_clips(
        [c1, c2],
        500.0,
        _cfg(max_dynamic_clip_seconds=45.0, phase_merge_duration_tolerance=0.0),
    )

    assert len(result) == 1
    merged = result[0]
    assert merged.action_start == 108.0
    assert merged.action_end == 145.0
    assert (merged.end - merged.start) <= 45.0
    assert merged.score_breakdown.get("final_overlap_context_trimmed") == 1.0


def test_no_merge_if_action_itself_exceeds_limit():
    c1 = _candidate("raw-0001", 100.0, 150.0)
    c1.action_start = 102.0
    c1.action_end = 160.0
    c2 = _candidate("raw-0002", 130.0, 185.0)
    c2.action_start = 120.0
    c2.action_end = 170.0

    result = merge_overlapping_final_clips(
        [c1, c2],
        500.0,
        _cfg(max_dynamic_clip_seconds=45.0, phase_merge_duration_tolerance=0.0),
    )

    assert len(result) == 2


def test_no_merge_if_minimum_context_cannot_fit():
    c1 = _candidate("raw-0001", 100.0, 165.0)
    c1.action_start = 106.0
    c1.action_end = 145.0
    c2 = _candidate("raw-0002", 130.0, 172.0)
    c2.action_start = 118.0
    c2.action_end = 162.5

    result = merge_overlapping_final_clips(
        [c1, c2],
        500.0,
        _cfg(
            max_dynamic_clip_seconds=45.0,
            phase_merge_duration_tolerance=0.0,
            phase_merge_min_pre_roll_seconds=2.0,
            phase_merge_min_post_roll_seconds=2.0,
        ),
    )

    assert len(result) == 2


def test_low_overlap_below_ratio_is_not_merged():
    c1 = _candidate("raw-0001", 10.0, 30.0)
    c2 = _candidate("raw-0002", 27.0, 47.0)

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(final_overlap_merge_min_ratio=0.8))

    assert len(result) == 2
    assert result[1].score_breakdown.get("final_overlap_merge_applied") == 0.0


def test_identical_clips_are_merged_to_single_export():
    c1 = _candidate("raw-0001", 100.0, 120.0)
    c2 = _candidate("raw-0002", 100.0, 120.0)

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(max_dynamic_clip_seconds=60.0))

    assert len(result) == 1
    assert "raw-0002" in result[0].merged_from


def test_different_keeper_not_merged():
    c1 = _candidate("raw-0001", 100.0, 130.0, keeper="Keeper #1")
    c2 = _candidate("raw-0002", 110.0, 138.0, keeper="Keeper #2")

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(max_dynamic_clip_seconds=60.0))

    assert len(result) == 2


def test_independent_restart_not_merged():
    c1 = _candidate("raw-0001", 100.0, 130.0, category="distribution")
    c2 = _candidate("raw-0002", 110.0, 138.0, category="distribution")

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(max_dynamic_clip_seconds=60.0))

    assert len(result) == 2
    assert result[1].score_breakdown.get("final_overlap_restart_detected") == 1.0


def test_small_gap_can_merge_when_configured():
    c1 = _candidate("raw-0001", 100.0, 130.0)
    c2 = _candidate("raw-0002", 130.7, 150.0)

    result = merge_overlapping_final_clips(
        [c1, c2],
        500.0,
        _cfg(final_overlap_merge_max_gap_seconds=1.0, max_dynamic_clip_seconds=60.0),
    )

    assert len(result) == 1
    assert result[0].end == 150.0


def test_large_gap_not_merged():
    c1 = _candidate("raw-0001", 100.0, 130.0)
    c2 = _candidate("raw-0002", 132.5, 150.0)

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(final_overlap_merge_max_gap_seconds=1.0))

    assert len(result) == 2


def test_rejected_plus_accepted_not_merged():
    c1 = _candidate("raw-0001", 100.0, 130.0, accepted=True)
    c2 = _candidate("raw-0002", 110.0, 138.0, accepted=False)

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(max_dynamic_clip_seconds=60.0))

    assert len(result) == 2


def test_both_rejected_not_merged():
    c1 = _candidate("raw-0001", 100.0, 130.0, accepted=False)
    c2 = _candidate("raw-0002", 110.0, 138.0, accepted=False)

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(max_dynamic_clip_seconds=60.0))

    assert len(result) == 2


def test_merged_from_and_parent_ids_are_preserved_uniquely():
    c1 = _candidate("raw-0001", 100.0, 140.0)
    c1.merged_from = ["legacy-a", "legacy-b"]
    c1.parent_candidate_ids = ["p1", "p2"]
    c2 = _candidate("raw-0002", 110.0, 150.0)
    c2.merged_from = ["legacy-b", "legacy-c"]
    c2.parent_candidate_ids = ["p2", "p3"]

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(max_dynamic_clip_seconds=60.0))

    assert len(result) == 1
    assert result[0].merged_from == ["legacy-a", "legacy-b", "raw-0002", "legacy-c"]
    assert result[0].parent_candidate_ids == ["p1", "p2", "raw-0002", "p3"]


def test_recovery_window_end_uses_latest_boundary():
    c1 = _candidate("raw-0001", 100.0, 140.0)
    c1.recovery_window_end = 141.0
    c2 = _candidate("raw-0002", 110.0, 150.0)
    c2.recovery_window_end = 153.0

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(max_dynamic_clip_seconds=60.0))

    assert len(result) == 1
    assert result[0].recovery_window_end == 153.0


def test_diagnostics_contain_original_and_effective_context_values():
    c1 = _candidate("raw-0001", 218.24, 263.24)
    c1.action_start = 228.24
    c1.action_end = 263.24
    c2 = _candidate("raw-0002", 236.72, 273.00)
    c2.action_start = 240.72
    c2.action_end = 264.00

    result = merge_overlapping_final_clips(
        [c1, c2],
        3000.0,
        _cfg(max_dynamic_clip_seconds=45.0, phase_merge_duration_tolerance=0.08),
    )

    assert len(result) == 1
    breakdown = result[0].score_breakdown
    for key in [
        "final_overlap_original_union_duration",
        "final_overlap_action_duration",
        "final_overlap_original_pre_roll",
        "final_overlap_original_post_roll",
        "final_overlap_effective_pre_roll",
        "final_overlap_effective_post_roll",
        "final_overlap_trimmed_duration",
        "final_overlap_duration_limit",
    ]:
        assert key in breakdown


def test_clip_end_reason_priority_prefers_controlled_release():
    c1 = _candidate("raw-0001", 100.0, 140.0, clip_end_reason="recovery_window_tail")
    c2 = _candidate("raw-0002", 110.0, 150.0, clip_end_reason="controlled_release")

    result = merge_overlapping_final_clips([c1, c2], 500.0, _cfg(max_dynamic_clip_seconds=60.0))

    assert len(result) == 1
    assert result[0].clip_end_reason == "controlled_release"


def test_no_forced_merge_when_union_exceeds_duration_limit():
    c1 = _candidate("raw-0001", 100.0, 130.0)
    c2 = _candidate("raw-0002", 128.0, 165.0)

    result = merge_overlapping_final_clips(
        [c1, c2],
        500.0,
        _cfg(max_dynamic_clip_seconds=45.0, phase_merge_duration_tolerance=0.0),
    )

    assert len(result) == 2
