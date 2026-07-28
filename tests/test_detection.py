from goalkeeper_highlights.detection import merge_candidates
from goalkeeper_highlights.models import Candidate


def candidate(start: float, end: float, distance: float = 1.0) -> Candidate:
    return Candidate(start, end, (start + end) / 2, distance, 1)


def test_merge_overlapping_candidates() -> None:
    merged = merge_candidates([candidate(1, 5), candidate(4, 8, 0.5)], gap=0, duration=20)
    assert len(merged) == 1
    assert merged[0].start == 1
    assert merged[0].end == 8
    assert merged[0].min_normalized_distance == 0.5


def test_keep_separate_candidates() -> None:
    merged = merge_candidates([candidate(1, 2), candidate(10, 11)], gap=2, duration=20)
    assert len(merged) == 2


def test_merge_preserves_category_specific_acceptance_of_stronger_candidate():
    from goalkeeper_highlights.detection import merge_candidates
    from goalkeeper_highlights.models import Candidate

    weaker = Candidate(
        start=10.0, end=15.0, trigger_time=12.0, min_normalized_distance=0.4,
        keeper_track_id=1, accepted=False, category="interaction", confidence=0.20,
        description="interaction", event_score=0.20, acceptance_threshold=0.38,
        rejection_reason="event_score_below_category_threshold", score_breakdown={"proximity": 0.1}
    )
    stronger = Candidate(
        start=14.0, end=18.0, trigger_time=16.0, min_normalized_distance=0.3,
        keeper_track_id=1, accepted=True, category="distribution", confidence=0.316,
        description="distribution", event_score=0.316, acceptance_threshold=0.22,
        rejection_reason="", score_breakdown={"confirmed_keeper_contact": 0.08}
    )

    result = merge_candidates([weaker, stronger], gap=0.5, duration=60.0)

    assert len(result) == 1
    merged = result[0]
    assert merged.accepted is True
    assert merged.category == "distribution"
    assert merged.acceptance_threshold == 0.22
    assert merged.rejection_reason == ""
    assert merged.score_breakdown == {"confirmed_keeper_contact": 0.08}


def test_dynamic_tail_extends_accepted_scene() -> None:
    from goalkeeper_highlights.detection import extend_and_chain_clip_windows

    item = candidate(100.0, 110.0)
    item.accepted = True
    result = extend_and_chain_clip_windows([item], 200.0, {
        "dynamic_end_enabled": True,
        "activity_tail_seconds": 8.0,
        "continuation_gap_seconds": 15.0,
        "final_keeper_contact_tail_seconds": 8.0,
        "max_dynamic_clip_seconds": 40.0,
    })
    assert result[0].end == 118.0


def test_dynamic_window_chains_follow_up_keeper_event() -> None:
    from goalkeeper_highlights.detection import extend_and_chain_clip_windows

    first = candidate(100.0, 110.0)
    first.accepted = True
    first.category = "distribution"
    second = candidate(120.0, 126.0)
    second.accepted = True
    second.category = "catch_or_control"

    result = extend_and_chain_clip_windows([first, second], 200.0, {
        "dynamic_end_enabled": True,
        "activity_tail_seconds": 8.0,
        "continuation_gap_seconds": 15.0,
        "final_keeper_contact_tail_seconds": 8.0,
        "max_dynamic_clip_seconds": 40.0,
    })
    assert len(result) == 1
    assert result[0].end == 134.0
    assert result[0].score_breakdown["chained_event_count"] == 2


def test_dynamic_window_respects_maximum_duration() -> None:
    from goalkeeper_highlights.detection import extend_and_chain_clip_windows

    first = candidate(100.0, 125.0)
    first.accepted = True
    second = candidate(130.0, 140.0)
    second.accepted = True

    result = extend_and_chain_clip_windows([first, second], 300.0, {
        "dynamic_end_enabled": True,
        "activity_tail_seconds": 8.0,
        "continuation_gap_seconds": 15.0,
        "final_keeper_contact_tail_seconds": 8.0,
        "max_dynamic_clip_seconds": 40.0,
    })
    assert result[0].end <= result[0].start + 40.0


def test_category_specific_windows_make_catch_start_earlier_and_distribution_shorter() -> None:
    from goalkeeper_highlights.detection import extend_and_chain_clip_windows

    catch = Candidate(386.6, 399.0, 393.6, 0.1, 1, accepted=True, category="catch_or_control")
    distribution = Candidate(2251.16, 2267.96, 2258.16, 0.1, 1, accepted=True, category="distribution")
    cfg = {
        "dynamic_end_enabled": True,
        "seconds_before": 7.0,
        "seconds_after": 5.0,
        "category_pre_roll_seconds": {"catch_or_control": 10.0, "distribution": 5.0},
        "category_post_roll_seconds": {"catch_or_control": 4.0, "distribution": 2.0},
        "activity_tail_seconds": 0.0,
        "continuation_gap_seconds": 0.0,
        "final_keeper_contact_tail_seconds": 8.0,
        "max_dynamic_clip_seconds": 40.0,
    }

    result = extend_and_chain_clip_windows([catch, distribution], 3000.0, cfg)

    # Legacy/cached candidates without action_start/action_end preserve their
    # existing diagnostic windows in 0.13.
    assert result[0].start == 386.6
    assert result[0].end == 399.0
    assert result[1].start == 2251.16
    assert result[1].end == 2267.96
