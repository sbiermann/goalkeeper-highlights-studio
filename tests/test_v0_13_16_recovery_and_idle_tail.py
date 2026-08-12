from goalkeeper_highlights.detection import (
    extend_and_chain_clip_windows,
    is_valid_recovery_continuation,
)
from goalkeeper_highlights.models import Candidate


def _cfg() -> dict:
    return {
        "max_dynamic_clip_seconds": 45.0,
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_post_roll_seconds": {"catch_or_control": 11.0},
        "continuation_gap_seconds": 12.0,
        "phase_merge_gap_seconds": 30.0,
        "phase_merge_duration_tolerance": 0.08,
        "phase_merge_min_pre_roll_seconds": 2.0,
        "phase_merge_min_post_roll_seconds": 2.0,
        "final_keeper_contact_tail_seconds": 5.0,
        "controlled_release_enabled": True,
        "controlled_release_minimum_possession_seconds": 0.5,
        "controlled_release_minimum_departure_speed": 0.35,
        "controlled_release_safety_tail_seconds": 4.0,
        "recovery_distribution_continuation_enabled": True,
        "recovery_distribution_search_seconds": 12.0,
        "recovery_distribution_safety_tail_seconds": 4.0,
        "recovery_distribution_allow_rejected_candidate": True,
        "recovery_window_tail_fallback_enabled": True,
        "recovery_window_tail_max_extension_seconds": 8.0,
        "recovery_window_tail_safety_seconds": 0.0,
        "recovery_window_tail_require_timeout": True,
        "recovery_continuation_require_ball_dynamics": True,
        "recovery_continuation_min_interaction_score": 0.30,
        "recovery_continuation_min_ball_confidence": 0.25,
        "recovery_continuation_min_possession_seconds": 0.45,
        "recovery_continuation_min_dynamic_signal": 0.12,
        "recovery_continuation_min_contact_frames": 2,
        "catch_control_dynamic_post_roll_enabled": True,
        "catch_control_idle_tail_seconds": 3.0,
        "catch_control_max_post_roll_seconds": 11.0,
        "interaction_validation": {"enabled": False},
    }


def test_weak_recovery_with_only_keeper_motion_not_absorbed():
    accepted = Candidate(
        candidate_id="raw-main",
        start=561.68,
        end=570.96,
        trigger_time=565.68,
        action_start=565.68,
        action_end=566.96,
        accepted=True,
        category="catch_or_control",
        keeper_label="Keeper #1",
        keeper_track_id=1,
        min_normalized_distance=0.1,
        contact_frames=4,
    )
    weak_recovery = Candidate(
        candidate_id="diag-weak",
        start=592.0,
        end=594.0,
        trigger_time=592.0,
        action_start=592.0,
        action_end=594.0,
        accepted=False,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        keeper_label="Keeper #1",
        keeper_track_id=1,
        min_normalized_distance=0.1,
        contact_frames=1,
        possession_duration=0.0,
        approach_speed=0.0,
        departure_speed=0.0,
        direction_change=0.0,
        keeper_motion=0.55,
    )
    results = extend_and_chain_clip_windows([accepted, weak_recovery], 1000.0, _cfg())
    assert len(results) == 2
    assert weak_recovery.continuation_absorbed is False


def test_recovery_with_real_ball_dynamics_can_be_absorbed():
    accepted = Candidate(100.0, 110.0, 104.0, 0.1, 1, candidate_id="base", accepted=True, category="catch_or_control", action_start=104.0, action_end=106.0, keeper_label="Keeper #1", contact_frames=4)
    dyn = Candidate(
        112.0,
        118.0,
        113.0,
        0.1,
        1,
        candidate_id="dyn",
        accepted=False,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=113.0,
        action_end=116.0,
        keeper_label="Keeper #1",
        contact_frames=3,
        ball_confidence=0.4,
        approach_speed=0.25,
        departure_speed=0.28,
        direction_change=0.2,
    )
    results = extend_and_chain_clip_windows([accepted, dyn], 1000.0, _cfg())
    assert len(results) == 1
    assert dyn.continuation_absorbed is True


def test_recovery_with_ball_control_can_be_absorbed():
    accepted = Candidate(10.0, 20.0, 12.0, 0.1, 1, candidate_id="a", accepted=True, category="catch_or_control", action_start=12.0, action_end=15.0, keeper_label="Keeper #1", contact_frames=4)
    controlled = Candidate(
        22.0,
        28.0,
        23.0,
        0.1,
        1,
        candidate_id="b",
        accepted=False,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=23.0,
        action_end=25.0,
        keeper_label="Keeper #1",
        contact_frames=1,
        possession_duration=0.7,
        ball_confidence=0.35,
    )
    results = extend_and_chain_clip_windows([accepted, controlled], 1000.0, _cfg())
    assert len(results) == 1
    assert controlled.continuation_absorbed is True


def test_catch_or_control_without_follow_action_uses_idle_tail():
    c = Candidate(0.0, 0.0, 10.0, 0.1, 1, candidate_id="catch", accepted=True, category="catch_or_control", action_start=10.0, action_end=10.0, keeper_label="Keeper #1", contact_frames=4)
    result = extend_and_chain_clip_windows([c], 100.0, _cfg())[0]
    assert result.end == 13.0
    assert result.clip_end_reason == "dynamic_idle_tail"


def test_catch_or_control_with_distribution_not_cut_too_early():
    c = Candidate(0.0, 0.0, 10.0, 0.1, 1, candidate_id="catch", accepted=True, category="catch_or_control", action_start=10.0, action_end=10.0, keeper_label="Keeper #1", contact_frames=4)
    dist = Candidate(16.0, 19.0, 16.0, 0.1, 1, candidate_id="dist", accepted=True, category="distribution", action_start=16.0, action_end=17.0, keeper_label="Keeper #1", contact_frames=3, departure_speed=0.6)
    result = extend_and_chain_clip_windows([c, dist], 100.0, _cfg())[0]
    assert result.end >= 20.0


def test_controlled_release_priority_is_preserved():
    c = Candidate(0.0, 0.0, 10.0, 0.1, 1, candidate_id="release", accepted=True, category="catch_or_control", action_start=10.0, action_end=12.0, keeper_label="Keeper #1", contact_frames=4, possession_duration=1.0, departure_speed=0.6)
    result = extend_and_chain_clip_windows([c], 100.0, _cfg())[0]
    assert result.clip_end_reason == "controlled_release"


def test_recovery_window_tail_priority_is_preserved():
    c = Candidate(
        candidate_id="tail",
        start=100.0,
        end=114.0,
        trigger_time=104.0,
        action_start=104.0,
        action_end=110.0,
        accepted=True,
        category="catch_or_control",
        recovery_candidate=True,
        recovery_window_start=104.0,
        recovery_window_end=122.0,
        keeper_label="Keeper #1",
        keeper_track_id=1,
        min_normalized_distance=0.1,
        contact_frames=4,
    )
    result = extend_and_chain_clip_windows([c], 1000.0, _cfg())[0]
    assert result.clip_end_reason == "recovery_window_tail"


def test_only_keeper_motion_after_action_is_not_relevant_activity():
    c = Candidate(0.0, 0.0, 10.0, 0.1, 1, candidate_id="catch", accepted=True, category="catch_or_control", action_start=10.0, action_end=10.0, keeper_label="Keeper #1", contact_frames=4)
    weak = Candidate(12.0, 14.0, 12.0, 0.1, 1, candidate_id="weak", accepted=False, category="recovery_uncovered_activity", recovery_candidate=True, action_start=12.0, action_end=13.0, keeper_label="Keeper #1", contact_frames=1, keeper_motion=0.7)
    result = extend_and_chain_clip_windows([c, weak], 100.0, _cfg())[0]
    assert result.end == 13.0


def test_ball_interaction_inside_max_post_roll_keeps_post_roll_open():
    c = Candidate(0.0, 0.0, 10.0, 0.1, 1, candidate_id="catch", accepted=True, category="catch_or_control", action_start=10.0, action_end=10.0, keeper_label="Keeper #1", contact_frames=4)
    follow = Candidate(18.0, 20.0, 18.0, 0.1, 1, candidate_id="follow", accepted=True, category="interaction", action_start=18.0, action_end=19.0, keeper_label="Keeper #1", contact_frames=3, ball_confidence=0.5)
    result = extend_and_chain_clip_windows([c, follow], 100.0, _cfg())[0]
    assert result.end >= 21.0


def test_ball_interaction_after_max_post_roll_does_not_extend_indefinitely():
    c = Candidate(0.0, 0.0, 10.0, 0.1, 1, candidate_id="catch", accepted=True, category="catch_or_control", action_start=10.0, action_end=10.0, keeper_label="Keeper #1", contact_frames=4)
    late = Candidate(25.0, 27.0, 25.0, 0.1, 2, candidate_id="late", accepted=True, category="interaction", action_start=25.0, action_end=26.0, keeper_label="Keeper #2", contact_frames=3, ball_confidence=0.5)
    result = extend_and_chain_clip_windows([c, late], 100.0, _cfg())[0]
    assert result.end == 13.0


def test_recovery_continuation_helper_marks_weak_case_invalid():
    weak = Candidate(
        start=0.0,
        end=1.0,
        trigger_time=0.5,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        accepted=False,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        contact_frames=1,
        possession_duration=0.0,
        approach_speed=0.0,
        departure_speed=0.0,
        direction_change=0.0,
        keeper_motion=0.8,
    )
    res = is_valid_recovery_continuation(weak, _cfg())
    assert res["recovery_continuation_valid"] == 0.0
    assert res["recovery_continuation_blocked_weak_ball_signal"] == 1.0