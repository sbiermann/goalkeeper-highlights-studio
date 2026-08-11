from goalkeeper_highlights.detection import extend_and_chain_clip_windows
from goalkeeper_highlights.models import Candidate


def _cfg() -> dict:
    return {
        "max_dynamic_clip_seconds": 45.0,
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "continuation_gap_seconds": 12.0,
        "controlled_release_enabled": True,
        "controlled_release_minimum_possession_seconds": 0.5,
        "recovery_distribution_continuation_enabled": True,
        "recovery_distribution_search_seconds": 12.0,
        "recovery_distribution_safety_tail_seconds": 4.0,
        "recovery_distribution_allow_rejected_candidate": True,
        "recovery_window_tail_fallback_enabled": True,
        "recovery_window_tail_max_extension_seconds": 8.0,
        "recovery_window_tail_safety_seconds": 0.0,
        "recovery_window_tail_require_timeout": True,
        "interaction_validation": {"enabled": False},
    }


def test_recovery_window_tail_timeout_extension_applied():
    c = Candidate(
        candidate_id="raw-0012",
        start=320.32,
        end=340.0,
        trigger_time=324.32,
        action_start=324.32,
        action_end=336.0,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        recovery_window_start=326.0,
        recovery_window_end=345.0,
        possession_duration=0.08,
        departure_speed=0.05,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=5,
        keeper_motion=0.2,
        interaction_score=0.6,
    )
    result = extend_and_chain_clip_windows([c], 1000.0, _cfg())[0]
    assert result.action_end == 336.0
    assert result.end >= 345.0
    assert result.clip_end_reason == "recovery_window_tail"
    assert result.score_breakdown["recovery_tail_applied"] == 1.0


def test_recovery_window_tail_blocked_by_controlled_release():
    c = Candidate(
        candidate_id="raw-release",
        start=100.0,
        end=110.0,
        trigger_time=105.0,
        action_start=105.0,
        action_end=110.0,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        recovery_window_start=104.0,
        recovery_window_end=120.0,
        possession_duration=1.0,
        departure_speed=0.5,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=5,
        keeper_motion=0.2,
    )
    result = extend_and_chain_clip_windows([c], 1000.0, _cfg())[0]
    assert result.clip_end_reason == "controlled_release"
    assert result.score_breakdown["recovery_tail_applied"] == 0.0
    assert result.score_breakdown["recovery_tail_blocked_by_release"] == 1.0


def test_recovery_window_tail_clamped_by_max_extension():
    c = Candidate(
        candidate_id="raw-long-tail",
        start=10.0,
        end=20.0,
        trigger_time=12.0,
        action_start=12.0,
        action_end=16.0,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        recovery_window_start=11.0,
        recovery_window_end=30.0,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=5,
        keeper_motion=0.2,
    )
    result = extend_and_chain_clip_windows([c], 1000.0, _cfg())[0]
    assert result.score_breakdown["recovery_tail_clamped"] == 1.0
    assert result.end <= 28.0


def test_recovery_window_tail_blocked_for_other_keeper():
    c1 = Candidate(
        candidate_id="raw-main",
        start=50.0,
        end=60.0,
        trigger_time=52.0,
        action_start=52.0,
        action_end=56.0,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        recovery_window_start=52.0,
        recovery_window_end=66.0,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=5,
        keeper_motion=0.2,
    )
    c2 = Candidate(
        candidate_id="other-attack",
        start=60.2,
        end=63.0,
        trigger_time=60.2,
        action_start=60.2,
        action_end=62.0,
        accepted=True,
        category="distribution",
        keeper_label="Keeper #2",
        min_normalized_distance=0.1,
        keeper_track_id=2,
        contact_frames=4,
        keeper_motion=0.2,
    )
    result = extend_and_chain_clip_windows([c1, c2], 1000.0, _cfg())[0]
    assert result.score_breakdown["recovery_tail_blocked_by_restart"] == 1.0
    assert result.score_breakdown["recovery_tail_applied"] == 0.0