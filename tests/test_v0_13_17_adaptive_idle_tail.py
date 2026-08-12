from goalkeeper_highlights.detection import apply_dynamic_catch_control_idle_tail, extend_and_chain_clip_windows
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
        "catch_control_idle_tail_low_seconds": 3.0,
        "catch_control_idle_tail_medium_seconds": 6.0,
        "catch_control_idle_tail_high_seconds": 7.0,
        "catch_control_medium_min_contact_frames": 4,
        "catch_control_medium_min_possession_seconds": 0.25,
        "catch_control_high_min_contact_frames": 10,
        "catch_control_high_min_possession_seconds": 1.0,
        "catch_control_high_min_interaction_score": 0.45,
        "catch_control_max_post_roll_seconds": 11.0,
        "interaction_validation": {"enabled": False},
    }


def test_low_interaction_remains_short_idle_tail():
    c = Candidate(
        start=10.0,
        end=21.0,
        trigger_time=10.0,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        candidate_id="low",
        accepted=True,
        category="catch_or_control",
        action_start=10.0,
        action_end=10.0,
        keeper_label="Keeper #1",
        contact_frames=1,
        possession_duration=0.0,
        interaction_score=0.05,
        keeper_motion=0.9,
    )
    diag = apply_dynamic_catch_control_idle_tail(c, [c], _cfg(), 1000.0)
    assert diag["catch_control_idle_level"] == 1.0
    assert diag["catch_control_selected_idle_tail"] == 3.0
    assert c.end == 13.0


def test_medium_realistic_clip9_style_case_is_not_low():
    c = Candidate(
        start=1162.56,
        end=1178.68,
        trigger_time=1166.56,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        candidate_id="synthetic-medium",
        accepted=True,
        category="catch_or_control",
        action_start=1166.56,
        action_end=1167.68,
        keeper_label="Keeper #1",
        contact_frames=8,
        possession_duration=0.56,
        interaction_score=0.276,
    )
    diag = apply_dynamic_catch_control_idle_tail(c, [c], _cfg(), 2000.0)
    assert diag["catch_control_idle_level"] >= 2.0
    assert diag["catch_control_selected_idle_tail"] > 3.0
    assert c.end == 1173.68
    assert c.end < 1178.68


def test_high_interaction_uses_high_tail_and_respects_max_post_roll():
    c = Candidate(
        start=20.0,
        end=40.0,
        trigger_time=24.0,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        candidate_id="high",
        accepted=True,
        category="catch_or_control",
        action_start=24.0,
        action_end=25.0,
        keeper_label="Keeper #1",
        contact_frames=14,
        possession_duration=1.3,
        interaction_score=0.62,
    )
    diag = apply_dynamic_catch_control_idle_tail(c, [c], _cfg(), 2000.0)
    assert diag["catch_control_idle_level"] == 3.0
    assert diag["catch_control_selected_idle_tail"] == 7.0
    assert c.end <= 36.0


def test_low_interaction_score_with_clear_control_is_medium():
    c = Candidate(
        start=40.0,
        end=56.0,
        trigger_time=44.0,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        candidate_id="score-low-control-clear",
        accepted=True,
        category="catch_or_control",
        action_start=44.0,
        action_end=45.0,
        keeper_label="Keeper #1",
        contact_frames=6,
        possession_duration=0.5,
        interaction_score=0.20,
    )
    diag = apply_dynamic_catch_control_idle_tail(c, [c], _cfg(), 2000.0)
    assert diag["catch_control_idle_level"] == 2.0
    assert c.end == 51.0


def test_high_keeper_motion_without_ball_evidence_stays_low():
    c = Candidate(
        start=60.0,
        end=76.0,
        trigger_time=64.0,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        candidate_id="motion-only",
        accepted=True,
        category="catch_or_control",
        action_start=64.0,
        action_end=65.0,
        keeper_label="Keeper #1",
        contact_frames=1,
        possession_duration=0.0,
        interaction_score=0.08,
        keeper_motion=1.0,
    )
    diag = apply_dynamic_catch_control_idle_tail(c, [c], _cfg(), 2000.0)
    assert diag["catch_control_idle_level"] == 1.0
    assert c.end == 68.0


def test_controlled_release_priority_kept_over_idle_tail():
    c = Candidate(0.0, 0.0, 10.0, 0.1, 1, candidate_id="release", accepted=True, category="catch_or_control", action_start=10.0, action_end=12.0, keeper_label="Keeper #1", contact_frames=4, possession_duration=1.0, departure_speed=0.6)
    result = extend_and_chain_clip_windows([c], 100.0, _cfg())[0]
    assert result.clip_end_reason == "controlled_release"


def test_distribution_continuation_priority_kept_over_idle_tail():
    c = Candidate(
        start=8.0,
        end=18.0,
        trigger_time=10.0,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        candidate_id="base",
        accepted=True,
        category="catch_or_control",
        action_start=10.0,
        action_end=10.5,
        keeper_label="Keeper #1",
        contact_frames=4,
        clip_end_reason="recovery_distribution_continuation",
    )
    diag = apply_dynamic_catch_control_idle_tail(c, [c], _cfg(), 100.0)
    assert c.clip_end_reason == "recovery_distribution_continuation"
    assert c.end == 18.0
    assert diag["catch_control_dynamic_post_roll_applied"] == 0.0


def test_recovery_window_tail_priority_kept_over_idle_tail():
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


def test_original_clip_end_is_hard_upper_bound():
    c = Candidate(
        start=30.0,
        end=52.0,
        trigger_time=34.0,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        candidate_id="bounded",
        accepted=True,
        category="catch_or_control",
        action_start=34.0,
        action_end=35.0,
        keeper_label="Keeper #1",
        contact_frames=6,
        possession_duration=0.8,
        interaction_score=0.35,
    )
    diag = apply_dynamic_catch_control_idle_tail(c, [c], _cfg(), 2000.0)
    assert c.end <= diag["catch_control_original_clip_end"]


def test_diagnostics_include_numeric_level_and_matches():
    c = Candidate(
        start=80.0,
        end=96.0,
        trigger_time=84.0,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        candidate_id="diag",
        accepted=True,
        category="catch_or_control",
        action_start=84.0,
        action_end=85.0,
        keeper_label="Keeper #1",
        contact_frames=10,
        possession_duration=1.0,
        interaction_score=0.5,
    )
    diag = apply_dynamic_catch_control_idle_tail(c, [c], _cfg(), 2000.0)
    assert diag["catch_control_idle_level"] in {1.0, 2.0, 3.0}
    assert "catch_control_medium_contact_match" in diag
    assert "catch_control_medium_possession_match" in diag
    assert "catch_control_high_contact_match" in diag
    assert "catch_control_high_possession_match" in diag
