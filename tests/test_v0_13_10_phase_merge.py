import pytest
from goalkeeper_highlights.models import Candidate
from goalkeeper_highlights.detection import extend_and_chain_clip_windows

def test_phase_merge_issue_repro():
    # Clip A
    # clip window = 218.24–240.60
    # action window = 228.24–229.60
    c1 = Candidate(
        candidate_id="c1",
        start=218.24, end=240.60,
        trigger_time=228.24,
        action_start=228.24, action_end=229.60,
        accepted=True,
        category="catch_or_control",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        min_normalized_distance=0.1, keeper_track_id=1
    )
    
    # Clip B
    # clip window = 236.72–268.00
    # action window = 240.72–264.00
    c2 = Candidate(
        candidate_id="c2",
        start=236.72, end=268.00,
        trigger_time=240.72,
        action_start=240.72, action_end=264.00,
        accepted=True,
        category="recovery_uncovered_activity",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        recovery_candidate=True,
        min_normalized_distance=0.1, keeper_track_id=1
    )
    
    # Duration: 500s (enough)
    duration = 500.0
    clips_cfg = {
        "interaction_validation": {"enabled": False},
        "max_dynamic_clip_seconds": 45.0,
        "phase_merge_duration_tolerance": 0.08, # 45 * 1.08 = 48.6
        "phase_merge_gap_seconds": 30.0,
        "category_pre_roll_seconds": {"catch_or_control": 10.0, "recovery_uncovered_activity": 4.0},
        "category_post_roll_seconds": {"catch_or_control": 11.0, "recovery_uncovered_activity": 4.0},
        "phase_merge_min_pre_roll_seconds": 2.0,
        "phase_merge_min_post_roll_seconds": 2.0,
    }
    
    # Der aktuelle Stand sollte fehlschlagen (kein Merge, da trimmed_duration > limit_with_tolerance)
    # Aktueller Algorithmus:
    # min_combined_start = 228.24
    # p_before = 10.0
    # c_after = 4.0
    # safe_start = 228.24 - 10.0 = 218.24
    # safe_end = 264.00 + 4.0 = 268.00
    # trimmed_duration = 268.00 - 218.24 = 49.76
    # 49.76 > 48.60 -> Merge schlägt fehl im alten Code
    
    results = extend_and_chain_clip_windows([c1, c2], duration, clips_cfg)
    
    # Wir erwarten jetzt, dass es gemergt wird
    assert len(results) == 1
    merged = results[0]
    assert "c2" in merged.merged_from
    assert merged.start >= 0
    assert merged.end <= duration
    assert merged.end - merged.start <= 48.60
    # Action window MUST be preserved: 228.24 to 264.00
    assert merged.start <= 228.24
    assert merged.end >= 264.00
    
    # Diagnostik prüfen
    sb = merged.score_breakdown
    assert sb["phase_merge_original_pre_roll"] == 10.0
    assert sb["phase_merge_effective_pre_roll"] < 10.0
    assert sb["phase_merge_effective_pre_roll"] >= 2.0
    assert sb["phase_merge_trimmed_duration"] <= 48.60
    assert sb["phase_merge_action_duration"] == 264.00 - 228.24

def test_continuation_absorption_trimming():
    # Testet, ob auch abgelehnte Recovery-Kandidaten (Absorption) vom Trimming profitieren
    c1 = Candidate(
        candidate_id="c1", start=100, end=120, trigger_time=110,
        action_start=110, action_end=115, accepted=True, category="catch",
        keeper_label="Keeper #1", clip_end_reason="timeout",
        min_normalized_distance=0.1, keeper_track_id=1
    )
    c2 = Candidate(
        candidate_id="c2", start=140, end=170, trigger_time=150,
        action_start=150, action_end=165, accepted=False, category="recovery_uncovered_activity",
        keeper_label="Keeper #1", recovery_candidate=True,
        min_normalized_distance=0.1, keeper_track_id=1
    )
    # Action span: 110 bis 165 = 55s. Max: 45s.
    # Selbst mit min_pre=2, min_post=2 -> 55 + 2 + 2 = 59s.
    # 59 > 48.6 -> Sollte NICHT gemergt werden.
    
    clips_cfg = {
        "interaction_validation": {"enabled": False},
        "max_dynamic_clip_seconds": 45.0,
        "phase_merge_duration_tolerance": 0.08,
        "phase_merge_gap_seconds": 30.0,
        "phase_merge_min_pre_roll_seconds": 2.0,
        "phase_merge_min_post_roll_seconds": 2.0,
    }
    results = extend_and_chain_clip_windows([c1, c2], 500.0, clips_cfg)
    assert len(results) == 2
    assert results[1].candidate_id == "c2" # Nicht absorbiert
    
def test_different_keepers_no_merge():
    # Gap zwischen den Clipfenstern vergrößern, um Auto-Chaining zu vermeiden
    # c1: action 228.24-229.60. before 5s -> start 223.24. after 4s -> end 233.60
    # c2: action 250.72-264.00. before 5s -> start 245.72. after 4s -> end 268.00
    # Gap: 245.72 - 233.60 = 12.12s.
    # Continuation gap is 12s. So they won't chain in pass A.
    c1 = Candidate(
        candidate_id="c1",
        start=218.24, end=240.60,
        trigger_time=228.24,
        action_start=228.24, action_end=229.60,
        accepted=True,
        category="catch_or_control",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        min_normalized_distance=0.1, keeper_track_id=1
    )
    c2 = Candidate(
        candidate_id="c2",
        start=236.72, end=268.00,
        trigger_time=250.72,
        action_start=250.72, action_end=264.00,
        accepted=True,
        category="recovery_uncovered_activity",
        keeper_label="Keeper #2",
        clip_end_reason="timeout",
        min_normalized_distance=0.1, keeper_track_id=2
    )
    results = extend_and_chain_clip_windows([c1, c2], 500.0, {"max_dynamic_clip_seconds": 45.0, "interaction_validation": {"enabled": False}, "continuation_gap_seconds": 12.0})
    assert len(results) == 2


def test_accepted_weak_phase_is_absorbed_without_boundary_extension():
    primary = Candidate(
        candidate_id="raw-a",
        start=88.0,
        end=109.0,
        trigger_time=95.0,
        action_start=95.0,
        action_end=102.0,
        accepted=True,
        category="catch_or_control",
        keeper_label="Keeper #1",
        clip_end_reason="dynamic_idle_tail",
        interaction_score=0.62,
        contact_frames=12,
        possession_duration=1.4,
        ball_confidence=0.82,
        min_normalized_distance=0.1,
        keeper_track_id=1,
    )
    weak_followup = Candidate(
        candidate_id="raw-b",
        start=103.0,
        end=120.0,
        trigger_time=106.0,
        action_start=106.0,
        action_end=110.0,
        accepted=True,
        category="interaction",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        interaction_score=0.11,
        contact_frames=2,
        possession_duration=0.10,
        ball_confidence=0.21,
        min_normalized_distance=0.1,
        keeper_track_id=1,
    )
    clips_cfg = {
        "interaction_validation": {"enabled": False},
        "max_dynamic_clip_seconds": 45.0,
        "phase_merge_gap_seconds": 30.0,
        "continuation_gap_seconds": 1.0,
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"catch_or_control": 10.0, "interaction": 5.0},
        "category_post_roll_seconds": {"catch_or_control": 11.0, "interaction": 8.0},
        "phase_core_weak_absorption_enabled": True,
        "phase_core_weak_absorption_delta": 0.35,
        "phase_core_weak_absorption_max_action_seconds": 12.0,
    }

    results = extend_and_chain_clip_windows([primary, weak_followup], 500.0, clips_cfg)

    assert len(results) == 1
    merged = results[0]
    assert merged.candidate_id == "raw-a"
    assert "raw-b" in merged.merged_from
    assert merged.start == pytest.approx(85.0)
    assert merged.end == pytest.approx(113.0)
    assert merged.phase_merge_reason == "same_keeper_weak_phase_absorbed"


def test_isolated_catch_or_control_gets_core_trimmed_tail():
    candidate = Candidate(
        candidate_id="raw-c",
        start=273.76,
        end=294.84,
        trigger_time=283.76,
        action_start=283.76,
        action_end=287.84,
        accepted=True,
        category="catch_or_control",
        keeper_label="Keeper #1",
        clip_end_reason="dynamic_idle_tail",
        interaction_score=0.58,
        contact_frames=12,
        possession_duration=1.2,
        ball_confidence=0.7,
        min_normalized_distance=0.1,
        keeper_track_id=1,
    )
    clips_cfg = {
        "interaction_validation": {"enabled": False},
        "max_dynamic_clip_seconds": 45.0,
        "continuation_gap_seconds": 12.0,
        "category_pre_roll_seconds": {"catch_or_control": 10.0},
        "category_post_roll_seconds": {"catch_or_control": 11.0},
        "catch_control_dynamic_post_roll_enabled": True,
        "catch_control_isolated_dynamic_idle_tail_seconds": 1.0,
    }

    results = extend_and_chain_clip_windows([candidate], 500.0, clips_cfg)

    assert len(results) == 1
    trimmed = results[0]
    assert trimmed.start == pytest.approx(275.76)
    assert trimmed.end == pytest.approx(290.84)
    assert trimmed.clip_boundary_reason == "isolated_action_core_rebalanced"


def test_compact_rebalance_does_not_touch_shorter_isolated_catch():
    candidate = Candidate(
        candidate_id="raw-stable",
        start=92.84,
        end=106.08,
        trigger_time=102.84,
        action_start=102.84,
        action_end=105.08,
        accepted=True,
        category="catch_or_control",
        keeper_label="Keeper #1",
        clip_end_reason="dynamic_idle_tail",
        interaction_score=0.52,
        contact_frames=8,
        possession_duration=0.8,
        ball_confidence=0.62,
        min_normalized_distance=0.1,
        keeper_track_id=1,
    )
    clips_cfg = {
        "interaction_validation": {"enabled": False},
        "max_dynamic_clip_seconds": 45.0,
        "continuation_gap_seconds": 12.0,
        "category_pre_roll_seconds": {"catch_or_control": 10.0},
        "category_post_roll_seconds": {"catch_or_control": 1.0},
        "catch_control_dynamic_post_roll_enabled": False,
        "catch_control_isolated_dynamic_idle_tail_seconds": 1.0,
    }

    results = extend_and_chain_clip_windows([candidate], 500.0, clips_cfg)

    assert len(results) == 1
    stable = results[0]
    assert stable.start == pytest.approx(92.84)
    assert stable.end == pytest.approx(106.08)
    assert stable.clip_boundary_reason == "observed_action_window"

def test_action_too_long_no_merge():
    # Action span is 200 to 250 = 50s. Max is 45s.
    c1 = Candidate(
        candidate_id="c1",
        start=190.0, end=210.0,
        trigger_time=200.0,
        action_start=200.0, action_end=205.0,
        accepted=True,
        category="catch_or_control",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        min_normalized_distance=0.1, keeper_track_id=1
    )
    c2 = Candidate(
        candidate_id="c2",
        start=240.0, end=260.0,
        trigger_time=250.0,
        action_start=245.0, action_end=250.0,
        accepted=True,
        category="recovery_uncovered_activity",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        min_normalized_distance=0.1, keeper_track_id=1
    )
    # Action span 200.0 - 250.0 = 50.0.
    # 50.0 > 45.0 * 1.08 = 48.6
    results = extend_and_chain_clip_windows([c1, c2], 500.0, {"max_dynamic_clip_seconds": 45.0, "phase_merge_duration_tolerance": 0.08, "interaction_validation": {"enabled": False}})
    assert len(results) == 2

def test_chaining_pass_different_keepers():
    # Testet, ob der erste Chaining-Pass Kandidaten unterschiedlicher Keeper trennt,
    # selbst wenn die Lücke klein genug wäre.
    c1 = Candidate(
        candidate_id="c1",
        start=100, end=110, trigger_time=105,
        action_start=105, action_end=108,
        accepted=True, category="catch_or_control",
        keeper_label="Keeper #1",
        min_normalized_distance=0.1, keeper_track_id=1
    )
    c2 = Candidate(
        candidate_id="c2",
        start=112, end=120, trigger_time=115,
        action_start=112, action_end=115,
        accepted=True, category="distribution",
        keeper_label="Keeper #2",
        min_normalized_distance=0.1, keeper_track_id=2
    )
    # Action gap: 112 - 108 = 4s.
    # Continuation gap is 12s (default).
    
    duration = 500.0
    clips_cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "continuation_gap_seconds": 12.0,
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], duration, clips_cfg)
    
    # Sollten getrennt bleiben wegen unterschiedlicher Keeper
    assert len(results) == 2
    assert results[0].keeper_label == "Keeper #1"
    assert results[1].keeper_label == "Keeper #2"

def test_classic_rescue_does_not_use_extended_phase_merge_window_like_raw_0068_0069():
    rescued_contact = Candidate(
        candidate_id="rescued-contact",
        start=2141.92,
        end=2150.56,
        trigger_time=2145.92,
        action_start=2145.92,
        action_end=2146.56,
        accepted=True,
        category="interaction",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        min_normalized_distance=0.43925,
        keeper_track_id=2,
        contact_frames=2,
        possession_duration=0.08,
        ball_confidence=0.37297,
        keeper_motion=0.06313,
        keeper_lateral_motion=0.19202,
        score_breakdown={"classic_action_rescue": 1.0},
    )
    later_control = Candidate(
        candidate_id="later-control",
        start=2160.16,
        end=2171.44,
        trigger_time=2164.16,
        action_start=2164.16,
        action_end=2167.44,
        accepted=True,
        category="catch_or_control",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        min_normalized_distance=0.0,
        keeper_track_id=2,
        contact_frames=29,
        possession_duration=2.72,
        ball_confidence=0.683,
        keeper_motion=0.061,
    )

    results = extend_and_chain_clip_windows(
        [rescued_contact, later_control],
        3000.0,
        {
            "interaction_validation": {"enabled": False},
            "continuation_gap_seconds": 12.0,
            "phase_merge_gap_seconds": 30.0,
            "max_dynamic_clip_seconds": 45.0,
            "category_pre_roll_seconds": {"interaction": 4.0, "catch_or_control": 10.0},
            "category_post_roll_seconds": {"interaction": 4.0, "catch_or_control": 11.0},
        },
    )

    accepted = [candidate for candidate in results if candidate.accepted]
    assert len(accepted) == 2
    assert [candidate.candidate_id for candidate in accepted] == ["rescued-contact", "later-control"]
    assert accepted[0].start == pytest.approx(2141.92)
    assert accepted[0].end == pytest.approx(2150.56)
    assert accepted[1].score_breakdown["phase_merge_classic_rescue_gap_guard"] == 1.0
    assert "later-control" not in accepted[0].merged_from


def test_classic_rescue_can_still_chain_inside_normal_continuation_window():
    rescued_contact = Candidate(
        candidate_id="rescued-contact",
        start=100.0,
        end=108.0,
        trigger_time=104.0,
        action_start=104.0,
        action_end=105.0,
        accepted=True,
        category="interaction",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        min_normalized_distance=0.3,
        keeper_track_id=1,
        contact_frames=2,
        score_breakdown={"classic_action_rescue": 1.0},
    )
    follow_up = Candidate(
        candidate_id="follow-up",
        start=110.0,
        end=118.0,
        trigger_time=112.0,
        action_start=112.0,
        action_end=114.0,
        accepted=True,
        category="catch_or_control",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=8,
        possession_duration=1.0,
    )

    results = extend_and_chain_clip_windows(
        [rescued_contact, follow_up],
        500.0,
        {
            "interaction_validation": {"enabled": False},
            "continuation_gap_seconds": 12.0,
            "phase_merge_gap_seconds": 30.0,
            "max_dynamic_clip_seconds": 45.0,
            "category_pre_roll_seconds": {"interaction": 4.0, "catch_or_control": 10.0},
            "category_post_roll_seconds": {"interaction": 4.0, "catch_or_control": 11.0},
        },
    )

    assert len([candidate for candidate in results if candidate.accepted]) == 1
    assert results[0].clip_boundary_reason == "chained_keeper_phase"


def test_long_inactive_gap_splits_previous_action_and_routes_setup_forward_like_raw_0061_0063_0064():
    previous = Candidate(
        candidate_id="previous-control",
        start=1985.20, end=2005.00, trigger_time=1995.20,
        action_start=1995.20, action_end=1998.00,
        accepted=True, category="catch_or_control", keeper_label="Keeper #1",
        clip_end_reason="dynamic_idle_tail", min_normalized_distance=0.0, keeper_track_id=1,
        contact_frames=15, possession_duration=0.80, ball_confidence=0.68,
        interaction_score=0.78, keeper_motion=0.81,
        merged_from=["same-action-child"],
    )
    setup = Candidate(
        candidate_id="leading-setup",
        start=2029.12, end=2038.08, trigger_time=2033.12,
        action_start=2033.12, action_end=2034.08,
        accepted=True, category="distribution", keeper_label="Keeper #1",
        clip_end_reason="timeout", min_normalized_distance=0.2699, keeper_track_id=1,
        contact_frames=6, possession_duration=0.10, ball_confidence=0.714,
        interaction_score=0.20, keeper_motion=0.047,
    )
    followup = Candidate(
        candidate_id="strong-followup",
        start=2038.60, end=2074.40, trigger_time=2055.92,
        action_start=2047.60, action_end=2070.40,
        accepted=True, category="diving_save", keeper_label="Keeper #1",
        clip_end_reason="controlled_release", min_normalized_distance=0.0, keeper_track_id=1,
        contact_frames=95, possession_duration=4.48, ball_confidence=0.701,
        interaction_score=1.0, keeper_motion=0.713, approach_speed=0.367,
        departure_speed=6.95, direction_change=0.997,
    )

    results = extend_and_chain_clip_windows(
        [previous, setup, followup],
        3000.0,
        {
            "interaction_validation": {"enabled": False},
            "continuation_gap_seconds": 12.0,
            "phase_merge_gap_seconds": 30.0,
            "max_dynamic_clip_seconds": 45.0,
            "category_pre_roll_seconds": {
                "catch_or_control": 10.0,
                "distribution": 4.0,
                "diving_save": 9.0,
            },
            "category_post_roll_seconds": {
                "catch_or_control": 11.0,
                # Real config gives distributions a much longer generic tail.
                # V10 must still recognise this as compact leading context.
                "distribution": 12.0,
                "diving_save": 11.0,
            },
            "leading_context_absorb_max_gap_seconds": 1.0,
            "phase_split_previous_pre_roll_seconds": 2.0,
            "phase_split_previous_post_roll_seconds": 3.0,
        },
    )

    accepted = [candidate for candidate in results if candidate.accepted]
    assert len(accepted) == 2
    assert [candidate.candidate_id for candidate in accepted] == ["previous-control", "strong-followup"]

    first, second = accepted
    assert first.start == pytest.approx(1993.20)
    assert first.end == pytest.approx(2001.00)
    assert first.clip_boundary_reason == "inactive_gap_phase_split"
    assert first.score_breakdown["phase_split_inactive_gap_applied"] == 1.0

    assert second.start == pytest.approx(2029.12)
    assert second.end == pytest.approx(2074.40)
    assert "leading-setup" in second.merged_from
    assert second.score_breakdown["leading_context_absorbed"] == 1.0
    assert second.score_breakdown["phase_merge_absorbed_leading_boundary_guard"] == 1.0
    assert "strong-followup" not in first.merged_from


def test_v13_long_internal_gap_splits_catch_and_distribution_into_two_highlights_like_clip_42():
    first = Candidate(candidate_id="catch-first", start=3485.12, end=3494.48, trigger_time=3489.12, action_start=3489.12, action_end=3490.48, accepted=True, category="catch_or_control", keeper_label="Keeper #1", clip_end_reason="timeout", min_normalized_distance=.1, keeper_track_id=1, contact_frames=4, possession_duration=.8)
    second = Candidate(candidate_id="distribution-second", start=3501.28, end=3510.24, trigger_time=3505.28, action_start=3505.28, action_end=3506.24, accepted=True, category="distribution", keeper_label="Keeper #1", clip_end_reason="timeout", min_normalized_distance=.1, keeper_track_id=1, contact_frames=6, possession_duration=.4)
    result = extend_and_chain_clip_windows([first, second], 5000.0, {"interaction_validation":{"enabled":False}, "seconds_before":4.0, "seconds_after":4.0, "category_pre_roll_seconds":{"catch_or_control":10.0,"distribution":4.0}, "category_post_roll_seconds":{"catch_or_control":11.0,"distribution":12.0}, "continuation_gap_seconds":12.0, "phase_merge_gap_seconds":30.0, "max_dynamic_clip_seconds":45.0, "catch_control_distribution_split_min_action_gap_seconds":10.0})
    accepted = [c for c in result if c.accepted]
    assert len(accepted) == 2
    assert accepted[0].start == pytest.approx(3479.12)
    assert accepted[0].end == pytest.approx(3492.48)
    assert accepted[1].start == pytest.approx(3501.28)
    assert accepted[1].end == pytest.approx(3513.24)
    assert accepted[0].clip_boundary_reason == "internal_phase_gap_split"
    assert accepted[1].clip_boundary_reason == "internal_phase_gap_split"
