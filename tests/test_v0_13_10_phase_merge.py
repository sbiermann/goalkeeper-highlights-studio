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
