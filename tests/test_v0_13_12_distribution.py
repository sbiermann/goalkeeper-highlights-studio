import pytest
from goalkeeper_highlights.models import Candidate
from goalkeeper_highlights.detection import extend_and_chain_clip_windows

def test_recovery_distribution_continuation_success():
    # Case 1 & 2: Recovery-Candidate mit extrem kurzer possession_duration (0.08s) 
    # und direkt nachfolgendem Distribution-Candidate desselben Keepers.
    c1 = Candidate(
        candidate_id="raw-0012",
        start=320.32, end=340.00,
        trigger_time=324.32,
        action_start=324.32, action_end=336.00,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        possession_duration=0.08, # Unreliable
        departure_speed=0.05, # Low departure so v0.13.11 logic skips it
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=5,
        keeper_motion=0.2,
        interaction_score=0.6
    )
    
    c2 = Candidate(
        candidate_id="dist-0013",
        start=338.0, end=345.0,
        trigger_time=340.0,
        action_start=340.0, action_end=342.0,
        accepted=True,
        category="distribution",
        departure_speed=0.7,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=5,
        keeper_motion=0.2
    )
    
    cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "controlled_release_enabled": True, # Enabled but should skip c1 because of low possession
        "controlled_release_minimum_possession_seconds": 0.5,
        "recovery_distribution_continuation_enabled": True,
        "recovery_distribution_search_seconds": 12.0,
        "recovery_distribution_safety_tail_seconds": 4.0,
        "recovery_distribution_max_gap_seconds": 8.0,
        "recovery_distribution_allow_rejected_candidate": True,
        "continuation_gap_seconds": 12.0,
        "final_keeper_contact_tail_seconds": 4.0,
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, cfg)
    
    # They should be merged/chained
    assert len(results) == 1
    merged = results[0]
    # In this case, since c2 is a strong distribution, it gets marked as kick/throw
    # and then upgraded to controlled_release during chaining.
    assert merged.clip_end_reason == "controlled_release"
    # Action end should be extended to at least 342 (end of c2)
    assert merged.action_end >= 342.0
    # End should include safety tail
    assert merged.end >= 342.0 + 4.0
    assert merged.score_breakdown["recovery_distribution_detected"] == 1.0
    assert merged.score_breakdown["controlled_release_detected"] == 0.0

def test_recovery_distribution_rejected_absorption():
    # Case 3: Recovery-Candidate mit nachfolgendem rejected Distribution-Candidate.
    c1 = Candidate(
        candidate_id="raw-0012",
        start=100.0, end=115.0,
        trigger_time=105.0,
        action_start=105.0, action_end=110.0,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        possession_duration=0.1,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=5,
        keeper_motion=0.2,
        interaction_score=0.6
    )
    
    c2 = Candidate(
        candidate_id="dist-rejected",
        start=112.0, end=120.0,
        trigger_time=115.0,
        action_start=115.0, action_end=116.0,
        accepted=False, # REJECTED
        category="distribution",
        departure_speed=0.6,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=2,
        keeper_motion=0.2
    )
    
    cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "recovery_distribution_continuation_enabled": True,
        "recovery_distribution_search_seconds": 12.0,
        "recovery_distribution_safety_tail_seconds": 3.0,
        "recovery_distribution_allow_rejected_candidate": True,
        "continuation_gap_seconds": 12.0,
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, cfg)
    
    # Only c1 should be accepted, but extended by c2
    accepted_results = [r for r in results if r.accepted]
    assert len(accepted_results) == 1
    merged = accepted_results[0]
    assert merged.candidate_id == "raw-0012"
    assert merged.action_end >= 116.0
    assert merged.end >= 116.0 + 3.0
    assert merged.score_breakdown["recovery_distribution_absorbed_rejected_candidate"] == 1.0

def test_recovery_distribution_different_keeper():
    # Case 5: Nachfolgender Candidate gehört zu einem anderen keeper_label.
    c1 = Candidate(
        candidate_id="raw-0012",
        start=100.0, end=115.0,
        trigger_time=105.0,
        action_start=105.0, action_end=110.0,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1
    )
    
    c2 = Candidate(
        candidate_id="dist-other",
        start=112.0, end=120.0,
        trigger_time=115.0,
        action_start=115.0, action_end=116.0,
        accepted=True,
        category="distribution",
        keeper_label="Keeper #2", # OTHER KEEPER
        min_normalized_distance=0.1,
        keeper_track_id=2
    )
    
    cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "recovery_distribution_continuation_enabled": True,
        "recovery_distribution_search_seconds": 12.0,
        "continuation_gap_seconds": 12.0,
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, cfg)
    # No absorption -> 2 separate clips
    assert len(results) == 2
    assert results[0].score_breakdown.get("recovery_distribution_detected", 0.0) == 0.0

def test_recovery_distribution_clamped():
    # Case 9: Distribution überschreitet maximale Clipdauer.
    c1 = Candidate(
        candidate_id="raw-0012",
        start=100.0, end=110.0, # Proposed start would be 96 (100-4)
        trigger_time=100.0,
        action_start=100.0, action_end=110.0,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1
    )
    
    c2 = Candidate(
        candidate_id="dist-late",
        start=150.0, end=160.0,
        trigger_time=155.0,
        action_start=155.0, action_end=156.0,
        accepted=True,
        category="distribution",
        departure_speed=0.6,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1
    )
    
    cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "recovery_distribution_continuation_enabled": True,
        "recovery_distribution_search_seconds": 60.0,
        "recovery_distribution_safety_tail_seconds": 10.0,
        "continuation_gap_seconds": 12.0, # This prevents normal chaining
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, cfg)
    
    # c1 start = 96.0
    # c1 proposed end = 156 + 10 = 166.0
    # duration = 166 - 96 = 70.0 > 45.0
    
    assert results[0].end - results[0].start <= 45.0
    assert results[0].score_breakdown["recovery_distribution_clamped_by_max_duration"] == 1.0

def test_v0_13_11_regression_still_works():
    # Case 10: Direkte Controlled-Release-Erkennung aus Version 0.13.11 funktioniert weiterhin.
    c1 = Candidate(
        candidate_id="raw-0001",
        start=100.0, end=110.0,
        trigger_time=105.0,
        action_start=105.0, action_end=106.0,
        accepted=True,
        category="catch_or_control",
        possession_duration=2.0, # Strong possession
        departure_speed=0.8, # Strong departure in same candidate
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=10,
        keeper_motion=0.2
    )
    
    cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "controlled_release_enabled": True,
        "controlled_release_minimum_possession_seconds": 0.5,
        "controlled_release_minimum_departure_speed": 0.35,
        "controlled_release_safety_tail_seconds": 5.0,
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1], 1000.0, cfg)
    assert results[0].clip_end_reason == "controlled_release"
    assert results[0].score_breakdown["controlled_release_detected"] == 1.0

def test_v0_13_10_phase_merge_regression():
    # Case 11: Phase-Merge-Regression aus Version 0.13.10.
    c1 = Candidate(
        candidate_id="c1",
        start=218.24, end=240.60,
        trigger_time=228.24,
        action_start=228.24, action_end=229.60,
        accepted=True,
        category="catch_or_control",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        min_normalized_distance=0.1,
        keeper_track_id=1
    )
    c2 = Candidate(
        candidate_id="c2",
        start=236.72, end=268.00,
        trigger_time=240.72,
        action_start=240.72, action_end=264.00,
        accepted=True,
        category="recovery_uncovered_activity",
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        min_normalized_distance=0.1,
        keeper_track_id=1
    )
    
    cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "phase_merge_duration_tolerance": 0.08,
        "phase_merge_min_pre_roll_seconds": 2.0,
        "phase_merge_min_post_roll_seconds": 2.0,
        "seconds_before": 10.0,
        "seconds_after": 11.0,
        "continuation_gap_seconds": 12.0,
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, cfg)
    assert len(results) == 1
    assert results[0].score_breakdown["phase_merge_decision"] == 1.0
    assert results[0].end - results[0].start <= 45.0 * 1.08
