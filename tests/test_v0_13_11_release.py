import pytest
from goalkeeper_highlights.models import Candidate
from goalkeeper_highlights.detection import extend_and_chain_clip_windows

def test_controlled_release_punt():
    # Case 1: Kontrollierte Ballannahme mit anschließendem klar bestätigtem Abschlag.
    c1 = Candidate(
        candidate_id="raw-0001",
        start=320.0, end=335.0,
        trigger_time=324.0,
        action_start=324.0, action_end=330.0,
        accepted=True,
        category="catch_or_control",
        possession_duration=2.0,
        departure_speed=0.1,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=10,
        keeper_motion=0.2
    )
    
    # Subsequent release event
    c2 = Candidate(
        candidate_id="raw-0002",
        start=332.0, end=345.0,
        trigger_time=335.0,
        action_start=335.0, action_end=336.0,
        accepted=True,
        category="distribution",
        departure_speed=0.8,
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
        "controlled_release_enabled": True,
        "controlled_release_search_seconds": 12.0,
        "controlled_release_safety_tail_seconds": 4.0,
        "controlled_release_minimum_possession_seconds": 0.5,
        "controlled_release_minimum_departure_speed": 0.35,
        "continuation_gap_seconds": 12.0,
        "final_keeper_contact_tail_seconds": 4.0,
        "phase_merge_gap_seconds": 30.0,
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, cfg)
    
    # They should be merged to one clip
    assert len(results) == 1
    merged = results[0]
    assert merged.clip_end_reason == "controlled_release"
    # Action end should be extended to at least 336 (end of c2)
    assert merged.action_end >= 336.0
    # End should include safety tail
    assert merged.end >= 336.0 + 4.0

def test_controlled_release_recovery():
    # Case 3: Recovery-Kandidat mit ausreichender Besitz- und Release-Evidenz.
    # Repro for the problem clip 6 (raw-0012)
    c1 = Candidate(
        candidate_id="raw-0012",
        start=320.32, end=340.00,
        trigger_time=324.32,
        action_start=324.32, action_end=336.00,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        possession_duration=2.0,
        departure_speed=0.1,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=10,
        keeper_motion=0.2,
        interaction_score=0.6
    )
    
    # Subsequent release in recovery
    c2 = Candidate(
        candidate_id="rec-0013",
        start=338.0, end=342.0,
        trigger_time=339.0,
        action_start=339.0, action_end=340.0,
        accepted=True, # MUST BE ACCEPTED FOR CHAINING/MERGING TEST
        recovery_candidate=True,
        category="distribution", # Release should be distribution category
        departure_speed=0.6,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=5,
        keeper_motion=0.2,
        interaction_score=0.6
    )
    
    cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "controlled_release_enabled": True,
        "controlled_release_search_seconds": 12.0,
        "controlled_release_safety_tail_seconds": 4.0,
        "controlled_release_minimum_possession_seconds": 0.5,
        "controlled_release_minimum_departure_speed": 0.35,
        "continuation_gap_seconds": 12.0,
        "final_keeper_contact_tail_seconds": 4.0,
        "phase_merge_gap_seconds": 30.0,
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, cfg)
    
    assert len(results) == 1
    merged = results[0]
    assert merged.candidate_id == "raw-0012"
    assert merged.clip_end_reason == "controlled_release"
    assert merged.action_end >= 340.0
    assert merged.end >= 340.0 + 4.0
    assert merged.score_breakdown["controlled_release_detected"] == 1.0

def test_controlled_release_clamped():
    # Case 7: Ballfreigabe liegt hinter der maximal zulässigen Clipdauer.
    c1 = Candidate(
        candidate_id="raw-0001",
        start=100.0, end=110.0,
        trigger_time=110.0,
        action_start=110.0, action_end=120.0,
        accepted=True,
        category="catch_or_control",
        possession_duration=10.0,
        departure_speed=0.1,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=10,
        keeper_motion=0.2
    )
    
    # Release far away
    c2 = Candidate(
        candidate_id="raw-0002",
        start=150.0, end=160.0,
        trigger_time=155.0,
        action_start=155.0, action_end=156.0,
        accepted=True,
        category="distribution",
        departure_speed=0.8,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=5,
        keeper_motion=0.2
    )
    
    cfg = {
        "max_dynamic_clip_seconds": 45.0, # Limit!
        "seconds_before": 10.0, # Start at 100
        "seconds_after": 10.0,
        "controlled_release_enabled": True,
        "controlled_release_search_seconds": 40.0,
        "controlled_release_safety_tail_seconds": 5.0,
        "controlled_release_minimum_possession_seconds": 0.5,
        "controlled_release_minimum_departure_speed": 0.35,
        "continuation_gap_seconds": 40.0,
        "final_keeper_contact_tail_seconds": 5.0,
        "interaction_validation": {"enabled": False}
    }
    
    # c1 is distribution, so it chains to c2 if within 40s gap.
    # action_gap = 155 - 120 = 35. 35 <= 40.
    # Clip 1 will be extended by release logic (detects c2).
    # Then chaining in Pass A:
    # Gap: action_start(c2)[155] - action_end(c1)[156] = -1.
    # But proposed_end = 156 + 5 = 161.
    # proposed_end (161) - c1.start (90) = 71 > 45.
    # Max duration exceeded -> No chaining in A!
    # They should remain as 2 clips because even together they exceed max_duration.
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, cfg)
    
    assert len(results) == 2
    # But c1 should be clamped to 45s
    assert results[0].end - results[0].start <= 45.0
    assert results[0].score_breakdown["controlled_release_clamped_by_max_duration"] == 1.0

def test_controlled_release_no_possession():
    # Case 4: Recovery-Kandidat ohne kontrollierten Ballbesitz.
    c1 = Candidate(
        candidate_id="raw-0001",
        start=100.0, end=110.0,
        trigger_time=105.0,
        action_start=105.0, action_end=106.0,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        possession_duration=0.1, # Tiny possession
        departure_speed=0.1,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=10,
        keeper_motion=0.2,
        interaction_score=0.6
    )
    
    c2 = Candidate(
        candidate_id="raw-0002",
        start=108.0, end=115.0,
        trigger_time=110.0,
        action_start=110.0, action_end=111.0,
        accepted=True,
        category="distribution",
        departure_speed=0.8,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=5,
        keeper_motion=0.2
    )
    
    cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "controlled_release_enabled": True,
        "controlled_release_minimum_possession_seconds": 0.5,
        "continuation_gap_seconds": 12.0,
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, cfg)
    assert len(results) == 1
    assert results[0].score_breakdown["controlled_release_detected"] == 0.0

def test_controlled_release_different_keepers():
    # Case 8: Unterschiedliche keeper_label.
    c1 = Candidate(
        candidate_id="raw-0001",
        start=100.0, end=110.0,
        trigger_time=105.0,
        action_start=105.0, action_end=106.0,
        accepted=True,
        category="catch_or_control",
        possession_duration=2.0,
        departure_speed=0.1,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=10,
        keeper_motion=0.2
    )
    
    c2 = Candidate(
        candidate_id="raw-0002",
        start=108.0, end=115.0,
        trigger_time=110.0,
        action_start=110.0, action_end=111.0,
        accepted=True,
        category="distribution",
        departure_speed=0.8,
        keeper_label="Keeper #2", # DIFFERENT KEEPER
        min_normalized_distance=0.1,
        keeper_track_id=2,
        contact_frames=5,
        keeper_motion=0.2
    )
    
    cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "controlled_release_enabled": True,
        "controlled_release_minimum_possession_seconds": 0.5,
        "continuation_gap_seconds": 12.0,
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, cfg)
    # They should NOT chain because of different keepers
    assert len(results) == 2
    assert results[0].score_breakdown["controlled_release_detected"] == 0.0

def test_controlled_release_rejected_regression():
    # Case 10: Regression für abgelehnte Kandidaten.
    c1 = Candidate(
        candidate_id="raw-0001",
        start=100.0, end=110.0,
        trigger_time=105.0,
        action_start=105.0, action_end=106.0,
        accepted=False, # REJECTED
        category="catch_or_control",
        possession_duration=2.0,
        departure_speed=0.1,
        keeper_label="Keeper #1",
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=10,
        keeper_motion=0.2
    )
    
    results = extend_and_chain_clip_windows([c1], 1000.0, {"interaction_validation": {"enabled": False}})
    # Still rejected
    assert len(results) == 1
    assert results[0].accepted == False

def test_regression_v0_13_10_phase_merge():
    # Regression für den Phase-Merge aus Version 0.13.10.
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
        keeper_track_id=1,
        contact_frames=10,
        keeper_motion=0.2
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
        recovery_candidate=True,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        contact_frames=10,
        keeper_motion=0.2,
        interaction_score=0.6
    )
    
    clips_cfg = {
        "max_dynamic_clip_seconds": 45.0,
        "phase_merge_duration_tolerance": 0.08,
        "phase_merge_gap_seconds": 30.0,
        "category_pre_roll_seconds": {"catch_or_control": 10.0},
        "category_post_roll_seconds": {"catch_or_control": 11.0},
        "phase_merge_min_pre_roll_seconds": 2.0,
        "phase_merge_min_post_roll_seconds": 2.0,
        "continuation_gap_seconds": 1.0, # Force no chaining in pass A
        "interaction_validation": {"enabled": False}
    }
    
    results = extend_and_chain_clip_windows([c1, c2], 1000.0, clips_cfg)
    assert len(results) == 1
    assert "c2" in results[0].merged_from
    assert results[0].end - results[0].start <= 48.6
