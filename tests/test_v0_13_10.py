from goalkeeper_highlights.detection import merge_candidates, _has_real_keeper_interaction, extend_and_chain_clip_windows
from goalkeeper_highlights.models import Candidate
from goalkeeper_highlights import __version__
import subprocess
import sys

def test_v0_13_18_version_consistency():
    # Central source must be 0.13.29
    assert __version__ == "0.13.29"

def test_v0_13_18_cli_version():
    # CLI must report 0.13.29
    result = subprocess.run([sys.executable, "-m", "goalkeeper_highlights.cli", "--version"],
                            capture_output=True, text=True, check=True)
    # Output is usually "goalkeeper-highlights 0.13.29" or similar depending on prog name
    assert "0.13.29" in result.stdout

def test_v0_13_10_merge_with_gap():
    # Candidates with 2.2s gap should NOT be merged in v0.13.10 if no possession flow
    c1 = Candidate(10.0, 15.0, 12.0, 0.1, 1, keeper_label="Keeper #1", contact_frames=2, approach_speed=0.2)
    c2 = Candidate(17.2, 20.0, 18.0, 0.1, 1, keeper_label="Keeper #1", contact_frames=2, approach_speed=0.2)
    
    merged = merge_candidates([c1, c2], gap=0.5, duration=100.0)
    assert len(merged) == 2

def test_v0_13_10_merge_possession_flow():
    # Candidates with 2.2s gap AND possession flow should be merged
    c1 = Candidate(10.0, 15.0, 12.0, 0.1, 1, keeper_label="Keeper #1", category="catch_or_control", contact_frames=2, approach_speed=0.2)
    c2 = Candidate(17.2, 20.0, 18.0, 0.1, 1, keeper_label="Keeper #1", category="distribution", contact_frames=2, approach_speed=0.2)
    
    merged = merge_candidates([c1, c2], gap=0.5, duration=100.0)
    assert len(merged) == 1
    assert merged[0].merged_reason == "same_keeper_possession_flow"

def test_v0_13_10_merge_different_keeper():
    # Candidates with 0.1s gap but DIFFERENT keeper should NOT be merged
    c1 = Candidate(10.0, 15.0, 12.0, 0.1, 1, keeper_label="Keeper #1", contact_frames=2, approach_speed=0.2)
    c2 = Candidate(15.1, 20.0, 18.0, 0.1, 2, keeper_label="Keeper #2", contact_frames=2, approach_speed=0.2)
    
    merged = merge_candidates([c1, c2], gap=0.5, duration=100.0)
    assert len(merged) == 2

def test_v0_13_10_merge_short_gap_no_possession():
    # Candidates with 0.3s gap should be merged even without possession flow
    c1 = Candidate(10.0, 15.0, 12.0, 0.1, 1, keeper_label="Keeper #1", contact_frames=2, approach_speed=0.2)
    c2 = Candidate(15.3, 20.0, 18.0, 0.1, 1, keeper_label="Keeper #1", contact_frames=2, approach_speed=0.2)
    
    merged = merge_candidates([c1, c2], gap=0.5, duration=100.0)
    assert len(merged) == 1
    assert merged[0].merged_reason == "same_keeper_within_merge_window"

def test_v0_13_10_interaction_validator_rejection():
    # Low dynamics, few contact frames -> reject
    c = Candidate(10.0, 12.0, 11.0, 0.1, 1, contact_frames=1, approach_speed=0.01, departure_speed=0.01, keeper_motion=0.01)
    cfg = {"interaction_validation": {"enabled": True}}
    accepted = _has_real_keeper_interaction(c, cfg)
    assert accepted is False
    assert c.rejection_reason == "insufficient_interaction_dynamics"

def test_v0_13_10_interaction_validator_acceptance():
    # Good dynamics -> accept
    c = Candidate(10.0, 12.0, 11.0, 0.1, 1, contact_frames=2, approach_speed=0.2, keeper_motion=0.1)
    cfg = {"interaction_validation": {"enabled": True}}
    accepted = _has_real_keeper_interaction(c, cfg)
    assert accepted is True
    assert c.interaction_score > 0

def test_v0_13_10_dynamic_clip_end_kick():
    # Distribution with high departure speed -> kick
    c = Candidate(10.0, 15.0, 12.0, 0.1, 1, category="distribution", action_start=11.0, action_end=14.0, departure_speed=0.5, approach_speed=0.1, contact_frames=2)
    c.merged_from = []
    cfg = {"seconds_before": 2, "seconds_after": 2, "interaction_validation": {"enabled": True}}
    result = extend_and_chain_clip_windows([c], 100.0, cfg)
    assert result[0].clip_end_reason == "kick"

def test_v0_13_10_dynamic_clip_end_throw():
    # Distribution with high departure speed and approach > departure
    c = Candidate(10.0, 15.0, 12.0, 0.1, 1, category="distribution", action_start=11.0, action_end=14.0, departure_speed=0.5, approach_speed=0.6, contact_frames=2)
    c.merged_from = []
    cfg = {"seconds_before": 2, "seconds_after": 2, "interaction_validation": {"enabled": True}}
    result = extend_and_chain_clip_windows([c], 100.0, cfg)
    assert result[0].clip_end_reason == "throw"

def test_v0_13_10_recovery_rejection():
    # Diagnostic-recovery-0003 case: recovery candidate with low score should be rejected
    c = Candidate(10.0, 12.0, 11.0, 0.1, 1, 
                  category="recovery_uncovered_activity", 
                  contact_frames=1, 
                  approach_speed=0.0, 
                  departure_speed=0.0, 
                  direction_change=0.0,
                  keeper_motion=0.1,
                  recovery_candidate=True)
    cfg = {"interaction_validation": {"enabled": True, "minimum_recovery_interaction_score": 0.45}}
    accepted = _has_real_keeper_interaction(c, cfg)
    assert accepted is False
    assert c.rejection_reason == "insufficient_recovery_interaction_score"

def test_v0_13_10_recovery_acceptance():
    # Recovery candidate with high dynamics should be accepted
    c = Candidate(10.0, 12.0, 11.0, 0.1, 1, 
                  category="recovery_uncovered_activity", 
                  contact_frames=1, 
                  approach_speed=0.5, 
                  departure_speed=0.5, 
                  direction_change=0.5,
                  keeper_motion=0.2,
                  recovery_candidate=True)
    cfg = {"interaction_validation": {"enabled": True, "minimum_recovery_interaction_score": 0.45}}
    accepted = _has_real_keeper_interaction(c, cfg)
    assert accepted is True

def test_v0_13_10_phase_merge():
    # Two related clips with a gap of 20s should be merged in Stufe B
    c1 = Candidate(10.0, 15.0, 12.0, 0.1, 1, keeper_label="Keeper #1", 
                   category="catch_or_control", action_start=11.0, action_end=14.0, 
                   contact_frames=5, keeper_motion=0.5)
    # Clip window for c1: 6.0 to 18.0 (default 5s before, 4s after)
    c2 = Candidate(40.0, 45.0, 42.0, 0.1, 1, keeper_label="Keeper #1", 
                   category="distribution", action_start=41.0, action_end=44.0, 
                   contact_frames=5, keeper_motion=0.5)
    # Clip window for c2: 36.0 to 48.0
    
    # Gap between clips: 36.0 - 18.0 = 18.0s
    
    c1.accepted = c2.accepted = True
    c1.clip_end_reason = "timeout"
    c2.clip_end_reason = "kick"
    
    cfg = {"continuation_gap_seconds": 12.0, "phase_merge_gap_seconds": 30.0, "max_dynamic_clip_seconds": 60.0,
           "interaction_validation": {"enabled": False}}
    
    result = extend_and_chain_clip_windows([c1, c2], 100.0, cfg)
    assert len(result) == 1
    assert result[0].score_breakdown["phase_merge_decision"] == 1.0
    assert result[0].clip_end_reason == "kick"

def test_v0_13_10_phase_merge_prevention():
    # Gap too large
    c1 = Candidate(10.0, 15.0, 12.0, 0.1, 1, keeper_label="Keeper #1", 
                   category="catch_or_control", action_start=11.0, action_end=14.0, contact_frames=5)
    c2 = Candidate(100.0, 105.0, 102.0, 0.1, 1, keeper_label="Keeper #1", 
                   category="distribution", action_start=101.0, action_end=104.0, contact_frames=5)
    
    c1.accepted = c2.accepted = True
    c1.clip_end_reason = "timeout"
    
    cfg = {"continuation_gap_seconds": 12.0, "phase_merge_gap_seconds": 30.0, "max_dynamic_clip_seconds": 120.0}
    
    result = extend_and_chain_clip_windows([c1, c2], 200.0, cfg)
    assert len(result) == 2
