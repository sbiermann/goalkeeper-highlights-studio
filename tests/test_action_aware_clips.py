from types import SimpleNamespace

from goalkeeper_highlights.detection import (
    clamp_clip_windows_to_sources,
    extend_and_chain_clip_windows,
)
from goalkeeper_highlights.models import Candidate


def test_action_boundaries_drive_clip_window():
    candidate = Candidate(90, 120, 100, 0.1, 1, accepted=True, category="save_or_deflection", action_start=96, action_end=104)
    result = extend_and_chain_clip_windows([candidate], 200, {
        "seconds_before": 4,
        "seconds_after": 4,
        "category_pre_roll_seconds": {"save_or_deflection": 6},
        "category_post_roll_seconds": {"save_or_deflection": 7},
        "continuation_gap_seconds": 12,
        "final_keeper_contact_tail_seconds": 5,
        "minimum_clip_seconds": 6,
        "max_dynamic_clip_seconds": 45,
        "interaction_validation": {"enabled": True},
    })
    assert result[0].start == 90
    assert result[0].end == 111
    assert result[0].clip_boundary_reason == "observed_action_window"


def test_static_long_contact_is_rejected():
    candidate = Candidate(10, 20, 15, 0.0, 1, accepted=True, category="catch_or_control", contact_frames=120, approach_speed=0.0, departure_speed=0.0, direction_change=0.0, keeper_motion=0.0, action_start=14, action_end=16)
    result = extend_and_chain_clip_windows([candidate], 100, {
        "interaction_validation": {"enabled": True, "extreme_contact_frames": 80, "minimum_motion_signal": 0.08},
    })
    assert result[0].accepted is False
    assert result[0].rejection_reason == "implausible_static_long_contact"


def test_source_boundary_is_hard_by_default():
    candidate = Candidate(95, 105, 99, 0.1, 1, accepted=True, action_start=97, action_end=103)
    manifest = SimpleNamespace(files=[
        SimpleNamespace(global_start_seconds=0.0, global_end_seconds=100.0),
        SimpleNamespace(global_start_seconds=100.0, global_end_seconds=200.0),
    ])
    clamp_clip_windows_to_sources([candidate], manifest, {"allow_cross_source_clips": False})
    assert candidate.end == 100.0
    assert candidate.clip_boundary_reason == "source_boundary_clamp"


def test_unrelated_goal_kick_is_not_chained_to_previous_restart():
    first = Candidate(10, 15, 12, .1, 1, accepted=True, category="distribution", action_start=11, action_end=13)
    second = Candidate(20, 25, 22, .1, 1, accepted=True, category="keeper_clearance", action_start=21, action_end=23)
    result = extend_and_chain_clip_windows([first, second], 100, {
        "seconds_before": 4, "seconds_after": 4,
        "category_pre_roll_seconds": {"distribution": 4, "keeper_clearance": 6},
        "category_post_roll_seconds": {"distribution": 12, "keeper_clearance": 11},
        "continuation_gap_seconds": 12, "final_keeper_contact_tail_seconds": 5,
        "minimum_clip_seconds": 6, "max_dynamic_clip_seconds": 45,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 2
    assert result[1].start == 15
    assert result[1].end == 34
