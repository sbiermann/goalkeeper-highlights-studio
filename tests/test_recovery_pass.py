from goalkeeper_highlights.detection import recover_missed_keeper_actions, _has_real_keeper_interaction
from goalkeeper_highlights.models import Candidate


class FakeStore:
    def recovery_observations(self):
        return [
            {"frame_index": 1, "timestamp": 10.0, "keeper_track_id": 7,
             "kx1": 100, "ky1": 100, "kx2": 200, "ky2": 300,
             "ball_confidence": .8, "bx1": 190, "by1": 180, "bx2": 202, "by2": 192},
            {"frame_index": 2, "timestamp": 10.08, "keeper_track_id": 7,
             "kx1": 112, "ky1": 100, "kx2": 212, "ky2": 300,
             "ball_confidence": .82, "bx1": 200, "by1": 180, "bx2": 212, "by2": 192},
        ]


def test_recovery_pass_is_generic_and_creates_candidate():
    config = {"event_engine": {"recovery_pass": {"enabled": True, "minimum_close_frames": 2}}}
    items = recover_missed_keeper_actions(FakeStore(), [], 100.0, config)
    assert len(items) == 1
    assert items[0].recovery_candidate is True
    assert items[0].category == "recovery_keeper_interaction"
    assert items[0].start < 10.0 < items[0].end


def test_recovery_pass_masks_existing_events():
    existing = [Candidate(8, 12, 10, .2, 7)]
    config = {"event_engine": {"recovery_pass": {"enabled": True, "minimum_close_frames": 2}}}
    assert recover_missed_keeper_actions(FakeStore(), existing, 100.0, config) == []


def test_irrelevant_central_restart_is_rejected():
    candidate = Candidate(0, 10, 5, .1, 1, category="distribution", contact_frames=20,
                          possession_duration=4.0, keeper_y_normalized=.5,
                          approach_speed=.01, departure_speed=.01, direction_change=.01,
                          keeper_motion=.01)
    cfg = {"interaction_validation": {"enabled": True, "suspicious_contact_frames": 12,
                                       "minimum_motion_signal": .08,
                                       "central_field_y_min": .36, "central_field_y_max": .64,
                                       "outside_box_restart_min_seconds": 2.5}}
    assert _has_real_keeper_interaction(candidate, cfg) is False
    assert candidate.rejection_reason == "irrelevant_outside_box_restart"


def test_strong_distribution_survives_outside_box_restart_guard():
    candidate = Candidate(
        920.24,
        943.52,
        924.24,
        0.0,
        1,
        accepted=True,
        category="distribution",
        contact_frames=31,
        possession_duration=2.8,
        keeper_y_normalized=0.5,
        approach_speed=0.01,
        departure_speed=1.64,
        direction_change=0.01,
        keeper_motion=4.13,
        ball_confidence=0.736,
    )
    cfg = {
        "interaction_validation": {
            "enabled": True,
            "suspicious_contact_frames": 12,
            "minimum_motion_signal": 0.08,
            "central_field_y_min": 0.36,
            "central_field_y_max": 0.64,
            "outside_box_restart_min_seconds": 2.5,
        }
    }
    assert _has_real_keeper_interaction(candidate, cfg) is True
    assert candidate.accepted is True
    assert candidate.score_breakdown.get("restart_relevance_rescue_applied") == 1.0


def test_weak_isolated_restart_remains_rejected_after_rescue_rule():
    candidate = Candidate(
        100.0,
        120.0,
        110.0,
        0.2,
        1,
        accepted=True,
        category="distribution",
        contact_frames=12,
        possession_duration=2.8,
        keeper_y_normalized=0.5,
        approach_speed=0.01,
        departure_speed=0.2,
        direction_change=0.01,
        keeper_motion=0.02,
        ball_confidence=0.40,
    )
    cfg = {
        "interaction_validation": {
            "enabled": True,
            "suspicious_contact_frames": 12,
            "minimum_motion_signal": 0.08,
            "central_field_y_min": 0.36,
            "central_field_y_max": 0.64,
            "outside_box_restart_min_seconds": 2.5,
        }
    }
    assert _has_real_keeper_interaction(candidate, cfg) is False
    assert candidate.rejection_reason == "irrelevant_outside_box_restart"


def test_contextual_recovery_rescue_accepts_compact_valid_recovery_window():
    candidate = Candidate(
        1566.0,
        1585.0,
        1574.0,
        0.80,
        1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=1574.0,
        action_end=1576.0,
        recovery_window_start=1574.0,
        recovery_window_end=1576.0,
        event_score=0.494,
        acceptance_threshold=0.42,
        interaction_score=0.149,
        ball_confidence=0.294,
        keeper_motion=0.198,
        contact_frames=1,
        possession_duration=0.0,
    )
    cfg = {
        "interaction_validation": {
            "enabled": True,
            "suspicious_contact_frames": 12,
            "minimum_motion_signal": 0.08,
            "minimum_recovery_interaction_score": 0.45,
        }
    }
    assert _has_real_keeper_interaction(candidate, cfg) is True
    assert candidate.accepted is True
    assert candidate.score_breakdown.get("recovery_contextual_rescue_applied") == 1.0


def test_context_free_recovery_stays_rejected_even_if_interaction_score_is_higher():
    candidate = Candidate(
        1500.0,
        1519.0,
        1510.0,
        0.95,
        1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=1510.0,
        action_end=1512.0,
        recovery_window_start=1510.0,
        recovery_window_end=1517.0,
        event_score=0.49,
        acceptance_threshold=0.42,
        interaction_score=0.26,
        ball_confidence=0.31,
        keeper_motion=0.19,
        contact_frames=0,
        possession_duration=0.0,
    )
    cfg = {
        "interaction_validation": {
            "enabled": True,
            "suspicious_contact_frames": 12,
            "minimum_motion_signal": 0.08,
            "minimum_recovery_interaction_score": 0.45,
        }
    }
    assert _has_real_keeper_interaction(candidate, cfg) is False
    assert candidate.rejection_reason == "insufficient_recovery_interaction_score"


class StaticFakeStore:
    def recovery_observations(self):
        return [
            {"frame_index": 1, "timestamp": 10.0, "keeper_track_id": 7,
             "kx1": 100, "ky1": 100, "kx2": 200, "ky2": 300,
             "ball_confidence": .9, "bx1": 150, "by1": 180, "bx2": 162, "by2": 192},
            {"frame_index": 2, "timestamp": 10.08, "keeper_track_id": 7,
             "kx1": 100, "ky1": 100, "kx2": 200, "ky2": 300,
             "ball_confidence": .9, "bx1": 150, "by1": 180, "bx2": 162, "by2": 192},
        ]


def test_recovery_pass_does_not_accept_static_overlap():
    config = {"event_engine": {"recovery_pass": {"enabled": True, "minimum_close_frames": 2}}}
    assert recover_missed_keeper_actions(StaticFakeStore(), [], 100.0, config) == []
