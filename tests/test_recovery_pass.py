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
