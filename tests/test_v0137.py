from pathlib import Path

from goalkeeper_highlights.classification import classify
from goalkeeper_highlights.detection import recover_uncovered_activity_windows
from goalkeeper_highlights.keeper_bootstrap import AutomaticGoalkeeperDetector
from goalkeeper_highlights.models import Box, Candidate


class RecoveryStore:
    def recovery_observations(self):
        return [
            {"frame_index": 1, "timestamp": 10.0, "keeper_track_id": 7,
             "kx1": 100, "ky1": 100, "kx2": 150, "ky2": 200,
             "ball_confidence": .85, "bx1": 145, "by1": 130, "bx2": 155, "by2": 140},
            {"frame_index": 2, "timestamp": 10.4, "keeper_track_id": 9,
             "kx1": 130, "ky1": 100, "kx2": 180, "ky2": 200,
             "ball_confidence": .80, "bx1": 150, "by1": 135, "bx2": 160, "by2": 145},
        ]


def test_uncovered_activity_becomes_recovery_candidate():
    config = {"diagnostics": {"window_seconds": 2.0}, "event_engine": {"recovery_pass": {
        "diagnostic_window_recovery_enabled": True,
        "diagnostic_min_score": .3,
        "diagnostic_min_ball_confidence": .2,
        "diagnostic_min_keeper_motion": .01,
        "diagnostic_max_distance": 1.5,
        "seconds_before": 4,
        "seconds_after": 5,
    }}}
    result = recover_uncovered_activity_windows(RecoveryStore(), [], 30, config)
    assert len(result) == 1
    assert result[0].recovery_candidate
    assert result[0].category == "recovery_uncovered_activity"
    assert result[0].keeper_track_id in {7, 9}


def test_qwen_disabled_still_records_routing():
    c = Candidate(0, 2, 1, .1, 1, event_score=.9, accepted=True)
    config = {"qwen": {"enabled": False, "routing": {"enabled": True, "high_threshold": .85, "low_threshold": .15}}}
    stats = classify(None, [c], config)
    assert c.routing_category == "HIGH"
    assert c.routing_score >= .85
    assert stats["heuristic_seconds"] >= 0


def test_bootstrap_requires_winner_margin(monkeypatch):
    detector = AutomaticGoalkeeperDetector({"bootstrap_min_score": .4, "bootstrap_min_confidence": .4,
                                             "bootstrap_min_margin": .05}, 1000, 500)
    box1 = Box(1, 1, 20, 50, .9, 0, 1)
    box2 = Box(30, 1, 50, 50, .9, 0, 2)
    class Evidence:
        last_box = box1
        last_frame = None
    detector.tracks[1] = Evidence()
    monkeypatch.setattr(detector, "rank", lambda: [{"track_id": 1, "score": .7}, {"track_id": 2, "score": .68}])
    box, frame, result = detector.select()
    assert box is None
    assert not result["selected"]
    assert result["margin_to_second"] < .05
