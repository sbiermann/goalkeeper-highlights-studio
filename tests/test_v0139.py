import numpy as np

from goalkeeper_highlights.keeper_bootstrap import AutomaticGoalkeeperDetector
from goalkeeper_highlights.models import Box


def _frame(box, colour=(0, 0, 255)):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[int(box.y1):int(box.y2), int(box.x1):int(box.x2)] = colour
    return frame


def test_restart_context_defers_field_excursion_penalty_and_records_context():
    cfg = {
        "goal_regions": [[0.0, 0.0, 1.0, 0.25], [0.0, 0.75, 1.0, 1.0]],
        "bootstrap_min_observations": 1,
        "bootstrap_min_score": 2.0,
        "bootstrap_min_confidence": 2.0,
        "bootstrap_min_margin": 2.0,
        "bootstrap_restart_context_seconds": 45.0,
        "bootstrap_restart_context_ratio": 0.25,
        "bootstrap_restart_field_penalty_factor": 0.15,
    }
    detector = AutomaticGoalkeeperDetector(cfg, 640, 360)
    advanced = Box(295, 150, 345, 250, .9, 0, 7)
    for timestamp in (0.0, 5.0, 10.0, 15.0):
        detector.observe(_frame(advanced), [advanced], [], timestamp)
    _, _, result = detector.select()
    row = result["ranking"][0]
    assert result["start_context"] == "restart_or_break"
    assert result["automatic_selection_deferred"] is True
    assert row["effective_field_excursion"] < row["field_excursion"]


def test_return_to_goal_adds_evidence_after_advanced_start():
    cfg = {
        "goal_regions": [[0.0, 0.0, 1.0, 0.25], [0.0, 0.75, 1.0, 1.0]],
        "goal_region_falloff": 0.05,
        "bootstrap_min_observations": 1,
        "bootstrap_identity_hist_distance": 1.0,
        "bootstrap_identity_max_gap_seconds": 120.0,
        "bootstrap_return_to_goal_target": 1,
    }
    detector = AutomaticGoalkeeperDetector(cfg, 640, 360)
    advanced = Box(295, 150, 345, 250, .9, 0, 7)
    near_goal = Box(295, 285, 345, 355, .9, 0, 7)
    detector.observe(_frame(advanced), [advanced], [], 0.0)
    detector.observe(_frame(near_goal), [near_goal], [], 60.0)
    row = detector.rank()[0]
    assert row["return_to_goal"] > 0.0
