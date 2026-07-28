import numpy as np

from goalkeeper_highlights.keeper_bootstrap import AutomaticGoalkeeperDetector
from goalkeeper_highlights.models import Box


def test_bootstrap_prefers_large_near_goal_unique_track():
    cfg = {
        "goal_regions": [[0.0, 0.62, 1.0, 1.0]],
        "goal_region_falloff": 0.2,
        "bootstrap_min_observations": 2,
        "bootstrap_observation_target": 2,
        "bootstrap_min_score": 0.2,
        "bootstrap_min_confidence": 0.1,
        "bootstrap_weights": {
            "shirt_uniqueness": 0.15,
            "camera_proximity": 0.35,
            "goal_area": 0.35,
            "low_movement": 0.05,
            "ball_contact": 0.05,
            "persistence": 0.05,
        },
    }
    detector = AutomaticGoalkeeperDetector(cfg, 1000, 600)
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    frame[:, :] = (30, 120, 30)
    for t in (0.0, 1.0, 2.0):
        keeper = Box(400, 390, 560, 590, .9, 0, 7)
        outfield = Box(100 + 40*t, 180, 170 + 40*t, 330, .9, 0, 12)
        frame[int(keeper.y1):int(keeper.y2), int(keeper.x1):int(keeper.x2)] = (0, 0, 255)
        frame[int(outfield.y1):int(outfield.y2), int(outfield.x1):int(outfield.x2)] = (255, 0, 0)
        detector.observe(frame, [keeper, outfield], [], t)
    box, _, result = detector.select()
    assert box is not None
    assert box.track_id == 7
    assert result["selected_track_id"] == 7
    assert result["keeper_label"] == "Keeper #1"


def test_bootstrap_returns_explainable_ranking():
    cfg = {"bootstrap_min_observations": 1, "bootstrap_min_score": 0.0, "bootstrap_min_confidence": 0.0}
    detector = AutomaticGoalkeeperDetector(cfg, 640, 360)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detector.observe(frame, [Box(250, 240, 360, 355, .9, 0, 4)], [], 0.0)
    _, _, result = detector.select()
    assert result["method"] == "automatic_initial_window"
    assert result["ranking"]
    assert "shirt_uniqueness" in result["ranking"][0]
