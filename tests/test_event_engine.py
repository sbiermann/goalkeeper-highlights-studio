from goalkeeper_highlights.event_engine import GoalkeeperEventEngine
from goalkeeper_highlights.models import Box


def person(x1=100, y1=100, x2=180, y2=260, track=1):
    return Box(x1, y1, x2, y2, 0.9, 0, track)


def ball(x, y=170, conf=0.8):
    return Box(x-5, y-5, x+5, y+5, conf, 32, 9)


def cfg():
    return {
        "contact_distance_factor": 1.15,
        "release_distance_factor": 1.4,
        "minimum_contact_frames": 2,
        "minimum_event_duration_seconds": 0.08,
        "ball_lost_finalize_seconds": 0.3,
        "trajectory_lookback_seconds": 1.2,
        "event_cooldown_seconds": 0.2,
        "minimum_event_score": 0.20,
        "category_thresholds": {"catch_or_control": 0.20, "interaction": 0.20},
        "possession_bonus": {0.5: 0.02, 1.5: 0.05, 3.0: 0.08, 5.0: 0.12},
        "approach_speed_target": 0.75,
        "departure_speed_target": 0.75,
        "possession_min_seconds": 0.4,
        "possession_duration_target": 0.8,
        "keeper_motion_target": 0.75,
    }


def test_trajectory_event_is_emitted():
    engine = GoalkeeperEventEngine(cfg(), {"seconds_before": 2, "seconds_after": 2}, 20)
    keeper = person()
    emitted = []
    # Ball approaches, remains near for two frames, then departs.
    for t, x in [(0.0, 520), (0.2, 380), (0.4, 260), (0.6, 190), (0.8, 180), (1.0, 320), (1.2, 520)]:
        emitted += engine.update(t, keeper, ball(x), 0.9)
    assert emitted
    event = emitted[0]
    assert event.accepted
    assert event.event_score > 0
    assert event.approach_speed >= 0
    assert event.category in {"save_or_deflection", "ball_contact", "interaction", "distribution", "catch_or_control"}


def test_single_close_detection_is_not_event():
    engine = GoalkeeperEventEngine(cfg(), {"seconds_before": 2, "seconds_after": 2}, 20)
    keeper = person()
    emitted = engine.update(0.0, keeper, ball(180), 0.9)
    emitted += engine.update(0.5, keeper, None, 0.9)
    assert not emitted


def test_possession_bonus_and_explainable_score():
    engine = GoalkeeperEventEngine(cfg(), {"seconds_before": 2, "seconds_after": 2}, 20)
    keeper = person()
    emitted = []
    for t, x in [(0.0, 300), (0.2, 190), (0.4, 180), (1.0, 180), (1.8, 180), (2.0, 520)]:
        emitted += engine.update(t, keeper, ball(x), 0.9)
    assert emitted
    event = emitted[0]
    assert event.possession_bonus >= 0.02
    assert "possession_bonus" in event.score_breakdown
    assert event.acceptance_threshold > 0
    assert event.keeper_label == "Keeper #1"


def test_strong_contact_is_accepted_as_distribution_or_clearance():
    config = cfg()
    config.update({
        "category_thresholds": {"interaction": 0.80, "distribution": 0.22, "keeper_clearance": 0.26},
        "strong_contact_identity": 0.70,
        "strong_contact_ball_confidence": 0.40,
        "strong_contact_frames": 3,
        "strong_contact_max_distance": 0.35,
        "strong_contact_bonus": 0.08,
    })
    engine = GoalkeeperEventEngine(config, {"seconds_before": 2, "seconds_after": 2}, 20)
    keeper = person()
    emitted = []
    for t, x in [(0.0, 300), (0.2, 185), (0.4, 180), (0.6, 180), (0.8, 180), (1.0, 600)]:
        emitted += engine.update(t, keeper, ball(x, conf=0.8), 0.8)
    assert emitted
    event = emitted[0]
    assert event.accepted
    assert event.category in {"distribution", "keeper_clearance", "catch_or_control"}
    assert "confirmed_keeper_contact" in event.score_breakdown
