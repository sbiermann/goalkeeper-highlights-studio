from goalkeeper_highlights.detection import recover_missed_keeper_actions, _has_real_keeper_interaction, rescue_classic_keeper_actions
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
        nearest_previous_accepted_keeper_gap=40.0,
        nearest_previous_accepted_category="ball_contact",
        nearest_next_accepted_keeper_gap=50.0,
        nearest_next_accepted_category="distribution",
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


def test_contextual_recovery_rescue_accepts_clip_17_like_long_action_span_with_compact_recovery_window():
    candidate = Candidate(
        1566.0,
        1585.0,
        1574.0,
        0.80,
        1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=1566.0,
        action_end=1585.0,
        recovery_window_start=1574.0,
        recovery_window_end=1576.0,
        event_score=0.494,
        acceptance_threshold=0.42,
        interaction_score=0.149,
        ball_confidence=0.294,
        keeper_motion=0.198,
        contact_frames=1,
        possession_duration=0.0,
        nearest_previous_accepted_keeper_gap=40.0,
        nearest_previous_accepted_category="ball_contact",
        nearest_next_accepted_keeper_gap=50.0,
        nearest_next_accepted_category="distribution",
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


def test_clip_13_like_recovery_stays_rejected_without_compact_context():
    candidate = Candidate(
        1510.0,
        1529.0,
        1518.0,
        0.82,
        1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=1510.0,
        action_end=1529.0,
        recovery_window_start=1518.0,
        recovery_window_end=1522.5,
        event_score=0.51,
        acceptance_threshold=0.42,
        interaction_score=0.21,
        ball_confidence=0.31,
        keeper_motion=0.20,
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
    assert _has_real_keeper_interaction(candidate, cfg) is False
    assert candidate.rejection_reason == "insufficient_recovery_interaction_score"


def test_clip_14_like_recovery_stays_rejected_with_weak_contextual_signal():
    candidate = Candidate(
        1535.0,
        1554.0,
        1543.0,
        0.91,
        1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=1540.0,
        action_end=1544.5,
        recovery_window_start=1540.0,
        recovery_window_end=1544.5,
        event_score=0.47,
        acceptance_threshold=0.42,
        interaction_score=0.19,
        ball_confidence=0.27,
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



def test_contextual_recovery_neighbor_rescue_accepts_clip_17_like_wider_distance():
    candidate = Candidate(
        1566.0,
        1585.0,
        1574.0,
        1.02348,
        1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=1574.0,
        action_end=1576.0,
        recovery_window_start=1574.0,
        recovery_window_end=1576.0,
        event_score=0.49385,
        acceptance_threshold=0.42,
        interaction_score=0.149,
        ball_confidence=0.29393,
        keeper_motion=0.19838,
        contact_frames=1,
        possession_duration=0.0,
        nearest_previous_accepted_keeper_gap=40.96,
        nearest_previous_accepted_category="ball_contact",
        nearest_next_accepted_keeper_gap=57.76,
        nearest_next_accepted_category="distribution",
    )
    cfg = {
        "interaction_validation": {
            "enabled": True,
            "minimum_recovery_interaction_score": 0.45,
        }
    }
    assert _has_real_keeper_interaction(candidate, cfg) is True
    assert candidate.accepted is True
    assert candidate.score_breakdown.get("recovery_contextual_rescue_applied") == 1.0
    assert candidate.score_breakdown.get("recovery_neighbor_context_rescue") == 1.0


def test_contextual_recovery_neighbor_rescue_keeps_clip_13_like_distribution_context_rejected():
    candidate = Candidate(
        958.0,
        977.0,
        966.0,
        1.05951,
        1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=966.0,
        action_end=968.0,
        recovery_window_start=966.0,
        recovery_window_end=968.0,
        event_score=0.69640,
        acceptance_threshold=0.42,
        interaction_score=0.149,
        ball_confidence=0.58185,
        keeper_motion=0.32992,
        contact_frames=1,
        possession_duration=0.0,
        nearest_previous_accepted_keeper_gap=26.48,
        nearest_previous_accepted_category="distribution",
        nearest_next_accepted_keeper_gap=507.28,
        nearest_next_accepted_category="distribution",
    )
    cfg = {
        "interaction_validation": {
            "enabled": True,
            "minimum_recovery_interaction_score": 0.45,
        }
    }
    assert _has_real_keeper_interaction(candidate, cfg) is False
    assert candidate.accepted is False
    assert candidate.rejection_reason == "insufficient_recovery_interaction_score"


def test_contextual_recovery_neighbor_rescue_keeps_clip_14_like_weak_context_rejected():
    candidate = Candidate(
        1188.0,
        1207.0,
        1196.0,
        0.40987,
        1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=1196.0,
        action_end=1198.0,
        recovery_window_start=1196.0,
        recovery_window_end=1198.0,
        event_score=0.43948,
        acceptance_threshold=0.42,
        interaction_score=0.149,
        ball_confidence=0.32309,
        keeper_motion=0.11136,
        contact_frames=1,
        possession_duration=0.0,
        nearest_previous_accepted_keeper_gap=256.48,
        nearest_previous_accepted_category="distribution",
        nearest_next_accepted_keeper_gap=277.28,
        nearest_next_accepted_category="distribution",
    )
    cfg = {
        "interaction_validation": {
            "enabled": True,
            "minimum_recovery_interaction_score": 0.45,
        }
    }
    assert _has_real_keeper_interaction(candidate, cfg) is False
    assert candidate.accepted is False
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


def test_diagnostic_recovery_close_geometry_without_neighbor_context_is_rejected_like_false_positive_21():
    candidate = Candidate(
        1842.0, 1861.0, 1850.0, 0.14610, 1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=1850.0,
        action_end=1852.0,
        recovery_window_start=1850.0,
        recovery_window_end=1852.0,
        event_score=0.71109,
        acceptance_threshold=0.42,
        ball_confidence=0.45121,
        keeper_motion=0.49822,
        contact_frames=1,
        nearest_previous_accepted_keeper_gap=160.32,
        nearest_previous_accepted_category="catch_or_control",
        nearest_next_accepted_keeper_gap=87.28,
        nearest_next_accepted_category="catch_or_control",
    )
    cfg = {"interaction_validation": {"enabled": True, "minimum_recovery_interaction_score": 0.45}}
    assert _has_real_keeper_interaction(candidate, cfg) is False
    assert candidate.rejection_reason == "insufficient_recovery_interaction_score"


def test_diagnostic_recovery_two_frame_motion_does_not_bypass_recovery_threshold_like_false_positive_30():
    candidate = Candidate(
        2356.0, 2379.0, 2364.0, 0.95133, 1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=2364.0,
        action_end=2370.0,
        recovery_window_start=2364.0,
        recovery_window_end=2370.0,
        event_score=0.64511,
        acceptance_threshold=0.42,
        ball_confidence=0.46379,
        keeper_motion=0.78102,
        contact_frames=2,
        nearest_previous_accepted_keeper_gap=64.88,
        nearest_previous_accepted_category="catch_or_control",
        nearest_next_accepted_keeper_gap=197.76,
        nearest_next_accepted_category="distribution",
    )
    cfg = {"interaction_validation": {"enabled": True, "minimum_recovery_interaction_score": 0.45}}
    assert _has_real_keeper_interaction(candidate, cfg) is False
    assert candidate.rejection_reason == "insufficient_recovery_interaction_score"


def test_strong_outside_box_catch_control_is_rescued_like_raw_0057():
    candidate = Candidate(
        1935.28, 1971.0, 1939.28, 0.0, 1,
        accepted=True,
        category="catch_or_control",
        contact_frames=71,
        possession_duration=4.96,
        keeper_y_normalized=0.514,
        approach_speed=0.0664,
        departure_speed=0.0,
        direction_change=0.0,
        keeper_motion=2.285,
        ball_confidence=0.765,
    )
    cfg = {"interaction_validation": {"enabled": True, "minimum_motion_signal": 0.08, "outside_box_restart_min_seconds": 2.5}}
    assert _has_real_keeper_interaction(candidate, cfg) is True
    assert candidate.score_breakdown.get("restart_control_rescue_applied") == 1.0
    assert candidate.score_breakdown.get("restart_control_rescue_original_start") == 1935.28


def test_strong_outside_box_catch_control_low_motion_is_rescued_like_raw_0069():
    candidate = Candidate(
        2160.16, 2171.44, 2164.16, 0.0, 1,
        accepted=True,
        category="catch_or_control",
        contact_frames=29,
        possession_duration=2.72,
        keeper_y_normalized=0.513,
        approach_speed=0.0,
        departure_speed=0.0,
        direction_change=0.0,
        keeper_motion=0.061,
        ball_confidence=0.683,
    )
    cfg = {"interaction_validation": {"enabled": True, "minimum_motion_signal": 0.08, "outside_box_restart_min_seconds": 2.5}}
    assert _has_real_keeper_interaction(candidate, cfg) is True
    assert candidate.score_breakdown.get("restart_control_rescue_applied") == 1.0


def test_outside_box_control_rescue_still_rejects_weak_ball_control():
    candidate = Candidate(
        100.0, 112.0, 104.0, 0.1, 1,
        accepted=True, category="catch_or_control", contact_frames=21, possession_duration=2.7,
        keeper_y_normalized=0.5, approach_speed=0.0, departure_speed=0.0, direction_change=0.0,
        keeper_motion=0.02, ball_confidence=0.40,
    )
    cfg = {"interaction_validation": {"enabled": True, "minimum_motion_signal": 0.08, "outside_box_restart_min_seconds": 2.5}}
    assert _has_real_keeper_interaction(candidate, cfg) is False
    assert candidate.rejection_reason == "irrelevant_outside_box_restart"



def test_short_lateral_contact_rescue_accepts_raw_0068_like_interaction():
    candidate = Candidate(
        2141.92, 2150.56, 2145.92, 0.43925, 2,
        accepted=False, category="interaction", contact_frames=2,
        ball_confidence=0.37297, identity_confidence=0.58591,
        keeper_motion=0.06313, keeper_lateral_motion=0.19202,
        possession_duration=0.08, event_score=0.19421,
        acceptance_threshold=0.38, rejection_reason="event_score_below_category_threshold",
        action_start=2145.92, action_end=2146.56,
    )
    cfg = {
        "classic_action_rescue": {"enabled": True},
        "interaction_validation": {"enabled": True, "minimum_motion_signal": 0.08},
    }
    rescue_classic_keeper_actions([candidate], cfg)
    assert candidate.accepted is True
    assert candidate.score_breakdown.get("classic_action_rescue") == 1.0
    assert _has_real_keeper_interaction(candidate, cfg) is True
    assert candidate.accepted is True


def test_short_lateral_contact_rescue_keeps_known_weak_interactions_rejected():
    cases = [
        Candidate(129.44, 138.16, 133.44, 0.49150, 1, accepted=False, category="interaction",
                  contact_frames=2, ball_confidence=0.30920, identity_confidence=0.95974,
                  keeper_motion=0.00692, keeper_lateral_motion=0.01584, possession_duration=0.16,
                  event_score=0.21474, acceptance_threshold=0.38,
                  rejection_reason="event_score_below_category_threshold"),
        Candidate(691.12, 708.08, 695.12, 0.42133, 1, accepted=False, category="interaction",
                  contact_frames=4, ball_confidence=0.66133, identity_confidence=0.93996,
                  keeper_motion=0.01172, keeper_lateral_motion=0.01805, possession_duration=0.08,
                  event_score=0.21955, acceptance_threshold=0.38,
                  rejection_reason="event_score_below_category_threshold"),
        Candidate(1645.52, 1654.24, 1649.52, 0.0, 1, accepted=False, category="interaction",
                  contact_frames=2, ball_confidence=0.28754, identity_confidence=0.81344,
                  keeper_motion=0.00210, keeper_lateral_motion=0.00585, possession_duration=0.16,
                  event_score=0.29626, acceptance_threshold=0.38,
                  rejection_reason="event_score_below_category_threshold"),
    ]
    cfg = {
        "classic_action_rescue": {"enabled": True},
        "interaction_validation": {"enabled": True, "minimum_motion_signal": 0.08},
    }
    rescue_classic_keeper_actions(cases, cfg)
    assert all(candidate.accepted is False for candidate in cases)
