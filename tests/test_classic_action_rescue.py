from goalkeeper_highlights.detection import rescue_classic_keeper_actions
from goalkeeper_highlights.models import Candidate


def test_classic_dynamic_keeper_contact_is_promoted():
    candidate = Candidate(
        10, 20, 15, 0.2, 7, accepted=False, category="save_or_deflection",
        rejection_reason="event_score_below_category_threshold", contact_frames=2,
        identity_confidence=.8, ball_confidence=.6, approach_speed=.4,
        event_score=.38, acceptance_threshold=.40,
    )
    rescue_classic_keeper_actions([candidate], {"classic_action_rescue": {"enabled": True}})
    assert candidate.accepted is True
    assert candidate.rejection_reason == ""
    assert candidate.score_breakdown["classic_action_rescue"] == 1.0


def test_cooldown_rejection_is_not_promoted():
    candidate = Candidate(
        10, 20, 15, 0.2, 7, accepted=False, category="save_or_deflection",
        rejection_reason="cooldown_lower_score_than_previous_candidate", contact_frames=3,
        identity_confidence=.9, ball_confidence=.9, approach_speed=.8,
    )
    rescue_classic_keeper_actions([candidate], {"classic_action_rescue": {"enabled": True}})
    assert candidate.accepted is False
