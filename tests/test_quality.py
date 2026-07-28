from goalkeeper_highlights.detection import heuristic_score, merge_candidates
from goalkeeper_highlights.models import Candidate


def test_heuristic_score_rewards_confirmed_close_contact():
    cfg = {"distance_factor": 1.35, "minimum_contact_frames": 2}
    strong = heuristic_score(0.05, 3, 0.8, 0.9, cfg)
    weak = heuristic_score(1.2, 1, 0.16, 0.45, cfg)
    assert strong > weak
    assert strong > 0.7


def test_merge_preserves_best_quality_and_sums_confirmations():
    first = Candidate(1, 5, 3, 0.2, 1, contact_frames=2, heuristic_score=0.6, quality_score=0.6)
    second = Candidate(4, 8, 6, 0.1, 2, contact_frames=3, heuristic_score=0.8, quality_score=0.8)
    merged = merge_candidates([first, second], gap=0.5, duration=20)
    assert len(merged) == 1
    assert merged[0].contact_frames == 5
    assert merged[0].quality_score == 0.8
    assert merged[0].keeper_track_id == 2
