import unittest
from unittest.mock import MagicMock
from goalkeeper_highlights.models import Candidate
from goalkeeper_highlights.classification import calculate_routing, classify

class TestRoutingAndClassification(unittest.TestCase):
    def setUp(self):
        self.config = {
            "qwen": {
                "enabled": True,
                "model": "dummy",
                "routing": {
                    "enabled": True,
                    "high_threshold": 0.85,
                    "low_threshold": 0.15,
                    "recovery_force_medium": True,
                    "retry_enabled": True,
                    "retry_min_confidence": 0.40,
                    "retry_max_confidence": 0.70
                }
            },
            "runtime": {"verbose_console": False}
        }

    def test_routing_high(self):
        c = Candidate(start=0, end=5, trigger_time=2, min_normalized_distance=0.1, keeper_track_id=1)
        c.event_score = 0.90
        calculate_routing(c, self.config)
        self.assertEqual(c.routing_category, "HIGH")

    def test_routing_low(self):
        c = Candidate(start=0, end=5, trigger_time=2, min_normalized_distance=0.5, keeper_track_id=1)
        c.event_score = 0.10
        calculate_routing(c, self.config)
        self.assertEqual(c.routing_category, "LOW")

    def test_routing_medium(self):
        c = Candidate(start=0, end=5, trigger_time=2, min_normalized_distance=0.3, keeper_track_id=1)
        c.event_score = 0.50
        calculate_routing(c, self.config)
        self.assertEqual(c.routing_category, "MEDIUM")

    def test_routing_possession_bonus(self):
        c = Candidate(start=0, end=5, trigger_time=2, min_normalized_distance=0.2, keeper_track_id=1)
        c.event_score = 0.82
        c.possession_duration = 1.0
        calculate_routing(c, self.config)
        # 0.82 + 0.05 = 0.87 -> HIGH
        self.assertEqual(c.routing_category, "HIGH")

    def test_recovery_force_medium(self):
        c = Candidate(start=0, end=5, trigger_time=2, min_normalized_distance=0.1, keeper_track_id=1)
        c.event_score = 0.90
        c.recovery_candidate = True
        calculate_routing(c, self.config)
        self.assertEqual(c.routing_category, "MEDIUM")

    def test_classify_high_skips_qwen(self):
        c = Candidate(start=0, end=5, trigger_time=2, min_normalized_distance=0.1, keeper_track_id=1)
        c.event_score = 0.95
        candidates = [c]
        # We don't need a real video object because Qwen won't be called
        classify(None, candidates, self.config)
        self.assertTrue(c.accepted)
        self.assertEqual(c.category, "heuristic_high")
        self.assertEqual(c.qwen_retry_count, 0)

    def test_classify_low_skips_qwen(self):
        c = Candidate(start=0, end=5, trigger_time=2, min_normalized_distance=0.5, keeper_track_id=1)
        c.event_score = 0.05
        candidates = [c]
        classify(None, candidates, self.config)
        self.assertFalse(c.accepted)
        self.assertEqual(c.rejection_reason, "heuristic_low_score")
        self.assertEqual(c.qwen_retry_count, 0)

if __name__ == "__main__":
    unittest.main()

class TestQwenRetryDecision(unittest.TestCase):
    def setUp(self):
        self.config = {
            "qwen": {"routing": {
                "retry_enabled": True,
                "retry_min_confidence": 0.40,
                "retry_max_confidence": 0.70,
                "recovery_retry_min_confidence": 0.30,
                "short_action_motion_threshold": 0.30,
            }}
        }

    def test_uncertain_confidence_retries(self):
        from goalkeeper_highlights.classification import should_retry_qwen
        c = Candidate(0, 4, 2, 0.2, 1)
        self.assertTrue(should_retry_qwen(confidence=0.55, contact=True, parse_failed=False, candidate=c, config=self.config))

    def test_clear_confidence_does_not_retry(self):
        from goalkeeper_highlights.classification import should_retry_qwen
        c = Candidate(0, 4, 2, 0.2, 1)
        self.assertFalse(should_retry_qwen(confidence=0.92, contact=True, parse_failed=False, candidate=c, config=self.config))

    def test_parse_failure_retries(self):
        from goalkeeper_highlights.classification import should_retry_qwen
        c = Candidate(0, 4, 2, 0.2, 1)
        self.assertTrue(should_retry_qwen(confidence=0.0, contact=False, parse_failed=True, candidate=c, config=self.config))

    def test_no_retry_loop(self):
        from goalkeeper_highlights.classification import should_retry_qwen
        c = Candidate(0, 4, 2, 0.2, 1)
        c.qwen_retry_count = 1
        self.assertFalse(should_retry_qwen(confidence=0.55, contact=True, parse_failed=False, candidate=c, config=self.config))
