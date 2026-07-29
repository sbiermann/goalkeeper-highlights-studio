import json
import os
from pathlib import Path

import numpy as np

from goalkeeper_highlights.keeper_bootstrap import AutomaticGoalkeeperDetector
from goalkeeper_highlights.models import Box, Candidate
from goalkeeper_highlights.diagnostics import create_debug_package


def test_fragmented_tracks_are_grouped_as_logical_identity():
    cfg = {
        "bootstrap_min_observations": 1,
        "bootstrap_identity_hist_distance": .35,
        "bootstrap_identity_max_gap_seconds": 20,
        "bootstrap_min_score": 0,
        "bootstrap_min_confidence": 0,
        "bootstrap_min_margin": 0,
    }
    detector = AutomaticGoalkeeperDetector(cfg, 640, 360)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[180:340, 280:360] = (0, 0, 255)
    detector.observe(frame, [Box(280, 180, 360, 340, .9, 0, 11)], [], 0.0)
    detector.observe(frame, [Box(282, 181, 362, 341, .9, 0, 29)], [], 2.0)
    ranking = detector.rank()
    assert any(set(row["member_track_ids"]) == {11, 29} for row in ranking)


def test_candidate_has_lifecycle_event_container():
    candidate = Candidate(0, 1, .5, .2, 1)
    candidate.lifecycle_events.append({"stage": "raw"})
    assert candidate.as_dict()["lifecycle_events"][0]["stage"] == "raw"


def test_opencv_attempts_are_forced_on_package_import():
    import goalkeeper_highlights
    assert os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] == os.environ.get("GOALKEEPER_OPENCV_READ_ATTEMPTS", "65536")
