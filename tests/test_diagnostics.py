from pathlib import Path
from types import SimpleNamespace
import json
import zipfile

from goalkeeper_highlights.diagnostics import create_debug_package
from goalkeeper_highlights.models import Candidate


class FakeStore:
    def checkpoint(self):
        self.did_checkpoint = True

    def recovery_observations(self):
        return [
            {"timestamp": 10.0, "kx1": 100, "ky1": 100, "kx2": 150, "ky2": 200,
             "bx1": 160, "by1": 120, "bx2": 170, "by2": 130, "ball_confidence": 0.8},
            {"timestamp": 11.0, "kx1": 150, "ky1": 100, "kx2": 200, "ky2": 200,
             "bx1": 180, "by1": 130, "bx2": 190, "by2": 140, "ball_confidence": 0.7},
        ]


def test_debug_package_contains_no_video(tmp_path: Path):
    candidate = Candidate(0, 3, 1, 0.2, 1, accepted=False, rejection_reason="test")
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "scene.mp4").write_bytes(b"video")
    (tmp_path / "events.json").write_text("[]")
    store = FakeStore()
    archive = create_debug_package(tmp_path, [candidate], {"accepted": 0}, {}, {"diagnostics": {}}, store)
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "debug/all_candidates.json" in names
    assert "debug/uncovered_suspicious_windows.json" in names
    assert "events.json" in names
    assert not any(name.endswith(".mp4") for name in names)
    assert store.did_checkpoint


def test_candidate_decision_path_is_written(tmp_path: Path):
    candidate = Candidate(0, 3, 1, 0.2, 1, routing_category="MEDIUM", qwen_first_pass_called=True)
    archive = create_debug_package(tmp_path, [candidate], {}, {}, {"diagnostics": {}}, FakeStore())
    data = json.loads((tmp_path / "debug" / "all_candidates.json").read_text())
    assert data[0]["decision_path"]["routing"] == "MEDIUM"
    assert data[0]["decision_path"]["qwen_first_pass"] is True
