import pytest
from pathlib import Path
from goalkeeper_highlights.models import Candidate
from goalkeeper_highlights.reporting import _breakdown, write_reports

def test_breakdown_with_non_numeric():
    # Candidate mit gemischtem score_breakdown
    c = Candidate(
        start=10.0, end=20.0, trigger_time=15.0, min_normalized_distance=0.1, keeper_track_id=1,
        score_breakdown={
            "numeric_val": 0.5,
            "string_val": "should_be_ignored",
            "bool_val": True,
            "negative_val": -0.2
        }
    )
    
    html_output = _breakdown(c)
    
    # Prüfen, dass numerische Werte vorhanden sind
    assert "Numeric Val" in html_output
    assert "0.500" in html_output
    assert "Negative Val" in html_output
    assert "0.200" in html_output
    
    # Prüfen, dass nicht-numerische Werte/Bools ignoriert wurden
    assert "String Val" not in html_output
    assert "should_be_ignored" not in html_output
    assert "Bool Val" not in html_output

def test_report_generation_with_phase_merge_reason(tmp_path):
    c1 = Candidate(
        start=100.0, end=110.0, trigger_time=105.0, min_normalized_distance=0.1, keeper_track_id=1,
        candidate_id="c1", accepted=True, category="catch",
        phase_merge_reason="absorbed_recovery_continuation",
        score_breakdown={"some_score": 1.0}
    )
    
    # write_reports erzeugt report.html und events.json
    write_reports(tmp_path, [c1])
    
    # Prüfen ob events.json den Grund enthält
    import json
    events_file = tmp_path / "events.json"
    assert events_file.exists()
    events_data = json.loads(events_file.read_text(encoding="utf-8"))
    assert events_data[0]["phase_merge_reason"] == "absorbed_recovery_continuation"
    
    # Prüfen ob score_breakdown in events.json nur numerisch ist
    # (Obwohl write_reports nur das Candidate-Dict dumped, 
    # stellen wir sicher dass wir keine Strings reingesteckt haben)
    assert events_data[0]["score_breakdown"]["some_score"] == 1.0
    assert len(events_data[0]["score_breakdown"]) == 1

def test_breakdown_empty_after_filtering():
    c = Candidate(
        start=10.0, end=20.0, trigger_time=15.0, min_normalized_distance=0.1, keeper_track_id=1,
        score_breakdown={"only_string": "ignored"}
    )
    assert _breakdown(c) == ""
