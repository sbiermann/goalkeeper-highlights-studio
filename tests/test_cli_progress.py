from __future__ import annotations

from goalkeeper_highlights.cli import TerminalProgress, _format_duration, _print_summary


def test_terminal_progress_hides_internal_tqdm_fields(capsys):
    progress = TerminalProgress(width=10)
    progress.update(0.03, "1.3/42.9 min, 2.31x realtime, raw candidates: 0")
    progress.close()
    output = capsys.readouterr().out
    assert "29/1000" not in output
    assert "‰" not in output
    assert "%/s" not in output
    assert "1.3/42.9 min" in output
    assert "Kandidaten 0" in output
    assert "2.31x" in output
    assert "ETA" in output


def test_terminal_progress_uses_phase_names(capsys):
    progress = TerminalProgress(width=10)
    progress.update(0.97, "Erstelle Clips 3/11")
    progress.close()
    output = capsys.readouterr().out
    assert "Clips" in output
    assert "3/11" in output


def test_format_duration():
    assert _format_duration(17) == "00:17"
    assert _format_duration(3671) == "1:01:11"


def test_summary_contains_release_0104_statistics(capsys):
    _print_summary({
        "video_name": "match.mp4",
        "accepted": 11,
        "rejected": 0,
        "raw_candidates": 18,
        "merged_candidates": 7,
        "keeper_label": "Keeper #1",
        "keeper_confidence": 0.91,
        "keeper_reidentifications": 42,
        "analysis_seconds": 1491.0,
        "realtime_factor": 1.73,
        "clip_creation_seconds": 11.0,
        "concat_seconds": 1.0,
        "total_seconds": 1504.0,
        "encoder": "h264_nvenc",
        "clip_mode": "accurate",
        "parallel_jobs": 2,
        "output": "C:/video/match_goalkeeper_highlights",
    })
    output = capsys.readouterr().out
    assert "Kandidaten:          18" in output
    assert "Zusammengeführt:     7" in output
    assert "Konfidenz:           91 %" in output
    assert "Re-Identifikationen: 42" in output
    assert "Geschwindigkeit:     1.73× Echtzeit" in output
    assert "match.mp4" in output
