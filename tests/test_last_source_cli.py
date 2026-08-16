from pathlib import Path

from goalkeeper_highlights.cli import parser
from goalkeeper_highlights.sources import discover_video_files


def test_cli_accepts_only_last_source():
    args = parser().parse_args(["analyze", "videos", "--only-last-source"])
    assert args.only_last_source is True


def test_cli_accepts_analyze_duration():
    args = parser().parse_args(["analyze", "videos", "--duration", "625"])
    assert args.duration == 625.0


def test_natural_last_source_selection(tmp_path: Path):
    for name in ["MatchTeil22.mp4", "MatchTeil1.mp4", "MatchTeil21.mp4"]:
        (tmp_path / name).write_bytes(b"")
    assert discover_video_files(tmp_path)[-1].name == "MatchTeil22.mp4"
