from pathlib import Path
from unittest.mock import patch

from goalkeeper_highlights.sources import prepare_source_timeline


def test_last_source_selection_keeps_only_naturally_last_file(tmp_path: Path):
    source = tmp_path / "match"
    source.mkdir()
    for name in ["SpielTeil22.mp4", "SpielTeil1.mp4", "SpielTeil21.mp4"]:
        (source / name).write_bytes(b"x")
    output = tmp_path / "out"
    with patch("goalkeeper_highlights.sources.duration_seconds", return_value=100.0):
        manifest = prepare_source_timeline(source, output, source_selection="last")
    assert manifest.source_type == "directory-last"
    assert [item.name for item in manifest.files] == ["SpielTeil22.mp4"]
    assert manifest.total_duration_seconds == 100.0


def test_all_source_selection_preserves_natural_order(tmp_path: Path):
    source = tmp_path / "match"
    source.mkdir()
    for name in ["SpielTeil22.mp4", "SpielTeil1.mp4", "SpielTeil21.mp4"]:
        (source / name).write_bytes(b"x")
    output = tmp_path / "out"
    with patch("goalkeeper_highlights.sources.duration_seconds", return_value=100.0):
        manifest = prepare_source_timeline(source, output, source_selection="all")
    assert [item.name for item in manifest.files] == ["SpielTeil1.mp4", "SpielTeil21.mp4", "SpielTeil22.mp4"]
    assert manifest.total_duration_seconds == 300.0
