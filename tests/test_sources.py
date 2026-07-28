from __future__ import annotations

from pathlib import Path

import pytest

from goalkeeper_highlights.sources import discover_video_files, natural_sort_key


def test_directory_videos_are_naturally_sorted(tmp_path: Path):
    for name in ["MVI_10.MP4", "MVI_2.MP4", "MVI_1.MP4", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    result = discover_video_files(tmp_path)
    assert [item.name for item in result] == ["MVI_1.MP4", "MVI_2.MP4", "MVI_10.MP4"]


def test_directory_scan_is_not_recursive(tmp_path: Path):
    (tmp_path / "MVI_1.MP4").write_bytes(b"")
    nested = tmp_path / "archive"
    nested.mkdir()
    (nested / "MVI_2.MP4").write_bytes(b"")
    result = discover_video_files(tmp_path)
    assert [item.name for item in result] == ["MVI_1.MP4"]


def test_generated_highlight_video_is_ignored(tmp_path: Path):
    (tmp_path / "MVI_1.MP4").write_bytes(b"")
    (tmp_path / "MVI_1_goalkeeper_highlights.mp4").write_bytes(b"")
    result = discover_video_files(tmp_path)
    assert [item.name for item in result] == ["MVI_1.MP4"]


def test_empty_directory_fails(tmp_path: Path):
    with pytest.raises(ValueError, match="No supported video files"):
        discover_video_files(tmp_path)


def test_real_world_part_names_are_sorted_in_recording_order(tmp_path: Path):
    names = [
        "FCWittlingen-SFETeil22-Tonasync.mp4",
        "FCWittlingen-SFETeil11.MP4",
        "FCWittlingen-SFETeil21.MP4",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in discover_video_files(tmp_path)] == [
        "FCWittlingen-SFETeil11.MP4",
        "FCWittlingen-SFETeil21.MP4",
        "FCWittlingen-SFETeil22-Tonasync.mp4",
    ]


def test_source_timeline_file_is_ignored(tmp_path: Path):
    (tmp_path / "MVI_1.MP4").write_bytes(b"")
    (tmp_path / "source_timeline.mp4").write_bytes(b"")
    assert [p.name for p in discover_video_files(tmp_path)] == ["MVI_1.MP4"]


def test_exact_reported_fc_wittlingen_order(tmp_path: Path):
    for name in [
        "FCWittlingen-SFETeil22-Tonasync.mp4",
        "FCWittlingen-SFETeil1.MP4",
        "FCWittlingen-SFETeil21.MP4",
    ]:
        (tmp_path / name).write_bytes(b"")
    result = discover_video_files(tmp_path)
    assert [p.name for p in result] == [
        "FCWittlingen-SFETeil1.MP4",
        "FCWittlingen-SFETeil21.MP4",
        "FCWittlingen-SFETeil22-Tonasync.mp4",
    ]


def test_sequence_number_wins_over_inconsistent_text_prefix(tmp_path: Path):
    # Real-world camera files can contain a typo in one filename. The numeric
    # segment number must still define recording order.
    for name in [
        "FCWittlinen-SFETeil22-Tonasync.mp4",
        "FCWittlingen-SFETeil1.MP4",
        "FCWittlingen-SFETeil21.MP4",
    ]:
        (tmp_path / name).write_bytes(b"")
    result = discover_video_files(tmp_path)
    assert [p.name for p in result] == [
        "FCWittlingen-SFETeil1.MP4",
        "FCWittlingen-SFETeil21.MP4",
        "FCWittlinen-SFETeil22-Tonasync.mp4",
    ]
