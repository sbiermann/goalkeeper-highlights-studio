from pathlib import Path
from unittest.mock import patch

import pytest

from goalkeeper_highlights.video import (
    _highlight_category_title,
    concatenate_highlights,
)


def test_category_titles_are_human_readable():
    assert _highlight_category_title("diving_save") == "Diving Save"
    assert _highlight_category_title("catch_or_control") == "Catch / Control"


def test_disabled_separator_uses_plain_concat(tmp_path: Path):
    clips = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    with patch("goalkeeper_highlights.video.concatenate") as concat:
        concatenate_highlights("ffmpeg", clips, ["diving_save", "distribution"], tmp_path / "out.mp4", tmp_path, {"mode": "accurate"}, {"separator": {"enabled": False}}, "libx264")
    concat.assert_called_once()


def test_fast_mode_rejects_enabled_separators(tmp_path: Path):
    clips = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    with pytest.raises(RuntimeError, match="clips.mode=accurate"):
        concatenate_highlights("ffmpeg", clips, ["diving_save", "distribution"], tmp_path / "out.mp4", tmp_path, {"mode": "fast"}, {"separator": {"enabled": True}}, None)


def test_separator_is_only_inserted_between_clips_and_cleaned_up(tmp_path: Path):
    clips = [tmp_path / "one.mp4", tmp_path / "two.mp4", tmp_path / "three.mp4"]
    for clip in clips:
        clip.touch()
    with patch(
        "goalkeeper_highlights.video._probe_complete_video_profile",
        return_value={"width": 1920, "height": 1080, "frame_rate": "30/1"},
    ), patch("goalkeeper_highlights.video._create_highlight_separator") as create_separator, patch(
        "goalkeeper_highlights.video._concat_filter_command", return_value=["ffmpeg"]
    ) as build_command, patch("goalkeeper_highlights.video.run_checked"):
        concatenate_highlights("ffmpeg", clips, ["catch_or_control", "diving_save", "distribution"], tmp_path / "out.mp4", tmp_path, {"mode": "accurate"}, {"separator": {"enabled": True}}, "libx264")
    assert create_separator.call_count == 2
    sequence = build_command.call_args.args[1]
    assert sequence[0] == clips[0]
    assert sequence[2] == clips[1]
    assert sequence[4] == clips[2]
    assert len(sequence) == 5
    assert not (tmp_path / ".highlight_separators").exists()
