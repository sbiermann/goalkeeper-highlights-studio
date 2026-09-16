from pathlib import Path
from unittest.mock import patch

from goalkeeper_highlights.video import (
    _concat_filter_command,
    categories_from_clip_filenames,
    rebuild_highlights_from_existing_clips,
)


def test_concat_filter_resets_timestamps_and_does_not_stream_copy(tmp_path: Path):
    sequence = [tmp_path / "one.mp4", tmp_path / "separator.mp4", tmp_path / "two.mp4"]
    command = _concat_filter_command(
        "ffmpeg", sequence, tmp_path / "out.mp4", {}, "libx264",
        {"width": 3840, "height": 2160, "frame_rate": "50/1"},
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "setpts=PTS-STARTPTS" in graph
    assert "asetpts=PTS-STARTPTS" in graph
    assert "concat=n=3:v=1:a=1" in graph
    assert "scale=3840:2160" in graph
    assert "fps=50/1" in graph
    assert "settb=AVTB" in graph
    assert "-c" not in command or "copy" not in command


def test_categories_are_recovered_from_existing_clip_names(tmp_path: Path):
    clips = [
        tmp_path / "002_000123.45_catch_or_control.mp4",
        tmp_path / "007_000456.78_diving_save.mp4",
    ]
    assert categories_from_clip_filenames(clips) == ["catch_or_control", "diving_save"]


def test_rebuild_uses_existing_clips_only(tmp_path: Path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    first = clips_dir / "002_000123.45_catch_or_control.mp4"
    second = clips_dir / "007_000456.78_diving_save.mp4"
    first.touch()
    second.touch()

    with patch("goalkeeper_highlights.video.resolve_encoder", return_value="libx264"), patch(
        "goalkeeper_highlights.video.concatenate_highlights"
    ) as concat:
        final = rebuild_highlights_from_existing_clips(
            "ffmpeg",
            tmp_path,
            {"mode": "accurate"},
            {"separator": {"enabled": True}},
        )

    assert final == tmp_path / "goalkeeper_highlights.mp4"
    args = concat.call_args.args
    assert args[1] == [first, second]
    assert args[2] == ["catch_or_control", "diving_save"]
