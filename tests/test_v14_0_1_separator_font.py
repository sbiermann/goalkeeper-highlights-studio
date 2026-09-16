from pathlib import Path
from unittest.mock import patch

import pytest

from goalkeeper_highlights.video import _create_highlight_separator


def test_separator_uses_explicit_font_file(tmp_path: Path):
    font = tmp_path / "arial.ttf"
    font.touch()
    output = tmp_path / "separator.mp4"

    with patch("goalkeeper_highlights.video.run_checked") as run:
        _create_highlight_separator(
            "ffmpeg",
            output,
            2,
            "catch_or_control",
            {"font_file": str(font)},
            {"mode": "accurate"},
            "libx264",
        )

    command = run.call_args.args[0]
    drawtext = command[command.index("-vf") + 1]
    assert "fontfile='" in drawtext
    assert "arial.ttf" in drawtext
    assert "Highlight 2 - Catch / Control" in drawtext
    assert run.call_args.kwargs["capture"] is True


def test_separator_error_includes_ffmpeg_stderr(tmp_path: Path):
    import subprocess

    font = tmp_path / "arial.ttf"
    font.touch()
    error = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="drawtext failed")

    with patch("goalkeeper_highlights.video.run_checked", side_effect=error):
        with pytest.raises(RuntimeError, match="drawtext failed"):
            _create_highlight_separator(
                "ffmpeg",
                tmp_path / "separator.mp4",
                2,
                "diving_save",
                {"font_file": str(font)},
                {"mode": "accurate"},
                "libx264",
            )
