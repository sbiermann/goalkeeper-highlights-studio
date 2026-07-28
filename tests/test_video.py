from pathlib import Path
from unittest.mock import patch

from goalkeeper_highlights.video import cut_clip, resolve_encoder


def test_accurate_cut_uses_input_seeking(tmp_path: Path):
    source = tmp_path / "in.mp4"
    output = tmp_path / "out.mp4"
    source.touch()
    cfg = {"mode": "accurate", "encoder": "libx264", "preset": "fast", "crf": 20}
    with patch("goalkeeper_highlights.video.run_checked") as run:
        cut_clip("ffmpeg", source, output, 100.0, 112.0, cfg, "libx264")
    command = run.call_args.args[0]
    assert command.index("-ss") < command.index("-i")


def test_fast_cut_uses_stream_copy(tmp_path: Path):
    source = tmp_path / "in.mp4"
    output = tmp_path / "out.mp4"
    source.touch()
    with patch("goalkeeper_highlights.video.run_checked") as run:
        cut_clip("ffmpeg", source, output, 10.0, 20.0, {"mode": "fast"})
    command = run.call_args.args[0]
    assert command[command.index("-c") + 1] == "copy"


def test_auto_encoder_prefers_nvenc():
    with patch("goalkeeper_highlights.video.available_encoders", return_value={"libx264", "h264_nvenc"}):
        assert resolve_encoder("ffmpeg", {"encoder": "auto"}) == "h264_nvenc"
