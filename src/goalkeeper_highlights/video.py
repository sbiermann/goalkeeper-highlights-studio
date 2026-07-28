from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_VERBOSE = False


def set_verbose(enabled: bool) -> None:
    global _VERBOSE
    _VERBOSE = bool(enabled)


def require_tool(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"Required executable not found in PATH: {name}")


def run_checked(command: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    if _VERBOSE:
        print("$", subprocess.list2cmdline(command))
    # Quiet mode keeps FFmpeg/FFprobe output away from the progress bar.  Output
    # is still captured so callers receive useful diagnostics on failure.
    quiet_capture = capture or not _VERBOSE
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if quiet_capture else None,
        stderr=subprocess.PIPE if quiet_capture else None,
    )


def duration_seconds(video: Path, ffprobe: str = "ffprobe") -> float:
    result = run_checked(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture=True,
    )
    return float(result.stdout.strip())


def available_encoders(ffmpeg: str) -> set[str]:
    try:
        result = run_checked([ffmpeg, "-hide_banner", "-encoders"], capture=True)
    except (OSError, subprocess.CalledProcessError):
        return set()
    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6:
            encoders.add(parts[1])
    return encoders


def resolve_encoder(ffmpeg: str, cfg: dict) -> str:
    requested = str(cfg.get("encoder", "auto")).lower()
    if requested != "auto":
        return requested
    encoders = available_encoders(ffmpeg)
    if "h264_nvenc" in encoders:
        return "h264_nvenc"
    return "libx264"


def _video_encode_args(encoder: str, cfg: dict) -> list[str]:
    if encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", str(cfg.get("nvenc_preset", "p4")),
            "-tune", str(cfg.get("nvenc_tune", "hq")),
            "-rc", "vbr",
            "-cq", str(cfg.get("cq", 20)),
            "-b:v", "0",
        ]
    return [
        "-c:v", encoder,
        "-preset", str(cfg.get("preset", "fast")),
        "-crf", str(cfg.get("crf", 20)),
    ]


def cut_clip(ffmpeg: str, source: Path, output: Path, start: float, end: float, cfg: dict, encoder: str | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    mode = str(cfg.get("mode", "accurate")).lower()
    common = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]

    if mode == "fast":
        # Keyframe-accurate stream copy. Extremely fast and useful for review clips.
        command = common + [
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
            "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(output),
        ]
    else:
        selected = encoder or resolve_encoder(ffmpeg, cfg)
        # Input seeking is much faster than placing -ss after -i, while re-encoding
        # keeps the requested clip start frame-accurate.
        command = common + [
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a?",
            *_video_encode_args(selected, cfg),
            "-c:a", "aac", "-b:a", str(cfg.get("audio_bitrate", "160k")),
            "-movflags", "+faststart", str(output),
        ]
    run_checked(command)


def concatenate(ffmpeg: str, clips: list[Path], output: Path, work_dir: Path, cfg: dict | None = None, encoder: str | None = None) -> None:
    if not clips:
        return
    concat_file = work_dir / "concat.txt"
    with concat_file.open("w", encoding="utf-8") as handle:
        for clip in clips:
            escaped = str(clip.resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
    cfg = cfg or {}

    # All clips are produced with identical parameters, so stream copy avoids a
    # complete second encoding pass. Fall back to re-encoding only if a source
    # file is incompatible.
    copy_command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", str(output),
    ]
    try:
        run_checked(copy_command)
        return
    except subprocess.CalledProcessError:
        if _VERBOSE:
            print("Stream-copy concat failed; falling back to re-encoding.")

    selected = encoder or resolve_encoder(ffmpeg, cfg)
    run_checked([
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), *_video_encode_args(selected, cfg),
        "-c:a", "aac", "-b:a", str(cfg.get("audio_bitrate", "160k")),
        "-movflags", "+faststart", str(output),
    ])


def concatenate_sources(ffmpeg: str, sources: list[Path], output: Path, work_dir: Path) -> None:
    """Build a lossless logical timeline from sequential camera files.

    Camera-generated segment files normally share identical stream parameters,
    so concat stream-copy is fast and avoids quality loss. A clear error is
    raised when the files are incompatible instead of silently re-encoding a
    potentially very large source timeline.
    """
    if not sources:
        raise ValueError("No source videos supplied")
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_file = work_dir / "source_concat.txt"
    with concat_file.open("w", encoding="utf-8") as handle:
        for source in sources:
            escaped = str(source.resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
    try:
        run_checked([
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", str(output),
        ])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "The video files could not be joined losslessly. Ensure all files "
            "come from the same camera and use identical resolution, codecs and "
            "frame rate. Run with --verbose for FFmpeg details."
        ) from exc


def cut_virtual_clip(ffmpeg: str, manifest, output: Path, start: float, end: float, cfg: dict, encoder: str | None = None) -> None:
    """Cut one global interval from one or more original source files."""
    overlapping = []
    for item in manifest.files:
        part_start = max(start, item.global_start_seconds)
        part_end = min(end, item.global_end_seconds)
        if part_end > part_start:
            overlapping.append((item, part_start - item.global_start_seconds, part_end - item.global_start_seconds))
    if not overlapping:
        raise ValueError(f"Clip interval outside virtual timeline: {start:.3f}-{end:.3f}")
    if len(overlapping) == 1:
        item, local_start, local_end = overlapping[0]
        cut_clip(ffmpeg, Path(item.path), output, local_start, local_end, cfg, encoder)
        return
    work = output.parent / f".{output.stem}_parts"
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    try:
        for index, (item, local_start, local_end) in enumerate(overlapping, 1):
            part = work / f"part_{index:03d}.mp4"
            cut_clip(ffmpeg, Path(item.path), part, local_start, local_end, cfg, encoder)
            parts.append(part)
        concatenate(ffmpeg, parts, output, work, cfg, encoder)
    finally:
        shutil.rmtree(work, ignore_errors=True)
