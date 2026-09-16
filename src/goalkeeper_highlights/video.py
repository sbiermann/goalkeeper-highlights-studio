from __future__ import annotations

import os
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


def run_checked(
    command: list[str],
    capture: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
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
        timeout=timeout,
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
    timeout_seconds = max(10.0, float(cfg.get("clip_export_timeout_seconds", 90.0)))
    try:
        run_checked(command, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        output.unlink(missing_ok=True)
        if mode == "fast" or selected == "libx264":
            raise
        fallback_command = common + [
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a?",
            *_video_encode_args("libx264", cfg),
            "-c:a", "aac", "-b:a", str(cfg.get("audio_bitrate", "160k")),
            "-movflags", "+faststart", str(output),
        ]
        if _VERBOSE:
            print(
                f"Clip export timed out after {timeout_seconds:.0f}s with {selected}; "
                "retrying once with libx264."
            )
        try:
            run_checked(fallback_command, timeout=timeout_seconds)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            output.unlink(missing_ok=True)
            raise


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



def _highlight_category_title(category: str) -> str:
    """Return a compact, human-readable title for an internal category."""
    titles = {
        "catch_or_control": "Catch / Control",
        "cross_claim_or_high_catch": "Cross Claim / High Catch",
        "distribution": "Distribution",
        "keeper_clearance": "Keeper Clearance",
        "ball_contact": "Ball Contact",
        "save_or_deflection": "Save / Deflection",
        "punch_clearance": "Punch Clearance",
        "sweep_or_one_on_one": "Sweep / One-on-One",
        "diving_save": "Diving Save",
        "interaction": "Interaction",
        "unclassified": "Unclassified",
    }
    key = str(category or "unclassified").strip().lower()
    return titles.get(key, key.replace("_", " " ).replace("-", " " ).title())


def _drawtext_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _resolve_separator_font(separator_cfg: dict) -> Path:
    configured = str(separator_cfg.get("font_file", "")).strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())

    windows_dir = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates.extend([
        windows_dir / "Fonts" / "arial.ttf",
        windows_dir / "Fonts" / "segoeui.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if configured:
        raise RuntimeError(f"Configured highlight separator font does not exist: {configured}")
    raise RuntimeError(
        "No usable font file found for highlight separators. Set "
        "highlights_complete.separator.font_file to a TrueType font, for example "
        "C:/Windows/Fonts/arial.ttf on Windows."
    )


def _drawtext_fontfile_escape(path: Path) -> str:
    # FFmpeg filter syntax treats ':' as a separator even in Windows paths.
    return _drawtext_escape(path.resolve().as_posix())


def _probe_complete_video_profile(ffprobe: str, clip: Path) -> dict[str, object]:
    result = run_checked([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate",
        "-of", "json", str(clip),
    ], capture=True)
    import json
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in highlight clip: {clip}")
    stream = streams[0]
    width = int(stream.get("width") or 1920)
    height = int(stream.get("height") or 1080)
    frame_rate = str(stream.get("avg_frame_rate") or "30/1")
    if frame_rate in {"0/0", "N/A", ""}:
        frame_rate = "30/1"
    return {"width": width, "height": height, "frame_rate": frame_rate}


def _create_highlight_separator(
    ffmpeg: str,
    output: Path,
    number: int,
    category: str,
    separator_cfg: dict,
    clips_cfg: dict,
    encoder: str | None,
    video_profile: dict[str, object] | None = None,
) -> None:
    duration = max(0.1, float(separator_cfg.get("duration_seconds", 1.0)))
    font_size = max(12, int(separator_cfg.get("font_size", 72)))
    show_number = bool(separator_cfg.get("show_clip_number", True))
    show_category = bool(separator_cfg.get("show_category", True))
    parts = []
    if show_number:
        parts.append(f"Highlight {number}")
    if show_category:
        parts.append(_highlight_category_title(category))
    title = " - ".join(parts) or f"Highlight {number}"
    escaped_title = _drawtext_escape(title)
    font_file = _resolve_separator_font(separator_cfg)
    escaped_font_file = _drawtext_fontfile_escape(font_file)
    selected = encoder or resolve_encoder(ffmpeg, clips_cfg)
    profile = video_profile or {}
    width = max(16, int(profile.get("width", 1920)))
    height = max(16, int(profile.get("height", 1080)))
    frame_rate = str(profile.get("frame_rate", "30/1"))
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={frame_rate}:d={duration:.3f}",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf",
        (
            f"drawtext=fontfile='{escaped_font_file}':text='{escaped_title}':"
            f"fontcolor=white:fontsize={font_size}:x=(w-text_w)/2:y=(h-text_h)/2"
        ),
        "-t", f"{duration:.3f}", *_video_encode_args(selected, clips_cfg),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", str(clips_cfg.get("audio_bitrate", "160k")),
        "-shortest", "-movflags", "+faststart", str(output),
    ]
    try:
        run_checked(command, capture=True)
    except subprocess.CalledProcessError as exc:
        output.unlink(missing_ok=True)
        details = (exc.stderr or exc.stdout or "").strip()
        message = f"FFmpeg failed while creating highlight separator {number}."
        if details:
            message += f"\nFFmpeg output:\n{details}"
        raise RuntimeError(message) from exc


def _concat_filter_command(
    ffmpeg: str,
    sequence: list[Path],
    output: Path,
    cfg: dict,
    encoder: str | None,
    video_profile: dict[str, object] | None = None,
) -> list[str]:
    """Build a timestamp-safe concat-filter command for the complete video.

    Every segment is decoded and gets fresh video/audio timestamps before the
    concat filter joins them. This intentionally re-encodes only the final
    complete video; the existing individual highlight clips are never touched.
    """
    selected = encoder or resolve_encoder(ffmpeg, cfg)
    profile = video_profile or {}
    width = max(16, int(profile.get("width", 1920)))
    height = max(16, int(profile.get("height", 1080)))
    frame_rate = str(profile.get("frame_rate", "30/1"))
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for item in sequence:
        command.extend(["-i", str(item)])

    filters: list[str] = []
    concat_inputs: list[str] = []
    for index in range(len(sequence)):
        filters.append(
            f"[{index}:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={frame_rate},settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[v{index}]"
        )
        filters.append(
            f"[{index}:a:0]aresample=48000:async=1:first_pts=0,"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
        concat_inputs.extend([f"[v{index}]", f"[a{index}]"])
    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(sequence)}:v=1:a=1[outv][outa]"
    )

    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[outv]", "-map", "[outa]",
        *_video_encode_args(selected, cfg),
        "-c:a", "aac", "-b:a", str(cfg.get("audio_bitrate", "160k")),
        "-movflags", "+faststart", str(output),
    ])
    return command


def concatenate_highlights(
    ffmpeg: str,
    clips: list[Path],
    categories: list[str],
    output: Path,
    work_dir: Path,
    cfg: dict,
    complete_cfg: dict | None = None,
    encoder: str | None = None,
    ffprobe: str = "ffprobe",
) -> None:
    """Build the complete highlight video from existing accepted clips.

    V14.0.2 deliberately avoids concat stream-copy when separators are enabled.
    Mixing generated title cards with camera-derived clips can preserve
    incompatible timestamps/time-bases and cause frozen/fast-forward video while
    audio keeps playing. The final complete video is therefore encoded once via
    the concat filter after resetting timestamps for every segment. The source
    highlight clips themselves are left untouched.
    """
    if not clips:
        return
    separator_cfg = (complete_cfg or {}).get("separator", {})
    if not bool(separator_cfg.get("enabled", True)) or len(clips) < 2:
        concatenate(ffmpeg, clips, output, work_dir, cfg, encoder)
        return
    if str(cfg.get("mode", "accurate")).lower() == "fast":
        raise RuntimeError(
            "Highlight separators require clips.mode=accurate because generated title cards "
            "cannot be safely stream-copied between arbitrary source codecs."
        )
    video_profile = _probe_complete_video_profile(ffprobe, clips[0])
    separator_dir = work_dir / ".highlight_separators"
    separator_dir.mkdir(parents=True, exist_ok=True)
    sequence: list[Path] = []
    try:
        for index, clip in enumerate(clips):
            if index:
                separator = separator_dir / f"separator_{index:03d}.mp4"
                category = categories[index] if index < len(categories) else "unclassified"
                _create_highlight_separator(
                    ffmpeg, separator, index + 1, category, separator_cfg, cfg, encoder, video_profile
                )
                sequence.append(separator)
            sequence.append(clip)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = _concat_filter_command(ffmpeg, sequence, output, cfg, encoder, video_profile)
        try:
            run_checked(command, capture=True)
        except subprocess.CalledProcessError as exc:
            output.unlink(missing_ok=True)
            details = (exc.stderr or exc.stdout or "").strip()
            message = "FFmpeg failed while building the complete highlight video."
            if details:
                message += f"\nFFmpeg output:\n{details}"
            raise RuntimeError(message) from exc
    finally:
        shutil.rmtree(separator_dir, ignore_errors=True)


def categories_from_clip_filenames(clips: list[Path]) -> list[str]:
    """Recover the exported category from standard clip filenames."""
    categories: list[str] = []
    for clip in clips:
        parts = clip.stem.split("_", 2)
        categories.append(parts[2] if len(parts) == 3 and parts[2] else "unclassified")
    return categories


def rebuild_highlights_from_existing_clips(
    ffmpeg: str,
    output_dir: Path,
    cfg: dict,
    complete_cfg: dict | None = None,
    ffprobe: str = "ffprobe",
) -> Path:
    """Rebuild only goalkeeper_highlights.mp4 from already exported clips.

    No detection, classification, source decoding or clip extraction is run.
    """
    clips_dir = output_dir / "clips"
    clips = sorted(clips_dir.glob("*.mp4")) if clips_dir.is_dir() else []
    if not clips:
        raise RuntimeError(f"No existing highlight clips found in: {clips_dir}")
    if str(cfg.get("mode", "accurate")).lower() == "fast" and bool((complete_cfg or {}).get("separator", {}).get("enabled", True)):
        raise RuntimeError(
            "Highlight separators require clips.mode=accurate. Rebuild with --clip-mode accurate."
        )
    encoder = resolve_encoder(ffmpeg, cfg)
    final = output_dir / "goalkeeper_highlights.mp4"
    categories = categories_from_clip_filenames(clips)
    concatenate_highlights(
        ffmpeg, clips, categories, final, output_dir, cfg, complete_cfg, encoder, ffprobe
    )
    return final

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
