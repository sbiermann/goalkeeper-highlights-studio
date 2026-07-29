from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .video import duration_seconds

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".avi"}


def natural_sort_key(path: Path) -> tuple[tuple[int, object], ...]:
    """Return a deterministic recording-order key for camera files.

    Camera filenames are not always perfectly consistent. For example one
    segment can contain a typo in the text prefix while the numeric sequence
    (``Teil1``, ``Teil21``, ``Teil22``) is still authoritative. Therefore the
    last numeric chunk is compared first. Remaining chunks and the complete
    filename only act as deterministic tie breakers.

    This also preserves the expected order for common camera names such as
    ``MVI_0540`` ... ``MVI_0544`` and ``Spiel_1`` ... ``Spiel_10``.
    """
    folded = path.name.casefold()
    sequence_source = path.stem.casefold()
    numbers = [int(value) for value in re.findall(r"\d+", sequence_source)]
    sequence = numbers[-1] if numbers else -1
    parts = re.split(r"(\d+)", folded)
    chunks = tuple((1, int(part)) if part.isdigit() else (0, part) for part in parts if part)
    return ((0, sequence),) + chunks + ((2, folded),)


def discover_video_files(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.casefold() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    videos = [
        item.resolve()
        for item in source.iterdir()
        if item.is_file()
        and item.suffix.casefold() in SUPPORTED_VIDEO_EXTENSIONS
        and "_goalkeeper_highlights" not in item.stem.casefold()
        and item.name.casefold() != "source_timeline.mp4"
    ]
    # Never rely on filesystem enumeration order. Windows Explorer order and
    # Path.iterdir() order are unrelated and may vary between runs.
    videos.sort(key=natural_sort_key)
    if not videos:
        raise ValueError(f"No supported video files found in directory: {source}")
    return videos


@dataclass(slots=True)
class SourceItem:
    name: str
    path: str
    duration_seconds: float
    global_start_seconds: float
    global_end_seconds: float


@dataclass(slots=True)
class SourceManifest:
    source_type: str
    source_path: str
    total_duration_seconds: float
    files: list[SourceItem]
    analysis_source: str = "virtual"

    def as_dict(self) -> dict[str, object]:
        return {
            "source_type": self.source_type,
            "source_path": self.source_path,
            "total_duration_seconds": self.total_duration_seconds,
            "analysis_source": self.analysis_source,
            "files": [asdict(item) for item in self.files],
        }

    def locate(self, global_seconds: float) -> tuple[SourceItem, float]:
        value = max(0.0, min(float(global_seconds), self.total_duration_seconds))
        for item in self.files:
            if value < item.global_end_seconds or item is self.files[-1]:
                return item, max(0.0, value - item.global_start_seconds)
        raise RuntimeError("Virtual source manifest contains no files")


def prepare_source_timeline(
    source: Path,
    output: Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    overwrite: bool = False,
    progress_callback=None,
    source_selection: str = "all",
) -> SourceManifest:
    """Build metadata for a virtual multi-file timeline without concatenating.

    No temporary source_timeline.mp4 is created. Frames and clips are read from
    the original files while all timestamps remain global across the directory.
    """
    del ffmpeg, overwrite  # retained in signature for backwards compatibility
    files = discover_video_files(source)
    # Defensive second sort: callers and future refactorings must not be able to
    # accidentally reintroduce filesystem order before probing/decoding.
    files = sorted(files, key=natural_sort_key)
    if source_selection == "last" and files:
        files = [files[-1]]
    elif source_selection != "all":
        raise ValueError(f"Unsupported source_selection: {source_selection}")
    if progress_callback:
        progress_callback(0.005, "Quellreihenfolge: " + " -> ".join(video.name for video in files))
    items: list[SourceItem] = []
    offset = 0.0
    for index, video in enumerate(files, 1):
        if progress_callback:
            progress_callback(0.01 + 0.03 * (index - 1) / max(1, len(files)), f"Prüfe Quelldateien {index}/{len(files)}")
        duration = duration_seconds(video, ffprobe)
        items.append(SourceItem(
            name=video.name,
            path=str(video),
            duration_seconds=duration,
            global_start_seconds=offset,
            global_end_seconds=offset + duration,
        ))
        offset += duration

    manifest = SourceManifest(
        source_type=("directory-last" if source.is_dir() and source_selection == "last" else "directory" if source.is_dir() else "file"),
        source_path=str(source.resolve()),
        total_duration_seconds=offset,
        files=items,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_manifest.json").write_text(
        json.dumps(manifest.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if progress_callback:
        progress_callback(0.04, f"Virtuelle Timeline bereit: {len(items)} Dateien")
    return manifest
