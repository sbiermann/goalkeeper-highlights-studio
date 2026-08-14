from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# OpenCV reads this during backend initialization. Set it before importing the pipeline.
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = os.environ.get("GOALKEEPER_OPENCV_READ_ATTEMPTS", "65536")

from .benchmark import run_benchmark
from .config import load_config
from .pipeline import run
from .sources import discover_video_files
from . import __version__


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("video", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--frame-stride", type=int)
    p.add_argument("--decoder", choices=["pyav", "opencv"])
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Goalkeeper video analysis and reproducible benchmarks.")
    sub = p.add_subparsers(dest="command")

    analyze = sub.add_parser("analyze", help="Run the complete analysis pipeline")
    _common(analyze)
    analyze.add_argument("--overwrite", action="store_true")
    analyze.add_argument("--qwen", action=argparse.BooleanOptionalAction, default=None)
    analyze.add_argument("--max-candidates", type=int)
    analyze.add_argument("--ffmpeg", default="ffmpeg")
    analyze.add_argument("--ffprobe", default="ffprobe")
    analyze.add_argument("--profiling", action=argparse.BooleanOptionalAction, default=None)
    analyze.add_argument("--profiling-interval", type=float)
    analyze.add_argument("--export-rejected", action=argparse.BooleanOptionalAction, default=None, help="Export rejected clips (default: enabled; use --no-export-rejected to disable)")
    analyze.add_argument("--clip-mode", choices=["accurate", "fast"], help="accurate re-encodes frame-exact clips; fast uses keyframe stream copy")
    analyze.add_argument("--encoder", help="Video encoder: auto, h264_nvenc, libx264, ...")
    analyze.add_argument("--parallel-jobs", type=int, help="Number of parallel FFmpeg clip jobs")
    analyze.add_argument("--verbose", action="store_true", help="Show detailed detector, profiler and FFmpeg output")
    analyze.add_argument("--only-last-source", action="store_true", help="When VIDEO is a directory, analyze only the naturally sorted final source file")

    benchmark = sub.add_parser("benchmark", help="Run short, reproducible performance benchmark without clip export")
    _common(benchmark)
    benchmark.add_argument("--duration", type=float, default=300.0)
    benchmark.add_argument("--start", type=float, default=0.0)
    benchmark.add_argument("--baseline", type=Path, help="Path to baseline benchmark.json for before/after diff")
    benchmark.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False, help="Enable YOLO FP16 for controlled A/B benchmark")
    benchmark.add_argument("--track-path", choices=["legacy", "optimized"], default="legacy", help="Select model.track execution path for overhead A/B")
    benchmark.add_argument("--decoder-mode", choices=["legacy", "prefetch"], default="prefetch", help="Select OpenCV decoder execution mode for A/B benchmark")
    benchmark.add_argument("--decoder-prefetch-queue-size", type=int, default=4, help="Bounded queue size for decoder prefetch mode")
    benchmark.add_argument("--ffmpeg", default="ffmpeg")
    benchmark.add_argument("--ffprobe", default="ffprobe")
    return p


def _legacy_args(argv: list[str]) -> list[str]:
    if argv and argv[0] not in {"analyze", "benchmark", "-h", "--help"}:
        return ["analyze", *argv]
    return argv


class TerminalProgress:
    """Single-line, user-facing progress display without tqdm internals."""

    def __init__(self, width: int = 32) -> None:
        self.width = width
        self.started = time.monotonic()
        self.last_length = 0
        self.closed = False

    @staticmethod
    def _format_seconds(seconds: float | None) -> str:
        if seconds is None or seconds < 0 or seconds == float("inf"):
            return "--:--"
        seconds = int(round(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def update(self, value: float, message: str) -> None:
        if self.closed:
            return
        value = max(0.0, min(1.0, value))
        percent = int(round(value * 100))
        filled = int(round(self.width * value))
        bar = "█" * filled + "░" * (self.width - filled)

        phase = "Analyse"
        details: list[str] = []
        if message.startswith("Prüfe Quelldateien"):
            phase = "Quellen"
            match = re.search(r"(\d+)/(\d+)", message)
            if match:
                details.append(f"{match.group(1)}/{match.group(2)} Dateien")
        elif message.startswith("Virtuelle Timeline"):
            phase = "Timeline"
            details.append(message.split(":", 1)[-1].strip())
        elif message.startswith("Erstelle Clips"):
            phase = "Clips"
            match = re.search(r"(\d+)/(\d+)", message)
            if match:
                details.append(f"{match.group(1)}/{match.group(2)}")
        elif message.startswith("Füge Highlights"):
            phase = "Gesamtvideo"
        elif message.startswith("Klassifiziere"):
            phase = "Auswahl"
        elif message.startswith("Fertig"):
            phase = "Fertig"
        else:
            match = re.search(
                r"(?P<current>[0-9.]+)/(?P<total>[0-9.]+) min, "
                r"(?P<rate>[0-9.]+)x realtime, raw candidates: (?P<candidates>\d+)",
                message,
            )
            if match:
                current = float(match.group("current"))
                total = float(match.group("total"))
                rate = float(match.group("rate"))
                candidates = int(match.group("candidates"))
                remaining_seconds = ((total - current) * 60 / rate) if rate > 0 else None
                details.extend([
                    f"{current:.1f}/{total:.1f} min",
                    f"ETA {self._format_seconds(remaining_seconds)}",
                    f"Kandidaten {candidates}",
                    f"{rate:.2f}x",
                ])
            elif "Torwarterkennung" in message:
                phase = "Torwart"

        text = f"{phase:<11} {bar} {percent:3d}%"
        if details:
            text += " | " + " | ".join(details)
        padding = " " * max(0, self.last_length - len(text))
        print("\r" + text + padding, end="", flush=True)
        self.last_length = len(text)

    def close(self) -> None:
        if not self.closed:
            print()
            self.closed = True


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _print_summary(summary: dict[str, object]) -> None:
    width = 64
    accepted = int(summary.get("accepted", 0))
    rejected = int(summary.get("rejected", 0))
    raw_candidates = int(summary.get("raw_candidates", accepted + rejected))
    merged_candidates = int(summary.get("merged_candidates", max(0, raw_candidates - accepted - rejected)))
    keeper_label = str(summary.get("keeper_label", "Keeper #1"))
    keeper_confidence = float(summary.get("keeper_confidence", 0.0))
    keeper_reids = int(summary.get("keeper_reidentifications", 0))
    realtime = float(summary.get("realtime_factor", 0.0))

    print("\n" + "=" * width)
    print("Analyse abgeschlossen")
    print("=" * width)

    print("\nVideo")
    print(f"  {summary.get('video_name', '–')}")
    source_count = int(summary.get("source_file_count", 1))
    if source_count > 1:
        print(f"  Quelldateien:        {source_count}")

    print("\nErgebnis")
    print(f"  Kandidaten:          {raw_candidates}")
    print(f"  Highlights:          {accepted}")
    print(f"  Verworfen:           {rejected}")
    print(f"  Zusammengeführt:     {merged_candidates}")

    print("\nTorwart")
    print(f"  Identität:           {keeper_label}")
    print(f"  Konfidenz:           {keeper_confidence * 100:.0f} %")
    print(f"  Re-Identifikationen: {keeper_reids}")

    print("\nLeistung")
    print(f"  Analysezeit:         {_format_duration(float(summary['analysis_seconds']))}")
    print(f"  Geschwindigkeit:     {realtime:.2f}× Echtzeit")
    print(f"  Gesamtzeit:          {_format_duration(float(summary['total_seconds']))}")

    recoveries = int(summary.get("decoder_read_recoveries", 0))
    if recoveries:
        print("\nHinweise")
        print(f"  Decoder-Neustarts:   {recoveries}")
        print("  Die Analyse wurde nach Lesefehlern automatisch fortgesetzt.")

    print("\nFFmpeg")
    print(f"  Clip-Erstellung:     {_format_duration(float(summary['clip_creation_seconds']))}")
    print(f"  Zusammenfügen:       {_format_duration(float(summary['concat_seconds']))}")
    print(f"  Encoder:             {summary['encoder']}")
    print(f"  Clip-Modus:          {summary['clip_mode']}")
    print(f"  Parallele Jobs:      {summary['parallel_jobs']}")

    print("\nErstellt")
    print("  ✓ Einzelclips")
    print("  ✓ Rejected-Clips")
    print("  ✓ Gesamtvideo")
    print("  ✓ HTML-Report")
    print("  ✓ CSV/JSON-Auswertung")
    print("  ✓ Analyse-Datenbank")

    print("\nAusgabeverzeichnis")
    print(f"  {summary['output']}")
    print("=" * width)


def main() -> int:
    args = parser().parse_args(_legacy_args(sys.argv[1:]))
    if not args.command:
        parser().print_help()
        return 2
    video = args.video.expanduser().resolve()
    if not video.exists():
        print(f"Source not found: {video}", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    if args.frame_stride is not None:
        cfg["yolo"]["frame_stride"] = max(1, args.frame_stride)
    if args.decoder:
        cfg.setdefault("decoder", {})["backend"] = args.decoder
    original_video = video
    if args.command == "analyze" and getattr(args, "only_last_source", False):
        if not video.is_dir():
            print("--only-last-source requires a directory source.", file=sys.stderr)
            return 2
        ordered_sources = discover_video_files(video)
        video = ordered_sources[-1]
        print(f"Nur letzte Quelldatei: {video.name}")

    if args.output:
        output = args.output.expanduser().resolve()
    elif args.command == "benchmark":
        output = video.with_name(video.stem + "_goalkeeper_benchmark")
    elif getattr(args, "only_last_source", False):
        output = original_video.with_name(original_video.stem + "_last_source_goalkeeper_highlights")
    else:
        output = video.with_name(video.stem + "_goalkeeper_highlights")
    try:
        if args.command == "benchmark":
            report = run_benchmark(
                video,
                output,
                cfg,
                duration_seconds=max(1.0, float(args.duration)),
                start_seconds=max(0.0, float(args.start)),
                baseline_path=args.baseline.expanduser().resolve() if args.baseline else None,
                fp16=bool(args.fp16),
                track_path=str(args.track_path),
                decoder_mode=str(args.decoder_mode),
                decoder_prefetch_queue_size=max(1, int(args.decoder_prefetch_queue_size)),
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
            )
            print(f"Benchmark written to: {output / 'benchmark'}")
            print(f"  Analysezeit: {report['analysis_seconds']:.2f}s")
            print(f"  FPS: {report['processed_fps']:.2f}")
            print(f"  Echtzeitfaktor: {report['realtime_factor']:.2f}x")
            print(f"  Candidates/Accepted/Rejected: {report['candidates']}/{report['accepted']}/{report['rejected']}")
            print(f"  Keeper: {report['keeper']}")
            diff = report.get("baseline_diff")
            if isinstance(diff, dict):
                print(f"  Diff zur Baseline: {diff.get('improvement_percent', 0.0):.2f}%")
        else:
            if args.qwen is not None:
                cfg["qwen"]["enabled"] = args.qwen
            if args.max_candidates is not None:
                cfg["clips"]["max_candidates"] = max(0, args.max_candidates)
            if args.profiling is not None:
                cfg.setdefault("profiling", {})["enabled"] = args.profiling
            if args.profiling_interval is not None:
                cfg.setdefault("profiling", {})["sample_interval_seconds"] = max(0.5, args.profiling_interval)
            if args.export_rejected is not None:
                cfg.setdefault("runtime", {})["export_rejected"] = args.export_rejected
            if args.clip_mode is not None:
                cfg.setdefault("clips", {})["mode"] = args.clip_mode
            if args.encoder is not None:
                cfg.setdefault("clips", {})["encoder"] = args.encoder
            if args.parallel_jobs is not None:
                cfg.setdefault("clips", {})["parallel_jobs"] = max(1, args.parallel_jobs)
            cfg.setdefault("runtime", {})["verbose_console"] = bool(args.verbose)

            if video.is_dir():
                ordered_sources = discover_video_files(video)
                print("\nQuellreihenfolge")
                for index, source_file in enumerate(ordered_sources, 1):
                    print(f"  {index}. {source_file.name}")
                print()

            terminal_progress = None if args.verbose else TerminalProgress()
            progress = terminal_progress.update if terminal_progress else None
            try:
                summary = run(video, output, cfg, args.overwrite, args.ffmpeg, args.ffprobe, progress_callback=progress)
            finally:
                if terminal_progress is not None:
                    terminal_progress.close()
            _print_summary(summary)
        return 0
    except KeyboardInterrupt:
        print("\nAbgebrochen.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
