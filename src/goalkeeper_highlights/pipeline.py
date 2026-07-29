from __future__ import annotations

import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .classification import classify
from .detection import clamp_clip_windows_to_sources, detect
from .diagnostics import create_debug_package
from .profiling import PerformanceProfiler
from .reporting import write_reports
from .store import AnalysisStore
from .sources import prepare_source_timeline
from .video import concatenate, cut_virtual_clip, require_tool, resolve_encoder, set_verbose

ProgressCallback = Callable[[float, str], None]


def run(source: Path, output: Path, config: dict, overwrite: bool, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", progress_callback: ProgressCallback | None = None) -> dict[str, object]:
    verbose = bool(config.get("runtime", {}).get("verbose_console", False))
    set_verbose(verbose)
    require_tool(ffmpeg)
    require_tool(ffprobe)
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not overwrite and not config["runtime"]["resume"]:
        raise RuntimeError(f"Output directory is not empty: {output}")

    store = AnalysisStore(output / "analysis.sqlite3")
    profiling_cfg = config.get("profiling", {})
    profiler = PerformanceProfiler(
        output,
        enabled=bool(profiling_cfg.get("enabled", True)),
        sample_interval_seconds=float(profiling_cfg.get("sample_interval_seconds", 5.0)),
        store=store,
    )
    timings: dict[str, float | str | int] = {}
    total_started = time.perf_counter()
    try:
        source_manifest = prepare_source_timeline(source, output, ffmpeg, ffprobe, overwrite, progress_callback)
        analysis_video = source_manifest
        duration = source_manifest.total_duration_seconds
        if verbose:
            print(f"Source: {source}\nFiles: {len(source_manifest.files)}\nDuration: {duration/60:.1f} minutes")
        detection_started = time.perf_counter()
        candidates = store.load_candidates() if config["runtime"]["resume"] and store.get_state("detection_complete", False) else []
        if candidates:
            if verbose:
                print(f"Resume: using {len(candidates)} cached candidates.")
        else:
            detection_progress = None
            if progress_callback:
                detection_progress = lambda value, message: progress_callback(0.04 + 0.91 * max(0.0, min(1.0, value)), message)
            candidates = detect(analysis_video, duration, config, store=store, progress_callback=detection_progress, profiler=profiler)
            store.replace_candidates(candidates)
            store.set_state("detection_complete", True)
        timings["analysis_seconds"] = time.perf_counter() - detection_started
        timings["video_duration_seconds"] = duration
        timings["video_name"] = source.name
        timings["source_type"] = source_manifest.source_type
        timings["source_file_count"] = len(source_manifest.files)
        analysis_seconds = float(timings["analysis_seconds"])
        timings["realtime_factor"] = duration / analysis_seconds if analysis_seconds > 0 else 0.0

        if progress_callback:
            progress_callback(0.96, "Klassifiziere Kandidaten")
        
        classification_started = time.perf_counter()
        classification_stats = classify(analysis_video, candidates, config)
        timings["classification_seconds"] = time.perf_counter() - classification_started
        timings.update(classification_stats)
        
        clips_cfg = config["clips"]
        clamp_clip_windows_to_sources(candidates, source_manifest, clips_cfg)
        store.replace_candidates(candidates)

        encoder = resolve_encoder(ffmpeg, clips_cfg) if str(clips_cfg.get("mode", "accurate")) != "fast" else "stream_copy"
        timings["encoder"] = encoder
        timings["clip_mode"] = str(clips_cfg.get("mode", "accurate"))
        parallel_jobs = max(1, int(clips_cfg.get("parallel_jobs", 1)))
        timings["parallel_jobs"] = parallel_jobs
        clips_dir = output / "clips"
        rejected_dir = output / "rejected"
        export_rejected = bool(config.get("runtime", {}).get("export_rejected", True))

        jobs: list[tuple[Path, object]] = []
        accepted: list[Path] = []
        for index, candidate in enumerate(candidates, 1):
            category = re.sub(r"[^a-zA-Z0-9_-]+", "_", candidate.category)[:30]
            if candidate.accepted:
                clip = clips_dir / f"{index:03d}_{candidate.trigger_time:09.2f}_{category}.mp4"
                candidate.clip_path = str(clip)
                accepted.append(clip)
                jobs.append((clip, candidate))
            elif export_rejected and not candidate.continuation_absorbed:
                clip = rejected_dir / f"{index:03d}_{candidate.trigger_time:09.2f}_{category}_score{candidate.event_score:.3f}.mp4"
                candidate.clip_path = str(clip)
                jobs.append((clip, candidate))
                reason = clip.with_suffix(".json")
                reason.parent.mkdir(parents=True, exist_ok=True)
                reason.write_text(json.dumps(candidate.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        clip_started = time.perf_counter()
        def create(job: tuple[Path, object]) -> Path:
            clip, candidate = job
            if not clip.exists() or overwrite:
                cut_virtual_clip(ffmpeg, source_manifest, clip, candidate.start, candidate.end, clips_cfg, None if encoder == "stream_copy" else encoder)
            return clip

        if jobs:
            with ThreadPoolExecutor(max_workers=parallel_jobs, thread_name_prefix="ffmpeg-clip") as executor:
                futures = [executor.submit(create, job) for job in jobs]
                completed_jobs = 0
                for future in as_completed(futures):
                    future.result()
                    completed_jobs += 1
                    if progress_callback:
                        progress_callback(0.96 + 0.025 * completed_jobs / max(1, len(futures)), f"Erstelle Clips {completed_jobs}/{len(futures)}")
        timings["clip_creation_seconds"] = time.perf_counter() - clip_started

        final = output / "goalkeeper_highlights.mp4"
        concat_started = time.perf_counter()
        if accepted and (not final.exists() or overwrite):
            if progress_callback:
                progress_callback(0.99, "Füge Highlights zusammen")
            concatenate(ffmpeg, accepted, final, output, clips_cfg, None if encoder == "stream_copy" else encoder)
        timings["concat_seconds"] = time.perf_counter() - concat_started
        timings["accepted"] = len(accepted)
        timings["rejected"] = len(candidates) - len(accepted)
        
        # Routing stats
        timings["routing_high"] = sum(1 for c in candidates if c.routing_category == "HIGH")
        timings["routing_medium"] = sum(1 for c in candidates if c.routing_category == "MEDIUM")
        timings["routing_low"] = sum(1 for c in candidates if c.routing_category == "LOW")
        timings["directly_accepted"] = sum(1 for c in candidates if c.routing_category == "HIGH" and c.accepted)
        timings["early_rejected"] = sum(1 for c in candidates if c.routing_category == "LOW" and not c.accepted)
        timings["qwen_first_pass"] = sum(1 for c in candidates if c.qwen_first_pass_called)
        timings["qwen_second_pass"] = sum(1 for c in candidates if c.qwen_second_pass_called)
        timings["qwen_saved"] = sum(1 for c in candidates if c.qwen_second_pass_rescued)
        timings["recovery_candidates"] = sum(1 for c in candidates if c.recovery_candidate)
        
        # Calculate saved calls
        total_candidates = len(candidates)
        if total_candidates > 0:
            saved_calls = timings["routing_high"] + timings["routing_low"]
            timings["qwen_calls_saved"] = saved_calls
        else:
            timings["qwen_calls_saved"] = 0

        timings["total_seconds"] = time.perf_counter() - total_started

        keeper_detection = store.get_state("keeper_detection", {})
        detection_stats = store.get_state("detection_stats", {})
        decoder_stats = store.get_state("decoder_stats", {})
        if isinstance(decoder_stats, dict):
            timings["decoder_read_recoveries"] = int(decoder_stats.get("read_recoveries", 0))
        if isinstance(detection_stats, dict):
            timings["raw_candidates"] = int(detection_stats.get("raw_candidates", len(candidates)))
            timings["final_candidates"] = int(detection_stats.get("final_candidates", len(candidates)))
            timings["merged_candidates"] = int(detection_stats.get("merged_candidates", 0))
        else:
            timings["raw_candidates"] = len(candidates)
            timings["final_candidates"] = len(candidates)
            timings["merged_candidates"] = 0
        if isinstance(keeper_detection, dict):
            timings["keeper_label"] = str(keeper_detection.get("keeper_label", "Keeper #1"))
            timings["keeper_confidence"] = float(keeper_detection.get("stabilized_confidence", keeper_detection.get("confidence", 0.0)))
            timings["keeper_reidentifications"] = int(keeper_detection.get("reidentification_count", 0))
        write_reports(output, candidates, timings, keeper_detection if isinstance(keeper_detection, dict) else {})
        if bool(config.get("diagnostics", {}).get("enabled", True)):
            debug_archive = create_debug_package(
                output, candidates, timings,
                keeper_detection if isinstance(keeper_detection, dict) else {},
                config, store,
            )
            timings["debug_package"] = str(debug_archive)
        if progress_callback:
            progress_callback(1.0, f"Fertig: {len(accepted)} Szenen")
        if verbose:
            print(
                f"Finished: {len(accepted)} accepted scenes. Output: {output}\n"
                f"FFmpeg: {timings['clip_creation_seconds']:.1f}s clips, {timings['concat_seconds']:.1f}s concat, "
                f"encoder={encoder}, mode={timings['clip_mode']}, jobs={parallel_jobs}"
            )
        return {"output": str(output), **timings}
    finally:
        profiler.finish({"video": str(source), "output": str(output), **timings})
        store.close()
