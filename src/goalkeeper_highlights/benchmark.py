from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from .pipeline import run


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _benchmark_config(config: dict[str, Any], *, start_seconds: float, duration_seconds: float, fp16: bool) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    runtime = cfg.setdefault("runtime", {})
    runtime["benchmark_mode"] = True
    runtime["benchmark_start_seconds"] = max(0.0, float(start_seconds))
    runtime["benchmark_duration_seconds"] = max(1.0, float(duration_seconds))
    runtime["export_rejected"] = False
    runtime["verbose_console"] = False
    cfg.setdefault("profiling", {})["enabled"] = True
    cfg.setdefault("diagnostics", {})["enabled"] = False
    cfg.setdefault("qwen", {})["enabled"] = False
    cfg.setdefault("yolo", {})["half"] = bool(fp16)
    return cfg


def _stage_averages(summary: dict[str, Any]) -> dict[str, float]:
    values = summary.get("stage_averages_ms", {})
    if not isinstance(values, dict):
        return {}
    return {str(k): _as_float(v) for k, v in values.items()}


def _metrics(summary: dict[str, Any], *, start: float, duration: float, fp16: bool) -> dict[str, Any]:
    analysis_seconds = _as_float(summary.get("analysis_seconds"))
    processed_frames = int(summary.get("processed_frames", 0) or 0)
    if processed_frames <= 0:
        source_rows = summary.get("source_performance", [])
        if isinstance(source_rows, list):
            processed_frames = sum(int(row.get("processed_frames", 0) or 0) for row in source_rows if isinstance(row, dict))
    stage = _stage_averages(summary)
    video_seconds = min(duration, max(0.0, _as_float(summary.get("video_duration_seconds")) - start))
    fps = processed_frames / max(analysis_seconds, 1e-6)
    realtime = video_seconds / max(analysis_seconds, 1e-6)
    return {
        "version": str(summary.get("version", "")),
        "start_seconds": round(start, 3),
        "duration_seconds": round(duration, 3),
        "fp16": bool(fp16),
        "analysis_seconds": round(analysis_seconds, 3),
        "video_seconds": round(video_seconds, 3),
        "processed_frames": processed_frames,
        "processed_fps": round(fps, 3),
        "realtime_factor": round(realtime, 3),
        "candidates": int(summary.get("final_candidates", 0) or 0),
        "accepted": int(summary.get("accepted", 0) or 0),
        "rejected": int(summary.get("rejected", 0) or 0),
        "keeper": str(summary.get("keeper_label", "Keeper #1")),
        "stage_averages_ms": stage,
        "source_performance": summary.get("source_performance", []),
    }


def _profiling_summary(output: Path) -> dict[str, Any]:
    path = output / "profiling" / "summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _diff(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    before = _as_float(baseline.get("analysis_seconds"), 0.0)
    after = _as_float(current.get("analysis_seconds"), 0.0)
    speedup = ((before - after) / before * 100.0) if before > 0 else 0.0
    return {
        "analysis_seconds_before": round(before, 3),
        "analysis_seconds_after": round(after, 3),
        "improvement_percent": round(speedup, 3),
        "realtime_before": round(_as_float(baseline.get("realtime_factor")), 3),
        "realtime_after": round(_as_float(current.get("realtime_factor")), 3),
        "processed_frames_before": int(baseline.get("processed_frames", 0) or 0),
        "processed_frames_after": int(current.get("processed_frames", 0) or 0),
        "candidates_before": int(baseline.get("candidates", 0) or 0),
        "candidates_after": int(current.get("candidates", 0) or 0),
        "accepted_before": int(baseline.get("accepted", 0) or 0),
        "accepted_after": int(current.get("accepted", 0) or 0),
        "rejected_before": int(baseline.get("rejected", 0) or 0),
        "rejected_after": int(current.get("rejected", 0) or 0),
        "keeper_before": str(baseline.get("keeper", "Keeper #1")),
        "keeper_after": str(current.get("keeper", "Keeper #1")),
    }


def run_benchmark(
    source: Path,
    output: Path,
    config: dict[str, Any],
    *,
    duration_seconds: float,
    start_seconds: float = 0.0,
    baseline_path: Path | None = None,
    fp16: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cfg = _benchmark_config(config, start_seconds=start_seconds, duration_seconds=duration_seconds, fp16=fp16)
    started = time.perf_counter()
    summary = run(source, output, cfg, overwrite=True, ffmpeg=ffmpeg, ffprobe=ffprobe, progress_callback=None)
    wall_seconds = time.perf_counter() - started
    profiling_summary = _profiling_summary(output)
    merged_summary = {**profiling_summary, **summary}
    benchmark_output = output / "benchmark"
    benchmark_output.mkdir(parents=True, exist_ok=True)
    payload = _metrics(merged_summary, start=start_seconds, duration=duration_seconds, fp16=fp16)
    payload["wall_seconds"] = round(wall_seconds, 3)
    baseline = _load_json(baseline_path)
    payload["baseline_diff"] = _diff(payload, baseline)
    (benchmark_output / "benchmark.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (benchmark_output / "benchmark.html").write_text(_html(payload), encoding="utf-8")
    return payload


def _html(report: dict[str, Any]) -> str:
    pretty = json.dumps(report, indent=2, ensure_ascii=False)
    return f"""<!doctype html><html lang='de'><head><meta charset='utf-8'><title>Benchmark</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#e6edf3;max-width:1000px;margin:30px auto}}pre{{background:#161b22;padding:20px;border-radius:10px;overflow:auto}}</style></head>
<body><h1>Goalkeeper Highlights Benchmark</h1><pre>{pretty}</pre></body></html>"""
