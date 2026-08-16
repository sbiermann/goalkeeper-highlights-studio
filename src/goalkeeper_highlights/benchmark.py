from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from . import __version__
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


def _benchmark_config(
    config: dict[str, Any],
    *,
    start_seconds: float,
    duration_seconds: float,
    fp16: bool,
    track_path: str = "legacy",
    decoder_mode: str = "prefetch",
    decoder_prefetch_queue_size: int = 4,
    backend: str = "pytorch",
) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    runtime = cfg.setdefault("runtime", {})
    runtime["benchmark_mode"] = True
    runtime["benchmark_start_seconds"] = max(0.0, float(start_seconds))
    runtime["benchmark_duration_seconds"] = max(1.0, float(duration_seconds))
    runtime["track_execution_mode"] = str(track_path or "legacy").strip().lower()
    runtime["decoder_execution_mode"] = str(decoder_mode or "prefetch").strip().lower()
    runtime["decoder_prefetch_queue_size"] = max(1, int(decoder_prefetch_queue_size))
    runtime["export_rejected"] = False
    runtime["verbose_console"] = False
    cfg.setdefault("profiling", {})["enabled"] = True
    cfg.setdefault("diagnostics", {})["enabled"] = False
    cfg.setdefault("qwen", {})["enabled"] = False
    cfg.setdefault("yolo", {})["half"] = bool(fp16)
    cfg.setdefault("yolo", {})["backend"] = str(backend or "pytorch").strip().lower()
    return cfg


def _stage_averages(summary: dict[str, Any]) -> dict[str, float]:
    values = summary.get("stage_averages_ms", {})
    if not isinstance(values, dict):
        return {}
    return {str(k): _as_float(v) for k, v in values.items()}


def _improvement_percent(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (before - after) / before * 100.0


def _increase_percent(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (after - before) / before * 100.0


def _safe_round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _compare_detections(
    current_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    iou_tolerance: float = 0.85,
    coord_tolerance_px: float = 3.0,
    confidence_tolerance: float = 0.03,
) -> dict[str, Any]:
    baseline_map = {(int(r.get("frame", -1)), int(r.get("track_id", -1)), int(r.get("class_id", -1))): r for r in baseline_rows}
    total = 0
    equivalent = 0
    class_mismatch = 0
    confidence_mismatch = 0
    bbox_mismatch = 0
    missing = 0
    for row in current_rows:
        key = (int(row.get("frame", -1)), int(row.get("track_id", -1)), int(row.get("class_id", -1)))
        base = baseline_map.get(key)
        total += 1
        if base is None:
            missing += 1
            continue
        if int(base.get("class_id", -1)) != int(row.get("class_id", -1)):
            class_mismatch += 1
            continue
        conf_delta = abs(_as_float(base.get("confidence")) - _as_float(row.get("confidence")))
        if conf_delta > confidence_tolerance:
            confidence_mismatch += 1
        x1, y1, x2, y2 = (_as_float(row.get("x1")), _as_float(row.get("y1")), _as_float(row.get("x2")), _as_float(row.get("y2")))
        bx1, by1, bx2, by2 = (_as_float(base.get("x1")), _as_float(base.get("y1")), _as_float(base.get("x2")), _as_float(base.get("y2")))
        inter_x1 = max(x1, bx1)
        inter_y1 = max(y1, by1)
        inter_x2 = min(x2, bx2)
        inter_y2 = min(y2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter_area
        iou = inter_area / union if union > 0 else 0.0
        coord_delta = max(abs(x1 - bx1), abs(y1 - by1), abs(x2 - bx2), abs(y2 - by2))
        if iou < iou_tolerance and coord_delta > coord_tolerance_px:
            bbox_mismatch += 1
        if conf_delta <= confidence_tolerance and (iou >= iou_tolerance or coord_delta <= coord_tolerance_px):
            equivalent += 1
    return {
        "rows_compared": total,
        "rows_equivalent": equivalent,
        "missing_rows": missing,
        "class_mismatch_rows": class_mismatch,
        "confidence_mismatch_rows": confidence_mismatch,
        "bbox_mismatch_rows": bbox_mismatch,
        "iou_tolerance": iou_tolerance,
        "coord_tolerance_px": coord_tolerance_px,
        "confidence_tolerance": confidence_tolerance,
    }


def _compare_candidates(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "processed_frames",
        "candidates",
        "accepted",
        "rejected",
        "merged",
        "keeper",
        "keeper_confidence",
    ]
    differences: dict[str, dict[str, Any]] = {}
    for field in fields:
        left = current.get(field)
        right = baseline.get(field)
        if left != right:
            differences[field] = {"baseline": right, "current": left}
    equivalent = len(differences) == 0
    return {"equivalent": equivalent, "differences": differences}


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
        "version": str(summary.get("version") or __version__),
        "start_seconds": round(start, 3),
        "duration_seconds": round(duration, 3),
        "fp16": bool(fp16),
        "requested_precision": "FP16" if bool(summary.get("requested_fp16", fp16)) else "FP32",
        "effective_precision": "FP16" if bool(summary.get("effective_fp16", fp16)) else "FP32",
        "fp16_requested": bool(summary.get("requested_fp16", fp16)),
        "fp16_effective": bool(summary.get("effective_fp16", fp16)),
        "fp16_fallback_reason": summary.get("fp16_fallback_reason"),
        "requested_backend": str(summary.get("requested_backend", "pytorch")),
        "effective_backend": str(summary.get("effective_backend", summary.get("requested_backend", "pytorch"))),
        "backend_fallback_reason": summary.get("backend_fallback_reason"),
        "model_format": str(summary.get("model_format", "pytorch")),
        "engine_cached": bool(summary.get("engine_cached", False)),
        "engine_build_seconds": round(_as_float(summary.get("engine_build_seconds")), 6),
        "backend_load_seconds": round(_as_float(summary.get("backend_load_seconds")), 6),
        "backend_warmup_seconds": round(_as_float(summary.get("backend_warmup_seconds")), 6),
        "cuda_available": bool(summary.get("cuda_available", False)),
        "device": str(summary.get("device", "")),
        "gpu": str(((summary.get("system") or {}).get("gpu", ""))),
        "gpu_name": str(summary.get("gpu_name", "")),
        "tensorrt_version": str(summary.get("tensorrt_version", "")),
        "onnxruntime_version": str(summary.get("onnxruntime_version", "")),
        "onnx_execution_provider": str(summary.get("onnx_execution_provider", "")),
        "analysis_seconds": round(analysis_seconds, 3),
        "video_seconds": round(video_seconds, 3),
        "processed_frames": processed_frames,
        "processed_fps": round(fps, 3),
        "realtime_factor": round(realtime, 3),
        "candidates": int(summary.get("final_candidates", 0) or 0),
        "accepted": int(summary.get("accepted", 0) or 0),
        "rejected": int(summary.get("rejected", 0) or 0),
        "merged": int(summary.get("merged_candidates", 0) or 0),
        "keeper": str(summary.get("keeper_label", "Keeper #1")),
        "keeper_confidence": round(_as_float(summary.get("keeper_confidence", 0.0)), 6),
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
    speedup = _improvement_percent(before, after)
    stage_before = baseline.get("stage_averages_ms", {}) if isinstance(baseline.get("stage_averages_ms"), dict) else {}
    stage_after = current.get("stage_averages_ms", {}) if isinstance(current.get("stage_averages_ms"), dict) else {}
    infer_before = _as_float(stage_before.get("yolo_inference_ms"))
    infer_after = _as_float(stage_after.get("yolo_inference_ms"))
    model_before = _as_float(stage_before.get("model_track_wall_ms"))
    model_after = _as_float(stage_after.get("model_track_wall_ms"))
    loop_before = _as_float(stage_before.get("loop_ms"))
    loop_after = _as_float(stage_after.get("loop_ms"))
    fps_before = _as_float(baseline.get("processed_fps"))
    fps_after = _as_float(current.get("processed_fps"))
    detection_comparison = _compare_detections(
        current.get("detections", []) if isinstance(current.get("detections"), list) else [],
        baseline.get("detections", []) if isinstance(baseline.get("detections"), list) else [],
    )
    candidate_comparison = _compare_candidates(current, baseline)
    return {
        "analysis_seconds_before": round(before, 3),
        "analysis_seconds_after": round(after, 3),
        "improvement_percent": _safe_round(speedup),
        "analysis_time_improvement_percent": _safe_round(speedup),
        "inference_improvement_percent": _safe_round(_improvement_percent(infer_before, infer_after)),
        "model_track_improvement_percent": _safe_round(_improvement_percent(model_before, model_after)),
        "loop_improvement_percent": _safe_round(_improvement_percent(loop_before, loop_after)),
        "fps_improvement_percent": _safe_round(_increase_percent(fps_before, fps_after)),
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
        "merged_before": int(baseline.get("merged", 0) or 0),
        "merged_after": int(current.get("merged", 0) or 0),
        "keeper_before": str(baseline.get("keeper", "Keeper #1")),
        "keeper_after": str(current.get("keeper", "Keeper #1")),
        "detection_comparison": detection_comparison,
        "candidate_comparison": candidate_comparison,
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
    track_path: str = "optimized",
    decoder_mode: str = "legacy",
    decoder_prefetch_queue_size: int = 4,
    backend: str = "pytorch",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cfg = _benchmark_config(
        config,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        fp16=fp16,
        track_path=track_path,
        decoder_mode=decoder_mode,
        decoder_prefetch_queue_size=decoder_prefetch_queue_size,
        backend=backend,
    )
    started = time.perf_counter()
    summary = run(source, output, cfg, overwrite=True, ffmpeg=ffmpeg, ffprobe=ffprobe, progress_callback=None)
    wall_seconds = time.perf_counter() - started
    profiling_summary = _profiling_summary(output)
    merged_summary = {**profiling_summary, **summary}
    benchmark_output = output / "benchmark"
    benchmark_output.mkdir(parents=True, exist_ok=True)
    payload = _metrics(merged_summary, start=start_seconds, duration=duration_seconds, fp16=fp16)
    payload["wall_seconds"] = round(wall_seconds, 3)
    payload["track_path"] = str(track_path)
    payload["decoder_mode"] = str(decoder_mode)
    payload["decoder_prefetch_queue_size"] = int(decoder_prefetch_queue_size)
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
