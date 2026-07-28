from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .decoder import OpenCVDecoder, PyAVDecoder
from .store import AnalysisStore


def _decode_benchmark(decoder, seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    frames = 0
    video_seconds = 0.0
    try:
        for item in decoder:
            frames += 1
            video_seconds = item.timestamp
            if video_seconds >= seconds:
                break
    finally:
        decoder.close()
    wall = time.perf_counter() - started
    return {
        "frames": frames,
        "video_seconds": round(video_seconds, 3),
        "wall_seconds": round(wall, 3),
        "decode_fps": round(frames / max(wall, 0.001), 3),
        "realtime_factor": round(video_seconds / max(wall, 0.001), 3),
    }


def _yolo_benchmark(video: Path, config: dict, seconds: float, backend: str) -> dict[str, Any]:
    from ultralytics import YOLO

    stride = max(1, int(config["yolo"].get("frame_stride", 1)))
    decoder = PyAVDecoder(video, stride) if backend == "pyav" else OpenCVDecoder(video, stride)
    device = config["yolo"].get("device", "auto")
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(config["yolo"]["model"])
    frames = 0
    video_seconds = 0.0
    started = time.perf_counter()
    inference_ms: list[float] = []
    try:
        for item in decoder:
            result = model.predict(
                source=item.image,
                classes=[0, 32],
                conf=float(config["yolo"]["confidence"]),
                iou=float(config["yolo"]["iou"]),
                imgsz=int(config["yolo"]["image_size"]),
                device=device,
                verbose=False,
            )[0]
            inference_ms.append(float((result.speed or {}).get("inference", 0.0)))
            frames += 1
            video_seconds = item.timestamp
            if video_seconds >= seconds:
                break
    finally:
        decoder.close()
    wall = time.perf_counter() - started
    return {
        "backend": backend,
        "device": str(device),
        "frames": frames,
        "video_seconds": round(video_seconds, 3),
        "wall_seconds": round(wall, 3),
        "pipeline_fps": round(frames / max(wall, 0.001), 3),
        "realtime_factor": round(video_seconds / max(wall, 0.001), 3),
        "mean_inference_ms": round(sum(inference_ms) / max(1, len(inference_ms)), 3),
    }


def run_benchmark(video: Path, output: Path, config: dict, seconds: float = 30.0, include_yolo: bool = True) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "video": str(video),
        "sample_seconds": seconds,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "decoders": {},
    }
    for backend, factory in (("opencv", lambda: OpenCVDecoder(video, 1)), ("pyav", lambda: PyAVDecoder(video, 1))):
        try:
            report["decoders"][backend] = _decode_benchmark(factory(), seconds)
        except Exception as exc:
            report["decoders"][backend] = {"error": str(exc)}
    if include_yolo:
        backend = str(config.get("decoder", {}).get("backend", "pyav"))
        report["yolo"] = _yolo_benchmark(video, config, seconds, backend)
    target = output / "benchmark"
    target.mkdir(parents=True, exist_ok=True)
    (target / "benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (target / "benchmark.html").write_text(_html(report), encoding="utf-8")
    store = AnalysisStore(output / "analysis.sqlite3")
    try:
        store.save_benchmark("decoder-and-yolo", report)
    finally:
        store.close()
    return report


def _html(report: dict[str, Any]) -> str:
    pretty = json.dumps(report, indent=2, ensure_ascii=False)
    return f"""<!doctype html><html lang='de'><head><meta charset='utf-8'><title>Benchmark</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#e6edf3;max-width:1000px;margin:30px auto}}pre{{background:#161b22;padding:20px;border-radius:10px;overflow:auto}}</style></head>
<body><h1>Goalkeeper Highlights Benchmark</h1><pre>{pretty}</pre></body></html>"""
