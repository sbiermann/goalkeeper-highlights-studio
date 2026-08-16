from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from goalkeeper_highlights.benchmark import _benchmark_config, _compare_candidates, _compare_detections, _diff, _metrics
from goalkeeper_highlights.cli import parser
from goalkeeper_highlights.detection import _resolve_fp16_state, boxes_from_result, boxes_from_result_legacy


def test_v0_13_20_benchmark_cli_args() -> None:
    args = parser().parse_args(
        [
            "benchmark",
            "video.mp4",
            "--duration",
            "300",
            "--start",
            "30",
            "--fp16",
            "--backend",
            "tensorrt",
            "--imgsz",
            "576",
            "--tf32",
            "--cudnn-benchmark",
        ]
    )
    assert args.command == "benchmark"
    assert args.duration == 300.0
    assert args.start == 30.0
    assert args.fp16 is True
    assert args.track_path == "legacy"
    assert args.decoder_mode == "prefetch"
    assert args.decoder_prefetch_queue_size == 4
    assert args.backend == "tensorrt"
    assert args.imgsz == 576
    assert args.tf32 is True
    assert args.cudnn_benchmark is True


def test_v0_13_20_benchmark_config_disables_clip_export() -> None:
    cfg = _benchmark_config(
        {"runtime": {}, "profiling": {}, "diagnostics": {}, "qwen": {}, "yolo": {}},
        start_seconds=10.0,
        duration_seconds=300.0,
        fp16=False,
        backend="onnx",
        image_size=512,
        tf32=True,
        cudnn_benchmark=True,
    )
    assert cfg["runtime"]["benchmark_mode"] is True
    assert cfg["runtime"]["export_rejected"] is False
    assert cfg["runtime"]["benchmark_start_seconds"] == 10.0
    assert cfg["runtime"]["benchmark_duration_seconds"] == 300.0
    assert cfg["runtime"]["track_execution_mode"] == "legacy"
    assert cfg["runtime"]["decoder_execution_mode"] == "prefetch"
    assert cfg["runtime"]["decoder_prefetch_queue_size"] == 4
    assert cfg["runtime"]["tf32"] is True
    assert cfg["runtime"]["cudnn_benchmark"] is True
    assert cfg["yolo"]["backend"] == "onnx"
    assert cfg["yolo"]["image_size"] == 512


def test_v0_13_22_benchmark_track_path_override() -> None:
    cfg = _benchmark_config(
        {"runtime": {}, "profiling": {}, "diagnostics": {}, "qwen": {}, "yolo": {}},
        start_seconds=0.0,
        duration_seconds=120.0,
        fp16=False,
        track_path="legacy",
    )
    assert cfg["runtime"]["track_execution_mode"] == "legacy"


def test_v0_13_23_benchmark_decoder_mode_override() -> None:
    cfg = _benchmark_config(
        {"runtime": {}, "profiling": {}, "diagnostics": {}, "qwen": {}, "yolo": {}},
        start_seconds=0.0,
        duration_seconds=120.0,
        fp16=False,
        decoder_mode="prefetch",
        decoder_prefetch_queue_size=7,
    )
    assert cfg["runtime"]["decoder_execution_mode"] == "prefetch"
    assert cfg["runtime"]["decoder_prefetch_queue_size"] == 7


def test_v0_13_21_fp16_cuda_guard_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("goalkeeper_highlights.detection.torch.cuda.is_available", lambda: False)
    effective, reason = _resolve_fp16_state("cpu", True)
    assert effective is False
    assert reason == "cuda_unavailable"


def test_v0_13_21_fp16_cuda_guard_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("goalkeeper_highlights.detection.torch.cuda.is_available", lambda: True)
    effective, reason = _resolve_fp16_state("0", True)
    assert effective is True
    assert reason is None


def test_v0_13_21_benchmark_metrics_precision_fields() -> None:
    summary = {
        "version": "0.13.21",
        "analysis_seconds": 10.0,
        "video_duration_seconds": 20.0,
        "processed_frames": 250,
        "requested_fp16": True,
        "effective_fp16": False,
        "fp16_fallback_reason": "cuda_unavailable",
        "requested_backend": "tensorrt",
        "effective_backend": "pytorch",
        "backend_fallback_reason": "tensorrt_unavailable",
        "model_format": "pytorch",
        "engine_cached": False,
        "engine_build_seconds": 0.0,
        "backend_load_seconds": 0.2,
        "backend_warmup_seconds": 0.1,
        "cuda_available": False,
        "device": "cpu",
        "system": {"gpu": ""},
        "keeper_label": "Keeper #1",
        "keeper_confidence": 0.77,
        "accepted": 2,
        "rejected": 1,
        "final_candidates": 3,
        "merged_candidates": 1,
        "stage_averages_ms": {"yolo_inference_ms": 20.0, "model_track_wall_ms": 60.0, "loop_ms": 70.0},
    }
    payload = _metrics(summary, start=0.0, duration=20.0, fp16=True)
    assert payload["requested_precision"] == "FP16"
    assert payload["effective_precision"] == "FP32"
    assert payload["fp16_requested"] is True
    assert payload["fp16_effective"] is False
    assert payload["fp16_fallback_reason"] == "cuda_unavailable"
    assert payload["requested_backend"] == "tensorrt"
    assert payload["effective_backend"] == "pytorch"
    assert payload["backend_fallback_reason"] == "tensorrt_unavailable"


def test_v0_13_21_detection_numeric_tolerance_equivalent() -> None:
    baseline = [{"frame": 10, "track_id": 1, "class_id": 0, "confidence": 0.9, "x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0}]
    current = [{"frame": 10, "track_id": 1, "class_id": 0, "confidence": 0.885, "x1": 10.5, "y1": 10.3, "x2": 30.2, "y2": 29.9}]
    comparison = _compare_detections(current, baseline)
    assert comparison["rows_compared"] == 1
    assert comparison["rows_equivalent"] == 1
    assert comparison["bbox_mismatch_rows"] == 0


def test_v0_13_21_candidate_difference_not_equivalent() -> None:
    baseline = {"processed_frames": 3001, "candidates": 5, "accepted": 3, "rejected": 2, "merged": 1, "keeper": "Keeper #1", "keeper_confidence": 0.8}
    current = {"processed_frames": 3001, "candidates": 6, "accepted": 3, "rejected": 3, "merged": 1, "keeper": "Keeper #1", "keeper_confidence": 0.8}
    comparison = _compare_candidates(current, baseline)
    assert comparison["equivalent"] is False
    assert "candidates" in comparison["differences"]


def test_v0_13_21_keeper_difference_not_equivalent() -> None:
    baseline = {"processed_frames": 3001, "candidates": 5, "accepted": 3, "rejected": 2, "merged": 1, "keeper": "Keeper #1", "keeper_confidence": 0.8}
    current = {"processed_frames": 3001, "candidates": 5, "accepted": 3, "rejected": 2, "merged": 1, "keeper": "Keeper #2", "keeper_confidence": 0.8}
    comparison = _compare_candidates(current, baseline)
    assert comparison["equivalent"] is False
    assert "keeper" in comparison["differences"]


def test_v0_13_20_benchmark_baseline_diff() -> None:
    current = {
        "analysis_seconds": 200.0,
        "realtime_factor": 2.0,
        "processed_frames": 100,
        "candidates": 10,
        "accepted": 3,
        "rejected": 7,
        "keeper": "Keeper #1",
    }
    baseline = {
        "analysis_seconds": 250.0,
        "realtime_factor": 1.6,
        "processed_frames": 100,
        "candidates": 10,
        "accepted": 3,
        "rejected": 7,
        "keeper": "Keeper #1",
    }
    diff = _diff(current, baseline)
    assert diff is not None
    assert diff["improvement_percent"] == pytest.approx(20.0)
    assert diff["keeper_before"] == "Keeper #1"


def test_v0_13_20_boxes_packed_equals_legacy() -> None:
    class _Boxes:
        def __init__(self) -> None:
            self.xyxy = torch.tensor([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
            self.conf = torch.tensor([0.91, 0.44], dtype=torch.float32)
            self.cls = torch.tensor([0.0, 32.0], dtype=torch.float32)
            self.id = torch.tensor([7.0, 13.0], dtype=torch.float32)

        def __len__(self) -> int:
            return int(self.xyxy.shape[0])

    class _Result:
        boxes = _Boxes()

    packed = boxes_from_result(_Result())
    legacy = boxes_from_result_legacy(_Result())
    assert len(packed) == len(legacy)
    for left, right in zip(packed, legacy):
        assert left.class_id == right.class_id
        assert left.track_id == right.track_id
        assert left.confidence == pytest.approx(right.confidence, abs=1e-6)
        assert left.x1 == pytest.approx(right.x1, abs=1e-6)
        assert left.y1 == pytest.approx(right.y1, abs=1e-6)
        assert left.x2 == pytest.approx(right.x2, abs=1e-6)
        assert left.y2 == pytest.approx(right.y2, abs=1e-6)
