from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from goalkeeper_highlights.benchmark import _benchmark_config, _diff
from goalkeeper_highlights.cli import parser
from goalkeeper_highlights.detection import boxes_from_result, boxes_from_result_legacy


def test_v0_13_20_benchmark_cli_args() -> None:
    args = parser().parse_args(["benchmark", "video.mp4", "--duration", "300", "--start", "30", "--fp16"])
    assert args.command == "benchmark"
    assert args.duration == 300.0
    assert args.start == 30.0
    assert args.fp16 is True


def test_v0_13_20_benchmark_config_disables_clip_export() -> None:
    cfg = _benchmark_config({"runtime": {}, "profiling": {}, "diagnostics": {}, "qwen": {}, "yolo": {}}, start_seconds=10.0, duration_seconds=300.0, fp16=False)
    assert cfg["runtime"]["benchmark_mode"] is True
    assert cfg["runtime"]["export_rejected"] is False
    assert cfg["runtime"]["benchmark_start_seconds"] == 10.0
    assert cfg["runtime"]["benchmark_duration_seconds"] == 300.0


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
