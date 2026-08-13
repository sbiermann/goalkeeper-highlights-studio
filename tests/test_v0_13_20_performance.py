from pathlib import Path

from goalkeeper_highlights.profiling import PerformanceProfiler


def _frame(source_index: int = 0, source_name: str = "Teil1.mp4", **overrides):
    base = {
        "source_index": source_index,
        "source_name": source_name,
        "frame_index": 1,
        "video_seconds": 1.0,
        "decoded_frames": 1,
        "processed_frames": 1,
        "decoder_next_ms": 2.0,
        "frame_prepare_ms": 1.0,
        "model_track_wall_ms": 10.0,
        "yolo_preprocess_ms": 1.0,
        "yolo_inference_ms": 6.0,
        "yolo_postprocess_ms": 1.0,
        "track_overhead_ms": 2.0,
        "boxes_from_result_ms": 0.4,
        "keeper_identity_ms": 0.2,
        "keeper_reid_ms": 0.2,
        "keeper_histogram_ms": 0.2,
        "ball_selection_ms": 0.1,
        "event_engine_ms": 0.3,
        "candidate_processing_ms": 0.05,
        "database_buffer_ms": 0.1,
        "database_flush_ms": 0.0,
        "diagnostics_ms": 0.0,
        "progress_reporting_ms": 0.05,
        "preview_ms": 0.0,
        "profiler_overhead_ms": 0.02,
        "other_loop_ms": 1.0,
        "loop_ms": 13.22,
    }
    base.update(overrides)
    return base


def test_v0_13_20_profiling_aggregation(tmp_path: Path):
    profiler = PerformanceProfiler(tmp_path, enabled=True, sample_interval_seconds=9999)
    profiler.record_stage_frame(_frame(loop_ms=10.0, decoder_next_ms=3.0))
    profiler.record_stage_frame(_frame(frame_index=2, video_seconds=2.0, decoded_frames=2, processed_frames=2, loop_ms=14.0, decoder_next_ms=5.0))

    summary = profiler._summary({})
    assert summary["stage_averages_ms"]["loop_ms"] == 12.0
    assert summary["stage_averages_ms"]["decoder_next_ms"] == 4.0
    assert summary["stage_totals_ms"]["loop_ms"] == 24.0


def test_v0_13_20_track_overhead_formula(tmp_path: Path):
    profiler = PerformanceProfiler(tmp_path, enabled=True, sample_interval_seconds=9999)
    profiler.record_stage_frame(_frame(model_track_wall_ms=20.0, yolo_preprocess_ms=3.0, yolo_inference_ms=12.0, yolo_postprocess_ms=2.0, track_overhead_ms=3.0))
    summary = profiler._summary({})
    assert summary["stage_averages_ms"]["track_overhead_ms"] == 3.0


def test_v0_13_20_decode_is_separate_stage(tmp_path: Path):
    profiler = PerformanceProfiler(tmp_path, enabled=True, sample_interval_seconds=9999)
    profiler.record_stage_frame(_frame(decoder_next_ms=7.5, model_track_wall_ms=0.0, yolo_inference_ms=0.0))
    summary = profiler._summary({})
    assert summary["stage_averages_ms"]["decoder_next_ms"] == 7.5
    assert summary["stage_averages_ms"]["model_track_wall_ms"] == 0.0


def test_v0_13_20_source_performance_stats(tmp_path: Path):
    profiler = PerformanceProfiler(tmp_path, enabled=True, sample_interval_seconds=9999)
    profiler.record_stage_frame(_frame(source_index=0, source_name="Teil1.mp4", frame_index=1, video_seconds=1.0, decoded_frames=1, processed_frames=1, loop_ms=10.0))
    profiler.record_stage_frame(_frame(source_index=0, source_name="Teil1.mp4", frame_index=2, video_seconds=2.0, decoded_frames=2, processed_frames=2, loop_ms=12.0))
    profiler.record_stage_frame(_frame(source_index=1, source_name="Teil2.mp4", frame_index=1, video_seconds=1.5, decoded_frames=1, processed_frames=1, loop_ms=8.0))

    summary = profiler._summary({})
    sources = summary["source_performance"]
    assert len(sources) == 2
    assert sources[0]["source_name"] == "Teil1.mp4"
    assert "p95_loop_ms" in sources[0]


def test_v0_13_20_no_negative_timings_in_summary(tmp_path: Path):
    profiler = PerformanceProfiler(tmp_path, enabled=True, sample_interval_seconds=9999)
    profiler.record_stage_frame(_frame(decoder_next_ms=-5.0, loop_ms=-1.0, other_loop_ms=-2.0))
    summary = profiler._summary({})
    assert summary["stage_averages_ms"]["decoder_next_ms"] >= 0.0
    assert summary["stage_averages_ms"]["loop_ms"] >= 0.0
    assert summary["stage_averages_ms"]["other_loop_ms"] >= 0.0
