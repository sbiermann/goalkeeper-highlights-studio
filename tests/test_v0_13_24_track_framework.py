from __future__ import annotations

from pathlib import Path

from goalkeeper_highlights.profiling import PerformanceProfiler


def test_v0_13_24_track_framework_stage_order_contains_new_substages(tmp_path: Path) -> None:
    profiler = PerformanceProfiler(output_dir=tmp_path, enabled=False)
    for stage in (
        "track_callback_ms",
        "track_callback_dispatch_ms",
        "track_callback_predict_start_ms",
        "track_callback_batch_start_ms",
        "track_callback_postprocess_end_ms",
        "track_callback_batch_end_ms",
        "track_callback_predict_end_ms",
        "track_callback_other_ms",
        "track_predictor_pre_ms",
        "track_pre_source_setup_ms",
        "track_pre_batch_prepare_ms",
        "track_pre_other_ms",
        "track_predictor_post_ms",
        "track_tracker_update_ms",
        "track_result_build_ms",
        "track_result_wrap_ms",
        "track_ultralytics_misc_ms",
        "track_framework_other_ms",
    ):
        assert stage in profiler.stage_order
