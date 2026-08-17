from pathlib import Path
from types import SimpleNamespace

import yaml

from goalkeeper_highlights.detection import (
    clamp_clip_windows_to_sources,
    extend_and_chain_clip_windows,
)
from goalkeeper_highlights.models import Candidate


def test_action_boundaries_drive_clip_window():
    candidate = Candidate(90, 120, 100, 0.1, 1, accepted=True, category="save_or_deflection", action_start=96, action_end=104, contact_frames=2, approach_speed=0.2)
    candidate.merged_from = []
    result = extend_and_chain_clip_windows([candidate], 200, {
        "seconds_before": 4,
        "seconds_after": 4,
        "category_pre_roll_seconds": {"save_or_deflection": 6},
        "category_post_roll_seconds": {"save_or_deflection": 7},
        "continuation_gap_seconds": 12,
        "final_keeper_contact_tail_seconds": 5,
        "minimum_clip_seconds": 6,
        "max_dynamic_clip_seconds": 45,
        "interaction_validation": {"enabled": True},
    })
    assert result[0].start == 90
    assert result[0].end == 111
    assert result[0].clip_boundary_reason == "observed_action_window"


def test_static_long_contact_is_rejected():
    candidate = Candidate(10, 20, 15, 0.0, 1, accepted=True, category="catch_or_control", contact_frames=120, approach_speed=0.0, departure_speed=0.0, direction_change=0.0, keeper_motion=0.0, action_start=14, action_end=16)
    candidate.merged_from = []
    result = extend_and_chain_clip_windows([candidate], 100, {
        "interaction_validation": {"enabled": True, "extreme_contact_frames": 80, "minimum_motion_signal": 0.08},
    })
    assert result[0].accepted is False
    assert result[0].rejection_reason == "insufficient_interaction_dynamics"


def test_source_boundary_is_hard_by_default():
    candidate = Candidate(95, 105, 99, 0.1, 1, accepted=True, action_start=97, action_end=103)
    manifest = SimpleNamespace(files=[
        SimpleNamespace(global_start_seconds=0.0, global_end_seconds=100.0),
        SimpleNamespace(global_start_seconds=100.0, global_end_seconds=200.0),
    ])
    clamp_clip_windows_to_sources([candidate], manifest, {"allow_cross_source_clips": False})
    assert candidate.end == 100.0
    assert candidate.clip_boundary_reason == "source_boundary_clamp"


def test_unrelated_goal_kick_is_not_chained_to_previous_restart():
    first = Candidate(10, 15, 12, .1, 1, accepted=True, category="distribution", action_start=11, action_end=13)
    second = Candidate(20, 25, 22, .1, 1, accepted=True, category="keeper_clearance", action_start=21, action_end=23)
    result = extend_and_chain_clip_windows([first, second], 100, {
        "seconds_before": 4, "seconds_after": 4,
        "category_pre_roll_seconds": {"distribution": 4, "keeper_clearance": 6},
        "category_post_roll_seconds": {"distribution": 12, "keeper_clearance": 11},
        "continuation_gap_seconds": 12, "final_keeper_contact_tail_seconds": 5,
        "minimum_clip_seconds": 6, "max_dynamic_clip_seconds": 45,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 2
    assert result[1].start == 15
    assert result[1].end == 34


def test_isolated_short_keeper_clearance_gets_conservative_tail():
    clearance = Candidate(
        candidate_id="raw-clear-short",
        start=602.72,
        end=620.12,
        trigger_time=608.72,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        accepted=True,
        category="keeper_clearance",
        action_start=608.72,
        action_end=609.12,
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
    )
    result = extend_and_chain_clip_windows([clearance], 700.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"keeper_clearance": 6.0},
        "category_post_roll_seconds": {"keeper_clearance": 11.0},
        "continuation_gap_seconds": 12.0,
        "final_keeper_contact_tail_seconds": 5.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].start == 602.72
    assert result[0].end == 612.72
    assert result[0].clip_boundary_reason == "isolated_clearance_safety_tail"


def test_keeper_clearance_with_continuation_keeps_longer_context():
    clearance = Candidate(
        candidate_id="raw-clear-chain",
        start=602.72,
        end=620.12,
        trigger_time=608.72,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        accepted=True,
        category="keeper_clearance",
        action_start=608.72,
        action_end=609.12,
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
    )
    follow_up = Candidate(
        candidate_id="raw-follow",
        start=610.0,
        end=617.0,
        trigger_time=611.0,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        accepted=True,
        category="interaction",
        action_start=611.0,
        action_end=613.0,
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
    )
    result = extend_and_chain_clip_windows([clearance, follow_up], 700.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"keeper_clearance": 6.0, "interaction": 5.0},
        "category_post_roll_seconds": {"keeper_clearance": 11.0, "interaction": 8.0},
        "continuation_gap_seconds": 0.2,
        "phase_merge_gap_seconds": 0.2,
        "final_keeper_contact_tail_seconds": 5.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "interaction_validation": {"enabled": False},
    })
    clearance_result = next(c for c in result if c.candidate_id == "raw-clear-chain")
    assert clearance_result.end > 612.72
    assert clearance_result.clip_boundary_reason != "isolated_clearance_safety_tail"


def test_restart_rescued_distribution_gets_small_additional_tail():
    candidate = Candidate(
        start=920.24,
        end=943.52,
        trigger_time=924.24,
        min_normalized_distance=0.0,
        keeper_track_id=1,
        accepted=True,
        category="distribution",
        action_start=924.24,
        action_end=939.52,
        keeper_label="Keeper #1",
        clip_end_reason="controlled_release",
        score_breakdown={"restart_relevance_rescue_applied": 1.0},
        merged_from=["raw-0035", "raw-0036", "raw-0037"],
        departure_speed=1.64,
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 4.0},
        "category_post_roll_seconds": {"distribution": 4.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].accepted is True
    assert result[0].end == 944.52
    assert result[0].clip_boundary_reason == "restart_rescue_distribution_tail"


def test_short_distribution_gets_preparation_pre_roll_like_raw_0021():
    candidate = Candidate(
        start=640.28,
        end=655.28,
        trigger_time=640.28,
        min_normalized_distance=0.0,
        keeper_track_id=1,
        accepted=True,
        category="distribution",
        action_start=640.28,
        action_end=641.60,
        keeper_label="Keeper #1",
        clip_end_reason="kick",
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 0.0},
        "category_post_roll_seconds": {"distribution": 11.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].start == 638.28
    assert result[0].end == 652.6
    assert result[0].clip_boundary_reason == "distribution_preparation_pre_roll"


def test_short_distribution_with_single_merge_gets_preparation_pre_roll_like_raw_0025():
    candidate = Candidate(
        start=729.16,
        end=744.16,
        trigger_time=729.16,
        min_normalized_distance=0.0,
        keeper_track_id=1,
        accepted=True,
        category="distribution",
        action_start=729.16,
        action_end=731.30,
        keeper_label="Keeper #1",
        clip_end_reason="kick",
        merged_from=["raw-0024"],
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 0.0},
        "category_post_roll_seconds": {"distribution": 11.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].start == 727.16
    assert result[0].end == 742.3
    assert result[0].clip_boundary_reason == "distribution_preparation_pre_roll"


def test_distribution_preparation_pre_roll_does_not_reexpand_long_merged_distribution():
    candidate = Candidate(
        start=920.24,
        end=944.52,
        trigger_time=924.24,
        min_normalized_distance=0.0,
        keeper_track_id=1,
        accepted=True,
        category="distribution",
        action_start=924.24,
        action_end=939.52,
        keeper_label="Keeper #1",
        clip_end_reason="controlled_release",
        merged_from=["raw-0035", "raw-0036", "raw-0037"],
        score_breakdown={"restart_relevance_rescue_applied": 1.0},
        departure_speed=1.64,
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 4.0},
        "category_post_roll_seconds": {"distribution": 4.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].start == 920.24
    assert result[0].end == 944.52


def test_long_multi_distribution_is_trimmed_to_compact_core_window():
    candidate = Candidate(
        start=1470.16,
        end=1497.60,
        trigger_time=1476.0,
        min_normalized_distance=0.0,
        keeper_track_id=1,
        accepted=True,
        category="distribution",
        action_start=1474.0,
        action_end=1485.0,
        keeper_label="Keeper #1",
        clip_end_reason="controlled_release",
        merged_from=["raw-0039", "raw-0040", "raw-0042"],
        departure_speed=7.7,
        score_breakdown={"phase_merge_action_duration": 18.0},
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 4.0},
        "category_post_roll_seconds": {"distribution": 11.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].accepted is True
    assert (result[0].end - result[0].start) == 15.0
    assert result[0].clip_boundary_reason == "distribution_compact_core_window"


def test_recovery_contextual_rescue_gets_compact_window_instead_of_generic_recovery_span():
    candidate = Candidate(
        start=1566.0,
        end=1585.0,
        trigger_time=1574.0,
        min_normalized_distance=0.80,
        keeper_track_id=1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=1574.0,
        action_end=1576.0,
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        score_breakdown={"recovery_contextual_rescue_applied": 1.0},
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 8.0,
        "seconds_after": 9.0,
        "category_pre_roll_seconds": {"recovery_uncovered_activity": 8.0},
        "category_post_roll_seconds": {"recovery_uncovered_activity": 9.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].accepted is True
    assert (result[0].end - result[0].start) == 14.0
    assert result[0].clip_boundary_reason == "recovery_context_rescue_window"


def test_long_multi_catch_final_overlap_phase_is_core_trimmed():
    candidate = Candidate(
        start=1659.12,
        end=1707.72,
        trigger_time=1661.12,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        accepted=True,
        category="catch_or_control",
        action_start=1661.12,
        action_end=1663.04,
        keeper_label="Keeper #1",
        clip_end_reason="dynamic_idle_tail",
        clip_boundary_reason="final_overlap_merged",
        merged_from=["raw-0052", "raw-0053", "raw-0054", "raw-0055", "raw-0056", "diagnostic-recovery-0007"],
        score_breakdown={
            "final_overlap_merge_applied": 1.0,
            "final_overlap_original_union_duration": 61.88,
            "final_overlap_trimmed_duration": 48.60,
        },
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"catch_or_control": 10.0},
        "category_post_roll_seconds": {"catch_or_control": 11.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].accepted is True
    assert (result[0].end - result[0].start) < 48.6
    assert result[0].end <= 1696.12


def test_default_catch_control_final_overlap_core_max_seconds_is_24():
    loaded = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    packaged = yaml.safe_load(Path("src/goalkeeper_highlights/default.yaml").read_text(encoding="utf-8"))
    assert loaded["clips"]["catch_control_final_overlap_core_max_seconds"] == 24.0
    assert packaged["clips"]["catch_control_final_overlap_core_max_seconds"] == 24.0


def test_long_multi_catch_final_overlap_uses_default_24s_core_limit():
    cfg = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))["clips"]
    cfg["interaction_validation"] = {"enabled": False}
    candidate = Candidate(
        start=1674.0,
        end=1709.0,
        trigger_time=1661.12,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        accepted=True,
        category="catch_or_control",
        action_start=1661.12,
        action_end=1663.04,
        keeper_label="Keeper #1",
        clip_end_reason="dynamic_idle_tail",
        clip_boundary_reason="final_overlap_merged",
        merged_from=["raw-0052", "raw-0053", "raw-0054", "raw-0055", "raw-0056", "diagnostic-recovery-0007"],
        score_breakdown={
            "final_overlap_merge_applied": 1.0,
            "final_overlap_original_union_duration": 61.88,
            "final_overlap_trimmed_duration": 35.0,
        },
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, cfg)
    assert len(result) == 1
    assert result[0].accepted is True
    assert (result[0].end - result[0].start) <= 25.0
