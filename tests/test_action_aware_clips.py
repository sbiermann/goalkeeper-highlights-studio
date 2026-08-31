from pathlib import Path
from types import SimpleNamespace

import pytest
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



def test_compact_distribution_core_keeps_two_second_preparation_context_like_raw_0021():
    candidate = Candidate(
        start=622.08,
        end=655.28,
        trigger_time=628.08,
        min_normalized_distance=0.0,
        keeper_track_id=1,
        accepted=True,
        category="distribution",
        action_start=628.08,
        action_end=651.28,
        keeper_label="Keeper #1",
        clip_end_reason="controlled_release",
        merged_from=["raw-0022"],
        departure_speed=6.21,
        possession_duration=18.8,
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 6.0},
        "category_post_roll_seconds": {"distribution": 12.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "distribution_preparation_pre_roll_seconds": 2.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].start == 638.28
    assert result[0].end == 655.28
    assert result[0].clip_boundary_reason == "distribution_preparation_pre_roll"
    assert result[0].score_breakdown["distribution_preparation_pre_roll_anchor"] == "compact_core_start"


def test_compact_multi_merge_distribution_keeps_two_second_preparation_context_like_raw_0025():
    candidate = Candidate(
        start=708.4,
        end=744.16,
        trigger_time=714.4,
        min_normalized_distance=0.0,
        keeper_track_id=1,
        accepted=True,
        category="distribution",
        action_start=714.4,
        action_end=740.16,
        keeper_label="Keeper #1",
        clip_end_reason="controlled_release",
        merged_from=["raw-0026", "raw-0027", "raw-0028", "raw-0029"],
        departure_speed=7.39,
        possession_duration=6.8,
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 6.0},
        "category_post_roll_seconds": {"distribution": 12.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "distribution_preparation_pre_roll_seconds": 2.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].start == 727.16
    assert result[0].end == 744.16
    assert result[0].clip_boundary_reason == "distribution_preparation_pre_roll"
    assert result[0].score_breakdown["distribution_preparation_pre_roll_anchor"] == "compact_core_start"

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


def test_neighbor_context_recovery_adds_one_second_pre_roll_without_extending_end():
    candidate = Candidate(
        start=1570.0,
        end=1580.0,
        trigger_time=1574.0,
        min_normalized_distance=1.02348,
        keeper_track_id=1,
        accepted=True,
        category="recovery_uncovered_activity",
        recovery_candidate=True,
        action_start=1574.0,
        action_end=1576.0,
        keeper_label="Keeper #1",
        clip_end_reason="timeout",
        score_breakdown={
            "recovery_contextual_rescue_applied": 1.0,
            "recovery_neighbor_context_rescue": 1.0,
        },
    )
    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"recovery_uncovered_activity": 4.0},
        "category_post_roll_seconds": {"recovery_uncovered_activity": 4.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "recovery_neighbor_context_extra_pre_roll_seconds": 1.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].start == 1569.0
    assert result[0].end == 1580.0
    assert (result[0].end - result[0].start) == 11.0
    assert result[0].clip_boundary_reason == "recovery_neighbor_context_pre_roll"
    assert result[0].score_breakdown["recovery_neighbor_context_pre_roll_applied"] == 1.0


def test_contextual_recovery_without_neighbor_rescue_does_not_get_extra_pre_roll():
    candidate = Candidate(
        start=1570.0,
        end=1580.0,
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
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"recovery_uncovered_activity": 4.0},
        "category_post_roll_seconds": {"recovery_uncovered_activity": 4.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "recovery_neighbor_context_extra_pre_roll_seconds": 1.0,
        "interaction_validation": {"enabled": False},
    })
    assert len(result) == 1
    assert result[0].start == 1570.0
    assert result[0].end == 1580.0


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


def test_restart_control_rescue_preserves_start_and_caps_long_clip_to_18_seconds():
    candidate = Candidate(
        start=1935.28, end=1971.0, trigger_time=1939.28, min_normalized_distance=0.0, keeper_track_id=1,
        accepted=True, category="catch_or_control", action_start=1939.28, action_end=1962.0,
        contact_frames=71, ball_confidence=0.765, keeper_motion=2.285, possession_duration=4.96,
        keeper_y_normalized=0.514, approach_speed=0.0664,
    )
    result = extend_and_chain_clip_windows([candidate], 3000.0, {
        "seconds_before": 4.0, "seconds_after": 4.0,
        "category_pre_roll_seconds": {"catch_or_control": 10.0},
        "category_post_roll_seconds": {"catch_or_control": 11.0},
        "restart_control_rescue_max_clip_seconds": 18.0,
        "interaction_validation": {"enabled": True, "minimum_motion_signal": 0.08, "outside_box_restart_min_seconds": 2.5},
    })
    assert result[0].accepted is True
    assert result[0].start == 1935.28
    assert result[0].end == 1953.28
    assert result[0].clip_boundary_reason == "restart_control_rescue_compact_window"


def test_restart_control_rescue_keeps_short_existing_window_like_raw_0069():
    candidate = Candidate(
        start=2160.16, end=2171.44, trigger_time=2164.16, min_normalized_distance=0.0, keeper_track_id=1,
        accepted=True, category="catch_or_control", action_start=2164.16, action_end=2167.44,
        contact_frames=29, ball_confidence=0.683, keeper_motion=0.061, possession_duration=2.72,
        keeper_y_normalized=0.513,
    )
    result = extend_and_chain_clip_windows([candidate], 3000.0, {
        "seconds_before": 4.0, "seconds_after": 4.0,
        "category_pre_roll_seconds": {"catch_or_control": 10.0},
        "category_post_roll_seconds": {"catch_or_control": 11.0},
        "restart_control_rescue_max_clip_seconds": 18.0,
        "interaction_validation": {"enabled": True, "minimum_motion_signal": 0.08, "outside_box_restart_min_seconds": 2.5},
    })
    assert result[0].accepted is True
    assert result[0].start == 2160.16
    assert result[0].end == 2171.44


def test_rejected_leading_distribution_is_absorbed_into_immediate_strong_keeper_action():
    leading = Candidate(
        start=2029.12, end=2038.08, trigger_time=2033.12, min_normalized_distance=0.2699, keeper_track_id=1,
        accepted=False, category="distribution", rejection_reason="insufficient_interaction_dynamics",
        action_start=2033.12, action_end=2034.08, contact_frames=6, ball_confidence=0.714, keeper_motion=0.047,
    )
    highlight = Candidate(
        start=2038.60, end=2074.40, trigger_time=2055.92, min_normalized_distance=0.0, keeper_track_id=1,
        accepted=True, category="diving_save", action_start=2047.60, action_end=2070.40, contact_frames=95,
        ball_confidence=0.701, keeper_motion=0.713, possession_duration=4.48, approach_speed=0.367,
        departure_speed=6.95, direction_change=0.997, clip_end_reason="controlled_release",
    )
    result = extend_and_chain_clip_windows([leading, highlight], 3000.0, {
        "seconds_before": 4.0, "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 4.0, "diving_save": 9.0},
        "category_post_roll_seconds": {"distribution": 4.0, "diving_save": 11.0},
        "leading_context_absorb_max_gap_seconds": 1.0,
        "interaction_validation": {"enabled": False},
    })
    accepted = [c for c in result if c.accepted]
    assert len(accepted) == 1
    assert accepted[0].candidate_id == highlight.candidate_id
    assert accepted[0].start == 2029.12
    assert accepted[0].end == 2074.40
    assert leading.continuation_absorbed is True
    assert leading.absorbed_into_candidate_id == highlight.candidate_id
    assert accepted[0].score_breakdown.get("leading_context_absorbed") == 1.0


def test_restart_rescue_with_diagnostic_recovery_merge_is_capped_to_10_second_core():
    candidate = Candidate(
        start=2247.44, end=2274.0, trigger_time=2251.44, min_normalized_distance=0.0, keeper_track_id=1,
        accepted=True, category="distribution", action_start=2251.44, action_end=2270.0, contact_frames=41,
        ball_confidence=0.759, keeper_motion=1.566, possession_duration=3.2, departure_speed=1.116,
        clip_end_reason="controlled_release", merged_from=["raw-child-a", "raw-child-b", "diagnostic-recovery-child"],
        score_breakdown={"restart_relevance_rescue_applied": 1.0},
    )
    result = extend_and_chain_clip_windows([candidate], 3000.0, {
        "seconds_before": 4.0, "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 4.0},
        "category_post_roll_seconds": {"distribution": 12.0},
        "distribution_restart_rescue_extra_tail_seconds": 1.0,
        "restart_diagnostic_recovery_core_max_seconds": 10.0,
        "interaction_validation": {"enabled": False},
    })
    assert result[0].start == 2247.44
    assert result[0].end == 2257.44
    assert result[0].clip_boundary_reason == "restart_diagnostic_recovery_compact_core"


def test_restart_rescue_without_diagnostic_recovery_merge_keeps_existing_baseline_window():
    candidate = Candidate(
        start=919.20, end=943.52, trigger_time=924.24, min_normalized_distance=0.0, keeper_track_id=1,
        accepted=True, category="distribution", action_start=923.20, action_end=939.52, contact_frames=64,
        ball_confidence=0.736, keeper_motion=4.13, possession_duration=2.8, departure_speed=1.64,
        clip_end_reason="controlled_release", merged_from=["raw-a", "raw-b", "raw-c"],
        score_breakdown={"restart_relevance_rescue_applied": 1.0},
    )
    result = extend_and_chain_clip_windows([candidate], 3000.0, {
        "seconds_before": 4.0, "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 4.0},
        "category_post_roll_seconds": {"distribution": 12.0},
        "distribution_restart_rescue_extra_tail_seconds": 1.0,
        "restart_diagnostic_recovery_core_max_seconds": 10.0,
        "interaction_validation": {"enabled": False},
    })
    assert result[0].start == 919.20
    assert result[0].end == 944.52


def test_accepted_leading_distribution_prefers_immediate_strong_followup():
    leading = Candidate(
        candidate_id="leading-accepted",
        start=2029.12, end=2038.08, trigger_time=2033.12, min_normalized_distance=0.2699, keeper_track_id=1,
        accepted=True, category="distribution", action_start=2033.12, action_end=2034.08,
        contact_frames=6, ball_confidence=0.714, keeper_motion=0.047, interaction_score=0.20,
        keeper_label="Keeper #1", clip_end_reason="timeout",
    )
    highlight = Candidate(
        candidate_id="strong-followup",
        start=2038.60, end=2074.40, trigger_time=2055.92, min_normalized_distance=0.0, keeper_track_id=1,
        accepted=True, category="diving_save", action_start=2047.60, action_end=2070.40,
        contact_frames=95, ball_confidence=0.701, keeper_motion=0.713, interaction_score=1.0,
        possession_duration=4.48, approach_speed=0.367, departure_speed=6.95, direction_change=0.997,
        keeper_label="Keeper #1", clip_end_reason="controlled_release",
    )

    result = extend_and_chain_clip_windows([leading, highlight], 3000.0, {
        "seconds_before": 4.0, "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 4.0, "diving_save": 9.0},
        "category_post_roll_seconds": {"distribution": 4.0, "diving_save": 11.0},
        "leading_context_absorb_max_gap_seconds": 1.0,
        "interaction_validation": {"enabled": False},
    })

    accepted = [candidate for candidate in result if candidate.accepted]
    assert len(accepted) == 1
    assert accepted[0].candidate_id == "strong-followup"
    assert accepted[0].start == 2029.12
    assert accepted[0].end == 2074.40
    assert "leading-accepted" in accepted[0].merged_from
    assert accepted[0].score_breakdown["leading_context_absorbed"] == 1.0
    assert leading.continuation_absorbed is True
    assert leading.absorbed_into_candidate_id == "strong-followup"


def test_accepted_leading_distribution_with_expanded_planning_tail_still_routes_forward():
    leading = Candidate(
        candidate_id="leading-expanded-tail",
        start=2029.12, end=2038.08, trigger_time=2033.12, min_normalized_distance=0.2699, keeper_track_id=1,
        accepted=True, category="distribution", action_start=2033.12, action_end=2034.08,
        contact_frames=6, ball_confidence=0.714, keeper_motion=0.047, interaction_score=0.20,
        keeper_label="Keeper #1", clip_end_reason="timeout",
    )
    highlight = Candidate(
        candidate_id="strong-followup-expanded-tail",
        start=2043.60, end=2074.40, trigger_time=2055.92, min_normalized_distance=0.0, keeper_track_id=1,
        accepted=True, category="diving_save", action_start=2047.60, action_end=2070.40,
        contact_frames=95, ball_confidence=0.701, keeper_motion=0.713, interaction_score=1.0,
        possession_duration=4.48, approach_speed=0.367, departure_speed=6.95, direction_change=0.997,
        keeper_label="Keeper #1", clip_end_reason="controlled_release",
    )

    result = extend_and_chain_clip_windows([leading, highlight], 3000.0, {
        "seconds_before": 4.0, "seconds_after": 4.0,
        "category_pre_roll_seconds": {"distribution": 4.0, "diving_save": 9.0},
        "category_post_roll_seconds": {"distribution": 12.0, "diving_save": 11.0},
        "leading_context_absorb_max_gap_seconds": 1.0,
        "leading_context_max_post_roll_seconds": 4.0,
        "interaction_validation": {"enabled": False},
    })

    accepted = [candidate for candidate in result if candidate.accepted]
    assert len(accepted) == 1
    assert accepted[0].candidate_id == "strong-followup-expanded-tail"
    assert accepted[0].start == 2029.12
    assert accepted[0].end == 2074.40
    assert "leading-expanded-tail" in accepted[0].merged_from
    assert accepted[0].score_breakdown["leading_context_absorbed"] == 1.0
    assert accepted[0].score_breakdown["leading_context_compact_end"] == 2038.08


def test_final_overlap_with_diagnostic_recovery_keeps_early_10_second_core_like_clip_20():
    candidate = Candidate(
        candidate_id="merged-catch-with-recovery",
        start=1659.12,
        end=1709.0,
        trigger_time=1661.12,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        accepted=True,
        category="catch_or_control",
        action_start=1661.12,
        # The merged recovery/release tail has shifted the effective action end late.
        action_end=1704.0,
        keeper_label="Keeper #1",
        # The real Clip-20 final candidate has already inherited the release
        # reason from the later merged phase. Using dynamic_idle_tail here would
        # trigger the earlier 18 s merged-idle guard and never exercise the
        # final-overlap core that this regression is meant to cover.
        clip_end_reason="controlled_release",
        clip_boundary_reason="final_overlap_merged",
        merged_from=[
            "raw-child-a", "raw-child-b", "raw-child-c", "raw-child-d",
            "diagnostic-recovery-child",
        ],
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
        "catch_control_final_overlap_core_max_seconds": 24.0,
        "catch_control_final_overlap_recovery_core_max_seconds": 10.0,
        "catch_control_final_overlap_recovery_extra_pre_roll_seconds": 1.0,
        "interaction_validation": {"enabled": False},
    })

    assert len(result) == 1
    assert result[0].start == pytest.approx(1684.0)
    assert result[0].end == pytest.approx(1694.0)
    assert result[0].clip_boundary_reason == "final_overlap_recovery_compact_core"
    assert result[0].score_breakdown["final_overlap_recovery_compact_core_applied"] == 1.0


def test_final_overlap_without_diagnostic_recovery_keeps_existing_24_second_core_behavior():
    candidate = Candidate(
        candidate_id="merged-catch-no-recovery",
        start=1659.12,
        end=1709.0,
        trigger_time=1661.12,
        min_normalized_distance=0.1,
        keeper_track_id=1,
        accepted=True,
        category="catch_or_control",
        action_start=1661.12,
        action_end=1704.0,
        keeper_label="Keeper #1",
        # Keep the control fixture on the same final-overlap/release path; the
        # only difference from the case above is the missing diagnostic child.
        clip_end_reason="controlled_release",
        clip_boundary_reason="final_overlap_merged",
        merged_from=["raw-child-a", "raw-child-b", "raw-child-c", "raw-child-d"],
        score_breakdown={"final_overlap_merge_applied": 1.0},
    )

    result = extend_and_chain_clip_windows([candidate], 2000.0, {
        "seconds_before": 4.0,
        "seconds_after": 4.0,
        "category_pre_roll_seconds": {"catch_or_control": 10.0},
        "category_post_roll_seconds": {"catch_or_control": 11.0},
        "continuation_gap_seconds": 12.0,
        "minimum_clip_seconds": 6.0,
        "max_dynamic_clip_seconds": 45.0,
        "catch_control_final_overlap_core_max_seconds": 24.0,
        "interaction_validation": {"enabled": False},
    })

    assert result[0].start == pytest.approx(1685.0)
    assert result[0].end == pytest.approx(1709.0)
    assert result[0].clip_boundary_reason == "final_overlap_compact_core"
    assert "final_overlap_recovery_compact_core_applied" not in result[0].score_breakdown
