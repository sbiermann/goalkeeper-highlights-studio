from __future__ import annotations

import gc
import hashlib
import importlib.util
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, cast

import cv2
import numpy as np
import torch

from .decoder import DecoderItem, PrefetchDecoder, create_decoder
from .models import Box, Candidate
from .event_engine import GoalkeeperEventEngine
from .profiling import PerformanceProfiler
from .keeper_bootstrap import AutomaticGoalkeeperDetector

# v0.13.11: Helper to detect controlled goalkeeper releases (punts, throws, rolls, passes)
def detect_controlled_release(candidate: Candidate, items: list[Candidate], clips_cfg: dict[str, Any]) -> dict[str, Any]:
    """Search for evidence of a controlled release after ball possession.
    
    Returns a dict with detection results for score_breakdown and timing.
    """
    res = {
        "controlled_release_checked": 1.0,
        "controlled_release_detected": 0.0,
        "controlled_release_time": 0.0,
        "controlled_release_possession_duration": candidate.possession_duration,
        "controlled_release_departure_speed": candidate.departure_speed,
        "controlled_release_clamped_by_max_duration": 0.0
    }
    
    if not clips_cfg.get("controlled_release_enabled", True):
        return res
        
    min_possession = float(clips_cfg.get("controlled_release_minimum_possession_seconds", 0.5))
    search_window = float(clips_cfg.get("controlled_release_search_seconds", 12.0))
    min_departure = float(clips_cfg.get("controlled_release_minimum_departure_speed", 0.35))
    
    # We need some initial possession evidence
    if candidate.possession_duration < min_possession:
        return res

    # 1. Check if the candidate itself already has a strong departure signal
    # If category is distribution, we already have some evidence.
    if candidate.departure_speed >= min_departure:
        res["controlled_release_detected"] = 1.0
        res["controlled_release_time"] = candidate.action_end or candidate.trigger_time
        return res

    # 2. Look for subsequent candidates of the same keeper within the search window
    # that might represent the release (e.g. a recovery candidate or a distribution)
    anchor = candidate.action_end or candidate.trigger_time
    for other in items:
        if other.candidate_id == candidate.candidate_id:
            continue
        
        other_start = other.action_start or other.trigger_time
        gap = other_start - anchor
        
        if 0 <= gap <= search_window and other.keeper_label == candidate.keeper_label:
            # If the next event is a distribution or has strong departure, it's our release
            if other.category == "distribution" or other.departure_speed >= min_departure:
                res["controlled_release_detected"] = 1.0
                res["controlled_release_time"] = other.action_end or other.trigger_time
                return res
            
            # If it's a generic recovery with some motion, it could be the release
            if other.recovery_candidate and (other.departure_speed >= 0.25 or other.keeper_motion >= 0.2):
                res["controlled_release_detected"] = 1.0
                res["controlled_release_time"] = other.action_end or other.trigger_time
                return res
                
    return res


def _has_strong_restart_distribution_evidence(candidate: Candidate, clips_cfg: dict[str, Any]) -> bool:
    validation = clips_cfg.get("interaction_validation", {}) or {}
    restart_categories = {"distribution", "goalkeeper_distribution", "keeper_clearance"}
    if candidate.category not in restart_categories:
        return False

    min_contacts = int(validation.get("restart_rescue_min_contact_frames", 20))
    min_possession = float(validation.get("restart_rescue_min_possession_seconds", 1.5))
    min_interaction = float(validation.get("restart_rescue_min_interaction_score", 0.65))
    min_ball_confidence = float(validation.get("restart_rescue_min_ball_confidence", 0.55))
    min_departure = float(validation.get("restart_rescue_min_departure_speed", 1.0))

    strong_release_signal = (
        candidate.departure_speed >= min_departure
        or candidate.clip_end_reason == "controlled_release"
        or float(candidate.score_breakdown.get("controlled_release_detected", 0.0)) > 0.0
    )
    return (
        candidate.contact_frames >= min_contacts
        and candidate.possession_duration >= min_possession
        and candidate.interaction_score >= min_interaction
        and candidate.ball_confidence >= min_ball_confidence
        and strong_release_signal
    )


def _has_contextual_recovery_rescue(candidate: Candidate, clips_cfg: dict[str, Any]) -> bool:
    validation = clips_cfg.get("interaction_validation", {}) or {}
    if not (candidate.recovery_candidate or candidate.category == "recovery_uncovered_activity"):
        return False

    min_event_margin = float(validation.get("recovery_context_rescue_min_event_margin", 0.05))
    min_ball_confidence = float(validation.get("recovery_context_rescue_min_ball_confidence", 0.28))
    max_distance = float(validation.get("recovery_context_rescue_max_distance", 0.85))
    min_keeper_motion = float(validation.get("recovery_context_rescue_min_keeper_motion", 0.18))
    min_contacts = int(validation.get("recovery_context_rescue_min_contact_frames", 1))
    max_window_seconds = float(validation.get("recovery_context_rescue_max_window_seconds", 3.5))

    action_start = candidate.action_start or candidate.trigger_time
    action_end = max(candidate.action_end or candidate.trigger_time, candidate.trigger_time)
    recovery_window_start = candidate.recovery_window_start if candidate.recovery_window_start > 0.0 else action_start
    recovery_window_end = candidate.recovery_window_end if candidate.recovery_window_end > 0.0 else action_end
    recovery_window_span = max(0.0, recovery_window_end - recovery_window_start)
    event_margin = candidate.event_score - candidate.acceptance_threshold

    return (
        event_margin >= min_event_margin
        and candidate.ball_confidence >= min_ball_confidence
        and candidate.min_normalized_distance <= max_distance
        and candidate.keeper_motion >= min_keeper_motion
        and candidate.contact_frames >= min_contacts
        and recovery_window_span <= max_window_seconds
    )

# v0.13.12: Secondary path for recovery candidates with unreliable possession_duration
def find_recovery_distribution_continuation(candidate: Candidate, items: list[Candidate], clips_cfg: dict[str, Any]) -> dict[str, Any]:
    """Find a subsequent distribution candidate that belongs to the same goalkeeper phase.
    
    This handles cases where possession_duration is too low for detect_controlled_release.
    """
    res = {
        "recovery_distribution_checked": 1.0,
        "recovery_distribution_detected": 0.0,
        "recovery_distribution_candidate_found": 0.0,
        "recovery_distribution_candidate_id": "",
        "recovery_distribution_candidate_gap": 0.0,
        "recovery_distribution_candidate_start": 0.0,
        "recovery_distribution_candidate_end": 0.0,
        "recovery_distribution_candidate_accepted": 0.0,
        "recovery_distribution_same_keeper": 0.0,
        "recovery_distribution_original_action_end": candidate.action_end or candidate.trigger_time,
        "recovery_distribution_effective_action_end": candidate.action_end or candidate.trigger_time,
        "recovery_distribution_safety_tail": float(clips_cfg.get("recovery_distribution_safety_tail_seconds", 4.0)),
        "recovery_distribution_clamped_by_max_duration": 0.0,
        "recovery_distribution_absorbed_rejected_candidate": 0.0
    }
    
    if not clips_cfg.get("recovery_distribution_continuation_enabled", True):
        return res
        
    # Only recovery candidates or generic activity need this secondary path
    if not (candidate.recovery_candidate or candidate.category == "recovery_uncovered_activity"):
        return res

    search_window = float(clips_cfg.get("recovery_distribution_search_seconds", 12.0))
    max_gap = float(clips_cfg.get("recovery_distribution_max_gap_seconds", 8.0))
    allow_rejected = bool(clips_cfg.get("recovery_distribution_allow_rejected_candidate", True))
    
    # Plausible distribution categories in this project
    # Based on event_engine.py and common labels
    distribution_categories = {
        "distribution", "goalkeeper_distribution", "goalkeeper_clearance", 
        "punt", "kick", "throw", "roll", "controlled_pass", "pass", "clearance",
        "controlled_release"
    }

    anchor = candidate.action_end or candidate.trigger_time
    best_match = None
    min_found_gap = float('inf')

    for other in items:
        if other.candidate_id == candidate.candidate_id:
            continue
            
        other_start = other.action_start or other.trigger_time
        gap = other_start - anchor
        
        # Must be in future, within search window, and not too huge a gap between actual actions
        if 0 <= gap <= search_window:
            # Same keeper check
            if other.keeper_label != candidate.keeper_label:
                continue
                
            # Category check
            is_dist = other.category in distribution_categories or other.departure_speed >= 0.35
            
            if is_dist:
                # If rejected, only use if explicitly allowed
                if not other.accepted and not allow_rejected:
                    continue
                
                if gap < min_found_gap:
                    min_found_gap = gap
                    best_match = other

    if best_match:
        res["recovery_distribution_detected"] = 1.0
        res["recovery_distribution_candidate_found"] = 1.0
        res["recovery_distribution_candidate_id"] = best_match.candidate_id
        res["recovery_distribution_candidate_gap"] = min_found_gap
        res["recovery_distribution_candidate_start"] = best_match.action_start or best_match.trigger_time
        res["recovery_distribution_candidate_end"] = best_match.action_end or best_match.trigger_time
        res["recovery_distribution_candidate_accepted"] = 1.0 if best_match.accepted else 0.0
        res["recovery_distribution_same_keeper"] = 1.0
        res["recovery_distribution_effective_action_end"] = best_match.action_end or best_match.trigger_time
        if not best_match.accepted:
            res["recovery_distribution_absorbed_rejected_candidate"] = 1.0
            
    return res


def is_valid_recovery_continuation(candidate: Candidate, clips_cfg: dict[str, Any]) -> dict[str, float]:
    """Validate whether a recovery candidate has sufficient keeper/ball continuation evidence."""
    min_interaction = float(clips_cfg.get("recovery_continuation_min_interaction_score", 0.30))
    min_ball_confidence = float(clips_cfg.get("recovery_continuation_min_ball_confidence", 0.25))
    min_possession = float(clips_cfg.get("recovery_continuation_min_possession_seconds", 0.45))
    min_dynamic_signal = float(clips_cfg.get("recovery_continuation_min_dynamic_signal", 0.12))
    min_contact_frames = int(clips_cfg.get("recovery_continuation_min_contact_frames", 2))
    require_ball_dynamics = bool(clips_cfg.get("recovery_continuation_require_ball_dynamics", True))

    approach = float(candidate.approach_speed)
    departure = float(candidate.departure_speed)
    direction = float(candidate.direction_change)
    contact_frames = int(candidate.contact_frames)
    possession_duration = float(candidate.possession_duration)
    interaction_score = float(candidate.interaction_score)
    ball_confidence = float(candidate.ball_confidence)

    ball_dynamics = max(approach, departure, direction)
    has_ball_dynamics = ball_dynamics >= min_dynamic_signal
    has_distribution_signal = (
        candidate.category in {"distribution", "goalkeeper_distribution", "controlled_release", "kick", "throw", "pass", "clearance"}
        or departure >= float(clips_cfg.get("controlled_release_minimum_departure_speed", 0.35))
    )

    weak_signature = (
        contact_frames <= 1
        and possession_duration <= 0.0
        and approach <= 0.0
        and departure <= 0.0
        and direction <= 0.0
        and interaction_score < min_interaction
        and ball_confidence < min_ball_confidence
    )

    variant_a = (contact_frames >= min_contact_frames) and (
        has_ball_dynamics or interaction_score >= min_interaction or ball_confidence >= min_ball_confidence
    )
    variant_b = (possession_duration >= min_possession) and (
        ball_confidence >= min_ball_confidence or has_ball_dynamics or interaction_score >= min_interaction
    )
    variant_c = (
        (approach >= min_dynamic_signal and departure >= min_dynamic_signal) or direction >= min_dynamic_signal
    ) and (
        ball_confidence >= min_ball_confidence or interaction_score >= min_interaction
    )
    variant_d = has_distribution_signal

    valid = (variant_a or variant_b or variant_c or variant_d) and not weak_signature
    if require_ball_dynamics and not has_distribution_signal and not variant_b and not has_ball_dynamics:
        valid = False

    return {
        "recovery_continuation_checked": 1.0,
        "recovery_continuation_valid": 1.0 if valid else 0.0,
        "recovery_continuation_interaction_score": interaction_score,
        "recovery_continuation_contact_frames": float(contact_frames),
        "recovery_continuation_possession_duration": possession_duration,
        "recovery_continuation_ball_dynamics": ball_dynamics,
        "recovery_continuation_keeper_motion": float(candidate.keeper_motion),
        "recovery_continuation_blocked_weak_ball_signal": 1.0 if (not valid and weak_signature) else 0.0,
    }


def apply_dynamic_catch_control_idle_tail(
    candidate: Candidate,
    items: list[Candidate],
    clips_cfg: dict[str, Any],
    duration: float,
) -> dict[str, float]:
    """Shorten catch/control post-roll conservatively with adaptive idle levels."""
    action_end = max(candidate.action_end or candidate.trigger_time, candidate.trigger_time)
    max_post_roll = float(
        clips_cfg.get(
            "catch_control_max_post_roll_seconds",
            clips_cfg.get("category_post_roll_seconds", {}).get("catch_or_control", 11.0),
        )
    )
    idle_tail_low = float(
        clips_cfg.get(
            "catch_control_idle_tail_low_seconds",
            clips_cfg.get("catch_control_idle_tail_seconds", 3.0),
        )
    )
    idle_tail_medium = float(clips_cfg.get("catch_control_idle_tail_medium_seconds", 6.0))
    idle_tail_high = float(clips_cfg.get("catch_control_idle_tail_high_seconds", 7.0))
    medium_min_contact = int(clips_cfg.get("catch_control_medium_min_contact_frames", 4))
    medium_min_possession = float(clips_cfg.get("catch_control_medium_min_possession_seconds", 0.25))
    high_min_contact = int(clips_cfg.get("catch_control_high_min_contact_frames", 10))
    high_min_possession = float(clips_cfg.get("catch_control_high_min_possession_seconds", 1.0))
    high_min_interaction = float(clips_cfg.get("catch_control_high_min_interaction_score", 0.45))
    dynamic_enabled = bool(clips_cfg.get("catch_control_dynamic_post_roll_enabled", True))

    contact_frames = max(0, int(candidate.contact_frames))
    possession_duration = max(0.0, float(candidate.possession_duration))
    interaction_score = max(0.0, float(candidate.interaction_score))

    medium_contact_match = 1.0 if contact_frames >= medium_min_contact else 0.0
    medium_possession_match = 1.0 if possession_duration >= medium_min_possession else 0.0
    high_contact_match = 1.0 if contact_frames >= high_min_contact else 0.0
    high_possession_match = 1.0 if possession_duration >= high_min_possession else 0.0

    idle_level = 1.0
    idle_tail = idle_tail_low
    if (high_contact_match > 0 and high_possession_match > 0) or (
        high_contact_match > 0 and interaction_score >= high_min_interaction
    ):
        idle_level = 3.0
        idle_tail = idle_tail_high
    elif medium_contact_match > 0 and medium_possession_match > 0:
        idle_level = 2.0
        idle_tail = idle_tail_medium

    res = {
        "catch_control_idle_checked": 1.0,
        "catch_control_contact_frames": float(contact_frames),
        "catch_control_possession_duration": possession_duration,
        "catch_control_interaction_score": interaction_score,
        "catch_control_idle_level": idle_level,
        "catch_control_medium_contact_match": medium_contact_match,
        "catch_control_medium_possession_match": medium_possession_match,
        "catch_control_high_contact_match": high_contact_match,
        "catch_control_high_possession_match": high_possession_match,
        "catch_control_idle_tail_low": idle_tail_low,
        "catch_control_idle_tail_medium": idle_tail_medium,
        "catch_control_idle_tail_high": idle_tail_high,
        "catch_control_selected_idle_tail": idle_tail,
        "catch_control_last_activity_time": action_end,
        "catch_control_idle_seconds": 0.0,
        "catch_control_dynamic_post_roll_applied": 0.0,
        "catch_control_original_clip_end": candidate.end,
        "catch_control_effective_clip_end": candidate.end,
        "catch_control_max_post_roll": max_post_roll,
        "catch_control_idle_tail": idle_tail,
    }

    if not dynamic_enabled or candidate.category != "catch_or_control":
        return res
    if candidate.clip_end_reason in {"controlled_release", "recovery_distribution_continuation", "recovery_window_tail"}:
        return res

    search_end = action_end + max_post_roll
    min_interaction = float(clips_cfg.get("recovery_continuation_min_interaction_score", 0.30))
    min_ball_confidence = float(clips_cfg.get("recovery_continuation_min_ball_confidence", 0.25))
    min_dynamic_signal = float(clips_cfg.get("recovery_continuation_min_dynamic_signal", 0.12))

    last_activity = action_end
    for other in items:
        if other.candidate_id == candidate.candidate_id or other.keeper_label != candidate.keeper_label:
            continue
        other_start = other.action_start or other.trigger_time
        if other_start < action_end or other_start > search_end:
            continue

        dynamic_signal = max(float(other.approach_speed), float(other.departure_speed), float(other.direction_change))
        has_keeper_ball_interaction = (
            int(other.contact_frames) >= 2
            or float(other.possession_duration) >= 0.45
            or float(other.interaction_score) >= min_interaction
            or float(other.ball_confidence) >= min_ball_confidence
            or dynamic_signal >= min_dynamic_signal
        )
        same_phase_event = other.accepted and other.category in {
            "distribution",
            "goalkeeper_distribution",
            "keeper_clearance",
            "catch_or_control",
            "cross_claim_or_high_catch",
            "save_or_deflection",
            "diving_save",
            "interaction",
            "recovery_keeper_interaction",
            "recovery_uncovered_activity",
        }
        if has_keeper_ball_interaction or same_phase_event:
            last_activity = max(last_activity, other.action_end or other.trigger_time)

    max_end = min(duration, action_end + max_post_roll)
    idle_end = min(max_end, last_activity + idle_tail)
    effective_end = min(candidate.end, idle_end)
    if effective_end < candidate.end:
        candidate.end = max(action_end, effective_end)
        candidate.clip_end_reason = "dynamic_idle_tail"
        res["catch_control_dynamic_post_roll_applied"] = 1.0

    res["catch_control_last_activity_time"] = last_activity
    res["catch_control_idle_seconds"] = max(0.0, candidate.end - last_activity)
    res["catch_control_effective_clip_end"] = candidate.end
    return res


def apply_recovery_window_tail_fallback(
    candidate: Candidate,
    items: list[Candidate],
    clips_cfg: dict[str, Any],
    duration: float,
    max_duration: float,
    normal_clip_end: float,
    controlled_release_detected: bool,
    recovery_distribution_detected: bool,
) -> dict[str, Any]:
    """Conservative fallback: extend clip_end to existing recovery window boundary evidence."""
    res: dict[str, Any] = {
        "recovery_tail_checked": 1.0,
        "recovery_tail_available": 0.0,
        "recovery_tail_source_count": 0.0,
        "recovery_tail_original_clip_end": normal_clip_end,
        "recovery_tail_window_end": 0.0,
        "recovery_tail_extension_seconds": 0.0,
        "recovery_tail_max_extension_seconds": float(clips_cfg.get("recovery_window_tail_max_extension_seconds", 8.0)),
        "recovery_tail_applied": 0.0,
        "recovery_tail_effective_clip_end": normal_clip_end,
        "recovery_tail_clamped": 0.0,
        "recovery_tail_same_keeper": 0.0,
        "recovery_tail_blocked_by_release": 1.0 if controlled_release_detected else 0.0,
        "recovery_tail_blocked_by_distribution": 1.0 if recovery_distribution_detected else 0.0,
        "recovery_tail_blocked_by_restart": 0.0,
        "recovery_tail_blocked_by_max_duration": 0.0,
    }

    candidate.recovery_tail_reason = "not_checked"
    if not bool(clips_cfg.get("recovery_window_tail_fallback_enabled", True)):
        candidate.recovery_tail_reason = "disabled"
        return res
    if not candidate.accepted:
        candidate.recovery_tail_reason = "candidate_not_accepted"
        return res
    if candidate.category not in {"recovery_uncovered_activity", "recovery_keeper_interaction"} and not candidate.recovery_candidate:
        candidate.recovery_tail_reason = "not_recovery_candidate"
        return res
    if controlled_release_detected:
        candidate.recovery_tail_reason = "blocked_by_controlled_release"
        return res
    if recovery_distribution_detected:
        candidate.recovery_tail_reason = "blocked_by_distribution"
        return res
    if bool(clips_cfg.get("recovery_window_tail_require_timeout", True)) and candidate.clip_end_reason not in {"timeout", "observed_action_window", "chained_keeper_phase", ""}:
        candidate.recovery_tail_reason = "blocked_by_clip_end_reason"
        return res

    anchor = candidate.action_end or candidate.trigger_time
    recovery_sources: list[Candidate] = []
    allowed_ids = {candidate.candidate_id, *candidate.parent_candidate_ids, *candidate.merged_from}
    for other in items:
        if other.keeper_label != candidate.keeper_label:
            continue
        if not (other.recovery_candidate or other.category == "recovery_uncovered_activity"):
            continue
        if other.candidate_id in allowed_ids:
            recovery_sources.append(other)
            continue
        other_start = other.action_start or other.trigger_time
        gap = abs(other_start - anchor)
        if gap <= float(clips_cfg.get("recovery_window_tail_source_search_seconds", 20.0)):
            recovery_sources.append(other)

    if not recovery_sources:
        candidate.recovery_tail_reason = "no_recovery_source"
        return res

    res["recovery_tail_source_count"] = float(len(recovery_sources))
    res["recovery_tail_same_keeper"] = 1.0
    recovery_end = max(
        max(src.recovery_window_end, src.end, src.action_end or src.trigger_time)
        for src in recovery_sources
    )
    res["recovery_tail_window_end"] = recovery_end

    if recovery_end <= normal_clip_end:
        candidate.recovery_tail_reason = "window_not_beyond_normal_end"
        return res

    res["recovery_tail_available"] = 1.0
    extension = recovery_end - normal_clip_end
    res["recovery_tail_extension_seconds"] = extension
    max_extension = float(clips_cfg.get("recovery_window_tail_max_extension_seconds", 8.0))
    if extension > max_extension:
        recovery_end = normal_clip_end + max_extension
        res["recovery_tail_clamped"] = 1.0

    restart_window = float(clips_cfg.get("recovery_window_tail_restart_guard_seconds", 1.2))
    for other in items:
        if other.candidate_id == candidate.candidate_id or other.keeper_label == candidate.keeper_label:
            continue
        other_start = other.action_start or other.trigger_time
        if normal_clip_end < other_start <= recovery_end + restart_window and other.accepted:
            res["recovery_tail_blocked_by_restart"] = 1.0
            candidate.recovery_tail_reason = "blocked_by_other_keeper_activity"
            return res

    effective_end = min(duration, recovery_end + float(clips_cfg.get("recovery_window_tail_safety_seconds", 0.0)))
    if effective_end - candidate.start > max_duration:
        effective_end = candidate.start + max_duration
        res["recovery_tail_blocked_by_max_duration"] = 1.0
        res["recovery_tail_clamped"] = 1.0
    if effective_end <= normal_clip_end:
        candidate.recovery_tail_reason = "blocked_by_max_duration"
        return res

    candidate.end = effective_end
    candidate.clip_end_reason = "recovery_window_tail"
    candidate.recovery_window_end = max(candidate.recovery_window_end, recovery_end)
    candidate.recovery_tail_reason = "applied"
    candidate.lifecycle_events.append({
        "stage": "clip_planning",
        "event": "recovery_window_tail",
        "source_count": len(recovery_sources),
        "original_clip_end": normal_clip_end,
        "recovery_window_end": recovery_end,
        "effective_clip_end": effective_end,
    })
    res["recovery_tail_applied"] = 1.0
    res["recovery_tail_effective_clip_end"] = effective_end
    return res

PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32
ProgressCallback = Callable[[float, str], None]


def boxes_from_result(result: Any) -> list[Box]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes
    if boxes.id is not None:
        packed = torch.cat((boxes.xyxy, boxes.conf.unsqueeze(1), boxes.cls.unsqueeze(1), boxes.id.unsqueeze(1)), dim=1).detach().cpu().numpy()
        return [
            Box(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), int(v[5]), int(v[6]))
            for v in packed
        ]
    packed = torch.cat((boxes.xyxy, boxes.conf.unsqueeze(1), boxes.cls.unsqueeze(1)), dim=1).detach().cpu().numpy()
    return [
        Box(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), int(v[5]), None)
        for v in packed
    ]


def boxes_from_result_legacy(result: Any) -> list[Box]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    ids = result.boxes.id.detach().cpu().numpy().astype(int) if result.boxes.id is not None else None
    return [
        Box(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(conf[i]), int(cls[i]), int(ids[i]) if ids is not None else None)
        for i, v in enumerate(xyxy)
    ]


def normalized_distance(ball: Box, keeper: Box) -> float:
    bx, by = ball.center
    dx = max(keeper.x1 - bx, 0.0, bx - keeper.x2)
    dy = max(keeper.y1 - by, 0.0, by - keeper.y2)
    return math.hypot(dx, dy) / keeper.diagonal


def merge_candidates(items: list[Candidate], gap: float, duration: float) -> list[Candidate]:
    if not items:
        return []
    ordered = sorted(items, key=lambda c: c.start)
    merged = [ordered[0]]
    # v0.13.10 extended merge logic
    # Criteria: same keeper, ball possession continuity, gap <= 2.5s, same phase
    # Note: current gap argument might be smaller than 2.5s, but we use the larger of them for logic
    effective_gap = max(gap, 2.5)
    
    for candidate in ordered[1:]:
        current = merged[-1]
        
        same_keeper = current.keeper_label == candidate.keeper_label
        time_gap = candidate.start - current.end
        
        # Possession check: if both have possession or the gap is very small
        # In the engine, 'distribution', 'catch_or_control' imply possession.
        possession_categories = {"catch_or_control", "distribution", "cross_claim_or_high_catch"}
        
        # v0.13.10: Refined possession flow logic
        # A flow exists if there is continuous possession or a direct transition between candidates
        has_possession_flow = (
            (current.category in possession_categories and candidate.category in possession_categories) or
            (current.category == "catch_or_control" and candidate.category == "distribution") or
            (current.possession_duration > 0.5 and candidate.possession_duration > 0.5 and time_gap < 1.0)
        )
        
        # v0.13.10: Merge if same keeper, within gap, and possession flow
        # If gap is very small (<0.5s), we merge even without explicit possession flow
        if same_keeper and time_gap <= effective_gap and (has_possession_flow or time_gap < 0.5):
            # print(f"[DEBUG_LOG] CONDITION MET: time_gap={time_gap} <= {effective_gap}")
            current.end = min(duration, max(current.end, candidate.end))
            current.action_start = min(current.action_start or current.trigger_time, candidate.action_start or candidate.trigger_time)
            current.action_end = max(current.action_end or current.trigger_time, candidate.action_end or candidate.trigger_time)
            current.contact_frames += candidate.contact_frames
            current.ball_confidence = max(current.ball_confidence, candidate.ball_confidence)
            current.identity_confidence = max(current.identity_confidence, candidate.identity_confidence)
            current.heuristic_score = max(current.heuristic_score, candidate.heuristic_score)
            current.quality_score = max(current.quality_score, candidate.quality_score)
            
            if candidate.event_score > current.event_score:
                current.category = candidate.category
                current.description = candidate.description
                current.event_score = candidate.event_score
                current.approach_speed = candidate.approach_speed
                current.departure_speed = candidate.departure_speed
                current.direction_change = candidate.direction_change
                current.keeper_motion = candidate.keeper_motion
                current.possession_duration = max(current.possession_duration, candidate.possession_duration)
                current.accepted = candidate.accepted
                current.acceptance_threshold = candidate.acceptance_threshold
                current.rejection_reason = candidate.rejection_reason
                current.score_breakdown = dict(candidate.score_breakdown)
                current.possession_bonus = candidate.possession_bonus
                current.cooldown_penalty = candidate.cooldown_penalty
            
            current.recovery_candidate = current.recovery_candidate or candidate.recovery_candidate
            current.recovery_window_start = min(
                (current.recovery_window_start or current.start),
                (candidate.recovery_window_start or candidate.start),
            ) if (current.recovery_window_start or candidate.recovery_window_start) else 0.0
            current.recovery_window_end = max(
                current.recovery_window_end,
                candidate.recovery_window_end,
                current.end,
                candidate.end,
            ) if (current.recovery_candidate or candidate.recovery_candidate) else current.recovery_window_end
            current.parent_candidate_ids = list(dict.fromkeys([*current.parent_candidate_ids, current.candidate_id, *candidate.parent_candidate_ids, candidate.candidate_id]))
            current.merged_from.append(candidate.candidate_id)
            current.merged_reason = "same_keeper_possession_flow" if has_possession_flow else "same_keeper_within_merge_window"
            current.merged_duration = current.end - current.start
            
            current.lifecycle_events.append({
                "stage": "merge", 
                "merged_candidate_id": candidate.candidate_id, 
                "reason": current.merged_reason
            })
            
            if candidate.min_normalized_distance < current.min_normalized_distance:
                current.min_normalized_distance = candidate.min_normalized_distance
                current.trigger_time = candidate.trigger_time
                current.keeper_track_id = candidate.keeper_track_id
        else:
            merged.append(candidate)
    return merged



def _has_real_keeper_interaction(candidate: Candidate, clips_cfg: dict[str, Any]) -> bool:
    """Reject geometrically implausible long contacts caused by a bad ball track."""
    validation = clips_cfg.get("interaction_validation", {}) or {}
    if not bool(validation.get("enabled", True)):
        return True
    
    extreme_frames = int(validation.get("extreme_contact_frames", 80))
    suspicious_frames = int(validation.get("suspicious_contact_frames", 12))
    motion_floor = float(validation.get("minimum_motion_signal", 0.08))
    
    # Interaction Score V2
    # Base components: contact frames, trajectory dynamics, keeper motion
    # Derive from existing scores
    dynamics = max(candidate.approach_speed, candidate.departure_speed, candidate.direction_change)
    
    # Calculate interaction score [0.0 - 1.0]
    # More weight on direction change and approach for saves
    interaction_score = (
        min(1.0, candidate.contact_frames / 10.0) * 0.3 +
        min(1.0, dynamics / 0.5) * 0.4 +
        min(1.0, candidate.keeper_motion / 0.5) * 0.3
    )
    candidate.interaction_score = interaction_score

    static_contact = candidate.contact_frames >= suspicious_frames and dynamics < motion_floor and candidate.keeper_motion < motion_floor
    
    # New logic: genuine interaction requires either dynamics or enough contact frames with some motion
    # or it's a confirmed possession
    genuine_interaction = (
        candidate.contact_frames >= 2 and 
        (dynamics >= motion_floor or candidate.keeper_motion >= motion_floor or candidate.possession_duration >= 0.5)
    )
    
    central_min = float(validation.get("central_field_y_min", 0.36))
    central_max = float(validation.get("central_field_y_max", 0.64))
    irrelevant_restart = (
        candidate.category in {"distribution", "catch_or_control", "interaction"}
        and candidate.possession_duration >= float(validation.get("outside_box_restart_min_seconds", 2.5))
        and central_min <= candidate.keeper_y_normalized <= central_max
        and candidate.approach_speed < motion_floor
        and candidate.direction_change < motion_floor
    )
    
    if not genuine_interaction:
        if candidate.recovery_candidate:
            # Stricter validation for recovery candidates: single frame requires high dynamics
            # If interaction_score is low, reject even if it's a recovery candidate
            if interaction_score < float(validation.get("minimum_recovery_interaction_score", 0.45)):
                if _has_contextual_recovery_rescue(candidate, clips_cfg):
                    candidate.score_breakdown["recovery_contextual_rescue_applied"] = 1.0
                    candidate.score_breakdown["interaction_validation"] = 1.0
                    return True
                candidate.accepted = False
                candidate.rejection_reason = "insufficient_recovery_interaction_score"
                candidate.score_breakdown["interaction_validation"] = -1.0
                return False
        else:
            candidate.accepted = False
            candidate.rejection_reason = "insufficient_interaction_dynamics"
            candidate.score_breakdown["interaction_validation"] = -1.0
            return False

    if irrelevant_restart and _has_strong_restart_distribution_evidence(candidate, clips_cfg):
        candidate.score_breakdown["restart_relevance_rescue_applied"] = 1.0
        candidate.score_breakdown["interaction_validation"] = 1.0
        return True

    if not static_contact and not irrelevant_restart and candidate.contact_frames < extreme_frames:
        return True
    
    if not static_contact and not irrelevant_restart and (dynamics >= motion_floor or candidate.keeper_motion >= motion_floor):
        return True

    candidate.accepted = False
    candidate.rejection_reason = "irrelevant_outside_box_restart" if irrelevant_restart else "implausible_static_long_contact"
    candidate.score_breakdown["interaction_validation"] = -1.0
    return False


def recover_missed_keeper_actions(store, existing: list[Candidate], duration: float, config: dict[str, Any]) -> list[Candidate]:
    """Generic second pass for missed fast saves and controls.

    The pass deliberately requires visible dynamics. A stationary ball overlapping
    the goalkeeper is not enough, while a single sampled frame may be recovered
    when the incoming ball or goalkeeper movement is sufficiently strong.
    """
    recovery_cfg = (config.get("event_engine", {}).get("recovery_pass", {}) or {})
    if store is None or not bool(recovery_cfg.get("enabled", True)):
        return []
    rows = store.recovery_observations()
    if not rows:
        return []
    min_ball = float(recovery_cfg.get("minimum_ball_confidence", 0.28))
    close_limit = float(recovery_cfg.get("maximum_normalized_distance", 0.85))
    motion_limit = float(recovery_cfg.get("minimum_keeper_motion", 0.10))
    ball_motion_limit = float(recovery_cfg.get("minimum_ball_motion", 0.35))
    approach_limit = float(recovery_cfg.get("minimum_approach_rate", 0.30))
    min_frames = int(recovery_cfg.get("minimum_close_frames", 2))
    single_frame_confidence = float(recovery_cfg.get("single_frame_min_ball_confidence", 0.50))
    group_gap = float(recovery_cfg.get("group_gap_seconds", 1.25))
    mask_margin = float(recovery_cfg.get("existing_candidate_mask_seconds", 1.0))
    before = float(recovery_cfg.get("seconds_before", 8.0))
    after = float(recovery_cfg.get("seconds_after", 9.0))

    best_by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        if float(row["ball_confidence"]) < min_ball:
            continue
        best_by_frame.setdefault(int(row["frame_index"]), row)

    signals: list[tuple[float, float, float, float, float, dict[str, Any]]] = []
    previous = None
    previous_distance = None
    for row in best_by_frame.values():
        kw, kh = max(1.0, row["kx2"]-row["kx1"]), max(1.0, row["ky2"]-row["ky1"])
        diag = math.hypot(kw, kh)
        bx, by = (row["bx1"]+row["bx2"])/2, (row["by1"]+row["by2"])/2
        dx = max(row["kx1"]-bx, 0.0, bx-row["kx2"])
        dy = max(row["ky1"]-by, 0.0, by-row["ky2"])
        distance = math.hypot(dx, dy)/diag
        kx, ky = (row["kx1"]+row["kx2"])/2, (row["ky1"]+row["ky2"])/2
        keeper_motion = ball_motion = approach = 0.0
        if previous is not None:
            dt = max(0.04, float(row["timestamp"])-float(previous["timestamp"]))
            pkx, pky = (previous["kx1"]+previous["kx2"])/2, (previous["ky1"]+previous["ky2"])/2
            pbx, pby = (previous["bx1"]+previous["bx2"])/2, (previous["by1"]+previous["by2"])/2
            keeper_motion = math.hypot(kx-pkx, ky-pky)/diag/dt
            ball_motion = math.hypot(bx-pbx, by-pby)/diag/dt
            if previous_distance is not None:
                approach = max(0.0, previous_distance-distance)/dt
        previous = row
        previous_distance = distance
        dynamic = keeper_motion >= motion_limit or ball_motion >= ball_motion_limit or approach >= approach_limit
        if distance <= close_limit and dynamic:
            signals.append((float(row["timestamp"]), distance, keeper_motion, ball_motion, approach, row))

    groups: list[list[tuple[float, float, float, float, float, dict[str, Any]]]] = []
    for signal in signals:
        if not groups or signal[0]-groups[-1][-1][0] > group_gap:
            groups.append([signal])
        else:
            groups[-1].append(signal)

    recovered: list[Candidate] = []
    accepted_mask = [c for c in existing if c.accepted]
    for group in groups:
        max_keeper_motion = max(v[2] for v in group)
        max_ball_motion = max(v[3] for v in group)
        max_approach = max(v[4] for v in group)
        avg_ball = sum(float(v[5]["ball_confidence"]) for v in group)/len(group)
        single_frame_dynamic = (
            len(group) == 1 and avg_ball >= single_frame_confidence
            and (max_approach >= approach_limit*1.5 or max_ball_motion >= ball_motion_limit*1.5 or max_keeper_motion >= motion_limit*1.5)
        )
        if len(group) < min_frames and not single_frame_dynamic:
            continue
        start_t, end_t = group[0][0], group[-1][0]
        if any(start_t <= c.end + mask_margin and end_t >= c.start - mask_margin for c in accepted_mask):
            continue
        min_distance = min(v[1] for v in group)
        score = min(0.82, 0.36 + 0.18*(1-min_distance/close_limit)
                    + 0.10*min(1.0, max_keeper_motion/max(motion_limit, .01))
                    + 0.10*min(1.0, max_ball_motion/max(ball_motion_limit, .01))
                    + 0.10*min(1.0, max_approach/max(approach_limit, .01)) + 0.06*avg_ball)
        recovered.append(Candidate(
            start=max(0.0, start_t-before), end=min(duration, end_t+after), trigger_time=start_t,
            min_normalized_distance=min_distance, keeper_track_id=group[0][5].get("keeper_track_id"),
            accepted=True, category="recovery_keeper_interaction", confidence=score,
            description="Zusätzlicher dynamischer Suchlauf: mögliche verpasste Parade oder Torwartaktion.",
            identity_confidence=0.75, contact_frames=len(group), ball_confidence=avg_ball,
            heuristic_score=score, quality_score=score, event_score=score, acceptance_threshold=0.40,
            score_breakdown={"recovery_proximity": 1-min_distance/close_limit, "recovery_keeper_motion": max_keeper_motion,
                             "recovery_ball_motion": max_ball_motion, "recovery_approach": max_approach,
                             "recovery_frames": len(group)},
            action_start=max(0.0, start_t-before), action_end=end_t,
            clip_boundary_reason="dynamic_recovery_pass", recovery_candidate=True,
        ))
    return recovered


def recover_uncovered_activity_windows(store, existing: list[Candidate], duration: float, config: dict[str, Any]) -> list[Candidate]:
    """Turn strong uncovered diagnostic activity into conservative recovery candidates.

    This path is deliberately independent from a continuous ByteTrack id. It uses the
    logical keeper box recorded for each processed frame and tolerates sparse ball
    detections. It is intended for short saves that disappear before the normal event
    engine can collect enough consecutive contact frames.
    """
    cfg = (config.get("event_engine", {}).get("recovery_pass", {}) or {})
    if store is None or not bool(cfg.get("diagnostic_window_recovery_enabled", True)):
        return []
    rows = store.recovery_observations()
    if not rows:
        return []
    bucket_seconds = max(0.5, float(config.get("diagnostics", {}).get("window_seconds", 2.0)))
    min_score = float(cfg.get("diagnostic_min_score", 0.42))
    min_ball = float(cfg.get("diagnostic_min_ball_confidence", 0.20))
    min_motion = float(cfg.get("diagnostic_min_keeper_motion", 0.03))
    max_distance = float(cfg.get("diagnostic_max_distance", 1.45))
    group_gap = float(cfg.get("diagnostic_group_gap_seconds", 2.5))
    mask = float(cfg.get("existing_candidate_mask_seconds", 1.0))
    before = float(cfg.get("seconds_before", 8.0))
    after = float(cfg.get("seconds_after", 9.0))

    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(int(float(row["timestamp"]) // bucket_seconds), []).append(row)

    signals: list[dict[str, Any]] = []
    previous_center: tuple[float, float] | None = None
    for bucket, group in sorted(buckets.items()):
        start = bucket * bucket_seconds
        end = start + bucket_seconds
        if any(c.start - mask <= end and c.end + mask >= start for c in existing):
            continue
        centers: list[tuple[float, float]] = []
        ball_conf = 0.0
        min_distance = math.inf
        best_track = None
        for row in group:
            kx = (float(row["kx1"]) + float(row["kx2"])) / 2
            ky = (float(row["ky1"]) + float(row["ky2"])) / 2
            bx = (float(row["bx1"]) + float(row["bx2"])) / 2
            by = (float(row["by1"]) + float(row["by2"])) / 2
            diag = max(1.0, math.hypot(float(row["kx2"]) - float(row["kx1"]), float(row["ky2"]) - float(row["ky1"])))
            centers.append((kx, ky))
            ball_conf = max(ball_conf, float(row["ball_confidence"] or 0.0))
            min_distance = min(min_distance, math.hypot(kx - bx, ky - by) / diag)
            best_track = row.get("keeper_track_id")
        motion = 0.0
        if centers:
            motion = math.hypot(centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1]) / 1000.0
            if previous_center is not None:
                motion = max(motion, math.hypot(centers[0][0] - previous_center[0], centers[0][1] - previous_center[1]) / 1000.0)
            previous_center = centers[-1]
        score = min(1.0, ball_conf * 0.48 + min(motion * 4.0, 1.0) * 0.32 + (1.0 / (1.0 + min_distance)) * 0.20)
        if ball_conf >= min_ball and min_distance <= max_distance and (motion >= min_motion or score >= min_score):
            signals.append({"start": start, "end": end, "score": score, "ball": ball_conf, "motion": motion,
                            "distance": min_distance, "track": best_track, "observations": len(group)})

    grouped: list[list[dict[str, Any]]] = []
    for signal in signals:
        if not grouped or signal["start"] - grouped[-1][-1]["end"] > group_gap:
            grouped.append([signal])
        else:
            grouped[-1].append(signal)

    recovered: list[Candidate] = []
    for index, group in enumerate(grouped, 1):
        start_t = group[0]["start"]
        end_t = group[-1]["end"]
        score = max(item["score"] for item in group)
        ball = max(item["ball"] for item in group)
        motion = max(item["motion"] for item in group)
        distance = min(item["distance"] for item in group)
        candidate = Candidate(
            start=max(0.0, start_t-before), end=min(duration, end_t+after), trigger_time=start_t,
            min_normalized_distance=distance, keeper_track_id=group[0]["track"], accepted=True,
            category="recovery_uncovered_activity", confidence=score,
            description="Nicht abgedeckte Ball-/Torwartaktivität aus dem Diagnosepfad als Recovery-Kandidat übernommen.",
            identity_confidence=0.65, contact_frames=sum(item["observations"] for item in group),
            ball_confidence=ball, heuristic_score=score, quality_score=score, event_score=score,
            acceptance_threshold=min_score, keeper_motion=motion,
            score_breakdown={"diagnostic_window_score": score, "diagnostic_ball_confidence": ball,
                             "diagnostic_keeper_motion": motion, "diagnostic_min_distance": distance,
                             "diagnostic_window_count": len(group)},
            action_start=start_t, action_end=end_t, clip_boundary_reason="uncovered_activity_recovery",
            recovery_candidate=True, recovery_window_start=start_t, recovery_window_end=end_t,
            candidate_id=f"diagnostic-recovery-{index:04d}",
            lifecycle_stage="recovered", lifecycle_reason="uncovered_suspicious_window",
        )
        recovered.append(candidate)
    return recovered


def rescue_classic_keeper_actions(items: list[Candidate], clips_cfg: dict[str, Any]) -> list[Candidate]:
    """Promote reliable classic keeper contacts that narrowly missed scoring.

    This is intentionally limited to score-threshold rejections. Cooldown,
    implausible static contacts and central-field restarts remain rejected.
    """
    cfg = clips_cfg.get("classic_action_rescue", {}) or {}
    if not bool(cfg.get("enabled", True)):
        return items
    allowed = set(cfg.get("categories", ["catch_or_control", "cross_claim_or_high_catch", "save_or_deflection", "diving_save", "keeper_clearance", "ball_contact", "interaction"]))
    min_contacts = int(cfg.get("minimum_contact_frames", 2))
    max_distance = float(cfg.get("maximum_normalized_distance", 0.55))
    min_identity = float(cfg.get("minimum_identity_confidence", 0.58))
    min_ball = float(cfg.get("minimum_ball_confidence", 0.25))
    min_dynamic = float(cfg.get("minimum_dynamic_signal", 0.12))
    for candidate in items:
        if candidate.accepted or candidate.rejection_reason != "event_score_below_category_threshold" or candidate.category not in allowed:
            continue
        dynamic = max(candidate.approach_speed, candidate.departure_speed, candidate.direction_change, candidate.keeper_motion)
        reliable_contact = (candidate.contact_frames >= min_contacts
                            and candidate.min_normalized_distance <= max_distance
                            and candidate.identity_confidence >= min_identity
                            and candidate.ball_confidence >= min_ball
                            and dynamic >= min_dynamic)
        if reliable_contact:
            candidate.accepted = True
            candidate.rejection_reason = ""
            candidate.event_score = max(candidate.event_score, candidate.acceptance_threshold + 0.01)
            candidate.confidence = max(candidate.confidence, candidate.event_score)
            candidate.quality_score = max(candidate.quality_score, candidate.event_score)
            candidate.score_breakdown["classic_action_rescue"] = 1.0
            candidate.description = (candidate.description + " Klassische Torwartaktion durch bestätigten dynamischen Kontakt gerettet.").strip()
    return items


def _clip_end_reason_priority(reason: str) -> int:
    priorities = {
        "controlled_release": 4,
        "recovery_distribution_continuation": 3,
        "recovery_window_tail": 2,
        "timeout": 1,
    }
    return priorities.get(reason or "", 0)


def _merge_unique_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def _candidate_action_bounds(candidate: Candidate) -> tuple[float, float]:
    action_start = candidate.action_start or candidate.trigger_time
    action_end = candidate.action_end or candidate.trigger_time
    return action_start, max(action_start, action_end)


def _candidate_relevance_score(candidate: Candidate) -> float:
    """Generalized relevance score used for conservative weak-phase absorption."""
    interaction = max(0.0, float(candidate.interaction_score))
    contact = min(12.0, max(0.0, float(candidate.contact_frames))) / 12.0
    possession = min(2.0, max(0.0, float(candidate.possession_duration))) / 2.0
    ball_confidence = max(0.0, min(1.0, float(candidate.ball_confidence)))
    recovery_bonus = 0.15 if (candidate.recovery_candidate or candidate.category.startswith("recovery_")) else 0.0
    accepted_bonus = 0.10 if candidate.accepted else 0.0
    return (interaction * 1.8) + (contact * 1.1) + (possession * 0.9) + (ball_confidence * 0.4) + recovery_bonus + accepted_bonus


def _has_independent_restart_between(previous: Candidate, current: Candidate) -> bool:
    restart_categories = {"distribution", "keeper_clearance"}
    previous_cat = previous.category
    current_cat = current.category
    if current_cat in restart_categories and previous_cat not in {"catch_or_control", "cross_claim_or_high_catch", "recovery_uncovered_activity"}:
        return True
    if previous_cat in restart_categories and current_cat in restart_categories:
        return True
    return False


def trim_context_to_duration_limit(
    union_start: float,
    union_end: float,
    action_start: float,
    action_end: float,
    duration_limit: float,
    min_pre_roll: float,
    min_post_roll: float,
) -> tuple[float, float, float, float, float, float, float, bool, bool]:
    """Trim only outer context while preserving full action span.

    Returns:
        (trimmed_start, trimmed_end, action_duration, original_pre_roll,
         original_post_roll, effective_pre_roll, effective_post_roll,
         context_trimmed, can_fit)
    """
    bounded_action_end = max(action_start, action_end)
    action_duration = max(0.0, bounded_action_end - action_start)
    original_pre_roll = max(0.0, action_start - union_start)
    original_post_roll = max(0.0, union_end - bounded_action_end)
    union_duration = max(0.0, union_end - union_start)

    if union_duration <= duration_limit:
        return (
            union_start,
            union_end,
            action_duration,
            original_pre_roll,
            original_post_roll,
            original_pre_roll,
            original_post_roll,
            False,
            True,
        )

    min_context_duration = action_duration + min_pre_roll + min_post_roll
    if min_context_duration > duration_limit:
        return (
            union_start,
            union_end,
            action_duration,
            original_pre_roll,
            original_post_roll,
            original_pre_roll,
            original_post_roll,
            False,
            False,
        )

    effective_pre_roll = original_pre_roll
    effective_post_roll = original_post_roll
    over_limit = union_duration - duration_limit

    reducible_pre = max(0.0, effective_pre_roll - min_pre_roll)
    reduce_pre = min(reducible_pre, over_limit)
    effective_pre_roll -= reduce_pre
    over_limit -= reduce_pre

    reducible_post = max(0.0, effective_post_roll - min_post_roll)
    reduce_post = min(reducible_post, over_limit)
    effective_post_roll -= reduce_post
    over_limit -= reduce_post

    if over_limit > 1e-6:
        return (
            union_start,
            union_end,
            action_duration,
            original_pre_roll,
            original_post_roll,
            effective_pre_roll,
            effective_post_roll,
            False,
            False,
        )

    trimmed_start = action_start - effective_pre_roll
    trimmed_end = bounded_action_end + effective_post_roll
    context_trimmed = (effective_pre_roll < original_pre_roll) or (effective_post_roll < original_post_roll)

    return (
        trimmed_start,
        trimmed_end,
        action_duration,
        original_pre_roll,
        original_post_roll,
        effective_pre_roll,
        effective_post_roll,
        context_trimmed,
        True,
    )


def merge_overlapping_final_clips(items: list[Candidate], duration: float, clips_cfg: dict[str, Any]) -> list[Candidate]:
    if not items:
        return []
    if not bool(clips_cfg.get("final_overlap_merge_enabled", True)):
        return items

    min_ratio = max(0.0, min(1.0, float(clips_cfg.get("final_overlap_merge_min_ratio", 0.60))))
    max_gap = max(0.0, float(clips_cfg.get("final_overlap_merge_max_gap_seconds", 1.0)))
    require_same_keeper = bool(clips_cfg.get("final_overlap_merge_require_same_keeper", True))
    max_duration = max(1.0, float(clips_cfg.get("max_dynamic_clip_seconds", 45.0)))
    duration_tolerance = float(clips_cfg.get("phase_merge_duration_tolerance", 0.08))
    duration_limit = max_duration * (1.0 + duration_tolerance)
    min_pre_roll = max(0.0, float(clips_cfg.get("phase_merge_min_pre_roll_seconds", 2.0)))
    min_post_roll = max(0.0, float(clips_cfg.get("phase_merge_min_post_roll_seconds", 2.0)))

    ordered = sorted(items, key=lambda c: (c.start, c.end, c.trigger_time))
    merged: list[Candidate] = []

    for candidate in ordered:
        if candidate.score_breakdown is None:
            candidate.score_breakdown = {}

        if not merged:
            merged.append(candidate)
            continue

        previous = merged[-1]
        if previous.score_breakdown is None:
            previous.score_breakdown = {}

        if not previous.accepted or not candidate.accepted:
            merged.append(candidate)
            continue

        previous_duration = max(0.001, previous.end - previous.start)
        current_duration = max(0.001, candidate.end - candidate.start)
        overlap_seconds = max(0.0, min(previous.end, candidate.end) - max(previous.start, candidate.start))
        shorter_duration = min(previous_duration, current_duration)
        overlap_shorter_ratio = overlap_seconds / shorter_duration if shorter_duration > 0 else 0.0
        union_start = min(previous.start, candidate.start)
        union_end = max(previous.end, candidate.end)
        union_duration = max(0.001, union_end - union_start)
        overlap_iou = overlap_seconds / union_duration if union_duration > 0 else 0.0
        gap = candidate.start - previous.end
        same_keeper = previous.keeper_label == candidate.keeper_label
        restart_detected = _has_independent_restart_between(previous, candidate)
        prev_action_start, prev_action_end = _candidate_action_bounds(previous)
        cand_action_start, cand_action_end = _candidate_action_bounds(candidate)
        action_span_start = min(prev_action_start, cand_action_start)
        action_span_end = max(prev_action_end, cand_action_end)
        (
            trimmed_start,
            trimmed_end,
            action_duration,
            original_pre_roll,
            original_post_roll,
            effective_pre_roll,
            effective_post_roll,
            context_trimmed,
            can_fit_with_trimming,
        ) = trim_context_to_duration_limit(
            union_start,
            union_end,
            action_span_start,
            action_span_end,
            duration_limit,
            min_pre_roll,
            min_post_roll,
        )
        trimmed_duration = max(0.001, trimmed_end - trimmed_start)

        candidate.score_breakdown.update({
            "final_overlap_checked": 1.0,
            "final_overlap_seconds": overlap_seconds,
            "final_overlap_shorter_ratio": overlap_shorter_ratio,
            "final_overlap_iou": overlap_iou,
            "final_overlap_same_keeper": 1.0 if same_keeper else 0.0,
            "final_overlap_restart_detected": 1.0 if restart_detected else 0.0,
            "final_overlap_gap": gap,
            "final_overlap_original_union_duration": union_duration,
            "final_overlap_action_duration": action_duration,
            "final_overlap_original_pre_roll": original_pre_roll,
            "final_overlap_original_post_roll": original_post_roll,
            "final_overlap_effective_pre_roll": effective_pre_roll,
            "final_overlap_effective_post_roll": effective_post_roll,
            "final_overlap_trimmed_duration": trimmed_duration,
            "final_overlap_context_trimmed": 1.0 if context_trimmed else 0.0,
            "final_overlap_union_duration": union_duration,
            "final_overlap_duration_limit": duration_limit,
            "final_overlap_merge_applied": 0.0,
        })

        if require_same_keeper and not same_keeper:
            merged.append(candidate)
            continue
        if restart_detected:
            merged.append(candidate)
            continue
        if not can_fit_with_trimming:
            merged.append(candidate)
            continue

        strong_overlap = overlap_seconds > 0.0 and overlap_shorter_ratio >= min_ratio
        small_gap_continuation = overlap_seconds <= 0.0 and gap <= max_gap
        if not strong_overlap and not small_gap_continuation:
            merged.append(candidate)
            continue

        previous_end_before_merge = previous.end
        previous.start = max(0.0, trimmed_start)
        previous.end = min(duration, trimmed_end)
        previous.action_start = action_span_start
        previous.action_end = action_span_end
        previous.recovery_window_start = min(
            (x for x in [previous.recovery_window_start, candidate.recovery_window_start] if x > 0.0),
            default=0.0,
        )
        previous.recovery_window_end = max(previous.recovery_window_end, candidate.recovery_window_end)
        previous.contact_frames += candidate.contact_frames
        previous.ball_confidence = max(previous.ball_confidence, candidate.ball_confidence)
        previous.identity_confidence = max(previous.identity_confidence, candidate.identity_confidence)
        previous.clip_boundary_reason = "final_overlap_merged"
        previous.phase_merge_reason = previous.phase_merge_reason or "same_keeper_related_phase"
        previous.merged_reason = "same_keeper_high_overlap"

        primary_reason = previous.clip_end_reason
        secondary_reason = candidate.clip_end_reason
        primary_priority = _clip_end_reason_priority(primary_reason)
        secondary_priority = _clip_end_reason_priority(secondary_reason)
        if secondary_priority > primary_priority:
            previous.clip_end_reason = secondary_reason
        elif secondary_priority == primary_priority and candidate.end >= previous_end_before_merge:
            previous.clip_end_reason = secondary_reason

        previous.merged_from = _merge_unique_ids(
            previous.merged_from + [candidate.candidate_id] + candidate.merged_from
        )
        previous.parent_candidate_ids = _merge_unique_ids(
            previous.parent_candidate_ids + [candidate.candidate_id] + candidate.parent_candidate_ids
        )
        previous.lifecycle_events.append(
            {
                "stage": "final_overlap_merge",
                "reason": "same_keeper_high_overlap",
                "merged_candidate_id": candidate.candidate_id,
            }
        )
        previous.lifecycle_reason = "same_keeper_high_overlap"

        previous.score_breakdown.update(
            {
                "final_overlap_checked": 1.0,
                "final_overlap_seconds": overlap_seconds,
                "final_overlap_shorter_ratio": overlap_shorter_ratio,
                "final_overlap_iou": overlap_iou,
                "final_overlap_same_keeper": 1.0 if same_keeper else 0.0,
                "final_overlap_restart_detected": 1.0 if restart_detected else 0.0,
                "final_overlap_gap": gap,
                "final_overlap_original_union_duration": union_duration,
                "final_overlap_action_duration": action_duration,
                "final_overlap_original_pre_roll": original_pre_roll,
                "final_overlap_original_post_roll": original_post_roll,
                "final_overlap_effective_pre_roll": effective_pre_roll,
                "final_overlap_effective_post_roll": effective_post_roll,
                "final_overlap_trimmed_duration": trimmed_duration,
                "final_overlap_context_trimmed": 1.0 if context_trimmed else 0.0,
                "final_overlap_union_duration": trimmed_duration,
                "final_overlap_duration_limit": duration_limit,
                "final_overlap_merge_applied": 1.0,
            }
        )

        candidate.continuation_absorbed = True
        candidate.absorbed_into_candidate_id = previous.candidate_id
        candidate.continuation_absorb_reason = "final_overlap_merge_absorbed"

    return merged

def extend_and_chain_clip_windows(items: list[Candidate], duration: float, clips_cfg: dict[str, Any]) -> list[Candidate]:
    """Plan clips around observed action boundaries instead of fixed trigger windows.

    ``action_start``/``action_end`` come directly from the temporal event engine.
    Small category-specific context margins are added around that observed action.
    Closely following keeper events can extend a phase of play, but an isolated
    distribution no longer receives a large generic tail.
    """
    if not items:
        return []

    # Compatibility for cached/pre-0.13 candidates that do not yet contain
    # observed action boundaries. Preserve their existing windows and legacy
    # tail/chaining semantics instead of fabricating action timestamps.
    if all(not c.action_start and not c.action_end for c in items):
        idle_tail = max(0.0, float(clips_cfg.get("activity_tail_seconds", 0.0)))
        continuation_gap = max(0.0, float(clips_cfg.get("continuation_gap_seconds", 15.0)))
        final_tail = max(0.0, float(clips_cfg.get("final_keeper_contact_tail_seconds", 8.0)))
        max_duration = max(1.0, float(clips_cfg.get("max_dynamic_clip_seconds", 40.0)))
        legacy: list[Candidate] = []
        for candidate in sorted(items, key=lambda c: c.start):
            original_end = candidate.end
            if candidate.accepted:
                candidate.end = min(duration, candidate.end + idle_tail)
            if legacy and candidate.accepted and legacy[-1].accepted and candidate.start <= legacy[-1].end + continuation_gap:
                current = legacy[-1]
                current.end = min(duration, max(current.end, original_end + final_tail), current.start + max_duration)
                current.score_breakdown["chained_event_count"] = int(current.score_breakdown.get("chained_event_count", 1)) + 1
            else:
                legacy.append(candidate)
        return legacy

    default_before = max(0.0, float(clips_cfg.get("seconds_before", 5.0)))
    default_after = max(0.0, float(clips_cfg.get("seconds_after", 4.0)))
    category_before = clips_cfg.get("category_pre_roll_seconds", {}) or {}
    category_after = clips_cfg.get("category_post_roll_seconds", {}) or {}
    continuation_gap = max(0.0, float(clips_cfg.get("continuation_gap_seconds", 12.0)))
    final_tail = max(0.0, float(clips_cfg.get("final_keeper_contact_tail_seconds", 4.0)))
    max_duration = max(1.0, float(clips_cfg.get("max_dynamic_clip_seconds", 45.0)))
    minimum_duration = max(1.0, float(clips_cfg.get("minimum_clip_seconds", 6.0)))

    ordered = sorted(items, key=lambda c: (c.action_start or c.trigger_time, c.trigger_time))
    planned: list[Candidate] = []
    
    # Stufe B: Phase Merging
    # We first process the regular chaining, then we check for larger gaps
    # with activity in between.
        
    def same_keeper_phase(previous: Candidate, current: Candidate) -> bool:
        # Do not glue unrelated restarts together merely because they happen
        # within the generic continuation window. A distribution may continue a
        # catch/control, but an isolated goal kick starts its own clip.
        restart_categories = {"distribution", "keeper_clearance"}
        
        # v0.13.11: Support recovery categories in chaining logic
        previous_cat = previous.category
        current_cat = current.category
        
        if current_cat in restart_categories and previous_cat not in {"catch_or_control", "cross_claim_or_high_catch", "recovery_uncovered_activity"}:
            return False
        if previous_cat in restart_categories and current_cat in restart_categories:
            return False
        return True
    for candidate in ordered:
        # v0.13.11: Ensure score_breakdown is initialized before use
        if candidate.score_breakdown is None:
            candidate.score_breakdown = {}
        
        # print(f"[DEBUG_LOG] Processing candidate {candidate.candidate_id}, accepted={candidate.accepted}, category={candidate.category}")
            
        _has_real_keeper_interaction(candidate, clips_cfg)
        # print(f"[DEBUG_LOG] After interaction check: {candidate.candidate_id}, accepted={candidate.accepted}")
        if not candidate.accepted:
            # v0.13.11: Diagnostics for rejected candidates too
            release_res = detect_controlled_release(candidate, ordered, clips_cfg)
            candidate.score_breakdown.update(release_res)
            planned.append(candidate)
            continue
            
        action_start = candidate.action_start or candidate.trigger_time
        action_end = max(candidate.action_end or candidate.trigger_time, candidate.trigger_time)
        before = max(0.0, float(category_before.get(candidate.category, default_before)))
        after = max(0.0, float(category_after.get(candidate.category, default_after)))
        
        # v0.13.10 Dynamic Clip End
        # If keeper controlled the ball, extend until restart (kick/throw) is detected
        # In the engine, 'distribution' already implies a departure (kick/throw)
        # We can use the departure_speed as a proxy for kick/throw in distribution category.
        if candidate.category == "distribution" and candidate.departure_speed > 0.35:
            candidate.clip_end_reason = "kick" if candidate.approach_speed < candidate.departure_speed else "throw"
            # Keep the existing 'after' context from the departure point
        
        candidate.start = max(0.0, action_start - before)
        candidate.end = min(duration, action_end + after)
        if candidate.recovery_candidate or candidate.category == "recovery_uncovered_activity":
            if candidate.recovery_window_start <= 0.0:
                candidate.recovery_window_start = action_start
            candidate.recovery_window_end = max(candidate.recovery_window_end, action_end, candidate.end)
        candidate.clip_boundary_reason = "observed_action_window"
        
        # Default end reason
        if not candidate.clip_end_reason:
            candidate.clip_end_reason = "timeout"

        # v0.13.11: Enhance clip end for controlled releases
        release_res = detect_controlled_release(candidate, ordered, clips_cfg)
        if release_res["controlled_release_detected"] > 0:
            release_time = release_res["controlled_release_time"]
            tail = float(clips_cfg.get("controlled_release_safety_tail_seconds", 4.0))
            
            original_action_end = action_end
            original_clip_end = candidate.end
            
            # Extension: action_end should at least cover the release
            effective_action_end = max(action_end, release_time)
            effective_clip_end = min(duration, effective_action_end + tail)
            
            # Respect max duration
            if effective_clip_end - candidate.start > max_duration:
                effective_clip_end = candidate.start + max_duration
                release_res["controlled_release_clamped_by_max_duration"] = 1.0
                # If even the new action_end is beyond max_duration, clamp it too for consistency
                effective_action_end = min(effective_action_end, effective_clip_end)
            
            candidate.action_end = effective_action_end
            candidate.end = effective_clip_end
            candidate.clip_end_reason = "controlled_release"
            
            # Also update the locally scoped action_end for pass A chaining
            action_end = effective_action_end
            
            release_res.update({
                "controlled_release_original_action_end": original_action_end,
                "controlled_release_effective_action_end": effective_action_end,
                "controlled_release_original_clip_end": original_clip_end,
                "controlled_release_effective_clip_end": effective_clip_end
            })
        
        candidate.score_breakdown.update(release_res)

        # v0.13.12: Secondary path for recovery-distribution absorption
        recovery_dist_res = find_recovery_distribution_continuation(candidate, ordered, clips_cfg)
        if recovery_dist_res["recovery_distribution_detected"] > 0:
            dist_end = recovery_dist_res["recovery_distribution_effective_action_end"]
            tail = recovery_dist_res["recovery_distribution_safety_tail"]
            
            original_action_end = action_end
            original_clip_end = candidate.end
            
            effective_action_end = max(action_end, dist_end)
            effective_clip_end = min(duration, effective_action_end + tail)
            
            # Respect max duration
            if effective_clip_end - candidate.start > max_duration:
                effective_clip_end = candidate.start + max_duration
                recovery_dist_res["recovery_distribution_clamped_by_max_duration"] = 1.0
                effective_action_end = min(effective_action_end, effective_clip_end)
                
            candidate.action_end = effective_action_end
            candidate.end = effective_clip_end
            
            # If we don't already have a more specific reason like controlled_release, use this one
            if candidate.clip_end_reason in {"timeout", None}:
                candidate.clip_end_reason = "recovery_distribution_continuation"
            
            action_end = effective_action_end
            
            recovery_dist_res.update({
                "recovery_distribution_original_action_end": original_action_end,
                "recovery_distribution_effective_action_end": effective_action_end,
                "recovery_distribution_original_clip_end": original_clip_end,
                "recovery_distribution_effective_clip_end": effective_clip_end
            })
            
        candidate.score_breakdown.update(recovery_dist_res)

        recovery_tail_res = apply_recovery_window_tail_fallback(
            candidate,
            ordered,
            clips_cfg,
            duration,
            max_duration,
            candidate.end,
            controlled_release_detected=release_res.get("controlled_release_detected", 0.0) > 0,
            recovery_distribution_detected=recovery_dist_res.get("recovery_distribution_detected", 0.0) > 0,
        )
        candidate.score_breakdown.update(recovery_tail_res)

        catch_idle_res = apply_dynamic_catch_control_idle_tail(candidate, ordered, clips_cfg, duration)
        candidate.score_breakdown.update(catch_idle_res)

        if planned and planned[-1].accepted:
            previous = planned[-1]
            # action_start is for current candidate, previous.action_end is for previous
            gap = action_start - (previous.action_end or previous.trigger_time)
            # print(f"[DEBUG_LOG] Chaining check: {previous.candidate_id} -> {candidate.candidate_id}, gap={gap}, same_keeper={previous.keeper_label == candidate.keeper_label}")
            
            # v0.13.11: More flexible chaining: also allow when the clips overlap (negative gap)
            if gap <= continuation_gap and previous.keeper_label == candidate.keeper_label and same_keeper_phase(previous, candidate):
                # Keep one continuous phase when a second keeper event follows,
                # e.g. distribution -> turnover -> shot -> catch.
                proposed_end = min(duration, action_end + final_tail)
                if proposed_end - previous.start <= max_duration:
                    previous.end = max(previous.end, proposed_end)
                    previous.action_end = max(previous.action_end, action_end)
                    previous.contact_frames += candidate.contact_frames
                    previous.ball_confidence = max(previous.ball_confidence, candidate.ball_confidence)
                    previous.identity_confidence = max(previous.identity_confidence, candidate.identity_confidence)
                    previous.description = f"{previous.description}; continued with {candidate.category}".strip("; ")
                    previous.score_breakdown["chained_event_count"] = int(previous.score_breakdown.get("chained_event_count", 1)) + 1
                    previous.clip_boundary_reason = "chained_keeper_phase"
                    
                    # Update clip end reason for the chained sequence
                    # v0.13.11: Use controlled_release if it was already detected for the current candidate
                    if candidate.clip_end_reason == "controlled_release":
                        previous.clip_end_reason = "controlled_release"
                    elif candidate.clip_end_reason == "recovery_distribution_continuation":
                        # v0.13.12: Propagate the new continuation reason
                        previous.clip_end_reason = "recovery_distribution_continuation"
                    elif candidate.category == "distribution" and candidate.departure_speed > 0.35:
                        previous.clip_end_reason = "kick" if candidate.approach_speed < candidate.departure_speed else "throw"
                    elif candidate.category in {"catch_or_control", "cross_claim_or_high_catch"}:
                        previous.clip_end_reason = "timeout"
                    
                    # Extension: Check if we can find a release event within the candidate action
                    if previous.clip_end_reason in {"timeout", "dynamic_idle_tail", "kick", "throw"} and candidate.departure_speed > 0.25 and candidate.clip_end_reason not in {"controlled_release", "recovery_distribution_continuation"}:
                        previous.clip_end_reason = "controlled_release"
                    
                    continue

        if candidate.end - candidate.start < minimum_duration:
            candidate.end = min(duration, candidate.start + minimum_duration)
        candidate.end = min(candidate.end, candidate.start + max_duration)
        planned.append(candidate)
    
    # Final pass: check for very related clips that should be merged despite large gaps (Stufe B)
    # This specifically addresses the 565s -> 592s case if there's evidence they belong together.
    final_clips: list[Candidate] = []
    phase_gap_limit = max(continuation_gap, float(clips_cfg.get("phase_merge_gap_seconds", 30.0)))
    
    # v0.13.10 constants
    duration_tolerance_default = 0.08
    duration_tolerance = float(clips_cfg.get("phase_merge_duration_tolerance", duration_tolerance_default))
    min_pre_roll = float(clips_cfg.get("phase_merge_min_pre_roll_seconds", 2.0))
    min_post_roll = float(clips_cfg.get("phase_merge_min_post_roll_seconds", 2.0))

    for candidate in planned:
        # v0.13.11: Ensure score_breakdown exists
        if candidate.score_breakdown is None:
            candidate.score_breakdown = {}
            
        if final_clips:
            previous = final_clips[-1]
            
            # Diagnose v0.13.10 phase merge diagnostics (for ALL adjacent pairs)
            gap = candidate.start - previous.end
            action_gap = (candidate.action_start or candidate.trigger_time) - (previous.action_end or previous.trigger_time)
            same_keeper = (previous.keeper_label == candidate.keeper_label)
            
            # Restart check
            restart_categories = {"distribution", "keeper_clearance"}
            is_unrelated_restart = (candidate.category in restart_categories and previous.category not in {"catch_or_control", "cross_claim_or_high_catch"})
            if previous.category in restart_categories and candidate.category in restart_categories:
                is_unrelated_restart = True
            
            # Ensure score_breakdown exists (already done above, but for clarity in merge logic)
            if candidate.score_breakdown is None:
                candidate.score_breakdown = {}
            
            # Store diagnostics in the candidate's score_breakdown
            candidate.score_breakdown.update({
                "phase_merge_checked": 1.0,
                "phase_merge_gap": gap,
                "phase_merge_action_gap": action_gap,
                "phase_merge_same_keeper": 1.0 if same_keeper else 0.0,
                "phase_merge_restart_detected": 1.0 if is_unrelated_restart else 0.0,
                "phase_merge_decision": 0.0
            })

            # Check for regular Phase Merge (both accepted) or Continuation Absorption (right is rejected recovery)
            is_continuation_candidate = (not candidate.accepted and candidate.recovery_candidate)
            continuation_validation = is_valid_recovery_continuation(candidate, clips_cfg)
            candidate.score_breakdown.update(continuation_validation)
            if is_continuation_candidate and continuation_validation["recovery_continuation_valid"] <= 0.0:
                candidate.continuation_absorb_reason = "rejected_recovery_continuation_weak_ball_signal"
                is_continuation_candidate = False
            
            if previous.accepted and (candidate.accepted or is_continuation_candidate):
                within_limit = (gap <= phase_gap_limit)

                weak_phase_categories = {
                    "interaction",
                    "recovery_keeper_interaction",
                    "recovery_uncovered_activity",
                }
                weak_absorb_enabled = bool(clips_cfg.get("phase_core_weak_absorption_enabled", True))
                weak_absorb_delta = float(clips_cfg.get("phase_core_weak_absorption_delta", 0.35))
                weak_absorb_max_action = float(clips_cfg.get("phase_core_weak_absorption_max_action_seconds", 12.0))
                previous_score = _candidate_relevance_score(previous)
                candidate_score = _candidate_relevance_score(candidate)
                candidate_action_start, candidate_action_end = _candidate_action_bounds(candidate)
                candidate_action_duration = max(0.0, candidate_action_end - candidate_action_start)

                weak_phase_absorb = (
                    weak_absorb_enabled
                    and candidate.accepted
                    and same_keeper
                    and within_limit
                    and not is_unrelated_restart
                    and previous.clip_end_reason in {"timeout", "dynamic_idle_tail"}
                    and candidate.clip_end_reason in {"timeout", "dynamic_idle_tail"}
                    and candidate.category in weak_phase_categories
                    and candidate_action_duration <= weak_absorb_max_action
                    and candidate_score + weak_absorb_delta < previous_score
                )
                if weak_phase_absorb:
                    previous.merged_from.append(candidate.candidate_id)
                    previous.phase_merge_reason = "same_keeper_weak_phase_absorbed"
                    candidate.continuation_absorbed = True
                    candidate.absorbed_into_candidate_id = previous.candidate_id
                    candidate.continuation_absorb_reason = "accepted_weak_phase_absorbed"
                    previous.score_breakdown.update(
                        {
                            "phase_merge_checked": 1.0,
                            "phase_merge_decision": 1.0,
                            "phase_merge_weak_absorb_applied": 1.0,
                            "phase_merge_weak_absorb_previous_score": previous_score,
                            "phase_merge_weak_absorb_candidate_score": candidate_score,
                            "phase_merge_weak_absorb_score_delta": previous_score - candidate_score,
                            "phase_merge_weak_absorb_candidate_action_seconds": candidate_action_duration,
                        }
                    )
                    continue
                
                # Intelligent duration handling
                raw_combined_duration = candidate.end - previous.start
                limit_with_tolerance = max_duration * (1.0 + duration_tolerance)
                
                if same_keeper and within_limit and not is_unrelated_restart and previous.clip_end_reason in {"timeout", "dynamic_idle_tail"}:
                    # Can we fit this into a trimmed window?
                    # We MUST preserve action times:
                    action_span_start = min(previous.action_start or previous.trigger_time, candidate.action_start or candidate.trigger_time)
                    action_span_end = max(previous.action_end or previous.trigger_time, candidate.action_end or candidate.trigger_time)
                    action_duration = action_span_end - action_span_start
                    
                    # We subtract the desired pre-roll
                    desired_pre_roll = max(0.0, float(category_before.get(previous.category, default_before)))
                    desired_post_roll = max(0.0, float(category_after.get(candidate.category, default_after)))
                    
                    effective_pre_roll = desired_pre_roll
                    effective_post_roll = desired_post_roll
                    
                    limit_with_tolerance = max_duration * (1.0 + duration_tolerance)
                    
                    # First try with desired context
                    if (action_duration + effective_pre_roll + effective_post_roll) > max_duration:
                        # Too long, reduce pre-roll first
                        excess = (action_duration + effective_pre_roll + effective_post_roll) - max_duration
                        reduction = min(excess, effective_pre_roll - min_pre_roll)
                        if reduction > 0:
                            effective_pre_roll -= reduction
                        
                        # If still too long, reduce post-roll
                        if (action_duration + effective_pre_roll + effective_post_roll) > max_duration:
                            excess = (action_duration + effective_pre_roll + effective_post_roll) - max_duration
                            reduction = min(excess, effective_post_roll - min_post_roll)
                            if reduction > 0:
                                effective_post_roll -= reduction
                    
                    safe_start = max(0.0, action_span_start - effective_pre_roll)
                    safe_end = min(duration, action_span_end + effective_post_roll)
                    trimmed_duration = safe_end - safe_start
                    
                    if trimmed_duration <= limit_with_tolerance:
                        # Success! Merge or Absorb
                        previous.start = safe_start
                        previous.end = safe_end
                        previous.action_end = action_span_end
                        previous.contact_frames += candidate.contact_frames
                        previous.ball_confidence = max(previous.ball_confidence, candidate.ball_confidence)
                        previous.identity_confidence = max(previous.identity_confidence, candidate.identity_confidence)
                        
                        if candidate.accepted:
                            previous.description = f"{previous.description}; phase-merged with {candidate.category}".strip("; ")
                            previous.score_breakdown["phase_merge_decision"] = 1.0
                            previous.phase_merge_reason = "same_keeper_related_phase"
                            previous.merged_from.append(candidate.candidate_id)
                        else:
                            previous.description = f"{previous.description}; continuation absorbed from {candidate.category}".strip("; ")
                            candidate.continuation_absorbed = True
                            candidate.absorbed_into_candidate_id = previous.candidate_id
                            candidate.continuation_absorb_reason = "rejected_recovery_continuation_same_keeper"
                            previous.score_breakdown["phase_merge_decision"] = 1.0
                            previous.phase_merge_reason = "absorbed_recovery_continuation"
                            previous.merged_from.append(candidate.candidate_id)
                        
                        previous.clip_end_reason = candidate.clip_end_reason
                        previous.score_breakdown.update({
                            "phase_merge_original_pre_roll": desired_pre_roll,
                            "phase_merge_original_post_roll": desired_post_roll,
                            "phase_merge_effective_pre_roll": effective_pre_roll,
                            "phase_merge_effective_post_roll": effective_post_roll,
                            "phase_merge_action_duration": action_duration,
                            "phase_merge_original_duration": raw_combined_duration,
                            "phase_merge_trimmed_duration": trimmed_duration,
                            "phase_merge_duration_limit": max_duration,
                        })
                        continue

        if candidate.accepted or not candidate.continuation_absorbed:
            final_clips.append(candidate)
        
    final_clips = merge_overlapping_final_clips(final_clips, duration, clips_cfg)

    isolated_dynamic_tail = max(0.0, float(clips_cfg.get("catch_control_isolated_dynamic_idle_tail_seconds", 3.0)))
    isolated_compact_action_max = max(
        0.0,
        float(clips_cfg.get("catch_control_isolated_compact_action_max_seconds", 4.5)),
    )
    isolated_compact_min_clip = max(
        0.0,
        float(clips_cfg.get("catch_control_isolated_compact_min_clip_seconds", 20.0)),
    )
    isolated_rebalance_pre_shift = max(
        0.0,
        float(clips_cfg.get("catch_control_isolated_rebalance_pre_shift_seconds", 2.0)),
    )
    isolated_rebalance_post_bonus = max(
        0.0,
        float(clips_cfg.get("catch_control_isolated_rebalance_post_bonus_seconds", 2.0)),
    )
    isolated_rebalance_min_pre = max(
        0.0,
        float(clips_cfg.get("catch_control_isolated_rebalance_min_pre_roll_seconds", 8.0)),
    )
    isolated_rebalance_max_post = max(
        0.0,
        float(clips_cfg.get("catch_control_isolated_rebalance_max_post_roll_seconds", 5.0)),
    )
    clearance_short_action_max = max(
        0.0,
        float(clips_cfg.get("clearance_isolated_short_action_max_seconds", 1.5)),
    )
    clearance_isolated_safety_tail = max(
        0.0,
        float(clips_cfg.get("clearance_isolated_safety_tail_seconds", 3.6)),
    )
    clearance_continuation_search = max(
        0.0,
        float(clips_cfg.get("clearance_continuation_search_seconds", 12.0)),
    )
    distribution_preparation_pre_roll = max(
        0.0,
        float(clips_cfg.get("distribution_preparation_pre_roll_seconds", 2.0)),
    )
    distribution_preparation_action_max = max(
        0.0,
        float(clips_cfg.get("distribution_preparation_max_action_seconds", 2.5)),
    )
    distribution_preparation_clip_max = max(
        8.0,
        float(clips_cfg.get("distribution_preparation_max_clip_seconds", 18.0)),
    )
    distribution_restart_rescue_extra_tail = max(
        0.0,
        float(clips_cfg.get("distribution_restart_rescue_extra_tail_seconds", 1.0)),
    )
    distribution_long_phase_departure_floor = max(
        0.0,
        float(clips_cfg.get("distribution_long_phase_departure_speed_floor", 3.0)),
    )
    distribution_long_phase_core_seconds = max(
        8.0,
        float(clips_cfg.get("distribution_long_phase_core_seconds", 11.0)),
    )
    distribution_long_phase_tail_seconds = max(
        1.0,
        float(clips_cfg.get("distribution_long_phase_tail_seconds", 4.0)),
    )
    distribution_long_phase_max_clip_seconds = max(
        12.0,
        float(clips_cfg.get("distribution_long_phase_max_clip_seconds", 15.0)),
    )
    recovery_context_rescue_pre_roll_seconds = max(
        0.0,
        float(clips_cfg.get("recovery_context_rescue_pre_roll_seconds", 6.0)),
    )
    recovery_context_rescue_post_roll_seconds = max(
        0.0,
        float(clips_cfg.get("recovery_context_rescue_post_roll_seconds", 6.0)),
    )
    catch_final_overlap_core_max_seconds = max(
        20.0,
        float(clips_cfg.get("catch_control_final_overlap_core_max_seconds", 24.0)),
    )
    merged_dynamic_tail_cap = max(6.0, float(clips_cfg.get("catch_control_merged_phase_max_seconds", 18.0)))
    single_merge_dynamic_tail = max(0.5, float(clips_cfg.get("catch_control_single_merge_dynamic_tail_seconds", 4.0)))
    single_merge_pre_roll_cap = max(0.0, float(clips_cfg.get("catch_control_single_merge_pre_roll_seconds", 3.0)))
    controlled_release_long_phase_min_action = max(1.0, float(clips_cfg.get("catch_control_controlled_release_long_phase_min_action_seconds", 40.0)))
    controlled_release_trailing_window = max(6.0, float(clips_cfg.get("catch_control_controlled_release_trailing_core_seconds", 24.0)))
    controlled_release_trailing_tail = max(0.0, float(clips_cfg.get("catch_control_controlled_release_trailing_tail_seconds", 5.0)))
    merged_dynamic_idle_tail_cap = max(
        0.0,
        float(clips_cfg.get("catch_control_merged_dynamic_idle_tail_cap_seconds", 4.0)),
    )
    merged_dynamic_pre_roll_cap = max(
        0.0,
        float(clips_cfg.get("catch_control_merged_dynamic_pre_roll_cap_seconds", 3.0)),
    )
    phase_min_pre_roll = float(clips_cfg.get("phase_merge_min_pre_roll_seconds", 2.0))
    for candidate in final_clips:
        if (
            candidate.accepted
            and candidate.category == "catch_or_control"
            and candidate.clip_end_reason == "dynamic_idle_tail"
            and len(candidate.merged_from) >= 2
        ):
            capped_end = min(duration, candidate.start + merged_dynamic_tail_cap)
            if capped_end < candidate.end:
                candidate.end = capped_end
                candidate.clip_boundary_reason = "merged_action_core"
                candidate.score_breakdown.update(
                    {
                        "catch_control_merged_core_trim_applied": 1.0,
                        "catch_control_merged_core_max_seconds": merged_dynamic_tail_cap,
                        "catch_control_merged_core_effective_end": candidate.end,
                    }
                )

        if (
            candidate.accepted
            and candidate.category == "catch_or_control"
            and candidate.clip_end_reason == "dynamic_idle_tail"
            and len(candidate.merged_from) == 1
            and candidate.phase_merge_reason == "same_keeper_weak_phase_absorbed"
            and float(candidate.score_breakdown.get("phase_merge_action_duration", 0.0)) >= 12.0
            and "catch_control_last_activity_time" in candidate.score_breakdown
        ):
            last_activity = float(
                candidate.score_breakdown.get(
                    "catch_control_last_activity_time",
                    candidate.action_end or candidate.trigger_time,
                )
            )
            trimmed_end = min(candidate.end, min(duration, last_activity + single_merge_dynamic_tail))
            action_start = candidate.action_start or candidate.trigger_time
            trimmed_start = max(candidate.start, action_start - single_merge_pre_roll_cap)
            if trimmed_start > candidate.start:
                candidate.start = trimmed_start
            if trimmed_end < candidate.end:
                candidate.end = max(candidate.start, trimmed_end)
            if trimmed_start > candidate.start or trimmed_end < candidate.end:
                candidate.clip_boundary_reason = "single_merge_action_core"
                candidate.score_breakdown.update(
                    {
                        "catch_control_single_merge_core_applied": 1.0,
                        "catch_control_single_merge_core_tail_seconds": single_merge_dynamic_tail,
                        "catch_control_single_merge_core_pre_roll_seconds": single_merge_pre_roll_cap,
                        "catch_control_single_merge_core_effective_start": candidate.start,
                        "catch_control_single_merge_core_effective_end": candidate.end,
                    }
                )

        if (
            candidate.accepted
            and candidate.category == "catch_or_control"
            and candidate.clip_end_reason == "controlled_release"
            and bool(candidate.merged_from)
        ):
            action_end = max(candidate.action_end or candidate.trigger_time, candidate.trigger_time)
            phase_action_duration = float(candidate.score_breakdown.get("phase_merge_action_duration", 0.0))
            effective_pre_roll = float(candidate.score_breakdown.get("phase_merge_effective_pre_roll", 0.0))
            if phase_action_duration >= controlled_release_long_phase_min_action and effective_pre_roll <= phase_min_pre_roll + 1e-6:
                trailing_start = max(0.0, action_end - controlled_release_trailing_window)
                trailing_end = min(duration, action_end + controlled_release_trailing_tail)
                if trailing_start > candidate.start:
                    candidate.start = trailing_start
                if trailing_end > candidate.end:
                    candidate.end = trailing_end
                candidate.clip_boundary_reason = "controlled_release_trailing_core"
                candidate.score_breakdown.update(
                    {
                        "catch_control_controlled_release_core_applied": 1.0,
                        "catch_control_controlled_release_core_window": controlled_release_trailing_window,
                        "catch_control_controlled_release_core_tail": controlled_release_trailing_tail,
                        "catch_control_controlled_release_core_effective_start": candidate.start,
                        "catch_control_controlled_release_core_effective_end": candidate.end,
                    }
                )

        if (
            candidate.accepted
            and candidate.category == "catch_or_control"
            and candidate.clip_end_reason == "dynamic_idle_tail"
            and not candidate.merged_from
        ):
            action_start = candidate.action_start or candidate.trigger_time
            action_end = max(candidate.action_end or candidate.trigger_time, candidate.trigger_time)
            action_duration = max(0.0, action_end - action_start)
            pre_roll = max(0.0, action_start - candidate.start)
            post_roll = max(0.0, candidate.end - action_end)
            clip_duration = max(0.0, candidate.end - candidate.start)

            rebalance_shift = min(
                isolated_rebalance_pre_shift,
                max(0.0, pre_roll - isolated_rebalance_min_pre),
            )
            rebalance_bonus = min(isolated_rebalance_post_bonus, isolated_rebalance_max_post)
            rebalance_new_start = candidate.start + rebalance_shift
            rebalance_new_end = min(duration, candidate.end + rebalance_bonus)
            rebalance_applied = False

            if (
                action_duration <= isolated_compact_action_max
                and clip_duration >= isolated_compact_min_clip
                and pre_roll >= isolated_rebalance_min_pre
                and post_roll >= 6.0
                and rebalance_shift > 0.0
                and rebalance_new_start <= action_start
            ):
                candidate.start = rebalance_new_start
                base_core_end = min(candidate.end, min(duration, action_end + isolated_dynamic_tail))
                candidate.end = max(action_end, min(duration, base_core_end + rebalance_bonus))
                candidate.clip_boundary_reason = "isolated_action_core_rebalanced"
                candidate.score_breakdown.update(
                    {
                        "catch_control_isolated_core_rebalance_applied": 1.0,
                        "catch_control_isolated_core_rebalance_shift_seconds": rebalance_shift,
                        "catch_control_isolated_core_rebalance_post_bonus_seconds": rebalance_bonus,
                        "catch_control_isolated_core_rebalance_base_tail_seconds": max(0.0, base_core_end - action_end),
                        "catch_control_isolated_core_rebalance_effective_start": candidate.start,
                        "catch_control_isolated_core_rebalance_effective_end": candidate.end,
                    }
                )
                rebalance_applied = True

            if not rebalance_applied:
                core_end = min(duration, action_end + isolated_dynamic_tail)
                if core_end < candidate.end:
                    candidate.end = max(action_end, core_end)
                    candidate.clip_boundary_reason = "isolated_action_core"
                    candidate.score_breakdown.update(
                        {
                            "catch_control_isolated_core_trim_applied": 1.0,
                            "catch_control_isolated_core_tail_seconds": isolated_dynamic_tail,
                            "catch_control_isolated_core_effective_end": candidate.end,
                            "catch_control_isolated_core_rebalance_kept": 0.0,
                        }
                    )
            else:
                candidate.score_breakdown.update(
                    {
                        "catch_control_isolated_core_trim_applied": 0.0,
                        "catch_control_isolated_core_tail_seconds": isolated_dynamic_tail,
                        "catch_control_isolated_core_effective_end": candidate.end,
                        "catch_control_isolated_core_rebalance_kept": 1.0,
                    }
                )

        if (
            candidate.accepted
            and candidate.category == "catch_or_control"
            and candidate.clip_end_reason == "dynamic_idle_tail"
            and bool(candidate.merged_from)
            and "catch_control_last_activity_time" in candidate.score_breakdown
            and float(candidate.score_breakdown.get("phase_merge_effective_pre_roll", 0.0)) >= 6.0
            and float(candidate.score_breakdown.get("phase_merge_effective_post_roll", 0.0)) >= 10.0
        ):
            last_activity = float(
                candidate.score_breakdown.get(
                    "catch_control_last_activity_time",
                    candidate.action_end or candidate.trigger_time,
                )
            )
            capped_end = min(duration, last_activity + merged_dynamic_idle_tail_cap)
            if capped_end < candidate.end:
                candidate.end = max(candidate.start, capped_end)
                action_start = candidate.action_start or candidate.trigger_time
                capped_start = max(0.0, action_start - merged_dynamic_pre_roll_cap)
                if capped_start > candidate.start:
                    candidate.start = min(capped_start, candidate.end)
                candidate.clip_boundary_reason = "merged_dynamic_idle_tail_cap"
                candidate.score_breakdown.update(
                    {
                        "catch_control_merged_dynamic_tail_cap_applied": 1.0,
                        "catch_control_merged_dynamic_tail_cap_seconds": merged_dynamic_idle_tail_cap,
                        "catch_control_merged_dynamic_pre_roll_cap_seconds": merged_dynamic_pre_roll_cap,
                        "catch_control_merged_dynamic_tail_cap_effective_start": candidate.start,
                        "catch_control_merged_dynamic_tail_cap_effective_end": candidate.end,
                    }
                )

        if (
            candidate.accepted
            and candidate.category in {"keeper_clearance", "distribution", "goalkeeper_distribution"}
            and candidate.clip_end_reason in {"timeout", "observed_action_window", "dynamic_idle_tail"}
            and not candidate.merged_from
        ):
            action_end = max(candidate.action_end or candidate.trigger_time, candidate.trigger_time)
            action_start = candidate.action_start or candidate.trigger_time
            action_duration = max(0.0, action_end - action_start)
            if action_duration <= clearance_short_action_max:
                continuation_detected = False
                search_end = min(duration, action_end + clearance_continuation_search)
                for other in final_clips:
                    if other.candidate_id == candidate.candidate_id:
                        continue
                    if other.keeper_label != candidate.keeper_label:
                        continue
                    if not other.accepted:
                        continue
                    other_start = other.action_start or other.trigger_time
                    if other_start <= action_end or other_start > search_end:
                        continue
                    if other.category in {
                        "distribution",
                        "goalkeeper_distribution",
                        "keeper_clearance",
                        "catch_or_control",
                        "cross_claim_or_high_catch",
                        "save_or_deflection",
                        "diving_save",
                        "interaction",
                        "recovery_keeper_interaction",
                        "recovery_uncovered_activity",
                    }:
                        continuation_detected = True
                        break

                if not continuation_detected:
                    capped_end = min(duration, action_end + clearance_isolated_safety_tail)
                    if capped_end < candidate.end:
                        candidate.end = max(action_end, capped_end)
                        candidate.clip_boundary_reason = "isolated_clearance_safety_tail"
                        candidate.score_breakdown.update(
                            {
                                "clearance_isolated_tail_applied": 1.0,
                                "clearance_isolated_tail_seconds": clearance_isolated_safety_tail,
                                "clearance_isolated_tail_effective_end": candidate.end,
                                "clearance_isolated_tail_continuation_detected": 0.0,
                            }
                        )
                else:
                    candidate.score_breakdown["clearance_isolated_tail_continuation_detected"] = 1.0

        if (
            candidate.accepted
            and candidate.category in {"distribution", "goalkeeper_distribution", "keeper_clearance"}
            and len(candidate.merged_from) <= 1
        ):
            action_start = candidate.action_start or candidate.trigger_time
            action_end = max(candidate.action_end or candidate.trigger_time, candidate.trigger_time)
            action_duration = max(0.0, action_end - action_start)
            clip_duration = max(0.0, candidate.end - candidate.start)
            execution_anchor = candidate.trigger_time
            current_pre_roll = max(0.0, execution_anchor - candidate.start)
            if (
                action_duration <= distribution_preparation_action_max
                and clip_duration <= distribution_preparation_clip_max
                and current_pre_roll + 1e-6 < distribution_preparation_pre_roll
            ):
                desired_start = max(0.0, execution_anchor - distribution_preparation_pre_roll)
                if desired_start < candidate.start:
                    candidate.start = desired_start
                    candidate.clip_boundary_reason = "distribution_preparation_pre_roll"
                    candidate.score_breakdown.update(
                        {
                            "distribution_preparation_pre_roll_applied": 1.0,
                            "distribution_preparation_pre_roll_seconds": distribution_preparation_pre_roll,
                            "distribution_preparation_pre_roll_effective_start": candidate.start,
                        }
                    )

        if (
            candidate.accepted
            and candidate.category in {"distribution", "goalkeeper_distribution"}
            and float(candidate.score_breakdown.get("restart_relevance_rescue_applied", 0.0)) > 0.0
        ):
            extended_end = min(duration, candidate.end + distribution_restart_rescue_extra_tail)
            if extended_end > candidate.end:
                candidate.end = extended_end
                candidate.clip_boundary_reason = "restart_rescue_distribution_tail"
                candidate.score_breakdown["distribution_restart_rescue_tail_seconds"] = distribution_restart_rescue_extra_tail

        if (
            candidate.accepted
            and candidate.category in {"distribution", "goalkeeper_distribution"}
            and bool(candidate.merged_from)
            and candidate.departure_speed >= distribution_long_phase_departure_floor
            and (candidate.end - candidate.start) > distribution_long_phase_max_clip_seconds
        ):
            action_end = max(candidate.action_end or candidate.trigger_time, candidate.trigger_time)
            compact_start = max(candidate.start, action_end - distribution_long_phase_core_seconds)
            compact_end = min(duration, action_end + distribution_long_phase_tail_seconds)
            if compact_end > compact_start and (compact_end - compact_start) < (candidate.end - candidate.start):
                candidate.start = compact_start
                candidate.end = compact_end
                candidate.clip_boundary_reason = "distribution_compact_core_window"
                candidate.score_breakdown.update(
                    {
                        "distribution_compact_core_applied": 1.0,
                        "distribution_compact_core_seconds": distribution_long_phase_core_seconds,
                        "distribution_compact_tail_seconds": distribution_long_phase_tail_seconds,
                    }
                )

        if (
            candidate.accepted
            and candidate.category == "recovery_uncovered_activity"
            and float(candidate.score_breakdown.get("recovery_contextual_rescue_applied", 0.0)) > 0.0
            and (candidate.end - candidate.start) > (recovery_context_rescue_pre_roll_seconds + recovery_context_rescue_post_roll_seconds + 1.0)
        ):
            action_start = candidate.action_start or candidate.trigger_time
            action_end = max(candidate.action_end or candidate.trigger_time, candidate.trigger_time)
            compact_start = max(0.0, action_start - recovery_context_rescue_pre_roll_seconds)
            compact_end = min(duration, action_end + recovery_context_rescue_post_roll_seconds)
            if compact_end > compact_start:
                candidate.start = compact_start
                candidate.end = compact_end
                candidate.clip_boundary_reason = "recovery_context_rescue_window"
                candidate.score_breakdown.update(
                    {
                        "recovery_context_rescue_window_applied": 1.0,
                        "recovery_context_rescue_pre_roll_seconds": recovery_context_rescue_pre_roll_seconds,
                        "recovery_context_rescue_post_roll_seconds": recovery_context_rescue_post_roll_seconds,
                    }
                )

        if (
            candidate.accepted
            and candidate.category == "catch_or_control"
            and candidate.clip_boundary_reason == "final_overlap_merged"
            and len(candidate.merged_from) >= 3
            and (candidate.end - candidate.start) > catch_final_overlap_core_max_seconds
        ):
            action_end = max(candidate.action_end or candidate.trigger_time, candidate.trigger_time)
            compact_start = max(candidate.start, action_end - (catch_final_overlap_core_max_seconds - 5.0))
            compact_end = min(duration, action_end + 5.0)
            if compact_end > compact_start and (compact_end - compact_start) < (candidate.end - candidate.start):
                candidate.start = compact_start
                candidate.end = compact_end
                candidate.clip_boundary_reason = "final_overlap_compact_core"
                candidate.score_breakdown.update(
                    {
                        "final_overlap_compact_core_applied": 1.0,
                        "final_overlap_compact_core_max_seconds": catch_final_overlap_core_max_seconds,
                    }
                )

    return final_clips


def clamp_clip_windows_to_sources(items: list[Candidate], manifest, clips_cfg: dict[str, Any]) -> list[Candidate]:
    """Prevent a clip from leaking into another recording/half by default."""
    if bool(clips_cfg.get("allow_cross_source_clips", False)) or len(manifest.files) <= 1:
        return items
    for candidate in items:
        anchor = candidate.trigger_time
        source = next((f for f in manifest.files if f.global_start_seconds <= anchor < f.global_end_seconds), None)
        if source is None:
            continue
        old_start, old_end = candidate.start, candidate.end
        candidate.start = max(candidate.start, source.global_start_seconds)
        candidate.end = min(candidate.end, source.global_end_seconds)
        if candidate.start != old_start or candidate.end != old_end:
            candidate.clip_boundary_reason = "source_boundary_clamp"
    return items

def choose_keeper_auto(persons: list[Box], width: int, height: int, roi: list[float]) -> Optional[Box]:
    if not persons:
        return None
    x1, y1, x2, y2 = roi
    px = (x1 * width, y1 * height, x2 * width, y2 * height)
    center = ((px[0] + px[2]) / 2, (px[1] + px[3]) / 2)
    in_roi = [p for p in persons if px[0] <= p.center[0] <= px[2] and px[1] <= p.center[1] <= px[3]] or persons
    return min(in_roi, key=lambda p: math.hypot(p.center[0]-center[0], p.center[1]-center[1]) - 0.25*p.height)


def select_keeper(frame, persons: list[Box]) -> Optional[Box]:
    if not persons:
        return None
    selected: dict[str, Optional[Box]] = {"box": None}
    base = frame.copy()
    for index, box in enumerate(persons, 1):
        cv2.rectangle(base, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 2)
        cv2.putText(base, str(index), (int(box.x1), max(20, int(box.y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    def click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            matches = [b for b in persons if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2]
            if matches:
                selected["box"] = min(matches, key=lambda b: b.width*b.height)

    title = "Goalkeeper anklicken, dann ENTER (ESC = Abbruch)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, click)
    while True:
        shown = base.copy()
        if selected["box"]:
            b = selected["box"]
            cv2.rectangle(shown, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0,0,255), 4)
        cv2.imshow(title, shown)
        key = cv2.waitKey(50) & 0xFF
        if key in (10, 13) and selected["box"]:
            break
        if key == 27:
            selected["box"] = None
            break
    cv2.destroyWindow(title)
    return selected["box"]


def _crop(frame: np.ndarray, box: Box) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(width, int(box.x2)), min(height, int(box.y2))
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None
    # Use torso area; legs and pitch colour are less useful for identity.
    torso_y2 = y1 + max(8, int((y2 - y1) * 0.68))
    return frame[y1:torso_y2, x1:x2]


def _appearance_histogram(frame: np.ndarray, box: Box) -> np.ndarray | None:
    crop = _crop(frame, box)
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    # Central upper torso: reduces grass, shorts, skin and neighbouring players.
    crop = crop[max(0, int(h*0.08)):max(1, int(h*0.62)), max(0, int(w*0.18)):max(1, int(w*0.82))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    mask = ((saturation >= 35) & (value >= 35)).astype(np.uint8) * 255
    # Exclude likely pitch-green pixels. This is intentionally broad and only
    # removes pixels when they are sufficiently saturated.
    green = ((hue >= 32) & (hue <= 92) & (saturation >= 55))
    mask[green] = 0
    if cv2.countNonZero(mask) < 20:
        mask = None
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
    return hist


def _hist_similarity(reference: np.ndarray | None, current: np.ndarray | None) -> float:
    if reference is None or current is None:
        return 0.5
    distance = float(cv2.compareHist(reference, current, cv2.HISTCMP_BHATTACHARYYA))
    return max(0.0, min(1.0, 1.0-distance))


def _shirt_contrast(candidate: np.ndarray | None, peers: list[np.ndarray | None]) -> float:
    """How different a shirt is from the most similar visible outfield shirts."""
    if candidate is None:
        return 0.5
    distances = []
    for peer in peers:
        if peer is None:
            continue
        distances.append(float(cv2.compareHist(candidate, peer, cv2.HISTCMP_BHATTACHARYYA)))
    if not distances:
        return 0.5
    distances.sort()
    # Median of the three closest peers is robust against one similarly coloured referee.
    nearest = distances[:min(3, len(distances))]
    return max(0.0, min(1.0, float(np.median(nearest)) / 0.70))


@dataclass
class KeeperMatch:
    box: Box
    confidence: float
    reidentified: bool = False


class KeeperIdentity:
    """Stable goalkeeper identity independent from ByteTrack IDs.

    Version 0.7 weighting:
      * 45% shirt-colour contrast against all other visible people
      * 30% location in a configured goal/penalty-area region
      * 15% spatial continuity
      * 10% similarity to the initially selected goalkeeper shirt

    A referee can also have a unique shirt, but normally receives a low goal-area
    score. ByteTrack IDs are treated as temporary implementation details.
    """

    def __init__(self, frame: np.ndarray, box: Box, cfg: dict, width: int, height: int) -> None:
        self.cfg = cfg
        self.width = width
        self.height = height
        self.track_id = box.track_id
        self.last_box = box
        self.reference_hist = _appearance_histogram(frame, box)
        self.last_seen_timestamp = 0.0
        self.pending_track_id: int | None = None
        self.pending_count = 0
        self.last_switch_timestamp = -999.0
        self.stable_identity_id = 1

    def _rect_score(self, box: Box, rect: list[float]) -> float:
        x1, y1, x2, y2 = rect
        cx, cy = box.center[0]/self.width, box.center[1]/self.height
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return 1.0
        dx = max(x1-cx, 0.0, cx-x2)
        dy = max(y1-cy, 0.0, cy-y2)
        return math.exp(-math.hypot(dx, dy)/max(float(self.cfg.get("goal_region_falloff", 0.20)), .01))

    def _goal_area_score(self, box: Box) -> float:
        regions = self.cfg.get("goal_regions") or [self.cfg.get("goal_roi", [0, 0, 1, 1])]
        score = max(self._rect_score(box, list(region)) for region in regions)
        # The referee usually remains around the central corridor rather than either goal.
        cx, cy = box.center[0]/self.width, box.center[1]/self.height
        central = self.cfg.get("referee_exclusion_region", [0.15, 0.35, 0.85, 0.65])
        if central[0] <= cx <= central[2] and central[1] <= cy <= central[3]:
            score *= float(self.cfg.get("referee_central_penalty", 0.35))
        return score

    def _continuity_score(self, box: Box) -> float:
        last_cx, last_cy = self.last_box.center
        cx, cy = box.center
        scale = max(self.last_box.diagonal*float(self.cfg.get("reid_position_scale", 4.0)), 1.0)
        return math.exp(-math.hypot(cx-last_cx, cy-last_cy)/scale)

    def _score_all(self, frame: np.ndarray, persons: list[Box]) -> list[tuple[float, Box, dict[str, float]]]:
        descriptors = [_appearance_histogram(frame, p) for p in persons]
        scored = []
        for index, box in enumerate(persons):
            peers = descriptors[:index] + descriptors[index+1:]
            contrast = _shirt_contrast(descriptors[index], peers)
            goal_area = self._goal_area_score(box)
            continuity = self._continuity_score(box)
            reference = _hist_similarity(self.reference_hist, descriptors[index])
            w_contrast = float(self.cfg.get("reid_shirt_contrast_weight", 0.20))
            w_goal = float(self.cfg.get("reid_goal_area_weight", 0.25))
            w_continuity = float(self.cfg.get("reid_continuity_weight", 0.30))
            w_reference = float(self.cfg.get("reid_reference_weight", 0.25))
            weight_sum = max(0.001, w_contrast + w_goal + w_continuity + w_reference)
            score = (w_contrast*contrast + w_goal*goal_area + w_continuity*continuity + w_reference*reference) / weight_sum
            scored.append((score, box, {
                "shirt_contrast": contrast,
                "goal_area": goal_area,
                "continuity": continuity,
                "reference_colour": reference,
            }))
        return sorted(scored, key=lambda item: item[0], reverse=True)

    def match(self, frame: np.ndarray, persons: list[Box], timestamp: float) -> KeeperMatch | None:
        if not persons:
            return None
        scored = self._score_all(frame, persons)
        score, best, components = scored[0]
        exact = next((item for item in scored if self.track_id is not None and item[1].track_id == self.track_id), None)
        # Keep the exact track if it remains plausible and is not clearly beaten.
        if exact is not None:
            exact_score, exact_box, _ = exact
            margin = float(self.cfg.get("reid_switch_margin", 0.08))
            if exact_score >= float(self.cfg.get("reid_keep_confidence", 0.50)) and score-exact_score < margin:
                self.last_box = exact_box
                self.last_seen_timestamp = timestamp
                current = _appearance_histogram(frame, exact_box)
                if current is not None and exact_box.confidence >= 0.55:
                    self.reference_hist = current if self.reference_hist is None else cv2.addWeighted(self.reference_hist, .97, current, .03, 0)
                return KeeperMatch(exact_box, exact_score, False)
        if score < float(self.cfg.get("reid_min_confidence", 0.62)):
            self.pending_track_id, self.pending_count = None, 0
            return None
        required = max(1, int(self.cfg.get("reid_confirmation_frames", 4)))
        cooldown = float(self.cfg.get("reid_switch_cooldown_seconds", .8))
        if best.track_id == self.pending_track_id:
            self.pending_count += 1
        else:
            self.pending_track_id, self.pending_count = best.track_id, 1
        confirmed = self.pending_count >= required and timestamp-self.last_switch_timestamp >= cooldown
        if confirmed:
            old_track = self.track_id
            self.track_id, self.last_box = best.track_id, best
            self.last_seen_timestamp, self.last_switch_timestamp = timestamp, timestamp
            self.pending_track_id, self.pending_count = None, 0
            return KeeperMatch(best, score, old_track != best.track_id)
        return KeeperMatch(best, score*.9, False)

    def reset_tracking_state_for_new_source(self) -> None:
        """Reset source-local tracking continuity while keeping semantic identity."""
        self.track_id = None
        self.last_box = Box(0.0, 0.0, float(self.width), float(self.height), 1.0, PERSON_CLASS, None)
        self.pending_track_id = None
        self.pending_count = 0
        self.last_switch_timestamp = -999.0

def heuristic_score(distance: float, contact_frames: int, ball_confidence: float, identity_confidence: float, cfg: dict) -> float:
    threshold = max(float(cfg.get("distance_factor", 1.35)), 0.01)
    distance_score = max(0.0, min(1.0, 1.0 - distance / threshold))
    frame_target = max(1, int(cfg.get("minimum_contact_frames", 2)))
    frame_score = min(1.0, contact_frames / frame_target)
    ball_score = max(0.0, min(1.0, ball_confidence))
    return max(0.0, min(1.0, 0.40*distance_score + 0.25*frame_score + 0.20*ball_score + 0.15*identity_confidence))


def _draw_preview(frame, persons: list[Box], balls: list[Box], keeper: Optional[Box], timestamp: float, candidate_count: int, identity_confidence: float):
    preview = frame.copy()
    for p in persons:
        cv2.rectangle(preview, (int(p.x1), int(p.y1)), (int(p.x2), int(p.y2)), (255, 120, 0), 1)
    for b in balls:
        cv2.rectangle(preview, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 0, 255), 3)
    if keeper:
        cv2.rectangle(preview, (int(keeper.x1), int(keeper.y1)), (int(keeper.x2), int(keeper.y2)), (0, 255, 0), 3)
    cv2.putText(preview, f"{timestamp/60:.1f} min | candidates {candidate_count} | keeper {identity_confidence:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow("Goalkeeper Highlights - Live Preview (Q to hide)", preview)
    return (cv2.waitKey(1) & 0xFF) != ord('q')


def _resolve_fp16_state(device: str, requested_fp16: bool) -> tuple[bool, str | None]:
    if not requested_fp16:
        return False, None
    if not torch.cuda.is_available():
        return False, "cuda_unavailable"
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return False, "device_cpu"
    return True, None


def _normalize_backend(value: Any) -> str:
    backend = str(value or "pytorch").strip().lower()
    return backend if backend in {"pytorch", "tensorrt", "onnx"} else "pytorch"


def _model_format_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".engine":
        return "tensorrt_engine"
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".pt":
        return "pytorch"
    return "unknown"


def _backend_cache_key(*, model: Path, image_size: int, precision: str, backend: str, backend_version: str) -> str:
    raw = f"{model.resolve()}|{model.stat().st_mtime_ns}|{image_size}|{precision}|{backend}|{backend_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class _TrackRuntimeBreakdown:
    callback_ms: float = 0.0
    callback_dispatch_ms: float = 0.0
    callback_predict_start_ms: float = 0.0
    callback_batch_start_ms: float = 0.0
    callback_postprocess_end_ms: float = 0.0
    callback_batch_end_ms: float = 0.0
    callback_predict_end_ms: float = 0.0
    callback_other_ms: float = 0.0
    predictor_pre_ms: float = 0.0
    pre_source_setup_ms: float = 0.0
    pre_batch_prepare_ms: float = 0.0
    pre_other_ms: float = 0.0
    predictor_post_ms: float = 0.0
    tracker_update_ms: float = 0.0
    result_build_ms: float = 0.0
    result_wrap_ms: float = 0.0
    ultralytics_misc_ms: float = 0.0
    framework_other_ms: float = 0.0


class _TrackRunner:
    def __init__(self, model: Any, *, yolo: dict[str, Any], device: str, fp16_enabled: bool, mode: str):
        self.model = model
        self.mode = mode if mode in {"legacy", "optimized"} else "optimized"
        self.kwargs = {
            "tracker": yolo["tracker"],
            "classes": [PERSON_CLASS, SPORTS_BALL_CLASS],
            "conf": float(yolo["confidence"]),
            "iou": float(yolo["iou"]),
            "imgsz": int(yolo["image_size"]),
            "device": device,
            "verbose": False,
        }
        if fp16_enabled:
            self.kwargs["half"] = True
        self._callback_accum_ms = 0.0
        self._callback_event_ms: dict[str, float] = {}
        self._callback_event_calls: dict[str, int] = {}
        self._tracker_update_accum_ms = 0.0
        self._callback_wrapped = False
        self._predictor_initialized = False
        self._last_callback_calls: dict[str, int] = {}
        self._last_callback_ms: dict[str, float] = {}

    def _wrap_tracker_updates(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if not trackers:
            return
        for tracker in trackers:
            if getattr(tracker, "_gh_update_wrapped", False):
                continue
            original = tracker.update

            def wrapped_update(*args, __orig=original, **kwargs):
                started = time.perf_counter()
                out = __orig(*args, **kwargs)
                self._tracker_update_accum_ms += (time.perf_counter() - started) * 1000.0
                return out

            tracker.update = wrapped_update
            setattr(tracker, "_gh_update_wrapped", True)

    def _wrap_callbacks(self) -> None:
        if self._callback_wrapped:
            return
        callbacks = getattr(self.model, "callbacks", None)
        if not isinstance(callbacks, dict):
            return
        for event_name, items in callbacks.items():
            if not isinstance(items, list):
                continue
            for idx, callback in enumerate(items):
                if getattr(callback, "_gh_timing_wrapped", False):
                    continue
                if str(event_name) not in {
                    "on_predict_start",
                    "on_predict_batch_start",
                    "on_predict_postprocess_end",
                    "on_predict_batch_end",
                    "on_predict_end",
                }:
                    continue

                event_key = str(event_name)

                def wrapped_cb(*args, __cb=callback, __event=event_key, **kwargs):
                    started = time.perf_counter()
                    out = __cb(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    self._callback_accum_ms += elapsed_ms
                    self._callback_event_ms[__event] = self._callback_event_ms.get(__event, 0.0) + elapsed_ms
                    self._callback_event_calls[__event] = self._callback_event_calls.get(__event, 0) + 1
                    return out

                setattr(wrapped_cb, "_gh_timing_wrapped", True)
                items[idx] = wrapped_cb
        self._callback_wrapped = True

    def run(self, frame: np.ndarray, *, source_changed: bool) -> tuple[Any, _TrackRuntimeBreakdown]:
        self._callback_accum_ms = 0.0
        self._callback_event_ms = {}
        self._callback_event_calls = {}
        self._tracker_update_accum_ms = 0.0

        track_started = time.perf_counter()
        if self.mode == "legacy":
            result = self.model.track(source=frame, persist=not source_changed, **self.kwargs)[0]
            self._wrap_callbacks()
            self._wrap_tracker_updates()
        else:
            if source_changed or not self._predictor_initialized or getattr(self.model, "predictor", None) is None:
                result = self.model.track(source=frame, persist=False, **self.kwargs)[0]
                self._predictor_initialized = True
                self._wrap_callbacks()
                self._wrap_tracker_updates()
            else:
                self._wrap_callbacks()
                predictor = self.model.predictor
                results = predictor(source=frame, stream=False)
                result = results[0]
                self._wrap_tracker_updates()
        track_wall_ms = (time.perf_counter() - track_started) * 1000.0

        callback_ms = max(0.0, self._callback_accum_ms)
        callback_predict_start_ms = max(0.0, self._callback_event_ms.get("on_predict_start", 0.0))
        callback_batch_start_ms = max(0.0, self._callback_event_ms.get("on_predict_batch_start", 0.0))
        callback_postprocess_end_ms = max(0.0, self._callback_event_ms.get("on_predict_postprocess_end", 0.0))
        callback_batch_end_ms = max(0.0, self._callback_event_ms.get("on_predict_batch_end", 0.0))
        callback_predict_end_ms = max(0.0, self._callback_event_ms.get("on_predict_end", 0.0))
        callback_known_ms = (
            callback_predict_start_ms
            + callback_batch_start_ms
            + callback_postprocess_end_ms
            + callback_batch_end_ms
            + callback_predict_end_ms
        )
        callback_other_ms = max(0.0, callback_ms - callback_known_ms)
        callback_dispatch_ms = max(0.0, track_wall_ms - callback_ms)

        predictor_pre_ms = max(0.0, callback_predict_start_ms + callback_batch_start_ms)
        pre_source_setup_ms = callback_predict_start_ms
        pre_batch_prepare_ms = callback_batch_start_ms
        pre_other_ms = max(0.0, predictor_pre_ms - pre_source_setup_ms - pre_batch_prepare_ms)
        predictor_post_ms = max(0.0, callback_postprocess_end_ms + callback_batch_end_ms + callback_predict_end_ms)
        tracker_update_ms = max(0.0, self._tracker_update_accum_ms)
        self._last_callback_calls = dict(self._callback_event_calls)
        self._last_callback_ms = dict(self._callback_event_ms)
        return result, _TrackRuntimeBreakdown(
            callback_ms=callback_ms,
            callback_dispatch_ms=callback_dispatch_ms,
            callback_predict_start_ms=callback_predict_start_ms,
            callback_batch_start_ms=callback_batch_start_ms,
            callback_postprocess_end_ms=callback_postprocess_end_ms,
            callback_batch_end_ms=callback_batch_end_ms,
            callback_predict_end_ms=callback_predict_end_ms,
            callback_other_ms=callback_other_ms,
            predictor_pre_ms=predictor_pre_ms,
            pre_source_setup_ms=pre_source_setup_ms,
            pre_batch_prepare_ms=pre_batch_prepare_ms,
            pre_other_ms=pre_other_ms,
            predictor_post_ms=predictor_post_ms,
            tracker_update_ms=tracker_update_ms,
            result_build_ms=0.0,
            result_wrap_ms=0.0,
            ultralytics_misc_ms=0.0,
            framework_other_ms=0.0,
        )


def detect(video, duration: float, config: dict, store=None, progress_callback: ProgressCallback | None = None, profiler: PerformanceProfiler | None = None) -> list[Candidate]:
    yolo = config["yolo"]
    keeper_cfg = config["keeper"]
    clips = config["clips"]
    runtime = config.get("runtime", {})
    benchmark_mode = bool(runtime.get("benchmark_mode", False))
    benchmark_force_noninteractive_keeper_selection = bool(runtime.get("benchmark_force_noninteractive_keeper_selection", False))
    device = yolo["device"]
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    requested_fp16 = bool(yolo.get("half", False))
    effective_fp16, fp16_fallback_reason = _resolve_fp16_state(device, requested_fp16)
    requested_precision = "FP16" if requested_fp16 else "FP32"
    effective_precision = "FP16" if effective_fp16 else "FP32"
    cuda_available = bool(torch.cuda.is_available())
    requested_tf32 = runtime.get("tf32")
    requested_cudnn_benchmark = runtime.get("cudnn_benchmark")
    tf32_enabled = bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)) if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul") else False
    cudnn_benchmark_enabled = bool(getattr(torch.backends.cudnn, "benchmark", False)) if hasattr(torch.backends, "cudnn") else False
    if requested_tf32 is not None:
        tf32_enabled = bool(requested_tf32)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32_enabled)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(tf32_enabled)
    if requested_cudnn_benchmark is not None and hasattr(torch.backends, "cudnn"):
        cudnn_benchmark_enabled = bool(requested_cudnn_benchmark)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark_enabled)
    track_execution_mode = str(runtime.get("track_execution_mode", "legacy")).strip().lower()
    if track_execution_mode not in {"legacy", "optimized"}:
        track_execution_mode = "legacy"
    decoder_execution_mode = str(runtime.get("decoder_execution_mode", "legacy")).strip().lower()
    if decoder_execution_mode not in {"legacy", "prefetch"}:
        decoder_execution_mode = "legacy"
    prefetch_queue_size = max(1, int(runtime.get("decoder_prefetch_queue_size", 4)))
    stride = max(1, int(yolo.get("frame_stride", 1)))
    decoder = create_decoder(video, config, stride)
    decoder_backend = str(config.get("decoder", {}).get("backend", "pyav")).strip().lower()
    if decoder_execution_mode == "prefetch" and decoder_backend == "opencv":
        decoder = PrefetchDecoder(decoder, queue_size=prefetch_queue_size)
    else:
        decoder_execution_mode = "legacy"
    from ultralytics import YOLO
    requested_backend = _normalize_backend(yolo.get("backend", "pytorch"))
    effective_backend = requested_backend
    backend_fallback_reason: str | None = None
    model_path = Path(yolo["model"])
    model_format = _model_format_from_path(model_path)
    tensorrt_version = ""
    onnxruntime_version = ""
    onnx_execution_provider = ""
    engine_cached = False
    engine_build_seconds = 0.0
    backend_load_seconds = 0.0
    warmup_seconds = 0.0

    tensorrt_available = importlib.util.find_spec("tensorrt") is not None
    onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None

    load_started = time.perf_counter()
    if requested_backend == "tensorrt" and not tensorrt_available:
        effective_backend = "pytorch"
        backend_fallback_reason = "tensorrt_unavailable"
    elif requested_backend == "onnx" and not onnxruntime_available:
        effective_backend = "pytorch"
        backend_fallback_reason = "onnxruntime_unavailable"

    if effective_backend == "tensorrt":
        import tensorrt as trt  # type: ignore[import-not-found]

        tensorrt_version = str(getattr(trt, "__version__", ""))
        cache_dir = Path("models") / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        precision_tag = "fp16" if effective_fp16 else "fp32"
        cache_key = _backend_cache_key(
            model=model_path,
            image_size=int(yolo["image_size"]),
            precision=effective_precision,
            backend="tensorrt",
            backend_version=tensorrt_version,
        )
        engine_path = cache_dir / f"{model_path.stem}-{int(yolo['image_size'])}-{precision_tag}-{cache_key}.engine"
        engine_cached = engine_path.exists()
        if not engine_cached:
            build_started = time.perf_counter()
            exporter = YOLO(str(model_path))
            exported = exporter.export(format="engine", imgsz=int(yolo["image_size"]), half=bool(effective_fp16), device=str(device), verbose=False)
            exported_path = Path(str(exported))
            if exported_path != engine_path:
                engine_path.parent.mkdir(parents=True, exist_ok=True)
                engine_path.write_bytes(exported_path.read_bytes())
            engine_build_seconds = time.perf_counter() - build_started
        model = YOLO(str(engine_path))
        model_format = "tensorrt_engine"
    elif effective_backend == "onnx":
        import onnxruntime as ort  # type: ignore[import-not-found]

        onnxruntime_version = str(getattr(ort, "__version__", ""))
        providers = list(ort.get_available_providers())
        onnx_execution_provider = providers[0] if providers else ""
        cache_dir = Path("models") / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        precision_tag = "fp16" if effective_fp16 else "fp32"
        cache_key = _backend_cache_key(
            model=model_path,
            image_size=int(yolo["image_size"]),
            precision=effective_precision,
            backend="onnx",
            backend_version=onnxruntime_version,
        )
        onnx_path = cache_dir / f"{model_path.stem}-{int(yolo['image_size'])}-{precision_tag}-{cache_key}.onnx"
        engine_cached = onnx_path.exists()
        if not engine_cached:
            build_started = time.perf_counter()
            exporter = YOLO(str(model_path))
            exported = exporter.export(format="onnx", imgsz=int(yolo["image_size"]), half=bool(effective_fp16), device=str(device), verbose=False)
            exported_path = Path(str(exported))
            if exported_path != onnx_path:
                onnx_path.parent.mkdir(parents=True, exist_ok=True)
                onnx_path.write_bytes(exported_path.read_bytes())
            engine_build_seconds = time.perf_counter() - build_started
        model = YOLO(str(onnx_path))
        model_format = "onnx"
    else:
        model = YOLO(yolo["model"])

    backend_load_seconds = time.perf_counter() - load_started

    warmup_started = time.perf_counter()
    warmup_frame = np.zeros((max(16, int(yolo["image_size"])), max(16, int(yolo["image_size"])), 3), dtype=np.uint8)
    warmup_kwargs: dict[str, Any] = {
        "source": warmup_frame,
        "persist": False,
        "tracker": yolo["tracker"],
        "classes": [PERSON_CLASS, SPORTS_BALL_CLASS],
        "conf": float(yolo["confidence"]),
        "iou": float(yolo["iou"]),
        "imgsz": int(yolo["image_size"]),
        "device": device,
        "verbose": False,
    }
    if effective_fp16:
        warmup_kwargs["half"] = True
    _ = model.track(**warmup_kwargs)
    warmup_seconds = time.perf_counter() - warmup_started
    track_runner = _TrackRunner(
        model,
        yolo=yolo,
        device=str(device),
        fp16_enabled=effective_fp16,
        mode=track_execution_mode,
    )
    if store is not None:
        store.save_video(path=Path(video.source_path) if hasattr(video, "source_path") else Path(video), duration=duration, fps=decoder.fps, width=decoder.width, height=decoder.height, frame_count=decoder.frame_count, decoder=config.get("decoder", {}).get("backend", "pyav"))

    identity: KeeperIdentity | None = None
    automatic_enabled = bool(keeper_cfg.get("automatic_initial_detection", True))
    bootstrap_seconds = max(0.0, float(keeper_cfg.get("bootstrap_seconds", 15.0)))
    bootstrap_max_seconds = max(bootstrap_seconds, float(keeper_cfg.get("bootstrap_max_seconds", 45.0)))
    bootstrap_recheck_seconds = max(1.0, float(keeper_cfg.get("bootstrap_recheck_seconds", 5.0)))
    next_bootstrap_check = bootstrap_seconds
    bootstrap = AutomaticGoalkeeperDetector(keeper_cfg, decoder.width, decoder.height) if automatic_enabled else None
    bootstrap_complete = not automatic_enabled
    bootstrap_result: dict[str, Any] = {"selected": False, "method": "disabled"}
    reid_count = 0
    reid_confidences: list[float] = []
    tracking_stable_reported = False
    reid_verbose = bool(keeper_cfg.get("reid_verbose_console", False))
    confidence_lock_after = max(0.0, float(keeper_cfg.get("confidence_lock_after_seconds", 30.0)))
    last_keeper: Optional[Box] = None
    interactive_done = False
    interactive_wait_seconds = 0.0
    candidates: list[Candidate] = []
    started = time.time()
    last_progress = 0.0
    preview_enabled = bool(runtime.get("live_preview", False))
    verbose_console = bool(runtime.get("verbose_console", False))
    detection_buffer: list[tuple] = []
    frame_buffer: list[tuple] = []
    processed_frames = 0
    callback_counts: dict[str, int] = {}
    callback_ms_map: dict[str, float] = {}
    minimum_ball_confidence = float(keeper_cfg.get("minimum_ball_confidence", 0.15))
    minimum_identity_confidence = float(keeper_cfg.get("minimum_identity_confidence", 0.45))
    event_cfg = config.get("event_engine", {})
    event_engine = GoalkeeperEventEngine(event_cfg, clips, duration, decoder.width, decoder.height)
    current_source_index: int | None = None
    source_stats: dict[int, dict[str, Any]] = {}
    benchmark_start = max(0.0, float(runtime.get("benchmark_start_seconds", 0.0)))
    benchmark_duration = max(0.0, float(runtime.get("benchmark_duration_seconds", 0.0)))
    benchmark_end = min(duration, benchmark_start + benchmark_duration) if benchmark_duration > 0 else duration
    effective_detection_duration = benchmark_end if benchmark_duration > 0 else duration
    box_mode = str(runtime.get("boxes_from_result_mode", "packed")).lower()
    box_converter = boxes_from_result_legacy if box_mode == "legacy" else boxes_from_result

    def _source_bucket(decoded_frame) -> dict[str, Any]:
        index = int(getattr(decoded_frame, "source_index", 0))
        name = str(getattr(decoded_frame, "source_name", ""))
        bucket = source_stats.setdefault(index, {
            "source_index": index,
            "source_name": name,
            "source_global_offset": float(decoded_frame.timestamp - getattr(decoded_frame, "source_local_timestamp", decoded_frame.timestamp)),
            "source_duration": 0.0,
            "frames_decoded": 0,
            "frames_sampled": 0,
            "frames_processed": 0,
            "keeper_frames": 0,
            "ball_frames": 0,
            "raw_candidates_created": 0,
            "accepted_candidates": 0,
            "rejected_candidates": 0,
            "keeper_identity_at_start": "Keeper #1" if identity is not None else "pending",
            "keeper_identity_at_end": "pending",
            "source_state_reset_performed": False,
            "keeper_reidentifications": 0,
            "decoder_restarts": 0,
            "ball_track_resets": 0,
            "keeper_track_resets": 0,
            "candidate_state_resets": 0,
        })
        if not bucket["source_name"] and name:
            bucket["source_name"] = name
        return bucket

    try:
        decoder_iterator = iter(decoder)
        decoded_frames = 0
        while True:
            decoder_read_ms = decoder_queue_wait_ms = consumer_queue_wait_ms = 0.0
            decode_started = time.perf_counter()
            try:
                decoder_item = next(decoder_iterator)
            except StopIteration:
                break
            if isinstance(decoder_item, DecoderItem):
                consumer_queue_wait_ms = max(0.0, float(decoder_item.queue_wait_ms))
                decoder_queue_wait_ms = max(0.0, float(decoder_item.producer_queue_wait_ms))
                signal = decoder_item.signal
                if signal is not None:
                    if signal.kind == "source_end":
                        continue
                    if signal.kind == "global_end":
                        break
                    if signal.kind == "exception":
                        raise RuntimeError(f"Decoder prefetch failed: {signal.error or 'unknown_error'}")
                decoded = cast(Any, decoder_item.frame)
                if decoded is None:
                    continue
                decoder_read_ms = max(0.0, float(decoder_item.read_ms))
                decoder_next_ms = consumer_queue_wait_ms
            else:
                decoded = cast(Any, decoder_item)
                decoder_next_ms = (time.perf_counter() - decode_started) * 1000.0
                decoder_read_ms = decoder_next_ms
            decoded_frames += 1
            if decoded.timestamp < benchmark_start:
                continue
            if decoded.timestamp > benchmark_end:
                break
            loop_started = time.perf_counter()
            database_ms = candidate_ms = preview_ms = 0.0
            frame_prepare_ms = keeper_identity_ms = keeper_reid_ms = keeper_histogram_ms = ball_selection_ms = event_engine_ms = 0.0
            progress_reporting_ms = profiler_overhead_ms = boxes_from_result_ms = 0.0
            database_buffer_ms = database_flush_ms = 0.0
            source_changed = current_source_index is None or decoded.source_index != current_source_index
            frame_prepare_started = time.perf_counter()
            if source_changed:
                if current_source_index is not None:
                    candidates.extend(event_engine.finish(decoded.timestamp))
                event_engine = GoalkeeperEventEngine(event_cfg, clips, duration, decoder.width, decoder.height)
                if identity is not None:
                    identity.reset_tracking_state_for_new_source()
                last_keeper = None
                current_source_index = decoded.source_index
                source_bucket = _source_bucket(decoded)
                source_bucket["source_state_reset_performed"] = True
                source_bucket["keeper_track_resets"] += 1
                source_bucket["ball_track_resets"] += 1
                source_bucket["candidate_state_resets"] += 1
            else:
                source_bucket = _source_bucket(decoded)
            frame_prepare_ms = (time.perf_counter() - frame_prepare_started) * 1000.0

            model_track_started = time.perf_counter()
            result, track_breakdown = track_runner.run(decoded.image, source_changed=source_changed)
            model_track_wall_ms = (time.perf_counter() - model_track_started) * 1000.0
            yolo_speed = getattr(result, "speed", None) or {}
            yolo_preprocess_ms = max(0.0, float(yolo_speed.get("preprocess", 0.0) or 0.0))
            yolo_inference_ms = max(0.0, float(yolo_speed.get("inference", 0.0) or 0.0))
            yolo_postprocess_ms = max(0.0, float(yolo_speed.get("postprocess", 0.0) or 0.0))
            track_overhead_ms = max(0.0, model_track_wall_ms - yolo_preprocess_ms - yolo_inference_ms - yolo_postprocess_ms)
            track_ultralytics_misc_ms = max(
                0.0,
                track_overhead_ms
                - track_breakdown.callback_ms
                - track_breakdown.predictor_pre_ms
                - track_breakdown.predictor_post_ms
                - track_breakdown.tracker_update_ms
                - track_breakdown.result_build_ms
                - track_breakdown.result_wrap_ms,
            )
            track_breakdown.ultralytics_misc_ms = track_ultralytics_misc_ms
            track_breakdown.framework_other_ms = track_ultralytics_misc_ms
            track_framework_other_ms = track_breakdown.framework_other_ms
            callback_counts = dict(track_runner._last_callback_calls)
            callback_ms_map = dict(track_runner._last_callback_ms)
            processed_frames += 1
            source_bucket["frames_decoded"] += 1
            source_bucket["frames_sampled"] += 1
            source_bucket["frames_processed"] += 1
            source_bucket["source_duration"] = max(source_bucket["source_duration"], float(decoded.source_local_timestamp))
            boxes_started = time.perf_counter()
            boxes = box_converter(result)
            boxes_from_result_ms = (time.perf_counter() - boxes_started) * 1000.0
            persons = [b for b in boxes if b.class_id == PERSON_CLASS]
            balls = [b for b in boxes if b.class_id == SPORTS_BALL_CLASS and b.confidence >= minimum_ball_confidence]
            if balls:
                source_bucket["ball_frames"] += 1
            keeper: Box | None = None
            identity_confidence = 0.0

            if identity is not None:
                keeper_identity_started = time.perf_counter()
                match = identity.match(decoded.image, persons, decoded.timestamp)
                keeper_identity_ms = (time.perf_counter() - keeper_identity_started) * 1000.0
                if match is not None:
                    keeper, identity_confidence = match.box, match.confidence
                    if match.reidentified:
                        reid_count += 1
                        source_bucket["keeper_reidentifications"] += 1
                        reid_confidences.append(identity_confidence)
                        if reid_verbose:
                            if verbose_console:
                                print(f"  Re-identified Keeper #1: ByteTrack {keeper.track_id}, confidence {identity_confidence:.2f}")
                    elif identity_confidence > 0:
                        reid_confidences.append(identity_confidence)
                    if (
                        not tracking_stable_reported
                        and decoded.timestamp >= confidence_lock_after
                        and len(reid_confidences) >= 8
                    ):
                        recent = reid_confidences[-min(60, len(reid_confidences)):]
                        stabilized = float(np.median(recent))
                        bootstrap_result["initial_confidence"] = float(bootstrap_result.get("confidence", stabilized))
                        bootstrap_result["stabilized_confidence"] = stabilized
                        bootstrap_result["confidence"] = max(float(bootstrap_result.get("confidence", 0.0)), stabilized)
                        bootstrap_result["confidence_locked_at_seconds"] = decoded.timestamp
                        if verbose_console:
                            print(f"Keeper tracking stable: Keeper #1, confidence {bootstrap_result['confidence']:.2f}")
                        tracking_stable_reported = True
            if keeper_identity_ms > 0:
                keeper_histogram_ms = keeper_identity_ms
                keeper_reid_ms = keeper_identity_ms

            # Version 0.13.10: keep gathering logical-person evidence beyond the initial
            # window. This handles recordings that start after a break while the
            # goalkeeper is advanced near midfield. Manual selection is only used
            # after the configured deferred evidence horizon.
            if identity is None and bootstrap is not None and not bootstrap_complete:
                bootstrap.observe(decoded.image, persons, balls, decoded.timestamp)
                if progress_callback:
                    progress_callback(min(0.03, decoded.timestamp / max(duration, 1)), f"Torwart-Evidenz wird gesammelt: {decoded.timestamp:.1f}/{bootstrap_max_seconds:.1f}s")
                if decoded.timestamp >= next_bootstrap_check:
                    selected_box, selected_frame, bootstrap_result = bootstrap.select()
                    next_bootstrap_check += bootstrap_recheck_seconds
                    bootstrap_result["observation_elapsed_seconds"] = decoded.timestamp
                    bootstrap_complete = selected_box is not None or decoded.timestamp >= bootstrap_max_seconds
                    if selected_box is not None and selected_frame is not None:
                        identity = KeeperIdentity(selected_frame, selected_box, keeper_cfg, decoder.width, decoder.height)
                        keeper = selected_box
                        identity_confidence = float(bootstrap_result.get("confidence", .7))
                        if verbose_console:
                            print(f"Automatic goalkeeper detection: Keeper #1 = ByteTrack {keeper.track_id}, confidence {identity_confidence:.2f}")
                    elif bootstrap_complete and bootstrap_result.get("ranking"):
                        if verbose_console:
                            print("Automatic goalkeeper detection remained inconclusive; using configured fallback.")
                    if store is not None:
                        store.set_state("keeper_detection", bootstrap_result)

            interactive_selection_enabled = bool(keeper_cfg.get("interactive_selection", True)) and not (
                benchmark_mode and benchmark_force_noninteractive_keeper_selection
            )

            if keeper is None and identity is None and bootstrap_complete and interactive_selection_enabled and not interactive_done and persons:
                if progress_callback:
                    progress_callback(min(0.03, decoded.timestamp / max(duration, 1)), "Automatische Erkennung unsicher – warte auf Torwartauswahl")
                selection_started = time.perf_counter()
                keeper = select_keeper(decoded.image, persons)
                interactive_wait_seconds += max(0.0, time.perf_counter() - selection_started)
                interactive_done = True
                if keeper:
                    identity = KeeperIdentity(decoded.image, keeper, keeper_cfg, decoder.width, decoder.height)
                    identity_confidence = 1.0
                    bootstrap_result = {"selected": True, "method": "interactive_fallback", "keeper_label": "Keeper #1", "selected_track_id": keeper.track_id,
                                        "confidence": 1.0, "manual_confirmation_confidence": 1.0,
                                        "automatic_confidence": float(bootstrap_result.get("confidence", 0.0)),
                                        "automatic_margin_to_second": float(bootstrap_result.get("margin_to_second", 0.0)),
                                        "ranking": bootstrap_result.get("ranking", [])}
                    if store is not None:
                        store.set_state("keeper_detection", bootstrap_result)
                    if verbose_console:
                        print(f"Selected Keeper #1: ByteTrack {keeper.track_id}")
            if keeper is None and identity is None and bootstrap_complete and not interactive_selection_enabled:
                keeper = choose_keeper_auto(persons, decoder.width, decoder.height, keeper_cfg["goal_roi"])
                if keeper:
                    identity = KeeperIdentity(decoded.image, keeper, keeper_cfg, decoder.width, decoder.height)
                    identity_confidence = 0.55
                    bootstrap_result = {"selected": True, "method": "single_frame_fallback", "keeper_label": "Keeper #1", "selected_track_id": keeper.track_id, "confidence": identity_confidence, "ranking": bootstrap_result.get("ranking", [])}
                    if store is not None:
                        store.set_state("keeper_detection", bootstrap_result)
            if keeper is None and last_keeper is not None:
                # Keep the old box only for display. It is deliberately not used to trigger candidates.
                display_keeper = last_keeper
            else:
                display_keeper = keeper
                if keeper is not None:
                    last_keeper = keeper

            candidate_started = time.perf_counter()
            closest_ball = None
            ball_selection_started = time.perf_counter()
            if keeper is not None and identity_confidence >= minimum_identity_confidence and balls:
                closest_ball = min(balls, key=lambda ball: normalized_distance(ball, keeper))
            ball_selection_ms = (time.perf_counter() - ball_selection_started) * 1000.0
            event_engine_started = time.perf_counter()
            emitted = event_engine.update(
                decoded.timestamp,
                keeper if identity_confidence >= minimum_identity_confidence else None,
                closest_ball,
                identity_confidence,
            )
            event_engine_ms = (time.perf_counter() - event_engine_started) * 1000.0
            candidates.extend(emitted)
            source_bucket["raw_candidates_created"] += len(emitted)
            if keeper is not None and identity_confidence >= minimum_identity_confidence:
                source_bucket["keeper_frames"] += 1
                source_bucket["keeper_identity_at_end"] = "Keeper #1"
            candidate_ms = (time.perf_counter() - candidate_started) * 1000

            if store is not None and runtime.get("store_detections", True):
                database_buffer_started = time.perf_counter()
                detection_buffer.extend((decoded.frame_index, decoded.timestamp, b.track_id, b.class_id, b.confidence, b.x1, b.y1, b.x2, b.y2) for b in boxes)
                frame_buffer.append((decoded.frame_index, decoded.timestamp, len(boxes), len(persons), len(balls), keeper.track_id if keeper else None))
                database_buffer_ms = (time.perf_counter() - database_buffer_started) * 1000.0
                if len(frame_buffer) >= 250:
                    db_started = time.perf_counter()
                    if detection_buffer:
                        store.append_detections(detection_buffer)
                    store.append_frames(frame_buffer)
                    detection_buffer.clear(); frame_buffer.clear()
                    database_ms = (time.perf_counter() - db_started) * 1000
                    database_flush_ms = database_ms

            if preview_enabled and processed_frames % max(1, int(runtime.get("preview_stride", 3))) == 0:
                preview_started = time.perf_counter()
                preview_enabled = _draw_preview(decoded.image, persons, balls, display_keeper, decoded.timestamp, len(candidates), identity_confidence)
                preview_ms = (time.perf_counter() - preview_started) * 1000

            loop_ms = (time.perf_counter() - loop_started) * 1000
            if profiler is not None:
                profiler_started = time.perf_counter()
                sample = profiler.sample(video_seconds=decoded.timestamp, frame_index=decoded.frame_index, processed_frames=processed_frames, loop_ms=loop_ms, speed=yolo_speed, candidate_ms=candidate_ms, database_ms=database_ms, preview_ms=preview_ms, raw_candidates=len(candidates), detections=len(boxes), persons=len(persons), balls=len(balls))
                profiler_overhead_ms = (time.perf_counter() - profiler_started) * 1000.0
                other_loop_ms = max(
                    0.0,
                    loop_ms
                    - frame_prepare_ms
                    - model_track_wall_ms
                    - boxes_from_result_ms
                    - keeper_identity_ms
                    - ball_selection_ms
                    - event_engine_ms
                    - candidate_ms
                    - database_buffer_ms
                    - database_flush_ms
                    - progress_reporting_ms
                    - preview_ms
                    - profiler_overhead_ms,
                )
                profiler.record_stage_frame(
                    {
                        "source_index": int(getattr(decoded, "source_index", 0)),
                        "source_name": str(getattr(decoded, "source_name", "")),
                        "frame_index": int(decoded.frame_index),
                        "video_seconds": float(decoded.timestamp),
                        "decoded_frames": decoded_frames,
                        "processed_frames": processed_frames,
                        "decoder_next_ms": decoder_next_ms,
                        "decoder_read_ms": decoder_read_ms,
                        "decoder_queue_wait_ms": decoder_queue_wait_ms,
                        "consumer_queue_wait_ms": consumer_queue_wait_ms,
                        "decoder_prefetch_frames": float(getattr(getattr(decoder, "stats", None), "prefetch_frames", 0)),
                        "decoder_queue_max_depth": float(getattr(getattr(decoder, "stats", None), "queue_max_depth", 0)),
                        "frame_prepare_ms": frame_prepare_ms,
                        "model_track_wall_ms": model_track_wall_ms,
                        "yolo_preprocess_ms": yolo_preprocess_ms,
                        "yolo_inference_ms": yolo_inference_ms,
                        "yolo_postprocess_ms": yolo_postprocess_ms,
                        "track_overhead_ms": track_overhead_ms,
                        "track_callback_ms": track_breakdown.callback_ms,
                        "track_callback_dispatch_ms": track_breakdown.callback_dispatch_ms,
                        "track_callback_predict_start_ms": track_breakdown.callback_predict_start_ms,
                        "track_callback_batch_start_ms": track_breakdown.callback_batch_start_ms,
                        "track_callback_postprocess_end_ms": track_breakdown.callback_postprocess_end_ms,
                        "track_callback_batch_end_ms": track_breakdown.callback_batch_end_ms,
                        "track_callback_predict_end_ms": track_breakdown.callback_predict_end_ms,
                        "track_callback_other_ms": track_breakdown.callback_other_ms,
                        "track_predictor_pre_ms": track_breakdown.predictor_pre_ms,
                        "track_pre_source_setup_ms": track_breakdown.pre_source_setup_ms,
                        "track_pre_batch_prepare_ms": track_breakdown.pre_batch_prepare_ms,
                        "track_pre_other_ms": track_breakdown.pre_other_ms,
                        "track_predictor_post_ms": track_breakdown.predictor_post_ms,
                        "track_tracker_update_ms": track_breakdown.tracker_update_ms,
                        "track_result_build_ms": track_breakdown.result_build_ms,
                        "track_result_wrap_ms": track_breakdown.result_wrap_ms,
                        "track_ultralytics_misc_ms": track_breakdown.ultralytics_misc_ms,
                        "track_framework_other_ms": track_framework_other_ms,
                        "boxes_from_result_ms": boxes_from_result_ms,
                        "keeper_identity_ms": keeper_identity_ms,
                        "keeper_reid_ms": keeper_reid_ms,
                        "keeper_histogram_ms": keeper_histogram_ms,
                        "ball_selection_ms": ball_selection_ms,
                        "event_engine_ms": event_engine_ms,
                        "candidate_processing_ms": candidate_ms,
                        "database_buffer_ms": database_buffer_ms,
                        "database_flush_ms": database_flush_ms,
                        "diagnostics_ms": 0.0,
                        "progress_reporting_ms": progress_reporting_ms,
                        "preview_ms": preview_ms,
                        "profiler_overhead_ms": profiler_overhead_ms,
                        "other_loop_ms": other_loop_ms,
                        "loop_ms": loop_ms,
                    }
                )
                if sample is not None:
                    if verbose_console:
                        print("  " + profiler.format_console(sample))
            now = time.time()
            if now - last_progress >= float(runtime["progress_interval_seconds"]):
                progress_started = time.perf_counter()
                rate = decoded.timestamp / max(0.001, now-started)
                msg = f"{decoded.timestamp/60:.1f}/{duration/60:.1f} min, {rate:.2f}x realtime, raw candidates: {len(candidates)}"
                if verbose_console:
                    print("  " + msg)
                if progress_callback:
                    progress_callback(min(0.95, decoded.timestamp / max(duration, 1)), msg)
                last_progress = now
                progress_reporting_ms = (time.perf_counter() - progress_started) * 1000.0
        candidates.extend(event_engine.finish(effective_detection_duration))
    finally:
        decoder.close()
        if detection_buffer and store is not None:
            store.append_detections(detection_buffer)
        if frame_buffer and store is not None:
            store.append_frames(frame_buffer)
        if preview_enabled:
            cv2.destroyAllWindows()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if store is not None:
        store.set_state("decoder_stats", {"read_recoveries": int(getattr(decoder, "read_recoveries", 0))})
        store.set_state(
            "precision",
            {
                "requested_fp16": requested_fp16,
                "effective_fp16": effective_fp16,
                "fp16_fallback_reason": fp16_fallback_reason,
                "requested_precision": requested_precision,
                "effective_precision": effective_precision,
                "requested_tf32": requested_tf32,
                "effective_tf32": bool(tf32_enabled),
                "requested_cudnn_benchmark": requested_cudnn_benchmark,
                "effective_cudnn_benchmark": bool(cudnn_benchmark_enabled),
                "requested_backend": requested_backend,
                "effective_backend": effective_backend,
                "backend_fallback_reason": backend_fallback_reason,
                "model_format": model_format,
                "engine_cached": engine_cached,
                "engine_build_seconds": engine_build_seconds,
                "backend_load_seconds": backend_load_seconds,
                "backend_warmup_seconds": warmup_seconds,
                "cuda_available": cuda_available,
                "device": str(device),
                "cuda_device": str(torch.cuda.current_device()) if cuda_available else "",
                "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "",
                "tensorrt_version": tensorrt_version,
                "onnxruntime_version": onnxruntime_version,
                "onnx_execution_provider": onnx_execution_provider,
                "python_version": platform.python_version(),
                "track_execution_mode": track_execution_mode,
                "decoder_execution_mode": decoder_execution_mode,
                "track_callback_last_calls": callback_counts,
                "track_callback_last_ms": callback_ms_map,
            },
        )
        for item in source_stats.values():
            item["decoder_restarts"] = int(getattr(decoder, "read_recoveries", 0))

    if reid_confidences:
        bootstrap_result["reidentification_count"] = reid_count
        bootstrap_result["mean_tracking_confidence"] = float(np.mean(reid_confidences))
        bootstrap_result["median_tracking_confidence"] = float(np.median(reid_confidences))
    bootstrap_result["selection_timed"] = bool(interactive_wait_seconds > 0.0)
    bootstrap_result["interactive_wait_seconds"] = float(interactive_wait_seconds)
    if store is not None:
        store.set_state("keeper_detection", bootstrap_result)
        ordered_source_stats = [source_stats[key] for key in sorted(source_stats)]
        store.set_state("source_diagnostics", ordered_source_stats)
    if reid_count:
        if verbose_console:
            print(f"Keeper tracking summary: Keeper #1 re-identified {reid_count}x, median confidence {bootstrap_result.get('median_tracking_confidence', 0.0):.2f}")

    for index, candidate in enumerate(candidates, 1):
        if not candidate.candidate_id:
            candidate.candidate_id = f"raw-{index:04d}"
        candidate.lifecycle_stage = "raw"
        candidate.lifecycle_events.append({"stage": "raw", "reason": "event_engine_candidate_created"})
    raw_candidate_count = len(candidates)
    if store is not None:
        store.set_state("raw_candidates", [c.as_dict() for c in candidates])

    recovered = recover_missed_keeper_actions(store, candidates, duration, config)
    for index, candidate in enumerate(recovered, 1):
        if not candidate.candidate_id:
            candidate.candidate_id = f"recovery-{index:04d}"
        candidate.lifecycle_stage = "recovered"
        candidate.lifecycle_reason = candidate.lifecycle_reason or "dynamic_recovery_pass"
        candidate.lifecycle_events.append({"stage": "recovery", "reason": candidate.lifecycle_reason})
    diagnostic_recovered = recover_uncovered_activity_windows(store, candidates + recovered, duration, config)
    candidates.extend(recovered)
    candidates.extend(diagnostic_recovered)
    if store is not None:
        store.set_state("recovery_candidates", [c.as_dict() for c in recovered + diagnostic_recovered])

    candidates = rescue_classic_keeper_actions(candidates, clips)
    for candidate in candidates:
        candidate.lifecycle_events.append({"stage": "validation", "accepted": candidate.accepted, "reason": candidate.rejection_reason or "validated"})
    if store is not None:
        store.set_state("validated_candidates", [c.as_dict() for c in candidates])
    merged = merge_candidates(candidates, float(clips["merge_gap_seconds"]), duration)
    for index, candidate in enumerate(merged, 1):
        if not candidate.candidate_id:
            candidate.candidate_id = f"merged-{index:04d}"
        candidate.lifecycle_stage = "merged"
        candidate.lifecycle_events.append({"stage": "merged", "reason": "merge_stage_complete"})
    if store is not None:
        store.set_state("merged_candidates", [c.as_dict() for c in merged])
    merged = extend_and_chain_clip_windows(merged, duration, clips)
    for candidate in merged:
        candidate.lifecycle_stage = "final"
        candidate.lifecycle_events.append({"stage": "clip_planning", "reason": candidate.clip_boundary_reason or "final_window_planned"})
    if store is not None:
        store.set_state(
            "detection_stats",
            {
                "raw_candidates": raw_candidate_count,
                "final_candidates": len(merged),
                "merged_candidates": max(0, raw_candidate_count + len(recovered) - len(merged)),
                "recovery_candidates": len(recovered) + len(diagnostic_recovered),
                "diagnostic_recovery_candidates": len(diagnostic_recovered),
                "processed_frames": processed_frames,
                "interactive_wait_seconds": float(interactive_wait_seconds),
                "keeper_selection_timed": bool(interactive_wait_seconds > 0.0),
                "requested_fp16": requested_fp16,
                "effective_fp16": effective_fp16,
                "fp16_fallback_reason": fp16_fallback_reason,
                "requested_precision": requested_precision,
                "effective_precision": effective_precision,
                "requested_tf32": requested_tf32,
                "effective_tf32": bool(tf32_enabled),
                "requested_cudnn_benchmark": requested_cudnn_benchmark,
                "effective_cudnn_benchmark": bool(cudnn_benchmark_enabled),
                "requested_backend": requested_backend,
                "effective_backend": effective_backend,
                "backend_fallback_reason": backend_fallback_reason,
                "model_format": model_format,
                "engine_cached": engine_cached,
                "engine_build_seconds": engine_build_seconds,
                "backend_load_seconds": backend_load_seconds,
                "backend_warmup_seconds": warmup_seconds,
                "cuda_available": cuda_available,
                "device": str(device),
                "track_execution_mode": track_execution_mode,
            },
        )
    # Acceptance is decided by GoalkeeperEventEngine using the threshold of the
    # final event category. Do not overwrite it here with the global threshold.
    # This is important for distribution and keeper_clearance, which intentionally
    # use lower category-specific thresholds.
    limit = int(clips.get("max_candidates", 0))
    return merged[:limit] if limit > 0 else merged
