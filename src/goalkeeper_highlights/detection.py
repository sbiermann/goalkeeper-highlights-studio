from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch

from .decoder import create_decoder
from .models import Box, Candidate
from .event_engine import GoalkeeperEventEngine
from .profiling import PerformanceProfiler
from .keeper_bootstrap import AutomaticGoalkeeperDetector

PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32
ProgressCallback = Callable[[float, str], None]


def boxes_from_result(result: Any) -> list[Box]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    ids = result.boxes.id.detach().cpu().numpy().astype(int) if result.boxes.id is not None else None
    return [Box(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(conf[i]), int(cls[i]), int(ids[i]) if ids is not None else None) for i, v in enumerate(xyxy)]


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
                candidate.accepted = False
                candidate.rejection_reason = "insufficient_recovery_interaction_score"
                candidate.score_breakdown["interaction_validation"] = -1.0
                return False
        else:
            candidate.accepted = False
            candidate.rejection_reason = "insufficient_interaction_dynamics"
            candidate.score_breakdown["interaction_validation"] = -1.0
            return False

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
            recovery_candidate=True, candidate_id=f"diagnostic-recovery-{index:04d}",
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
        if current.category in restart_categories and previous.category not in {"catch_or_control", "cross_claim_or_high_catch"}:
            return False
        if previous.category in restart_categories and current.category in restart_categories:
            return False
        return True
    for candidate in ordered:
        _has_real_keeper_interaction(candidate, clips_cfg)
        if not candidate.accepted:
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
        elif candidate.category in {"catch_or_control", "cross_claim_or_high_catch"}:
            # These usually end with the keeper holding the ball. 
            # If they are NOT chained to a distribution, they might end abruptly.
            candidate.clip_end_reason = "timeout"
        
        candidate.start = max(0.0, action_start - before)
        candidate.end = min(duration, action_end + after)
        candidate.clip_boundary_reason = "observed_action_window"

        if planned and planned[-1].accepted:
            previous = planned[-1]
            gap = action_start - (previous.action_end or previous.trigger_time)
            if 0.0 <= gap <= continuation_gap and same_keeper_phase(previous, candidate):
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
                    if candidate.category == "distribution" and candidate.departure_speed > 0.35:
                        previous.clip_end_reason = "kick" if candidate.approach_speed < candidate.departure_speed else "throw"
                    elif candidate.category in {"catch_or_control", "cross_claim_or_high_catch"}:
                        previous.clip_end_reason = "timeout"
                    
                    # Extension: Check if we can find a release event within the candidate action
                    if candidate.clip_end_reason == "timeout" and candidate.departure_speed > 0.25:
                        candidate.clip_end_reason = "controlled_release"
                        previous.clip_end_reason = "controlled_release"
                    
                    continue

        if candidate.end - candidate.start < minimum_duration:
            candidate.end = min(duration, candidate.start + minimum_duration)
        candidate.end = min(candidate.end, candidate.start + max_duration)
        planned.append(candidate)
    
    # Final pass: check for very related clips that should be merged despite large gaps (Stufe B)
    # This specifically addresses the 565s -> 592s case if there's evidence they belong together.
    # For now we implement the logic to check but keep it conservative.
    final_clips: list[Candidate] = []
    phase_gap_limit = max(continuation_gap, float(clips_cfg.get("phase_merge_gap_seconds", 30.0)))
    
    for candidate in planned:
        if not candidate.accepted:
            final_clips.append(candidate)
            continue
            
        if final_clips and final_clips[-1].accepted:
            previous = final_clips[-1]
            gap = candidate.start - previous.end
            
            # Diagnose v0.13.10 phase merge
            candidate.score_breakdown["phase_merge_checked"] = 1.0
            candidate.score_breakdown["phase_merge_gap"] = gap
            candidate.score_breakdown["phase_merge_decision"] = 0.0
            
            # Phase merge criteria:
            # 1. Same keeper
            # 2. Gap within limit
            # 3. No other conflicting events in between (already filtered here since we only see accepted)
            # 4. Total duration within limit
            same_keeper = (previous.keeper_label == candidate.keeper_label)
            within_limit = (gap <= phase_gap_limit)
            within_duration = (candidate.end - previous.start <= max_duration)
            
            # In a real run, we would check SQLite for activity in the gap.
            # Here we use a simplified version: if it's the same keeper and within a reasonable window
            # and the previous one ended with 'timeout' (suggesting it didn't clearly finish), we consider merging.
            # v0.13.10: Also ensure we don't merge unrelated restarts (distribution -> clearance)
            restart_categories = {"distribution", "keeper_clearance"}
            is_unrelated_restart = (candidate.category in restart_categories and previous.category not in {"catch_or_control", "cross_claim_or_high_catch"})
            
            should_phase_merge = same_keeper and within_limit and within_duration and previous.clip_end_reason == "timeout" and not is_unrelated_restart
            
            if should_phase_merge:
                previous.end = candidate.end
                previous.action_end = max(previous.action_end, candidate.action_end)
                previous.contact_frames += candidate.contact_frames
                previous.ball_confidence = max(previous.ball_confidence, candidate.ball_confidence)
                previous.description = f"{previous.description}; phase-merged with {candidate.category}".strip("; ")
                previous.score_breakdown["phase_merge_decision"] = 1.0
                previous.score_breakdown["phase_merge_reason"] = "same_keeper_related_phase"
                previous.clip_end_reason = candidate.clip_end_reason
                continue
                
        final_clips.append(candidate)
        
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


def detect(video, duration: float, config: dict, store=None, progress_callback: ProgressCallback | None = None, profiler: PerformanceProfiler | None = None) -> list[Candidate]:
    yolo = config["yolo"]
    keeper_cfg = config["keeper"]
    clips = config["clips"]
    runtime = config.get("runtime", {})
    device = yolo["device"]
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    stride = max(1, int(yolo.get("frame_stride", 1)))
    decoder = create_decoder(video, config, stride)
    from ultralytics import YOLO
    model = YOLO(yolo["model"])
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
    candidates: list[Candidate] = []
    started = time.time()
    last_progress = 0.0
    preview_enabled = bool(runtime.get("live_preview", False))
    verbose_console = bool(runtime.get("verbose_console", False))
    detection_buffer: list[tuple] = []
    frame_buffer: list[tuple] = []
    processed_frames = 0
    minimum_ball_confidence = float(keeper_cfg.get("minimum_ball_confidence", 0.15))
    minimum_identity_confidence = float(keeper_cfg.get("minimum_identity_confidence", 0.45))
    event_cfg = config.get("event_engine", {})
    event_engine = GoalkeeperEventEngine(event_cfg, clips, duration, decoder.width, decoder.height)
    try:
        for decoded in decoder:
            loop_started = time.perf_counter()
            database_ms = candidate_ms = preview_ms = 0.0
            result = model.track(
                source=decoded.image,
                persist=True,
                tracker=yolo["tracker"],
                classes=[PERSON_CLASS, SPORTS_BALL_CLASS],
                conf=float(yolo["confidence"]),
                iou=float(yolo["iou"]),
                imgsz=int(yolo["image_size"]),
                device=device,
                verbose=False,
            )[0]
            processed_frames += 1
            boxes = boxes_from_result(result)
            persons = [b for b in boxes if b.class_id == PERSON_CLASS]
            balls = [b for b in boxes if b.class_id == SPORTS_BALL_CLASS and b.confidence >= minimum_ball_confidence]
            keeper: Box | None = None
            identity_confidence = 0.0

            if identity is not None:
                match = identity.match(decoded.image, persons, decoded.timestamp)
                if match is not None:
                    keeper, identity_confidence = match.box, match.confidence
                    if match.reidentified:
                        reid_count += 1
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

            if keeper is None and identity is None and bootstrap_complete and keeper_cfg.get("interactive_selection", True) and not interactive_done and persons:
                if progress_callback:
                    progress_callback(min(0.03, decoded.timestamp / max(duration, 1)), "Automatische Erkennung unsicher – warte auf Torwartauswahl")
                keeper = select_keeper(decoded.image, persons)
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
            if keeper is None and identity is None and bootstrap_complete and not keeper_cfg.get("interactive_selection", True):
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
            if keeper is not None and identity_confidence >= minimum_identity_confidence and balls:
                closest_ball = min(balls, key=lambda ball: normalized_distance(ball, keeper))
            emitted = event_engine.update(
                decoded.timestamp,
                keeper if identity_confidence >= minimum_identity_confidence else None,
                closest_ball,
                identity_confidence,
            )
            candidates.extend(emitted)
            candidate_ms = (time.perf_counter() - candidate_started) * 1000

            if store is not None and runtime.get("store_detections", True):
                detection_buffer.extend((decoded.frame_index, decoded.timestamp, b.track_id, b.class_id, b.confidence, b.x1, b.y1, b.x2, b.y2) for b in boxes)
                frame_buffer.append((decoded.frame_index, decoded.timestamp, len(boxes), len(persons), len(balls), keeper.track_id if keeper else None))
                if len(frame_buffer) >= 250:
                    db_started = time.perf_counter()
                    if detection_buffer:
                        store.append_detections(detection_buffer)
                    store.append_frames(frame_buffer)
                    detection_buffer.clear(); frame_buffer.clear()
                    database_ms = (time.perf_counter() - db_started) * 1000

            if preview_enabled and processed_frames % max(1, int(runtime.get("preview_stride", 3))) == 0:
                preview_started = time.perf_counter()
                preview_enabled = _draw_preview(decoded.image, persons, balls, display_keeper, decoded.timestamp, len(candidates), identity_confidence)
                preview_ms = (time.perf_counter() - preview_started) * 1000

            loop_ms = (time.perf_counter() - loop_started) * 1000
            if profiler is not None:
                sample = profiler.sample(video_seconds=decoded.timestamp, frame_index=decoded.frame_index, processed_frames=processed_frames, loop_ms=loop_ms, speed=getattr(result, "speed", None), candidate_ms=candidate_ms, database_ms=database_ms, preview_ms=preview_ms, raw_candidates=len(candidates), detections=len(boxes), persons=len(persons), balls=len(balls))
                if sample is not None:
                    if verbose_console:
                        print("  " + profiler.format_console(sample))
            now = time.time()
            if now - last_progress >= float(runtime["progress_interval_seconds"]):
                rate = decoded.timestamp / max(0.001, now-started)
                msg = f"{decoded.timestamp/60:.1f}/{duration/60:.1f} min, {rate:.2f}x realtime, raw candidates: {len(candidates)}"
                if verbose_console:
                    print("  " + msg)
                if progress_callback:
                    progress_callback(min(0.95, decoded.timestamp / max(duration, 1)), msg)
                last_progress = now
        candidates.extend(event_engine.finish(duration))
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

    if reid_confidences:
        bootstrap_result["reidentification_count"] = reid_count
        bootstrap_result["mean_tracking_confidence"] = float(np.mean(reid_confidences))
        bootstrap_result["median_tracking_confidence"] = float(np.median(reid_confidences))
    if store is not None:
        store.set_state("keeper_detection", bootstrap_result)
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
            },
        )
    # Acceptance is decided by GoalkeeperEventEngine using the threshold of the
    # final event category. Do not overwrite it here with the global threshold.
    # This is important for distribution and keeper_clearance, which intentionally
    # use lower category-specific thresholds.
    limit = int(clips.get("max_candidates", 0))
    return merged[:limit] if limit > 0 else merged
