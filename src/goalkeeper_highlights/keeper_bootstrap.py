from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np

from .models import Box


def _torso_histogram(frame: np.ndarray, box: Box) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(width, int(box.x2)), min(height, int(box.y2))
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None
    h = y2 - y1
    w = x2 - x1
    crop = frame[y1 + int(h * .08): y1 + max(2, int(h * .62)), x1 + int(w * .18): x1 + max(2, int(w * .82))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val, hue = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
    mask = ((sat >= 35) & (val >= 35)).astype(np.uint8) * 255
    mask[((hue >= 32) & (hue <= 92) & (sat >= 55))] = 0
    if cv2.countNonZero(mask) < 20:
        mask = None
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
    return hist


def _hist_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return .5
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def _rect_score(box: Box, width: int, height: int, regions: list[list[float]], falloff: float) -> float:
    cx, cy = box.center[0] / width, box.center[1] / height
    best = 0.0
    for x1, y1, x2, y2 in regions:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return 1.0
        dx = max(x1 - cx, 0.0, cx - x2)
        dy = max(y1 - cy, 0.0, cy - y2)
        best = max(best, math.exp(-math.hypot(dx, dy) / max(falloff, .01)))
    return best


@dataclass(slots=True)
class TrackEvidence:
    track_id: int
    observations: int = 0
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0
    area_sum: float = 0.0
    goal_area_sum: float = 0.0
    camera_proximity_sum: float = 0.0
    movement_sum: float = 0.0
    ball_contact_frames: int = 0
    confidence_sum: float = 0.0
    histograms: list[np.ndarray] = field(default_factory=list)
    last_center: tuple[float, float] | None = None
    last_box: Box | None = None
    last_frame: np.ndarray | None = None


class AutomaticGoalkeeperDetector:
    """Collects evidence during an initial window and selects Keeper #1.

    The stable keeper identity is independent from ByteTrack IDs. Bootstrap uses
    shirt uniqueness, apparent camera proximity/size, goal-area occupancy,
    movement pattern and observed ball contacts. Interactive selection remains a
    fallback when confidence is insufficient.
    """

    def __init__(self, cfg: dict[str, Any], width: int, height: int) -> None:
        self.cfg = cfg
        self.width = width
        self.height = height
        self.tracks: dict[int, TrackEvidence] = {}
        self.frame_peer_hists: list[list[tuple[int, np.ndarray | None]]] = []

    def observe(self, frame: np.ndarray, persons: list[Box], balls: list[Box], timestamp: float) -> None:
        regions = self.cfg.get("goal_regions") or [self.cfg.get("goal_roi", [0, 0, 1, 1])]
        falloff = float(self.cfg.get("goal_region_falloff", .20))
        frame_hists: list[tuple[int, np.ndarray | None]] = []
        for box in persons:
            if box.track_id is None:
                continue
            hist = _torso_histogram(frame, box)
            frame_hists.append((box.track_id, hist))
            evidence = self.tracks.setdefault(box.track_id, TrackEvidence(track_id=box.track_id, first_timestamp=timestamp))
            evidence.observations += 1
            evidence.last_timestamp = timestamp
            evidence.area_sum += box.area / max(1.0, self.width * self.height)
            evidence.goal_area_sum += _rect_score(box, self.width, self.height, regions, falloff)
            # Behind-goal footage: the near goalkeeper is usually large and low in the image.
            normalized_y = box.center[1] / self.height
            normalized_area = min(1.0, box.area / max(1.0, self.width * self.height * .06))
            evidence.camera_proximity_sum += .55 * normalized_y + .45 * normalized_area
            if evidence.last_center is not None:
                dx = (box.center[0] - evidence.last_center[0]) / self.width
                dy = (box.center[1] - evidence.last_center[1]) / self.height
                evidence.movement_sum += math.hypot(dx, dy)
            evidence.last_center = box.center
            evidence.last_box = box
            evidence.last_frame = frame.copy()
            evidence.confidence_sum += box.confidence
            if hist is not None and len(evidence.histograms) < int(self.cfg.get("bootstrap_histogram_samples", 20)):
                evidence.histograms.append(hist)
            for ball in balls:
                bx, by = ball.center
                dx = max(box.x1 - bx, 0.0, bx - box.x2)
                dy = max(box.y1 - by, 0.0, by - box.y2)
                if math.hypot(dx, dy) / max(box.diagonal, 1.0) <= float(self.cfg.get("bootstrap_ball_contact_distance", 1.25)):
                    evidence.ball_contact_frames += 1
                    break
        self.frame_peer_hists.append(frame_hists)

    def _mean_hist(self, evidence: TrackEvidence) -> np.ndarray | None:
        if not evidence.histograms:
            return None
        result = np.mean(np.stack(evidence.histograms), axis=0).astype(np.float32)
        cv2.normalize(result, result, 1, 0, cv2.NORM_L1)
        return result

    def rank(self) -> list[dict[str, Any]]:
        minimum_observations = max(1, int(self.cfg.get("bootstrap_min_observations", 4)))
        eligible = [e for e in self.tracks.values() if e.observations >= minimum_observations]
        means = {e.track_id: self._mean_hist(e) for e in eligible}
        rows: list[dict[str, Any]] = []
        for evidence in eligible:
            n = max(1, evidence.observations)
            peers = [_hist_distance(means[evidence.track_id], means[p.track_id]) for p in eligible if p.track_id != evidence.track_id]
            peers.sort()
            shirt_uniqueness = min(1.0, (float(np.median(peers[:3])) if peers else .35) / .70)
            goal_area = min(1.0, evidence.goal_area_sum / n)
            camera_proximity = min(1.0, evidence.camera_proximity_sum / n)
            mean_movement = evidence.movement_sum / max(1, n - 1)
            low_movement = math.exp(-mean_movement / max(.001, float(self.cfg.get("bootstrap_movement_scale", .025))))
            ball_contact = min(1.0, evidence.ball_contact_frames / max(1, int(self.cfg.get("bootstrap_ball_contact_target", 3))))
            persistence = min(1.0, n / max(1, int(self.cfg.get("bootstrap_observation_target", 12))))
            weights = self.cfg.get("bootstrap_weights", {})
            score = (
                float(weights.get("shirt_uniqueness", .32)) * shirt_uniqueness
                + float(weights.get("camera_proximity", .22)) * camera_proximity
                + float(weights.get("goal_area", .22)) * goal_area
                + float(weights.get("low_movement", .10)) * low_movement
                + float(weights.get("ball_contact", .08)) * ball_contact
                + float(weights.get("persistence", .06)) * persistence
            )
            rows.append({
                "track_id": evidence.track_id,
                "score": max(0.0, min(1.0, score)),
                "shirt_uniqueness": shirt_uniqueness,
                "camera_proximity": camera_proximity,
                "goal_area": goal_area,
                "low_movement": low_movement,
                "ball_contact": ball_contact,
                "persistence": persistence,
                "observations": evidence.observations,
                "last_timestamp": evidence.last_timestamp,
            })
        return sorted(rows, key=lambda row: row["score"], reverse=True)

    def select(self) -> tuple[Box | None, np.ndarray | None, dict[str, Any]]:
        ranking = self.rank()
        if not ranking:
            return None, None, {"selected": False, "reason": "no_eligible_tracks", "ranking": []}
        best = ranking[0]
        second = ranking[1]["score"] if len(ranking) > 1 else 0.0
        confidence = max(0.0, min(1.0, .65 * best["score"] + .35 * max(0.0, best["score"] - second) / .35))
        selected = best["score"] >= float(self.cfg.get("bootstrap_min_score", .48)) and confidence >= float(self.cfg.get("bootstrap_min_confidence", .52))
        evidence = self.tracks[int(best["track_id"])]
        result = {
            "selected": selected,
            "keeper_label": "Keeper #1",
            "selected_track_id": int(best["track_id"]) if selected else None,
            "score": best["score"],
            "confidence": confidence,
            "margin_to_second": best["score"] - second,
            "window_seconds": float(self.cfg.get("bootstrap_seconds", 8.0)),
            "ranking": ranking[:10],
            "weights": dict(self.cfg.get("bootstrap_weights", {
                "shirt_uniqueness": .17,
                "camera_proximity": .27,
                "goal_area": .22,
                "low_movement": .10,
                "ball_contact": .08,
                "persistence": .06,
            })),
            "method": "automatic_initial_window",
        }
        return (evidence.last_box if selected else None), (evidence.last_frame if selected else None), result
