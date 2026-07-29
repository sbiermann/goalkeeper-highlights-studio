from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    h, w = y2 - y1, x2 - x1
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
        dx, dy = max(x1 - cx, 0.0, cx - x2), max(y1 - cy, 0.0, cy - y2)
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
    horizontal_movement_sum: float = 0.0
    vertical_movement_sum: float = 0.0
    center_goal_corridor_sum: float = 0.0
    isolation_sum: float = 0.0
    depth_sum: float = 0.0
    depth_sq_sum: float = 0.0
    field_excursion_frames: int = 0
    ball_contact_frames: int = 0
    confidence_sum: float = 0.0
    histograms: list[np.ndarray] = field(default_factory=list)
    last_center: tuple[float, float] | None = None
    last_box: Box | None = None
    last_frame: np.ndarray | None = None


class AutomaticGoalkeeperDetector:
    """Behaviour-first goalkeeper bootstrap with cross-track identity clustering.

    ByteTrack IDs are treated as temporary observations. Similar shirt appearance,
    compatible position and non-overlapping lifetimes are aggregated into a logical
    identity before ranking. This prevents a real keeper split across several short
    tracks from losing against a persistent field player.
    """

    def __init__(self, cfg: dict[str, Any], width: int, height: int) -> None:
        self.cfg, self.width, self.height = cfg, width, height
        self.tracks: dict[int, TrackEvidence] = {}
        self.rank_history: list[dict[str, Any]] = []

    def observe(self, frame: np.ndarray, persons: list[Box], balls: list[Box], timestamp: float) -> None:
        regions = self.cfg.get("goal_regions") or [self.cfg.get("goal_roi", [0, 0, 1, 1])]
        falloff = float(self.cfg.get("goal_region_falloff", .20))
        for box in persons:
            if box.track_id is None:
                continue
            hist = _torso_histogram(frame, box)
            e = self.tracks.setdefault(box.track_id, TrackEvidence(track_id=box.track_id, first_timestamp=timestamp))
            e.observations += 1
            e.last_timestamp = timestamp
            e.area_sum += box.area / max(1.0, self.width * self.height)
            e.goal_area_sum += _rect_score(box, self.width, self.height, regions, falloff)
            nx, ny = box.center[0] / self.width, box.center[1] / self.height
            normalized_area = min(1.0, box.area / max(1.0, self.width * self.height * .06))
            e.camera_proximity_sum += .55 * ny + .45 * normalized_area
            # A behind-goal keeper is usually close to the central goal axis, but can
            # be at very different image depths depending on which half is recorded.
            e.center_goal_corridor_sum += math.exp(-abs(nx - .5) / .20)
            e.depth_sum += ny
            e.depth_sq_sum += ny * ny
            if .25 <= nx <= .75 and .30 <= ny <= .78:
                e.field_excursion_frames += 1
            peer_distances = [math.hypot(box.center[0]-p.center[0], box.center[1]-p.center[1]) / max(box.diagonal, 1.0)
                              for p in persons if p.track_id != box.track_id]
            e.isolation_sum += min(1.0, (min(peer_distances) if peer_distances else 3.0) / 3.0)
            if e.last_center is not None:
                dx, dy = (box.center[0] - e.last_center[0]) / self.width, (box.center[1] - e.last_center[1]) / self.height
                e.movement_sum += math.hypot(dx, dy)
                e.horizontal_movement_sum += abs(dx)
                e.vertical_movement_sum += abs(dy)
            e.last_center, e.last_box, e.last_frame = box.center, box, frame.copy()
            e.confidence_sum += box.confidence
            if hist is not None and len(e.histograms) < int(self.cfg.get("bootstrap_histogram_samples", 30)):
                e.histograms.append(hist)
            for ball in balls:
                bx, by = ball.center
                dx, dy = max(box.x1-bx, 0.0, bx-box.x2), max(box.y1-by, 0.0, by-box.y2)
                if math.hypot(dx, dy) / max(box.diagonal, 1.0) <= float(self.cfg.get("bootstrap_ball_contact_distance", 1.5)):
                    e.ball_contact_frames += 1
                    break

    @staticmethod
    def _mean_hist(e: TrackEvidence) -> np.ndarray | None:
        if not e.histograms:
            return None
        result = np.mean(np.stack(e.histograms), axis=0).astype(np.float32)
        cv2.normalize(result, result, 1, 0, cv2.NORM_L1)
        return result

    def _clusters(self, eligible: list[TrackEvidence]) -> list[list[TrackEvidence]]:
        threshold = float(self.cfg.get("bootstrap_identity_hist_distance", .30))
        max_gap = float(self.cfg.get("bootstrap_identity_max_gap_seconds", 18.0))
        means = {e.track_id: self._mean_hist(e) for e in eligible}
        clusters: list[list[TrackEvidence]] = []
        for e in sorted(eligible, key=lambda x: (x.first_timestamp, x.track_id)):
            best_cluster, best_distance = None, 9.0
            for cluster in clusters:
                latest = max(cluster, key=lambda x: x.last_timestamp)
                # Two simultaneously visible players cannot be the same logical identity.
                # Allow only a small overlap caused by tracker hand-off.
                overlap_tolerance = float(self.cfg.get("bootstrap_identity_overlap_tolerance_seconds", .5))
                temporal_ok = (e.first_timestamp >= latest.last_timestamp - overlap_tolerance
                               and e.first_timestamp <= latest.last_timestamp + max_gap)
                distance = min(_hist_distance(means[e.track_id], means[m.track_id]) for m in cluster)
                if temporal_ok and distance < threshold and distance < best_distance:
                    best_cluster, best_distance = cluster, distance
            if best_cluster is None:
                clusters.append([e])
            else:
                best_cluster.append(e)
        return clusters

    def rank(self) -> list[dict[str, Any]]:
        minimum = max(1, int(self.cfg.get("bootstrap_min_observations", 4)))
        eligible = [e for e in self.tracks.values() if e.observations >= minimum]
        means = {e.track_id: self._mean_hist(e) for e in eligible}
        rows: list[dict[str, Any]] = []
        for cluster_index, cluster in enumerate(self._clusters(eligible), 1):
            n = max(1, sum(e.observations for e in cluster))
            member_ids = [e.track_id for e in cluster]
            peer_hists = [means[e.track_id] for e in eligible if e.track_id not in member_ids]
            cluster_hist = next((means[e.track_id] for e in cluster if means[e.track_id] is not None), None)
            peers = sorted(_hist_distance(cluster_hist, p) for p in peer_hists)
            shirt_uniqueness = min(1.0, (float(np.median(peers[:5])) if peers else .35) / .65)
            goal_area = min(1.0, sum(e.goal_area_sum for e in cluster) / n)
            camera_proximity = min(1.0, sum(e.camera_proximity_sum for e in cluster) / n)
            center_corridor = min(1.0, sum(e.center_goal_corridor_sum for e in cluster) / n)
            isolation = min(1.0, sum(e.isolation_sum for e in cluster) / n)
            movement = sum(e.movement_sum for e in cluster) / max(1, n-len(cluster))
            horizontal = sum(e.horizontal_movement_sum for e in cluster) / max(1, n-len(cluster))
            vertical = sum(e.vertical_movement_sum for e in cluster) / max(1, n-len(cluster))
            low_movement = math.exp(-movement / max(.001, float(self.cfg.get("bootstrap_movement_scale", .025))))
            patrol = min(1.0, horizontal / max(.001, vertical + .004)) * min(1.0, movement / .012)
            mean_depth = sum(e.depth_sum for e in cluster) / n
            variance = max(0.0, sum(e.depth_sq_sum for e in cluster)/n - mean_depth*mean_depth)
            depth_stability = math.exp(-math.sqrt(variance) / .08)
            ball_contact = min(1.0, sum(e.ball_contact_frames for e in cluster) / max(1, int(self.cfg.get("bootstrap_ball_contact_target", 3))))
            persistence = min(1.0, n / max(1, int(self.cfg.get("bootstrap_observation_target", 40))))
            field_excursion = sum(e.field_excursion_frames for e in cluster) / n
            weights = self.cfg.get("bootstrap_weights", {})
            score = (
                float(weights.get("shirt_uniqueness", .24))*shirt_uniqueness +
                float(weights.get("goal_area", .10))*goal_area +
                float(weights.get("camera_proximity", .08))*camera_proximity +
                float(weights.get("center_corridor", .18))*center_corridor +
                float(weights.get("isolation", .10))*isolation +
                float(weights.get("depth_stability", .10))*depth_stability +
                float(weights.get("horizontal_patrol", .07))*patrol +
                float(weights.get("ball_contact", .08))*ball_contact +
                float(weights.get("persistence", .10))*persistence +
                float(weights.get("low_movement", .05))*low_movement -
                float(weights.get("field_excursion_penalty", .08))*field_excursion
            )
            latest = max(cluster, key=lambda x: x.last_timestamp)
            rows.append({
                "logical_identity_id": f"keeper-candidate-{cluster_index:03d}", "track_id": latest.track_id,
                "member_track_ids": member_ids, "score": max(0.0, min(1.0, score)),
                "shirt_uniqueness": shirt_uniqueness, "camera_proximity": camera_proximity,
                "goal_area": goal_area, "center_corridor": center_corridor, "isolation": isolation,
                "depth_stability": depth_stability, "horizontal_patrol": patrol,
                "low_movement": low_movement, "ball_contact": ball_contact, "persistence": persistence,
                "field_excursion": field_excursion, "observations": n, "last_timestamp": latest.last_timestamp,
            })
        ranking = sorted(rows, key=lambda row: row["score"], reverse=True)
        self.rank_history.append({"timestamp": max((e.last_timestamp for e in eligible), default=0.0), "ranking": ranking[:8]})
        return ranking

    def select(self) -> tuple[Box | None, np.ndarray | None, dict[str, Any]]:
        ranking = self.rank()
        if not ranking:
            return None, None, {"selected": False, "reason": "no_eligible_tracks", "ranking": [], "rank_history": self.rank_history}
        best, second = ranking[0], (ranking[1]["score"] if len(ranking) > 1 else 0.0)
        margin = best["score"] - second
        confidence = max(0.0, min(1.0, .72*best["score"] + .28*min(1.0, margin/.12)))
        selected = best["score"] >= float(self.cfg.get("bootstrap_min_score", .50)) and confidence >= float(self.cfg.get("bootstrap_min_confidence", .56)) and margin >= float(self.cfg.get("bootstrap_min_margin", .035))
        evidence = self.tracks[int(best["track_id"])]
        result = {
            "selected": selected, "keeper_label": "Keeper #1", "selected_track_id": int(best["track_id"]) if selected else None,
            "logical_identity_id": best.get("logical_identity_id", f"keeper-candidate-track-{best['track_id']}"),
            "member_track_ids": best.get("member_track_ids", [best["track_id"]]),
            "score": best["score"], "confidence": confidence, "margin_to_second": margin,
            "window_seconds": float(self.cfg.get("bootstrap_max_seconds", 60.0)),
            "ranking": ranking[:max(1, int(self.cfg.get("bootstrap_top_candidates", 8)))],
            "rank_history": self.rank_history, "weights": dict(self.cfg.get("bootstrap_weights", {})),
            "method": "automatic_initial_window",
            "strategy": "multistage_behaviour_clustering",
            "selection_failure_reason": "" if selected else ("insufficient_margin" if margin < float(self.cfg.get("bootstrap_min_margin", .035)) else "insufficient_confidence"),
        }
        return (evidence.last_box if selected else None), (evidence.last_frame if selected else None), result
