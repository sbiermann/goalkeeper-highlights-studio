from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Any

from .models import Candidate
from . import __version__


def _candidate_record(index: int, candidate: Candidate) -> dict[str, Any]:
    data = candidate.as_dict()
    data["candidate_index"] = index
    data["decision_path"] = {
        "routing": candidate.routing_category,
        "routing_score": candidate.routing_score,
        "routing_reason": candidate.routing_reason,
        "qwen_first_pass": candidate.qwen_first_pass_called,
        "qwen_second_pass": candidate.qwen_second_pass_called,
        "rescued_by_second_pass": candidate.qwen_second_pass_rescued,
        "final_accepted": candidate.accepted,
        "rejection_reason": candidate.rejection_reason,
        "interaction_score": candidate.interaction_score,
        "clip_end_reason": candidate.clip_end_reason,
        "merged_reason": candidate.merged_reason,
    }
    return data


def _uncovered_suspicious_windows(store, candidates: list[Candidate], config: dict) -> list[dict[str, Any]]:
    """Find timeline regions worth reviewing even when no candidate survived.

    This is diagnostic only: it never changes classification. It deliberately uses
    loose thresholds so missed short saves remain visible in the review package.
    """
    rows = store.recovery_observations()
    if not rows:
        return []
    cfg = config.get("diagnostics", {}) or {}
    bucket_seconds = max(0.5, float(cfg.get("window_seconds", 2.0)))
    mask = float(cfg.get("candidate_mask_seconds", 1.0))
    min_ball = float(cfg.get("minimum_ball_confidence", 0.12))
    min_motion = float(cfg.get("minimum_keeper_motion", 0.035))
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(int(float(row["timestamp"]) // bucket_seconds), []).append(row)

    windows: list[dict[str, Any]] = []
    previous_keeper: tuple[float, float] | None = None
    for bucket, group in sorted(buckets.items()):
        start = bucket * bucket_seconds
        end = start + bucket_seconds
        covered = any(c.start - mask <= end and c.end + mask >= start for c in candidates)
        centers: list[tuple[float, float]] = []
        ball_conf = 0.0
        min_distance = math.inf
        for row in group:
            kx = (float(row["kx1"]) + float(row["kx2"])) / 2
            ky = (float(row["ky1"]) + float(row["ky2"])) / 2
            bx = (float(row["bx1"]) + float(row["bx2"])) / 2
            by = (float(row["by1"]) + float(row["by2"])) / 2
            diag = max(1.0, math.hypot(float(row["kx2"]) - float(row["kx1"]), float(row["ky2"]) - float(row["ky1"])))
            centers.append((kx, ky))
            ball_conf = max(ball_conf, float(row["ball_confidence"] or 0.0))
            min_distance = min(min_distance, math.hypot(kx - bx, ky - by) / diag)
        keeper_motion = 0.0
        if centers:
            first, last = centers[0], centers[-1]
            keeper_motion = math.hypot(last[0] - first[0], last[1] - first[1]) / 1000.0
            if previous_keeper is not None:
                keeper_motion = max(keeper_motion, math.hypot(first[0] - previous_keeper[0], first[1] - previous_keeper[1]) / 1000.0)
            previous_keeper = last
        score = min(1.0, ball_conf * 0.45 + min(keeper_motion * 4.0, 1.0) * 0.35 + (1.0 / (1.0 + min_distance)) * 0.20)
        if not covered and ball_conf >= min_ball and (keeper_motion >= min_motion or min_distance <= 1.25):
            windows.append({
                "start": round(start, 3), "end": round(end, 3),
                "diagnostic_score": round(score, 4),
                "ball_confidence": round(ball_conf, 4),
                "keeper_motion": round(keeper_motion, 4),
                "minimum_normalized_distance": round(min_distance, 4),
                "reason": "uncovered_ball_keeper_activity",
                "observation_count": len(group),
            })
    return sorted(windows, key=lambda item: item["diagnostic_score"], reverse=True)


def create_debug_package(output: Path, candidates: list[Candidate], timings: dict[str, Any], keeper_detection: dict[str, Any], config: dict, store) -> Path:
    debug_dir = output / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    records = [_candidate_record(i, c) for i, c in enumerate(candidates, 1)]
    (debug_dir / "final_candidates.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    # Compatibility name retained for existing review scripts.
    (debug_dir / "all_candidates.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    get_state = getattr(store, "get_state", lambda _key, default=None: default)
    stage_files = {
        "raw_candidates.json": get_state("raw_candidates", []),
        "validated_candidates.json": get_state("validated_candidates", []),
        "merged_candidates.json": get_state("merged_candidates", []),
        "recovery_candidates.json": get_state("recovery_candidates", []),
    }
    for name, payload in stage_files.items():
        (debug_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Complete machine-readable trace keyed by stable candidate id. Parent ids make
    # merge ancestry explicit instead of forcing reviewers to infer it from times.
    trace: dict[str, dict[str, Any]] = {}
    for stage_name, payload in stage_files.items():
        stage = stage_name.replace("_candidates.json", "")
        for item in payload or []:
            cid = str(item.get("candidate_id") or f"{stage}-anonymous-{len(trace)+1:04d}")
            node = trace.setdefault(cid, {"candidate_id": cid, "stages": [], "parent_candidate_ids": []})
            node["stages"].append({"stage": stage, "payload": item})
            node["parent_candidate_ids"] = list(dict.fromkeys([*node["parent_candidate_ids"], *(item.get("parent_candidate_ids") or [])]))
    for item in records:
        cid = str(item.get("candidate_id") or f"final-anonymous-{len(trace)+1:04d}")
        node = trace.setdefault(cid, {"candidate_id": cid, "stages": [], "parent_candidate_ids": []})
        node["stages"].append({"stage": "final", "payload": item})
        node["parent_candidate_ids"] = list(dict.fromkeys([*node["parent_candidate_ids"], *(item.get("parent_candidate_ids") or [])]))
    (debug_dir / "candidate_pipeline_trace.json").write_text(json.dumps(list(trace.values()), indent=2, ensure_ascii=False), encoding="utf-8")

    (debug_dir / "pipeline_summary.json").write_text(json.dumps({"timings": timings, "keeper_detection": keeper_detection}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (debug_dir / "effective_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    windows = _uncovered_suspicious_windows(store, candidates, config)
    (debug_dir / "suspicious_windows.json").write_text(json.dumps(windows, indent=2, ensure_ascii=False), encoding="utf-8")
    (debug_dir / "uncovered_suspicious_windows.json").write_text(json.dumps(windows, indent=2, ensure_ascii=False), encoding="utf-8")
    recovered_payload = stage_files.get("recovery_candidates.json", []) or []
    extended = []
    for window in windows:
        overlaps = [item for item in recovered_payload if float(item.get("start", 0.0)) <= window["end"] and float(item.get("end", 0.0)) >= window["start"]]
        extended.append({**window, "recovery_candidate_created": bool(overlaps),
                         "recovery_candidate_ids": [item.get("candidate_id", "") for item in overlaps],
                         "outcome": "promoted_to_recovery" if overlaps else "still_uncovered",
                         "next_debug_focus": "classification_or_merge" if overlaps else "keeper_or_ball_observation"})
    (debug_dir / "extended_recovery_analysis.json").write_text(json.dumps(extended, indent=2, ensure_ascii=False), encoding="utf-8")

    timeline = store.keeper_identity_timeline() if hasattr(store, "keeper_identity_timeline") else []
    with (debug_dir / "keeper_identity_timeline.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["frame_index", "timestamp", "keeper_track_id", "person_count", "ball_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(timeline)
    reid_events = []
    previous = None
    for row in timeline:
        current = row.get("keeper_track_id")
        if current is not None and previous is not None and current != previous:
            reid_events.append({"timestamp": row["timestamp"], "from_track_id": previous, "to_track_id": current})
        if current is not None:
            previous = current
    (debug_dir / "keeper_reid_events.json").write_text(json.dumps(reid_events, indent=2, ensure_ascii=False), encoding="utf-8")
    ball_gaps = store.ball_detection_gaps() if hasattr(store, "ball_detection_gaps") else []
    (debug_dir / "ball_detection_gaps.json").write_text(json.dumps(ball_gaps, indent=2, ensure_ascii=False), encoding="utf-8")

    with (debug_dir / "candidate_decisions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["candidate_index", "candidate_id", "lifecycle_stage", "lifecycle_reason", "trigger_time", "start", "end", "action_start", "action_end", "category", "accepted", "rejection_reason", "routing_category", "routing_score", "routing_reason", "event_score", "quality_score", "keeper_motion", "approach_speed", "departure_speed", "direction_change", "ball_confidence", "identity_confidence", "contact_frames", "recovery_candidate", "qwen_first_pass_called", "qwen_second_pass_called", "qwen_second_pass_rescued"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow({key: item.get(key, "") for key in fields})

    lifecycle_rows = []
    for stage_name, payload in stage_files.items():
        stage = stage_name.replace("_candidates.json", "")
        for index, item in enumerate(payload or [], 1):
            lifecycle_rows.append({"stage": stage, "index": index, "candidate_id": item.get("candidate_id", ""),
                                   "trigger_time": item.get("trigger_time", ""), "category": item.get("category", ""),
                                   "accepted": item.get("accepted", ""), "reason": item.get("lifecycle_reason", item.get("rejection_reason", ""))})
    for index, item in enumerate(records, 1):
        lifecycle_rows.append({"stage": "final", "index": index, "candidate_id": item.get("candidate_id", ""),
                               "trigger_time": item.get("trigger_time", ""), "category": item.get("category", ""),
                               "accepted": item.get("accepted", ""), "reason": item.get("rejection_reason", "")})
    with (debug_dir / "candidate_lifecycle.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["stage", "index", "candidate_id", "trigger_time", "category", "accepted", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(lifecycle_rows)

    readme = f"# Debug package v{__version__}\n" + """
This archive intentionally contains no video files.

Missed-save investigation order:
1. suspicious_windows.json
2. recovery_candidates.json
3. candidate_pipeline_trace.json (complete id/parent based lineage)
4. raw_candidates.json -> validated_candidates.json -> merged_candidates.json -> final_candidates.json
5. candidate_lifecycle.csv
5. keeper_identity_timeline.csv and keeper_reid_events.json
6. ball_detection_gaps.json

The package records every available candidate stage, logical keeper tracking changes,
ball-detection gaps and uncovered activity windows so a missed action can be assigned
to detection, keeper identity, recovery, merge, routing or final classification.
"""
    (debug_dir / "README_DEBUG.md").write_text(readme, encoding="utf-8")

    store.checkpoint()
    archive = output / f"goalkeeper_highlights_debug_v{__version__}.zip"
    excluded_suffixes = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path == archive or path.suffix.lower() in excluded_suffixes:
                continue
            zf.write(path, path.relative_to(output))
    return archive
