from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    track_id: Optional[int] = None

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)


@dataclass(slots=True)
class Candidate:
    start: float
    end: float
    trigger_time: float
    min_normalized_distance: float
    keeper_track_id: Optional[int]
    accepted: bool = True
    category: str = "unclassified"
    confidence: float = 0.0
    description: str = ""
    qwen_raw: str = ""
    clip_path: str = ""
    identity_confidence: float = 0.0
    contact_frames: int = 0
    ball_confidence: float = 0.0
    heuristic_score: float = 0.0
    quality_score: float = 0.0
    rejection_reason: str = ""
    approach_speed: float = 0.0
    departure_speed: float = 0.0
    direction_change: float = 0.0
    keeper_motion: float = 0.0
    possession_duration: float = 0.0
    event_score: float = 0.0
    relative_ball_height: float = 0.0
    aerial_score: float = 0.0
    keeper_lateral_motion: float = 0.0
    keeper_label: str = "Keeper #1"
    acceptance_threshold: float = 0.0
    possession_bonus: float = 0.0
    cooldown_penalty: float = 0.0
    score_breakdown: dict[str, Any] = None  # type: ignore[assignment]
    action_start: float = 0.0
    action_end: float = 0.0
    clip_boundary_reason: str = ""
    keeper_x_normalized: float = 0.0
    keeper_y_normalized: float = 0.0
    recovery_candidate: bool = False
    routing_score: float = 0.0
    routing_category: str = "MEDIUM"
    routing_reason: str = ""
    qwen_retry_count: int = 0
    qwen_retry_confidence: float = 0.0
    qwen_first_pass_called: bool = False
    qwen_second_pass_called: bool = False
    qwen_second_pass_rescued: bool = False
    qwen_first_pass_seconds: float = 0.0
    qwen_second_pass_seconds: float = 0.0
    candidate_id: str = ""
    parent_candidate_ids: list[str] = None  # type: ignore[assignment]
    lifecycle_stage: str = "final"
    lifecycle_reason: str = ""
    lifecycle_events: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.score_breakdown is None:
            self.score_breakdown = {}
        if self.parent_candidate_ids is None:
            self.parent_candidate_ids = []
        if self.lifecycle_events is None:
            self.lifecycle_events = []

    def as_dict(self) -> dict:
        return asdict(self)
