from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .models import Box, Candidate


@dataclass(slots=True)
class Observation:
    timestamp: float
    ball_x: float
    ball_y: float
    keeper_x: float
    keeper_y: float
    distance: float
    ball_confidence: float
    identity_confidence: float
    keeper_track_id: int | None
    keeper_diagonal: float
    keeper_width: float
    keeper_height: float
    relative_ball_x: float
    relative_ball_y: float


@dataclass(slots=True)
class PendingInteraction:
    start_time: float
    first_close_time: float
    last_ball_time: float
    minimum_distance: float
    contact_frames: int
    ball_confidence_sum: float
    identity_confidence_sum: float
    track_id: int | None
    incoming: list[Observation]
    close: list[Observation]
    outgoing: list[Observation]


class GoalkeeperEventEngine:
    """Temporal goalkeeper event detector for saves, catches, punches and distributions.

    The detector uses only geometry and motion produced by YOLO/ByteTrack. Labels are
    conservative: ambiguous cases remain ``interaction`` and may later be refined by Qwen.
    """

    def __init__(self, cfg: dict, clips_cfg: dict, duration: float, frame_width: int = 1, frame_height: int = 1) -> None:
        self.cfg = cfg
        self.clips_cfg = clips_cfg
        self.duration = duration
        self.history: deque[Observation] = deque()
        self.pending: PendingInteraction | None = None
        self.last_emitted = -999.0
        self.last_accepted: Candidate | None = None
        self.frame_width = max(1, frame_width)
        self.frame_height = max(1, frame_height)

    def update(self, timestamp: float, keeper: Box | None, ball: Box | None, identity_confidence: float) -> list[Candidate]:
        emitted: list[Candidate] = []
        lookback = float(self.cfg.get("trajectory_lookback_seconds", 1.5))
        while self.history and timestamp - self.history[0].timestamp > lookback:
            self.history.popleft()

        if keeper is not None and ball is not None:
            kx, ky = keeper.center
            bx, by = ball.center
            dx = max(keeper.x1 - bx, 0.0, bx - keeper.x2)
            dy = max(keeper.y1 - by, 0.0, by - keeper.y2)
            distance = math.hypot(dx, dy) / keeper.diagonal
            obs = Observation(
                timestamp, bx, by, kx, ky, distance, ball.confidence,
                identity_confidence, keeper.track_id, keeper.diagonal,
                keeper.width, keeper.height,
                (bx - keeper.x1) / keeper.width,
                (by - keeper.y1) / keeper.height,
            )
            self.history.append(obs)
            close_threshold = float(self.cfg.get("contact_distance_factor", 1.15))
            release_threshold = float(self.cfg.get("release_distance_factor", 1.55))

            if self.pending is None and distance <= close_threshold:
                incoming = [item for item in self.history if item.timestamp < timestamp]
                self.pending = PendingInteraction(
                    start_time=incoming[0].timestamp if incoming else timestamp,
                    first_close_time=timestamp,
                    last_ball_time=timestamp,
                    minimum_distance=distance,
                    contact_frames=1,
                    ball_confidence_sum=ball.confidence,
                    identity_confidence_sum=identity_confidence,
                    track_id=keeper.track_id,
                    incoming=incoming,
                    close=[obs],
                    outgoing=[],
                )
            elif self.pending is not None:
                p = self.pending
                p.last_ball_time = timestamp
                p.minimum_distance = min(p.minimum_distance, distance)
                p.ball_confidence_sum += ball.confidence
                p.identity_confidence_sum += identity_confidence
                if distance <= close_threshold:
                    p.contact_frames += 1
                    p.close.append(obs)
                else:
                    p.outgoing.append(obs)
                    if distance >= release_threshold and timestamp - p.first_close_time >= float(self.cfg.get("minimum_event_duration_seconds", 0.16)):
                        candidate = self._finalize(timestamp)
                        if candidate is not None:
                            emitted.append(candidate)
        elif self.pending is not None:
            if timestamp - self.pending.last_ball_time >= float(self.cfg.get("ball_lost_finalize_seconds", 0.55)):
                candidate = self._finalize(timestamp)
                if candidate is not None:
                    emitted.append(candidate)
        return emitted

    def finish(self, timestamp: float) -> list[Candidate]:
        if self.pending is None:
            return []
        candidate = self._finalize(timestamp)
        return [candidate] if candidate is not None else []

    @staticmethod
    def _distance_rate(a: Observation, b: Observation) -> float:
        dt = max(0.04, b.timestamp - a.timestamp)
        return (a.distance - b.distance) / dt

    @staticmethod
    def _direction_change(incoming: list[Observation], outgoing: list[Observation]) -> float:
        if len(incoming) < 2 or len(outgoing) < 2:
            return 0.0
        a, b = incoming[-2], incoming[-1]
        c, d = outgoing[0], outgoing[-1]
        v1 = (b.ball_x-a.ball_x, b.ball_y-a.ball_y)
        v2 = (d.ball_x-c.ball_x, d.ball_y-c.ball_y)
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cosine = max(-1.0, min(1.0, (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)))
        return (1.0-cosine)/2.0

    @staticmethod
    def _range(values: list[float]) -> float:
        return max(values)-min(values) if values else 0.0

    def _finalize(self, end_time: float) -> Candidate | None:
        p, self.pending = self.pending, None
        if p is None:
            return None
        minimum_frames = int(self.cfg.get("minimum_contact_frames", 2))
        if p.contact_frames < minimum_frames:
            # Recovery path for a fast save that is visible for only one sampled
            # frame.  Require strong incoming motion and reliable identities so
            # this does not broadly lower the normal contact threshold.
            if not (
                p.contact_frames == 1
                and p.incoming
                and p.close
                and max(0.0, self._distance_rate(p.incoming[0], p.close[0]))
                    >= float(self.cfg.get("single_frame_save_min_approach", 0.90))
                and (p.identity_confidence_sum / max(1, p.contact_frames + len(p.outgoing)))
                    >= float(self.cfg.get("single_frame_save_min_identity", 0.75))
                and (p.ball_confidence_sum / max(1, p.contact_frames + len(p.outgoing)))
                    >= float(self.cfg.get("single_frame_save_min_ball_confidence", 0.45))
            ):
                return None
        all_obs = p.incoming + p.close + p.outgoing
        if not all_obs:
            return None

        approach = max(0.0, self._distance_rate(p.incoming[0], p.close[0])) if p.incoming and p.close else 0.0
        departure = max(0.0, -self._distance_rate(p.close[-1], p.outgoing[-1])) if p.close and p.outgoing else 0.0
        direction_change = self._direction_change(p.incoming, p.outgoing)
        possession = max(0.0, p.close[-1].timestamp-p.first_close_time) if p.close else 0.0
        first, last = all_obs[0], all_obs[-1]
        keeper_dx = abs(last.keeper_x-first.keeper_x)/max(first.keeper_width, 1.0)
        keeper_dy = abs(last.keeper_y-first.keeper_y)/max(first.keeper_height, 1.0)
        keeper_motion = math.hypot(last.keeper_x-first.keeper_x, last.keeper_y-first.keeper_y)/max(first.keeper_diagonal, 1.0)
        avg_ball = p.ball_confidence_sum/max(1, p.contact_frames+len(p.outgoing))
        avg_identity = p.identity_confidence_sum/max(1, p.contact_frames+len(p.outgoing))
        relative_y = sum(o.relative_ball_y for o in p.close)/max(1, len(p.close))
        relative_x = sum(o.relative_ball_x for o in p.close)/max(1, len(p.close))
        ball_vertical_travel = self._range([o.ball_y for o in all_obs])/max(first.keeper_height, 1.0)

        approach_s = min(1.0, approach/float(self.cfg.get("approach_speed_target", 0.75)))
        departure_s = min(1.0, departure/float(self.cfg.get("departure_speed_target", 0.75)))
        proximity_s = max(0.0, 1.0-p.minimum_distance/max(float(self.cfg.get("contact_distance_factor", 1.15)), .01))
        duration_s = min(1.0, possession/float(self.cfg.get("possession_duration_target", .8)))
        motion_s = min(1.0, keeper_motion/float(self.cfg.get("keeper_motion_target", .75)))
        high_ball_s = max(0.0, min(1.0, (float(self.cfg.get("high_ball_relative_y", .48))-relative_y)/.48))
        aerial_s = min(1.0, high_ball_s*.75 + min(1.0, ball_vertical_travel/1.2)*.25)
        deflection_s = max(direction_change, departure_s)

        score_breakdown = {
            "proximity": .20 * proximity_s,
            "approach": .20 * approach_s,
            "departure": .13 * departure_s,
            "direction_change": .12 * direction_change,
            "contact_duration": .10 * duration_s,
            "keeper_motion": .08 * motion_s,
            "aerial": .06 * aerial_s,
            "ball_confidence": .05 * avg_ball,
            "identity_confidence": .06 * avg_identity,
        }
        base_score = sum(score_breakdown.values())
        event_score = max(0.0, min(1.0, base_score))

        possession_min = float(self.cfg.get("possession_min_seconds", .55))
        short_contact_max = float(self.cfg.get("punch_max_possession_seconds", .42))
        category = "interaction"
        description = "Bestätigte Torwartaktion anhand von Balltrajektorie und Torwartbewegung."

        # Order matters: aerial actions and saves must be decided before generic control.
        if aerial_s >= float(self.cfg.get("aerial_action_threshold", .42)) and approach_s >= .18:
            if possession >= possession_min:
                category = "cross_claim_or_high_catch"
                description = "Torwart fängt oder kontrolliert einen hohen Ball beziehungsweise eine Flanke."
                event_score = max(event_score, .42 + .18*aerial_s + .10*duration_s)
            elif possession <= short_contact_max and deflection_s >= .30:
                category = "punch_clearance"
                description = "Torwart klärt einen hohen Ball wahrscheinlich mit der Faust oder lenkt ihn deutlich ab."
                event_score = max(event_score, .40 + .20*aerial_s + .15*deflection_s)
        if category == "interaction" and approach_s >= .35 and deflection_s >= .28:
            if keeper_dx >= float(self.cfg.get("dive_lateral_motion_target", .75)) or keeper_dy >= .45:
                category = "diving_save"
                description = "Torwart bewegt sich deutlich seitlich oder vertikal und wehrt den ankommenden Ball ab."
                event_score = max(event_score, .43 + .18*approach_s + .14*deflection_s + .10*motion_s)
            else:
                category = "save_or_deflection"
                description = "Torwart pariert oder lenkt einen ankommenden Ball ab."
                event_score = max(event_score, .42 + .18*approach_s + .15*deflection_s)
        if category == "interaction" and motion_s >= .50 and approach_s >= .22 and possession < possession_min:
            category = "sweep_or_one_on_one"
            description = "Torwart kommt dem Ball deutlich entgegen, vermutlich beim Herauslaufen oder im Eins-gegen-eins."
            event_score = max(event_score, .40 + .18*motion_s + .12*approach_s)
        if category == "interaction" and possession >= possession_min:
            if departure_s >= float(self.cfg.get("distribution_departure_threshold", .45)) and possession >= float(self.cfg.get("distribution_min_possession_seconds", .75)):
                category = "distribution"
                description = "Torwart kontrolliert den Ball und eröffnet das Spiel durch Abwurf, Abschlag oder Pass."
            else:
                category = "catch_or_control"
                description = "Torwart kontrolliert oder fängt den Ball."
        elif category == "interaction" and approach_s >= .22:
            category = "ball_contact"
            description = "Ball nähert sich dem Torwart und wird im Kontaktbereich bestätigt."

        # A confirmed multi-frame keeper/ball contact is strong evidence even when
        # trajectory features are weak (for example a calm pass or clearance).
        strong_contact = (
            avg_identity >= float(self.cfg.get("strong_contact_identity", .70))
            and avg_ball >= float(self.cfg.get("strong_contact_ball_confidence", .40))
            and p.contact_frames >= int(self.cfg.get("strong_contact_frames", 3))
            and p.minimum_distance <= float(self.cfg.get("strong_contact_max_distance", .35))
        )
        if strong_contact and category in {"interaction", "ball_contact"}:
            if possession >= float(self.cfg.get("distribution_min_possession_seconds", .75)) or p.contact_frames >= 4:
                category = "distribution"
                description = "Bestätigter Ballkontakt des Torwarts mit anschließender Spieleröffnung."
            else:
                category = "keeper_clearance"
                description = "Bestätigter Ballkontakt des Torwarts bei einer Klärungs- oder Abspielaktion."

        # Long, stable possession is strong evidence for a genuine goalkeeper action.
        bonus_steps = self.cfg.get("possession_bonus", {})
        possession_bonus = 0.0
        for seconds, bonus in sorted(((float(k), float(v)) for k, v in bonus_steps.items())):
            if possession >= seconds:
                possession_bonus = bonus
        if category in {"catch_or_control", "cross_claim_or_high_catch", "distribution"}:
            event_score += possession_bonus
            score_breakdown["possession_bonus"] = possession_bonus

        thresholds = self.cfg.get("category_thresholds", {})
        minimum_score = float(thresholds.get(category, self.cfg.get("minimum_event_score", .34)))
        if strong_contact:
            contact_bonus = float(self.cfg.get("strong_contact_bonus", .08))
            score_breakdown["confirmed_keeper_contact"] = contact_bonus
            event_score = max(event_score + contact_bonus, minimum_score + .01)

        event_score = max(0.0, min(1.0, event_score))
        cooldown = float(self.cfg.get("event_cooldown_seconds", 1.0))
        within_cooldown = p.first_close_time - self.last_emitted < cooldown
        accepted = event_score >= minimum_score
        rejection_reason = "" if accepted else "event_score_below_category_threshold"
        cooldown_penalty = 0.0

        candidate = Candidate(
            start=max(0.0, p.first_close_time-float(self.clips_cfg["seconds_before"])),
            end=min(self.duration, max(end_time, p.last_ball_time)+float(self.clips_cfg["seconds_after"])),
            trigger_time=p.first_close_time,
            min_normalized_distance=p.minimum_distance,
            keeper_track_id=p.track_id,
            accepted=accepted,
            category=category,
            confidence=event_score,
            description=description,
            identity_confidence=avg_identity,
            contact_frames=p.contact_frames,
            ball_confidence=avg_ball,
            heuristic_score=event_score,
            quality_score=event_score,
            rejection_reason=rejection_reason,
            approach_speed=approach,
            departure_speed=departure,
            direction_change=direction_change,
            keeper_motion=keeper_motion,
            possession_duration=possession,
            event_score=event_score,
            relative_ball_height=relative_y,
            aerial_score=aerial_s,
            keeper_lateral_motion=keeper_dx,
            keeper_label="Keeper #1",
            acceptance_threshold=minimum_score,
            possession_bonus=possession_bonus,
            cooldown_penalty=cooldown_penalty,
            score_breakdown=score_breakdown,
            action_start=all_obs[0].timestamp,
            action_end=max(end_time, p.last_ball_time),
            keeper_x_normalized=sum(o.keeper_x for o in all_obs) / max(1, len(all_obs)) / self.frame_width,
            keeper_y_normalized=sum(o.keeper_y for o in all_obs) / max(1, len(all_obs)) / self.frame_height,
        )

        # Cooldown collisions retain the stronger event rather than blindly keeping the first.
        if candidate.accepted and within_cooldown and self.last_accepted is not None:
            if candidate.event_score > self.last_accepted.event_score:
                self.last_accepted.accepted = False
                self.last_accepted.rejection_reason = "superseded_by_stronger_cooldown_candidate"
                self.last_accepted.cooldown_penalty = -1.0
                self.last_accepted.score_breakdown["cooldown"] = -1.0
            else:
                candidate.accepted = False
                candidate.rejection_reason = "cooldown_lower_score_than_previous_candidate"
                candidate.cooldown_penalty = -1.0
                candidate.score_breakdown["cooldown"] = -1.0
        if candidate.accepted:
            self.last_emitted = p.first_close_time
            self.last_accepted = candidate
        return candidate
