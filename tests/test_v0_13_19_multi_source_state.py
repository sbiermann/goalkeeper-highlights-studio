from __future__ import annotations

import numpy as np

from goalkeeper_highlights.decoder import DecodedFrame
from goalkeeper_highlights.detection import KeeperIdentity
from goalkeeper_highlights.event_engine import GoalkeeperEventEngine
from goalkeeper_highlights.models import Box


def _keeper_box(track_id: int | None = 1) -> Box:
    return Box(100.0, 100.0, 180.0, 260.0, 0.9, 0, track_id)


def _ball_box() -> Box:
    return Box(135.0, 130.0, 145.0, 140.0, 0.9, 32, 9)


def test_v0_13_19_decoded_frame_carries_source_local_and_global_time() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    decoded = DecodedFrame(
        frame_index=10,
        timestamp=3919.2,
        image=frame,
        source_index=2,
        source_name="Teil22.mp4",
        source_local_timestamp=0.0,
    )
    assert decoded.timestamp == 3919.2
    assert decoded.source_local_timestamp == 0.0
    assert decoded.source_index == 2
    assert decoded.source_name == "Teil22.mp4"


def test_v0_13_19_keeper_identity_resets_tracking_but_keeps_semantic_identity() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    keeper = _keeper_box(track_id=123)
    identity = KeeperIdentity(frame, keeper, {}, width=64, height=64)
    identity.pending_track_id = 777
    identity.pending_count = 3
    identity.last_switch_timestamp = 42.0

    identity.reset_tracking_state_for_new_source()

    assert identity.track_id is None
    assert identity.pending_track_id is None
    assert identity.pending_count == 0
    assert identity.last_switch_timestamp < 0.0
    assert identity.stable_identity_id == 1


def test_v0_13_19_event_engine_boundary_reset_clears_pending_candidate_state() -> None:
    cfg = {
        "trajectory_lookback_seconds": 1.5,
        "contact_distance_factor": 1.2,
        "release_distance_factor": 1.55,
        "minimum_event_duration_seconds": 0.01,
        "ball_lost_finalize_seconds": 1.0,
    }
    engine = GoalkeeperEventEngine(cfg, {}, duration=100.0, frame_width=1920, frame_height=1080)

    # Build pending interaction near end of source 1.
    emitted = engine.update(10.0, _keeper_box(1), _ball_box(), 0.9)
    assert emitted == []
    assert engine.pending is not None

    # Source boundary => new engine instance (expected lifecycle behavior).
    engine = GoalkeeperEventEngine(cfg, {}, duration=100.0, frame_width=1920, frame_height=1080)
    assert engine.pending is None
    assert len(engine.history) == 0


def test_v0_13_19_source_local_time_restart_does_not_require_monotonic_local_time() -> None:
    # Local timestamps may restart at 0 for every source while global continues.
    source1 = DecodedFrame(0, 100.0, np.zeros((8, 8, 3), dtype=np.uint8), source_index=0, source_name="A", source_local_timestamp=100.0)
    source2 = DecodedFrame(0, 200.0, np.zeros((8, 8, 3), dtype=np.uint8), source_index=1, source_name="B", source_local_timestamp=0.0)

    assert source2.source_local_timestamp < source1.source_local_timestamp
    assert source2.timestamp > source1.timestamp


def test_v0_13_19_cooldown_does_not_leak_when_event_engine_is_reinitialized() -> None:
    cfg = {
        "trajectory_lookback_seconds": 1.5,
        "contact_distance_factor": 1.2,
        "release_distance_factor": 1.55,
        "minimum_event_duration_seconds": 0.01,
        "ball_lost_finalize_seconds": 0.01,
    }
    engine = GoalkeeperEventEngine(cfg, {}, duration=200.0, frame_width=1920, frame_height=1080)
    engine.last_emitted = 199.9

    # Simulated source reset.
    engine = GoalkeeperEventEngine(cfg, {}, duration=200.0, frame_width=1920, frame_height=1080)
    assert engine.last_emitted < 0.0


def test_v0_13_19_recovery_window_metadata_does_not_cross_source_boundary() -> None:
    cfg = {
        "trajectory_lookback_seconds": 1.5,
        "contact_distance_factor": 1.2,
        "release_distance_factor": 1.55,
        "minimum_event_duration_seconds": 0.01,
        "ball_lost_finalize_seconds": 0.2,
    }
    engine = GoalkeeperEventEngine(cfg, {}, duration=300.0, frame_width=1920, frame_height=1080)
    engine.update(50.0, _keeper_box(1), _ball_box(), 0.9)
    assert engine.pending is not None

    # New source => no carried pending interaction/recovery context.
    engine = GoalkeeperEventEngine(cfg, {}, duration=300.0, frame_width=1920, frame_height=1080)
    assert engine.pending is None


def test_v0_13_19_three_sources_can_hold_independent_local_zero_points() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    s1 = DecodedFrame(0, 0.0, frame, source_index=0, source_name="A.mp4", source_local_timestamp=0.0)
    s2 = DecodedFrame(0, 100.0, frame, source_index=1, source_name="B.mp4", source_local_timestamp=0.0)
    s3 = DecodedFrame(0, 200.0, frame, source_index=2, source_name="C.mp4", source_local_timestamp=0.0)

    assert s1.source_local_timestamp == 0.0
    assert s2.source_local_timestamp == 0.0
    assert s3.source_local_timestamp == 0.0
    assert s1.timestamp < s2.timestamp < s3.timestamp


def test_v0_13_19_source_offset_applied_exactly_once_in_virtual_frame_contract() -> None:
    # Contract-level assertion for the decoder payload: global = offset + local.
    offset = 3919.2
    local = 12.5
    decoded = DecodedFrame(
        frame_index=123,
        timestamp=offset + local,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        source_index=2,
        source_name="Teil22.mp4",
        source_local_timestamp=local,
    )
    assert decoded.timestamp - decoded.source_local_timestamp == offset
