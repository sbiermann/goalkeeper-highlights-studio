from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pytest

from goalkeeper_highlights.decoder import DecodedFrame, DecoderItem, PrefetchDecoder


@dataclass
class _FakeDecoder:
    frames: list[DecodedFrame]
    fps: float = 25.0
    width: int = 64
    height: int = 64
    frame_count: int = 0
    read_recoveries: int = 0
    closed: bool = False

    def __post_init__(self) -> None:
        self.frame_count = len(self.frames)

    def __iter__(self) -> Iterator[DecodedFrame]:
        yield from self.frames

    def close(self) -> None:
        self.closed = True


class _FailingDecoder(_FakeDecoder):
    def __iter__(self) -> Iterator[DecodedFrame]:
        for idx, frame in enumerate(self.frames):
            if idx == 1:
                raise RuntimeError("boom")
            yield frame


def _frame(index: int, source_index: int, source_name: str, ts: float) -> DecodedFrame:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return DecodedFrame(index, ts, image, source_index=source_index, source_name=source_name, source_local_timestamp=ts)


def test_v0_13_23_prefetch_keeps_frame_order_and_source_transition_signal() -> None:
    base = _FakeDecoder([
        _frame(0, 0, "a.mp4", 0.0),
        _frame(2, 0, "a.mp4", 0.08),
        _frame(4, 1, "b.mp4", 0.0),
    ])
    decoder = PrefetchDecoder(base, queue_size=2)
    items = list(decoder)
    decoder.close()

    kinds = [it.signal.kind for it in items if it.signal is not None]
    assert "source_end" in kinds
    assert "global_end" in kinds

    frames = [it.frame for it in items if it.frame is not None]
    assert [f.frame_index for f in frames] == [0, 2, 4]
    assert [f.source_index for f in frames] == [0, 0, 1]


def test_v0_13_23_prefetch_raises_decoder_exception_signal() -> None:
    base = _FailingDecoder([_frame(0, 0, "a.mp4", 0.0), _frame(2, 0, "a.mp4", 0.08)])
    decoder = PrefetchDecoder(base, queue_size=1)
    items = list(decoder)
    decoder.close()
    assert any(it.signal is not None and it.signal.kind == "exception" for it in items)


def test_v0_13_23_prefetch_runtime_stats_non_negative() -> None:
    base = _FakeDecoder([_frame(i * 2, 0, "a.mp4", i * 0.08) for i in range(5)])
    decoder = PrefetchDecoder(base, queue_size=1)
    items = list(decoder)
    decoder.close()

    assert any(isinstance(item, DecoderItem) and item.frame is not None for item in items)
    assert decoder.stats.read_ms >= 0.0
    assert decoder.stats.producer_queue_wait_ms >= 0.0
    assert decoder.stats.consumer_queue_wait_ms >= 0.0
    assert decoder.stats.prefetch_frames == 5
    assert decoder.stats.queue_max_depth >= 1
