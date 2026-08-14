from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import queue
import threading
import time
from typing import Iterator, Protocol

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = os.environ.get("GOALKEEPER_OPENCV_READ_ATTEMPTS", "65536")
import cv2
import numpy as np


@dataclass(slots=True)
class DecodedFrame:
    frame_index: int
    timestamp: float
    image: np.ndarray
    source_index: int = 0
    source_name: str = ""
    source_local_timestamp: float = 0.0


@dataclass(slots=True)
class DecoderSignal:
    kind: str
    source_index: int | None = None
    source_name: str = ""
    error: str | None = None


@dataclass(slots=True)
class DecoderItem:
    frame: DecodedFrame | None = None
    signal: DecoderSignal | None = None
    read_ms: float = 0.0
    producer_queue_wait_ms: float = 0.0
    queue_wait_ms: float = 0.0


@dataclass(slots=True)
class DecoderRuntimeStats:
    read_ms: float = 0.0
    producer_queue_wait_ms: float = 0.0
    consumer_queue_wait_ms: float = 0.0
    prefetch_frames: int = 0
    queue_max_depth: int = 0


class VideoDecoder(Protocol):
    fps: float
    width: int
    height: int
    frame_count: int
    def __iter__(self) -> Iterator[DecodedFrame]: ...
    def close(self) -> None: ...


class OpenCVDecoder:
    def __init__(self, path: Path, stride: int = 1, read_attempts: int = 65536, reopen_retries: int = 3) -> None:
        self.path = path
        self.stride = max(1, stride)
        self.read_attempts = max(4096, int(read_attempts))
        self.reopen_retries = max(0, int(reopen_retries))
        self.read_recoveries = 0
        os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(self.read_attempts)
        try:
            cv2.setLogLevel(2)  # errors only; keep OpenCV warnings out of the progress line
        except Exception:
            pass
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Video cannot be opened with OpenCV: {path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __iter__(self) -> Iterator[DecodedFrame]:
        index = 0
        consecutive_failures = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                # At the real end of file OpenCV reports the current frame near frame_count.
                current = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES) or index)
                if self.frame_count > 0 and current >= self.frame_count - 1:
                    break
                if consecutive_failures >= self.reopen_retries:
                    break
                consecutive_failures += 1
                self.read_recoveries += 1
                target = max(index + 1, current + 1)
                self.capture.release()
                self.capture = cv2.VideoCapture(str(self.path))
                if not self.capture.isOpened():
                    break
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                index = target
                continue
            consecutive_failures = 0
            if index % self.stride == 0:
                local_timestamp = index / self.fps
                yield DecodedFrame(
                    index,
                    local_timestamp,
                    frame,
                    source_index=0,
                    source_name=self.path.name,
                    source_local_timestamp=local_timestamp,
                )
            index += 1

    def close(self) -> None:
        self.capture.release()


class PrefetchDecoder:
    """Consumes frames from a source decoder in a dedicated producer thread."""

    def __init__(self, decoder: VideoDecoder, queue_size: int = 4) -> None:
        self.decoder = decoder
        self.queue_size = max(1, int(queue_size))
        self.fps = decoder.fps
        self.width = decoder.width
        self.height = decoder.height
        self.frame_count = decoder.frame_count
        self.read_recoveries = int(getattr(decoder, "read_recoveries", 0))
        self.stats = DecoderRuntimeStats()
        self._queue: queue.Queue[DecoderItem] = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._source_active_index: int | None = None
        self._source_active_name: str = ""

    def _put_item(self, item: DecoderItem) -> None:
        while not self._stop_event.is_set():
            wait_started = time.perf_counter()
            try:
                self._queue.put(item, timeout=0.1)
                wait_ms = (time.perf_counter() - wait_started) * 1000.0
                self.stats.producer_queue_wait_ms += wait_ms
                item.producer_queue_wait_ms += wait_ms
                self.stats.queue_max_depth = max(self.stats.queue_max_depth, int(self._queue.qsize()))
                return
            except queue.Full:
                self.stats.producer_queue_wait_ms += (time.perf_counter() - wait_started) * 1000.0

    def _producer_loop(self) -> None:
        iterator = iter(self.decoder)
        try:
            while not self._stop_event.is_set():
                read_started = time.perf_counter()
                try:
                    decoded = next(iterator)
                except StopIteration:
                    self._put_item(DecoderItem(signal=DecoderSignal(kind="global_end")))
                    return
                except Exception as exc:
                    self._put_item(DecoderItem(signal=DecoderSignal(kind="exception", error=str(exc))))
                    return
                read_ms = (time.perf_counter() - read_started) * 1000.0
                self.stats.read_ms += read_ms
                self.stats.prefetch_frames += 1

                if self._source_active_index is None:
                    self._source_active_index = int(getattr(decoded, "source_index", 0))
                    self._source_active_name = str(getattr(decoded, "source_name", ""))
                elif int(getattr(decoded, "source_index", 0)) != self._source_active_index:
                    self._put_item(
                        DecoderItem(
                            signal=DecoderSignal(
                                kind="source_end",
                                source_index=self._source_active_index,
                                source_name=self._source_active_name,
                            )
                        )
                    )
                    self._source_active_index = int(getattr(decoded, "source_index", 0))
                    self._source_active_name = str(getattr(decoded, "source_name", ""))

                self._put_item(DecoderItem(frame=decoded, read_ms=read_ms))

            self._put_item(DecoderItem(signal=DecoderSignal(kind="global_end")))
        finally:
            if self._source_active_index is not None:
                self._put_item(
                    DecoderItem(
                        signal=DecoderSignal(
                            kind="source_end",
                            source_index=self._source_active_index,
                            source_name=self._source_active_name,
                        )
                    )
                )

    def __iter__(self) -> Iterator[DecoderItem]:
        if self._thread is not None:
            raise RuntimeError("PrefetchDecoder iterator may only be consumed once")
        self._thread = threading.Thread(target=self._producer_loop, name="decoder-prefetch", daemon=True)
        self._thread.start()
        while True:
            wait_started = time.perf_counter()
            try:
                item = self._queue.get(timeout=0.1)
                queue_wait_ms = (time.perf_counter() - wait_started) * 1000.0
                self.stats.consumer_queue_wait_ms += queue_wait_ms
                item.queue_wait_ms = queue_wait_ms
                yield item
                if item.signal is not None and item.signal.kind in {"global_end", "exception"}:
                    return
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                self.stats.consumer_queue_wait_ms += (time.perf_counter() - wait_started) * 1000.0

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.read_recoveries = int(getattr(self.decoder, "read_recoveries", 0))
        self.decoder.close()


class PyAVDecoder:
    def __init__(self, path: Path, stride: int = 1, threads: int = 0) -> None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV is not installed. Run: python -m pip install av") from exc
        self.path = path
        self.stride = max(1, stride)
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        if threads > 0:
            self.stream.thread_count = threads
        self.stream.thread_type = "AUTO"
        rate = self.stream.average_rate or self.stream.guessed_rate
        self.fps = float(rate) if rate else 25.0
        self.width = int(self.stream.codec_context.width or 0)
        self.height = int(self.stream.codec_context.height or 0)
        self.frame_count = int(self.stream.frames or 0)

    def __iter__(self) -> Iterator[DecodedFrame]:
        for decoded_index, frame in enumerate(self.container.decode(self.stream)):
            if decoded_index % self.stride != 0:
                continue
            timestamp = float(frame.time) if frame.time is not None else decoded_index / self.fps
            yield DecodedFrame(
                decoded_index,
                timestamp,
                frame.to_ndarray(format="bgr24"),
                source_index=0,
                source_name=self.path.name,
                source_local_timestamp=timestamp,
            )

    def close(self) -> None:
        self.container.close()


class VirtualTimelineDecoder:
    """Sequentially decode original files and expose one global timestamp axis."""
    def __init__(self, manifest, config: dict, stride: int) -> None:
        self.manifest = manifest
        self.config = config
        self.stride = stride
        first = create_decoder(Path(manifest.files[0].path), config, stride)
        self.fps, self.width, self.height = first.fps, first.width, first.height
        first.close()
        self.frame_count = 0
        for item in manifest.files:
            probe = create_decoder(Path(item.path), config, stride)
            self.frame_count += probe.frame_count
            probe.close()
        self._current = None
        self.read_recoveries = 0

    def __iter__(self) -> Iterator[DecodedFrame]:
        global_frame = 0
        for source_index, item in enumerate(self.manifest.files):
            decoder = create_decoder(Path(item.path), self.config, self.stride)
            self._current = decoder
            try:
                for decoded in decoder:
                    yield DecodedFrame(
                        global_frame,
                        item.global_start_seconds + decoded.source_local_timestamp,
                        decoded.image,
                        source_index=source_index,
                        source_name=item.name,
                        source_local_timestamp=decoded.source_local_timestamp,
                    )
                    global_frame += self.stride
            finally:
                self.read_recoveries += int(getattr(decoder, "read_recoveries", 0))
                decoder.close()
        self._current = None

    def close(self) -> None:
        if self._current is not None:
            self._current.close()


def create_decoder(source, config: dict, stride: int) -> VideoDecoder:
    if hasattr(source, "files") and hasattr(source, "total_duration_seconds"):
        return VirtualTimelineDecoder(source, config, stride)
    path = Path(source)
    decoder_cfg = config.get("decoder", {})
    backend = str(decoder_cfg.get("backend", "pyav")).lower()
    if backend == "opencv":
        return OpenCVDecoder(
            path, stride,
            int(decoder_cfg.get("opencv_read_attempts", 65536)),
            int(decoder_cfg.get("opencv_reopen_retries", 3)),
        )
    if backend == "pyav":
        return PyAVDecoder(path, stride, int(decoder_cfg.get("threads", 0)))
    raise ValueError(f"Unknown decoder backend: {backend}. Supported: pyav, opencv")
