from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator, Protocol

os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "65536")
import cv2
import numpy as np


@dataclass(slots=True)
class DecodedFrame:
    frame_index: int
    timestamp: float
    image: np.ndarray


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
                yield DecodedFrame(index, index / self.fps, frame)
            index += 1

    def close(self) -> None:
        self.capture.release()


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
            yield DecodedFrame(decoded_index, timestamp, frame.to_ndarray(format="bgr24"))

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
        for item in self.manifest.files:
            decoder = create_decoder(Path(item.path), self.config, self.stride)
            self._current = decoder
            try:
                for decoded in decoder:
                    yield DecodedFrame(global_frame, item.global_start_seconds + decoded.timestamp, decoded.image)
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
