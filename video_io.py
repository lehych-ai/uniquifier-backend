"""Shared video I/O for the GPU backend.

Why this exists: the old code wrote frames with cv2.VideoWriter(*"mp4v") — the
MPEG-4 Part 2 codec. That looks soft, balloons file size, and gets re-encoded
again by the desktop Unifier, stacking quality loss. Here we instead pipe raw
BGR frames straight into an ffmpeg `libx264` encoder and remux the original
audio back in a single pass. Same approach for face swap and color swap.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

FFMPEG = "ffmpeg"


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    total_frames: int


def open_video(path: str) -> tuple[cv2.VideoCapture, VideoMeta]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    return cap, VideoMeta(fps=fps, width=w, height=h, total_frames=total)


def iter_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


class FfmpegWriter:
    """Pipes raw BGR frames into ffmpeg → libx264, muxing original audio.

    Usage:
        with FfmpegWriter(out, meta, audio_from=src) as w:
            for frame in ...:
                w.write(frame)
    """

    def __init__(
        self,
        out_path: str,
        meta: VideoMeta,
        audio_from: str | None = None,
        crf: int = 18,
        preset: str = "fast",
    ) -> None:
        self.out_path = out_path
        self.meta = meta
        size = f"{meta.width}x{meta.height}"
        cmd: list[str] = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            # raw video on stdin
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", size,
            "-r", f"{meta.fps:.6f}", "-i", "pipe:0",
        ]
        if audio_from:
            cmd += ["-i", audio_from]
        cmd += [
            "-map", "0:v:0",
        ]
        if audio_from:
            # `?` makes the audio stream optional — source may be silent.
            cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k", "-shortest"]
        else:
            cmd += ["-an"]
        cmd += [
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.meta.width or frame.shape[0] != self.meta.height:
            frame = cv2.resize(frame, (self.meta.width, self.meta.height))
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder exited {rc} for {self.out_path}")

    def __enter__(self) -> "FfmpegWriter":
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.close()
        except Exception:
            if exc[0] is None:
                raise
