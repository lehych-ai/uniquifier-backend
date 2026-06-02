"""Shared video I/O for the GPU backend.

Why this exists: the old code wrote frames with cv2.VideoWriter(*"mp4v") — the
MPEG-4 Part 2 codec. That looks soft, balloons file size, and gets re-encoded
again by the desktop Unifier, stacking quality loss. Here we instead pipe raw
BGR frames straight into an ffmpeg `libx264` encoder and remux the original
audio back in a single pass. Same approach for face swap and color swap.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


def _encoders(bin_path: str) -> str:
    try:
        return subprocess.run([bin_path, "-hide_banner", "-encoders"],
                              capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return ""


def _pick_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg_bin, h264_encoder). Conda ships a minimal ffmpeg without
    libx264 (so -c:v libx264 makes ffmpeg exit immediately → BrokenPipe on the
    first frame write). Prefer an ffmpeg that actually has an H.264 encoder; fall
    back across candidates and encoder names so we never pipe into a dead ffmpeg."""
    env = os.environ.get("FFMPEG_BIN")
    cands = [env] if env else []
    cands += ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", shutil.which("ffmpeg") or "ffmpeg"]
    seen, ordered = set(), []
    for c in cands:
        if c and c not in seen and (os.path.exists(c) or c == "ffmpeg"):
            seen.add(c)
            ordered.append(c)
    # Prefer the best encoder ACROSS binaries: a binary with libx264 beats one
    # that only has libopenh264, even if the latter is earlier on the path. (The
    # image's /usr/bin/ffmpeg has libopenh264 only; the static build we drop into
    # /usr/local/bin has libx264.)
    bins_enc = [(b, _encoders(b)) for b in ordered]
    for name in ("libx264", "h264_nvenc", "libopenh264"):
        for b, enc in bins_enc:
            if f" {name} " in enc or f" {name}\n" in enc:
                return b, name
    # nothing with H.264 — last resort: first binary + mpeg4 (always present)
    return (ordered[0] if ordered else "ffmpeg"), "mpeg4"


FFMPEG, H264_ENC = _pick_ffmpeg()


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
        cmd += ["-c:v", H264_ENC]
        # each encoder takes different rate-control flags:
        if H264_ENC == "libx264":
            cmd += ["-preset", preset, "-crf", str(crf)]
        elif H264_ENC == "h264_nvenc":
            cmd += ["-preset", "p5", "-rc", "vbr", "-cq", str(crf)]
        elif H264_ENC == "libopenh264":
            # libopenh264 has no -preset/-crf; drive it by bitrate.
            cmd += ["-b:v", "6M"]
        else:  # mpeg4
            cmd += ["-q:v", "3"]
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
        self._cmd = cmd
        # capture stderr so a dead ffmpeg surfaces its reason instead of an opaque
        # BrokenPipe; loglevel=error keeps it tiny so the PIPE never deadlocks.
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _ffmpeg_err(self) -> str:
        try:
            err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
        except Exception:  # noqa: BLE001
            err = ""
        return err.strip()[-600:]

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.meta.width or frame.shape[0] != self.meta.height:
            frame = cv2.resize(frame, (self.meta.width, self.meta.height))
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError:
            raise RuntimeError(
                f"ffmpeg died (enc={H264_ENC}, bin={FFMPEG}): {self._ffmpeg_err()}"
            ) from None

    def close(self) -> None:
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder exited {rc} (enc={H264_ENC}): {self._ffmpeg_err()}")

    def __enter__(self) -> "FfmpegWriter":
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.close()
        except Exception:
            if exc[0] is None:
                raise
