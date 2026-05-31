"""Face swap — InsightFace inswapper_128 + optional GFPGAN restoration.

What changed vs. the original:
  1. Optional GFPGAN face enhancer. inswapper_128 emits a 128x128 face; on HD
     footage that reads as a soft, low-res patch. Roop/FaceFusion always pair it
     with a restorer. We load GFPGAN if its weights are present and run it on the
     swapped result; if not present we degrade gracefully (no crash).
  2. Fixed `skip_frames`. The old code wrote the *entire previous frame* on
     skipped frames, freezing the whole picture (background + body), which looks
     like a 15fps stutter. Now skipped frames keep the live frame and only the
     cached swapped *face* is re-pasted at the current detected box.
  3. Real libx264 output via the ffmpeg pipe (see video_io), not mp4v.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from video_io import FfmpegWriter, open_video

log = logging.getLogger("gpu.face_swap")

MODELS_DIR = os.environ.get("MODELS_DIR", "/root/models")
INSWAPPER_PATH = os.path.join(MODELS_DIR, "inswapper_128.onnx")
GFPGAN_PATH = os.path.join(MODELS_DIR, "GFPGANv1.4.pth")
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

ProgressCb = Optional[Callable[[float, str], None]]

_analyzer = None
_swapper = None
_enhancer: object | None = None
_enhancer_tried = False


def get_models():
    global _analyzer, _swapper
    if _analyzer is None:
        log.info("loading FaceAnalysis (buffalo_l)…")
        _analyzer = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
        _analyzer.prepare(ctx_id=0, det_size=(640, 640))
    if _swapper is None:
        log.info("loading inswapper_128…")
        _swapper = get_model(INSWAPPER_PATH, providers=PROVIDERS)
    return _analyzer, _swapper


def get_enhancer():
    """Lazy-load GFPGAN. Returns None (and logs once) if unavailable."""
    global _enhancer, _enhancer_tried
    if _enhancer_tried:
        return _enhancer
    _enhancer_tried = True
    if not os.path.isfile(GFPGAN_PATH):
        log.warning("GFPGAN weights not found at %s — running without enhancer", GFPGAN_PATH)
        return None
    try:
        from gfpgan import GFPGANer

        _enhancer = GFPGANer(
            model_path=GFPGAN_PATH,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        log.info("GFPGAN enhancer loaded")
    except Exception as exc:  # noqa: BLE001
        log.warning("GFPGAN load failed (%s) — running without enhancer", exc)
        _enhancer = None
    return _enhancer


def models_status() -> dict:
    return {
        "inswapper": os.path.isfile(INSWAPPER_PATH),
        "gfpgan": os.path.isfile(GFPGAN_PATH),
    }


# ─────────────────────────────────────────
# detection / swap
# ─────────────────────────────────────────

def detect_faces(analyzer, image: np.ndarray):
    faces = analyzer.get(image)
    return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)


def get_source_face(analyzer, face_image_path: str):
    img = cv2.imread(face_image_path)
    if img is None:
        raise ValueError(f"cannot load face image: {face_image_path}")
    faces = detect_faces(analyzer, img)
    log.info("source photo '%s': detected %d face(s)", face_image_path, len(faces))
    if not faces:
        raise ValueError("no face detected in the source photo")
    return faces[0]


def _enhance_face_region(frame: np.ndarray, box) -> np.ndarray:
    """Run GFPGAN on just the swapped face crop and feather it back in.

    Why not the whole frame: the old code enhanced the entire image every frame
    (only_center_face=False). That's the slowest step in the pipeline, it also
    "restores" the background and any bystanders, and can over-smooth. inswapper
    already gives us the box, so we crop a padded region, restore only that face,
    and blend it back — ~2x faster and it never touches the background.
    """
    enhancer = get_enhancer()
    if enhancer is None:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    # pad ~35% so GFPGAN has context to detect + align the face
    pw, ph = int((x2 - x1) * 0.35), int((y2 - y1) * 0.35)
    cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
    cx2, cy2 = min(w, x2 + pw), min(h, y2 + ph)
    if cx2 <= cx1 or cy2 <= cy1:
        return frame
    crop = frame[cy1:cy2, cx1:cx2]
    try:
        _, _, restored = enhancer.enhance(crop, has_aligned=False, only_center_face=True, paste_back=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("enhance failed: %s", exc)
        return frame
    if restored is None:
        return frame
    if restored.shape[:2] != crop.shape[:2]:
        restored = cv2.resize(restored, (crop.shape[1], crop.shape[0]))
    # feathered elliptical blend so the patch melts into the original frame
    mask = np.zeros(crop.shape[:2], np.float32)
    cv2.ellipse(mask, ((cx2 - cx1) // 2, (cy2 - cy1) // 2),
                ((cx2 - cx1) // 2, (cy2 - cy1) // 2), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2, (cx2 - cx1) // 12))[..., None]
    frame[cy1:cy2, cx1:cx2] = (restored * mask + crop * (1 - mask)).astype(np.uint8)
    return frame


def swap_single_image(image, source_face, analyzer, swapper, swap_all=False, enhance=True):
    target_faces = detect_faces(analyzer, image)
    log.info("target frame %s: detected %d face(s), swap_all=%s, enhance=%s",
             image.shape[:2], len(target_faces), swap_all, enhance)
    if not target_faces:
        log.warning("no face in target frame — returning original (no swap)")
        return image, []
    result = image.copy()
    targets = target_faces if swap_all else target_faces[:1]
    for face in targets:
        result = swapper.get(result, face, source_face, paste_back=True)
    boxes = [f.bbox.astype(int) for f in targets]
    if enhance:
        for bx in boxes:
            result = _enhance_face_region(result, bx)
    return result, boxes


def face_swap_preview(frame_path, face_path, output_path, swap_all=False, enhance=True):
    analyzer, swapper = get_models()
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"cannot load frame: {frame_path}")
    source_face = get_source_face(analyzer, face_path)
    result, _ = swap_single_image(frame, source_face, analyzer, swapper, swap_all, enhance)
    cv2.imwrite(output_path, result)
    return output_path


# ─────────────────────────────────────────
# video
# ─────────────────────────────────────────

def _paste_face(dst: np.ndarray, cached: np.ndarray, box) -> np.ndarray:
    """Paste a cached swapped-face crop into `dst` at the current bbox, feathered.

    Used on skipped frames so the body/background stay live while the (already
    swapped) face simply follows the head — far better than freezing the frame.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = dst.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return dst
    patch = cv2.resize(cached, (x2 - x1, y2 - y1))
    mask = np.zeros((y2 - y1, x2 - x1), np.float32)
    cv2.ellipse(mask, ((x2 - x1) // 2, (y2 - y1) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2, (x2 - x1) // 12))[..., None]
    dst[y1:y2, x1:x2] = (patch * mask + dst[y1:y2, x1:x2] * (1 - mask)).astype(np.uint8)
    return dst


def face_swap_video(
    video_path,
    face_path,
    output_path,
    swap_all=False,
    skip_frames=0,
    enhance=True,
    on_progress: ProgressCb = None,
):
    analyzer, swapper = get_models()
    source_face = get_source_face(analyzer, face_path)

    cap, meta = open_video(video_path)
    cached_face_crop: np.ndarray | None = None
    cached_box = None

    with FfmpegWriter(output_path, meta, audio_from=video_path) as writer:
        idx = 0
        for frame in _frames(cap):
            heavy = skip_frames <= 0 or idx % (skip_frames + 1) == 0
            if heavy:
                result, boxes = swap_single_image(frame, source_face, analyzer, swapper, swap_all, enhance)
                if boxes:
                    bx = boxes[0]
                    x1, y1, x2, y2 = [int(v) for v in bx]
                    x1, y1 = max(0, x1), max(0, y1)
                    cached_face_crop = result[y1:y2, x1:x2].copy()
                    cached_box = bx
                writer.write(result)
            elif cached_face_crop is not None:
                # light frame: re-detect (cheap) to follow the head, paste cached face
                faces = detect_faces(analyzer, frame)
                box = faces[0].bbox if faces else cached_box
                writer.write(_paste_face(frame.copy(), cached_face_crop, box))
            else:
                writer.write(frame)

            idx += 1
            if on_progress and meta.total_frames and idx % 15 == 0:
                on_progress(idx / meta.total_frames, f"face swap {idx}/{meta.total_frames}")

    cap.release()
    if on_progress:
        on_progress(1.0, "done")
    return output_path


def _frames(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
