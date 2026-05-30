import cv2
import numpy as np
import os
import subprocess
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


# ─────────────────────────────────────────
# ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ (один раз)
# ─────────────────────────────────────────

_face_analyzer = None
_face_swapper = None

def get_models():
    global _face_analyzer, _face_swapper

    if _face_analyzer is None:
        print("Loading FaceAnalysis model...")
        _face_analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    if _face_swapper is None:
        print("Loading inswapper_128 model...")
        _face_swapper = get_model(
            "/root/models/inswapper_128.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    return _face_analyzer, _face_swapper


# ─────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────

def detect_faces(analyzer, image: np.ndarray):
    faces = analyzer.get(image)
    # Сортируем — самое большое лицо первым
    faces = sorted(faces, key=lambda x: x.bbox[2] - x.bbox[0], reverse=True)
    return faces

def get_source_face(analyzer, face_image_path: str):
    img = cv2.imread(face_image_path)
    if img is None:
        raise ValueError(f"Cannot load face image: {face_image_path}")

    faces = detect_faces(analyzer, img)
    if not faces:
        raise ValueError("No face detected in source image")

    return faces[0]


# ─────────────────────────────────────────
# SWAP ОДНОГО ИЗОБРАЖЕНИЯ
# ─────────────────────────────────────────

def swap_single_image(
    image: np.ndarray,
    source_face,
    analyzer,
    swapper,
    swap_all: bool = False,
) -> np.ndarray:
    target_faces = detect_faces(analyzer, image)

    if not target_faces:
        return image  # нет лица — возвращаем оригинал

    result = image.copy()

    if swap_all:
        for face in target_faces:
            result = swapper.get(result, face, source_face, paste_back=True)
    else:
        result = swapper.get(result, target_faces[0], source_face, paste_back=True)

    return result


# ─────────────────────────────────────────
# PREVIEW — первый кадр
# ─────────────────────────────────────────

def face_swap_preview(
    frame_path: str,
    face_path: str,
    output_path: str,
    swap_all: bool = False,
) -> str:
    analyzer, swapper = get_models()

    frame = cv2.imread(frame_path)
    source_face = get_source_face(analyzer, face_path)

    result = swap_single_image(frame, source_face, analyzer, swapper, swap_all)

    cv2.imwrite(output_path, result)
    print(f"Preview saved: {output_path}")
    return output_path


# ─────────────────────────────────────────
# APPLY TO VIDEO
# ─────────────────────────────────────────

def face_swap_video(
    video_path: str,
    face_path: str,
    output_path: str,
    swap_all: bool = False,
    skip_frames: int = 0,
) -> str:
    analyzer, swapper = get_models()

    # Загружаем лицо один раз
    source_face = get_source_face(analyzer, face_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_video = output_path.replace(".mp4", "_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))

    frame_idx = 0
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # skip_frames=1 → каждый второй кадр берём из предыдущего
        # Ускоряет в 2x, почти незаметно на видео
        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0 and last_result is not None:
            out.write(last_result)
        else:
            result = swap_single_image(frame, source_face, analyzer, swapper, swap_all)
            out.write(result)
            last_result = result

        frame_idx += 1

        if frame_idx % 30 == 0:
            progress = round((frame_idx / total_frames) * 100)
            print(f"Face swap: {progress}% ({frame_idx}/{total_frames})")

    cap.release()
    out.release()

    # Возвращаем звук
    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path
    ], capture_output=True)

    os.remove(temp_video)
    print(f"Done: {output_path}")
    return output_path
