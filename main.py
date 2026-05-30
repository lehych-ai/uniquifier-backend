from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/uploads"
OUTPUT_DIR = "/tmp/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────
# СТАТУС
# ─────────────────────────────────────────

@app.get("/api/status")
async def status():
    return {"status": "ok"}


# ─────────────────────────────────────────
# ЗАГРУЗКА ВИДЕО + ПЕРВЫЙ КАДР
# ─────────────────────────────────────────

@app.post("/api/extract-frame")
async def extract_frame(video: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    video_path = f"{UPLOAD_DIR}/{video_id}.mp4"
    frame_path = f"{UPLOAD_DIR}/{video_id}_frame0.jpg"

    with open(video_path, "wb") as f:
        f.write(await video.read())

    os.system(f'ffmpeg -y -i "{video_path}" -vf "select=eq(n\\,0)" -q:v 2 "{frame_path}"')

    return {
        "video_id": video_id,
        "frame_url": f"/files/uploads/{video_id}_frame0.jpg"
    }


# ─────────────────────────────────────────
# COLOR SWAP — PREVIEW
# Принимает комбинацию тумблеров из UI
# ─────────────────────────────────────────

@app.post("/api/color-swap-preview")
async def api_color_swap_preview(
    video_id: str = Form(...),

    # Тумблеры первой строки
    random_color: bool = Form(False),
    random_clothes: bool = Form(False),
    random_bg: bool = Form(False),

    # Тумблеры второй строки
    upper_clothes: bool = Form(False),
    lower_clothes: bool = Form(False),
    full_outfit: bool = Form(False),

    # Промпты
    prompt_upper: str = Form(""),
    prompt_lower: str = Form(""),
    prompt_full: str = Form(""),
):
    from color_swap import color_swap_preview

    frame_path = f"{UPLOAD_DIR}/{video_id}_frame0.jpg"
    output_path = f"{OUTPUT_DIR}/{video_id}_color_preview.jpg"
    mask_path = f"{UPLOAD_DIR}/{video_id}_mask.npy"
    ref_path = f"{UPLOAD_DIR}/{video_id}_ref.jpg"

    # Определяем режим
    if random_bg:
        mode = "random_bg"
        region = "full"
        prompt = ""
    elif random_clothes:
        mode = "random_clothes"
        if upper_clothes:
            region = "upper"
            prompt = prompt_upper
        elif lower_clothes:
            region = "lower"
            prompt = prompt_lower
        else:
            region = "full"
            prompt = prompt_full
    else:
        # random_color
        mode = "random_color"
        if upper_clothes:
            region = "upper"
        elif lower_clothes:
            region = "lower"
        else:
            region = "full"
        prompt = ""

    color_swap_preview(
        frame_path=frame_path,
        output_path=output_path,
        mask_save_path=mask_path,
        ref_save_path=ref_path,
        mode=mode,
        region=region,
        prompt=prompt,
    )

    return FileResponse(output_path, media_type="image/jpeg")


# ─────────────────────────────────────────
# COLOR SWAP — APPLY TO VIDEO
# ─────────────────────────────────────────

@app.post("/api/color-swap-video")
async def api_color_swap_video(
    video_id: str = Form(...),
    random_color: bool = Form(False),
    random_clothes: bool = Form(False),
    random_bg: bool = Form(False),
):
    from color_swap import color_swap_video

    video_path = f"{UPLOAD_DIR}/{video_id}.mp4"
    output_path = f"{OUTPUT_DIR}/{video_id}_color_result.mp4"
    mask_path = f"{UPLOAD_DIR}/{video_id}_mask.npy"
    ref_path = f"{UPLOAD_DIR}/{video_id}_ref.jpg"

    if random_bg:
        mode = "random_bg"
    elif random_clothes:
        mode = "random_clothes"
    else:
        mode = "random_color"

    color_swap_video(
        video_path=video_path,
        output_path=output_path,
        mask_path=mask_path,
        ref_path=ref_path,
        mode=mode,
    )

    return {"output_url": f"/files/outputs/{video_id}_color_result.mp4"}


# ─────────────────────────────────────────
# FACE SWAP — PREVIEW
# ─────────────────────────────────────────

@app.post("/api/face-swap-preview")
async def api_face_swap_preview(
    video_id: str = Form(...),
    swap_all: bool = Form(False),
    face: UploadFile = File(...),
):
    from face_swap import face_swap_preview

    frame_path = f"{UPLOAD_DIR}/{video_id}_frame0.jpg"
    face_path = f"{UPLOAD_DIR}/{video_id}_face.jpg"
    output_path = f"{OUTPUT_DIR}/{video_id}_face_preview.jpg"

    with open(face_path, "wb") as f:
        f.write(await face.read())

    face_swap_preview(
        frame_path=frame_path,
        face_path=face_path,
        output_path=output_path,
        swap_all=swap_all,
    )

    return FileResponse(output_path, media_type="image/jpeg")


# ─────────────────────────────────────────
# FACE SWAP — APPLY TO VIDEO
# ─────────────────────────────────────────

@app.post("/api/face-swap-video")
async def api_face_swap_video(
    video_id: str = Form(...),
    swap_all: bool = Form(False),
    skip_frames: int = Form(0),
):
    from face_swap import face_swap_video

    video_path = f"{UPLOAD_DIR}/{video_id}.mp4"
    face_path = f"{UPLOAD_DIR}/{video_id}_face.jpg"
    output_path = f"{OUTPUT_DIR}/{video_id}_face_result.mp4"

    face_swap_video(
        video_path=video_path,
        face_path=face_path,
        output_path=output_path,
        swap_all=swap_all,
        skip_frames=skip_frames,
    )

    return {"output_url": f"/files/outputs/{video_id}_face_result.mp4"}


# ─────────────────────────────────────────
# ОТДАЁМ ФАЙЛЫ
# ─────────────────────────────────────────

@app.get("/files/uploads/{filename}")
async def get_upload(filename: str):
    return FileResponse(f"{UPLOAD_DIR}/{filename}")

@app.get("/files/outputs/{filename}")
async def get_output(filename: str):
    return FileResponse(f"{OUTPUT_DIR}/{filename}")
