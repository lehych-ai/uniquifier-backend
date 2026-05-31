"""CloserAI GPU backend — FastAPI server that runs on a Vast.ai pod.

Contract (consumed by the desktop sidecar's VastBackend):

  GET  /api/status                      -> health + which weights are present
  POST /api/extract-frame   (video)     -> {video_id, frame_url}
  POST /api/color-swap-preview (form)   -> image/jpeg  (+ saves a reusable plan)
  POST /api/color-swap-video  (form)    -> {job_id, status:"started"}
  POST /api/face-swap-preview (face)    -> image/jpeg
  POST /api/face-swap-video   (form)    -> {job_id, status:"started"}
  GET  /api/progress/{video_id}         -> {percent, status, output_url?, error?}
  GET  /files/{kind}/{name}             -> serve uploaded/generated files

Long jobs run on a background thread and report into PROGRESS so the sidecar can
poll /api/progress and forward it to the renderer over SSE — no blocking POST.

Set API_TOKEN to require `Authorization: Bearer <token>` on every /api route;
leave it unset for open access (e.g. behind a private Vast proxy).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/outputs")
API_TOKEN = os.environ.get("API_TOKEN", "").strip()
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# video_id -> {"percent": float, "status": str, "output_url": str|None, "error": str|None}
PROGRESS: dict[str, dict] = {}

app = FastAPI(title="CloserAI GPU backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def require_token(authorization: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        return
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "invalid or missing API token")


def _up(video_id: str, name: str) -> str:
    return os.path.join(UPLOAD_DIR, f"{video_id}_{name}")


def _set_progress(video_id: str, percent: float, status: str, **extra) -> None:
    PROGRESS[video_id] = {"percent": round(percent, 4), "status": status,
                          "output_url": None, "error": None, **extra}


# ─────────────────────────────────────────
# status / files
# ─────────────────────────────────────────

@app.get("/api/status")
def status():
    import face_swap
    import color_swap
    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        pass
    return {
        "status": "ok",
        "gpu": gpu,
        "models": {**face_swap.models_status(), **color_swap.models_status()},
        "auth": bool(API_TOKEN),
    }


@app.get("/files/{kind}/{name}")
def serve_file(kind: str, name: str):
    base = UPLOAD_DIR if kind == "uploads" else OUTPUT_DIR
    path = os.path.join(base, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "file not found")
    return FileResponse(path)


# ─────────────────────────────────────────
# upload + first frame
# ─────────────────────────────────────────

@app.post("/api/extract-frame")
async def extract_frame(video: UploadFile = File(...), _=Depends(require_token)):
    video_id = uuid.uuid4().hex[:16]
    video_path = _up(video_id, "src.mp4")
    frame_path = _up(video_id, "frame0.jpg")
    with open(video_path, "wb") as f:
        f.write(await video.read())
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", video_path, "-frames:v", "1", "-q:v", "2", frame_path],
        check=False,
    )
    return {"video_id": video_id, "frame_url": f"/files/uploads/{video_id}_frame0.jpg"}


# ─────────────────────────────────────────
# color swap
# ─────────────────────────────────────────

def _color_mode(random_color: bool, random_clothes: bool, random_bg: bool) -> str:
    if random_bg:
        return "random_bg"
    if random_clothes:
        return "random_clothes"
    return "random_color"


def _color_region(upper: bool, lower: bool, full: bool) -> str:
    if full:
        return "full"
    if lower:
        return "lower"
    return "upper"


@app.post("/api/color-swap-preview")
def color_swap_preview_route(
    video_id: str = Form(...),
    random_color: bool = Form(False),
    random_clothes: bool = Form(False),
    random_bg: bool = Form(False),
    upper_clothes: bool = Form(False),
    lower_clothes: bool = Form(False),
    full_outfit: bool = Form(False),
    prompt_upper: str = Form(""),
    prompt_lower: str = Form(""),
    prompt_full: str = Form(""),
    seed: int | None = Form(None),
    _=Depends(require_token),
):
    from color_swap import color_swap_preview

    mode = _color_mode(random_color, random_clothes, random_bg)
    region = _color_region(upper_clothes, lower_clothes, full_outfit)
    prompt = {"upper": prompt_upper, "lower": prompt_lower, "full": prompt_full}[region]

    out = _up(video_id, "color_preview.jpg")
    plan = _up(video_id, "color_plan.npz")
    color_swap_preview(
        frame_path=_up(video_id, "frame0.jpg"),
        output_path=out,
        plan_save_path=plan,
        mode=mode, region=region, prompt=prompt, seed=seed,
    )
    return FileResponse(out, media_type="image/jpeg")


@app.post("/api/color-swap-video")
def color_swap_video_route(video_id: str = Form(...), _=Depends(require_token)):
    from color_swap import color_swap_video

    src = _up(video_id, "src.mp4")
    plan = _up(video_id, "color_plan.npz")
    if not os.path.isfile(plan):
        raise HTTPException(400, "run color-swap-preview first (no plan found)")
    out = os.path.join(OUTPUT_DIR, f"{video_id}_color.mp4")
    job_id = uuid.uuid4().hex[:12]

    def work():
        _set_progress(video_id, 0.0, "running")
        try:
            color_swap_video(src, out, plan,
                             on_progress=lambda p, m: _set_progress(video_id, p, "running"))
            PROGRESS[video_id] = {"percent": 1.0, "status": "done",
                                  "output_url": f"/files/outputs/{video_id}_color.mp4", "error": None}
        except Exception as exc:  # noqa: BLE001
            PROGRESS[video_id] = {"percent": 0.0, "status": "failed", "output_url": None, "error": str(exc)}

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id, "video_id": video_id, "status": "started"}


# ─────────────────────────────────────────
# face swap
# ─────────────────────────────────────────

@app.post("/api/face-swap-preview")
async def face_swap_preview_route(
    video_id: str = Form(...),
    swap_all: bool = Form(False),
    enhance: bool = Form(True),
    face: UploadFile = File(...),
    _=Depends(require_token),
):
    from face_swap import face_swap_preview

    face_path = _up(video_id, "face.jpg")
    with open(face_path, "wb") as f:
        f.write(await face.read())
    out = _up(video_id, "face_preview.jpg")
    face_swap_preview(_up(video_id, "frame0.jpg"), face_path, out, swap_all=swap_all, enhance=enhance)
    return FileResponse(out, media_type="image/jpeg")


@app.post("/api/face-swap-video")
def face_swap_video_route(
    video_id: str = Form(...),
    swap_all: bool = Form(False),
    skip_frames: int = Form(0),
    enhance: bool = Form(True),
    _=Depends(require_token),
):
    from face_swap import face_swap_video

    src = _up(video_id, "src.mp4")
    face_path = _up(video_id, "face.jpg")
    if not os.path.isfile(face_path):
        raise HTTPException(400, "run face-swap-preview first (no face uploaded)")
    out = os.path.join(OUTPUT_DIR, f"{video_id}_face.mp4")
    job_id = uuid.uuid4().hex[:12]

    def work():
        _set_progress(video_id, 0.0, "running")
        try:
            face_swap_video(src, face_path, out, swap_all=swap_all, skip_frames=skip_frames,
                            enhance=enhance,
                            on_progress=lambda p, m: _set_progress(video_id, p, "running"))
            PROGRESS[video_id] = {"percent": 1.0, "status": "done",
                                  "output_url": f"/files/outputs/{video_id}_face.mp4", "error": None}
        except Exception as exc:  # noqa: BLE001
            PROGRESS[video_id] = {"percent": 0.0, "status": "failed", "output_url": None, "error": str(exc)}

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id, "video_id": video_id, "status": "started"}


# ─────────────────────────────────────────
# progress
# ─────────────────────────────────────────

@app.get("/api/progress/{video_id}")
def progress(video_id: str, _=Depends(require_token)):
    return PROGRESS.get(video_id, {"percent": 0.0, "status": "unknown", "output_url": None, "error": None})


# ─────────────────────────────────────────
# ComfyUI control plane (drive the head-swap engine over HTTP, no terminal/SSH)
#
# Narrow surface — NOT arbitrary exec: install runs a fixed script, log tails a
# fixed file, diag lists fixed dirs. All gated behind the API token.
# ─────────────────────────────────────────

COMFY_DIR = os.environ.get("COMFY_DIR", "/root/ComfyUI")
COMFY_LOG = "/root/comfy.log"
COMFY_SETUP = "/root/comfy_setup.sh"
COMFY_SETUP_URL = "https://raw.githubusercontent.com/lehych-ai/uniquifier-backend/main/comfy_setup.sh"


def _comfy_running() -> bool:
    return subprocess.run(["pgrep", "-f", "comfy_setup.sh"], capture_output=True).returncode == 0


def _port_open(port: int) -> bool:
    import socket
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        s.close()


@app.post("/api/comfy/install")
def comfy_install(_=Depends(require_token)):
    """Pull the latest comfy_setup.sh and run it in the background (idempotent)."""
    if _comfy_running():
        return {"started": False, "reason": "comfy_setup already running"}
    subprocess.run(["wget", "-qO", COMFY_SETUP, COMFY_SETUP_URL], check=False)
    logf = open(COMFY_LOG, "w")
    subprocess.Popen(["bash", COMFY_SETUP], stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True)
    return {"started": True}


@app.get("/api/comfy/log")
def comfy_log(lines: int = 120, _=Depends(require_token)):
    try:
        with open(COMFY_LOG, errors="replace") as f:
            tail = f.readlines()[-lines:]
        return {"log": "".join(tail), "installing": _comfy_running()}
    except FileNotFoundError:
        return {"log": "", "installing": _comfy_running()}


@app.get("/api/comfy/diag")
def comfy_diag(_=Depends(require_token)):
    def ls(p: str) -> list[str]:
        try:
            return sorted(os.listdir(p))
        except Exception:
            return []
    m = os.path.join(COMFY_DIR, "models")
    return {
        "comfy_up": _port_open(8188),
        "installing": _comfy_running(),
        "nodes": ls(os.path.join(COMFY_DIR, "custom_nodes")),
        "models": {
            "checkpoints": ls(os.path.join(m, "checkpoints")),
            "vae": ls(os.path.join(m, "vae")),
            "animatediff_models": ls(os.path.join(m, "animatediff_models")),
            "controlnet": ls(os.path.join(m, "controlnet")),
            "ipadapter": ls(os.path.join(m, "ipadapter")),
            "clip_vision": ls(os.path.join(m, "clip_vision")),
            "loras": ls(os.path.join(m, "loras")),
            "insightface": ls(os.path.join(m, "insightface", "models")),
        },
    }


# ---- proxy to the local ComfyUI server (8188, not publicly exposed) ----

COMFY_API = "http://127.0.0.1:8188"


def _comfy_req(path: str, data: bytes | None = None, timeout: float = 30):
    req = urllib.request.Request(
        COMFY_API + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


@app.get("/api/comfy/nodes")
def comfy_nodes(_=Depends(require_token)):
    """List installed node class_types — used to author the workflow with exact names."""
    try:
        info = json.loads(_comfy_req("/object_info"))
        return {"count": len(info), "nodes": sorted(info.keys())}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"ComfyUI not reachable: {exc}")


@app.get("/api/comfy/node/{name}")
def comfy_node(name: str, _=Depends(require_token)):
    """Full input schema for one node (so the workflow wires inputs correctly)."""
    try:
        return json.loads(_comfy_req("/object_info/" + name))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"ComfyUI not reachable: {exc}")


@app.post("/api/comfy/prompt")
async def comfy_prompt(req: Request, _=Depends(require_token)):
    """Forward a workflow ({"prompt": {...}}) to ComfyUI; returns {prompt_id}."""
    body = await req.body()
    try:
        return json.loads(_comfy_req("/prompt", data=body, timeout=60))
    except urllib.error.HTTPError as exc:
        raise HTTPException(400, exc.read().decode(errors="replace"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"ComfyUI not reachable: {exc}")


@app.get("/api/comfy/result/{prompt_id}")
def comfy_result(prompt_id: str, _=Depends(require_token)):
    """Poll a submitted prompt: returns its /history entry (outputs when done)."""
    try:
        hist = json.loads(_comfy_req("/history/" + prompt_id))
        return hist.get(prompt_id, {"status": "pending"})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"ComfyUI not reachable: {exc}")


@app.post("/api/admin/redeploy")
def admin_redeploy(_=Depends(require_token)):
    """Pull the latest backend code and restart uvicorn (detached, survives our exit).
    Lets me ship new endpoints without a terminal — the running process can't
    hot-reload, so we relaunch it via setup.sh."""
    script = (
        "sleep 1; pkill -f 'uvicorn main:app' 2>/dev/null; "
        "wget -qO /root/setup.sh https://raw.githubusercontent.com/lehych-ai/uniquifier-backend/main/setup.sh; "
        "nohup bash /root/setup.sh > /root/setup.log 2>&1 &"
    )
    subprocess.Popen(["bash", "-c", script], start_new_session=True)
    return {"restarting": True}
