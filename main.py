"""Uniquifier GPU backend — FastAPI orchestrator on a Vast pod.

New stack (2026-06): drives ComfyUI workflows instead of in-process SD.
  Flux.2 Klein  → first-frame edits (head swap / clothes / bg / color)
  Wan 2.2 Animate (Kijai WanVideoWrapper) → full video gen (movement/static)

Contract:
  GET  /api/version                      -> git HEAD of running code
  GET  /api/status                       -> health + comfy state
  POST /api/extract-frame   (video)      -> {video_id, frame_url}   (stores src.mp4 + frame0.jpg)
  POST /api/flux-edit        (form)      -> edited first frame (image/jpeg)
  POST /api/wan-animate      (json)      -> {status:"started"}  (poll /api/progress)
  GET  /api/progress/{video_id}          -> {percent,status,output_url?,error?}
  GET  /files/{kind}/{name}              -> serve uploaded/generated files
  comfy control: /api/comfy/{install,log,diag,nodes,node,prompt,result}
  admin: /api/admin/{redeploy,log}

Long jobs run on a thread + report into PROGRESS (no blocking POST).
"""
from __future__ import annotations

import json
import os
import shutil
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

PROGRESS: dict[str, dict] = {}

app = FastAPI(title="Uniquifier GPU backend", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
# status / version / files
# ─────────────────────────────────────────

@app.get("/api/version")
def version():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, cwd=os.path.dirname(__file__) or ".").stdout.strip()
    except Exception:  # noqa: BLE001
        h = "unknown"
    return {"head": h}


@app.get("/api/status")
def status():
    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        pass
    return {"status": "ok", "gpu": gpu, "comfy_up": _port_open(8188),
            "comfy_installing": _comfy_running(), "auth": bool(API_TOKEN)}


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
def extract_frame(video: UploadFile = File(...), _=Depends(require_token)):
    video_id = uuid.uuid4().hex[:16]
    video_path = _up(video_id, "src.mp4")
    frame_path = _up(video_id, "frame0.jpg")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video_path, "-frames:v", "1", "-q:v", "2", frame_path], check=False)
    return {"video_id": video_id, "frame_url": f"/files/uploads/{video_id}_frame0.jpg"}


@app.post("/api/upload-face")
def upload_face(video_id: str = Form(...), face: UploadFile = File(...), _=Depends(require_token)):
    face_path = _up(video_id, "face.jpg")
    with open(face_path, "wb") as f:
        shutil.copyfileobj(face.file, f)
    return {"video_id": video_id, "stored": True}


# ─────────────────────────────────────────
# Flux first-frame edit + Wan video gen  (filled in by workflows.py)
# ─────────────────────────────────────────

@app.post("/api/flux-edit")
def flux_edit(video_id: str = Form(...), kind: str = Form("face"), prompt: str = Form(""),
              region: str = Form(""), face: UploadFile | None = File(default=None),
              _=Depends(require_token)):
    import workflows
    frame = _up(video_id, "frame0.jpg")
    if not os.path.isfile(frame):
        raise HTTPException(400, "no first frame; call /api/extract-frame first")
    face_path = None
    if face is not None:
        face_path = _up(video_id, "face.jpg")
        with open(face_path, "wb") as f:
            shutil.copyfileobj(face.file, f)
    elif kind == "face":
        face_path = _up(video_id, "face.jpg")
        if not os.path.isfile(face_path):
            raise HTTPException(400, "face kind needs a reference face (upload-face or face field)")
    out = _up(video_id, "edited.png")
    try:
        workflows.run_flux_edit(kind=kind, frame_path=frame, face_path=face_path,
                                prompt=prompt, region=region, out_path=out)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"flux edit failed: {exc}")
    return FileResponse(out, media_type="image/png")


@app.post("/api/wan-animate")
async def wan_animate(req: Request, _=Depends(require_token)):
    import workflows
    body = await req.json()
    video_id = body.get("video_id", "")
    bg_source = body.get("bg_source", "movement")  # movement | static
    params = body.get("params", {}) or {}
    src = _up(video_id, "src.mp4")
    edited = _up(video_id, "edited.png")
    if not os.path.isfile(src):
        raise HTTPException(400, "no source video for this id")
    if not os.path.isfile(edited):
        raise HTTPException(400, "no edited first frame; run /api/flux-edit first")
    out_final = os.path.join(OUTPUT_DIR, f"{video_id}_wan.mp4")

    def work():
        _set_progress(video_id, 0.0, "running")
        try:
            out = workflows.run_wan_animate(src_video=src, ref_image=edited, bg_source=bg_source,
                                            params=params,
                                            on_progress=lambda p, m: _set_progress(video_id, p, "running", msg=m))
            shutil.copy2(out, out_final)
            PROGRESS[video_id] = {"percent": 1.0, "status": "done",
                                  "output_url": f"/files/outputs/{video_id}_wan.mp4", "error": None}
        except Exception as exc:  # noqa: BLE001
            PROGRESS[video_id] = {"percent": 0.0, "status": "failed", "output_url": None, "error": str(exc)}

    threading.Thread(target=work, daemon=True).start()
    return {"video_id": video_id, "status": "started"}


@app.get("/api/progress/{video_id}")
def progress(video_id: str, _=Depends(require_token)):
    return PROGRESS.get(video_id, {"percent": 0.0, "status": "unknown", "output_url": None, "error": None})


# ─────────────────────────────────────────
# ComfyUI control plane
# ─────────────────────────────────────────

COMFY_DIR = os.environ.get("COMFY_DIR", "/root/ComfyUI")
COMFY_LOG = "/root/comfy.log"
COMFY_SETUP = os.path.join(os.path.dirname(__file__) or ".", "comfy_setup.sh")
COMFY_API = "http://127.0.0.1:8188"


def _comfy_running() -> bool:
    return subprocess.run(["pgrep", "-f", "comfy_setup.sh"], capture_output=True).returncode == 0


def _port_open(port: int) -> bool:
    import socket
    s = socket.socket(); s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port)); return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        s.close()


@app.post("/api/comfy/install")
def comfy_install(_=Depends(require_token)):
    if _comfy_running():
        return {"started": False, "reason": "comfy_setup already running"}
    logf = open(COMFY_LOG, "a")
    subprocess.Popen(["bash", COMFY_SETUP], stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    return {"started": True}


@app.get("/api/comfy/log")
def comfy_log(lines: int = 160, _=Depends(require_token)):
    try:
        with open(COMFY_LOG, errors="replace") as f:
            return {"log": "".join(f.readlines()[-lines:]), "installing": _comfy_running(), "up": _port_open(8188)}
    except FileNotFoundError:
        return {"log": "", "installing": _comfy_running(), "up": _port_open(8188)}


@app.get("/api/comfy/diag")
def comfy_diag(_=Depends(require_token)):
    def ls(p):
        try:
            return sorted(os.listdir(p))
        except Exception:  # noqa: BLE001
            return []
    m = os.path.join(COMFY_DIR, "models")
    return {"comfy_up": _port_open(8188), "installing": _comfy_running(),
            "nodes": ls(os.path.join(COMFY_DIR, "custom_nodes")),
            "models": {k: ls(os.path.join(m, k)) for k in
                       ("diffusion_models", "vae", "text_encoders", "loras", "controlnet",
                        "clip_vision", "detection")}}


def _comfy_req(path: str, data: bytes | None = None, timeout: float = 30):
    req = urllib.request.Request(COMFY_API + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {},
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


@app.get("/api/comfy/nodes")
def comfy_nodes(_=Depends(require_token)):
    try:
        info = json.loads(_comfy_req("/object_info"))
        return {"count": len(info), "nodes": sorted(info.keys())}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"ComfyUI not reachable: {exc}")


@app.get("/api/comfy/node/{name}")
def comfy_node(name: str, _=Depends(require_token)):
    try:
        return json.loads(_comfy_req("/object_info/" + name))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"ComfyUI not reachable: {exc}")


@app.post("/api/comfy/prompt")
async def comfy_prompt(req: Request, _=Depends(require_token)):
    body = await req.body()
    try:
        return json.loads(_comfy_req("/prompt", data=body, timeout=60))
    except urllib.error.HTTPError as exc:
        raise HTTPException(400, exc.read().decode(errors="replace"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"ComfyUI not reachable: {exc}")


@app.get("/api/comfy/result/{prompt_id}")
def comfy_result(prompt_id: str, _=Depends(require_token)):
    try:
        hist = json.loads(_comfy_req("/history/" + prompt_id))
        return hist.get(prompt_id, {"status": "pending"})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"ComfyUI not reachable: {exc}")


# ─────────────────────────────────────────
# admin
# ─────────────────────────────────────────

@app.post("/api/admin/redeploy")
def admin_redeploy(_=Depends(require_token)):
    script = (
        "sleep 1; pkill -f '[u]vicorn' 2>/dev/null; sleep 1; "
        "wget -qO /root/setup.sh https://raw.githubusercontent.com/lehych-ai/uniquifier-backend/main/setup.sh; "
        "nohup bash /root/setup.sh > /root/setup.log 2>&1 &"
    )
    subprocess.Popen(["bash", "-c", script], start_new_session=True)
    return {"restarting": True}


@app.get("/api/admin/log")
def admin_log(lines: int = 160, _=Depends(require_token)):
    for path in ("/root/setup.log", "/root/onstart.log"):
        try:
            with open(path, errors="replace") as f:
                return {"path": path, "log": "".join(f.readlines()[-lines:])}
        except FileNotFoundError:
            continue
    return {"path": None, "log": ""}
