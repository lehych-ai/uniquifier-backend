"""Full head + hair swap via ComfyUI (AnimateDiff vid2vid + IPAdapter-FaceID).

Why ComfyUI and not raw diffusers: temporal coherence ("no flicker") comes from
the AnimateDiff motion module + Evolved sampling, which ComfyUI/AnimateDiff-Evolved
implement robustly. We drive ComfyUI's local API (127.0.0.1:8188) from here.

Pipeline (Strategy A — full vid2vid, identity-led):
  source frames --VAEEncode--> latent  (img2img, denoise keeps composition)
  source frames --DWPose-----> ControlNet(openpose)  (locks pose/motion)
  reference face --IPAdapter-FaceID--> patches MODEL with the reference identity
  MODEL --AnimateDiff--> temporal model  --> KSampler --> VAEDecode --> mp4 (+audio)

The reference identity (incl. hair/face shape, via FaceID + moderate denoise)
takes over the head; ControlNet + low-ish denoise keep the body/scene close.
Everything is tunable from the request so we can dial quality on the pod.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request

COMFY_API = "http://127.0.0.1:8188"
COMFY_DIR = os.environ.get("COMFY_DIR", "/root/ComfyUI")
COMFY_INPUT = os.path.join(COMFY_DIR, "input")
COMFY_OUTPUT = os.path.join(COMFY_DIR, "output")


def _post(path: str, body: dict, timeout: float = 60) -> dict:
    req = urllib.request.Request(
        COMFY_API + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(path: str, timeout: float = 60) -> dict:
    with urllib.request.urlopen(COMFY_API + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_prompt(video_name: str, face_name: str, p: dict) -> dict:
    """Return a ComfyUI API-format prompt graph. `p` carries tunables."""
    frames = int(p.get("frames", 16))
    w, h = int(p.get("width", 512)), int(p.get("height", 896))
    fps = int(p.get("fps", 16))
    denoise = float(p.get("denoise", 0.65))
    steps = int(p.get("steps", 22))
    cfg = float(p.get("cfg", 7.0))
    seed = int(p.get("seed", 12345))
    ip_weight = float(p.get("ip_weight", 1.0))
    ip_v2 = float(p.get("ip_weight_faceidv2", 1.0))
    cn_pose = float(p.get("cn_pose", 0.85))
    pos = p.get("positive", "a person, photorealistic, natural skin, sharp focus, high detail")
    neg = p.get("negative", "blurry, low quality, deformed, distorted face, extra limbs, watermark, text, cartoon")

    g: dict = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "realisticVision_v51.safetensors"}},
        "2": {"class_type": "VAELoader",
              "inputs": {"vae_name": "vae-ft-mse-840000.safetensors"}},
        "3": {"class_type": "VHS_LoadVideo",
              "inputs": {"video": video_name, "force_rate": fps, "force_size": "Custom",
                         "custom_width": w, "custom_height": h, "frame_load_cap": frames,
                         "skip_first_frames": 0, "select_every_nth": 1}},
        "4": {"class_type": "LoadImage", "inputs": {"image": face_name}},
        "5": {"class_type": "DWPreprocessor",
              "inputs": {"image": ["3", 0], "detect_hand": "enable", "detect_body": "enable",
                         "detect_face": "enable", "resolution": w}},
        "6": {"class_type": "IPAdapterUnifiedLoaderFaceID",
              "inputs": {"model": ["1", 0], "preset": "FACEID PLUS V2",
                         "lora_strength": 0.6, "provider": "CUDA"}},
        "7": {"class_type": "IPAdapterFaceID",
              "inputs": {"model": ["6", 0], "ipadapter": ["6", 1], "image": ["4", 0],
                         "weight": ip_weight, "weight_faceidv2": ip_v2, "weight_type": "linear",
                         "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0,
                         "embeds_scaling": "V only"}},
        "8": {"class_type": "ADE_AnimateDiffLoaderWithContext",
              "inputs": {"model": ["7", 0], "model_name": "v3_sd15_mm.ckpt",
                         "beta_schedule": "sqrt_linear (AnimateDiff)"}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}},
        "11": {"class_type": "ControlNetLoaderAdvanced",
               "inputs": {"control_net_name": "control_v11p_sd15_openpose.pth"}},
        "12": {"class_type": "ACN_AdvancedControlNetApply",
               "inputs": {"positive": ["9", 0], "negative": ["10", 0], "control_net": ["11", 0],
                          "image": ["5", 0], "strength": cn_pose,
                          "start_percent": 0.0, "end_percent": 1.0}},
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["2", 0]}},
        "14": {"class_type": "KSampler",
               "inputs": {"model": ["8", 0], "positive": ["12", 0], "negative": ["12", 1],
                          "latent_image": ["13", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "euler", "scheduler": "normal", "denoise": denoise}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["2", 0]}},
        "16": {"class_type": "VHS_VideoCombine",
               "inputs": {"images": ["15", 0], "frame_rate": fps, "loop_count": 0,
                          "filename_prefix": "headswap", "format": "video/h264-mp4",
                          "pingpong": False, "save_output": True, "audio": ["3", 2]}},
    }

    # AnimateDiff needs sliding-window context only past its native 16-frame window.
    if frames > 16:
        g["20"] = {"class_type": "ADE_StandardUniformContextOptions",
                   "inputs": {"context_length": 16, "context_stride": 1,
                              "context_overlap": 4, "fuse_method": "pyramid"}}
        g["8"]["inputs"]["context_options"] = ["20", 0]
    return g


def run_head_swap(src_video: str, face_image: str, params: dict,
                  on_progress=None) -> str:
    """Stage inputs into ComfyUI, submit the graph, poll, return the output mp4 path."""
    os.makedirs(COMFY_INPUT, exist_ok=True)
    vid_name = "hs_" + os.path.basename(src_video)
    face_name = "hs_" + os.path.basename(face_image)
    shutil.copy2(src_video, os.path.join(COMFY_INPUT, vid_name))
    shutil.copy2(face_image, os.path.join(COMFY_INPUT, face_name))

    # ComfyUI runs one prompt at a time and its HTTP server stalls while busy.
    # Clear any stale queue (e.g. leftovers from a previous attempt) and stop a
    # running job so ours starts promptly, then submit with retries — the POST
    # itself can be refused for a while if the server is mid-execution.
    for ep, bd in (("/interrupt", {}), ("/queue", {"clear": True})):
        try:
            _post(ep, bd, timeout=10)
        except Exception:  # noqa: BLE001
            pass

    graph = build_prompt(vid_name, face_name, params)
    resp = None
    for attempt in range(5):
        try:
            resp = _post("/prompt", {"prompt": graph}, timeout=120)
            break
        except Exception as exc:  # noqa: BLE001 — server busy; back off and retry
            if attempt == 4:
                raise RuntimeError(f"ComfyUI did not accept the prompt (busy): {exc}")
            if on_progress:
                on_progress(0.05, "waiting for ComfyUI queue")
            time.sleep(12)
    if resp.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected graph: {json.dumps(resp['node_errors'])[:800]}")
    prompt_id = resp["prompt_id"]
    if on_progress:
        on_progress(0.1, "queued on ComfyUI")

    # poll history until this prompt produces an output (or errors). ComfyUI's
    # HTTP server can lag for tens of seconds while it cold-loads models or pulls
    # a preprocessor weight on first run — a single timed-out poll must NOT fail
    # the whole job, so we swallow transient errors and keep polling.
    deadline = time.time() + 2400
    while time.time() < deadline:
        time.sleep(4)
        try:
            hist = _get("/history/" + prompt_id, timeout=90).get(prompt_id)
        except Exception:  # noqa: BLE001 — transient (server busy loading); retry
            if on_progress:
                on_progress(0.2, "loading models / rendering")
            continue
        if not hist:
            if on_progress:
                on_progress(0.25, "rendering")
            continue
        status = hist.get("status", {})
        if status.get("status_str") == "error" or status.get("completed") is False and status.get("messages"):
            # surface the first execution error
            raise RuntimeError(f"ComfyUI run failed: {json.dumps(status.get('messages'))[:800]}")
        outputs = hist.get("outputs", {})
        node16 = outputs.get("16", {})
        files = node16.get("gifs") or node16.get("videos") or []
        if files:
            fn = files[0]["filename"]
            sub = files[0].get("subfolder", "")
            out_path = os.path.join(COMFY_OUTPUT, sub, fn)
            if on_progress:
                on_progress(1.0, "done")
            return out_path
        if on_progress:
            on_progress(0.6, "rendering")
    raise RuntimeError("ComfyUI head-swap timed out")
