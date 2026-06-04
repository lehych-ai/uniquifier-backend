"""Drive the user's ComfyUI workflows (Flux.2 first-frame edits + Wan2.2 Animate).

The workflows are UI-export JSONs (workflows/*.json) with virtual nodes
(SetNode/GetNode, Anything Everywhere). Converting those to ComfyUI's API format
by hand is error-prone, so we use ComfyUI's OWN `app.graphToPrompt()` via a
headless Playwright session against the live frontend (127.0.0.1:8188) — correct
by construction. API keys equal the UI node ids, so we inject inputs by node id
after conversion (cached per workflow).
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
WF_DIR = os.path.join(HERE, "workflows")
API_CACHE = os.path.join(WF_DIR, "_api")
COMFY_DIR = os.environ.get("COMFY_DIR", "/root/ComfyUI")
COMFY_INPUT = os.path.join(COMFY_DIR, "input")
COMFY_OUTPUT = os.path.join(COMFY_DIR, "output")
COMFY_API = "http://127.0.0.1:8188"

WF_FILES = {
    "face": "Flux-FaceSwap.json",
    "clothes": "Flux-ClothesSwap.json",
    "movement": "WanAnimate-movement.json",
    "static": "WanAnimate-static.json",
}

# injection node ids (stable in the user's files)
FACE_BASE, FACE_REF, FACE_OUT = "63", "64", "62"
CLOTHES_FRAME, CLOTHES_PROMPT, CLOTHES_OUT = "202", "107", "9"
WAN_VIDEO, WAN_REF, WAN_OUT = "75", "76", "547"

# the 6 clothes prompts (modes 4/5/6 get the UI word appended)
CLOTHES_PROMPTS = {
    "random_color": "Change only the color of the clothing item in the image to a random new color. Choose the color yourself randomly each time. The new color should look natural, realistic, and suitable for the clothing and the photo.\nPreserve everything else exactly the same: the person’s identity, face, body shape, pose, hairstyle, facial expression, skin tone, background, lighting, shadows, camera angle, clothing shape, fit, fabric texture, folds, seams, patterns, logos, accessories, and overall photo realism.",
    "random_clothes": "Replace the clothing in the image with a different random outfit. Choose the new clothing yourself randomly each time. The new outfit should look natural, realistic, and visually appropriate for the person, pose, and scene.\nPreserve everything else exactly the same: the person’s identity, face, body shape, pose, hairstyle, facial expression, skin tone, background, lighting, shadows, camera angle, proportions, and overall photo realism.\nKeep the result realistic and coherent. Do not change the person or the environment. Only modify the clothing.",
    "random_background": "Replace the background in the image with a new random background. Choose the background yourself randomly each time. The new background should look realistic, natural, and visually appropriate for the person and the scene.\nPreserve everything else exactly the same: the person’s identity, face, body shape, pose, hairstyle, facial expression, skin tone, clothing, accessories, camera angle, proportions, and overall photo realism.\nKeep the subject unchanged and only modify the background. Blend the subject naturally into the new environment with realistic perspective, lighting, and depth.",
    "upper": "Replace only the upper-body clothing in the image with a new garment. Change the top clothing item while preserving everything else exactly the same. Keep the same person, face, identity, body shape, pose, hairstyle, facial expression, skin tone, lower-body clothing, accessories, background, lighting, shadows, camera angle, proportions, and overall photo realism. The new upper-body clothing should look realistic, natural, and visually appropriate for the subject and scene. Do not modify anything except the upper-body clothing. New upper-body clothing:",
    "lower": "Replace only the lower-body clothing in the image with a new garment. Change the bottom clothing item while preserving everything else exactly the same. Keep the same person, face, identity, body shape, pose, hairstyle, facial expression, skin tone, upper-body clothing, accessories, background, lighting, shadows, camera angle, proportions, and overall photo realism. The new lower-body clothing should look realistic, natural, and visually appropriate for the subject and scene. Do not modify anything except the lower-body clothing. New lower-body clothing:",
    "full": "Replace the clothing in the image with a new outfit while preserving everything else exactly the same. Keep the same person, face, identity, body shape, pose, hairstyle, facial expression, skin tone, accessories, background, lighting, shadows, camera angle, proportions, and overall photo realism. The new outfit should look realistic, natural, and visually appropriate for the subject and scene. Do not modify the person or the environment. Only change the outfit. New outfit:",
}


# ── ComfyUI HTTP ──────────────────────────────────────────────────────────────

def _get(path: str, timeout: float = 30):
    with urllib.request.urlopen(COMFY_API + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(path: str, body: dict, timeout: float = 60):
    req = urllib.request.Request(COMFY_API + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ── UI → API conversion via headless Playwright (ComfyUI's own graphToPrompt) ──

def _convert_ui_to_api(ui_path: str) -> dict:
    from playwright.sync_api import sync_playwright
    with open(ui_path, encoding="utf-8") as f:
        ui = json.load(f)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        page = browser.new_page()
        page.goto(COMFY_API, wait_until="domcontentloaded", timeout=180000)
        page.wait_for_function("() => window.app && window.app.graphToPrompt && window.app.loadGraphData",
                               timeout=180000)
        api = page.evaluate(
            """async (graph) => {
                await window.app.loadGraphData(graph, true, false);
                const p = await window.app.graphToPrompt();
                return p.output;
            }""", ui)
        browser.close()
    if not api:
        raise RuntimeError("graphToPrompt returned empty (missing nodes?)")
    return api


def _api_graph(kind: str) -> dict:
    os.makedirs(API_CACHE, exist_ok=True)
    cache = os.path.join(API_CACHE, kind + ".json")
    if os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)
    api = _convert_ui_to_api(os.path.join(WF_DIR, WF_FILES[kind]))
    with open(cache, "w") as f:
        json.dump(api, f)
    return api


# ── input staging + graph injection ───────────────────────────────────────────

def _stage(path: str, name: str) -> str:
    os.makedirs(COMFY_INPUT, exist_ok=True)
    dst = os.path.join(COMFY_INPUT, name)
    shutil.copy2(path, dst)
    return name


def _set_in(graph: dict, node: str, key: str, val) -> None:
    if node in graph:
        graph[node].setdefault("inputs", {})[key] = val


def _submit_and_wait(graph: dict, out_node: str, on_progress=None, timeout: float = 3600) -> str:
    # clear any stale queue, then submit with retries (HTTP stalls while busy)
    for ep, bd in (("/interrupt", {}), ("/queue", {"clear": True})):
        try:
            _post(ep, bd, timeout=10)
        except Exception:  # noqa: BLE001
            pass
    resp = None
    for attempt in range(5):
        try:
            resp = _post("/prompt", {"prompt": graph}, timeout=120)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 4:
                raise RuntimeError(f"ComfyUI rejected prompt: {exc}")
            time.sleep(8)
    if resp.get("node_errors"):
        raise RuntimeError("node_errors: " + json.dumps(resp["node_errors"])[:800])
    pid = resp["prompt_id"]
    if on_progress:
        on_progress(0.1, "queued")
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(4)
        try:
            hist = _get("/history/" + pid, timeout=90).get(pid)
        except Exception:  # noqa: BLE001
            if on_progress:
                on_progress(0.3, "rendering")
            continue
        if not hist:
            if on_progress:
                on_progress(0.3, "rendering")
            continue
        st = hist.get("status", {})
        if st.get("status_str") == "error":
            raise RuntimeError("ComfyUI run error: " + json.dumps(st.get("messages"))[:800])
        outs = (hist.get("outputs") or {}).get(out_node, {})
        files = outs.get("images") or outs.get("gifs") or outs.get("videos") or []
        if files:
            f0 = files[0]
            path = os.path.join(COMFY_OUTPUT, f0.get("subfolder", ""), f0["filename"])
            if on_progress:
                on_progress(1.0, "done")
            return path
        if on_progress:
            on_progress(0.6, "rendering")
    raise RuntimeError("ComfyUI job timed out")


# ── public API ─────────────────────────────────────────────────────────────────

def run_flux_edit(kind: str, frame_path: str, face_path: str | None,
                  prompt: str = "", region: str = "", out_path: str = "") -> str:
    """kind='face' (head swap, fixed prompt) or 'clothes' (region selects 1 of 6 prompts)."""
    graph = copy.deepcopy(_api_graph("face" if kind == "face" else "clothes"))
    uid = uuid.uuid4().hex[:8]
    fn = _stage(frame_path, f"{uid}_frame{os.path.splitext(frame_path)[1] or '.png'}")
    if kind == "face":
        _set_in(graph, FACE_BASE, "image", fn)
        rf = _stage(face_path, f"{uid}_face{os.path.splitext(face_path)[1] or '.png'}")
        _set_in(graph, FACE_REF, "image", rf)
        out_node = FACE_OUT
    else:
        _set_in(graph, CLOTHES_FRAME, "image", fn)
        text = CLOTHES_PROMPTS.get(region or "random_clothes", CLOTHES_PROMPTS["random_clothes"])
        if region in ("upper", "lower", "full") and prompt.strip():
            text = text + " " + prompt.strip()
        _set_in(graph, CLOTHES_PROMPT, "text", text)
        out_node = CLOTHES_OUT
    produced = _submit_and_wait(graph, out_node, timeout=600)
    if out_path:
        shutil.copy2(produced, out_path)
        return out_path
    return produced


def run_wan_animate(src_video: str, ref_image: str, bg_source: str = "movement",
                    params: dict | None = None, on_progress=None) -> str:
    """bg_source='movement' (bg from driving video, SAM2) or 'static' (bg from photo)."""
    kind = "static" if bg_source == "static" else "movement"
    graph = copy.deepcopy(_api_graph(kind))
    uid = uuid.uuid4().hex[:8]
    v = _stage(src_video, f"{uid}_src.mp4")
    r = _stage(ref_image, f"{uid}_ref{os.path.splitext(ref_image)[1] or '.png'}")
    _set_in(graph, WAN_VIDEO, "video", v)
    _set_in(graph, WAN_REF, "image", r)
    return _submit_and_wait(graph, WAN_OUT, on_progress=on_progress, timeout=3600)
