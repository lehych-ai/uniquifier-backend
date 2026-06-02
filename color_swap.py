"""Color swap — rewritten to process every frame instead of stamping frame 0.

The original pipeline edited frame 0 once, then pasted those frozen pixels onto
every frame through a static mask. On any camera/subject motion that reads as a
sticker floating in place. This rewrite recomputes a mask **per frame** for all
three modes, so the edit tracks the moving subject:

  random_color  : per-frame clothing mask (rembg u2net_cloth_seg) + a consistent
                  hue rotation (chosen once from the seed → temporally stable).
                  Hue is *rotated*, not flattened, so garment texture survives.
  random_bg     : background generated once (SD2 inpainting), then per-frame
                  person matte (feathered + temporally smoothed) composites the
                  live subject over it.
  random_clothes: a new garment look is generated once on a reference frame
                  (SD2 inpainting of the clothing region from the prompt), then
                  per frame we LAB-transfer that look onto the freshly segmented
                  clothing region. Motion-correct, no frozen patch.

Full generative garment *replacement* per frame (changing the garment shape, not
just its look) needs a motion-aware video model — that is the Wan2.1 VACE I2V
upgrade noted in the project README and is intentionally out of scope here.

Everything degrades gracefully: if a heavy model/weight is missing we fall back
to a person matte or a plain hue shift rather than crashing.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from video_io import FfmpegWriter, open_video

log = logging.getLogger("gpu.color_swap")

MODELS_DIR = os.environ.get("MODELS_DIR", "/root/models")
# stabilityai/* inpainting repos are gated (HTTP 401); default to a public mirror.
SD_INPAINT_MODEL = os.environ.get("SD_INPAINT_MODEL", "botp/stable-diffusion-v1-5-inpainting")

ProgressCb = Optional[Callable[[float, str], None]]

# ─────────────────────────────────────────
# lazy singletons
# ─────────────────────────────────────────

_person_session = None
_cloth_session = None
_inpaint_pipe = None
_translator_ok = True
_clothes_seg = None  # (model, processor) | False

# mattmdjaga/segformer_b2_clothes class ids → which count as "the garment" per region.
#  0 bg 1 hat 2 hair 3 sunglasses 4 upper-clothes 5 skirt 6 pants 7 dress 8 belt
#  9/10 shoes 11 face 12/13 legs 14/15 arms 16 bag 17 scarf
CLOTHES_CLASSES = {
    "upper": {4, 7, 17},          # upper-clothes, dress, scarf
    "lower": {5, 6, 7},           # skirt, pants, dress
    "full": {1, 4, 5, 6, 7, 17},  # hat + upper + skirt + pants + dress + scarf
}


def _get_person_session():
    global _person_session
    if _person_session is None:
        from rembg import new_session
        _person_session = new_session("u2net")
    return _person_session


def _get_cloth_session():
    """u2net_cloth_seg segments garments (upper/lower/full). Falls back to None."""
    global _cloth_session
    if _cloth_session is None:
        try:
            from rembg import new_session
            _cloth_session = new_session("u2net_cloth_seg")
        except Exception as exc:  # noqa: BLE001
            log.warning("cloth-seg model unavailable (%s); using person-band fallback", exc)
            _cloth_session = False
    return _cloth_session or None


def _get_inpaint_pipe():
    global _inpaint_pipe
    if _inpaint_pipe is None:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        _inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL, torch_dtype=torch.float16
        ).to("cuda")
        _inpaint_pipe.safety_checker = None
        # Memory-efficient attention without the xformers dependency. Prefer
        # xformers if it happens to be present; otherwise fall back to PyTorch's
        # built-in sliced attention. Both are no-ops on failure.
        try:
            _inpaint_pipe.enable_xformers_memory_efficient_attention()
        except Exception:  # noqa: BLE001
            try:
                _inpaint_pipe.enable_attention_slicing()
            except Exception:  # noqa: BLE001
                pass
        log.info("SD inpainting loaded (%s)", SD_INPAINT_MODEL)
    return _inpaint_pipe


def models_status() -> dict:
    return {"sd_inpaint_model": SD_INPAINT_MODEL}


# ─────────────────────────────────────────
# prompt translation (write prompts in any language)
# ─────────────────────────────────────────

def translate_prompt(text: str) -> str:
    global _translator_ok
    if not text or not text.strip() or not _translator_ok:
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as exc:  # noqa: BLE001
        log.debug("translate failed: %s", exc)
        _translator_ok = False
        return text


# ─────────────────────────────────────────
# masks (per frame)
# ─────────────────────────────────────────

def _rembg_alpha(session, frame_bgr: np.ndarray) -> np.ndarray:
    from rembg import remove
    h, w = frame_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    # only_mask=True returns a single-channel mask directly. The cloth-seg model
    # also emits masks at its own resolution, so always resize back to the frame
    # — otherwise cv2.bitwise_and(alpha, band) fails on a size mismatch.
    out = remove(pil, session=session, only_mask=True)
    arr = np.array(out)
    if arr.ndim == 3:
        arr = arr[:, :, 3] if arr.shape[2] == 4 else cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    if arr.shape[:2] != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.uint8)


def person_mask(frame_bgr: np.ndarray) -> np.ndarray:
    return _rembg_alpha(_get_person_session(), frame_bgr)


def _region_band(h: int, w: int, region: str) -> np.ndarray:
    # Bands target the TORSO (below the head) so a "recolor" never lands on the
    # face. "upper" = chest/shirt, "lower" = pants area, "full" = whole torso+legs.
    band = np.zeros((h, w), np.uint8)
    if region == "upper":
        band[int(h * 0.40):int(h * 0.80), :] = 255
    elif region == "lower":
        band[int(h * 0.62):int(h * 0.97), :] = 255
    else:
        band[int(h * 0.38):int(h * 0.97), :] = 255
    return band


def _skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Skin pixels (YCrCb) — subtracted from the garment mask so face/neck/arms
    are never recolored even if the band/segmentation overlaps them."""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8))
    return cv2.dilate(skin, np.ones((5, 5), np.uint8), iterations=2)


def _get_clothes_segformer():
    """Human-parsing SegFormer — clean per-class garment masks (upper/dress/pants/
    skirt), with hair/skin/face/background as their own classes. Falls to False."""
    global _clothes_seg
    if _clothes_seg is None:
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            mid = os.environ.get("CLOTHES_SEG_MODEL", "mattmdjaga/segformer_b2_clothes")
            proc = SegformerImageProcessor.from_pretrained(mid)
            model = SegformerForSemanticSegmentation.from_pretrained(mid).to("cuda").eval()
            _clothes_seg = (model, proc)
            log.info("clothes SegFormer loaded (%s)", mid)
        except Exception as exc:  # noqa: BLE001
            log.warning("clothes SegFormer unavailable (%s); using rembg fallback", exc)
            _clothes_seg = False
    return _clothes_seg or None


def _segformer_clothes_mask(frame_bgr: np.ndarray, region: str) -> np.ndarray | None:
    seg = _get_clothes_segformer()
    if seg is None:
        return None
    import torch
    model, proc = seg
    h, w = frame_bgr.shape[:2]
    rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = proc(images=rgb, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, C, h', w')
    up = torch.nn.functional.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
    pred = up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    classes = CLOTHES_CLASSES.get(region, CLOTHES_CLASSES["upper"])
    return (np.isin(pred, list(classes)).astype(np.uint8) * 255)


def clothes_mask(frame_bgr: np.ndarray, region: str = "upper") -> np.ndarray:
    """Per-frame garment mask. Prefer the human-parsing SegFormer (clean garment
    classes); else rembg cloth-seg / person∩band. Always protect skin."""
    h, w = frame_bgr.shape[:2]
    # 1) best path: SegFormer human parsing — already excludes hair/skin/bg.
    seg = _segformer_clothes_mask(frame_bgr, region)
    if seg is not None and int((seg > 0).sum()) > 0.003 * h * w:
        return cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return _clothes_mask_fallback(frame_bgr, region)


def _clothes_mask_fallback(frame_bgr: np.ndarray, region: str = "upper") -> np.ndarray:
    """rembg cloth-seg (or person∩band) fallback when SegFormer isn't available."""
    h, w = frame_bgr.shape[:2]
    band = _region_band(h, w, region)
    person = person_mask(frame_bgr)
    base = None
    sess = _get_cloth_session()
    if sess is not None:
        alpha = _rembg_alpha(sess, frame_bgr)
        # use the FULL garment mask (cloth-seg is garment-only) when coverage is
        # plausible — clipping it to a band turned a whole sweater into a stripe.
        area = int((alpha > 30).sum())
        if 0.01 * h * w < area < 0.75 * h * w:
            base = alpha
    if base is None:
        # fallback: person silhouette ∩ torso band
        base = cv2.bitwise_and(person, band)
    # the garment must lie ON the person — this kills background leak (e.g. a sofa
    # behind her at torso height) far better than a full-width band ever could.
    base = cv2.bitwise_and(base, person)
    # never touch skin (face/neck/arms)
    base = cv2.bitwise_and(base, cv2.bitwise_not(_skin_mask(frame_bgr)))
    # close small holes so the garment reads as one solid region
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return base


def _feather(mask: np.ndarray, k: int = 9) -> np.ndarray:
    m = cv2.GaussianBlur(mask, (0, 0), sigmaX=k)
    return (m.astype(np.float32) / 255.0)


# ─────────────────────────────────────────
# per-frame operations
# ─────────────────────────────────────────

def recolor_frame(frame_bgr: np.ndarray, mask: np.ndarray, hue_target: int, sat_mul: float) -> np.ndarray:
    """Colorize the masked garment to a vivid target hue.

    A plain hue *rotation* is invisible on black/grey/low-saturation clothes
    (rotating the hue of a near-greyscale pixel changes nothing). Instead we set
    the garment to a fixed target hue, floor the saturation so the colour
    actually shows, and lift only the deep shadows so a near-black garment reads
    as "that colour, dark" while folds/highlights (texture) survive untouched.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    new_h = np.full_like(h_ch, float(hue_target % 180))
    # saturation floor (~70) so grey/black garments take colour; then scale up
    new_s = np.clip(np.maximum(s_ch, 70.0) * sat_mul, 45.0, 255.0)
    # raise only the darkest pixels toward ~70 so the hue is visible; bright
    # folds stay put, preserving the garment's shading/texture.
    new_v = np.clip(v_ch + np.clip(72.0 - v_ch, 0.0, None) * 0.7, 0.0, 255.0)
    colored = cv2.cvtColor(cv2.merge([new_h, new_s, new_v]).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    a = _feather(mask)[..., None]
    return (colored * a + frame_bgr.astype(np.float32) * (1 - a)).astype(np.uint8)


def composite_person_over_bg(frame_bgr: np.ndarray, bg: np.ndarray, prev_alpha: np.ndarray | None):
    """Composite the live subject over a static generated background.

    Temporal EMA on the matte tames rembg's per-frame edge flicker.
    """
    h, w = frame_bgr.shape[:2]
    pm = person_mask(frame_bgr).astype(np.float32) / 255.0
    if prev_alpha is not None and prev_alpha.shape == pm.shape:
        pm = 0.6 * pm + 0.4 * prev_alpha
    cur_alpha = pm
    a = cv2.GaussianBlur(pm, (0, 0), sigmaX=2.0)[..., None]
    bg_r = cv2.resize(bg, (w, h))
    out = (frame_bgr.astype(np.float32) * a + bg_r.astype(np.float32) * (1 - a)).astype(np.uint8)
    return out, cur_alpha


def lab_transfer(frame_bgr: np.ndarray, mask: np.ndarray, ref_lab_stats) -> np.ndarray:
    """Map the masked region's LAB mean/std to the generated garment's stats."""
    a = _feather(mask)
    region = a > 0.05
    if not region.any():
        return frame_bgr
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    (rm, rs) = ref_lab_stats
    for c in range(3):
        ch = lab[:, :, c]
        cm, cs = ch[region].mean(), ch[region].std() + 1e-5
        ch_new = (ch - cm) / cs * rs[c] + rm[c]
        lab[:, :, c] = ch_new
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    a3 = a[..., None]
    return (out * a3 + frame_bgr * (1 - a3)).astype(np.uint8)


# ─────────────────────────────────────────
# generation (once, on a reference frame)
# ─────────────────────────────────────────

def _sd_inpaint(image_bgr: np.ndarray, mask_255: np.ndarray, prompt: str,
               seed: int | None = None, steps: int = 25) -> np.ndarray:
    pipe = _get_inpaint_pipe()
    h, w = image_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).resize((512, 512))
    msk = Image.fromarray(mask_255).resize((512, 512)).convert("RGB")
    gen = None
    if seed is not None:
        import torch
        gen = torch.Generator(device="cuda").manual_seed(int(seed) & 0x7FFFFFFF)
    out = pipe(
        prompt=prompt + ", high quality, photorealistic, detailed",
        negative_prompt="blurry, low quality, deformed, extra limbs, watermark, text",
        image=img, mask_image=msk, num_inference_steps=steps, guidance_scale=7.5,
        generator=gen,
    ).images[0]
    out = out.resize((w, h), Image.LANCZOS)
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


RANDOM_BG_PROMPTS = [
    "busy city street with people walking, urban photography",
    "cozy coffee shop interior, warm lighting",
    "tropical beach with palm trees, sunny day",
    "green park with trees, daytime",
    "modern shopping mall interior, bright lights",
    "rooftop terrace with city skyline at sunset",
    "luxury hotel lobby, elegant interior",
    "fashion studio with soft lighting",
    "concrete urban wall with graffiti, street photography",
    "mountain landscape with clear blue sky",
]


def _lab_stats(image_bgr: np.ndarray, mask: np.ndarray):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    region = mask > 128
    if not region.any():
        region = np.ones(mask.shape, bool)
    means = [lab[:, :, c][region].mean() for c in range(3)]
    stds = [lab[:, :, c][region].std() + 1e-5 for c in range(3)]
    return means, stds


# ─────────────────────────────────────────
# PREVIEW (frame 0) — also computes the reusable "plan" for the video pass
# ─────────────────────────────────────────

def color_swap_preview(frame_path, output_path, plan_save_path, mode, region, prompt="", seed=None):
    """Render the frame-0 preview and persist a small reusable plan (npz)."""
    image = cv2.imread(frame_path)
    if image is None:
        raise ValueError(f"cannot load frame: {frame_path}")
    rng = random.Random(seed if seed is not None else random.randint(0, 2**31))

    if mode == "random_color":
        # full hue wheel for variety; sat>1 keeps the colour punchy (recolor_frame
        # floors/​lifts so even a black garment reads as the chosen colour).
        hue = rng.randint(0, 179)
        sat = rng.uniform(1.3, 1.8)
        mask = clothes_mask(image, region)
        result = recolor_frame(image, mask, hue, sat)
        np.savez(plan_save_path, mode=mode, region=region, hue=hue, sat=sat)

    elif mode == "random_bg":
        bg_prompt = rng.choice(RANDOM_BG_PROMPTS)
        pm = person_mask(image)
        bg_mask = cv2.bitwise_not(pm)
        bg = _sd_inpaint(image, bg_mask, bg_prompt)
        result, _ = composite_person_over_bg(image, bg, None)
        np.savez(plan_save_path, mode=mode, region=region, bg=bg)

    elif mode == "random_clothes":
        prompt_en = translate_prompt(prompt) or "new stylish outfit"
        cloth_seed = rng.randint(0, 2**31 - 1)
        mask = clothes_mask(image, region)
        garment = _sd_inpaint(image, mask, prompt_en, seed=cloth_seed)
        # Composite the ACTUAL inpainted garment into the masked region (feathered)
        # so the preview shows a real garment change, not just a tint. The video
        # pass re-runs this same seeded inpaint per frame (real replacement), with
        # a LAB-stats fallback if the SD pipe is unavailable on the pod.
        a = _feather(mask, k=6)[..., None]
        result = (garment.astype(np.float32) * a + image.astype(np.float32) * (1 - a)).astype(np.uint8)
        means, stds = _lab_stats(garment, mask)
        np.savez(plan_save_path, mode=mode, region=region, prompt=prompt_en, seed=cloth_seed,
                 lab_mean=np.array(means), lab_std=np.array(stds))

    else:
        result = image
        np.savez(plan_save_path, mode="none", region=region)

    cv2.imwrite(output_path, result)
    return output_path


# ─────────────────────────────────────────
# APPLY TO VIDEO (per frame)
# ─────────────────────────────────────────

def color_swap_video(video_path, output_path, plan_path, on_progress: ProgressCb = None):
    plan = np.load(plan_path, allow_pickle=True)
    mode = str(plan["mode"])
    region = str(plan["region"]) if "region" in plan.files else "upper"

    cap, meta = open_video(video_path)
    bg = plan["bg"] if mode == "random_bg" and "bg" in plan.files else None
    ref_stats = None
    if mode == "random_clothes":
        ref_stats = (list(plan["lab_mean"]), list(plan["lab_std"]))
    cloth_prompt = str(plan["prompt"]) if "prompt" in plan.files else ""
    cloth_seed = int(plan["seed"]) if "seed" in plan.files else 0
    hue = int(plan["hue"]) if mode == "random_color" else 0
    sat = float(plan["sat"]) if mode == "random_color" else 1.0

    # For real garment replacement we re-inpaint every frame with the SAME seed
    # and prompt, so the new garment is generated (shape + texture), not just
    # recoloured. Per-frame SD flickers, so we temporally smooth the generated
    # garment (EMA) inside the mask. If the SD pipe can't load, fall back to the
    # cheap LAB colour transfer so the job never hard-fails.
    use_inpaint = False
    if mode == "random_clothes":
        try:
            _get_inpaint_pipe()
            use_inpaint = True
        except Exception as exc:  # noqa: BLE001
            log.warning("clothes inpaint pipe unavailable (%s); LAB-transfer fallback", exc)

    prev_alpha = None
    prev_gen: np.ndarray | None = None
    with FfmpegWriter(output_path, meta, audio_from=video_path) as writer:
        idx = 0
        for frame in _frames(cap):
            if mode == "random_color":
                m = clothes_mask(frame, region)
                out = recolor_frame(frame, m, hue, sat)
            elif mode == "random_bg":
                out, prev_alpha = composite_person_over_bg(frame, bg, prev_alpha)
            elif mode == "random_clothes":
                m = clothes_mask(frame, region)
                if use_inpaint:
                    gen = _sd_inpaint(frame, m, cloth_prompt, seed=cloth_seed, steps=18)
                    if prev_gen is not None and prev_gen.shape == gen.shape:
                        gen = (0.5 * gen.astype(np.float32) + 0.5 * prev_gen.astype(np.float32)).astype(np.uint8)
                    prev_gen = gen
                    a = _feather(m, k=6)[..., None]
                    out = (gen.astype(np.float32) * a + frame.astype(np.float32) * (1 - a)).astype(np.uint8)
                else:
                    out = lab_transfer(frame, m, ref_stats)
            else:
                out = frame
            writer.write(out)
            idx += 1
            if on_progress and meta.total_frames and idx % 5 == 0:
                on_progress(idx / meta.total_frames, f"color swap {idx}/{meta.total_frames}")

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
