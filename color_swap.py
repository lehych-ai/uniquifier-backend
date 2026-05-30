import cv2
import numpy as np
import os
import subprocess
import random
from rembg import remove
from PIL import Image
import torch
from deep_translator import GoogleTranslator


# ─────────────────────────────────────────
# ПЕРЕВОД ПРОМПТА
# ─────────────────────────────────────────

def translate_prompt(text: str) -> str:
    """Переводит любой текст в английский"""
    if not text or not text.strip():
        return text
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        print(f"Prompt translated: '{text}' → '{translated}'")
        return translated
    except Exception as e:
        print(f"Translation failed, using original: {e}")
        return text


# ─────────────────────────────────────────
# РАНДОМНЫЕ ФОНЫ
# ─────────────────────────────────────────

RANDOM_BG_PROMPTS = [
    "busy city street with people walking, urban photography",
    "modern shopping mall interior, bright lights",
    "cozy coffee shop interior, warm lighting",
    "tropical beach with palm trees, sunny day",
    "green park with trees and benches, daytime",
    "subway station interior, urban environment",
    "rooftop terrace with city skyline at sunset",
    "modern gym interior with equipment",
    "luxury hotel lobby, elegant interior",
    "outdoor market street with colorful stalls",
    "university campus outdoor area, sunny day",
    "modern office space with glass windows",
    "nightclub interior with colored lights",
    "outdoor basketball court, urban setting",
    "fashion studio with white background and soft lighting",
    "concrete urban wall with graffiti, street photography",
    "airport terminal interior, modern architecture",
    "restaurant interior with warm ambient lighting",
    "empty warehouse with industrial aesthetic",
    "mountain landscape with clear blue sky",
]

def get_random_bg_prompt() -> str:
    return random.choice(RANDOM_BG_PROMPTS)


# ─────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────

def load_image_cv(path):
    return cv2.imread(path)

def save_image_cv(img, path):
    cv2.imwrite(path, img)

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def save_mask(mask: np.ndarray, path: str):
    np.save(path, mask)

def load_mask(path: str) -> np.ndarray:
    return np.load(path)


# ─────────────────────────────────────────
# МОДЕЛИ — загружаются один раз
# ─────────────────────────────────────────

_sam_predictor = None
_inpaint_pipe = None
_ootd = None

def get_sam():
    global _sam_predictor
    if _sam_predictor is None:
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry["vit_h"](
            checkpoint="/root/models/sam_vit_h_4b8939.pth"
        )
        sam.to("cuda")
        _sam_predictor = SamPredictor(sam)
        print("SAM loaded")
    return _sam_predictor

def get_inpaint_pipe():
    global _inpaint_pipe
    if _inpaint_pipe is None:
        from diffusers import StableDiffusionInpaintPipeline
        _inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        ).to("cuda")
        _inpaint_pipe.safety_checker = None
        print("SD Inpainting loaded")
    return _inpaint_pipe

def get_ootd():
    global _ootd
    if _ootd is None:
        import sys
        sys.path.insert(0, "/root/OOTDiffusion")
        from run_ootd import OOTDiffusion
        _ootd = OOTDiffusion(gpu_id=0)
        print("OOTDiffusion loaded")
    return _ootd


# ─────────────────────────────────────────
# МАСКА ЧЕРЕЗ SAM
# ─────────────────────────────────────────

def get_mask_sam(image_cv: np.ndarray, region: str = "upper") -> np.ndarray:
    predictor = get_sam()
    h, w = image_cv.shape[:2]

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    if region == "upper":
        point = np.array([[w // 2, int(h * 0.38)]])
    elif region == "lower":
        point = np.array([[w // 2, int(h * 0.70)]])
    else:
        point = np.array([[w // 2, int(h * 0.50)]])

    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=np.array([1]),
        multimask_output=True
    )

    best = masks[np.argmax(scores)]
    return (best * 255).astype(np.uint8)


def get_person_mask(image_cv: np.ndarray) -> np.ndarray:
    pil = cv_to_pil(image_cv)
    removed = remove(pil)
    alpha = np.array(removed)[:, :, 3]
    return alpha


# ─────────────────────────────────────────
# RANDOM COLOR
# ─────────────────────────────────────────

def random_color_shift(image_cv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hue = random.randint(0, 179)
    saturation = random.randint(-30, 30)
    value = random.randint(-20, 20)

    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV).astype(np.int32)
    mask_bool = mask > 128

    hsv[mask_bool, 0] = hue
    hsv[mask_bool, 1] = np.clip(hsv[mask_bool, 1] + saturation, 0, 255)
    hsv[mask_bool, 2] = np.clip(hsv[mask_bool, 2] + value, 0, 255)

    hsv = hsv.astype(np.uint8)
    converted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = image_cv.copy()
    result[mask_bool] = converted[mask_bool]
    return result


# ─────────────────────────────────────────
# BACKGROUND SWAP — SD Inpainting
# ─────────────────────────────────────────

def generate_background_preview(
    image_cv: np.ndarray,
) -> tuple:
    """
    Каждый раз берёт рандомный промпт из списка.
    Человек остаётся, фон генерируется новый.
    """
    pipe = get_inpaint_pipe()
    h, w = image_cv.shape[:2]

    # Рандомный фон из списка — каждый раз разный
    bg_prompt = get_random_bg_prompt()
    print(f"Background prompt: {bg_prompt}")

    person_mask = get_person_mask(image_cv)
    bg_mask = cv2.bitwise_not(person_mask)

    pil_image = cv_to_pil(image_cv)
    pil_mask = Image.fromarray(bg_mask).convert("RGB")

    pil_image_512 = pil_image.resize((512, 512))
    pil_mask_512 = pil_mask.resize((512, 512))

    negative_prompt = "blurry, low quality, deformed, artifacts, watermark, duplicate person"

    result_pil = pipe(
        prompt=bg_prompt + ", high quality, photorealistic, 8k",
        negative_prompt=negative_prompt,
        image=pil_image_512,
        mask_image=pil_mask_512,
        num_inference_steps=25,
        guidance_scale=7.5,
    ).images[0]

    result_pil = result_pil.resize((w, h), Image.LANCZOS)
    result_cv = pil_to_cv(result_pil)

    return result_cv, bg_mask, bg_prompt


def apply_background_to_frame(
    frame_cv: np.ndarray,
    reference_result: np.ndarray,
) -> np.ndarray:
    h, w = frame_cv.shape[:2]

    person_mask = get_person_mask(frame_cv)
    alpha = person_mask.astype(np.float32) / 255.0
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)

    bg_resized = cv2.resize(reference_result, (w, h))

    result = (
        frame_cv.astype(np.float32) * alpha_3ch +
        bg_resized.astype(np.float32) * (1 - alpha_3ch)
    ).astype(np.uint8)

    return result


# ─────────────────────────────────────────
# CLOTHES SWAP — OOTDiffusion
# ─────────────────────────────────────────

def generate_clothes_preview(
    image_cv: np.ndarray,
    prompt: str,
    region: str = "upper",
) -> tuple:
    # Переводим промпт в английский (можно писать на русском)
    prompt_en = translate_prompt(prompt)
    print(f"Clothes prompt (EN): {prompt_en}")

    ootd = get_ootd()
    pil_image = cv_to_pil(image_cv)

    model_type = {
        "upper": "half_body",
        "lower": "lower_body",
        "full": "full_body",
    }.get(region, "half_body")

    result_pil = ootd.run(
        model_type=model_type,
        image_garm=None,
        image_vton=pil_image,
        prompt=prompt_en,
        n_samples=1,
        n_steps=20,
        image_scale=2.0,
        seed=-1,
    )[0]

    result_cv = pil_to_cv(result_pil)
    clothes_mask = get_mask_sam(result_cv, region)

    return result_cv, clothes_mask


def apply_clothes_to_frame(
    frame_cv: np.ndarray,
    clothes_mask: np.ndarray,
    reference_result: np.ndarray,
) -> np.ndarray:
    h, w = frame_cv.shape[:2]

    mask_resized = cv2.resize(clothes_mask, (w, h))
    ref_resized = cv2.resize(reference_result, (w, h))

    mask_bool = mask_resized > 128
    result = frame_cv.copy()
    result[mask_bool] = ref_resized[mask_bool]

    return result


# ─────────────────────────────────────────
# PREVIEW — первый кадр
# ─────────────────────────────────────────

def color_swap_preview(
    frame_path: str,
    output_path: str,
    mask_save_path: str,
    ref_save_path: str,
    mode: str,
    region: str,
    prompt: str = "",
) -> str:
    image = load_image_cv(frame_path)

    if mode == "random_color":
        mask = get_mask_sam(image, region)
        result = random_color_shift(image, mask)
        save_mask(mask, mask_save_path)
        save_image_cv(result, ref_save_path)

    elif mode == "random_bg":
        # Промпт не нужен — берётся рандомный из списка
        result, bg_mask, used_prompt = generate_background_preview(image)
        save_mask(bg_mask, mask_save_path)
        save_image_cv(result, ref_save_path)

    elif mode == "random_clothes":
        # Промпт пишешь на русском — переводится автоматически
        result, clothes_mask = generate_clothes_preview(image, prompt, region)
        save_mask(clothes_mask, mask_save_path)
        save_image_cv(result, ref_save_path)

    else:
        result = image

    save_image_cv(result, output_path)
    return output_path


# ─────────────────────────────────────────
# APPLY TO VIDEO
# ─────────────────────────────────────────

def color_swap_video(
    video_path: str,
    output_path: str,
    mask_path: str,
    ref_path: str,
    mode: str,
) -> str:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_video = output_path.replace(".mp4", "_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))

    mask = load_mask(mask_path)
    reference = load_image_cv(ref_path)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "random_color":
            mask_resized = cv2.resize(mask, (w, h))
            ref_resized = cv2.resize(reference, (w, h))
            mask_bool = mask_resized > 128
            result = frame.copy()
            result[mask_bool] = ref_resized[mask_bool]

        elif mode == "random_bg":
            # Фон из референса (сгенерирован один раз на preview)
            # Человек вырезается из каждого кадра
            result = apply_background_to_frame(frame, reference)

        elif mode == "random_clothes":
            result = apply_clothes_to_frame(frame, mask, reference)

        else:
            result = frame

        out.write(result)
        frame_idx += 1

        if frame_idx % 30 == 0:
            progress = round((frame_idx / total_frames) * 100)
            print(f"Processing: {progress}% ({frame_idx}/{total_frames})")

    cap.release()
    out.release()

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
