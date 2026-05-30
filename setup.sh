#!/bin/bash
set -e

echo "================================================"
echo "=== UNIQUIFIER BACKEND SETUP ==="
echo "================================================"


# ─────────────────────────────────────────
# ШАГ 1 — СИСТЕМНЫЕ ПАКЕТЫ
# ─────────────────────────────────────────

echo ""
echo "=== [1/6] Installing system packages ==="
apt-get update -q
apt-get install -y -q \
    ffmpeg \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev


# ─────────────────────────────────────────
# ШАГ 2 — КЛОНИРУЕМ РЕПО
# ─────────────────────────────────────────

echo ""
echo "=== [2/6] Cloning repo ==="
cd /root

if [ -d "uniquifier-backend" ]; then
    echo "Repo exists, pulling latest..."
    cd uniquifier-backend
    git pull
else
    git clone https://github.com/ТВОЙ_ЮЗЕР/uniquifier-backend.git
    cd uniquifier-backend
fi


# ─────────────────────────────────────────
# ШАГ 3 — PYTHON БИБЛИОТЕКИ
# ─────────────────────────────────────────

echo ""
echo "=== [3/6] Installing Python packages ==="

pip install --upgrade pip -q
pip install -r requirements.txt -q

# segment-anything с GitHub
pip install git+https://github.com/facebookresearch/segment-anything.git -q

# diffusers для SD Inpainting
pip install diffusers transformers accelerate xformers -q

# Кешируем buffalo_l для InsightFace
python -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print('buffalo_l ready')
"


# ─────────────────────────────────────────
# ШАГ 4 — OOTDIFFUSION
# ─────────────────────────────────────────

echo ""
echo "=== [4/6] Setting up OOTDiffusion ==="

if [ ! -d "/root/OOTDiffusion" ]; then
    git clone https://github.com/levihsu/OOTDiffusion.git /root/OOTDiffusion
fi

cd /root/OOTDiffusion
pip install -r requirements.txt -q

# Скачиваем веса OOTDiffusion с HuggingFace (~7gb)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'levihsu/OOTDiffusion',
    local_dir='/root/OOTDiffusion/checkpoints',
    ignore_patterns=['*.msgpack', '*.h5']
)
print('OOTDiffusion weights ready')
"

cd /root/uniquifier-backend


# ─────────────────────────────────────────
# ШАГ 5 — СКАЧИВАЕМ МОДЕЛИ
# ─────────────────────────────────────────

echo ""
echo "=== [5/6] Downloading models ==="

mkdir -p /root/models

# Face swap
if [ ! -f "/root/models/inswapper_128.onnx" ]; then
    echo "Downloading inswapper_128.onnx (~500mb)..."
    wget -q --show-progress \
        "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx" \
        -O /root/models/inswapper_128.onnx
else
    echo "inswapper_128.onnx exists, skipping"
fi

# SAM
if [ ! -f "/root/models/sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM vit_h (~2.4gb)..."
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
        -O /root/models/sam_vit_h_4b8939.pth
else
    echo "SAM exists, skipping"
fi

# SD Inpainting — кешируем веса с HuggingFace (~4gb)
python -c "
from diffusers import StableDiffusionInpaintPipeline
import torch
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting',
    torch_dtype=torch.float16,
)
print('SD Inpainting weights cached')
"

# rembg u2net
python -c "
from rembg import remove
from PIL import Image
import numpy as np
img = Image.fromarray(np.zeros((64,64,3), dtype=np.uint8))
remove(img)
print('rembg u2net ready')
"


# ─────────────────────────────────────────
# ШАГ 6 — ЗАПУСК
# ─────────────────────────────────────────

echo ""
echo "=== [6/6] Starting server ==="

mkdir -p /tmp/uploads /tmp/outputs

cd /root/uniquifier-backend

echo ""
echo "================================================"
echo "Server running on port 8000"
echo "================================================"
echo ""

uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
