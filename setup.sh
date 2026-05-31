#!/usr/bin/env bash
# Provision a Vast.ai (or any CUDA) pod for the CloserAI GPU backend.
# Idempotent: safe to re-run. Honors env overrides:
#   MODELS_DIR (default /root/models), APP_DIR (default /root/app),
#   PORT (default 8000), API_TOKEN (optional), SD_INPAINT_MODEL.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/root/models}"
APP_DIR="${APP_DIR:-/root/app}"
PORT="${PORT:-8000}"
REPO="${REPO:-https://github.com/lehych-ai/uniquifier-backend}"

echo "==> system packages"
apt-get update -y
# build-essential is required: insightface compiles a Cython extension from
# source, and the pytorch *-runtime images ship without a compiler (no g++).
apt-get install -y --no-install-recommends \
  build-essential python3-dev \
  ffmpeg git wget curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev

echo "==> code"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only || true
else
  git clone --depth 1 "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"
# If this script lives in a gpu-backend/ subdir of the desktop repo, use that.
[ -f "$APP_DIR/gpu-backend/main.py" ] && cd "$APP_DIR/gpu-backend"

echo "==> python deps"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> model weights -> $MODELS_DIR"
mkdir -p "$MODELS_DIR"

dl() {  # dl <url> <dest>
  if [ ! -f "$2" ]; then
    echo "    downloading $(basename "$2")"
    wget -q --show-progress -O "$2" "$1"
  else
    echo "    have $(basename "$2")"
  fi
}

# Face swap: inswapper (~500MB) + GFPGAN restorer (~340MB)
dl "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" \
   "$MODELS_DIR/inswapper_128.onnx"
dl "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
   "$MODELS_DIR/GFPGANv1.4.pth"

# Pre-fetch the InsightFace detector pack (buffalo_l ~280MB). Without this the
# first face-swap request stalls while it downloads mid-request.
echo "==> prefetch insightface buffalo_l"
python - <<'PY'
try:
    from insightface.app import FaceAnalysis
    FaceAnalysis(name="buffalo_l")  # construction downloads the model pack
    print("    buffalo_l cached")
except Exception as e:
    print("    buffalo_l prefetch skipped (will lazy-load):", e)
PY

# Warm rembg model cache (person matte + clothes segmentation)
echo "==> warming rembg models"
python - <<'PY'
try:
    from rembg import new_session
    new_session("u2net")
    new_session("u2net_cloth_seg")
    print("    rembg models cached")
except Exception as e:
    print("    rembg warm failed (will lazy-load):", e)
PY

# Pre-fetch SD inpainting weights. stabilityai/* went gated (HTTP 401), so use a
# public SD1.5-inpainting mirror by default. Non-fatal — falls back to lazy load.
SD_INPAINT_MODEL="${SD_INPAINT_MODEL:-botp/stable-diffusion-v1-5-inpainting}"
export SD_INPAINT_MODEL
echo "==> prefetch SD inpainting (${SD_INPAINT_MODEL})"
python - <<'PY'
import os
mid = os.environ.get("SD_INPAINT_MODEL")
try:
    from huggingface_hub import snapshot_download
    snapshot_download(mid, allow_patterns=["*.json","*.txt","*.safetensors","*.bin"])
    print("    cached", mid)
except Exception as e:
    print("    SD prefetch skipped (will lazy-load):", e)
PY

mkdir -p /tmp/uploads /tmp/outputs

# onnxruntime-gpu needs libcudnn.so.8 / libcublas, which this image keeps inside
# torch/conda rather than on the default loader path → "libcudnn.so.8: cannot
# open shared object file". Locate them and export LD_LIBRARY_PATH before launch.
echo "==> wiring cuDNN onto LD_LIBRARY_PATH for onnxruntime"
CUDNN_LIB="$(find /opt/conda -name 'libcudnn.so.8*' 2>/dev/null | head -1)"
TORCH_LIB="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${CUDNN_LIB:+$(dirname "$CUDNN_LIB")}:${TORCH_LIB}:/opt/conda/lib:${LD_LIBRARY_PATH:-}"
echo "    cudnn=$CUDNN_LIB"
echo "    LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "==> launching API on :$PORT"
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1 --log-level info
