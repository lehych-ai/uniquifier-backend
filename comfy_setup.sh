#!/usr/bin/env bash
# Phase 1 — stand up ComfyUI (headless) for the full head+hair video swap engine.
# AnimateDiff-Evolved gives temporal coherence (the "no flicker"), ControlNet
# drives pose/motion from the source clip, IPAdapter-FaceID injects the reference
# identity. This script installs ComfyUI + the required custom nodes + models and
# launches the API on :8188. Idempotent — safe to re-run. Non-fatal model DLs so a
# single 404 doesn't abort the whole provision; check the summary at the end.
#
# Re-run on the pod:
#   pkill -f 'ComfyUI/main.py' 2>/dev/null
#   wget -qO /root/comfy_setup.sh https://raw.githubusercontent.com/lehych-ai/uniquifier-backend/main/comfy_setup.sh
#   nohup bash /root/comfy_setup.sh > /root/comfy.log 2>&1 &
#   tail -f /root/comfy.log
set -uo pipefail

COMFY_DIR="${COMFY_DIR:-/root/ComfyUI}"
PORT="${COMFY_PORT:-8188}"

echo "==> system packages"
apt-get update -y
apt-get install -y --no-install-recommends git wget curl ffmpeg libgl1 libglib2.0-0

echo "==> ComfyUI core (pinned to a torch-2.1-compatible era)"
# The pod's torch is 2.1.0 (required by onnxruntime-gpu/inswapper on CUDA 11.8).
# Recent ComfyUI needs torch>=2.4 (torch.library.custom_op via comfy_kitchen) and
# pulls numpy 2.x — both break this image. Pin ComfyUI + every node to a 2024
# commit that still runs on torch 2.1, and force numpy<2 at the end.
PIN_DATE="${COMFY_PIN_DATE:-2024-09-15}"

pin_repo() {  # pin_repo <dir>
  local dir="$1" c
  git -C "$dir" fetch -q --unshallow 2>/dev/null || git -C "$dir" fetch -q || true
  c="$(git -C "$dir" rev-list -1 --before="$PIN_DATE 00:00" HEAD 2>/dev/null || true)"
  if [ -n "$c" ]; then
    git -C "$dir" checkout -q "$c" && echo "    pinned $(basename "$dir") -> ${c:0:8} (<=$PIN_DATE)"
  else
    echo "    !! could not pin $(basename "$dir")"
  fi
}

[ -d "$COMFY_DIR/.git" ] || git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
pin_repo "$COMFY_DIR"
cd "$COMFY_DIR"
pip install --upgrade pip
pip install -r requirements.txt

NODES="$COMFY_DIR/custom_nodes"
mkdir -p "$NODES"
clone_node() {  # clone_node <giturl>
  local url="$1" name
  name="$(basename "$url" .git)"
  [ -d "$NODES/$name/.git" ] || git clone "$url" "$NODES/$name" || { echo "    !! clone failed: $name"; return; }
  pin_repo "$NODES/$name"
  [ -f "$NODES/$name/requirements.txt" ] && pip install -r "$NODES/$name/requirements.txt" || true
}

echo "==> custom nodes (pinned)"
clone_node https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved  # temporal (anti-flicker)
clone_node https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet  # AD-compatible controlnet
clone_node https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite     # load/save video
clone_node https://github.com/Fannovel16/comfyui_controlnet_aux        # openpose/depth preprocessors
clone_node https://github.com/cubiq/ComfyUI_IPAdapter_plus             # IPAdapter-FaceID
clone_node https://github.com/cubiq/ComfyUI_essentials                 # mask helpers

# ─────────────────────────────────────────
# models
# ─────────────────────────────────────────
CK="$COMFY_DIR/models/checkpoints"
VAE="$COMFY_DIR/models/vae"
CN="$COMFY_DIR/models/controlnet"
AD="$COMFY_DIR/models/animatediff_models"
IPA="$COMFY_DIR/models/ipadapter"
CLIPV="$COMFY_DIR/models/clip_vision"
LORA="$COMFY_DIR/models/loras"
mkdir -p "$CK" "$VAE" "$CN" "$AD" "$IPA" "$CLIPV" "$LORA"

declare -a MISSING=()
dl() {  # dl <url> <dest>
  if [ -f "$2" ]; then echo "    have $(basename "$2")"; return; fi
  echo "    downloading $(basename "$2")"
  if ! wget -q --show-progress -O "$2" "$1"; then
    rm -f "$2"; echo "    !! FAILED $(basename "$2")"; MISSING+=("$(basename "$2")")
  fi
}

echo "==> SD1.5 realistic checkpoint"
dl "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors" \
   "$CK/realisticVision_v51.safetensors"

echo "==> VAE"
dl "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors" \
   "$VAE/vae-ft-mse-840000.safetensors"

echo "==> AnimateDiff motion module (v3)"
dl "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt" \
   "$AD/v3_sd15_mm.ckpt"

echo "==> ControlNet (openpose + depth)"
dl "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" \
   "$CN/control_v11p_sd15_openpose.pth"
dl "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth" \
   "$CN/control_v11f1p_sd15_depth.pth"

echo "==> IPAdapter-FaceID (+ lora) + CLIP vision"
dl "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin" \
   "$IPA/ip-adapter-faceid-plusv2_sd15.bin"
dl "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors" \
   "$LORA/ip-adapter-faceid-plusv2_sd15_lora.safetensors"
dl "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
   "$CLIPV/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

# IPAdapter-FaceID also needs the InsightFace buffalo_l pack where ComfyUI looks.
echo "==> InsightFace buffalo_l for FaceID"
pip install -q insightface onnxruntime-gpu==1.16.3 2>/dev/null || true
python - <<'PY' || true
import os, zipfile, urllib.request
dest = os.path.expanduser("/root/ComfyUI/models/insightface/models")
os.makedirs(dest, exist_ok=True)
if not os.path.isdir(os.path.join(dest, "buffalo_l")):
    z = os.path.join(dest, "buffalo_l.zip")
    urllib.request.urlretrieve("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip", z)
    with zipfile.ZipFile(z) as f: f.extractall(os.path.join(dest, "buffalo_l"))
    os.remove(z)
    print("    buffalo_l ready")
else:
    print("    buffalo_l present")
PY

# ComfyUI/node pip installs may have pulled numpy 2.x — force it back below 2 so
# torch/insightface/onnxruntime keep their ABI ("_ARRAY_API not found" otherwise).
echo "==> pinning numpy<2 (shared with the inswapper backend)"
pip install -q "numpy<2" || true

echo "==> wiring cuDNN onto LD_LIBRARY_PATH (onnxruntime for FaceID)"
CUDNN_LIB="$(find /opt/conda -name 'libcudnn.so.8*' 2>/dev/null | head -1)"
TORCH_LIB="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${CUDNN_LIB:+$(dirname "$CUDNN_LIB")}:${TORCH_LIB}:/opt/conda/lib:${LD_LIBRARY_PATH:-}"

echo ""
echo "==================== SUMMARY ===================="
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "MISSING MODELS (fix URLs / download manually):"
  for m in "${MISSING[@]}"; do echo "   - $m"; done
else
  echo "all models present"
fi
echo "================================================="
echo "==> launching ComfyUI on :$PORT"
exec python main.py --listen 0.0.0.0 --port "$PORT"
