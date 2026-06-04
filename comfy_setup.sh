#!/usr/bin/env bash
# Provision ComfyUI for the Uniquifier video stack: Flux.2 Klein (first-frame
# edits) + Wan 2.2 Animate (video gen, Kijai WanVideoWrapper). Idempotent.
# Target: a modern CUDA 12.x / torch 2.x image on an RTX PRO 6000 (96GB).
set -uo pipefail

COMFY_DIR="${COMFY_DIR:-/root/ComfyUI}"
APP_DIR="${APP_DIR:-/root/app}"
MANIFEST="${MANIFEST:-$APP_DIR/gpu-backend/models_manifest.txt}"
[ -f "$MANIFEST" ] || MANIFEST="$(dirname "$0")/models_manifest.txt"
LOG(){ echo "[comfy_setup] $*"; }

export DEBIAN_FRONTEND=noninteractive
export PIP_ROOT_USER_ACTION=ignore
export HF_HUB_ENABLE_HF_TRANSFER=0

LOG "apt deps"
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y --no-install-recommends git wget curl aria2 ffmpeg build-essential \
  libgl1 libglib2.0-0 python3-dev >/dev/null 2>&1 || true

# ── ComfyUI core (latest — Flux.2 + Wan2.2 need recent) ───────────────────────
if [ ! -d "$COMFY_DIR/.git" ]; then
  LOG "clone ComfyUI"
  git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
else
  LOG "update ComfyUI"; git -C "$COMFY_DIR" pull --ff-only || true
fi
python -m pip install --upgrade pip >/dev/null 2>&1 || true
pip install -r "$COMFY_DIR/requirements.txt" || true

CN="$COMFY_DIR/custom_nodes"
mkdir -p "$CN"

# ── custom node packs (clone + install deps; non-fatal) ───────────────────────
clone_node(){  # clone_node <repo_url>
  local url="$1" name; name="$(basename "$url" .git)"
  if [ ! -d "$CN/$name" ]; then
    LOG "node: $name"; git clone --depth 1 "$url" "$CN/$name" || { LOG "FAILED clone $name"; return; }
  else
    git -C "$CN/$name" pull --ff-only 2>/dev/null || true
  fi
  [ -f "$CN/$name/requirements.txt" ] && pip install -r "$CN/$name/requirements.txt" 2>&1 | tail -1 || true
}
clone_node https://github.com/ltdrdata/ComfyUI-Manager
clone_node https://github.com/kijai/ComfyUI-WanVideoWrapper
clone_node https://github.com/kijai/ComfyUI-KJNodes
clone_node https://github.com/kijai/ComfyUI-segment-anything-2
clone_node https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
clone_node https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
clone_node https://github.com/rgthree/rgthree-comfy
clone_node https://github.com/chrisgoringe/cg-use-everywhere
clone_node https://github.com/yolain/ComfyUI-Easy-Use
clone_node https://github.com/pythongosssss/ComfyUI-Custom-Scripts
clone_node https://github.com/scraed/LanPaint
clone_node https://github.com/zcfrank1st/Comfyui-Toolbox
clone_node https://github.com/cubiq/ComfyUI_essentials

# common runtime deps these nodes need (sdpa attention path → no flash/sage required)
pip install "numpy<2" opencv-python imageio-ffmpeg onnxruntime-gpu segment-anything \
  accelerate sentencepiece ftfy einops 2>&1 | tail -1 || true

# ── models → ComfyUI/models/<subdir> (resumable) ──────────────────────────────
dl(){  # dl <url> <subdir>
  local url="$1" sub="$2" dir="$COMFY_DIR/models/$2" fn
  fn="$(basename "$url")"; fn="${fn//%20/ }"
  mkdir -p "$dir"
  if [ -s "$dir/$fn" ]; then LOG "have $sub/$fn"; return; fi
  LOG "download $sub/$fn"
  if command -v aria2c >/dev/null; then
    aria2c -x8 -s8 -c --summary-interval=0 --console-log-level=warn -d "$dir" -o "$fn" "$url" \
      || wget -c -q -O "$dir/$fn" "$url" || LOG "FAILED $fn"
  else
    wget -c -q -O "$dir/$fn" "$url" || LOG "FAILED $fn"
  fi
}
if [ -f "$MANIFEST" ]; then
  LOG "models from $MANIFEST"
  while IFS= read -r line; do
    case "$line" in ''|\#*) continue;; esac
    url="${line%%|*}"; sub="${line##*|}"
    dl "$url" "$sub"
  done < "$MANIFEST"
else
  LOG "NO MANIFEST at $MANIFEST"
fi

# ── launch ComfyUI on 8188 (local only; backend proxies it) ───────────────────
LOG "launching ComfyUI :8188"
cd "$COMFY_DIR"
exec python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --highvram
