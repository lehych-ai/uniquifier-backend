#!/usr/bin/env bash
# Provision a Vast pod for the Uniquifier backend (Flux.2 + Wan2.2 via ComfyUI).
# Idempotent. Boots the FastAPI orchestrator on :8000 and kicks ComfyUI
# provisioning (comfy_setup.sh) in the background so the pod self-installs.
set -uo pipefail

APP_DIR="${APP_DIR:-/root/app}"
PORT="${PORT:-8000}"
REPO="${REPO:-https://github.com/lehych-ai/uniquifier-backend}"

echo "==> system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y || true
apt-get install -y --no-install-recommends \
  git wget curl aria2 ffmpeg build-essential python3-dev \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev || true

echo "==> code"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --depth 1 origin main || git -C "$APP_DIR" fetch origin main
  git -C "$APP_DIR" reset --hard FETCH_HEAD
else
  git clone --depth 1 "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"
git rev-parse --short HEAD | sed 's/^/    code @ /'
[ -f "$APP_DIR/gpu-backend/main.py" ] && cd "$APP_DIR/gpu-backend"

echo "==> backend python deps (minimal)"
pip install --upgrade pip || true
pip install -r requirements.txt || true

echo "==> kick ComfyUI provisioning in background (Flux.2 + Wan2.2 + models)"
# idempotent: comfy_setup is also guarded by /api/comfy/install (pgrep). It runs
# long (clones nodes + downloads ~70GB) and ends by serving ComfyUI on :8188.
chmod +x comfy_setup.sh 2>/dev/null || true
if ! pgrep -f comfy_setup.sh >/dev/null 2>&1; then
  nohup bash comfy_setup.sh > /root/comfy.log 2>&1 &
fi

echo "==> launching API on :$PORT"
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1 --log-level info
