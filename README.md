# CloserAI GPU backend

FastAPI service that runs the heavy AI for **Face Swap** and **Color Swap** on a
rented GPU pod (Vast.ai / RunPod / any CUDA box). The desktop app's Python
sidecar talks to it through `VastBackend`; the renderer never calls it directly.

## What it does

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | health + which model weights are present + GPU flag |
| `POST /api/extract-frame` | upload video → `{video_id, frame_url}` |
| `POST /api/color-swap-preview` | render frame-0 preview + save a reusable plan |
| `POST /api/color-swap-video` | start full-video job (returns immediately) |
| `POST /api/face-swap-preview` | upload face → frame-0 preview |
| `POST /api/face-swap-video` | start full-video job |
| `GET /api/progress/{video_id}` | `{percent, status, output_url?, error?}` |
| `GET /files/{kind}/{name}` | serve uploaded/generated files |

Long jobs run on a background thread and report into an in-memory progress map,
so the sidecar polls `/api/progress` and relays it to the UI over SSE.

## Quality notes (why it's not the old code)

**Face swap** — `inswapper_128` + **GFPGAN** restoration (the 128px swap alone
looks soft on HD). `skip_frames` no longer freezes the whole frame: skipped
frames stay live and only the cached face is re-pasted at the current box.
Output is real **libx264**, not `mp4v`.

**Color swap** — every mode recomputes its mask **per frame** (rembg
`u2net_cloth_seg` / person matte), so the edit tracks the moving subject instead
of stamping frame 0:
- `random_color` — consistent hue rotation on the per-frame garment mask.
- `random_bg` — background generated once (SD2 inpainting), live subject
  composited per frame with a feathered, temporally-smoothed matte.
- `random_clothes` — garment look generated once, then LAB-transferred onto the
  per-frame clothing region.

> Full generative garment **replacement** per frame (new garment *shape*, not
> just look) needs a motion-aware video model — that's the Wan2.1 VACE I2V
> upgrade path, intentionally not implemented here.

Everything degrades gracefully: missing weights → fall back (no enhancer / plain
hue shift / person-band mask) instead of crashing.

## Run on a pod

```bash
# optional: require a bearer token on every /api call
export API_TOKEN=your-secret
bash setup.sh           # installs deps, downloads weights, launches uvicorn :8000
```

Override paths/model with `MODELS_DIR`, `UPLOAD_DIR`, `OUTPUT_DIR`,
`SD_INPAINT_MODEL`, `PORT`. See [.env.example](.env.example).

## Auth

If `API_TOKEN` is set, every `/api/*` request must send
`Authorization: Bearer <token>`. Leave unset only behind a private proxy.
