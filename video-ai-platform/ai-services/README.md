# AI Service — VideoAI Platform

FastAPI-based AI processing service for the VideoAI Platform. Provides video stabilization (RAFT), heavy rain removal (HeavyRainRemoval), and visibility enhancement (PromptIR).

---

## Folder Structure

```
ai-services/
├── app/
│   ├── api/
│   │   ├── process.py          # POST /process endpoint
│   │   └── health.py           # GET /health endpoint
│   ├── pipeline/
│   │   └── pipeline.py         # Pipeline orchestrator (no AI logic)
│   ├── streaming/
│   │   ├── reader.py           # VideoReader — frame-by-frame streaming
│   │   └── writer.py           # VideoWriter — output video writer
│   ├── models/
│   │   ├── base.py             # Abstract model interface
│   │   ├── stabilize.py        # RAFT stabilization
│   │   ├── heavy_rain_remove.py# HeavyRainRemoval
│   │   └── video_visibility.py # PromptIR visibility enhancement
│   ├── config/
│   │   └── settings.py         # All config via pydantic-settings + .env
│   ├── utils/
│   │   ├── logger.py           # Structured logging
│   │   └── video_utils.py      # Video helpers
│   └── main.py                 # FastAPI app factory + lifespan
├── notebooks/                  # Source notebooks (reference only)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

### 1. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. PyTorch (with CUDA support)

Visit https://pytorch.org/get-started/locally/ and install the correct version for your CUDA version.

Example for CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Example for CPU only:
```bash
pip install torch torchvision
```

### 3. Other dependencies

```bash
pip install -r requirements.txt
```

---

## Model Setup

### RAFT (Stabilization)

```bash
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT && bash download_models.sh
cd ..
```

Required file: `RAFT/models/raft-sintel.pth`

### HeavyRainRemoval

The Heavy Rain Removal model is fully integrated.
- **Repository**: Cloned automatically on first run to `HeavyRainRemoval/`
- **Checkpoint**: Downloaded automatically on first run to `pretrained/HeavyRainRemoval/checkpoint/HeavyRain-stage2-2019-05-11-76_ckpt.pth.tar`

No manual setup is required for this model.

### PromptIR (Video Visibility)

The PromptIR Video Visibility Enhancement model is fully integrated.
- **Repository**: Cloned automatically on first run to `PromptIR/`
- **Checkpoint**: Downloaded automatically on first run to `pretrained/PromptIR/checkpoint/model.ckpt`

No manual setup is required for this model. It also preserves the tiled inference algorithm and all post-processing enhancements (CLAHE, Sharpening, Contrast) from the original notebook.

---

## Configuration

```bash
cp .env.example .env
# Edit .env to set correct paths and device
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `PORT` | `8000` | HTTP port |
| `RAFT_MODEL_PATH` | `RAFT/models/raft-sintel.pth` | RAFT weights |
| `HEAVY_RAIN_CHECKPOINT` | `HeavyRainRemoval/.ckpt/...` | Heavy rain checkpoint |
| `PROMPTIR_CHECKPOINT` | `models_weights/promptir_model.ckpt` | PromptIR weights |
| `OUTPUT_DIR` | `output` | Processed video output directory |

---

## Running

```bash
# From the ai-services/ directory
python -m app.main

# Or directly with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Development (no reload — GPU models must stay in memory)
python app/main.py
```

The service starts on **http://localhost:8000**

Interactive API docs: **http://localhost:8000/docs**

---

## API

### `GET /health`

```json
{
  "status": "running",
  "device": "cuda",
  "models_loaded": {
    "stabilization": true,
    "heavy_rain_removal": true,
    "video_visibility": true
  }
}
```

---

### `POST /process`

**Request:**
```json
{
  "videoPath": "/path/to/video.mp4",
  "stabilization": true,
  "heavyRainRemoval": false,
  "videoVisibility": true
}
```

**Response:**
```json
{
  "status": "completed",
  "outputVideo": "output/video_processed.mp4",
  "executionTime": "1m 23.45s",
  "logs": [
    "Request validated successfully.",
    "Video loaded: ... | 1920x1080 @ 30.00fps | 900 frames",
    "Pipeline: Input → VideoStabilization → VideoVisibility → Output",
    "Stage: VideoStabilization — processing 900 frames…",
    "Stage VideoStabilization complete — 900 frames out | 45.2s",
    "Stage: VideoVisibility — processing 900 frames…",
    "Stage VideoVisibility complete — 900 frames out | 38.1s",
    "Processing completed in 1m 23.45s — saved to output/video_processed.mp4"
  ],
  "error": null
}
```

---

## Node.js Backend Integration

The Node.js backend calls this service at `AI_SERVICE_URL` (default `http://localhost:8000`).

Update `backend/.env`:
```
AI_SERVICE_URL=http://localhost:8000
```

The `aiService.js` in the backend should be updated to forward requests to `POST /process` and stream progress via WebSocket.

---

## Pipeline Execution Order

When multiple features are enabled, they always run in this fixed order:

```
Input Video
    ↓
Stabilization    (RAFT optical flow)
    ↓
Heavy Rain Removal  (HeavyRainRemoval network)
    ↓
Video Visibility  (PromptIR + CLAHE + sharpening)
    ↓
Output Video
```

Disabled features are skipped entirely — no model is loaded or executed.

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Invalid video path | HTTP 422 with validation error |
| No features enabled | HTTP 422 with message |
| Model weights missing | Warning at startup; HTTP 500 if requested |
| CUDA out of memory | Caught; returned as failed result with suggestion |
| Corrupted video | Caught; returned as failed result |
| OpenCV error | Caught; returned as failed result |
| Unhandled exception | Global handler; HTTP 500, server stays alive |

---

## Performance Notes

- Models load **once** at startup and stay in GPU memory for the lifetime of the service.
- Use `DEVICE=cpu` if GPU memory is insufficient.
- For large videos, reduce `PROMPTIR_TILE_SIZE` to lower GPU memory usage.
- RAFT requires the full frame sequence in RAM for trajectory smoothing. For very long videos (>1000 frames) consider splitting into segments.
