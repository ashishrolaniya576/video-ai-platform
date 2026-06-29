# Installation Report — VideoAI Platform

**Generated:** 2026-06-29 18:31:50  
**Host:** ramanlab  
**Python:** 3.10.13  
**Repo:** /home/kuldeep/video-ai-platform/video-ai-platform  

## Overall Status: WARN

| Metric | Count |
|--------|-------|
| Total Checks | 48 |
| Passed | 47 |
| Warnings | 1 |
| Failed | 0 |

## Detailed Results

| Status | Component | Detail |
|--------|-----------|--------|
| ✅ PASS | ai-services/.env | 0.0 MB  /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/.env |
| ✅ PASS | backend/.env | 0.0 MB  /home/kuldeep/video-ai-platform/video-ai-platform/backend/.env |
| ✅ PASS | ai-services/output | /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/output |
| ✅ PASS | ai-services/temp | /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/temp |
| ✅ PASS | ai-services/models_weights | /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/models_weights |
| ✅ PASS | ai-services/pretrained/HeavyRainRemoval/checkpoint | /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/pretrained/HeavyRainRemoval/checkpoint |
| ✅ PASS | ai-services/pretrained/PromptIR/checkpoint | /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/pretrained/PromptIR/checkpoint |
| ✅ PASS | logs/ | /home/kuldeep/video-ai-platform/video-ai-platform/logs |
| ✅ PASS | RAFT/core | /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/RAFT/core |
| ✅ PASS | RAFT/core/raft.py | 0.0 MB  /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/RAFT/core/raft.py |
| ✅ PASS | PromptIR/net | /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/PromptIR/net |
| ✅ PASS | PromptIR/net/model.py | 0.0 MB  /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/PromptIR/net/model.py |
| ✅ PASS | HeavyRainRemoval/ | /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/HeavyRainRemoval |
| ✅ PASS | HeavyRainRemoval/helper.py | 0.0 MB  /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/HeavyRainRemoval/helper.py |
| ✅ PASS | RAFT raft-sintel.pth | 20.1 MB  /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/RAFT/models/raft-sintel.pth |
| ✅ PASS | PromptIR model.ckpt | 388.2 MB  /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/pretrained/PromptIR/checkpoint/model.ckpt |
| ✅ PASS | HeavyRain checkpoint | 192.1 MB  /home/kuldeep/video-ai-platform/video-ai-platform/ai-services/pretrained/HeavyRainRemoval/checkpoint/HeavyRain-stage2-2019-05-11-76_ckpt.pth.tar |
| ✅ PASS | import fastapi | importable |
| ✅ PASS | import uvicorn | importable |
| ✅ PASS | import pydantic | importable |
| ✅ PASS | import torch | importable |
| ✅ PASS | import torchvision | importable |
| ✅ PASS | import cv2 | importable |
| ✅ PASS | import numpy | importable |
| ✅ PASS | import scipy | importable |
| ✅ PASS | import PIL | importable |
| ✅ PASS | import einops | importable |
| ⚠️ WARN | import lightning | No module named 'lightning' |
| ✅ PASS | import tqdm | importable |
| ✅ PASS | import skimage | importable |
| ✅ PASS | import gdown | importable |
| ✅ PASS | import psutil | importable |
| ✅ PASS | PyTorch CUDA | GPU: NVIDIA A10 |
| ✅ PASS | backend node_modules | /home/kuldeep/video-ai-platform/video-ai-platform/backend/node_modules/express |
| ✅ PASS | backend socket.io | /home/kuldeep/video-ai-platform/video-ai-platform/backend/node_modules/socket.io |
| ✅ PASS | backend axios | /home/kuldeep/video-ai-platform/video-ai-platform/backend/node_modules/axios |
| ✅ PASS | frontend node_modules | /home/kuldeep/video-ai-platform/video-ai-platform/frontend/node_modules/react |
| ✅ PASS | frontend vite | /home/kuldeep/video-ai-platform/video-ai-platform/frontend/node_modules/vite |
| ✅ PASS | import app.config.settings | importable |
| ✅ PASS | import app.utils.logger | importable |
| ✅ PASS | import app.models.base | importable |
| ✅ PASS | import app.models.stabilize | importable |
| ✅ PASS | import app.models.heavy_rain_remove | importable |
| ✅ PASS | import app.models.video_visibility | importable |
| ✅ PASS | import app.pipeline.pipeline | importable |
| ✅ PASS | import app.api.process | importable |
| ✅ PASS | import app.api.health | importable |
| ✅ PASS | FastAPI create_app() | routes: ['/openapi.json', '/docs', '/docs/oauth2-redirect', '/redoc', '/health', '/progress', '/cancel', '/process'] |

## Next Steps

```bash
# Start all services:
bash scripts/start_all.sh

# Check health:
bash scripts/health_check.sh
```
