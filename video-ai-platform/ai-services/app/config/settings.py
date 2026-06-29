"""
Application settings loaded from environment variables via pydantic-settings.
All configuration is centralized here. No hardcoded values elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the ai-services/ directory, regardless of launch CWD.
# settings.py lives at  ai-services/app/config/settings.py  → parents[2] == ai-services/
AI_SERVICES_DIR: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info", alias="LOG_LEVEL"
    )

    # ── Device ────────────────────────────────────────────────
    device: Literal["auto", "cuda", "cpu"] = Field(default="auto", alias="DEVICE")

    # ── Directories ───────────────────────────────────────────
    output_dir: Path = Field(default=Path("output"), alias="OUTPUT_DIR")
    temp_dir: Path = Field(default=Path("temp"), alias="TEMP_DIR")
    models_dir: Path = Field(default=Path("models_weights"), alias="MODELS_DIR")

    # ── Streaming ─────────────────────────────────────────────
    frame_buffer_size: int = Field(default=32, alias="FRAME_BUFFER_SIZE")
    max_resolution: int = Field(default=1920, alias="MAX_RESOLUTION")

    # ── RAFT Stabilizer ───────────────────────────────────────
    raft_model_path: Path = Field(
        default=Path("RAFT/models/raft-sintel.pth"), alias="RAFT_MODEL_PATH"
    )
    raft_repo_path: Path = Field(
        default=Path("RAFT"), alias="RAFT_REPO_PATH"
    )
    raft_smoothing_radius: int = Field(default=30, alias="RAFT_SMOOTHING_RADIUS")
    raft_crop_ratio: float = Field(default=0.07, alias="RAFT_CROP_RATIO")
    raft_flow_width: int = Field(default=960, alias="RAFT_FLOW_WIDTH")
    raft_flow_height: int = Field(default=540, alias="RAFT_FLOW_HEIGHT")
    raft_iters: int = Field(default=20, alias="RAFT_ITERS")

    # ── Heavy Rain Removal ────────────────────────────────────
    heavy_rain_repo_path: Path = Field(
        default=Path("HeavyRainRemoval"), alias="HEAVY_RAIN_REPO_PATH"
    )
    heavy_rain_checkpoint: Path = Field(
        default=Path("pretrained/HeavyRainRemoval/checkpoint/HeavyRain-stage2-2019-05-11-76_ckpt.pth.tar"),
        alias="HEAVY_RAIN_CHECKPOINT",
    )
    heavy_rain_checkpoint_url: str = Field(
        default="https://www.dropbox.com/s/h8x6xl6epc45ngn/HeavyRain-stage2-2019-05-11-76_ckpt.pth.tar?dl=1",
        alias="HEAVY_RAIN_CHECKPOINT_URL",
    )

    # ── Video Visibility (PromptIR) ───────────────────────────
    promptir_checkpoint: Path = Field(
        default=Path("pretrained/PromptIR/checkpoint/model.ckpt"), alias="PROMPTIR_CHECKPOINT"
    )
    promptir_checkpoint_url: str = Field(
        default="https://github.com/va1shn9v/PromptIR/releases/download/v1.0/model.ckpt",
        alias="PROMPTIR_CHECKPOINT_URL"
    )
    promptir_repo_path: Path = Field(
        default=Path("PromptIR"), alias="PROMPTIR_REPO_PATH"
    )
    promptir_tile_size: int = Field(default=512, alias="PROMPTIR_TILE_SIZE")
    promptir_tile_overlap: int = Field(default=32, alias="PROMPTIR_TILE_OVERLAP")
    promptir_contrast_alpha: float = Field(default=1.3, alias="PROMPTIR_CONTRAST_ALPHA")
    promptir_contrast_beta: float = Field(default=10.0, alias="PROMPTIR_CONTRAST_BETA")
    promptir_clahe_clip: float = Field(default=1.5, alias="PROMPTIR_CLAHE_CLIP")

    # ── Distance Estimation ───────────────────────────────────
    distance_weights_path: Path = Field(
        default=Path("../distanceEstimation_d2/best.pth"), alias="DISTANCE_WEIGHTS_PATH"
    )
    distance_yaml_path: Path = Field(
        default=Path("../distanceEstimation_d2/data.yaml"), alias="DISTANCE_YAML_PATH"
    )
    distance_confidence_threshold: float = Field(default=0.3, alias="DISTANCE_CONFIDENCE_THRESHOLD")

    @field_validator("output_dir", "temp_dir", "models_dir", mode="after")
    @classmethod
    def ensure_directories_exist(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def resolve_raft_paths_to_absolute(self) -> "Settings":
        """
        Convert RAFT paths to absolute paths anchored at AI_SERVICES_DIR.

        If the path supplied via .env (or default) is relative, prefix it with
        AI_SERVICES_DIR so the FastAPI process finds RAFT regardless of the
        working directory from which uvicorn is launched.
        """
        if not self.raft_repo_path.is_absolute():
            self.raft_repo_path = (AI_SERVICES_DIR / self.raft_repo_path).resolve()
        if not self.raft_model_path.is_absolute():
            self.raft_model_path = (AI_SERVICES_DIR / self.raft_model_path).resolve()
        if not self.distance_weights_path.is_absolute():
            self.distance_weights_path = (AI_SERVICES_DIR / self.distance_weights_path).resolve()
        if not self.distance_yaml_path.is_absolute():
            self.distance_yaml_path = (AI_SERVICES_DIR / self.distance_yaml_path).resolve()
        return self

    def resolve_device(self) -> str:
        """Return the concrete torch device string after checking availability."""
        if self.device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


settings = Settings()
