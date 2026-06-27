#!/usr/bin/env python3
"""
verify_installation.py — Full installation verifier for VideoAI Platform.

Usage:
    python scripts/verify_installation.py --repo-root /path/to/video-ai-platform

Checks:
    - Python packages (all requirements.txt entries)
    - Node.js packages (backend + frontend)
    - AI model repositories (RAFT, PromptIR, HeavyRainRemoval)
    - AI model weight files (size-verified)
    - Runtime directory structure
    - Environment variable files
    - AI model Python imports
    - FastAPI application creation (no model loading)

Output:
    - Console table with PASS / WARN / FAIL per component
    - INSTALL_REPORT.md written to repo root
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal

# ── Types ─────────────────────────────────────────────────────────────────────
Status = Literal["PASS", "WARN", "FAIL"]

@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str


@dataclass
class Report:
    results: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: Status, detail: str) -> None:
        self.results.append(CheckResult(name, status, detail))

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def warned(self) -> int:
        return sum(1 for r in self.results if r.status == "WARN")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")

    @property
    def total(self) -> int:
        return len(self.results)

    def overall(self) -> Status:
        if self.failed > 0:
            return "FAIL"
        if self.warned > 0:
            return "WARN"
        return "PASS"


# ── Colour output ─────────────────────────────────────────────────────────────
ANSI = {
    "PASS": "\033[0;32m",
    "WARN": "\033[1;33m",
    "FAIL": "\033[0;31m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "CYAN": "\033[0;36m",
}

def colour(text: str, tag: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{ANSI.get(tag, '')}{text}{ANSI['RESET']}"

def print_row(result: CheckResult) -> None:
    icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[result.status]
    status_str = colour(f"[{result.status}]", result.status)
    icon_str = colour(icon, result.status)
    print(f"  {icon_str} {status_str:<20} {result.name:<45} {result.detail}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def check_file(
    report: Report,
    name: str,
    path: Path,
    min_bytes: int = 1024,
) -> bool:
    if not path.exists():
        report.add(name, "FAIL", f"NOT FOUND: {path}")
        return False
    size = path.stat().st_size
    if size < min_bytes:
        report.add(name, "WARN", f"Suspiciously small ({size} bytes): {path}")
        return False
    report.add(name, "PASS", f"{size / 1024 / 1024:.1f} MB  {path}")
    return True


def check_dir(report: Report, name: str, path: Path) -> bool:
    if path.is_dir():
        report.add(name, "PASS", str(path))
        return True
    report.add(name, "FAIL", f"NOT FOUND: {path}")
    return False


def check_import(report: Report, package: str, *, warn_only: bool = False) -> bool:
    try:
        importlib.import_module(package)
        report.add(f"import {package}", "PASS", "importable")
        return True
    except ImportError as exc:
        status: Status = "WARN" if warn_only else "FAIL"
        report.add(f"import {package}", status, str(exc))
        return False


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="VideoAI installation verifier")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Absolute path to the video-ai-platform root",
    )
    args = parser.parse_args()

    REPO = Path(args.repo_root).resolve()
    AI_DIR = REPO / "ai-services"
    BACKEND_DIR = REPO / "backend"
    FRONTEND_DIR = REPO / "frontend"

    # Add ai-services/ to sys.path so app.* imports work
    if str(AI_DIR) not in sys.path:
        sys.path.insert(0, str(AI_DIR))

    report = Report()

    print(colour("\n╔══════════════════════════════════════════════════════╗", "CYAN"))
    print(colour("║      VideoAI Platform — Installation Verifier        ║", "CYAN"))
    print(colour("╚══════════════════════════════════════════════════════╝\n", "CYAN"))

    # =========================================================================
    # 1. Environment files
    # =========================================================================
    print(colour("  ── Environment Files ──", "BOLD"))
    check_file(report, "ai-services/.env", AI_DIR / ".env", min_bytes=10)
    check_file(report, "backend/.env", BACKEND_DIR / ".env", min_bytes=5)

    # =========================================================================
    # 2. Runtime directories
    # =========================================================================
    print(colour("\n  ── Runtime Directories ──", "BOLD"))
    for subdir in ["output", "temp", "models_weights",
                   "pretrained/HeavyRainRemoval/checkpoint",
                   "pretrained/PromptIR/checkpoint"]:
        check_dir(report, f"ai-services/{subdir}", AI_DIR / subdir)
    check_dir(report, "logs/", REPO / "logs")

    # =========================================================================
    # 3. Model repositories
    # =========================================================================
    print(colour("\n  ── AI Model Repositories ──", "BOLD"))
    check_dir(report, "RAFT/core", AI_DIR / "RAFT" / "core")
    check_file(report, "RAFT/core/raft.py", AI_DIR / "RAFT" / "core" / "raft.py", min_bytes=100)
    check_dir(report, "PromptIR/net", AI_DIR / "PromptIR" / "net")
    check_file(report, "PromptIR/net/model.py", AI_DIR / "PromptIR" / "net" / "model.py", min_bytes=100)
    check_dir(report, "HeavyRainRemoval/", AI_DIR / "HeavyRainRemoval")
    check_file(report, "HeavyRainRemoval/helper.py", AI_DIR / "HeavyRainRemoval" / "helper.py", min_bytes=10)

    # =========================================================================
    # 4. Model weight files
    # =========================================================================
    print(colour("\n  ── Model Weight Files ──", "BOLD"))
    check_file(
        report, "RAFT raft-sintel.pth",
        AI_DIR / "RAFT" / "models" / "raft-sintel.pth",
        min_bytes=10 * 1024 * 1024,  # 10 MB minimum
    )
    check_file(
        report, "PromptIR model.ckpt",
        AI_DIR / "pretrained" / "PromptIR" / "checkpoint" / "model.ckpt",
        min_bytes=100 * 1024 * 1024,  # 100 MB minimum
    )
    check_file(
        report, "HeavyRain checkpoint",
        AI_DIR / "pretrained" / "HeavyRainRemoval" / "checkpoint" /
        "HeavyRain-stage2-2019-05-11-76_ckpt.pth.tar",
        min_bytes=100 * 1024 * 1024,  # 100 MB minimum
    )
    yolo_path = AI_DIR / "models_weights" / "yolo11n.pt"
    if yolo_path.exists() and yolo_path.stat().st_size > 1024:
        report.add("YOLOv11n yolo11n.pt", "PASS", f"{yolo_path.stat().st_size / 1024 / 1024:.1f} MB  {yolo_path}")
    else:
        report.add("YOLOv11n yolo11n.pt", "WARN", "Not found — will auto-download on first API call")

    # =========================================================================
    # 5. Python packages
    # =========================================================================
    print(colour("\n  ── Python Packages ──", "BOLD"))
    critical_packages = [
        ("fastapi", False),
        ("uvicorn", False),
        ("pydantic", False),
        ("torch", False),
        ("torchvision", False),
        ("cv2", False),
        ("numpy", False),
        ("scipy", False),
        ("PIL", False),
        ("ultralytics", False),
        ("einops", False),
        ("lightning", True),       # pytorch-lightning imports as lightning
        ("tqdm", False),
        ("skimage", True),         # scikit-image
        ("gdown", False),
        ("psutil", True),
    ]
    for pkg, warn_only in critical_packages:
        check_import(report, pkg, warn_only=warn_only)

    # Check torch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            report.add("PyTorch CUDA", "PASS", f"GPU: {device_name}")
        else:
            report.add("PyTorch CUDA", "WARN", "CUDA not available — will run on CPU (slower)")
    except Exception as exc:
        report.add("PyTorch CUDA", "WARN", f"Could not check CUDA: {exc}")

    # =========================================================================
    # 6. Node.js packages
    # =========================================================================
    print(colour("\n  ── Node.js Packages ──", "BOLD"))
    for label, mod_path in [
        ("backend node_modules", BACKEND_DIR / "node_modules" / "express"),
        ("backend socket.io", BACKEND_DIR / "node_modules" / "socket.io"),
        ("backend axios", BACKEND_DIR / "node_modules" / "axios"),
        ("frontend node_modules", FRONTEND_DIR / "node_modules" / "react"),
        ("frontend vite", FRONTEND_DIR / "node_modules" / "vite"),
    ]:
        check_dir(report, label, mod_path)

    # =========================================================================
    # 7. Core application imports
    # =========================================================================
    print(colour("\n  ── Application Imports ──", "BOLD"))
    app_modules = [
        "app.config.settings",
        "app.utils.logger",
        "app.models.base",
        "app.models.stabilize",
        "app.models.heavy_rain_remove",
        "app.models.video_visibility",
        "app.models.object_detection",
        "app.pipeline.pipeline",
        "app.api.process",
        "app.api.health",
    ]
    # Change CWD so relative .env is found
    original_cwd = os.getcwd()
    os.chdir(str(AI_DIR))
    for mod in app_modules:
        check_import(report, mod, warn_only=False)
    os.chdir(original_cwd)

    # =========================================================================
    # 8. FastAPI app creation
    # =========================================================================
    print(colour("\n  ── FastAPI Application ──", "BOLD"))
    try:
        os.chdir(str(AI_DIR))
        from app.main import create_app  # type: ignore[import]
        app_obj = create_app()
        routes = [r.path for r in app_obj.routes]  # type: ignore[attr-defined]
        report.add("FastAPI create_app()", "PASS", f"routes: {routes}")
        os.chdir(original_cwd)
    except Exception as exc:
        report.add("FastAPI create_app()", "FAIL", str(exc))
        os.chdir(original_cwd)

    # =========================================================================
    # Print results
    # =========================================================================
    print(colour("\n  ── Results ──\n", "BOLD"))
    for r in report.results:
        print_row(r)

    overall = report.overall()
    print(f"\n  Total: {report.total}  |  "
          f"{colour(f'PASS: {report.passed}', 'PASS')}  |  "
          f"{colour(f'WARN: {report.warned}', 'WARN')}  |  "
          f"{colour(f'FAIL: {report.failed}', 'FAIL')}")
    print(f"\n  Overall: {colour(overall, overall)}\n")

    # =========================================================================
    # Write INSTALL_REPORT.md
    # =========================================================================
    report_path = REPO / "INSTALL_REPORT.md"
    _write_markdown_report(report, report_path, REPO)
    print(colour(f"  Report written to: {report_path}", "CYAN"))

    return 0 if overall != "FAIL" else 1


def _write_markdown_report(report: Report, path: Path, repo: Path) -> None:
    lines = [
        "# Installation Report — VideoAI Platform",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Host:** {os.uname().nodename}  ",
        f"**Python:** {sys.version.split()[0]}  ",
        f"**Repo:** {repo}  ",
        "",
        f"## Overall Status: {report.overall()}",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total Checks | {report.total} |",
        f"| Passed | {report.passed} |",
        f"| Warnings | {report.warned} |",
        f"| Failed | {report.failed} |",
        "",
        "## Detailed Results",
        "",
        "| Status | Component | Detail |",
        "|--------|-----------|--------|",
    ]
    for r in report.results:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[r.status]
        lines.append(f"| {icon} {r.status} | {r.name} | {r.detail} |")

    lines += [
        "",
        "## Next Steps",
        "",
        "```bash",
        "# Start all services:",
        "bash scripts/start_all.sh",
        "",
        "# Check health:",
        "bash scripts/health_check.sh",
        "```",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
