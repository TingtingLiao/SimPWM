"""Locations of everything outside the Python package. Each can be overridden with the environment variable named
next to it; the defaults follow the repository layout described in the README."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESOURCES = REPO / "resources"  # files shipped with the code: splits, camera calibrations, robot configs


def _env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default)


ASSETS = _env("SIMPWM_ASSETS", REPO / "assets")  # robot models: MuJoCo Menagerie, GR1 URDF
DATA = _env("SIMPWM_DATA", REPO / "data")  # datasets and files derived from them
DEPS = _env("SIMPWM_DEPS", REPO / "deps")  # external code: LTX-2, FoundationStereo
RUNS = _env("SIMPWM_RUNS", REPO / "runs")  # default output root

DROID_ROOT = _env("DROID_ROOT", DATA / "droid")  # RLDS 1.0.1 release (optional)
DROID_RAW = _env("DROID_RAW", DATA / "droid_raw")  # per-episode raw release directories
GR1_ROOT = _env("GR1_ROOT", DATA / "gr1/GR1_robot")  # GR00T-Teleop-GR1, subset GR1_robot (LeRobot v2)
H3_MODEL = _env("H3_DIFFUSERS", REPO / "models/MiniMax-H3-diffusers-ref2va")  # MiniMax-H3 Ref2VA, diffusers layout
SAM3_PATH = os.environ.get("SAM3_PATH", "facebook/sam3")  # HF repo id or local snapshot of SAM 3
