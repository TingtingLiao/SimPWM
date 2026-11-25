"""GR1 data: episode index, DreamDojo test split, states, frames, captions and head cameras.

Source: nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1, subset GR1_robot (LeRobot v2; GR1T1 with Fourier hands, ego/head
camera 1280x800 at 20 fps) under simpwm.paths.GR1_ROOT; states are read from its parquet files (pyarrow).

Clips: N raw frames at stride 2 (10 fps, DreamDojo's GR1 sampling) from a raw start frame. The real clip and first
frame are stored stretched 1280x800 -> 1280x768 (SQ_WH); LTX trains and generates at 800x480 (GEN_WH, both sides
multiples of 32; the trainer stretches 1280x768 to it, same aspect). The robot is rendered at 500x800 (WORK_HW, the
recording's 16:10) and the RGB proxy is stored stretched to 800x480 (REF_WH = GEN_WH: reference downscale factor 1).

Test split: DreamDojo's GR1_robot data_split=test (the last 5% of its 49-frame stride-2 windows, episodes 21705-22208);
those episodes are never used for training or camera calibration.
Caption: TEMPLATE, with the episode task stripped of its "locked waist: " / "unlocked waist: " prefix.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from simpwm.gr1 import calib as calib_gr1
from simpwm.paths import GR1_ROOT, RESOURCES

TEST_SPLIT = RESOURCES / "gr1/dreamdojo_test_split.json"  # {test_episodes, samples: [[episode, base frame], ...]}
TEST_WINDOWS = RESOURCES / "gr1/test_windows.json"  # [{k, ep, base, start}]: the 100 DreamDojo test windows
SESSIONS = RESOURCES / "gr1/sessions.json"  # {session: [episodes]} for the per-session camera refit
SESSION_CAMERAS = RESOURCES / "gr1/session_cameras"  # <session>.json written by gr1/session_calib.py

SQ_WH = (1280, 768)  # stored real clip + first frame
GEN_WH = (800, 480)  # LTX training / generation size
REF_WH = GEN_WH  # RGB proxy (reference) size
WORK_HW = (500, 800)  # render / perception size (16:10 like the recording; stretched to REF_WH when written)
FPS_SRC = 20.0
STRIDE = 2
CLIP_FRAMES = {"train": 97, "eval": 49}  # training clip / DreamDojo test window (frames at STRIDE)
TEMPLATE = ("A head-mounted camera films a Fourier GR1 humanoid robot with two dexterous hands working at a table. "
            "The robot performs the task: {instr}. Realistic lighting, natural colors.")


@lru_cache(None)
def episodes():
    return {e["episode_index"]: e for e in map(json.loads, open(GR1_ROOT / "meta/episodes.jsonl"))}


@lru_cache(None)
def test_episodes():
    return frozenset(json.loads(TEST_SPLIT.read_text())["test_episodes"])


def session(ep):
    """Recording session = trajectory_id without its counter (GR1 unit + start time)."""
    return episodes()[ep]["trajectory_id"].rsplit("-", 1)[0]


def unit(ep):
    return episodes()[ep]["trajectory_id"].split("-")[-2]


def instr(ep):
    t = episodes()[ep]["tasks"][0]
    for p in ("locked waist:", "unlocked waist:"):
        if t.startswith(p):
            t = t[len(p):]
    return t.strip()


def caption(ep):
    return TEMPLATE.format(instr=instr(ep))


def states(ep, key="observation.state"):
    """(T, 44) state vectors of episode ep (radians, see gr1/embodiment.py)."""
    import pyarrow.parquet as pq

    t = pq.read_table(GR1_ROOT / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet", columns=[key])
    return np.array(t.column(key).to_pylist(), np.float64)


def video_path(ep):
    return GR1_ROOT / f"videos/chunk-{ep // 1000:03d}/observation.images.ego_view_freq20/episode_{ep:06d}.mp4"


def read_frames(ep, ts, HW=None):
    """{t: frame (H, W, 3) uint8 RGB} for the raw frame indices ts (decoding stops after max(ts)); HW None: 800x1280."""
    import av

    need, out = set(int(t) for t in ts), {}
    with av.open(str(video_path(ep))) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in need:
                a = f.to_ndarray(format="rgb24")
                out[i] = a if HW is None or a.shape[:2] == HW else cv2.resize(a, HW[::-1], interpolation=cv2.INTER_AREA)
            if i >= max(need):
                break
    return out


def clip_ts(start, n=97, stride=STRIDE):
    return start + stride * np.arange(n)


def camera_calib(ep, cam_dir=SESSION_CAMERAS):
    """(calibration dict, source) of episode ep: the session's refit (gr1/session_calib.py) in cam_dir when there is
    one, else the stored per-session / per-unit / dataset calibration (gr1/calib.py). Its "dt" (raw frames, default 0)
    is the state/video time offset: the state at t + dt drives frame t."""
    s = session(ep)
    f = Path(cam_dir) / f"{s}.json"
    if f.exists():
        r = json.loads(f.read_text())
        return dict(r["calib"], dt=int(r.get("dt", 0))), r.get("source", "session_refit")
    base = calib_gr1.load_calib()
    if s in base.get("per_session", {}):
        return base["per_session"][s], "per_session"
    if unit(ep) in base.get("per_robot", {}):
        return base["per_robot"][unit(ep)], "per_robot"
    return base, "dataset"


def cameras(emb, q, calib, HW=WORK_HW):
    """Per-frame head cameras for the states q."""
    return [calib_gr1.head_camera(emb, qt, calib, *HW) for qt in q]
