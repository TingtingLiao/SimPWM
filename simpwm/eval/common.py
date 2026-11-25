"""Helpers shared by the evaluation modules: DROID episodes (raw recording, logged states), proxy-run cameras, video
frames, id lists, task captions, bootstrap intervals and small mask utilities."""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

from simpwm import paths
from simpwm.core.camera import Camera
from simpwm.droid import data as droid_data
from simpwm.video.common import caption_task, parser, read_frames, read_jsonl  # noqa: F401  (parser: re-exported)

STEREO_DIR = paths.RUNS / "stereo/depth"  # first-frame FoundationStereo depth <idx>_ext1.npy (droid/stereo)
STEREO_SEQ_DIR = paths.RUNS / "stereo_seq/depth"  # later-step stereo depth <idx>_ext1_t<t>.npy (droid/stereo svoseq)


def raw_dir(idx) -> Path:
    """Raw-release directory of DROID episode idx (ext1.mp4, trajectory.h5)."""
    from simpwm.video.dataset import ep_raw_dir  # (lazy: video.dataset imports droid.run, which imports eval.tracks)

    return droid_data.DROID_RAW / ep_raw_dir(idx)


def states(idx, n=None):
    """Logged joint positions and gripper closure (observed) of episode idx, first n steps."""
    with h5py.File(raw_dir(idx) / "trajectory.h5", "r") as h:
        return h["observation/robot_state/joint_positions"][:n], h["observation/robot_state/gripper_position"][:n]


def real_frames(idx, n, wh) -> list:
    """The first n frames of the real ext1 recording, resized (INTER_AREA) to wh = (W, H)."""
    return read_frames(raw_dir(idx) / "ext1.mp4", n, wh)


def episode_dir(run, idx) -> Path:
    return Path(run) / f"droid_{idx:05d}"


def run_camera(info, H, W):
    """(Camera at H x W, gripper yaw in degrees) of a proxy run's info.json."""
    yaw = re.search(r"'gripper_yaw': (-?[\d.]+)", str(info.get("camera_calibration", "")))
    return Camera(info["K"], info["T_wc"], H, W, info.get("dist", (0.0, 0.0))), float(yaw.group(1)) if yaw else 90.0


def read_ids(path, shard=0, n=1) -> list[int]:
    """The shard-th of n interleaved shards of a whitespace-separated id list."""
    return [int(i) for i in Path(path).read_text().split()][shard::n]


def read_indices(path, want) -> dict:
    """{k: RGB frame} for the frame indices k in `want` (decoding stops after the last one)."""
    r = imageio.get_reader(str(path), "ffmpeg")
    out = {}
    for k, f in enumerate(r):
        if k in want:
            out[k] = f
        if k >= max(want):
            break
    r.close()
    return out


def read_steps(path, steps, size, stride=1) -> dict:
    """{step: frame resized (INTER_AREA) to size} of a video whose frame k is recorded step k * stride (steps that are
    not a multiple of the stride are left out)."""
    want = {s // stride: s for s in steps if s % stride == 0}
    return {want[k]: cv2.resize(f, size, interpolation=cv2.INTER_AREA) for k, f in read_indices(path, want).items()}


def centroid(m):
    """(x, y) mean of a mask's pixels, None if empty."""
    y, x = np.nonzero(m)
    return np.array([x.mean(), y.mean()]) if len(x) else None


def magenta(p):
    """The proxy's "unknown" pixels (MARK_RGB after video compression) of an RGB frame."""
    return (p[..., 0] > 200) & (p[..., 2] > 200) & (p[..., 1] < 80)


def task_texts(dataset_jsonl) -> dict:
    """{idx: task instruction} from the captions of a video dataset."""
    return {int(r["idx"]): caption_task(r["caption"]) for r in read_jsonl(dataset_jsonl)}


def load_dir(d) -> dict:
    """{int(stem): record} of the per-episode JSON files of a directory."""
    return {int(p.stem): json.loads(p.read_text()) for p in Path(d).glob("*.json")}


def ci(x, B=2000, seed=0):
    """Mean with a bootstrap 95% interval (B resamples of the values, numpy default_rng(seed))."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(mean=None, ci95=None, n=0)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return dict(mean=float(x.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(x))


def paired_bootstrap(a, b, rng, B=2000):
    """B bootstrap means of the paired differences b - a (resampled episodes)."""
    i = rng.integers(0, len(a), (B, len(a)))
    return (b[i] - a[i]).mean(1)


def each(ids, fn, done=None):
    """fn(idx) for every id not `done`; a failing episode is reported and does not stop the others."""
    for idx in ids:
        if done is not None and done(idx):
            continue
        try:
            fn(idx)
        except Exception:
            print("FAILED", idx, traceback.format_exc(), flush=True)


def grid(imgs, width=640, interpolation=cv2.INTER_AREA):
    """Two rows of four frames, each resized to `width` keeping its aspect."""
    th = [cv2.resize(i, (width, int(round(width * i.shape[0] / i.shape[1]))), interpolation=interpolation)
          for i in imgs]
    return np.concatenate([np.concatenate(th[:4], 1), np.concatenate(th[4:], 1)], 0)


def write_jpg(path, rgb, quality=90):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
