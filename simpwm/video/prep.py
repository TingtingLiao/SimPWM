"""Inputs of the zero-training proxy -> video baseline (pretrained LTX IC-LoRA pipeline), one folder per DROID episode.

From RUN/droid_XXXXX/conditions (droid/run.py output, 24 fps), in OUT/<idx>/ at 1024x576 (divisible by 64 for the
two-stage IC-LoRA pipeline), --clip-frames frames (8k + 1):
  depth.mp4    the proxy depth as a grayscale video (inverse depth, near = bright, per-episode 2-98% range)
  first.png    the real first frame (the image condition), from the raw ext1 MP4
  real.mp4     the real video on the same time axis (24 fps resample of the raw MP4), for side-by-side comparison
  prompt.txt   the caption of the task instruction

usage: python -m simpwm.video.prep RUN OUT IDX [IDX ...] [--clip-frames N]
"""

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from simpwm.droid import data as droid_data
from simpwm.droid.run import CWM_FPS, CWM_HW
from simpwm.video.common import parser, write_video
from simpwm.video.dataset import prompt, raw_frames

W, H = 1024, 576
CLIP_FRAMES = 121


def depth_video(cond, n=CLIP_FRAMES):
    d = np.stack([np.fromfile(cond / f"{k:06d}.depth.f32", np.float32).reshape(CWM_HW) for k in range(n)])
    inv = np.where(d > 0.05, 1.0 / np.maximum(d, 0.05), 0.0)
    lo, hi = np.percentile(inv[inv > 0], [2, 98])
    g = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    g = (g * 255).astype(np.uint8)
    return [cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2RGB) for f in g]


def build(run, out, idxs, clip_frames=CLIP_FRAMES):
    ok = [r for r in droid_data.manifest() if r["status"] == "ok"]
    for idx in idxs:
        src, dst = Path(run) / f"droid_{idx:05d}", Path(out) / str(idx)
        dst.mkdir(parents=True, exist_ok=True)
        ep = droid_data.load(ok[idx])
        T = len(ep["frames"])
        src_t = np.minimum(np.round(np.arange(clip_frames) * ep["fps"] / CWM_FPS).astype(int), T - 1)
        frames = raw_frames(idx, ep, set(src_t.tolist()) | {0})  # the raw 720p recording, never the 320x180 frames
        if frames is None:
            print("skip (no matching raw mp4)", idx, flush=True)
            continue
        real = [cv2.resize(frames[t], (W, H), interpolation=cv2.INTER_AREA) for t in src_t]
        imageio.imwrite(dst / "first.png", real[0])
        write_video(dst / "real.mp4", real, CWM_FPS, quality=8)
        write_video(dst / "depth.mp4", depth_video(src / "conditions", clip_frames), CWM_FPS, quality=8)
        (dst / "prompt.txt").write_text(prompt(ep["text"]))
        print("prep", idx, flush=True)


def main():
    p = parser(__doc__)
    p.add_argument("run", help="proxy run directory (droid/run.py output)")
    p.add_argument("out", help="output directory (one folder per episode)")
    p.add_argument("idxs", nargs="+", type=int, metavar="IDX", help="episode indices (@FILE reads them from a file)")
    p.add_argument("--clip-frames", type=int, default=CLIP_FRAMES, help="clip length in frames (default %(default)s)")
    a = p.parse_args()
    build(a.run, a.out, a.idxs, a.clip_frames)


if __name__ == "__main__":
    main()
