"""Per-session refinement of the GR1 head camera with SAM 3 robot masks and the GPU rasterizer.

For each recording session: its episodes from SESSIONS_JSON (training episodes only, never DreamDojo test episodes),
FRAMES frames each spread over a 97-frame stride-2 clip, SAM 3 robot instances at 500x800 (gr1/calib.py). The 9
camera parameters of calib.apply (rotation vector, translation, log focal, k1, k2) are fitted by Nelder-Mead, then the
state/video time offset dt (raw frames, -6..6; the state at t + dt drives frame t) by grid search, and the 9
parameters again with it, maximising the pooled robot IoU (fit at 250x400, scored at 500x800): per frame, the robot
mask of the instances touching the rasterized FK render against the render, pixel-weighted over all frames. Start =
the stored per-session calibration if any, else the unit's, else the dataset-level one. The refit is kept only if it
improves the IoU; OUT/<session>.json is written either way (source "session_refit" or "start:<start>").

usage: python -m simpwm.gr1.session_calib --out OUT [--sessions SESSIONS_JSON] [--shard K --num-shards N]
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import minimize

from simpwm.core.camera import Camera
from simpwm.core.raster_gpu import GpuRaster
from simpwm.gr1 import calib as calib_gr1
from simpwm.gr1 import data as gr1_data
from simpwm.gr1.embodiment import GR1Embodiment
from simpwm.paths import RUNS

CLIP_SPAN = 195  # raw frames covered by a 97-frame stride-2 clip (with margin)


def start_calib(ep):
    base = calib_gr1.load_calib()
    s, u = gr1_data.session(ep), gr1_data.unit(ep)
    if s in base.get("per_session", {}):
        return base["per_session"][s], "per_session"
    if u in base.get("per_robot", {}):
        return base["per_robot"][u], "per_robot"
    return {k: base[k] for k in calib_gr1.CALIB_KEYS}, "dataset"


def robot_mask_t(ins, r):
    """Torch version of calib.robot_mask: union of the instances `ins` (n, H, W) touching the render `r` (H, W)."""
    if ins is None:
        return torch.zeros_like(r)
    touch = (ins & r).flatten(1).sum(1) >= calib_gr1.TOUCH * ins.flatten(1).sum(1).clamp(min=1)
    return ins[touch].any(0) if touch.any() else torch.zeros_like(r)


class Scorer:
    """Pooled robot IoU of a calibration over fixed frames (states, head poses, SAM 3 instances) on the GPU."""

    def __init__(self, emb, gr, items, HW):
        """items: [(q44, [instance mask, ...])]; HW: the scoring size (instances are resized to it)."""
        self.gr, self.HW = gr, HW
        self.heads, self.V, self.ins = [], [], []
        for q, inst in items:
            emb.set_state(q)
            self.heads.append(emb.head_pose())
            self.V.append(gr.world_verts([(q, None)]))
            m = [torch.from_numpy(np.ascontiguousarray(
                cv2.resize(x.astype(np.uint8), HW[::-1], interpolation=cv2.INTER_NEAREST) > 0)) for x in inst]
            self.ins.append(torch.stack(m).to(gr.dev) if m else None)

    def score(self, calib):
        H, W = self.HW
        K = calib_gr1.intrinsics(calib, H, W)
        T_hc = np.asarray(calib["T_hc"])
        inter = union = 0.0
        for V, head, ins in zip(self.V, self.heads, self.ins):
            r = self.gr.parts(V, Camera(K, head @ T_hc, H, W, tuple(calib["dist"])))[0] >= 0
            m = robot_mask_t(ins, r)
            inter, union = inter + float((m & r).sum()), union + float((m | r).sum())
        return inter / max(union, 1.0)


def nelder_mead(f, x0, step, maxfev):
    """Two Nelder-Mead rounds from axis simplices of size `step`, then `step` / 4; a round is kept if not worse."""
    best = x0
    for s in (step, step / 4):
        simplex = np.vstack([best] + [best + np.eye(len(best))[k] * s[k] for k in range(len(best))])
        r = minimize(f, best, method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, maxfev=maxfev, xatol=1e-4, fatol=1e-4))
        best = r.x if r.fun <= f(best) else best
    return best


def camera_objective(calib, scorer):
    """-IoU of calib.apply(calib, x), 1 outside the distortion's valid range."""

    def f(x):
        if not (-0.6 < x[7] < 0.4 and -0.3 < x[8] < 0.3):
            return 1.0
        return -scorer.score(calib_gr1.apply(calib, x))

    return f


def fit_session(emb, gr, sess, eps, n_frames=12, log=print):
    calib0, src = start_calib(eps[0])
    raw = []
    for ep in eps:
        q = gr1_data.states(ep)
        n = min(len(q), CLIP_SPAN)
        ts = np.unique(np.linspace(0, n - 1, n_frames).round().astype(int))
        fr = gr1_data.read_frames(ep, ts, gr1_data.WORK_HW)
        ts = [t for t in ts if t in fr]
        inst = [calib_gr1.robot_instances(fr[t]) for t in ts]
        raw += [(q, t, i) for t, i in zip(ts, inst)]
    lo_hw = (gr1_data.WORK_HW[0] // 2, gr1_data.WORK_HW[1] // 2)

    def scorers(dt):
        items = [(q[min(max(t + dt, 0), len(q) - 1)], i) for q, t, i in raw]
        return Scorer(emb, gr, items, gr1_data.WORK_HW), Scorer(emb, gr, items, lo_hw)

    step = np.r_[0.04, 0.04, 0.04, 0.02, 0.02, 0.02, 0.05, 0.05, 0.03]
    hi0, lo0 = scorers(0)
    iou0 = hi0.score(calib0)
    x = nelder_mead(camera_objective(calib0, lo0), np.zeros(9), step, 500)
    c1 = calib_gr1.apply(calib0, x)
    sc = {dt: scorers(dt)[1].score(c1) if dt else lo0.score(c1) for dt in range(-6, 7)}
    dt = max(sc, key=sc.get)
    if dt != 0:
        hi, lo = scorers(dt)
        x = nelder_mead(camera_objective(calib0, lo), x, step, 500)
    else:
        hi = hi0
    calib1 = calib_gr1.apply(calib0, x)
    iou1 = hi.score(calib1)
    keep = iou1 > iou0
    out = dict(session=sess, unit=gr1_data.unit(eps[0]), episodes=eps, n_frames=len(raw), start=src, iou_start=iou0,
               iou_refit=iou1, x=x.tolist(), dt=int(dt) if keep else 0, dt_scores={str(k): v for k, v in sc.items()},
               source="session_refit" if keep else f"start:{src}",
               calib={k: (calib1 if keep else calib0)[k] for k in calib_gr1.CALIB_KEYS})
    log(f"{sess}: {len(eps)} eps, {len(raw)} frames, start {src} iou {iou0:.3f} -> {iou1:.3f} dt {dt} "
        f"({'kept' if keep else 'rejected'})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=RUNS / "gr1/session_cameras", help="output dir (<session>.json)")
    ap.add_argument("--sessions", type=Path, default=gr1_data.SESSIONS, help="{session: [episode, ...]} json")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard of the sorted sessions")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--frames", type=int, default=12, help="frames per episode")
    ap.add_argument("--reverse", action="store_true", help="walk the shard from its end (a second worker on it)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sessions = json.loads(a.sessions.read_text())
    todo = sorted(sessions)[a.shard::a.num_shards]
    if a.reverse:
        todo = todo[::-1]
    emb = GR1Embodiment(verbose=False)
    gr = GpuRaster(emb)
    for s in todo:
        f = a.out / f"{s}.json"
        if f.exists():
            continue
        t0 = time.time()
        try:
            r = fit_session(emb, gr, s, sessions[s], a.frames, log=lambda m: print(m, flush=True))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print("FAILED", s, e, flush=True)
            continue
        r["seconds"] = time.time() - t0
        f.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
