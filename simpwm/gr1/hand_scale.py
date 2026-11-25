"""Fit of the per-hand closure scale (gr1/run.py --hand-scale) on training candidates with a session camera.

Per episode, scored on 13 clip frames (every 8th of the candidate's 97-frame stride-2 clip) at 500x800 with the robot
IoU of gr1/session_calib.Scorer:
  S   the session camera (and its dt) as is
  E5  per-episode fit of the 9 camera parameters + a closure scale per hand (0.5-2.5) on the 6 hand values
  E6  E5 + arm joint offsets (7 per arm, |offset| <= 0.12 rad), fitted jointly
The fitted scales (fs: left, right) are summarised over the episodes; the paper uses 0.89:1.28 (the Fourier-hand
mapping under-closes the right hand).

usage: python -m simpwm.gr1.hand_scale OUT_DIR [--episodes 16] [--candidates JSON] [--cameras DIR]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from simpwm.core.camera import Camera
from simpwm.core.raster_gpu import GpuRaster
from simpwm.gr1 import calib as calib_gr1
from simpwm.gr1 import data as gr1_data
from simpwm.gr1.embodiment import SL, GR1Embodiment
from simpwm.gr1.session_calib import nelder_mead, robot_mask_t
from simpwm.paths import RUNS

ARM = list(range(SL["left"].start, SL["left"].stop)) + list(range(SL["right"].start, SL["right"].stop))


def modified_state(q, fs, off):
    q = q.copy()
    q[SL["left_hand"]] *= fs[0]
    q[SL["right_hand"]] *= fs[1]
    q[ARM] += off
    return q


class HandScorer:
    def __init__(self):
        self.emb = GR1Embodiment(verbose=False)
        self.gr = GpuRaster(self.emb)

    def score(self, calib, qs, ins, fs=(1, 1), off=np.zeros(14), hw=gr1_data.WORK_HW):
        """Pooled robot IoU over the states qs (instances ins, at WORK_HW) with hand scales fs and arm offsets off."""
        h, w = hw
        K = calib_gr1.intrinsics(calib, h, w)
        T_hc = np.asarray(calib["T_hc"])
        inter = union = 0.0
        for q, m_ in zip(qs, ins):
            qt = modified_state(q, fs, off)
            self.emb.set_state(qt)
            cam = Camera(K, self.emb.head_pose() @ T_hc, h, w, tuple(calib["dist"]))
            r = self.gr.parts(self.gr.world_verts([(qt, None)]), cam)[0] >= 0
            if m_ is not None and m_.shape[-2:] != r.shape:
                m_ = torch.nn.functional.interpolate(m_[:, None].float(), size=r.shape)[:, 0] > 0
            m = robot_mask_t(m_, r)
            inter, union = inter + float((m & r).sum()), union + float((m | r).sum())
        return inter / max(union, 1)


def fit_episode(hs, c, cam_dir):
    t0 = time.time()
    H, W = gr1_data.WORK_HW
    calib, src = gr1_data.camera_calib(c["ep"], cam_dir)
    q_all = gr1_data.states(c["ep"])
    ts = np.minimum(gr1_data.clip_ts(c["start"], 97)[::8], len(q_all) - 1)
    fr = gr1_data.read_frames(c["ep"], ts, gr1_data.WORK_HW)
    ts = [t for t in ts if t in fr]
    qs = [q_all[min(t + calib.get("dt", 0), len(q_all) - 1)] for t in ts]
    ins = []
    for t in ts:
        m = calib_gr1.robot_instances(fr[t])
        ins.append(torch.from_numpy(np.stack(m)).to(hs.gr.dev) if m else None)
    s0 = hs.score(calib, qs, ins)
    lo = (H // 2, W // 2)

    def f5(x):
        if not (-0.6 < x[7] < 0.4 and -0.3 < x[8] < 0.3 and 0.5 < x[9] < 2.5 and 0.5 < x[10] < 2.5):
            return 1.0
        return -hs.score(calib_gr1.apply(calib, x[:9]), qs, ins, x[9:11], hw=lo)

    st5 = np.r_[0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.03, 0.03, 0.02, 0.2, 0.2]
    x5 = nelder_mead(f5, np.r_[np.zeros(9), 1.0, 1.0], st5, 400)
    s5 = hs.score(calib_gr1.apply(calib, x5[:9]), qs, ins, x5[9:11])

    def f6(x):
        if (not (-0.6 < x[7] < 0.4 and -0.3 < x[8] < 0.3 and 0.5 < x[9] < 2.5 and 0.5 < x[10] < 2.5)
                or np.abs(x[11:]).max() > 0.12):
            return 1.0
        return -hs.score(calib_gr1.apply(calib, x[:9]), qs, ins, x[9:11], x[11:], hw=lo)

    x6 = nelder_mead(f6, np.r_[x5, np.zeros(14)], np.r_[st5 / 2, np.full(14, 0.04)], 1200)
    s6 = hs.score(calib_gr1.apply(calib, x6[:9]), qs, ins, x6[9:11], x6[11:])
    return dict(ep=c["ep"], session=c["session"], src=src, S=s0, E5=s5, E6=s6, fs=x5[9:11].tolist(),
                off=np.round(x6[11:], 3).tolist(), seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="output dir (hand_scale.json)")
    ap.add_argument("--episodes", type=int, default=16, help="number of candidates to fit")
    ap.add_argument("--candidates", type=Path, default=RUNS / "gr1/candidates.json", help="gr1/select.py output")
    ap.add_argument("--cameras", type=Path, default=gr1_data.SESSION_CAMERAS, help="per-session camera dir")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in a.cameras.glob("*.json")}
    items = [c for c in json.loads(a.candidates.read_text()) if c["session"] in done][:a.episodes]
    hs = HandScorer()
    res = []
    for c in items:
        r = fit_episode(hs, c, a.cameras)
        res.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    (a.out / "hand_scale.json").write_text(json.dumps(res, indent=1))
    for k in ("S", "E5", "E6"):
        print(k, "mean IoU", round(float(np.mean([r[k] for r in res])), 3), flush=True)
    fs = np.array([r["fs"] for r in res])
    print("hand scale (left, right): mean", np.round(fs.mean(0), 3).tolist(), "median",
          np.round(np.median(fs, 0), 3).tolist(), flush=True)


if __name__ == "__main__":
    main()
