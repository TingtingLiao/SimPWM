"""Clip-level proxy metrics of DROID proxy runs over the first --clip-frames recorded steps, against the shared
real-video masks of real_dyn.py (Table 2, the paper's Metrics protocol; evaluation only).

  arm_iou    robot IoU at steps 0, 8, ..., N-1 (arm_iou.score with the run's camera on the cached SAM 3 instances)
  rgb_l1/l2  RGB proxy (conditions/ref_rgb_1280x704.mp4) against the real video at 640x352, steps 1..N-1, only over the
             moving region: the real robot and grasped objects (real_dyn) united with the proxy's robot and held objects
             (proxy_cat.npz), magenta (unknown) pixels excluded; per step, then the mean over steps
  depth_l2   depth proxy (conditions/ref_depth_1280x704.mp4, decoded to metres) against stereo depth at DEPTH_STEPS
             (droid/stereo svoseq + FoundationStereo) at 1280x704, over the pixels where both lie in 0.2-3 m and either
             the real or the proxy depth moved more than CHANGE from step 0 (real step 0: the first-frame stereo depth)

Stages:
  run      one JSON line per episode of RUN -> OUT_JSONL
  summary  Table 2's automatic rows: per run (--row NAME=RUN, in order), over the episodes every row has, the mean and
           a bootstrap 95% interval [mean, lo, hi, n] (2000 resamples, one numpy default_rng(0) stream) of arm IoU,
           RGB L1 / L2, depth L2 / L1 (METRICS_DIR/<NAME>_*.jsonl) and of the grasp_iou.py results in each run
           (--grasp-name): grasp success (same + none over the decided episodes), object IoU (all decided grasps;
           same object only), end error; plus the outcome counts
usage: python -m simpwm.eval.proxy_metrics run OUT_JSONL RUN REAL_DYN_DIR --ids IDS_TXT [--shard T N] [--clip-frames N]
       python -m simpwm.eval.proxy_metrics summary OUT_JSON METRICS_DIR --ids IDS_TXT --row NAME=RUN [...] [--grasp-name F]
"""

import glob
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from simpwm.core.state import C_OBJ0, is_robot
from simpwm.eval import arm_iou, grasp_iou, real_dyn
from simpwm.eval.common import (
    STEREO_DIR,
    STEREO_SEQ_DIR,
    episode_dir,
    magenta,
    parser,
    read_frames,
    read_ids,
    real_frames,
    states,
)
from simpwm.video.dataset import D_FAR, D_NEAR, decode_depth

CLIP_FRAMES = 97
WH, DWH = (640, 352), (1280, 704)  # RGB / depth comparison sizes
DEPTH_STEPS = [24, 48, 72, 96]
CHANGE = 0.02  # m: "moved" depth change from step 0
RGB, DEPTH = "conditions/ref_rgb_1280x704.mp4", "conditions/ref_depth_1280x704.mp4"
GRASP_NAME = "grasp_iou.json"


def rgb_errors(px, real, dyn, proxy_mask, T):
    """Per-step RGB L1 / L2 of proxy frames px against real frames over the moving region: dyn[t - 1] (real) united
    with proxy_mask(t), magenta excluded; steps with fewer than 50 such pixels are skipped."""
    l1, l2 = [], []
    for t in range(1, min(T, len(real), len(px))):
        pm = proxy_mask(t)
        p = px[t]
        m = (dyn[t - 1] | pm) & ~magenta(p)
        if m.sum() < 50:
            continue
        e = p[m].astype(np.float32) / 255 - real[t][m].astype(np.float32) / 255
        l1.append(float(np.abs(e).mean()))
        l2.append(float((e ** 2).mean()))
    return l1, l2


def stereo_depth(f):
    """A stereo depth map (m) resized (nearest) to DWH."""
    return cv2.resize(np.load(f).astype(np.float32), DWH, interpolation=cv2.INTER_NEAREST)


def decoded_depth(path, steps):
    """{t: metric depth at DWH} of a depth-proxy video at step 0 and at `steps`."""
    r = imageio.get_reader(path)
    out = {t: decode_depth(cv2.resize(f[..., 0], DWH, interpolation=cv2.INTER_NEAREST)) for t, f in enumerate(r)
           if t == 0 or t in steps}
    r.close()
    return out


def valid_depth(p):
    return np.isfinite(p) & (p >= D_NEAR) & (p <= D_FAR)


def change_mask(p, p0, gt, g0):
    """Pixels where real (gt) and proxy (p) depth both lie in D_NEAR-D_FAR and either moved more than CHANGE from its
    step 0 (g0, p0)."""
    ok = np.isfinite(gt) & np.isfinite(p) & (gt >= D_NEAR) & (gt <= D_FAR) & (p >= D_NEAR) & (p <= D_FAR)
    moved = (np.isfinite(g0) & (np.abs(gt - g0) > CHANGE)) | (np.isfinite(p0) & (np.abs(p - p0) > CHANGE))
    return ok & moved


def episode(run, real_dir, idx, emb, n=CLIP_FRAMES, stereo_dir=STEREO_DIR, stereo_seq_dir=STEREO_SEQ_DIR):
    d = episode_dir(run, idx)
    q, g = states(idx, n)
    T = min(n, len(q))
    res = dict(idx=idx)
    inst, dyn, _ = real_dyn.load(real_dir, idx, T, max(8, n // 15))
    # arm IoU on the cached instances
    cam, yaw = arm_iou.run_camera(d)
    res["arm_iou"] = arm_iou.summary(arm_iou.score(emb, cam, yaw, inst, q, g))["iou"]
    # dynamic-region RGB difference
    cat = np.load(d / "proxy_cat.npz")["cat"]
    held = sorted(grasp_iou.held_within(json.loads((d / "info.json").read_text()), T))

    def proxy_mask(t):
        c = cat[min(t, len(cat) - 1)]
        pm = is_robot(c) | np.isin(c, [C_OBJ0 + k for k in held])
        return cv2.resize(pm.astype(np.uint8), WH, interpolation=cv2.INTER_NEAREST) > 0

    l1, l2 = rgb_errors(read_frames(d / RGB, T, WH), real_frames(idx, T, WH), dyn, proxy_mask, T)
    res.update(rgb_l1=float(np.mean(l1)) if l1 else None, rgb_l2=float(np.mean(l2)) if l2 else None, rgb_n=len(l1))
    # change-masked depth error
    f0 = Path(stereo_dir) / f"{idx}_ext1.npy"
    if f0.exists():
        g0 = stereo_depth(f0)
        pd = decoded_depth(d / DEPTH, DEPTH_STEPS)
        dl2, dl1 = [], []
        for t in DEPTH_STEPS:
            ft = Path(stereo_seq_dir) / f"{idx}_ext1_t{t}.npy"
            if t >= T or t not in pd or not ft.exists():
                continue
            gt = stereo_depth(ft)
            p = pd[t]
            m = change_mask(p, pd[0], gt, g0)
            if m.sum() < 100:
                continue
            e = p[m] - gt[m]
            dl2.append(float((e ** 2).mean()))
            dl1.append(float(np.abs(e).mean()))
        res.update(depth_l2=float(np.mean(dl2)) if dl2 else None, depth_l1=float(np.mean(dl1)) if dl1 else None,
                   depth_n=len(dl2))
    return res


def _load(f):
    try:
        return json.loads(Path(f).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def summary(out, metrics_dir, rows, ids, grasp_name=GRASP_NAME):
    """Table 2's automatic metrics per row (see the module docstring) -> out {n_common, variants: {name: ...}}."""
    rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray([a for a in x if a is not None], float)
        if not len(x):
            return None
        b = rng.choice(x, (2000, len(x))).mean(1)
        return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(x)]

    def grasp_stats(gs):
        dec = [g for g in gs if g and g["grasp"] != "undecided"]
        return dict(grasp_success=ci([float(g["grasp"] in ("same", "none")) for g in dec]),
                    object_iou_all=ci([g["object"] for g in dec if g["grasp"] != "none"]),
                    object_iou_same=ci([g["object"] for g in dec if g["grasp"] == "same"]),
                    end_err=ci([g.get("end_err") for g in dec if g["grasp"] == "same"]),
                    counts={k: sum(g["grasp"] == k for g in dec)
                            for k in ("same", "none", "diff", "nested", "miss", "fp")})

    m = {v: {r["idx"]: r for f in glob.glob(str(Path(metrics_dir) / f"{v}_*.jsonl")) for r in map(json.loads, open(f))}
         for v in rows}
    g = {v: {i: _load(episode_dir(run, i) / grasp_name) for i in ids} for v, run in rows.items()}
    common = [i for i in ids if all(i in m[v] and g[v][i] for v in rows)]
    res = dict(n_common=len(common), variants={})
    for v in rows:
        e = [m[v][i] for i in common]
        res["variants"][v] = dict(arm_iou=ci([r["arm_iou"] for r in e]),
                                  rgb_l1=ci([r["rgb_l1"] for r in e]), rgb_l2=ci([r["rgb_l2"] for r in e]),
                                  depth_l2=ci([r.get("depth_l2") for r in e]),
                                  depth_l1=ci([r.get("depth_l1") for r in e]),
                                  **grasp_stats([g[v][i] for i in common]))
    Path(out).write_text(json.dumps(res, indent=1))
    print("test episodes in every row:", len(common))
    for v, s in res["variants"].items():
        print(f"{v:5s}", " ".join(f"{k}={s[k][0]:.4f}" for k in s if k != "counts" and s[k]), s["counts"])
    return res


def main():
    p = parser(__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    r = sub.add_parser("run", help="per-episode metrics of one run")
    r.add_argument("out", type=Path, help="output JSONL")
    r.add_argument("run", type=Path, help="proxy run directory")
    r.add_argument("real", type=Path, help="real_dyn.py output directory")
    r.add_argument("--ids", required=True, help="episode id list")
    r.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("T", "N"), help="T-th of N shards")
    r.add_argument("--clip-frames", type=int, default=CLIP_FRAMES, help="clip length in steps (default %(default)s)")
    r.add_argument("--stereo-dir", type=Path, default=STEREO_DIR, help="first-frame stereo depth (default %(default)s)")
    r.add_argument("--stereo-seq-dir", type=Path, default=STEREO_SEQ_DIR,
                   help="later-step stereo depth (default %(default)s)")
    s = sub.add_parser("summary", help="Table 2 automatic rows")
    s.add_argument("out", type=Path, help="output JSON")
    s.add_argument("metrics", type=Path, help="directory of the run stage's <NAME>_*.jsonl")
    s.add_argument("--ids", required=True, help="test episode id list")
    s.add_argument("--row", action="append", required=True, metavar="NAME=RUN", help="row name and its proxy run")
    s.add_argument("--grasp-name", default=GRASP_NAME, help="grasp_iou.py output file name (default %(default)s)")
    a = p.parse_args()
    if a.stage == "summary":
        summary(a.out, a.metrics, dict(x.split("=", 1) for x in a.row), read_ids(a.ids), a.grasp_name)
        return
    from simpwm.core.embodiment import Embodiment

    emb = Embodiment("franka_robotiq")
    with open(a.out, "w") as f:
        for idx in read_ids(a.ids, *a.shard):
            if not (a.real / f"{idx}.npz").exists():
                continue
            try:
                res = episode(a.run, a.real, idx, emb, a.clip_frames, a.stereo_dir, a.stereo_seq_dir)
                f.write(json.dumps(res) + "\n")
                f.flush()
            except Exception as e:  # one bad episode must not stop the shard
                print("FAILED", idx, repr(e), flush=True)


if __name__ == "__main__":
    main()
