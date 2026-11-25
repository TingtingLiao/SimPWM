"""Robot (arm + gripper) alignment of a DROID proxy run over the LTX clip: the camera-alignment filter of the training
set and the robot IoU of the proxy metrics (Table 2). Objects are not scored.

  real robot   SAM 3 "robot arm" / "robot" instances (score >= 0.5) over the whole frame; the instances that touch the
               render are kept WHOLE (the render only picks instances, so a second robot elsewhere in view is ignored,
               and it never clips them)
  proxy robot  the FK arm + gripper rendered (core/raster_gpu.py) with the run's camera: droid/episode_calib.py's
               cam_refined.json when present, else info.json's K / T_wc / dist and calibrated gripper yaw
  per frame    intersection and union of the two (a frame where only one side shows a robot adds only to the union);
               the episode score pools them over the clip (pixel-weighted IoU)
Every EVERY-th recorded step of the clip (max(8, N // 15): 13 of 97 steps, 16 of 241) at 640x352 (the 1280x720
recording stretched as the 1280x704 proxy is). A metric that only keeps detections overlapping the render, clipped to a
band around it, would score a wrong camera ~0.6 on whatever lies under the render; this one does not.

Writes RUN/droid_XXXXX/<--out-name> {iou (pixel-weighted), iou_frames (per-frame mean), frac_good (frames with
IoU >= 0.6), robot_px, n_frames, frames {t: [intersection, union]}}; episodes with that file are skipped.
usage: python -m simpwm.eval.arm_iou RUN --ids IDS_TXT [--shard T N] [--clip-frames N] [--out-name arm_iou.json]
"""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from simpwm.core.camera import Camera
from simpwm.core.raster_gpu import PosedRobot
from simpwm.eval import common
from simpwm.eval.common import episode_dir, parser, raw_dir, read_ids, states  # noqa: F401  (states: public API)

N = 97  # clip length (recorded steps)
EVERY = max(8, N // 15)
WH = (640, 352)
OUT = "arm_iou.json"
PHRASES = ["robot arm", "robot"]
MIN_SCORE = 0.5
MIN_PX = 50  # smaller masks count as no robot (at 640x352)
TOUCH = 0.05  # an instance is ours if this fraction of it lies on the render


def run_camera(d):
    """(Camera at 320x180, gripper yaw) the proxy run in dir d used: cam_refined.json if present, else info.json's."""
    f = d / "cam_refined.json"
    if f.exists():
        c = json.loads(f.read_text())
        return Camera(np.array(c["K"]), np.array(c["T_wc"]), 180, 320, tuple(c["dist"])), c["gripper_yaw"]
    return common.run_camera(json.loads((d / "info.json").read_text()), 180, 320)


def instances(idx, ts, wh=WH):
    """{t: [SAM 3 robot instance masks at wh]} for the recorded frames ts (resized from 1280x720 to wh)."""
    from simpwm.perception import sam3

    out, ts = {}, set(ts)
    for t, im in zip(range(max(ts) + 1), imageio.get_reader(raw_dir(idx) / "ext1.mp4")):
        if t in ts:
            im = cv2.resize(im, wh, interpolation=cv2.INTER_AREA)
            out[t] = [i["mask"] for i in sam3.segment_concepts(im, PHRASES, threshold=0.3)
                      if i["score"] >= MIN_SCORE and i["mask"].sum() >= MIN_PX * wh[0] / WH[0]]
    return out


def render_mask(emb, cam, q, g, min_px=MIN_PX):
    r = PosedRobot(emb, [(q, g)]).masks(cam)[0]
    return r if r.sum() >= min_px else r & False


def pick(ins, r):
    """Union of the instances that touch the render r (empty if none)."""
    ours = [m for m in ins if (m & r).sum() >= TOUCH * m.sum()]
    return np.any(ours, 0) if ours else np.zeros(r.shape, bool)


def score(emb, cam, yaw, inst, q, g):
    """Per frame {t: (intersection, union) pixels} of the FK robot rendered with cam (at 320x180; scaled to the
    instance masks' size) and the real robot."""
    H, W = next(v[0].shape for v in inst.values() if v) if any(inst.values()) else WH[::-1]
    c = cam.scaled(H, W)
    emb.set_gripper_yaw(yaw)
    ts = list(inst)
    rs = PosedRobot(emb, [(q[t], g[t]) for t in ts]).masks(c)
    per = {}
    for t, r in zip(ts, rs):
        ins = inst[t]
        r = r if r.sum() >= MIN_PX else r & False
        if not ins and not r.any():
            continue
        m = pick(ins, r)
        per[t] = (int((m & r).sum()), int((m | r).sum()))
    return per


def summary(per):
    """iou: pixel-weighted over the clip (sum of intersections / sum of unions), so frames where the robot only
    grazes the image border (a few pixels, IoU 0 or 1 by chance) barely count; iou_frames: the plain per-frame mean;
    robot_px: mean union per frame (how much robot the clip shows)."""
    if not per:
        return dict(iou=None, iou_frames=None, frac_good=None, robot_px=0.0, n_frames=0, frames={})
    i, u = np.array(list(per.values()), float).T
    f = i / np.maximum(u, 1)
    return dict(iou=float(i.sum() / max(u.sum(), 1)), iou_frames=float(f.mean()), frac_good=float((f >= 0.6).mean()),
                robot_px=float(u.mean()), n_frames=len(f),
                frames={str(k): [int(a), int(b)] for k, (a, b) in per.items()})


def episode(run, emb, idx, n=N, out=OUT):
    d = episode_dir(run, idx)
    cam, yaw = run_camera(d)
    q, g = states(idx, n)
    inst = instances(idx, range(0, min(n, len(q)), max(8, n // 15)))
    res = summary(score(emb, cam, yaw, inst, q, g))
    (d / out).write_text(json.dumps(res))
    return res


def main():
    p = parser(__doc__)
    p.add_argument("run", type=Path, help="proxy run directory (droid/run.py output)")
    p.add_argument("--ids", required=True, help="episode id list")
    p.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("T", "N"), help="T-th of N shards")
    p.add_argument("--clip-frames", type=int, default=N, help="clip length in steps (default %(default)s)")
    p.add_argument("--out-name", default=OUT, help="per-episode output file name (default %(default)s)")
    a = p.parse_args()
    from simpwm.core.embodiment import Embodiment

    emb = Embodiment("franka_robotiq")
    for idx in read_ids(a.ids, *a.shard):
        if (episode_dir(a.run, idx) / a.out_name).exists():
            continue
        try:
            r = episode(a.run, emb, idx, a.clip_frames, a.out_name)
            print(idx, r["iou"] if r["iou"] is None else round(r["iou"], 3), flush=True)
        except Exception as e:  # one bad episode must not stop the shard
            print("FAILED", idx, repr(e), flush=True)


if __name__ == "__main__":
    main()
