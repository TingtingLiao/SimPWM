"""Per-episode camera fit on the episode's own recording (the cameras of the LTX training and test proxies).

Setup cameras (droid/setup_calib.py) are far off for many episodes: the arm + gripper IoU (eval/arm_iou.py) varies
mostly between setups, and the misses are whole-arm offsets (a failed setup fit, or a camera moved without a new logged
extrinsic). Here the camera is refitted per episode, before any proxy is rendered:
  masks   SAM 3 robot instances over the whole frame (no render prior) on frames of the clip plus a few from the rest of
          the episode; after a first fit only the instances touching the render are kept (a second robot in view
          drops out) and the fit is repeated
  starts  the setup camera (resources/droid/setup_cameras/<setup>.json) and the DROID-logged extrinsic, each with
          sim/camera_calib.py's coarse grid; the start with the lower fit cost is refined further
  score   the clip's pixel-weighted arm IoU (eval/arm_iou.py at 640x352); the refined camera is kept only if it beats
          the starting camera
Silhouettes are GPU-rasterized (core/raster_gpu.py), so full-length fits take seconds; SAM 3 dominates the runtime.

Writes OUT_DIR/<idx>.json {K, T_wc, dist (at 320x180), gripper_yaw, start, iou_before, iou_after, arm (eval/arm_iou
summary of the kept camera), T}, the input of `droid/run.py --camera-dir OUT_DIR`. Finished episodes are skipped.
usage: python -m simpwm.droid.episode_calib OUT_DIR IDX [IDX ...] [--clip-frames 241]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from simpwm.core.camera import Camera
from simpwm.droid import data, setup_calib
from simpwm.eval import arm_iou
from simpwm.sim import camera_calib
from simpwm.video.dataset import ep_raw_dir

FIT_WH = (640, 360)  # SAM 3 input for the fit masks, downsampled to the 320x180 camera
N_CLIP, N_EXTRA = 12, 8  # fit frames in the clip / from the rest of the episode (better-conditioned extrinsics)
FAST = dict(scale=0.5, maxfev=1500, top=3)  # first round, per start: coarse grid + local fit at 160x90
FINE = dict(scale=1.0, maxfev=1500)  # second round at 320x180, from the start with the lower fit cost


def logged_camera(idx):
    meta = json.loads((data.DROID_RAW / ep_raw_dir(idx) / "metadata.json").read_text())
    return data.logged_camera(meta, "ext1", 180, 320)


def setup_camera(idx):
    """(Camera, gripper yaw) of the episode's setup calibration, or (None, 90) if its setup is not calibrated."""
    key, _ = setup_calib.load_index()
    f = setup_calib.CACHE / f"{key.get(idx)}.json"
    if not f.exists():
        return None, 90.0
    c = json.loads(f.read_text())
    return Camera(np.array(c["K"]), np.array(c["T_wc"]), 180, 320, tuple(c["dist"])), float(c.get("gripper_yaw", 90))


def small(m):
    return cv2.resize(m.astype(np.uint8), (320, 180), interpolation=cv2.INTER_AREA) > 0


def masks_all(inst):
    m = {t: np.any(v, 0) for t, v in inst.items() if v}
    return {t: x for t, x in m.items() if x.sum() >= 50}


def masks_ours(emb, ep, inst, cam):
    """Per frame, the instances touching cam's render (a second robot in view drops out)."""
    m = {t: arm_iou.pick(v, arm_iou.render_mask(emb, cam, ep["arm_q"][t], ep["grip"][t], 10)) for t, v in inst.items()}
    return {t: x for t, x in m.items() if x.sum() >= 50}


def episode(emb, idx, clip_frames=97):
    """Fit the camera of episode idx; the arm IoU is scored on its first clip_frames recorded steps."""
    every = max(8, clip_frames // 15)  # eval/arm_iou.py's frame step for a clip of this length
    q, g = arm_iou.states(idx)
    T = len(q) - 1  # the recording has one frame fewer than the trajectory (frame i <-> step i)
    cam_s, yaw = setup_camera(idx)
    cam_l = logged_camera(idx)
    clip = range(0, min(clip_frames, T), every)
    inst_eval = arm_iou.instances(idx, clip)
    arm = lambda c: arm_iou.summary(arm_iou.score(emb, c, yaw, inst_eval, q, g))  # noqa: E731
    base_cam, start = (cam_s, "setup") if cam_s is not None else (cam_l, "logged")
    best = (arm(base_cam), base_cam, start + " (unrefined)")
    base = best[0]["iou"] or 0.0
    fit_ts = sorted(set(np.linspace(4, min(clip_frames, T) - 1, N_CLIP).astype(int).tolist())
                    | set(np.linspace(min(clip_frames, T - 1), T - 1, N_EXTRA).astype(int).tolist()))
    inst_fit = {t: [small(m) for m in v]
                for t, v in arm_iou.instances(idx, [t for t in fit_ts if t < T], FIT_WH).items()}
    ep = dict(arm_q=q, grip=g)
    emb.set_gripper_yaw(yaw)
    m1 = masks_all(inst_fit)
    if len(m1) >= 3:
        starts = [(n, c) for n, c in (("setup", cam_s), ("logged", cam_l)) if c is not None]
        fits = [(camera_calib.refine(emb, ep, m1, c, coarse=True, n_params=9, **FAST), n) for n, c in starts]
        (cam1, _), name = min(fits, key=lambda f: f[0][1]["cost"])
        m2 = masks_ours(emb, ep, inst_fit, cam1)
        cam2 = camera_calib.refine(emb, ep, m2, cam1, n_params=9, **FINE)[0] if len(m2) >= 3 else cam1
        s2 = arm(cam2)
        if (s2["iou"] or 0.0) > base:
            best = (s2, cam2, name)
    a, c, name = best
    return dict(K=c.K.tolist(), T_wc=c.T_wc.tolist(), dist=list(c.dist), gripper_yaw=yaw, start=name,
                iou_before=base, iou_after=a["iou"] or 0.0, arm=a, T=int(T))


def main():
    ap = argparse.ArgumentParser(prog="python -m simpwm.droid.episode_calib", description=__doc__.split("\n\n")[0],
                                 fromfile_prefix_chars="@")
    ap.convert_arg_line_to_args = str.split  # @FILE: a whitespace-separated id list
    ap.add_argument("out_dir", type=Path, help="output directory of the per-episode cameras <idx>.json")
    ap.add_argument("idxs", type=int, nargs="+", metavar="IDX",
                    help="episode indices (droid/data.py manifest); @FILE reads them from a file")
    ap.add_argument("--clip-frames", type=int, default=97, help="clip length (recorded steps) the arm IoU is scored on")
    args = ap.parse_args()

    from simpwm.core.embodiment import Embodiment

    emb = Embodiment("franka_robotiq")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for idx in args.idxs:
        f = args.out_dir / f"{idx}.json"
        if f.exists():
            continue
        try:
            t0 = time.time()
            r = episode(emb, idx, args.clip_frames)
            f.write_text(json.dumps(r, default=float))
            print("refine", idx, round(r["iou_before"], 3), "->", round(r["iou_after"], 3), r["start"],
                  f"{time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # one bad episode must not stop the shard
            print("FAILED", idx, repr(e), flush=True)


if __name__ == "__main__":
    main()
