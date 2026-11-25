"""Grasp agreement and grasped-object IoU of a DROID proxy run over its clip (the first --clip-frames recorded steps):
the automatic grasp metrics of Table 2.

Which object the real gripper grasps comes from the real video only (gt_targets): per real grasp in the clip (a closure
that stops short of fully closed, grasp_gt.real_grasps), SAM 2 segments the thing between the fingers at the stall
frame (grasp_gt.object_masks) and tracks it back to step 0. Strict mode (default) also rejects candidates larger than
MAX_FRAC of the image, that do not move with the gripper, or whose back-track drifts. The proxy's grasped objects are
the info.json held_objects attached within the clip (step-0 masks: masks0, or the rendered footprint for phantoms).
Per episode:
  same       the proxy holds a real target (grasp_gt.same_object on the step-0 masks): object IoU = mean over the
             clip of proxy C_OBJ0 + id against the SAM 3 track of that target (from its step-0 mask)
  diff / nested / miss / fp
             the proxy grasps another object, a container of / content in it, nothing, or something while the real
             gripper grasps nothing: object IoU 0
  none       neither grasps: no object IoU
  undecided  the real gripper grasps but no target could be extracted: no object IoU
Success = same or none, over the decided episodes. Every EVERY-th step from EVERY on (step 0 is the real frame); a frame
where only one side shows the object scores 0. Means over the clip and over its first 97 steps.

Writes RUN/droid_XXXXX/<--out-name> {grasp, object, object_97, object_static, object_static_97, n_real_grasps,
n_targets, held, frames, end_err, end_t}. object_static: the target's step-0 mask against its track (a proxy that never
moves the object). end_err (same only): centroid distance in pixels at 1280 wide between the proxy object and the real
target at the last step of the final quarter of the clip where both are present. --gt-run RUN: real targets from RUN's
camera and robot render (ablations: every variant is scored against the same targets).
usage: python -m simpwm.eval.grasp_iou RUN --ids IDS_TXT [--shard T N] [--clip-frames N] [--lenient] [--out-name F]
           [--gt-run RUN]
"""

import json
from pathlib import Path

import cv2
import numpy as np

from simpwm.core.state import C_OBJ0, is_robot
from simpwm.eval.common import centroid, episode_dir, parser, read_ids, real_frames, run_camera, states
from simpwm.eval.grasp_gt import ROBOT_MAX, nested, object_masks, real_grasps, same_object

N = 241  # clip length (recorded steps)
EVERY, N97 = 8, 97
MIN_PX = 8  # at the proxy's 320x180: smaller masks count as absent
MAX_FRAC = 0.04  # strict: a grasped object covers at most this fraction of the image
FOLLOW_PX, FOLLOW_MIN, FOLLOW_T = 12, 0.4, 30  # strict: motion test (TCP pixels moved, fraction followed, steps)
CYCLE_MIN = 0.5  # strict: IoU of the forward re-track of the step-0 mask with the stall-frame mask
END_LAST = 0.25  # end error: the last step, within the final quarter of the clip, where both masks are present


def out_name(strict=True):
    return "grasp_iou2.json" if strict else "grasp_iou.json"


def _iou(a, b):
    a, b = (a if a.sum() >= MIN_PX else a & False), (b if b.sum() >= MIN_PX else b & False)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else None


def _mean(d, hi):
    v = [x for t, x in d.items() if t < hi and x is not None]
    return float(np.mean(v)) if v else None


def held_within(info, n):
    """Objects the proxy holds within its first n steps: attached in a logged state (every 5th step) before n. The
    run's held_objects covers its whole simulated clip, which can be longer than the evaluated one."""
    held = {int(k) for k in info.get("held_objects", [])}
    att = {int(o["id"]) for s in info.get("states", []) if s["t"] < n for o in s["objects"] if o.get("attached")}
    return held & att if info.get("states") else held


def _tcp_px(cam, emb, q, g, t):
    emb.set_state(q[t], g[t])
    uv, z = cam.project(emb.keypoints()["tcp"][None])
    return uv[0].astype(float), z[0] > 0


def _follows(frames, m, cam, emb, q, g, ts, te):
    """False when the gripper visibly moves between ts and te while the stall-frame mask stays put (background, a
    table or wall region): the grasped object must move with the jaws. True when it follows or cannot be tested."""
    from simpwm.eval import tracks

    te = min(te, ts + FOLLOW_T, len(frames) - 1)
    (a, za), (b, zb) = _tcp_px(cam, emb, q, g, ts), _tcp_px(cam, emb, q, g, te)
    d = b - a
    if te <= ts or not (za and zb) or np.linalg.norm(d) < FOLLOW_PX:
        return True
    tr = tracks.track(frames[ts:te + 1].copy(), {0: m})[0]
    for k in range(len(tr) - 1, 0, -1):  # the last step where the object is still visible
        if tr[k].sum() >= 0.3 * m.sum():
            return float(np.dot(centroid(tr[k]) - centroid(m), d) / np.dot(d, d)) >= FOLLOW_MIN
    return True  # lost (occluded by the hand): cannot tell


def find_targets(frames, cat, cam, emb, q, g, strict=True):
    """([dict(m0=step-0 mask, s0, s1, ts)], real grasps): at most one target per real grasp (closure start s0, end s1,
    stall step ts) within the clip; frames and cat (T, H, W) at the same size, cam at that size. Strict mode rejects
    candidates larger than MAX_FRAC of the image, that do not move with the gripper, or whose back-track to step 0
    drifts (area changes by more than 3x, or the forward re-track misses the stall mask)."""
    from simpwm.eval import tracks

    T, H, W = cat.shape
    rob0 = is_robot(cat[0])
    out, grasps = [], real_grasps(g[:T])
    for s0, s1, ts in grasps:
        (u, v), zf = _tcp_px(cam, emb, q, g, ts)
        u, v = int(u), int(v)
        if not (zf and 0 <= u < W and 0 <= v < H):
            continue
        rob = cv2.dilate(is_robot(cat[ts]).astype(np.uint8), np.ones((3, 3))).astype(bool)
        for k, (m, _) in enumerate(object_masks(frames[ts], rob, u, v, H, W)):
            if k >= 3:  # at most three back-tracks per grasp
                break
            if strict and (m.sum() > MAX_FRAC * H * W or not _follows(frames, m, cam, emb, q, g, ts, s1)):
                continue
            m0 = m if ts == 0 else tracks.track(frames[ts::-1].copy(), {0: m})[0][-1]
            if strict and ts and not (1 / 3 <= m0.sum() / m.sum() <= 3 and _iou(
                    tracks.track(frames[:ts + 1].copy(), {0: m0})[0][-1], m) or 0) >= CYCLE_MIN:
                continue  # the back-track drifted: tracking its step-0 mask forward must land on the stall mask
            if m0.sum() >= 25 and (m0 & rob0).sum() <= ROBOT_MAX * m0.sum():
                out.append(dict(m0=m0, s0=int(s0), s1=int(s1), ts=int(ts)))
                break
    return out, grasps


def gt_targets(frames, cat, cam, emb, q, g, strict=True):
    """(step-0 masks of the objects the real gripper grasps within the clip, number of real grasps)."""
    tg, grasps = find_targets(frames, cat, cam, emb, q, g, strict)
    return [t["m0"] for t in tg], len(grasps)


def episode(run, idx, emb, n=N, strict=True, out=None, gt_run=None):
    d = episode_dir(run, idx)
    z = np.load(d / "proxy_cat.npz")
    cat = z["cat"][:n]
    info = json.loads((d / "info.json").read_text())
    T, H, W = cat.shape
    frames = np.stack(real_frames(idx, T, (W, H)))
    T = min(T, len(frames))
    q, g = states(idx, T)
    cam, yaw = run_camera(info, H, W)
    emb.set_gripper_yaw(yaw)
    if gt_run:
        gd = episode_dir(gt_run, idx)
        gcam, gyaw = run_camera(json.loads((gd / "info.json").read_text()), H, W)
        emb.set_gripper_yaw(gyaw)
        tg, n_real = gt_targets(frames, np.load(gd / "proxy_cat.npz")["cat"][:T], gcam, emb, q, g, strict)
        emb.set_gripper_yaw(yaw)
    else:
        tg, n_real = gt_targets(frames, cat, cam, emb, q, g, strict)
    held = held_within(info, T)
    hm = {int(k): m for k, m in zip(z["obj_ids"], z["masks0"]) if int(k) in held}
    hm.update({int(k): m for k, m in zip(z["phantom_ids"], z["phantom_masks0"]) if int(k) in held})
    pairs = [(k, j) for k, a in hm.items() for j, b in enumerate(tg) if same_object(a, b)]
    if n_real and hm:
        grasp = "undecided" if not tg else "same" if pairs else "nested" if any(
            nested(a, b) for a in hm.values() for b in tg) else "diff"
    else:
        grasp = "miss" if n_real else "fp" if hm else "none"

    ts = list(range(EVERY, T, EVERY))
    res = dict(grasp=grasp, n_real_grasps=n_real, n_targets=len(tg), held=sorted(hm))
    obj, st = {}, {}
    if grasp == "same":
        from simpwm.eval import tracks_sam3

        js = sorted({j for _, j in pairs})
        tr = tracks_sam3.track(frames[:T], {j: tg[j] for j in js})
        for t in ts:
            o = [_iou(cat[t] == C_OBJ0 + k, tr[j][t]) for k, j in pairs]
            s = [_iou(tr[j][0], tr[j][t]) for j in js]
            o, s = [x for x in o if x is not None], [x for x in s if x is not None]
            obj[t], st[t] = (float(np.mean(o)) if o else None), (float(np.mean(s)) if s else None)
        # end error: centroid distance (pixels at 1280 wide) of the proxy object and its real target at the last step
        # where both are present, searched back over the final quarter of the clip
        for t in reversed([t for t in ts if t >= (1 - END_LAST) * T]):
            e = [np.linalg.norm(centroid(cat[t] == C_OBJ0 + k) - centroid(tr[j][t])) * 1280 / W for k, j in pairs
                 if (cat[t] == C_OBJ0 + k).sum() >= MIN_PX and tr[j][t].sum() >= MIN_PX]
            if e:
                res["end_err"], res["end_t"] = float(np.mean(e)), t
                break
    elif grasp in ("diff", "nested", "miss", "fp"):
        obj = {t: 0.0 for t in ts}
    for name, per in (("object", obj), ("object_static", st)):
        res[name], res[name + "_97"] = _mean(per, T), _mean(per, N97)
    res["frames"] = obj
    (d / (out or out_name(strict))).write_text(json.dumps(res))
    return res


def main():
    p = parser(__doc__)
    p.add_argument("run", type=Path, help="proxy run directory (droid/run.py output)")
    p.add_argument("--ids", required=True, help="episode id list")
    p.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("T", "N"), help="T-th of N shards")
    p.add_argument("--clip-frames", type=int, default=N, help="clip length in steps (default %(default)s)")
    p.add_argument("--lenient", action="store_true", help="no strict target checks (Table 2's grasp metric)")
    p.add_argument("--out-name", help="per-episode output file (default grasp_iou2.json; --lenient: grasp_iou.json)")
    p.add_argument("--gt-run", type=Path, help="proxy run whose camera / robot render extracts the real targets")
    a = p.parse_args()
    from simpwm.core.embodiment import Embodiment

    strict, out = not a.lenient, a.out_name or out_name(not a.lenient)
    emb = Embodiment("franka_robotiq")
    for idx in read_ids(a.ids, *a.shard):
        if (episode_dir(a.run, idx) / out).exists():
            continue
        try:
            r = episode(a.run, idx, emb, a.clip_frames, strict, out, a.gt_run)
            print(idx, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "frames"},
                  flush=True)
        except Exception as e:  # one bad episode must not stop the shard
            print("FAILED", idx, repr(e), flush=True)


if __name__ == "__main__":
    main()
