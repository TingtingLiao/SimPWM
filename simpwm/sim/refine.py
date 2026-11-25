"""Action-conditioned refinement of the initial state, from the gripper signal of the robot-state input.

  snap     monocular depth leaves each object's distance along its camera rays uncertain (~10 cm), exactly the error
           that makes a grasp miss. Where the effector closes on a pixel covered by an object in the first frame, that
           object must be at the effector's depth there: scale it about the camera centre (which keeps its image
           footprint) so that the two agree.
  empty    a closure in which the observed gripper closes fully held nothing: no proposal, snap or grasp there.
  phantom  a closure that stops short holds something; if no object of the initial state lies between the pads, a
           small box is added there.

Embodiment-agnostic: only the closure signal and the effector keypoints (tcp, pad_l, pad_r) are used.
"""

from __future__ import annotations

import cv2
import numpy as np

from simpwm import config
from simpwm.core.state import Obj
from simpwm.sim import scale as sc
from simpwm.sim.engine import JAW_HALF_W, JAW_PALM, JAW_TIP, JAW_Y_MARGIN, effector_frame, grasp_key, jaw_inside


def closure_onsets(grip, thr):
    """Steps at which the closure crosses thr upwards."""
    g = np.asarray(grip)
    return [t for t in range(1, len(g)) if g[t] > thr >= g[t - 1]]


def closure_windows(grip, thr, eps=0.01):
    """(onset, end) per closing: from crossing thr until the gripper stops closing (contact or fully closed)."""
    g = np.asarray(grip)
    out = []
    for t0 in closure_onsets(g, thr):
        t1 = t0
        while t1 + 4 < len(g) and g[t1 + 4] > g[t1] + eps:  # still closing over the next few steps
            t1 += 1
        out.append((t0, min(t1 + 4, len(g) - 1)))
    return out


def empty_closure_steps(grip, thr=0.12, empty=0.975):
    """Steps of the closed segments (runs of grip > thr) in which the gripper closes fully, i.e. on nothing: the
    median closure of the segment (after its first step) is >= `empty`. This needs the observed gripper position
    (a closure that stops short of fully closed holds something), not the commanded one."""
    g = np.asarray(grip)
    out, t = set(), 0
    while t < len(g):
        if g[t] > thr:
            s = t
            while t < len(g) and g[t] > thr:
                t += 1
            q = g[s:t]
            if len(q) < 3 or np.median(q[1:]) >= empty:
                out |= set(range(s, t))
        t += 1
    return out


def lowest_tcp(emb, arm_q, grip, t0, t1):
    """Lowest tcp position within steps t0..t1 (where the jaws close on something) and its step."""
    ps = []
    for t in range(t0, t1 + 1):
        emb.set_state(arm_q[t], grip[t])
        ps.append((emb.keypoints()["tcp"], t))
    return min(ps, key=lambda x: x[0][2])


def lift_above_support(pts, n, d0):
    """Shift points along the support normal n so that none lies below the support plane n.x + d0 = 0."""
    below = -(pts @ n + d0).min()
    return pts + n * below if below > 0 else pts


def jaw_footprint(emb, cam, q, g, n=9):
    """(H, W) bool image footprint of the jaw volume (engine.jaw_inside) at state (q, g): a grid of points inside it,
    projected, dilated by one pixel."""
    emb.set_state(q, g)
    o, R, kp = effector_frame(emb)
    pl, pr = (kp["pad_l"] - o) @ R, (kp["pad_r"] - o) @ R
    zc = 0.5 * (pl[2] + pr[2])
    xs = np.linspace(min(pl[0], pr[0]), max(pl[0], pr[0]), n)
    ys = np.linspace(-(JAW_HALF_W + JAW_Y_MARGIN), JAW_HALF_W + JAW_Y_MARGIN, n)
    zs = np.linspace(zc - JAW_PALM, zc + JAW_TIP, 2 * n)
    P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3) @ R.T + o
    uv, z = cam.project(P)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.W) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.H)
    m = np.zeros((cam.H, cam.W), np.uint8)
    m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = 1
    return cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool)


def snap_to_grasps(objs, masks, emb, cam, arm_q, grip, plane, thr=0.12, px=5, hit_px=12, ratio_range=(0.35, 2.5),
                   skip=frozenset()):
    """The snap, per closing window (except those starting in `skip`): the object whose first-frame mask, dilated by
    hit_px, contains the closing point is scaled about the camera centre so that its points within px pixels of the
    closing point sit at the effector's depth, if that ratio is within ratio_range (closing is strong evidence, so the
    range is wide). Each object snaps at most once. Returns a log: one record per window, "applied" if moved.
    px / hit_px are in 320x180 pixels (scaled to the simulation resolution, sim/scale.py)."""
    px, hit_px = sc.px(px), sc.px(hit_px)
    H, W = cam.H, cam.W
    c = cam.T_wc[:3, 3]
    n, d0 = np.asarray(plane[:3]), plane[3]
    log, done = [], set()
    dil = [cv2.dilate(m.astype(np.uint8), np.ones((2 * hit_px + 1, 2 * hit_px + 1))).astype(bool) for m in masks]
    for t0, t1 in closure_windows(grip, thr):
        if t0 in skip:
            continue
        p, t = lowest_tcp(emb, arm_q, grip, t0, t1)
        uv, z = cam.project(p[None])
        u, v = int(uv[0, 0]), int(uv[0, 1])
        if z[0] <= 0 or not (0 <= u < W and 0 <= v < H):
            continue
        hit = [i for i, m in enumerate(dil) if m[v, u] and objs[i].id not in done]
        if not hit:
            log.append(dict(t=t, px=[u, v], hit=None, window=[t0, t1]))
            continue
        if config.cfg.jaw_inside:
            # the object that covers the jaws' image footprint at the closing pose (the projected jaw volume,
            # engine.jaw_inside), most covered pixels first, the smaller object on ties (a cup on a tray, a lid on a
            # pot); an object only beside the fingers covers none of it and is not snapped into the jaws
            foot = jaw_footprint(emb, cam, arm_q[t], grip[t])
            cov = {k: int((masks[k] & foot).sum()) for k in hit}
            i = max(hit, key=lambda k: (cov[k], -masks[k].sum()))
            if cov[i] == 0:
                log.append(dict(t=t, px=[u, v], hit=None, window=[t0, t1], jaw_footprint=int(foot.sum())))
                continue
        else:
            # the object nearest to the closing point in the image (the smaller one on ties: a pepper on a plate)
            dist = {k: np.min(np.hypot(*(np.argwhere(masks[k]) - [v, u]).T)) for k in hit}
            i = min(hit, key=lambda k: (round(dist[k] / sc.px(3)), masks[k].sum()))
        ob = objs[i]
        puv, pz = cam.project(ob.pts)
        near = np.linalg.norm(puv - uv[0], axis=1) < px
        zo = float(np.median(pz[near] if near.sum() >= sc.cnt(3) else pz))
        r = float(z[0] / zo)
        rec = dict(t=t, px=[u, v], obj=ob.id, ratio=r, window=[t0, t1])
        lo, hi = ratio_range
        if lo < r < hi:
            ob.pts = lift_above_support(c + (ob.pts - c) * r, n, d0)
            done.add(ob.id)
            rec["applied"] = True
        log.append(rec)
    return log


def phantom_grasps(objs, emb, arm_q, grip, plane, thr=0.12, skip=frozenset(), min_pts=6, margin=0.012, depth=0.03,
                   n_pts=600, only=None):
    """Closures that hold something (`skip` = the empty closures) but where no object of the initial state lies
    between the pads at any step of the closing window (the segmentation missed it, or it is not visible in the first
    frame): add a small box between the pads at the end of the window, as wide as the jaw opening there, pinned until
    grasped. only: restrict to these window onsets (two-pass rule: windows where a rollout grasped nothing).
    Returns a log; the new objects are appended to `objs`."""
    n, d0 = np.asarray(plane[:3]), plane[3]
    max_open = emb.cfg.get("max_open_m", 0.10)
    rng = np.random.default_rng(0)
    log = []
    for t0, t1 in closure_windows(grip, thr):
        if t0 in skip or (only is not None and t0 not in only):
            continue
        poses = []
        for t in range(t0, t1 + 1):
            emb.set_state(arm_q[t], grip[t])
            o, R, kp = effector_frame(emb)
            poses.append((t, o, R, {k: kp[k] for k in ("pad_l", "pad_r", "tcp")}))
        if config.cfg.jaw_inside:  # something is already inside the jaw volume at some step of the window
            if any(jaw_inside(ob.pts, kp, o, R)[0].sum() >= sc.cnt(min_pts) for ob in objs for _, o, R, kp in poses):
                continue
        elif any(grasp_key(ob.pts, kp, o, margin, depth, max_open)[1] >= sc.cnt(min_pts)
                 for ob in objs for _, o, _, kp in poses):
            continue
        t, o, R, kp = poses[-1]  # the gripper stalls at the end of the closing window
        w = float(np.clip(np.linalg.norm(kp["pad_r"] - kp["pad_l"]), 0.01, max_open))
        half = np.array([w / 2, 0.6 * w / 2 + 0.005, 0.6 * w / 2 + 0.005])  # x = across the jaws (effector frame)
        local = rng.uniform(-1, 1, (sc.cnt(n_pts), 3)) * half
        pts = lift_above_support(local @ R.T + o, n, d0)
        ob = Obj(id=len(objs), name=f"phantom{len(objs)}", pts=pts, radius=0.004, pix=0, pinned=True)
        objs.append(ob)
        log.append(dict(t=int(t), obj=ob.id, width_m=round(w, 4)))
    return log
