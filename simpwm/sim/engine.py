"""Rule engine: scene state s_t -> s_{t+1} given the next robot state.

RuleEngine only uses the effector keypoints of the embodiment (tcp and the two pads of a parallel gripper) and the
object point sets, never object categories:
  grasp    the effector closes and >= min_pts object points lie between the pads -> the points follow the effector
  release  the effector opens -> the object becomes free
  gravity  free, unpinned objects fall (g = 9.81) onto a height map of the static scene and the other objects
  soft     a held soft object (cloth, cable) hangs from the jaws, or slides as a sheet when the jaws are low
  contact  per-object contact strength each frame (1 = held, (0, 0.5] = touching), exported as a proxy channel

Any engine with the same constructor and `step` can replace it (pipeline.generate's engine_cls).
"""

from __future__ import annotations

import numpy as np

from simpwm import config
from simpwm.core.state import State
from simpwm.sim import scale as sc

G = 9.81
SOFT_HALF_WIDTH = 0.12  # lateral extent (m) of a draped object along the pad axis
SOFT_SLIDE = 0.04  # jaws below this height (m) above the support: a held cloth slides as a sheet


class HeightMap:
    """Max-height grid (cell metres) over the xy plane of the static support points, plus moving objects on demand."""

    def __init__(self, pts, cell=0.01, reach=2.5):
        self.cell = cell
        # only the robot's surroundings matter; a bad depth scale can scatter support points tens of metres away and
        # the grid would then need hundreds of GB
        near = np.abs(pts[:, :2]).max(1) <= reach
        pts = pts[near] if near.any() else np.zeros((1, 3))
        self.lo = pts[:, :2].min(0) - 0.5
        ij = ((pts[:, :2] - self.lo) / cell).astype(int)
        self.shape = tuple(ij.max(0) + 2 + int(1.0 / cell))
        self.h = np.full(self.shape, -np.inf)
        np.maximum.at(self.h, (ij[:, 0], ij[:, 1]), pts[:, 2])
        self.base = self.h.copy()

    def query(self, xy):
        ij = np.clip(((xy - self.lo) / self.cell).astype(int), 0, np.array(self.shape) - 1)
        return self.h[ij[:, 0], ij[:, 1]]

    def with_points(self, pts):
        """A copy of the static map with the points `pts` (N, 3) added."""
        hm = HeightMap.__new__(HeightMap)
        hm.cell, hm.lo, hm.shape, hm.base = self.cell, self.lo, self.shape, self.base
        hm.h = self.base.copy()
        if len(pts):
            ij = np.clip(((pts[:, :2] - self.lo) / self.cell).astype(int), 0, np.array(self.shape) - 1)
            np.maximum.at(hm.h, (ij[:, 0], ij[:, 1]), pts[:, 2])
        return hm


def grasp_region(pts, kp, margin=0.012, depth=0.03):
    """Points between the pads: along the pad axis (with a margin) and within `depth` of it."""
    a, b = kp["pad_l"], kp["pad_r"]
    ab = b - a
    L = np.linalg.norm(ab) + 1e-9
    s = (pts - a) @ ab / L
    perp = np.linalg.norm((pts - a) - s[:, None] * ab[None] / L, axis=1)
    return (s > -margin) & (s < L + margin) & (perp < depth)


def grasp_key(pts, kp, o, margin=0.012, depth=0.03, max_open=0.10):
    """RuleEngine's grasp test for one object: (fits between the open jaws, number of points between the pads)."""
    ab = kp["pad_r"] - kp["pad_l"]
    ab = ab / (np.linalg.norm(ab) + 1e-9)
    n_in = int(grasp_region(pts, kp, margin, depth).sum())
    # local width across the jaws, measured on the part of the object near the grasp point
    rel = pts - o
    near = np.linalg.norm(rel - (rel @ ab)[:, None] * ab[None], axis=1) < 0.03
    span = np.ptp(pts[near] @ ab) if near.sum() >= sc.cnt(3) else np.inf
    # a preference, not a filter, so no tolerance: a cloth measuring just over the opening must not beat the one that
    # has the most points between the jaws
    return (bool(span <= max_open), n_in)


# config.cfg.jaw_inside: the jaw volume of the parallel gripper, in the effector frame of effector_frame (x = pad axis,
# z = towards the fingertips, y = across the pads' width), relative to the pad keypoints (the pads' inner contact
# surfaces; Robotiq 2F-85 pads: half-width 1.1 cm, half-length 1.9 cm along z). A point is inside the jaws iff it lies
# strictly between the two pad surfaces (inside the current opening), within the pad width (+ JAW_Y_MARGIN) and, along
# the approach axis, from JAW_TIP beyond the pad centres (fingertips) back to JAW_PALM behind them (towards the palm).
JAW_HALF_W, JAW_Y_MARGIN = 0.011, 0.005
JAW_TIP, JAW_PALM = 0.019 + 0.005, 0.019 + 0.025


def jaw_inside(pts, kp, o, R):
    """(N,) bool: points inside the jaw volume, and their approach coordinate z (effector frame)."""
    p = (pts - o) @ R
    pl, pr = (kp["pad_l"] - o) @ R, (kp["pad_r"] - o) @ R
    lo, hi = min(pl[0], pr[0]), max(pl[0], pr[0])
    zc = 0.5 * (pl[2] + pr[2])
    ins = ((p[:, 0] > lo) & (p[:, 0] < hi) & (np.abs(p[:, 1]) < JAW_HALF_W + JAW_Y_MARGIN)
           & (p[:, 2] < zc + JAW_TIP) & (p[:, 2] > zc - JAW_PALM))
    return ins, p[:, 2]


def jaw_pick(objs, kp, o, R, min_pts):
    """Grasp choice with config.cfg.jaw_inside: among the objects with >= min_pts points inside the jaw volume, the
    topmost one along the approach (smallest median approach coordinate of its inside points: nearest the palm, i.e.
    on top of a stack); ties by more points inside. Returns (object or None, {object id: (n inside, median z)})."""
    cand, info = [], {}
    for ob in objs:
        ins, z = jaw_inside(ob.pts, kp, o, R)
        n = int(ins.sum())
        if n:
            info[ob.id] = (n, float(np.median(z[ins])))
        if n >= min_pts:
            cand.append((float(np.median(z[ins])), -n, ob.id, ob))
    return (min(cand, key=lambda c: c[:3])[3] if cand else None), info


# config.cfg.carry_children: a free object is carried with the grasped (rigid) parent if its lowest points rest on the
# parent: >= CHILD_FRAC of its bottom CHILD_BOTTOM fraction (by height) lie within CHILD_DIST of the parent's points,
# and it is not below the parent (its median height is above the parent's lowest point + 1 cm: a support under the
# grasped object is never a child). Carried objects are not "held" (contact CARRIED_CONTACT < 1).
CHILD_DIST, CHILD_BOTTOM, CHILD_FRAC, CARRIED_CONTACT = 0.015, 0.2, 0.3, 0.99


def carried_children(parent, objs, plane=None):
    """Free objects resting on `parent` (see above). With the support plane given, an object whose lowest points are
    within 1 cm of the support is not carried: it stands on the table next to the parent."""
    from scipy.spatial import cKDTree

    tree, out = None, []
    zmin = parent.pts[:, 2].min()
    for ob in objs:
        if ob is parent or ob.attached or len(ob.pts) == 0:
            continue
        if np.median(ob.pts[:, 2]) <= zmin + 0.01:
            continue
        lo = ob.pts[ob.pts[:, 2] <= np.quantile(ob.pts[:, 2], CHILD_BOTTOM)]
        if plane is not None and np.median(lo @ np.asarray(plane[:3]) + plane[3]) <= 0.01:
            continue
        if tree is None:
            tree = cKDTree(parent.pts)
        d, _ = tree.query(lo, distance_upper_bound=CHILD_DIST)
        if np.mean(np.isfinite(d)) >= CHILD_FRAC:
            out.append(ob)
    return out


def effector_frame(emb):
    """Effector frame (origin, R, keypoints): origin at the tcp keypoint, x along the pad axis, z the tcp site's z
    axis made orthogonal to it."""
    kp = emb.keypoints()
    o = kp["tcp"]
    x = kp["pad_r"] - kp["pad_l"]
    x /= np.linalg.norm(x) + 1e-9
    _, Rs = emb.site_pose(emb.cfg["keypoints"]["tcp"]["site"])
    z = Rs[:, 2] - x * (Rs[:, 2] @ x)
    z /= np.linalg.norm(z) + 1e-9
    return o, np.stack([x, np.cross(z, x), z], 1), kp


def _drape_coords(pts, o, R):
    """At the grasp of a soft object: per point (offset along the pad axis, length of material to the jaws)."""
    e = (pts - o) @ R
    x = np.clip(e[:, 0], -SOFT_HALF_WIDTH, SOFT_HALF_WIDTH)
    return np.stack([x, np.hypot(np.linalg.norm(e[:, 1:], axis=1), e[:, 0] - x)], 1)


def _drape(ob, o, R, plane):
    """A held soft object hangs from the jaws: a point with material length r to the jaws hangs r below them if the
    jaws are at least r above the support, else it hangs down to the support and the rest lies on it, where it is
    dragged towards the jaws only once taut (a rope on a table)."""
    n, d = plane[:3], plane[3]
    H = max(float(o @ n + d), 0.0)
    x, r = ob.drape[:, 0], ob.drape[:, 1]
    top = o + x[:, None] * R[:, 0][None]
    pts = top - np.minimum(r, H)[:, None] * n[None]
    lie = r > H
    if lie.any():
        foot = top[lie] - H * n[None]
        q = ob.pts[lie] - foot
        q -= (q @ n)[:, None] * n[None]
        if H < SOFT_SLIDE:
            # jaws (nearly) on the support: the cloth is dragged as a sheet (wiping, sliding), every lying point moves
            # with the jaws in the support plane
            mv = o - ob.drape_o
            mv -= (mv @ n) * n
            pts[lie] = ob.pts[lie] + mv
        else:
            L = r[lie] - H
            dist = np.linalg.norm(q, axis=1) + 1e-9
            pts[lie] = foot + q * np.minimum(1.0, L / dist)[:, None]
    ob.drape_o = o.copy()
    ob.pts = pts


class RuleEngine:
    """close_thr / open_drop: closure above which the gripper is closing / drop from its maximum that counts as
    opening; min_pts, margin, depth: grasp test (grasp_region); max_open: jaw opening (m) for grasp_key."""

    def __init__(self, emb, background, dt, close_thr=0.12, open_drop=0.12, min_pts=6, margin=0.012, depth=0.03,
                 max_open=None):
        self.emb, self.bg, self.dt = emb, background, dt
        self.close_thr, self.open_drop, self.min_pts = close_thr, open_drop, sc.cnt(min_pts)  # points: sim/scale.py
        self.margin, self.depth = margin, depth
        self.hm = HeightMap(background.support_pts)
        self.max_open = max_open if max_open is not None else emb.cfg.get("max_open_m", 0.10)
        self.max_grip = 0.0
        self.no_grasp_steps = frozenset()  # steps of empty closures (refine.empty_closure_steps): no grasp there

    def step(self, st: State, arm_q, grip):
        emb = self.emb
        emb.set_state(arm_q, grip)
        o, R, kp = effector_frame(emb)
        closing = grip > self.close_thr
        self.max_grip = max(self.max_grip, grip) if closing else grip
        opening = grip < self.max_grip - self.open_drop or grip < self.close_thr
        contacts = {}
        # at most one new grasp per step: the graspable object with the most points between the pads (objects that
        # fit between the open jaws are preferred; a wider one, a cloth or a lid edge, is pinched)
        grab = None
        can_grasp = (closing and not opening and st.t + 1 not in self.no_grasp_steps
                     and not any(x.attached for x in st.objects))
        if can_grasp and config.cfg.jaw_inside:
            grab, _ = jaw_pick(st.objects, kp, o, R, self.min_pts)
        elif can_grasp:
            best = (False, 0)
            for ob in st.objects:
                key = grasp_key(ob.pts, kp, o, self.margin, self.depth, self.max_open)
                if key[1] >= self.min_pts and key > best:
                    grab, best = ob, key
        for ob in st.objects:
            if ob.attached:
                if opening:
                    ob.attached, ob.T_rel = False, None
                    ob.carried = False
                    ob.vel = np.zeros(3)
                elif ob.drape is not None:
                    _drape(ob, o, R, self.bg.plane)
                    contacts[ob.id] = 1.0
                    continue
                else:
                    ob.pts = ob.T_rel @ R.T + o
                    contacts[ob.id] = CARRIED_CONTACT if ob.carried else 1.0
                    continue
            if ob is grab:
                ob.attached, ob.pinned = True, False
                ob.T_rel = (ob.pts - o) @ R  # points in the effector frame
                if ob.soft and config.cfg.soft_objects:
                    ob.drape = _drape_coords(ob.pts, o, R)
                    ob.drape_o = o.copy()
                elif config.cfg.carry_children:  # what rests on / in the grasped object moves with it
                    for ch in carried_children(ob, st.objects, self.bg.plane):
                        ch.attached, ch.pinned, ch.carried = True, False, True
                        ch.T_rel = (ch.pts - o) @ R
                        contacts[ch.id] = CARRIED_CONTACT
                contacts[ob.id] = 1.0
                continue
            # proximity contact (touch without grasp)
            dmin = min(np.linalg.norm(ob.pts - kp[k], axis=1).min() for k in ("pad_l", "pad_r", "tcp"))
            if dmin < 0.02:
                contacts[ob.id] = float(np.clip(1 - dmin / 0.02, 0, 1)) * 0.5
        # gravity for free objects, resting on the scene and the other objects
        for ob in st.objects:
            if ob.attached or ob.pinned:
                continue
            others = [x.pts for x in st.objects if x is not ob]
            hm = self.hm.with_points(np.concatenate(others)) if others else self.hm
            if ob.drape is not None:  # a released soft object: every point falls on its own and settles where it lands
                # cells without support (-inf) fall back to the support plane: a point beside the table must not fall
                # forever
                n, d = self.bg.plane[:3], self.bg.plane[3]
                floor = np.maximum(hm.query(ob.pts[:, :2]), -(ob.pts[:, :2] @ n[:2] + d) / max(n[2], 1e-6))
                gap = np.maximum(ob.pts[:, 2] - floor, 0.0)
                if (gap > 0.004).any():
                    ob.vel[2] -= G * self.dt
                    ob.pts[:, 2] -= np.minimum(-ob.vel[2] * self.dt, gap)
                else:
                    ob.vel[:] = 0
                continue
            gap = (ob.pts[:, 2] - hm.query(ob.pts[:, :2])).min()
            if gap > 0.004 or ob.vel[2] < 0:
                ob.vel[2] -= G * self.dt
                dz = max(ob.vel[2] * self.dt, -gap)
                ob.pts = ob.pts + np.array([0, 0, dz])
                if dz == -gap:
                    ob.vel[:] = 0
        return State(t=st.t + 1, arm_q=np.asarray(arm_q), grip=float(grip), objects=st.objects, contacts=contacts)
