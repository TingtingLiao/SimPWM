"""Real-to-sim initialisation: first frame + robot state + calibrated camera -> initial scene state (objects as point
sets, static background, support plane).

Class-agnostic: an object is whatever rests on the support surface, so nothing here knows object categories.
  robot mask    Grounding DINO + SAM2 near the forward-kinematics render of the robot
  metric depth  stereo depth when the runner provides it (STEREO), else Depth Anything V2 metric depth rescaled so that
                the robot pixels match the rendered robot depth (the robot is the ruler)
  support       the horizontal surface the effector works on (gravity is -z in the robot base frame); its exact
                ray / plane depth corrects the whole depth map (affine in inverse depth)
  objects       proposals (SAM2 at the action's closing points first, then SAM 3 concept instances, holes in the
                support, SAM2 grid prompts in large holes), claimed smallest first, placed on the support and completed
                behind their visible surface
"""

from __future__ import annotations

import cv2
import numpy as np

from simpwm import config, perception
from simpwm.core.state import Background, Obj
from simpwm.perception import sam3
from simpwm.sim import scale as sc
from simpwm.sim.camera_calib import ROBOT_PHRASES, box_overlap, rendered_robot

N_PTS = 4000  # points per object
SAM3_PHRASES = ["object", "item", "container", "cloth", "toy", "tool", "food", "bottle"]  # generic concept prompts
# config.cfg.parts: part / sub-object prompts as well, so a lid, cap or knob becomes its own instance inside its pot /
# bottle / drawer (the candidates are claimed smallest first, so a nested part keeps its pixels and the parent loses
# them: no duplicated points); the report's "parts" records which object lies inside which
SAM3_PART_PHRASES = ["lid", "pot lid", "cover", "cap", "knob", "handle", "drawer", "door"]
SUPPORT_PHRASES = ["table", "countertop", "desk", "shelf", "floor"]
MODEL_HFOV_DEG = 60.0  # horizontal field of view the metric depth model assumes
THICK_MAX = 0.10  # largest thickness (m) of the completed back side of an object

# set by the dataset runners
STEREO = None  # first-frame metric depth (m) from the camera's stereo pair, set per episode, or None (monocular)
KNOWN_SUPPORT_Z = None  # world height of the support surface when it is known, else estimated
# config.cfg.hires: the full-resolution first frame (set per episode by pipeline.generate). SAM 3 then segments it
# instead of the simulation frame; its masks are area-downsampled for the simulation and kept at full resolution in
# HIRES_MASKS as (lowres, hires) pairs, which pipeline.hires_object_masks matches to the final object masks
FRAME_HI = None
HIRES_MASKS = []
HIRES_MASK_AREA = 0.3  # a block of the full-res mask counts as object in the simulation frame from 30% coverage


def _sam3_concepts(frame, phrases, **kw):
    """sam3.segment_concepts on `frame`, or with config.cfg.hires (and FRAME_HI set) on the full-resolution first
    frame, the masks area-downsampled to frame's size (the full-resolution masks are recorded in HIRES_MASKS)."""
    if not (config.cfg.hires and FRAME_HI is not None):
        return sam3.segment_concepts(frame, phrases, **kw)
    H, W = frame.shape[:2]
    out = []
    for d in sam3.segment_concepts(FRAME_HI, phrases, **kw):
        hi = d["mask"]
        lo = cv2.resize(hi.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) >= HIRES_MASK_AREA
        HIRES_MASKS.append((lo, hi))
        out.append(dict(d, mask=lo, mask_hi=hi))
    return out


def robot_mask(frame, emb, cam):
    """First-frame robot mask: Grounding DINO + SAM2 near the robot rendered in its current state, plus the rendered
    silhouette itself. Returns (mask, rendered silhouette)."""
    rend, rb = rendered_robot(emb, cam)
    if rb is None:
        return rend, rend
    pad = 0.15 * max(rb[2] - rb[0], rb[3] - rb[1]) + 6
    dets = perception.detect(frame, ROBOT_PHRASES)
    boxes = [d["box"] for d in dets if box_overlap(d["box"], rb + [-pad, -pad, pad, pad]) > 0.3]
    m = perception.segment(frame, boxes=boxes).any(0) if boxes else rend.copy()
    # segmentation spill onto chairs / stands behind the arm: keep the mask near the rendered robot (cables stay in)
    k = int(0.08 * max(rb[2] - rb[0], rb[3] - rb[1])) + sc.px(6)
    m &= cv2.dilate(rend.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1))).astype(bool)
    return m | cv2.dilate(rend.astype(np.uint8), np.ones((sc.k(3), sc.k(3)))).astype(bool), rend


# ---- metric depth ----
def metric_depth(frame, emb, cam, rmask, min_ruler_px=None):
    """First-frame z-depth (H, W) in metres and a report: stereo if STEREO is set, else monocular."""
    if STEREO is not None:
        return stereo_depth(frame, emb, cam, rmask, min_ruler_px)
    return mono_depth(frame, emb, cam, rmask, min_ruler_px)


def stereo_depth(frame, emb, cam, rmask, min_ruler_px=None):
    """STEREO resampled to the frame (rectified left view, same field of view). Pixels without stereo support (the
    left border the right camera does not see, occlusions) take the monocular depth, scaled onto the stereo depth.
    The robot-ruler error is reported but not applied: it measures how well the camera agrees with the stereo depth."""
    min_ruler_px = sc.cnt(200) if min_ruler_px is None else min_ruler_px
    H, W = frame.shape[:2]
    z = cv2.resize(STEREO.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    if np.nanmedian(z) > 20:  # millimetres (the ZED default unit) would silently fall back to monocular depth
        raise ValueError(f"stereo depth median {np.nanmedian(z):.0f}: expected metres")
    ok = np.isfinite(z) & (z > 0.05) & (z < 10)
    zm, minfo = mono_depth(frame, emb, cam, rmask, min_ruler_px)
    fit = ok & ~rmask
    s = float(np.median(z[fit] / np.maximum(zm[fit], 1e-3))) if fit.sum() > sc.cnt(100) else 1.0
    z = np.where(ok, z, zm * s).astype(np.float32)
    zr, part = emb.render(cam)
    core = cv2.erode(((part >= 0) & rmask).astype(np.uint8), np.ones((sc.k(5), sc.k(5)))).astype(bool) & ok
    info = dict(source="stereo", stereo_valid_frac=float(ok.mean()), mono_fill_scale=s, mono=minfo,
                ruler_px=int(core.sum()), scale=1.0,
                ruler_med_err_m=float(np.median(np.abs(z[core] - zr[core]))) if core.sum() >= min_ruler_px else None,
                ruler_med_ratio=float(np.median(zr[core] / z[core])) if core.sum() >= min_ruler_px else None)
    return z, info


def mono_depth(frame, emb, cam, rmask, min_ruler_px=None):
    """Metric monocular depth, rescaled by one factor so the robot pixels match the rendered robot depth. With too
    little robot in view the model's own metric scale is kept."""
    min_ruler_px = sc.cnt(200) if min_ruler_px is None else min_ruler_px
    # the depth model assumes its training field of view; metric depth scales with the true focal length
    s0 = float(cam.K[0, 0] / (0.5 * cam.W / np.tan(np.deg2rad(MODEL_HFOV_DEG / 2))))
    z = perception.metric_depth(frame) * s0
    zr, part = emb.render(cam)
    core = cv2.erode(((part >= 0) & rmask).astype(np.uint8), np.ones((sc.k(5), sc.k(5)))).astype(bool)
    b = 8  # parts cut by the image border (e.g. the base right under the camera) make a poor ruler
    core[:b], core[-b:], core[:, :b], core[:, -b:] = False, False, False, False
    info = dict(ruler_px=int(core.sum()), focal_scale=s0, scale=1.0, ruler_med_err_m=None)
    if core.sum() >= min_ruler_px:
        r = zr[core] / np.maximum(z[core], 1e-3)
        s = float(np.clip(np.median(r), 0.6, 1.6))
        z = z * s
        info.update(scale=s, ruler_spread=float(np.percentile(r, 75) - np.percentile(r, 25)),
                    ruler_med_err_m=float(np.median(np.abs(z[core] - zr[core]))))
    return z.astype(np.float32), info


def _inverse_depth_fit(depth, target, sel):
    """Least-squares (a, b) of 1 / target = a / depth + b on the pixels `sel`."""
    A = np.stack([1.0 / depth[sel], np.ones(sel.sum())], 1)
    return np.linalg.lstsq(A, 1.0 / target[sel], rcond=None)[0]


def _apply_inverse_depth(depth, a, b):
    return (1.0 / np.maximum(a / depth + b, 1e-3)).astype(np.float32)


# ---- support surface ----
def support_and_correct(depth, cam, rmask, tcp_zmin=None, min_frac=0.02):
    """Find the support surface (horizontal: gravity is known in the robot base frame) and use it to correct the
    depth map. The lowest effector point over the action, tcp_zmin, bounds the support height from above.

    Support pixels get an exact depth from the ray / plane intersection; an affine inverse-depth correction fitted on
    them is applied to the whole map, which removes most of the depth model's global scale / tilt error.
    Returns (depth, plane (4,), report)."""
    H, W = depth.shape
    rays_o, rays_d = cam.rays()
    info = {}
    # support height if no horizontal surface is found at all (the loop below breaks before estimating one)
    h = KNOWN_SUPPORT_Z if KNOWN_SUPPORT_Z is not None else (tcp_zmin - 0.01 if tcp_zmin is not None else 0.0)
    for it in range(2):
        P = cam.unproject(depth)
        gy, gx = np.gradient(P, axis=(0, 1))
        nrm = np.cross(gx, gy)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
        horiz = (np.abs(nrm[..., 2]) > 0.9) & ~rmask & (depth < 3.0)
        if horiz.sum() < sc.cnt(50) and KNOWN_SUPPORT_Z is not None:
            # no horizontal surface in a badly scaled monocular depth: take the loosely horizontal non-robot pixels,
            # the known support height fixes the rest
            horiz = (np.abs(nrm[..., 2]) > 0.6) & ~rmask & (depth < 3.0)
        if horiz.sum() < sc.cnt(50):
            break
        z = P[..., 2][horiz]
        bins = np.arange(z.min() - 0.02, z.max() + 0.02, 0.01)
        hist, edges = np.histogram(z, bins)
        hist = np.convolve(hist, [1, 2, 3, 2, 1], "same")
        centers = (edges[:-1] + edges[1:]) / 2
        peaks = [i for i in range(1, len(hist) - 1) if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]
                 and hist[i] > min_frac * horiz.sum() * 3]
        if tcp_zmin is not None:
            # the effector goes down to the surface it works on: among the strong horizontal surfaces, take the one
            # closest to the lowest effector point (monocular depth can put the table a few cm above it)
            strong = [i for i in peaks if hist[i] >= 0.1 * max(hist[j] for j in peaks)] if peaks else []
            ok = [i for i in strong if tcp_zmin - 0.30 <= centers[i] <= tcp_zmin + 0.12]
            pick = min(ok, key=lambda i: abs(centers[i] - (tcp_zmin - 0.01))) if ok else None
        else:
            pick = None
        if pick is not None:
            h = float(centers[pick])
        elif tcp_zmin is not None:  # no surface in range: the effector went down to it, so it sits just below
            h = tcp_zmin - 0.01
        else:
            h = float(centers[max(peaks, key=lambda i: hist[i]) if peaks else int(np.argmax(hist))])
        table = horiz & (np.abs(P[..., 2] - h) < 0.04)
        if table.sum() > sc.cnt(50):
            h = float(np.median(P[..., 2][table]))
        if tcp_zmin is not None:
            h = min(h, tcp_zmin + 0.01)  # the effector cannot go through the support
            table = horiz & (np.abs(P[..., 2] - h) < 0.04)
        if KNOWN_SUPPORT_Z is not None:
            # known support height: the surface found above only selects the support pixels; their exact depth comes
            # from the known plane, so the correction below also fixes the depth scale
            h = KNOWN_SUPPORT_Z
        t = (h - rays_o[2]) / np.where(np.abs(rays_d[:, 2]) > 1e-6, rays_d[:, 2], 1e-6)
        zpl = (t * cam.ray_z).reshape(H, W)
        sel = table & (zpl > 0.05) & (zpl < 3.0)
        if sel.sum() < sc.cnt(50):
            break
        a, b = _inverse_depth_fit(depth, zpl, sel)
        if a <= 0:
            break
        depth = _apply_inverse_depth(depth, a, b)
        info[f"iter{it}"] = dict(h=h, a=float(a), b=float(b), table_px=int(sel.sum()))
    info["h"] = h
    info["tcp_zmin"] = tcp_zmin
    return depth, np.array([0.0, 0.0, 1.0, -h]), info


def support_mask(frame, cam, rmask, P, height, far, work_xy, d0):
    """Mask of the surface the effector works on: SAM2 on the support under the lowest effector point `work_xy`
    (inside the smallest support box containing it, else a point prompt there); fallback: the large regions of
    plane-consistent pixels within reach. Returns (mask, report)."""
    H, W = frame.shape[:2]
    mask = None
    if work_xy is not None:
        uv, zc = cam.project(np.array([[work_xy[0], work_xy[1], -d0]]))
        u, v = uv[0]
        if zc[0] > 0 and 0 <= u < W and 0 <= v < H:
            boxes = [d["box"] for d in perception.detect(frame, SUPPORT_PHRASES, box_thr=0.25)
                     if d["box"][0] <= u <= d["box"][2] and d["box"][1] <= v <= d["box"][3]]
            if boxes:
                bt = min(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                sm = perception.segment(frame, boxes=[bt])[0]
            else:
                sm = perception.segment(frame, points=[[u, v]], labels=[1])[0]
            mask = sm & ~rmask
            info = dict(work_px=[float(u), float(v)], via="box" if boxes else "point")
    if mask is None or mask.sum() < sc.cnt(200):
        reach = np.linalg.norm(P[..., :2], axis=-1) < 1.2
        on = (np.abs(height) < 0.03) & ~rmask & ~far & reach
        on = cv2.morphologyEx(on.astype(np.uint8), cv2.MORPH_OPEN, np.ones((sc.k(3), sc.k(3))))
        nlab, lab, stats, _ = cv2.connectedComponentsWithStats(on, 8)
        keep = [i for i in range(1, nlab) if stats[i, 4] >= 0.15 * stats[1:, 4].max()] if nlab > 1 else []
        mask = np.isin(lab, keep)
        info = dict(via="plane")
    return mask, info


# ---- objects ----
def _complete(vis, cam, mask, n_up, plane_d):
    """Shape completion that keeps the first-frame silhouette: the visible surface, a copy of it pushed back along each
    camera ray by the object's thickness (its smaller horizontal extent) but no further than where the ray meets the
    support, and side points from the visible surface down to the support that project inside the first-frame mask."""
    ext = np.ptp(vis[:, :2], 0)
    thick = float(np.clip(ext.min(), 0.01, THICK_MAX))
    ray = vis - cam.T_wc[:3, 3]
    ray /= np.linalg.norm(ray, axis=1, keepdims=True) + 1e-9
    hgt = vis @ n_up + plane_d
    down = np.maximum(-(ray @ n_up), 1e-6)  # height lost per metre along the ray
    t = np.where(ray @ n_up < 0, np.clip(hgt / down, 0, thick), thick)
    back = vis + t[:, None] * ray
    side = vis - (hgt * np.random.default_rng(0).uniform(0, 1, len(vis)))[:, None] * n_up[None]
    uv, z = cam.project(side)
    u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    inb = (z > 0) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
    keep = np.zeros(len(side), bool)
    keep[inb] = mask[v[inb], u[inb]]
    return np.concatenate([vis, back, side[keep]])


def build(frame, emb, cam, arm_q, grip, phrases=(), tcp_zmin=None, work_xy=None, grasp_px=(), min_px=25,
          max_extent=0.45, verbose=True):
    """Initial scene state from the first frame (H, W, 3) uint8, the first robot state and the calibrated camera.

    phrases: extra SAM 3 concept prompts. tcp_zmin / work_xy: height / (x, y) of the lowest effector point over the
    action (bounds the support height / locates the support surface). grasp_px: the action's closing points in the
    first frame, segmented first as grasp proposals. min_px: smallest object (pixels); max_extent: largest object (m),
    0.8 m for grasp proposals (a towel). Pixel constants, min_px included, are in 320x180 pixels (sim/scale.py).
    Returns (objects, Background, report, dbg); dbg["obj_masks"] is the first-frame mask of each object."""
    min_px = sc.cnt(min_px)
    emb.set_state(arm_q, grip)
    H, W = frame.shape[:2]
    rmask, rend = robot_mask(frame, emb, cam)
    depth, dinfo = metric_depth(frame, emb, cam, rmask)
    depth, plane, pinfo = support_and_correct(depth, cam, rmask, tcp_zmin)
    dinfo["support"] = pinfo
    P = cam.unproject(depth)
    far = depth > 3.0
    n, d0 = plane[:3], plane[3]
    h = P @ n + d0  # height above the plane
    rays_o, rays_d = cam.rays()
    denom = rays_d @ n
    tpl = -(rays_o @ n + d0) / np.where(np.abs(denom) > 1e-6, denom, 1e-6)
    zpl = (tpl * cam.ray_z).reshape(H, W)  # z-depth of the support plane per pixel
    sup_mask, sup_info = support_mask(frame, cam, rmask, P, h, far, work_xy, d0)
    dinfo["support_mask"] = dict(sup_info, px=int(sup_mask.sum()))
    # the support surface gets the exact ray / plane depth; refit the global depth correction on it
    sel = sup_mask & (zpl > 0.05) & (zpl < 3.0)
    if sel.sum() > sc.cnt(200):
        a, b = _inverse_depth_fit(depth, zpl, sel)
        if a > 0:
            depth = _apply_inverse_depth(depth, a, b)
        depth = np.where(sel, zpl, depth).astype(np.float32)
        h = cam.unproject(depth) @ n + d0

    # things resting on the support show up as holes inside its hull
    hull = np.zeros((H, W), np.uint8)
    if sup_mask.any():
        ys, xs = np.nonzero(sup_mask)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32)), 1)
    holes = hull.astype(bool) & ~sup_mask & ~rmask
    holes = cv2.morphologyEx(holes.astype(np.uint8), cv2.MORPH_OPEN, np.ones((sc.k(3), sc.k(3))))
    spill = sup_mask & ~cv2.dilate(holes, np.ones((sc.k(3), sc.k(3)))).astype(bool)  # SAM spill onto the support
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    sup_area = max(int(sup_mask.sum()), 1)
    boxes = [[stats[i, 0], stats[i, 1], stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]] for i in range(1, nlab)
             if min_px <= stats[i, 4] <= 0.25 * sup_area]
    # large holes are usually several touching things (cups next to a towel) or a container (a sink with items in
    # it): split them with SAM2 point prompts on a grid inside the hole
    pts_prompts = []
    for i in range(1, nlab):
        if stats[i, 4] < sc.cnt(400):
            continue
        comp = cv2.erode((lab == i).astype(np.uint8), np.ones((sc.k(5), sc.k(5)))).astype(bool)
        ys, xs = np.nonzero(comp)
        if len(ys) == 0:
            continue
        stride = max(sc.px(8), int(np.sqrt(len(ys) / 24)))
        g = comp & ((np.arange(H)[:, None] % stride) == 0) & ((np.arange(W)[None] % stride) == 0)
        gy, gx = np.nonzero(g)
        pts_prompts += [(int(x), int(y)) for x, y in zip(gx, gy)]
    # SAM 3 concept instances find whole objects far more reliably than Grounding DINO "object" boxes
    sam3_masks, part_flags = [], []
    if config.cfg.sam3_proposals:
        part_phrases = SAM3_PART_PHRASES if config.cfg.parts else []
        for d in _sam3_concepts(frame, SAM3_PHRASES + part_phrases + list(phrases)):
            m = d["mask"] & ~rmask
            if (min_px <= m.sum() <= max(0.25 * sup_area, 0.05 * H * W)
                    and (d["mask"] & rmask).sum() < 0.5 * d["mask"].sum()):
                sam3_masks.append(m)
                part_flags.append(d["phrase"] in SAM3_PART_PHRASES)
        if config.cfg.parts_nested:
            # a part-prompt instance is kept only inside a generic object instance (a lid in a pot), not on furniture
            # (cabinet doors, drawer fronts, handles of the room): >= 80% of it within another, larger kept mask
            gen = [m for m, pf in zip(sam3_masks, part_flags) if not pf]
            sam3_masks = [m for m, pf in zip(sam3_masks, part_flags)
                          if not pf or any((m & g).sum() >= 0.8 * m.sum() and g.sum() > m.sum() for g in gen)]
    else:  # Grounding DINO boxes whose bottom centre lies on the support
        for d in perception.detect(frame, ["object"] + list(phrases), box_thr=0.3):
            x0, y0, x1, y1 = d["box"]
            foot = (int(np.clip((x0 + x1) / 2, 0, W - 1)), int(np.clip(y1 - 2, 0, H - 1)))
            if hull[foot[1], foot[0]] and (x1 - x0) * (y1 - y0) < 0.25 * H * W:
                boxes.append(d["box"])
    cands = []
    for bx in boxes:
        m = perception.segment(frame, boxes=[bx])[0] & ~rmask & ~spill
        if min_px <= m.sum() <= 0.25 * sup_area:
            cands.append(m)
    for u, v in pts_prompts:
        m = perception.segment(frame, points=[[u, v]], labels=[1])[0] & ~rmask & ~spill
        if min_px <= m.sum() <= 0.25 * sup_area and m[v, u]:
            cands.append(m)
    cands += sam3_masks
    cands.sort(key=lambda m: m.sum())  # small first: a pepper on a plate becomes its own object
    # action-conditioned proposals: wherever the effector closes, something is being grasped. They are claimed before
    # everything else (the detectors may have missed them): the smallest SAM 3 instance at the closing point, else
    # a SAM2 point prompt there
    gcands = []
    for u, v in grasp_px:
        if not (0 <= u < W and 0 <= v < H) or rmask[v, u]:
            continue
        near = [m for m in sam3_masks if cv2.dilate(m.astype(np.uint8), np.ones((sc.k(9), sc.k(9))))[v, u]]
        if near:
            gcands.append(min(near, key=lambda m: m.sum()))
            continue
        m = perception.segment(frame, points=[[u, v]], labels=[1])[0] & ~rmask
        if (m & ~spill)[v, u]:  # else the support mask swallowed the object: keep SAM's own mask
            m = m & ~spill
        if min_px <= m.sum() <= max(0.25 * sup_area, 0.05 * H * W) and m[v, u]:
            gcands.append(m)
    n_grasp_props = len(gcands)

    objs, obj_masks, used = [], [], np.zeros((H, W), bool)
    for ci, m0 in enumerate(gcands + cands):
        m = m0 & ~used
        if m.sum() < min_px or m.sum() < 0.5 * m0.sum():  # mostly claimed already: duplicate proposal
            continue
        # anchor to the support: the object's lowest image rows touch the plane; keep the model's relative depth
        ys, xs = np.nonzero(m)
        foot = ys >= np.percentile(ys, 90)
        zc = np.median(zpl[ys[foot], xs[foot]])
        if not (0.05 < zc < 3.0):
            continue
        zo = depth[m] - np.median(depth[ys[foot], xs[foot]]) + zc
        zo = np.minimum(zo, zpl[m])  # nothing below the support
        dm = np.zeros((H, W), np.float32)
        dm[m] = zo
        vis = cam.unproject(dm)[m]
        if np.ptp(vis, 0).max() > (0.8 if ci < n_grasp_props else max_extent):  # shelf, wall, drawer front, ...
            continue
        used |= m
        obj_masks.append(m)
        full = _complete(vis, cam, m, n, d0)
        sel = np.random.default_rng(len(objs)).choice(len(full), min(sc.cnt(N_PTS), len(full)), replace=False)
        rad = float(np.clip(0.9 * np.median(zo) / cam.K[0, 0], 0.002, 0.02))  # ~ one pixel footprint
        objs.append(Obj(id=len(objs), name=f"obj{len(objs)}", pts=full[sel], radius=rad, pix=int(m.sum())))
        depth = np.where(m, dm, depth)

    # static background: robot and objects removed, the support plane under objects, inpainting elsewhere
    hole = cv2.dilate((rmask | used).astype(np.uint8), np.ones((sc.k(5), sc.k(5))))
    bg = depth.copy()
    on_plane = used & (zpl > 0) & (zpl < 3.0)
    scale = max(float(np.percentile(depth, 99)), 1e-3)
    inp = cv2.inpaint((np.clip(bg / scale, 0, 1) * 65535).astype(np.uint16), hole, sc.px(5),
                      cv2.INPAINT_TELEA).astype(np.float32) / 65535 * scale
    bg = np.where(hole > 0, inp, bg)
    bg = np.where(on_plane, zpl, bg)
    bgP = cam.unproject(bg)
    hb = bgP @ n + d0
    # support label from the segmented surface (+ the plane under objects), not a per-pixel height threshold on noisy
    # monocular depth, which gives a speckled map
    sem = cv2.morphologyEx(((sup_mask & (np.abs(hb) < 0.05)) | on_plane).astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((sc.k(5), sc.k(5))))
    support = bgP[(hb > -0.02) & (bg < 3.0)].reshape(-1, 3)
    background = Background(depth=bg, pts=bgP.reshape(-1, 3), sem=sem, plane=plane, support_pts=support[::2])
    info = dict(depth=dinfo, plane=plane.tolist(), n_objects=len(objs), obj_px=[o.pix for o in objs],
                grasp_proposals=dict(points=[list(map(int, p)) for p in grasp_px], kept=n_grasp_props),
                robot_px=int(rmask.sum()), config=config.snapshot())
    if config.cfg.parts:  # part hierarchy: object a is a part of b if >= 50% of a's mask lies in b's mask dilated
        k = np.ones((sc.k(9), sc.k(9)), np.uint8)
        dil = [cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in obj_masks]
        info["parts"] = [[int(objs[a].id), int(objs[b].id)] for a in range(len(obj_masks))
                         for b in range(len(obj_masks))
                         if a != b and obj_masks[a].sum() < obj_masks[b].sum()
                         and (obj_masks[a] & dil[b]).sum() >= 0.5 * obj_masks[a].sum()]
    if verbose:
        print("init:", info, flush=True)
    dbg = dict(rmask=rmask, rend=rend, depth=depth, height=h, used=used, obj_masks=obj_masks, support=sup_mask,
               holes=holes.astype(bool))
    return objs, background, info, dbg
