"""Camera calibration against the robot: refine the camera extrinsics, focal length and radial distortion (and the
gripper mount yaw) so that the forward-kinematics render of the robot matches the robot segmented in the video.

The robot is the one thing in every frame whose 3D shape and pose are known exactly, so it is the calibration target
(and later the metric ruler of the monocular depth, sim/init_state.py). Objective: truncated symmetric chamfer distance
between the rendered robot silhouette and the segmented robot mask, averaged over frames with different arm poses;
derivative-free (Powell, then Nelder-Mead) because the silhouette renderer is not differentiable.
"""

from __future__ import annotations

from functools import partial

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from simpwm import perception
from simpwm.core.camera import Camera
from simpwm.core.raster_gpu import PosedRobot
from simpwm.perception import sam3
from simpwm.sim import scale as sc

ROBOT_PHRASES = ["robot arm", "robot gripper"]  # Grounding DINO prompts for the robot


def rendered_robot(emb, cam):
    """Silhouette (H, W) bool of the robot in its current state and its bounding box [x0, y0, x1, y1], or None as the
    box when the robot covers fewer than 20 pixels (at 320x180; pixel constants follow sim/scale.py)."""
    rend = emb.render(cam)[1] >= 0
    if rend.sum() < sc.cnt(20):
        return rend, None
    ys, xs = np.nonzero(rend)
    return rend, np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


def box_overlap(b, ref):
    """Fraction of box b's area inside box ref (boxes [x0, y0, x1, y1])."""
    x0, y0 = max(b[0], ref[0]), max(b[1], ref[1])
    x1, y1 = min(b[2], ref[2]), min(b[3], ref[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))


def robot_masks(frames, emb, ep, idx, cam, gate=True, source="sam3"):
    """Robot masks of frames `idx` (ep: dict with arm_q, grip) near the robot rendered with the roughly calibrated
    camera. Returns {k: (H, W) bool mask} for the frames where the robot is found.

    source "sam3": SAM 3 "robot arm" instances overlapping the render (cables and held objects are left out of the
    robot more often); "sam2": Grounding DINO boxes overlapping the render, segmented by SAM2 (used by the robot-IoU
    metrics). gate=False is for a camera that may be far off (a wrong logged extrinsic): confident instances are kept
    regardless of the render, and the mask is not clipped to the render's neighbourhood."""
    out = {}
    for k in idx:
        emb.set_state(ep["arm_q"][k], ep["grip"][k])
        rend, rb = rendered_robot(emb, cam)
        if rb is None:
            continue
        pad = 0.25 * max(rb[2] - rb[0], rb[3] - rb[1]) + sc.px(10)
        rb_d = rb + [-pad, -pad, pad, pad]
        if source == "sam3":
            ins = sam3.segment_concepts(frames[k], ["robot arm"], threshold=0.3)
            ms = [d["mask"] for d in ins if (box_overlap(d["box"], rb_d) > 0.3 if gate else d["score"] > 0.5)]
            if not ms:
                continue
            m = np.any(ms, 0)
        else:
            dets = perception.detect(frames[k], ROBOT_PHRASES)
            boxes = [d["box"] for d in dets if (box_overlap(d["box"], rb_d) > 0.3 if gate else d["score"] > 0.35)]
            if not boxes:
                continue
            m = perception.segment(frames[k], boxes=boxes).any(0)
        if gate:  # segmentation spill onto chairs / stands behind the arm: stay near the render
            kk = int(0.15 * max(rb[2] - rb[0], rb[3] - rb[1])) + sc.px(10)
            m &= cv2.dilate(rend.astype(np.uint8), np.ones((2 * kk + 1, 2 * kk + 1))).astype(bool)
        out[k] = m
    return out


def _dt(mask, trunc):
    return np.minimum(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3), trunc)


def apply(cam, x):
    """Camera updated by x = [rotvec (3) and translation (3, m) in the camera frame, log focal scale, (k1, k2) added
    to the current radial distortion]; shorter x leave the remaining parameters unchanged (6 = pose only)."""
    T = cam.T_wc.copy()
    dR = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, :3] = cam.T_wc[:3, :3] @ dR
    T[:3, 3] = cam.T_wc[:3, 3] + cam.T_wc[:3, :3] @ x[3:6]
    K = cam.K.copy()
    if len(x) > 6:
        K[0, 0] *= np.exp(x[6])
        K[1, 1] *= np.exp(x[6])
    dist = cam.dist if len(x) < 9 else (cam.dist[0] + x[7], cam.dist[1] + x[8])
    return Camera(K, T, cam.H, cam.W, dist)


def _silhouette_cost(cam, masks, ks, scale, trunc):
    """Calibration objective at `scale` times the image resolution: cost(x, robot) is the mean over frames ks of the
    truncated symmetric chamfer distance (pixels) between the robot (PosedRobot) rendered with apply(cam, x) and its
    mask."""
    H, W = int(cam.H * scale), int(cam.W * scale)
    small = {k: cv2.resize(masks[k].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) for k in ks}
    dts = {k: _dt(small[k], trunc * scale) for k in ks}

    def cost(x, robot):
        c = apply(cam, x).scaled(H, W)
        # a distortion that folds inside the image (radius past its turning point) is not a lens: reject it
        inval = 1.0 - c.valid_rays().mean()
        if inval > 0:
            return trunc * scale * 2 * (1 + inval)
        tot = 0.0
        for k, r in zip(ks, robot.masks(c)):
            tot += (dts[k][r].mean() + _dt(r, trunc * scale)[small[k]].mean()) if r.any() else trunc * scale * 2
        return tot / len(ks)

    return cost


def refine(emb, ep, masks, cam, scale=0.5, trunc=12.0, yaws=None, coarse=False, n_params=9, maxfev=1500, top=3):
    """Fit the camera (n_params of `apply`) to the robot masks {k: mask} of frames k of ep (dict with arm_q, grip),
    for each gripper mount yaw in `yaws` (None: keep the current one); the best yaw is set on emb. coarse=True starts
    from the `top` best points of a coarse grid (for a far-off initial camera). Returns (camera, report)."""
    ks = sorted(k for k in masks if masks[k].sum() >= sc.cnt(50))  # empty / tiny masks would make the cost NaN
    cost = _silhouette_cost(cam, masks, ks, scale, trunc)
    states = [(ep["arm_q"][k], ep["grip"][k]) for k in ks]
    best = None
    for yaw in (yaws if yaws is not None else [None]):
        if yaw is not None:
            emb.set_gripper_yaw(yaw)
        f = partial(cost, robot=PosedRobot(emb, states))
        x0 = np.zeros(n_params)
        if coarse:
            starts = _coarse_starts(emb, ep, masks, cam, ks, top=top)
            xs = [np.pad(x, (0, max(0, n_params - 7)))[:n_params] for x in starts]
        else:
            xs = [x0]
        rs = [minimize(f, xi, method="Powell", options=dict(maxfev=maxfev, xtol=1e-4, ftol=1e-4)) for xi in xs]
        r = min(rs, key=lambda q: q.fun)
        r2 = minimize(f, r.x, method="Nelder-Mead", options=dict(maxfev=maxfev, xatol=1e-5, fatol=1e-5))
        res = (r2.fun, r2.x, yaw, f(x0))
        print(f"  yaw={yaw} cost {res[3]:.2f} -> {res[0]:.2f}  x={np.round(r2.x, 4)}", flush=True)
        if best is None or res[0] < best[0]:
            best = res
    if best[2] is not None:
        emb.set_gripper_yaw(best[2])
    cam2 = apply(cam, best[1])
    return cam2, dict(cost0=best[3], cost=best[0], x=best[1].tolist(), gripper_yaw=best[2], dist=list(cam2.dist))


def _coarse_starts(emb, ep, masks, cam, ks, scale=0.25, trunc=12.0, top=3):
    """Grid over camera rotation (about its x / y axes) and translation at low resolution, to escape the local minimum
    of a far-off initial extrinsic. Returns the best `top` start vectors (7 parameters)."""
    cost = partial(_silhouette_cost(cam, masks, ks, scale, trunc),
                   robot=PosedRobot(emb, [(ep["arm_q"][k], ep["grip"][k]) for k in ks]))
    a, t = (-0.15, 0.0, 0.15), (-0.12, 0.0, 0.12)
    grid = [np.array([rx, ry, 0, tx, ty, tz, 0]) for rx in a for ry in a for tx in t for ty in t for tz in t]
    cs = [cost(x) for x in grid]
    order = np.argsort(cs)[:top]
    print(f"  coarse: start cost {cost(np.zeros(7)):.2f}, best {[round(cs[i], 2) for i in order]}", flush=True)
    return [grid[i] for i in order]


def iou_report(emb, ep, masks, cam):
    """Robot IoU (rendered vs mask) per mask {k: mask} of frames k of ep. Returns (mean, list)."""
    ious = []
    for k, m in masks.items():
        emb.set_state(ep["arm_q"][k], ep["grip"][k])
        _, part = emb.render(cam)
        r = part >= 0
        ious.append((r & m).sum() / max(1, (r | m).sum()))
    return float(np.mean(ious)), ious


# ---- full calibration of a camera setup (droid/setup_calib.py) ----
SAM3_POLISH_MAX_SAM2_DROP = 0.02


def calibrate_camera(frames, arm_q, grip, cam, emb, n=10, yaws=(0, 90)):
    """Camera calibration from robot silhouettes only (no object information), on a calibration sequence of the robot
    moving in front of this camera: SAM2 robot masks, a coarse search when the initial camera is far off, a fit per
    gripper mount yaw, a second fit on masks re-collected with the fitted camera, then sam3_polish.
    Returns (camera, report)."""
    T = len(frames)
    idx = np.linspace(0, T - 1, n).astype(int).tolist()
    ep = dict(arm_q=arm_q, grip=grip)
    masks = robot_masks(frames, emb, ep, idx, cam, source="sam2")
    iou0 = iou_report(emb, ep, masks, cam)[0] if masks else 0.0
    coarse = iou0 < 0.3  # logged extrinsic far off: masks must not depend on it, and a local search is not enough
    if coarse:
        masks = robot_masks(frames, emb, ep, idx, cam, gate=False, source="sam2")
    if len(masks) < 2:
        return cam, dict(skipped=True, n_masks=len(masks))
    iou0 = iou_report(emb, ep, masks, cam)[0]
    cam2, info = refine(emb, ep, masks, cam, yaws=list(yaws), coarse=coarse)
    # re-collect the masks with the camera now roughly right (less spill, right boxes), and polish
    m2 = robot_masks(frames, emb, ep, idx, cam2, source="sam2")
    if len(m2) >= 2:
        masks = m2
        cam2, info2 = refine(emb, ep, masks, cam2, yaws=[info["gripper_yaw"]])
        info = dict(info2, coarse_stage=info)
    iou1, _ = iou_report(emb, ep, masks, cam2)
    info = dict(info, iou_before=iou0, iou_after=iou1, n_masks=len(masks), coarse=coarse)
    return sam3_polish(frames, ep, cam2, emb, info)


def sam3_polish(frames, ep, cam, emb, info):
    """Last calibration stage: SAM 3 robot masks (gated by the camera so far) and one more local fit, gripper yaw
    kept. SAM 3 leaves cables and held objects out of the robot more often than SAM2, so the camera is no longer
    pulled towards them. The polished camera is kept only if, on these frames, its SAM 3 robot IoU goes up and its
    SAM2 robot IoU drops by at most SAM3_POLISH_MAX_SAM2_DROP (guards against fits to bad SAM 3 masks)."""
    idx = list(range(len(frames)))
    m3 = robot_masks(frames, emb, ep, idx, cam, source="sam3")
    if len(m3) < 2:
        return cam, dict(info, sam3_polish=dict(skipped=True, n_masks=len(m3)))
    yaw = info.get("gripper_yaw", 90)
    emb.set_gripper_yaw(yaw)
    m2 = robot_masks(frames, emb, ep, idx, cam, source="sam2")
    before = iou_report(emb, ep, m3, cam)[0]
    before2 = iou_report(emb, ep, m2, cam)[0] if m2 else None
    cam3, info3 = refine(emb, ep, m3, cam, yaws=[yaw])
    after = iou_report(emb, ep, m3, cam3)[0]
    after2 = iou_report(emb, ep, m2, cam3)[0] if m2 else None
    accepted = after > before and (before2 is None or after2 >= before2 - SAM3_POLISH_MAX_SAM2_DROP)
    rep = dict(n_masks=len(m3), iou_before=before, iou_after=after, sam2_before=before2, sam2_after=after2,
               accepted=bool(accepted), x=info3["x"])
    return (cam3 if accepted else cam), dict(info, sam3_polish=rep)
