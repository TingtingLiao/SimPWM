"""Rendering: scene state -> proxy frame (z-depth, category map, contact map and optionally the RGB of the dynamic
things), plus the mapping of the categories to the semantic ids of the video model's control input.

Categories: 0 background, 1 support surface, 2 robot arm, 3 effector, 4 effector pads, 5 + id for object id
(constants in core/state.py). Objects are z-buffered disk splats of their points; the static background is the
first-frame background depth (nearest-upsampled when rendering the same camera at a higher resolution).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from simpwm import config
from simpwm.core.camera import Camera
from simpwm.core.embodiment import PART_ARM, PART_GRIPPER, PART_PAD
from simpwm.core.state import C_ARM, C_BG, C_EFF, C_OBJ0, C_PAD, C_SUPPORT, is_robot  # noqa: F401  (re-exported)


def splat(cam, pts, rad, depth, label, value, max_px=4, index=None):
    """z-buffered disk splats (radius rad metres, at most max_px pixels) of world points into depth / label, in place.
    `index` (H, W) int32, optional: receives the index of the point that won each pixel (for per-point colours)."""
    uv, z = cam.project(pts)
    ok = (z > 0.05) & (uv[:, 0] >= -max_px) & (uv[:, 0] < cam.W + max_px) & (uv[:, 1] >= -max_px) \
        & (uv[:, 1] < cam.H + max_px)
    pid = np.nonzero(ok)[0]
    uv, z = uv[ok], z[ok]
    r = np.clip(cam.K[0, 0] * rad / z, 0.7, max_px)
    for dy in range(-max_px, max_px + 1):
        for dx in range(-max_px, max_px + 1):
            sel = dx * dx + dy * dy <= r * r
            if not sel.any():
                continue
            x = np.floor(uv[sel, 0] + dx).astype(int)
            y = np.floor(uv[sel, 1] + dy).astype(int)
            zz, ii = z[sel], pid[sel]
            inb = (x >= 0) & (x < cam.W) & (y >= 0) & (y < cam.H)
            x, y, zz, ii = x[inb], y[inb], zz[inb], ii[inb]
            flat = y * cam.W + x
            cur = depth.reshape(-1)
            order = np.argsort(-zz)  # write far first so near wins
            flat, zz, ii = flat[order], zz[order], ii[order]
            better = (cur[flat] == 0) | (zz < cur[flat])
            cur[flat[better]] = zz[better]
            label.reshape(-1)[flat[better]] = value
            if index is not None:
                index.reshape(-1)[flat[better]] = ii[better]


def color_objects(objs, masks, frame0, cam):
    """Per-point colour (Obj.rgb) of every object from the first frame: points inside the object's first-frame mask take
    the pixel they project to, the others (completed back side, points off the mask) the colour of the nearest such
    point. Objects with fewer than 3 such points stay grey."""
    H, W = frame0.shape[:2]
    for ob, mk in zip(objs, masks):
        uv, z = cam.project(ob.pts)
        u = np.clip(np.floor(uv[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.floor(uv[:, 1]).astype(int), 0, H - 1)
        col = frame0[v, u].astype(np.float32) / 255
        seen = (z > 0) & mk[v, u]
        if seen.sum() >= 3:
            col[~seen] = col[seen][cKDTree(ob.pts[seen]).query(ob.pts[~seen])[1]] if (~seen).any() else col[~seen]
            ob.rgb = col


CROP_MARGIN = 12  # px around an object's projected points: > splat radius (max_px 4) + the 5x5 closing / erosion


def _texture_from_first_frame(cam, ob, d, pidx, valid, tex, cam_full=None):
    """Object colours at full resolution (config.cfg.hires). Each object pixel (depth d, winning point pidx) is moved
    back to the first frame with its point's displacement (pts - pts0) and samples the full-resolution first frame
    there, if it lands on the object's full-resolution first-frame mask. cam may be a crop camera of the frame: then
    cam_full is the camera of the whole frame. Returns (rgb (H, W, 3) float in [0, 1], used (H, W) bool)."""
    H, W = d.shape
    cam_full = cam if cam_full is None else cam_full
    Hf, Wf = tex["frame0"].shape[:2]
    pts0, mask0 = tex["pts0"].get(ob.id), tex["masks0"].get(ob.id)
    used = np.zeros((H, W), bool)
    rgb = np.zeros((H, W, 3), np.float32)
    if pts0 is None or mask0 is None or len(pts0) != len(ob.pts) or not valid.any():
        return rgb, used
    v, u = np.nonzero(valid)
    Xc = cam._pix_dirs(u + 0.5, v + 0.5) * d[v, u][:, None]
    X = Xc @ cam.T_wc[:3, :3].T + cam.T_wc[:3, 3]
    i = pidx[v, u]
    uv0, z0 = cam_full.project(X - (ob.pts[i] - pts0[i]))
    x0, y0 = uv0[:, 0] - 0.5, uv0[:, 1] - 0.5  # pixel-centre convention of Camera.project / unproject
    xi, yi = np.round(x0).astype(int), np.round(y0).astype(int)
    ok = (z0 > 0.05) & (xi >= 0) & (xi < Wf) & (yi >= 0) & (yi < Hf)
    ok[ok] &= mask0[yi[ok], xi[ok]]
    if ok.any():
        mx = np.zeros((H, W), np.float32)
        my = np.zeros((H, W), np.float32)
        mx[v[ok], u[ok]], my[v[ok], u[ok]] = x0[ok], y0[ok]
        samp = cv2.remap(tex["frame0"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        used[v[ok], u[ok]] = True
        rgb[used] = samp[used].astype(np.float32) / 255
    return rgb, used


def _render_object_cropped(cam, ob, depth, cat, rgb, shade, tex, max_px=4):
    """render()'s per-object step on the object's bounding-box crop (config.cfg.hires; much faster at the higher
    resolution): a crop camera (principal point shifted, same rays) splats, closes the gaps, colours and textures the
    object there, and the result is composited into the full depth / cat / rgb as render() does for the full frame."""
    H, W = cam.H, cam.W
    uv, z = cam.project(ob.pts)
    ok = (z > 0.05) & (uv[:, 0] >= -max_px) & (uv[:, 0] < W + max_px) & (uv[:, 1] >= -max_px) & (uv[:, 1] < H + max_px)
    if not ok.any():
        return
    x0 = int(max(0, np.floor(uv[ok, 0].min()) - CROP_MARGIN))
    x1 = int(min(W, np.ceil(uv[ok, 0].max()) + CROP_MARGIN + 1))
    y0 = int(max(0, np.floor(uv[ok, 1].min()) - CROP_MARGIN))
    y1 = int(min(H, np.ceil(uv[ok, 1].max()) + CROP_MARGIN + 1))
    if x1 <= x0 or y1 <= y0:
        return
    K = cam.K.copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    cc = Camera(K, cam.T_wc, y1 - y0, x1 - x0, cam.dist)
    h, w = y1 - y0, x1 - x0
    d = np.zeros((h, w), np.float32)
    lab = np.full((h, w), -1, np.int16)
    pidx = np.zeros((h, w), np.int32) if shade else None
    splat(cc, ob.pts, ob.radius, d, lab, C_OBJ0 + ob.id, index=pidx, max_px=max_px)
    hit = lab >= 0
    if not hit.any():
        return
    fill = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5))).astype(bool) & ~hit
    if fill.any():
        dn = cv2.erode(np.where(hit, d, 1e3).astype(np.float32), np.ones((5, 5)))
        d[fill], lab[fill] = dn[fill], C_OBJ0 + ob.id
    col = None
    if shade:
        col = ob.rgb[pidx] if ob.rgb is not None else np.full((h, w, 3), 0.5, np.float32)
        if fill.any():
            _, (iy, ix) = distance_transform_edt(~hit, return_indices=True)
            col = col[iy, ix]
            pidx = pidx[iy, ix]
        if tex is not None:
            trgb, tused = _texture_from_first_frame(cc, ob, d, pidx, lab >= 0, tex, cam_full=cam)
            col = np.where(tused[..., None], trgb, col)
            tex.setdefault("stats", []).append((int(tused.sum()), int((lab >= 0).sum())))
    D, C = depth[y0:y1, x0:x1], cat[y0:y1, x0:x1]
    m = (lab >= 0) & ((d < D) | (D == 0))
    D[m], C[m] = d[m], lab[m]
    if shade:
        rgb[y0:y1, x0:x1][m] = col[m]


def render(cam, emb, st, bg, shade=False, tex=None):
    """Render state `st` from `cam` (the first-frame camera, possibly at another resolution) over the background bg.
    Returns z-depth (H, W) float32 (0 = nothing), category map (H, W) int16, contact map (H, W) float32 and, with
    shade=True, the RGB (H, W, 3) of the dynamic things: the shaded robot mesh and the objects in their first-frame
    colours (0 elsewhere).

    tex (config.cfg.hires, optional): dict(frame0: (H, W, 3) uint8 full-resolution first frame of `cam`, pts0: {object
    id: first-frame points}, masks0: {object id: (H, W) bool full-resolution first-frame mask}): the object colours
    are sampled from the full-resolution first frame (_texture_from_first_frame), and a background bg.depth_hi of
    `cam`'s size replaces the upsampled bg.depth."""
    H, W = cam.H, cam.W
    depth = bg.depth.copy()
    cat = np.where(bg.sem > 0, C_SUPPORT, C_BG).astype(np.int16)
    if depth.shape != (H, W):
        depth_hi = getattr(bg, "depth_hi", None) if tex is not None else None
        if depth_hi is not None and depth_hi.shape == (H, W):
            depth = depth_hi.copy()
        else:
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        cat = cv2.resize(cat, (W, H), interpolation=cv2.INTER_NEAREST)
    rgb = np.zeros((H, W, 3), np.float32) if shade else None
    if config.cfg.hires:  # the per-object step below, on each object's bounding-box crop
        for ob in st.objects:
            _render_object_cropped(cam, ob, depth, cat, rgb, shade, tex)
    for ob in () if config.cfg.hires else st.objects:
        d = np.zeros((H, W), np.float32)
        lab = np.full((H, W), -1, np.int16)
        pidx = np.zeros((H, W), np.int32) if shade else None
        splat(cam, ob.pts, ob.radius, d, lab, C_OBJ0 + ob.id, index=pidx)
        # close the gaps between splats (sparse points in near / enlarged views) with the nearest splat depth
        hit = lab >= 0
        fill = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5))).astype(bool) & ~hit
        if fill.any():
            dn = cv2.erode(np.where(hit, d, 1e3).astype(np.float32), np.ones((5, 5)))
            d[fill], lab[fill] = dn[fill], C_OBJ0 + ob.id
        if shade:  # the winning point's first-frame colour (grey without colours); filled gaps take the nearest
            col = ob.rgb[pidx] if ob.rgb is not None else np.full((H, W, 3), 0.5, np.float32)
            if fill.any():
                _, (iy, ix) = distance_transform_edt(~hit, return_indices=True)
                col = col[iy, ix]
        m = (lab >= 0) & ((d < depth) | (depth == 0))
        depth[m], cat[m] = d[m], lab[m]
        if shade:
            rgb[m] = col[m]
    emb.set_state(st.arm_q, st.grip)
    zr, part, *rob = emb.render(cam, shade=True) if shade else (*emb.render(cam),)
    m = (part >= 0) & ((zr < depth) | (depth == 0))
    depth[m] = zr[m]
    cat[m] = np.select([part[m] == PART_ARM, part[m] == PART_GRIPPER, part[m] == PART_PAD], [C_ARM, C_EFF, C_PAD])
    contact = np.zeros((H, W), np.float32)
    for oid, s in st.contacts.items():
        contact[(cat == C_OBJ0 + oid) | ((cat == C_PAD) & (s >= 1.0))] = s
    if shade:
        rgb[m] = rob[0][m]
        return depth, cat, contact, rgb
    return depth, cat, contact


# Semantic ids of the video model's control input (CWM release): void 0, sky 1, water 2, terrain 3, road 4,
# vegetation 5, building 6, infrastructure 7, human 8, animal 9, vehicle 10, prop 11.
CWM_MAP = {C_BG: 6, C_SUPPORT: 7, C_ARM: 8, C_EFF: 8, C_PAD: 8}
CWM_OBJ = 10  # every object


def to_cwm(cat):
    """Category map -> uint8 semantic id map of the control input."""
    out = np.full(cat.shape, CWM_OBJ, np.uint8)
    for k, v in CWM_MAP.items():
        out[cat == k] = v
    return out
