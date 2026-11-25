"""Proxy generation: first frame + robot states (the actions) + calibrated camera -> per-step proxy frames.

  (1) real-to-sim  init_state.build turns the first frame into objects (point sets), a static background and the
                   support plane; the action refines it (refine.py): objects the gripper closes on are snapped to the
                   gripper's depth and pinned, closures on nothing are ignored, closures on something unseen add a
                   phantom object
  (2) simulation   engine.RuleEngine steps the object states along the robot states (grasp, release, gravity, drape)
  (3) rendering    render.render draws each state as depth / category / contact maps, at the simulation resolution and
                   optionally at the reference resolution of the video model's control input (with the RGB of the
                   robot and the objects)

Embodiment-specific knowledge enters only through `Embodiment` (its JSON config); objects are class-agnostic. The
rules are switched by `simpwm.config` (ablations).
"""

from __future__ import annotations

import copy
import re

import cv2
import numpy as np

from simpwm import config
from simpwm.core.state import State
from simpwm.sim import camera_calib as calibrate
from simpwm.sim import init_state
from simpwm.sim import refine as refine_state
from simpwm.sim import render as raster
from simpwm.sim import scale as sc
from simpwm.sim.engine import RuleEngine

CLOSE_THR = 0.12  # closure above which the gripper counts as closing (RuleEngine's close_thr)

# VLM grasp phrases that name a deformable thing
SOFT_WORDS = re.compile(r"towel|cloth|rag|napkin|tissue|fabric|shirt|sock|blanket|scarf|bandana|glove|cable|rope|wire|"
                        r"string|cord|ribbon|clothes|clothing|garment", re.I)
NOT_SOFT = re.compile(r"box|container|dispenser|holder|roll\b", re.I)  # "tissue box", "napkin holder"
SOFT_MAX_HEIGHT = 0.06  # a soft object lies flat-ish on the support at t0 (a towel, not a stuffed bag)

HIRES_MATCH_IOU = 0.5  # a final object mask takes the full-resolution SAM 3 mask it came from at this IoU

# Counterfactual rollouts (droid/run_counterfactual.py): (arm_q, grip) arrays that replace the recorded robot states
# for the rollout and the renders only. Everything built before the rollout (support height bound, grasp proposals,
# snap, phantoms, empty closures) still uses the recorded states, so the scene is the unedited one and only the
# motion differs. None = off.
ROLLOUT_ACTIONS = None


def is_soft(phrase):
    return bool(phrase and SOFT_WORDS.search(phrase) and not NOT_SOFT.search(phrase))


def mark_soft(objs, masks, soft_px, plane):
    """Mark as soft the flat objects whose first-frame mask (dilated) contains a closing point the VLM named as a
    deformable thing. Returns the ids of the soft objects."""
    k = np.ones((sc.k(9), sc.k(9)), np.uint8)
    for o, m in zip(objs, masks):
        if not soft_px:
            break
        md = cv2.dilate(m.astype(np.uint8), k).astype(bool)
        hit = any(0 <= v < md.shape[0] and 0 <= u < md.shape[1] and md[v, u] for u, v in soft_px)
        o.soft = bool(hit and np.percentile(o.pts @ plane[:3] + plane[3], 95) < SOFT_MAX_HEIGHT)
    return [o.id for o in objs if o.soft]


def hires_object_masks(obj_masks, hw):
    """config.cfg.hires: the full-resolution (hw) first-frame mask of each final object mask: the full-resolution
    SAM 3 mask whose area-downsampled version matches it best (IoU >= HIRES_MATCH_IOU), clipped to the object's
    upsampled mask dilated by 2 simulation pixels; else the nearest-upsampled simulation mask.
    Returns (masks, sources)."""
    H, W = hw
    out, src = [], []
    for m in obj_masks:
        up = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        best, bi = 0.0, None
        for lo, hi in init_state.HIRES_MASKS:
            if lo.shape != m.shape:
                continue
            iou = (lo & m).sum() / max(1, (lo | m).sum())
            if iou > best:
                best, bi = iou, hi
        if bi is not None and best >= HIRES_MATCH_IOU:
            k = 2 * max(1, round(2 * W / m.shape[1])) + 1
            out.append(bi & (cv2.dilate(up.astype(np.uint8), np.ones((k, k), np.uint8)) > 0))
            src.append(dict(source="sam3_fullres", iou=round(float(best), 3)))
        else:
            out.append(up)
            src.append(dict(source="upsampled_320x180", iou=round(float(best), 3)))
    return out, src


def hires_background_depth(bg_depth, holes, masks_hi, hw, stereo):
    """config.cfg.hires: the static background depth at full resolution (hw): the stereo depth (metres, full
    resolution) where it agrees with the nearest-upsampled simulation background within 5 cm, outside the object /
    robot holes (which the simulation background fills in); the upsampled background elsewhere.
    Returns (depth, report)."""
    H, W = hw
    up = cv2.resize(bg_depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    if stereo is None:
        return up, dict(source="upsampled_320x180")
    z = stereo.astype(np.float32)
    if np.nanmedian(z) > 20:
        z = z / 1000.0
    if z.shape != (H, W):
        z = cv2.resize(z, (W, H), interpolation=cv2.INTER_NEAREST)
    excl = cv2.resize(holes.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    for m in masks_hi:
        excl |= m
    k = 2 * round(6 * W / 1280) + 1
    excl = cv2.dilate(excl.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    ok = np.isfinite(z) & (z > 0.05) & ~excl & (np.abs(z - up) < 0.05)
    return np.where(ok, z, up).astype(np.float32), dict(source="stereo_fullres+upsampled", stereo_frac=float(ok.mean()))


def closing_points(arm_q, grip, cam, emb, skip=frozenset()):
    """Where the effector closes (the lowest tcp of each closing window, except windows starting in `skip`),
    projected into the first frame: list of (u, v) pixels."""
    grasp_px = []
    for t0, t1 in refine_state.closure_windows(grip, CLOSE_THR):
        if t0 in skip:
            continue
        p, _ = refine_state.lowest_tcp(emb, arm_q, grip, t0, t1)
        uv, z = cam.project(p[None])
        if z[0] > 0:
            grasp_px.append((int(uv[0, 0]), int(uv[0, 1])))
    return grasp_px


def _rollout_held(engine_cls, emb, bg, fps, objs, arm_q, grip, empty):
    """Per step: does a rollout of the initial state hold any object (engine only, no rendering)."""
    eng = engine_cls(emb, bg, 1.0 / fps)
    eng.no_grasp_steps = empty
    st = State(t=0, arm_q=arm_q[0], grip=float(grip[0]), objects=copy.deepcopy(objs))
    held = [False]
    for t in range(1, len(arm_q)):
        st = eng.step(st, arm_q[t], grip[t])
        held.append(any(o.attached for o in st.objects))
    return held


def _rollout_inputs(arm_q, grip):
    if ROLLOUT_ACTIONS is None:
        return arm_q, grip
    q, g = ROLLOUT_ACTIONS
    assert len(q) >= len(arm_q) and len(g) >= len(grip)
    return np.asarray(q[:len(arm_q)]), np.asarray(g[:len(grip)])


def generate(frame0, arm_q, grip, cam, emb, fps, refine_cam=True, phrases=(), engine_cls=RuleEngine, snap=True,
             grasp_props=True, soft_px=(), hires=None, frame0_hi=None):
    """Proxy of one clip.

    frame0: first frame (H, W, 3) uint8 at the simulation resolution; arm_q (T, n_joints) and grip (T,) in [0, 1]: the
    robot states at `fps`; cam: the calibrated camera (camera_calib.calibrate_camera), polished on frame 0 when refine_cam.
    phrases: extra SAM 3 prompts; snap / grasp_props: action snap / grasp proposals (ablations; the snap also follows
    config.cfg.snap); soft_px: closing points (pixels) the VLM named as deformable things (perception/vlm_objects.py).
    hires: optional (camera, steps): those steps are also rendered with that camera, the same camera at the reference
    resolution of the video model's control input (Camera.scaled).
    frame0_hi (config.cfg.hires, with hires): the first frame at the resolution of hires[0]. SAM 3 proposals run on
    it, the objects get full-resolution masks, point colours and first-frame textures in the hires renders, and the
    hires background depth comes from the full-resolution stereo depth (init_state.STEREO).

    Returns dict(cam, bg, frames: (depth, cat, contact) per step, frames_hi: {step: (depth, cat, contact)} and
    shade_hi: {step: RGB of robot and objects} of the hires renders, states: per-step summary, info: report,
    dbg: from init_state.build, objects0: the objects before the rollout)."""
    use_hi = config.cfg.hires and hires is not None and frame0_hi is not None
    if config.cfg.hires and hires is not None and frame0_hi is None:
        raise RuntimeError("config.cfg.hires needs the full-resolution first frame")
    if use_hi:
        if frame0_hi.shape[:2] != (hires[0].H, hires[0].W):
            raise RuntimeError(f"hires: first frame {frame0_hi.shape[:2]} != hires camera {(hires[0].H, hires[0].W)}")
        init_state.FRAME_HI = frame0_hi
        init_state.HIRES_MASKS.clear()
    snap = snap and config.cfg.snap
    info = {}
    if refine_cam:  # small polish on frame 0 only
        ep0 = dict(arm_q=arm_q[:1], grip=grip[:1])
        masks = calibrate.robot_masks(frame0[None], emb, ep0, [0], cam)
        if masks:
            iou0, _ = calibrate.iou_report(emb, ep0, masks, cam)
            cam, cinfo = calibrate.refine(emb, ep0, masks, cam)
            iou1, _ = calibrate.iou_report(emb, ep0, masks, cam)
            info["calib"] = dict(cinfo, iou_before=iou0, iou_after=iou1)

    # (1) real-to-sim initialisation, refined by the action
    low = []  # lowest effector point per step: the lowest one bounds the support height from above
    for q, g in zip(arm_q, grip):
        emb.set_state(q, g)
        low.append(min(emb.keypoints().values(), key=lambda p: p[2]))
    low = np.array(low)
    k = int(np.argmin(low[:, 2]))
    empty = refine_state.empty_closure_steps(grip) if config.cfg.empty_closures else frozenset()
    info["empty_closure_steps"] = len(empty)
    grasp_px = closing_points(arm_q, grip, cam, emb, skip=empty)
    objs, bg, iinfo, dbg = init_state.build(frame0, emb, cam, arm_q[0], grip[0], phrases=phrases,
                                            tcp_zmin=float(low[k, 2]), work_xy=low[k, :2],
                                            grasp_px=grasp_px if grasp_props else ())
    info["init"] = iinfo
    tex = None  # full-resolution first frame, masks and points of the objects: textures of the hires renders
    if use_hi:
        init_state.FRAME_HI = None
        hw = (hires[0].H, hires[0].W)
        masks_hi, msrc = hires_object_masks(dbg["obj_masks"], hw)
        dbg["obj_masks_hi"] = masks_hi
        raster.color_objects(objs, masks_hi, frame0_hi, hires[0])
        bg.depth_hi, dsrc = hires_background_depth(bg.depth, dbg["holes"], masks_hi, hw, init_state.STEREO)
        info["hires"] = dict(first_frame_size=list(frame0_hi.shape[:2]), sim_frame_size=list(frame0.shape[:2]),
                             sam3_input_size=list(frame0_hi.shape[:2]) if config.cfg.sam3_proposals else None,
                             n_sam3_masks_fullres=len(init_state.HIRES_MASKS), object_masks=msrc,
                             object_mask_size=list(hw), point_colours_from=list(frame0_hi.shape[:2]),
                             stereo_size=list(init_state.STEREO.shape) if init_state.STEREO is not None else None,
                             background_depth=dsrc)
        tex = dict(frame0=frame0_hi, masks0={o.id: m for o, m in zip(objs, masks_hi)})
    else:
        raster.color_objects(objs, dbg["obj_masks"], frame0, cam)
    if config.cfg.soft_objects:
        info["soft"] = mark_soft(objs, dbg["obj_masks"], soft_px, bg.plane)
    if snap:
        info["snap"] = refine_state.snap_to_grasps(objs, dbg["obj_masks"], emb, cam, arm_q, grip, bg.plane, skip=empty)
        if config.cfg.pin:
            # the snap moves an object along its camera ray to where the jaws close, which can leave it above the
            # support; without pinning it would fall before the jaws close
            snapped = {r["obj"] for r in info["snap"] if r.get("applied")}
            for o in objs:
                o.pinned = o.id in snapped
    if config.cfg.phantoms:
        only = None
        if config.cfg.phantom_two_pass:
            # only where a rollout of the current state grasps nothing during the closing window, so a phantom never
            # takes the place of a real object that the engine would have grasped
            held = _rollout_held(engine_cls, emb, bg, fps, objs, arm_q, grip, empty)
            only = {t0 for t0, t1 in refine_state.closure_windows(grip, CLOSE_THR)
                    if t0 not in empty and not any(held[t0:min(t1 + 5, len(held))])}
        info["phantoms"] = refine_state.phantom_grasps(objs, emb, arm_q, grip, bg.plane, skip=empty, only=only)
    objects0 = copy.deepcopy(objs)
    if tex is not None:
        tex["pts0"] = {o.id: o.pts.copy() for o in objects0}

    # (2) simulation and (3) rendering
    arm_q, grip = _rollout_inputs(arm_q, grip)
    eng = engine_cls(emb, bg, 1.0 / fps)
    eng.no_grasp_steps = empty
    st = State(t=0, arm_q=arm_q[0], grip=float(grip[0]), objects=objs)
    states, frames, frames_hi, shade_hi = [], [], {}, {}
    for t in range(len(arm_q)):
        if t > 0:
            st = eng.step(st, arm_q[t], grip[t])
        emb.set_state(arm_q[t], grip[t])
        states.append(dict(t=t, grip=st.grip, contacts=dict(st.contacts), tcp=emb.keypoints()["tcp"].tolist(),
                           objects=[dict(id=o.id, centroid=o.pts.mean(0).tolist(), attached=o.attached)
                                    for o in st.objects]))
        frames.append(raster.render(cam, emb, st, bg))
        if hires is not None and t in hires[1]:
            d, c, con, shade_hi[t] = raster.render(hires[0], emb, st, bg, shade=True, tex=tex)
            frames_hi[t] = (d, c, con)
    if tex is not None:
        st_ = tex.get("stats", [])
        info["hires"]["object_pixels_textured_frac"] = float(sum(a for a, _ in st_) / max(1, sum(b for _, b in st_)))
    return dict(cam=cam, bg=bg, frames=frames, frames_hi=frames_hi, shade_hi=shade_hi, states=states, info=info,
                dbg=dbg, objects0=objects0)
